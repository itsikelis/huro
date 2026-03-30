#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float32
from unitree_go.msg import SportModeState
from unitree_hg.msg import IMUState, LowCmd, LowState

from huro_py.crc_hg import Crc

G1_NUM_MOTOR = 29

BASE_CMD_TOPIC = "/g1_hand_base/cmd_vel"
HEIGHT_TOPIC = "/g1_hand_base/height"
LEFT_HAND_TOPIC = "/g1_hand_base/left_hand_target"
RIGHT_HAND_TOPIC = "/g1_hand_base/right_hand_target"

WORLD_FRAME = "world"

HARDWARE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)
DEFAULT_TARGET_HEIGHT_M = 0.793
DEFAULT_LEFT_HAND_POS_B = np.array([0.052136, 0.217596, -0.020992], dtype=np.float64)
DEFAULT_LEFT_HAND_QUAT_B = np.array([0.885975, 0.124298, 0.446219, 0.022052], dtype=np.float64)
DEFAULT_RIGHT_HAND_POS_B = np.array([0.047405, -0.224646, -0.026772], dtype=np.float64)
DEFAULT_RIGHT_HAND_QUAT_B = np.array([0.902447, -0.075884, 0.415953, -0.082552], dtype=np.float64)


def _csv(value: str, cast=str) -> list:
    if not value:
        return []
    return [cast(part.strip()) for part in value.split(",")]


def _expand_vector(values: list[float], size: int, *, name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"Missing values for `{name}`.")
    if len(values) == 1:
        values = values * size
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"`{name}` expected shape ({size},), got {vector.shape}.")
    return vector


def _parse_scalar_int(value: str, *, key: str) -> int:
    try:
        return int(value)
    except ValueError:
        parsed = _csv(value, int)
        if len(parsed) != 1:
            raise ValueError(f"Expected scalar integer for `{key}`, got `{value}`.")
        return int(parsed[0])


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dtype=np.float64,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(np.asarray(q, dtype=np.float64))
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_mul(_quat_mul(_quat_conj(q), qv), q)[1:]


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    w, x, y, z = q
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rot6d_from_quat(q: np.ndarray) -> np.ndarray:
    return _quat_to_rotmat(q)[:, :2].T.reshape(-1)


def _msg_quat_xyzw_to_wxyz(x: float, y: float, z: float, w: float) -> np.ndarray:
    return _quat_normalize(np.array([w, x, y, z], dtype=np.float64))


@dataclass(frozen=True)
class ObsTerm:
    name: str
    flat_dim: int
    history_length: int

    @property
    def step_dim(self) -> int:
        if self.history_length <= 0:
            return self.flat_dim
        if self.flat_dim % self.history_length != 0:
            raise ValueError(
                f"Observation term `{self.name}` has flat_dim={self.flat_dim} and "
                f"history_length={self.history_length}, which do not divide cleanly."
            )
        return self.flat_dim // self.history_length


@dataclass(frozen=True)
class PolicySpec:
    control_dt: float
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_semantics: str
    action_dim: int
    action_target_names: tuple[str, ...]
    action_scale: np.ndarray
    action_offset: np.ndarray
    observation_dim: int
    observation_terms: tuple[ObsTerm, ...]
    motion_command_class: str
    motion_command_representation: str
    motion_command_dim: int

    @classmethod
    def from_ort_session(cls, session: ort.InferenceSession) -> "PolicySpec":
        metadata = dict(session.get_modelmeta().custom_metadata_map)

        def require(key: str) -> str:
            if key not in metadata:
                raise KeyError(
                    f"Missing ONNX metadata key `{key}`. "
                    "Export the policy with the current ONNX exporter."
                )
            return metadata[key]

        action_dim = _parse_scalar_int(require("action_dim"), key="action_dim")
        joint_names = tuple(_csv(require("joint_names")))
        observation_terms = tuple(
            ObsTerm(
                name=str(entry["name"]),
                flat_dim=int(entry["flat_dim"]),
                history_length=int(entry["history_length"]),
            )
            for entry in json.loads(require("observation_terms_layout"))
        )

        return cls(
            control_dt=float(require("control_dt")),
            joint_names=joint_names,
            default_joint_pos=np.asarray(
                _csv(require("default_joint_pos"), float), dtype=np.float64
            ),
            joint_stiffness=np.asarray(
                _csv(require("joint_stiffness"), float), dtype=np.float64
            ),
            joint_damping=np.asarray(
                _csv(require("joint_damping"), float), dtype=np.float64
            ),
            action_semantics=require("action_semantics"),
            action_dim=action_dim,
            action_target_names=tuple(_csv(require("action_target_names"))),
            action_scale=_expand_vector(
                _csv(require("action_scale"), float),
                action_dim,
                name="action_scale",
            ),
            action_offset=_expand_vector(
                _csv(metadata.get("action_offset", "0.0"), float),
                action_dim,
                name="action_offset",
            ),
            observation_dim=_parse_scalar_int(
                require("observation_dim"), key="observation_dim"
            ),
            observation_terms=observation_terms,
            motion_command_class=require("motion_command_class"),
            motion_command_representation=metadata.get(
                "motion_command_representation_name", "default"
            ),
            motion_command_dim=_parse_scalar_int(
                metadata.get("motion_command_dim", "0"), key="motion_command_dim"
            ),
        )

    def validate(self) -> None:
        num_joints = len(self.joint_names)
        if num_joints == 0:
            raise ValueError("Exported policy does not list any joints.")
        if self.default_joint_pos.shape != (num_joints,):
            raise ValueError(
                f"default_joint_pos shape {self.default_joint_pos.shape} does not "
                f"match num_joints={num_joints}."
            )
        if self.joint_stiffness.shape != (num_joints,):
            raise ValueError(
                f"joint_stiffness shape {self.joint_stiffness.shape} does not match "
                f"num_joints={num_joints}."
            )
        if self.joint_damping.shape != (num_joints,):
            raise ValueError(
                f"joint_damping shape {self.joint_damping.shape} does not match "
                f"num_joints={num_joints}."
            )
        if self.motion_command_representation != "default":
            raise ValueError(
                "This deploy app only supports policies that consume "
                "the default motion command representation."
            )
        if self.motion_command_class not in {
            "HandBaseMotionCommand",
            "HandBaseTeacherStudentMotionCommand",
        }:
            raise ValueError(
                "This deploy app only supports Hand-Base policies, got "
                f"`{self.motion_command_class}`."
            )
        if self.motion_command_dim != 22:
            raise ValueError(
                "This deploy app expects the current 22-D Hand-Base command, got "
                f"motion_command_dim={self.motion_command_dim}."
            )
        if self.action_semantics != "joint_position":
            raise ValueError(
                "Hand-Base deployment expects `joint_position` actions. "
                f"Loaded policy uses `{self.action_semantics}`."
            )
        if len(self.action_target_names) != self.action_dim:
            raise ValueError(
                f"Expected action_target_names to have length {self.action_dim}, got "
                f"{len(self.action_target_names)}."
            )


@dataclass
class HandTarget:
    pos_w: np.ndarray
    quat_wxyz: np.ndarray


class G1HandBasePolicyRunner(Node):
    """Run a G1 Hand-Base policy from topic-driven hand and base commands."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("g1_hand_base_policy_runner")

        self.mode_pr = 0
        self.mode_machine = 0
        self.motors_on = 1
        self.run_time_s = 0.0
        self._last_missing_state_log_s = -1.0

        self._hardware_joint_to_index = {
            name: idx for idx, name in enumerate(HARDWARE_JOINT_NAMES)
        }

        # Load the exported policy metadata and keep the deployed control path
        # aligned with the training-time Hand-Base interface.
        self._ort_sess = ort.InferenceSession(
            str(args.onnx_path),
            sess_options=ort.SessionOptions(),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._ort_sess.get_inputs()[0].name
        self._output_name = self._ort_sess.get_outputs()[0].name
        self.spec = PolicySpec.from_ort_session(self._ort_sess)
        self.spec.validate()
        self._validate_policy_against_hardware()

        self.control_dt = self.spec.control_dt

        self._policy_joint_hw_indices = self._hardware_indices(self.spec.joint_names)
        self._target_hw_indices = self._hardware_indices(self.spec.action_target_names)

        self.joint_positions_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_velocities_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_stiffness_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_damping_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw[self._policy_joint_hw_indices] = self.spec.default_joint_pos
        self.joint_stiffness_hw[self._policy_joint_hw_indices] = self.spec.joint_stiffness
        self.joint_damping_hw[self._policy_joint_hw_indices] = self.spec.joint_damping

        self.odom_state = SportModeState()
        self.imu = IMUState()
        self.lowstate_received = False
        self.odom_received = False

        self.base_cmd_b = np.zeros(3, dtype=np.float64)
        self.target_height_m = float(DEFAULT_TARGET_HEIGHT_M)
        self.left_hand_target: HandTarget | None = None
        self.right_hand_target: HandTarget | None = None

        self.actions = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.obs_history: dict[str, np.ndarray] = {}

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        self.odommodestate_sub = self.create_subscription(
            SportModeState, "/odommodestate", self.odom_handler, 10
        )
        self.base_cmd_sub = self.create_subscription(
            Twist, BASE_CMD_TOPIC, self.base_cmd_handler, 10
        )
        self.height_sub = self.create_subscription(
            Float32, HEIGHT_TOPIC, self.height_handler, 10
        )
        self.left_hand_sub = self.create_subscription(
            PoseStamped, LEFT_HAND_TOPIC, self.left_hand_handler, 10
        )
        self.right_hand_sub = self.create_subscription(
            PoseStamped, RIGHT_HAND_TOPIC, self.right_hand_handler, 10
        )
        self.timer = self.create_timer(self.control_dt, self.control)

        self.get_logger().info(
            "Loaded topic-driven Hand-Base policy: "
            f"{args.onnx_path} | control_dt={self.control_dt:.4f}s | "
            f"obs_dim={self.spec.observation_dim} | "
            f"action_dim={self.spec.action_dim}"
        )
        self.get_logger().info(
            "Running the policy with zero base velocity, a fixed height, and fixed "
            "default hand references until teleop topics overwrite them."
        )

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------

    def _validate_policy_against_hardware(self) -> None:
        if len(self.spec.joint_names) != G1_NUM_MOTOR:
            raise ValueError(
                f"This deploy app expects a 29-DOF G1 policy, got "
                f"{len(self.spec.joint_names)} joints."
            )
        unknown_joint_names = sorted(
            set(self.spec.joint_names) - set(self._hardware_joint_to_index)
        )
        if unknown_joint_names:
            raise ValueError(
                "Policy references joints that this G1 deploy app does not know: "
                f"{unknown_joint_names}."
            )
        unknown_action_targets = sorted(
            set(self.spec.action_target_names) - set(self._hardware_joint_to_index)
        )
        if unknown_action_targets:
            raise ValueError(
                "Policy action targets are unknown to this G1 deploy app: "
                f"{unknown_action_targets}."
            )

    # ------------------------------------------------------------------
    # Main control loop.
    # ------------------------------------------------------------------

    def control(self) -> None:
        if not self.lowstate_received:
            self._log_missing_state("lowstate")
            return
        if not self.odom_received:
            self._log_missing_state("odommodestate")
            return

        low_cmd = LowCmd()
        low_cmd.mode_pr = self.mode_pr
        low_cmd.mode_machine = self.mode_machine

        terms = self._term_values()
        self._append_history(terms)
        obs = self._build_observation(terms)
        raw_action = self._get_raw_action(obs)
        desired_q_hw = self._desired_positions_from_action(raw_action)
        self.actions = raw_action.astype(np.float32, copy=True)

        self._fill_low_cmd(low_cmd, desired_q_hw)
        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

        self.run_time_s += self.control_dt

    # ------------------------------------------------------------------
    # Command synthesis.
    # ------------------------------------------------------------------

    def _fill_low_cmd(self, low_cmd: LowCmd, desired_q_hw: np.ndarray) -> None:
        for idx in range(G1_NUM_MOTOR):
            cmd = low_cmd.motor_cmd[idx]
            cmd.mode = self.motors_on
            cmd.q = float(desired_q_hw[idx])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = float(self.joint_stiffness_hw[idx])
            cmd.kd = float(self.joint_damping_hw[idx])

    def _desired_positions_from_action(self, raw_action: np.ndarray) -> np.ndarray:
        processed = (
            raw_action.astype(np.float64) * self.spec.action_scale + self.spec.action_offset
        )
        desired_q_hw = self.default_joint_pos_hw.copy()
        desired_q_hw[self._target_hw_indices] = processed
        return desired_q_hw

    # ------------------------------------------------------------------
    # Observation reconstruction.
    # ------------------------------------------------------------------

    def _anchor_pose_w(self) -> tuple[np.ndarray, np.ndarray]:
        anchor_pos_w = np.array(
            [
                self.odom_state.position[0],
                self.odom_state.position[1],
                self.odom_state.position[2],
            ],
            dtype=np.float64,
        )
        anchor_quat_wxyz = np.array(
            [
                self.odom_state.imu_state.quaternion[0],
                self.odom_state.imu_state.quaternion[1],
                self.odom_state.imu_state.quaternion[2],
                self.odom_state.imu_state.quaternion[3],
            ],
            dtype=np.float64,
        )
        return anchor_pos_w, _quat_normalize(anchor_quat_wxyz)

    def _current_hand_targets_b(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.left_hand_target is None or self.right_hand_target is None:
            return (
                DEFAULT_LEFT_HAND_POS_B,
                DEFAULT_LEFT_HAND_QUAT_B,
                DEFAULT_RIGHT_HAND_POS_B,
                DEFAULT_RIGHT_HAND_QUAT_B,
            )

        anchor_pos_w, anchor_quat_wxyz = self._anchor_pose_w()
        left_pos_b = quat_rotate_inverse(
            anchor_quat_wxyz, self.left_hand_target.pos_w - anchor_pos_w
        )
        right_pos_b = quat_rotate_inverse(
            anchor_quat_wxyz, self.right_hand_target.pos_w - anchor_pos_w
        )
        left_quat_b = _quat_mul(_quat_conj(anchor_quat_wxyz), self.left_hand_target.quat_wxyz)
        right_quat_b = _quat_mul(
            _quat_conj(anchor_quat_wxyz), self.right_hand_target.quat_wxyz
        )
        return left_pos_b, left_quat_b, right_pos_b, right_quat_b

    def _hand_base_command(self) -> np.ndarray:
        left_pos_b, left_quat_b, right_pos_b, right_quat_b = self._current_hand_targets_b()

        return np.concatenate(
            (
                left_pos_b,
                right_pos_b,
                _rot6d_from_quat(left_quat_b),
                _rot6d_from_quat(right_quat_b),
                self.base_cmd_b[:2],
                self.base_cmd_b[2:3],
                np.array([self.target_height_m], dtype=np.float64),
            )
        ).astype(np.float32)

    def _term_values(self) -> dict[str, np.ndarray]:
        quat = np.array(
            [
                self.imu.quaternion[0],
                self.imu.quaternion[1],
                self.imu.quaternion[2],
                self.imu.quaternion[3],
            ],
            dtype=np.float64,
        )
        base_lin_vel = np.array(
            [
                self.odom_state.velocity[0],
                self.odom_state.velocity[1],
                self.odom_state.velocity[2],
            ],
            dtype=np.float64,
        )
        joint_pos = self.joint_positions_hw[self._policy_joint_hw_indices]
        joint_vel = self.joint_velocities_hw[self._policy_joint_hw_indices]

        return {
            "command": self._hand_base_command(),
            "base_lin_vel": base_lin_vel.astype(np.float32),
            "base_ang_vel": np.array(
                [
                    self.imu.gyroscope[0],
                    self.imu.gyroscope[1],
                    self.imu.gyroscope[2],
                ],
                dtype=np.float32,
            ),
            "projected_gravity": quat_rotate_inverse(quat, WORLD_GRAVITY).astype(np.float32),
            "joint_pos": (joint_pos - self.spec.default_joint_pos).astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "actions": self.actions.astype(np.float32, copy=False),
        }

    def _append_history(self, terms: dict[str, np.ndarray]) -> None:
        for term in self.spec.observation_terms:
            if term.name not in terms:
                continue
            value = terms[term.name]
            if term.history_length <= 0:
                continue
            if value.shape != (term.step_dim,):
                raise ValueError(
                    f"Observation term `{term.name}` has shape {value.shape}, expected "
                    f"({term.step_dim},)."
                )
            if term.name not in self.obs_history:
                self.obs_history[term.name] = np.repeat(
                    value[None, :], term.history_length, axis=0
                ).astype(np.float32)
            else:
                self.obs_history[term.name][:-1] = self.obs_history[term.name][1:]
                self.obs_history[term.name][-1] = value

    def _build_observation(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for term in self.spec.observation_terms:
            if term.name not in terms:
                raise KeyError(
                    f"Unsupported observation term `{term.name}`. "
                    "Extend this deploy app to provide it."
                )
            value = terms[term.name]
            if term.history_length > 0:
                if term.name not in self.obs_history:
                    raise RuntimeError(
                        f"History for observation term `{term.name}` is not initialized."
                    )
                parts.append(self.obs_history[term.name].reshape(-1))
            else:
                if value.shape != (term.step_dim,):
                    raise ValueError(
                        f"Observation term `{term.name}` has shape {value.shape}, expected "
                        f"({term.step_dim},)."
                    )
                parts.append(value)
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (self.spec.observation_dim,):
            raise ValueError(
                f"Constructed observation has shape {obs.shape}, expected "
                f"({self.spec.observation_dim},)."
            )
        return obs

    def _get_raw_action(self, obs: np.ndarray) -> np.ndarray:
        outputs = self._ort_sess.run(
            [self._output_name],
            {self._input_name: obs.reshape(1, -1).astype(np.float32, copy=False)},
        )
        return outputs[0].reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Topic callbacks.
    # ------------------------------------------------------------------

    def low_state_handler(self, msg: LowState) -> None:
        self.lowstate_received = True
        self.mode_machine = msg.mode_machine
        self.imu = msg.imu_state
        for idx in range(G1_NUM_MOTOR):
            self.joint_positions_hw[idx] = msg.motor_state[idx].q
            self.joint_velocities_hw[idx] = msg.motor_state[idx].dq

    def odom_handler(self, msg: SportModeState) -> None:
        self.odom_received = True
        self.odom_state = msg

    def base_cmd_handler(self, msg: Twist) -> None:
        self.base_cmd_b[0] = float(msg.linear.x)
        self.base_cmd_b[1] = float(msg.linear.y)
        self.base_cmd_b[2] = float(msg.angular.z)

    def height_handler(self, msg: Float32) -> None:
        self.target_height_m = float(msg.data)

    def left_hand_handler(self, msg: PoseStamped) -> None:
        self.left_hand_target = self._hand_target_from_msg(msg)

    def right_hand_handler(self, msg: PoseStamped) -> None:
        self.right_hand_target = self._hand_target_from_msg(msg)

    def _hand_target_from_msg(self, msg: PoseStamped) -> HandTarget:
        if msg.header.frame_id and msg.header.frame_id != "world":
            self.get_logger().warn(
                f"Received hand target in frame `{msg.header.frame_id}`. "
                "This policy node currently expects `world`."
            )
        return HandTarget(
            pos_w=np.array(
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ],
                dtype=np.float64,
            ),
            quat_wxyz=_msg_quat_xyzw_to_wxyz(
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ),
        )

    # ------------------------------------------------------------------
    # Small helpers.
    # ------------------------------------------------------------------

    def _log_missing_state(self, source: str) -> None:
        if self.run_time_s - self._last_missing_state_log_s < 1.0:
            return
        self._last_missing_state_log_s = self.run_time_s
        self.get_logger().warn(f"Waiting for `{source}` before sending commands.")

    def _hardware_indices(self, joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.asarray(
            [self._hardware_joint_to_index[name] for name in joint_names],
            dtype=np.int32,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a topic-driven G1 Hand-Base policy through HURo."
        )
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="ONNX policy with embedded deployment metadata.",
    )
    return parser


def main(args=None) -> None:
    parser = _build_argparser()
    cli_args = parser.parse_args(
        remove_ros_args(args=sys.argv if args is None else args)[1:]
    )
    cli_args.onnx_path = cli_args.onnx_path.expanduser().resolve()

    rclpy.init(args=args)
    node = None
    try:
        node = G1HandBasePolicyRunner(cli_args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
