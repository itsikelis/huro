#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from unitree_go.msg import SportModeState
from unitree_hg.msg import IMUState, LowCmd, LowState

from huro_py.crc_hg import Crc

G1_NUM_MOTOR = 29

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
REFERENCE_TRANSITION_DURATION_S = 2.0

# Frozen standing pose derived from the mean joint position of
# `resources/motions/A1___Stand.npz`. This is used as the idle motion-command
# reference when playback is not active.
IDLE_REFERENCE_JOINT_POS = np.array(
    [
        0.131463,
        0.003387,
        0.15493,
        0.060897,
        -0.084281,
        0.052526,
        0.139701,
        -0.016189,
        -0.229621,
        0.029136,
        -0.071553,
        -0.079474,
        -0.078601,
        0.02141,
        0.043132,
        0.432435,
        0.219976,
        -0.011892,
        0.46946,
        0.0,
        0.0,
        0.0,
        0.34118,
        -0.224489,
        -0.000747,
        0.473817,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


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


def _parse_scalar_int(value: str, *, key: str) -> int:
    try:
        return int(value)
    except ValueError:
        parsed = _csv(value, int)
        if len(parsed) != 1:
            raise ValueError(f"Expected scalar integer for `{key}`, got `{value}`.")
        return int(parsed[0])


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
        if len(set(self.joint_names)) != num_joints:
            raise ValueError("Exported policy joint_names contain duplicates.")
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
        expected_command_dim = 2 * num_joints
        if self.motion_command_dim != expected_command_dim:
            raise ValueError(
                "This deploy app only supports current joint-reference commands with "
                f"dim = 2 * num_joints. Got motion_command_dim={self.motion_command_dim}, "
                f"num_joints={num_joints}."
            )
        if self.action_semantics not in {
            "residual_joint_position",
            "joint_position",
        }:
            raise NotImplementedError(
                "This deploy app currently supports only `residual_joint_position` "
                f"and `joint_position`, got `{self.action_semantics}`."
            )
        if len(self.action_target_names) != self.action_dim:
            raise ValueError(
                f"Expected action_target_names to have length {self.action_dim}, got "
                f"{len(self.action_target_names)}."
            )


@dataclass(frozen=True)
class MotionFrame:
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    @classmethod
    def stationary(cls, joint_pos: np.ndarray) -> "MotionFrame":
        return cls(
            joint_pos=np.asarray(joint_pos, dtype=np.float64).copy(),
            joint_vel=np.zeros_like(joint_pos, dtype=np.float64),
        )

    def copy(self) -> "MotionFrame":
        return MotionFrame(
            joint_pos=self.joint_pos.copy(),
            joint_vel=self.joint_vel.copy(),
        )

    @staticmethod
    def blend(start: "MotionFrame", end: "MotionFrame", alpha: float) -> "MotionFrame":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return MotionFrame(
            joint_pos=(1.0 - alpha) * start.joint_pos + alpha * end.joint_pos,
            joint_vel=(1.0 - alpha) * start.joint_vel + alpha * end.joint_vel,
        )


@dataclass
class ReferenceTransition:
    start_frame: MotionFrame
    end_frame: MotionFrame
    target_mode: str
    duration_s: float
    elapsed_s: float = 0.0

    def frame(self) -> MotionFrame:
        return MotionFrame.blend(
            self.start_frame,
            self.end_frame,
            self.elapsed_s / max(self.duration_s, 1.0e-6),
        )

    def advance(self, dt: float) -> bool:
        self.elapsed_s += dt
        return self.elapsed_s > self.duration_s


class MotionClip:
    def __init__(self, npz_path: Path, expected_joint_dim: int):
        with np.load(npz_path, allow_pickle=False) as data:
            self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
            self.fps = self._extract_fps(data)

        if self.joint_pos.ndim != 2 or self.joint_pos.shape[1] != expected_joint_dim:
            raise ValueError(
                f"`joint_pos` in {npz_path} has shape {self.joint_pos.shape}, expected "
                f"(num_frames, {expected_joint_dim})."
            )
        if self.joint_vel.shape != self.joint_pos.shape:
            raise ValueError(
                f"`joint_vel` in {npz_path} has shape {self.joint_vel.shape}, expected "
                f"{self.joint_pos.shape}."
            )

    @staticmethod
    def _extract_fps(data) -> float:
        if "fps" not in data:
            return 30.0
        values = np.asarray(data["fps"]).reshape(-1)
        return float(values[0]) if values.size > 0 else 30.0

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def length_s(self) -> float:
        return float(max(self.num_frames - 1, 0)) / max(self.fps, 1.0e-6)

    def sample(self, time_s: float) -> MotionFrame:
        if self.num_frames <= 1:
            idx0 = idx1 = 0
            blend = 0.0
        else:
            clip_time = float(np.clip(time_s, 0.0, self.length_s))
            phase = clip_time * self.fps
            idx0 = int(np.floor(phase))
            idx1 = min(idx0 + 1, self.num_frames - 1)
            blend = float(phase - idx0)

        return MotionFrame(
            joint_pos=(1.0 - blend) * self.joint_pos[idx0]
            + blend * self.joint_pos[idx1],
            joint_vel=(1.0 - blend) * self.joint_vel[idx0]
            + blend * self.joint_vel[idx1],
        )


class G1MotionTrackingRunner(Node):
    """Run the motion-tracking policy continuously, with a switchable motion reference.

    By default it uses a frozen standing pose as the motion-command reference.
    When the user presses the start key, the reference transitions into the motion
    clip start. Once the clip ends, the reference transitions back to the default pose.
    """

    # ------------------------------------------------------------------
    # Initialization and ROS wiring.
    # ------------------------------------------------------------------

    def __init__(self, args: argparse.Namespace):
        super().__init__("g1_motion_tracking_runner")

        # The reference can be idle, transitioning, or following the active clip.
        self.mode_pr = 0
        self.mode_machine = 0
        self.motors_on = 1
        self.time_s = 0.0
        self.motion_elapsed_s = 0.0
        self.play_motion = False
        self.transition_duration_s = REFERENCE_TRANSITION_DURATION_S
        self.transition: ReferenceTransition | None = None
        self._last_missing_state_log_s = -1.0
        self._last_missing_odom_log_s = -1.0
        self._stdin_fd: int | None = None
        self._stdin_termios: list | None = None
        self._keyboard_enabled = False

        self._hardware_joint_to_index = {
            name: idx for idx, name in enumerate(HARDWARE_JOINT_NAMES)
        }

        # Load the exported policy and the reference clip it will consume.
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
        self.motion_clip = MotionClip(
            args.motion_npz,
            expected_joint_dim=len(self.spec.joint_names),
        )

        # Convert between policy joint ordering and the fixed hardware motor ordering.
        self._reference_hw_indices = self._hardware_indices(self.spec.joint_names)
        self._target_hw_indices = self._hardware_indices(self.spec.action_target_names)
        self._reference_action_indices = np.asarray(
            [self.spec.joint_names.index(name) for name in self.spec.action_target_names],
            dtype=np.int32,
        )

        self.joint_positions_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_velocities_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_stiffness_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_damping_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw[self._reference_hw_indices] = self.spec.default_joint_pos
        self.joint_stiffness_hw[self._reference_hw_indices] = self.spec.joint_stiffness
        self.joint_damping_hw[self._reference_hw_indices] = self.spec.joint_damping

        self.idle_reference_frame = MotionFrame.stationary(IDLE_REFERENCE_JOINT_POS)

        # Live robot state and observation-history buffers.
        self.odom_state = SportModeState()
        self.imu = IMUState()
        self.lowstate_received = False
        self.odom_received = False

        self.actions = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.obs_history: dict[str, np.ndarray] = {}

        # ROS interfaces and keyboard setup.
        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        self.odommodestate_sub = self.create_subscription(
            SportModeState, "/odommodestate", self.odom_handler, 10
        )
        self._setup_keyboard()
        self.timer = self.create_timer(self.control_dt, self.control)

        self.get_logger().info(
            "Loaded motion-tracking policy: "
            f"{args.onnx_path} | motion: {args.motion_npz} | "
            f"control_dt={self.control_dt:.4f}s | obs_dim={self.spec.observation_dim} | "
            f"action_dim={self.spec.action_dim}"
        )
        if self._keyboard_enabled:
            self.get_logger().info(
                "Starting in default-reference mode. Press SPACE or ENTER to "
                "play the motion clip, and `x` to disable the motors."
            )
        else:
            self.get_logger().warn(
                "Starting in default-reference mode, but keyboard control is "
                "disabled because stdin is not a TTY."
            )
        self.get_logger().info(
            "Idle reference uses a frozen standing pose"
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
        unsupported_joint_names = sorted(
            set(self.spec.joint_names) - set(self._hardware_joint_to_index)
        )
        if unsupported_joint_names:
            raise ValueError(
                "Policy references joints that this G1 deploy app does not know: "
                f"{unsupported_joint_names}."
            )
        unsupported_action_targets = sorted(
            set(self.spec.action_target_names) - set(self._hardware_joint_to_index)
        )
        if unsupported_action_targets:
            raise ValueError(
                "Policy action targets are unknown to this G1 deploy app: "
                f"{unsupported_action_targets}."
            )
        input_shape = self._ort_sess.get_inputs()[0].shape
        if len(input_shape) != 2 or input_shape[1] != self.spec.observation_dim:
            raise ValueError(
                f"ONNX input shape {input_shape} does not match metadata "
                f"observation_dim={self.spec.observation_dim}."
            )
        output_shape = self._ort_sess.get_outputs()[0].shape
        if len(output_shape) != 2 or output_shape[1] != self.spec.action_dim:
            raise ValueError(
                f"ONNX output shape {output_shape} does not match metadata "
                f"action_dim={self.spec.action_dim}."
            )

    # ------------------------------------------------------------------
    # Main control loop.
    # ------------------------------------------------------------------

    def control(self) -> None:
        self._poll_keyboard()

        if not self.lowstate_received:
            self._log_missing_state("lowstate")
            return

        self.time_s += self.control_dt

        low_cmd = LowCmd()
        low_cmd.mode_pr = self.mode_pr
        low_cmd.mode_machine = self.mode_machine

        # The policy always runs. The reference is either:
        # - the idle standing pose,
        # - a smooth transition frame,
        # - or the active motion clip.
        frame = self._current_reference_frame()
        terms = self._term_values(frame)
        self._append_history(terms)
        obs = self._build_observation(terms)
        raw_action = self._get_raw_action(obs)
        desired_q_hw = self._desired_positions_from_action(raw_action, frame)
        self.actions = raw_action.astype(np.float32, copy=True)

        self._fill_low_cmd(low_cmd, desired_q_hw)
        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

        self._advance_reference_state(frame)

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

    def _desired_positions_from_action(
        self,
        raw_action: np.ndarray,
        frame: MotionFrame,
    ) -> np.ndarray:
        processed = raw_action.astype(np.float64) * self.spec.action_scale + self.spec.action_offset
        desired_q_hw = self.default_joint_pos_hw.copy()
        desired_q_hw[self._reference_hw_indices] = frame.joint_pos

        if self.spec.action_semantics == "residual_joint_position":
            desired_q_hw[self._target_hw_indices] = (
                frame.joint_pos[self._reference_action_indices] + processed
            )
        elif self.spec.action_semantics == "joint_position":
            desired_q_hw[self._target_hw_indices] = processed
        else:
            raise NotImplementedError(
                f"Unsupported action semantics `{self.spec.action_semantics}`."
            )
        return desired_q_hw

    # ------------------------------------------------------------------
    # Observation reconstruction.
    # ------------------------------------------------------------------

    def _term_values(self, frame: MotionFrame) -> dict[str, np.ndarray]:
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
        if not self.odom_received and self.time_s - self._last_missing_odom_log_s >= 1.0:
            self._last_missing_odom_log_s = self.time_s
            self.get_logger().warn(
                "No `/odommodestate` received yet; using zero base linear velocity."
            )

        joint_pos = self.joint_positions_hw[self._reference_hw_indices]
        joint_vel = self.joint_velocities_hw[self._reference_hw_indices]

        return {
            "command": np.concatenate((frame.joint_pos, frame.joint_vel)).astype(np.float32),
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
    # ROS callbacks and playback state changes.
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

    def _reset_policy_state(self, frame: MotionFrame) -> None:
        # When we switch references, restart action/history so the policy sees a
        # clean first step for the new reference source.
        self.actions[:] = 0.0
        self.obs_history.clear()
        if self.lowstate_received:
            self._append_history(self._term_values(frame))

    def _start_motion_reference(self) -> None:
        if self.play_motion or (
            self.transition is not None and self.transition.target_mode == "motion"
        ):
            return

        self._begin_transition(
            self._current_reference_frame(),
            self.motion_clip.sample(0.0),
            target_mode="motion",
            message="Transitioning into motion playback.",
        )

    def _switch_to_default_reference(self, message: str | None = None) -> None:
        self.play_motion = False
        self.transition = None
        self.motion_elapsed_s = 0.0
        self._reset_policy_state(self.idle_reference_frame)
        if message:
            self.get_logger().info(message)

    # ------------------------------------------------------------------
    # Small helpers.
    # ------------------------------------------------------------------

    def _setup_keyboard(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_termios = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._keyboard_enabled = True
        except (termios.error, ValueError, OSError) as exc:
            self.get_logger().warn(f"Failed to enable keyboard control: {exc}")
            self._stdin_fd = None
            self._stdin_termios = None
            self._keyboard_enabled = False

    def _restore_keyboard(self) -> None:
        if self._stdin_fd is None or self._stdin_termios is None:
            return
        try:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_termios)
        except (termios.error, ValueError, OSError):
            pass
        self._stdin_fd = None
        self._stdin_termios = None
        self._keyboard_enabled = False

    def _poll_keyboard(self) -> None:
        if not self._keyboard_enabled or self._stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self._stdin_fd], [], [], 0.0)
            if not ready:
                break
            key = sys.stdin.read(1)
            if not key:
                break
            self._handle_keypress(key)

    def _handle_keypress(self, key: str) -> None:
        if key in {" ", "\n", "\r"}:
            self.motors_on = 1
            self._start_motion_reference()
            return

        if key in {"x", "X"}:
            self.motors_on = 0
            self._switch_to_default_reference("Motor disable requested from keyboard.")
            self.get_logger().warn("Motor disable requested from keyboard.")

    def _log_missing_state(self, source: str) -> None:
        if self.time_s - self._last_missing_state_log_s < 1.0:
            return
        self._last_missing_state_log_s = self.time_s
        self.get_logger().warn(f"Waiting for `{source}` before sending commands.")

    def _current_reference_frame(self) -> MotionFrame:
        if self.transition is not None:
            return self.transition.frame()
        if self.play_motion:
            return self.motion_clip.sample(self.motion_elapsed_s)
        return self.idle_reference_frame

    def _advance_reference_state(self, current_frame: MotionFrame) -> None:
        if self.transition is not None:
            if self.transition.advance(self.control_dt):
                target_mode = self.transition.target_mode
                self.transition = None
                if target_mode == "motion":
                    self.play_motion = True
                    self.motion_elapsed_s = min(self.control_dt, self.motion_clip.length_s)
                    self.get_logger().info("Motion playback started.")
                else:
                    self.play_motion = False
                    self.motion_elapsed_s = 0.0
                    self.get_logger().info("Returned to standing reference.")
            return

        if self.play_motion:
            self.motion_elapsed_s += self.control_dt
            # Stop after the final frame has been used once, then transition to idle.
            if self.motion_elapsed_s > self.motion_clip.length_s:
                self._begin_transition(
                    current_frame,
                    self.idle_reference_frame,
                    target_mode="idle",
                    message="Motion clip completed. Transitioning back to standing reference.",
                )

    def _begin_transition(
        self,
        start_frame: MotionFrame,
        end_frame: MotionFrame,
        *,
        target_mode: str,
        message: str | None = None,
    ) -> None:
        self.play_motion = False
        self.motion_elapsed_s = 0.0
        self.transition = ReferenceTransition(
            start_frame=start_frame.copy(),
            end_frame=end_frame.copy(),
            target_mode=target_mode,
            duration_s=self.transition_duration_s,
        )
        self._reset_policy_state(self.transition.start_frame)
        if message:
            self.get_logger().info(message)

    def close(self) -> None:
        self._restore_keyboard()

    def _hardware_indices(self, joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.asarray(
            [self._hardware_joint_to_index[name] for name in joint_names],
            dtype=np.int32,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a G1 motion-tracking policy through HURo."
        )
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="ONNX policy with embedded deployment metadata.",
    )
    parser.add_argument(
        "--motion-npz",
        type=Path,
        required=True,
        help="Reference motion clip in NPZ format.",
    )
    return parser


def main(args=None) -> None:
    parser = _build_argparser()
    cli_args = parser.parse_args(remove_ros_args(args=sys.argv if args is None else args)[1:])
    cli_args.onnx_path = cli_args.onnx_path.expanduser().resolve()
    cli_args.motion_npz = cli_args.motion_npz.expanduser().resolve()

    rclpy.init(args=args)
    node = None
    try:
        node = G1MotionTrackingRunner(cli_args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
