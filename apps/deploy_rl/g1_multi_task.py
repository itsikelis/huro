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
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Joy
from unitree_hg.msg import IMUState, LowCmd, LowState

from huro_py.crc_hg import Crc

G1_NUM_MOTOR = 29
WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)

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

    @classmethod
    def from_ort_session(cls, session: ort.InferenceSession) -> "PolicySpec":
        metadata = dict(session.get_modelmeta().custom_metadata_map)

        def require(key: str) -> str:
            if key not in metadata:
                raise KeyError(
                    f"Missing ONNX metadata key `{key}`. "
                    "Export the policy with deployment metadata."
                )
            return metadata[key]

        action_dim = _parse_scalar_int(require("action_dim"), key="action_dim")
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
            joint_names=tuple(_csv(require("joint_names"))),
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
                _csv(require("action_scale"), float), action_dim, name="action_scale"
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
        )

    def validate(self) -> None:
        num_joints = len(self.joint_names)
        if num_joints != G1_NUM_MOTOR:
            raise ValueError(f"Expected a 29-DOF G1 policy, got {num_joints} joints.")
        if len(set(self.joint_names)) != num_joints:
            raise ValueError("Exported policy joint_names contain duplicates.")
        for name, values in (
            ("default_joint_pos", self.default_joint_pos),
            ("joint_stiffness", self.joint_stiffness),
            ("joint_damping", self.joint_damping),
        ):
            if values.shape != (num_joints,):
                raise ValueError(
                    f"{name} shape {values.shape} does not match num_joints={num_joints}."
                )
        if self.action_semantics not in {"joint_position", "residual_joint_position"}:
            raise NotImplementedError(
                "This deploy app supports `joint_position` and "
                f"`residual_joint_position`, got `{self.action_semantics}`."
            )
        if len(self.action_target_names) != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action target names, got "
                f"{len(self.action_target_names)}."
            )
        required_terms = {
            "command",
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "history",
        }
        exported_terms = {term.name for term in self.observation_terms}
        missing = sorted(required_terms - exported_terms)
        if missing:
            raise ValueError(f"Policy is missing required observation terms: {missing}.")


class G1MultiTaskRunner(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("g1_multi_task_runner")

        self.mode_machine = 0
        self.motors_on = 1
        self.run_policy = False
        self.time = 0.0
        self.init_duration_s = args.init_duration
        self.lowstate_received = False
        self._waiting_for_lowstate_logged = False

        self.max_vx_forward = args.max_vx_forward
        self.max_vx_backward = args.max_vx_backward
        self.max_vy = args.max_vy
        self.max_wz = args.max_wz
        self.deadband = args.deadband
        self.start_button = args.start_button
        self.disable_button = args.disable_button
        if args.control_policy_ratio < 1:
            raise ValueError(
                f"--control-policy-ratio must be >= 1, got {args.control_policy_ratio}."
            )
        self.control_policy_ratio = args.control_policy_ratio

        self._hardware_joint_to_index = {
            name: idx for idx, name in enumerate(HARDWARE_JOINT_NAMES)
        }

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
        self._validate_model_io()

        self.policy_dt = self.spec.control_dt
        self.control_dt = self.policy_dt / self.control_policy_ratio
        self._policy_joint_hw_indices = self._hardware_indices(self.spec.joint_names)
        self._target_hw_indices = self._hardware_indices(self.spec.action_target_names)
        self._target_policy_indices = np.asarray(
            [self.spec.joint_names.index(name) for name in self.spec.action_target_names],
            dtype=np.int32,
        )

        self.joint_positions_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_velocities_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_stiffness_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_damping_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw[self._policy_joint_hw_indices] = self.spec.default_joint_pos
        self.joint_stiffness_hw[self._policy_joint_hw_indices] = self.spec.joint_stiffness
        self.joint_damping_hw[self._policy_joint_hw_indices] = self.spec.joint_damping

        self.imu = IMUState()
        self.joystick = Joy()
        self.actions = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.obs_history: dict[str, np.ndarray] = {}
        self._interp_step = self.control_policy_ratio
        self._interp_start_q_hw = self.default_joint_pos_hw.copy()
        self._interp_end_q_hw = self.default_joint_pos_hw.copy()
        self._last_desired_q_hw = self.default_joint_pos_hw.copy()

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        self.joystick_sub = self.create_subscription(Joy, "/joy", self.joy_handler, 10)
        self.timer = self.create_timer(self.control_dt, self.control)

        self.get_logger().info(
            "Loaded G1 multi-task locomotion policy: "
            f"{args.onnx_path} | policy_dt={self.policy_dt:.4f}s | "
            f"control_dt={self.control_dt:.4f}s | "
            f"control_policy_ratio={self.control_policy_ratio} | "
            f"obs_dim={self.spec.observation_dim} | action_dim={self.spec.action_dim}"
        )
        self.get_logger().info(
            "Press joystick button "
            f"{self.start_button} to start policy; button {self.disable_button} "
            "disables motor commands."
        )

    def control(self) -> None:
        if not self.lowstate_received:
            self._log_waiting_for_lowstate()
            return

        self.time += self.control_dt
        low_cmd = LowCmd()
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = self.mode_machine

        if not self.run_policy:
            desired_q_hw = self._startup_desired_positions()
            self._sync_interpolator(desired_q_hw)
        else:
            if self._interp_step >= self.control_policy_ratio:
                self._start_policy_interpolation_segment()
            desired_q_hw = self._interpolated_desired_positions()
            self._interp_step += 1

        self._fill_low_cmd(low_cmd, desired_q_hw)
        self._last_desired_q_hw = desired_q_hw.copy()
        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

    def _startup_desired_positions(self) -> np.ndarray:
        ratio = np.clip(self.time / max(self.init_duration_s, 1.0e-6), 0.0, 1.0)
        return (1.0 - ratio) * self.joint_positions_hw + ratio * self.default_joint_pos_hw

    def _fill_low_cmd(self, low_cmd: LowCmd, desired_q_hw: np.ndarray) -> None:
        for idx in range(G1_NUM_MOTOR):
            cmd = low_cmd.motor_cmd[idx]
            cmd.mode = self.motors_on
            cmd.q = float(desired_q_hw[idx])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = float(self.joint_stiffness_hw[idx])
            cmd.kd = float(self.joint_damping_hw[idx])

    def _sync_interpolator(self, desired_q_hw: np.ndarray) -> None:
        self._interp_step = self.control_policy_ratio
        self._interp_start_q_hw = desired_q_hw.copy()
        self._interp_end_q_hw = desired_q_hw.copy()

    def _start_policy_interpolation_segment(self) -> None:
        terms = self._term_values()
        self._initialize_history(terms)
        obs = self._build_observation(terms)
        raw_action = self._get_raw_action(obs)
        desired_q_hw = self._desired_positions_from_action(raw_action)
        self._append_history(terms)
        self.actions = raw_action.astype(np.float32, copy=True)

        self._interp_start_q_hw = self._last_desired_q_hw.copy()
        self._interp_end_q_hw = desired_q_hw
        self._interp_step = 0

    def _interpolated_desired_positions(self) -> np.ndarray:
        alpha = (self._interp_step + 1) / self.control_policy_ratio
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (
            (1.0 - alpha) * self._interp_start_q_hw + alpha * self._interp_end_q_hw
        )

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
        joint_pos = self.joint_positions_hw[self._policy_joint_hw_indices]
        joint_vel = self.joint_velocities_hw[self._policy_joint_hw_indices]

        return {
            "command": self._joystick_command(),
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

    def _joystick_command(self) -> np.ndarray:
        vx_axis = self._axis(3)
        vx_scale = self.max_vx_forward if vx_axis >= 0.0 else self.max_vx_backward
        return np.array(
            [
                self._deadband(vx_axis) * vx_scale,
                self._deadband(self._axis(2)) * self.max_vy,
                self._deadband(self._axis(0)) * self.max_wz,
            ],
            dtype=np.float32,
        )

    def _proprio_history_block(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            (
                terms["base_ang_vel"],
                terms["projected_gravity"],
                terms["joint_pos"],
                terms["joint_vel"],
                terms["actions"],
            )
        ).astype(np.float32)

    def _initialize_history(self, terms: dict[str, np.ndarray]) -> None:
        for term in self.spec.observation_terms:
            if term.history_length > 0:
                value = terms[term.name]
                self.obs_history.setdefault(
                    term.name,
                    np.repeat(value[None, :], term.history_length, axis=0).astype(
                        np.float32
                    ),
                )
                continue

            if term.name == "history" and term.name not in self.obs_history:
                current = self._proprio_history_block(terms)
                if term.flat_dim % current.shape[0] != 0:
                    raise ValueError(
                        "Cannot initialize locomotion history term: "
                        f"history dim={term.flat_dim}, proprio dim={current.shape[0]}."
                    )
                history_length = term.flat_dim // current.shape[0]
                self.obs_history[term.name] = np.repeat(
                    current[None, :], history_length, axis=0
                ).astype(np.float32)

    def _append_history(self, terms: dict[str, np.ndarray]) -> None:
        for term in self.spec.observation_terms:
            if term.history_length > 0:
                self.obs_history[term.name][:-1] = self.obs_history[term.name][1:]
                self.obs_history[term.name][-1] = terms[term.name]
                continue
            if term.name == "history" and term.name in self.obs_history:
                self.obs_history[term.name][:-1] = self.obs_history[term.name][1:]
                self.obs_history[term.name][-1] = self._proprio_history_block(terms)

    def _build_observation(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for term in self.spec.observation_terms:
            if term.history_length > 0 or term.name in self.obs_history:
                parts.append(self.obs_history[term.name].reshape(-1))
                continue
            if term.name not in terms:
                raise KeyError(
                    f"Unsupported observation term `{term.name}`. "
                    "Extend this deploy app to provide it."
                )
            value = terms[term.name]
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

    def _desired_positions_from_action(self, raw_action: np.ndarray) -> np.ndarray:
        processed = (
            raw_action.astype(np.float64) * self.spec.action_scale + self.spec.action_offset
        )
        desired_q_hw = self.default_joint_pos_hw.copy()
        if self.spec.action_semantics == "joint_position":
            desired_q_hw[self._target_hw_indices] = processed
        elif self.spec.action_semantics == "residual_joint_position":
            desired_q_hw[self._target_hw_indices] = (
                self.spec.default_joint_pos[self._target_policy_indices] + processed
            )
        else:
            raise NotImplementedError(
                f"Unsupported action semantics `{self.spec.action_semantics}`."
            )
        return desired_q_hw

    def low_state_handler(self, msg: LowState) -> None:
        self.lowstate_received = True
        self.mode_machine = msg.mode_machine
        self.imu = msg.imu_state
        for idx in range(G1_NUM_MOTOR):
            self.joint_positions_hw[idx] = msg.motor_state[idx].q
            self.joint_velocities_hw[idx] = msg.motor_state[idx].dq

    def joy_handler(self, msg: Joy) -> None:
        self.joystick = msg
        if self._button(self.start_button):
            self.run_policy = True
            self.get_logger().info("Policy enabled from joystick.")
        if self._button(self.disable_button):
            self.motors_on = 0
            self.get_logger().warn("Motor disable requested from joystick.")

    def _axis(self, idx: int) -> float:
        return float(self.joystick.axes[idx]) if idx < len(self.joystick.axes) else 0.0

    def _button(self, idx: int) -> bool:
        return idx < len(self.joystick.buttons) and self.joystick.buttons[idx] == 1

    def _deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self.deadband else value

    def _validate_policy_against_hardware(self) -> None:
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

    def _validate_model_io(self) -> None:
        model_input = self._ort_sess.get_inputs()[0]
        model_output = self._ort_sess.get_outputs()[0]
        if model_input.shape[1] != self.spec.observation_dim:
            raise ValueError(
                f"ONNX input dim {model_input.shape[1]} does not match metadata "
                f"observation_dim={self.spec.observation_dim}."
            )
        if model_output.shape[1] != self.spec.action_dim:
            raise ValueError(
                f"ONNX output dim {model_output.shape[1]} does not match metadata "
                f"action_dim={self.spec.action_dim}."
            )

    def _hardware_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [self._hardware_joint_to_index[name] for name in joint_names],
            dtype=np.int32,
        )

    def _log_waiting_for_lowstate(self) -> None:
        if self._waiting_for_lowstate_logged:
            return
        self._waiting_for_lowstate_logged = True
        self.get_logger().warn("Waiting for `/lowstate` before sending commands.")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a G1 multi-task locomotion ONNX policy through HURo."
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the multi-task ONNX policy.",
    )
    parser.add_argument("--max-vx-forward", type=float, default=1.5)
    parser.add_argument("--max-vx-backward", type=float, default=0.7)
    parser.add_argument("--max-vy", type=float, default=0.4)
    parser.add_argument("--max-wz", type=float, default=1.2)
    parser.add_argument("--deadband", type=float, default=0.05)
    parser.add_argument("--init-duration", type=float, default=3.0)
    parser.add_argument("--start-button", type=int, default=1)
    parser.add_argument("--disable-button", type=int, default=0)
    parser.add_argument(
        "--control-policy-ratio",
        "--policy-ratio",
        "--ratio",
        dest="control_policy_ratio",
        type=int,
        default=1,
        help=(
            "Number of high-frequency PD command updates per ONNX policy inference. "
            "The policy still runs at the exported control_dt."
        ),
    )
    return parser


def main(args=None) -> None:
    parser = _build_argparser()
    cli_args = parser.parse_args(remove_ros_args(args=sys.argv if args is None else args)[1:])
    cli_args.onnx_path = cli_args.onnx_path.expanduser().resolve()
    if not cli_args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX policy not found: {cli_args.onnx_path}")

    rclpy.init(args=args)
    node = None
    try:
        node = G1MultiTaskRunner(cli_args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
