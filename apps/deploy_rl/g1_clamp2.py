#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
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
SUPPORTED_MOTION_COMMAND_CLASS = "JointRefAnchorRpMotionCommand"
REFERENCE_TRANSITION_DURATION_S = 2.0

IDLE_REFERENCE_HEIGHT_M = 0.78


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


def _decode_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _extract_fps(data) -> float:
    if "fps" not in data:
        return 30.0
    values = np.asarray(data["fps"]).reshape(-1)
    return float(values[0]) if values.size > 0 else 30.0


def _extract_body_names(data) -> tuple[str, ...]:
    for key in ("body_names", "body_link_names"):
        if key in data:
            values = np.asarray(data[key]).reshape(-1).tolist()
            return tuple(_decode_name(value) for value in values)
    raise ValueError("Motion npz must contain `body_names` or `body_link_names`.")


def _lerp(a: np.ndarray, b: np.ndarray, blend: float) -> np.ndarray:
    return (1.0 - blend) * a + blend * b


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


def _quat_roll_pitch_yaw(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, blend: float) -> np.ndarray:
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(_lerp(q0, q1, blend))

    dot = float(np.clip(dot, 0.0, 1.0))
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * blend
    sin_theta_0 = float(np.sin(theta_0))
    s0 = np.sin(theta_0 - theta) / max(sin_theta_0, 1.0e-8)
    s1 = np.sin(theta) / max(sin_theta_0, 1.0e-8)
    return _quat_normalize(s0 * q0 + s1 * q1)


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
    root_body_name: str

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
            root_body_name=metadata.get("root_body_name", "pelvis"),
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

        if self.motion_command_class != SUPPORTED_MOTION_COMMAND_CLASS:
            raise NotImplementedError(
                f"This deploy app supports only `{SUPPORTED_MOTION_COMMAND_CLASS}`. "
                f"Got `{self.motion_command_class}`."
            )

        expected_command_dim = 2 * num_joints + 6
        if self.motion_command_dim != expected_command_dim:
            raise ValueError(
                f"motion_command_dim={self.motion_command_dim} does not match expected "
                f"dim={expected_command_dim} for `{SUPPORTED_MOTION_COMMAND_CLASS}`."
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
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray

    @classmethod
    def stationary(cls, joint_pos: np.ndarray, *, height_m: float) -> "MotionFrame":
        return cls(
            joint_pos=np.asarray(joint_pos, dtype=np.float64).copy(),
            joint_vel=np.zeros_like(joint_pos, dtype=np.float64),
            root_pos_w=np.array([0.0, 0.0, height_m], dtype=np.float64),
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            root_lin_vel_w=np.zeros(3, dtype=np.float64),
            root_ang_vel_w=np.zeros(3, dtype=np.float64),
        )

    def copy(self) -> "MotionFrame":
        return MotionFrame(
            joint_pos=self.joint_pos.copy(),
            joint_vel=self.joint_vel.copy(),
            root_pos_w=self.root_pos_w.copy(),
            root_quat_wxyz=self.root_quat_wxyz.copy(),
            root_lin_vel_w=self.root_lin_vel_w.copy(),
            root_ang_vel_w=self.root_ang_vel_w.copy(),
        )

    @staticmethod
    def blend(start: "MotionFrame", end: "MotionFrame", alpha: float) -> "MotionFrame":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return MotionFrame(
            joint_pos=_lerp(start.joint_pos, end.joint_pos, alpha),
            joint_vel=_lerp(start.joint_vel, end.joint_vel, alpha),
            root_pos_w=_lerp(start.root_pos_w, end.root_pos_w, alpha),
            root_quat_wxyz=_quat_slerp(start.root_quat_wxyz, end.root_quat_wxyz, alpha),
            root_lin_vel_w=_lerp(start.root_lin_vel_w, end.root_lin_vel_w, alpha),
            root_ang_vel_w=_lerp(start.root_ang_vel_w, end.root_ang_vel_w, alpha),
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
        return self.elapsed_s >= self.duration_s


class NpzEpisodeLogger:
    def __init__(self, log_dir: Path, ros_logger, *, log_label: str | None = None):
        self.root_log_dir = log_dir
        self.log_label = (log_label or "").strip()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_label = self._safe_name(self.log_label)
        self.run_id = f"{timestamp}_{safe_label}" if safe_label else timestamp
        self.log_dir = log_dir / self.run_id
        self.ros_logger = ros_logger
        self.episode_index = 0
        self.active_name: str | None = None
        self.active_metadata: dict[str, object] = {}
        self.samples: dict[str, list[np.ndarray | float | int]] = {}
        self.start_time_ns: int | None = None
        self.saved_active_episode = False

    @property
    def active(self) -> bool:
        return self.active_name is not None

    def start_episode(self, name: str, metadata: dict[str, object]) -> None:
        if self.active:
            self.finish_episode(completed=False, suffix="interrupted")
        self.episode_index += 1
        self.active_name = name
        self.active_metadata = dict(metadata)
        self.samples = {}
        self.start_time_ns = None
        self.saved_active_episode = False

    def record(self, now_ns: int, sample: dict[str, np.ndarray | float | int]) -> None:
        if not self.active:
            return
        if self.start_time_ns is None:
            self.start_time_ns = now_ns
        sample = dict(sample)
        sample["time_s"] = (now_ns - self.start_time_ns) * 1.0e-9
        for key, value in sample.items():
            self.samples.setdefault(key, []).append(value)

    def finish_episode(
        self,
        *,
        completed: bool,
        suffix: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path | None:
        if not self.active:
            return None
        if self.saved_active_episode:
            return None

        if not self.samples:
            self._clear_episode()
            return None

        all_metadata = dict(self.active_metadata)
        if metadata:
            all_metadata.update(metadata)
        all_metadata["completed"] = bool(completed)
        all_metadata["episode_index"] = int(self.episode_index)
        all_metadata["run_id"] = self.run_id
        all_metadata["log_label"] = self.log_label

        arrays = {
            key: np.asarray(values)
            for key, values in self.samples.items()
        }
        if "control_dt" in all_metadata:
            arrays["control_dt"] = np.asarray(float(all_metadata["control_dt"]))
        for key, value in all_metadata.items():
            arrays[f"meta_{key}"] = self._metadata_value(value)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        filename = self._episode_filename(self.active_name or "episode", suffix=suffix)
        path = self.log_dir / filename
        np.savez_compressed(path, **arrays)
        self.saved_active_episode = True
        self.ros_logger.info(f"Saved log episode to {path}.")
        self._clear_episode()
        return path

    def _clear_episode(self) -> None:
        self.active_name = None
        self.active_metadata = {}
        self.samples = {}
        self.start_time_ns = None
        self.saved_active_episode = False

    def _episode_filename(self, name: str, *, suffix: str | None) -> str:
        safe_name = self._safe_name(name) or "episode"
        suffix_text = f"_{suffix}" if suffix else ""
        return f"{self.episode_index:04d}_{safe_name}{suffix_text}.npz"

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

    @staticmethod
    def _metadata_value(value: object) -> np.ndarray:
        if isinstance(value, Path):
            return np.asarray(str(value))
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return np.asarray(value)


class MotionClip:
    def __init__(self, npz_path: Path, *, spec: PolicySpec):
        with np.load(npz_path, allow_pickle=False) as data:
            self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
            self.fps = _extract_fps(data)
            self.body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
            self.body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
            self.body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
            self.body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
            self.body_names = _extract_body_names(data)

        expected_joint_dim = len(spec.joint_names)
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

        self._body_name_to_index = {name: idx for idx, name in enumerate(self.body_names)}
        self.root_body_index = self.body_index(spec.root_body_name)

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def length_s(self) -> float:
        return float(max(self.num_frames - 1, 0)) / max(self.fps, 1.0e-6)

    def body_index(self, body_name: str) -> int:
        if body_name not in self._body_name_to_index:
            raise ValueError(f"Body `{body_name}` not found in loaded motion clip.")
        return self._body_name_to_index[body_name]

    def sample(self, time_s: float) -> MotionFrame:
        idx0, idx1, blend = self._sample_indices(time_s)
        root_pos_w = _lerp(
            self.body_pos_w[idx0, self.root_body_index],
            self.body_pos_w[idx1, self.root_body_index],
            blend,
        )
        root_quat_wxyz = _quat_slerp(
            self.body_quat_w[idx0, self.root_body_index],
            self.body_quat_w[idx1, self.root_body_index],
            blend,
        )
        root_lin_vel_w = _lerp(
            self.body_lin_vel_w[idx0, self.root_body_index],
            self.body_lin_vel_w[idx1, self.root_body_index],
            blend,
        )
        root_ang_vel_w = _lerp(
            self.body_ang_vel_w[idx0, self.root_body_index],
            self.body_ang_vel_w[idx1, self.root_body_index],
            blend,
        )

        return MotionFrame(
            joint_pos=_lerp(self.joint_pos[idx0], self.joint_pos[idx1], blend),
            joint_vel=_lerp(self.joint_vel[idx0], self.joint_vel[idx1], blend),
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
        )

    def _sample_indices(self, time_s: float) -> tuple[int, int, float]:
        if self.num_frames <= 1 or self.length_s <= 0.0:
            return 0, 0, 0.0
        clip_time = float(np.clip(time_s, 0.0, self.length_s))
        phase = clip_time * self.fps
        idx0 = int(np.floor(phase))
        idx1 = min(idx0 + 1, self.num_frames - 1)
        return idx0, idx1, float(phase - idx0)


class G1Clamp2Runner(Node):
    """Run a G1 CLAMP/CLAMP2 ONNX policy through HURo."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(getattr(args, "node_name", "g1_clamp2_runner"))

        self.mode_machine = 0
        self.motors_on = 1
        self.motion_time_s = 0.0
        self._waiting_for_lowstate_logged = False
        self._stdin_fd: int | None = None
        self._stdin_termios: list | None = None
        self._keyboard_enabled = False

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

        self.control_dt = self.spec.control_dt
        self.idle_reference_frame = MotionFrame.stationary(
            self.spec.default_joint_pos,
            height_m=IDLE_REFERENCE_HEIGHT_M,
        )
        self.imu = IMUState()
        self.episode_logger = (
            NpzEpisodeLogger(
                args.log_dir,
                self.get_logger(),
                log_label=getattr(args, "log_label", None),
            )
            if getattr(args, "log_dir", None) is not None
            else None
        )
        self.onnx_path = args.onnx_path
        self._init_reference(args)

        self._policy_joint_hw_indices = self._hardware_indices(self.spec.joint_names)
        self._target_hw_indices = self._hardware_indices(self.spec.action_target_names)
        self._target_frame_joint_indices = np.asarray(
            [self.spec.joint_names.index(name) for name in self.spec.action_target_names],
            dtype=np.int32,
        )

        self.joint_positions_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_velocities_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_accelerations_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_torques_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.motor_temperature_ch0_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.motor_temperature_ch1_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.motor_voltage_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.motor_state_flags_hw = np.zeros(G1_NUM_MOTOR, dtype=np.int64)
        self.motor_mode_hw = np.zeros(G1_NUM_MOTOR, dtype=np.int64)
        self.default_joint_pos_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_stiffness_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.joint_damping_hw = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.default_joint_pos_hw[self._policy_joint_hw_indices] = self.spec.default_joint_pos
        self.joint_stiffness_hw[self._policy_joint_hw_indices] = self.spec.joint_stiffness
        self.joint_damping_hw[self._policy_joint_hw_indices] = self.spec.joint_damping

        self.lowstate_received = False

        self.actions = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.obs_history: dict[str, np.ndarray] = {}

        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowstate_sub = self.create_subscription(
            LowState, "/lowstate", self.low_state_handler, 10
        )
        self._setup_keyboard()
        self.timer = self.create_timer(self.control_dt, self.control)

        self.get_logger().info(
            "Loaded G1 CLAMP2 policy: "
            f"{args.onnx_path} | {self._reference_description(args)} | "
            f"command={self.spec.motion_command_class} | "
            f"control_dt={self.control_dt:.4f}s | obs_dim={self.spec.observation_dim} | "
            f"action_dim={self.spec.action_dim}"
        )
        if self._keyboard_enabled:
            self.get_logger().info(self._keyboard_help())
        else:
            self.get_logger().warn("Keyboard control is disabled because stdin is not a TTY.")

    def _init_reference(self, args: argparse.Namespace) -> None:
        self.motion_clip = MotionClip(args.motion_npz, spec=self.spec)
        self.play_motion = False
        self.transition: ReferenceTransition | None = None

    def _reference_description(self, args: argparse.Namespace) -> str:
        return f"motion: {args.motion_npz}"

    def _keyboard_help(self) -> str:
        return (
            "Starting from the fixed default reference. Press SPACE or ENTER "
            "to toggle motion playback, and `x` to disable the motors."
        )

    def _start_log_episode(
        self,
        name: str,
        *,
        phase_names: tuple[str, ...],
        metadata: dict[str, object],
    ) -> None:
        if self.episode_logger is None:
            return
        common_metadata = {
            "control_dt": float(self.control_dt),
            "phase_names": phase_names,
            "hardware_joint_names": HARDWARE_JOINT_NAMES,
            "policy_joint_names": self.spec.joint_names,
            "onnx_name": self.onnx_path.name,
            "onnx_path": self.onnx_path,
            "action_target_names": self.spec.action_target_names,
            "action_semantics": self.spec.action_semantics,
            "imu_accelerometer_available": self._imu_accelerometer_available(),
        }
        common_metadata.update(metadata)
        self.episode_logger.start_episode(name, common_metadata)

    def _finish_log_episode(
        self,
        *,
        completed: bool,
        suffix: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.episode_logger is None:
            return
        self.episode_logger.finish_episode(
            completed=completed,
            suffix=suffix,
            metadata=metadata,
        )

    def _record_log_sample(
        self,
        frame: MotionFrame,
        raw_action: np.ndarray,
        desired_q_hw: np.ndarray,
    ) -> None:
        if self.episode_logger is None or not self.episode_logger.active:
            return
        now_ns = self.get_clock().now().nanoseconds
        self.episode_logger.record(
            now_ns,
            {
                "phase_id": int(self._current_log_phase_id()),
                "joint_positions_hw": self.joint_positions_hw.copy(),
                "joint_velocities_hw": self.joint_velocities_hw.copy(),
                "joint_accelerations_hw": self.joint_accelerations_hw.copy(),
                "joint_torques_hw": self.joint_torques_hw.copy(),
                "motor_temperature_ch0_hw": self.motor_temperature_ch0_hw.copy(),
                "motor_temperature_ch1_hw": self.motor_temperature_ch1_hw.copy(),
                "motor_voltage_hw": self.motor_voltage_hw.copy(),
                "motor_state_flags_hw": self.motor_state_flags_hw.copy(),
                "motor_mode_hw": self.motor_mode_hw.copy(),
                "imu_quaternion": self._imu_vector("quaternion", 4),
                "imu_gyroscope": self._imu_vector("gyroscope", 3),
                "imu_accelerometer": self._imu_accelerometer_vector(),
                "reference_joint_pos": frame.joint_pos.copy(),
                "reference_joint_vel": frame.joint_vel.copy(),
                "reference_root_pos_w": frame.root_pos_w.copy(),
                "reference_root_quat_wxyz": frame.root_quat_wxyz.copy(),
                "reference_root_lin_vel_w": frame.root_lin_vel_w.copy(),
                "reference_root_ang_vel_w": frame.root_ang_vel_w.copy(),
                "raw_action": raw_action.copy(),
                "desired_q_hw": desired_q_hw.copy(),
                "motors_on": int(self.motors_on),
                "lowstate_mode_machine": int(self.mode_machine),
            },
        )

    def _current_log_phase_id(self) -> int:
        return -1

    def _imu_vector(self, attr_name: str, size: int) -> np.ndarray:
        value = getattr(self.imu, attr_name, None)
        if value is None:
            return np.zeros(size, dtype=np.float64)
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size < size:
            padded = np.zeros(size, dtype=np.float64)
            padded[: vector.size] = vector
            return padded
        return vector[:size].copy()

    def _imu_accelerometer_available(self) -> bool:
        return any(
            hasattr(self.imu, attr_name)
            for attr_name in ("accelerometer", "accelerometer_body", "accel")
        )

    def _imu_accelerometer_vector(self) -> np.ndarray:
        for attr_name in ("accelerometer", "accelerometer_body", "accel"):
            if hasattr(self.imu, attr_name):
                return self._imu_vector(attr_name, 3)
        return np.zeros(3, dtype=np.float64)

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

    def control(self) -> None:
        self._poll_keyboard()

        if not self.lowstate_received:
            self._log_waiting_for_lowstate()
            return

        frame = self._current_reference_frame()
        terms = self._term_values(frame)
        self._append_history(terms)
        obs = self._build_observation(terms)
        raw_action = self._get_raw_action(obs)
        desired_q_hw = self._desired_positions_from_action(raw_action, frame)
        self.actions = raw_action.astype(np.float32, copy=True)
        self._record_log_sample(frame, raw_action, desired_q_hw)

        low_cmd = LowCmd()
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = self.mode_machine
        self._fill_low_cmd(low_cmd, desired_q_hw)
        low_cmd.crc = Crc(low_cmd)
        self.lowcmd_pub.publish(low_cmd)

        self._advance_reference_state()

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
        processed = (
            raw_action.astype(np.float64) * self.spec.action_scale + self.spec.action_offset
        )
        desired_q_hw = self.default_joint_pos_hw.copy()
        desired_q_hw[self._policy_joint_hw_indices] = frame.joint_pos
        if self.spec.action_semantics == "residual_joint_position":
            desired_q_hw[self._target_hw_indices] = (
                frame.joint_pos[self._target_frame_joint_indices] + processed
            )
        elif self.spec.action_semantics == "joint_position":
            desired_q_hw[self._target_hw_indices] = processed
        else:
            raise NotImplementedError(
                f"Unsupported action semantics `{self.spec.action_semantics}`."
            )
        return desired_q_hw

    def _command_value(self, frame: MotionFrame) -> np.ndarray:
        anchor_lin_vel_b = quat_rotate_inverse(frame.root_quat_wxyz, frame.root_lin_vel_w)
        anchor_ang_vel_b = quat_rotate_inverse(frame.root_quat_wxyz, frame.root_ang_vel_w)
        roll, pitch, _ = _quat_roll_pitch_yaw(frame.root_quat_wxyz)
        return np.concatenate(
            (
                frame.joint_pos,
                frame.joint_vel,
                anchor_lin_vel_b[:2],
                anchor_ang_vel_b[2:3],
                frame.root_pos_w[2:3],
                np.asarray([roll, pitch], dtype=np.float64),
            )
        ).astype(np.float32)

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
        joint_pos = self.joint_positions_hw[self._policy_joint_hw_indices]
        joint_vel = self.joint_velocities_hw[self._policy_joint_hw_indices]

        return {
            "command": self._command_value(frame),
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

    def _current_observation_block(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            (
                terms["command"],
                terms["base_ang_vel"],
                terms["projected_gravity"],
                terms["joint_pos"],
                terms["joint_vel"],
                terms["actions"],
            )
        ).astype(np.float32)

    def _append_history(self, terms: dict[str, np.ndarray]) -> None:
        for term in self.spec.observation_terms:
            if term.history_length > 0:
                value = terms[term.name]
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
                continue

            if term.name == "history" and term.name not in terms:
                current = self._current_observation_block(terms)
                if term.flat_dim % current.shape[0] != 0:
                    raise ValueError(
                        "Cannot initialize CLAMP2 history term: "
                        f"history dim={term.flat_dim}, current block dim={current.shape[0]}."
                    )
                history_length = term.flat_dim // current.shape[0]
                if term.name not in self.obs_history:
                    self.obs_history[term.name] = np.repeat(
                        current[None, :], history_length, axis=0
                    ).astype(np.float32)
                else:
                    self.obs_history[term.name][:-1] = self.obs_history[term.name][1:]
                    self.obs_history[term.name][-1] = current

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

    def low_state_handler(self, msg: LowState) -> None:
        self.lowstate_received = True
        self.mode_machine = msg.mode_machine
        self.imu = msg.imu_state
        for idx in range(G1_NUM_MOTOR):
            motor_state = msg.motor_state[idx]
            self.joint_positions_hw[idx] = motor_state.q
            self.joint_velocities_hw[idx] = motor_state.dq
            self.joint_accelerations_hw[idx] = getattr(motor_state, "ddq", 0.0)
            self.joint_torques_hw[idx] = getattr(motor_state, "tau_est", 0.0)
            temperature = getattr(motor_state, "temperature", ())
            if len(temperature) > 0:
                self.motor_temperature_ch0_hw[idx] = temperature[0]
            if len(temperature) > 1:
                self.motor_temperature_ch1_hw[idx] = temperature[1]
            self.motor_voltage_hw[idx] = getattr(motor_state, "vol", 0.0)
            self.motor_state_flags_hw[idx] = getattr(motor_state, "motorstate", 0)
            self.motor_mode_hw[idx] = getattr(motor_state, "mode", 0)

    def _current_reference_frame(self) -> MotionFrame:
        if self.transition is not None:
            return self.transition.frame()
        if self.play_motion:
            return self.motion_clip.sample(self.motion_time_s)
        return self.idle_reference_frame

    def _advance_reference_state(self) -> None:
        if self.transition is not None:
            if self.transition.advance(self.control_dt):
                target_mode = self.transition.target_mode
                self.transition = None
                self.play_motion = target_mode == "motion"
                self.motion_time_s = 0.0
                if self.play_motion:
                    self.get_logger().info("Motion playback started.")
                else:
                    self.get_logger().info("Returned to fixed default reference.")
            return

        if self.play_motion:
            self.motion_time_s += self.control_dt
            if self.motion_time_s >= self.motion_clip.length_s:
                self._begin_reference_transition(
                    self.motion_clip.sample(self.motion_clip.length_s),
                    self.idle_reference_frame,
                    target_mode="idle",
                    message="Motion clip completed. Transitioning back to fixed default reference.",
                )

    def _toggle_motion_reference(self) -> None:
        current_frame = self._current_reference_frame()
        if self.play_motion or (
            self.transition is not None and self.transition.target_mode == "motion"
        ):
            self._begin_reference_transition(
                current_frame,
                self.idle_reference_frame,
                target_mode="idle",
                message="Transitioning back to fixed default reference.",
            )
            return

        self.motion_time_s = 0.0
        self._begin_reference_transition(
            current_frame,
            self.motion_clip.sample(0.0),
            target_mode="motion",
            message="Transitioning into motion playback.",
        )

    def _begin_reference_transition(
        self,
        start_frame: MotionFrame,
        end_frame: MotionFrame,
        *,
        target_mode: str,
        message: str,
    ) -> None:
        self.play_motion = False
        self.transition = ReferenceTransition(
            start_frame=start_frame.copy(),
            end_frame=end_frame.copy(),
            target_mode=target_mode,
            duration_s=REFERENCE_TRANSITION_DURATION_S,
        )
        self.get_logger().info(message)

    def _log_waiting_for_lowstate(self) -> None:
        if self._waiting_for_lowstate_logged:
            return
        self._waiting_for_lowstate_logged = True
        self.get_logger().warn("Waiting for `/lowstate` before sending commands.")

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
            if key in {" ", "\n", "\r"}:
                self._toggle_motion_reference()
                continue
            if key in {"x", "X"}:
                self.motors_on = 0
                self.get_logger().warn("Motor disable requested from keyboard.")

    def destroy_node(self) -> bool:
        self._finish_log_episode(completed=False, suffix="interrupted")
        self._restore_keyboard()
        return super().destroy_node()

    def _hardware_indices(self, joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.asarray(
            [self._hardware_joint_to_index[name] for name in joint_names],
            dtype=np.int32,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a G1 CLAMP2 ONNX policy through HURo."
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="ONNX policy with embedded deployment metadata.",
    )
    parser.add_argument(
        "--motion-npz",
        "--motion-file",
        dest="motion_npz",
        type=Path,
        required=True,
        help="Reference motion clip in CLAMP NPZ format.",
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
        node = G1Clamp2Runner(cli_args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
