#!/usr/bin/env python3

from __future__ import annotations

import argparse
import select
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.utilities import remove_ros_args

from g1_clamp2 import (
    IDLE_REFERENCE_HEIGHT_M,
    G1Clamp2Runner,
    MotionFrame,
)


def _package_resource_path(*parts: str) -> Path | None:
    try:
        return Path(get_package_share_directory("huro"), *parts)
    except PackageNotFoundError:
        return None


def _source_resource_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / Path(*parts)


def _default_resource_path(*parts: str) -> Path:
    installed = _package_resource_path("resources", *parts)
    if installed is not None and installed.exists():
        return installed
    return _source_resource_path(*parts)


def _default_log_path(*parts: str) -> Path:
    docker_source_resources = Path("/huro_ws/src/huro/resources")
    if docker_source_resources.exists():
        return docker_source_resources / "log" / Path(*parts)

    local_source_resources = Path(__file__).resolve().parents[2] / "resources"
    if local_source_resources.exists():
        return local_source_resources / "log" / Path(*parts)

    installed = _package_resource_path("resources", "log", *parts)
    if installed is not None:
        return installed
    return local_source_resources / "log" / Path(*parts)


@dataclass(frozen=True)
class StanceSpec:
    label: str
    description: str
    joint_overrides: dict[str, float]


PREDEFINED_STANCES = {
    "bent_forearms": StanceSpec(
        label="Bent forearms",
        description=(
            "Default standing with bent arms."
        ),
        joint_overrides={
            "left_elbow_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
    ),
    "arms_forward": StanceSpec(
        label="Straight arms forward",
        description=(
            "Default standing with arms forward."
        ),
        joint_overrides={
            "left_shoulder_pitch_joint": -0.90,
            "left_shoulder_roll_joint": 0.20,
            "left_elbow_joint": 0.90,
            "right_shoulder_pitch_joint": -0.90,
            "right_shoulder_roll_joint": -0.20,
            "right_elbow_joint": 0.90,
        },
    ),
}


def _apply_joint_overrides(
    joint_names: tuple[str, ...],
    base_joint_pos: np.ndarray,
    overrides: dict[str, float],
) -> np.ndarray:
    joint_pos = np.asarray(base_joint_pos, dtype=np.float64).copy()
    joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, value in overrides.items():
        if joint_name not in joint_name_to_index:
            raise ValueError(f"Policy does not expose required joint `{joint_name}`.")
        joint_pos[joint_name_to_index[joint_name]] = value
    return joint_pos


class G1PredefinedStanceRunner(G1Clamp2Runner):
    """Run a G1 CLAMP2 policy against a hardcoded stationary reference pose."""

    def __init__(self, args: argparse.Namespace):
        self.selected_stance_name = args.pose
        self.hold_reference_frame: MotionFrame | None = None
        self.stance_frames: dict[str, MotionFrame] = {}
        args.node_name = "g1_predefined_stance_runner"
        super().__init__(args)
        self._print_stance_menu()

    def _init_reference(self, args: argparse.Namespace) -> None:
        self.play_motion = False
        self.transition = None
        self.stance_frames = {}
        for name, stance in PREDEFINED_STANCES.items():
            joint_pos = _apply_joint_overrides(
                self.spec.joint_names,
                self.spec.default_joint_pos,
                stance.joint_overrides,
            )
            self.stance_frames[name] = MotionFrame.stationary(
                joint_pos,
                height_m=IDLE_REFERENCE_HEIGHT_M,
            )

        self.hold_reference_frame = self.idle_reference_frame
        self._begin_reference_transition(
            self.idle_reference_frame,
            self.stance_frames[self.selected_stance_name],
            target_mode="stance",
            message=(
                f"Transitioning from default stance to "
                f"`{self.selected_stance_name}`."
            ),
        )

    def _reference_description(self, args: argparse.Namespace) -> str:
        return f"predefined pose: {args.pose}"

    def _keyboard_help(self) -> str:
        return (
            "Press 1/2 to select a stance, SPACE or ENTER to toggle default/selected "
            "stance, `l` to list poses, and `x` to disable the motors."
        )

    def _print_stance_menu(self) -> None:
        lines = ["Available predefined stances:"]
        for idx, (name, stance) in enumerate(PREDEFINED_STANCES.items(), start=1):
            marker = "*" if name == self.selected_stance_name else " "
            lines.append(f"  {marker} {idx}. {name}: {stance.label}")
            lines.append(f"       {stance.description}")
        self.get_logger().info("\n".join(lines))

    def _begin_reference_transition(
        self,
        start_frame,
        end_frame,
        *,
        target_mode: str,
        message: str,
    ) -> None:
        if target_mode == "stance":
            self._start_stance_log_episode()
        super()._begin_reference_transition(
            start_frame,
            end_frame,
            target_mode=target_mode,
            message=message,
        )

    def _start_stance_log_episode(self) -> None:
        stance = PREDEFINED_STANCES[self.selected_stance_name]
        self._start_log_episode(
            self.selected_stance_name,
            phase_names=("transition_in", "hold", "transition_out"),
            metadata={
                "episode_type": "predefined_stance",
                "pose_name": self.selected_stance_name,
                "pose_label": stance.label,
                "pose_description": stance.description,
            },
        )

    def _current_reference_frame(self) -> MotionFrame:
        if self.transition is not None:
            return self.transition.frame()
        if self.hold_reference_frame is None:
            return self.idle_reference_frame
        return self.hold_reference_frame

    def _advance_reference_state(self) -> None:
        previous_transition = self.transition
        if self.transition is None:
            return
        if not self.transition.advance(self.control_dt):
            return

        target_mode = self.transition.target_mode
        self.hold_reference_frame = self.transition.end_frame.copy()
        self.transition = None
        if target_mode == "stance":
            self.get_logger().info(f"Holding `{self.selected_stance_name}` stance.")
        else:
            self.get_logger().info("Returned to fixed default reference.")
        if previous_transition is not None and previous_transition.target_mode == "idle":
            self._finish_log_episode(completed=True)

    def _current_log_phase_id(self) -> int:
        if self.transition is not None:
            return 0 if self.transition.target_mode == "stance" else 2
        if self._is_targeting_or_holding_stance():
            return 1
        return -1

    def _toggle_stance_reference(self) -> None:
        current_frame = self._current_reference_frame()
        if self._is_targeting_or_holding_stance():
            self._begin_reference_transition(
                current_frame,
                self.idle_reference_frame,
                target_mode="idle",
                message="Transitioning back to fixed default reference.",
            )
            return

        self._begin_reference_transition(
            current_frame,
            self.stance_frames[self.selected_stance_name],
            target_mode="stance",
            message=f"Transitioning into `{self.selected_stance_name}` stance.",
        )

    def _is_targeting_or_holding_stance(self) -> bool:
        if self.transition is not None:
            return self.transition.target_mode == "stance"
        if self.hold_reference_frame is None:
            return False
        return not np.allclose(
            self.hold_reference_frame.joint_pos,
            self.idle_reference_frame.joint_pos,
            atol=1.0e-6,
        )

    def _select_stance_index(self, index: int) -> None:
        stance_names = tuple(PREDEFINED_STANCES)
        if index < 1 or index > len(stance_names):
            self.get_logger().warn(f"Stance number {index} is outside 1..{len(stance_names)}.")
            return
        self.selected_stance_name = stance_names[index - 1]
        self.get_logger().info(f"Selected `{self.selected_stance_name}` stance.")
        self._begin_reference_transition(
            self._current_reference_frame(),
            self.stance_frames[self.selected_stance_name],
            target_mode="stance",
            message=f"Transitioning into `{self.selected_stance_name}` stance.",
        )

    def _poll_keyboard(self) -> None:
        if not self._keyboard_enabled or self._stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self._stdin_fd], [], [], 0.0)
            if not ready:
                break
            key = sys.stdin.read(1)
            if key in {" ", "\n", "\r"}:
                self._toggle_stance_reference()
                continue
            if key in {"1", "2"}:
                self._select_stance_index(int(key))
                continue
            if key in {"l", "L"}:
                self._print_stance_menu()
                continue
            if key in {"x", "X"}:
                self.motors_on = 0
                self.get_logger().warn("Motor disable requested from keyboard.")
                continue
            if key in {"q", "Q"}:
                self.get_logger().info("Quit requested from keyboard.")
                rclpy.shutdown()
                return


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a G1 CLAMP2 ONNX policy against a hardcoded stance pose."
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=_default_resource_path("policies", "g1", "g1_clamp2.onnx"),
        help="ONNX policy with embedded deployment metadata.",
    )
    parser.add_argument(
        "--pose",
        choices=tuple(PREDEFINED_STANCES),
        default="bent_forearms",
        help="Predefined stance reference to hold.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_default_log_path("g1_predefined_stance"),
        help="Directory where per-stance NPZ log episodes are saved.",
    )
    return parser


def main(args=None) -> None:
    parser = _build_argparser()
    cli_args = parser.parse_args(remove_ros_args(args=sys.argv if args is None else args)[1:])
    cli_args.onnx_path = cli_args.onnx_path.expanduser().resolve()
    cli_args.log_dir = cli_args.log_dir.expanduser().resolve()

    rclpy.init(args=args)
    node = None
    try:
        node = G1PredefinedStanceRunner(cli_args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
