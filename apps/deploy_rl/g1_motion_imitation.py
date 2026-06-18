#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.utilities import remove_ros_args

from g1_clamp2 import G1Clamp2Runner, MotionClip

ANSI_STYLES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
}


def _colors_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _style(text: str, *styles: str, enabled: bool) -> str:
    if not enabled:
        return text
    prefix = "".join(ANSI_STYLES[name] for name in styles)
    return f"{prefix}{text}{ANSI_STYLES['reset']}"


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


def _discover_motion_paths(motions_dir: Path) -> list[Path]:
    return sorted(
        (path for path in motions_dir.rglob("*.npz") if path.is_file()),
        key=lambda path: path.relative_to(motions_dir).as_posix().lower(),
    )


def _resolve_initial_motion(
    initial_motion: str | None,
    *,
    motions_dir: Path,
    motion_paths: list[Path],
) -> Path:
    if not motion_paths:
        raise FileNotFoundError(f"No .npz motion files found under {motions_dir}.")
    if initial_motion is None:
        return motion_paths[0]

    initial_path = Path(initial_motion).expanduser()
    if initial_path.exists():
        return initial_path.resolve()

    matches = [
        path
        for path in motion_paths
        if path.name == initial_motion
        or path.stem == initial_motion
        or path.relative_to(motions_dir).as_posix() == initial_motion
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(path.relative_to(motions_dir).as_posix() for path in matches)
        raise ValueError(f"Initial motion `{initial_motion}` is ambiguous: {names}")
    raise FileNotFoundError(
        f"Initial motion `{initial_motion}` was not found as a file or under {motions_dir}."
    )


class G1MotionImitationRunner(G1Clamp2Runner):
    """Run a G1 CLAMP2 policy with interactive selection from a motion library."""

    def __init__(self, args: argparse.Namespace):
        self.motions_dir = args.motions_dir
        self.motion_paths = _discover_motion_paths(self.motions_dir)
        args.motion_npz = _resolve_initial_motion(
            args.initial_motion,
            motions_dir=self.motions_dir,
            motion_paths=self.motion_paths,
        )
        if args.motion_npz not in self.motion_paths:
            raise ValueError(
                f"Initial motion {args.motion_npz} is not inside {self.motions_dir}."
            )
        self.selected_motion_path = args.motion_npz
        self.selected_motion_index = self.motion_paths.index(args.motion_npz)
        self._command_buffer = ""
        self._colors_enabled = _colors_enabled(args.color)
        args.node_name = "g1_motion_imitation_runner"

        super().__init__(args)

        self.get_logger().info(
            self._paint(
                f"Loaded {len(self.motion_paths)} motions from {self.motions_dir}.",
                "cyan",
            )
        )
        self._print_motion_menu()
        self.get_logger().info(
            self._paint("Controls: ", "bold")
            + "type "
            + self._paint("motion number + ENTER", "yellow", "bold")
            + " to play, "
            + self._paint("SPACE", "yellow")
            + " or empty "
            + self._paint("ENTER", "yellow")
            + " to replay, "
            + self._paint("l", "yellow")
            + " to list, "
            + self._paint("r", "yellow")
            + " to rescan, "
            + self._paint("x", "yellow")
            + " to disable motors."
        )

    def _paint(self, text: str, *styles: str) -> str:
        return _style(text, *styles, enabled=self._colors_enabled)

    def _print_motion_menu(self) -> None:
        lines = [self._paint("Available motions:", "cyan", "bold")]
        for idx, path in enumerate(self.motion_paths, start=1):
            selected = path == self.selected_motion_path
            marker = self._paint("*", "green", "bold") if selected else " "
            number = self._paint(f"{idx:2d}", "yellow", "bold" if selected else "dim")
            rel_path = path.relative_to(self.motions_dir).as_posix()
            if selected:
                rel_path = self._paint(rel_path, "green", "bold")
            lines.append(f"  {marker} {number}. {rel_path}")
        self.get_logger().info("\n".join(lines))

    def _reload_motion_library(self) -> None:
        previous = self.selected_motion_path
        self.motion_paths = _discover_motion_paths(self.motions_dir)
        if not self.motion_paths:
            self.get_logger().error(f"No .npz motion files found under {self.motions_dir}.")
            return
        if previous not in self.motion_paths:
            previous = self.motion_paths[0]
            self._load_selected_motion(previous)
        else:
            self.selected_motion_index = self.motion_paths.index(previous)
        self._print_motion_menu()

    def _load_selected_motion(self, motion_path: Path) -> None:
        self.motion_clip = MotionClip(motion_path, spec=self.spec)
        self.selected_motion_path = motion_path
        self.selected_motion_index = self.motion_paths.index(motion_path)
        self.motion_time_s = 0.0
        rel_path = motion_path.relative_to(self.motions_dir).as_posix()
        self.get_logger().info(
            self._paint(
                f"Selected motion {self.selected_motion_index + 1}:",
                "green",
                "bold",
            )
            + f" {self._paint(rel_path, 'green')} "
            + self._paint(
                f"({self.motion_clip.length_s:.2f}s, "
                f"{self.motion_clip.num_frames} frames).",
                "dim",
            )
        )

    def _select_motion_number(self, text: str) -> None:
        try:
            selected = int(text)
        except ValueError:
            self.get_logger().warn(f"Ignoring unknown motion command `{text}`.")
            return

        if selected < 1 or selected > len(self.motion_paths):
            self.get_logger().warn(
                f"Motion number {selected} is outside 1..{len(self.motion_paths)}."
            )
            return

        motion_path = self.motion_paths[selected - 1]
        current_frame = self._current_reference_frame()
        if self.play_motion or self.transition is not None:
            self._begin_reference_transition(
                current_frame,
                self.idle_reference_frame,
                target_mode="idle",
                message=self._paint(
                    "Stopping current motion before switching selection.", "magenta"
                ),
            )
            self._load_selected_motion(motion_path)
            return

        self._load_selected_motion(motion_path)
        self._begin_reference_transition(
            current_frame,
            self.motion_clip.sample(0.0),
            target_mode="motion",
            message=self._paint("Transitioning into selected motion playback.", "magenta"),
        )

    def _poll_keyboard(self) -> None:
        if not self._keyboard_enabled or self._stdin_fd is None:
            return
        while True:
            ready, _, _ = self._select_stdin()
            if not ready:
                break
            key = sys.stdin.read(1)
            if key in {" ", "\n", "\r"}:
                if self._command_buffer:
                    command = self._command_buffer
                    self._command_buffer = ""
                    self._select_motion_number(command)
                else:
                    self._toggle_motion_reference()
                continue
            if key in {"\x7f", "\b"}:
                self._command_buffer = self._command_buffer[:-1]
                continue
            if key in {"l", "L"} and not self._command_buffer:
                self._print_motion_menu()
                continue
            if key in {"r", "R"} and not self._command_buffer:
                self._reload_motion_library()
                continue
            if key in {"x", "X"} and not self._command_buffer:
                self.motors_on = 0
                self.get_logger().warn("Motor disable requested from keyboard.")
                continue
            if key.isdigit():
                self._command_buffer += key
                continue
            if key in {"q", "Q"} and not self._command_buffer:
                self.get_logger().info("Quit requested from keyboard.")
                rclpy.shutdown()
                return
            self.get_logger().warn(f"Ignoring key `{key!r}`.")

    def _select_stdin(self):
        import select

        return select.select([self._stdin_fd], [], [], 0.0)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a G1 CLAMP2 ONNX policy and interactively choose motions from "
            "the motions folder."
        )
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=_default_resource_path("policies", "g1", "g1_clamp2.onnx"),
        help="ONNX policy with embedded deployment metadata.",
    )
    parser.add_argument(
        "--motions-dir",
        type=Path,
        default=_default_resource_path("motions"),
        help="Folder containing CLAMP NPZ motion clips.",
    )
    parser.add_argument(
        "--initial-motion",
        help=(
            "Optional initial motion file, filename, stem, or path relative to "
            "--motions-dir. The robot still starts from the default stance."
        ),
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize terminal log output.",
    )
    return parser


def main(args=None) -> None:
    parser = _build_argparser()
    cli_args = parser.parse_args(remove_ros_args(args=sys.argv if args is None else args)[1:])
    cli_args.onnx_path = cli_args.onnx_path.expanduser().resolve()
    cli_args.motions_dir = cli_args.motions_dir.expanduser().resolve()

    rclpy.init(args=args)
    node = None
    try:
        node = G1MotionImitationRunner(cli_args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
