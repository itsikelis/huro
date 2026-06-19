#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def _load_matplotlib(show: bool):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "huro_matplotlib_cache"),
    )
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _as_strings(array: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(array).reshape(-1).tolist()]


def _scalar_string(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "log"


def _discover_npz(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.npz")))
        elif path.is_file() and path.suffix == ".npz":
            files.append(path)
    return sorted(dict.fromkeys(path.resolve() for path in files))


def _reference_to_hardware_order(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    reference_q = np.asarray(data["reference_joint_pos"], dtype=np.float64)
    reference_dq = np.asarray(data["reference_joint_vel"], dtype=np.float64)
    hardware_names = _as_strings(data["meta_hardware_joint_names"])
    policy_names = _as_strings(data["meta_policy_joint_names"])
    policy_index = {name: idx for idx, name in enumerate(policy_names)}

    reference_q_hw = np.full((reference_q.shape[0], len(hardware_names)), np.nan)
    reference_dq_hw = np.full((reference_dq.shape[0], len(hardware_names)), np.nan)
    for hw_idx, joint_name in enumerate(hardware_names):
        if joint_name not in policy_index:
            continue
        policy_idx = policy_index[joint_name]
        reference_q_hw[:, hw_idx] = reference_q[:, policy_idx]
        reference_dq_hw[:, hw_idx] = reference_dq[:, policy_idx]
    return reference_q_hw, reference_dq_hw


def _parse_joint_selection(selection: str | None, joint_names: list[str]) -> list[int]:
    if not selection:
        return list(range(len(joint_names)))
    selected: list[int] = []
    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    for raw_token in re.split(r"[,\s]+", selection.strip()):
        if not raw_token:
            continue
        if raw_token.isdigit():
            idx = int(raw_token)
            if idx < 0 or idx >= len(joint_names):
                raise ValueError(f"Joint index {idx} outside 0..{len(joint_names) - 1}.")
            selected.append(idx)
            continue
        if raw_token not in name_to_index:
            matches = [idx for idx, name in enumerate(joint_names) if raw_token in name]
            if len(matches) == 1:
                selected.append(matches[0])
                continue
            raise ValueError(f"Unknown or ambiguous joint selector `{raw_token}`.")
        selected.append(name_to_index[raw_token])
    return sorted(dict.fromkeys(selected))


def _phase_segments(data: np.lib.npyio.NpzFile) -> list[tuple[int, int, int]]:
    if "phase_id" not in data:
        return []
    phase_id = np.asarray(data["phase_id"], dtype=np.int64)
    if phase_id.size == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(phase_id[0])
    for idx in range(1, phase_id.size):
        value = int(phase_id[idx])
        if value == current:
            continue
        segments.append((start, idx - 1, current))
        start = idx
        current = value
    segments.append((start, phase_id.size - 1, current))
    return segments


def _shade_phases(ax, time_s: np.ndarray, data: np.lib.npyio.NpzFile) -> None:
    phase_names = _as_strings(data["meta_phase_names"]) if "meta_phase_names" in data else []
    colors = {
        0: "#e8f1ff",
        1: "#edf8ed",
        2: "#fff2df",
    }
    for start, end, phase in _phase_segments(data):
        if phase < 0 or start >= len(time_s) or end >= len(time_s):
            continue
        label = phase_names[phase] if phase < len(phase_names) else f"phase {phase}"
        ax.axvspan(time_s[start], time_s[end], color=colors.get(phase, "#eeeeee"), alpha=0.28)
        if start == end:
            continue
        y_top = ax.get_ylim()[1]
        ax.text(
            0.5 * (time_s[start] + time_s[end]),
            y_top,
            label,
            fontsize=7,
            ha="center",
            va="top",
            alpha=0.6,
        )


def _paginate(indices: list[int], page_size: int) -> list[list[int]]:
    return [indices[start : start + page_size] for start in range(0, len(indices), page_size)]


def _plot_joint_pages(
    *,
    plt,
    output_dir: Path,
    stem: str,
    title: str,
    suffix: str,
    time_s: np.ndarray,
    joint_names: list[str],
    joint_indices: list[int],
    series: list[tuple[str, np.ndarray, str]],
    data: np.lib.npyio.NpzFile,
    max_joints_per_figure: int,
    show: bool,
) -> None:
    for page_idx, page_joint_indices in enumerate(
        _paginate(joint_indices, max_joints_per_figure),
        start=1,
    ):
        ncols = 3
        nrows = int(math.ceil(len(page_joint_indices) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False)
        for ax, joint_idx in zip(axes.flat, page_joint_indices):
            for label, values, style in series:
                ax.plot(time_s, values[:, joint_idx], style, linewidth=1.1, label=label)
            _shade_phases(ax, time_s, data)
            ax.set_title(f"{joint_idx}: {joint_names[joint_idx]}", fontsize=9)
            ax.set_xlabel("time [s]")
            ax.grid(True, alpha=0.25)
        for ax in axes.flat[len(page_joint_indices) :]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{title} - {suffix} page {page_idx}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 0.98, 0.95))
        output_path = output_dir / f"{stem}_{suffix}_page{page_idx:02d}.png"
        fig.savefig(output_path, dpi=160)
        if show:
            plt.show()
        plt.close(fig)


def _plot_rmse(
    *,
    plt,
    output_dir: Path,
    stem: str,
    time_s: np.ndarray,
    joint_names: list[str],
    joint_indices: list[int],
    real_q: np.ndarray,
    reference_q_hw: np.ndarray,
    desired_q_hw: np.ndarray,
    show: bool,
) -> None:
    del time_s
    ref_rmse = np.sqrt(np.nanmean((real_q - reference_q_hw) ** 2, axis=0))
    desired_rmse = np.sqrt(np.nanmean((real_q - desired_q_hw) ** 2, axis=0))
    ranked = sorted(joint_indices, key=lambda idx: ref_rmse[idx], reverse=True)
    top = ranked[: min(12, len(ranked))]

    print(f"\n{stem}: largest q-real vs reference RMSE")
    for idx in top:
        print(
            f"  {idx:2d} {joint_names[idx]:28s} "
            f"ref={ref_rmse[idx]:8.4f} desired={desired_rmse[idx]:8.4f}"
        )

    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.18, ref_rmse[top], width=0.36, label="real - reference")
    ax.bar(x + 0.18, desired_rmse[top], width=0.36, label="real - desired")
    ax.set_xticks(x)
    ax.set_xticklabels([joint_names[idx] for idx in top], rotation=35, ha="right")
    ax.set_ylabel("RMSE [rad]")
    ax.set_title(f"{stem} joint position RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_joint_position_rmse.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def _plot_root_reference(
    *,
    plt,
    output_dir: Path,
    stem: str,
    time_s: np.ndarray,
    data: np.lib.npyio.NpzFile,
    show: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    items = (
        ("reference_root_pos_w", ("x", "y", "z")),
        ("reference_root_lin_vel_w", ("vx", "vy", "vz")),
        ("reference_root_ang_vel_w", ("wx", "wy", "wz")),
    )
    for ax, (key, labels) in zip(axes, items):
        values = np.asarray(data[key], dtype=np.float64)
        for dim, label in enumerate(labels):
            ax.plot(time_s, values[:, dim], linewidth=1.1, label=label)
        _shade_phases(ax, time_s, data)
        ax.set_title(key)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{stem} root reference")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / f"{stem}_root_reference.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_log(path: Path, args: argparse.Namespace) -> None:
    plt = _load_matplotlib(args.show)
    data = np.load(path, allow_pickle=False)
    required = {
        "time_s",
        "joint_positions_hw",
        "joint_velocities_hw",
        "reference_joint_pos",
        "reference_joint_vel",
        "desired_q_hw",
        "raw_action",
        "meta_hardware_joint_names",
        "meta_policy_joint_names",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")

    stem = _safe_stem(path)
    output_dir = args.output_dir if args.output_dir is not None else path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    time_s = np.asarray(data["time_s"], dtype=np.float64)
    joint_names = _as_strings(data["meta_hardware_joint_names"])
    joint_indices = _parse_joint_selection(args.joints, joint_names)
    real_q = np.asarray(data["joint_positions_hw"], dtype=np.float64)
    real_dq = np.asarray(data["joint_velocities_hw"], dtype=np.float64)
    desired_q_hw = np.asarray(data["desired_q_hw"], dtype=np.float64)
    reference_q_hw, reference_dq_hw = _reference_to_hardware_order(data)

    title_parts = [
        stem,
        _scalar_string(data, "meta_episode_type"),
        _scalar_string(data, "meta_motion_name", _scalar_string(data, "meta_pose_name")),
        _scalar_string(data, "meta_onnx_name"),
    ]
    title = " | ".join(part for part in title_parts if part)

    _plot_joint_pages(
        plt=plt,
        output_dir=output_dir,
        stem=stem,
        title=title,
        suffix="joint_positions",
        time_s=time_s,
        joint_names=joint_names,
        joint_indices=joint_indices,
        series=[
            ("real q", real_q, "-"),
            ("reference q", reference_q_hw, "--"),
            ("desired q", desired_q_hw, ":"),
        ],
        data=data,
        max_joints_per_figure=args.max_joints_per_figure,
        show=args.show,
    )

    if not args.no_velocity:
        _plot_joint_pages(
            plt=plt,
            output_dir=output_dir,
            stem=stem,
            title=title,
            suffix="joint_velocities",
            time_s=time_s,
            joint_names=joint_names,
            joint_indices=joint_indices,
            series=[
                ("real dq", real_dq, "-"),
                ("reference dq", reference_dq_hw, "--"),
            ],
            data=data,
            max_joints_per_figure=args.max_joints_per_figure,
            show=args.show,
        )

    if "joint_torques_hw" in data and not args.no_torque:
        joint_torques = np.asarray(data["joint_torques_hw"], dtype=np.float64)
        _plot_joint_pages(
            plt=plt,
            output_dir=output_dir,
            stem=stem,
            title=title,
            suffix="joint_torques",
            time_s=time_s,
            joint_names=joint_names,
            joint_indices=joint_indices,
            series=[
                ("tau_est", joint_torques, "-"),
            ],
            data=data,
            max_joints_per_figure=args.max_joints_per_figure,
            show=args.show,
        )

    if "motor_temperature_ch0_hw" in data and not args.no_temperature:
        temperature_series = [
            ("temperature ch0", np.asarray(data["motor_temperature_ch0_hw"], dtype=np.float64), "-"),
        ]
        if "motor_temperature_ch1_hw" in data:
            temperature_series.append(
                (
                    "temperature ch1",
                    np.asarray(data["motor_temperature_ch1_hw"], dtype=np.float64),
                    "--",
                )
            )
        _plot_joint_pages(
            plt=plt,
            output_dir=output_dir,
            stem=stem,
            title=title,
            suffix="motor_temperature",
            time_s=time_s,
            joint_names=joint_names,
            joint_indices=joint_indices,
            series=temperature_series,
            data=data,
            max_joints_per_figure=args.max_joints_per_figure,
            show=args.show,
        )

    if "motor_voltage_hw" in data and not args.no_voltage:
        _plot_joint_pages(
            plt=plt,
            output_dir=output_dir,
            stem=stem,
            title=title,
            suffix="motor_voltage",
            time_s=time_s,
            joint_names=joint_names,
            joint_indices=joint_indices,
            series=[
                ("voltage", np.asarray(data["motor_voltage_hw"], dtype=np.float64), "-"),
            ],
            data=data,
            max_joints_per_figure=args.max_joints_per_figure,
            show=args.show,
        )

    _plot_rmse(
        plt=plt,
        output_dir=output_dir,
        stem=stem,
        time_s=time_s,
        joint_names=joint_names,
        joint_indices=joint_indices,
        real_q=real_q,
        reference_q_hw=reference_q_hw,
        desired_q_hw=desired_q_hw,
        show=args.show,
    )

    if not args.no_root:
        _plot_root_reference(
            plt=plt,
            output_dir=output_dir,
            stem=stem,
            time_s=time_s,
            data=data,
            show=args.show,
        )

    print(f"Saved plots for {path} to {output_dir}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot HURo deploy NPZ logs against their motion/stance references."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="NPZ files or directories containing NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for PNG outputs. Defaults to a `plots` folder next to each NPZ.",
    )
    parser.add_argument(
        "--joints",
        help=(
            "Comma/space-separated joint names, substrings, or hardware indices. "
            "Defaults to all joints."
        ),
    )
    parser.add_argument(
        "--max-joints-per-figure",
        type=int,
        default=12,
        help="Number of joint subplots per PNG page.",
    )
    parser.add_argument("--no-velocity", action="store_true", help="Skip velocity plots.")
    parser.add_argument("--no-torque", action="store_true", help="Skip torque plots.")
    parser.add_argument("--no-temperature", action="store_true", help="Skip temperature plots.")
    parser.add_argument("--no-voltage", action="store_true", help="Skip voltage plots.")
    parser.add_argument("--no-root", action="store_true", help="Skip root reference plots.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.max_joints_per_figure <= 0:
        raise ValueError("--max-joints-per-figure must be positive.")

    files = _discover_npz(args.paths)
    if not files:
        raise FileNotFoundError("No .npz logs found.")
    for path in files:
        plot_log(path, args)


if __name__ == "__main__":
    main()
