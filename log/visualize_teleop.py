#!/usr/bin/env python3
"""Quick textual visualization of teleop target poses per frame index."""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJ = SCRIPT_DIR.parent / "demo" / "teleop_20251107-115048.pkl"


def load_episode(path: Path, index: int) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not payload:
        raise ValueError(f"No episodes stored in {path}")
    try:
        episode = payload[index]
    except IndexError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Trajectory index {index} out of range (available: {len(payload)})") from exc
    if not isinstance(episode, dict):
        raise TypeError("Unexpected trajectory structure; expected dict episodes")
    return episode


def extract_pose_series(episode: dict[str, Any], side: str) -> np.ndarray:
    bucket = episode.get(f"grippers_{side}")
    if not bucket:
        raise ValueError(f"No target data recorded for {side}")
    tcp_pose = np.asarray(bucket[0]["tcp_pose"], dtype=float)
    if tcp_pose.ndim != 2 or tcp_pose.shape[1] != 6:
        raise ValueError(f"Unexpected tcp_pose shape for {side}: {tcp_pose.shape}")
    return tcp_pose


def load_targets(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode = load_episode(path, index)
    left = extract_pose_series(episode, "left")
    right = extract_pose_series(episode, "right")
    timestamps = np.asarray(episode.get("target_timestamps", []), dtype=float)
    sample_count = min(len(left), len(right), len(timestamps))
    if sample_count < 1:
        raise ValueError("Trajectory does not contain any synchronized samples")
    return left[:sample_count], right[:sample_count], timestamps[:sample_count]


def fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{val:.4f}" for val in vec) + "]"


def fmt_ts(ts: float) -> str:
    human = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
    return f"{ts:.3f}s ({human})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-frame teleop target poses for the left and right hands.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJ,
        help=f"Trajectory pickle to inspect (default: {DEFAULT_TRAJ.name}).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Episode index inside the trajectory pickle.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index to print (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last frame index to print (exclusive). Defaults to the final sample.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride between printed frames (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left, right, timestamps = load_targets(args.trajectory, args.index)
    total = len(timestamps)

    start = max(0, int(args.start))
    if start >= total:
        raise ValueError(f"--start {start} exceeds available samples ({total})")

    end = total if args.end is None else min(total, int(args.end))
    if end <= start:
        raise ValueError(f"--end {end} must be greater than --start {start}")

    step = int(args.step)
    if step <= 0:
        raise ValueError("--step must be positive")

    print(
        f"[visualize_teleop] frames {start}..{end - 1} (step={step}, total={total}) "
        f"from {args.trajectory.name}"
    )
    for idx in range(start, end, step):
        ts = timestamps[idx]
        l_pos = left[idx, :3]
        l_rot = left[idx, 3:]
        r_pos = right[idx, :3]
        r_rot = right[idx, 3:]
        print(
            f"idx {idx:04d} | t={fmt_ts(ts)} \n"
            f"L pos={fmt_vec(l_pos)} rotvec={fmt_vec(l_rot)} \n"
            f"R pos={fmt_vec(r_pos)} rotvec={fmt_vec(r_rot)}"
        )


if __name__ == "__main__":
    main()
