#!/usr/bin/env python3
"""Inspect a particular target jump and compare it with controller samples."""

from __future__ import annotations

import argparse
import json
import pickle
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = SCRIPT_DIR / "teleop_20251107-115048.jsonl"
DEFAULT_TRAJ = SCRIPT_DIR.parent / "demo" / "teleop_20251107-115048.pkl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["_line"] = idx
            entries.append(data)
    return entries


def load_episode(path: Path, index: int) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not payload:
        raise ValueError(f"No episodes stored in {path}")
    try:
        episode = payload[index]
    except IndexError as exc:
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


def fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{val:.4f}" for val in vec) + "]"


def fmt_ts(ts: float) -> str:
    human = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
    return f"{ts:.3f}s ({human})"


def summarize_delta(label: str, before: np.ndarray, after: np.ndarray) -> None:
    delta = after - before
    print(f"  {label} before={fmt_vec(before)}")
    print(f"  {label}  after={fmt_vec(after)}")
    print(f"  {label}  delta={fmt_vec(delta)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a jump between two consecutive target indices and compare controller data.",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Controller jsonl log path.")
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJ, help="Trajectory pickle.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to inspect.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=207,
        help="Baseline target index; the next index will be treated as the 'after' sample.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.1,
        help="Padding (seconds) before/after the two target timestamps when searching controller logs.",
    )
    parser.add_argument(
        "--print-window",
        action="store_true",
        help="Dump every controller sample in the padded window for manual inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller_log = load_jsonl(args.log)
    episode = load_episode(args.trajectory, args.episode_index)
    timestamps = np.asarray(episode.get("target_timestamps", []), dtype=float)
    if timestamps.size == 0:
        raise ValueError("Trajectory does not contain target_timestamps")

    idx_before = int(args.start_index)
    idx_after = idx_before + 1
    if idx_after >= len(timestamps):
        raise ValueError("start-index must leave room for an after sample")

    left = extract_pose_series(episode, "left")
    right = extract_pose_series(episode, "right")

    print(f"Analyzing indices {idx_before} -> {idx_after}")
    ts_before = timestamps[idx_before]
    ts_after = timestamps[idx_after]
    print(f"  target ts before={fmt_ts(ts_before)} after={fmt_ts(ts_after)}")

    summarize_delta("LEFT  pos", left[idx_before, :3], left[idx_after, :3])
    summarize_delta("RIGHT pos", right[idx_before, :3], right[idx_after, :3])

    valid_ctrl = [
        entry for entry in controller_log if isinstance(entry.get("timestamp"), (int, float))
    ]
    if not valid_ctrl:
        raise ValueError("Controller log does not contain timestamped entries")
    valid_ctrl.sort(key=lambda entry: entry["timestamp"])
    ctrl_ts = [entry["timestamp"] for entry in valid_ctrl]

    def ctrl_before(ts: float) -> dict[str, Any] | None:
        idx = bisect_right(ctrl_ts, ts) - 1
        if idx < 0:
            return None
        return valid_ctrl[idx]

    ctrl_before_sample = ctrl_before(ts_before)
    ctrl_after_sample = ctrl_before(ts_after)

    def extract_pos(entry: dict[str, Any] | None, side: str) -> np.ndarray | None:
        if not entry:
            return None
        pose = entry.get(side) or {}
        pos = pose.get("position")
        if pos is None:
            return None
        return np.asarray(pos, dtype=float)

    ctrl_left_before = extract_pos(ctrl_before_sample, "left")
    ctrl_left_after = extract_pos(ctrl_after_sample, "left")
    ctrl_right_before = extract_pos(ctrl_before_sample, "right")
    ctrl_right_after = extract_pos(ctrl_after_sample, "right")

    if ctrl_right_before is None or ctrl_right_after is None:
        print("Controller samples missing right-hand pose; cannot compute delta.")
    else:
        print("Controller pairing (latest sample before each target timestamp):")
        print(
            f"  before line {ctrl_before_sample.get('_line')} t={fmt_ts(ctrl_before_sample['timestamp'])}"
        )
        print(
            f"  after  line {ctrl_after_sample.get('_line')} t={fmt_ts(ctrl_after_sample['timestamp'])}"
        )
        summarize_delta("CTRL right", ctrl_right_before, ctrl_right_after)
        if ctrl_left_before is not None and ctrl_left_after is not None:
            summarize_delta("CTRL left ", ctrl_left_before, ctrl_left_after)
        else:
            print("  Left-hand pose missing in controller samples.")
    if not args.print_window:
        return

    window_lo = min(ts_before, ts_after) - float(args.pad)
    window_hi = max(ts_before, ts_after) + float(args.pad)
    print(f"\nController samples between {fmt_ts(window_lo)} and {fmt_ts(window_hi)}:")
    for entry in controller_log:
        ts = entry.get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        if not (window_lo <= ts <= window_hi):
            continue
        left_pos = extract_pos(entry, "left")
        right_pos = extract_pos(entry, "right")
        left_str = fmt_vec(left_pos) if left_pos is not None else "[missing]"
        right_str = fmt_vec(right_pos) if right_pos is not None else "[missing]"
        print(
            f"line {entry.get('_line'):>5} | t={fmt_ts(ts)} | left={left_str} | right={right_str}"
        )
if __name__ == "__main__":
    main()