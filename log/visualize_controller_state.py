#!/usr/bin/env python3
"""Inspect controller log samples corresponding to a range of target indices."""

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


def fmt_vec(vec: np.ndarray, precision: int = 4) -> str:
    return "[" + ", ".join(f"{val:.{precision}f}" for val in vec) + "]"


def fmt_ts(ts: float) -> str:
    human = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
    return f"{ts:.3f}s ({human})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print controller samples whose timestamps align with a target index window.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Controller jsonl log (default: {DEFAULT_LOG.name}).",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJ,
        help=f"Trajectory pickle (default: {DEFAULT_TRAJ.name}).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Episode index inside the trajectory pickle.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First target index to inspect (inclusive).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last target index to inspect (exclusive). Defaults to the final sample.",
    )
    parser.add_argument(
        "--pad-before",
        type=float,
        default=0.1,
        help="Seconds to include before the first target timestamp (default: 0.1).",
    )
    parser.add_argument(
        "--pad-after",
        type=float,
        default=0.1,
        help="Seconds to include after the last target timestamp (default: 0.1).",
    )
    parser.add_argument(
        "--show-targets",
        action="store_true",
        help="Also print the left/right target poses for the selected indices.",
    )
    parser.add_argument(
        "--print-controller-samples",
        action="store_true",
        help="Dump every controller sample inside the window (can be verbose).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller_log = load_jsonl(args.log)
    episode = load_episode(args.trajectory, args.index)
    timestamps = np.asarray(episode.get("target_timestamps", []), dtype=float)
    if timestamps.size == 0:
        raise ValueError("Trajectory does not contain target_timestamps")

    total = len(timestamps)
    start_idx = max(0, int(args.start_index))
    if start_idx >= total:
        raise ValueError(f"--start-index {start_idx} exceeds available samples ({total})")

    end_idx = total if args.end_index is None else min(total, int(args.end_index))
    if end_idx <= start_idx:
        raise ValueError("--end-index must be greater than --start-index")

    start_ts = timestamps[start_idx]
    end_ts = timestamps[end_idx - 1]
    lo = start_ts - float(args.pad_before)
    hi = end_ts + float(args.pad_after)

    left = extract_pose_series(episode, "left")
    right = extract_pose_series(episode, "right")

    print(
        f"[controller] target idx {start_idx}..{end_idx - 1} | window {fmt_ts(lo)} -> {fmt_ts(hi)}"
    )

    if args.show_targets:
        for idx in range(start_idx, end_idx):
            ts = timestamps[idx]
            l_pos = left[idx, :3]
            r_pos = right[idx, :3]
            print(
                f"  target idx {idx:04d} @ {fmt_ts(ts)} | "
                f"L pos={fmt_vec(l_pos)} | R pos={fmt_vec(r_pos)}"
            )

    valid_ctrl = [
        entry for entry in controller_log if isinstance(entry.get("timestamp"), (int, float))
    ]
    if not valid_ctrl:
        raise ValueError("Controller log does not contain timestamped entries")
    valid_ctrl.sort(key=lambda entry: entry["timestamp"])
    ctrl_ts = [entry["timestamp"] for entry in valid_ctrl]

    def find_ctrl_for_target(ts: float) -> dict[str, Any] | None:
        idx = bisect_right(ctrl_ts, ts) - 1
        if idx < 0:
            return None
        return valid_ctrl[idx]

    def right_x_from_entry(entry: dict[str, Any] | None) -> float | None:
        if not entry:
            return None
        pose = entry.get("right") or {}
        pos = pose.get("position")
        if not pos:
            return None
        try:
            return float(pos[0])
        except (TypeError, ValueError):
            return None

    def fmt_delta(value: float | None) -> str:
        return "Δ={:+.4f}".format(value) if value is not None else "Δ=--"

    prev_target_x: float | None = None
    prev_ctrl_x: float | None = None
    print("[delta-x] target vs controller (right hand)")
    for idx in range(start_idx, end_idx):
        tgt_ts = timestamps[idx]
        tgt_x = float(right[idx, 0])
        tgt_dx = None if prev_target_x is None else tgt_x - prev_target_x

        ctrl_entry = find_ctrl_for_target(tgt_ts)
        ctrl_ts_val = ctrl_entry.get("timestamp") if ctrl_entry else None
        ctrl_line = ctrl_entry.get("_line") if ctrl_entry else None
        ctrl_x = right_x_from_entry(ctrl_entry)
        ctrl_dx = None if prev_ctrl_x is None or ctrl_x is None else ctrl_x - prev_ctrl_x

        target_part = (
            f"idx {idx:04d} | target t={fmt_ts(tgt_ts)} x={tgt_x:.4f} {fmt_delta(tgt_dx)}"
        )
        if ctrl_entry and ctrl_ts_val is not None:
            ctrl_x_str = "x={:.4f}".format(ctrl_x) if ctrl_x is not None else "x=--"
            ctrl_part = (
                f"ctrl line {ctrl_line} t={fmt_ts(ctrl_ts_val)} {ctrl_x_str} {fmt_delta(ctrl_dx)}"
            )
        else:
            ctrl_part = "ctrl: no prior sample"
        print(f"  {target_part} | {ctrl_part}")

        prev_target_x = tgt_x
        if ctrl_x is not None:
            prev_ctrl_x = ctrl_x

    if not args.print_controller_samples:
        return

    matching = [entry for entry in controller_log if lo <= entry.get("timestamp", 0.0) <= hi]
    if not matching:
        print("No controller samples found in this window.")
        return

    for entry in matching:
        ts = entry.get("timestamp")
        print(
            f"line {entry.get('_line')} | t={fmt_ts(ts)} | "
            f"keys={sorted(k for k in entry.keys() if not k.startswith('_'))}"
        )
        for side in ("left", "right", "head"):
            pose = entry.get(side)
            if not pose:
                continue
            pos = pose.get("position") or []
            rot = pose.get("rotation") or []
            print(
                f"    {side:>5} pos="
                f"[{', '.join(f'{p:.4f}' for p in pos)}] rot="
                f"[{', '.join(f'{r:.4f}' for r in rot)}]"
            )
        ctrl_state = entry.get("controller_state")
        if ctrl_state:
            print(f"    controller_state keys: {list(ctrl_state.keys())}")


if __name__ == "__main__":
    main()
