#!/usr/bin/env python3
"""Analyze teleop controller logs and recorded targets for sudden EE pose shifts."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = SCRIPT_DIR / "teleop_20251107-115048.jsonl"
DEFAULT_TRAJ = SCRIPT_DIR.parent / "demo" / "teleop_20251107-115048.pkl"


@dataclass(slots=True)
class ShiftEvent:
    idx_before: int
    idx_after: int
    ts_before: float
    ts_after: float
    pos_before: np.ndarray
    pos_after: np.ndarray
    rotvec_before: np.ndarray
    rotvec_after: np.ndarray
    diff_m: float


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


def load_episode(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not payload:
        raise ValueError("Empty trajectory pickle")
    episode = payload[0]
    if not isinstance(episode, dict):
        raise TypeError("Unexpected episode structure")
    return episode


def extract_target_series(
    episode: dict[str, Any], hand: str
) -> tuple[np.ndarray, np.ndarray]:
    payloads = episode.get(f"grippers_{hand}")
    if not payloads:
        raise ValueError(f"No target data recorded for {hand}")
    target_block = payloads[0]
    tcp_pose = np.asarray(target_block["tcp_pose"], dtype=float)
    timestamps = np.asarray(episode.get("target_timestamps", []), dtype=float)
    sample_count = min(len(timestamps), len(tcp_pose))
    if sample_count < 2:
        raise ValueError("Not enough samples to compute differences")
    return tcp_pose[:sample_count], timestamps[:sample_count]


def detect_largest_jump(
    tcp_pose: np.ndarray, timestamps: np.ndarray
) -> ShiftEvent:
    positions = tcp_pose[:, :3]
    rotvecs = tcp_pose[:, 3:]
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    idx = int(np.argmax(diffs))
    return ShiftEvent(
        idx_before=idx,
        idx_after=idx + 1,
        ts_before=float(timestamps[idx]),
        ts_after=float(timestamps[idx + 1]),
        pos_before=positions[idx],
        pos_after=positions[idx + 1],
        rotvec_before=rotvecs[idx],
        rotvec_after=rotvecs[idx + 1],
        diff_m=float(diffs[idx]),
    )


def controller_entries_span(
    entries: list[dict[str, Any]], start_ts: float, end_ts: float
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    lo = min(start_ts, end_ts)
    hi = max(start_ts, end_ts)

    valid = [entry for entry in entries if isinstance(entry.get("timestamp"), (int, float))]
    if not valid:
        return [], {}
    valid.sort(key=lambda e: e["timestamp"])

    between = [entry for entry in valid if lo <= entry["timestamp"] <= hi]
    before_neighbor = max(
        (entry for entry in valid if entry["timestamp"] < lo),
        default=None,
        key=lambda e: e["timestamp"],
    )
    if before_neighbor is None:
        before_neighbor = max(
            (entry for entry in valid if entry["timestamp"] <= lo),
            default=None,
            key=lambda e: e["timestamp"],
        )
    after_neighbor = next((entry for entry in valid if entry["timestamp"] > hi), None)
    if after_neighbor is None:
        after_neighbor = next((entry for entry in valid if entry["timestamp"] >= hi), None)

    ordered: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def add_entry(entry: dict[str, Any]) -> None:
        if entry is None or id(entry) in seen_ids:
            return
        seen_ids.add(id(entry))
        ts_value = entry["timestamp"]
        for idx, existing in enumerate(ordered):
            if ts_value < existing["timestamp"]:
                ordered.insert(idx, entry)
                break
        else:
            ordered.append(entry)

    add_entry(before_neighbor)
    for entry in between:
        add_entry(entry)
    add_entry(after_neighbor)

    highlight_targets: dict[int, list[str]] = {}

    def mark_prev_sample(ts: float, label: str) -> None:
        candidate = max(
            (entry for entry in valid if entry["timestamp"] < ts),
            default=None,
            key=lambda e: e["timestamp"],
        )
        if candidate is None:
            candidate = max(
                (entry for entry in valid if entry["timestamp"] <= ts),
                default=None,
                key=lambda e: e["timestamp"],
            )
        if candidate is None:
            candidate = min(valid, key=lambda e: abs(e["timestamp"] - ts))
        add_entry(candidate)
        highlight_targets.setdefault(id(candidate), []).append(label)

    mark_prev_sample(start_ts, "BEFORE target sample")
    mark_prev_sample(end_ts, "AFTER target sample")

    highlights = {entry_id: " & ".join(labels) for entry_id, labels in highlight_targets.items()}
    return ordered, highlights


def fmt_vec(vec: np.ndarray, precision: int = 4) -> str:
    return "[" + ", ".join(f"{val:.{precision}f}" for val in vec) + "]"


def fmt_ts(ts: float) -> str:
    human = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
    return f"{ts:.3f}s ({human})"


def describe_shift(event: ShiftEvent) -> None:
    print("=== Largest target EE jump ===")
    print(
        f"Samples {event.idx_before} -> {event.idx_after} | "
        f"Δpos = {event.diff_m*1000:.1f} mm"
    )
    print(f"Before : t={fmt_ts(event.ts_before)} pos={fmt_vec(event.pos_before)}")
    print(f"After  : t={fmt_ts(event.ts_after)} pos={fmt_vec(event.pos_after)}")
    rot_delta = np.linalg.norm(event.rotvec_after - event.rotvec_before)
    print(f"Δrotation (rotvec norm): {rot_delta:.4f} rad")


def describe_controller_window(
    entries: list[dict[str, Any]],
    start_ts: float,
    end_ts: float,
    highlights: dict[int, str],
) -> None:
    if not entries:
        print("No controller samples between these timestamps.")
        return
    print(
        "=== Controller samples between target timestamps "
        f"{fmt_ts(start_ts)} -> {fmt_ts(end_ts)} (n={len(entries)}) ==="
    )
    for entry in entries:
        ts = entry.get("timestamp")
        label = highlights.get(id(entry))
        label_suffix = f" [{label}]" if label else ""
        print(
            f"line {entry.get('_line')} | t={fmt_ts(ts)} | "
            f"keys={sorted(k for k in entry.keys() if not k.startswith('_'))}"
            f"{label_suffix}"
        )
        for side in ("left", "right"):
            pose = entry.get(side)
            if not pose:
                continue
            pos = pose.get("position") or []
            rot = pose.get("rotation") or []
            print(
                f"  {side:>5} pos="
                f"[{', '.join(f'{p:.4f}' for p in pos)}] rot="
                f"[{', '.join(f'{r:.4f}' for r in rot)}]"
            )
        ctrl_state = entry.get("controller_state")
        if ctrl_state:
            print(f"  controller_state keys: {list(ctrl_state.keys())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find sudden EE pose jumps and inspect controller state."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Controller jsonl log (default: {DEFAULT_LOG.name})",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJ,
        help=f"Trajectory pickle (default: {DEFAULT_TRAJ.name})",
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right"),
        default="left",
        help="Which hand/EE to inspect (default: left).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller_log = load_jsonl(args.log)
    episode = load_episode(args.trajectory)
    tcp_pose, timestamps = extract_target_series(episode, args.hand)
    shift_event = detect_largest_jump(tcp_pose, timestamps)
    describe_shift(shift_event)
    matches, highlights = controller_entries_span(
        controller_log, shift_event.ts_before, shift_event.ts_after
    )
    describe_controller_window(matches, shift_event.ts_before, shift_event.ts_after, highlights)


if __name__ == "__main__":
    main()
