#!/usr/bin/env python3
"""Offline plotter for pose handler intervals recorded by teleop_iphone."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - convenience script
    raise SystemExit(
        "matplotlib is required for interval_plotter.py. Install it with `pip install matplotlib`."
    ) from exc


LABELS = ("left", "right", "head")
COLORS = {
    "left": "tab:blue",
    "right": "tab:orange",
    "head": "tab:green",
}


def read_recent_rows(path: Path, max_rows: int | None) -> List[Dict[str, str]]:
    """Load rows from the CSV log, optionally keeping only the newest `max_rows`."""
    if not path.exists():
        return []
    rows: deque[Dict[str, str]] | List[Dict[str, str]]
    rows = deque(maxlen=max_rows) if max_rows else []
    with path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return list(rows)


def prepare_series(rows: List[Dict[str, str]]):
    """Group samples by controller label."""
    series = {label: {"x": [], "y": []} for label in LABELS}
    if not rows:
        return series
    base_time = None
    for row in rows:
        label = row.get("label", "").lower()
        if label not in series:
            continue
        try:
            perf = float(row["perf_counter_s"])
            delta = float(row["delta_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if base_time is None:
            base_time = perf
        series[label]["x"].append(perf - base_time)
        series[label]["y"].append(delta)
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pose handler intervals from a saved CSV.")
    parser.add_argument(
        "log_file",
        type=str,
        help="CSV log file produced by TeleopIphone (iphone_pose_intervals_*.csv).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Number of most recent samples to plot (0 means plot the entire file).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file).expanduser().resolve()
    if not log_path.exists():
        raise SystemExit(f"CSV log {log_path} does not exist.")

    max_rows = args.window if args.window > 0 else None
    rows = read_recent_rows(log_path, max_rows=max_rows)
    if not rows:
        raise SystemExit(f"No interval samples found in {log_path}.")

    fig, ax = plt.subplots()
    lines = {}
    for label in LABELS:
        (line,) = ax.plot([], [], label=label.capitalize(), color=COLORS[label])
        lines[label] = line

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Interval (ms)")
    ax.set_title(f"iPhone Teleop Pose Handler Intervals\n{log_path.name}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    grouped = prepare_series(rows)
    for label, line in lines.items():
        xs = grouped[label]["x"]
        ys = grouped[label]["y"]
        line.set_data(xs, ys)
    ax.relim()
    ax.autoscale_view()
    plt.show()


if __name__ == "__main__":
    main()
