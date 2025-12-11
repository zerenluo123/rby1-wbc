#!/usr/bin/env python3
"""Measure end-to-end camera latency using QR-coded timestamps.

This script follows the procedure outlined in the UMI appendix: display a QR
code containing the host's current timestamp, stream frames from the wrist or
head cameras, decode the QR payload, and compute the latency as
``t_recv - t_display - l_display``. The results are printed and optionally
written to CSV for later analysis.
"""

'''
python - <<'PY'
import time, cv2, numpy as np, qrcode
while True:
    ts = time.time()
    qr = qrcode.make(f"{ts:.6f}").convert("RGB")
    img = cv2.cvtColor(np.array(qr), cv2.COLOR_RGB2BGR)
    cv2.imshow("QR timestamp", cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST))
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
PY

'''

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.camera_stream import AravisCameraStreamer

DEFAULT_CAMERA_SERIAL_MAP: Mapping[str, str] = {
    # "camera_head_main_rgb": "FLIR-Blackfly S BFS-PGE-50S5C-25260985",
    # "camera_head_main_right_rgb": "FLIR-Blackfly S BFS-PGE-50S5C-25272263",
    # "camera_head_ultrawide_rgb": "FLIR-Blackfly S BFS-PGE-50S5C-25260989",
    # "camera_left_main_rgb": "FLIR-Blackfly S BFS-PGE-23S3C-24260091",
    "camera_right_main_rgb": "FLIR-Blackfly S BFS-PGE-23S3C-24260092",
}


def _parse_camera_map(entries: Sequence[str] | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not entries:
        return mapping
    for entry in entries:
        if "=" in entry:
            key, serial = entry.split("=", 1)
        else:
            key, serial = entry, entry
        key = key.strip()
        serial = serial.strip()
        if not key or not serial:
            raise ValueError(f"Invalid camera specification '{entry}'")
        mapping[key] = serial
    return mapping


def _decode_qr_timestamp(image: np.ndarray, detector: cv2.QRCodeDetector) -> float | None:
    data, _, _ = detector.detectAndDecode(image)
    if not data:
        return None
    try:
        return float(data)
    except ValueError:
        return None


def _summarize(latencies: Sequence[float]) -> str:
    arr = np.asarray(list(latencies), dtype=float)
    if arr.size == 0:
        return "no samples"
    return (
        f"n={arr.size} min/med/p95/max = "
        f"{arr.min():.4f}s/{np.median(arr):.4f}s/{np.percentile(arr, 95):.4f}s/{arr.max():.4f}s"
    )


def _install_sigint_handler(stop_flag: MutableMapping[str, bool]) -> None:
    import signal

    def _handler(signum: int, frame: object | None) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure camera latency via QR-coded timestamps")
    parser.add_argument("--camera", action="append", help="Camera mapping KEY=SERIAL; repeatable")
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Observation key to decode (defaults to the first configured camera)",
    )
    parser.add_argument("--buffer-size", type=int, default=8, help="Frames to keep per camera")
    parser.add_argument("--samples", type=int, default=200, help="Number of decoded frames to collect")
    parser.add_argument("--display-latency", type=float, default=0.017, help="Known display latency in seconds")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV to write decoded samples")
    args = parser.parse_args()

    requested = _parse_camera_map(args.camera)
    camera_map = requested or dict(DEFAULT_CAMERA_SERIAL_MAP)

    target_key = args.camera_key or next(iter(camera_map))
    if target_key not in camera_map:
        raise SystemExit(f"Camera key '{target_key}' is not in configured mapping: {list(camera_map)}")

    detector = cv2.QRCodeDetector()
    streamer = AravisCameraStreamer(camera_map, buffer_size=args.buffer_size)
    streamer.start()
    streamer.wait_until_ready(min_frames=1, timeout=5.0)

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    rows: list[tuple[float, float, float]] = []
    start_time = time.time()
    print(f"Collecting {args.samples} QR decodes from {target_key}...")
    while not stop_flag["stop"] and len(rows) < args.samples:
        frames, timestamps = streamer.get_observation_window(1, include_timestamps=True)
        image = frames[target_key][0]
        decoded_ts = _decode_qr_timestamp(image, detector)
        if decoded_ts is None:
            continue
        recv_ts = time.time()
        latency = recv_ts - decoded_ts - float(args.display_latency)
        rows.append((recv_ts, decoded_ts, latency))
        if len(rows) % 20 == 0:
            print(f"  sample {len(rows)}: latency={latency:.4f}s")

    elapsed = time.time() - start_time
    streamer.stop()

    latencies = [row[2] for row in rows]
    print(f"Finished in {elapsed:.1f}s; {_summarize(latencies)}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["recv_timestamp", "qr_timestamp", "latency_seconds"])
            writer.writerows(rows)
        print(f"Wrote {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
