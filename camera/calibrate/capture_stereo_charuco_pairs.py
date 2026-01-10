#!/usr/bin/env python3
"""
Capture synchronized stereo image pairs (left/right) from AravisCameraStreamer.

Left key:  camera_head_main_rgb
Right key: camera_head_main_right_rgb   (per your setup)

Usage:
  mkdir -p stereo_calib_pairs
  python capture_stereo_charuco_pairs.py --out stereo_calib_pairs --interactive --show

  # or auto:
  python capture_stereo_charuco_pairs.py --out stereo_calib_pairs --count 150 --rate 5 --show

Interactive tips:
- Click the OpenCV window once to give it keyboard focus.
- Press SPACE (or 's') to save a pair.
- Press 'q' or ESC to quit.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, MutableMapping

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.camera_stream import AravisCameraStreamer


def _install_sigint_handler(stop_flag: MutableMapping[str, bool]) -> None:
    import signal

    def _handler(signum: int, frame: object | None) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _ensure_u8_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Got None image")

    if img.dtype != np.uint8:
        arr = img.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = arr

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def _stack_preview(left: np.ndarray, right: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h = min(left.shape[0], right.shape[0])
    left = left[:h]
    right = right[:h]
    vis = np.concatenate([left, right], axis=1)
    if scale != 1.0:
        vis = cv2.resize(
            vis,
            (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST,
        )
    return vis


def _find_next_index(out_dir: Path, ext: str) -> int:
    pattern = re.compile(rf"^(left|right)_(\d+)\.{re.escape(ext)}$")
    max_idx = -1
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        match = pattern.match(p.name)
        if match:
            idx = int(match.group(2))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-key", default="camera_head_main_rgb")
    ap.add_argument("--right-key", default="camera_head_main_right_rgb")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--ext", default="png", choices=["png", "jpg"])
    ap.add_argument("--interactive", action="store_true", help="SPACE/'s' to save pair, q/ESC quit")
    ap.add_argument("--count", type=int, default=0, help="If >0, auto save this many pairs")
    ap.add_argument("--rate", type=float, default=5.0, help="Auto save rate (Hz) when --count>0")
    ap.add_argument("--show", action="store_true", help="Show preview")
    ap.add_argument("--preview-scale", type=float, default=2.0)
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--waitkey-ms", type=int, default=30, help="waitKey delay; larger is more reliable for keypresses")
    ap.add_argument("--debug-keys", action="store_true", help="Print received keycodes (debug)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    streamer = AravisCameraStreamer()
    available = list(getattr(streamer, "_camera_map", {}).keys())
    for k in [args.left_key, args.right_key]:
        if k not in getattr(streamer, "_camera_map", {}):
            raise SystemExit(f"Camera key '{k}' not found. Available: {available}")

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    streamer.start()
    streamer.wait_until_ready(min_frames=1, timeout=5.0)

    win = "stereo_preview"
    if args.show or args.interactive:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    next_index = _find_next_index(out_dir, args.ext)
    if next_index > 0:
        print(f"Found existing images, starting at index {next_index:04d}.")
    saved = 0
    last_save_t = 0.0
    next_auto_t = time.time()

    print(f"Left: {args.left_key}")
    print(f"Right: {args.right_key}")
    print(f"Saving pairs to: {out_dir.resolve()}")
    if args.interactive:
        print("Interactive mode: click the preview window once, then press SPACE (or 's') to save.")

    try:
        while not stop_flag["stop"]:
            frames, _ts = streamer.get_observation_window(1, include_timestamps=True)
            L = _ensure_u8_bgr(frames[args.left_key][0])
            R = _ensure_u8_bgr(frames[args.right_key][0])

            if args.show or args.interactive:
                vis = _stack_preview(L, R, scale=args.preview_scale)
                cv2.imshow(win, vis)

            now = time.time()
            should_save = False

            if args.interactive:
                # More reliable than waitKey(1); also allows WM to deliver key events
                key = cv2.waitKey(args.waitkey_ms)
                key8 = key & 0xFF

                if args.debug_keys and key != -1:
                    print(f"[debug] key={key} key8={key8}")

                if key8 in (27, ord("q")):  # ESC or q
                    break

                # SPACE sometimes doesn't come through unless window has focus; also accept 's'
                if key8 == 32 or key8 == ord("s"):
                    should_save = True
            else:
                if args.show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if args.count > 0 and now >= next_auto_t:
                    should_save = True
                    next_auto_t = now + (1.0 / max(1e-6, args.rate))

            if should_save and args.min_interval > 0.0 and (now - last_save_t) < args.min_interval:
                should_save = False

            if should_save:
                left_path = out_dir / f"left_{next_index:04d}.{args.ext}"
                right_path = out_dir / f"right_{next_index:04d}.{args.ext}"

                okL, okR = True, True
                if args.ext == "jpg":
                    okL = cv2.imwrite(str(left_path), L, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    okR = cv2.imwrite(str(right_path), R, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    okL = cv2.imwrite(str(left_path), L)
                    okR = cv2.imwrite(str(right_path), R)

                if not okL or not okR:
                    print(f"[warn] Failed to write: {left_path.name} or {right_path.name}")
                else:
                    saved += 1
                    next_index += 1
                    last_save_t = now
                    print(f"Saved pair {saved}: {left_path.name}, {right_path.name}")

                if args.count > 0 and saved >= args.count:
                    break

    finally:
        streamer.stop()
        if args.show or args.interactive:
            cv2.destroyAllWindows()

    print(f"Done. Saved {saved} stereo pairs.")


if __name__ == "__main__":
    main()
