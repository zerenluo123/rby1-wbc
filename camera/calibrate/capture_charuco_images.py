#!/usr/bin/env python3
"""
Capture and save ChArUco calibration images from AravisCameraStreamer.

Usage examples:
  # Interactive: press SPACE to save, q to quit
  python capture_charuco_images.py --camera-key camera_head_rgb --out calib_imgs --interactive

  # Automatic: save N images at ~rate Hz (best if you move the camera smoothly)
  python capture_charuco_images.py --camera-key camera_head_rgb --out calib_imgs --count 120 --rate 5

Notes:
- Use the EXACT streaming mode you will use later (480x300, 4x4 binning).
- Lock exposure/gain/whitebalance before capture (outside this script).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, MutableMapping

import cv2
import numpy as np

# Make project imports work like in your existing script
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
    """
    Try to convert common formats to uint8 BGR for display and saving.
    - If already uint8 and 3-ch: assume BGR.
    - If RGB: caller can swap, but we don't know; many pipelines already give BGR.
    - If mono: convert to BGR.
    - If float: scale to 0..255 using min/max (fallback).
    """
    if img is None:
        raise ValueError("Got None image")

    if img.dtype != np.uint8:
        # best-effort conversion
        arr = img.astype(np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = arr

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture ChArUco calibration images")
    parser.add_argument("--camera-key", default=None, help="Observation key to capture (defaults to first camera)")
    parser.add_argument("--out", required=True, help="Output directory for saved images")
    parser.add_argument("--prefix", default="frame", help="Filename prefix")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"], help="Image extension")
    parser.add_argument("--interactive", action="store_true", help="Press SPACE to save; q/ESC to quit")
    parser.add_argument("--count", type=int, default=0, help="If >0, save this many frames automatically")
    parser.add_argument("--rate", type=float, default=5.0, help="Auto-capture rate (Hz) when --count > 0")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--downscale-display", type=float, default=1.0, help="Preview scale factor (e.g., 2.0)")
    parser.add_argument("--min-interval", type=float, default=0.0, help="Min seconds between saved frames (extra safety)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    streamer = AravisCameraStreamer()

    # Choose camera key
    if args.camera_key is None:
        # AravisCameraStreamer keeps mapping in _camera_map in your snippet
        try:
            target_key = next(iter(streamer._camera_map))
        except Exception as e:
            raise SystemExit("Could not infer default camera key; pass --camera-key") from e
    else:
        target_key = args.camera_key

    if not hasattr(streamer, "_camera_map") or target_key not in streamer._camera_map:
        available = list(getattr(streamer, "_camera_map", {}).keys())
        raise SystemExit(f"Camera key '{target_key}' not found. Available: {available}")

    stop_flag: Dict[str, bool] = {"stop": False}
    _install_sigint_handler(stop_flag)

    streamer.start()
    streamer.wait_until_ready(min_frames=1, timeout=5.0)

    print(f"Capturing from camera key: {target_key}")
    print(f"Saving to: {out_dir.resolve()}")

    saved = 0
    last_save_t = 0.0
    next_auto_t = time.time()

    win = "preview"
    if args.show or args.interactive:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while not stop_flag["stop"]:
            frames, _timestamps = streamer.get_observation_window(1, include_timestamps=True)
            img = frames[target_key][0]
            img = _ensure_u8_bgr(img)

            if args.show or args.interactive:
                disp = img
                if args.downscale_display != 1.0:
                    disp = cv2.resize(
                        img,
                        (int(img.shape[1] * args.downscale_display), int(img.shape[0] * args.downscale_display)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                cv2.imshow(win, disp)

            now = time.time()

            should_save = False
            if args.interactive:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or q
                    break
                if key == ord(" "):  # SPACE
                    should_save = True
            else:
                # Non-interactive mode: still pump window events if shown
                if args.show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if args.count > 0 and now >= next_auto_t:
                    should_save = True
                    next_auto_t = now + (1.0 / max(1e-6, args.rate))

            # Rate limit saves if requested
            if should_save and args.min_interval > 0.0 and (now - last_save_t) < args.min_interval:
                should_save = False

            if should_save:
                fname = f"{args.prefix}_{saved:04d}.{args.ext}"
                path = out_dir / fname
                # Use PNG for lossless corners (recommended)
                if args.ext == "jpg":
                    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(str(path), img)

                saved += 1
                last_save_t = now
                print(f"Saved {saved}: {path.name}")

                if args.count > 0 and saved >= args.count:
                    break

    finally:
        streamer.stop()
        if args.show or args.interactive:
            cv2.destroyAllWindows()

    print(f"Done. Saved {saved} images.")


if __name__ == "__main__":
    main()
