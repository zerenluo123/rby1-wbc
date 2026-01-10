#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.camera_stream import AravisCameraStreamer


@dataclass
class StereoFrameData:
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    capture_time: float
    receive_time: float


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


class AravisStereoCamera:
    def __init__(self, left_key: str, right_key: str):
        self.left_key = left_key
        self.right_key = right_key
        self.streamer = AravisCameraStreamer()

        available = list(getattr(self.streamer, "_camera_map", {}).keys())
        for k in [left_key, right_key]:
            if k not in getattr(self.streamer, "_camera_map", {}):
                raise SystemExit(f"Camera key '{k}' not found. Available: {available}")

        self.running = False

    def start(self):
        self.streamer.start()
        self.streamer.wait_until_ready(min_frames=1, timeout=5.0)
        self.running = True

    def stop(self):
        self.streamer.stop()
        self.running = False

    def read(self, latency: float = 0.0) -> Optional[StereoFrameData]:
        if not self.running:
            return None
        frames, _ts = self.streamer.get_observation_window(1, include_timestamps=True)
        if self.left_key not in frames or self.right_key not in frames:
            return None
        L = _ensure_u8_bgr(frames[self.left_key][0])
        R = _ensure_u8_bgr(frames[self.right_key][0])
        receive_time = time.monotonic()
        capture_time = receive_time - float(latency)
        return StereoFrameData(L, R, capture_time, receive_time)


class StereoFisheyeChessboardCalibrator:
    def __init__(
        self,
        camera: AravisStereoCamera,
        image_save_path: str = "./images",
        calibration_save_path: str = "./camera_calibration",
        latency: float = 0.0,
    ):
        self.cam = camera
        self.latency = float(latency)

        self.image_save_path = image_save_path
        self.calibration_save_path = calibration_save_path
        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.calibration_save_path, exist_ok=True)

        self.image_size: Optional[Tuple[int, int]] = None  # (W,H)

        # Per-view points (we’ll store corresponding points per accepted stereo frame)
        self.objpoints: List[np.ndarray] = []
        self.imgpoints_L: List[np.ndarray] = []
        self.imgpoints_R: List[np.ndarray] = []

        # Results
        self.K1 = np.eye(3, dtype=np.float64)
        self.D1 = np.zeros((4, 1), dtype=np.float64)
        self.K2 = np.eye(3, dtype=np.float64)
        self.D2 = np.zeros((4, 1), dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.T = np.zeros((3, 1), dtype=np.float64)

        self.rectify_maps = None

    def collect_stereo_frames(
        self,
        chessboard_size: Tuple[int, int],
        square_size_m: float,
        num_images: int = 40,
        min_interval_s: float = 0.8,
        show: bool = True,
        save_pairs: bool = True,
    ):
        """
        Collect stereo frames where BOTH left and right detect the chessboard.
        Controls:
          - SPACE: capture if detected
          - q/ESC: quit
        """
        print(f"Show a {chessboard_size[0]}x{chessboard_size[1]} chessboard (inner corners).")
        print("Controls: SPACE capture, q/ESC quit")
        print("Tip: fill the frame, avoid blur/glare, vary pose.")

        # Object points in chessboard coords
        objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= float(square_size_m)

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

        self.cam.start()
        last_cap = time.time() - 10
        captured = 0
        idx = 0

        win = "stereo_chessboard"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        try:
            while captured < num_images:
                frame = self.cam.read(latency=self.latency)
                if frame is None:
                    continue

                L = frame.left_bgr
                R = frame.right_bgr

                gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
                gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
                if self.image_size is None:
                    self.image_size = (gL.shape[1], gL.shape[0])

                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                retL, cornersL = cv2.findChessboardCorners(gL, chessboard_size, flags)
                retR, cornersR = cv2.findChessboardCorners(gR, chessboard_size, flags)

                disp = np.concatenate([L, R], axis=1)

                if retL:
                    cornersL = cv2.cornerSubPix(gL, cornersL, (5, 5), (-1, -1), subpix_criteria)
                    cv2.drawChessboardCorners(disp[:, :L.shape[1]], chessboard_size, cornersL, retL)

                if retR:
                    cornersR = cv2.cornerSubPix(gR, cornersR, (5, 5), (-1, -1), subpix_criteria)
                    cv2.drawChessboardCorners(disp[:, L.shape[1]:], chessboard_size, cornersR, retR)

                cv2.putText(
                    disp,
                    f"Captured {captured}/{num_images} | L:{int(retL)} R:{int(retR)} | SPACE to save, q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if show:
                    cv2.imshow(win, disp)

                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break

                if key == 32:  # SPACE
                    now = time.time()
                    if (now - last_cap) < min_interval_s:
                        continue
                    if not (retL and retR):
                        print("[skip] need BOTH left and right chessboard detections")
                        continue

                    self.objpoints.append(objp.copy())
                    self.imgpoints_L.append(cornersL.astype(np.float64))
                    self.imgpoints_R.append(cornersR.astype(np.float64))
                    last_cap = now

                    if save_pairs:
                        lp = os.path.join(self.image_save_path, f"left_{idx:04d}.png")
                        rp = os.path.join(self.image_save_path, f"right_{idx:04d}.png")
                        cv2.imwrite(lp, L)
                        cv2.imwrite(rp, R)

                    idx += 1
                    captured += 1
                    print(f"[saved] {captured}/{num_images}")

        finally:
            self.cam.stop()
            if show:
                cv2.destroyAllWindows()

        if captured < 10:
            raise RuntimeError(f"Too few captured stereo frames: {captured}. Move closer / improve sharpness.")

    def calibrate(self, baseline_m: float = 0.1, balance: float = 0.0):
        if self.image_size is None:
            raise RuntimeError("No image_size; collect frames first.")
        if len(self.objpoints) < 15:
            raise RuntimeError(f"Need ~15+ frames; have {len(self.objpoints)}")

        image_size = self.image_size  # (W,H)

        # Fisheye calibration flags (same style as your Insta360 class)
        calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

        # cv2.fisheye.calibrate expects:
        # objectPoints: list of (1,N,3), imagePoints: list of (N,1,2)
        obj = [op.astype(np.float64) for op in self.objpoints]
        imgL = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in self.imgpoints_L]
        imgR = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in self.imgpoints_R]

        print("Calibrating LEFT intrinsics...")
        rmsL, self.K1, self.D1, *_ = cv2.fisheye.calibrate(
            obj, imgL, image_size, np.eye(3), np.zeros((4, 1)), None, None,
            flags=calib_flags, criteria=criteria
        )

        print("Calibrating RIGHT intrinsics...")
        rmsR, self.K2, self.D2, *_ = cv2.fisheye.calibrate(
            obj, imgR, image_size, np.eye(3), np.zeros((4, 1)), None, None,
            flags=calib_flags, criteria=criteria
        )

        print("Stereo calibrating (FIX_INTRINSIC)...")
        stereo_flags = cv2.fisheye.CALIB_FIX_INTRINSIC
        rmsS, _K1, _D1, _K2, _D2, self.R, self.T = cv2.fisheye.stereoCalibrate(
            obj, imgL, imgR,
            self.K1, self.D1, self.K2, self.D2,
            image_size,
            None, None,
            flags=stereo_flags,
            criteria=criteria
        )

        baseline_est = float(np.linalg.norm(self.T))

        # Rectify + maps
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            self.K1, self.D1, self.K2, self.D2,
            image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=float(balance),
            fov_scale=1.0
        )
        map1L, map2L = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, R1, P1, image_size, cv2.CV_16SC2)
        map1R, map2R = cv2.fisheye.initUndistortRectifyMap(self.K2, self.D2, R2, P2, image_size, cv2.CV_16SC2)
        self.rectify_maps = (map1L, map2L, map1R, map2R)

        # Save JSON + maps
        out = {
            "model": "opencv_fisheye_stereo_chessboard",
            "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
            "left": {"K": self.K1.tolist(), "D": self.D1.tolist(), "rms": float(rmsL)},
            "right": {"K": self.K2.tolist(), "D": self.D2.tolist(), "rms": float(rmsR)},
            "stereo": {
                "rms": float(rmsS),
                "R": self.R.tolist(),
                "T": self.T.tolist(),
                "baseline_m": baseline_est,
                "baseline_expected_m": float(baseline_m),
            },
            "rectify": {"R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist()},
            "n_frames": int(len(obj)),
        }

        json_path = os.path.join(self.calibration_save_path, "stereo_fisheye_chessboard.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

        npz_path = os.path.join(self.calibration_save_path, "rectify_maps_chessboard.npz")
        np.savez_compressed(npz_path, map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R)

        print("\n=== Calibration OK ===")
        print(f"Frames: {len(obj)}")
        print(f"Left RMS:  {float(rmsL):.4f}")
        print(f"Right RMS: {float(rmsR):.4f}")
        print(f"Stereo RMS:{float(rmsS):.4f}")
        print(f"Baseline |T|: {baseline_est:.4f} m (expected ~{baseline_m:.4f} m)")
        print(f"Saved: {json_path}")
        print(f"Saved: {npz_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_key", default="camera_head_main_rgb")
    ap.add_argument("--right_key", default="camera_head_main_right_rgb")
    ap.add_argument("--out_images", default="./images")
    ap.add_argument("--out_calib", default="./camera_calibration")
    ap.add_argument("--latency", type=float, default=0.0)

    # IMPORTANT: chessboard_size is INNER corners (columns, rows)
    ap.add_argument("--chessboard_cols", type=int, required=True)
    ap.add_argument("--chessboard_rows", type=int, required=True)
    ap.add_argument("--square_size_m", type=float, required=True)

    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--min_interval", type=float, default=0.8)
    ap.add_argument("--baseline_m", type=float, default=0.1)
    ap.add_argument("--balance", type=float, default=0.0)
    args = ap.parse_args()

    cam = AravisStereoCamera(args.left_key, args.right_key)
    calib = StereoFisheyeChessboardCalibrator(
        camera=cam,
        image_save_path=args.out_images,
        calibration_save_path=args.out_calib,
        latency=args.latency,
    )

    calib.collect_stereo_frames(
        chessboard_size=(args.chessboard_cols, args.chessboard_rows),
        square_size_m=args.square_size_m,
        num_images=args.num,
        min_interval_s=args.min_interval,
        show=True,
        save_pairs=True,
    )
    calib.calibrate(baseline_m=args.baseline_m, balance=args.balance)


if __name__ == "__main__":
    main()
