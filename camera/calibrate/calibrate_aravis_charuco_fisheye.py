#!/usr/bin/env python3
import os
import sys
import time
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import argparse

# --- adjust if needed ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.camera_stream import AravisCameraStreamer


@dataclass
class StereoFrameData:
    left_bgr: np.ndarray
    right_bgr: np.ndarray
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


DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
}


def _make_detector_params() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    # More robust at low-res / blur:
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    return params


def _detect_charuco(gray: np.ndarray, aruco_dict, board):
    params = _make_detector_params()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None, None, corners, ids

    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)

    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
    )
    if n is None or n < 4 or ch_corners is None or ch_ids is None:
        return None, None, corners, ids

    # Subpixel refine ChArUco corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    cv2.cornerSubPix(gray, ch_corners, (5, 5), (-1, -1), criteria)

    return ch_corners, ch_ids, corners, ids


class AravisStereoSource:
    def __init__(self, left_key: str, right_key: str):
        self.left_key = left_key
        self.right_key = right_key
        self.streamer = AravisCameraStreamer()

        available = list(getattr(self.streamer, "_camera_map", {}).keys())
        for k in [left_key, right_key]:
            if k not in getattr(self.streamer, "_camera_map", {}):
                raise SystemExit(f"Camera key '{k}' not found. Available: {available}")

    def start(self):
        self.streamer.start()
        self.streamer.wait_until_ready(min_frames=1, timeout=5.0)

    def stop(self):
        self.streamer.stop()

    def read(self) -> Optional[StereoFrameData]:
        frames, _ts = self.streamer.get_observation_window(1, include_timestamps=True)
        if self.left_key not in frames or self.right_key not in frames:
            return None
        L = _ensure_u8_bgr(frames[self.left_key][0])
        R = _ensure_u8_bgr(frames[self.right_key][0])
        return StereoFrameData(L, R, time.time())


class FisheyeStereoCharucoCalibrator:
    """
    Calibrate fisheye intrinsics (left/right) + stereo extrinsics using a ChArUco board.
    This matches the calib.io board you printed.
    """
    def __init__(
        self,
        left_key: str,
        right_key: str,
        out_dir: str,
        squares_x: int,
        squares_y: int,
        square_length_m: float,
        marker_length_m: float,
        dict_name: str,
        baseline_m: float = 0.1,
    ):
        self.left_key = left_key
        self.right_key = right_key
        self.baseline_m = float(baseline_m)

        if dict_name not in DICT_MAP:
            raise SystemExit(f"Unknown dict {dict_name}. Choose from: {list(DICT_MAP.keys())}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            float(square_length_m),
            float(marker_length_m),
            self.aruco_dict,
        )
        self.chessboard_corners_3d = self.board.getChessboardCorners()  # (Nc,3)

        self.source = AravisStereoSource(left_key, right_key)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "pairs").mkdir(parents=True, exist_ok=True)

        # Collected points
        self.obj_L: List[np.ndarray] = []
        self.img_L: List[np.ndarray] = []
        self.obj_R: List[np.ndarray] = []
        self.img_R: List[np.ndarray] = []

        self.obj_st: List[np.ndarray] = []
        self.imgL_st: List[np.ndarray] = []
        self.imgR_st: List[np.ndarray] = []

        self.image_size: Optional[Tuple[int, int]] = None  # (W,H)
        self.saved = 0

    def collect(
        self,
        need: int = 40,
        min_each: int = 15,
        min_common: int = 8,
        save_pairs: bool = True,
        show: bool = True,
    ):
        """
        Collect 'need' good stereo frames.
        Controls:
          - SPACE: save if it meets thresholds
          - f: force save (relaxed, still requires >=4 common)
          - q/ESC: quit
        """
        print("Collecting stereo frames...")
        print("Controls: SPACE save, f force-save, q/ESC quit")
        print("Tip: board should fill most of the 480x300 image; avoid blur/glare.")

        self.source.start()
        cv2.namedWindow("stereo", cv2.WINDOW_NORMAL)

        try:
            while self.saved < need:
                data = self.source.read()
                if data is None:
                    continue

                L = data.left_bgr
                R = data.right_bgr

                gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
                gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
                if self.image_size is None:
                    self.image_size = (gL.shape[1], gL.shape[0])

                chL, idL, _, _ = _detect_charuco(gL, self.aruco_dict, self.board)
                chR, idR, _, _ = _detect_charuco(gR, self.aruco_dict, self.board)

                nL = 0 if chL is None else int(len(chL))
                nR = 0 if chR is None else int(len(chR))
                common_n = 0

                common = None
                if chL is not None and chR is not None:
                    idsL = idL.flatten().astype(int)
                    idsR = idR.flatten().astype(int)
                    common = np.intersect1d(idsL, idsR)
                    common_n = int(common.size)

                if show:
                    vis = np.concatenate([L, R], axis=1)
                    cv2.putText(
                        vis,
                        f"L corners:{nL}  R corners:{nR}  common:{common_n}  saved:{self.saved}/{need}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                    cv2.imshow("stereo", vis)

                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break

                want_save = (key == 32)  # SPACE
                force_save = (key == ord("f"))

                if not (want_save or force_save):
                    continue

                # Save only if it has at least some correspondences
                if common is None or common_n < 4:
                    print("[skip] too few common corners")
                    continue

                ok_thresh = (nL >= min_each and nR >= min_each and common_n >= min_common)
                if (not ok_thresh) and (not force_save):
                    print(f"[skip] below thresholds (L={nL}, R={nR}, common={common_n}); press 'f' to force")
                    continue

                # Build matched arrays
                idsL = idL.flatten().astype(int)
                idsR = idR.flatten().astype(int)
                idxL = np.array([np.where(idsL == c)[0][0] for c in common], dtype=int)
                idxR = np.array([np.where(idsR == c)[0][0] for c in common], dtype=int)

                obj = self.chessboard_corners_3d[common].reshape(-1, 1, 3).astype(np.float64)
                imgpL = chL[idxL].reshape(-1, 1, 2).astype(np.float64)
                imgpR = chR[idxR].reshape(-1, 1, 2).astype(np.float64)

                self.obj_L.append(obj); self.img_L.append(imgpL)
                self.obj_R.append(obj); self.img_R.append(imgpR)
                self.obj_st.append(obj); self.imgL_st.append(imgpL); self.imgR_st.append(imgpR)

                if save_pairs:
                    cv2.imwrite(str(self.out_dir / "pairs" / f"left_{self.saved:04d}.png"), L)
                    cv2.imwrite(str(self.out_dir / "pairs" / f"right_{self.saved:04d}.png"), R)

                self.saved += 1
                print(f"[saved] {self.saved}/{need} (L={nL}, R={nR}, common={common_n})")

        finally:
            self.source.stop()
            cv2.destroyAllWindows()

        if self.saved < 10:
            raise RuntimeError(f"Too few saved frames ({self.saved}). Move board closer / improve sharpness.")

    def calibrate(self, balance: float = 0.0):
        if self.image_size is None:
            raise RuntimeError("No images collected")
        if len(self.obj_st) < 15:
            raise RuntimeError(f"Need ~15+ frames; have {len(self.obj_st)}")

        image_size = self.image_size

        def fisheye_calibrate(objpoints, imgpoints):
            K = np.eye(3, dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_CHECK_COND
                | cv2.fisheye.CALIB_FIX_SKEW
            )
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
            rms, K, D, *_ = cv2.fisheye.calibrate(
                objpoints, imgpoints, image_size, K, D, None, None, flags=flags, criteria=criteria
            )
            return float(rms), K, D

        # Intrinsics
        rmsL, K1, D1 = fisheye_calibrate(self.obj_L, self.img_L)
        rmsR, K2, D2 = fisheye_calibrate(self.obj_R, self.img_R)

        # Stereo (fix intrinsics)
        flags_stereo = cv2.fisheye.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        rms_stereo, _, _, _, _, R, T = cv2.fisheye.stereoCalibrate(
            self.obj_st,
            self.imgL_st,
            self.imgR_st,
            K1, D1, K2, D2,
            image_size,
            None, None,
            flags=flags_stereo,
            criteria=criteria,
        )

        baseline = float(np.linalg.norm(T))

        # Rectify + maps
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=float(balance),
            fov_scale=1.0,
        )
        map1L, map2L = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
        map1R, map2R = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

        # Save
        out_json = self.out_dir / "stereo_fisheye_charuco.json"
        out_npz = self.out_dir / "rectify_maps.npz"
        calib = {
            "board": {
                "squares_x": int(self.board.getChessboardSize()[0]),
                "squares_y": int(self.board.getChessboardSize()[1]),
            },
            "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
            "left": {"K": K1.tolist(), "D": D1.tolist(), "rms": rmsL},
            "right": {"K": K2.tolist(), "D": D2.tolist(), "rms": rmsR},
            "stereo": {
                "rms": float(rms_stereo),
                "R": R.tolist(),
                "T": T.tolist(),
                "baseline_m": baseline,
                "baseline_expected_m": float(self.baseline_m),
            },
            "rectify": {"R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist()},
            "accepted_frames": int(len(self.obj_st)),
            "pairs_dir": str(self.out_dir / "pairs"),
        }
        out_json.write_text(json.dumps(calib, indent=2))
        np.savez_compressed(out_npz, map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R)

        print("\n=== DONE ===")
        print(f"frames used: {len(self.obj_st)}")
        print(f"Left RMS:  {rmsL:.4f}")
        print(f"Right RMS: {rmsR:.4f}")
        print(f"Stereo RMS:{float(rms_stereo):.4f}")
        print(f"Baseline |T|: {baseline:.4f} m (expected ~{self.baseline_m:.4f} m)")
        print(f"Saved: {out_json}")
        print(f"Saved: {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_key", default="camera_head_main_rgb")
    ap.add_argument("--right_key", default="camera_head_main_right_rgb")
    ap.add_argument("--out_dir", default="stereo_calib_out_live")

    # calib.io board you described
    ap.add_argument("--squares_x", type=int, default=11)
    ap.add_argument("--squares_y", type=int, default=8)
    ap.add_argument("--square_length_m", type=float, default=0.015)
    ap.add_argument("--marker_length_m", type=float, default=0.011)
    ap.add_argument("--dict", default="DICT_4X4_1000")

    ap.add_argument("--baseline_m", type=float, default=0.1)
    ap.add_argument("--need", type=int, default=40)
    ap.add_argument("--min_each", type=int, default=15)
    ap.add_argument("--min_common", type=int, default=8)
    ap.add_argument("--balance", type=float, default=0.0)
    ap.add_argument("--no_save_pairs", action="store_true")
    args = ap.parse_args()

    calib = FisheyeStereoCharucoCalibrator(
        left_key=args.left_key,
        right_key=args.right_key,
        out_dir=args.out_dir,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        dict_name=args.dict,
        baseline_m=args.baseline_m,
    )

    # Collect then calibrate
    calib.collect(
        need=args.need,
        min_each=args.min_each,
        min_common=args.min_common,
        save_pairs=not args.no_save_pairs,
        show=True,
    )
    calib.calibrate(balance=args.balance)


if __name__ == "__main__":
    main()
