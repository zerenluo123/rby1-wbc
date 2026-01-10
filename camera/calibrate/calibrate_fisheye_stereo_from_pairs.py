#!/usr/bin/env python3
"""
Fisheye stereo calibration using ChArUco on saved stereo pairs.

- Detect ChArUco corners in each left/right image.
- Calibrate fisheye intrinsics per camera (cv2.fisheye.calibrate).
- Stereo-calibrate extrinsics (cv2.fisheye.stereoCalibrate).
- Stereo-rectify + produce rectification maps (initUndistortRectifyMap).
- Save everything to JSON.

Example:
  python calibrate_fisheye_stereo_from_pairs.py \
    --pairs_dir stereo_calib_pairs \
    --out stereo_calib_out \
    --squares_x 11 --squares_y 8 \
    --square_length_m 0.015 --marker_length_m 0.011 \
    --dict DICT_4X4_1000 \
    --baseline_m 0.1 \
    --show
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


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


def detect_charuco(gray, aruco_dict, board):
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None, None, corners, ids
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if n is None or n < 6 or ch_corners is None or ch_ids is None:
        return None, None, corners, ids
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    cv2.cornerSubPix(gray, ch_corners, (5, 5), (-1, -1), criteria)
    return ch_corners, ch_ids, corners, ids


def load_pairs(pairs_dir: Path) -> List[Tuple[Path, Path]]:
    lefts = sorted(pairs_dir.glob("left_*.png")) + sorted(pairs_dir.glob("left_*.jpg"))
    pairs = []
    for lp in lefts:
        idx = lp.stem.split("_")[-1]
        rp_png = pairs_dir / f"right_{idx}.png"
        rp_jpg = pairs_dir / f"right_{idx}.jpg"
        rp = rp_png if rp_png.exists() else rp_jpg if rp_jpg.exists() else None
        if rp is not None and rp.exists():
            pairs.append((lp, rp))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out", default="stereo_calib_out")
    ap.add_argument("--squares_x", type=int, default=7)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_length_m", type=float, required=True)
    ap.add_argument("--marker_length_m", type=float, required=True)
    ap.add_argument("--dict", default="DICT_5X5_1000")
    ap.add_argument("--min_corners", type=int, default=12)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--baseline_m", type=float, default=0.1, help="Expected baseline (for sanity check only)")
    ap.add_argument("--balance", type=float, default=0.0, help="0=crop more, 1=keep more FOV in rectified images")
    args = ap.parse_args()

    if args.dict not in DICT_MAP:
        raise SystemExit(f"Unknown dict {args.dict}. Choose from: {list(DICT_MAP.keys())}")

    pairs_dir = Path(args.pairs_dir)
    pairs = load_pairs(pairs_dir)
    print(f"Found {len(pairs)} stereo pairs in {pairs_dir}")
    if not pairs:
        raise SystemExit(f"No pairs found in {pairs_dir} with left_####.(png/jpg) and right_####.(png/jpg)")

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length_m,
        args.marker_length_m,
        aruco_dict,
    )
    chessboard_corners_3d = board.getChessboardCorners()  # (Nc,3), float32

    obj_L, img_L = [], []
    obj_R, img_R = [], []
    obj_stereo, imgL_stereo, imgR_stereo = [], [], []

    image_size = None
    used = 0

    if args.show:
        cv2.namedWindow("detections", cv2.WINDOW_NORMAL)

    for lp, rp in pairs:
        L = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        R = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if L is None or R is None:
            continue

        gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gL.shape[1], gL.shape[0])

        chL, idL, mkcL, mkidL = detect_charuco(gL, aruco_dict, board)
        chR, idR, mkcR, mkidR = detect_charuco(gR, aruco_dict, board)
        if chL is None or chR is None:
            continue

        idsL = idL.flatten().astype(int)
        idsR = idR.flatten().astype(int)

        # Only keep corners that appear in BOTH images (important for stereo)
        common = np.intersect1d(idsL, idsR)
        if common.size < args.min_corners:
            continue

        # Map common IDs to per-image indices
        idxL = np.array([np.where(idsL == c)[0][0] for c in common], dtype=int)
        idxR = np.array([np.where(idsR == c)[0][0] for c in common], dtype=int)

        obj = chessboard_corners_3d[common].reshape(-1, 1, 3).astype(np.float64)
        imgpL = chL[idxL].reshape(-1, 1, 2).astype(np.float64)
        imgpR = chR[idxR].reshape(-1, 1, 2).astype(np.float64)

        # For individual intrinsics we can also use per-side points (using common is fine too)
        obj_L.append(obj)
        img_L.append(imgpL)
        obj_R.append(obj)
        img_R.append(imgpR)

        obj_stereo.append(obj)
        imgL_stereo.append(imgpL)
        imgR_stereo.append(imgpR)

        used += 1

        if args.show:
            vis = np.concatenate([L, R], axis=1)
            cv2.imshow("detections", vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    if args.show:
        cv2.destroyAllWindows()

    if used < 15:
        raise SystemExit(f"Too few good stereo frames ({used}). Capture more / improve coverage.")

    assert image_size is not None

    def fisheye_calibrate(objpoints, imgpoints, image_size):
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_size, K, D, None, None, flags=flags, criteria=criteria
        )
        return rms, K, D

    # 1) Intrinsics
    rmsL, K1, D1 = fisheye_calibrate(obj_L, img_L, image_size)
    rmsR, K2, D2 = fisheye_calibrate(obj_R, img_R, image_size)

    print(f"[Left]  RMS={rmsL:.6f}\nK1=\n{K1}\nD1={D1.ravel()}")
    print(f"[Right] RMS={rmsR:.6f}\nK2=\n{K2}\nD2={D2.ravel()}")

    # 2) Stereo extrinsics (R, T)
    # Fix intrinsics while optimizing extrinsics is typically most stable
    flags_stereo = cv2.fisheye.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    rms_stereo, K1o, D1o, K2o, D2o, R, T = cv2.fisheye.stereoCalibrate(
        objectPoints=obj_stereo,
        imagePoints1=imgL_stereo,
        imagePoints2=imgR_stereo,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        imageSize=image_size,
        R=None,
        T=None,
        flags=flags_stereo,
        criteria=criteria,
    )

    baseline_est = float(np.linalg.norm(T))
    print(f"[Stereo] RMS={rms_stereo:.6f}  |T|={baseline_est:.4f} m (expected ~{args.baseline_m:.4f} m)")

    # 3) Stereo rectify + maps
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, balance=args.balance, fov_scale=1.0
    )

    map1L, map2L = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1R, map2R = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = {
        "model": "opencv_fisheye_stereo_charuco",
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": float(args.square_length_m),
            "marker_length_m": float(args.marker_length_m),
            "dictionary": args.dict,
        },
        "left": {"K": K1.tolist(), "D": D1.tolist(), "rms": float(rmsL)},
        "right": {"K": K2.tolist(), "D": D2.tolist(), "rms": float(rmsR)},
        "stereo": {
            "rms": float(rms_stereo),
            "R": R.tolist(),
            "T": T.tolist(),
            "baseline_m": baseline_est,
        },
        "rectify": {
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "balance": float(args.balance),
        },
        # Maps can be big; save as .npz instead of JSON
        "maps_npz": "rectify_maps.npz",
        "frames_used": int(used),
        "frames_total": int(len(pairs)),
    }

    (out_dir / "stereo_fisheye_calibration.json").write_text(json.dumps(calib, indent=2))
    np.savez_compressed(
        out_dir / "rectify_maps.npz",
        map1L=map1L, map2L=map2L, map1R=map1R, map2R=map2R
    )

    print(f"Saved: {out_dir / 'stereo_fisheye_calibration.json'}")
    print(f"Saved: {out_dir / 'rectify_maps.npz'}")


if __name__ == "__main__":
    main()
