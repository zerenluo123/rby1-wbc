#!/usr/bin/env python3
"""
Fisheye calibration from ChArUco images using OpenCV.

Requires: opencv-contrib-python (for cv2.aruco)

Example:
  python calibrate_charuco_fisheye.py \
    --images "calib_imgs/*.png" \
    --out calib_fisheye_out \
    --squares_x 7 --squares_y 5 \
    --square_length_m 0.036 \
    --marker_length_m 0.027 \
    --dict DICT_5X5_1000 \
    --min_corners 12 \
    --show
"""

import argparse
import glob
import json
from pathlib import Path

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

    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
    )
    if n is None or n < 6 or ch_corners is None or ch_ids is None:
        return None, None, corners, ids

    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    cv2.cornerSubPix(gray, ch_corners, (5, 5), (-1, -1), criteria)

    return ch_corners, ch_ids, corners, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Glob for images, e.g. 'calib_imgs/*.png'")
    ap.add_argument("--out", default="calib_fisheye_out", help="Output directory")
    ap.add_argument("--squares_x", type=int, default=7)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_length_m", type=float, required=True)
    ap.add_argument("--marker_length_m", type=float, required=True)
    ap.add_argument("--dict", default="DICT_5X5_1000")
    ap.add_argument("--min_corners", type=int, default=12)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--fix_skew", action="store_true", help="Often reasonable to fix skew for most cameras")
    ap.add_argument("--fix_principal_point", action="store_true", help="Only if you really want to constrain cx,cy")
    ap.add_argument("--init_fx_fy", type=float, default=0.0, help="Optional initial focal (pixels). 0 = auto")
    args = ap.parse_args()

    if args.dict not in DICT_MAP:
        raise SystemExit(f"Unknown dict {args.dict}. Choose from: {list(DICT_MAP.keys())}")

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"No images matched: {args.images}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length_m,
        args.marker_length_m,
        aruco_dict,
    )

    # 3D locations of ALL ChArUco chessboard corners (in board coords)
    # shape: (Ncorners, 3), float32
    chessboard_corners_3d = board.getChessboardCorners()

    objpoints = []  # list of (N,1,3) float64
    imgpoints = []  # list of (N,1,2) float64
    image_size = None
    used = 0

    if args.show:
        cv2.namedWindow("detections", cv2.WINDOW_NORMAL)

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        ch_corners, ch_ids, mk_corners, mk_ids = detect_charuco(gray, aruco_dict, board)
        if ch_corners is None or ch_ids is None:
            continue

        # ch_ids is (N,1). Flatten to (N,)
        ids = ch_ids.flatten().astype(int)
        if len(ids) < args.min_corners:
            continue

        # Build 3D-2D correspondences by indexing the known 3D corner locations
        obj = chessboard_corners_3d[ids].reshape(-1, 1, 3).astype(np.float64)
        imgp = ch_corners.reshape(-1, 1, 2).astype(np.float64)

        objpoints.append(obj)
        imgpoints.append(imgp)
        used += 1

        if args.show:
            vis = img.copy()
            if mk_ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, mk_corners, mk_ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
            cv2.imshow("detections", vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    if args.show:
        cv2.destroyAllWindows()

    if used < 15:
        raise SystemExit(f"Too few good frames ({used}). Capture more / improve views.")

    assert image_size is not None

    # Initialize K, D
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    if args.init_fx_fy > 0:
        K[0, 0] = args.init_fx_fy
        K[1, 1] = args.init_fx_fy
        K[0, 2] = image_size[0] / 2.0
        K[1, 2] = image_size[1] / 2.0

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    if not args.fix_skew:
        flags &= ~cv2.fisheye.CALIB_FIX_SKEW
    if args.fix_principal_point:
        flags |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        image_size=image_size,
        K=K,
        D=D,
        rvecs=None,
        tvecs=None,
        flags=flags,
        criteria=criteria,
    )

    print(f"Used frames: {used}/{len(paths)}")
    print(f"Image size: {image_size}")
    print(f"Fisheye RMS reprojection error: {rms:.6f}")
    print("K:\n", K)
    print("D (k1,k2,k3,k4):\n", D.ravel())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": "opencv_fisheye_charuco",
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": float(args.square_length_m),
            "marker_length_m": float(args.marker_length_m),
            "dictionary": args.dict,
        },
        "K": K.tolist(),
        "D": D.tolist(),
        "rms": float(rms),
        "frames_used": int(used),
        "frames_total": int(len(paths)),
        "flags": int(flags),
    }
    (out_dir / "calibration_fisheye.json").write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_dir / 'calibration_fisheye.json'}")

    # Simple ROS-ish YAML (note: distortion_model differs from plumb_bob)
    yaml_txt = f"""image_width: {image_size[0]}
image_height: {image_size[1]}
camera_name: blackfly_s_fisheye
camera_matrix:
  rows: 3
  cols: 3
  data: [{K[0,0]}, {K[0,1]}, {K[0,2]},
         {K[1,0]}, {K[1,1]}, {K[1,2]},
         {K[2,0]}, {K[2,1]}, {K[2,2]}]
distortion_model: fisheye
distortion_coefficients:
  rows: 1
  cols: 4
  data: [{D[0,0]}, {D[1,0]}, {D[2,0]}, {D[3,0]}]
"""
    (out_dir / "camera_info_fisheye.yaml").write_text(yaml_txt)
    print(f"Saved: {out_dir / 'camera_info_fisheye.yaml'}")


if __name__ == "__main__":
    main()
