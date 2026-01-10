#!/usr/bin/env python3
"""
Calibrate a camera using ChArUco images with OpenCV (pinhole model).

Example:
  python calibrate_charuco_from_images.py \
    --images "calib_imgs/*.png" \
    --out calib_out \
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
        board=board
    )
    if n is None or n < 6 or ch_corners is None or ch_ids is None:
        return None, None, corners, ids

    # subpixel refine charuco corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    cv2.cornerSubPix(gray, ch_corners, (5, 5), (-1, -1), criteria)

    return ch_corners, ch_ids, corners, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Glob pattern for images, e.g. 'calib_imgs/*.png'")
    ap.add_argument("--out", default="calib_out", help="Output directory")
    ap.add_argument("--squares_x", type=int, default=7)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_length_m", type=float, required=True)
    ap.add_argument("--marker_length_m", type=float, required=True)
    ap.add_argument("--dict", default="DICT_5X5_1000")
    ap.add_argument("--min_corners", type=int, default=12)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--use_rational_model", action="store_true", help="Enable k4,k5,k6 (sometimes helps wide lenses)")
    args = ap.parse_args()

    if args.dict not in DICT_MAP:
        raise SystemExit(f"Unknown dict {args.dict}. Choose from: {list(DICT_MAP.keys())}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length_m,
        args.marker_length_m,
        aruco_dict
    )

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"No images matched {args.images}")

    all_corners, all_ids = [], []
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
        if len(ch_ids) < args.min_corners:
            continue

        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
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
        raise SystemExit(f"Too few good frames ({used}). Capture more/better images.")

    flags = 0
    if args.use_rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
    )

    print(f"Used frames: {used}/{len(paths)}")
    print(f"Image size: {image_size}")
    print(f"RMS reprojection error: {rms:.6f}")
    print("K:\n", K)
    print("dist:\n", dist.ravel())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": "opencv_pinhole_charuco",
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length_m,
            "marker_length_m": args.marker_length_m,
            "dictionary": args.dict,
        },
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rms": float(rms),
        "frames_used": int(used),
        "frames_total": int(len(paths)),
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_dir / 'calibration.json'}")

    # ROS-style camera_info.yaml (plumb_bob)
    d = dist.ravel().tolist()
    yaml_txt = f"""image_width: {image_size[0]}
image_height: {image_size[1]}
camera_name: blackfly_s_charuco
camera_matrix:
  rows: 3
  cols: 3
  data: [{K[0,0]}, {K[0,1]}, {K[0,2]},
         {K[1,0]}, {K[1,1]}, {K[1,2]},
         {K[2,0]}, {K[2,1]}, {K[2,2]}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {len(d)}
  data: {d}
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0,
         0, 1, 0,
         0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [{K[0,0]}, 0, {K[0,2]}, 0,
         0, {K[1,1]}, {K[1,2]}, 0,
         0, 0, 1, 0]
"""
    (out_dir / "camera_info.yaml").write_text(yaml_txt)
    print(f"Saved: {out_dir / 'camera_info.yaml'}")


if __name__ == "__main__":
    main()
