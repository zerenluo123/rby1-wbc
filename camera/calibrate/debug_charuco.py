#!/usr/bin/env python3
import cv2, numpy as np
from pathlib import Path
import argparse, glob

DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}

def detect(gray, board, aruco_dict):
    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rej = det.detectMarkers(gray)
    if ids is None:
        return 0, 0
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rej)
    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    nchar = int(n) if n is not None else 0
    return int(len(ids)), nchar

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob, e.g. 'stereo_calib_pairs/left_*.png'")
    ap.add_argument("--squares_x", type=int, required=True)
    ap.add_argument("--squares_y", type=int, required=True)
    ap.add_argument("--square_length_m", type=float, required=True)
    ap.add_argument("--marker_length_m", type=float, required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit("No images matched")

    for name, did in DICT_MAP.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(did)
        board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            args.square_length_m,
            args.marker_length_m,
            aruco_dict,
        )

        total_m, total_c, used = 0, 0, 0
        for p in paths[:80]:  # sample first 80
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            nm, nc = detect(img, board, aruco_dict)
            total_m += nm
            total_c += nc
            used += 1
        print(f"{name}: avg markers {total_m/max(1,used):.2f}, avg charuco corners {total_c/max(1,used):.2f}")
