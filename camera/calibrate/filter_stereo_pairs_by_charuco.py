#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}

def count_charuco(gray, board, aruco_dict):
    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rej = det.detectMarkers(gray)
    if ids is None:
        return 0
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rej)
    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    return int(n) if n is not None else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--squares_x", type=int, required=True)
    ap.add_argument("--squares_y", type=int, required=True)
    ap.add_argument("--square_length_m", type=float, required=True)
    ap.add_argument("--marker_length_m", type=float, required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--min_left", type=int, default=20)
    ap.add_argument("--min_right", type=int, default=20)
    ap.add_argument("--max_pairs", type=int, default=80)
    args = ap.parse_args()

    if args.dict not in DICT_MAP:
        raise SystemExit(f"Unknown dict {args.dict}. Choose from: {list(DICT_MAP.keys())}")

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y),
                                   args.square_length_m, args.marker_length_m, aruco_dict)

    lefts = sorted(pairs_dir.glob("left_*.png")) + sorted(pairs_dir.glob("left_*.jpg"))
    scored = []

    for lp in lefts:
        idx = lp.stem.split("_")[-1]
        rp = None
        for ext in ("png", "jpg"):
            cand = pairs_dir / f"right_{idx}.{ext}"
            if cand.exists():
                rp = cand
                break
        if rp is None:
            continue

        L = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            continue

        nL = count_charuco(L, board, aruco_dict)
        nR = count_charuco(R, board, aruco_dict)

        if nL >= args.min_left and nR >= args.min_right:
            # score by total corners
            scored.append((nL + nR, nL, nR, lp, rp))

    scored.sort(reverse=True, key=lambda x: x[0])
    keep = scored[: args.max_pairs]

    print(f"Found {len(scored)} pairs meeting thresholds (L>={args.min_left}, R>={args.min_right}). Keeping {len(keep)} best.")

    for i, (tot, nL, nR, lp, rp) in enumerate(keep):
        new_idx = f"{i:04d}"
        shutil.copy2(lp, out_dir / f"left_{new_idx}{lp.suffix}")
        shutil.copy2(rp, out_dir / f"right_{new_idx}{rp.suffix}")

    if keep:
        best = keep[0]
        print(f"Best pair corners: total={best[0]} left={best[1]} right={best[2]}")
    print(f"Wrote filtered pairs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
