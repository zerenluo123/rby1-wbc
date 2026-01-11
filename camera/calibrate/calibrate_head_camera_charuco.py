#!/usr/bin/env python3
"""Estimate head->camera extrinsics using a ChArUco board.

We assume the ChArUco board pose is known in the head-site frame (from CAD or
measurement). For each detection, we estimate the board pose in the camera
frame and solve:

    T_head_cam = T_head_board @ inv(T_cam_board)

The output is printed in the roll-pitch-yaw + translation format expected by
make_transform in rby1/frame_transforms.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "OpenCV (cv2) with aruco support is required. Install opencv-contrib-python."
    ) from exc


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
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class Sample:
    t_head_cam: np.ndarray


def _read_matrix(entry: object) -> np.ndarray:
    if isinstance(entry, dict) and "data" in entry:
        data = np.asarray(entry["data"], dtype=float)
        rows = entry.get("rows")
        cols = entry.get("cols")
        if rows and cols:
            return data.reshape(int(rows), int(cols))
        return data
    return np.asarray(entry, dtype=float)


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    if path.suffix.lower() in {".txt", ".dat"}:
        with path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Intrinsics text file is empty")
        values = [float(v) for v in lines[0].split()]
        if len(values) < 9:
            raise ValueError("First line must contain at least 9 values for K")
        camera_matrix = np.asarray(values[:9], dtype=float).reshape(3, 3)
        dist_coeffs = np.zeros(5, dtype=float)
    elif path.suffix.lower() == ".npz":
        data = np.load(str(path))
        camera_matrix = data.get("camera_matrix") or data.get("K")
        dist_coeffs = data.get("dist_coeffs") or data.get("distortion_coefficients") or data.get("D")
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        camera_matrix = _read_matrix(payload.get("camera_matrix") or payload.get("K"))
        dist_coeffs = _read_matrix(
            payload.get("dist_coeffs")
            or payload.get("distortion_coefficients")
            or payload.get("D")
            or []
        )
    else:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        camera_matrix = _read_matrix(payload.get("camera_matrix") or payload.get("K"))
        dist_coeffs = _read_matrix(
            payload.get("dist_coeffs")
            or payload.get("distortion_coefficients")
            or payload.get("D")
            or []
        )

    if camera_matrix is None or np.asarray(camera_matrix).shape != (3, 3):
        raise ValueError("camera_matrix/K must be a 3x3 matrix")
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=float)
    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1)

    return np.asarray(camera_matrix, dtype=float), dist_coeffs


def load_transform(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")
    if path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(str(path))
        if isinstance(data, np.lib.npyio.NpzFile):
            if "transform" in data:
                return np.asarray(data["transform"], dtype=float)
            if "T" in data:
                return np.asarray(data["T"], dtype=float)
            raise ValueError("NPZ must contain 'transform' or 'T' array")
        return np.asarray(data, dtype=float)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if isinstance(payload, dict) and "transform" in payload:
        payload = payload["transform"]
    return np.asarray(payload, dtype=float)


def load_transform_sequence(path: Path) -> List[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Transform list not found: {path}")

    if path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(str(path))
        if isinstance(data, np.lib.npyio.NpzFile):
            if "poses" in data:
                arr = np.asarray(data["poses"], dtype=float)
            elif "transforms" in data:
                arr = np.asarray(data["transforms"], dtype=float)
            else:
                raise ValueError("NPZ must contain 'poses' or 'transforms'")
        else:
            arr = np.asarray(data, dtype=float)
    else:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if isinstance(payload, dict):
            payload = payload.get("poses") or payload.get("transforms") or payload
        arr = np.asarray(payload, dtype=float)

    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError("Expected a list/array with shape (N, 4, 4)")
    return [np.asarray(mat, dtype=float) for mat in arr]


def build_transform(rpy: Sequence[float], xyz: Sequence[float]) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()
    mat[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return mat


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return mat


def rotation_mean(rotations: List[Rotation]) -> Rotation:
    quats = np.stack([rot.as_quat() for rot in rotations], axis=0)
    mean = np.sum(quats, axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        return rotations[0]
    mean /= norm
    return Rotation.from_quat(mean)


def average_transforms(transforms: List[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("No transforms to average")
    translations = np.stack([t[:3, 3] for t in transforms], axis=0)
    rotations = [Rotation.from_matrix(t[:3, :3]) for t in transforms]
    avg_rot = rotation_mean(rotations)
    avg = np.eye(4, dtype=float)
    avg[:3, :3] = avg_rot.as_matrix()
    avg[:3, 3] = np.mean(translations, axis=0)
    return avg


def compute_residuals(avg: np.ndarray, transforms: List[np.ndarray]) -> Tuple[float, float]:
    if not transforms:
        return 0.0, 0.0
    rot_errors = []
    trans_errors = []
    avg_inv = np.linalg.inv(avg)
    for t in transforms:
        delta = avg_inv @ t
        rot = Rotation.from_matrix(delta[:3, :3])
        angle = rot.magnitude()
        rot_errors.append(float(angle))
        trans_errors.append(float(np.linalg.norm(delta[:3, 3])))
    return float(np.mean(rot_errors)), float(np.mean(trans_errors))


def split_rt(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return transform[:3, :3].copy(), transform[:3, 3].copy()


def hand_eye_calibrate(
    world_to_head_list: List[np.ndarray],
    cam_to_board_list: List[np.ndarray],
    method: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(world_to_head_list) != len(cam_to_board_list):
        raise ValueError("Head pose and board pose counts must match")
    if len(world_to_head_list) < 3:
        raise ValueError("Need at least 3 samples for hand-eye calibration")

    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []
    for world_to_head, cam_to_board in zip(world_to_head_list, cam_to_board_list):
        head_to_world = np.linalg.inv(world_to_head)
        board_to_cam = np.linalg.inv(cam_to_board)
        r_g2b, t_g2b = split_rt(head_to_world)
        r_t2c, t_t2c = split_rt(board_to_cam)
        r_gripper2base.append(r_g2b)
        t_gripper2base.append(t_g2b)
        r_target2cam.append(r_t2c)
        t_target2cam.append(t_t2c)

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    cam_to_head = np.eye(4, dtype=float)
    cam_to_head[:3, :3] = r_cam2gripper
    cam_to_head[:3, 3] = np.asarray(t_cam2gripper, dtype=float).reshape(3)
    head_to_cam = np.linalg.inv(cam_to_head)

    world_to_board = []
    for world_to_head, cam_to_board in zip(world_to_head_list, cam_to_board_list):
        world_to_board.append(world_to_head @ head_to_cam @ cam_to_board)
    world_to_board_avg = average_transforms(world_to_board)

    return head_to_cam, world_to_board_avg


def format_rpy_xyz(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_matrix(transform[:3, :3])
    rpy = rot.as_euler("xyz", degrees=False)
    xyz = transform[:3, 3]
    return rpy, xyz


def iter_frames_from_video(path: Path) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def iter_frames_from_images(paths: Sequence[Path]) -> Iterable[np.ndarray]:
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        yield image


def iter_frames_from_device(device_index: int) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open device: {device_index}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def iter_frames_from_aravis(config_path: Path, camera_key: str) -> Iterable[np.ndarray]:
    from camera.camera_stream import AravisCameraStreamer

    streamer = AravisCameraStreamer(str(config_path))
    streamer.start()
    try:
        streamer.wait_until_ready(min_frames=1, timeout=5.0)
        while True:
            obs = streamer.get_observation_window(horizon=1)
            if camera_key not in obs:
                raise KeyError(f"Camera key '{camera_key}' not found in config")
            frame = obs[camera_key][0]
            yield frame
    finally:
        streamer.stop()


def detect_charuco_pose(
    image: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    board: cv2.aruco.CharucoBoard,
    detector_params: cv2.aruco.DetectorParameters,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_corners: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None

    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
    )
    if n is None or ch_corners is None or ch_ids is None or int(n) < min_corners:
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners,
        ch_ids,
        board,
        camera_matrix,
        dist_coeffs,
        None,
        None,
    )
    if not ok:
        return None

    cam_from_board = rvec_tvec_to_matrix(rvec, tvec)
    return cam_from_board, ch_corners, ch_ids, corners, ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate head->camera extrinsics using a ChArUco board."
    )
    parser.add_argument(
        "--mode",
        choices=("head_tag", "world_tag"),
        default="head_tag",
        help="head_tag: board pose in head is known. world_tag: board fixed in world (hand-eye).",
    )
    parser.add_argument("--intrinsics", type=Path, required=True, help="Path to camera intrinsics")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length-m", type=float, required=True)
    parser.add_argument("--marker-length-m", type=float, required=True)
    parser.add_argument("--dict", default="DICT_5X5_1000")
    parser.add_argument("--min-corners", type=int, default=12)

    parser.add_argument("--head-to-board-rpy", type=float, nargs=3, default=None)
    parser.add_argument("--head-to-board-xyz", type=float, nargs=3, default=None)
    parser.add_argument("--head-to-board-mat", type=Path, default=None)
    parser.add_argument(
        "--head-poses",
        type=Path,
        default=None,
        help="Path to world->head pose list for hand-eye mode (Nx4x4).",
    )
    parser.add_argument(
        "--handeye-method",
        type=str,
        default="TSAI",
        help="Hand-eye method: TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS.",
    )

    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-period", type=float, default=0.2)
    parser.add_argument("--display", action="store_true", help="Show detections in a window")

    parser.add_argument("--camera-config", type=Path, default=Path("config/camera.yaml"))
    parser.add_argument("--camera-key", type=str, default="camera_head_main_rgb")

    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--image-glob", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)

    parser.add_argument("--output", type=Path, default=None, help="Write YAML with result")

    args = parser.parse_args()

    head_to_board = None
    if args.mode == "head_tag":
        head_to_board = load_transform(args.head_to_board_mat)
        if head_to_board is None:
            if args.head_to_board_rpy is None or args.head_to_board_xyz is None:
                raise SystemExit(
                    "Provide --head-to-board-mat or both --head-to-board-rpy and --head-to-board-xyz."
                )
            head_to_board = build_transform(args.head_to_board_rpy, args.head_to_board_xyz)
    else:
        if args.head_poses is None:
            raise SystemExit("--head-poses is required for --mode world_tag.")
        head_poses = load_transform_sequence(args.head_poses)
        if not head_poses:
            raise SystemExit("No head poses loaded.")

    if args.dict not in DICT_MAP:
        raise SystemExit(f"Unknown dict {args.dict}. Choose from: {list(DICT_MAP.keys())}")

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length_m,
        args.marker_length_m,
        aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()

    if args.video:
        frame_iter = iter_frames_from_video(args.video)
    elif args.image_glob:
        paths = sorted(Path().glob(args.image_glob))
        if not paths:
            raise SystemExit(f"No images matched glob: {args.image_glob}")
        frame_iter = iter_frames_from_images(paths)
    elif args.device is not None:
        frame_iter = iter_frames_from_device(args.device)
    else:
        frame_iter = iter_frames_from_aravis(args.camera_config, args.camera_key)

    samples: List[Sample] = []
    cam_to_board_samples: List[np.ndarray] = []
    world_to_head_samples: List[np.ndarray] = []
    last_sample_time = 0.0

    if args.display:
        cv2.namedWindow("charuco", cv2.WINDOW_NORMAL)

    for frame in frame_iter:
        now = time.monotonic()
        if args.num_samples and len(samples) >= args.num_samples:
            break
        if now - last_sample_time < args.sample_period:
            continue

        detection = detect_charuco_pose(
            frame,
            aruco_dict,
            board,
            detector_params,
            camera_matrix,
            dist_coeffs,
            args.min_corners,
        )
        if detection is None:
            if args.display:
                cv2.imshow("charuco", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        cam_from_board, ch_corners, ch_ids, mk_corners, mk_ids = detection
        if args.mode == "head_tag":
            head_to_cam = head_to_board @ np.linalg.inv(cam_from_board)
            samples.append(Sample(t_head_cam=head_to_cam))
        else:
            pose_idx = len(cam_to_board_samples)
            if pose_idx >= len(head_poses):
                break
            cam_to_board_samples.append(cam_from_board)
            world_to_head_samples.append(head_poses[pose_idx])
        last_sample_time = now

        if args.display:
            vis = frame.copy()
            if mk_ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, mk_corners, mk_ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
            cv2.imshow("charuco", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.display:
        cv2.destroyAllWindows()

    if args.mode == "head_tag":
        if not samples:
            raise SystemExit("No valid ChArUco detections; no calibration computed.")

        transforms = [s.t_head_cam for s in samples]
        avg = average_transforms(transforms)
        mean_rot_err, mean_trans_err = compute_residuals(avg, transforms)
        rpy, xyz = format_rpy_xyz(avg)

        print("Estimated head->camera transform:")
        print("- rpy (rad):", np.array2string(rpy, precision=6))
        print("- xyz (m): ", np.array2string(xyz, precision=6))
        print("- mean rot error (rad):", f"{mean_rot_err:.6f}")
        print("- mean trans error (m):", f"{mean_trans_err:.6f}")
        print()
        print("Drop into rby1/rby1/frame_transforms.py as:")
        print("MODEL_TO_TCP_FRAME['head'] = make_transform(")
        print(f"    [{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}],")
        print(f"    [{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}],")
        print(")")
    else:
        if not cam_to_board_samples:
            raise SystemExit("No valid ChArUco detections; no calibration computed.")
        method_name = args.handeye_method.upper()
        method_map = {
            "TSAI": cv2.CALIB_HAND_EYE_TSAI,
            "PARK": cv2.CALIB_HAND_EYE_PARK,
            "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
            "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
            "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        if method_name not in method_map:
            raise SystemExit(f"Unknown hand-eye method: {args.handeye_method}")
        head_to_cam, world_to_board = hand_eye_calibrate(
            world_to_head_samples,
            cam_to_board_samples,
            method_map[method_name],
        )
        rpy, xyz = format_rpy_xyz(head_to_cam)
        board_rpy, board_xyz = format_rpy_xyz(world_to_board)
        print("Estimated head->camera transform (hand-eye):")
        print("- rpy (rad):", np.array2string(rpy, precision=6))
        print("- xyz (m): ", np.array2string(xyz, precision=6))
        print()
        print("Estimated world->board transform:")
        print("- rpy (rad):", np.array2string(board_rpy, precision=6))
        print("- xyz (m): ", np.array2string(board_xyz, precision=6))
        print()
        print("Drop into rby1/rby1/frame_transforms.py as:")
        print("MODEL_TO_TCP_FRAME['head'] = make_transform(")
        print(f"    [{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}],")
        print(f"    [{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}],")
        print(")")

    if args.output is not None:
        payload = {
            "model_to_tcp_head": {
                "rpy": [float(v) for v in rpy],
                "xyz": [float(v) for v in xyz],
                "matrix": head_to_cam.tolist() if args.mode == "world_tag" else avg.tolist(),
            }
        }
        if args.mode == "head_tag":
            payload["model_to_tcp_head"]["mean_rot_error_rad"] = float(mean_rot_err)
            payload["model_to_tcp_head"]["mean_trans_error_m"] = float(mean_trans_err)
        else:
            payload["world_to_board"] = {
                "rpy": [float(v) for v in board_rpy],
                "xyz": [float(v) for v in board_xyz],
                "matrix": world_to_board.tolist(),
            }
        with args.output.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"Wrote calibration to {args.output}")


if __name__ == "__main__":
    main()
