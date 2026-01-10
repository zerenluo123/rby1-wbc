#!/usr/bin/env python3
"""Estimate the head camera pose relative to the model's head site using ArUco.

This script detects a single ArUco marker and uses the known pose of that
marker in the head-site frame to solve for the camera extrinsics:

    T_head_cam = T_head_tag @ inv(T_cam_tag)

Where T_cam_tag comes from OpenCV ArUco pose estimation, and T_head_tag is
provided by the user (measured or from CAD).

In hand-eye mode ("world_tag"), the tag is fixed in the world and the head
pose in world is provided per sample. The script solves for the head->camera
transform and also estimates the world->tag transform. Optionally, this script
can command the head look-at targets directly and record those head poses.

The output is printed in the roll-pitch-yaw + translation format expected by
make_transform in rby1/frame_transforms.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

try:
    import cv2
    from cv2 import aruco
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "OpenCV (cv2) with aruco support is required. Install opencv-contrib-python."
    ) from exc


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
        # Match policy_server.py convention: line 2 is baseline, no distortion.
        dist_coeffs = np.zeros(5, dtype=float)
    elif path.suffix.lower() == ".npz":
        data = np.load(str(path))
        camera_matrix = data.get("camera_matrix") or data.get("K")
        dist_coeffs = data.get("dist_coeffs") or data.get("distortion_coefficients") or data.get("D")
    elif path.suffix.lower() in {".json"}:
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
    cam_to_tag_list: List[np.ndarray],
    method: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(world_to_head_list) != len(cam_to_tag_list):
        raise ValueError("Head pose and tag pose counts must match")
    if len(world_to_head_list) < 3:
        raise ValueError("Need at least 3 samples for hand-eye calibration")

    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []
    for world_to_head, cam_to_tag in zip(world_to_head_list, cam_to_tag_list):
        head_to_world = np.linalg.inv(world_to_head)
        tag_to_cam = np.linalg.inv(cam_to_tag)
        r_g2b, t_g2b = split_rt(head_to_world)
        r_t2c, t_t2c = split_rt(tag_to_cam)
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

    world_to_tag = []
    for world_to_head, cam_to_tag in zip(world_to_head_list, cam_to_tag_list):
        world_to_tag.append(world_to_head @ head_to_cam @ cam_to_tag)
    world_to_tag_avg = average_transforms(world_to_tag)

    return head_to_cam, world_to_tag_avg


def get_aruco_dict(name: str) -> aruco.Dictionary:
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def collect_handeye_samples_from_robot(
    config_path: Path,
    use_sim: bool,
    num_yaw: int,
    num_pitch: int,
    yaw_range_deg: float,
    pitch_range_deg: float,
    distance: float,
    settle_time: float,
    sample_delay: float,
    samples_per_target: int,
    shuffle: bool,
    relative_to_head: bool,
    frame_iter: Iterable[np.ndarray],
    aruco_dict: aruco.Dictionary,
    detector_params: aruco.DetectorParameters,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    marker_id: Optional[int],
    display: bool,
    detect_timeout: float,
    scan_dicts: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    import signal as _signal
    from control.rby1_policy import RBY1PolicyRobot
    from rby1.frame_transforms import MODEL_TO_TCP_FRAME

    def _install_sigint_handler(stop_flag: dict[str, bool]) -> None:
        def _handler(signum: int, frame: object | None) -> None:
            stop_flag["stop"] = True

        _signal.signal(_signal.SIGINT, _handler)
        _signal.signal(_signal.SIGTERM, _handler)

    def _build_targets() -> List[np.ndarray]:
        yaw_vals = np.linspace(-yaw_range_deg / 2.0, yaw_range_deg / 2.0, num=max(num_yaw, 1))
        pitch_vals = np.linspace(
            -pitch_range_deg / 2.0, pitch_range_deg / 2.0, num=max(num_pitch, 1)
        )
        points: List[np.ndarray] = []
        for yaw_deg in yaw_vals:
            for pitch_deg in pitch_vals:
                yaw = math.radians(float(yaw_deg))
                pitch = math.radians(float(pitch_deg))
                direction = np.array(
                    [
                        math.cos(pitch) * math.cos(yaw),
                        math.cos(pitch) * math.sin(yaw),
                        math.sin(pitch),
                    ],
                    dtype=float,
                )
                if relative_to_head:
                    world_dir = head_tcp_tf[:3, :3] @ direction
                else:
                    world_dir = direction
                points.append(center + distance * world_dir)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(points)
        return points

    robot = RBY1PolicyRobot(config_path=str(config_path), use_sim=use_sim)
    robot.start()
    robot.wait_until_ready(timeout=10.0)

    robot.wait_for_observations(count=2, timeout=2.0)
    obs = robot.get_latest_observation()
    if obs is None:
        robot.stop()
        raise RuntimeError("No robot observations available.")
    left_tf = obs.left_tf
    right_tf = obs.right_tf
    left_width = float(obs.left_width)
    right_width = float(obs.right_width)
    head_tf = obs.head_tf
    head_tcp_tf = head_tf @ MODEL_TO_TCP_FRAME["head"]
    center = head_tcp_tf[:3, 3].copy()

    targets = _build_targets()
    poses: List[np.ndarray] = []
    cam_to_tag_samples: List[np.ndarray] = []
    stop_flag = {"stop": False}
    _install_sigint_handler(stop_flag)

    dt = max(robot.dt, 0.02)
    for target_idx, lookat in enumerate(targets, start=1):
        if stop_flag["stop"]:
            break
        start = time.monotonic()
        while not stop_flag["stop"] and (time.monotonic() - start) < settle_time:
            tick = time.monotonic()
            payload = {
                "gripper_left_tf": left_tf,
                "gripper_right_tf": right_tf,
                "gripper_left_gripper_width": left_width,
                "gripper_right_gripper_width": right_width,
                "camera_head_lookatpoint": lookat,
            }
            robot.apply_action(payload, duration=dt, timestamp=tick)
            sleep_dt = dt - (time.monotonic() - tick)
            if sleep_dt > 0:
                time.sleep(min(sleep_dt, dt))

        time.sleep(max(sample_delay, 0.0))
        collected = 0
        detect_deadline = time.monotonic() + max(detect_timeout, 0.1)
        while (
            not stop_flag["stop"]
            and collected < max(samples_per_target, 1)
            and time.monotonic() <= detect_deadline
        ):
            try:
                frame = next(frame_iter)
            except StopIteration:
                stop_flag["stop"] = True
                break
            if scan_dicts:
                detection = scan_common_dictionaries(
                    frame,
                    detector_params,
                    camera_matrix,
                    dist_coeffs,
                    marker_length,
                )
                if detection is not None and display:
                    _, size_px, _, dict_name, detected_id = detection
                    cv2.putText(
                        frame,
                        f"{dict_name} id={detected_id} size={size_px:.1f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                detection = detect_pose(
                    frame,
                    aruco_dict,
                    detector_params,
                    camera_matrix,
                    dist_coeffs,
                    marker_length,
                    marker_id,
                )
                if detection is not None and display:
                    _, size_px, _, detected_id = detection
                    cv2.putText(
                        frame,
                        f"id={detected_id} size={size_px:.1f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            if display:
                cv2.imshow("aruco", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_flag["stop"] = True
                    break
            if detection is None:
                continue
            if scan_dicts:
                cam_to_tag = detection[0]
                print(f"[handeye] detected {detection[3]} id={detection[4]}")
            else:
                cam_to_tag = detection[0]
            obs = robot.get_latest_observation()
            if obs is None:
                continue
            head_tf = obs.head_tf
            if not np.isfinite(head_tf).all():
                continue
            poses.append(head_tf)
            cam_to_tag_samples.append(cam_to_tag)
            collected += 1
            print(f"[handeye] target {target_idx}/{len(targets)} detection {collected}/{samples_per_target}")
            time.sleep(max(sample_delay, 0.0))
        if collected < max(samples_per_target, 1):
            print(
                f"[handeye] target {target_idx}/{len(targets)} timed out: "
                f"{collected}/{samples_per_target} detections"
            )

    robot.stop()
    if display:
        cv2.destroyAllWindows()
    return poses, cam_to_tag_samples


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


def detect_pose(
    image: np.ndarray,
    aruco_dict: aruco.Dictionary,
    detector_params: aruco.DetectorParameters,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    marker_id: Optional[int],
) -> Optional[Tuple[np.ndarray, float, np.ndarray, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    if ids is None or len(ids) == 0:
        return None
    ids = ids.flatten()
    if marker_id is not None:
        matches = np.where(ids == marker_id)[0]
        if len(matches) == 0:
            return None
        idx = int(matches[0])
    else:
        idx = 0

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs
    )
    rvec = rvecs[idx].reshape(3)
    tvec = tvecs[idx].reshape(3)
    marker_corners = corners[idx].reshape(4, 2)
    edges = np.linalg.norm(marker_corners - np.roll(marker_corners, -1, axis=0), axis=1)
    size_px = float(np.mean(edges))
    return rvec_tvec_to_matrix(rvec, tvec), size_px, marker_corners, int(ids[idx])


def scan_common_dictionaries(
    image: np.ndarray,
    detector_params: aruco.DetectorParameters,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
) -> Optional[Tuple[np.ndarray, float, np.ndarray, str, int]]:
    dict_names = [
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_4X4_1000",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_6X6_50",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_50",
        "DICT_7X7_100",
        "DICT_7X7_250",
        "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
    ]
    best = None
    for name in dict_names:
        if not hasattr(aruco, name):
            continue
        aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, name))
        detection = detect_pose(
            image,
            aruco_dict,
            detector_params,
            camera_matrix,
            dist_coeffs,
            marker_length,
            marker_id=None,
        )
        if detection is None:
            continue
        cam_to_tag, size_px, corners, detected_id = detection
        if best is None or size_px > best[1]:
            best = (cam_to_tag, size_px, corners, name, detected_id)
    return best


def format_rpy_xyz(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_matrix(transform[:3, :3])
    rpy = rot.as_euler("xyz", degrees=False)
    xyz = transform[:3, 3]
    return rpy, xyz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate head camera pose relative to the head site using ArUco."
    )
    parser.add_argument(
        "--mode",
        choices=("head_tag", "world_tag"),
        default="head_tag",
        help="head_tag: tag fixed on head. world_tag: tag fixed in world (hand-eye).",
    )
    parser.add_argument("--intrinsics", type=Path, required=True, help="Path to camera intrinsics")
    parser.add_argument("--marker-size", type=float, required=True, help="Marker size in meters")
    parser.add_argument("--marker-id", type=int, default=None, help="Specific marker ID to use")
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name",
    )

    parser.add_argument("--head-to-tag-rpy", type=float, nargs=3, default=None)
    parser.add_argument("--head-to-tag-xyz", type=float, nargs=3, default=None)
    parser.add_argument("--head-to-tag-mat", type=Path, default=None)
    parser.add_argument(
        "--head-poses",
        type=Path,
        default=None,
        help="Path to world->head pose list for hand-eye mode (Nx4x4).",
    )
    parser.add_argument(
        "--sample-head-poses",
        action="store_true",
        help="Sample world->head poses by commanding head look-at targets.",
    )
    parser.add_argument("--head-config", type=Path, default=Path("config/wbc.yaml"))
    parser.add_argument("--head-use-sim", action="store_true")
    parser.add_argument("--head-num-yaw", type=int, default=5)
    parser.add_argument("--head-num-pitch", type=int, default=3)
    parser.add_argument("--head-yaw-range-deg", type=float, default=40.0)
    parser.add_argument("--head-pitch-range-deg", type=float, default=30.0)
    parser.add_argument("--head-distance", type=float, default=1.0)
    parser.add_argument("--head-settle-time", type=float, default=0.6)
    parser.add_argument("--head-sample-delay", type=float, default=0.05)
    parser.add_argument("--head-samples-per-target", type=int, default=1)
    parser.add_argument("--head-shuffle", action="store_true")
    parser.add_argument(
        "--head-absolute",
        action="store_false",
        dest="head_relative",
        help="Sweep in world axes instead of around current head orientation.",
    )
    parser.add_argument("--head-detect-timeout", type=float, default=3.0)
    parser.add_argument(
        "--scan-dicts",
        action="store_true",
        help="Scan common ArUco dictionaries and print detected family/id.",
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

    head_to_tag = None
    if args.mode == "head_tag":
        head_to_tag = load_transform(args.head_to_tag_mat)
        if head_to_tag is None:
            if args.head_to_tag_rpy is None or args.head_to_tag_xyz is None:
                raise SystemExit(
                    "Provide --head-to-tag-mat or both --head-to-tag-rpy and --head-to-tag-xyz."
                )
            head_to_tag = build_transform(args.head_to_tag_rpy, args.head_to_tag_xyz)
    else:
        if args.head_poses is None and not args.sample_head_poses:
            raise SystemExit("--head-poses is required for --mode world_tag (or use --sample-head-poses).")

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)
    aruco_dict = get_aruco_dict(args.aruco_dict)
    detector_params = aruco.DetectorParameters()
    head_poses = None
    if args.mode == "world_tag":
        if not args.sample_head_poses:
            head_poses = load_transform_sequence(args.head_poses)
            if not head_poses:
                raise SystemExit("No head poses loaded.")

    samples: List[Sample] = []
    last_sample_time = 0.0
    cam_to_tag_samples: List[np.ndarray] = []
    world_to_head_samples: List[np.ndarray] = []

    if args.mode == "world_tag" and args.sample_head_poses:
        if args.video or args.image_glob:
            raise SystemExit("--sample-head-poses requires a live camera stream.")
        if args.device is not None:
            frame_iter = iter_frames_from_device(args.device)
        else:
            frame_iter = iter_frames_from_aravis(args.camera_config, args.camera_key)
        head_poses, cam_to_tag_samples = collect_handeye_samples_from_robot(
            args.head_config,
            args.head_use_sim,
            args.head_num_yaw,
            args.head_num_pitch,
            args.head_yaw_range_deg,
            args.head_pitch_range_deg,
            args.head_distance,
            args.head_settle_time,
            args.head_sample_delay,
            args.head_samples_per_target,
            args.head_shuffle,
            args.head_relative,
            frame_iter,
            aruco_dict,
            detector_params,
            camera_matrix,
            dist_coeffs,
            args.marker_size,
            args.marker_id,
            args.display,
            args.head_detect_timeout,
            args.scan_dicts,
        )
        world_to_head_samples = head_poses
    else:
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

        for frame in frame_iter:
            now = time.monotonic()
            if args.num_samples and len(samples) >= args.num_samples:
                break
            if now - last_sample_time < args.sample_period:
                continue

            if args.scan_dicts:
                detection = scan_common_dictionaries(
                    frame,
                    detector_params,
                    camera_matrix,
                    dist_coeffs,
                    args.marker_size,
                )
            else:
                detection = detect_pose(
                    frame,
                    aruco_dict,
                    detector_params,
                    camera_matrix,
                    dist_coeffs,
                    args.marker_size,
                    args.marker_id,
                )
            if detection is None:
                if args.display:
                    cv2.imshow("aruco", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            if args.scan_dicts:
                cam_to_tag, size_px, _, dict_name, detected_id = detection
                if args.display:
                    cv2.putText(
                        frame,
                        f"{dict_name} id={detected_id} size={size_px:.1f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                print(f"[aruco] detected {dict_name} id={detected_id}")
            else:
                cam_to_tag, size_px, _, detected_id = detection
                if args.display:
                    cv2.putText(
                        frame,
                        f"id={detected_id} size={size_px:.1f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            if args.mode == "head_tag":
                head_to_cam = head_to_tag @ np.linalg.inv(cam_to_tag)
                samples.append(Sample(t_head_cam=head_to_cam))
            else:
                pose_idx = len(cam_to_tag_samples)
                if head_poses is None or pose_idx >= len(head_poses):
                    break
                cam_to_tag_samples.append(cam_to_tag)
                world_to_head_samples.append(head_poses[pose_idx])
            last_sample_time = now

            if args.display:
                cv2.imshow("aruco", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if args.display:
            cv2.destroyAllWindows()

    if args.mode == "head_tag":
        if not samples:
            raise SystemExit("No valid ArUco detections; no calibration computed.")
        transforms = [s.t_head_cam for s in samples]
        avg = average_transforms(transforms)
        mean_rot_err, mean_trans_err = compute_residuals(avg, transforms)
        rpy, xyz = format_rpy_xyz(avg)

        print("Estimated MODEL_TO_TCP['head'] transform:")
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
        if not cam_to_tag_samples:
            raise SystemExit("No valid ArUco detections; no calibration computed.")
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
        head_to_cam, world_to_tag = hand_eye_calibrate(
            world_to_head_samples,
            cam_to_tag_samples,
            method_map[method_name],
        )
        rpy, xyz = format_rpy_xyz(head_to_cam)
        tag_rpy, tag_xyz = format_rpy_xyz(world_to_tag)
        print("Estimated MODEL_TO_TCP['head'] transform (hand-eye):")
        print("- rpy (rad):", np.array2string(rpy, precision=6))
        print("- xyz (m): ", np.array2string(xyz, precision=6))
        print()
        print("Estimated world->tag transform:")
        print("- rpy (rad):", np.array2string(tag_rpy, precision=6))
        print("- xyz (m): ", np.array2string(tag_xyz, precision=6))
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
            payload["world_to_tag"] = {
                "rpy": [float(v) for v in tag_rpy],
                "xyz": [float(v) for v in tag_xyz],
                "matrix": world_to_tag.tolist(),
            }
        with args.output.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"Wrote calibration to {args.output}")


if __name__ == "__main__":
    main()
