"""Render dataset action trajectories using whole-body IK."""

from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
    raise ModuleNotFoundError(
        "mujoco is required for rendering. Please install mujoco in the current environment."
    ) from exc

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
    raise ModuleNotFoundError(
        "opencv-python is required for image export. Please install it in the current environment."
    ) from exc

try:
    import hydra
    from omegaconf import OmegaConf
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
    raise ModuleNotFoundError(
        "hydra-core and omegaconf are required for dataset loading."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to make umi_day importable when running from rby1 checkout.
UMI_ROOT = PROJECT_ROOT.parents[0] / "umi_day"
if UMI_ROOT.exists() and str(UMI_ROOT) not in sys.path:
    sys.path.append(str(UMI_ROOT))

try:
    from umi.common.pose_util import pose10d_to_mat, pose_to_mat
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    added = False
    for parent in UMI_ROOT.parents:
        candidate = parent / "umi_day" / "deps" / "universal_manipulation_interface"
        if candidate.exists():
            sys.path.append(str(candidate))
            added = True
            break
    if not added:
        raise
    from umi.common.pose_util import pose10d_to_mat, pose_to_mat

from demo.trajectory_loader import MODEL_TO_FRAME, TCP_TO_MODEL_FRAME
from rby1.whole_body_ik import RBY1WholeBodyIK


def _load_task_config(task_path: str):
    task_cfg = OmegaConf.load(task_path)
    wrapped = OmegaConf.create({})
    wrapped["task"] = task_cfg
    for key in task_cfg.keys():
        wrapped[key] = task_cfg[key]
    OmegaConf.resolve(wrapped)
    return wrapped.task


def _decode_episode_names(names: Iterable) -> List[str]:
    decoded = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8"))
        else:
            decoded.append(str(name))
    return decoded


def _resolve_dataset_index_from_episode_name(dataset, episode_name: str) -> int:
    names = _decode_episode_names(dataset.replay_buffer.episode_names[:])
    if episode_name not in names:
        raise ValueError(f"Episode name '{episode_name}' not found.")
    episode_idx = names.index(episode_name)
    if not hasattr(dataset, "sampler"):
        raise RuntimeError("Dataset sampler unavailable; cannot resolve episode name.")
    best_idx = None
    fallback_idx = None
    for idx, (within_idx, ep_idx, _task_idx) in enumerate(dataset.sampler.indices):
        if ep_idx != episode_idx:
            continue
        if fallback_idx is None:
            fallback_idx = idx
        if within_idx == 0:
            best_idx = idx
            break
    if best_idx is None and fallback_idx is not None:
        best_idx = fallback_idx
    if best_idx is None:
        raise RuntimeError(f"No dataset index found for episode '{episode_name}'.")
    return best_idx


def _extract_action_trajectories(
    action: np.ndarray, action_indexing: Dict[str, Tuple[int, int]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray]]:
    trajectories: Dict[str, np.ndarray] = {}
    widths: Dict[str, np.ndarray] = {}
    lookat = None

    for key, (start, end) in action_indexing.items():
        if key.endswith("lookatpoint"):
            lookat = action[:, start:end].astype(np.float32, copy=False)

    for key, (start, end) in action_indexing.items():
        if key.endswith("_gripper_width"):
            prefix = key[: -len("_gripper_width")]
            widths[prefix] = action[:, start:end].reshape(-1).astype(np.float32, copy=False)

    for key, (start, end) in action_indexing.items():
        if not key.endswith("_eef_pos"):
            continue
        prefix = key[: -len("_eef_pos")]
        rot_key = f"{prefix}_eef_rot_axis_angle"
        if rot_key not in action_indexing:
            continue
        pos = action[:, start:end]
        rot_start, rot_end = action_indexing[rot_key]
        rot = action[:, rot_start:rot_end]
        if rot.shape[-1] == 3:
            pose_vec = np.concatenate([pos, rot], axis=-1)
            trajectories[prefix] = pose_to_mat(pose_vec)
        elif rot.shape[-1] == 6:
            pose_vec = np.concatenate([pos, rot], axis=-1)
            trajectories[prefix] = pose10d_to_mat(pose_vec)
        else:
            raise ValueError(f"Unexpected rotation dimension {rot.shape[-1]} for {prefix}")

    return trajectories, widths, lookat


def _extract_action_trajectories_raw(
    action_raw: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray]]:
    trajectories: Dict[str, np.ndarray] = {}
    widths: Dict[str, np.ndarray] = {}
    lookat = None

    if "camera_head_lookatpoint" in action_raw:
        lookat = np.asarray(action_raw["camera_head_lookatpoint"], dtype=np.float32)

    for key, data in action_raw.items():
        if key.endswith("_gripper_width"):
            prefix = key[: -len("_gripper_width")]
            widths[prefix] = np.asarray(data, dtype=np.float32).reshape(-1)

    for key, data in action_raw.items():
        if not key.endswith("_eef_pos"):
            continue
        prefix = key[: -len("_eef_pos")]
        rot_key = f"{prefix}_eef_rot_axis_angle"
        if rot_key not in action_raw:
            continue
        pos = np.asarray(data, dtype=np.float32)
        rot = np.asarray(action_raw[rot_key], dtype=np.float32)
        pose_vec = np.concatenate([pos, rot], axis=-1)
        trajectories[prefix] = pose_to_mat(pose_vec)

    return trajectories, widths, lookat


def _resolve_current_index(dataset, dataset_index: int) -> Tuple[int, int, int, int]:
    sampler = dataset.sampler
    if getattr(sampler, "use_prompting", False):
        raise RuntimeError("Prompting datasets are not supported by action-only sampling.")

    if dataset_index < 0 or dataset_index >= len(dataset):
        raise IndexError(
            f"dataset-index {dataset_index} outside valid range (0-{len(dataset) - 1})."
        )

    within_idx, episode_idx, task_idx = sampler.indices[dataset_index]
    replay_buffer = dataset.replay_buffer
    if sampler.sample_type == "task":
        segment_end = int(replay_buffer.task_data_ends[task_idx])
        segment_length = int(replay_buffer.task_lengths[task_idx])
        start_data_idx = segment_end - segment_length
    else:
        segment_end = int(replay_buffer.episode_ends[episode_idx])
        start_data_idx = 0 if episode_idx == 0 else int(replay_buffer.episode_ends[episode_idx - 1])

    current_data_idx = int(start_data_idx + within_idx)
    return current_data_idx, int(start_data_idx), int(segment_end), int(episode_idx)


def _sample_action_only(dataset, dataset_index: int) -> Tuple[np.ndarray, int, int, int]:
    sampler = dataset.sampler
    current_data_idx, start_data_idx, segment_end, episode_idx = _resolve_current_index(
        dataset, dataset_index
    )
    input_arr = sampler.in_memory_replay_buffer["action"]
    action_horizon = sampler.key_horizon["action"]
    action_down_sample_steps = sampler.key_down_sample_steps["action"]
    slice_end = min(
        segment_end,
        current_data_idx + (action_horizon - 1) * action_down_sample_steps + 1,
    )
    output = input_arr[current_data_idx:slice_end:action_down_sample_steps]
    if not sampler.action_padding:
        if output.shape[0] != action_horizon:
            raise RuntimeError(
                f"Action chunk too short ({output.shape[0]} vs {action_horizon}); "
                "enable action_padding in the dataset config."
            )
        return output, current_data_idx, start_data_idx, episode_idx

    if output.shape[0] < action_horizon:
        padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
        output = np.concatenate([output, padding], axis=0)
    return output, current_data_idx, start_data_idx, episode_idx


def _sample_action_raw_only(
    dataset, dataset_index: int
) -> Tuple[Dict[str, np.ndarray], int, int, int]:
    sampler = dataset.sampler
    current_data_idx, _start_data_idx, segment_end, episode_idx = _resolve_current_index(
        dataset, dataset_index
    )
    if "action_raw" not in sampler.in_memory_replay_buffer:
        raise RuntimeError("action_raw not available in replay buffer.")
    input_arr = sampler.in_memory_replay_buffer["action_raw"]
    action_horizon = sampler.key_horizon["action"]
    action_down_sample_steps = sampler.key_down_sample_steps["action"]
    slice_end = min(
        segment_end,
        current_data_idx + (action_horizon - 1) * action_down_sample_steps + 1,
    )
    output_raw: Dict[str, np.ndarray] = {}
    for key, data in input_arr.items():
        output = data[current_data_idx:slice_end:action_down_sample_steps]
        if not sampler.action_padding:
            if output.shape[0] != action_horizon:
                raise RuntimeError(
                    f"Action chunk too short ({output.shape[0]} vs {action_horizon}); "
                    "enable action_padding in the dataset config."
                )
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        output_raw[key] = output
    return output_raw, current_data_idx, _start_data_idx, episode_idx


def _body_pose_mat(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[bid].copy()
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.xmat[bid])
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def _align_trajectories_relative(
    trajectories: Dict[str, np.ndarray],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    prefix_map: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    converted: Dict[str, np.ndarray] = {}
    tcp_poses: Dict[str, np.ndarray] = {}
    for prefix, traj in trajectories.items():
        model_name = prefix_map.get(prefix)
        if model_name is None:
            converted[prefix] = traj
            continue
        tcp_to_model = TCP_TO_MODEL_FRAME.get(model_name)
        if tcp_to_model is None:
            converted[prefix] = traj
            continue
        body_name = MODEL_TO_FRAME.get(model_name)
        if body_name is None:
            converted[prefix] = traj
            continue
        robot_pose = _body_pose_mat(model, data, body_name)
        target0_model = traj[0] @ tcp_to_model.as_matrix()
        align = robot_pose @ np.linalg.inv(target0_model)
        out = np.zeros_like(traj)
        tcp_out = np.zeros_like(traj)
        model_to_tcp = tcp_to_model.inverse().as_matrix()
        for idx, rel_pose in enumerate(traj):
            target_model = rel_pose @ tcp_to_model.as_matrix()
            aligned = align @ target_model
            out[idx] = aligned
            tcp_out[idx] = aligned @ model_to_tcp
        converted[prefix] = out
        tcp_poses[prefix] = tcp_out
    return converted, tcp_poses


def _apply_relative_lookat_sequence(
    lookat: Optional[np.ndarray],
    head_tcp: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if lookat is None or head_tcp is None:
        return lookat
    ones = np.ones((lookat.shape[0], 1), dtype=lookat.dtype)
    points_h = np.concatenate([lookat, ones], axis=-1)
    world = np.einsum("tij,tj->ti", head_tcp, points_h)
    return world[:, :3]


def _select_action_indexing(action: np.ndarray, dataset) -> Dict[str, Tuple[int, int]]:
    indexing = dataset.action_indexing
    max_end = max(end for _, end in indexing.values())
    if max_end <= action.shape[-1]:
        return indexing
    return dataset.action_indexing_raw


def _pose_to_wxyz(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = pose[:3, 3].astype(float)
    quat_xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
    return pos, quat_wxyz


def _build_target_sequence(
    trajectories: Dict[str, np.ndarray],
    widths: Dict[str, np.ndarray],
    lookat: Optional[np.ndarray],
    left_prefix: str,
    right_prefix: str,
    head_prefix: Optional[str],
) -> Tuple[List[Dict[str, Optional[np.ndarray]]], List[Dict[str, float]], Optional[np.ndarray]]:
    left_traj = trajectories.get(left_prefix)
    right_traj = trajectories.get(right_prefix)
    head_traj = trajectories.get(head_prefix) if head_prefix else None

    left_width = widths.get(left_prefix)
    right_width = widths.get(right_prefix)

    lengths = [
        seq.shape[0]
        for seq in (left_traj, right_traj, head_traj)
        if seq is not None
    ]
    if not lengths:
        raise RuntimeError("No end-effector trajectories found in action chunk.")
    num_steps = max(lengths)

    poses_list: List[Dict[str, Optional[np.ndarray]]] = []
    widths_list: List[Dict[str, float]] = []

    for idx in range(num_steps):
        pose_entry: Dict[str, Optional[np.ndarray]] = {}
        if left_traj is not None and idx < left_traj.shape[0]:
            pose_entry["left"] = left_traj[idx]
        if right_traj is not None and idx < right_traj.shape[0]:
            pose_entry["right"] = right_traj[idx]
        if head_traj is not None and idx < head_traj.shape[0]:
            pose_entry["head"] = head_traj[idx]

        width_entry = {
            "left": float(left_width[idx]) if left_width is not None and idx < left_width.shape[0] else 0.0,
            "right": float(right_width[idx]) if right_width is not None and idx < right_width.shape[0] else 0.0,
        }
        poses_list.append(pose_entry)
        widths_list.append(width_entry)

    lookat_seq = lookat if lookat is None else lookat[:num_steps]
    return poses_list, widths_list, lookat_seq


def _copy_camera(cam: mujoco.MjvCamera) -> mujoco.MjvCamera:
    new_cam = mujoco.MjvCamera()
    new_cam.type = cam.type
    new_cam.fixedcamid = cam.fixedcamid
    new_cam.trackbodyid = cam.trackbodyid
    new_cam.lookat[:] = cam.lookat
    new_cam.distance = cam.distance
    new_cam.azimuth = cam.azimuth
    new_cam.elevation = cam.elevation
    if hasattr(cam, "fovy"):
        new_cam.fovy = cam.fovy
    new_cam.orthographic = cam.orthographic
    return new_cam


def _apply_camera(src: mujoco.MjvCamera, dst: mujoco.MjvCamera) -> None:
    dst.type = src.type
    dst.fixedcamid = src.fixedcamid
    dst.trackbodyid = src.trackbodyid
    dst.lookat[:] = src.lookat
    dst.distance = src.distance
    dst.azimuth = src.azimuth
    dst.elevation = src.elevation
    if hasattr(src, "fovy") and hasattr(dst, "fovy"):
        dst.fovy = src.fovy
    dst.orthographic = src.orthographic


def _add_axis_marker(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    quat: np.ndarray,
    axis_length: float,
    alpha: float,
    line_width: float,
) -> None:
    if scene.ngeom + 3 >= scene.maxgeom:
        return
    pos_arr = np.asarray(pos, dtype=np.float64)
    quat_arr = np.asarray(quat, dtype=np.float64)

    line_size = np.zeros(3, dtype=np.float64)
    identity_mat = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64
    )
    axis_colors = (
        np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, alpha], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, alpha], dtype=np.float32),
    )

    frame_mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(frame_mat, quat_arr)
    rot = frame_mat.reshape(3, 3)

    for axis_idx, axis_color in enumerate(axis_colors):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            line_size,
            pos_arr,
            identity_mat,
            axis_color,
        )
        endpoint = pos_arr + rot[:, axis_idx] * axis_length
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, line_width, pos_arr, endpoint
        )
        scene.ngeom += 1


def _add_lookat_marker(
    scene: mujoco.MjvScene,
    point: np.ndarray,
    radius: float,
    color: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    size = np.array([radius, 0.0, 0.0], dtype=np.float64)
    mat = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        np.asarray(point, dtype=np.float64),
        mat,
        color,
    )
    scene.ngeom += 1


def _render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    target_pose: Dict[str, Optional[np.ndarray]],
    lookat: Optional[np.ndarray],
    axis_length: float,
    line_width: float,
    lookat_radius: float,
    targets_only: bool,
    scene_option: Optional[mujoco.MjvOption] = None,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    scene = renderer.scene
    if targets_only:
        scene.ngeom = 0

    for key in ("left", "right", "head"):
        pose = target_pose.get(key)
        if pose is None:
            continue
        pos, quat = _pose_to_wxyz(pose)
        _add_axis_marker(scene, pos, quat, axis_length, alpha=0.8, line_width=line_width)

    if lookat is not None:
        _add_lookat_marker(
            scene,
            lookat,
            lookat_radius,
            np.array([0.9, 0.2, 0.2, 0.8], dtype=np.float32),
        )

    rgb = renderer.render()
    return rgb


def _transparent_rgba(rgb: np.ndarray, tolerance: int) -> np.ndarray:
    if rgb.dtype != np.uint8:
        img = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        img = rgb
    h, w, _ = img.shape
    samples = [
        img[0, 0],
        img[0, w - 1],
        img[h - 1, 0],
        img[h - 1, w - 1],
    ]
    colors = []
    for col in samples:
        if not any(np.all(col == existing) for existing in colors):
            colors.append(col)
    tol = int(max(tolerance, 0))
    mask = np.zeros((h, w), dtype=bool)
    for col in colors:
        diff = np.abs(img.astype(np.int16) - col.astype(np.int16))
        close = np.all(diff <= tol, axis=-1)
        mask |= close
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([img, alpha])
    return rgba


def _load_camera_config(path: Path, camera: mujoco.MjvCamera) -> None:
    payload = json.loads(path.read_text())
    if "lookat" in payload:
        camera.lookat[:] = np.asarray(payload["lookat"], dtype=float)
    if "distance" in payload:
        camera.distance = float(payload["distance"])
    if "azimuth" in payload:
        camera.azimuth = float(payload["azimuth"])
    if "elevation" in payload:
        camera.elevation = float(payload["elevation"])
    if "fovy" in payload and hasattr(camera, "fovy"):
        camera.fovy = float(payload["fovy"])


def _save_camera_config(path: Path, camera: mujoco.MjvCamera) -> None:
    payload = {
        "lookat": camera.lookat.tolist(),
        "distance": float(camera.distance),
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
    }
    if hasattr(camera, "fovy"):
        payload["fovy"] = float(camera.fovy)
    path.write_text(json.dumps(payload, indent=2))


def _wait_for_camera_adjust(viewer) -> mujoco.MjvCamera:
    print("Adjust the MuJoCo camera, then press Enter to render.")
    while True:
        viewer.sync()
        if not viewer.is_running():
            raise RuntimeError("Viewer closed before camera capture.")
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            sys.stdin.readline()
            break
    return _copy_camera(viewer.cam)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render dataset action chunk trajectories using whole-body IK."
    )
    parser.add_argument(
        "task_config",
        type=str,
        help="Path to a task config yaml (e.g. train_network/config/task/ego3d_lookat_policy.yaml).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset_path inside the task config.",
    )
    parser.add_argument(
        "--reference-frame",
        type=str,
        default=None,
        help="Override dataset reference frame (defaults to task.reference_frame if present).",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Index to sample from the dataset.",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Episode name to sample (overrides dataset-index if provided).",
    )
    parser.add_argument(
        "--left-prefix",
        type=str,
        default="gripper_left",
        help="Action prefix for the left end-effector pose.",
    )
    parser.add_argument(
        "--right-prefix",
        type=str,
        default="gripper_right",
        help="Action prefix for the right end-effector pose.",
    )
    parser.add_argument(
        "--head-prefix",
        type=str,
        default="gripper_head",
        help="Action prefix for the head end-effector pose.",
    )
    parser.add_argument(
        "--action-fps",
        type=float,
        default=10.0,
        help="Action frequency used to step the IK simulation.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=None,
        help="Render frequency; when provided, computes stride from action-fps.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth action step (ignored if render-fps is set).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to render (-1 keeps all).",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.12,
        help="Length of each target axis.",
    )
    parser.add_argument(
        "--axis-line-width",
        type=float,
        default=2.5,
        help="Line width for axis markers.",
    )
    parser.add_argument(
        "--lookat-radius",
        type=float,
        default=0.02,
        help="Radius of lookat point markers.",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=960,
        help="Width of rendered images.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=720,
        help="Height of rendered images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Defaults to dataset directory.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=("mujoco", "blender"),
        default="mujoco",
        help="Rendering backend to use.",
    )
    parser.add_argument(
        "--blender-exec",
        type=str,
        default="blender",
        help="Blender executable (used only when renderer=blender).",
    )
    parser.add_argument(
        "--blender-script",
        type=str,
        default=None,
        help="Blender script to run (used only when renderer=blender).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show a GUI to adjust the camera before rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--targets-overlay",
        action="store_true",
        help="Render all targets into one frame instead of per-frame outputs.",
    )
    parser.add_argument(
        "--overlay-stride",
        type=int,
        default=1,
        help="Stride for targets overlay rendering.",
    )
    parser.add_argument(
        "--transparent-bg",
        action="store_true",
        help="Write PNGs with transparent background by chroma-keying corner colors.",
    )
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help="Render only target axes/lookat points (omit robot geometry).",
    )
    parser.add_argument(
        "--target-prefix",
        type=str,
        default="targets",
        help="Filename prefix for target-only renders.",
    )
    parser.add_argument(
        "--bg-tolerance",
        type=int,
        default=6,
        help="Color tolerance for background chroma-keying.",
    )
    parser.add_argument(
        "--camera-load",
        type=str,
        default=None,
        help="Load camera config JSON before rendering.",
    )
    parser.add_argument(
        "--camera-save",
        type=str,
        default=None,
        help="Save camera config JSON after GUI/auto setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_cfg = _load_task_config(args.task_config)
    dataset_cfg = task_cfg.get("dataset")
    if dataset_cfg is None:
        raise ValueError("Task config does not define a dataset block.")

    if args.dataset_path:
        dataset_cfg.dataset_path = args.dataset_path
    task_reference_frame = task_cfg.get("reference_frame")
    if args.reference_frame:
        dataset_cfg.reference_frame = args.reference_frame
    elif task_reference_frame is not None:
        dataset_cfg.reference_frame = task_reference_frame

    dataset_path_value = dataset_cfg.get("dataset_path")
    if dataset_path_value is None:
        raise ValueError("dataset.dataset_path must be specified in the task config.")
    dataset_path = Path(dataset_path_value)

    dataset = hydra.utils.instantiate(dataset_cfg)
    if args.episode_name:
        args.dataset_index = _resolve_dataset_index_from_episode_name(
            dataset, args.episode_name
        )
    ik = RBY1WholeBodyIK()
    current_qpos = ik._get_nominal_posture(ik.data.qpos.copy())

    model = mujoco.MjModel.from_xml_path(ik.model_path)
    data = mujoco.MjData(model)
    data.qpos[:] = current_qpos
    mujoco.mj_forward(model, data)

    prefix_map = {
        "gripper_left": "left_arm",
        "gripper_right": "right_arm",
        "gripper_head": "head",
    }
    if "action_raw" in dataset.sampler.in_memory_replay_buffer:
        action_raw, _current_data_idx, _start_data_idx, _episode_idx = _sample_action_raw_only(
            dataset, args.dataset_index
        )
        trajectories, widths, lookat = _extract_action_trajectories_raw(action_raw)
    else:
        action_np, _current_data_idx, _start_data_idx, _episode_idx = _sample_action_only(
            dataset, args.dataset_index
        )
        action_indexing = _select_action_indexing(action_np, dataset)
        trajectories, widths, lookat = _extract_action_trajectories(
            action_np, action_indexing
        )

    trajectories, tcp_poses = _align_trajectories_relative(
        trajectories, model, data, prefix_map
    )
    if lookat is not None and args.head_prefix:
        head_tcp = tcp_poses.get(args.head_prefix)
        lookat = _apply_relative_lookat_sequence(lookat, head_tcp)

    poses_list, widths_list, lookat_seq = _build_target_sequence(
        trajectories,
        widths,
        lookat,
        left_prefix=args.left_prefix,
        right_prefix=args.right_prefix,
        head_prefix=None,
    )

    if args.output is None:
        out_dir = dataset_path.parent
    else:
        out_dir = Path(args.output)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_label = args.episode_name.strip() if args.episode_name else None
    sample_prefix = (
        episode_label if episode_label else f"{dataset_path.stem}_idx{args.dataset_index:05d}"
    )

    action_fps = float(args.action_fps)
    if action_fps <= 0.0:
        raise ValueError("action-fps must be > 0")
    if args.render_fps is not None:
        stride = max(1, int(round(action_fps / float(args.render_fps))))
    else:
        stride = max(1, int(args.frame_stride))

    render_width = int(args.render_width)
    render_height = int(args.render_height)

    total_frames = len(poses_list)
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    frame_indices = list(range(0, total_frames, stride))
    if not frame_indices:
        raise RuntimeError("No frames selected after applying stride/max-frames.")

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    targets_scene_option = None
    if args.targets_only:
        targets_scene_option = mujoco.MjvOption()
        targets_scene_option.geomgroup[:] = 0
        targets_scene_option.sitegroup[:] = 0
    if args.camera_load:
        _load_camera_config(Path(args.camera_load), camera)

    if args.gui:
        first_pose = poses_list[0]
        left_pose = first_pose.get("left")
        right_pose = first_pose.get("right")
        head_pose = first_pose.get("head")
        left_pos, left_quat = (_pose_to_wxyz(left_pose) if left_pose is not None else (None, None))
        right_pos, right_quat = (_pose_to_wxyz(right_pose) if right_pose is not None else (None, None))
        head_pos, head_quat = (_pose_to_wxyz(head_pose) if head_pose is not None else (None, None))
        preview_qpos, _preview_qvel, _success, _info = ik.solve(
            left_target_pos=left_pos,
            left_target_quat=left_quat,
            right_target_pos=right_pos,
            right_target_quat=right_quat,
            head_target_pos=head_pos,
            head_target_quat=head_quat,
            current_qpos=current_qpos,
            dt=1.0 / action_fps,
        )
        data.qpos[:] = preview_qpos
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            _apply_camera(camera, viewer.cam)
            viewer.user_scn.ngeom = 0
            for key in ("left", "right", "head"):
                if first_pose.get(key) is None:
                    continue
                pos, quat = _pose_to_wxyz(first_pose[key])
                _add_axis_marker(
                    viewer.user_scn,
                    pos,
                    quat,
                    axis_length=args.axis_length,
                    alpha=0.7,
                    line_width=args.axis_line_width,
                )
            if lookat_seq is not None:
                _add_lookat_marker(
                    viewer.user_scn,
                    lookat_seq[0],
                    radius=args.lookat_radius,
                    color=np.array([0.9, 0.2, 0.2, 0.7], dtype=np.float32),
                )
            camera = _wait_for_camera_adjust(viewer)

    if args.camera_save:
        _save_camera_config(Path(args.camera_save), camera)

    if args.renderer == "blender":
        if args.blender_script is None:
            raise ValueError("--blender-script is required when renderer=blender")
        job = {
            "model_path": ik.model_path,
            "qpos": [],
            "targets": [],
            "lookat": [] if lookat_seq is not None else None,
            "camera": {
                "lookat": camera.lookat.tolist(),
                "distance": float(camera.distance),
                "azimuth": float(camera.azimuth),
                "elevation": float(camera.elevation),
            },
            "render_width": render_width,
            "render_height": render_height,
            "output_dir": str(out_dir),
            "sample_prefix": sample_prefix,
            "frame_indices": frame_indices,
        }
        if hasattr(camera, "fovy"):
            job["camera"]["fovy"] = float(camera.fovy)

    renderer = None
    max_w = max_h = 0
    if hasattr(model.vis, "global_"):
        max_w = int(getattr(model.vis.global_, "offwidth", 0) or 0)
        max_h = int(getattr(model.vis.global_, "offheight", 0) or 0)
        if max_w > 0 and render_width > max_w:
            print(f"[wbik_render] Reducing render width to {max_w} (offscreen limit).")
            render_width = max_w
        if max_h > 0 and render_height > max_h:
            print(f"[wbik_render] Reducing render height to {max_h} (offscreen limit).")
            render_height = max_h
    if args.renderer == "mujoco":
        swap_dims = False
        if max_w > 0 and max_h > 0:
            if (
                render_width <= max_h
                and render_height <= max_w
                and (render_width > max_w or render_height > max_h)
            ):
                swap_dims = True
        try:
            if swap_dims:
                renderer = mujoco.Renderer(model, render_height, render_width)
                render_width, render_height = render_height, render_width
            else:
                renderer = mujoco.Renderer(model, render_width, render_height)
        except ValueError as exc:
            if "Image height" in str(exc) or "Image width" in str(exc):
                renderer = mujoco.Renderer(model, render_height, render_width)
                render_width, render_height = render_height, render_width
            else:
                raise

    if args.targets_overlay:
        overlay_stride = max(1, int(args.overlay_stride))
        renderer.update_scene(data, camera=camera, scene_option=targets_scene_option)
        scene = renderer.scene
        scene.ngeom = 0
        for idx in range(0, total_frames, overlay_stride):
            target_pose = poses_list[idx]
            for key in ("left", "right", "head"):
                pose = target_pose.get(key)
                if pose is None:
                    continue
                pos, quat = _pose_to_wxyz(pose)
                _add_axis_marker(
                    scene,
                    pos,
                    quat,
                    axis_length=args.axis_length,
                    alpha=0.7,
                    line_width=args.axis_line_width,
                )
            if lookat_seq is not None and idx < len(lookat_seq):
                _add_lookat_marker(
                    scene,
                    lookat_seq[idx],
                    radius=args.lookat_radius,
                    color=np.array([0.9, 0.2, 0.2, 0.7], dtype=np.float32),
                )
        rgb = renderer.render()
        if args.targets_only and args.target_prefix:
            name = f"{sample_prefix}_{args.target_prefix}_overlay.png"
        else:
            name = f"{sample_prefix}_overlay.png"
        output_path = out_dir / name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
        if args.transparent_bg:
            rgba = _transparent_rgba(rgb, args.bg_tolerance)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            if not cv2.imwrite(str(output_path), bgra):
                raise RuntimeError(f"Failed to write render to {output_path}")
        else:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), rgb_bgr):
                raise RuntimeError(f"Failed to write render to {output_path}")
        print(f"[wbik_render] Overlay saved to {output_path}")
        return

    for idx in range(total_frames):
        target_pose = poses_list[idx]
        left_pose = target_pose.get("left")
        right_pose = target_pose.get("right")
        head_pose = None

        left_pos, left_quat = (None, None)
        right_pos, right_quat = (None, None)
        head_pos, head_quat = (None, None)

        if left_pose is not None:
            left_pos, left_quat = _pose_to_wxyz(left_pose)
        if right_pose is not None:
            right_pos, right_quat = _pose_to_wxyz(right_pose)
        if head_pose is not None:
            head_pos, head_quat = _pose_to_wxyz(head_pose)

        sol_qpos, _sol_qvel, success, info = ik.solve(
            left_target_pos=left_pos,
            left_target_quat=left_quat,
            right_target_pos=right_pos,
            right_target_quat=right_quat,
            head_target_pos=None,
            head_target_quat=None,
            current_qpos=current_qpos,
            dt=1.0 / action_fps,
        )
        if not success:
            print(f"[wbik_render] IK solve failed at step {idx}: {info}")
        current_qpos = sol_qpos

        if args.renderer == "blender":
            job["qpos"].append(sol_qpos.tolist())
            job["targets"].append(
                {
                    "left": left_pose.tolist() if left_pose is not None else None,
                    "right": right_pose.tolist() if right_pose is not None else None,
                    "head": head_pose.tolist() if head_pose is not None else None,
                }
            )
            if lookat_seq is not None:
                job["lookat"].append(lookat_seq[idx].tolist())

        if idx not in frame_indices or args.renderer == "blender":
            continue

        data.qpos[:] = sol_qpos
        mujoco.mj_forward(model, data)

        lookat_point = lookat_seq[idx] if lookat_seq is not None else None
        rgb = _render_frame(
            renderer,
            data,
            camera,
            target_pose,
            lookat_point,
            axis_length=args.axis_length,
            line_width=args.axis_line_width,
            lookat_radius=args.lookat_radius,
            targets_only=args.targets_only,
            scene_option=targets_scene_option,
        )
        if args.targets_only and args.target_prefix:
            name = f"{sample_prefix}_{args.target_prefix}_frame{idx:05d}.png"
        else:
            name = f"{sample_prefix}_frame{idx:05d}.png"
        output_path = out_dir / name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
        if args.transparent_bg:
            rgba = _transparent_rgba(rgb, args.bg_tolerance)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            if not cv2.imwrite(str(output_path), bgra):
                raise RuntimeError(f"Failed to write render to {output_path}")
        else:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), rgb_bgr):
                raise RuntimeError(f"Failed to write render to {output_path}")

    if args.renderer == "blender":
        job_path = out_dir / f"{sample_prefix}_blender_job.json"
        if job_path.exists() and not args.overwrite:
            raise FileExistsError(f"{job_path} already exists. Use --overwrite to replace it.")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        cmd = [args.blender_exec, "-b", "--python", args.blender_script, "--", str(job_path)]
        print("[wbik_render] Launching blender:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Blender failed with exit code {result.returncode}")
        return

    camera_path = out_dir / f"{sample_prefix}_camera.json"
    if camera_path.exists() and not args.overwrite:
        raise FileExistsError(f"{camera_path} already exists. Use --overwrite to replace it.")
    camera_payload = {
        "lookat": camera.lookat.tolist(),
        "distance": float(camera.distance),
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
        "render_width": render_width,
        "render_height": render_height,
        "action_fps": action_fps,
        "frame_stride": stride,
    }
    if hasattr(camera, "fovy"):
        camera_payload["fovy"] = float(camera.fovy)
    with open(camera_path, "w", encoding="utf-8") as f:
        json.dump(camera_payload, f, indent=2)

    print(f"[wbik_render] Rendered {len(frame_indices)} frames to {out_dir}")


if __name__ == "__main__":
    main()
