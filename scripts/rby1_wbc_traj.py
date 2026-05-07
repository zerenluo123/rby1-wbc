"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import hydra
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    hydra = None
    OmegaConf = None

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to make umi_day importable when running from rby1 checkout.
UMI_ROOT = Path(__file__).resolve().parents[2] / "umi_day"
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
        pose10d_to_mat = None
        pose_to_mat = None
    else:
        from umi.common.pose_util import pose10d_to_mat, pose_to_mat

from control.rby1_wbc import RBY1WBC
from demo.trajectory_loader import load_trajectory
from rby1.ee_targets import EETargets
from rby1.rby1_wbc_app import RBY1WBCApp


def _load_task_config(task_path: str):
    if OmegaConf is None:
        raise ModuleNotFoundError("omegaconf is required for dataset loading.")
    task_cfg = OmegaConf.load(task_path)
    wrapped = OmegaConf.create({})
    wrapped["task"] = task_cfg
    for key in task_cfg.keys():
        wrapped[key] = task_cfg[key]
    OmegaConf.resolve(wrapped)
    return wrapped.task


def _decode_episode_names(names) -> list[str]:
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


def _extract_action_trajectories(action: np.ndarray, action_indexing: dict):
    trajectories: dict[str, np.ndarray] = {}
    widths: dict[str, np.ndarray] = {}

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

    return trajectories, widths


def _load_dataset_trajectory(args) -> tuple[list[dict], list[dict]]:
    if hydra is None:
        raise ModuleNotFoundError("hydra-core is required for dataset loading.")
    if pose10d_to_mat is None or pose_to_mat is None:
        raise ModuleNotFoundError("umi.common.pose_util is required.")

    task_cfg = _load_task_config(args.dataset_task_config)
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

    dataset = hydra.utils.instantiate(dataset_cfg)
    dataset_index = args.dataset_index
    if args.episode_name:
        dataset_index = _resolve_dataset_index_from_episode_name(dataset, args.episode_name)
    if dataset_index < 0 or dataset_index >= len(dataset):
        raise IndexError(
            f"dataset-index {dataset_index} outside valid range (0-{len(dataset) - 1})."
        )

    sample = dataset[dataset_index]
    action_np = sample["action"].detach().cpu().numpy()

    trajectories, widths = _extract_action_trajectories(action_np, dataset.action_indexing)

    left_traj = trajectories.get(args.left_prefix)
    right_traj = trajectories.get(args.right_prefix)
    head_traj = trajectories.get(args.head_prefix) if args.head_prefix else None

    if left_traj is None or right_traj is None:
        raise RuntimeError("Left/right action trajectories not found in dataset action chunk.")

    num_steps = min(left_traj.shape[0], right_traj.shape[0])
    if head_traj is not None:
        num_steps = min(num_steps, head_traj.shape[0])

    left_width = widths.get(args.left_prefix)
    right_width = widths.get(args.right_prefix)

    poses_list = []
    widths_list = []
    for idx in range(num_steps):
        pose_entry = {
            "left_arm": left_traj[idx],
            "right_arm": right_traj[idx],
        }
        if head_traj is not None:
            pose_entry["head"] = head_traj[idx]
        poses_list.append(pose_entry)

        width_entry = {
            "left_width": float(left_width[idx]) if left_width is not None and idx < left_width.shape[0] else 0.0,
            "right_width": float(right_width[idx]) if right_width is not None and idx < right_width.shape[0] else 0.0,
        }
        widths_list.append(width_entry)

    return poses_list, widths_list


class RBY1WBCTrajectory(RBY1WBCApp):
    def __init__(
        self,
        wbc: RBY1WBC,
        headless: bool = False,
        poses_list: list[dict] | None = None,
        widths_list: list[dict] | None = None,
    ) -> None:
        self.poses_list = poses_list if poses_list is not None else []
        self.widths_list = widths_list if widths_list is not None else []
        self.trajectory_index = 0
        self.num_steps = min(len(self.poses_list), len(self.widths_list))
        super().__init__(wbc=wbc, headless=headless)

    @staticmethod
    def _transform_to_pose(transform) -> tuple[np.ndarray, np.ndarray]:
        mat = (
            transform.as_matrix()
            if hasattr(transform, "as_matrix")
            else np.asarray(transform)
        )
        pos = mat[:3, 3].astype(float)
        quat_xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
        )
        return pos, quat_wxyz

    def on_target_rejected(self, target: EETargets) -> None:
        rejected_index = max(self.trajectory_index - 1, 0)
        print(f"[trajectory] target {rejected_index} rejected; skipping to next command.")

    def get_target(self) -> Optional[EETargets]:
        if self.trajectory_index == 0:
            input("Press [Enter] to start streaming the trajectory.")

        if self.trajectory_index >= self.num_steps:
            print("[trajectory] streaming complete.")
            return None

        pose_entry = self.poses_list[self.trajectory_index]
        left_transform = pose_entry["left_arm"]
        right_transform = pose_entry["right_arm"]
        left_pos, left_quat = self._transform_to_pose(left_transform)
        right_pos, right_quat = self._transform_to_pose(right_transform)

        head_pos = head_quat = None
        if "head" in pose_entry:
            head_transform = pose_entry["head"]
            head_pos, head_quat = self._transform_to_pose(head_transform)

        width_entry = self.widths_list[self.trajectory_index]
        left_width = width_entry["left_width"]
        right_width = width_entry["right_width"]

        self.trajectory_index += 1
        return EETargets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=head_pos,
            head_quat=head_quat,
            duration=self.trajectory_rate.dt,
            timestamp=time.monotonic(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body IK GUI decoupled from WBC thread"
    )
    parser.add_argument(
        "--trajectory",
        default=PROJECT_ROOT + "/demo/dataset_plan_waiter3.pkl",
        help="Path to a pickle file containing trajectory episodes",
    )
    parser.add_argument(
        "--index",
        default=0,
        type=int,
        help="Index of the trajectory episode to use",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer (useful for debugging controller only).",
    )
    parser.add_argument(
        "--dataset-task-config",
        type=str,
        default=None,
        help="Use a UMI dataset action chunk from a task config yaml instead of a pkl trajectory.",
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
        help="Action prefix for the head end-effector pose (set empty to disable).",
    )
    args = parser.parse_args()

    wbc = RBY1WBC()
    wbc.start()

    poses_list = None
    widths_list = None
    if args.dataset_task_config:
        if args.head_prefix == "":
            args.head_prefix = None
        poses_list, widths_list = _load_dataset_trajectory(args)
    else:
        poses_list, widths_list = load_trajectory(
            traj_dir=args.trajectory,
            client=wbc,
            index=args.index,
            use_head=True,
            align_mode="relative",
        )

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    gui = None
    try:
        gui = RBY1WBCTrajectory(
            wbc=wbc,
            headless=args.headless,
            poses_list=poses_list,
            widths_list=widths_list,
        )
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()


if __name__ == "__main__":
    main()
