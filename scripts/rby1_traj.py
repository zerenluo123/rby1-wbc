import argparse
import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import mujoco
from scipy.spatial.transform import Rotation
from transforms3d import affines


class RigidTransform:
    """Lightweight SE(3) transform helper with Drake-compatible operations."""

    __slots__ = ("_matrix",)

    def __init__(self, rotation=None, translation=None):
        if translation is None:
            if rotation is None:
                self._matrix = np.eye(4, dtype=float)
            else:
                matrix = np.asarray(rotation, dtype=float)
                if matrix.shape != (4, 4):
                    raise ValueError("Provide (4,4) matrix or (3,3)+(3,) pair")
                self._matrix = matrix.copy()
        else:
            rot = np.asarray(rotation, dtype=float).reshape(3, 3)
            trans = np.asarray(translation, dtype=float).reshape(3)
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = rot
            matrix[:3, 3] = trans
            self._matrix = matrix

    @classmethod
    def identity(cls):
        return cls()

    def copy(self):
        return RigidTransform(self._matrix)

    def as_matrix(self):
        return self._matrix.copy()

    def rotation_matrix(self):
        return self._matrix[:3, :3].copy()

    def translation(self):
        return self._matrix[:3, 3].copy()

    def inverse(self):
        rot = self._matrix[:3, :3]
        trans = self._matrix[:3, 3]
        rot_inv = rot.T
        trans_inv = -rot_inv @ trans
        return RigidTransform(rot_inv, trans_inv)

    def __matmul__(self, other):
        if not isinstance(other, RigidTransform):
            return NotImplemented
        return RigidTransform(self._matrix @ other._matrix)

def rpy_to_matrix(rpy):
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()

MODEL_TO_FRAME = {
    "right_arm": "EE_BODY_R",
    "left_arm": "EE_BODY_L",
    "head": "link_head_2"
}

CAMERA_TO_MODEL_FRAME = {
    "head": RigidTransform(rpy_to_matrix([-1.57079632679, 0.0, -1.57079632679]), [-0.020851, 0.0, 0.0601]).inverse(),
    "left_arm": RigidTransform(rpy_to_matrix([np.pi, 0.0, 0.0]), [0.0, 0.055, 0.003355]).inverse(),
    "right_arm": RigidTransform(rpy_to_matrix([0.0, np.pi, 0.0]), [0.0, -0.055, 0.003355]).inverse(),
}

TCP_TO_MODEL_FRAME = {
    "head": RigidTransform(rpy_to_matrix([-1.57079632679, 0.0, -1.57079632679]), [0.04, 0.0, 0.0601]).inverse(),
    "left_arm": RigidTransform(rpy_to_matrix([np.pi, 0.0, 0.0]), [0.0, 0.0, -0.2]).inverse(),
    "right_arm": RigidTransform(rpy_to_matrix([0.0, np.pi, 0.0]), [0.0, 0.0, -0.2]).inverse(),
}

def load_pose_and_gripper(fname, idx=0):
    with open(fname, 'rb') as f:
        plan = pickle.load(f)
    
    try:
        plan_episode = plan[idx]
    except Exception as e:
        raise ValueError(f"Failed to get episode index {idx} from plan with {len(plan)} episodes: {e}")

    grippers_by_side = {k.split('grippers_')[1]: v for k, v in plan_episode.items() if k.startswith('grippers_')}
    poses = {}
    widths = {}
    for side, grippers in grippers_by_side.items():
        gripper = grippers[0]  # Assuming only one gripper per side in the plan
        eef_pose = gripper['tcp_pose']
        # transform to matrix (N, 7) -> (N, 4, 4)
        eef_pose_matrix = [RigidTransform(affines.compose(
            pose[:3],  # x, y, z
            Rotation.from_rotvec(pose[3:]).as_matrix(),
            np.ones(3)  # scale (not used)
        )) for pose in eef_pose]
        poses[side] = eef_pose_matrix
        if side != "head":  # head does not have gripper width
            gripper_widths = np.expand_dims(gripper['gripper_width'], axis=-1).astype(np.float32)
            widths[side] = gripper_widths
    return poses, widths

def load_trajectory(traj_dir, client, index: int = 0, use_head: bool = True, align_mode: str = "relative"):
    # Load raw pose_data
    traj_dir = Path(traj_dir)

    model_names = ["left_arm", "right_arm"]
    if use_head:
        model_names.append("head")

    poses, gripper_widths = load_pose_and_gripper(traj_dir, idx=index)
    if not traj_dir.exists():
        raise FileNotFoundError(f"Missing: {traj_dir}")
    # change keys to match model names
    pose_data = {m: poses[m.replace("_arm", "")] for m in model_names}
    gripper_data = {m: gripper_widths[m.replace("_arm", "")] for m in model_names if m.replace("_arm", "") in gripper_widths}

    T = min(len(pose_data[m]) for m in model_names)

    for m in model_names:
        if len(pose_data[m]) != T:
            raise ValueError(f"Pose data for {m} has length {len(pose_data[m])}, expected {T}")

    if align_mode == "relative":
        poses_all = [pose_data[m][0] @ TCP_TO_MODEL_FRAME[m] for m in model_names]

        snapshot = client.robot_state.load()
        if snapshot is None or not snapshot.is_valid:
            snapshot = client.wait_for_first_state()
        with client._model_lock:
            qpos = client._snapshot_to_qpos(snapshot)
            client.data.qpos[:] = qpos
            mujoco.mj_forward(client.model, client.data)
            robot_poses = []
            for m in model_names:
                body_id = mujoco.mj_name2id(client.model, mujoco.mjtObj.mjOBJ_BODY, MODEL_TO_FRAME[m])
                rot = client.data.xmat[body_id].reshape(3, 3)
                pos = client.data.xpos[body_id].copy()
                robot_poses.append(RigidTransform(rot, pos))

        # robot_poses = [from_lcm_pose(next(m.pose_actual for m in status_msg.models if m.model_name == m_name)) for m_name in model_names]

        T_transform_all = {key: robot_poses[i] @ poses_all[i].inverse() for i, key in enumerate(model_names)}

    elif align_mode == "none":
        # No alignment, just use the raw poses
        T_transform_all = {key: RigidTransform.identity() for key in model_names}

    # Apply transform to all poses
    poses_list = []
    widths_list = []

    for t in range(T):
        poses = {}
        for m in model_names:
            T_model = pose_data[m][t] @ TCP_TO_MODEL_FRAME[m]
            T_aligned = T_transform_all[m] @ T_model
            poses[m] = T_aligned

        widths = {}
        for m in model_names:
            if m in gripper_data:
                widths[f"{m.replace('_arm', '')}_width"] = gripper_data[m][t]
        poses_list.append(poses)
        widths_list.append(widths)

    return poses_list, widths_list
