import argparse
import time
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from transforms3d import affines
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.geometry import Meshcat, MeshcatParams, Rgba, Cylinder
from pydrake.common.eigen_geometry import Quaternion

from anzu.intuitive.visuomotor.multiarm_robot_client import (
    MultiarmRobotClient,
    make_multi_frame_decoupled_message,
    move_to_initial,
)
from anzu.common.lcm_geometry import from_lcm_pose

MODEL_TO_FRAME = {
    "right_arm": "right_arm::ee_right",
    "left_arm": "left_arm::ee_left",
    "head": "head::link_head_2",
    # "right_arm": "right_arm::wrist_camera_minus",     # can command with camera frame but somehow cannot get published pose
    # "left_arm": "left_arm::wrist_camera_plus",
    # "head": "head::head_center_camera",
}

CAMERA_TO_MODEL_FRAME = {
    "head": RigidTransform(
        RotationMatrix(RollPitchYaw([-1.57079632679, 0.0, -1.57079632679])),
        # RotationMatrix(RollPitchYaw([1.57079632679, 1.57079632679, 0])),
        [-0.020851, 0.0, 0.0601]
    ).inverse(),
    "left_arm": RigidTransform(
        RotationMatrix(RollPitchYaw([np.pi, 0.0, 0.0])),
        [0.0, 0.055, 0.003355]
    ).inverse(),
    "right_arm": RigidTransform(
        RotationMatrix(RollPitchYaw([0.0, np.pi, 0.0])),
        [0.0, -0.055, 0.003355]
    ).inverse(),
}

TCP_TO_MODEL_FRAME = {
    "head": RigidTransform(
        RotationMatrix(RollPitchYaw([-1.57079632679, 0.0, -1.57079632679])),
        [0.04, 0.0, 0.0601]
    ).inverse(),
    "left_arm": RigidTransform(
        RotationMatrix(RollPitchYaw([np.pi, 0.0, 0.0])),
        [0.0, 0.0, -0.2]
    ).inverse(),
    "right_arm": RigidTransform(
        RotationMatrix(RollPitchYaw([0.0, np.pi, 0.0])),
        [0.0, 0.0, -0.2]
    ).inverse(),
}

def load_pose_and_gripper(fname, idx=0):
    with open(fname, 'rb') as f:
        plan = pickle.load(f)
    plan_episode = plan[idx]

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

def load_trajectory(traj_dir, client, use_head: bool, align_mode: str = "relative"):
    # Load raw pose_data
    traj_dir = Path(traj_dir)

    model_names = ["left_arm", "right_arm"]
    if use_head:
        model_names.append("head")

    path = traj_dir / "dataset_plan.pkl"
    poses, grippers = load_pose_and_gripper(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    # change keys to match model names
    pose_data = {m: poses[m.replace("_arm", "")] for m in model_names}
    gripper_data = {m: grippers[m.replace("_arm", "")] for m in model_names if m.replace("_arm", "") in grippers}

    T = min(len(pose_data[m]) for m in model_names)

    for m in model_names:
        if len(pose_data[m]) != T:
            raise ValueError(f"Pose data for {m} has length {len(pose_data[m])}, expected {T}")

    if align_mode == "relative":
        poses_all = [pose_data[m][0] @ TCP_TO_MODEL_FRAME[m] for m in model_names]
        status_msg = client.get_status()
        robot_poses = [from_lcm_pose(next(m.pose_actual for m in status_msg.models if m.model_name == m_name)) for m_name in model_names]

        T_transform_all = {key: robot_poses[i] @ poses_all[i].inverse() for i, key in enumerate(model_names)}

    elif align_mode == "none":
        # No alignment, just use the raw poses
        T_transform_all = {key: RigidTransform(RotationMatrix(), np.zeros(3)) for key in model_names}

    # Apply transform to all poses
    poses_list = []
    grippers_list = []

    for t in range(T):
        poses = {}
        for m in model_names:
            T_model = pose_data[m][t] @ TCP_TO_MODEL_FRAME[m]
            T_aligned = T_transform_all[m] @ T_model
            poses[m] = T_aligned

        grippers = {}
        for m in model_names:
            if m in gripper_data:
                grippers[f"{m.replace('_arm', '')}_gripper"] = gripper_data[m][t]
        poses_list.append(poses)
        grippers_list.append(grippers)

    return poses_list, grippers_list


def add_meshcat_triad(meshcat, path, length=0.1, radius=0.005, opacity=1.0):
    colors = {
        "x": Rgba(1.0, 0.0, 0.0, opacity),  # red
        "y": Rgba(0.0, 1.0, 0.0, opacity),  # green
        "z": Rgba(0.0, 0.0, 1.0, opacity),  # blue
    }

    rotations = {
        "x": RotationMatrix.MakeYRotation(np.pi / 2),
        "y": RotationMatrix.MakeXRotation(-np.pi / 2),
        "z": RotationMatrix(),
    }

    for axis in "xyz":
        meshcat.SetObject(
            f"{path}/{axis}",
            Cylinder(radius, length),
            colors[axis]
        )
        translation = [length / 2 if d == axis else 0.0 for d in "xyz"]
        X = RigidTransform(rotations[axis], translation)
        meshcat.SetTransform(f"{path}/{axis}", X)


def stream_trajectory(robot_client, poses_list, grippers_list, stream_dt, meshcat, use_head=True):
    for t, (poses, grippers) in enumerate(zip(poses_list, grippers_list)):
        t0 = time.time()

        # Send command
        cmd = make_multi_frame_decoupled_message(
            model_poses=poses,
            model_to_frame_name=MODEL_TO_FRAME,
            gripper_widths=grippers,
            stream_dt=stream_dt,
            skip_blocking_init=True,
        )
        robot_client.send_command(cmd)
        robot_client.spin_once()

        # Get robot feedback
        status_msg = robot_client.get_status()

        if meshcat:
            for model_name, target_pose in poses.items():
                meshcat_path = f"target_pose/{model_name}"
                add_meshcat_triad(meshcat, path=meshcat_path, length=0.1, radius=0.005, opacity=0.2)
                meshcat.SetTransform(meshcat_path, target_pose.GetAsMatrix4())

            for m in status_msg.models:
                feedback_pose = from_lcm_pose(m.pose_actual)
                meshcat_path = f"feedback_pose/{m.model_name}"
                add_meshcat_triad(meshcat, path=meshcat_path, length=0.1, radius=0.005, opacity=1.0)
                meshcat.SetTransform(meshcat_path, feedback_pose.GetAsMatrix4())

        print(f"[step {t}] utime = {status_msg.utime}")
        for m in status_msg.models:
            pose = from_lcm_pose(m.pose_actual)
            print(f"  {m.model_name} -> {pose.translation().tolist()}")
            if not m.model_name == "base" and not (not use_head and m.model_name == "head"):
                print(f" {m.model_name} command: {poses[m.model_name].translation().tolist()}")
        print("----")

        time.sleep(max(0.0, stream_dt - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--lcm_url", default="")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--visualize", action="store_true", help="Enable Meshcat visualization")
    parser.add_argument("--use_head", action="store_true", help="Include head trajectory if available")
    parser.add_argument(
        "--align_mode",
        choices=["relative", "gripper_center", "all_center", "none"],
        default="relative",
        help="Method to align trajectory with robot poses"
    )

    args = parser.parse_args()

    client = MultiarmRobotClient(args.lcm_url)
    client.wait_for_first()
    client.reset()

    # Move to initial config using built-in helper
    move_to_initial(
        client,
        model_position={
            "right_arm": np.array([0.0, -0.0873, 0.0, -2.0944, 0.0, 0.9599, 1.5708]),
            "left_arm": np.array([0.0, 0.0873, 0.0, -2.0944, 0.0, 0.9599, -1.5708]),
            "torso": np.array([0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0]),
            "head": np.array([0.0, 0.6109]),
        },
        gripper_position={
            "right_gripper": 0.1,
            "left_gripper": 0.1,
        },
    )

    meshcat = None
    if args.visualize:
        params = MeshcatParams()
        params.url = "http://localhost:7000"
        meshcat = Meshcat(params)
        # meshcat.Delete()
        meshcat.SetProperty("target_pose", "visible", True)
        meshcat.SetProperty("feedback_pose", "visible", True)

    # Get robot's initial pose
    status_msg = client.get_status()

    print("=== Reported Poses in get_status ===")
    for m in status_msg.models:
        pose = from_lcm_pose(m.pose_actual)
        print(f"  {m.model_name} -> {pose.translation().tolist()}")

    # Load and transform trajectory
    poses, grippers = load_trajectory(args.traj_dir, client, args.use_head, align_mode=args.align_mode)

    # Stream to robot
    stream_trajectory(client, poses, grippers, args.dt, meshcat, args.use_head)


if __name__ == "__main__":
    main()