"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional
import mujoco
import mujoco.viewer
import numpy as np
import pickle
import threading
from scipy.spatial.transform import Rotation

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.rby1_wbc import RBY1WBC
from scripts.rby1_traj import load_trajectory


def _quat_angle_error(target_quat: np.ndarray, actual_quat: np.ndarray) -> float:
    target = np.asarray(target_quat, dtype=np.float64)
    actual = np.asarray(actual_quat, dtype=np.float64)
    target_norm = np.linalg.norm(target)
    actual_norm = np.linalg.norm(actual)
    if target_norm < 1e-9 or actual_norm < 1e-9:
        return float("nan")
    target /= target_norm
    actual /= actual_norm
    dot = float(np.dot(target, actual))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


class RBY1WBCTrajectory:
    def __init__(self, model_path: str, wbc: RBY1WBC, headless: bool = False, trajectory_frequency_hz: float = 10.0, poses_list: list[dict] | None = None, widths_list: list[dict] | None = None) -> None:
        self.model_path = model_path
        self.wbc = wbc
        self.headless = headless

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        
        snapshot = self._wait_for_initial_snapshot()
        if snapshot is None:
            raise RuntimeError("Failed to receive initial robot state snapshot")
        self._apply_snapshot(snapshot)

        # Viewer setup
        self.viewer = None
        self.viewer_rate = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
            )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        # Trajectory Streamer Setup
        self.poses_list = poses_list if poses_list is not None else []
        self.widths_list = widths_list if widths_list is not None else []
        self.trajectory_rate = RateLimiter(frequency=trajectory_frequency_hz, warn=False)
        self.trajectory_index = 0

        self._left_ee_body_id = self.model.body("EE_BODY_R").id
        self._right_ee_body_id = self.model.body("EE_BODY_L").id

    def _wait_for_initial_snapshot(self, timeout_sec: float = 5.0):
        deadline = time.monotonic() + timeout_sec
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = self.wbc.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                break
            time.sleep(0.005)
        return snapshot if snapshot is not None and snapshot.is_valid else None

    def _apply_snapshot(self, snapshot) -> None:
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        if qpos is None:
            return
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def site_pos(self, site_name: str, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()

    def visualize_loop(self) -> None:
        if self.viewer is None or self.viewer_rate is None:
            return

        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is not None and snapshot.is_valid:
            try:
                self._apply_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[visualize] snapshot apply error: {exc}")

        policy_left_pos, policy_left_quat, _plw, policy_right_pos, policy_right_quat, _prw, _hpp, _hpq = self.wbc.shared_targets.get_target()
        actual_left_pos, actual_left_quat = self._get_body_pose(self._left_ee_body_id)
        actual_right_pos, actual_right_quat = self._get_body_pose(self._right_ee_body_id)

        self._update_pose_markers(
            policy_left_pos,
            policy_left_quat,
            policy_right_pos,
            policy_right_quat,
            actual_left_pos,
            actual_left_quat,
            actual_right_pos,
            actual_right_quat,
        )
        left_err = self._format_ee_error("L", policy_left_pos, policy_left_quat, actual_left_pos, actual_left_quat)
        right_err = self._format_ee_error("R", policy_right_pos, policy_right_quat, actual_right_pos, actual_right_quat)
        print(f"[EE error] {left_err} | {right_err}", end="\r", flush=True)

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()

    def _update_pose_markers(
        self,
        policy_left_pos: Optional[np.ndarray],
        policy_left_quat: Optional[np.ndarray],
        policy_right_pos: Optional[np.ndarray],
        policy_right_quat: Optional[np.ndarray],
        actual_left_pos: np.ndarray,
        actual_left_quat: np.ndarray,
        actual_right_pos: np.ndarray,
        actual_right_quat: np.ndarray,
    ) -> None:
        if self.viewer is None:
            return
        scene = self.viewer.user_scn
        if scene is None:
            return

        scene.ngeom = 0
        policy_alpha = 0.5
        policy_width = 1.5
        actual_alpha = 0.9
        actual_width = 3.0

        self._add_pose_marker(scene, policy_left_pos, policy_left_quat, policy_alpha, policy_width)
        self._add_pose_marker(scene, policy_right_pos, policy_right_quat, policy_alpha, policy_width)
        self._add_pose_marker(scene, actual_left_pos, actual_left_quat, actual_alpha, actual_width)
        self._add_pose_marker(scene, actual_right_pos, actual_right_quat, actual_alpha, actual_width)

    def _add_pose_marker(
        self,
        scene: mujoco.MjvScene,
        pos: Optional[np.ndarray],
        quat: Optional[np.ndarray],
        alpha: float,
        line_width: float,
    ) -> None:
        if pos is None or quat is None or scene.ngeom >= scene.maxgeom:
            return

        pos_arr = np.asarray(pos, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)

        axis_length = 0.15
        line_size = np.zeros(3, dtype=np.float64)
        identity_mat = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        axis_colors = (
            np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, alpha], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, alpha], dtype=np.float32),
        )

        frame_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(frame_mat, quat_arr)
        rot = frame_mat.reshape(3, 3, order="F")

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
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                line_width,
                pos_arr,
                endpoint,
            )
            scene.ngeom += 1

    def _get_body_pose(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        return pos, quat

    def _format_ee_error(
        self,
        label: str,
        target_pos: Optional[np.ndarray],
        target_quat: Optional[np.ndarray],
        actual_pos: np.ndarray,
        actual_quat: np.ndarray,
    ) -> str:
        if target_pos is None or target_quat is None:
            return f"{label}: no target"
        pos_err = np.asarray(target_pos, dtype=np.float64) - np.asarray(actual_pos, dtype=np.float64)
        pos_err_norm = float(np.linalg.norm(pos_err))
        quat_err = _quat_angle_error(target_quat, actual_quat)
        return f"{label} dpos={pos_err_norm:.4f} dq={quat_err:.4f}"

    @staticmethod
    def _transform_to_pose(transform) -> tuple[np.ndarray, np.ndarray]:
        mat = transform.as_matrix() if hasattr(transform, "as_matrix") else np.asarray(transform)
        pos = mat[:3, 3].astype(float)
        quat_xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
        return pos, quat_wxyz

    def trajectory_loop(self) -> None:
        """Stream the full trajectory to the WBC at trajectory_rate until finished."""
        input("Press Enter to start streaming the trajectory?")
        num_steps = min(len(self.poses_list), len(self.widths_list))
        while self.trajectory_index < num_steps:

            # Get Gripper (and Head) Poses
            pose_entry = self.poses_list[self.trajectory_index]
            left_transform = pose_entry["left_arm"]
            right_transform = pose_entry["right_arm"]
            left_pos, left_quat = self._transform_to_pose(left_transform)
            right_pos, right_quat = self._transform_to_pose(right_transform)

            head_pos = head_quat = None
            if "head" in pose_entry:
                head_transform = pose_entry["head"]
                head_pos, head_quat = self._transform_to_pose(head_transform)

            # Get Gripper Widths
            width_entry = self.widths_list[self.trajectory_index]
            left_width = width_entry["left_width"]
            right_width = width_entry["right_width"]

            duration = self.trajectory_rate.dt
            timestamp = time.monotonic()

            self.wbc.update_targets(
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                left_width=left_width,
                right_width=right_width,
                head_pos=head_pos,
                head_quat=head_quat,
                duration=duration,
                timestamp=timestamp,
            )
            self.trajectory_index += 1 
            self.trajectory_rate.sleep()  

        print("[trajectory] streaming complete.")


    def run(self) -> None:
        self._trajectory_thread = threading.Thread(
            target=self.trajectory_loop, name="trajectory_streamer", daemon=True
        )
        self._trajectory_thread.start()

        try:
            if self.headless:
                while True:
                    self.viewer_rate.sleep()
            else:
                while self.viewer.is_running():
                    self.visualize_loop()
        except KeyboardInterrupt:
            return

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body IK GUI decoupled from WBC thread")
    parser.add_argument(
        "--address",
        default=os.environ.get("RBY1_ROBOT", "localhost:50051"),
        help="Robot gRPC address (default: env RBY1_ROBOT or localhost:50051)",
    )
    parser.add_argument(
        "--model",
        default=PROJECT_ROOT + "/model/rby1/rby1.xml",
        help="Path to the MuJoCo model to visualize",
    )
    parser.add_argument(
        "--trajectory",
        default=PROJECT_ROOT + "/demo/dataset_plan.pkl",
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
    
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    wbc = RBY1WBC(model_path=args.model, address=args.address, ik_frequency_hz=100.0, trajectory_frequency_hz=10.0)
    wbc.start()

    poses_list, widths_list = load_trajectory(traj_dir=args.trajectory, client=wbc, index=args.index, use_head=False, align_mode="relative")
    gui = None
    try:
        gui = RBY1WBCTrajectory(model_path=args.model, wbc=wbc, headless=args.headless, trajectory_frequency_hz=10.0, poses_list=poses_list, widths_list=widths_list)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()

if __name__ == "__main__":
    main()
