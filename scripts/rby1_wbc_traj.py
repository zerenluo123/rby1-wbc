"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
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

        self.viewer = None
        self.viewer_rate = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
            )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        self.poses_list = poses_list if poses_list is not None else []
        self.widths_list = widths_list if widths_list is not None else []
        self.trajectory_rate = RateLimiter(frequency=trajectory_frequency_hz, warn=False)
        self.trajectory_index = 0

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

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()

    @staticmethod
    def _transform_to_pose(transform) -> tuple[np.ndarray, np.ndarray]:
        mat = transform.as_matrix() if hasattr(transform, "as_matrix") else np.asarray(transform)
        pos = mat[:3, 3].astype(float)
        quat_xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
        return pos, quat_wxyz

    def trajectory_loop(self) -> None:
        """Stream the full trajectory to the WBC at trajectory_rate until finished."""
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

            self.wbc.update_targets(
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                left_width=left_width,
                right_width=right_width,
                head_pos=head_pos,
                head_quat=head_quat,
            )
            print(f"[trajectory] step {self.trajectory_index}/{num_steps-1}")
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
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer (useful for debugging controller only).",
    )
    
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    wbc = RBY1WBC(model_path=args.model, address=args.address, ik_frequency_hz=100.0)
    wbc.start()

    poses_list, widths_list = load_trajectory(traj_dir=args.trajectory, client=wbc, use_head=False, align_mode="relative")
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
