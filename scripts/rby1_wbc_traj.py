"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import pickle
import threading
from scipy.spatial.transform import Rotation

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from demo.trajectory_loader import load_trajectory
from rby1.state_visualizer import StateVisualizer


class RBY1WBCTrajectory:
    def __init__(
        self,
        wbc: RBY1WBC,
        headless: bool = False,
        poses_list: list[dict] | None = None,
        widths_list: list[dict] | None = None,
    ) -> None:
        self.wbc = wbc
        self.headless = headless
        self.visualizer = None
        if not self.headless:
            snapshot = self.wbc.wait_for_first_state()
            qpos = self.wbc.snapshot_to_qpos(snapshot)
            self.visualizer = StateVisualizer(
                model_path=self.wbc.model_path, initial_qpos = qpos, print_errors=True
            )
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)


        # Trajectory Streamer Setup
        self.poses_list = poses_list if poses_list is not None else []
        self.widths_list = widths_list if widths_list is not None else []
        self.trajectory_rate = RateLimiter(
            frequency=self.wbc.trajectory_frequency_hz, warn=False
        )
        self.trajectory_index = 0

    def visualize_loop(self) -> None:
        snapshot = self.wbc.get_latest_robot_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        targets = self.wbc.ee_targets.get_target()
        self.visualizer.render(qpos, targets)
        self.viewer_rate.sleep()

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

    def trajectory_loop(self) -> None:
        """Stream the full trajectory to the WBC at trajectory_rate until finished."""
        input("Press [Enter] to start streaming the trajectory.")
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
                duration,
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                left_width=left_width,
                right_width=right_width,
                head_pos=head_pos,
                head_quat=head_quat,
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
                while self._trajectory_thread.is_alive():
                    time.sleep(0.01)
            else:
                viewer = self.visualizer.viewer
                while viewer.is_running():
                    self.visualize_loop()
        except KeyboardInterrupt:
            return

    def close(self) -> None:
        if self._trajectory_thread is not None and self._trajectory_thread.is_alive():
            self._trajectory_thread.join(timeout=1.0)
        if not self.headless:
            try:
                self.visualizer.viewer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body IK GUI decoupled from WBC thread"
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

    wbc = RBY1WBC()
    wbc.start()

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
