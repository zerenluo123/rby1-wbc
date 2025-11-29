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

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from demo.trajectory_loader import load_trajectory
from rby1.ee_targets import EETargets
from rby1.rby1_wbc_app import WBCStreamingApp


class RBY1WBCTrajectory(WBCStreamingApp):
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
