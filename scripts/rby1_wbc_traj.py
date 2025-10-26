"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import pickle

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.rby1_wbc import RBY1WBC
from demo.trajectory import Trajectory

class RBY1WBCTrajectory:
    def __init__(self, model_path: str, wbc: RBY1WBC, headless: bool = False, trajectory: Trajectory = None):
        self.model_path = model_path
        self.wbc = wbc
        self.headless = headless

        self.viewer = None if headless else self._init_viewer(model_path)
        self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        self.trajectory = trajectory

    def _init_viewer(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        
        snapshot = self._wait_for_initial_snapshot()
        if snapshot is None:
            raise RuntimeError("Failed to receive initial robot state snapshot")
        self._apply_snapshot(snapshot)

        viewer = mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        )
        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
        return viewer

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
        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is not None and snapshot.is_valid:
            try:
                self._apply_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[visualize] snapshot apply error: {exc}")

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()

    def target_loop(self) -> None:
        # TODO: update_target based upon trajectory
        self.wbc.update_targets(left_pos, left_quat, right_pos, right_quat, self.data.qpos)

    def run(self) -> None:
        if self.headless:
            try:
                while True:
                    self.viewer_rate.sleep()
            except KeyboardInterrupt:
                return
        else:
            try:
                while self.viewer.is_running():
                    self.visualize_loop()
            except KeyboardInterrupt:
                pass

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

def load_trajectory(path: str) -> Trajectory | None:
    # Load Trajectory
    trajectory = None
    try:
        with open(path, "rb") as f:
            trajectory_data = pickle.load(f)

        num_eps = len(trajectory_data)
        if num_eps == 0:
            print("[Trajectory] WARNING: demo dataset is empty, proceeding with no trajectory")
            trajectory = None
        else:        
            print(f"[Trajectory] Loaded {path} with {num_eps} episodes found")
            # user_input = input(f"Enter episode index to use (0 to {num_eps - 1}, default: 0): ").strip()
            # episode_idx = int(user_input) if user_input.isdigit() else 0
            episode_idx = 0 # currently fixed
            trajectory = Trajectory(trajectory_data[episode_idx])
            print(f"[Trajectory] Loaded a trajectory for episode {trajectory.episode_name} (Length: {trajectory.length} | Frequency: {trajectory.frequency} Hz)")
        
    except FileNotFoundError:
        print(f"[Trajectory] WARNING: demo file not found at {path}, proceeding without trajectory")
        trajectory = None
    except Exception as exc:
        print(f"[Trajectory] WARNING: failed to load demo from {path}: {exc}")
        trajectory = None

    return trajectory

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

    trajectory = load_trajectory(args.trajectory)
    if not trajectory:
        raise Exception("No valid trajectory loaded, cannot proceed.")

    gui = None
    try:
        gui = RBY1WBCTrajectory(model_path=args.model, wbc=wbc, headless=args.headless, trajectory=trajectory)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()

if __name__ == "__main__":
    main()
