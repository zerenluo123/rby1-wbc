"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import threading

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from rby1.state_visualizer import StateVisualizer

class RBY1WBCGui:
    def __init__(self, wbc: RBY1WBC):
        self.wbc = wbc
         
        # Initialize visualizer
        snapshot = self.wbc.wait_for_first_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        self.visualizer = StateVisualizer(
            model_path=PROJECT_ROOT + "/model/rby1/rby1_mocap.xml", initial_qpos = qpos, print_errors=True
        )
        self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        # Initialize mocap targets via visualizer helper
        self.mocap_ids = self.visualizer.init_mocap_targets()

        # Trajectory Streamer Setup
        self.trajectory_rate = RateLimiter(
            frequency=self.wbc.trajectory_frequency_hz, warn=False
        )

    def visualize_loop(self) -> None:
        snapshot = self.wbc.get_latest_robot_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        targets = self.wbc.ee_targets.get_target()
        self.visualizer.render(qpos, targets)
        self.viewer_rate.sleep()

    def trajectory_loop(self) -> None:
        while True:
            (
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                head_pos,
                head_quat,
            ) = self.visualizer.get_mocap_targets(self.mocap_ids)
            
            duration = 1.0 / float(self.wbc.trajectory_frequency_hz)
            timestamp = time.monotonic()

            self.wbc.update_targets(
                duration,
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                left_width=None,
                right_width=None,
                head_pos=head_pos,
                head_quat=head_quat,
                timestamp=timestamp,
            )
            self.trajectory_rate.sleep()

    def run(self) -> None:
        self._trajectory_thread = threading.Thread(
            target=self.trajectory_loop, name="trajectory_streamer", daemon=True
        )
        self._trajectory_thread.start()

        try:
            viewer = self.visualizer.viewer
            while viewer.is_running():
                self.visualize_loop()
        except KeyboardInterrupt:
            return

    def close(self) -> None:        
        if self._trajectory_thread is not None and self._trajectory_thread.is_alive():
            self._trajectory_thread.join(timeout=1.0)
        try:
            self.visualizer.viewer.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body IK GUI decoupled from WBC thread")
    args = parser.parse_args()

    wbc = RBY1WBC()
    wbc.start()
    gui = None
    try:
        gui = RBY1WBCGui(wbc=wbc)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()

if __name__ == "__main__":
    main()
