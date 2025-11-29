"""Meta Quest teleoperation frontend for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from rby1.ee_targets import EETargets
from rby1.state_visualizer import StateVisualizer
from teleop.teleop_vr import TeleopVR

class RBY1WBCTeleopVR:
    def __init__(self, wbc: RBY1WBC, teleop: TeleopVR, headless: bool = False) -> None:
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

        # Trajectory streamer setup
        self.teleop = teleop
        self.trajectory_rate = RateLimiter(
            frequency=self.wbc.trajectory_frequency_hz, warn=False
        )
        self._stop_event = threading.Event()
        self._trajectory_thread: Optional[threading.Thread] = None
        self.teleop.start()

    def visualize_loop(self) -> None:
        snapshot = self.wbc.get_latest_robot_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        targets = self.wbc.ee_targets.get_target()
        self.visualizer.render(qpos, targets)
        self.viewer_rate.sleep()

    def trajectory_loop(self) -> None:
        """Stream live targets from the Meta Quest to the WBC."""
        while not self._stop_event.is_set():
            target: EETargets = self.teleop.compute_target()
            if target is None:
                self.trajectory_rate.sleep()
                continue

            duration = self.trajectory_rate.dt
            timestamp = time.monotonic()

            self.wbc.update_targets(
                duration,
                left_pos=target.left_pos,
                left_quat=target.left_quat,
                right_pos=target.right_pos,
                right_quat=target.right_quat,
                left_width=target.left_width,
                right_width=target.right_width,
                head_pos=target.head_pos,
                head_quat=target.head_quat,
                timestamp=timestamp,
            )
            self.trajectory_rate.sleep()

    def run(self) -> None:
        self._trajectory_thread = threading.Thread(
            target=self.trajectory_loop, name="teleop_streamer", daemon=True
        )
        self._trajectory_thread.start()

        try:
            if self.headless:
                while not self._stop_event.is_set():
                    time.sleep(0.01)
            else:
                viewer = self.visualizer.viewer
                while viewer.is_running():
                    self.visualize_loop()
        except KeyboardInterrupt:
            self._stop_event.set()
            return

    def close(self) -> None:
        self._stop_event.set()
        if self._trajectory_thread is not None:
            self._trajectory_thread.join(timeout=1.0)
        self.teleop.stop()
        if not self.headless:
            try:
                self.visualizer.viewer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body teleop GUI driven by Meta Quest"
    )
    parser.add_argument(
        "--local_ip",
        required=True,
        help="Local Wi-Fi (or LAN) IP address where the Meta Quest should stream controller poses.",
    )
    parser.add_argument(
        "--meta_quest_ip",
        required=True,
        help="IP address of the Meta Quest headset on the same network.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer (useful for debugging controller only).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist computed teleop trajectory as a dataset-style pickle under demo/.",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    wbc = RBY1WBC()
    wbc.start()
    teleop = TeleopVR(
        wbc=wbc,
        local_ip=args.local_ip,
        meta_quest_ip=args.meta_quest_ip,
        save_trajectory=args.save,
    )
    if not teleop.initialize():
        raise Exception("Teleoperation can not be initialized!")

    gui = None
    try:
        gui = RBY1WBCTeleopVR(wbc=wbc, teleop=teleop, headless=args.headless)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()


if __name__ == "__main__":
    main()
