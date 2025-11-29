"""Unified teleoperation frontend for the RBY1 whole-body controller."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from rby1.ee_targets import EETargets
from rby1.state_visualizer import StateVisualizer
from teleop.teleop_iphone import TeleopIphone
from teleop.teleop_vr import TeleopVR


class RBY1WBCTeleop:
    def __init__(self, wbc: RBY1WBC, teleop: Any, headless: bool = False) -> None:
        self.wbc = wbc
        self.teleop = teleop
        self.headless = headless

        self.visualizer = None
        if not self.headless:
            snapshot = self.wbc.wait_for_first_state()
            qpos = self.wbc.snapshot_to_qpos(snapshot)
            self.visualizer = StateVisualizer(
                model_path=self.wbc.model_path, initial_qpos=qpos, print_errors=True
            )
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        # Trajectory streamer setup
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
        while not self._stop_event.is_set():
            target: Optional[EETargets] = self.teleop.compute_target()
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
            except Exception:
                pass

def build_teleop(mode:str, wbc: RBY1WBC, config: Dict[str, Any], save_trajectory: bool) -> Any:
    if mode == "vr":
        local_ip = config.get("local_ip")
        meta_quest_ip = config.get("meta_quest_ip")
        local_port = int(config.get("local_port", 5005))
        meta_quest_port = int(config.get("meta_quest_port", 6000))
        if not local_ip or not meta_quest_ip:
            raise ValueError("Meta Quest mode requires 'local_ip' and 'meta_quest_ip' in config/teleop_vr.yaml.")
        teleop = TeleopVR(
            wbc=wbc,
            local_ip=str(local_ip),
            meta_quest_ip=str(meta_quest_ip),
            local_port=local_port,
            meta_quest_port=meta_quest_port,
            save_trajectory=save_trajectory,
        )
    elif mode == "iphone":
        host = str(config.get("host", "0.0.0.0"))
        port = int(config.get("port", 5555))
        portrait = bool(config.get("portrait", False))
        teleop = TeleopIphone(
            wbc=wbc,
            host=host,
            port=port,
            save_trajectory=save_trajectory,
            use_portrait_mode=portrait,
        )
    else:
        raise ValueError(f"Unsupported teleop mode: {mode}")
    return teleop

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBY1 whole-body teleop frontend for VR or iPhone clients."
    )
    parser.add_argument(
        "--mode",
        choices=["vr", "iphone"],
        help="Select teleop mode; defaults to 'vr' when not provided.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist computed teleop trajectory as a dataset-style pickle under demo/.",
    )
    args = parser.parse_args()

    mode = args.mode or "vr"
    headless = bool(args.headless)
    save_trajectory = bool(args.save)
    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # Load config
    config_path = Path(PROJECT_ROOT + f"/config/teleop_{mode}.yaml")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Exception while loading config file: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"WBC config at {config_path} must be a mapping.")

    # Run 
    wbc = RBY1WBC()
    wbc.start()
    teleop = build_teleop(mode, wbc, config, save_trajectory=save_trajectory)
    if not teleop.initialize():
        wbc.stop()
        raise RuntimeError("Teleoperation can not be initialized!")

    gui: Optional[RBY1WBCTeleop] = None
    try:
        gui = RBY1WBCTeleop(wbc=wbc, teleop=teleop, headless=headless)
        gui.run()
    finally:
        try:
            if gui is not None:
                gui.close()
            else:
                teleop.stop()
        finally:
            wbc.stop()


if __name__ == "__main__":
    main()
