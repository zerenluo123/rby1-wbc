from __future__ import annotations

import threading
import time
from typing import Optional

from loop_rate_limiters import RateLimiter

from .ee_targets import EETargets
from .state_visualizer import StateVisualizer
from .whole_body_control import RBY1WBC
from .whole_body_ik import RBY1WholeBodyIK

class RBY1WBCApp:
    """Reusable visualizer + trajectory loop for streaming WBC targets."""

    def __init__(
        self,
        wbc: RBY1WBC,
        headless: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        self.wbc = wbc
        self.headless = headless
        self.model_path = model_path or self.wbc.model_path

        self.visualizer: Optional[StateVisualizer] = None
        self.viewer_rate: Optional[RateLimiter] = None
        if not self.headless:
            snapshot = self.wbc.wait_for_first_state()
            qpos = self.wbc.snapshot_to_qpos(snapshot)
            self.visualizer = StateVisualizer(
                model_path=self.model_path, initial_qpos=qpos, print_errors=False
            )
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        self.trajectory_rate = RateLimiter(
            frequency=self.wbc.trajectory_frequency_hz, warn=False
        )

        self._stop_event = threading.Event()
        self._trajectory_thread: Optional[threading.Thread] = None

    # ----- Hooks for subclasses -------------------------------------------------
    def get_target(self) -> Optional[EETargets]:
        raise NotImplementedError

    # ----- Runtime --------------------------------------------------------------
    def visualize_loop(self) -> None:
        snapshot = self.wbc.get_latest_robot_state()
        qpos = self.wbc.snapshot_to_qpos(snapshot)
        targets = self.wbc.ee_targets.get_target()
        self.visualizer.render(qpos, targets)
        self.viewer_rate.sleep()

    def trajectory_loop(self) -> None:
        while not self._stop_event.is_set():
            target = self.get_target()
            if target is None:
                self.trajectory_rate.sleep()
                continue

            duration = target.duration if target.duration and target.duration > 0.0 else self.trajectory_rate.dt
            timestamp = target.timestamp if target.timestamp and target.timestamp > 0.0 else time.monotonic()
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

        self._stop_event.set()

    def run(self) -> None:
        self._trajectory_thread = threading.Thread(
            target=self.trajectory_loop, name="trajectory_streamer", daemon=True
        )
        self._trajectory_thread.start()

        try:
            if self.headless or self.visualizer is None:
                while not self._stop_event.is_set() and self._trajectory_thread.is_alive():
                    time.sleep(0.01)
            else:
                viewer = self.visualizer.viewer
                while viewer.is_running() and not self._stop_event.is_set():
                    self.visualize_loop()
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            self._stop_event.set()

    def close(self) -> None:
        self._stop_event.set()
        if self._trajectory_thread is not None:
            self._trajectory_thread.join(timeout=1.0)
            
        if not self.headless and self.visualizer is not None:
            try:
                self.visualizer.viewer.close()
            except Exception:
                pass