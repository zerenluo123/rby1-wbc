"""High-level robot interface tailored for policy streaming.

This module exposes :class:`RBY1PolicyRobot`, a thin convenience wrapper around
the existing :class:`~control.rby1_wbc.RBY1WBC` whole-body controller.  The
class keeps a short history of end-effector poses expressed in Cartesian space
and provides a helper for dispatching Cartesian targets produced by a learned
policy.  The implementation intentionally mirrors the behaviour in
``scripts/rby1_wbc_traj.py`` where differential IK solutions are sent to the
pybind realtime driver.

Typical usage::

    robot = RBY1PolicyRobot()
    robot.start()
    robot.wait_until_ready()
    obs = robot.get_observation_window(horizon=2)
    robot.apply_action(policy_action)

The class is thread-safe and designed to run alongside a camera streamer in a
policy inference loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .rby1_wbc import RBY1WBC


@dataclass(frozen=True)
class RobotObservation:
    """Compact representation of a single policy observation."""

    timestamp: float
    left_pos: np.ndarray
    left_rot_axis_angle: np.ndarray
    right_pos: np.ndarray
    right_rot_axis_angle: np.ndarray
    head_pos: np.ndarray
    head_rot_axis_angle: np.ndarray
    left_width: float
    right_width: float


def _quat_wxyz_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector to a quaternion in (w, x, y, z) order."""

    axis_angle = np.asarray(axis_angle, dtype=float)
    if axis_angle.shape != (3,):  # pragma: no cover - defensive
        raise ValueError("Expected axis-angle vector with shape (3,)")
    quat_xyzw = Rotation.from_rotvec(axis_angle).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


class RBY1PolicyRobot:
    """High-level helper for streaming policy commands to the robot.

    The object internally owns a :class:`RBY1WBC` instance and mirrors the
    MuJoCo model so that policy-facing Cartesian states can be produced.  A
    fixed-size deque keeps the most recent observations which are returned in a
    format that matches the expected keys of the policy server (e.g.
    ``gripper_left_eef_pos``).
    """

    LEFT_SITE = "end_effector_l"
    RIGHT_SITE = "end_effector_r"
    HEAD_SITE = "head"

    def __init__(
        self,
        config_path: str | None = None,
        buffer_size: int = 64,
    ) -> None:
        self._wbc = RBY1WBC(config_path=config_path) if config_path else RBY1WBC()
        self._buffer: Deque[RobotObservation] = deque(maxlen=int(max(buffer_size, 1)))
        self._buffer_lock = threading.Lock()
        self._buffer_ready = threading.Event()
        self._stop_event = threading.Event()
        self._buffer_thread: Optional[threading.Thread] = None

        # Mirror of the MuJoCo model used for FK computations in the streaming
        # thread.  The worker keeps an internal model already, but sharing it
        # across threads is unsafe, so we create our own lightweight copy.
        self._model = mujoco.MjModel.from_xml_path(self._wbc.model_path)
        self._data = mujoco.MjData(self._model)
        self._left_site = self._model.site(self.LEFT_SITE).id
        self._right_site = self._model.site(self.RIGHT_SITE).id
        self._head_site = self._model.site(self.HEAD_SITE).id

    @property
    def dt(self) -> float:
        """Return the IK loop period of the underlying controller."""

        return 1.0 / float(self._wbc.trajectory_frequency_hz)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the realtime controller threads."""

        self._wbc.start()
        if self._buffer_thread is not None and self._buffer_thread.is_alive():
            return
        self._stop_event.clear()
        self._buffer_thread = threading.Thread(
            target=self._buffer_worker,
            name="policy-robot-buffer",
            daemon=True,
        )
        self._buffer_thread.start()

    def stop(self) -> None:
        """Stop controller threads and flush buffers."""

        self._stop_event.set()
        if self._buffer_thread is not None:
            self._buffer_thread.join(timeout=1.0)
            self._buffer_thread = None
        self._wbc.stop()
        with self._buffer_lock:
            self._buffer.clear()
        self._buffer_ready.clear()

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        """Block until the first valid snapshot is available."""

        deadline = time.monotonic() + max(timeout, 0.0)
        self._wbc.wait_for_first_state(timeout_sec=timeout)
        remaining = max(0.0, deadline - time.monotonic())
        if not self._buffer_ready.wait(timeout=remaining):
            raise TimeoutError("Timed out waiting for buffered robot observations")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _buffer_worker(self) -> None:
        """Continuously fetch robot snapshots and append them to the buffer."""

        poll_period = max(self.dt * 0.5, 0.01)
        while not self._stop_event.is_set():
            snapshot = self._wbc.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                qpos = self._wbc.snapshot_to_qpos(snapshot)
                if qpos is not None:
                    observation = self._qpos_to_observation(qpos, time.monotonic())
                    with self._buffer_lock:
                        self._buffer.append(observation)
                        self._buffer_ready.set()
            time.sleep(poll_period)

    def wait_for_observations(self, count: int, timeout: float = 2.0) -> bool:
        """Block until the buffer stores ``count`` observations.

        Returns ``True`` when the desired number of entries are available and
        ``False`` if the timeout expires first.
        """

        count = int(max(count, 1))
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() <= deadline:
            with self._buffer_lock:
                if len(self._buffer) >= count:
                    return True
            time.sleep(min(self.dt, 0.01))
        return False

    def _qpos_to_observation(self, qpos: np.ndarray, timestamp: float) -> RobotObservation:
        """Convert a joint configuration into Cartesian observations."""

        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)

        def _site_pose(site_id: int) -> tuple[np.ndarray, np.ndarray]:
            pos = self._data.site_xpos[site_id].astype(float)
            mat = self._data.site_xmat[site_id].reshape(3, 3, order="F")
            rotvec = Rotation.from_matrix(mat).as_rotvec()
            return pos.copy(), rotvec

        left_pos, left_rotvec = _site_pose(self._left_site)
        right_pos, right_rotvec = _site_pose(self._right_site)
        head_pos, head_rotvec = _site_pose(self._head_site)

        left_width, right_width = self._wbc.get_latest_gripper_widths()

        return RobotObservation(
            timestamp=float(timestamp),
            left_pos=left_pos,
            left_rot_axis_angle=left_rotvec,
            right_pos=right_pos,
            right_rot_axis_angle=right_rotvec,
            head_pos=head_pos,
            head_rot_axis_angle=head_rotvec,
            left_width=float(left_width),
            right_width=float(right_width),
        )

    def get_latest_observation(self) -> Optional[RobotObservation]:
        with self._buffer_lock:
            return self._buffer[-1] if self._buffer else None

    def get_observation_window(
        self,
        horizon: int,
        stride: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Return stacked arrays compatible with the policy server.

        Parameters
        ----------
        horizon:
            Number of historical steps to collect (oldest to newest).
        stride:
            Step between samples inside the internal buffer.  ``stride=1`` means
            consecutive observations, ``stride=2`` picks every other entry, etc.
        """

        horizon = int(max(horizon, 1))
        stride = int(max(stride, 1))
        with self._buffer_lock:
            if not self._buffer:
                raise RuntimeError("Robot observation buffer is empty")
            items: Iterable[RobotObservation] = list(self._buffer)[-horizon * stride :: stride]
        observations = list(items)
        if len(observations) < horizon:
            # If we could not gather enough data yet we simply replicate the
            # oldest entry to preserve shapes.  This mirrors the behaviour in
            # many robotics stacks where the first few iterations bootstrap the
            # history with the initial measurement.
            observations = [observations[0]] * (horizon - len(observations)) + observations

        def stack(attr: str) -> np.ndarray:
            return np.stack([getattr(obs, attr) for obs in observations], axis=0)

        timestamps = np.array([obs.timestamp for obs in observations], dtype=np.float64)
        return {
            "timestamp": timestamps,
            "gripper_left_eef_pos": stack("left_pos"),
            "gripper_left_eef_rot_axis_angle": stack("left_rot_axis_angle"),
            "gripper_right_eef_pos": stack("right_pos"),
            "gripper_right_eef_rot_axis_angle": stack("right_rot_axis_angle"),
            "head_eef_pos": stack("head_pos"),
            "head_eef_rot_axis_angle": stack("head_rot_axis_angle"),
            "gripper_left_gripper_width": stack("left_width").reshape(-1, 1),
            "gripper_right_gripper_width": stack("right_width").reshape(-1, 1),
        }

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def apply_action(
        self,
        action: Dict[str, np.ndarray],
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Forward a single Cartesian action to the WBC.

        Parameters
        ----------
        action:
            Dictionary with keys ``left_pos``, ``left_rot_axis_angle``,
            ``right_pos``, ``right_rot_axis_angle`` and optionally ``head_pos``,
            ``head_rot_axis_angle`` and gripper widths.  Values are numpy arrays
            of shape ``(3,)`` (for poses) or scalars for grippers.
        duration:
            Desired execution time.  Defaults to the WBC trajectory period.
        timestamp:
            Optional timestamp to feed into the shared target queue.
        """

        required = [
            "left_pos",
            "left_rot_axis_angle",
            "right_pos",
            "right_rot_axis_angle",
        ]
        for key in required:
            if key not in action:
                raise KeyError(f"Missing required action field '{key}'")

        left_quat = _quat_wxyz_from_axis_angle(action["left_rot_axis_angle"])
        right_quat = _quat_wxyz_from_axis_angle(action["right_rot_axis_angle"])

        head_pos = action.get("head_pos")
        head_quat = None
        if head_pos is not None and action.get("head_rot_axis_angle") is not None:
            head_quat = _quat_wxyz_from_axis_angle(action["head_rot_axis_angle"])
        else:
            head_pos = None

        left_width = float(action["left_gripper_width"]) if "left_gripper_width" in action else None
        right_width = float(action["right_gripper_width"]) if "right_gripper_width" in action else None

        self._wbc.update_targets(
            left_pos=np.asarray(action["left_pos"], dtype=float),
            left_quat=left_quat,
            right_pos=np.asarray(action["right_pos"], dtype=float),
            right_quat=right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=None if head_pos is None else np.asarray(head_pos, dtype=float),
            head_quat=head_quat,
            duration=self.dt if duration is None else float(duration),
            timestamp=timestamp,
        )


__all__ = ["RBY1PolicyRobot", "RobotObservation"]