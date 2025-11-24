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
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional, Union

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import yaml

from ik.rby1_whole_body_ik import RBY1WholeBodyIK
from .rby1_wbc import RBY1WBC

try:  # pragma: no cover - optional dependency
    import mujoco.viewer as mj_viewer
except (ImportError, AttributeError):
    mj_viewer = None


@dataclass(frozen=True)
class RobotObservation:
    """Compact representation of a single policy observation."""

    timestamp: float
    left_tf: np.ndarray
    right_tf: np.ndarray
    head_tf: np.ndarray
    left_width: float
    right_width: float


@dataclass(frozen=True)
class _SimSnapshot:
    qpos: np.ndarray
    is_valid: bool = True


def _build_interp1d(t: np.ndarray, values: np.ndarray) -> interp1d:
    return interp1d(
        t,
        values,
        axis=0,
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )


class _PoseInterpolator:
    def __init__(self, timestamps: np.ndarray, poses: np.ndarray) -> None:
        self._t = np.asarray(timestamps, dtype=float)
        self._t_min = float(self._t[0])
        self._t_max = float(self._t[-1])
        self._pos_interp = _build_interp1d(self._t, poses[:, :3])
        if len(self._t) >= 2:
            self._rot_slerp = Slerp(self._t, Rotation.from_rotvec(poses[:, 3:]))
            self._rot_static = None
        else:
            self._rot_slerp = None
            self._rot_static = Rotation.from_rotvec(poses[0, 3:])

    def __call__(self, query_t: np.ndarray) -> np.ndarray:
        query = np.asarray(query_t, dtype=float)
        needs_expand = query.ndim == 0
        if needs_expand:
            query = query[None]
        query = np.clip(query, self._t_min, self._t_max)
        pos = self._pos_interp(query)
        if self._rot_slerp is not None:
            rot = self._rot_slerp(query).as_rotvec()
        else:
            rot = np.tile(self._rot_static.as_rotvec(), (len(query), 1))
        pose = np.concatenate([pos, rot], axis=-1)
        if needs_expand:
            pose = pose[0]
        return pose

def _tf_to_posevec(tf: np.ndarray) -> np.ndarray:
    pose = np.zeros(6, dtype=float)
    pose[:3] = tf[:3, 3]
    pose[3:] = Rotation.from_matrix(tf[:3, :3]).as_rotvec()
    return pose


def _posevec_to_tf(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    tf = np.tile(np.eye(4), (pose.shape[0], 1, 1))
    tf[:, :3, 3] = pose[:, :3]
    tf[:, :3, :3] = Rotation.from_rotvec(pose[:, 3:]).as_matrix()
    return tf


def _pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pos_dist = np.linalg.norm(b[:3] - a[:3])
    rot_a = Rotation.from_rotvec(a[3:])
    rot_b = Rotation.from_rotvec(b[3:])
    rot_dist = (rot_b * rot_a.inv()).magnitude()
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        times = np.asarray(times, dtype=float)
        poses = np.asarray(poses, dtype=float)
        assert times.ndim == 1 and poses.ndim == 2
        assert times.size == poses.shape[0]
        if times.size == 1:
            self._single_step = True
            self._times = times
            self._poses = poses
        else:
            assert np.all(times[1:] >= times[:-1])
            self._single_step = False
            self._times = times
            self._pos_interp = interp1d(times, poses[:, :3], axis=0, assume_sorted=True)
            self._rot_interp = Slerp(times, Rotation.from_rotvec(poses[:, 3:]))

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def poses(self) -> np.ndarray:
        if self._single_step:
            return self._poses
        poses = np.zeros((len(self._times), 6), dtype=float)
        poses[:, :3] = self._pos_interp(self._times)
        poses[:, 3:] = self._rot_interp(self._times).as_rotvec()
        return poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        start_t = float(start_t)
        end_t = float(end_t)
        assert start_t <= end_t
        times = self.times
        keep = (start_t < times) & (times < end_t)
        keep_times = times[keep]
        new_times = np.concatenate([[start_t], keep_times, [end_t]])
        new_times = np.unique(new_times)
        poses = self(new_times)
        return PoseTrajectoryInterpolator(new_times, poses)

    def drive_to_waypoint(
        self,
        pose: np.ndarray,
        time: float,
        curr_time: float,
        max_pos_speed: float,
        max_rot_speed: float,
    ) -> "PoseTrajectoryInterpolator":
        pose = np.asarray(pose, dtype=float)
        time = max(float(time), float(curr_time))
        curr_pose = self(curr_time)
        pos_dist, rot_dist = _pose_distance(curr_pose, pose)
        duration = time - curr_time
        if max_pos_speed > 0:
            duration = max(duration, pos_dist / max_pos_speed)
        if max_rot_speed > 0:
            duration = max(duration, rot_dist / max_rot_speed)
        end_time = curr_time + max(duration, 0.0)
        trimmed = self.trim(curr_time, curr_time)
        new_times = np.append(trimmed.times, [end_time], axis=0)
        new_poses = np.append(trimmed.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(new_times, new_poses)

    def schedule_waypoint(
        self,
        pose: np.ndarray,
        time: float,
        curr_time: Optional[float],
        last_waypoint_time: Optional[float],
        max_pos_speed: float,
        max_rot_speed: float,
    ) -> "PoseTrajectoryInterpolator":
        pose = np.asarray(pose, dtype=float)
        time = float(time)
        start_time = self.times[0]
        end_time = self.times[-1]
        if curr_time is not None:
            curr_time = float(curr_time)
            if time <= curr_time:
                return self
            start_time = max(start_time, curr_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(curr_time, last_waypoint_time)
            else:
                end_time = curr_time
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        trimmed = self.trim(start_time, end_time)
        duration = time - end_time
        end_pose = trimmed(end_time)
        pos_dist, rot_dist = _pose_distance(pose, end_pose)
        if max_pos_speed > 0:
            duration = max(duration, pos_dist / max_pos_speed)
        if max_rot_speed > 0:
            duration = max(duration, rot_dist / max_rot_speed)
        final_time = end_time + max(duration, 0.0)
        new_times = np.append(trimmed.times, [final_time], axis=0)
        new_poses = np.append(trimmed.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(new_times, new_poses)

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        if self._single_step:
            if np.isscalar(t):
                return self._poses[0]
            return np.tile(self._poses[0], (len(t), 1))
        if np.isscalar(t):
            query = np.array([t], dtype=float)
            single = True
        else:
            query = np.asarray(t, dtype=float)
            single = False
        query = np.clip(query, self._times[0], self._times[-1])
        poses = np.zeros((len(query), 6), dtype=float)
        poses[:, :3] = self._pos_interp(query)
        poses[:, 3:] = self._rot_interp(query).as_rotvec()
        if single:
            return poses[0]
        return poses



class ScalarTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, values: np.ndarray):
        times = np.asarray(times, dtype=float).reshape(-1)
        values = np.asarray(values, dtype=float).reshape(-1)
        assert times.shape[0] == values.shape[0]
        self._times = times
        self._values = values

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def values(self) -> np.ndarray:
        return self._values

    def trim(self, start_t: float, end_t: float) -> "ScalarTrajectoryInterpolator":
        start_t = float(start_t)
        end_t = float(end_t)
        assert start_t <= end_t
        times = self._times
        keep = (start_t < times) & (times < end_t)
        keep_times = times[keep]
        new_times = np.concatenate([[start_t], keep_times, [end_t]])
        new_times = np.unique(new_times)
        values = self(new_times)
        return ScalarTrajectoryInterpolator(new_times, values)

    def drive_to_waypoint(self, value: float, time: float, curr_time: float, max_speed: float) -> "ScalarTrajectoryInterpolator":
        value = float(value)
        curr_time = float(curr_time)
        time = max(float(time), curr_time)
        trimmed = self.trim(curr_time, curr_time)
        new_times = np.append(trimmed.times, [time], axis=0)
        new_values = np.append(trimmed.values, [value], axis=0)
        return ScalarTrajectoryInterpolator(new_times, new_values)

    def schedule_waypoint(
        self,
        value: float,
        time: float,
        curr_time: Optional[float],
        last_waypoint_time: Optional[float],
        max_speed: float,
    ) -> "ScalarTrajectoryInterpolator":
        time = float(time)
        start_time = self._times[0]
        end_time = self._times[-1]
        if curr_time is not None:
            curr_time = float(curr_time)
            if time <= curr_time:
                return self
            start_time = max(start_time, curr_time)
            if last_waypoint_time is not None:
                last_waypoint_time = float(last_waypoint_time)
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(curr_time, last_waypoint_time)
            else:
                end_time = curr_time
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        trimmed = self.trim(start_time, end_time)
        end_value = float(trimmed(end_time))
        duration = time - end_time
        if max_speed > 0:
            delta = abs(float(value) - end_value)
            duration = max(duration, delta / max_speed)
        final_time = end_time + max(duration, 0.0)
        new_times = np.append(trimmed.times, [final_time], axis=0)
        new_values = np.append(trimmed.values, [float(value)], axis=0)
        return ScalarTrajectoryInterpolator(new_times, new_values)

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray | float:
        if np.isscalar(t):
            query = float(t)
            if len(self._times) == 1:
                return float(self._values[0])
            return float(np.interp(query, self._times, self._values))
        query = np.asarray(t, dtype=float)
        if len(self._times) == 1:
            return np.full_like(query, float(self._values[0]))
        return np.interp(query, self._times, self._values)
    
class _SimBackend:
    """Minimal drop-in replacement for :class:`RBY1WBC` used for debugging."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sim_frequency: float = 20.0,
        enable_viewer: bool = False,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default_model = project_root / "model" / "rby1" / "rby1.xml"
        default_config = project_root / "config" / "wbc.yaml"
        self.model_path = str(model_path or default_model)
        self.init_config_path = default_config
        try:
            with Path(self.init_config_path).open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception:
            config = {}
        self._init_position = config.get("init_position", {})
        self.trajectory_frequency_hz = sim_frequency
        self._model = mujoco.MjModel.from_xml_path(self.model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._ik = RBY1WholeBodyIK()
        self._state_lock = threading.Lock()
        self._apply_init_position()
        self._snapshot = _SimSnapshot(self._data.qpos.copy())
        self._stop = threading.Event()
        self._viewer_enabled = enable_viewer and (mj_viewer is not None)
        self._viewer_thread: Optional[threading.Thread] = None
        self._viewer: Optional[Any] = None
        self._viewer_data: Optional[mujoco.MjData] = None
        self._gripper_widths = (0.08, 0.08)
        self._apply_init_position()

    def start(self) -> None:
        if not self._viewer_enabled or self._viewer_thread is not None:
            return
        self._stop.clear()
        self._viewer_thread = threading.Thread(target=self._viewer_loop, name="sim-viewer", daemon=True)
        self._viewer_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
            self._viewer_thread = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._viewer = None
        self._viewer_data = None

    def wait_for_first_state(self, timeout_sec: float = 0.0) -> _SimSnapshot:
        return self._snapshot

    def get_latest_robot_state(self) -> _SimSnapshot:
        with self._state_lock:
            return _SimSnapshot(self._data.qpos.copy())

    def snapshot_to_qpos(self, snapshot: _SimSnapshot) -> np.ndarray:
        return snapshot.qpos.copy()

    def get_latest_gripper_widths(self) -> tuple[float, float]:
        return self._gripper_widths

    def update_targets(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        duration: float = 0.1,
        timestamp: Optional[float] = None,
    ) -> None:
        sol_qpos, sol_vel, success, _ = self._ik.solve(
            left_target_pos=left_pos,
            left_target_quat=left_quat,
            right_target_pos=right_pos,
            right_target_quat=right_quat,
            head_target_pos=head_pos,
            head_target_quat=head_quat,
            current_qpos=self._data.qpos.copy(),
            dt=max(duration, 1e-3),
        )
        if not success:
            print("[sim] IK failed to reach target")  # pragma: no cover - debug aid
            return
        with self._state_lock:
            self._data.qpos[:] = sol_qpos
            self._data.qvel[:] = sol_vel
            mujoco.mj_forward(self._model, self._data)
            self._snapshot = _SimSnapshot(sol_qpos.copy())
            lw, rw = self._gripper_widths
            if left_width is not None:
                lw = float(left_width)
            if right_width is not None:
                rw = float(right_width)
            self._gripper_widths = (lw, rw)
    def _apply_init_position(self) -> None:
        init = self._init_position
        if not init:
            return
        torso = np.asarray(init.get("torso", []), dtype=float) if init.get("torso") else None
        left = np.asarray(init.get("left_arm", []), dtype=float) if init.get("left_arm") else None
        right = np.asarray(init.get("right_arm", []), dtype=float) if init.get("right_arm") else None
        head = np.asarray(init.get("head", []), dtype=float) if init.get("head") else None
        if all(item is None for item in (torso, left, right, head)):
            return
        with self._state_lock:
            qpos = self._data.qpos.copy()
            if torso is not None:
                for idx, value in zip(self._ik.torso_qpos_indices, torso):
                    qpos[int(idx)] = float(value)
            if left is not None:
                for idx, value in zip(self._ik.left_arm_qpos_indices, left):
                    qpos[int(idx)] = float(value)
            if right is not None:
                for idx, value in zip(self._ik.right_arm_qpos_indices, right):
                    qpos[int(idx)] = float(value)
            if head is not None:
                for idx, value in zip(self._ik.head_qpos_indices, head):
                    qpos[int(idx)] = float(value)
            self._data.qpos[:] = qpos
            mujoco.mj_forward(self._model, self._data)

    def _viewer_loop(self) -> None:
        if mj_viewer is None:
            return
        try:
            self._viewer_data = mujoco.MjData(self._model)
            self._viewer = mj_viewer.launch_passive(
                model=self._model, data=self._viewer_data, show_left_ui=False, show_right_ui=False
            )
            mujoco.mjv_defaultFreeCamera(self._model, self._viewer.cam)
        except Exception as exc:  # pragma: no cover - optional viewer
            print(f"[sim] Failed to start viewer: {exc}")
            return
        while not self._stop.is_set():
            if self._viewer_data is None or self._viewer is None:
                break
            with self._state_lock:
                self._viewer_data.qpos[:] = self._data.qpos
                self._viewer_data.qvel[:] = self._data.qvel
                mujoco.mj_forward(self._model, self._viewer_data)
            self._viewer.sync()
            time.sleep(1.0 / 60.0)

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
        use_sim: bool = False,
        sim_model_path: Optional[str] = None,
        sim_viewer: bool = False,
        max_pos_speed: float = 0.5,
        max_rot_speed: float = 2.0,
        max_gripper_speed: float = 0.1,
        gripper_min_width: float = 0.0,
        gripper_max_width: float = 0.085,
    ) -> None:
        if use_sim:
            self._backend: Any = _SimBackend(model_path=sim_model_path, enable_viewer=sim_viewer)
        else:
            self._backend = RBY1WBC(config_path=config_path) if config_path else RBY1WBC()
        self._buffer: Deque[RobotObservation] = deque(maxlen=int(max(buffer_size, 1)))
        self._buffer_lock = threading.Lock()
        self._buffer_ready = threading.Event()
        self._stop_event = threading.Event()
        self._buffer_thread: Optional[threading.Thread] = None
        self._command_thread: Optional[threading.Thread] = None
        self._command_stop = threading.Event()
        self._traj_lock = threading.Lock()
        self._pose_traj: Dict[str, Optional[PoseTrajectoryInterpolator]] = {
            "left": None,
            "right": None,
            "head": None,
        }
        self._pose_last_waypoint: Dict[str, Optional[float]] = {
            "left": None,
            "right": None,
            "head": None,
        }
        self._pose_curr_time: Dict[str, Optional[float]] = {
            "left": None,
            "right": None,
            "head": None,
        }
        self._width_traj: Dict[str, Optional[ScalarTrajectoryInterpolator]] = {
            "left": None,
            "right": None,
        }
        self._width_last_waypoint: Dict[str, Optional[float]] = {
            "left": None,
            "right": None,
        }
        self._width_curr_time: Dict[str, Optional[float]] = {
            "left": None,
            "right": None,
        }
        self._max_pos_speed = float(max_pos_speed)
        self._max_rot_speed = float(max_rot_speed)
        self._max_gripper_speed = float(max_gripper_speed)
        self._gripper_limits = (float(gripper_min_width), float(gripper_max_width))

        # Mirror of the MuJoCo model used for FK computations in the streaming
        # thread.  The worker keeps an internal model already, but sharing it
        # across threads is unsafe, so we create our own lightweight copy.
        self._model = mujoco.MjModel.from_xml_path(self._backend.model_path)
        self._data = mujoco.MjData(self._model)
        self._left_site = self._model.site(self.LEFT_SITE).id
        self._right_site = self._model.site(self.RIGHT_SITE).id
        self._head_site = self._model.site(self.HEAD_SITE).id

    @property
    def dt(self) -> float:
        """Return the IK loop period of the underlying controller."""

        return 1.0 / float(self._backend.trajectory_frequency_hz)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the realtime controller threads."""

        self._backend.start()
        if self._buffer_thread is not None and self._buffer_thread.is_alive():
            return
        self._stop_event.clear()
        self._buffer_thread = threading.Thread(
            target=self._buffer_worker,
            name="policy-robot-buffer",
            daemon=True,
        )
        self._buffer_thread.start()
        self._start_command_thread()

    def stop(self) -> None:
        """Stop controller threads and flush buffers."""

        self._stop_event.set()
        if self._buffer_thread is not None:
            self._buffer_thread.join(timeout=1.0)
            self._buffer_thread = None
        self._command_stop.set()
        if self._command_thread is not None:
            self._command_thread.join(timeout=1.0)
            self._command_thread = None
        self._backend.stop()
        with self._buffer_lock:
            self._buffer.clear()
        self._buffer_ready.clear()
        self._command_stop.clear()
        with self._traj_lock:
            for key in self._pose_traj:
                self._pose_traj[key] = None
                self._pose_last_waypoint[key] = None
                self._pose_curr_time[key] = None
            for key in self._width_traj:
                self._width_traj[key] = None
                self._width_last_waypoint[key] = None
                self._width_curr_time[key] = None

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        """Block until the first valid snapshot is available."""

        deadline = time.monotonic() + max(timeout, 0.0)
        self._backend.wait_for_first_state(timeout_sec=timeout)
        remaining = max(0.0, deadline - time.monotonic())
        if not self._buffer_ready.wait(timeout=remaining):
            raise TimeoutError("Timed out waiting for buffered robot observations")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _buffer_worker(self) -> None:
        """Continuously fetch robot snapshots and append them to the buffer."""

        poll_period = max(self.dt, 0.01)
        while not self._stop_event.is_set():
            snapshot = self._backend.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                qpos = self._backend.snapshot_to_qpos(snapshot)
                if qpos is not None:
                    observation = self._qpos_to_observation(qpos, time.monotonic())
                    with self._buffer_lock:
                        self._buffer.append(observation)
                        self._buffer_ready.set()
            time.sleep(poll_period)

    def _start_command_thread(self) -> None:
        if self._command_thread is not None and self._command_thread.is_alive():
            return
        self._command_stop.clear()
        self._command_thread = threading.Thread(
            target=self._command_worker,
            name="policy-robot-command",
            daemon=True,
        )
        self._command_thread.start()

    def _command_worker(self) -> None:
        command_dt = 1.0 / float(self._backend.trajectory_frequency_hz)
        while not self._command_stop.is_set():
            now = time.monotonic()
            payload = self._sample_scheduled_payload(now)
            if payload is not None:
                self.apply_action(payload, duration=command_dt, timestamp=now)
            wait_time = max(0.0, command_dt - (time.monotonic() - now))
            if self._command_stop.wait(timeout=wait_time):
                break

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

        def _site_tf(site_id: int) -> np.ndarray:
            tf = np.eye(4, dtype=float)
            tf[:3, 3] = self._data.site_xpos[site_id].astype(float)
            tf[:3, :3] = self._data.site_xmat[site_id].reshape(3, 3).astype(float)
            return tf

        left_tf = _site_tf(self._left_site)
        right_tf = _site_tf(self._right_site)
        head_tf = _site_tf(self._head_site)

        left_width, right_width = self._backend.get_latest_gripper_widths()

        return RobotObservation(
            timestamp=float(timestamp),
            left_tf=left_tf,
            right_tf=right_tf,
            head_tf=head_tf,
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
        obs_frequency: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Return stacked arrays compatible with the policy server.

        Parameters
        ----------
        horizon:
            Number of historical steps to collect (oldest to newest).
        stride:
            Step between samples inside the internal buffer.  ``stride=1`` means
            consecutive observations, ``stride=2`` picks every other entry, etc.
        obs_frequency:
            If provided, overrides the internal buffer frequency to simulate a
            different observation rate. This only affects the timestamps returned
            and does not interpolate any data. should be combined with stride.
        """

        horizon = int(max(horizon, 1))
        stride = int(max(stride, 1))
        with self._buffer_lock:
            if not self._buffer:
                raise RuntimeError("Robot observation buffer is empty")
            buffer_list = list(self._buffer)
            if obs_frequency is None or len(buffer_list) < 2:
                items: Iterable[RobotObservation] = buffer_list[-horizon * stride :: stride]
                observations = list(items)
            else:
                latest_timestamp = buffer_list[-1].timestamp
                step = float(stride) / max(obs_frequency, 1e-6)
                desired_timestamps = latest_timestamp - step * np.arange(horizon - 1, -1, -1, dtype=float)
                buffer_timestamps = np.array([obs.timestamp for obs in buffer_list], dtype=float)

                def _build_tf_interp(get_tf: Callable[[RobotObservation], np.ndarray]) -> _PoseInterpolator:
                    poses = np.stack([_tf_to_posevec(get_tf(obs)) for obs in buffer_list], axis=0)
                    return _PoseInterpolator(buffer_timestamps, poses)

                left_interp = _build_tf_interp(lambda obs: obs.left_tf)
                right_interp = _build_tf_interp(lambda obs: obs.right_tf)
                head_interp = _build_tf_interp(lambda obs: obs.head_tf)

                def _interp_scalar(attr: str) -> np.ndarray:
                    values = np.stack([getattr(obs, attr) for obs in buffer_list], axis=0)
                    return _build_interp1d(buffer_timestamps, values.reshape(-1, 1))

                left_width_interp = _interp_scalar("left_width")
                right_width_interp = _interp_scalar("right_width")

                left_tf = _posevec_to_tf(left_interp(desired_timestamps))
                right_tf = _posevec_to_tf(right_interp(desired_timestamps))
                head_tf = _posevec_to_tf(head_interp(desired_timestamps))
                left_width = left_width_interp(desired_timestamps)
                right_width = right_width_interp(desired_timestamps)

                return {
                    "timestamp": desired_timestamps,
                    "gripper_left_tf": left_tf,
                    "gripper_right_tf": right_tf,
                    "head_tf": head_tf,
                    "gripper_left_gripper_width": left_width.reshape(-1, 1),
                    "gripper_right_gripper_width": right_width.reshape(-1, 1),
                }

        if len(observations) < horizon:
            observations = [observations[0]] * (horizon - len(observations)) + observations

        def stack_tf(attr: str) -> np.ndarray:
            return np.stack([getattr(obs, attr) for obs in observations], axis=0)

        def stack_scalar(attr: str) -> np.ndarray:
            return np.stack([getattr(obs, attr) for obs in observations], axis=0).reshape(-1, 1)

        timestamps = np.array([obs.timestamp for obs in observations], dtype=np.float64)
        return {
            "timestamp": timestamps,
            "gripper_left_tf": stack_tf("left_tf"),
            "gripper_right_tf": stack_tf("right_tf"),
            "head_tf": stack_tf("head_tf"),
            "gripper_left_gripper_width": stack_scalar("left_width"),
            "gripper_right_gripper_width": stack_scalar("right_width"),
        }

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def _clamp_gripper_width(self, value: float) -> float:
        return float(np.clip(value, self._gripper_limits[0], self._gripper_limits[1]))

    def _sample_scheduled_payload(self, now: float) -> Optional[Dict[str, np.ndarray]]:
        with self._traj_lock:
            payload: Dict[str, np.ndarray] = {}
            for effector, key in (("left", "left_tf"), ("right", "right_tf"), ("head", "head_tf")):
                traj = self._pose_traj.get(effector)
                if traj is None:
                    continue
                pose_vec = traj(now)
                payload[key] = _posevec_to_tf(pose_vec[None])[0]
                self._pose_curr_time[effector] = now
            for width_key, traj in self._width_traj.items():
                if traj is None:
                    continue
                value = float(traj(now))
                if width_key == "left":
                    payload["left_gripper_width"] = value
                else:
                    payload["right_gripper_width"] = value
                self._width_curr_time[width_key] = now
            return payload or None

    def schedule_waypoint(self, payload: Dict[str, np.ndarray], timestamp: float) -> None:
        target_time = float(timestamp)
        now = time.monotonic()
        current_obs: Optional[RobotObservation] = None
        obs_pose_map: Dict[str, np.ndarray] = {}
        obs_width_map: Dict[str, float] = {}

        def _ensure_current_obs() -> bool:
            nonlocal current_obs, obs_pose_map, obs_width_map
            if current_obs is not None:
                return True
            current_obs = self.get_latest_observation()
            if current_obs is None:
                return False
            obs_pose_map = {
                "left": _tf_to_posevec(current_obs.left_tf),
                "right": _tf_to_posevec(current_obs.right_tf),
                "head": _tf_to_posevec(current_obs.head_tf),
            }
            obs_width_map = {
                "left": float(current_obs.left_width),
                "right": float(current_obs.right_width),
            }
            return True

        with self._traj_lock:
            for effector, key in (("left", "left_tf"), ("right", "right_tf"), ("head", "head_tf")):
                tf = payload.get(key)
                if tf is None:
                    continue
                pose_vec = _tf_to_posevec(tf)
                traj = self._pose_traj.get(effector)
                curr_exec_time = self._pose_curr_time.get(effector) or now
                curr_time_eff = max(now, curr_exec_time)
                desired_time = target_time if target_time > curr_time_eff else curr_time_eff + 1e-3
                if traj is None:
                    if not _ensure_current_obs():
                        return
                    traj = PoseTrajectoryInterpolator(
                        times=np.array([curr_time_eff], dtype=float),
                        poses=np.array([obs_pose_map[effector]], dtype=float),
                    )
                    self._pose_last_waypoint[effector] = curr_time_eff
                    self._pose_curr_time[effector] = curr_time_eff
                last_waypoint_time = self._pose_last_waypoint.get(effector)
                traj = traj.schedule_waypoint(
                    pose=pose_vec,
                    time=desired_time,
                    curr_time=curr_time_eff,
                    last_waypoint_time=last_waypoint_time,
                    max_pos_speed=self._max_pos_speed,
                    max_rot_speed=self._max_rot_speed,
                )
                self._pose_traj[effector] = traj
                self._pose_last_waypoint[effector] = float(traj.times[-1])

            for width_key, payload_key in (("left", "left_gripper_width"), ("right", "right_gripper_width")):
                width_val = payload.get(payload_key)
                if width_val is None:
                    continue
                width_target = self._clamp_gripper_width(float(width_val))
                traj = self._width_traj.get(width_key)
                curr_exec_time = self._width_curr_time.get(width_key) or now
                curr_time_eff = max(now, curr_exec_time)
                desired_time = target_time if target_time > curr_time_eff else curr_time_eff + 1e-3
                if traj is None:
                    if not _ensure_current_obs():
                        return
                    traj = ScalarTrajectoryInterpolator(
                        times=np.array([curr_time_eff], dtype=float),
                        values=np.array([obs_width_map[width_key]], dtype=float),
                    )
                    self._width_last_waypoint[width_key] = curr_time_eff
                    self._width_curr_time[width_key] = curr_time_eff
                last_waypoint_time = self._width_last_waypoint.get(width_key)
                traj = traj.schedule_waypoint(
                    value=width_target,
                    time=desired_time,
                    curr_time=curr_time_eff,
                    last_waypoint_time=last_waypoint_time,
                    max_speed=self._max_gripper_speed,
                )
                self._width_traj[width_key] = traj
                self._width_last_waypoint[width_key] = float(traj.times[-1])

    def apply_action(
        self,
        action: Dict[str, np.ndarray],
        duration: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Forward a single Cartesian action to the WBC."""

        for key in ("left_tf", "right_tf"):
            if key not in action:
                raise KeyError(f"Missing required action field '{key}'")

        left_tf = np.asarray(action["left_tf"], dtype=float).reshape(4, 4)
        right_tf = np.asarray(action["right_tf"], dtype=float).reshape(4, 4)
        head_tf = action.get("head_tf")
        head_tf = None if head_tf is None else np.asarray(head_tf, dtype=float).reshape(4, 4)

        def _tf_to_pos_quat(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pos = tf[:3, 3]
            quat_xyzw = Rotation.from_matrix(tf[:3, :3]).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            return pos, quat_wxyz

        left_pos, left_quat = _tf_to_pos_quat(left_tf)
        right_pos, right_quat = _tf_to_pos_quat(right_tf)
        head_pos = None
        head_quat = None
        if head_tf is not None:
            head_pos, head_quat = _tf_to_pos_quat(head_tf)

        left_width = float(action["left_gripper_width"]) if "left_gripper_width" in action else None
        right_width = float(action["right_gripper_width"]) if "right_gripper_width" in action else None

        self._backend.update_targets(
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=None if head_pos is None else head_pos,
            head_quat=head_quat,
            duration=self.dt if duration is None else float(duration),
            timestamp=timestamp,
        )


__all__ = ["RBY1PolicyRobot", "RobotObservation"]
