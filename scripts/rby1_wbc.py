"""Whole-body controller thread decoupled from any GUI interaction."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Optional, Tuple

import mujoco
import numpy as np

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ik.rby1_whole_body_ik import RBY1WholeBodyIK

from rby1.control import (
    Config as ControllerConfig,
    RealtimeDriver,
    RobotSnapshot,
)

# TRI's IK runs at 500 hz and ours at 100 hz, so scale the gains by 5x
BASE_ERROR_GAIN = np.array([0.2, 0.2, 0.2], dtype=float)
# Might need to tune this more
BASE_VELOCITY_GAIN = np.array([0.09, 0.09, 0.5], dtype=float)

# Init Position
INIT_POSITION={ 
    "torso": np.array([0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0]), 
    "left_arm": np.array([0.0, 0.0873, 0.0, -2.0944, 0.0, 0.9599, -1.5708]),
    "right_arm": np.array([0.0, -0.0873, 0.0, -2.0944, 0.0, 0.9599, 1.5708]),
    "head": np.array([0.0, 0.6109]), 
    "grippers": np.array([0.1, 0.1])
}


class RobotStateBuffer:
    """Stores the latest robot snapshot retrieved from the controller."""
    def __init__(self):
        self._lock = threading.Lock()
        self.latest: Optional[RobotSnapshot] = None

    def store(self, snapshot: RobotSnapshot) -> None:
        with self._lock:
            self.latest = snapshot

    def load(self) -> Optional[RobotSnapshot]:
        with self._lock:
            return self.latest


@dataclass
class SharedTargets:
    """Thread-safe shared targets and current qpos snapshot for IK."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    left_gripper_pos: Optional[np.ndarray] = None
    left_gripper_quat: Optional[np.ndarray] = None
    left_gripper_width: Optional[float] = None

    right_gripper_pos: Optional[np.ndarray] = None
    right_gripper_quat: Optional[np.ndarray] = None
    right_gripper_width: Optional[float] = None
    
    head_target_pos: Optional[np.ndarray] = None
    head_target_quat: Optional[np.ndarray] = None

    def set_targets(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
    ) -> None:
        with self.lock:
            self.left_gripper_pos = left_pos.copy()
            self.left_gripper_quat = left_quat.copy()
            self.right_gripper_pos = right_pos.copy()
            self.right_gripper_quat = right_quat.copy()
            self.left_gripper_width = None if left_width is None else float(left_width)
            self.right_gripper_width = None if right_width is None else float(right_width)
            self.head_target_pos = None if head_pos is None else head_pos.copy()
            self.head_target_quat = None if head_quat is None else head_quat.copy()

    def get_for_ik(
        self,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        with self.lock:
            lt_p = None if self.left_gripper_pos is None else self.left_gripper_pos.copy()
            lt_q = None if self.left_gripper_quat is None else self.left_gripper_quat.copy()
            lw = self.left_gripper_width

            rt_p = None if self.right_gripper_pos is None else self.right_gripper_pos.copy()
            rt_q = None if self.right_gripper_quat is None else self.right_gripper_quat.copy()
            rw = self.right_gripper_width

            hp = None if self.head_target_pos is None else self.head_target_pos.copy()
            hq = None if self.head_target_quat is None else self.head_target_quat.copy()
        return lt_p, lt_q, lw, rt_p, rt_q, rw, hp, hq


class RBY1WBC:
    """Worker that streams IK solutions to the realtime controller."""

    def __init__(
        self,
        model_path: str,
        address: str = "localhost:50051",
        ik_frequency_hz: float = 100.0,
        state_frequency_hz: float = 200.0,
    ):
        self.model_path = model_path
        self.address = address
        self.ik_rate = RateLimiter(frequency=ik_frequency_hz, warn=False)
        self.state_poll_rate = RateLimiter(frequency=state_frequency_hz, warn=False)

        self.shared_targets = SharedTargets()
        self.robot_state = RobotStateBuffer()

        self._stop = threading.Event()
        self._threads_started = False
        self._model_lock = threading.Lock()

        self.controller = self._init_controller(address)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.ik_solver = RBY1WholeBodyIK()
        self.prev_qpos = self.data.qpos.copy()

        self._build_joint_mapping()
        self._extract_base_origin()

        self._state_thread: Optional[threading.Thread] = None
        self._ik_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._threads_started:
            return
        self._stop.clear()
        self._state_thread = threading.Thread(
            target=self._state_poll_loop, name="state-poll", daemon=True
        )
        self._ik_thread = threading.Thread(
            target=self._ik_loop, name="wbc", daemon=True
        )
        self._state_thread.start()
        self._ik_thread.start()
        self._threads_started = True
        self._set_init_position()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._ik_thread is not None:
            self._ik_thread.join(timeout=join_timeout)
        if self._state_thread is not None:
            self._state_thread.join(timeout=join_timeout)
        self.controller.stop()
        self._threads_started = False

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
    ) -> None:
        self.shared_targets.set_targets(
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            left_width=left_width,
            right_width=right_width,
            head_pos=head_pos,
            head_quat=head_quat,
        )

    def get_latest_robot_state(self) -> Optional[RobotSnapshot]:
        return self.robot_state.load()

    def wait_for_first_state(self, timeout_sec: float = 5.0) -> Optional[RobotSnapshot]:
        deadline = time.monotonic() + timeout_sec
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = self.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                return snapshot
            time.sleep(0.005)
        raise Exception("Timeout waiting for first valid robot state snapshot")

    def snapshot_to_qpos(self, snapshot: RobotSnapshot) -> Optional[np.ndarray]:
        if snapshot is None or not snapshot.is_valid:
            return None
        with self._model_lock:
            return self._snapshot_to_qpos(snapshot)

    def _init_controller(self, address: str) -> RealtimeDriver:
        config = ControllerConfig()
        config.robot_address = address
        controller = RealtimeDriver(config)
        controller.start()
        if not controller.wait_until_ready(timeout_sec=15.0):
            controller.stop()
            raise RuntimeError("Controller did not report ready within 15 seconds")
        print("Realtime controller ready.")
        return controller

    def _set_init_position(self) -> None:
        print("Setting initial positions...")
        sol_qpos = self.data.qpos.copy()

        torso_indices = getattr(self.ik_solver, "torso_qpos_indices", [])
        right_indices = getattr(self.ik_solver, "right_arm_qpos_indices", [])
        left_indices = getattr(self.ik_solver, "left_arm_qpos_indices", [])
        head_indices = getattr(self.ik_solver, "head_qpos_indices", [])

        for idx, value in zip(torso_indices, INIT_POSITION["torso"]):
            sol_qpos[int(idx)] = value
        for idx, value in zip(right_indices, INIT_POSITION["right_arm"]):
            sol_qpos[int(idx)] = value
        for idx, value in zip(left_indices, INIT_POSITION["left_arm"]):
            sol_qpos[int(idx)] = value
        for idx, value in zip(head_indices, INIT_POSITION["head"]):
            sol_qpos[int(idx)] = value

        target_body = self._compute_body_commands(sol_qpos)
        snapshot = self.wait_for_first_state()
        with self._model_lock:
            start_qpos = self._snapshot_to_qpos(snapshot)
        start_body = self._compute_body_commands(start_qpos)

        delta = target_body - start_body
        max_delta = float(np.max(np.abs(delta)))
        if max_delta < 1e-6:
            self.controller.set_body_position_targets(target_body.tolist())
        else:
            steps = min(200, max(5, int(np.ceil(max_delta / 0.05))))
            for step in range(1, steps + 1):
                alpha = step / steps
                cmd = start_body + alpha * delta
                self.controller.set_body_position_targets(cmd.tolist())
                time.sleep(0.02)
            self.controller.set_body_position_targets(target_body.tolist())

        self.prev_qpos = sol_qpos.copy()
        print("Initial positions set.")

    def _state_poll_loop(self) -> None:
        while not self._stop.is_set():
            snapshot = self.controller.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                self.robot_state.store(snapshot)
            self.state_poll_rate.sleep()

    def _ik_loop(self) -> None:
        while not self._stop.is_set():
            snapshot = self.robot_state.load()
            current_qpos: Optional[np.ndarray] = self.snapshot_to_qpos(snapshot)
            left_pos, left_quat, left_width, right_pos, right_quat, right_width, head_pos, head_quat = self.shared_targets.get_for_ik()

            if current_qpos is None or left_pos is None or right_pos is None:
                self.ik_rate.sleep()
                continue

            # TODO: Include head target in IK
            sol_qpos, sol_vel, success, _info = self.ik_solver.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                current_qpos=current_qpos,
                dt=self.ik_rate.dt,
            )
            if not success:
                print(f"[wbc] IK failed: {_info}")

            try:
                # TODO: Add gripper commands
                body_targets = self._compute_body_commands(sol_qpos)
                twist = self._compute_base_twist_command(sol_qpos, sol_vel, current_qpos)
                self.controller.set_body_position_targets(body_targets.tolist())
                self.controller.set_base_twist_command(twist)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[wbc] command error: {exc}")

            self.ik_rate.sleep()

    def _compute_body_commands(self, sol_qpos: np.ndarray) -> np.ndarray:
        positions_sdk = np.zeros(len(self._sdk_joint_names), dtype=float)
        for i, adr in enumerate(self._sdk_to_mj_qadr):
            if adr is not None:
                positions_sdk[i] = sol_qpos[adr]

        name_to_idx = {name: i for i, name in enumerate(self._sdk_joint_names)}
        torso_idxs = [name_to_idx[f"torso_{i}"] for i in range(6)]
        left_idxs = [name_to_idx[f"left_arm_{i}"] for i in range(7)]
        right_idxs = [name_to_idx[f"right_arm_{i}"] for i in range(7)]
        head_idxs = [name_to_idx[f"head_{i}"] for i in range(2)]
        ordered = torso_idxs + left_idxs + right_idxs + head_idxs
        body_targets = positions_sdk[ordered]
        return body_targets

    def _compute_base_twist_command(self, sol_qpos: np.ndarray, sol_qvel: np.ndarray, cur_qpos: np.ndarray) -> np.ndarray:
        measured_x = float(cur_qpos[0])
        measured_y = float(cur_qpos[1])
        measured_yaw = self._yaw_from_quat(cur_qpos[3:7])

        desired_x = float(sol_qpos[0])
        desired_y = float(sol_qpos[1])
        desired_yaw = self._yaw_from_quat(sol_qpos[3:7])

        error = np.array(
            [
                desired_x - measured_x,
                desired_y - measured_y,
                self._angle_difference(desired_yaw, measured_yaw),
            ],
            dtype=float,
        )

        feedback = BASE_ERROR_GAIN * error

        velocity_desired_world = np.array(
            [float(sol_qvel[0]), float(sol_qvel[1]), float(sol_qvel[5])],
            dtype=float,
        )
        velocity_command_world = feedback + BASE_VELOCITY_GAIN * velocity_desired_world

        cy = math.cos(measured_yaw)
        sy = math.sin(measured_yaw)
        vx_world = velocity_command_world[0]
        vy_world = velocity_command_world[1]
        vx_body = cy * vx_world + sy * vy_world
        vy_body = -sy * vx_world + cy * vy_world

        return np.array([vx_body, vy_body, velocity_command_world[2]], dtype=float)

    def _snapshot_to_qpos(self, snapshot: RobotSnapshot) -> np.ndarray:
        if hasattr(self, "prev_qpos") and self.prev_qpos is not None:
            qpos = self.prev_qpos.copy()
        else:
            qpos = np.zeros(self.model.nq, dtype=float)

        base_adr = getattr(self, "_base_free_adr", None)
        base_origin = getattr(self, "_base_origin_pos", np.zeros(3, dtype=float))
        if base_adr is not None:
            T = snapshot.odom_SE2
            x = float(T[0, 2])
            y = float(T[1, 2])
            yaw = math.atan2(T[1, 0], T[0, 0])
            qpos[base_adr + 0] = base_origin[0] + x
            qpos[base_adr + 1] = base_origin[1] + y
            qpos[base_adr + 2] = base_origin[2]
            qpos[base_adr + 3 : base_adr + 7] = self._quat_from_yaw(yaw)

        joint_positions = snapshot.joint_position
        mapping = getattr(self, "_sdk_to_mj_qadr", [])
        count = min(len(joint_positions), len(mapping))
        for idx in range(count):
            adr = mapping[idx]
            if adr is not None:
                qpos[adr] = joint_positions[idx]

        self.prev_qpos = qpos.copy()
        return qpos

    def _build_joint_mapping(self) -> None:
        self._sdk_joint_names = [
            "wheel_fr",
            "wheel_fl",
            "wheel_rr",
            "wheel_rl",
            "torso_0",
            "torso_1",
            "torso_2",
            "torso_3",
            "torso_4",
            "torso_5",
            "right_arm_0",
            "right_arm_1",
            "right_arm_2",
            "right_arm_3",
            "right_arm_4",
            "right_arm_5",
            "right_arm_6",
            "left_arm_0",
            "left_arm_1",
            "left_arm_2",
            "left_arm_3",
            "left_arm_4",
            "left_arm_5",
            "left_arm_6",
            "head_0",
            "head_1",
        ]

        self._mj_joint_qadr = {}
        self._base_free_adr = None
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = int(self.model.jnt_type[i])
            adr = int(self.model.jnt_qposadr[i])
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                if name == "world_j":
                    self._base_free_adr = adr
                continue
            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                self._mj_joint_qadr[name] = adr

        if self._base_free_adr is None:
            raise RuntimeError("Base free joint address not found (world_j)")

        self._sdk_to_mj_qadr = []
        missing = []
        for name in self._sdk_joint_names:
            adr = self._mj_joint_qadr.get(name)
            if adr is None:
                missing.append(name)
            self._sdk_to_mj_qadr.append(adr)
        if missing:
            print(f"[mapping] Missing joints in MuJoCo model: {missing}")

    def _extract_base_origin(self) -> None:
        self._base_origin_pos = self.data.qpos[
            self._base_free_adr : self._base_free_adr + 3
        ].copy()

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        half = 0.5 * yaw
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)

    @staticmethod
    def _yaw_from_quat(q: np.ndarray) -> float:
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @staticmethod
    def _angle_difference(target: float, source: float) -> float:
        diff = float(target) - float(source)
        return (diff + math.pi) % (2 * math.pi) - math.pi


__all__ = ["RBY1WBC", "SharedTargets"]
