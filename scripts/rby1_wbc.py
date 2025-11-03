"""Whole-body controller thread decoupled from any GUI interaction."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Mapping, Optional, Tuple, Union

import mujoco
import numpy as np
import yaml

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ik.rby1_whole_body_ik import RBY1WholeBodyIK

from rby1.control import (
    Config as ControllerConfig,
    RealtimeDriver,
    RobotSnapshot,
)

from gripper.gripper import Gripper

# TRI's IK runs at 500 hz and ours at 100 hz, so scale the gains by 5x
BASE_ERROR_GAIN = np.array([0.2, 0.2, 0.2], dtype=float)
# Might need to tune this more
BASE_VELOCITY_GAIN = np.array([0.09, 0.09, 0.1], dtype=float)

class RobotStateBuffer:
    """Stores the latest robot snapshot retrieved from the controller."""
    def __init__(self):
        self._lock = threading.Lock()
        self.latest: Optional[RobotSnapshot] = None
        self._gripper_widths: Tuple[float, float] = (0.0, 0.0)

    def store(self, snapshot: RobotSnapshot) -> None:
        with self._lock:
            self.latest = snapshot

    def load(self) -> Optional[RobotSnapshot]:
        with self._lock:
            return self.latest

    def store_gripper_widths(
        self,
        left_width: Optional[float],
        right_width: Optional[float],
    ) -> None:
        with self._lock:
            current_left, current_right = self._gripper_widths
            if left_width is not None:
                current_left = float(left_width)
            if right_width is not None:
                current_right = float(right_width)
            self._gripper_widths = (current_left, current_right)

    def load_gripper_widths(self) -> Tuple[float, float]:
        with self._lock:
            return self._gripper_widths


@dataclass
class SharedTargets:
    """Thread-safe shared targets and current qpos snapshot for IK."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    duration: float = 0.1 # 10Hz
    target_set_timestamp: float = 0.0

    left_gripper_pos_start: Optional[np.ndarray] = None
    left_gripper_quat_start: Optional[np.ndarray] = None
    left_gripper_width_start: Optional[float] = None
    left_gripper_pos: Optional[np.ndarray] = None
    left_gripper_quat: Optional[np.ndarray] = None
    left_gripper_width: Optional[float] = None

    right_gripper_pos_start: Optional[np.ndarray] = None
    right_gripper_quat_start: Optional[np.ndarray] = None
    right_gripper_width_start: Optional[float] = None
    right_gripper_pos: Optional[np.ndarray] = None
    right_gripper_quat: Optional[np.ndarray] = None
    right_gripper_width: Optional[float] = None

    head_target_pos_start: Optional[np.ndarray] = None
    head_target_quat_start: Optional[np.ndarray] = None
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
        duration: float = 0.1,
        timestamp: Optional[float] = None,
    ) -> None:
        with self.lock:
            now = time.monotonic() if timestamp is None else float(timestamp)

            # Store previous targets for interpolation
            prev_left_pos = self.left_gripper_pos.copy() if self.left_gripper_pos is not None else None
            prev_left_quat = self.left_gripper_quat.copy() if self.left_gripper_quat is not None else None
            prev_left_width = self.left_gripper_width

            prev_right_pos = self.right_gripper_pos.copy() if self.right_gripper_pos is not None else None
            prev_right_quat = self.right_gripper_quat.copy() if self.right_gripper_quat is not None else None
            prev_right_width = self.right_gripper_width

            prev_head_pos = self.head_target_pos.copy() if self.head_target_pos is not None else None
            prev_head_quat = self.head_target_quat.copy() if self.head_target_quat is not None else None

            self.left_gripper_pos_start = prev_left_pos if prev_left_pos is not None else left_pos.copy()
            self.left_gripper_quat_start = prev_left_quat if prev_left_quat is not None else left_quat.copy()
            self.left_gripper_width_start = prev_left_width if prev_left_width is not None else (None if left_width is None else float(left_width))

            self.right_gripper_pos_start = prev_right_pos if prev_right_pos is not None else right_pos.copy()
            self.right_gripper_quat_start = prev_right_quat if prev_right_quat is not None else right_quat.copy()
            self.right_gripper_width_start = prev_right_width if prev_right_width is not None else (None if right_width is None else float(right_width))

            self.head_target_pos_start = prev_head_pos if prev_head_pos is not None else (None if head_pos is None else head_pos.copy())
            self.head_target_quat_start = prev_head_quat if prev_head_quat is not None else (None if head_quat is None else head_quat.copy())

            # Set new targets
            self.left_gripper_pos = left_pos.copy()
            self.left_gripper_quat = left_quat.copy()
            self.right_gripper_pos = right_pos.copy()
            self.right_gripper_quat = right_quat.copy()
            self.left_gripper_width = None if left_width is None else float(left_width)
            self.right_gripper_width = None if right_width is None else float(right_width)
            self.head_target_pos = None if head_pos is None else head_pos.copy()
            self.head_target_quat = None if head_quat is None else head_quat.copy()

            self.duration = max(0.0, float(duration))
            self.target_set_timestamp = now

    def get_for_ik(self, use_interpolation: bool = False) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if not use_interpolation:
            return self.get_target()
        
        # Return the linearly interpolated targets based on elapsed time since setting
        current_time = time.monotonic()
        with self.lock:
            duration = max(self.duration, 0.0)
            elapsed = max(0.0, current_time - self.target_set_timestamp)
            alpha = min(1.0, elapsed / duration)

            lt_p = _lerp_value(self.left_gripper_pos_start, self.left_gripper_pos, alpha)
            lt_q = _slerp_quaternion(self.left_gripper_quat_start, self.left_gripper_quat, alpha)
            lw = _lerp_value(self.left_gripper_width_start, self.left_gripper_width, alpha)

            rt_p = _lerp_value(self.right_gripper_pos_start, self.right_gripper_pos, alpha)
            rt_q = _slerp_quaternion(self.right_gripper_quat_start, self.right_gripper_quat, alpha)
            rw = _lerp_value(self.right_gripper_width_start, self.right_gripper_width, alpha)

            hp = _lerp_value(self.head_target_pos_start, self.head_target_pos, alpha)
            hq = _slerp_quaternion(self.head_target_quat_start, self.head_target_quat, alpha)

        return lt_p, lt_q, lw, rt_p, rt_q, rw, hp, hq

    def get_target(self) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
        Optional[np.ndarray],
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


def _lerp_value(
    start: Optional[Union[np.ndarray, float]],
    end: Optional[Union[np.ndarray, float]],
    alpha: float,
):
    if end is None:
        return None
    if start is None:
        if isinstance(end, np.ndarray):
            return end.copy()
        return float(end)
    alpha_clamped = max(0.0, min(1.0, alpha))
    if isinstance(end, np.ndarray):
        return (1.0 - alpha_clamped) * start + alpha_clamped * end
    return float((1.0 - alpha_clamped) * start + alpha_clamped * end)


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _slerp_quaternion(
    start: Optional[np.ndarray],
    end: Optional[np.ndarray],
    alpha: float,
) -> Optional[np.ndarray]:
    if end is None:
        return None
    if start is None:
        return end.copy()

    start_norm = _normalize_quaternion(start)
    end_norm = _normalize_quaternion(end)

    dot = float(np.dot(start_norm, end_norm))
    if dot < 0.0:
        end_norm = -end_norm
        dot = -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        result = start_norm + alpha * (end_norm - start_norm)
        return _normalize_quaternion(result)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    if sin_theta_0 < 1e-6:
        return end_norm.copy()

    alpha_clamped = max(0.0, min(1.0, alpha))
    theta = theta_0 * alpha_clamped
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * start_norm + s1 * end_norm
    return _normalize_quaternion(result)


class RBY1WBC:
    """Worker that streams IK solutions to the realtime controller."""

    def __init__(self, config_path: str = PROJECT_ROOT + "/config/wbc.yaml"):
        # Load Controller Config
        try:
            config_path = Path(config_path)
            with config_path.open("r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) 
        except Exception as e:
            raise Exception(f"Exception while loading config file: {e}")
        
        self.address = self.config["address"]
        self.model_path = PROJECT_ROOT + self.config["model_path"]
        self.init_position = self.config["init_position"]
        self.state_frequency_hz = self.config["state_frequency_hz"]
        self.trajectory_frequency_hz = self.config["trajectory_frequency_hz"]
        self.ik_frequency_hz = self.config["ik_frequency_hz"]
        self.use_interpolation = self.config["use_interpolation"]
        self.command_timeout_sec = self.config["command_timeout_sec"]

        # Initialize Controller loops
        self.ik_rate = RateLimiter(frequency=self.ik_frequency_hz, warn=False)
        self.state_poll_rate = RateLimiter(frequency=self.state_frequency_hz, warn=False)

        self.shared_targets = SharedTargets()
        self.robot_state = RobotStateBuffer()

        self._stop = threading.Event()
        self._threads_started = False
        self._model_lock = threading.Lock()

        self.controller = self._init_controller(self.address, self.command_timeout_sec)

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.ik_solver = RBY1WholeBodyIK()
        self._build_joint_mapping()
        self._extract_base_origin()

        # Initialize Gripper
        self.gripper = Gripper()
        if self.gripper.initialize():
            self.gripper.start()
            print("Successfully initialized gripper")
        else:
            self.gripper = None
            print("Failed to initialize gripper")
        
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
        duration: float = 0.1,
        timestamp: Optional[float] = None,
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
            duration=duration,
            timestamp=timestamp,
        )

    def get_latest_robot_state(self) -> Optional[RobotSnapshot]:
        return self.robot_state.load()

    def get_latest_gripper_widths(self) -> Tuple[float, float]:
        return self.robot_state.load_gripper_widths()

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

    def _init_controller(
        self, address: str, command_timeout_sec: float
    ) -> RealtimeDriver:
        config = ControllerConfig()
        config.robot_address = address
        config.command_timeout_us = int(
            round(max(command_timeout_sec, 0.0) * 1_000_000.0)
        )
        controller = RealtimeDriver(config)
        controller.start()
        if not controller.wait_until_ready(timeout_sec=15.0):
            controller.stop()
            raise RuntimeError("Controller did not report ready within 15 seconds")
        print("Realtime controller ready.")
        return controller

    def _set_init_position(self) -> None:
        def _to_array(values: Optional[list[float]]) -> Optional[np.ndarray]:
            if values is None:
                return None
            return np.asarray(values, dtype=float)

        torso_targets = _to_array(self.init_position.get("torso"))
        right_targets = _to_array(self.init_position.get("right_arm"))
        left_targets = _to_array(self.init_position.get("left_arm"))
        head_targets = _to_array(self.init_position.get("head"))
        gripper_targets = _to_array(self.init_position.get("grippers"))

        if all(target is None for target in (torso_targets, right_targets, left_targets, head_targets, gripper_targets)):
            print(f"Init config at {self.init_config_path} did not contain any targets; skipping initial position command.")
            return

        print("Setting initial positions...")
        sol_qpos = self.data.qpos.copy()

        torso_indices = getattr(self.ik_solver, "torso_qpos_indices", [])
        right_indices = getattr(self.ik_solver, "right_arm_qpos_indices", [])
        left_indices = getattr(self.ik_solver, "left_arm_qpos_indices", [])
        head_indices = getattr(self.ik_solver, "head_qpos_indices", [])

        if torso_targets is not None:
            for idx, value in zip(torso_indices, torso_targets):
                sol_qpos[int(idx)] = float(value)
        if right_targets is not None:
            for idx, value in zip(right_indices, right_targets):
                sol_qpos[int(idx)] = float(value)
        if left_targets is not None:
            for idx, value in zip(left_indices, left_targets):
                sol_qpos[int(idx)] = float(value)
        if head_targets is not None:
            for idx, value in zip(head_indices, head_targets):
                sol_qpos[int(idx)] = float(value)

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
            INIT_POSITION_MAX_STEP_DELTA = 0.02
            steps = max(10, int(np.ceil(max_delta / INIT_POSITION_MAX_STEP_DELTA)))
            for step in range(1, steps + 1):
                alpha = step / steps
                cmd = start_body + alpha * delta
                self.controller.set_body_position_targets(cmd.tolist())
                self.ik_rate.sleep()
            self.controller.set_body_position_targets(target_body.tolist())

        if self.gripper and gripper_targets is not None:
            self.gripper.set_target(gripper_targets.tolist())
        self.robot_state.store_gripper_widths(float(gripper_targets[0]),  float(gripper_targets[1]))
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
            now = time.monotonic()
            left_pos, left_quat, left_width, right_pos, right_quat, right_width, head_pos, head_quat = self.shared_targets.get_for_ik(use_interpolation=self.use_interpolation)

            if current_qpos is None or left_pos is None or right_pos is None:
                self.ik_rate.sleep()
                continue

            sol_qpos, sol_vel, success, _info = self.ik_solver.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                head_target_pos=head_pos,
                head_target_quat=head_quat,
                current_qpos=current_qpos,
                dt=self.ik_rate.dt,
            )
            if not success:
                print(f"[wbc] IK failed: {_info}")

            try:
                if self.gripper and left_width is not None and right_width is not None:
                    self.gripper.set_target([right_width, left_width])
                self.robot_state.store_gripper_widths(left_width, right_width)
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

        # Gripper joints handling
        left_width, right_width = self.robot_state.load_gripper_widths()
        mujoco_mapping = getattr(self, "_mj_joint_qadr", {})
        l1_index, l2_index = mujoco_mapping["gripper_finger_l1"], mujoco_mapping["gripper_finger_l2"]
        r1_index, r2_index = mujoco_mapping["gripper_finger_r1"], mujoco_mapping["gripper_finger_r2"]
        qpos[l2_index], qpos[r2_index] = left_width/2, right_width/2
        qpos[l1_index], qpos[r1_index] = -left_width/2, -right_width/2
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
            elif jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
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
