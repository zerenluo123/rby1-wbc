"""Interactive GUI that streams IK solutions to the C++ realtime driver.

This script mirrors the behaviour of ``rby1_joint_wbc_gui.py`` but routes all
low-level control through the pybind-wrapped realtime controller residing in
``rby1/control``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ik.rby1_whole_body_ik import RBY1WholeBodyIK

from rby1.control import (
    Config as ControllerConfig,
    RealtimeDriver,
    RobotSnapshot,
)


class RobotStateBuffer:
    """Stores the latest robot snapshot retrieved from the controller."""

    def __init__(self):
        self._queue: SimpleQueue[RobotSnapshot] = SimpleQueue()
        self.latest: Optional[RobotSnapshot] = None

    def store(self, snapshot: RobotSnapshot) -> None:
        self._queue.put_nowait(snapshot)
        if self._queue.qsize() > 1:
            self._queue.get_nowait()

    def load(self) -> Optional[RobotSnapshot]:
        if not self._queue.empty():
            self.latest = self._queue.get_nowait()
        return self.latest


@dataclass
class SharedTargets:
    """Thread-safe shared targets and current qpos snapshot for IK."""

    lock: threading.Lock
    left_target_pos: Optional[np.ndarray] = None
    left_target_quat: Optional[np.ndarray] = None
    right_target_pos: Optional[np.ndarray] = None
    right_target_quat: Optional[np.ndarray] = None
    current_qpos: Optional[np.ndarray] = None

    def set_from_viewer(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        qpos: np.ndarray,
    ) -> None:
        with self.lock:
            self.left_target_pos = left_pos.copy()
            self.left_target_quat = left_quat.copy()
            self.right_target_pos = right_pos.copy()
            self.right_target_quat = right_quat.copy()
            self.current_qpos = qpos.copy()

    def get_for_ik(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            lt_p = None if self.left_target_pos is None else self.left_target_pos.copy()
            lt_q = None if self.left_target_quat is None else self.left_target_quat.copy()
            rt_p = None if self.right_target_pos is None else self.right_target_pos.copy()
            rt_q = None if self.right_target_quat is None else self.right_target_quat.copy()
            q = None if self.current_qpos is None else self.current_qpos.copy()
        return lt_p, lt_q, rt_p, rt_q, q


class PybindSimGUI:
    def __init__(self, model_path: str, address: str = "localhost:50051", headless: bool = False):
        self.headless = headless
        self.controller = self._init_controller(address)

        # Robot state buffer populated from the controller thread.
        self.robot_state = RobotStateBuffer()
        self.state_poll_rate = RateLimiter(frequency=200.0, warn=False)

        # Provide SDK joint names up-front so headless mode can run without MuJoCo
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

        self.viewer = None if self.headless else self._init_viewer(model_path)
        self.ik_rate = RateLimiter(frequency=200.0, warn=False)
        self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        self.shared = SharedTargets(lock=threading.Lock())
        self._stop = threading.Event()
        self._controller_thread: Optional[threading.Thread] = None

        # Kick off a polling thread that fetches the latest controller snapshot.
        self._state_thread = threading.Thread(
            target=self._state_poll_loop, name="state-poll", daemon=True
        )
        self._state_thread.start()

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

    def _state_poll_loop(self) -> None:
        while not self._stop.is_set():
            snapshot = self.controller.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                self.robot_state.store(snapshot)
            self.state_poll_rate.sleep()

    def _init_viewer(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ik_solver = RBY1WholeBodyIK()

        mujoco.mj_forward(self.model, self.data)
        self.current_qpos = self.data.qpos.copy()
        self.prev_qpos = self.current_qpos.copy()

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            start = self.model.jnt_qposadr[i]
            jtype = self.model.jnt_type[i]
            dim = 7 if jtype == mujoco.mjtJoint.mjJNT_FREE else (
                4 if jtype == mujoco.mjtJoint.mjJNT_BALL else 1
            )
            print(f"{name:20s} : qpos[{start}:{start + dim}]")

        self.ee_l_mid = self.model.body("ee_l_target").mocapid[0]
        self.ee_r_mid = self.model.body("ee_r_target").mocapid[0]

        self._build_joint_mapping()
        self._extract_base_origin()

        snapshot = None
        for _ in range(300):  # wait ~1.5s for the first snapshot
            snapshot = self.controller.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                break
            time.sleep(0.005)
        if snapshot is None or not snapshot.is_valid:
            raise RuntimeError("Failed to receive initial robot state snapshot")

        self._apply_snapshot_to_mujoco(snapshot)

        left_nominal = self.site_pos("end_effector_l", self.data.qpos)
        right_nominal = self.site_pos("end_effector_r", self.data.qpos)
        self.data.mocap_pos[self.ee_l_mid] = left_nominal
        self.data.mocap_pos[self.ee_r_mid] = right_nominal
        l_q = self.data.xquat[self.model.body("EE_BODY_L").id].copy()
        r_q = self.data.xquat[self.model.body("EE_BODY_R").id].copy()
        self.data.mocap_quat[self.ee_l_mid] = l_q
        self.data.mocap_quat[self.ee_r_mid] = r_q

        self._se2_prev = np.zeros(3, dtype=float)

        viewer = mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        )
        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
        return viewer

    def _apply_snapshot_to_mujoco(self, snapshot: RobotSnapshot) -> None:
        T = snapshot.odom_SE2
        x = float(T[0, 2])
        y = float(T[1, 2])
        yaw = math.atan2(T[1, 0], T[0, 0])
        self.data.qpos[self._base_free_adr + 0] = self._base_origin_pos[0] + x
        self.data.qpos[self._base_free_adr + 1] = self._base_origin_pos[1] + y
        self.data.qpos[self._base_free_adr + 2] = self._base_origin_pos[2]
        self.data.qpos[self._base_free_adr + 3 : self._base_free_adr + 7] = self._quat_from_yaw(yaw)

        if len(snapshot.joint_position) == len(self._sdk_joint_names):
            for sdk_idx, adr in enumerate(self._sdk_to_mj_qadr):
                if adr is not None:
                    self.data.qpos[adr] = snapshot.joint_position[sdk_idx]

        mujoco.mj_forward(self.model, self.data)
        self.prev_qpos = self.data.qpos.copy()

    def _snapshot_to_qpos(self, snapshot: RobotSnapshot) -> np.ndarray:
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("MuJoCo model not initialized; cannot map snapshot to qpos.")

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

        return qpos

    def site_pos(self, site_name: str, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()

    def controller_loop(self) -> None:
        # Body indices exclude the four wheel joints; derived from SDK joint order
        body_idx = [i for i, name in enumerate(self._sdk_joint_names) if not name.startswith("wheel_")]

        while not self._stop.is_set():
            snapshot = self.robot_state.load()
            qpos_from_snapshot: Optional[np.ndarray] = None
            if (
                snapshot is not None
                and snapshot.is_valid
                and hasattr(self, "model")
                and self.model is not None
            ):
                try:
                    qpos_from_snapshot = self._snapshot_to_qpos(snapshot)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[controller] snapshot conversion error: {exc}")

            left_pos, left_quat, right_pos, right_quat, qpos_viewer = self.shared.get_for_ik()
            qpos_for_ik = qpos_from_snapshot if qpos_from_snapshot is not None else qpos_viewer

            if isinstance(qpos_for_ik, RobotSnapshot):
                try:
                    qpos_for_ik = self._snapshot_to_qpos(qpos_for_ik)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[controller] snapshot-to-qpos fallback error: {exc}")
                    qpos_for_ik = None

            if qpos_for_ik is not None and not isinstance(qpos_for_ik, np.ndarray):
                qpos_for_ik = np.asarray(qpos_for_ik, dtype=float)

            if qpos_for_ik is None or left_pos is None or right_pos is None:
                self.ik_rate.sleep()
                continue

            sol_qpos, sol_vel, success, _info = self.ik_solver.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                current_qpos=qpos_for_ik,
                dt=self.ik_rate.dt,
            )
            if not success:
                self.ik_rate.sleep()
                continue

            try:
                body_targets, twist = self._compute_commands(sol_qpos, sol_vel, body_idx)
                self.controller.set_body_position_targets(body_targets.tolist())
                self.controller.set_base_twist_command(twist)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[controller] command error: {exc}")

            self.ik_rate.sleep()

    def _compute_commands(
        self,
        sol_qpos: np.ndarray,
        sol_qvel: np.ndarray,
        body_idx: list[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        positions_sdk = np.zeros(len(self._sdk_joint_names), dtype=float)
        for i, adr in enumerate(self._sdk_to_mj_qadr):
            if adr is not None:
                positions_sdk[i] = sol_qpos[adr]

        body_targets = positions_sdk[body_idx]
        twist = self._extract_base_twist(sol_qpos, sol_qvel)

        max_lin = 1.5
        max_ang = np.pi / 2
        twist[0] = float(np.clip(twist[0], -max_lin, max_lin))
        twist[1] = float(np.clip(twist[1], -max_lin, max_lin))
        twist[2] = float(np.clip(twist[2], -max_ang, max_ang))
        return body_targets, twist

    def _extract_base_twist(
        self, sol_qpos: np.ndarray, sol_qvel: np.ndarray
    ) -> np.ndarray:
        if sol_qvel.shape[0] < 6:
            return np.zeros(3, dtype=float)

        angular_world = sol_qvel[0:3]
        linear_world = sol_qvel[3:6]

        quat = sol_qpos[3:7]
        R_world_base = self._quat_to_matrix(quat)
        linear_body = R_world_base.T @ linear_world

        return np.array([linear_body[0], linear_body[1], angular_world[2]], dtype=float)

    def visualize_loop(self) -> None:
        if self.viewer is None:
            return
        snapshot = self.robot_state.load()
        if snapshot is not None and snapshot.is_valid:
            try:
                self._apply_snapshot_to_mujoco(snapshot)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[visualize] snapshot apply error: {exc}")

        left_pos = self.data.mocap_pos[self.ee_l_mid].copy()
        right_pos = self.data.mocap_pos[self.ee_r_mid].copy()
        left_quat = self.data.mocap_quat[self.ee_l_mid].copy()
        right_quat = self.data.mocap_quat[self.ee_r_mid].copy()

        self.shared.set_from_viewer(left_pos, left_quat, right_pos, right_quat, self.data.qpos)

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()

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
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def run(self) -> None:
        self._controller_thread = threading.Thread(
            target=self.controller_loop, name="controller", daemon=True
        )
        self._controller_thread.start()

        try:
            if self.headless:
                while not self._stop.is_set():
                    self.state_poll_rate.sleep()
            else:
                while self.viewer.is_running() and not self._stop.is_set():
                    self.visualize_loop()
        finally:
            self._stop.set()
            if self._controller_thread is not None:
                self._controller_thread.join(timeout=2.0)
            if self._state_thread is not None:
                self._state_thread.join(timeout=2.0)
            self.controller.stop()
            try:
                if self.viewer is not None:
                    self.viewer.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body IK GUI with realtime driver")
    parser.add_argument(
        "--address",
        default=os.environ.get("RBY1_ROBOT", "localhost:50051"),
        help="Robot gRPC address (default: env RBY1_ROBOT or localhost:50051)",
    )
    parser.add_argument(
        "--model",
        default=PROJECT_ROOT + "/model/rby1/rby1_mocap.xml",
        help="Path to the MuJoCo model to visualize",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip launching the MuJoCo viewer (useful for debugging controller only).",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    gui = PybindSimGUI(model_path=args.model, address=args.address, headless=args.headless)
    gui.run()


if __name__ == "__main__":
    main()
