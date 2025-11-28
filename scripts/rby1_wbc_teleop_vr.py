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

import mujoco
import mujoco.viewer
import numpy as np

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_control import RBY1WBC
from rby1.ee_targets import EETargets
from teleop.teleop_vr import TeleopVR


def _quat_angle_error(target_quat: np.ndarray, actual_quat: np.ndarray) -> float:
    target = np.asarray(target_quat, dtype=np.float64)
    actual = np.asarray(actual_quat, dtype=np.float64)
    target_norm = np.linalg.norm(target)
    actual_norm = np.linalg.norm(actual)
    if target_norm < 1e-9 or actual_norm < 1e-9:
        return float("nan")
    target /= target_norm
    actual /= actual_norm
    dot = float(np.dot(target, actual))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


class RBY1WBCTeleopVR:
    def __init__(self, wbc: RBY1WBC, teleop: TeleopVR, headless: bool = False) -> None:
        self.wbc = wbc
        self.teleop = teleop
        self.headless = headless

        self.model_path = self.wbc.model_path
        self.trajectory_frequency_hz = self.wbc.trajectory_frequency_hz

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        snapshot = self._wait_for_initial_snapshot()
        if snapshot is None:
            raise RuntimeError("Failed to receive initial robot state snapshot")
        self._apply_snapshot(snapshot)

        # Viewer setup
        self.viewer = None
        self.viewer_rate = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
            )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)
        else:
            self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

        # Trajectory streamer setup
        self.trajectory_rate = RateLimiter(frequency=self.trajectory_frequency_hz, warn=False)
        self._stop_event = threading.Event()
        self._trajectory_thread: Optional[threading.Thread] = None

        self._left_ee_site_id = self.model.site("end_effector_l").id
        self._right_ee_site_id = self.model.site("end_effector_r").id
        self._head_ee_site_id = self.model.site("head").id
        base_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, base_data)
        base_mat = base_data.site_xmat[self._head_ee_site_id].copy()
        self._head_site_base_rot = base_mat.reshape(3, 3, order="F")

        self.teleop.start()

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
        if self.viewer is None or self.viewer_rate is None:
            return

        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is not None and snapshot.is_valid:
            try:
                self._apply_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[visualize] snapshot apply error: {exc}")

        (
            policy_left_pos,
            policy_left_quat,
            _plw,
            policy_right_pos,
            policy_right_quat,
            _prw,
            policy_head_pos,
            policy_head_quat,
        ) = self.wbc.shared_targets.get_target()
        actual_left_pos, actual_left_quat = self._get_site_pose(self._left_ee_site_id)
        actual_right_pos, actual_right_quat = self._get_site_pose(self._right_ee_site_id)
        actual_head_pos, actual_head_quat = self._get_site_pose(self._head_ee_site_id)

        policy_head_pos_viz, policy_head_quat_viz = self._prepare_head_policy_pose(
            policy_head_pos,
            policy_head_quat,
            actual_head_pos,
        )

        self._update_pose_markers(
            policy_left_pos,
            policy_left_quat,
            policy_right_pos,
            policy_right_quat,
            policy_head_pos_viz,
            policy_head_quat_viz,
            actual_left_pos,
            actual_left_quat,
            actual_right_pos,
            actual_right_quat,
            actual_head_pos,
            actual_head_quat,
        )
        left_err = self._format_ee_error("L", policy_left_pos, policy_left_quat, actual_left_pos, actual_left_quat)
        right_err = self._format_ee_error("R", policy_right_pos, policy_right_quat, actual_right_pos, actual_right_quat)
        head_err = self._format_ee_error(
            "H",
            policy_head_pos_viz,
            policy_head_quat_viz,
            actual_head_pos,
            actual_head_quat,
            include_position=False,
        )

        errors = [err for err in (left_err, right_err, head_err) if err is not None]
        # if errors:
        #     msg = f"[EE error] {' | '.join(errors)}"
        #     sys.stdout.write(f"\r{msg}\x1b[K")
        #     sys.stdout.flush()

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()

    def _update_pose_markers(
        self,
        policy_left_pos: Optional[np.ndarray],
        policy_left_quat: Optional[np.ndarray],
        policy_right_pos: Optional[np.ndarray],
        policy_right_quat: Optional[np.ndarray],
        policy_head_pos: Optional[np.ndarray],
        policy_head_quat: Optional[np.ndarray],
        actual_left_pos: np.ndarray,
        actual_left_quat: np.ndarray,
        actual_right_pos: np.ndarray,
        actual_right_quat: np.ndarray,
        actual_head_pos: np.ndarray,
        actual_head_quat: np.ndarray,
    ) -> None:
        if self.viewer is None:
            return
        scene = self.viewer.user_scn
        if scene is None:
            return

        scene.ngeom = 0
        policy_alpha = 0.5
        policy_width = 1.5
        actual_alpha = 0.9
        actual_width = 3.0

        self._add_pose_marker(scene, policy_left_pos, policy_left_quat, policy_alpha, policy_width)
        self._add_pose_marker(scene, policy_right_pos, policy_right_quat, policy_alpha, policy_width)
        self._add_pose_marker(scene, policy_head_pos, policy_head_quat, policy_alpha, policy_width)
        self._add_pose_marker(scene, actual_left_pos, actual_left_quat, actual_alpha, actual_width)
        self._add_pose_marker(scene, actual_right_pos, actual_right_quat, actual_alpha, actual_width)
        self._add_pose_marker(scene, actual_head_pos, actual_head_quat, actual_alpha, actual_width)

    def _add_pose_marker(
        self,
        scene: mujoco.MjvScene,
        pos: Optional[np.ndarray],
        quat: Optional[np.ndarray],
        alpha: float,
        line_width: float,
    ) -> None:
        if pos is None or quat is None or scene.ngeom >= scene.maxgeom:
            return

        pos_arr = np.asarray(pos, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)

        axis_length = 0.15
        line_size = np.zeros(3, dtype=np.float64)
        identity_mat = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        axis_colors = (
            np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, alpha], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, alpha], dtype=np.float32),
        )

        frame_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(frame_mat, quat_arr)
        rot = frame_mat.reshape(3, 3)

        for axis_idx, axis_color in enumerate(axis_colors):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                line_size,
                pos_arr,
                identity_mat,
                axis_color,
            )
            endpoint = pos_arr + rot[:, axis_idx] * axis_length
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                line_width,
                pos_arr,
                endpoint,
            )
            scene.ngeom += 1

    def _get_site_pose(self, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[site_id].copy()
        mat = self.data.site_xmat[site_id].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat

    def _format_ee_error(
        self,
        label: str,
        target_pos: Optional[np.ndarray],
        target_quat: Optional[np.ndarray],
        actual_pos: np.ndarray,
        actual_quat: np.ndarray,
        include_position: bool = True,
    ) -> Optional[str]:
        if target_pos is None or target_quat is None:
            return None
        parts: list[str] = []
        if include_position:
            pos_err = np.asarray(target_pos, dtype=np.float64) - np.asarray(actual_pos, dtype=np.float64)
            pos_err_norm = float(np.linalg.norm(pos_err))
            parts.append(f"dpos={pos_err_norm:.4f}")
        quat_err = _quat_angle_error(target_quat, actual_quat)
        parts.append(f"dq={quat_err:.4f}")
        return f"{label} {' '.join(parts)}"

    def _prepare_head_policy_pose(
        self,
        policy_pos: Optional[np.ndarray],
        policy_quat: Optional[np.ndarray],
        actual_pos: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if policy_quat is None:
            return None, None
        projected_quat = self._project_head_orientation(policy_quat)
        if projected_quat is None:
            return None, None
        return actual_pos.copy(), projected_quat

    def _project_head_orientation(self, policy_quat: np.ndarray) -> Optional[np.ndarray]:
        quat_arr = np.asarray(policy_quat, dtype=np.float64)
        norm = np.linalg.norm(quat_arr)
        if norm < 1e-9:
            return None
        quat_arr = quat_arr / norm

        target_rot_flat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(target_rot_flat, quat_arr)
        target_rot = target_rot_flat.reshape(3, 3, order="F")

        relative_rot = self._head_site_base_rot.T @ target_rot

        rel_20 = float(np.clip(relative_rot[2, 0], -1.0, 1.0))
        pitch = -math.asin(rel_20)
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) < 1e-8:
            yaw = math.atan2(-relative_rot[0, 1], relative_rot[1, 1])
        else:
            yaw = math.atan2(relative_rot[1, 0], relative_rot[0, 0])

        projected_rot = self._head_site_base_rot @ self._rotation_z(yaw) @ self._rotation_y(pitch)
        projected_flat = projected_rot.reshape(9, order="F")
        projected_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(projected_quat, projected_flat)
        return projected_quat

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )

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
                timestamp=timestamp
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
                    self.viewer_rate.sleep()
            else:
                while self.viewer.is_running():
                    self.visualize_loop()
        except KeyboardInterrupt:
            self._stop_event.set()
            return

    def close(self) -> None:
        self._stop_event.set()
        if self._trajectory_thread is not None:
            self._trajectory_thread.join(timeout=1.0)
        self.teleop.stop()
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body teleop GUI driven by Meta Quest")
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
    teleop = TeleopVR(wbc=wbc, local_ip=args.local_ip, meta_quest_ip=args.meta_quest_ip, save_trajectory=args.save)
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
