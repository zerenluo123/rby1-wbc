"""GUI frontend that drives targets for the standalone RBY1 WBC thread."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.rby1_wbc import RBY1WBC


class RBY1WBCGui:
    def __init__(self, model_path: str, wbc: RBY1WBC, headless: bool = False):
        self.model_path = model_path
        self.wbc = wbc
        self.headless = headless

        self.viewer = None if headless else self._init_viewer(model_path)
        self.viewer_rate = RateLimiter(frequency=60.0, warn=False)

    def _init_viewer(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.ee_l_mid = self.model.body("ee_l_target").mocapid[0]
        self.ee_r_mid = self.model.body("ee_r_target").mocapid[0]
        self.head_mid = self.model.body("head_target").mocapid[0]
        self._head_site_id = self.model.site("head").id

        snapshot = self._wait_for_initial_snapshot()
        if snapshot is None:
            raise RuntimeError("Failed to receive initial robot state snapshot")
        self._apply_snapshot(snapshot)

        left_nominal = self.site_pos("end_effector_l", self.data.qpos)
        right_nominal = self.site_pos("end_effector_r", self.data.qpos)
        head_nominal_pos, head_nominal_quat = self.site_pose("head", self.data.qpos)
        self.data.mocap_pos[self.ee_l_mid] = left_nominal
        self.data.mocap_pos[self.ee_r_mid] = right_nominal
        self.data.mocap_pos[self.head_mid] = head_nominal_pos
        l_q = self.data.xquat[self.model.body("EE_BODY_L").id].copy()
        r_q = self.data.xquat[self.model.body("EE_BODY_R").id].copy()
        self.data.mocap_quat[self.ee_l_mid] = l_q
        self.data.mocap_quat[self.ee_r_mid] = r_q
        self.data.mocap_quat[self.head_mid] = head_nominal_quat

        viewer = mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        )
        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
        return viewer

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

    def site_pose(self, site_name: str, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = self.data.site_xpos[sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[sid])
        return pos, quat

    def site_pos(self, site_name: str, qpos: np.ndarray) -> np.ndarray:
        pos, _ = self.site_pose(site_name, qpos)
        return pos

    def visualize_loop(self) -> None:
        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is not None and snapshot.is_valid:
            try:
                self._apply_snapshot(snapshot)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[visualize] snapshot apply error: {exc}")

        left_pos = self.data.mocap_pos[self.ee_l_mid].copy()
        right_pos = self.data.mocap_pos[self.ee_r_mid].copy()
        head_pos = self.data.mocap_pos[self.head_mid].copy()
        left_quat = self.data.mocap_quat[self.ee_l_mid].copy()
        right_quat = self.data.mocap_quat[self.ee_r_mid].copy()
        head_quat = self.data.mocap_quat[self.head_mid].copy()

        self.wbc.update_targets(
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            head_pos=head_pos,
            head_quat=head_quat,
        )

        mujoco.mj_camlight(self.model, self.data)
        self.viewer.sync()
        self.viewer_rate.sleep()

    def run(self) -> None:
        if self.headless:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                return
        else:
            try:
                while self.viewer.is_running():
                    self.visualize_loop()
            except KeyboardInterrupt:
                pass

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 whole-body IK GUI decoupled from WBC thread")
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

    wbc = RBY1WBC(model_path=args.model, address=args.address, ik_frequency_hz=100.0)
    wbc.start()

    gui = None
    try:
        gui = RBY1WBCGui(model_path=args.model, wbc=wbc, headless=args.headless)
        gui.run()
    finally:
        if gui is not None:
            gui.close()
        wbc.stop()


if __name__ == "__main__":
    main()
