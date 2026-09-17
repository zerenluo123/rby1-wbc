"""MuJoCo preview for R1 Pro whole-body IK.

Loads config/wbik_r1pro.yaml → model/r1pro/r1pro_model.yaml.
Drag the red mocap spheres to move the hands. Head target is shown but not
sent to IK (same as rby1_wbik_gui.py).
"""
from pathlib import Path
import sys
import time
import math
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_ik import WholeBodyIK


def main():
    model = mujoco.MjModel.from_xml_path(PROJECT_ROOT + "/model/r1pro/r1pro_mocap.xml")
    data = mujoco.MjData(model)
    ik = WholeBodyIK(PROJECT_ROOT + "/config/wbik_r1pro.yaml")
    if model.nq != ik.model.nq:
        raise RuntimeError(
            f"Viewer nq={model.nq} != IK nq={ik.model.nq}. "
            "mocap wrapper must not add joints."
        )

    left_site = ik.left_ee_name
    right_site = ik.right_ee_name
    head_site = ik.head_name

    joint_entries = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_entries.append((model.joint(j).name, model.jnt_qposadr[j]))

    def site_pose(site_name: str, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = data.site_xpos[sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        return pos, quat

    def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        q1 = q1 / max(np.linalg.norm(q1), 1e-9)
        q2 = q2 / max(np.linalg.norm(q2), 1e-9)
        dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
        return math.degrees(2.0 * math.acos(abs(dot)))

    def _passes_incremental_safety(
        current_qpos: np.ndarray,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        head_pos: Optional[np.ndarray],
        head_quat: Optional[np.ndarray],
    ) -> bool:
        MAX_POS_DELTA = 0.1
        MAX_ROT_DELTA_DEG = 20.0
        cur_left_pos, cur_left_quat = site_pose(left_site, current_qpos)
        cur_right_pos, cur_right_quat = site_pose(right_site, current_qpos)
        left_delta = float(np.linalg.norm(left_pos - cur_left_pos))
        left_rot_delta = _quat_angle_deg(cur_left_quat, left_quat)
        if left_delta > MAX_POS_DELTA or left_rot_delta > MAX_ROT_DELTA_DEG:
            print(
                f"[r1pro_wbik_gui] reject left target: dpos={left_delta:.3f}m"
                f" drot={left_rot_delta:.1f}deg"
            )
            return False
        right_delta = float(np.linalg.norm(right_pos - cur_right_pos))
        right_rot_delta = _quat_angle_deg(cur_right_quat, right_quat)
        if right_delta > MAX_POS_DELTA or right_rot_delta > MAX_ROT_DELTA_DEG:
            print(
                f"[r1pro_wbik_gui] reject right target: dpos={right_delta:.3f}m"
                f" drot={right_rot_delta:.1f}deg"
            )
            return False
        return True

    mujoco.mj_forward(model, data)
    nominal_qpos = ik._get_nominal_posture(ik.data.qpos.copy())
    if ik.use_ik_adjustment:
        ik._apply_torso_qpos_constraints_inplace(nominal_qpos)
    ik.data.qpos[:] = nominal_qpos
    mujoco.mj_forward(ik.model, ik.data)
    ik.configuration.update(q=ik.data.qpos)

    data.qpos[:] = nominal_qpos
    current_qpos = data.qpos.copy()

    ee_l_mid = model.body("ee_l_target").mocapid[0]
    ee_r_mid = model.body("ee_r_target").mocapid[0]
    head_mid = model.body("head_target").mocapid[0]

    left_nominal_pos, left_nominal_quat = site_pose(left_site, current_qpos)
    right_nominal_pos, right_nominal_quat = site_pose(right_site, current_qpos)
    head_nominal_pos, head_nominal_quat = site_pose(head_site, current_qpos)

    data.mocap_pos[ee_l_mid] = left_nominal_pos
    data.mocap_pos[ee_r_mid] = right_nominal_pos
    data.mocap_pos[head_mid] = head_nominal_pos
    data.mocap_quat[ee_l_mid] = left_nominal_quat
    data.mocap_quat[ee_r_mid] = right_nominal_quat
    data.mocap_quat[head_mid] = head_nominal_quat

    print(
        f"[r1pro_wbik_gui] loaded wbik_r1pro.yaml / r1pro_model.yaml  "
        f"nq={model.nq}  chest={ik.torso5_name}  "
        f"ee=({left_site}, {right_site})"
    )

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # Visual meshes on; collision spheres stay in the model for IK but hidden
        # like rby1_wbik_gui.py. Toggle group 3 in the viewer if you want to inspect them.
        viewer.opt.geomgroup[2] = 1
        viewer.opt.geomgroup[3] = 0

        rate = RateLimiter(frequency=30.0, warn=False)
        last_print_time = 0.0
        print_period_s = 0.5

        while viewer.is_running():
            left_pos = data.mocap_pos[ee_l_mid].copy()
            left_quat = data.mocap_quat[ee_l_mid].copy()
            right_pos = data.mocap_pos[ee_r_mid].copy()
            right_quat = data.mocap_quat[ee_r_mid].copy()
            head_pos = data.mocap_pos[head_mid].copy()
            head_quat = data.mocap_quat[head_mid].copy()

            current_qpos = data.qpos.copy()
            if not _passes_incremental_safety(
                current_qpos,
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                head_pos,
                head_quat,
            ):
                print("[r1pro_wbik_gui] resetting targets to current pose")
                left_pos, left_quat = site_pose(left_site, current_qpos)
                right_pos, right_quat = site_pose(right_site, current_qpos)
                head_pos, head_quat = site_pose(head_site, current_qpos)
                data.mocap_pos[ee_l_mid] = left_pos
                data.mocap_pos[ee_r_mid] = right_pos
                data.mocap_pos[head_mid] = head_pos
                data.mocap_quat[ee_l_mid] = left_quat
                data.mocap_quat[ee_r_mid] = right_quat
                data.mocap_quat[head_mid] = head_quat

            sol_qpos, _sol_qvel, success, info = ik.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                head_target_pos=None,
                head_target_quat=None,
                current_qpos=current_qpos,
                dt=rate.dt,
            )
            if not success:
                print(f"[r1pro_wbik_gui] IK failed: {info}")

            data.qpos[:] = sol_qpos
            now = time.perf_counter()
            if now - last_print_time >= print_period_s:
                last_print_time = now
                torso = ", ".join(
                    f"{name}={sol_qpos[adr]:.3f}"
                    for name, adr in joint_entries
                    if name.startswith("torso_")
                )
                print(f"[r1pro_wbik_gui] {torso} success={success}")
            mujoco.mj_forward(model, data)
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
