from pathlib import Path
import sys
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

# Ensure project root is on sys.path regardless of current working directory.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rby1.whole_body_ik import RBY1WholeBodyIK

def main():
    # Viewer model/data (independent from IK's internal model)
    model = mujoco.MjModel.from_xml_path(PROJECT_ROOT + "/model/rby1/rby1_mocap.xml")
    data = mujoco.MjData(model)
    ik = RBY1WholeBodyIK()

    # Helper: FK using viewer model
    def site_pose(site_name: str, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = data.site_xpos[sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
        return pos, quat

    def site_pos(site_name: str, qpos: np.ndarray) -> np.ndarray:
        pos, _ = site_pose(site_name, qpos)
        return pos

    # Initialize from model default
    mujoco.mj_forward(model, data)
    # Use IK's nominal pose as initial posture for both IK and viewer
    nominal_qpos = ik._get_nominal_posture(ik.data.qpos.copy())
    ik.data.qpos[:] = nominal_qpos
    mujoco.mj_forward(ik.model, ik.data)
    ik.configuration.update(q=ik.data.qpos)

    data.qpos[:] = nominal_qpos
    current_qpos = data.qpos.copy()
    prev_qpos = current_qpos.copy()

    # Get mocap ids for target bodies and initialize them at current EE poses
    ee_l_mid = model.body("ee_l_target").mocapid[0]
    ee_r_mid = model.body("ee_r_target").mocapid[0]
    head_mid = model.body("head_target").mocapid[0]

    left_nominal_pos = site_pos("end_effector_l", current_qpos)
    right_nominal_pos = site_pos("end_effector_r", current_qpos)
    head_nominal_pos, head_nominal_quat = site_pose("head", current_qpos)

    left_nominal_quat = data.xquat[model.body("EE_BODY_L").id].copy()
    right_nominal_quat = data.xquat[model.body("EE_BODY_R").id].copy()

    data.mocap_pos[ee_l_mid] = left_nominal_pos
    data.mocap_pos[ee_r_mid] = right_nominal_pos
    data.mocap_pos[head_mid] = head_nominal_pos
    data.mocap_quat[ee_l_mid] = left_nominal_quat
    data.mocap_quat[ee_r_mid] = right_nominal_quat
    data.mocap_quat[head_mid] = head_nominal_quat

    # Passive viewer loop
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=30.0, warn=False)

        while viewer.is_running():
            # Read targets from mocap spheres (drag with mouse in viewer)
            left_pos = data.mocap_pos[ee_l_mid].copy()
            left_quat = data.mocap_quat[ee_l_mid].copy()
            right_pos = data.mocap_pos[ee_r_mid].copy()
            right_quat = data.mocap_quat[ee_r_mid].copy()
            head_pos = data.mocap_pos[head_mid].copy()
            head_quat = data.mocap_quat[head_mid].copy()

            # Run whole-body IK from current viewer state
            current_qpos = data.qpos.copy()
            ik_start = time.perf_counter()
            sol_qpos, sol_qvel, success, _info = ik.solve(
                left_target_pos=left_pos,
                left_target_quat=left_quat,
                right_target_pos=right_pos,
                right_target_quat=right_quat,
                head_target_pos=None,
                head_target_quat=None,
                current_qpos=current_qpos,
                dt=rate.dt,
            )
            ik_elapsed_ms = (time.perf_counter() - ik_start) * 1000.0
            # print(f"[wbik_gui] IK solve took {ik_elapsed_ms:.3f} ms")
            if not success:
                print(f"[wbc] IK failed: {_info}")

            # Apply solution to viewer
            data.qpos[:] = sol_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_camlight(model, data)

            # Render
            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()
