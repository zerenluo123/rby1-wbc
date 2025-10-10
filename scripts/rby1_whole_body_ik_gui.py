from pathlib import Path
import sys
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

# ensure project root on sys.path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from ik.rby1_whole_body_ik import RBY1WholeBodyIK


_PROJECT_ROOT = _THIS_DIR.parent
_XML = _PROJECT_ROOT / "model" / "rby1" / "rby1_mocap.xml"


def main():
    # Viewer model/data (independent from IK's internal model)
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # IK solver (loads same XML internally)
    ik = RBY1WholeBodyIK()

    # Helper: FK using viewer model
    def site_pos(site_name: str, qpos: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return data.site_xpos[sid].copy()

    # Initialize from model default
    mujoco.mj_forward(model, data)
    current_qpos = data.qpos.copy()
    prev_qpos = current_qpos.copy()

    # Get mocap ids for target bodies and initialize them at current EE poses
    ee_l_mid = model.body("ee_l_target").mocapid[0]
    ee_r_mid = model.body("ee_r_target").mocapid[0]
    # left_nominal = site_pos("end_effector_l", current_qpos)
    # right_nominal = site_pos("end_effector_r", current_qpos)
    # data.mocap_pos[ee_l_mid] = left_nominal
    # data.mocap_pos[ee_r_mid] = right_nominal
    # Passive viewer loop
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            # Read targets from mocap spheres (drag with mouse in viewer)
            left_target = data.mocap_pos[ee_l_mid].copy()
            right_target = data.mocap_pos[ee_r_mid].copy()
            # left_target[2] = max(left_target[2], 0.2)
            # right_target[2] = max(right_target[2], 0.2)

            # Run whole-body IK from current viewer state
            current_qpos = data.qpos.copy()
            time_start = time.time()
            sol_qpos, success, _info = ik.solve(
                left_target_pos=left_target,
                left_target_quat=None,
                right_target_pos=right_target,
                right_target_quat=None,
                current_qpos=current_qpos,
                dt=rate.dt
            )
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
            # Compute velocities (finite difference)
            dq = (sol_qpos - prev_qpos) / rate.dt
            # Base linear XY speed from free joint positions
            base_xy_speed = float(np.linalg.norm((sol_qpos[0:2] - prev_qpos[0:2]) / rate.dt))

            # Base yaw rate from quaternion difference
            def quat_to_yaw(q):
                w, x, y, z = q
                return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            yaw_now = quat_to_yaw(sol_qpos[3:7])
            yaw_prev = quat_to_yaw(prev_qpos[3:7])
            # Normalize yaw difference to [-pi, pi]
            dyaw = (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            base_yaw_rate = abs(dyaw) / rate.dt

            # Max joint velocity (exclude base qpos[0:7])
            joint_max_vel = float(np.max(np.abs(dq[7:])) if dq.size > 7 else 0.0)

            # Compute IK position error (EE vs mocap targets)
            ee_l_pos = site_pos("end_effector_l", sol_qpos)
            ee_r_pos = site_pos("end_effector_r", sol_qpos)
            l_err = float(np.linalg.norm(ee_l_pos - left_target))
            r_err = float(np.linalg.norm(ee_r_pos - right_target))

            # Print to stdout (may not show under mjpython without flush)
            print(
                f"max_joint_vel={joint_max_vel:.3f} rad/s | base_xy={base_xy_speed:.3f} m/s | base_yaw={base_yaw_rate:.3f} rad/s | l_err={l_err:.3f} m | r_err={r_err:.3f} m",
                flush=True,
            )

            # Apply solution to viewer
            data.qpos[:] = sol_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_camlight(model, data)

            # Update prev state
            prev_qpos = sol_qpos

            # Render
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()


