import os
import sys
import numpy as np
import mujoco
from pathlib import Path

# Ensure local imports work when executing from the repo root
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from ik.rby1_whole_body_ik import RBY1WholeBodyIK  # noqa: E402


def main(num_trials: int = 10, pos_noise: float = 0.05, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)

    # Create IK solver (independent from test FK model)
    solver = RBY1WholeBodyIK()

    # Load an independent MuJoCo model for FK evaluation
    model_path = os.path.join(_THIS_DIR.parent, "model", "rby1", "rby1.xml")
    fk_model = mujoco.MjModel.from_xml_path(model_path)
    fk_data = mujoco.MjData(fk_model)

    # Use the FK model's initial configuration as the seed
    current_qpos = fk_data.qpos.copy()
    mujoco.mj_forward(fk_model, fk_data)

    # Helper to compute FK site position without mutating IK solver
    def fk_site_position(site_name: str, qpos: np.ndarray) -> np.ndarray:
        fk_data.qpos[:] = qpos
        mujoco.mj_forward(fk_model, fk_data)
        site_id = mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return fk_data.site_xpos[site_id].copy()

    # Nominal EE positions from FK model
    left_nominal = fk_site_position("end_effector_l", current_qpos)
    right_nominal = fk_site_position("end_effector_r", current_qpos)

    left_errors = []
    right_errors = []
    successes = 0

    for i in range(num_trials):
        # Sample small random deltas around the nominal positions
        left_target = left_nominal + rng.uniform(-pos_noise, pos_noise, size=3)
        right_target = right_nominal + rng.uniform(-pos_noise, pos_noise, size=3)

        # Keep targets at reasonable height (avoid going underground)
        left_target[2] = max(left_target[2], 0.2)
        right_target[2] = max(right_target[2], 0.2)

        # Solve position-only IK (orientation unconstrained for robustness)
        sol_qpos, success, info = solver.solve(
            left_target_pos=left_target,
            left_target_quat=None,
            right_target_pos=right_target,
            right_target_quat=None,
            current_qpos=current_qpos,
        )

        # Forward kinematics on independent FK model to measure errors
        left_fk = fk_site_position("end_effector_l", sol_qpos)
        right_fk = fk_site_position("end_effector_r", sol_qpos)
        l_err = float(np.linalg.norm(left_fk - left_target))
        r_err = float(np.linalg.norm(right_fk - right_target))
        left_errors.append(l_err)
        right_errors.append(r_err)
        successes += int(bool(success))

        # Update current_qpos to the latest solution to keep sampling locally
        current_qpos = sol_qpos

        print(f"Trial {i+1:02d}: success={bool(success)} | left_err={l_err:.4f} m | right_err={r_err:.4f} m")

    # Report summary
    print("\nSummary:")
    print(f"  Trials: {num_trials}")
    print(f"  Successes: {successes}/{num_trials}")
    print(f"  Left  error (mean/std/max): {np.mean(left_errors):.4f} / {np.std(left_errors):.4f} / {np.max(left_errors):.4f} m")
    print(f"  Right error (mean/std/max): {np.mean(right_errors):.4f} / {np.std(right_errors):.4f} / {np.max(right_errors):.4f} m")

    # Return non-zero if too large average error
    mean_err = 0.5 * (np.mean(left_errors) + np.mean(right_errors))
    return 0 if mean_err < 0.05 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


