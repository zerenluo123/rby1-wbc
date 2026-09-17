"""Golden regression for RBY1 whole-body IK.

Record a snapshot from the current solver, then compare after refactors.

    python tests/test_rby1_wbik_regression.py --write-golden
    python tests/test_rby1_wbik_regression.py
    pytest tests/test_rby1_wbik_regression.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rby1.whole_body_ik import RBY1WholeBodyIK  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = FIXTURE_DIR / "rby1_wbik_golden.npz"
DT = 1e-3
ATOL_Q = 1e-8
ATOL_EE = 1e-7
TRAJ_STEPS = 20

# Exact unions from RBY1WholeBodyIK._build_limits_cache before the yaml extract.
# Used only when the solver does not yet expose collision_groups.
HARDCODED_RBY1_COLLISION_GROUPS = {
    "body": sorted(
        {
            "base_col_0",
            "base_col_1",
            "torso_0_col_0",
            "torso_0_col_1",
            "torso_1_col_0",
            "torso_1_col_1",
            "torso_1_col_2",
            "torso_1_col_3",
            "torso_1_col_4",
            "torso_1_col_5",
            "torso_1_col_6",
            "torso_1_col_7",
            "torso_1_col_8",
            "torso_1_col_9",
            "torso_1_col_10",
            "torso_2_col_0",
            "torso_2_col_1",
            "torso_2_col_2",
            "torso_2_col_3",
            "torso_2_col_4",
            "torso_2_col_5",
            "torso_2_col_6",
            "torso_2_col_7",
            "torso_2_col_8",
            "torso_2_col_9",
            "torso_2_col_10",
            "torso_4_col_0",
            "torso_4_col_1",
            "torso_4_col_2",
            "torso_4_col_3",
            "torso_5_col_0",
            "torso_5_col_1",
            "torso_5_col_2",
            "torso_5_col_3",
            "torso_5_col_4",
            "head_col_0",
        }
    ),
    "left_arm": sorted(
        {
            "left_arm_0_col_0",
            "left_arm_0_col_1",
            "left_arm_0_col_2",
            "left_arm_1_col_0",
            "left_arm_2_col_0",
            "left_arm_2_col_1",
            "left_arm_2_col_2",
            "left_arm_2_col_3",
            "left_arm_2_col_4",
            "left_arm_2_col_5",
            "left_arm_2_col_6",
            "left_arm_2_col_7",
            "left_arm_3_col_0",
            "left_arm_3_col_1",
            "left_arm_3_col_2",
            "left_arm_3_col_3",
            "left_arm_4_col_0",
            "left_arm_4_col_1",
            "left_arm_4_col_2",
            "left_arm_4_col_3",
            "left_arm_4_col_4",
            "left_arm_5_col_0",
            "left_arm_5_col_1",
            "left_arm_5_col_2",
            "left_arm_6_col_0",
            "left_arm_7_col_0",
            "left_wrist_cam_col_0",
            "left_wrist_cam_col_1",
            "left_wrist_cam_col_2",
            "left_ee_col_0",
            "left_ee_col_1",
            "left_ee_col_2",
            "left_ee_col_3",
            "left_ee_col_4",
            "left_finger_col_0",
            "left_finger_col_1",
        }
    ),
    "right_arm": sorted(
        {
            "right_arm_0_col_0",
            "right_arm_0_col_1",
            "right_arm_0_col_2",
            "right_arm_1_col_0",
            "right_arm_2_col_0",
            "right_arm_2_col_1",
            "right_arm_2_col_2",
            "right_arm_2_col_3",
            "right_arm_2_col_4",
            "right_arm_2_col_5",
            "right_arm_2_col_6",
            "right_arm_2_col_7",
            "right_arm_3_col_0",
            "right_arm_3_col_1",
            "right_arm_3_col_2",
            "right_arm_3_col_3",
            "right_arm_4_col_0",
            "right_arm_4_col_1",
            "right_arm_4_col_2",
            "right_arm_4_col_3",
            "right_arm_4_col_4",
            "right_arm_5_col_0",
            "right_arm_5_col_1",
            "right_arm_5_col_2",
            "right_arm_6_col_0",
            "right_arm_7_col_0",
            "right_wrist_cam_col_0",
            "right_wrist_cam_col_1",
            "right_wrist_cam_col_2",
            "right_ee_col_0",
            "right_ee_col_1",
            "right_ee_col_2",
            "right_ee_col_3",
            "right_ee_col_4",
            "right_finger_col_0",
            "right_finger_col_1",
        }
    ),
}

HARDCODED_ROBOT_BODIES = [
    "base",
    "wheel_fr_link",
    "wheel_fl_link",
    "wheel_rr_link",
    "wheel_rl_link",
    "link_torso_0",
    "link_torso_1",
    "link_torso_2",
    "link_torso_3",
    "link_torso_4",
    "link_torso_5",
    "link_head_1",
    "link_head_2",
    "link_right_arm_0",
    "link_right_arm_1",
    "link_right_arm_2",
    "link_right_arm_3",
    "link_right_arm_4",
    "link_right_arm_5",
    "link_right_arm_6",
    "FT_SENSOR_R",
    "EE_BODY_R",
    "link_left_arm_0",
    "link_left_arm_1",
    "link_left_arm_2",
    "link_left_arm_3",
    "link_left_arm_4",
    "link_left_arm_5",
    "link_left_arm_6",
    "FT_SENSOR_L",
    "EE_BODY_L",
]


def _as_str_list(values) -> list[str]:
    return [str(v) for v in values]


def _site_pose(ik: RBY1WholeBodyIK, name: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_SITE, name)
    pos = ik.data.site_xpos[site_id].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, ik.data.site_xmat[site_id])
    return pos, quat


def _forward(ik: RBY1WholeBodyIK, qpos: np.ndarray) -> None:
    ik.data.qpos[:] = qpos
    mujoco.mj_forward(ik.model, ik.data)


def _nominal_qpos(ik: RBY1WholeBodyIK) -> np.ndarray:
    qpos = ik._get_nominal_posture(ik.data.qpos.copy())
    if ik.use_ik_adjustment:
        ik._apply_torso_qpos_constraints_inplace(qpos)
    return qpos


def _collision_groups(ik: RBY1WholeBodyIK) -> dict[str, list[str]]:
    groups = getattr(ik, "collision_groups", None)
    if groups is None:
        return {key: list(names) for key, names in HARDCODED_RBY1_COLLISION_GROUPS.items()}
    return {key: sorted(groups[key]) for key in sorted(groups)}


def _robot_bodies(ik: RBY1WholeBodyIK) -> list[str]:
    names = getattr(ik, "robot_body_names", None)
    if names is None:
        return list(HARDCODED_ROBOT_BODIES)
    return _as_str_list(names)


def _geom_pair_keys(ik: RBY1WholeBodyIK) -> np.ndarray:
    keys = []
    for geom_a, geom_b in ik.collision_avoidance_limit.geom_id_pairs:
        name_a = mujoco.mj_id2name(ik.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_a))
        name_b = mujoco.mj_id2name(ik.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_b))
        first, second = sorted((name_a, name_b))
        keys.append(f"{first}|{second}")
    return np.array(sorted(set(keys)))


def _frame_snapshot(ik: RBY1WholeBodyIK) -> dict[str, str]:
    chest = getattr(ik, "chest_name", ik.torso5_name)
    return {
        "base": ik.base_name,
        "chest": chest,
        "end_effector_l": ik.left_ee_name,
        "end_effector_r": ik.right_ee_name,
        "head": ik.head_name,
    }


def _record_solve(
    ik: RBY1WholeBodyIK,
    qpos: np.ndarray,
    *,
    left_pos,
    left_quat,
    right_pos,
    right_quat,
    head_pos=None,
    head_quat=None,
) -> dict[str, np.ndarray]:
    sol_qpos, sol_qvel, success, _info = ik.solve(
        left_target_pos=left_pos,
        left_target_quat=left_quat,
        right_target_pos=right_pos,
        right_target_quat=right_quat,
        head_target_pos=head_pos,
        head_target_quat=head_quat,
        current_qpos=qpos,
        dt=DT,
    )
    _forward(ik, sol_qpos)
    left_ee, _ = _site_pose(ik, ik.left_ee_name)
    right_ee, _ = _site_pose(ik, ik.right_ee_name)
    head_ee, _ = _site_pose(ik, ik.head_name)
    return {
        "qpos": sol_qpos.copy(),
        "qvel": sol_qvel.copy(),
        "success": np.array([int(success)], dtype=np.int32),
        "left_ee": left_ee,
        "right_ee": right_ee,
        "head_ee": head_ee,
    }


def snapshot_ik(ik: RBY1WholeBodyIK | None = None) -> dict[str, np.ndarray]:
    if ik is None:
        ik = RBY1WholeBodyIK()

    groups = _collision_groups(ik)
    payload: dict[str, np.ndarray] = {
        "frames_json": np.array(json.dumps(_frame_snapshot(ik), sort_keys=True)),
        "torso_joint_names": np.array(ik.torso_joint_names),
        "left_arm_joint_names": np.array(ik.left_arm_joint_names),
        "right_arm_joint_names": np.array(ik.right_arm_joint_names),
        "head_joint_names": np.array(ik.head_joint_names),
        "wheel_joint_names": np.array(ik.wheel_joint_names),
        "wheel_names": np.array(ik.wheel_names),
        "torso_qpos_indices": np.asarray(ik.torso_qpos_indices, dtype=np.int32),
        "torso_dof_indices": np.asarray(ik.torso_dof_indices, dtype=np.int32),
        "left_arm_qpos_indices": np.asarray(ik.left_arm_qpos_indices, dtype=np.int32),
        "left_arm_dof_indices": np.asarray(ik.left_arm_dof_indices, dtype=np.int32),
        "right_arm_qpos_indices": np.asarray(ik.right_arm_qpos_indices, dtype=np.int32),
        "right_arm_dof_indices": np.asarray(ik.right_arm_dof_indices, dtype=np.int32),
        "head_qpos_indices": np.asarray(ik.head_qpos_indices, dtype=np.int32),
        "head_dof_indices": np.asarray(ik.head_dof_indices, dtype=np.int32),
        "com_over_base_xy_target": np.asarray(ik.com_over_base_xy_target, dtype=float),
        "collision_body": np.array(groups["body"]),
        "collision_left_arm": np.array(groups["left_arm"]),
        "collision_right_arm": np.array(groups["right_arm"]),
        "robot_bodies": np.array(_robot_bodies(ik)),
        "geom_pairs": _geom_pair_keys(ik),
    }

    q0 = _nominal_qpos(ik)
    _forward(ik, q0)
    left_pos, left_quat = _site_pose(ik, ik.left_ee_name)
    right_pos, right_quat = _site_pose(ik, ik.right_ee_name)
    head_pos, head_quat = _site_pose(ik, ik.head_name)
    payload["nominal_qpos"] = q0.copy()
    payload["nominal_left_ee"] = left_pos.copy()
    payload["nominal_right_ee"] = right_pos.copy()
    payload["nominal_head_ee"] = head_pos.copy()

    cases = {
        "hold": _record_solve(
            ik,
            q0,
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
        ),
        "left_reach": _record_solve(
            ik,
            q0,
            left_pos=left_pos + np.array([0.05, 0.0, 0.0]),
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
        ),
        "dual_reach": _record_solve(
            ik,
            q0,
            left_pos=left_pos + np.array([0.05, 0.0, 0.0]),
            left_quat=left_quat,
            right_pos=right_pos + np.array([0.05, 0.0, 0.0]),
            right_quat=right_quat,
        ),
        "with_head": _record_solve(
            ik,
            q0,
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
            head_pos=head_pos,
            head_quat=head_quat,
        ),
    }
    for name, record in cases.items():
        for key, value in record.items():
            payload[f"case_{name}_{key}"] = value

    q = q0.copy()
    traj_qpos = []
    traj_qvel = []
    traj_success = []
    traj_left = []
    traj_right = []
    traj_head = []
    for step in range(TRAJ_STEPS):
        alpha = (step + 1) / TRAJ_STEPS
        record = _record_solve(
            ik,
            q,
            left_pos=left_pos + np.array([0.05 * alpha, 0.0, 0.0]),
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
        )
        q = record["qpos"]
        traj_qpos.append(record["qpos"])
        traj_qvel.append(record["qvel"])
        traj_success.append(record["success"][0])
        traj_left.append(record["left_ee"])
        traj_right.append(record["right_ee"])
        traj_head.append(record["head_ee"])
    payload["traj_qpos"] = np.stack(traj_qpos, axis=0)
    payload["traj_qvel"] = np.stack(traj_qvel, axis=0)
    payload["traj_success"] = np.asarray(traj_success, dtype=np.int32)
    payload["traj_left_ee"] = np.stack(traj_left, axis=0)
    payload["traj_right_ee"] = np.stack(traj_right, axis=0)
    payload["traj_head_ee"] = np.stack(traj_head, axis=0)
    return payload


def write_golden(path: Path = GOLDEN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot_ik()
    np.savez_compressed(path, **payload)
    return path


def _assert_string_array_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    actual_list = [str(x) for x in actual.tolist()]
    expected_list = [str(x) for x in expected.tolist()]
    if actual_list != expected_list:
        missing = sorted(set(expected_list) - set(actual_list))
        extra = sorted(set(actual_list) - set(expected_list))
        raise AssertionError(
            f"{name} mismatch: missing={missing[:12]} extra={extra[:12]} "
            f"(counts actual={len(actual_list)} expected={len(expected_list)})"
        )


def compare_to_golden(path: Path = GOLDEN_PATH) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Golden fixture missing: {path}. Run with --write-golden first.")
    golden = np.load(path, allow_pickle=True)
    actual = snapshot_ik()

    for key in (
        "frames_json",
        "torso_joint_names",
        "left_arm_joint_names",
        "right_arm_joint_names",
        "head_joint_names",
        "wheel_joint_names",
        "wheel_names",
        "robot_bodies",
        "collision_body",
        "collision_left_arm",
        "collision_right_arm",
        "geom_pairs",
    ):
        _assert_string_array_equal(key, actual[key], golden[key])

    for key in (
        "torso_qpos_indices",
        "torso_dof_indices",
        "left_arm_qpos_indices",
        "left_arm_dof_indices",
        "right_arm_qpos_indices",
        "right_arm_dof_indices",
        "head_qpos_indices",
        "head_dof_indices",
        "traj_success",
    ):
        if not np.array_equal(actual[key], golden[key]):
            raise AssertionError(f"{key} mismatch:\n actual={actual[key]}\n golden={golden[key]}")

    def check_close(name: str, atol: float) -> None:
        if not np.allclose(actual[name], golden[name], atol=atol, rtol=0.0):
            delta = np.max(np.abs(actual[name] - golden[name]))
            raise AssertionError(f"{name} max abs delta={delta} exceeds atol={atol}")

    check_close("com_over_base_xy_target", ATOL_Q)
    check_close("nominal_qpos", ATOL_Q)
    for case in ("hold", "left_reach", "dual_reach", "with_head"):
        if not np.array_equal(actual[f"case_{case}_success"], golden[f"case_{case}_success"]):
            raise AssertionError(
                f"{case} success {actual[f'case_{case}_success']} != {golden[f'case_{case}_success']}"
            )
        check_close(f"case_{case}_qpos", ATOL_Q)
        check_close(f"case_{case}_qvel", ATOL_Q)
        check_close(f"case_{case}_left_ee", ATOL_EE)
        check_close(f"case_{case}_right_ee", ATOL_EE)
        check_close(f"case_{case}_head_ee", ATOL_EE)
    check_close("traj_qpos", ATOL_Q)
    check_close("traj_qvel", ATOL_Q)
    check_close("traj_left_ee", ATOL_EE)
    check_close("traj_right_ee", ATOL_EE)
    check_close("traj_head_ee", ATOL_EE)


def test_rby1_wbik_matches_golden() -> None:
    compare_to_golden()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="Record a new golden fixture from the current solver. Do not use after refactors.",
    )
    args = parser.parse_args(argv)
    if args.write_golden:
        written = write_golden()
        print(f"Wrote golden fixture: {written}")
        return 0
    compare_to_golden()
    print("RBY1 WBIK golden regression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
