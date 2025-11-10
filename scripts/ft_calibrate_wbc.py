#!/usr/bin/env python3
"""Force/torque calibration routine driven by the RBY1 WBC."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Ensure the project root is on the import path.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from control.rby1_wbc import RBY1WBC, RobotSnapshot


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


def _normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        raise ValueError("Quaternion norm is zero")
    return quat / norm


def _apply_global_tilts(initial_quat: np.ndarray, tilt_x: float, tilt_y: float, tilt_z: float) -> np.ndarray:
    """Returns the pose quaternion after tilting about world X, Y, then Z (degrees)."""
    base = Rotation.from_quat(_wxyz_to_xyzw(initial_quat))
    tilt = Rotation.from_euler("xyz", [tilt_x, tilt_y, tilt_z], degrees=True)
    result = tilt * base
    return _normalize(_xyzw_to_wxyz(result.as_quat()))


def _angle_error(target: np.ndarray, actual: np.ndarray) -> float:
    targ = Rotation.from_quat(_wxyz_to_xyzw(target))
    act = Rotation.from_quat(_wxyz_to_xyzw(actual))
    delta = act.inv() * targ
    return float(delta.magnitude())


def _linspace(min_val: float, max_val: float, count: int) -> Iterable[float]:
    if count < 2:
        return [float(min_val)]
    return np.linspace(min_val, max_val, count)


def _transform_wrench_to_ee(wrench: np.ndarray, arm: str) -> np.ndarray:
    transformed = wrench.copy()
    if arm == "left":
        left_wrench_copy = transformed.copy()
        transformed[0] = left_wrench_copy[1]
        transformed[1] = left_wrench_copy[0]
        transformed[2] = -left_wrench_copy[2]
        transformed[3] = left_wrench_copy[4]
        transformed[4] = left_wrench_copy[3]
        transformed[5] = -left_wrench_copy[5]
    else:
        right_wrench_copy = transformed.copy()
        transformed[0] = -right_wrench_copy[1]
        transformed[1] = -right_wrench_copy[0]
        transformed[2] = -right_wrench_copy[2]
        transformed[3] = -right_wrench_copy[4]
        transformed[4] = -right_wrench_copy[3]
        transformed[5] = -right_wrench_copy[5]
    return transformed


@dataclass
class CalibrationConfig:
    arm: str
    tilt_x_min: float
    tilt_x_max: float
    tilt_x_count: int
    tilt_y_min: float
    tilt_y_max: float
    tilt_y_count: int
    tilt_z_min: float
    tilt_z_max: float
    tilt_z_count: int
    move_duration: float
    settle_time: float
    sample_duration: float
    sample_rate_hz: float
    settle_tolerance_rad: float
    settle_timeout: float


class FTCalibrator:
    def __init__(self, config: CalibrationConfig, wbc: RBY1WBC):
        if config.arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        self.cfg = config
        self.wbc = wbc
        self._calib_arm_index = 0 if config.arm == "left" else 1

    def run(self) -> None:
        snapshot = self.wbc.wait_for_first_state(timeout_sec=10.0)
        ee_pose = self.wbc.get_end_effector_pose(snapshot)
        if ee_pose is None:
            raise RuntimeError("Unable to compute initial end-effector pose")
        left_pose, right_pose = ee_pose
        left_pos, left_quat = left_pose[:3].copy(), left_pose[3:].copy()
        right_pos, right_quat = right_pose[:3].copy(), right_pose[3:].copy()

        # Hold the current pose as the nominal setpoint.
        left_target_pos = left_pos.copy()
        left_target_quat = left_quat.copy()
        right_target_pos = right_pos.copy()
        right_target_quat = right_quat.copy()

        gripper_left, gripper_right = self.wbc.get_latest_gripper_widths()
        self.wbc.update_targets(
            left_target_pos,
            left_target_quat,
            right_target_pos,
            right_target_quat,
            left_width=gripper_left,
            right_width=gripper_right,
            duration=self.cfg.move_duration,
        )
        time.sleep(self.cfg.settle_time)

        initial_quat = _normalize(left_target_quat if self.cfg.arm == "left" else right_target_quat)

        tilt_x_vals = list(_linspace(self.cfg.tilt_x_min, self.cfg.tilt_x_max, self.cfg.tilt_x_count))
        tilt_y_vals = list(_linspace(self.cfg.tilt_y_min, self.cfg.tilt_y_max, self.cfg.tilt_y_count))
        tilt_z_vals = list(_linspace(self.cfg.tilt_z_min, self.cfg.tilt_z_max, self.cfg.tilt_z_count))

        quaternions: List[np.ndarray] = []
        wrenches: List[np.ndarray] = []

        print(f"Starting FT calibration on {self.cfg.arm} arm...")
        for tx in tilt_x_vals:
            for ty in tilt_y_vals:
                for tz in tilt_z_vals:
                    print(f"  -> Tilt X={tx:.1f} deg, Y={ty:.1f} deg, Z={tz:.1f} deg")
                    target_quat = _apply_global_tilts(initial_quat, tx, ty, tz)
                    if self.cfg.arm == "left":
                        left_target_quat = target_quat
                    else:
                        right_target_quat = target_quat

                    self.wbc.update_targets(
                        left_target_pos,
                        left_target_quat,
                        right_target_pos,
                        right_target_quat,
                        left_width=gripper_left,
                        right_width=gripper_right,
                        duration=self.cfg.move_duration,
                    )
                    self._wait_for_settle(target_quat)
                    wrench_avg = self._sample_wrench()
                    pose = self._capture_pose()
                    if pose is None:
                        raise RuntimeError("Failed to capture pose for current tilt")
                    quaternions.append(_normalize(pose[3:].copy()))
                    wrenches.append(wrench_avg)

        print("Returning to initial orientation...")
        if self.cfg.arm == "left":
            left_target_quat = initial_quat.copy()
        else:
            right_target_quat = initial_quat.copy()
        self.wbc.update_targets(
            left_target_pos,
            left_target_quat,
            right_target_pos,
            right_target_quat,
            left_width=gripper_left,
            right_width=gripper_right,
            duration=self.cfg.move_duration,
        )
        time.sleep(self.cfg.settle_time)

        print("Computing calibration...")
        result = self._solve_calibration(quaternions, wrenches)
        self._print_result(result)

    def _wait_for_settle(self, target_quat: np.ndarray) -> None:
        deadline = time.monotonic() + max(self.cfg.settle_timeout, 0.0)
        while time.monotonic() < deadline:
            pose = self._capture_pose()
            if pose is not None:
                quat = pose[3:]
                err = _angle_error(target_quat, quat)
                if err <= self.cfg.settle_tolerance_rad:
                    break
            time.sleep(0.05)
        time.sleep(self.cfg.settle_time)

    def _capture_pose(self) -> np.ndarray | None:
        snapshot = self.wbc.get_latest_robot_state()
        if snapshot is None or not snapshot.is_valid:
            return None
        poses = self.wbc.get_end_effector_pose(snapshot)
        if poses is None:
            return None
        return poses[self._calib_arm_index]

    def _sample_wrench(self) -> np.ndarray:
        samples: List[np.ndarray] = []
        dt = 1.0 / max(self.cfg.sample_rate_hz, 1.0)
        end_time = time.monotonic() + max(self.cfg.sample_duration, dt)
        while time.monotonic() < end_time:
            snapshot = self.wbc.get_latest_robot_state()
            if snapshot is not None and snapshot.is_valid:
                wrench = self._extract_wrench(snapshot)
                if wrench is not None:
                    samples.append(wrench)
            time.sleep(dt)
        if not samples:
            raise RuntimeError("No valid FT samples recorded for this tilt.")
        return np.mean(samples, axis=0)

    def _extract_wrench(self, snapshot: RobotSnapshot) -> np.ndarray | None:
        if self.cfg.arm == "left":
            if not snapshot.left_ft_valid:
                return None
            wrench = np.asarray(snapshot.left_ee_wrench, dtype=float)
            return _transform_wrench_to_ee(wrench, arm="left")
        if not snapshot.right_ft_valid:
            return None
        wrench = np.asarray(snapshot.right_ee_wrench, dtype=float)
        return _transform_wrench_to_ee(wrench, arm="right")

    def _solve_calibration(
        self, quaternions: List[np.ndarray], wrenches: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(quaternions) != len(wrenches):
            raise ValueError("Mismatched quaternion and wrench lists.")
        num_samples = len(quaternions)
        if num_samples < 4:
            raise ValueError("Need at least 4 samples to solve calibration.")

        A_force = np.zeros((num_samples * 3, 6), dtype=float)
        b_force = np.zeros((num_samples * 3,), dtype=float)
        for i, (quat, wrench) in enumerate(zip(quaternions, wrenches)):
            rot = Rotation.from_quat(_wxyz_to_xyzw(quat)).as_matrix()
            A_force[3 * i : 3 * i + 3, 0:3] = rot.T
            A_force[3 * i : 3 * i + 3, 3:6] = -np.eye(3)
            b_force[3 * i : 3 * i + 3] = wrench[:3]

        x_gf, *_ = np.linalg.lstsq(A_force, b_force, rcond=None)
        G = x_gf[:3]
        F = x_gf[3:]

        A_torque = np.zeros_like(A_force)
        b_torque = np.zeros_like(b_force)
        for i, wrench in enumerate(wrenches):
            F_hat = wrench[:3] + F
            cross = np.array(
                [
                    [0.0, F_hat[2], -F_hat[1]],
                    [-F_hat[2], 0.0, F_hat[0]],
                    [F_hat[1], -F_hat[0], 0.0],
                ],
                dtype=float,
            )
            A_torque[3 * i : 3 * i + 3, 0:3] = cross
            A_torque[3 * i : 3 * i + 3, 3:6] = -np.eye(3)
            b_torque[3 * i : 3 * i + 3] = wrench[3:]

        x_pt, *_ = np.linalg.lstsq(A_torque, b_torque, rcond=None)
        P = x_pt[:3]
        T = x_pt[3:]
        return G, F, P, T

    def _print_result(self, result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        G, F, P, T = result
        print("\nCalibration complete. Copy the following into your hardware config:\n")
        print("------------------- begin yaml -------------------")
        print("ftsensor:")
        print("  offset:")
        print(f"    fx: {F[0]:.6f}")
        print(f"    fy: {F[1]:.6f}")
        print(f"    fz: {F[2]:.6f}")
        print(f"    tx: {T[0]:.6f}")
        print(f"    ty: {T[1]:.6f}")
        print(f"    tz: {T[2]:.6f}")
        print("  gravity:")
        print(f"    x: {G[0]:.6f}")
        print(f"    y: {G[1]:.6f}")
        print(f"    z: {G[2]:.6f}")
        print("  COM:")
        print(f"    x: {P[0]:.6f}")
        print(f"    y: {P[1]:.6f}")
        print(f"    z: {P[2]:.6f}")
        print("------------------- end yaml ---------------------\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate FT sensor using the RBY1 WBC.")
    parser.add_argument("--wbc-config", default=str(Path(PROJECT_ROOT, "config/wbc.yaml")), help="Path to the WBC YAML config.")
    parser.add_argument("--arm", choices=("left", "right"), default="left", help="Which arm's FT sensor to calibrate.")
    parser.add_argument("--tilt-x-min", type=float, default=-45, help="Minimum X tilt in degrees.")
    parser.add_argument("--tilt-x-max", type=float, default=45.0, help="Maximum X tilt in degrees.")
    parser.add_argument("--tilt-x-count", type=int, default=3, help="Number of samples along the X tilt range.")
    parser.add_argument("--tilt-y-min", type=float, default=-45.0, help="Minimum Y tilt in degrees.")
    parser.add_argument("--tilt-y-max", type=float, default=45.0, help="Maximum Y tilt in degrees.")
    parser.add_argument("--tilt-y-count", type=int, default=3, help="Number of samples along the Y tilt range.")
    parser.add_argument("--tilt-z-min", type=float, default=-45.0, help="Minimum Z/yaw rotation in degrees.")
    parser.add_argument("--tilt-z-max", type=float, default=45.0, help="Maximum Z/yaw rotation in degrees.")
    parser.add_argument("--tilt-z-count", type=int, default=3, help="Number of samples along the Z rotation range.")
    parser.add_argument("--move-duration", type=float, default=10.0, help="Blend duration when updating IK targets.")
    parser.add_argument("--settle-time", type=float, default=0.5, help="Extra delay after settling before sampling (seconds).")
    parser.add_argument("--settle-tolerance-deg", type=float, default=2.0, help="Orientation error allowed before sampling.")
    parser.add_argument("--settle-timeout", type=float, default=7.0, help="Maximum wait for pose to settle (seconds).")
    parser.add_argument("--sample-duration", type=float, default=2.0, help="Duration to average wrench samples at each pose (seconds).")
    parser.add_argument("--sample-rate", type=float, default=250.0, help="Frequency to poll the FT sensor during sampling (Hz).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = CalibrationConfig(
        arm=args.arm,
        tilt_x_min=args.tilt_x_min,
        tilt_x_max=args.tilt_x_max,
        tilt_x_count=max(1, args.tilt_x_count),
        tilt_y_min=args.tilt_y_min,
        tilt_y_max=args.tilt_y_max,
        tilt_y_count=max(1, args.tilt_y_count),
        tilt_z_min=args.tilt_z_min,
        tilt_z_max=args.tilt_z_max,
        tilt_z_count=max(1, args.tilt_z_count),
        move_duration=max(0.1, args.move_duration),
        settle_time=max(0.0, args.settle_time),
        sample_duration=max(0.1, args.sample_duration),
        sample_rate_hz=max(1.0, args.sample_rate),
        settle_tolerance_rad=math.radians(max(0.1, args.settle_tolerance_deg)),
        settle_timeout=max(0.5, args.settle_timeout),
    )

    wbc = RBY1WBC(config_path=args.wbc_config)
    wbc.start()
    try:
        calibrator = FTCalibrator(cfg, wbc)
        calibrator.run()
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
    finally:
        wbc.stop()


if __name__ == "__main__":
    main()
