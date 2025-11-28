from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np


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


@dataclass
class EETargets:
    """Thread-safe shared end-effector targets and current qpos snapshot for IK."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    duration: float = 0.0
    timestamp: float = 0.0

    left_pos_start: Optional[np.ndarray] = None
    left_quat_start: Optional[np.ndarray] = None
    left_width_start: Optional[float] = None
    left_pos: Optional[np.ndarray] = None
    left_quat: Optional[np.ndarray] = None
    left_width: Optional[float] = None

    right_pos_start: Optional[np.ndarray] = None
    right_quat_start: Optional[np.ndarray] = None
    right_width_start: Optional[float] = None
    right_pos: Optional[np.ndarray] = None
    right_quat: Optional[np.ndarray] = None
    right_width: Optional[float] = None

    head_pos_start: Optional[np.ndarray] = None
    head_quat_start: Optional[np.ndarray] = None
    head_pos: Optional[np.ndarray] = None
    head_quat: Optional[np.ndarray] = None

    def set_targets(
        self,
        duration: float,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
        left_width: Optional[float] = None,
        right_width: Optional[float] = None,
        head_pos: Optional[np.ndarray] = None,
        head_quat: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        with self.lock:
            now = time.monotonic() if timestamp is None else float(timestamp)

            # Store previous targets for interpolation
            prev_left_pos = self.left_pos.copy() if self.left_pos is not None else None
            prev_left_quat = self.left_quat.copy() if self.left_quat is not None else None
            prev_left_width = self.left_width

            prev_right_pos = self.right_pos.copy() if self.right_pos is not None else None
            prev_right_quat = self.right_quat.copy() if self.right_quat is not None else None
            prev_right_width = self.right_width

            prev_head_pos = self.head_pos.copy() if self.head_pos is not None else None
            prev_head_quat = self.head_quat.copy() if self.head_quat is not None else None

            self.left_pos_start = prev_left_pos if prev_left_pos is not None else left_pos.copy()
            self.left_quat_start = prev_left_quat if prev_left_quat is not None else left_quat.copy()
            self.left_width_start = prev_left_width if prev_left_width is not None else (None if left_width is None else float(left_width))

            self.right_pos_start = prev_right_pos if prev_right_pos is not None else right_pos.copy()
            self.right_quat_start = prev_right_quat if prev_right_quat is not None else right_quat.copy()
            self.right_width_start = prev_right_width if prev_right_width is not None else (None if right_width is None else float(right_width))

            self.head_pos_start = prev_head_pos if prev_head_pos is not None else (None if head_pos is None else head_pos.copy())
            self.head_quat_start = prev_head_quat if prev_head_quat is not None else (None if head_quat is None else head_quat.copy())

            # Set new targets
            self.left_pos = left_pos.copy()
            self.left_quat = left_quat.copy()
            self.right_pos = right_pos.copy()
            self.right_quat = right_quat.copy()
            self.left_width = None if left_width is None else float(left_width)
            self.right_width = None if right_width is None else float(right_width)
            self.head_pos = None if head_pos is None else head_pos.copy()
            self.head_quat = None if head_quat is None else head_quat.copy()

            self.duration = max(0.0, float(duration))
            self.timestamp = now

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
            elapsed = max(0.0, current_time - self.timestamp)
            alpha = min(1.0, elapsed / duration) if duration > 0.0 else 1.0

            lt_p = _lerp_value(self.left_pos_start, self.left_pos, alpha)
            lt_q = _slerp_quaternion(self.left_quat_start, self.left_quat, alpha)
            lw = _lerp_value(self.left_width_start, self.left_width, alpha)

            rt_p = _lerp_value(self.right_pos_start, self.right_pos, alpha)
            rt_q = _slerp_quaternion(self.right_quat_start, self.right_quat, alpha)
            rw = _lerp_value(self.right_width_start, self.right_width, alpha)

            hp = _lerp_value(self.head_pos_start, self.head_pos, alpha)
            hq = _slerp_quaternion(self.head_quat_start, self.head_quat, alpha)

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
            lt_p = None if self.left_pos is None else self.left_pos.copy()
            lt_q = None if self.left_quat is None else self.left_quat.copy()
            lw = self.left_width

            rt_p = None if self.right_pos is None else self.right_pos.copy()
            rt_q = None if self.right_quat is None else self.right_quat.copy()
            rw = self.right_width

            hp = None if self.head_pos is None else self.head_pos.copy()
            hq = None if self.head_quat is None else self.head_quat.copy()

        return lt_p, lt_q, lw, rt_p, rt_q, rw, hp, hq


__all__ = ["EETargets"]
