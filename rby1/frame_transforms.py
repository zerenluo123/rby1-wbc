from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(rotation_rpy: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    """Build a homogeneous transform from roll-pitch-yaw and translation."""
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = Rotation.from_euler("xyz", np.asarray(rotation_rpy, dtype=float)).as_matrix()
    mat[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return mat


MODEL_TO_TCP_FRAME = {
    "left": make_transform([np.pi, 0.0, 0.0], [0.0, 0.0, -0.2]),
    "right": make_transform([0.0, np.pi, 0.0], [0.0, 0.0, -0.2]),
    "head": make_transform([-np.pi / 2.0, 0.0, -np.pi / 2.0], [0.04, 0.0, 0.0601]),
}
TCP_TO_MODEL_FRAME = {name: np.linalg.inv(mat) for name, mat in MODEL_TO_TCP_FRAME.items()}


def apply_transform_tf(tf: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a fixed transform to a pose matrix or position vector."""
    if tf.shape == (3,):
        tf_homogeneous = np.eye(4, dtype=float)
        tf_homogeneous[:3, 3] = tf
        return (transform @ tf_homogeneous)[:3, 3]
    return tf @ transform


__all__ = ["MODEL_TO_TCP_FRAME", "TCP_TO_MODEL_FRAME", "apply_transform_tf", "make_transform"]
