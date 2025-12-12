from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TeleopTargets:
    left_pos: Optional[np.ndarray] = None
    left_quat: Optional[np.ndarray] = None
    right_pos: Optional[np.ndarray] = None
    right_quat: Optional[np.ndarray] = None
    left_width: Optional[float] = None
    right_width: Optional[float] = None
    head_pos: Optional[np.ndarray] = None
    head_quat: Optional[np.ndarray] = None
