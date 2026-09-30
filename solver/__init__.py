"""G1-D WholeBodyIK variants that differ only in how Yaw_Joint (the nod) is gated.

Both classes subclass rby1.whole_body_ik.WholeBodyIK, which carries no gate
logic of its own; each gate adds its term through the ``_extra_tasks`` /
``_add_extra_inequalities`` hooks. Both keep the ``solve()`` signature, and
after every solve expose ``gate_info`` with at least ``step``, ``held`` (nod
held toward 0), ``event`` ("release" / "hold" / None) and ``yaw_deg``.
``reset_gate()`` restarts the gate state.

Import with the rby1-wbc root on ``sys.path`` (same as ``rby1``)::

    from solver import GATES, gate_config_path
    cls, _ = GATES["shadow"]
    ik = cls(gate_config_path("shadow"))
"""

from __future__ import annotations

from pathlib import Path

from .hard_gate_ik import HardGatedIK
from .shadow_gate_ik import ShadowGatedIK

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# gate name -> (class, yaml path relative to PROJECT_ROOT)
GATES = {
    "hard": (HardGatedIK, "config/wbik_g1d_hardgate.yaml"),
    "shadow": (ShadowGatedIK, "config/wbik_g1d_shadowgate.yaml"),
}


def gate_config_path(gate: str) -> str:
    return str(PROJECT_ROOT / GATES[gate][1])


__all__ = ["GATES", "HardGatedIK", "PROJECT_ROOT", "ShadowGatedIK", "gate_config_path"]
