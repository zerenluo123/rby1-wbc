"""Hard gate: yaw upright spring switched by the lower hand target's height.

This is the original G1-D scheme, driven by the ``yaw_upright_*`` keys of
wbik_g1d_hardgate.yaml:

* a spring pulls ``yaw_upright_joints`` (one joint, default Yaw_Joint) back
  to its nominal angle;
* spring weight ``yaw_upright_cost`` while the lower of the two hand targets
  is at or above ``yaw_upright_z_ref`` (world z), ``yaw_upright_cost_min``
  below it (``yaw_upright_gate_eps`` > 0 blends linearly just below);
* the spring's angle error is clipped to ``yaw_upright_error_clip`` per solve.
"""

from __future__ import annotations

from pathlib import Path

import mink
import numpy as np

from rby1.whole_body_ik import WholeBodyIK, _load_yaml_mapping


class JointUprightTask(mink.Task):
    """Scalar spring ``||w (Δq_i + (q_i - q*))||²`` on one hinge.

    Same Mink residual as other tasks: ``J=1``, ``e = q_i - q*``.
    """

    def __init__(
        self,
        dof_index: int,
        qpos_index: int,
        q_star: float = 0.0,
        cost: float = 0.0,
        error_clip: float | None = None,
    ):
        super().__init__(cost=np.array([float(cost)]), gain=1.0, lm_damping=0.0)
        self.dof_index = int(dof_index)
        self.qpos_index = int(qpos_index)
        self.q_star = float(q_star)
        # None = full angle error. A positive cap limits how far one solve
        # may pull the joint, so a door opening cannot spend the velocity box.
        self.error_clip = None if error_clip is None else float(error_clip)

    def set_cost(self, cost: float) -> None:
        self.cost[0] = max(0.0, float(cost))

    def compute_error(self, configuration) -> np.ndarray:
        error = float(configuration.q[self.qpos_index]) - self.q_star
        clip = self.error_clip
        if clip is not None and clip > 0.0:
            error = float(np.clip(error, -clip, clip))
        return np.array([error])

    def compute_jacobian(self, configuration) -> np.ndarray:
        jacobian = np.zeros((1, configuration.nv))
        jacobian[0, self.dof_index] = 1.0
        return jacobian


class HardGatedIK(WholeBodyIK):
    gate_name = "hard"

    def __init__(self, config_path: str):
        cfg = _load_yaml_mapping(Path(config_path))
        self.yaw_upright_cost = float(cfg.get("yaw_upright_cost", 0.0))
        if self.yaw_upright_cost <= 0.0 or "yaw_upright_z_ref" not in cfg:
            raise ValueError(
                f"{config_path}: the hard gate needs yaw_upright_cost > 0 and "
                "yaw_upright_z_ref (see wbik_g1d_hardgate.yaml)"
            )
        self.yaw_upright_cost_min = float(cfg.get("yaw_upright_cost_min", 0.0))
        self.yaw_upright_gate_eps = float(cfg.get("yaw_upright_gate_eps", 0.0))
        self.yaw_upright_z_ref = float(cfg["yaw_upright_z_ref"])
        clip = cfg.get("yaw_upright_error_clip")
        self.yaw_upright_error_clip = None if clip is None else float(clip)
        if self.yaw_upright_error_clip is not None and self.yaw_upright_error_clip < 0.0:
            raise ValueError("yaw_upright_error_clip must be >= 0")
        self.yaw_upright_joints = [
            str(name) for name in (cfg.get("yaw_upright_joints") or ["Yaw_Joint"])
        ]
        if len(self.yaw_upright_joints) != 1:
            raise ValueError(
                f"yaw_upright_joints must name exactly one joint, got {self.yaw_upright_joints}"
            )

        super().__init__(config_path)
        name = self.yaw_upright_joints[0]
        qpos, dofs = self._resolve_joint_qpos_dofs([name])
        self._gate_q = qpos[0]
        q_star = (
            float(self.nominal_torso_angles[self.torso_joint_names.index(name)])
            if name in self.torso_joint_names else 0.0
        )
        self._yaw_upright_task = JointUprightTask(
            dof_index=dofs[0],
            qpos_index=qpos[0],
            q_star=q_star,
            cost=self.yaw_upright_cost,
            error_clip=self.yaw_upright_error_clip,
        )
        self.reset_gate()

    def gate_params(self) -> dict:
        return dict(
            yaw_upright_joints=self.yaw_upright_joints,
            yaw_upright_cost=self.yaw_upright_cost,
            yaw_upright_cost_min=self.yaw_upright_cost_min,
            yaw_upright_z_ref=self.yaw_upright_z_ref,
            yaw_upright_gate_eps=self.yaw_upright_gate_eps,
            yaw_upright_error_clip=self.yaw_upright_error_clip,
        )

    def reset_gate(self) -> None:
        self._step = 0
        self._held: bool | None = None
        self.gate_info: dict = {}

    def _ee_target_z_min(self) -> float:
        zs = []
        for task in (self._left_ee_task, self._right_ee_task):
            target = getattr(task, "transform_target_to_world", None)
            if target is None:
                continue
            zs.append(float(target.wxyz_xyz[6]))
        return min(zs) if zs else float("inf")

    def _yaw_upright_weight(self) -> float:
        """Full at/above z_ref, ``cost_min`` below. Optional linear door."""
        w_max = self.yaw_upright_cost
        w_min = self.yaw_upright_cost_min
        z = self._ee_target_z_min()
        z_ref = self.yaw_upright_z_ref
        eps = self.yaw_upright_gate_eps
        if eps <= 0.0:
            return w_max if z >= z_ref else w_min
        t = (z - (z_ref - eps)) / eps
        t = min(1.0, max(0.0, t))
        return w_min + (w_max - w_min) * t

    def _extra_tasks(self, configuration) -> list:
        self._yaw_upright_task.set_cost(self._yaw_upright_weight())
        return [self._yaw_upright_task]

    def solve(self, *args, **kwargs):
        out = super().solve(*args, **kwargs)
        weight = float(self._yaw_upright_task.cost[0])
        held = weight > 0.0
        event = None
        if self._held is not None and held != self._held:
            event = "hold" if held else "release"
        self._held = held
        self.gate_info = dict(
            step=self._step, held=held, event=event,
            yaw_deg=float(np.degrees(out[0][self._gate_q])),
            spring_weight=weight,
            target_z_min=float(self._ee_target_z_min()),
            z_ref=float(self.yaw_upright_z_ref),
        )
        self._step += 1
        return out
