"""Shadow gate: release the nod only when nodding buys reach.

Replaces the hard gate's world-height door. Parameters come from the shadow
block of wbik_g1d_shadowgate.yaml. Every solve step, before the real robot:

* Two shadow robots (full WholeBodyIK copies with ``shadow_ee_lm_damping``)
  chase the hand positions extrapolated ``predictive_horizon`` ahead
  (velocity from the last ``predictive_velocity_frames`` targets, shift capped
  at ``predictive_max_shift``). The lock shadow has ``shadow_gate_joint``
  welded at 0; the free shadow may nod. The lock shadow stands for "the real
  robot right now, upright", so its reach must be one the real robot could
  make:
  - ``shadow_follow_real_chassis``: each step both shadows start from the
    real robot's chassis pose (their joints stay their own). Otherwise the
    shadow chassis drifts tens of cm and a shadow "reaches" by driving ahead.
  - ``shadow_nominal_posture_cost_arm``: arm nominal-posture cost for the
    shadows (missing = the robot's). The real robot's 50 is negligible next
    to ee_pos_cost; a shadow with it twists its shoulders to reach.
  - ``shadow_nominal_posture_cost_torso``: same for the torso group (lifts,
    nod, waist). Shadows keep their own joints between frames; at 50 the
    free shadow drifts to a deep nod with the lifts up while both reach.
* The nod is released when the free shadow beats the lock shadow by more
  than ``release_free_shadow_greater_than_lock_shadow``, and held again when
  that advantage drops below ``held_free_shadow_greater_than_lock_shadow``.
* While held, the real robot's gate joint must come back toward 0 by at
  least ``held_min_return_per_step`` per solve (straight to 0 when closer);
  it may come back faster when the hands pull it. At 0 it cannot nod forward.

Costs three QPs per step (real robot + two shadows).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from rby1.whole_body_ik import WholeBodyIK, _load_yaml_mapping

FLOAT_KEYS = (
    "shadow_ee_lm_damping",
    "release_free_shadow_greater_than_lock_shadow",
    "held_free_shadow_greater_than_lock_shadow",
    "held_min_return_per_step",
    "predictive_horizon",
    "predictive_max_shift",
)
INT_KEYS = ("predictive_velocity_frames",)
STR_KEYS = ("shadow_gate_joint",)


class ShadowGatedIK(WholeBodyIK):
    gate_name = "shadow"

    def __init__(self, config_path: str):
        cfg = _load_yaml_mapping(Path(config_path))
        missing = [k for k in FLOAT_KEYS + INT_KEYS + STR_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"{config_path}: shadow gate keys missing: {missing}")
        hard_keys = [k for k in cfg if k.startswith("yaw_upright_")]
        if hard_keys:
            raise ValueError(
                f"{config_path}: {hard_keys} are hard-gate keys (HardGatedIK); "
                "use wbik_g1d_shadowgate.yaml"
            )
        self.gate = {k: float(cfg[k]) for k in FLOAT_KEYS}
        self.gate.update({k: int(cfg[k]) for k in INT_KEYS})
        self.gate.update({k: str(cfg[k]) for k in STR_KEYS})
        self.shadow_follow_real_chassis = bool(cfg.get("shadow_follow_real_chassis", False))
        arm_cost = cfg.get("shadow_nominal_posture_cost_arm")
        self.shadow_nominal_posture_cost_arm = None if arm_cost is None else float(arm_cost)
        torso_cost = cfg.get("shadow_nominal_posture_cost_torso")
        self.shadow_nominal_posture_cost_torso = None if torso_cost is None else float(torso_cost)

        super().__init__(config_path)
        joint = self.gate["shadow_gate_joint"]
        qpos, dofs = self._resolve_joint_qpos_dofs([joint])
        self._gate_q, self._gate_d = qpos[0], dofs[0]
        self._gate_max_step: float | None = None
        self._chassis_idx = np.array(list(self.base_qpos_indices) + list(self.base_quat_indices))

        self._locked = self._make_shadow(config_path, lock=True)
        self._free = self._make_shadow(config_path, lock=False)
        self._shadow_seed = self._locked._get_nominal_posture(
            self._locked.data.qpos.copy()
        )
        self.reset_gate()

    def gate_params(self) -> dict:
        return dict(
            self.gate,
            shadow_follow_real_chassis=self.shadow_follow_real_chassis,
            shadow_nominal_posture_cost_arm=self.shadow_nominal_posture_cost_arm,
            shadow_nominal_posture_cost_torso=self.shadow_nominal_posture_cost_torso,
        )

    def _make_shadow(self, config_path: str, lock: bool) -> WholeBodyIK:
        sh = WholeBodyIK(config_path)
        sh.ee_lm_damping = self.gate["shadow_ee_lm_damping"]
        if self.shadow_nominal_posture_cost_arm is not None:
            for d in list(sh.left_arm_dof_indices) + list(sh.right_arm_dof_indices):
                sh.nominal_posture_cost_vector[d] = self.shadow_nominal_posture_cost_arm
        if self.shadow_nominal_posture_cost_torso is not None:
            for d in sh.torso_dof_indices:
                sh.nominal_posture_cost_vector[d] = self.shadow_nominal_posture_cost_torso
            if sh.nominal_posture_cost_waist is not None:
                d = sh._resolve_joint_qpos_dofs(["torso_Joint"])[1][0]
                sh.nominal_posture_cost_vector[d] = sh.nominal_posture_cost_waist
        sh._build_reusable_tasks()
        if lock:
            joint = self.gate["shadow_gate_joint"]
            sh.joint_locks = dict(sh.joint_locks)
            sh.joint_locks[joint] = 0.0
            sh._joint_qposadr[joint] = self._gate_q
            sh._joint_dofadr[joint] = self._gate_d
        return sh

    def reset_gate(self) -> None:
        self._q_locked = self._shadow_seed.copy()
        self._q_free = self._shadow_seed.copy()
        self._pos_hist: list[tuple[np.ndarray, np.ndarray]] = []
        self._held = True
        self._step = 0
        self.gate_info: dict = {}

    def _add_extra_inequalities(self, problem) -> None:
        if self._gate_max_step is None:
            return
        row = np.zeros((1, self.model.nv), dtype=float)
        row[0, self._gate_d] = 1.0
        if problem.G is None:
            problem.G, problem.h = row, np.array([self._gate_max_step])
        else:
            problem.G = np.vstack([problem.G, row])
            problem.h = np.hstack([problem.h, [self._gate_max_step]])

    def _predict(self, lp, rp, dt: float):
        self._pos_hist.append((np.asarray(lp, float).copy(), np.asarray(rp, float).copy()))
        m = min(self.gate["predictive_velocity_frames"], len(self._pos_hist) - 1)
        if m <= 0:
            return np.asarray(lp, float), np.asarray(rp, float)
        lp0, rp0 = self._pos_hist[-1 - m]
        cap = self.gate["predictive_max_shift"]
        horizon = self.gate["predictive_horizon"]

        def ahead(p_now, p_old):
            shift = (p_now - p_old) / (m * dt) * horizon
            norm = float(np.linalg.norm(shift))
            if norm > cap:
                shift *= cap / norm
            return p_now + shift

        return ahead(self._pos_hist[-1][0], lp0), ahead(self._pos_hist[-1][1], rp0)

    @staticmethod
    def _hands(ik: WholeBodyIK, q) -> list[np.ndarray]:
        ik.data.qpos[:] = q
        mujoco.mj_forward(ik.model, ik.data)
        out = []
        for site in (ik.left_ee_name, ik.right_ee_name):
            sid = mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_SITE, site)
            out.append(ik.data.site_xpos[sid].copy())
        return out

    @staticmethod
    def _hand_err(hands: list[np.ndarray], lp, rp) -> float:
        errs = [float(np.linalg.norm(h - np.asarray(p)))
                for h, p in zip(hands, (lp, rp)) if p is not None]
        return max(errs) if errs else 0.0

    def _update_gate(self, lp, lq, rp, rq_, dt: float) -> None:
        g = self.gate
        if lp is not None and rp is not None:
            lp_p, rp_p = self._predict(lp, rp, dt)
        else:
            lp_p, rp_p = lp, rp
        kw = dict(left_target_pos=lp_p, left_target_quat=lq,
                  right_target_pos=rp_p, right_target_quat=rq_, dt=dt)
        q_new, _v, ok, _ = self._locked.solve(current_qpos=self._q_locked, **kw)
        if ok:
            self._q_locked = q_new
        q_new, _v, ok, _ = self._free.solve(current_qpos=self._q_free, **kw)
        if ok:
            self._q_free = q_new
        hands_lock = self._hands(self._locked, self._q_locked)
        hands_free = self._hands(self._free, self._q_free)
        e_lock = self._hand_err(hands_lock, lp_p, rp_p)
        e_free = self._hand_err(hands_free, lp_p, rp_p)
        adv = e_lock - e_free

        event = None
        if self._held and adv > g["release_free_shadow_greater_than_lock_shadow"]:
            self._held, event = False, "release"
        elif not self._held and adv < g["held_free_shadow_greater_than_lock_shadow"]:
            self._held, event = True, "hold"
        self.gate_info = dict(
            step=self._step, held=self._held, event=event,
            e_lock=e_lock, e_free=e_free, adv=adv,
            pred_l=None if lp_p is None else np.asarray(lp_p, float).copy(),
            pred_r=None if rp_p is None else np.asarray(rp_p, float).copy(),
            q_locked=self._q_locked.copy(), q_free=self._q_free.copy(),
            hands_locked=hands_lock, hands_free=hands_free,
        )

    def solve(self, left_target_pos=None, left_target_quat=None,
              right_target_pos=None, right_target_quat=None,
              head_target_pos=None, head_target_quat=None,
              current_qpos=None, dt: float = 1e-3):
        q_cur = self.data.qpos.copy() if current_qpos is None else np.asarray(current_qpos)
        if self.shadow_follow_real_chassis:
            self._q_locked[self._chassis_idx] = q_cur[self._chassis_idx]
            self._q_free[self._chassis_idx] = q_cur[self._chassis_idx]
        self._update_gate(left_target_pos, left_target_quat,
                          right_target_pos, right_target_quat, float(dt))
        if self._held:
            q = float(q_cur[self._gate_q])
            self._gate_max_step = -min(q, self.gate["held_min_return_per_step"])
        else:
            self._gate_max_step = None
        out = super().solve(
            left_target_pos=left_target_pos, left_target_quat=left_target_quat,
            right_target_pos=right_target_pos, right_target_quat=right_target_quat,
            head_target_pos=head_target_pos, head_target_quat=head_target_quat,
            current_qpos=current_qpos, dt=dt,
        )
        self.gate_info["yaw_deg"] = float(np.degrees(out[0][self._gate_q]))
        self._step += 1
        return out
