"""Isaac Lab's pretrained Unitree G1 flat-terrain walking policy, as plain numpy.

The checkpoint is ``Isaac-Velocity-Flat-G1-v0`` (rsl_rl, md5 a4ca1e234d2b107f8a3b382c130651dc),
the file NVIDIA ships unchanged on its CDN under the 4.5, 5.0, 5.1 and 6.0 trees. Its training
environment is readable line by line in the local Isaac Lab checkout, and the G1 MDP config is
identical between Isaac Lab v2.0.0 (when the checkpoint was trained) and the local 2.2.1, apart
from two renamed actuator keys -- see policy_notes.md next to the spec for the file:line audit.

WHY NUMPY. Kit 6.0.1's Python has no torch (``python.sh -c "import torch"`` ->
ModuleNotFoundError); torch only exists inside the deprecated ``omni.isaac.ml_archive``
extension. The actor is a 123-256-128-128-37 ELU MLP: 4 matmuls at 50 Hz, which numpy does in
microseconds, and it removes the dependency outright. ``nav/tools/g1_export_policy.py`` checks
this implementation against torch to 1e-5.

THE CONTRACT this module enforces (everything else is the caller's job):
  * obs = [base_lin_vel_b(3), base_ang_vel_b(3), projected_gravity_b(3), command(3),
           q - q_default (37), dq (37), last_action (37)]           -> 123, no scaling, no clip
  * target_q = q_default + 0.5 * action                             (no clip)
  * the 37 joints are in the POLICY's order (``spec["policy_joint_names"]``). The simulator's
    DOF order is matched by NAME in ``bind()``, never assumed.
  * velocities are the root body's COM velocity rotated into the root (pelvis) frame; gravity is
    the unit vector (0, 0, -1) rotated into the same frame.
  * call ``step()`` once per 4 physics steps of 5 ms (50 Hz); hold the PD targets in between.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def quat_rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)^T v for a unit quaternion (w, x, y, z) -- Isaac Lab's ``quat_apply_inverse``."""
    w, x, y, z = (float(c) for c in quat_wxyz)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return r.T @ np.asarray(v, dtype=np.float64)


def base_frame_state(quat_wxyz, lin_vel_w, ang_vel_w):
    """(lin_vel_b, ang_vel_b, projected_gravity_b) from a root pose and root (COM) velocity."""
    lin_b = quat_rotate_inverse(quat_wxyz, lin_vel_w)
    ang_b = quat_rotate_inverse(quat_wxyz, ang_vel_w)
    grav_b = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))
    return lin_b, ang_b, grav_b


class G1WalkPolicy:
    """Deterministic actor (the Gaussian mean; rsl_rl ``act_inference``) plus its obs/action glue."""

    def __init__(self, spec_path: str | Path, weights_path: str | Path | None = None):
        spec_path = Path(spec_path)
        self.spec = json.loads(spec_path.read_text())
        net = self.spec["network"]
        if weights_path is None:
            weights_path = spec_path.parent / net["weights_npz"]
        w = np.load(weights_path)
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(net["layer_dims"]) - 1):
            W = w[f"W{i}"].astype(np.float32)
            b = w[f"b{i}"].astype(np.float32)
            self.layers.append((W, b))
        if net["activation"] != "elu":
            raise ValueError(f"only elu is implemented, spec says {net['activation']!r}")

        self.joint_names: list[str] = list(self.spec["policy_joint_names"])
        self.num_actions = len(self.joint_names)
        self.obs_dim = int(self.spec["observation"]["dim"])
        if self.layers[0][0].shape[1] != self.obs_dim or self.layers[-1][0].shape[0] != self.num_actions:
            raise ValueError(f"weights {self.layers[0][0].shape[1]}->{self.layers[-1][0].shape[0]} do not "
                             f"match spec obs {self.obs_dim} / actions {self.num_actions}")
        self.default_q = np.asarray(self.spec["default_joint_pos"], dtype=np.float64)
        # Training's reset pose: default clamped to the 0.9 soft limits (moves left/right_four/six to -/+0.092).
        # Use it for set_joint_positions at reset; the observation still subtracts default_q.
        self.reset_q = np.asarray(self.spec.get("reset_joint_pos", self.spec["default_joint_pos"]), dtype=np.float64)
        self.action_scale = float(self.spec["action"]["scale"])
        self.last_action = np.zeros(self.num_actions, dtype=np.float64)
        self._sim_from_policy: np.ndarray | None = None   # sim_idx[k] = sim DOF of policy joint k

    # ------------------------------------------------------------------ joint order
    def bind(self, sim_dof_names: list[str]) -> np.ndarray:
        """Match the simulator's DOF names to the policy's, by name. Returns sim indices in
        policy order. Raises if any policy joint is missing (a 29-DOF USD will not bind)."""
        index = {n: i for i, n in enumerate(sim_dof_names)}
        missing = [n for n in self.joint_names if n not in index]
        if missing:
            raise KeyError(f"{len(missing)} policy joints missing from the articulation: {missing[:6]}...")
        self._sim_from_policy = np.array([index[n] for n in self.joint_names], dtype=np.int64)
        return self._sim_from_policy

    def to_policy_order(self, sim_values) -> np.ndarray:
        return np.asarray(sim_values, dtype=np.float64)[self._sim_from_policy]

    def to_sim_order(self, policy_values, num_sim_dofs: int, fill=None) -> np.ndarray:
        out = np.zeros(num_sim_dofs) if fill is None else np.array(fill, dtype=np.float64, copy=True)
        out[self._sim_from_policy] = policy_values
        return out

    # ------------------------------------------------------------------ the network
    def forward(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < last:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU(alpha=1)
        return x.astype(np.float64)

    # ------------------------------------------------------------------ obs / action glue
    def reset(self) -> None:
        self.last_action[:] = 0.0

    def observation(self, lin_vel_b, ang_vel_b, gravity_b, command, q_policy, dq_policy) -> np.ndarray:
        obs = np.concatenate([
            np.asarray(lin_vel_b, dtype=np.float64),
            np.asarray(ang_vel_b, dtype=np.float64),
            np.asarray(gravity_b, dtype=np.float64),
            np.asarray(command, dtype=np.float64),
            np.asarray(q_policy, dtype=np.float64) - self.default_q,
            np.asarray(dq_policy, dtype=np.float64),          # default joint vel is 0
            self.last_action,
        ])
        assert obs.shape == (self.obs_dim,), obs.shape
        return obs

    def step(self, lin_vel_b, ang_vel_b, gravity_b, command, q_sim, dq_sim):
        """One 50 Hz policy tick. Inputs in SIM DOF order; returns (target_q_sim_order_for_bound
        joints in POLICY order, raw action). Use ``to_sim_order`` to scatter the targets."""
        if self._sim_from_policy is None:
            raise RuntimeError("call bind(sim_dof_names) first")
        q = self.to_policy_order(q_sim)
        dq = self.to_policy_order(dq_sim)
        obs = self.observation(lin_vel_b, ang_vel_b, gravity_b, command, q, dq)
        action = self.forward(obs)
        self.last_action = action
        target_q = self.default_q + self.action_scale * action
        return target_q, action, obs
