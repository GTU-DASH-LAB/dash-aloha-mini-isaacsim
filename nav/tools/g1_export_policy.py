"""Export Isaac Lab's pretrained G1 flat walking actor: TorchScript + numpy weights + JSON spec,
and prove the numpy re-implementation (nav/sim/g1/walk_policy.py) matches torch.

Runs in a torch env, NOT in Kit (Kit 6.0.1 has no torch):

    ~/envs/qvla/bin/python nav/tools/g1_export_policy.py \
        --ckpt <scratch>/g1/policy/v22/checkpoint.pt --probe <scratch>/g1/policy/probe_dofs.json \
        --outdir <scratch>/g1/policy --spec <scratch>/g1/policy_spec.json

The joint ORDER comes from --probe (a Kit run of nav/tools/g1_policy_probe.py on the same USD),
because the policy's order is PhysX's DOF order and nothing outside PhysX can produce it. Every
per-joint number is resolved here from the Isaac Lab config tables below with re.fullmatch, the
way Isaac Lab's string_utils.resolve_matching_names_values does it, and each table cites its
source line in IsaacLab-2.2.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]   # profile.py shadow
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ISAACLAB = "IsaacLab-2.2.1/source"
UNITREE = f"{ISAACLAB}/isaaclab_assets/isaaclab_assets/robots/unitree.py"
VELENV = f"{ISAACLAB}/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
G1ROUGH = f"{ISAACLAB}/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py"
G1FLAT = f"{ISAACLAB}/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py"
G1PPO = f"{ISAACLAB}/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py"

# G1_CFG.init_state.joint_pos -- unitree.py:289-302 (unlisted joints default to 0.0)
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_pitch_joint": 0.87, "left_shoulder_roll_joint": 0.16, "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16, "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0, "right_one_joint": -1.0, "left_two_joint": 0.52, "right_two_joint": -0.52,
}
# G1_CFG.actuators -- unitree.py:306-373. velocity_limit: v2.0.0 (the checkpoint's version) set
# velocity_limit=100.0 on "legs" and "arms" and wrote it to PhysX (v2.0.0 articulation.py:1343-1344);
# 2.2.1 dropped it (USD value). "feet" never had one.
ACTUATORS = {
    "legs": {"expr": [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
             "effort_limit": 300.0, "velocity_limit": 100.0,
             "stiffness": {".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0, ".*_hip_pitch_joint": 200.0,
                           ".*_knee_joint": 200.0, "torso_joint": 200.0},
             "damping": {".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0, ".*_hip_pitch_joint": 5.0,
                         ".*_knee_joint": 5.0, "torso_joint": 5.0},
             "armature": {".*_hip_.*": 0.01, ".*_knee_joint": 0.01, "torso_joint": 0.01},
             "lines": "unitree.py:307-335"},
    "feet": {"expr": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
             "effort_limit": 20.0, "velocity_limit": None,
             "stiffness": 20.0, "damping": 2.0, "armature": 0.01, "lines": "unitree.py:336-342"},
    "arms": {"expr": [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                      ".*_elbow_pitch_joint", ".*_elbow_roll_joint", ".*_five_joint", ".*_three_joint",
                      ".*_six_joint", ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint"],
             "effort_limit": 300.0, "velocity_limit": 100.0, "stiffness": 40.0, "damping": 10.0,
             "armature": {".*_shoulder_.*": 0.01, ".*_elbow_.*": 0.01, ".*_five_joint": 0.001,
                          ".*_three_joint": 0.001, ".*_six_joint": 0.001, ".*_four_joint": 0.001,
                          ".*_zero_joint": 0.001, ".*_one_joint": 0.001, ".*_two_joint": 0.001},
             "lines": "unitree.py:343-372"},
}


def resolve(name: str, table, default=None):
    """Isaac Lab semantics: keys are regexes matched with re.fullmatch; a name matching two keys
    is an error there, so it is an error here."""
    if not isinstance(table, dict):
        return table
    hits = [k for k in table if re.fullmatch(k, name)]
    if len(hits) > 1:
        raise ValueError(f"{name} matches {hits}")
    return table[hits[0]] if hits else default


def group_of(name: str) -> str:
    hits = [g for g, a in ACTUATORS.items() if any(re.fullmatch(e, name) for e in a["expr"])]
    if len(hits) != 1:
        raise ValueError(f"{name} in actuator groups {hits}")
    return hits[0]


def build_actor(sd: dict, prefix: str) -> nn.Sequential:
    idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)})
    layers: list[nn.Module] = []
    for n, i in enumerate(idx):
        W = sd[f"{prefix}{i}.weight"]
        lin = nn.Linear(W.shape[1], W.shape[0])
        lin.weight.data.copy_(W)
        lin.bias.data.copy_(sd[f"{prefix}{i}.bias"])
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def load_actor(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:          # rsl_rl <= 2.x (Isaac Lab 2.x checkpoints)
        sd, prefix, std = ck["model_state_dict"], "actor.", ck["model_state_dict"]["std"]
    elif "actor_state_dict" in ck:        # rsl_rl 3.x+ layout (the 6.0 tree's root checkpoints)
        sd, prefix = ck["actor_state_dict"], "mlp."
        std = sd["distribution.std_param"]
    else:
        raise KeyError(f"unknown checkpoint layout: {list(ck)}")
    return build_actor(sd, prefix), std, ck.get("iter")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--compare-ckpt", type=Path, default=None,
                    help="another checkpoint to compare architecture/weights against (report only)")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    actor, std, it = load_actor(a.ckpt)
    lins = [m for m in actor if isinstance(m, nn.Linear)]
    dims = [lins[0].in_features] + [m.out_features for m in lins]
    print(f"actor dims {dims}  iter {it}  std mean {float(std.mean()):.4f}")

    probe = json.loads(a.probe.read_text())
    if probe.get("status") != "ok":
        raise SystemExit(f"probe status {probe.get('status')}")
    names = probe["dof_names_physx_order"]
    n = len(names)
    obs_dim = 12 + 3 * n
    if dims[0] != obs_dim or dims[-1] != n:
        raise SystemExit(f"checkpoint {dims[0]}->{dims[-1]} but USD has {n} DOFs (obs would be {obs_dim})")

    # per-joint tables, POLICY order (== PhysX order of this USD)
    default_q = [float(resolve(j, DEFAULT_JOINT_POS, 0.0)) for j in names]
    groups = [group_of(j) for j in names]
    stiff = [float(resolve(j, ACTUATORS[g]["stiffness"])) for j, g in zip(names, groups)]
    damp = [float(resolve(j, ACTUATORS[g]["damping"])) for j, g in zip(names, groups)]
    arm = [float(resolve(j, ACTUATORS[g]["armature"])) for j, g in zip(names, groups)]
    eff = [float(ACTUATORS[g]["effort_limit"]) for g in groups]
    vlim = [ACTUATORS[g]["velocity_limit"] for g in groups]
    if any(v is None for v in stiff + damp + arm):
        raise SystemExit("unresolved gain")

    # default q must be inside the USD limits (Isaac Lab clamps otherwise, articulation.py:697-701)
    lims = np.array(probe["dof_limits_rad"])
    out_of_range = [(j, q, lims[k].tolist()) for k, (j, q) in enumerate(zip(names, default_q))
                    if not (lims[k][0] <= q <= lims[k][1])]
    # Training RESET pose is not exactly default_q: reset_joints_by_scale (events.py:1166-1197) clamps
    # default * U(1,1) to the SOFT limits, mean +- 0.5*range*0.9 (articulation.py:1593-1599,
    # soft_joint_pos_limit_factor=0.9 at unitree.py:305). On g1_minimal that moves the four hand joints whose
    # default 0.0 sits ON a hard limit (left/right_four, left/right_six) to -/+0.092. Verified in the running
    # Isaac Lab env (nav/tools/g1_isaaclab_reference.py first_obs). The obs still subtracts default_q.
    soft_factor = 0.9
    mid, rng_half = lims.mean(axis=1), 0.5 * (lims[:, 1] - lims[:, 0]) * soft_factor
    reset_q = np.clip(np.array(default_q), mid - rng_half, mid + rng_half).tolist()

    # ---------------------------------------------------------------- exports
    weights = a.outdir / "policy_weights.npz"
    np.savez(weights, **{f"W{i}": m.weight.detach().numpy().astype(np.float32) for i, m in enumerate(lins)},
             **{f"b{i}": m.bias.detach().numpy().astype(np.float32) for i, m in enumerate(lins)})
    ts_path = a.outdir / "policy_actor.pt"
    scripted = torch.jit.script(actor)
    scripted.save(str(ts_path))

    def term(name, start, dim, src, note=""):
        return {"name": name, "slice": [start, start + dim], "dim": dim, "scale": 1.0, "clip": None,
                "source": src, **({"note": note} if note else {})}

    o = 0
    terms = []
    for name, dim, src, note in [
        ("base_lin_vel", 3, "observations.py:53-57 root_lin_vel_b = root COM lin vel in root-link frame "
                            "(articulation_data.py:1012-1014, 793-799)", "training noise U(-0.1,0.1)"),
        ("base_ang_vel", 3, "observations.py:63-67 root_ang_vel_b (articulation_data.py:1017-1019)",
         "training noise U(-0.2,0.2)"),
        ("projected_gravity", 3, "observations.py:73-77 projected_gravity_b = R^T (0,0,-1) "
                                 "(articulation_data.py:56-60, 759-761)", "training noise U(-0.05,0.05)"),
        ("velocity_commands", 3, "observations.py:672-674 -> (vx, vy, wz) base frame "
                                 "(velocity_command.py vel_command_b)", "no noise"),
        ("joint_pos_rel", n, "observations.py:209-216 q - q_default, policy joint order",
         "training noise U(-0.01,0.01)"),
        ("joint_vel_rel", n, "observations.py:254-261 dq - 0", "training noise U(-1.5,1.5)"),
        ("last_action", n, "observations.py:654-661 env.action_manager.action = RAW policy output "
                           "(before scale/offset)", "zeros after reset (joint_actions.py:178)"),
    ]:
        terms.append(term(name, o, dim, src, note))
        o += dim
    assert o == obs_dim

    spec = {
        "name": "isaaclab_g1_flat_walk",
        "checkpoint": {
            "url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/"
                   "IsaacLab/PretrainedCheckpoints/2.2.0/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt",
            "md5": hashlib.md5(a.ckpt.read_bytes()).hexdigest(),
            "same_file_also_at": ["4.5/.../rsl_rl/", "5.0|5.1|6.0/.../PretrainedCheckpoints/{2.0.0,2.2.0,2.3.2}/rsl_rl/"],
            "iter": it, "framework": "rsl_rl (model_state_dict layout, rsl_rl<=2.x)",
            "task": "Isaac-Velocity-Flat-G1-v0", "trained_with": "Isaac Lab v2.0.0 (Jan 2025); G1 MDP identical in 2.2.1",
        },
        "robot": {
            "usd_url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
                       "IsaacLab/Robots/Unitree/G1/g1_minimal.usd",
            "source": f"{G1ROUGH}:111 (G1_MINIMAL_CFG) -> {UNITREE}:378-379",
            "articulation_root_in_probe": probe["articulation_roots"],
            "spawn_pos": [0.0, 0.0, 0.74], "spawn_source": f"{UNITREE}:288",
            "articulation_props": {"enabled_self_collisions": False, "solver_position_iteration_count": 8,
                                   "solver_velocity_iteration_count": 4, "source": f"{UNITREE}:283-285"},
            "rigid_body_props": {"linear_damping": 0.0, "angular_damping": 0.0, "max_linear_velocity": 1000.0,
                                 "max_angular_velocity_deg_s": 1000.0, "max_depenetration_velocity": 1.0,
                                 "retain_accelerations": False, "source": f"{UNITREE}:274-282"},
            "num_rigid_bodies": probe.get("num_rigid_bodies"),
        },
        "sim": {
            "physics_dt": 0.005, "decimation": 4, "policy_dt": 0.02, "policy_hz": 50,
            "source": f"{VELENV}:308-311",
            "gravity": [0.0, 0.0, -9.81],
            "solver_type": "TGS (PhysxCfg.solver_type=1)", "training_device": "cuda:0 (GPU pipeline)",
            "ground_material": {"static_friction": 1.0, "dynamic_friction": 1.0, "restitution": 0.0,
                                "friction_combine": "multiply", "restitution_combine": "multiply",
                                "source": f"{VELENV}:50-55 + {G1FLAT}:19 (plane)"},
            "robot_material": {"static_friction": 0.8, "dynamic_friction": 0.6, "restitution": 0.0,
                               "source": f"{VELENV}:154-164 (startup, degenerate ranges)"},
        },
        "policy_joint_names": names,
        "policy_order_rule": "PhysX DOF order of the USD (Isaac Lab find_joints('.*', preserve_order=False), "
                             "articulation.py:142 dof_names; joint_actions.py:62-63). Match by NAME at runtime.",
        "default_joint_pos": default_q,
        "default_joint_pos_source": f"{UNITREE}:289-302 (others 0.0)",
        "default_joint_pos_out_of_usd_limits": out_of_range,
        "reset_joint_pos": reset_q,
        "reset_joint_pos_source": "events.py:1166-1197 reset_joints_by_scale (position_range (1,1), "
                                  f"{G1ROUGH}:117) clamps default to soft limits, factor 0.9 ({UNITREE}:305, "
                                  "articulation.py:1593-1599). Root reset: pos (0,0,0.74) + U(+-0.5) xy, yaw U(+-pi), "
                                  "zero velocity (rough_env_cfg.py:118-128).",
        "actuators_policy_order": {
            "model": "implicit PD (PhysX joint drive, force mode); gains in SI rad units for the tensor API. "
                     "If authored in USD DriveAPI instead: stiffness/damping * pi/180 (USD angular units are deg).",
            "group": groups, "stiffness": stiff, "damping": damp, "effort_limit": eff,
            "velocity_limit": vlim, "armature": arm, "friction": "USD value (cfg None; 0.0 on g1_minimal)",
            "velocity_limit_note": "100 rad/s = Isaac Lab v2.0.0 (the checkpoint's version) for legs+arms; None = keep "
                                   "the USD value (20-53 rad/s). Isaac Lab 2.2.1 keeps the USD value on every joint. "
                                   "Closed-loop walking is identical either way (Kit 100 vs Isaac Lab 2.2.1 USD limits).",
            "source": {g: ACTUATORS[g]["lines"] for g in ACTUATORS},
        },
        "action": {
            "dim": n, "term": "JointPositionAction joint_names=['.*'] (all 37 actuated by the policy)",
            "target_q": "default_joint_pos + scale * action", "scale": 0.5, "offset": "default_joint_pos",
            "clip": None, "source": f"{VELENV}:112; joint_actions.py:166-196; rsl_rl clip_actions=None (rl_cfg.py:156)",
        },
        "observation": {
            "dim": obs_dim, "terms": terms, "concatenate": True, "normalization": "none "
            "(empirical_normalization=False, " + f"{G1PPO}:17)",
            "noise": "training only (enable_corruption); do NOT add at deploy",
        },
        "commands": {
            "frame": "base (vel_command_b)", "order": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
            "trained_ranges": {"lin_vel_x": [0.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-1.0, 1.0]},
            "ranges_source": f"{G1FLAT}:39-41",
            "heading_command": True, "rel_heading_envs": 1.0, "heading_control_stiffness": 0.5,
            "heading_note": "in training ang_vel_z was ALWAYS clip(0.5*wrap(heading_target-heading), -1, 1) "
                            "(velocity_command.py:149-155, velocity_env_cfg.py:94-104); deploy feeds wz directly",
            "rel_standing_envs": 0.02, "resampling_time_s": 10.0,
        },
        "network": {
            "layer_dims": dims, "activation": "elu", "output": "Gaussian mean (act_inference); std ignored",
            "std_param": [round(float(s), 4) for s in std], "weights_npz": weights.name,
            "torchscript": ts_path.name, "source": f"{G1PPO}:18-23, 45-48; rsl_rl actor_critic.py:41-50",
        },
        "rough_variant": {
            "task": "Isaac-Velocity-Rough-G1-v0", "obs_dim": obs_dim + 187, "hidden": [512, 256, 128],
            "extra_obs": "height_scan appended LAST: 187 = 17 (x, -0.8..0.8) x 11 (y, -0.5..0.5) grid @0.1 m, "
                         "x fastest; value = sensor_z - hit_z - 0.5, clip [-1, 1]; sensor on torso_link, "
                         "+20 m z offset, yaw-aligned rays straight down",
            "source": f"{VELENV}:66-73,134-139; {G1ROUGH}:112; observations.py:289-297; patterns.py:44-52",
            "commands": "lin_vel_x [0,1], lin_vel_y 0, ang_vel_z [-1,1] (rough_env_cfg.py:147-149)",
        },
    }
    # Two copies: one next to the weights, one at --spec. walk_policy.G1WalkPolicy resolves
    # network.weights_npz / torchscript relative to the spec file, so each copy gets its own
    # relative paths.
    import copy
    import os
    a.spec.parent.mkdir(parents=True, exist_ok=True)
    for path in {a.spec.resolve(), (a.outdir / "policy_spec.json").resolve()}:
        s = copy.deepcopy(spec)
        s["network"]["weights_npz"] = os.path.relpath(weights.resolve(), path.parent)
        s["network"]["torchscript"] = os.path.relpath(ts_path.resolve(), path.parent)
        path.write_text(json.dumps(s, indent=1))
    local_spec = a.outdir / "policy_spec.json"

    # ---------------------------------------------------------------- checks
    from walk_policy import G1WalkPolicy
    pol = G1WalkPolicy(local_spec, weights)
    reloaded = torch.jit.load(str(ts_path))
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, size=(2000, obs_dim)).astype(np.float32)
    with torch.no_grad():
        ref = actor(torch.from_numpy(obs)).numpy()
        ts = reloaded(torch.from_numpy(obs)).numpy()
    mine = np.stack([pol.forward(x) for x in obs])
    d_np = float(np.abs(mine - ref).max())
    d_ts = float(np.abs(ts - ref).max())

    standing = np.zeros(obs_dim, dtype=np.float32)
    standing[8] = -1.0           # projected gravity of an upright base
    with torch.no_grad():
        a_stand = actor(torch.from_numpy(standing)).numpy()
    a_stand_np = pol.forward(standing)
    tgt = np.array(default_q) + 0.5 * a_stand_np
    zero = np.zeros(obs_dim, dtype=np.float32)
    with torch.no_grad():
        a_zero = actor(torch.from_numpy(zero)).numpy()

    report = {
        "ckpt": str(a.ckpt), "dims": dims, "iter": it,
        "numpy_vs_torch_maxabs_2000_random_obs": d_np,
        "torchscript_vs_torch_maxabs": d_ts,
        "standing_obs_action_absmax": float(np.abs(a_stand).max()),
        "standing_obs_action_rms": float(np.sqrt(np.mean(a_stand ** 2))),
        "standing_obs_numpy_vs_torch": float(np.abs(a_stand - a_stand_np).max()),
        "standing_obs_target_minus_default_absmax_rad": float(np.abs(tgt - np.array(default_q)).max()),
        "standing_action_by_joint": {j: round(float(v), 4) for j, v in zip(names, a_stand)},
        "all_zero_obs_action_absmax": float(np.abs(a_zero).max()),
        "default_q_out_of_limits": out_of_range,
        "sha256": {"policy_actor.pt": sha256(ts_path), "policy_weights.npz": sha256(weights)},
    }
    if a.compare_ckpt:
        other, ostd, oit = load_actor(a.compare_ckpt)
        olins = [m for m in other if isinstance(m, nn.Linear)]
        odims = [olins[0].in_features] + [m.out_features for m in olins]
        with torch.no_grad():
            oa = other(torch.from_numpy(standing)).numpy()
        report["compare"] = {"ckpt": str(a.compare_ckpt), "dims": odims, "iter": oit,
                             "same_arch": odims == dims,
                             "weights_identical": all(torch.equal(x.weight, y.weight) for x, y in zip(lins, olins))
                             if odims == dims else False,
                             "standing_action_absmax": float(np.abs(oa).max())}
    (a.outdir / "export_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "standing_action_by_joint"}, indent=1))
    ok = d_np < 1e-5 and d_ts < 1e-6
    print("PASS" if ok else "FAIL", f"numpy-vs-torch {d_np:.2e}, torchscript-vs-torch {d_ts:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
