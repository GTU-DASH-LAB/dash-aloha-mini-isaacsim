"""Ground-truth run of the G1 flat walking policy INSIDE Isaac Lab 2.2.1 (Isaac Sim 5.0), to check
our plain-Kit 6.0.1 deployment (nav/sim/g1/walk_policy.py + nav/tools/g1_policy_probe.py) against
the environment the checkpoint was trained in.

Why this exists: everything in policy_spec.json was read off source files. This script asks the
running Isaac Lab env instead -- joint order, default pose, the gains Isaac Lab actually wrote into
PhysX, obs/action dims, the reset observation -- and then drives the robot through the SAME command
schedule as g1_policy_probe.py --walk, so pelvis height and speed tracking can be compared number
for number. --device cpu repeats it on CPU physics, which is what our Kit deploy uses (Kit 6.0.1 has
no torch, so it runs the numpy backend), separating "CPU vs GPU PhysX" from "our glue is wrong".

    ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-5.0.0/python.sh \
        nav/tools/g1_isaaclab_reference.py --out ref_gpu.json [--device cpu] [--policy policy.pt]

Runs under Isaac Sim 5.0's Python (3.11), where Isaac Lab 2.2.1 is installed. GPU0 only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]   # nav/tools/profile.py shadow

DEFAULT_POLICY = ("/home/gtu-dsa/robotics/IsaacLab-2.2.1/.pretrained_checkpoints/rsl_rl/"
                  "Isaac-Velocity-Flat-G1-v0/exported/policy.pt")   # play.py's own export, ckpt md5 a4ca1e23

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
ap.add_argument("--policy", default=DEFAULT_POLICY)
ap.add_argument("--seconds", type=float, default=16.0)
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

# GPU0 only, never multi-GPU (GPU1 belongs to the policy server). AppLauncher sets active_gpu and
# physics_gpu from the device id; multi_gpu passes straight through to SimulationApp.
app = AppLauncher({"headless": True, "device": "cuda:0" if args.device == "cpu" else args.device,
                   "multi_gpu": False}).app

result: dict = {"device": args.device, "policy": args.policy}
try:
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY

    cfg = G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # fixed, directly-set commands (same schedule as g1_policy_probe.py --walk)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 5.0)
    # deterministic spawn: no random xy/yaw offset
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

    env = ManagerBasedRLEnv(cfg)
    robot = env.scene["robot"]
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
    obs_dict, _ = env.reset()

    def t2l(x, nd=5):
        return [round(float(v), nd) for v in x.reshape(-1).tolist()]

    jt = env.action_manager.get_term("joint_pos")
    result["runtime"] = {
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "action_term_joint_names": list(jt._joint_names),
        "action_term_joint_ids": (list(range(robot.num_joints)) if isinstance(jt._joint_ids, slice)
                                  else [int(i) for i in jt._joint_ids]),
        "action_scale": float(jt._scale) if not torch.is_tensor(jt._scale) else t2l(jt._scale),
        "obs_dim_policy": list(env.observation_manager.group_obs_dim["policy"]),
        "obs_terms_policy": env.observation_manager.active_terms["policy"],
        "obs_term_dims_policy": [list(d) for d in env.observation_manager.group_obs_term_dim["policy"]],
        "action_dim": int(env.action_manager.total_action_dim),
        "sim_dt": float(env.sim.get_physics_dt()), "decimation": int(cfg.decimation), "step_dt": float(env.step_dt),
        "default_joint_pos": t2l(robot.data.default_joint_pos[0]),
        "joint_stiffness": t2l(robot.data.joint_stiffness[0]),
        "joint_damping": t2l(robot.data.joint_damping[0]),
        "joint_armature": t2l(robot.data.joint_armature[0]),
        "joint_effort_limits": t2l(robot.data.joint_effort_limits[0]),
        "joint_vel_limits": t2l(robot.data.joint_vel_limits[0]),
        "joint_pos_limits": [t2l(r) for r in robot.data.joint_pos_limits[0]],
        "physx_dof_names": list(robot.root_physx_view.shared_metatype.dof_names),
        "physx_max_velocities": t2l(robot.root_physx_view.get_dof_max_velocities()[0]),
        "physx_stiffnesses": t2l(robot.root_physx_view.get_dof_stiffnesses()[0]),
        "physx_dampings": t2l(robot.root_physx_view.get_dof_dampings()[0]),
        "physx_max_forces": t2l(robot.root_physx_view.get_dof_max_forces()[0]),
        "physx_armatures": t2l(robot.root_physx_view.get_dof_armatures()[0]),
        "physx_friction_coeffs": t2l(robot.root_physx_view.get_dof_friction_coefficients()[0]),
        "material_properties": t2l(robot.root_physx_view.get_material_properties()[0], 3),
        "num_shapes": int(robot.root_physx_view.max_shapes),
        "root_pos_after_reset": t2l(robot.data.root_pos_w[0]),
        "first_obs": t2l(obs_dict["policy"][0]),
    }
    print(f"[ref] joint order {result['runtime']['joint_names']}", flush=True)
    print(f"[ref] obs dim {result['runtime']['obs_dim_policy']} action dim {result['runtime']['action_dim']} "
          f"dt {result['runtime']['sim_dt']} decimation {cfg.decimation}", flush=True)

    cmd_term = env.command_manager.get_term("base_velocity")

    def command_at(t):
        if t < 2.0:
            return (0.0, 0.0, 0.0), "stand"
        if t < 8.0:
            return (0.5, 0.0, 0.0), "fwd0.5"
        if t < 11.0:
            return (0.8, 0.0, 0.0), "fwd0.8"
        if t < 14.0:
            return (0.3, 0.0, 0.6), "turn"
        return (0.0, 0.0, 0.0), "stand2"

    log = []
    steps = int(round(args.seconds / env.step_dt))
    resets = []
    t0 = time.time()
    first_action = None
    for k in range(steps):
        t = k * env.step_dt
        cmd, phase = command_at(t)
        cmd_term.vel_command_b[:] = torch.tensor(cmd, device=env.device)
        obs = env.observation_manager.compute()["policy"]   # re-read so this tick sees this command
        with torch.no_grad():
            act = policy(obs)
        if first_action is None:
            first_action = act[0].clone()
        _, _, term, trunc, _ = env.step(act)
        if bool(term[0]) or bool(trunc[0]):
            resets.append(round(t, 3))
        pos = robot.data.root_pos_w[0]
        q = robot.data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0] * q[3] + q[1] * q[2]), 1 - 2 * float(q[2] ** 2 + q[3] ** 2))
        lin_b = robot.data.root_lin_vel_b[0]
        ang_b = robot.data.root_ang_vel_b[0]
        log.append({"t": round(t, 3), "phase": phase, "cmd": cmd, "pos": t2l(pos, 4), "yaw": round(yaw, 4),
                    "lin_b": t2l(lin_b, 3), "ang_b": t2l(ang_b, 3),
                    "grav_z": round(float(robot.data.projected_gravity_b[0, 2]), 4),
                    "act_absmax": round(float(act.abs().max()), 3)})
        if k % 50 == 0:
            print(f"[ref] t={t:5.2f} {phase:6s} z={float(pos[2]):.3f} xy=({float(pos[0]):+.2f},{float(pos[1]):+.2f}) "
                  f"yaw={math.degrees(yaw):+6.1f} vb=({float(lin_b[0]):+.2f},{float(lin_b[1]):+.2f}) "
                  f"wz={float(ang_b[2]):+.2f} |a|max={float(act.abs().max()):.2f}", flush=True)

    def window(name):
        rows = [r for r in log if r["phase"] == name]
        if not rows:
            return None
        start = rows[0]["t"] + 1.0
        rows = [r for r in rows if r["t"] >= start]
        return {"n": len(rows), "cmd": rows[0]["cmd"],
                "mean_vx_b": round(float(np.mean([r["lin_b"][0] for r in rows])), 3),
                "mean_vy_b": round(float(np.mean([r["lin_b"][1] for r in rows])), 3),
                "mean_wz_b": round(float(np.mean([r["ang_b"][2] for r in rows])), 3),
                "min_z": round(float(min(r["pos"][2] for r in rows)), 3),
                "max_z": round(float(max(r["pos"][2] for r in rows)), 3)}

    result["walk"] = {
        "physics": f"Isaac Lab ManagerBasedRLEnv, device {args.device}, dt {env.sim.get_physics_dt()}, "
                   f"decimation {cfg.decimation}",
        "seconds_simulated": round(len(log) * env.step_dt, 2), "wall_s": round(time.time() - t0, 1),
        "resets_at_s": resets,
        "first_action_absmax": float(first_action.abs().max()),
        "first_action": t2l(first_action, 4),
        "final_pos": log[-1]["pos"], "final_yaw_deg": round(math.degrees(log[-1]["yaw"]), 2),
        "windows": {p: window(p) for p in ["stand", "fwd0.5", "fwd0.8", "turn", "stand2"]},
    }
    Path(args.out).with_suffix(".walklog.json").write_text(json.dumps(log))
    print(f"[ref] summary {json.dumps(result['walk']['windows'])}", flush=True)
    result["status"] = "ok"
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    result["status"] = f"error: {e!r}"
finally:
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(f"[ref] wrote {args.out} status={result.get('status')}", flush=True)
    sys.stdout.flush()
    # First version skipped env.close() and Isaac Sim 5.0's app.close() then spun at 100% CPU for 6 min
    # (result already written; killed via the slot wrapper). Isaac Lab's own scripts close the env first.
    try:
        env.close()  # noqa: F821 -- may not exist if setup failed
    except Exception:  # noqa: BLE001
        pass
    import os
    os._exit(0 if result.get("status") == "ok" else 1)   # do not risk a second hang in app.close()
