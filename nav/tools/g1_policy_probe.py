"""Probe the Isaac Lab G1 USD in Kit: PhysX DOF order, USD drive/limit data, and (``--walk``)
a closed-loop run of the pretrained flat walking policy on a flat plane.

The DOF order is the whole point of the first phase. Isaac Lab's ``JointPositionAction`` and
``joint_pos_rel`` use ``find_joints(".*", preserve_order=False)``, i.e. the order of
``root_physx_view.shared_metatype.dof_names`` -- the PhysX articulation's own DOF order, which
is not the USD authoring order and not alphabetical. isaacsim.core.prims.Articulation reads the
same tensor-API list, so a Kit probe of the SAME USD is the ground truth for the policy order.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/g1_policy_probe.py \
        --usd <g1_minimal.usd> --out probe.json [--walk --spec policy_spec.json]

Physics for --walk mirrors the training env (IsaacLab-2.2.1 velocity_env_cfg.py:305-314,
unitree.py:270-374): dt 5 ms, policy every 4 steps, ground mu 1.0/1.0 multiply, robot shapes
mu_s 0.8 / mu_d 0.6, solver 8/4, self-collisions off, gains written through the tensor API
(rad units, so no USD deg conversion). CPU physics (numpy backend; Kit has no torch), whereas
training ran the GPU pipeline -- noted in the result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# nav/tools/profile.py shadows the stdlib `profile`, which Kit's warp imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--walk", action="store_true")
ap.add_argument("--spec", default=None)
ap.add_argument("--seconds", type=float, default=16.0)
ap.add_argument("--try-ml-archive", action="store_true",
                help="also check whether enabling omni.isaac.ml_archive makes torch importable")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

result: dict = {"usd": args.usd}
try:
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02, backend="numpy")
    stage = omni.usd.get_context().get_stage()

    # ---------------------------------------------------------------- ground (Isaac Lab plane)
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.GetSizeAttr().Set(1.0)
    gx = UsdGeom.XformCommonAPI(ground.GetPrim())
    gx.SetTranslate(Gf.Vec3d(0, 0, -0.5))
    gx.SetScale(Gf.Vec3f(200.0, 200.0, 1.0))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    mat = UsdShade.Material.Define(stage, "/World/GroundMaterial")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(1.0)
    mapi.CreateDynamicFrictionAttr(1.0)
    mapi.CreateRestitutionAttr(0.0)
    pmat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pmat.CreateFrictionCombineModeAttr("multiply")
    pmat.CreateRestitutionCombineModeAttr("multiply")
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # ---------------------------------------------------------------- robot
    add_reference_to_stage(usd_path=args.usd, prim_path="/World/G1")
    for _ in range(5):
        kit.update()
    robot_prim = stage.GetPrimAtPath("/World/G1")
    roots, joints, bodies = [], [], []
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            j = {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
            jo = UsdPhysics.Joint(prim)
            b0 = jo.GetBody0Rel().GetTargets()
            b1 = jo.GetBody1Rel().GetTargets()
            j["body0"] = str(b0[0]) if b0 else None
            j["body1"] = str(b1[0]) if b1 else None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(prim)
                j["axis"] = rj.GetAxisAttr().Get()
                j["lower_deg"] = rj.GetLowerLimitAttr().Get()
                j["upper_deg"] = rj.GetUpperLimitAttr().Get()
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive and drive.GetStiffnessAttr().IsValid():
                j["usd_drive"] = {
                    "type": drive.GetTypeAttr().Get(),
                    "stiffness_per_deg": drive.GetStiffnessAttr().Get(),
                    "damping_per_deg": drive.GetDampingAttr().Get(),
                    "max_force": drive.GetMaxForceAttr().Get(),
                }
            pj = PhysxSchema.PhysxJointAPI(prim)
            if pj and pj.GetMaxJointVelocityAttr().IsValid():
                j["max_joint_velocity_deg_s"] = pj.GetMaxJointVelocityAttr().Get()
            if pj and pj.GetArmatureAttr().IsValid():
                j["armature"] = pj.GetArmatureAttr().Get()
            joints.append(j)
    result["articulation_roots"] = roots
    result["num_rigid_bodies"] = len(bodies)
    result["usd_joints"] = joints
    print(f"[probe] roots={roots} bodies={len(bodies)} joints={len(joints)}", flush=True)
    if len(roots) != 1:
        raise RuntimeError(f"expected one articulation root, found {roots}")

    # Isaac Lab spawn-time properties (G1_CFG.spawn, unitree.py:274-285).
    for p in bodies:
        prim = stage.GetPrimAtPath(p)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateDisableGravityAttr(False)
        rb.CreateRetainAccelerationsAttr(False)
        rb.CreateLinearDampingAttr(0.0)
        rb.CreateAngularDampingAttr(0.0)
        rb.CreateMaxLinearVelocityAttr(1000.0)
        rb.CreateMaxAngularVelocityAttr(1000.0)          # deg/s, written raw as Isaac Lab does
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
    root_prim = stage.GetPrimAtPath(roots[0])
    pa = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    pa.CreateEnabledSelfCollisionsAttr(False)
    pa.CreateSolverPositionIterationCountAttr(8)
    pa.CreateSolverVelocityIterationCountAttr(4)

    UsdGeom.XformCommonAPI(robot_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.74))

    art = Articulation(roots[0], name="g1")
    world.scene.add(art)
    world.reset()
    for _ in range(2):
        world.step(render=False)

    dof_names = list(art.dof_names)
    result["dof_names_physx_order"] = dof_names
    result["num_dof"] = len(dof_names)
    result["body_names_physx_order"] = list(art.body_names)
    lim = art.get_dof_limits()
    result["dof_limits_rad"] = np.asarray(lim).reshape(-1, 2).tolist()
    kps, kds = art.get_gains()
    result["usd_gains_tensor_api"] = {"kp": np.asarray(kps).reshape(-1).tolist(),
                                      "kd": np.asarray(kds).reshape(-1).tolist()}
    result["max_efforts"] = np.asarray(art.get_max_efforts()).reshape(-1).tolist()
    result["max_joint_velocities"] = np.asarray(art.get_joint_max_velocities()).reshape(-1).tolist()
    result["armatures"] = np.asarray(art.get_armatures()).reshape(-1).tolist()
    try:
        result["effort_modes"] = art.get_effort_modes()
    except Exception as e:  # noqa: BLE001
        result["effort_modes"] = f"unavailable: {e!r}"
    try:
        result["drive_types"] = art.get_drive_types()
    except Exception as e:  # noqa: BLE001
        result["drive_types"] = f"unavailable: {e!r}"
    print(f"[probe] {len(dof_names)} DOFs: {dof_names}", flush=True)

    # ---------------------------------------------------------------- closed-loop walk
    if args.walk:
        from walk_policy import G1WalkPolicy, base_frame_state

        pol = G1WalkPolicy(args.spec)
        result["spec_order_equals_physx_order"] = pol.joint_names == dof_names
        if pol.joint_names != dof_names:
            diffs = [(k, a, b) for k, (a, b) in enumerate(zip(pol.joint_names, dof_names)) if a != b]
            result["spec_vs_physx_first_diffs"] = diffs[:10]
        print(f"[walk] spec order == PhysX order: {result['spec_order_equals_physx_order']}", flush=True)
        sim_idx = pol.bind(dof_names)
        spec = pol.spec
        n = len(dof_names)
        act = spec["actuators_policy_order"]
        kp = pol.to_sim_order(np.array(act["stiffness"]), n)
        kd = pol.to_sim_order(np.array(act["damping"]), n)
        eff = pol.to_sim_order(np.array(act["effort_limit"]), n)
        arm = pol.to_sim_order(np.array(act["armature"]), n)
        q0 = pol.to_sim_order(pol.default_q, n)
        art.set_gains(kps=kp.reshape(1, -1), kds=kd.reshape(1, -1))
        art.set_max_efforts(eff.reshape(1, -1))
        art.set_armatures(arm.reshape(1, -1))
        vlim = act.get("velocity_limit")
        if vlim is not None:
            cur = np.asarray(art.get_joint_max_velocities()).reshape(-1)
            v = np.array([cur[sim_idx[k]] if vl is None else vl for k, vl in enumerate(vlim)])
            art.set_max_joint_velocities(pol.to_sim_order(v, n, fill=cur).reshape(1, -1))

        # Isaac Lab's randomize_rigid_body_material (velocity_env_cfg.py:154-164): every robot
        # shape mu_s 0.8, mu_d 0.6, e 0 (the ranges are degenerate, so no randomness).
        try:
            view = art._physics_view
            mats = view.get_material_properties()
            mats_np = np.asarray(mats).copy()
            mats_np[..., 0] = 0.8
            mats_np[..., 1] = 0.6
            mats_np[..., 2] = 0.0
            view.set_material_properties(mats_np, np.arange(1))
            result["robot_material"] = f"set on {mats_np.shape[1]} shapes"
        except Exception as e:  # noqa: BLE001
            result["robot_material"] = f"FAILED: {e!r}"

        def reset_pose():
            art.set_world_poses(positions=np.array([[0.0, 0.0, 0.74]]),
                                orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
            art.set_joint_positions(q0.reshape(1, -1))
            art.set_joint_velocities(np.zeros((1, n)))
            art.set_velocities(np.zeros((1, 6)))
            art.set_joint_position_targets(q0.reshape(1, -1))

        reset_pose()
        pol.reset()

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
        dt_policy = 0.02
        steps = int(round(args.seconds / dt_policy))
        fell_at = None
        t0 = time.time()
        first_action = None
        for k in range(steps):
            t = k * dt_policy
            cmd, phase = command_at(t)
            pos, quat = art.get_world_poses()
            vel = np.asarray(art.get_velocities()).reshape(-1)
            pos = np.asarray(pos).reshape(-1)
            quat = np.asarray(quat).reshape(-1)
            lin_b, ang_b, grav_b = base_frame_state(quat, vel[:3], vel[3:])
            q = np.asarray(art.get_joint_positions()).reshape(-1)
            dq = np.asarray(art.get_joint_velocities()).reshape(-1)
            target, action, obs = pol.step(lin_b, ang_b, grav_b, cmd, q, dq)
            if first_action is None:
                first_action = action.copy()
            art.set_joint_position_targets(pol.to_sim_order(target, n, fill=q0).reshape(1, -1))
            for _ in range(4):
                world.step(render=False)
            yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                             1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
            log.append({"t": round(t, 3), "phase": phase, "cmd": cmd,
                        "pos": [round(float(x), 4) for x in pos], "yaw": round(yaw, 4),
                        "lin_b": [round(float(x), 3) for x in lin_b],
                        "ang_b": [round(float(x), 3) for x in ang_b],
                        "grav_z": round(float(grav_b[2]), 4),
                        "act_absmax": round(float(np.abs(action).max()), 3)})
            if fell_at is None and (pos[2] < 0.45 or grav_b[2] > -0.6):
                fell_at = t
                print(f"[walk] FELL at t={t:.2f}s z={pos[2]:.3f} grav_z={grav_b[2]:.3f}", flush=True)
                break
            if k % 50 == 0:
                print(f"[walk] t={t:5.2f} {phase:6s} z={pos[2]:.3f} xy=({pos[0]:+.2f},{pos[1]:+.2f}) "
                      f"yaw={math.degrees(yaw):+6.1f} vb=({lin_b[0]:+.2f},{lin_b[1]:+.2f}) "
                      f"wz={ang_b[2]:+.2f} |a|max={np.abs(action).max():.2f}", flush=True)

        def window(name):
            rows = [r for r in log if r["phase"] == name and r["t"] >= (
                min(x["t"] for x in log if x["phase"] == name) + 1.0)]
            if not rows:
                return None
            return {
                "n": len(rows),
                "cmd": rows[0]["cmd"],
                "mean_vx_b": round(float(np.mean([r["lin_b"][0] for r in rows])), 3),
                "mean_vy_b": round(float(np.mean([r["lin_b"][1] for r in rows])), 3),
                "mean_wz_b": round(float(np.mean([r["ang_b"][2] for r in rows])), 3),
                "min_z": round(float(min(r["pos"][2] for r in rows)), 3),
                "max_z": round(float(max(r["pos"][2] for r in rows)), 3),
            }

        result["walk"] = {
            "physics": "CPU (numpy backend), dt 0.005, decimation 4",
            "seconds_simulated": round(len(log) * dt_policy, 2),
            "wall_s": round(time.time() - t0, 1),
            "fell_at_s": fell_at,
            "first_action_absmax": float(np.abs(first_action).max()) if first_action is not None else None,
            "first_action": [round(float(a), 4) for a in first_action] if first_action is not None else None,
            "final_pos": log[-1]["pos"] if log else None,
            "final_yaw_deg": round(math.degrees(log[-1]["yaw"]), 2) if log else None,
            "windows": {p: window(p) for p in ["stand", "fwd0.5", "fwd0.8", "turn", "stand2"]},
        }
        Path(args.out).with_suffix(".walklog.json").write_text(json.dumps(log))
        print(f"[walk] summary {json.dumps(result['walk']['windows'])}", flush=True)

    if args.try_ml_archive:
        try:
            import omni.kit.app
            em = omni.kit.app.get_app().get_extension_manager()
            ok = em.set_extension_enabled_immediate("omni.isaac.ml_archive", True)
            import torch  # noqa: F401
            result["ml_archive_torch"] = f"enabled={ok} torch {torch.__version__}"
        except Exception as e:  # noqa: BLE001
            result["ml_archive_torch"] = f"FAILED: {e!r}"
        print(f"[probe] ml_archive: {result['ml_archive_torch']}", flush=True)

    result["status"] = "ok"
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    result["status"] = f"error: {e!r}"
finally:
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(f"[probe] wrote {args.out} status={result.get('status')}", flush=True)
    rc = 0 if result.get("status") == "ok" else 1
    sys.stdout.flush()
    kit.close()
    sys.exit(rc)   # not reached in practice (close() ends the process); kept for clarity
