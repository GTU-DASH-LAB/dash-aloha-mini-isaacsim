"""Probe Isaac Sim 6.0.1's behavior-agent people stack before nav/sim/people.py is written.

DynaNav's pedestrians run on omni.anim.people 0.7.7 + IRA 0.7.19 (Isaac Sim 5.0). Neither
exists in 6.0.1: omni.anim.people is gone and IRA 1.6.8 rejects 0.7.x configs. The 6.0.1
replacement is omni.anim.behavior.core (C++ "behavior agents" driven by a HumanMotionLibrary).
This probe answers, on a tiny Z-up metre stage, the questions the people.py design depends on:

  1. Does CreateBehaviorAgentCommand work headless with the 6.0 CDN character + motion library,
     and does its `position` argument actually place the character?
  2. What physics does a behavior agent author under its SkelRoot (BodyCollider/AvoidanceTrigger:
     RigidBody? kinematic? CollisionAPI? trigger?), and does it FOLLOW the walking agent?
  3. Does omni.physx raycast_closest at z=0.30 m (our guard/lidar height) hit an agent, and which
     prim does it report (body capsule, or the wider avoidance trigger)?
  4. Do get_world_translation()/get_linear_velocity() report metres in a metre stage, at what
     walking speed, how far does the timeline advance per app.update(), and what does a step cost
     with N agents?
  5. Does an agent walking head-on at a moving KINEMATIC box (stand-in for the teleported robot)
     deviate (auto avoidance of rigid bodies)?
  6. Does a walking agent SHOVE a dynamic 50 kg box in its path (agent 0, "SolidBox"), and does a
     UsdPhysics.CollisionGroup authored BEFORE play (People group filtered against a "Robot"
     group) stop that for a second box (agent 2, "GhostBox")? -- this decides whether DynaNav's
     ghost people can be reproduced while keeping the capsules ray-visible.

Run ONLY through the GPU0 slot wrapper:
  ISAAC_SLOT_TAG=people-probe ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh \
      ~/robotics/isaacsim-6.0.1/python.sh nav/tools/probe_people_agents.py --out <dir> [--agents N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# nav/tools/profile.py shadows the stdlib `profile` module, which warp imports while Kit
# starts -- drop this directory from sys.path before SimulationApp exists.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--agents", type=int, default=4)
ap.add_argument("--frames", type=int, default=720)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

CDN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"
CHAR_URLS = [
    f"{CDN}/People/Characters/F_Business_02/F_Business_02.usd",
    f"{CDN}/People/Characters/M_Medical_01/M_Medical_01.usd",
    f"{CDN}/People/Characters/original_male_adult_police_04/male_adult_police_04.usd",
]
MOTION_LIB_URL = f"{CDN}/People/MotionLibrary/HumanMotionLibrary.usd"

report: dict = {"steps": [], "errors": [], "args": vars(args)}


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def save() -> None:
    (OUT / "probe_people_agents.json").write_text(json.dumps(report, indent=1, default=str))


def r3(v) -> list:
    return [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)]


app = omni.kit.app.get_app()
em = app.get_extension_manager()
for ext in ["omni.physx", "omni.anim.navigation.core", "omni.anim.behavior.core"]:
    ok = em.set_extension_enabled_immediate(ext, True)
    log(f"enable {ext}: {ok}")
    report[f"enable_{ext}"] = bool(ok)
for _ in range(5):
    app.update()

import NavSchema  # noqa: E402
import omni.anim.behavior.core as bh  # noqa: E402
import omni.anim.navigation.core as nav  # noqa: E402
import omni.physx  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

settings = carb.settings.get_settings()
report["kit_time_settings"] = {k: settings.get(k) for k in [
    "/app/runLoops/main/manualModeEnabled", "/app/runLoops/main/rateLimitFrequency",
    "/app/runLoops/main/rateLimitEnabled", "/app/player/useFixedTimeStepping"]}
# DynaNav's navmesh numbers (benchmark.py:982-985), translated to the 6.0.1 key names. 5.0 keys
# agentHeight/agentRadius no longer exist; setting them would be silently ignored.
NAV = "/exts/omni.anim.navigation.core/navMesh/config"
for k, v in {"agentMinHeight": 180, "agentMinRadius": 40, "agentMaxStepHeight": 20, "agentMaxFloorSlope": 45.0,
             "autoRebakeOnChanges": False}.items():
    settings.set(f"{NAV}/{k}", v)
    settings.set(f"/persistent{NAV}/{k}", v)
settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
report["behavior_settings"] = {k: settings.get(f"/exts/omni.anim.behavior.core/{k}") for k in
                               ["enableAutoAvoidance", "enableRagdollPhysics", "enableAutoImpact",
                                "defaultAutoAvoidanceMass", "navBlockedTimeout", "navBlockedSpeedThreshold",
                                "walkSpeedControlRatio", "agentResponseTime"]}
report["nav_settings_after_set"] = {k: settings.get(f"{NAV}/{k}") for k in
                                    ["agentMinHeight", "agentMinRadius", "agentMaxRadius", "agentMaxStepHeight",
                                     "agentMaxFloorSlope", "excludeRigidBodies", "autoRebakeOnChanges"]}

ctx = omni.usd.get_context()
ctx.new_stage()
for _ in range(5):
    app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
UsdLux.DistantLight.Define(stage, "/World/Light").GetIntensityAttr().Set(3000.0)
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr(9.81)


def box(path: str, center, size, rigid: str | None = None, mass: float = 50.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    x = UsdGeom.XformCommonAPI(cube.GetPrim())
    x.SetTranslate(Gf.Vec3d(*center))
    x.SetScale(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if rigid:
        rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        if rigid == "kinematic":
            rb.CreateKinematicEnabledAttr(True)
        else:
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass)
    return x


def mesh_box(path: str, center, size):
    # The navmesh baker wants UsdGeom.Mesh walkable geometry: a run with UsdGeom.Cube ground baked
    # nothing (start_navmesh_baking_and_wait() -> False in 0.0 s, "Behavior System disabled because
    # no navmesh is available", 0/4 agents). The nav stages are all Mesh, so this is probe-only.
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = center
    pts = [Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5])
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Boundable(m.GetPrim()).GetExtentAttr().Set([pts[0], pts[7]])
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())


mesh_box("/World/Ground", (0, 0, -0.05), (40, 40, 0.1))
mesh_box("/World/Wall", (6.0, 0.0, 1.0), (0.2, 3.0, 2.0))
# Dynamic boxes, NOT in the navmesh bake (excludeRigidBodies default true), so the walker's planned
# straight line goes right through them; what happens there is the measurement.
box("/World/SolidBox", (0.0, 0.0, 0.3), (0.6, 0.5, 0.6), rigid="dynamic")
box("/World/GhostBox", (0.0, -4.0, 0.3), (0.6, 0.5, 0.6), rigid="dynamic")
# Kinematic "robot" box: 0.6 x 0.5 x 0.6, starts at (-6, 2.5, 0.3), slides +X at 0.5 m/s.
rx = box("/World/RobotBox", (-6.0, 2.5, 0.3), (0.6, 0.5, 0.6), rigid="kinematic")

# Collision groups authored BEFORE play: everything under /World/Characters (including the
# BodyCollider the behavior system authors at play time) vs. a "Robot" group holding GhostBox.
gp = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/People")
gp.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Characters")
gr = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/Robot")
gr.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/GhostBox")
gp.CreateFilteredGroupsRel().AddTarget(gr.GetPath())

vol = NavSchema.NavMeshVolume.Define(stage, "/World/NavMeshVolume")
vol.GetNavVolumeTypeAttr().Set("Include")
NavSchema.NavMeshAreaAPI.Apply(vol.GetPrim())
UsdGeom.Boundable(vol.GetPrim()).GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
vx = UsdGeom.XformCommonAPI(vol.GetPrim())
vx.SetTranslate(Gf.Vec3d(0, 0, 0.5))
vx.SetScale(Gf.Vec3f(30, 30, 3))

for _ in range(10):
    app.update()

# Agents: 0 walks +X through SolidBox; 1 walks -X head-on at the kinematic RobotBox's lane (y=2.5);
# 2 walks +X through GhostBox (filtered); i>=3 are a cost crowd on a grid, odd ones walking.
spawn = [(-4.0, 0.0, 0.0), (4.5, 2.5, 0.0), (-4.0, -4.0, 0.0)]
goals = [(4.5, 0.0, 0.0), (-8.0, 2.5, 0.0), (4.5, -4.0, 0.0)]
for i in range(3, args.agents):
    j = i - 3
    sx, sy = -12.0 + (j % 10) * 2.6, -7.0 - (j // 10) * 2.0
    spawn.append((sx, sy, 0.0))
    goals.append((-sx, sy, 0.0) if i % 2 else None)
skel_paths: list[str] = []
t0 = time.time()
UsdGeom.Scope.Define(stage, "/World/Characters")
report["spawn_check"] = []
for i in range(args.agents):
    char_path = Sdf.Path(f"/World/Characters/Person_{i:02d}")
    try:
        omni.kit.commands.execute(
            "CreateBehaviorAgentCommand",
            character_prim_path=char_path,
            character_asset_url=CHAR_URLS[i % len(CHAR_URLS)],
            motion_library_prim_path=Sdf.Path("/World/Characters/HumanMotionLibrary"),
            motion_library_asset_url=MOTION_LIB_URL,
            motion_library_skeleton_rig="Human",
            position=Gf.Vec3d(*spawn[i]),
        )
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"CreateBehaviorAgentCommand {i}: {e!r}")
    app.update()
    prim = stage.GetPrimAtPath(char_path)
    authored = r3(omni.usd.get_world_transform_matrix(prim).ExtractTranslation()) if prim.IsValid() else None
    ops = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()] if prim.IsValid() else []
    if authored is not None and math.hypot(authored[0] - spawn[i][0], authored[1] - spawn[i][1]) > 1e-3:
        # position= only edits an EXISTING translate op; place it ourselves if it did not take.
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*spawn[i]))
    report["spawn_check"].append({"i": i, "requested": spawn[i], "after_command": authored, "xform_ops": ops})
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "SkelRoot":
            skel_paths.append(str(p.GetPath()))
            break
# wait for payload loads (motion library is ~444 MB the first time)
for k in range(1200):
    app.update()
    if k % 50 == 0 and k > 0 and not ctx.get_stage_loading_status()[2]:
        break
report["agents_created_s"] = round(time.time() - t0, 1)
report["skelroots"] = skel_paths
log(f"skelroots: {skel_paths[:4]}... ({len(skel_paths)}) in {report['agents_created_s']} s")
save()

# Bake AFTER the agents exist (the behavior core's own tests do it in this order, with the
# comment "seems create_human_agent() must be called before create_default_navmesh()").
inav = nav.acquire_interface()
t0 = time.time()
baked = inav.start_navmesh_baking_and_wait()
navmesh = inav.get_navmesh()
report["navmesh"] = {"bake_returned": baked, "has_navmesh": navmesh is not None, "bake_s": round(time.time() - t0, 2)}
if navmesh is None:
    for _ in range(30):
        app.update()
    t0 = time.time()
    baked = inav.start_navmesh_baking_and_wait()
    navmesh = inav.get_navmesh()
    report["navmesh"]["retry"] = {"bake_returned": baked, "has_navmesh": navmesh is not None,
                                  "bake_s": round(time.time() - t0, 2)}
if navmesh is not None:
    for fn in ["get_agent_min_height", "get_agent_min_radius", "get_agent_max_radius", "get_agent_max_slope",
               "get_agent_max_step_height", "get_agent_min_island_radius"]:
        try:
            report["navmesh"][fn] = getattr(navmesh, fn)()
        except Exception as e:  # noqa: BLE001
            report["navmesh"][fn] = f"ERR {e!r}"
    try:
        p = navmesh.query_shortest_path(carb.Float3(-4, 0, 0), carb.Float3(4.5, 0, 0))
        pts = p.get_points() if p is not None else None
        report["navmesh"]["path_m4_to_4p5"] = [r3((q.x, q.y, q.z)) for q in pts] if pts else None
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"query_shortest_path: {e!r}")
log(f"navmesh: {report['navmesh']}")
save()


def describe_physics(skel: str) -> dict:
    out = {}
    root = stage.GetPrimAtPath(f"{skel}/Physics")
    if not root.IsValid():
        return {"missing": True}
    for p in Usd.PrimRange(root):
        d = {"type": p.GetTypeName(), "apis": [str(a) for a in p.GetAppliedSchemas()]}
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            r = UsdPhysics.RigidBodyAPI(p)
            d["kinematic"] = r.GetKinematicEnabledAttr().Get()
            d["rb_enabled"] = r.GetRigidBodyEnabledAttr().Get()
        if p.HasAPI(UsdPhysics.CollisionAPI):
            d["collision_enabled"] = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        d["trigger"] = p.HasAPI(PhysxSchema.PhysxTriggerAPI)
        if p.IsA(UsdGeom.Capsule) or p.IsA(UsdGeom.Cylinder):
            g = UsdGeom.Capsule(p) if p.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(p)
            d["radius"], d["height"], d["axis"] = g.GetRadiusAttr().Get(), g.GetHeightAttr().Get(), g.GetAxisAttr().Get()
        if p.IsA(UsdGeom.Xformable):
            d["world_t"] = r3(omni.usd.get_world_transform_matrix(p).ExtractTranslation())
        d["props"] = sorted(a.GetName() for a in p.GetAuthoredAttributes())[:40]
        out[str(p.GetPath())] = d
    return out


timeline = omni.timeline.get_timeline_interface()
timeline.play()
agents = [None] * len(skel_paths)
t_wait = time.time()
for k in range(900):
    app.update()
    for i, s in enumerate(skel_paths):
        if agents[i] is None:
            agents[i] = bh.acquire_interface().get_agent(s)
    if all(a is not None for a in agents) and k >= 10:
        break
report["agents_found"] = [a is not None for a in agents]
report["agent_wait_frames"] = k + 1
report["agent_wait_s"] = round(time.time() - t_wait, 1)
report["physics_after_play"] = {s: describe_physics(s) for s in skel_paths[:3]}
if agents and agents[0] is not None:
    a0 = agents[0]
    report["agent_api"] = sorted(n for n in dir(a0) if not n.startswith("_"))
    for fn in ["get_height", "get_radius", "get_speed", "get_auto_avoidance_mass", "is_auto_avoidance_enabled",
               "is_obstacle_avoidance_enabled", "get_custom_action_names", "get_up_direction"]:
        try:
            v = getattr(a0, fn)()
            report[f"agent0_{fn}"] = v if isinstance(v, (int, float, str, bool, list)) else str(v)
        except Exception as e:  # noqa: BLE001
            report[f"agent0_{fn}"] = f"ERR {e!r}"
log(f"agents found: {sum(report['agents_found'])}/{len(agents)} after {report['agent_wait_frames']} frames")
save()

task_ids = {}
for i, a in enumerate(agents):
    if a is None:
        continue
    try:
        if goals[i] is not None:
            a.set_speed(1.2)  # stage units / s; stage is metres
            task_ids[i] = a.move_to(target=carb.Float3(*goals[i]))
        else:
            task_ids[i] = a.idle()
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"task {i}: {e!r}")
report["task_ids"] = {k: int(v) for k, v in task_ids.items()}

sq = omni.physx.get_physx_scene_query_interface()


def ray(origin, direction, dist=10.0):
    hit = sq.raycast_closest(carb.Float3(*origin), carb.Float3(*direction), dist)
    if not hit or not hit.get("hit"):
        return {"hit": False}
    return {"hit": True, "dist": round(hit["distance"], 3), "collision": hit.get("collision"),
            "rigidBody": hit.get("rigidBody")}


def side_ray(agent_xy, z, side):
    # perpendicular ray from 3 m to the side, so no other probe object sits between
    o = (agent_xy[0], agent_xy[1] + 3.0 * side, z)
    out = ray(o, (0.0, -side, 0.0))
    out["expected_center_dist"] = 3.0
    return out


def xyz(path):
    return r3(omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path)).ExtractTranslation())


robot_x = -6.0
t_steps, t_tl = [], []
min_d = {"SolidBox": 99.0, "GhostBox": 99.0, "RobotBox": 99.0}
lat_dev = {0: 0.0, 1: 0.0, 2: 0.0}
for f in range(args.frames):
    robot_x += 0.5 / 60.0
    rx.SetTranslate(Gf.Vec3d(robot_x, 2.5, 0.3))
    tl0 = timeline.get_current_time()
    ts = time.time()
    app.update()
    t_steps.append(time.time() - ts)
    t_tl.append(timeline.get_current_time() - tl0)
    pos = {}
    for i in (0, 1, 2):
        if i < len(agents) and agents[i] is not None:
            p = agents[i].get_world_translation()
            pos[i] = (p.x, p.y, p.z)
            lat_dev[i] = max(lat_dev[i], abs(p.y - spawn[i][1]))
    sb, gb = xyz("/World/SolidBox"), xyz("/World/GhostBox")
    if 0 in pos:
        min_d["SolidBox"] = min(min_d["SolidBox"], math.hypot(pos[0][0] - sb[0], pos[0][1] - sb[1]))
    if 2 in pos:
        min_d["GhostBox"] = min(min_d["GhostBox"], math.hypot(pos[2][0] - gb[0], pos[2][1] - gb[1]))
    if 1 in pos:
        min_d["RobotBox"] = min(min_d["RobotBox"], math.hypot(pos[1][0] - robot_x, pos[1][1] - 2.5))
    if f % 30 == 0:
        row = {"f": f, "tl": round(timeline.get_current_time(), 4), "robot_x": round(robot_x, 3),
               "SolidBox": sb, "GhostBox": gb,
               "ray_robotbox": ray((robot_x, 1.5, 0.3), (0.0, 1.0, 0.0)), "agents": []}
        for i in range(min(3, len(agents))):
            a = agents[i]
            if a is None:
                continue
            p = a.get_world_translation()
            v = a.get_linear_velocity()
            body = stage.GetPrimAtPath(f"{skel_paths[i]}/Physics/BodyCollider")
            st = a.get_task_status(task_ids[i]) if i in task_ids else None
            r = {"i": i, "pos": r3((p.x, p.y, p.z)), "vel": r3((v.x, v.y, v.z)),
                 "speed": round(math.hypot(v.x, v.y), 3),
                 "usd_skelroot": xyz(skel_paths[i]),
                 "usd_body": xyz(body.GetPath()) if body.IsValid() else None,
                 "task": str(st)}
            side = 1.0 if i == 1 else -1.0
            r["ray030"] = side_ray((p.x, p.y), 0.30, side)
            r["ray100"] = side_ray((p.x, p.y), 1.00, side)
            row["agents"].append(r)
        report["steps"].append(row)
        log(json.dumps(row)[:600])
        save()

report["step_ms"] = {"mean": round(1000 * sum(t_steps) / len(t_steps), 2),
                     "p50": round(1000 * sorted(t_steps)[len(t_steps) // 2], 2),
                     "max": round(1000 * max(t_steps), 2), "agents": len(agents)}
report["timeline_dt"] = {"mean": round(sum(t_tl) / len(t_tl), 6), "min": round(min(t_tl), 6),
                         "max": round(max(t_tl), 6)}
report["min_dist_agent_to_box_m"] = {k: round(v, 3) for k, v in min_d.items()}
report["max_lateral_deviation_m"] = {k: round(v, 3) for k, v in lat_dev.items()}
report["box_end"] = {"SolidBox": xyz("/World/SolidBox"), "GhostBox": xyz("/World/GhostBox")}
report["physics_end"] = {s: describe_physics(s) for s in skel_paths[:3]}
if len(agents) > 3:
    moved = []
    for i in range(3, len(agents)):
        if agents[i] is not None:
            p = agents[i].get_world_translation()
            moved.append(round(math.hypot(p.x - spawn[i][0], p.y - spawn[i][1]), 2))
    report["crowd_displacement_m"] = moved
timeline.stop()
for _ in range(5):
    app.update()
save()
log(f"done; step_ms={report['step_ms']}; dt={report['timeline_dt']}; min_d={report['min_dist_agent_to_box_m']}; "
    f"boxes={report['box_end']}; errors={report['errors']}")
kit.close()
