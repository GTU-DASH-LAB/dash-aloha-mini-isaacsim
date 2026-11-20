"""Second people probe (Isaac Sim 6.0.1): obstacle avoidance, ghost filtering, ray proxies, cost.

probe_people_agents.py established (scratchpad/people/probe3): a behavior agent authors a KINEMATIC
capsule <SkelRoot>/Physics/BodyCollider (r ~0.14 m) that PhysX moves with the walker and
raycast_closest hits, plus a 1.7 m kinematic TRIGGER cylinder that rays do not hit; USD transforms
stay at the spawn point (Fabric-only writes); a walker SHOVED a 50 kg dynamic box 2.5 m; a
CollisionGroup authored before play with includes=/World/Characters did NOT stop that (the capsule
is created after PhysX parsed the stage); the agent walked straight through a kinematic UsdGeom.Cube.

This probe answers what people.py needs next, one lane per question (agents walk +X, or -X on B):
  A (y=0)   dynamic MESH box (convex hull, like the behavior core's own move_to "Obstacle"):
            does the crowd sim avoid a rigid MESH body (the Cube in probe 1 was not avoided)?
  B (y=2.5) KINEMATIC mesh box sliding head-on at 0.5 m/s (robot stand-in): avoided?
  C (y=-4)  dynamic box + UsdPhysics.FilteredPairsAPI(box -> agent's BodyCollider) authored at
            RUNTIME, after the capsule exists: does the walker still push it?
  D (y=-8)  dynamic box + CollisionGroups authored at RUNTIME naming the BodyCollider explicitly.
  E (y=+6)  our OWN kinematic proxy capsule, authored BEFORE play in group "Proxies" filtered against
            group "RobotPre" (holding a dynamic target box), moved through the box via USD xform at
            1.2 m/s: ray-visible? does it push the filtered box?
  + a crowd of (agents-4) more walkers for the per-step cost, and at the end: does setting
    physics:collisionEnabled=False on a BodyCollider hide it from raycast_closest?

  ISAAC_SLOT_TAG=people-probe ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh \
      ~/robotics/isaacsim-6.0.1/python.sh nav/tools/probe_people_physics.py --out <dir> [--agents 20]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--agents", type=int, default=20)
ap.add_argument("--frames", type=int, default=720)
ap.add_argument("--contact-lanes", action="store_true",
                help="force contact: lanes A/B/C/D use UsdGeom.Cube boxes (the crowd sim does not avoid Cubes,\n"
                     "probe 1), B gets collisionEnabled=False on its walker's capsule, + unfiltered proxy lane F")
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
report: dict = {"args": vars(args), "errors": [], "rows": []}


def log(m):
    print(f"[probe2] {m}", flush=True)


def save():
    (OUT / "probe_people_physics.json").write_text(json.dumps(report, indent=1, default=str))


def r3(v):
    return [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)]


app = omni.kit.app.get_app()
em = app.get_extension_manager()
for ext in ["omni.physx", "omni.anim.navigation.core", "omni.anim.behavior.core"]:
    report[f"enable_{ext}"] = bool(em.set_extension_enabled_immediate(ext, True))
for _ in range(5):
    app.update()

import NavSchema  # noqa: E402
import omni.anim.behavior.core as bh  # noqa: E402
import omni.anim.navigation.core as nav  # noqa: E402
import omni.physx  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

s = carb.settings.get_settings()
NAV = "/exts/omni.anim.navigation.core/navMesh/config"
for k, v in {"agentMinHeight": 180, "agentMinRadius": 40, "agentMaxStepHeight": 20, "agentMaxFloorSlope": 45.0,
             "autoRebakeOnChanges": False}.items():
    s.set(f"{NAV}/{k}", v)
    s.set(f"/persistent{NAV}/{k}", v)

ctx = omni.usd.get_context()
ctx.new_stage()
for _ in range(5):
    app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
UsdLux.DistantLight.Define(stage, "/World/Light").GetIntensityAttr().Set(3000.0)
UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")


def mesh_box(path, center, size, rigid=None, mass=50.0):
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    pts = [Gf.Vec3f(sx * hx, sy * hy, sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5])
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Boundable(m.GetPrim()).GetExtentAttr().Set([pts[0], pts[7]])
    x = UsdGeom.XformCommonAPI(m.GetPrim())
    x.SetTranslate(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    if rigid:
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        rb = UsdPhysics.RigidBodyAPI.Apply(m.GetPrim())
        if rigid == "kinematic":
            rb.CreateKinematicEnabledAttr(True)
        else:
            UsdPhysics.MassAPI.Apply(m.GetPrim()).CreateMassAttr(mass)
    return x


mesh_box("/World/Ground", (0, 0, -0.05), (40, 40, 0.1))
BOX = (0.6, 0.5, 0.6)


def cube_box(path, center, size, mass=50.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    x = UsdGeom.XformCommonAPI(c.GetPrim())
    x.SetTranslate(Gf.Vec3d(*center))
    x.SetScale(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(mass)
    return x


if args.contact_lanes:
    cube_box("/World/BoxA", (0.0, 0.0, 0.3), BOX)
    rbx = cube_box("/World/BoxB_Robot", (0.0, 2.5, 0.3), BOX)  # static start; walker's capsule disabled at f=60
    cube_box("/World/BoxC_Pairs", (0.0, -4.0, 0.3), BOX)
    cube_box("/World/BoxD_Groups", (0.0, -8.0, 0.3), BOX)
    # Lane F control: an UNFILTERED proxy capsule through a dynamic box at y=9
    UsdGeom.Scope.Define(stage, "/World/ProxyControl")
    capf = UsdGeom.Capsule.Define(stage, "/World/ProxyControl/P1")
    capf.CreateRadiusAttr(0.15)
    capf.CreateHeightAttr(1.4)
    capf.CreateAxisAttr("Z")
    pfx = UsdGeom.XformCommonAPI(capf.GetPrim())
    pfx.SetTranslate(Gf.Vec3d(-6.0, 9.0, 0.85))
    UsdPhysics.CollisionAPI.Apply(capf.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(capf.GetPrim()).CreateKinematicEnabledAttr(True)
    mesh_box("/World/BoxF_Control", (0.0, 9.0, 0.3), BOX, "dynamic")
else:
    mesh_box("/World/BoxA", (0.0, 0.0, 0.3), BOX, "dynamic")
    rbx = mesh_box("/World/BoxB_Robot", (-6.0, 2.5, 0.3), BOX, "kinematic")
    mesh_box("/World/BoxC_Pairs", (0.0, -4.0, 0.3), BOX, "dynamic")
    mesh_box("/World/BoxD_Groups", (0.0, -8.0, 0.3), BOX, "dynamic")
mesh_box("/World/BoxE_Target", (0.0, 6.0, 0.3), BOX, "dynamic")

# Lane E: our own proxy capsule, pre-play, in a group filtered against the target's group.
UsdGeom.Scope.Define(stage, "/World/Proxies")
cap = UsdGeom.Capsule.Define(stage, "/World/Proxies/P0")
cap.CreateRadiusAttr(0.15)
cap.CreateHeightAttr(1.4)
cap.CreateAxisAttr("Z")
cap.CreatePurposeAttr("guide")  # invisible to cameras (render purpose), still a PhysX shape
px = UsdGeom.XformCommonAPI(cap.GetPrim())
px.SetTranslate(Gf.Vec3d(-6.0, 6.0, 0.85))
UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim()).CreateKinematicEnabledAttr(True)
g_prox = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/Proxies")
g_prox.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Proxies")
g_rob = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/RobotPre")
g_rob.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/BoxE_Target")
g_prox.CreateFilteredGroupsRel().AddTarget(g_rob.GetPath())

vol = NavSchema.NavMeshVolume.Define(stage, "/World/NavMeshVolume")
vol.GetNavVolumeTypeAttr().Set("Include")
NavSchema.NavMeshAreaAPI.Apply(vol.GetPrim())
UsdGeom.Boundable(vol.GetPrim()).GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
vx = UsdGeom.XformCommonAPI(vol.GetPrim())
vx.SetTranslate(Gf.Vec3d(0, 0, 0.5))
vx.SetScale(Gf.Vec3f(30, 30, 3))
for _ in range(10):
    app.update()

LANES = {0: ((-4.0, 0.0), (4.5, 0.0)), 1: ((4.5, 2.5), (-8.0, 2.5)), 2: ((-4.0, -4.0), (4.5, -4.0)),
         3: ((-4.0, -8.0), (4.5, -8.0))}
spawn, goals = [], []
for i in range(args.agents):
    if i in LANES:
        (sx, sy), g = LANES[i]
    else:
        j = i - 4
        sx, sy = -12.0 + (j % 10) * 2.6, -11.0 - (j // 10) * 2.0
        g = (-sx, sy)
    spawn.append((sx, sy, 0.0))
    goals.append((g[0], g[1], 0.0))

UsdGeom.Scope.Define(stage, "/World/Characters")
skel = []
t0 = time.time()
for i in range(args.agents):
    cp = Sdf.Path(f"/World/Characters/Person_{i:02d}")
    omni.kit.commands.execute("CreateBehaviorAgentCommand", character_prim_path=cp,
                              character_asset_url=CHAR_URLS[i % 3],
                              motion_library_prim_path=Sdf.Path("/World/Characters/HumanMotionLibrary"),
                              motion_library_asset_url=MOTION_LIB_URL, motion_library_skeleton_rig="Human",
                              position=Gf.Vec3d(*spawn[i]))
    app.update()
    sk = next((p for p in Usd.PrimRange(stage.GetPrimAtPath(cp)) if p.GetTypeName() == "SkelRoot"), None)
    skel.append(str(sk.GetPath()) if sk else None)
for k in range(1200):
    app.update()
    if k % 50 == 0 and k > 0 and not ctx.get_stage_loading_status()[2]:
        break
report["agents_created_s"] = round(time.time() - t0, 1)

inav = nav.acquire_interface()
t0 = time.time()
report["bake"] = {"ok": inav.start_navmesh_baking_and_wait(), "s": round(time.time() - t0, 2)}
log(f"bake {report['bake']}, agents created in {report['agents_created_s']} s")

# idle baseline cost: timeline stopped, then playing with agents but no tasks
tb = []
for _ in range(60):
    ts = time.time()
    app.update()
    tb.append(time.time() - ts)
report["step_ms_stopped"] = round(1000 * sum(tb) / len(tb), 2)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
agents = [None] * args.agents
for k in range(900):
    app.update()
    for i, p in enumerate(skel):
        if agents[i] is None and p:
            agents[i] = bh.acquire_interface().get_agent(p)
    if all(a is not None for a in agents) and k >= 10:
        break
report["agents_found"] = sum(a is not None for a in agents)
tb = []
for _ in range(60):
    ts = time.time()
    app.update()
    tb.append(time.time() - ts)
report["step_ms_playing_no_tasks"] = round(1000 * sum(tb) / len(tb), 2)
log(f"agents found {report['agents_found']}/{args.agents}; step ms stopped={report['step_ms_stopped']} "
    f"playing-idle={report['step_ms_playing_no_tasks']}")

for i, a in enumerate(agents):
    if a is not None:
        a.set_speed(1.2)
        a.move_to(target=carb.Float3(*goals[i]))

sq = omni.physx.get_physx_scene_query_interface()


def ray(o, d, dist=10.0):
    h = sq.raycast_closest(carb.Float3(*o), carb.Float3(*d), dist)
    if not h or not h.get("hit"):
        return {"hit": False}
    return {"hit": True, "dist": round(h["distance"], 3), "collision": h.get("collision")}


def pos_of(path):
    return r3(omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path)).ExtractTranslation())


def body(i):
    return f"{skel[i]}/Physics/BodyCollider"


start_boxes = {b: pos_of(f"/World/{b}") for b in ["BoxA", "BoxC_Pairs", "BoxD_Groups", "BoxE_Target"]
               + (["BoxB_Robot", "BoxF_Control"] if args.contact_lanes else [])}
mind = {i: 99.0 for i in range(4)}
latdev = {i: 0.0 for i in range(4)}
prev = {}
speeds = {i: [] for i in range(min(4, args.agents))}
robot_x, proxy_x = -6.0, -6.0
t_steps = []
runtime = {}
for f in range(args.frames):
    robot_x += 0.5 / 60.0
    proxy_x = min(6.0, proxy_x + 1.2 / 60.0)
    if args.contact_lanes:
        pfx.SetTranslate(Gf.Vec3d(proxy_x, 9.0, 0.85))
    else:
        rbx.SetTranslate(Gf.Vec3d(robot_x, 2.5, 0.3))
    px.SetTranslate(Gf.Vec3d(proxy_x, 6.0, 0.85))
    if f == 60 and args.contact_lanes:
        # Lane B (contact mode): walker 1's capsule collision off -- passes through? (rays: see the end test)
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(body(1))).CreateCollisionEnabledAttr(False)
        runtime["lane_b_capsule_disabled"] = True
    if f == 60:
        # Lane C: FilteredPairsAPI at runtime, box -> agent 2's capsule
        try:
            fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/World/BoxC_Pairs"))
            fp.CreateFilteredPairsRel().AddTarget(body(2))
            runtime["pairs_authored"] = True
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"pairs: {e!r}")
        # Lane D: collision groups at runtime naming agent 3's capsule explicitly
        try:
            gp = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/RtPeople")
            gp.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(body(3))
            gr = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroups/RtBox")
            gr.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/BoxD_Groups")
            gp.CreateFilteredGroupsRel().AddTarget(gr.GetPath())
            runtime["groups_authored"] = True
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"groups: {e!r}")
    ts = time.time()
    app.update()
    t_steps.append(time.time() - ts)
    boxes = {"BoxA": pos_of("/World/BoxA"), "BoxC_Pairs": pos_of("/World/BoxC_Pairs"),
             "BoxD_Groups": pos_of("/World/BoxD_Groups")}
    for i in range(min(4, args.agents)):
        a = agents[i]
        if a is None:
            continue
        p = a.get_world_translation()
        latdev[i] = max(latdev[i], abs(p.y - spawn[i][1]))
        bpos = pos_of("/World/BoxB_Robot") if args.contact_lanes else (robot_x, 2.5)
        ref = {0: boxes["BoxA"], 1: bpos, 2: boxes["BoxC_Pairs"], 3: boxes["BoxD_Groups"]}[i]
        mind[i] = min(mind[i], math.hypot(p.x - ref[0], p.y - ref[1]))
        if i in prev:
            speeds[i].append(math.hypot(p.x - prev[i][0], p.y - prev[i][1]) * 60.0)
        prev[i] = (p.x, p.y)
    if f % 60 == 0:
        row = {"f": f, "robot_x": round(robot_x, 2), "proxy_x": round(proxy_x, 2), "boxes": boxes,
               "boxE": pos_of("/World/BoxE_Target"),
               "ray_proxy030": ray((proxy_x, 3.0, 0.30), (0.0, 1.0, 0.0)), "agents": {}}
        for i in range(min(4, args.agents)):
            a = agents[i]
            if a is None:
                continue
            p = a.get_world_translation()
            row["agents"][i] = {"pos": r3((p.x, p.y, p.z)),
                                "ray030_side": ray((p.x, p.y - 3.0 if i != 1 else p.y + 3.0, 0.30),
                                                   (0.0, 1.0 if i != 1 else -1.0, 0.0))}
        report["rows"].append(row)
        log(json.dumps(row)[:500])
        save()

# crowd displacement (are the extra walkers actually walking?)
disp = []
for i in range(4, args.agents):
    if agents[i] is not None:
        p = agents[i].get_world_translation()
        disp.append(round(math.hypot(p.x - spawn[i][0], p.y - spawn[i][1]), 2))
report["crowd_displacement_m"] = disp
report["runtime_authoring"] = runtime
report["min_dist_agent_to_obstacle_m"] = {str(k): round(v, 3) for k, v in mind.items()}
report["max_lateral_deviation_m"] = {str(k): round(v, 3) for k, v in latdev.items()}
report["box_start"] = start_boxes
report["box_end"] = {b: pos_of(f"/World/{b}") for b in start_boxes}
report["walk_speed_mps"] = {str(i): {"median": round(sorted(v)[len(v) // 2], 3), "p90": round(sorted(v)[int(len(v) * 0.9)], 3)}
                            for i, v in speeds.items() if v}
report["step_ms"] = {"mean": round(1000 * sum(t_steps) / len(t_steps), 2),
                     "p50": round(1000 * sorted(t_steps)[len(t_steps) // 2], 2), "agents": args.agents}

# Does collisionEnabled=False on a BodyCollider hide it from raycast_closest?
try:
    a0 = agents[0]
    p = a0.get_world_translation()
    before = ray((p.x, p.y - 3.0, 0.30), (0.0, 1.0, 0.0))
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath(body(0))).CreateCollisionEnabledAttr(False)
    for _ in range(5):
        app.update()
    p = a0.get_world_translation()
    after = ray((p.x, p.y - 3.0, 0.30), (0.0, 1.0, 0.0))
    report["collision_disabled_ray"] = {"before": before, "after": after}
except Exception as e:  # noqa: BLE001
    report["errors"].append(f"collision toggle: {e!r}")

save()
log(f"min_dist={report['min_dist_agent_to_obstacle_m']} latdev={report['max_lateral_deviation_m']}")
log(f"box_start={start_boxes} box_end={report['box_end']}")
log(f"speed={report['walk_speed_mps']} step_ms={report['step_ms']} crowd={disp} "
    f"toggle={report.get('collision_disabled_ray')} errors={report['errors']}")
timeline.stop()
app.update()
kit.close()
