"""People on a REAL nav stage (Isaac Sim 6.0.1): navmesh bake, DynaNav spawns + commands, robot.

The empty-stage probes (probe_people_agents.py, probe_people_physics.py) settled the mechanics.
This one checks that they survive contact with assets/usd/nav_hospital.usda -- a CDN DynaNav scene
referenced under /World/Environment with the AlohaMini articulation in it -- and is the first cut of
the per-step sequencer nav/sim/people.py will own:

  * navmesh: DynaNav's EFFECTIVE hospital numbers (5.0 bake measured by probe_dynanav_people_5x.py:
    radius 0.40 m, height 1.80 m, step 0.40 m, slope 45 deg) in 6.0.1 key names; bake time; whether
    the NavMeshVolume composed from the referenced layer is found.
  * people: DynaNav's 10 hospital characters at the EXACT spawn points DynaNav's own pipeline produced
    (dn5 run, seed 666), in DynaNav's asset order, running DynaNav's command lists
    (generate_commands.py, seed 666) through a sequencer: GoTo -> move_to (skipped when the navmesh
    has no path, as omni.anim.people did), Idle d / LookAround d -> idle() for d sim seconds. Plus 2
    "crossers" whose straight line runs through the robot, to see whether the crowd sim steers
    around the AlohaMini and whether a capsule shoves it.
  * metrics people.py must produce: per-person live xy (behavior API, NOT USD), min distance to the
    robot, DynaNav's 0.2 m human-collision latch, raycast_closest hits on people at the guard's
    0.30 m height, step cost; plus the robot base displacement (pushed?).
  * frames: top-down overview and the robot's own camera_nav at t = 5/10/15/20/25 s.

  ISAAC_SLOT_TAG=people-navstage ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh \
      ~/robotics/isaacsim-6.0.1/python.sh nav/tools/probe_people_navstage.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
REPO = Path(__file__).resolve().parents[2]
DN = Path("/home/gtu-dsa/robotics/TIC-VLA/DynaNav")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--no-frames", action="store_true")
ap.add_argument("--crossers", choices=["avoid", "push", "ghost"], default="avoid",
                help="avoid: crowd-sim defaults; push: crossers' avoidance OFF so their capsules run into the robot; "
                     "ghost: as push, plus runtime FilteredPairsAPI from every BodyCollider to every robot rigid body")
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
MOTION_LIB_URL = f"{CDN}/People/MotionLibrary/HumanMotionLibrary.usd"
# DynaNav's asset order (IRA 0.7.19: np.random.seed(666) shuffle) -- CONFIRMED by the dn5 run, where
# Character..Character_09 referenced exactly these folders. Same folder names exist on the 6.0 CDN.
DN_ASSETS = ["original_male_adult_construction_02/male_adult_construction_02.usd",
             "original_male_adult_construction_03/male_adult_construction_03.usd",
             "original_female_adult_business_02/female_adult_business_02.usd",
             "original_male_adult_police_04/male_adult_police_04.usd",
             "original_male_adult_construction_01/male_adult_construction_01.usd",
             "original_male_adult_medical_01/male_adult_medical_01.usd",
             "female_adult_police_03_new/female_adult_police_03_new.usd",
             "male_adult_construction_03/male_adult_construction_03.usd",
             "male_adult_construction_05_new/male_adult_construction_05_new.usd",
             "F_Business_02/F_Business_02.usd"]
# DynaNav's own spawn points for hospital, seed 666 (dn5 usd_at_play, Isaac Sim 5.0 + IRA 0.7.19)
DN_SPAWN = [(15.243, 29.244), (-27.299, 9.783), (-22.362, 7.393), (5.374, 9.615), (-29.912, 19.516),
            (-40.33, 14.808), (-4.895, 6.079), (3.896, 11.468), (-5.634, 11.606), (9.502, 1.96)]
ROBOT_PREFIX = "/World/Aloha"
BASE = "/World/Aloha/Geometry/base_link"

report: dict = {"args": vars(args), "errors": [], "rows": []}


def log(m):
    print(f"[navstage] {m}", flush=True)


def save():
    (OUT / "probe_people_navstage.json").write_text(json.dumps(report, indent=1, default=str))


app = omni.kit.app.get_app()
em = app.get_extension_manager()
for ext in ["omni.anim.navigation.core", "omni.anim.behavior.core"]:
    report[f"enable_{ext}"] = bool(em.set_extension_enabled_immediate(ext, True))
app.update()

import NavSchema  # noqa: E402
import omni.anim.behavior.core as bh  # noqa: E402
import omni.anim.navigation.core as nav  # noqa: E402
import omni.physx  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

ctx = omni.usd.get_context()
scene = REPO / "assets/usd/nav_hospital.usda"
ctx.open_stage(str(scene.resolve()))
stage = ctx.get_stage()
t0 = time.time()
for _ in range(240):
    app.update()
stage.Load()
for _ in range(30):
    app.update()
report["stage_load_s"] = round(time.time() - t0, 1)
# Session layer for every edit: assets/usd/nav_*.usda is never modified or saved.
stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))

vols = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(NavSchema.NavMeshVolume)]
report["navmesh_volumes"] = vols
log(f"stage loaded in {report['stage_load_s']} s; NavMeshVolumes: {vols}")

s = carb.settings.get_settings()
NAV = "/exts/omni.anim.navigation.core/navMesh/config"
for k, v in {"agentMinHeight": 180.0, "agentMinRadius": 40.0, "agentMaxRadius": 50.0, "agentMaxStepHeight": 40.0,
             "agentMaxFloorSlope": 45.0, "autoRebakeOnChanges": False, "excludeRigidBodies": True}.items():
    s.set(f"{NAV}/{k}", v)

# Commands: DynaNav's generator, verbatim call (benchmark.py:1380-1387)
sys.path.insert(0, str(DN))
from generate_commands import generate_character_commands  # noqa: E402

cmd_file = OUT / "hospital_commands.txt"
generate_character_commands(targets="hospital", num_characters=10, num_commands=10, seed=666,
                            filename=str(cmd_file), base_name="Character")
names = ["Character"] + [f"Character_{i:02d}" for i in range(1, 10)]
cmds = {n: [] for n in names}
for line in cmd_file.read_text().splitlines():
    parts = line.split()
    cmds[parts[0]].append(parts[1:])

# The robot as placed by the stage (the runner teleports it to the episode start later).
robot0 = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(BASE)).ExtractTranslation()
rx0, ry0 = float(robot0[0]), float(robot0[1])
report["robot_start"] = [round(rx0, 3), round(ry0, 3)]
# two crossers: 5 m either side of the robot along X, walking through its position
if args.crossers == "avoid":
    crossers = [("Crosser_A", (rx0 - 5.0, ry0), (rx0 + 5.0, ry0), "F_Business_02/F_Business_02.usd"),
                ("Crosser_B", (rx0, ry0 + 5.0), (rx0, ry0 - 5.0), "M_Medical_01/M_Medical_01.usd")]
else:
    # both along the corridor (Y), straight through the robot's centre, from opposite ends, 1.5 s apart
    crossers = [("Crosser_A", (rx0, ry0 + 5.0), (rx0, ry0 - 5.0), "F_Business_02/F_Business_02.usd"),
                ("Crosser_B", (rx0 + 0.15, ry0 - 6.5), (rx0 + 0.15, ry0 + 5.0), "M_Medical_01/M_Medical_01.usd")]

UsdGeom.Scope.Define(stage, "/World/People")
people = []  # dicts: name, path, skel, spawn, commands
t0 = time.time()
for i, n in enumerate(names):
    people.append({"name": n, "asset": DN_ASSETS[i], "spawn": DN_SPAWN[i], "cmds": cmds[n]})
for n, a, b, asset in crossers:
    people.append({"name": n, "asset": asset, "spawn": a, "cmds": [["GoTo", f"{b[0]}", f"{b[1]}", "0.0", "_"]]})
for p in people:
    path = Sdf.Path(f"/World/People/{p['name']}")
    omni.kit.commands.execute("CreateBehaviorAgentCommand", character_prim_path=path,
                              character_asset_url=f"{CDN}/People/Characters/{p['asset']}",
                              motion_library_prim_path=Sdf.Path("/World/People/HumanMotionLibrary"),
                              motion_library_asset_url=MOTION_LIB_URL, motion_library_skeleton_rig="Human",
                              position=Gf.Vec3d(p["spawn"][0], p["spawn"][1], 0.0))
    omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=path, api=NavSchema.NavMeshExcludeAPI)
    app.update()
    sk = next((q for q in Usd.PrimRange(stage.GetPrimAtPath(path)) if q.GetTypeName() == "SkelRoot"), None)
    p["skel"] = str(sk.GetPath()) if sk else None
for k in range(1200):
    app.update()
    if k % 50 == 0 and k > 0 and not ctx.get_stage_loading_status()[2]:
        break
report["people_created_s"] = round(time.time() - t0, 1)
report["skelroots_found"] = sum(p["skel"] is not None for p in people)

inav = nav.acquire_interface()
t0 = time.time()
ok = inav.start_navmesh_baking_and_wait()
navmesh = inav.get_navmesh()
report["bake"] = {"ok": ok, "s": round(time.time() - t0, 2), "has_navmesh": navmesh is not None}
if navmesh is not None:
    for fn in ["get_agent_min_height", "get_agent_min_radius", "get_agent_max_step_height", "get_agent_max_slope"]:
        report["bake"][fn] = round(getattr(navmesh, fn)(), 3)
    # are DynaNav's spawn points on the 6.0.1 navmesh?
    snap = []
    for p in people:
        try:
            q, _ = navmesh.query_closest_point(target=carb.Float3(p["spawn"][0], p["spawn"][1], 0.0))
            snap.append(round(math.hypot(q[0] - p["spawn"][0], q[1] - p["spawn"][1]), 3) if q is not None else None)
        except Exception as e:  # noqa: BLE001
            snap.append(f"ERR {e!r}")
    report["spawn_to_navmesh_m"] = dict(zip([p["name"] for p in people], snap))
log(f"bake {report['bake']}; people {report['skelroots_found']}/{len(people)} in {report['people_created_s']} s")
save()

# frames
cams = {}
if not args.no_frames:
    try:
        from isaacsim.sensors.camera import Camera

        sys.path.insert(0, str(REPO / "scripts"))
        from alohamini1_specs import NAV_CAMERA_PRIM_PATH  # noqa: E402

        # Oblique overview (the hospital has a ceiling, so no top-down): 6 m in front of the robot
        # (it faces -Y), 2.4 m up, looking +Y and 25 deg down. A USD camera looks along -Z;
        # rotateX(65 deg) turns that into (0, 0.906, -0.423).
        top = UsdGeom.Camera.Define(stage, "/World/PeopleProbeOverviewCam")
        top.CreateFocalLengthAttr(12.0)
        top.CreateHorizontalApertureAttr(20.955)
        top.CreateVerticalApertureAttr(20.955 * 9 / 16)
        top.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
        ox = UsdGeom.XformCommonAPI(top.GetPrim())
        ox.SetTranslate(Gf.Vec3d(rx0, ry0 - 6.0, 2.4))
        ox.SetRotate(Gf.Vec3f(65.0, 0.0, 0.0))
        cams["overview"] = Camera(prim_path="/World/PeopleProbeOverviewCam", resolution=(960, 540))
        cams["nav"] = Camera(prim_path=NAV_CAMERA_PRIM_PATH, resolution=(960, 540))
        for c in cams.values():
            c.initialize()
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"cameras: {e!r}")
        cams = {}

timeline = omni.timeline.get_timeline_interface()
timeline.play()
pending = {p["name"]: p for p in people if p["skel"]}
for k in range(900):
    app.update()
    for n, p in list(pending.items()):
        a = bh.acquire_interface().get_agent(p["skel"])
        if a is not None:
            p["agent"] = a
            pending.pop(n)
    if not pending and k >= 10:
        break
report["agents_live"] = sum("agent" in p for p in people)
if args.crossers in ("push", "ghost"):
    for p in people:
        if p["name"].startswith("Crosser") and "agent" in p:
            p["agent"].set_auto_avoidance_enabled(False)
            p["agent"].set_obstacle_avoidance_enabled(False)
if args.crossers == "ghost":
    from pxr import UsdPhysics  # noqa: E402

    robot_bodies = [str(q.GetPath()) for q in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PREFIX))
                    if q.HasAPI(UsdPhysics.RigidBodyAPI)]
    n_ok = 0
    for p in people:
        bc = stage.GetPrimAtPath(f"{p['skel']}/Physics/BodyCollider") if p.get("skel") else None
        if bc is not None and bc.IsValid():
            rel = UsdPhysics.FilteredPairsAPI.Apply(bc).CreateFilteredPairsRel()
            for rb in robot_bodies:
                rel.AddTarget(rb)
            n_ok += 1
    report["ghost_filter"] = {"robot_bodies": len(robot_bodies), "capsules_filtered": n_ok}
    log(f"ghost filter: {report['ghost_filter']}")
for _ in range(30):  # camera pipeline warm-up (first renders are empty)
    app.update()

# ---------------- sequencer (the people.py core) ----------------
DT = 1.0 / 60.0
for p in people:
    p.update(ci=-1, t_end=None, task=None, path_m=0.0, last=None, log=[])


def start_next(p, t):
    a = p.get("agent")
    while True:
        p["ci"] += 1
        if p["ci"] >= len(p["cmds"]):
            p["task"] = a.idle() if a else None  # number_of_loop = 0: idle forever
            p["t_end"] = None
            p["log"].append((round(t, 2), "DONE"))
            return
        c = p["cmds"][p["ci"]]
        if c[0] == "GoTo":
            tgt = (float(c[1]), float(c[2]), float(c[3]))
            cur = a.get_world_translation()
            path = navmesh.query_shortest_path(start_pos=carb.Float3(cur.x, cur.y, cur.z),
                                               end_pos=carb.Float3(*tgt)) if navmesh else None
            if path is None:  # omni.anim.people: no path -> GoTo completes at once
                p["log"].append((round(t, 2), f"GoTo {tgt[:2]} SKIPPED (no path)"))
                continue
            a.set_speed(1.1)  # DynaNav walkers measured ~0.0183 m/update = 1.1 m/s at 60 Hz
            p["task"] = a.move_to(target=carb.Float3(*tgt))
            p["t_end"] = None
            p["log"].append((round(t, 2), f"GoTo {tgt[:2]}"))
            return
        d = float(c[1])  # Idle / LookAround (no LookAround clip in the 6.0 motion library)
        p["task"] = a.idle()
        p["t_end"] = t + d
        p["log"].append((round(t, 2), f"{c[0]} {d}"))
        return


sq = omni.physx.get_physx_scene_query_interface()


def people_ray_hits(cx, cy, n=72, z=0.30, rng=12.0, r0=0.45):
    hits = {}
    for k in range(n):
        th = 2 * math.pi * k / n
        d = (math.cos(th), math.sin(th), 0.0)
        o = (cx + d[0] * r0, cy + d[1] * r0, z)
        h = sq.raycast_closest(carb.Float3(*o), carb.Float3(*d), rng)
        if h and h.get("hit"):
            c = str(h.get("collision", ""))
            if c.startswith("/World/People/"):
                nm = c.split("/")[3]
                hits[nm] = min(hits.get(nm, 99.0), round(h["distance"] + r0, 2))
    return hits


t_steps = []
min_d = {p["name"]: 99.0 for p in people}
dn_collision = {}
frames_at = {int(round(x * 60)) for x in (5, 10, 15, 20, 25) if x < args.seconds}
nframes = int(args.seconds * 60)
for p in people:
    if "agent" in p:
        start_next(p, 0.0)
for f in range(nframes):
    t = f * DT
    ts = time.time()
    app.update()
    t_steps.append(time.time() - ts)
    bm = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(BASE))
    b = bm.ExtractTranslation()
    rbx, rby = float(b[0]), float(b[1])
    up = bm.TransformDir(Gf.Vec3d(0, 0, 1))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2] / max(1e-9, up.GetLength())))))
    report["robot_max_tilt_deg"] = max(report.get("robot_max_tilt_deg", 0.0), round(tilt, 2))
    report["robot_max_disp_m"] = max(report.get("robot_max_disp_m", 0.0),
                                     round(math.hypot(rbx - rx0, rby - ry0), 4))
    for p in people:
        a = p.get("agent")
        if a is None:
            continue
        q = a.get_world_translation()
        if p["last"] is not None:
            p["path_m"] += math.hypot(q.x - p["last"][0], q.y - p["last"][1])
        p["last"] = (q.x, q.y)
        dd = math.hypot(q.x - rbx, q.y - rby)
        min_d[p["name"]] = min(min_d[p["name"]], dd)
        if dd < 0.2 and p["name"] not in dn_collision:  # DynaNav benchmark.py:694-701
            dn_collision[p["name"]] = round(t, 2)
        # advance the command list
        if p["t_end"] is not None:
            if t >= p["t_end"]:
                a.cancel_task(p["task"])
                start_next(p, t)
        elif p["task"] is not None and p["ci"] < len(p["cmds"]):
            st = a.get_task_status(p["task"])
            if st != bh.BehaviorTaskStatus.RUNNING:
                p["log"].append((round(t, 2), f"  -> {str(st).split('.')[-1]}"))
                start_next(p, t)
    if f % 120 == 0:
        row = {"t": round(t, 2), "robot": [round(rbx, 3), round(rby, 3)],
               "people": {p["name"]: [round(p["last"][0], 2), round(p["last"][1], 2)] for p in people if p["last"]},
               "ray_hits_030": people_ray_hits(rbx, rby)}
        report["rows"].append(row)
        log(json.dumps({k: row[k] for k in ("t", "robot", "ray_hits_030")}))
        save()
    if f in frames_at and cams:
        try:
            from PIL import Image

            for nm, c in cams.items():
                rgba = c.get_rgba()
                if rgba is not None and rgba.size:
                    Image.fromarray(rgba[:, :, :3].astype("uint8")).save(OUT / f"{nm}_t{int(round(t)):02d}.jpg", quality=88)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"frame {f}: {e!r}")

b = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(BASE)).ExtractTranslation()
report["robot_end"] = [round(float(b[0]), 3), round(float(b[1]), 3)]
report["robot_displacement_m"] = round(math.hypot(float(b[0]) - rx0, float(b[1]) - ry0), 4)
report["min_dist_to_robot_m"] = {k: round(v, 3) for k, v in min_d.items()}
report["dynanav_human_collision_first_t"] = dn_collision
report["path_m"] = {p["name"]: round(p["path_m"], 2) for p in people}
report["command_log"] = {p["name"]: p["log"] for p in people}
report["step_ms"] = {"mean": round(1000 * sum(t_steps) / len(t_steps), 2),
                     "p50": round(1000 * sorted(t_steps)[len(t_steps) // 2], 2),
                     "max": round(1000 * max(t_steps), 2), "people": len(people), "cameras": len(cams)}
save()
log(f"robot displacement {report['robot_displacement_m']} m (max {report.get('robot_max_disp_m')}, "
    f"max tilt {report.get('robot_max_tilt_deg')} deg); min dist {report['min_dist_to_robot_m']}")
log(f"DN collisions {dn_collision}; path {report['path_m']}")
log(f"step_ms {report['step_ms']} errors {report['errors']}")
timeline.stop()
app.update()
kit.close()
