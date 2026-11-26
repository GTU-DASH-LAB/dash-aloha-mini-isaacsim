"""Prototype check for nav/sim/people.py on a real nav stage (Isaac Sim 6.0.1, GPU0 only).

Opens assets/usd/nav_<env>.usda IN MEMORY (session-layer edits only, never saved), puts the
AlohaMini on an episode's start pose, spawns people with nav/sim/people.py, and runs
--seconds of simulation:

  * t in [0, --drive-after): robot parked (articulation must not move or tilt);
  * t >= --drive-after: robot teleport-driven (tensor API, like KinematicBase) straight
    toward the episode goal at --drive-speed, stopping for non-people geometry within
    0.6 m (a 7-ray fan at 0.30 m, like the collision guard), NOT for people -- to see
    whether walkers avoid a moving kinematic robot and what the DynaNav latch does;
  * records camera_nav (the policy's camera), camera_chase and a fixed overview camera
    at --fps, per-person trajectories every 0.5 s, a 72-ray ring at 0.30 m around the
    robot every 1 s (do our raycasts see people?), per-person 4-side ray checks every 5 s,
    GPU memory of THIS process (nvidia-smi) and wall-clock cost per step;
  * then (--reset-check S) teleports robot + people back to the start with
    PeopleSim.reset() and runs S seconds again, to compare against the first S seconds
    (within-process repeatability of reset()).

--num-people 0 runs the identical loop without people (cost baseline; people.py is not
even imported into the sim). Compose the video afterwards WITHOUT Kit:

  python3 nav/tools/try_people.py --compose <out_dir> [--video <mp4>]
  python3 nav/tools/try_people.py --compare <run_a.json> <run_b.json>

Launch (always through the slot wrapper):
  ISAAC_SLOT_TAG=people-try ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh \\
      ~/robotics/isaacsim-6.0.1/python.sh nav/tools/try_people.py --out <dir> > <dir>.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# nav/tools/profile.py shadows the stdlib `profile`, which Kit's warp imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
REPO = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--env", default="hospital")
ap.add_argument("--episode", default="hospital_vending_machine")
ap.add_argument("--num-people", type=int, default=None,
                help="default: DynaNav's count for the episode (episodes_full.yaml); 3 if that is 0")
ap.add_argument("--seed", type=int, default=666)
ap.add_argument("--spawn", default="near", choices=["near", "dynanav", "sample"])
ap.add_argument("--interaction", default="ghost", choices=["ghost", "solid"])
ap.add_argument("--seconds", type=float, default=40.0)
ap.add_argument("--fps", type=float, default=6.0)
ap.add_argument("--no-record", action="store_true")
ap.add_argument("--early-attach", action="store_true",
                help="call PeopleSim.attach() right after the robot settles (default: lazily inside the first step)")
ap.add_argument("--drive-after", type=float, default=20.0, help="<0: never drive")
ap.add_argument("--drive-speed", type=float, default=0.5)
ap.add_argument("--reset-check", type=float, default=15.0)
ap.add_argument("--nav-res", default="1920x1080")
ap.add_argument("--ray-every", type=float, default=5.0,
                help="seconds between the per-person 4-side ray check (own-capsule hits at 0.30 m), >= 1")
ap.add_argument("--late-enable", action="store_true",
                help="enable the people extensions AFTER opening the stage (isolates the run-A crash)")
ap.add_argument("--compose", metavar="DIR")
ap.add_argument("--video")
ap.add_argument("--compare", nargs=2, metavar="JSON")
args = ap.parse_args()


# ============================================================ offline modes (no Kit)
def compose(d: Path, video: Path | None) -> None:
    from PIL import Image, ImageDraw, ImageFont

    meta = [json.loads(ln) for ln in (d / "frames.jsonl").read_text().splitlines() if ln.strip()]
    rep = json.loads((d / "try_people.json").read_text())
    comp = d / "composite"
    comp.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    W, TOP, BOT = 1280, 720, 480
    n = 0
    for m in meta:
        canvas = Image.new("RGB", (W, TOP + BOT + 40), (18, 18, 18))
        panels = [("nav", (0, 40), (W, TOP), "camera_nav: the policy's camera (90 deg HFOV, 0.35 m)"),
                  ("chase", (0, 40 + TOP), (640, BOT), "camera_chase (behind the robot)"),
                  ("overview", (640, 40 + TOP), (640, BOT), "fixed overview, looking back at the start")]
        for key, (x, y), size, label in panels:
            f = d / "frames" / f"{key}_{m['i']:05d}.jpg"
            if f.exists():
                canvas.paste(Image.open(f).convert("RGB").resize(size), (x, y))
            dr = ImageDraw.Draw(canvas)
            dr.rectangle([x, y, x + 8 + dr.textlength(label, font=small), y + 24], fill=(0, 0, 0))
            dr.text((x + 4, y + 3), label, fill=(255, 255, 255), font=small)
        dr = ImageDraw.Draw(canvas)
        near = f"{m['min_d']:.2f} m ({m['nearest']})" if m.get("min_d") is not None else "-"
        txt = (f"t={m['t']:5.1f} s   people={rep['num_people']} ({rep['spawn']}, seed {rep['seed']})   "
               f"robot {m['mode']}   nearest person {near}   DynaNav 0.2 m latch: "
               f"{'YES' if m.get('latch') else 'no'}")
        dr.text((10, 8), txt, fill=(255, 230, 120), font=font)
        canvas.save(comp / f"c_{n:05d}.jpg", quality=90)
        n += 1
    fps = rep["args"]["fps"]
    video = video or d / "hospital_people.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(comp / "c_%05d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(video)]
    r = subprocess.run(cmd)
    print(f"composed {n} frames -> {video} (ffmpeg rc {r.returncode})")
    plot_tracks(d, rep)


def plot_tracks(d: Path, rep: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, tr in rep["traj"].items():
        xs, ys = [r[1] for r in tr], [r[2] for r in tr]
        ax.plot(xs, ys, "-", lw=1.5, label=name)
        ax.plot(xs[0], ys[0], "o", ms=6, color=ax.lines[-1].get_color())
        ax.plot(xs[-1], ys[-1], "s", ms=6, color=ax.lines[-1].get_color())
    rt = rep["robot_traj"]
    ax.plot([r[1] for r in rt], [r[2] for r in rt], "k-", lw=3, label="AlohaMini base")
    ax.plot(*rep["goal"][:2], "r*", ms=14, label="episode goal")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"people tracks ({rep['env']}, {rep['episode']}, {rep['seconds']} s; o start, s end)")
    fig.savefig(d / "tracks.png", dpi=110, bbox_inches="tight")
    print(f"tracks -> {d / 'tracks.png'}")


def compare(a: Path, b: Path) -> None:
    A, B = json.loads(a.read_text()), json.loads(b.read_text())
    out = {"a": str(a), "b": str(b), "per_person": {}}
    worst = 0.0
    for name, ta in A["traj"].items():
        tb = B["traj"].get(name)
        if not tb:
            continue
        ib = {round(r[0], 2): r for r in tb}
        diffs = [(r[0], math.hypot(r[1] - ib[round(r[0], 2)][1], r[2] - ib[round(r[0], 2)][2]))
                 for r in ta if round(r[0], 2) in ib]
        mx = max(diffs, key=lambda x: x[1]) if diffs else (None, None)
        first = next((t for t, dd in diffs if dd > 0.05), None)
        out["per_person"][name] = {"samples": len(diffs), "max_diff_m": round(mx[1], 4) if diffs else None,
                                   "at_t": mx[0], "first_t_over_5cm": first}
        worst = max(worst, mx[1] or 0.0)
    out["max_diff_m"] = round(worst, 4)
    print(json.dumps(out, indent=1))


if args.compose:
    compose(Path(args.compose), Path(args.video) if args.video else None)
    sys.exit(0)
if args.compare:
    compare(Path(args.compare[0]), Path(args.compare[1]))
    sys.exit(0)
if not args.out:
    ap.error("--out is required for a simulation run")

# ============================================================ simulation run
OUT = Path(args.out)
(OUT / "frames").mkdir(parents=True, exist_ok=True)
(OUT / "frames.jsonl").unlink(missing_ok=True)
report: dict = {"args": vars(args), "errors": [], "notes": [], "mem_mib": {}, "rows": []}
T_START = time.time()


def log(m: str) -> None:
    print(f"[try_people {time.time() - T_START:7.1f}s] {m}", flush=True)


def save() -> None:
    (OUT / "try_people.json").write_text(json.dumps(report, indent=1, default=str))


def gpu_mem_mib() -> int | None:
    """This process's GPU memory (MiB) per nvidia-smi; Kit's python3 is os.getpid()."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        for ln in r.stdout.splitlines():
            pid, mem = [x.strip() for x in ln.split(",")[:2]]
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:  # noqa: BLE001
        return None
    return None


def mem(tag: str) -> None:
    report["mem_mib"][tag] = gpu_mem_mib()
    log(f"GPU mem [{tag}] = {report['mem_mib'][tag]} MiB")


import yaml  # noqa: E402

ep = yaml.safe_load((REPO / "nav/config/episodes.yaml").read_text())["episodes"][args.episode]
START = tuple(float(v) for v in ep["start"])
YAW = math.radians(float(ep["start_yaw_deg"]))
GOAL = tuple(float(v) for v in ep["goal"])
report.update(env=args.env, episode=args.episode, start=START, start_yaw_deg=float(ep["start_yaw_deg"]),
              goal=GOAL, seconds=args.seconds, seed=args.seed, spawn=args.spawn)

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
mem("kit_started")

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

sys.path.insert(0, str(REPO / "nav" / "sim"))
sys.path.insert(0, str(REPO / "scripts"))
from alohamini1_specs import CHASE_CAMERA_PRIM_PATH, NAV_CAMERA_PRIM_PATH  # noqa: E402

app = omni.kit.app.get_app()
WANT_PEOPLE = args.num_people is None or args.num_people > 0
if WANT_PEOPLE and not args.late_enable:
    import people as _people_mod  # noqa: E402

    _people_mod.enable_extensions(app)   # BEFORE open_stage (see people.enable_extensions)
    report["extensions_enabled"] = "before open_stage"
ctx = omni.usd.get_context()
scene = (REPO / f"assets/usd/nav_{args.env}.usda").resolve()
t0 = time.time()
ctx.open_stage(str(scene))
stage = ctx.get_stage()
for _ in range(240):
    app.update()
stage.Load()
for _ in range(30):
    app.update()
report["stage_load_s"] = round(time.time() - t0, 1)
stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))   # never touch nav_*.usda
log(f"stage {scene.name} loaded in {report['stage_load_s']} s")
mem("stage_loaded")

# ---------------------------------------------------------------- people
n_people = args.num_people
if n_people is None:
    import people as _p  # noqa: E402

    dn = _p.episode_num_people(args.episode)
    report["dynanav_num_people"] = dn
    n_people = dn if dn else 3
    if not dn:
        report["notes"].append(f"DynaNav's {args.episode} has num_people={dn}; using 3 for the prototype")
report["num_people"] = n_people
ppl = None
if n_people > 0:
    import people  # noqa: E402

    t0 = time.time()
    ppl = people.setup(stage, args.env, num_people=n_people, seed=args.seed, spawn=args.spawn,
                       center=START, spawn_radius_m=(2.0, 7.0), spawn_heading_rad=YAW,
                       spawn_sector_rad=math.radians(70.0), robot_interaction=args.interaction,
                       app=app, log=lambda m: log(f"[people] {m}"))
    report["people_setup_s"] = round(time.time() - t0, 1)
    report["command_lines"] = ppl.command_lines
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    mem("people_built")
save()

# ---------------------------------------------------------------- cameras
sq = omni.physx.get_physx_scene_query_interface()


def ray(o, d, rng):
    h = sq.raycast_closest(carb.Float3(*o), carb.Float3(*d), rng)
    if h and h.get("hit"):
        return float(h["distance"]), str(h.get("collision", ""))
    return None, None


cams = {}
REC_EVERY = max(1, int(round(60.0 / args.fps)))
if not args.no_record:
    from isaacsim.sensors.camera import Camera  # noqa: E402

    # Overview: ahead of the start, looking back toward it, below the ceiling.
    ox, oy, oz = START[0] + 7.0 * math.cos(YAW), START[1] + 7.0 * math.sin(YAW), 2.8   # z refined after play
    cam = UsdGeom.Camera.Define(stage, "/World/TryPeopleOverviewCam")
    cam.CreateFocalLengthAttr(10.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 3 / 4)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))
    xf = UsdGeom.XformCommonAPI(cam.GetPrim())
    xf.SetTranslate(Gf.Vec3d(ox, oy, oz))
    # USD camera looks down -Z. rotateX(60) -> looks +Y, 30 deg down; rotateZ(yaw+90) -> +Y turns to -heading.
    xf.SetRotate(Gf.Vec3f(60.0, 0.0, math.degrees(YAW) + 90.0))
    w, h = (int(v) for v in args.nav_res.split("x"))
    cams["nav"] = Camera(prim_path=NAV_CAMERA_PRIM_PATH, resolution=(w, h))
    cams["chase"] = Camera(prim_path=CHASE_CAMERA_PRIM_PATH, resolution=(640, 480))
    cams["overview"] = Camera(prim_path="/World/TryPeopleOverviewCam", resolution=(640, 480))
    for c in cams.values():
        c.initialize()

# Before play: the PhysX GPU capacities every entry point that drives AlohaMini raises
# (nav/sim/physics_setup.py), in the session layer so the stage file is never touched.
from physics_setup import raise_gpu_capacities  # noqa: E402

for _line in raise_gpu_capacities(stage):
    print(f"[physics] {_line}")

# ---------------------------------------------------------------- play, robot to start
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(10):
    app.update()
from isaacsim.core.prims import Articulation  # noqa: E402

art = Articulation("/World/Aloha/Geometry/base_link")
art.initialize()
NDOF = len(art.dof_names)
REST_Z = 0.0078   # base_link rest height above the floor (alohamini1_specs.BASE_LINK_REST_Z_M)


def teleport(x, y, yaw, vx=0.0, vy=0.0):
    art.set_world_poses(positions=np.array([[x, y, START[2] - 0.01 + REST_Z]], dtype=np.float32),
                        orientations=np.array([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]], dtype=np.float32))
    art.set_velocities(np.array([[vx, vy, 0, 0, 0, 0]], dtype=np.float32))
    if vx == 0.0 and vy == 0.0:
        art.set_joint_velocities(np.zeros((1, NDOF), dtype=np.float32))


def robot_pose():
    p, q = art.get_world_poses()
    p, q = p[0], q[0]                              # wxyz
    w, x, y, z = (float(v) for v in q)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    upz = 1 - 2 * (x * x + y * y)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, upz))))
    return float(p[0]), float(p[1]), float(p[2]), yaw, tilt


teleport(START[0], START[1], YAW)
if cams:   # scene queries only work while simulating: keep the overview camera under the ceiling
    ceil, cprim = ray((ox, oy, 0.5), (0.0, 0.0, 1.0), 20.0)
    oz = min(2.8, (0.5 + ceil - 0.3) if ceil else 2.8)
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/TryPeopleOverviewCam")).SetTranslate(Gf.Vec3d(ox, oy, oz))
    report["overview_cam"] = {"xyz": [round(ox, 2), round(oy, 2), round(oz, 2)], "ceiling_hit_m": ceil,
                              "ceiling_prim": cprim}
for _ in range(60):   # settle + camera pipeline warm-up (first renders are empty)
    app.update()
rx0, ry0, rz0, _, _ = robot_pose()
if ppl is not None and args.early_attach:
    t_att = time.time()
    ppl.attach()          # the one-off ghost-filter stall happens here, not inside the first step()
    report["attach_s"] = round(time.time() - t_att, 2)
    log(f"people attached in {report['attach_s']} s")
report["robot_settled"] = [round(rx0, 4), round(ry0, 4), round(rz0, 4)]
mem("after_play")
log(f"robot settled at {report['robot_settled']}")


def ring_people_hits(cx, cy, n=72, z=0.30, rng=12.0, r0=0.45):
    hits = {}
    for k in range(n):
        th = 2 * math.pi * k / n
        d = (math.cos(th), math.sin(th), 0.0)
        dist, prim = ray((cx + d[0] * r0, cy + d[1] * r0, z), d, rng)
        if prim and prim.startswith("/World/People/"):
            nm = prim.split("/")[3]
            hits[nm] = min(hits.get(nm, 99.0), round(dist + r0, 2))
    return hits


def ring_compare(cx, cy, n=72, z=0.30, rng=12.0, r0=0.45):
    """PhysX raycast_closest vs the analytic people test (PeopleSim.ray_distances_2d) on one ring.

    Per ray, restricted to rays where either one reports a person: 'agree' (both, within 3 cm),
    'physx_stale' (both hit a person but > 3 cm apart: PhysX is looking at an old pose),
    'physx_blind' (analytic hits a person, PhysX hits nothing or something nearer/other),
    'physx_phantom' (PhysX hits a person's capsule, analytic hits none or a different one).
    """
    import numpy as _np

    angs = [2 * math.pi * k / n for k in range(n)]
    ana, who = ppl.ray_distances_2d((cx, cy), angs, rng + r0, z=z)
    c = {"agree": 0, "physx_stale": 0, "physx_blind": 0, "physx_phantom": 0}
    bad = {}
    for k, th in enumerate(angs):
        d = (math.cos(th), math.sin(th), 0.0)
        dist, prim = ray((cx + d[0] * r0, cy + d[1] * r0, z), d, rng)
        px_person = bool(prim and prim.startswith("/World/People/"))
        an = math.isfinite(float(ana[k])) and ana[k] >= r0
        if not px_person and not an:
            continue
        who_name = ppl.people[int(who[k])].name if an else None
        if px_person and an:
            if prim.split("/")[3] == who_name and abs((dist + r0) - float(ana[k])) <= 0.03:
                c["agree"] += 1
            else:
                c["physx_stale"] += 1
                bad[prim.split("/")[3]] = bad.get(prim.split("/")[3], 0) + 1
        elif an:
            hit_other = prim is not None and dist + r0 < float(ana[k]) - 0.03
            if not hit_other:
                c["physx_blind"] += 1
                bad[who_name] = bad.get(who_name, 0) + 1
        else:
            c["physx_phantom"] += 1
            bad[prim.split("/")[3]] = bad.get(prim.split("/")[3], 0) + 1
    return c, bad


def fan_blocked(cx, cy, heading, z=0.30, stop=0.6, r0=0.45):
    """Guard-style 7-ray fan over +-35 deg; True if NON-people geometry is within `stop`."""
    for k in range(7):
        th = heading + math.radians(-35 + k * 70 / 6)
        d = (math.cos(th), math.sin(th), 0.0)
        dist, prim = ray((cx + d[0] * r0, cy + d[1] * r0, z), d, stop)
        if dist is not None and not prim.startswith("/World/People/") and not prim.startswith("/World/Aloha"):
            return True
    return False


# ---------------------------------------------------------------- main loop
DT = 1.0 / 60.0


def run_segment(seconds: float, tag: str, record: bool, drive_after: float):
    traj: dict[str, list] = {}
    robot_traj = []
    upd_ms, ppl_ms, cap_ms = [], [], []
    stats = {"max_disp_parked_m": 0.0, "max_tilt_deg": 0.0, "z_min": 9.0, "z_max": -9.0,
             "max_cmd_err_drive_m": 0.0, "max_root_speed_parked": 0.0, "drive_stopped_by_fan_t": None}
    latch_t = None
    cmd_x, cmd_y = START[0], START[1]
    gx, gy = GOAL[0] - START[0], GOAL[1] - START[1]
    gn = math.hypot(gx, gy)
    ux, uy = gx / gn, gy / gn
    driving, stopped = False, False
    fmeta = open(OUT / "frames.jsonl", "a") if record else None
    nsteps = int(round(seconds * 60))
    wall0 = time.time()
    for f in range(nsteps):
        t = f * DT
        if drive_after >= 0 and t >= drive_after and not stopped:
            if not driving:
                driving = True
                log(f"[{tag}] t={t:.2f} robot starts driving toward the goal at {args.drive_speed} m/s")
            if fan_blocked(cmd_x, cmd_y, math.atan2(uy, ux)) or math.hypot(GOAL[0] - cmd_x, GOAL[1] - cmd_y) < 1.0:
                stopped = True
                stats["drive_stopped_by_fan_t"] = round(t, 2)
                teleport(cmd_x, cmd_y, YAW)
                log(f"[{tag}] t={t:.2f} drive stopped (obstacle/goal) at ({cmd_x:.2f},{cmd_y:.2f})")
            else:
                cmd_x += ux * args.drive_speed * DT
                cmd_y += uy * args.drive_speed * DT
                teleport(cmd_x, cmd_y, YAW, ux * args.drive_speed, uy * args.drive_speed)
        ts = time.perf_counter()
        app.update()
        upd_ms.append((time.perf_counter() - ts) * 1000)
        x, y, z, yaw, tilt = robot_pose()
        stats["max_tilt_deg"] = max(stats["max_tilt_deg"], tilt)
        stats["z_min"], stats["z_max"] = min(stats["z_min"], z), max(stats["z_max"], z)
        if driving:
            stats["max_cmd_err_drive_m"] = max(stats["max_cmd_err_drive_m"], math.hypot(x - cmd_x, y - cmd_y))
        else:
            stats["max_disp_parked_m"] = max(stats["max_disp_parked_m"], math.hypot(x - rx0, y - ry0))
            if f % 6 == 0:
                v = art.get_velocities()[0]
                stats["max_root_speed_parked"] = max(stats["max_root_speed_parked"],
                                                     float(np.linalg.norm(v[:3])))
        ts = time.perf_counter()
        if ppl is not None:
            ppl.step(dt=DT, robot_xy=(x, y), robot_yaw=yaw)
            if latch_t is None and ppl.m["human_collision"]:
                latch_t = ppl.m["human_collision_t"]
        ppl_ms.append((time.perf_counter() - ts) * 1000)
        if f % 30 == 0:
            ts_ = round(t, 2)
            robot_traj.append([ts_, round(x, 4), round(y, 4), round(z, 4), round(yaw, 4), round(tilt, 3)])
            if ppl is not None:
                for name, px, py, pz, vx, vy in ppl.positions():
                    traj.setdefault(name, []).append([ts_, round(px, 4), round(py, 4), round(pz, 4),
                                                      round(vx, 3), round(vy, 3)])
        if f % 60 == 0 and ppl is not None:
            row = {"seg": tag, "t": round(t, 2), "robot": [round(x, 3), round(y, 3)],
                   "ring_030": ring_people_hits(x, y),
                   "people": {s["name"]: s for s in ppl.snapshot()}}
            if f % max(60, int(round(args.ray_every * 60))) == 0:
                # own hits out of 4 rays, plus what the missing rays hit instead (a wall or a
                # table between the origin and the person is not a failure to see the person)
                row["ray_visibility"] = {k: {"own": v["own_hits"], "at_spawn": v.get("own_hits_at_spawn"),
                                             "other": sorted({str(r["prim"]).rsplit("/", 1)[-1]
                                                              for r in v["rays"] if not r["own"] and r["prim"]})}
                                         for k, v in ppl.ray_visibility().items()}
            if f % 60 == 0:
                row["ring_compare"], row["ring_compare_by_person"] = ring_compare(x, y)
            if ppl is not None and f in (60, 600, 1200):
                row["capsules"] = ppl.capsule_report()
            report["rows"].append(row)
        if record and cams and f % REC_EVERY == 0:
            ts = time.perf_counter()
            from PIL import Image

            i = len(report.setdefault("frame_times", []))
            for nm, c in cams.items():
                rgba = c.get_rgba()
                if rgba is not None and rgba.size:
                    Image.fromarray(rgba[:, :, :3].astype("uint8")).save(OUT / "frames" / f"{nm}_{i:05d}.jpg",
                                                                         quality=88)
            near = None
            if ppl is not None and ppl.positions():
                near = min(((math.hypot(px - x, py - y), name) for name, px, py, *_ in ppl.positions()))
            fmeta.write(json.dumps({"i": i, "t": round(t, 2), "mode": "driving" if driving and not stopped else
                                    ("stopped" if stopped else "parked"),
                                    "min_d": round(near[0], 2) if near else None, "nearest": near[1] if near else None,
                                    "latch": latch_t is not None}) + "\n")
            report["frame_times"].append(round(t, 2))
            cap_ms.append((time.perf_counter() - ts) * 1000)
        if f % 600 == 0:
            log(f"[{tag}] t={t:.1f} robot=({x:.2f},{y:.2f}) tilt={tilt:.2f} "
                f"upd={sum(upd_ms[-600:]) / len(upd_ms[-600:]):.1f} ms"
                + (f" min_d={ppl.m['min_dist_m']:.2f}" if ppl is not None else ""))
            report.setdefault("mem_series", []).append([tag, round(t, 1), gpu_mem_mib()])
            save()
    wall = time.time() - wall0
    if fmeta:
        fmeta.close()
    cap_steps = set(range(0, nsteps, REC_EVERY)) if record and cams else set()
    upd_nocap = [u for k, u in enumerate(upd_ms) if k not in cap_steps] or upd_ms
    stats = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}
    cost = {"sim_s": round(seconds, 2), "wall_s": round(wall, 2), "wall_per_sim_s": round(wall / seconds, 3),
            "update_ms_mean": round(sum(upd_ms) / len(upd_ms), 2),
            "update_ms_p50": round(sorted(upd_ms)[len(upd_ms) // 2], 2),
            "update_ms_mean_noncapture_steps": round(sum(upd_nocap) / len(upd_nocap), 2),
            "people_step_ms_mean": round(sum(ppl_ms) / len(ppl_ms), 3),
            "capture_ms_mean": round(sum(cap_ms) / len(cap_ms), 1) if cap_ms else None,
            "captures": len(cap_ms)}
    return traj, robot_traj, stats, cost


traj, robot_traj, stats, cost = run_segment(args.seconds, "main", not args.no_record, args.drive_after)
report.update(traj=traj, robot_traj=robot_traj, robot=stats, cost=cost)
mem("after_main")
if ppl is not None:
    report["people_metrics"] = ppl.metrics()
log(f"main: robot {stats}; cost {cost}")
save()

# ---------------------------------------------------------------- reset check
if ppl is not None and args.reset_check > 0:
    teleport(START[0], START[1], YAW)
    for _ in range(30):
        app.update()
    rx0, ry0, rz0, _, _ = robot_pose()
    ppl.reset()
    traj2, robot_traj2, stats2, cost2 = run_segment(args.reset_check, "reset", False, -1.0)
    diffs = {}
    for name, tr in traj2.items():
        ref = {r[0]: r for r in traj.get(name, [])}
        dd = [(r[0], math.hypot(r[1] - ref[r[0]][1], r[2] - ref[r[0]][2])) for r in tr if r[0] in ref]
        if dd:
            mx = max(dd, key=lambda v: v[1])
            diffs[name] = {"samples": len(dd), "max_diff_m": round(mx[1], 4), "at_t": mx[0],
                           "first_t_over_5cm": next((t for t, v in dd if v > 0.05), None)}
    report["reset_check"] = {"seconds": args.reset_check, "vs_main_first_seconds": diffs, "robot": stats2,
                             "cost": cost2, "traj": traj2, "metrics": ppl.metrics()}
    log(f"reset check: {json.dumps({k: v['max_diff_m'] for k, v in diffs.items()})}")
    save()

report["wall_total_s"] = round(time.time() - T_START, 1)
mem("end")
save()
log(f"done; errors={report['errors']}")
if ppl is not None:
    ppl.teardown()
timeline.stop()
app.update()
kit.close()
