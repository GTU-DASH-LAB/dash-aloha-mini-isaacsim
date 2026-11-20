"""Replay a recorded run's trace through the simulated RPLIDAR C1, one pose at a time, at rest.

    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/replay_lidar_trace.py \\
        --result nav/results/20260905-024542_hospital_vending_machine_braking.json \\
        --every 2 --max-poses 8 --frames

WHAT IT IS FOR. Two consumers, and both need the same thing: what the 2D lidar returned
at the poses a real run actually visited.

  1. EXPLAINER MEDIA. A top-down "how the lidar sees" animation needs per-beam ranges,
     world hit points and the robot pose, and is far more convincing next to the robot's
     own `camera_nav` frame and a third-person `camera_chase` frame of the same instant.
  2. GATE G-a: "does a C1 ray along the TRUE goal bearing hit the target object or an
     occluder?" For that, every pose also records the goal's bearing and distance from the
     sensor, and the returns (range, prim) of every C1 beam within +/-3 deg of it.

WHY A NEW TOOL AND NOT THE GATES. `gate_lidar_opening.py` / `gate_lidar_blindside.py`
cast their own world-frame ray loop from a robot PARKED at its authored pose and skip
`/World/Aloha` hits for that reason (gate_lidar_opening.py:42-45, :149). That is right for
their question and wrong for this one: here the robot is TELEPORTED to each trace pose, so
its own colliders are where they really were and a self-hit is a true return of the real
mounted sensor, exactly as `lidar_2d.py:52-60` models it. And it reads the sensor through
`SweepingLidar2D` -- the very object the runner feeds the guard and the arc menu with
(run_navigation.py:649-650, :1135) -- rather than a parallel re-implementation.

CODE PATHS REUSED, so a difference between this and a live run can only be the motion:

  * Stage: `assets/usd/nav_<env>.usda`, keyed on `episode.env_name(ep.scene)`
    (run_navigation.py:315-317), opened by ABSOLUTE path with the same 240-update /
    `stage.Load()` / 30-update / `timeline.play()` sequence (run_navigation.py:619-632,
    gate_lidar_opening.py:127-138). Scene queries only see colliders while the timeline
    plays (check_lidar_2d.py:86-87).
  * Robot: `Articulation("/World/Aloha/Geometry/base_link")` + `KinematicBase`
    (run_navigation.py:634-639). Teleport with `KinematicBase.reset_to(snapped, yaw)`,
    `KinematicBase.settle()` (30 held updates), `sync_from_sim()` -- the runner's own
    episode-start sequence (run_episode). z is `ground_spawn_pose`'s floor under the
    tyres at each pose, probed from the previous pose's floor; it used to be the episode's
    z at every pose, which outdoors is DynaNav's Nova Carter height (0.14-2.21 m above the
    terrain), so those scans came from a falling robot. Results written before 2026-10-05
    evening (no `sampling.spawn` key) used that, which matters for the OUTDOOR rows of the
    P2 gates (CLAUDE.md, "G-a ... Outdoor gives 0/7"); indoor z_start is the floor.
    PhysX GPU capacities are raised as the runner does (physics_setup.py).
  * Lidar: `SweepingLidar2D(mount_height_m=...)` (lidar_2d.py:247-358), stepped once with
    `dt = 1 / scan_rate_hz`, i.e. exactly one revolution (500 rays) with the robot still,
    from `position = base.position()` and `base.yaw` as the runner passes them
    (run_navigation.py:834, :1135). The per-beam buffers are read directly (`_hit`, `_self`,
    `_prim`, `_wx`, `_wy`, `_stamp`) because `points_body()`/`prims()` drop misses and
    self-hits, and per-beam data is the point here.
  * Cameras: `NavCameraSource` (camera_nav, 1920x1080) with its 30-update warmup, and the
    chase camera via `try_create_chase()` -- created lazily, and only with --frames,
    because once an Isaac Sim Camera exists Kit renders it every frame
    (camera_source.py:12-15, run_navigation.py:641-642, :454-473).
  * Goal: `episode.load_episode(result["episode"]).goal` from nav/config/episodes.yaml --
    the result JSON carries no goal of its own. Cross-checked against the result's own
    `initial_distance_m` measured from `trace[0]` (run_navigation.py:726).

TRACE CLOCK. The trace stores no timestamps. The runner appends the start pose, then one
sample every 30 physics steps at 1/60 s (run_navigation.py:106, :751, :1218-1219), then the
final pose (run_navigation.py:1246). So t_i = 0.5 * i for every sample but the last, which
is at `elapsed_s`. This is CHECKED per result (`trace_clock_ok`) rather than assumed.

OUTPUT (default nav/results/replay/<result-stem>/):
  replay.npz    per pose p and beam i (500 beams, 0 = robot forward, CCW positive):
                  t, pose_trace (x, y, yaw wrapped), pose_sim (x, y, z, yaw after settle),
                  origin (sensor xyz), bearings_body, ranges (NaN = no return),
                  hit_world (x, y; NaN = no return), prim (collision prim path, "" = no
                  return), self_hit, goal_xy, goal_rel_bearing (rad, body frame),
                  goal_dist (from the sensor origin), goal_window (beams within
                  +/- --goal-window-deg), goal_beam (nearest beam), goal_exact_range /
                  goal_exact_prim (ONE reference ray exactly along the bearing -- not a C1
                  beam, which only exists every 0.72 deg), and trace_full / trace_t.
                Strings are fixed-width unicode, so `np.load` needs no allow_pickle.
  summary.json  run identity, stage used, sensor, timings, and per pose the goal-window
                beams as (beam, bearing, range, prim, hit-to-goal distance). With
                --frames also `cameras` per pose and `camera_check` overall: each
                camera's measured world heading against the lidar's yaw, its pitch and
                its mount offset in the body frame -- the independent check that lidar
                beam 0 is where camera_nav looks (see camera_pose()).
  frames/       nav_<k>.jpg (camera_nav) and chase_<k>.jpg, with --frames.

LIMITATIONS, stated so nobody has to rediscover them:
  * AT REST. The beam really sweeps over ~6 physics steps while the robot moves
    (lidar_2d.py:257-261); here the robot is still, so the 100 ms motion smear and the
    staleness of a live buffer are absent. This is a geometric snapshot of each pose.
  * Z is not in the trace. The robot is placed at `episode.start` z and allowed to settle,
    as the runner does at the start line; a run whose base drifted vertically (the known
    kinematic-base blow-ups) is NOT reproduced.
  * Arm and lift joint state is the stage's default, not the run's. The nav runner never
    commands them, so this matches every nav run on record.
  * The episode goal is a FLOOR point (z ~ 0.01), not an object. G-a must decide from the
    prim path and the hit-to-goal distance which return is "the target"; this tool does
    not guess.
  * Ground truth geometry: no noise, no dropouts, and glass is whatever collider PhysX
    has, the opposite of a real DTOF unit (lidar_2d.py:35-50).
  * One environment per process: Isaac Sim cannot swap stages in-process at this version
    (run_navigation.py:522-534), so results from two environments are refused up front.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# nav/tools/ must NOT be importable from here. Python puts a script's own directory first
# on sys.path, and nav/tools/profile.py then shadows the stdlib `profile` module -- which
# warp imports (`_pyprofile.run`) while Kit starts, so omni.warp.core, and with it
# isaacsim.core.simulation_manager / isaacsim.core.prims, fail to load and Kit cannot
# start. Measured, not guessed: first smoke run died with "module 'profile' has no
# attribute 'run'". Nothing here imports from nav/tools, so drop it (and a cwd entry
# pointing at it) before anything else is imported.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
for _p in (REPO / "nav" / "sim", REPO / "scripts"):
    sys.path.insert(0, str(_p))

from episode import env_name, load_episode  # noqa: E402

# The runner's clock, copied with citations rather than imported: importing
# run_navigation pulls in the policy client and controllers for two constants.
PHYSICS_DT = 1.0 / 60.0        # run_navigation.py:106
TRACE_EVERY_STEPS = 30         # run_navigation.py:1218  (`if step % 30 == 0`)
TRACE_DT = TRACE_EVERY_STEPS * PHYSICS_DT   # 0.5 s
SETTLE_STEPS = 30              # run_navigation.py:718 / :1337, after every teleport
DEFAULT_HEIGHT_M = 0.30        # run_navigation.py:122 (NAV_LIDAR_HEIGHT), lidar_2d.py:276


def wrap(a: float) -> float:
    """Into (-pi, pi]. The on-disk trace yaw is the integrated, UNWRAPPED heading."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def trace_times(n: int, elapsed_s: float) -> tuple[list[float], bool]:
    """Sim time of each trace sample, and whether the result agrees with that clock.

    Start pose at t=0, one sample per TRACE_DT, then the final pose at `elapsed_s`. The
    check is that the last regular sample sits no later than `elapsed_s` and no more than
    one period before it; a result written under a different cadence fails it.
    """
    if n <= 1:
        return [0.0] * n, n == 1
    times = [TRACE_DT * i for i in range(n - 1)] + [float(elapsed_s)]
    gap = float(elapsed_s) - times[-2]
    return times, (-1e-6 <= gap <= TRACE_DT + 1e-6)


def pick(times: list[float], every: float, max_poses: int | None) -> list[int]:
    """Indices spaced at least `every` seconds apart, from t=0, capped as a PREFIX.

    A prefix rather than a spread on purpose: the explainer media want uniform time
    spacing, and spreading N poses over a long trace would silently change `--every`.
    Raise --every to cover a whole episode in fewer poses.
    """
    out: list[int] = []
    next_t = 0.0
    for i, t in enumerate(times):
        if t + 1e-6 >= next_t:
            out.append(i)
            next_t = t + every
    return out[:max_poses] if max_poses else out


def camera_pose(source, base_xyz, yaw: float) -> dict | None:
    """Where a camera ACTUALLY is and looks, read back from the sim, in the robot's frame.

    This is the independent check on the lidar's frame convention. The lidar turns body
    bearings into world rays as `bearing + base.yaw` (lidar_2d.py:336); camera_nav is a
    child of base_link, so its view direction is the robot's physical heading and owes
    nothing to that arithmetic. heading_err ~ 0 at a non-zero yaw means beam 0 really is
    "where the camera looks". It also measures the mount: the offsets authored in
    scripts/pipeline/add_cameras.py:134 (nav: 0.25, 0, 0.346) and :152-153 (chase: -2.5,
    0, 1.8, pitched 27.5 deg down) should come back unchanged in the body frame.

    `camera_axes="usd"`: a USD camera looks down its local -Z, so the view vector is
    minus the third column of the rotation (isaacsim.sensors.camera Camera.get_world_pose,
    quaternion scalar-first).
    """
    try:
        pos, q = source._camera.get_world_pose(camera_axes="usd")
    except Exception:  # noqa: BLE001 -- a diagnostic must never cost the replay
        return None
    pos = np.asarray(pos, dtype=np.float64).ravel()
    w, qx, qy, qz = (float(v) for v in np.asarray(q, dtype=np.float64).ravel())
    vx = -2.0 * (qx * qz + w * qy)
    vy = -2.0 * (qy * qz - w * qx)
    vz = -(1.0 - 2.0 * (qx * qx + qy * qy))
    n = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    heading = math.atan2(vy, vx)
    dx, dy, dz = pos[0] - base_xyz[0], pos[1] - base_xyz[1], pos[2] - base_xyz[2]
    c, s = math.cos(yaw), math.sin(yaw)
    return {
        "heading_deg": round(math.degrees(heading), 3),
        "heading_err_deg": round(math.degrees(wrap(heading - yaw)), 3),
        "pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, vz / n)))), 3),
        # Mount offset in the BODY frame (x forward, y left, z up from the base origin).
        "mount_body_m": [round(dx * c + dy * s, 4), round(-dx * s + dy * c, 4), round(dz, 4)],
        "world_z_m": round(float(pos[2]), 4),
    }


def lidar_height_for(result: dict, override: float | None) -> tuple[float, str]:
    if override is not None:
        return override, "--height"
    label = str(result.get("lidar") or "")
    if label.startswith("c1@"):
        try:
            return float(label[3:].split("+")[0]), f"result lidar label {label!r}"
        except ValueError:
            pass
    return DEFAULT_HEIGHT_M, (f"default (result lidar label {label!r} is not a C1 run; "
                              "the replay still uses the C1 it would have had)")


# ----------------------------------------------------------------- argument parsing
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--result", nargs="+", required=True,
                help="One or more nav/results/*.json, all from ONE environment.")
ap.add_argument("--every", type=float, default=1.0,
                help="Sim seconds between sampled trace poses (trace cadence is 0.5 s).")
ap.add_argument("--frames", action="store_true",
                help="Also save a camera_nav and a camera_chase JPEG per pose.")
ap.add_argument("--max-poses", type=int, default=None,
                help="Cap on poses per result, taken as a prefix of the sampled poses.")
ap.add_argument("--indices", default=None,
                help="JSON file mapping a result path (or its stem) to the exact trace "
                     "indices to replay. Overrides --every/--max-poses for the results it "
                     "names, so a probe can render the poses it chose, not a fixed grid.")
ap.add_argument("--out", default=None,
                help="Output dir (default nav/results/replay/<result-stem>/). With several "
                     "results it is a parent and each gets <out>/<result-stem>/.")
ap.add_argument("--height", type=float, default=None,
                help="Mount height above the base, m. Default: from the result's "
                     "`lidar` label if it is c1@<h>, else 0.30.")
ap.add_argument("--goal-window-deg", type=float, default=3.0,
                help="Half-width of the beam window around the true goal bearing.")
ap.add_argument("--settle", type=int, default=SETTLE_STEPS,
                help="kit.update() calls after each teleport (runner uses 30).")
args = ap.parse_args()
chosen: dict[str, list[int]] = {}
if args.indices:
    for key, ids in json.loads(Path(args.indices).read_text()).items():
        chosen[Path(key).name.removesuffix(".json")] = sorted({int(i) for i in ids})

# ------------------------------------------------- everything checkable before Kit
jobs = []
for raw in args.result:
    path = Path(raw).resolve()
    if not path.is_file():
        raise SystemExit(f"no such result: {path}")
    res = json.loads(path.read_text())
    trace = res.get("trace") or []
    if not trace or any(len(s) < 3 for s in trace):
        raise SystemExit(f"{path.name}: trace missing or not (x, y, yaw) -- cannot replay")
    try:
        ep = load_episode(res["episode"])
    except KeyError as exc:
        raise SystemExit(f"{path.name}: {exc}") from None
    env = env_name(ep.scene)
    stage_path = REPO / "assets" / "usd" / f"nav_{env}.usda"
    if not stage_path.is_file():
        raise SystemExit(f"nav scene not built: {stage_path}\n  nav/sim/build_nav_scene.sh {env}")
    times, clock_ok = trace_times(len(trace), float(res.get("elapsed_s", 0.0)))
    idx = pick(times, args.every, args.max_poses)
    if path.stem in chosen:
        idx = chosen[path.stem]
        bad = [i for i in idx if not 0 <= i < len(trace)]
        if bad:
            raise SystemExit(f"{path.name}: --indices outside the trace (0..{len(trace) - 1}): {bad}")
    height, height_src = lidar_height_for(res, args.height)
    d0 = math.dist(trace[0][:2], ep.goal[:2])
    jobs.append({
        "path": path, "res": res, "ep": ep, "env": env, "stage": stage_path,
        "trace": trace, "times": times, "clock_ok": clock_ok, "idx": idx,
        "height": height, "height_src": height_src,
        "goal_check_m": abs(d0 - float(res.get("initial_distance_m", d0))),
    })

envs = sorted({j["env"] for j in jobs})
if len(envs) != 1:
    raise SystemExit(f"results span environments {envs}; Isaac Sim cannot swap stages "
                     f"in-process (run_navigation.py:526-529). Run once per environment.")
if len({str(j["stage"]) for j in jobs}) != 1:
    raise SystemExit("results map to different stage files")

for j in jobs:
    stem = j["path"].stem
    if args.out is None:
        j["out"] = REPO / "nav" / "results" / "replay" / stem
    elif len(jobs) == 1:
        j["out"] = Path(args.out).resolve()
    else:
        j["out"] = Path(args.out).resolve() / stem
    if not j["clock_ok"]:
        print(f"[replay] WARNING {j['path'].name}: trace length {len(j['trace'])} does not "
              f"match elapsed_s={j['res'].get('elapsed_s')} at {TRACE_DT} s/sample -- "
              f"t values are approximate", flush=True)
    if j["goal_check_m"] > 0.05:
        print(f"[replay] WARNING {j['path'].name}: episodes.yaml goal is "
              f"{j['goal_check_m']:.2f} m off the result's own initial_distance_m -- the "
              f"episode definition may have changed since the run", flush=True)
    print(f"[replay] {j['path'].name}: {len(j['idx'])} poses of {len(j['trace'])} "
          f"(t={j['times'][j['idx'][0]]:.1f}..{j['times'][j['idx'][-1]]:.1f} s of "
          f"{j['times'][-1]:.1f} s), lidar z+{j['height']:.2f} m ({j['height_src']})",
          flush=True)

STAGE = jobs[0]["stage"]

# --------------------------------------------------------------------------- Kit
t_launch = time.time()
from isaacsim import SimulationApp  # noqa: E402

# GPU0 through Kit's own config, never CUDA_VISIBLE_DEVICES (run_navigation.py:589-602).
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
t_app = time.time()

exit_code = 1
try:
    import omni.timeline  # noqa: E402
    import omni.usd  # noqa: E402
    from isaacsim.core.prims import Articulation  # noqa: E402

    from base_drive import KinematicBase, ground_spawn_pose  # noqa: E402
    from lidar_2d import RPLIDAR_C1, SweepingLidar2D  # noqa: E402
    from physics_setup import raise_gpu_capacities  # noqa: E402

    ctx = omni.usd.get_context()
    ctx.open_stage(str(STAGE.resolve()))           # ABSOLUTE, run_navigation.py:616-619
    stage = ctx.get_stage()
    for _ in range(240):                             # CDN streaming, run_navigation.py:621-627
        kit.update()
    stage.Load()
    for _ in range(30):
        kit.update()
    t_stage = time.time()

    for line in raise_gpu_capacities(stage):     # as run_navigation.setup() does
        print(f"[replay] physics: {line}", flush=True)
    omni.timeline.get_timeline_interface().play()
    for _ in range(10):
        kit.update()
    art = Articulation("/World/Aloha/Geometry/base_link")
    art.initialize()
    base = KinematicBase(art, list(art.dof_names))
    base.sync_from_sim()

    nav_cam = chase = None
    if args.frames:
        from camera_source import NavCameraSource, try_create_chase  # noqa: E402

        # scratch_dir is mandatory in the constructor and is mkdir'd there; frames are
        # written by this script under its own names via grab(), never grab_to_file().
        nav_cam = NavCameraSource(scratch_dir=jobs[0]["out"] / "frames")
        chase = try_create_chase()
        if chase is None:
            print("[replay] WARNING: stage has no camera_chase prim; chase frames skipped",
                  flush=True)
        nav_cam.warmup(kit)
    t_ready = time.time()
    startup = {
        "simulation_app_s": round(t_app - t_launch, 2),
        "stage_open_load_s": round(t_stage - t_app, 2),
        "play_articulation_cameras_s": round(t_ready - t_stage, 2),
        "total_s": round(t_ready - t_launch, 2),
    }
    print(f"[replay] Kit ready in {startup['total_s']:.1f} s ({startup})", flush=True)

    spec = RPLIDAR_C1
    N = spec.points_per_rev
    win = math.radians(args.goal_window_deg)
    z_start = None

    for j in jobs:
        out = j["out"]
        out.mkdir(parents=True, exist_ok=True)
        if args.frames:
            (out / "frames").mkdir(exist_ok=True)
        ep, trace, times = j["ep"], j["trace"], j["times"]
        gx, gy = float(ep.goal[0]), float(ep.goal[1])
        z_start = float(ep.start[2])
        z_ref = z_start          # where the spawn probe looks from; follows the floor
        no_floor = 0             # poses with no floor found under the probe
        P = len(j["idx"])

        t_arr = np.zeros(P)
        pose_trace = np.zeros((P, 3))
        pose_sim = np.zeros((P, 4))
        origin = np.zeros((P, 3))
        ranges = np.full((P, N), np.nan, dtype=np.float32)
        hit_world = np.full((P, N, 2), np.nan, dtype=np.float64)
        prims = np.full((P, N), "", dtype=object)
        self_hit = np.zeros((P, N), dtype=bool)
        goal_rel = np.zeros(P)
        goal_dist = np.zeros(P)
        goal_window = np.zeros((P, N), dtype=bool)
        goal_beam = np.zeros(P, dtype=np.int32)
        goal_exact_range = np.full(P, np.nan)
        goal_exact_prim = np.full(P, "", dtype=object)
        per_pose = []
        timing = []
        bearings = None

        for k, i in enumerate(j["idx"]):
            x, y, yaw_raw = (float(v) for v in trace[i][:3])
            yaw = wrap(yaw_raw)
            t = float(times[i])
            tp0 = time.time()

            # --- teleport exactly as the runner puts the robot on a start line: z snapped
            # to the floor under the tyres, then a held settle (run_episode). The trace has
            # no z, and the episode's own z is DynaNav's Nova Carter height -- up to 2.21 m
            # above the terrain outdoors, so a scan taken at "z_start" there was taken from
            # a robot still falling. The probe starts 0.30 m above the PREVIOUS pose's floor
            # (the episode z for the first), so it follows terrain that climbs between
            # samples (<= 0.75 m apart) instead of starting underground; with no floor found
            # it falls back to z_start as before, and says so.
            # The robot is still standing at the previous pose, <= 0.75 m away, i.e. inside
            # the probe's footprint columns: lift it out of the way first, exactly as
            # run_navigation._spawn_pose does, or a ray stepping out of its own tyre lands
            # under the floor it rests on (17 mm low, or a floor metres down; base_drive).
            base.park_for_probe(kit)
            snapped, snap = ground_spawn_pose((x, y, z_ref), yaw)
            if snap["ground_z"] is None:
                snapped = (x, y, z_start)
                no_floor += 1
            else:
                z_ref = snapped[2]
            base.reset_to(snapped, yaw)
            base.settle(kit, args.settle, PHYSICS_DT)
            base.sync_from_sim()
            pos = base.position()
            yaw_sim = base.yaw
            tp1 = time.time()

            # --- one full revolution, at rest
            # A `c1@<h>` label is a run from before the column was modelled, when the unit
            # sat at the centre and looked through it; `c1@<h>+<x>` is the mounted sensor.
            # Replay each the way it ran, or the replay is not of that run.
            label = str(j["res"].get("lidar") or "")
            fwd = float(label.split("+")[1]) if label.startswith("c1@") and "+" in label else 0.0
            lidar = SweepingLidar2D(spec=spec, mount_height_m=j["height"],
                                    mount_forward_m=fwd, model_column="+" in label)
            cast = lidar.step(pos, yaw_sim, 1.0 / spec.scan_rate_hz, t)
            if cast != N or not np.all(lidar._stamp == t):
                raise RuntimeError(f"pose {k}: expected one revolution of {N} rays, "
                                   f"cast {cast}")
            bearings = lidar.bearings_body.astype(np.float64)
            ox, oy, oz = pos[0], pos[1], pos[2] + lidar.mount_height_m
            hit = lidar._hit.copy()
            r = np.hypot(lidar._wx - ox, lidar._wy - oy)
            ranges[k, hit] = r[hit]
            hit_world[k, hit, 0] = lidar._wx[hit]
            hit_world[k, hit, 1] = lidar._wy[hit]
            self_hit[k] = lidar._self & hit
            for b in np.flatnonzero(hit):
                prims[k, b] = lidar._prim[b]

            # --- G-a: the true goal, seen from the sensor
            gb_world = math.atan2(gy - oy, gx - ox)
            rel = wrap(gb_world - yaw_sim)
            dist = math.hypot(gx - ox, gy - oy)
            off = (bearings - rel + math.pi) % (2.0 * math.pi) - math.pi
            goal_window[k] = np.abs(off) <= win
            goal_beam[k] = int(np.argmin(np.abs(off)))
            # Reference ray EXACTLY along the bearing, same interface, same min-range rule
            # (lidar_2d.py:344-348). Not a C1 beam -- it shows what the 0.72 deg
            # quantisation can and cannot miss.
            ex = lidar._scene_query().raycast_closest(
                (ox, oy, oz), (math.cos(gb_world), math.sin(gb_world), 0.0),
                spec.max_range_m)
            if ex and ex.get("hit") and float(ex.get("distance", 1e9)) >= spec.min_range_m:
                goal_exact_range[k] = float(ex["distance"])
                goal_exact_prim[k] = str(ex.get("collision", ""))
            tp2 = time.time()

            # --- frames, and where each camera really was when it took them
            frame_names = {}
            cams = {}
            if args.frames:
                cams["nav"] = camera_pose(nav_cam, pos, yaw_sim)
                if chase is not None:
                    cams["chase"] = camera_pose(chase, pos, yaw_sim)
                from PIL import Image

                img = nav_cam.grab()
                if img is not None:
                    name = f"nav_{k:03d}.jpg"
                    Image.fromarray(img).save(out / "frames" / name, quality=90)
                    frame_names["nav"] = f"frames/{name}"
                if chase is not None:
                    img = chase.grab()
                    if img is not None:
                        name = f"chase_{k:03d}.jpg"
                        Image.fromarray(img).save(out / "frames" / name, quality=90)
                        frame_names["chase"] = f"frames/{name}"
            tp3 = time.time()

            t_arr[k] = t
            pose_trace[k] = (x, y, yaw)
            pose_sim[k] = (pos[0], pos[1], pos[2], yaw_sim)
            origin[k] = (ox, oy, oz)
            goal_rel[k] = rel
            goal_dist[k] = dist

            ext = hit & ~self_hit[k]
            window = []
            for b in np.flatnonzero(goal_window[k]):
                rb = float(ranges[k, b]) if hit[b] else None
                window.append({
                    "beam": int(b),
                    "bearing_deg": round(math.degrees(float(bearings[b])), 2),
                    "off_goal_deg": round(math.degrees(float(off[b])), 2),
                    "range_m": round(rb, 3) if rb is not None else None,
                    "prim": prims[k, b] or None,
                    "self": bool(self_hit[k, b]),
                    # How far the return lands from the goal point. Small = the ray ends
                    # at the goal; a return well SHORT of goal_dist is something in
                    # between. Interpretation is G-a's, not this tool's.
                    "short_of_goal_m": round(dist - rb, 3) if rb is not None else None,
                    "hit_to_goal_m": (round(math.hypot(hit_world[k, b, 0] - gx,
                                                       hit_world[k, b, 1] - gy), 3)
                                      if rb is not None else None),
                })
            per_pose.append({
                "k": k, "trace_index": i, "t": round(t, 2),
                "trace_pose": [round(x, 4), round(y, 4), round(yaw, 4)],
                "sim_pose": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4),
                             round(yaw_sim, 4)],
                "pose_err_m": round(math.hypot(pos[0] - x, pos[1] - y), 4),
                "yaw_err_deg": round(math.degrees(wrap(yaw_sim - yaw)), 3),
                "returns": int(hit.sum()), "self_hits": int(self_hit[k].sum()),
                "external_min_m": (round(float(np.nanmin(ranges[k][ext])), 3)
                                   if ext.any() else None),
                "external_median_m": (round(float(np.nanmedian(ranges[k][ext])), 3)
                                      if ext.any() else None),
                "goal_rel_deg": round(math.degrees(rel), 2),
                "goal_dist_m": round(dist, 3),
                "goal_beam": int(goal_beam[k]),
                "goal_exact_ray": {
                    "range_m": (round(float(goal_exact_range[k]), 3)
                                if np.isfinite(goal_exact_range[k]) else None),
                    "prim": goal_exact_prim[k] or None,
                },
                "goal_window": window,
                "frames": frame_names,
                "cameras": cams,
            })
            timing.append({"teleport_settle_s": round(tp1 - tp0, 3),
                           "lidar_s": round(tp2 - tp1, 4),
                           "frames_s": round(tp3 - tp2, 3),
                           "total_s": round(tp3 - tp0, 3)})

            nb = int(goal_beam[k])
            nb_r = f"{ranges[k, nb]:6.2f}m" if hit[nb] else "  none "
            print(f"[replay] {k:3d} t={t:5.1f}s ({x:7.2f},{y:7.2f},{math.degrees(yaw):7.1f}d)"
                  f" err={per_pose[-1]['pose_err_m']:.3f}m"
                  f" ret={int(hit.sum()):3d}/{N} self={int(self_hit[k].sum())}"
                  f" goal {math.degrees(rel):+7.1f}d @{dist:5.2f}m -> beam {nb} {nb_r}"
                  f" {(prims[k, nb] or '').rsplit('/', 1)[-1][:40]}"
                  + (f" cam_nav {cams['nav']['heading_err_deg']:+.2f}d"
                     f" mount {cams['nav']['mount_body_m']}" if cams.get("nav") else "")
                  + f"  [{timing[-1]['total_s']:.2f}s]", flush=True)

        # --- write BEFORE anything else: code after kit.close() never runs
        trace_full = np.array([[float(s[0]), float(s[1]), wrap(float(s[2]))]
                               for s in trace])
        np.savez_compressed(
            out / "replay.npz",
            t=t_arr, pose_trace=pose_trace, pose_sim=pose_sim, origin=origin,
            bearings_body=bearings, ranges=ranges, hit_world=hit_world,
            prim=prims.astype(str), self_hit=self_hit,
            goal_xy=np.array([gx, gy]), goal_rel_bearing=goal_rel, goal_dist=goal_dist,
            goal_window=goal_window, goal_beam=goal_beam,
            goal_exact_range=goal_exact_range, goal_exact_prim=goal_exact_prim.astype(str),
            trace_full=trace_full, trace_t=np.array(times),
        )
        prim_counts: dict[str, int] = {}
        for pp in per_pose:
            for w in pp["goal_window"]:
                if w["prim"]:
                    prim_counts[w["prim"]] = prim_counts.get(w["prim"], 0) + 1
        per_pose_s = [tm["total_s"] for tm in timing]
        camera_check = {}
        for name in ("nav", "chase"):
            got = [pp["cameras"][name] for pp in per_pose if pp["cameras"].get(name)]
            if got:
                camera_check[name] = {
                    "poses": len(got),
                    "max_abs_heading_err_deg": max(abs(c["heading_err_deg"]) for c in got),
                    "pitch_deg": [min(c["pitch_deg"] for c in got),
                                  max(c["pitch_deg"] for c in got)],
                    "mount_body_m_mean": [round(float(np.mean([c["mount_body_m"][a]
                                                               for c in got])), 4)
                                          for a in range(3)],
                    "authored": ({"translate": [0.25, 0.0, 0.346], "pitch_deg": 0.0,
                                  "source": "scripts/pipeline/add_cameras.py:134"}
                                 if name == "nav" else
                                 {"translate": [-2.5, 0.0, 1.8], "pitch_deg": -27.5,
                                  "source": "scripts/pipeline/add_cameras.py:152-153"}),
                }
        summary = {
            "tool": "nav/tools/replay_lidar_trace.py",
            "result": str(j["path"]),
            "episode": ep.name,
            "environment": j["env"],
            "stage": str(j["stage"]),
            "instruction": j["res"].get("instruction"),
            "scored": (str(j["res"].get("instruction", "")).strip()
                       == ep.instruction.strip()),
            "run": {k_: j["res"].get(k_) for k_ in (
                "success", "elapsed_s", "final_distance_m", "initial_distance_m",
                "controller", "policy", "lidar", "plan_mode")},
            "goal": {"xy": [gx, gy], "source": "nav/config/episodes.yaml via "
                     "episode.load_episode", "initial_distance_check_m":
                     round(j["goal_check_m"], 4),
                     "success_threshold_m": ep.success_threshold_m},
            "sensor": {"name": spec.name, "points_per_rev": N,
                       "angular_resolution_deg": spec.angular_resolution_deg,
                       "range_m": [spec.min_range_m, spec.max_range_m],
                       "mount_height_m": j["height"], "height_source": j["height_src"],
                       "class": "SweepingLidar2D, one revolution at rest"},
            "sampling": {"every_s": args.every, "max_poses": args.max_poses,
                         "trace_dt_s": TRACE_DT, "trace_clock_ok": j["clock_ok"],
                         "trace_samples": len(trace), "poses": P,
                         "settle_updates": args.settle, "z_start": z_start,
                         "spawn": "ground_spawn_pose per pose, held settle",
                         "poses_without_floor": no_floor},
            "goal_window_deg": args.goal_window_deg,
            "goal_window_prims": dict(sorted(prim_counts.items(),
                                             key=lambda kv: -kv[1])),
            "timing": {"startup": startup,
                       "per_pose_mean_s": round(float(np.mean(per_pose_s)), 3),
                       "per_pose_max_s": round(float(np.max(per_pose_s)), 3),
                       "per_pose": timing},
            "frames": bool(args.frames),
            "camera_check": camera_check,
            "npz_keys": ["t", "pose_trace", "pose_sim", "origin", "bearings_body",
                         "ranges", "hit_world", "prim", "self_hit", "goal_xy",
                         "goal_rel_bearing", "goal_dist", "goal_window", "goal_beam",
                         "goal_exact_range", "goal_exact_prim", "trace_full", "trace_t"],
            "poses": per_pose,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[replay] wrote {out / 'replay.npz'} and summary.json "
              f"({P} poses, {np.mean(per_pose_s):.2f} s/pose mean)", flush=True)

    exit_code = 0
    print("[replay] REPLAY OK", flush=True)
except Exception as exc:  # noqa: BLE001 -- reported, and the exit code carries it
    import traceback

    traceback.print_exc()
    print(f"[replay] REPLAY FAILED: {exc}", flush=True)

# Last statement on purpose: nothing after SimulationApp.close() executes (CLAUDE.md), and
# `exit_code` is what close() hands to os._exit on the fast-shutdown path.
kit.close(exit_code=exit_code)
