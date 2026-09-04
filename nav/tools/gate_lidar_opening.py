"""Would a 2D scan have SEEN the turn the camera missed?

    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/gate_lidar_opening.py --env hospital
    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/gate_lidar_opening.py --env warehouse

THE DECISION THIS GATES, and why it is worth a tool rather than a weekend. Building an
occupancy map, a frontier selector and a channel into the prompt is days of work. The
question underneath all of it takes minutes: at the moments the reference policy went
the wrong way, was the right way VISIBLE TO A LIDAR? If the goal is behind a wall
geometrically, then a scan is as blind as the camera was and nothing built on top of it
can recover those episodes. Three architectures on this branch have already been killed
by a probe like this after the prompt-level version looked obviously right, so the rule
here is to measure the channel before wiring it.

WHY THE CAMERA CANNOT ANSWER IT. `camera_nav` is 90 degrees of horizontal field of view
pointed forward, and the measured failure is `TARGET: not visible` on 65% of all
decisions and 92% of the ones needing a turn past 90 degrees. A turn is needed exactly
where the goal is off to the side or behind -- outside the frame by construction. A 360
degree scan is the one cheap thing that sees there.

WHAT IS MEASURED. Every trace sample of every FAILED reference run, split by how far off
the nose the goal was at that moment:

  * FAILURE MOMENTS (|bearing error| >= --min-err): a turn was needed. The question is
    whether the scan shows drivable floor toward the goal.
  * CONTROL MOMENTS (|bearing error| <= --ctrl-err): straight was already correct. These
    exist to catch a metric that calls everything open -- a rule that fires everywhere
    is not a signal, and this is the same role the open-corridor frames played in
    `probe_arc_repair.py`.

Openness toward a bearing is the MEDIAN free range over a +/-`--sector` wedge, not the
max. A single ray slipping through a gap between two racks is not a corridor the robot
can drive, and the max would report one.

RANGE IS FREE HERE, so all three candidate parts are answered in one pass: every ray is
cast once at the longest range and the shorter ones are derived by thresholding, since a
return at 18 m simply IS a non-return for a 12 m device. That matters because the
warehouse start pose already measured 15.7-16.7 m to the nearest wall, which is outside
an RPLIDAR C1 entirely -- so this run decides the part number as well as the
architecture.

The robot's own prims are skipped. At 0.30 m it occludes nothing from its own centre
(measured: 0 self-hits of 500), but the robot is parked at its authored pose while these
scans are taken from poses all along the trace, so from a distance its wheel colliders
can intercept a ray that in reality would have moved with it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

from episode import env_name, load_episode  # noqa: E402
from summarize_runs import load_config, score  # noqa: E402

STUDY = REPO / "nav/results/hard_study"
ARMS = ("ref_s0", "ref_s1", "ref_s2", "refrep_s0")

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="hospital", help="Environment whose episodes to replay.")
ap.add_argument("--ranges", default="12,25,40", help="Candidate device ranges, metres.")
ap.add_argument("--rays", type=int, default=500, help="RPLIDAR C1 is 500/rev.")
ap.add_argument("--height", type=float, default=0.30, help="Mount height, measured clear.")
ap.add_argument("--sector", type=float, default=20.0, help="Half-width of the wedge, deg.")
ap.add_argument("--open-m", type=float, default=3.0, help="Free range that counts as open.")
ap.add_argument("--min-err", type=float, default=40.0)
ap.add_argument("--ctrl-err", type=float, default=15.0)
ap.add_argument("--max-poses", type=int, default=60, help="Poses sampled per run.")
args = ap.parse_args()

RANGES = sorted(float(r) for r in args.ranges.split(","))
MAXR = RANGES[-1]

cfg = load_config()

# Collect the work before paying for Isaac Sim: which failed runs belong to this
# environment, and where their traces go. A launch that finds nothing to do is a
# three-minute way to learn a one-second fact.
jobs = []
for arm in ARMS:
    for f in sorted((STUDY / arm).glob("*.json")):
        if f.stem == "arm":
            continue
        r = score(f, cfg)
        if r is None or r["success"]:
            continue                       # the gate is about the runs that lost
        ep = f.stem.split("_", 1)[1].rsplit("_", 1)[0]
        if ep not in cfg:
            continue
        # env_name() is a module function over the episode's scene path, not a method:
        # nav stages are keyed on the environment, and several episodes share one.
        try:
            if env_name(load_episode(ep).scene) != args.env:
                continue
        except Exception:
            continue
        trace = json.loads(f.read_text()).get("trace") or []
        if len(trace) < 8:
            continue
        jobs.append({"ep": ep, "arm": arm, "trace": trace,
                     "goal": tuple(cfg[ep]["goal"][:2])})

if not jobs:
    raise SystemExit(f"no failed reference runs on disk for env {args.env!r}")

scene = REPO / "assets" / "usd" / f"nav_{args.env}.usda"
if not scene.is_file():
    raise SystemExit(f"nav scene not built: {scene}")

print(f"\n{len(jobs)} failed runs to replay in {args.env}, "
      f"{args.rays} rays/scan at z+{args.height:.2f} m, up to {MAXR:.0f} m")

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(str(scene.resolve()))
stage = ctx.get_stage()
for _ in range(240):
    kit.update()
stage.Load()
for _ in range(30):
    kit.update()
omni.timeline.get_timeline_interface().play()
for _ in range(60):
    kit.update()
query = get_physx_scene_query_interface()

BEARINGS = [2.0 * math.pi * i / args.rays for i in range(args.rays)]


def scan(x: float, y: float, z: float) -> list[float]:
    """Free range along every bearing, world frame, capped at MAXR."""
    out = []
    for b in BEARINGS:
        d = (math.cos(b), math.sin(b), 0.0)
        r = query.raycast_closest((x, y, z), d, MAXR)
        if r and r.get("hit") and not str(r.get("collision", "")).startswith("/World/Aloha"):
            out.append(float(r["distance"]))
        else:
            out.append(MAXR)
    return out


def median_open(ranges: list[float], centre: float, half: float, cap: float) -> float:
    """Median free range in a wedge, as a device with `cap` metres would report it."""
    vals = []
    for b, d in zip(BEARINGS, ranges):
        off = (b - centre + math.pi) % (2.0 * math.pi) - math.pi
        if abs(off) <= half:
            vals.append(min(d, cap))
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


half = math.radians(args.sector)
# stats[episode][cap] = [fail_open, fail_n, ctrl_open, ctrl_n, fail_goal_beats_straight]
stats: dict[str, dict[float, list[int]]] = {}

for j in jobs:
    trace, (gx, gy) = j["trace"], j["goal"]
    step = max(1, len(trace) // args.max_poses)
    for s in trace[::step]:
        x, y, yaw = float(s[0]), float(s[1]), float(s[2])
        err = math.degrees(math.atan2(gy - y, gx - x) - yaw)
        err = (err + 180.0) % 360.0 - 180.0
        if abs(err) >= args.min_err:
            kind = "fail"
        elif abs(err) <= args.ctrl_err:
            kind = "ctrl"
        else:
            continue
        z = 0.01 + args.height
        ranges = scan(x, y, z)
        goal_b = math.atan2(gy - y, gx - x)
        per = stats.setdefault(j["ep"], {c: [0, 0, 0, 0, 0] for c in RANGES})
        for cap in RANGES:
            og = median_open(ranges, goal_b, half, cap)
            os_ = median_open(ranges, yaw, half, cap)
            row = per[cap]
            if kind == "fail":
                row[1] += 1
                row[0] += og >= args.open_m
                row[4] += og > os_
            else:
                row[3] += 1
                row[2] += os_ >= args.open_m

print(f"\nOPEN = median free range over a +/-{args.sector:.0f} deg wedge "
      f">= {args.open_m:.1f} m\n")
print(f"  {'episode':<30}{'range':>7}{'fail: open to goal':>21}"
      f"{'goal > straight':>18}{'ctrl: open ahead':>19}")
print("  " + "-" * 95)

tot = {c: [0, 0, 0, 0, 0] for c in RANGES}
for ep in sorted(stats):
    for cap in RANGES:
        a, n, c, m, b = stats[ep][cap]
        for i, v in enumerate((a, n, c, m, b)):
            tot[cap][i] += v
        fo = f"{a}/{n} ({a / n:.0%})" if n else "--"
        gb = f"{b}/{n} ({b / n:.0%})" if n else "--"
        co = f"{c}/{m} ({c / m:.0%})" if m else "--"
        print(f"  {ep if cap == RANGES[0] else '':<30}{cap:6.0f}m{fo:>21}{gb:>18}{co:>19}")
print("  " + "-" * 95)
for cap in RANGES:
    a, n, c, m, b = tot[cap]
    fo = f"{a}/{n} ({a / n:.0%})" if n else "--"
    gb = f"{b}/{n} ({b / n:.0%})" if n else "--"
    co = f"{c}/{m} ({c / m:.0%})" if m else "--"
    print(f"  {'ALL' if cap == RANGES[0] else '':<30}{cap:6.0f}m{fo:>21}{gb:>18}{co:>19}")

print("\n\nHOW TO READ IT\n")
print("  fail: open to goal  -- the scan shows drivable floor toward the goal at a moment")
print("                         the camera reported the target not visible. This is the")
print("                         bit the policy was missing; high is the gate passing.")
print("  goal > straight     -- and it is MORE open toward the goal than straight ahead,")
print("                         i.e. the scan does not merely permit the turn, it prefers")
print("                         it. This is the number a frontier selector would act on.")
print("  ctrl: open ahead    -- straight is open when straight was already correct. A")
print("                         metric that scores high on the failure set and LOW here")
print("                         has acquired a turning bias, not a goal channel.")
print("\n  Compare the ranges before ordering: a column that collapses at 12 m and holds")
print("  at 25 m is the C1 being too short for these scenes, not the architecture failing.")

kit.close()
