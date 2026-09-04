"""Does the scan report openings the CAMERA CANNOT SEE -- and are they the right ones?

    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/gate_lidar_blindside.py --env hospital
    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/gate_lidar_blindside.py --env warehouse

WHAT THE FIRST GATE SETTLED AND WHAT IT DID NOT. `gate_lidar_opening.py` established
that the turn was physically available (goal wedge drivable on 55% of hospital and 83%
of warehouse failure moments) and that openness does NOT prefer the goal over straight
ahead (40% / 43% against a control that holds at 81% / 65%). That killed a purely
geometric frontier selector. It did not test the surviving idea, and it could not: its
wedge is centred on the goal bearing, which the robot does not have -- the instruction
names a landmark, not coordinates.

THE SURVIVING IDEA. `camera_nav` is 90 degrees of horizontal field of view pointed
forward. The measured failure is `TARGET: not visible` on 65% of decisions and 92% of
those needing a turn past 90 degrees, because a turn is needed exactly where the goal is
off to the side or behind -- outside the frame by construction. So the scan's job is not
to CHOOSE; it is to tell the VLM about drivable openings that lie outside the frame,
which the VLM then reasons about with the instruction it alone holds.

THE MECHANISM USES NO GOAL KNOWLEDGE. Openings are contiguous sectors of free floor
found in the scan alone. The goal bearing enters only to GRADE them afterwards, the same
separation `EpisodeResult.plans` already keeps for `bearing_to_goal`: scoring only, never
fed to the policy.

PRECISION IS THE WHOLE QUESTION, NOT RECALL, and this is the lesson from the open-set
prompt that looked like a fix. Asked to list every drivable direction, the model named
the needed side on 15 of 20 failure moments against the old question's 2 -- and 10 of
those 20 answers listed ALL FIVE directions. Recall 100%, precision 0, useless. A scan
that reports four openings outside the frame, one of which is the goal, hands the VLM a
menu rather than an answer. So the number that decides this is the MEAN COUNT of
blind-side openings, reported next to the hit rate, plus how often the channel fires at
all on control moments where straight was already correct. A channel that always says
"there is an opening to your left" is noise with a good hit rate.
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
ap.add_argument("--env", default="hospital")
ap.add_argument("--rays", type=int, default=500)
ap.add_argument("--range", type=float, default=12.0, help="RPLIDAR C1.")
ap.add_argument("--height", type=float, default=0.30)
ap.add_argument("--fov", type=float, default=90.0, help="camera_nav HFOV; blind is outside.")
ap.add_argument("--open-m", type=float, default=3.0, help="Free range that counts as open.")
ap.add_argument("--min-width", type=float, default=15.0, help="Narrowest usable gap, deg.")
ap.add_argument("--min-err", type=float, default=40.0)
ap.add_argument("--ctrl-err", type=float, default=15.0)
ap.add_argument("--max-poses", type=int, default=60)
args = ap.parse_args()

cfg = load_config()

jobs = []
for arm in ARMS:
    for f in sorted((STUDY / arm).glob("*.json")):
        if f.stem == "arm":
            continue
        r = score(f, cfg)
        if r is None or r["success"]:
            continue
        ep = f.stem.split("_", 1)[1].rsplit("_", 1)[0]
        if ep not in cfg:
            continue
        try:
            if env_name(load_episode(ep).scene) != args.env:
                continue
        except Exception:
            continue
        trace = json.loads(f.read_text()).get("trace") or []
        if len(trace) < 8:
            continue
        jobs.append({"ep": ep, "trace": trace, "goal": tuple(cfg[ep]["goal"][:2])})

if not jobs:
    raise SystemExit(f"no failed reference runs on disk for env {args.env!r}")

scene = REPO / "assets" / "usd" / f"nav_{args.env}.usda"
print(f"\n{len(jobs)} failed runs in {args.env}; {args.rays} rays at z+{args.height:.2f} m, "
      f"{args.range:.0f} m; camera blind outside +/-{args.fov / 2:.0f} deg")

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

N = args.rays
STEP = 2.0 * math.pi / N
BEARINGS = [i * STEP for i in range(N)]
HALF_FOV = math.radians(args.fov / 2.0)
MIN_RUN = max(1, int(math.radians(args.min_width) / STEP))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def openings(x: float, y: float, z: float) -> tuple[int, list[bool]]:
    """How many drivable sectors the scan reports, and which rays lie in one.

    Found from the scan alone -- no goal, no instruction, and no yaw either: the count and
    the per-ray mask are both in the world frame, and the caller does the only thing that
    needs a heading, which is deciding what the camera could see.

    The mask is returned alongside the count because the two answer different halves of
    the question. The count is the precision denominator. The mask is what makes a NULL
    possible: the fraction of the blind arc that is open at all is exactly the chance a
    goal bearing lands in an opening for no reason, and a hit rate without it is
    uninterpretable.

    The index space is circular, so a sector spanning bearing 0 would be reported as two
    half-openings if the walk simply started at index 0. Starting it at the first BLOCKED
    bearing instead makes every run contiguous in the walk order, which is why the
    all-free case has to be handled before the search rather than inside it.
    """
    free = []
    for b in BEARINGS:
        d = (math.cos(b), math.sin(b), 0.0)
        r = query.raycast_closest((x, y, z), d, args.range)
        hit = r and r.get("hit") and not str(r.get("collision", "")).startswith("/World/Aloha")
        free.append((float(r["distance"]) if hit else args.range) >= args.open_m)
    if all(free):
        return 1, [True] * N
    base = free.index(False)
    in_open = [False] * N
    count, i = 0, 0
    while i < N:
        if not free[(base + i) % N]:
            i += 1
            continue
        j = i
        while j < N and free[(base + j) % N]:
            j += 1
        if j - i >= MIN_RUN:
            count += 1
            for k in range(i, j):
                in_open[(base + k) % N] = True
        i = j
    return count, in_open


# per episode: [fail_n, goal_blind, blind_hit, null_sum, count_sum, ctrl_n, ctrl_fire,
#               ctrl_count_sum]
stats: dict[str, list[float]] = {}

for j in jobs:
    trace, (gx, gy) = j["trace"], j["goal"]
    stepn = max(1, len(trace) // args.max_poses)
    for s in trace[::stepn]:
        x, y, yaw = float(s[0]), float(s[1]), float(s[2])
        rel = wrap(math.atan2(gy - y, gx - x) - yaw)
        deg = abs(math.degrees(rel))
        if deg >= args.min_err:
            kind = "fail"
        elif deg <= args.ctrl_err:
            kind = "ctrl"
        else:
            continue
        count, in_open = openings(x, y, 0.01 + args.height)

        # Blindness is per RAY, not per opening. An opening whose centre sits outside the
        # frame can still reach into it, so classifying by centre would both miscount the
        # blind arc and let the null disagree with the hit test it is the null FOR.
        blind_idx = [k for k in range(N) if abs(wrap(BEARINGS[k] - yaw)) > HALF_FOV]
        n_blind_open = sum(in_open[k] for k in blind_idx)
        # Distinct openings reaching into the blind arc: count transitions, so one wide
        # sector is one item however many rays it spans.
        n_ops = sum(1 for a, b in zip(blind_idx, blind_idx[1:] + blind_idx[:1])
                    if in_open[b] and not in_open[a])
        n_ops = max(n_ops, 1) if n_blind_open else 0

        row = stats.setdefault(j["ep"], [0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0])
        if kind == "fail":
            row[0] += 1
            if deg > args.fov / 2.0:                       # goal outside the frame
                row[1] += 1
                goal_ray = int(round((math.atan2(gy - y, gx - x) % (2.0 * math.pi)) / STEP)) % N
                row[2] += in_open[goal_ray]
                row[3] += n_blind_open / len(blind_idx)    # the null for that hit
            row[4] += n_ops
            row[8] += n_blind_open / len(blind_idx)        # same fraction, every moment
        else:
            row[5] += 1
            row[6] += bool(n_ops)
            row[7] += n_ops

BLIND_DEG = 360.0 - args.fov
HDR = ("episode", "goal outside", "hit", "null", "lift", "opens", "width", "ctrl fires")


def line(name, v):
    n, gb, hit, null, ops, cn, cf, co, frac = v
    h = 100.0 * hit / gb if gb else 0.0
    b = 100.0 * null / gb if gb else 0.0
    print(f"  {name:<30}{f'{gb}/{n} ({gb / n:.0%})' if n else '--':>16}"
          f"{f'{hit}/{gb} ({h:.0f}%)' if gb else '--':>17}"
          f"{f'{b:.0f}%' if gb else '--':>7}"
          f"{f'{h - b:+.0f}pp' if gb else '--':>8}"
          f"{ops / n if n else 0:8.2f}"
          f"{f'{frac * BLIND_DEG / ops:.0f}d' if ops else '--':>8}"
          f"{f'{cf}/{cn} ({cf / cn:.0%})' if cn else '--':>16}")


print(f"\nOPENING = contiguous sector, free range >= {args.open_m:.1f} m, "
      f"width >= {args.min_width:.0f} deg; blind arc is {BLIND_DEG:.0f} deg\n")
print(f"  {HDR[0]:<30}{HDR[1]:>16}{HDR[2]:>17}{HDR[3]:>7}{HDR[4]:>8}"
      f"{HDR[5]:>8}{HDR[6]:>8}{HDR[7]:>16}")
print("  " + "-" * 111)

T = [0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0]
for ep in sorted(stats):
    for i, x in enumerate(stats[ep]):
        T[i] += x
    line(ep, stats[ep])
print("  " + "-" * 111)
line("ALL", T)

print("\n\nHOW TO READ IT\n")
print("  goal outside  -- how often the thing the robot needed was structurally invisible")
print("                   to the camera. This is the hole the scan exists to fill.")
print("  hit           -- and the scan reported a drivable sector containing it. RECALL,")
print("                   and recall alone is what made the open-set prompt look like a fix.")
print("  null          -- the fraction of the blind arc that is open ANYWAY, i.e. the")
print("                   chance of that hit for no reason. Read `hit` only against this.")
print("  lift          -- hit minus null, in points. This is the entire result. Near zero")
print("                   means the scan reports that a room is a room, not where to go.")
print("  opens/width   -- how many blind-side sectors are offered at once, and how wide")
print("                   each is. READ THEM TOGETHER. A count near 1 looks like the")
print("                   channel naming ONE direction, and that reading is wrong when the")
print("                   width is most of the blind arc: one 200 deg opening is not a")
print("                   direction, it is the observation that the robot is indoors.")
print("  ctrl fires    -- how loud the channel is when straight was already correct. A")
print("                   channel that fires on nearly every control moment is noise.")

kit.close()
