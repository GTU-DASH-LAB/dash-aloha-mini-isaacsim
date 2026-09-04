"""Where can an RPLIDAR C1 actually go on this robot, and does the scan see anything?

    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/check_lidar_2d.py --episode warehouse

TWO QUESTIONS, and they are separate. `check_collision_guard.py` already established
that `raycast_closest` fires against real warehouse geometry, so "does the ray work" is
settled and is not what this tool is for.

  1. HOW BIG IS THE BLIND SECTOR. AlohaMini carries a lift column and two arms down its
     centreline, and a lidar bolted below them is looking through its own robot for part
     of every revolution. `collision_guard.py` skips self-hits by design and so could
     never have revealed this; `lidar_2d.py` keeps them precisely so it can be measured.
     A 360 degree part number does not buy 360 degrees of coverage on a robot with a
     mast in the way, and the mount height is what decides the difference. This sweeps
     candidate heights and reports the cost of each.

  2. IS THE SCAN INFORMATIVE THERE. A mount that is clear of the robot but above every
     wall would score a perfect zero blind sector and return nothing but max-range
     misses -- an unblocked view of empty air. So the same sweep reports how much of the
     revolution comes back as a real external return. The height to pick is the lowest
     one whose blind sector is small, not the one whose blind sector is smallest.

The measurement is taken from the robot's OWN pose read off the stage, not from
`episode.start`. Nav stages are keyed on the environment and the runner teleports to the
episode start at run time, so the authored robot pose and the episode start are not the
same point -- and self-occlusion measured from a spot the robot is not standing in would
be a confident number about nothing.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

from episode import load_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", default="warehouse")
ap.add_argument("--heights", default="0.20,0.30,0.50,0.80,1.10,1.30,1.50",
                help="Candidate mount heights above the robot base, metres.")
ap.add_argument("--spec", default="c1", choices=("c1", "ld19"))
args = ap.parse_args()

ep = load_episode(args.episode)
scene = REPO / "assets" / "usd" / f"nav_{ep.name}.usda"
if not scene.is_file():
    raise SystemExit(f"nav scene not built: {scene}\n  nav/sim/build_nav_scene.sh {ep.name}")

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from lidar_2d import LDROBOT_LD19, RPLIDAR_C1, Lidar2D  # noqa: E402

spec = RPLIDAR_C1 if args.spec == "c1" else LDROBOT_LD19

usd_context = omni.usd.get_context()
usd_context.open_stage(str(scene.resolve()))
stage = usd_context.get_stage()
for _ in range(240):
    kit.update()
stage.Load()
for _ in range(30):
    kit.update()

# Scene queries only report colliders while the timeline is running; querying a stopped
# timeline is its own way to get a confident, wrong "nothing is there".
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(30):
    kit.update()

robot = stage.GetPrimAtPath("/World/Aloha")
if not robot or not robot.IsValid():
    kit.close()
    raise SystemExit("no robot at /World/Aloha -- was this stage built by build_nav_scene.sh?")
m = UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
t = m.ExtractTranslation()
position = (float(t[0]), float(t[1]), float(t[2]))
yaw = 0.0


def longest_blind_arc_deg(self_hit) -> float:
    """Largest CONTIGUOUS blocked sector, wrapping around the seam.

    The total blind fraction understates the damage: 20% scattered as single dead rays
    between the arms is a usable sensor, while the same 20% in one block is a robot that
    cannot see one whole side. Only the contiguous number answers the mounting question.
    """
    n = len(self_hit)
    if self_hit.all():
        return 360.0
    if not self_hit.any():
        return 0.0
    best = run = 0
    # Two laps so a run spanning index 0 is counted whole rather than split at the seam.
    for i in range(2 * n):
        run = run + 1 if self_hit[i % n] else 0
        best = max(best, run)
    return min(best, n) * 360.0 / n


print(f"\n{spec.name}: {spec.min_range_m}-{spec.max_range_m} m, "
      f"{spec.points_per_rev} points/rev ({spec.angular_resolution_deg:.2f} deg), "
      f"{spec.scan_rate_hz:.0f} Hz")
print(f"scene {ep.name}, robot at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})\n")
print(f"  {'mount z':>8}  {'blind':>7}  {'worst arc':>10}  {'external':>9}  "
      f"{'max-range':>10}  {'nearest':>8}")
print("  " + "-" * 66)

rows = []
for text in args.heights.split(","):
    h = float(text)
    lidar = Lidar2D(spec=spec, mount_height_m=h)
    s = lidar.scan(position, yaw)
    external = s.hit & ~s.self_hit
    nearest = float(s.ranges[external].min()) if external.any() else float("nan")
    row = {
        "h": h,
        "blind": s.blind_fraction,
        "arc": longest_blind_arc_deg(s.self_hit),
        "external": float(external.mean()),
        "miss": float((~s.hit).mean()),
        "nearest": nearest,
    }
    rows.append(row)
    print(f"  {h:7.2f}m  {row['blind']:6.1%}  {row['arc']:9.1f}d  {row['external']:8.1%}  "
          f"{row['miss']:9.1%}  {nearest:7.2f}m")

print("  " + "-" * 66)

# The lowest mount that both clears the robot and still sees the room. Low is preferred
# on purpose: a mast is weight, wiring and a thing to catch on doorframes, and the whole
# argument for 2D over 3D was that the cheap, simple option is the one that transfers.
usable = [r for r in rows if r["arc"] <= 15.0 and r["external"] >= 0.10]
print()
if not usable:
    print("VERDICT: no candidate height both clears the robot and sees the room.")
    print("         Either the mast has to go higher than the sweep, or the unit has to")
    print("         be mounted forward of the column and accept a partial field of view")
    print("         -- set Lidar2D(mount_forward_m=...) and re-run before assuming a")
    print("         360 degree scan is available on this chassis.")
    code = 1
else:
    best = min(usable, key=lambda r: r["h"])
    print(f"VERDICT: mount at z={best['h']:.2f} m -- worst blind arc {best['arc']:.1f} deg, "
          f"{best['external']:.0%} of the revolution returns real geometry.")
    print(f"         Set Lidar2D(mount_height_m={best['h']:.2f}); the 1.30 default in")
    print("         lidar_2d.py is provisional and this number replaces it.")
    code = 0

kit.close()
raise SystemExit(code)
