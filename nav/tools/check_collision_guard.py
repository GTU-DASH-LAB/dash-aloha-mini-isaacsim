"""Does the collision guard actually see anything?

A guard that silently never fires is worse than no guard: the robot ghosts through
shelving (its base is kinematically teleported, so PhysX will not stop it) and the run
log reports "0 guard stops", which reads as "nothing was in the way".

The first real warehouse run drove 50.85 m down an aisle between pallet racks and
reported exactly 0 interventions. That is *possible* -- the aisle is wide -- but it is
equally what a broken raycast looks like, and the two must be told apart.

This sweeps the guard's ray fan through a full circle from a known pose and reports
what it hits. In a warehouse, some bearings must hit something.

Usage:
    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/check_collision_guard.py --episode warehouse
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
ap.add_argument("--bearings", type=int, default=16, help="Directions to sweep.")
ap.add_argument("--max-range", type=float, default=15.0)
args = ap.parse_args()

ep = load_episode(args.episode)
scene = REPO / "assets" / "usd" / f"nav_{ep.name}.usda"
if not scene.is_file():
    raise SystemExit(f"nav scene not built: {scene}\n  nav/sim/build_nav_scene.sh {ep.name}")

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(str(scene.resolve()))
stage = usd_context.get_stage()
for _ in range(240):
    kit.update()
stage.Load()
for _ in range(30):
    kit.update()

# The scene query interface only reports colliders once physics is actually running.
# Querying a stopped timeline is itself a way to get a confident, wrong "no hits".
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(30):
    kit.update()

query = get_physx_scene_query_interface()
origin_xy = ep.start[:2]
z = ep.start[2] + 0.30  # the guard's ray height

print(f"\nSweeping {args.bearings} bearings from {origin_xy} at z={z:.2f}, "
      f"range {args.max_range} m\n")
print(f"{'bearing':>8}  {'hit':>5}  {'dist':>7}   collider")
print("-" * 78)

hits = 0
for i in range(args.bearings):
    bearing = 2.0 * math.pi * i / args.bearings
    direction = (math.cos(bearing), math.sin(bearing), 0.0)
    origin = (
        origin_xy[0] + direction[0] * 0.35,   # guard's chassis_radius_m
        origin_xy[1] + direction[1] * 0.35,
        z,
    )
    result = query.raycast_closest(origin, direction, args.max_range)
    if result and result.get("hit"):
        hits += 1
        prim = str(result.get("collision", ""))
        print(f"{math.degrees(bearing):7.1f}deg  {'YES':>5}  "
              f"{float(result['distance']):6.2f}m   {prim[-58:]}")
    else:
        print(f"{math.degrees(bearing):7.1f}deg  {'--':>5}  {'':>7}   (nothing within range)")

print("-" * 78)
print(f"\n{hits}/{args.bearings} bearings hit something.")
if hits == 0:
    print("\nVERDICT: FAIL -- the raycast never fires. The guard is decorative and the")
    print("         robot will pass through walls. Check that the timeline is playing")
    print("         and that the environment's colliders actually composed.")
    code = 1
elif hits < args.bearings // 4:
    print("\nVERDICT: SUSPICIOUS -- very few hits for an indoor scene. Worth eyeballing.")
    code = 0
else:
    print("\nVERDICT: PASS -- the guard sees real geometry. A low intervention count in a")
    print("         run means the robot genuinely kept its distance, not that the guard")
    print("         was asleep.")
    code = 0

kit.close()
raise SystemExit(code)
