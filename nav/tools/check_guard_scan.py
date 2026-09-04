"""Does the lidar-fed guard agree with the 7-ray fan, and where does it disagree?

    ~/robotics/isaacsim-6.0.1/python.sh nav/tools/check_guard_scan.py \
        --episode hospital_exit_room --steps 900

ONE TRAJECTORY, TWO OBSERVERS. Both guards are checked on the SAME pose every step and
only the fan's answer is allowed to steer. Letting each drive its own robot would produce
two different trajectories, and every disagreement would then be attributable either to
the sensor or to the fact that the two robots were standing in different places -- which
is exactly the confound that makes a comparison worthless. The fan drives because it is
the arm every number in ../../CLAUDE.md was measured on.

WHAT THE COMPARISON IS FOR. The claim being tested is narrow and quantitative: the fan's
7 rays across +-35 degrees sit 11.7 degrees apart, which is a 12 cm gap at the 0.6 m stop
distance and 30 cm at 1.5 m, and the C1's 0.72 degrees closes those to 0.8 cm and 1.9 cm.
If that matters at all, it shows up as moments where the scan reports something inside the
band and the fan reports clear -- `scan-only` below. If it never happens, the density
argument is wrong on these scenes and should be dropped rather than shipped.

WHAT WOULD FALSIFY THE WIRING RATHER THAN THE IDEA, and the two are worth keeping apart:

  * `fan-only` moments, where the fan sees something the 16x denser sensor missed. A few
    are expected and are not a bug -- the buffer is up to one revolution old, so an
    obstacle that entered a bearing the beam has not reached yet is genuinely absent, and
    that is the freshness cost the raised stop distance is paying for. A LOT of them, or
    any at all once the robot is standing still, would mean the buffer is not being
    filled where it is being read.
  * `nearest` disagreeing by metres rather than centimetres. The two are measuring the
    same wedge of the same world; the scan is centre-relative by construction and the fan
    adds its chassis radius back to become so. A systematic 0.35 m offset between them is
    that correction having been applied twice or not at all.

The last section runs the buffer through `arc_clearance`, which is the OTHER consumer --
the menu filter the policy server applies. It is here rather than in its own tool because
it needs exactly this setup and answers a question of the same kind: whether the numbers
that reach the VLM are metres of drivable arc or an artifact of an empty buffer.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))
sys.path.insert(0, str(REPO / "nav"))

from episode import env_name, load_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", default="hospital_exit_room")
ap.add_argument("--steps", type=int, default=900, help="physics steps at 60 Hz")
ap.add_argument("--speed", type=float, default=0.6, help="commanded vx, m/s")
ap.add_argument("--height", type=float, default=0.30, help="lidar mount height, m")
ap.add_argument("--stop-fan", type=float, default=0.60)
ap.add_argument("--stop-scan", type=float, default=0.75)
args = ap.parse_args()

DT = 1.0 / 60.0

ep = load_episode(args.episode)
scene = REPO / "assets" / "usd" / f"nav_{env_name(ep.scene)}.usda"
if not scene.is_file():
    raise SystemExit(f"nav scene not built: {scene}\n  nav/sim/build_nav_scene.sh {env_name(ep.scene)}")

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

from arc_menu import arc_clearance, drivable, make_arcs  # noqa: E402
from base_drive import KinematicBase  # noqa: E402
from collision_guard import CollisionGuard  # noqa: E402
from lidar_2d import SweepingLidar2D  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(str(scene.resolve()))
stage = usd_context.get_stage()
for _ in range(240):
    kit.update()
stage.Load()
for _ in range(30):
    kit.update()

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(10):
    kit.update()

art = Articulation("/World/Aloha/Geometry/base_link")
art.initialize()
base = KinematicBase(art, list(art.dof_names))
base.sync_from_sim()
base.reset_to(tuple(ep.start), ep.start_yaw_rad)
for _ in range(10):
    kit.update()

lidar = SweepingLidar2D(mount_height_m=args.height)
guard_fan = CollisionGuard(stop_distance_m=args.stop_fan)
guard_scan = CollisionGuard(scan=lidar, stop_distance_m=args.stop_scan)
arcs = make_arcs()

rows = []          # one per step, both observers
clearance_rows = []  # sampled every 60 steps, for the menu-filter half
rays_total = 0

for step in range(args.steps):
    position = base.position()
    yaw = base.yaw
    rays_total += lidar.step(position, yaw, DT, step * DT)

    f = guard_fan.check(position, yaw, args.speed, 0.0)
    s = guard_scan.check(position, yaw, args.speed, 0.0)
    rows.append((step, f.distance_m, s.distance_m, f.blocked, s.blocked,
                 math.hypot(f.vx, f.vy), math.hypot(s.vx, s.vy)))

    if step % 60 == 0 and step > 0:
        pts = lidar.points_body(position, yaw)
        clear = arc_clearance(arcs, pts)
        kept, _, dropped = drivable(arcs, list(range(len(arcs))), clear, 1.0)
        clearance_rows.append((step, len(pts), lidar.max_age_s(step * DT), clear,
                               len(kept), len(dropped)))

    # Only the FAN steers. See the header -- this is what keeps the comparison a
    # comparison of sensors rather than of two different drives.
    base.apply(f.vx, f.vy, 0.0, DT)
    kit.update()

print(f"\nepisode {ep.name} ({env_name(ep.scene)}), {args.steps} steps at 60 Hz "
      f"= {args.steps * DT:.1f} s, vx={args.speed} m/s")
print(f"lidar z={args.height} m, {lidar.spec.name}, {rays_total} rays cast "
      f"({rays_total / args.steps:.1f}/step, {lidar.revolutions:.1f} revolutions)")
print(f"stop distance: fan {args.stop_fan} m, scan {args.stop_scan} m "
      f"(+{args.stop_scan - args.stop_fan:.2f} m for one revolution of staleness)\n")

# --- who saw what -----------------------------------------------------------------
fan_block = [r for r in rows if r[3]]
scan_block = [r for r in rows if r[4]]
scan_only = [r for r in rows if r[4] and not r[3]]
fan_only = [r for r in rows if r[3] and not r[4]]
both = [r for r in rows if r[3] and r[4]]

print(f"  blocked steps      fan {len(fan_block):5d}   scan {len(scan_block):5d}   "
      f"both {len(both):5d}")
print(f"  scan-only          {len(scan_only):5d}   (CONFOUNDED -- see below)")
print(f"  fan-only           {len(fan_only):5d}   (the freshness cost: a bearing the "
      f"beam had not reached)")
print(f"  interventions      fan {guard_fan.interventions:5d}   "
      f"scan {guard_scan.interventions:5d}")
print(f"\n  `scan-only` mixes two causes and must not be quoted as the density result: "
      f"the scan\n  guard also runs a HIGHER stop distance ({args.stop_scan} vs "
      f"{args.stop_fan} m), so anything between the\n  two thresholds lands in it "
      f"whether or not 7 rays would have found it. The matched-\n  threshold version is "
      f"the last line of the next section.")

# --- do they agree about DISTANCE -------------------------------------------------
# Only where both are inside the slow band; comparing infinities says nothing, and one
# observer reporting inf while the other reports 1.2 m is already counted above.
paired = [(r[1], r[2]) for r in rows if math.isfinite(r[1]) and math.isfinite(r[2])]
print()
if paired:
    diffs = [a - b for a, b in paired]
    # SPLIT BY SIGN, because the two signs are different claims and an absolute maximum
    # hides which one happened. `fan - scan > 0` is the scan finding something CLOSER
    # than 7 rays did, which is the density argument. `< 0` is the scan missing something
    # the fan caught, which is the staleness cost. Reporting max|d| answers neither.
    closer = [d for d in diffs if d > 0]
    missed = [-d for d in diffs if d < 0]
    print(f"  nearest, both finite   n={len(paired)}   "
          f"fan-minus-scan median {statistics.median(diffs):+.3f} m, "
          f"mean {statistics.fmean(diffs):+.3f} m")
    print(f"    scan found it CLOSER   {len(closer):5d} steps, worst "
          f"{max(closer) if closer else 0.0:.3f} m   (density)")
    print(f"    scan found it FARTHER  {len(missed):5d} steps, worst "
          f"{max(missed) if missed else 0.0:.3f} m   (staleness)")
    print("  (a systematic +-0.35 m in the mean is the chassis radius applied twice or "
          "not at all,")
    print("   not a sensor difference -- that would be a wiring bug and would not depend "
          "on the route)")

    # The threshold-crossing count above needs the route to actually deliver a near miss.
    # This is the same claim without that dependency: moments where the DENSER sensor
    # reports something inside the fan's own stop distance while the fan calls it clear.
    would = [r for r in rows if math.isfinite(r[2]) and r[2] <= args.stop_fan
             and not (math.isfinite(r[1]) and r[1] <= args.stop_fan)]
    print(f"\n  DENSITY, matched threshold: scan inside the fan's own {args.stop_fan} m "
          f"while the fan\n  called it clear -- {len(would)} steps. This is the number "
          f"the extra 493 rays bought.")
else:
    print("  nearest: never both finite -- nothing came within "
          f"{guard_fan.slow_distance_m} m on this route, so this run tests nothing.")
    print("  Pick an episode that drives at something, or raise --steps.")

if not fan_block and not scan_block:
    print("\n  NOTE: neither guard ever blocked. The distance channel above is still a "
          "real")
    print("  comparison, but the scan-only / fan-only counts are a NULL, not a result -- "
          "the")
    print("  robot never came within a stop distance of anything on this route. Do not "
          "read")
    print("  0 scan-only as evidence against the density claim; raise --steps or pick an "
          "episode")
    print("  that drives at something.")

# --- what the MENU consumer sees --------------------------------------------------
print(f"\n  {'step':>5}  {'points':>7}  {'oldest':>7}  {'kept':>5}  {'dropped':>7}  "
      f"clearance per arc (m)")
print("  " + "-" * 78)
for step, npts, age, clear, nkept, ndrop in clearance_rows:
    text = " ".join(f"{c:4.1f}" for c in clear)
    print(f"  {step:5d}  {npts:7d}  {age:6.3f}s  {nkept:5d}  {ndrop:7d}  {text}")

print()
if not clearance_rows:
    print("  no clearance samples -- --steps below 60.")
else:
    stuck = sum(1 for r in clearance_rows if r[4] == 1)
    print(f"  menu filter: {stuck} of {len(clearance_rows)} samples were reduced to a "
          f"single arc.")
    print("  `drivable` never returns an empty menu -- a shrunken menu is a claim about "
          "the room,")
    print("  and STOP-as-'everything is blocked' is the measured pathology it exists to "
          "avoid.")

kit.close()
