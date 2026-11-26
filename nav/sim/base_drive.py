"""Velocity command -> robot pose, via the kinematic base drive.

Used by the navigation runner AND by scripts/control/control_terminal.py (in its
"released" mode, see KinematicBase), so the two cannot drift apart: they had, and the
REPL kept the pre-fix step below for a day after this file was fixed. The reasoning
behind teleporting rather than driving the wheels lives in
../../CLAUDE.md; the short version is that real wheel traction never held up, and
set_world_poses() is the only way to actually move a body during simulation (raw USD
transform writes are ignored while physics is stepping -- PhysX only pushes
physics->USD, verified directly).

The wheel joints are still commanded at the kinematically correct rate. They carry no
load, but a robot gliding along with motionless wheels looks broken, and the wheel
speeds are the honest LeKiwi kinematics either way.

**A teleport writes a pose and nothing else, and the velocity it leaves behind is the
sim's.** Until 2026-10 this module wrote only poses, and every outdoor blow-up traced back
to that. DynaNav's start z is the Nova Carter's, not ours, so outdoor starts sat 0.14-2.21 m
above the terrain; the robot was still falling when the episode reset it, and `reset_to()`
moved it without stopping it -- measured at that instant, (-0.43, 0.74, -0.12) m/s and
(-2.24, -1.24, -0.42) rad/s on outdoor_umbrellas. It fell 0.99 m again while spinning,
landed at pitch 59.6 deg / roll 42 deg at 4.6 m/s and left the ground at 8.9 rad/s. From
there `apply()` forced the orientation level every step and never touched the angular
velocity, so the spin never decayed, and forced-level-plus-spin turned out to be a
reactionless thruster: +12 m/s^2 beyond gravity once airborne, 26 m up by t = 7.5 s, and
the km-scale "drive" in the trace once the sim's shove passed the resync tolerance.

So the pose and the velocity are now written together, always: `apply()` sets the root
velocity to the commanded planar velocity and (0, 0, omega) -- including on a zero
command, because a stopped robot has to stop spinning too, and on a zero command the pose
is held as well -- and `reset_to()` zeroes root and joint velocities. Replayed on the exact blown state, writing the velocity alone took z
span from 276 m to 0.058 m (max |v| 0.48 m/s against 0.445 commanded); the snap below
removes the drop that seeded it in the first place.

**A teleported body has no contact response of its own, so the base gets one before each
teleport.** PhysX never pushes back on a root that is pinned every step: replaying recorded
runs through this class alone, the arms drove into a staircase's underside (14.9 m/s),
the lift carriage into a chair back and the tyres through 12-15 cm kerbs. The contact layer
(nav/sim/kinematic_contact.py) sweeps the robot's whole envelope along each step's
displacement and clips it to stop 2 cm short and slide; it plugs into `motion_filter`
(see `_resolve_motion`). NavigationRunner.setup() attaches it (NAV_CONTACT=0 is the bare
base, for A/B runs), and so does everything that boots through the runner
(validate_episodes.py, route_drive.py) and scripts/control/control_terminal.py
(--no-contact). nav/tools/check_guard_scan.py still drives the BARE base: it compares two
guards on one fan-steered trajectory, the arm its recorded numbers were measured on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from alohamini1_specs import (  # noqa: E402
    BASE_LINK_REST_Z_M,
    WHEEL_ANGLES_RAD,
    WHEEL_MOUNT_RADIUS_M,
    WHEEL_NAMES,
    WHEEL_RADIUS_M,
    body_to_wheel_speeds,
)

# The joint each of body_to_wheel_speeds()'s outputs drives, IN ITS ORDER. That order is
# LeKiwi's angle slots [150, -90, 30] deg, which are wheel1, wheel3, wheel2 on this URDF
# (alohamini1_specs.py derives it from the joint origins). This used to be the naive
# ("wheel1", "wheel2", "wheel3"), so wheel2 got the -90 slot's speed and wheel3 the 30
# slot's -- invisible while the wheels carried no load, but traction is not zero, and two
# wheels spinning against the motion dragged the body sideways every step.
WHEEL_JOINTS = tuple(WHEEL_NAMES)

# --- spawning on the floor --------------------------------------------------------------
# Where `ground_spawn_pose()` starts looking, above the episode's own start z. An episode
# start is free space for DynaNav's Nova Carter, whose camera alone sits 0.346 m up, so the
# first 0.30 m above it cannot hold an umbrella, a canopy or a ceiling -- casting from
# higher would risk landing the robot ON one. It is also enough to recover a start a few
# cm BELOW its floor, which a hand-written episode can easily be.
SPAWN_PROBE_ABOVE_M = 0.30
# Measured drops from episode z to the collider under it: umbrellas 0.989, pillars 2.207,
# library 0.136, upsway 1.914, upsway_far 1.851, ramp_fountain 0.621; indoors ~0.01.
SPAWN_PROBE_RANGE_M = 10.0
# A surface the robot can stand on faces up. Steeper hits are the side of a kerb or a
# railing the ray grazed, and the next hit down is the floor.
SPAWN_WALKABLE_NORMAL_Z = 0.7
# Above rest, so the highest wheel starts just clear of its floor instead of embedded in
# it -- embedding is a separation impulse, the same class of kick as the drop.
SPAWN_CLEARANCE_M = 0.003
# A hit closer than this to the ray's origin is not a surface below it: PhysX reports a ray
# that STARTS inside a collider as a hit at distance 0, at the origin, with whatever normal
# the shape gives (+z for a box, measured). Taken as a floor, that placed the robot at the
# probe origin, inside the solid (0.31 m up a 0.45 m box). Skipped, and recorded, so a start
# that sits inside something says so instead of quietly spawning embedded.
SPAWN_ORIGIN_OVERLAP_M = 1e-3
# Iterated ray: after a hit that is not the floor (the robot's own collider, a steep face),
# re-cast from this far past it. Out of an initial overlap the step doubles up to
# SPAWN_OVERLAP_STEP_MAX_M, which walks out of a convex or box solid in a few casts.
SPAWN_RAY_SKIP_M = 1e-3
SPAWN_OVERLAP_STEP_MAX_M = 0.05
# Hard bound on casts per sample ray, whatever the scene holds there.
SPAWN_RAY_MAX_CASTS = 64
# Leaving a solid by a step can carry the origin PAST a surface that starts right under it,
# and the solid the probe meets most is the robot's own tyre, which rests ON the floor. The
# first version stepped straight through: on every start the robot was already standing on
# (hospital e02-e05 after e01, and every UI or route_drive repeat), the centre and inner
# rings found nothing -- the origin had left the r = 0.05 tyre below the ground plane's top,
# inside its half-space -- or a floor 2.78 m down (hospital e09/e10, below a one-sided floor
# mesh), and only the 0.75 r ring's graze survived: the robot spawned 17 mm into the floor
# (verify_env hospital, 2026-10-05). So on leaving a solid the step just taken is searched:
# into another solid, its top is bisected to SPAWN_BISECT_STEPS halvings (5 cm -> 12 um);
# into free space, a ray looks back up the step, both-sided so a one-sided floor's top face
# is seen from below.
SPAWN_BISECT_STEPS = 12
SPAWN_BISECT_RAY_M = 1e-5
# Where under each tyre the floor is sampled, as fractions of the tyre radius
# (WHEEL_RADIUS_M), each ring at SPAWN_FOOTPRINT_BEARINGS bearings, plus the centre.
#
# One ray at the tyre centre cannot see a floor feature beside it, and the sphere still sits
# on it: office_e41-45's tyre 1 is 6 mm past the edge of a 5 mm door sill
# (BP_Door_250cm4/SM_Door_250cm, top z = 0.005), the centre ray finds the floor at z = 0,
# and the r = 0.05 sphere starts embedded in the sill's edge -- 4.6 mm of lift predicted,
# 1.035 deg of tilt measured (scenes/office/probe_floor.json). A feature at horizontal
# distance d from the tyre centre with its top at z needs the centre at z + sqrt(r^2 - d^2),
# so each sample contributes that and the tyre's floor is the max. The ring a feature first
# falls inside under-reads it by at most the sphere's sag between rings:
# r - sqrt(r^2 - d^2) = 1.6 mm at 0.25 r, 6.7 at 0.5 r, 16.9 at 0.75 r, 28.2 at 0.9 r --
# the sill reads 3.4 of its 4.6 mm, a 13 cm kerb face anywhere inside 0.9 r lifts the tyre
# onto the kerb, which is the only non-penetrating place for a sphere whose x/y is fixed.
# Not at 1.0 r: a feature there touches the sphere's side only if it is taller than the
# tyre's centre, and a top surface AT 1.0 r would read as a lift onto it.
SPAWN_FOOTPRINT_RINGS = (0.25, 0.5, 0.75, 0.9)
SPAWN_FOOTPRINT_BEARINGS = 8


def _is_inside(hit) -> bool:
    return bool(hit) and bool(hit.get("hit")) \
        and float(hit.get("distance", 0.0)) < SPAWN_ORIGIN_OVERLAP_M


def _solid_top(query, x: float, y: float, z_out: float, z_in: float, prim: str) -> float:
    """Top of `prim` between z_out (origin NOT inside it) and z_in (inside it), by
    bisection on whether a ray from the midpoint starts inside `prim`."""
    hi, lo = z_out, z_in
    for _ in range(SPAWN_BISECT_STEPS):
        mid = 0.5 * (hi + lo)
        # A ray this short only reports what the midpoint is INSIDE: a top surface within
        # SPAWN_ORIGIN_OVERLAP_M below it would otherwise read as inside and lift the
        # answer by up to 1 mm (modelled: 1.0 mm on a slab under a resting tyre).
        hit = query.raycast_closest((x, y, mid), (0.0, 0.0, -1.0), SPAWN_BISECT_RAY_M)
        if _is_inside(hit) and str(hit.get("collision", "")) == prim:
            lo = mid
        else:
            hi = mid
    return hi


def _floor_below(query, x: float, y: float, z_top: float, robot_prim_prefix: str,
                 origin_inside: list[str], seen: dict) -> tuple[float, str] | None:
    """First upward-facing, non-robot surface straight below (x, y, z_top): (z, prim).

    Iterated `raycast_closest`, re-cast 1 mm past each hit that is not the floor. NEVER
    `raycast_all`: on the outdoor stage it segfaults Kit inside libomni.physx.plugin on a
    ray `raycast_closest` answers (nav/tools/validate_episodes.py, 3 of 3 runs, exit 139),
    and it reports one hit per shape, so a floor sharing a mesh with something above it is
    invisible to it. A hit at the very start of the ray is the origin INSIDE a solid
    (SPAWN_ORIGIN_OVERLAP_M): recorded in `origin_inside` if it is the probe's own origin,
    never taken as a floor, and stepped out of in doubling increments -- and the step that
    leaves it is searched for a surface it jumped (SPAWN_BISECT_STEPS). That search is a
    mitigation, not a cure: where a tyre rests EXACTLY on the floor (tangent), the step out
    of it lands on or under the floor's top, and a ground plane is not hit from below at
    all. So every ray that meets the robot is counted in `seen["robot_hits"]`, and callers
    move the robot out of the way first (KinematicBase.park_for_probe).
    """
    def robot(prim: str) -> bool:
        return prim.startswith(robot_prim_prefix)

    down = (0.0, 0.0, -1.0)
    travelled = 0.0
    step = SPAWN_RAY_SKIP_M
    inside: str | None = None   # the solid the last origin was found in
    inside_at = 0.0             # ...and how far down that origin was
    for _ in range(SPAWN_RAY_MAX_CASTS):
        remaining = SPAWN_PROBE_RANGE_M - travelled
        if remaining <= 1e-6:
            return None
        z = z_top - travelled
        hit = query.raycast_closest((x, y, z), down, remaining)
        got = bool(hit) and bool(hit.get("hit"))
        prim = str(hit.get("collision", "")) if got else ""
        if robot(prim):
            seen["robot_hits"] = seen.get("robot_hits", 0) + 1
        if _is_inside(hit):
            # Only the probe ORIGIN being inside is worth reporting -- 1 mm past an
            # accepted-and-skipped face is inside that face's solid by construction.
            if travelled == 0.0 and not robot(prim) and prim not in origin_inside:
                origin_inside.append(prim)
            if inside is not None and prim != inside and not robot(prim):
                # Stepped out of one solid straight into another: the second one's top
                # is inside the step, and it is what the first one stood on.
                return _solid_top(query, x, y, z_top - inside_at, z, prim), prim
            inside, inside_at = prim, travelled
            travelled += step
            step = min(2.0 * step, SPAWN_OVERLAP_STEP_MAX_M)
            continue
        if inside is not None:
            # Out of the solid into free space, by a step that may have jumped the top of
            # whatever the solid rested on. Look back up the step for it.
            gap = travelled - inside_at
            up = query.raycast_closest((x, y, z), (0.0, 0.0, 1.0), gap, True)
            up_prim = str(up.get("collision", "")) if up and up.get("hit") else ""
            if up_prim and up_prim != inside and not robot(up_prim):
                if _is_inside(up):
                    # Inside it after all (a half-space ground plane reads this way).
                    return _solid_top(query, x, y, z_top - inside_at, z, up_prim), up_prim
                return z + float(up["distance"]), up_prim
            inside = None
        if not got:
            return None
        step = SPAWN_RAY_SKIP_M
        dist = float(hit.get("distance", 0.0))
        if not robot(prim) and float(hit["normal"][2]) >= SPAWN_WALKABLE_NORMAL_Z:
            return float(hit["position"][2]), prim
        travelled += dist + SPAWN_RAY_SKIP_M
    return None


def ground_spawn_pose(
    start: tuple[float, float, float],
    yaw: float,
    robot_prim_prefix: str = "/World/Aloha",
) -> tuple[tuple[float, float, float], dict]:
    """`start` with its z replaced by where THIS robot rests on the floor under it.

    DynaNav's episode z is the Nova Carter's spawn height, not ours, and outdoors it sits
    0.14-2.21 m above the terrain. Spawning there drops the robot, and a robot still
    tumbling from a 1-2 m fall when the episode starts is what seeded every outdoor
    blow-up (module docstring). Indoors the same z is 2 mm above rest, so this changes
    nothing there.

    The floor is sampled under each WHEEL, not under the centre: base_link's own shell has
    no collider (fix_wheel_collision.py), so the three sphere tyres are the only supports,
    and on a 7 deg slope (outdoor_pillars, p95) they stand on floor 2-3 cm apart. And under
    each wheel it is sampled over the tyre's FOOTPRINT (SPAWN_FOOTPRINT_RINGS), not at one
    point, because a sphere rests on whatever is highest within its radius. The robot is
    placed so the HIGHEST tyre rests on its floor -- any lower and that tyre starts
    embedded, which is a separation impulse; the others drop the slope's few cm.

    Each ray is cast down from just above the episode z (SPAWN_PROBE_ABOVE_M), so roofs,
    umbrellas and ceilings above the start are never hit; the robot's own colliders are
    skipped, and the first upward-facing hit below is the floor. Returns the snapped
    position and a record of what was found; with no floor under any tyre the episode z
    is returned unchanged and `record["ground_z"]` is None, so a caller can refuse to run
    rather than spawn the robot onto a guess.
    """
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    x0, y0, z0 = (float(v) for v in start)
    origin_z = z0 + SPAWN_PROBE_ABOVE_M
    origin_inside: list[str] = []
    seen: dict = {"robot_hits": 0}
    rays = 0
    r = WHEEL_RADIUS_M
    offsets = [(0.0, 0.0, r)] + [
        (f * r * math.cos(2.0 * math.pi * k / SPAWN_FOOTPRINT_BEARINGS),
         f * r * math.sin(2.0 * math.pi * k / SPAWN_FOOTPRINT_BEARINGS),
         math.sqrt(r * r - (f * r) ** 2))
        for f in SPAWN_FOOTPRINT_RINGS for k in range(SPAWN_FOOTPRINT_BEARINGS)]

    def tyre_floor(cx: float, cy: float) -> tuple[float, str, float | None] | None:
        """(effective floor z, its prim, the centre ray's floor z) under one tyre.

        The effective floor is where a flat floor would have to be for the sphere to rest
        exactly as it must here: max over the footprint of (hit z + sqrt(r^2 - d^2)) - r.
        """
        nonlocal rays
        best: tuple[float, str] | None = None
        centre_z = None
        for dx, dy, rise in offsets:
            hit = _floor_below(query, cx + dx, cy + dy, origin_z, robot_prim_prefix,
                               origin_inside, seen)
            rays += 1
            if hit is None:
                continue
            if dx == 0.0 and dy == 0.0:
                centre_z = hit[0]
            eff = hit[0] + rise - r
            if best is None or eff > best[0]:
                best = (eff, hit[1])
        return None if best is None else (best[0], best[1], centre_z)

    tyres = [tyre_floor(x0 + WHEEL_MOUNT_RADIUS_M * math.cos(yaw + a),
                        y0 + WHEEL_MOUNT_RADIUS_M * math.sin(yaw + a))
             for a in WHEEL_ANGLES_RAD]
    found = [(t[0], t[1]) for t in tyres if t is not None]
    if not found:
        # A tyre over a gap still leaves the others; none at all, try the centre before
        # giving up -- a narrow beam or a grating can pass three footprints and stop one.
        centre = tyre_floor(x0, y0)
        found = [(centre[0], centre[1])] if centre is not None else []
    record: dict = {
        "episode_z": round(z0, 4),
        # Per tyre, in WHEEL_NAMES order: the effective floor the snap used...
        "tyre_ground_z": [None if t is None else round(t[0], 4) for t in tyres],
        # ...and what the single centre ray alone would have said, so the footprint's
        # contribution (a sill, a kerb edge) is visible in every result.
        "tyre_centre_ground_z": [None if t is None or t[2] is None else round(t[2], 4)
                                 for t in tyres],
        "rays": rays,
        # Colliders a probe ray STARTED inside (see SPAWN_ORIGIN_OVERLAP_M). Empty on
        # every start this repo has measured; non-empty means the start overlaps
        # something within SPAWN_PROBE_ABOVE_M of the episode z.
        "origin_inside": origin_inside,
        # Casts that met the robot's own colliders. 0 when the robot was moved out of the
        # way first (park_for_probe); otherwise the floor under a resting tyre can read
        # low or not at all -- see _floor_below.
        "robot_hits": seen["robot_hits"],
    }
    if not found:
        record.update(ground_z=None, ground_prim="", snapped_z=round(z0, 4),
                      drop_removed_m=0.0)
        return (x0, y0, z0), record
    ground_z, ground_prim = max(found)
    z = ground_z + BASE_LINK_REST_Z_M + SPAWN_CLEARANCE_M
    record.update(ground_z=round(ground_z, 4), ground_prim=ground_prim,
                  snapped_z=round(z, 4), drop_removed_m=round(z0 - z, 4))
    return (x0, y0, z), record


# --- clearing the robot out of its own floor probe ---------------------------------------
# ground_spawn_pose() cannot see past the robot: raycast_closest has no filter, and a tyre
# resting on the floor is tangent to it exactly where the probe samples (see _floor_below).
# The robot is usually standing on or next to the start it is about to be reset to -- the
# stage is authored with it on one episode's start, starts are shared (hospital e01-e05),
# and the UI and route_drive reset the same start repeatedly. park_for_probe() lifts it this
# far straight up for one Kit update, which is what PhysX's scene query needs to see it
# gone; reset_to() then puts it on the start and zeroes its velocity. 50 m clears every
# collider over a start on these stages (indoor ceilings ~3-4 m, warehouse pillars 9 m;
# outdoor buildings carry no colliders) and is 40x the probe's 0.30 m reach above a start.
PROBE_PARK_DZ_M = 50.0


# --- settling after a reset --------------------------------------------------------------
# KinematicBase.settle() holds the robot level and steps until it is AT REST, at least
# SETTLE_MIN_STEPS and at most the caller's cap. "At rest" is judged on the post-step state
# SETTLE_QUIET_STEPS times in a row:
#   * tilt <= SETTLE_TILT_DEG. Held, the orientation is pinned level every step, so the
#     post-step tilt is one step of rotation: 0.05-0.14 deg on all six outdoor starts,
#     pillars' 6.7 deg slope included (triage outdoor_verify, 30 held steps). A free settle
#     left pillars rolled 14.0 deg and still rocking -- that is what the hold is for;
#   * |v| <= SETTLE_SPEED_MPS. A held body reads the contact solver's per-step response, not
#     motion: 0.033-0.075 m/s on the five flat outdoor starts, 0.137-0.141 on pillars, where
#     the level-pinned body pivots on its high tyre each step and is put back. One step of
#     free fall reads g*dt = 0.16 m/s and two 0.33 (the written vz carries the sim's), so the
#     bound separates "standing" from "still dropping onto the floor";
#   * |dz| per step <= SETTLE_DZ_M: the highest tyre has landed its SPAWN_CLEARANCE_M.
# Not reaching that is recorded (`settled: False`) and announced, never silently cut short.
SETTLE_MIN_STEPS = 10
SETTLE_QUIET_STEPS = 10
SETTLE_TILT_DEG = 1.0
SETTLE_SPEED_MPS = 0.25
SETTLE_DZ_M = 1e-3


class KinematicBase:
    """Velocity command -> robot pose. See the module docstring.

    `hold_when_stopped` (default True, the navigation runner's mode): a zero command HOLDS
    the robot -- the pose is pinned level at the integrated x/y and yaw every step, exactly
    as a moving step pins its start, and only z follows the sim. So a stopped robot cannot
    creep, rock or keep a spin, the camera stays level, and the x/y the jump test compares
    against is always one step old. False (scripts/control/control_terminal.py): a zero
    command RELEASES the robot to the physics after one zero-velocity write -- the arms of
    a robot pinned every step do not converge (CLAUDE.md, "Kinematic root-teleporting
    fights concurrent arm-joint convergence"), and that tool commands arms with the base
    stopped. The next moving command then starts from wherever the physics left it.

    `velocity_targets`: a joint velocity-target array to write the wheel speeds INTO,
    when the caller keeps its own persistent one for the other joints (CLAUDE.md rule 6).
    Otherwise the class keeps a private array whose non-wheel entries stay 0.

    `motion_filter`: see `_resolve_motion()` -- the one place a commanded displacement
    becomes the next pose. None (the default) passes it through unchanged.
    """

    # If the sim's idea of where the robot is diverges from ours by more than this,
    # something moved it that was not us. Well above per-step wheel drag (~1 cm) and well
    # below any distance that matters to an episode. What it MEANS depends on whether we
    # were driving: see _judge.
    RESYNC_TOLERANCE_M = 0.25
    # The vertical twin, per PHYSICS step. The drive never writes z, but it writes vz
    # clamped to [VZ_MIN_MPS, VZ_MAX_MPS], so one step of our own motion moves z by at most
    # (2 + g*dt)*dt = 3.6 cm down and 0.5 cm up. Kerbs and step-downs add the contact
    # solver's share: the largest single step on any fixed-base run is 59 mm (the curb
    # wedge that ended 20261005-113221_outdoor_upsway_far), and a 15 cm island step-down
    # lands over ~10 steps. A z change past this in ONE step is the solver throwing the
    # body -- the jump test used to look at x/y only, so a 0.30 m vertical pop passed.
    Z_STEP_FAULT_M = 0.15
    # The sim's own vz, kept but clamped. Down to -2 m/s so a robot that is not on its
    # floor (a missed snap, a ledge) reaches it in a fraction of a second; up to only
    # +0.3 m/s so a kerb pop or a depenetration push cannot become a ballistic launch.
    # With planar speed capped at 1.5 m/s the sim then never reads faster than
    # hypot(1.5, 2 + g*dt) = 2.63 m/s on a healthy step (run_navigation.fault_speed_mps).
    VZ_MIN_MPS = -2.0
    VZ_MAX_MPS = 0.3

    def __init__(self, articulation, dof_names: list[str], *,
                 hold_when_stopped: bool = True,
                 velocity_targets: np.ndarray | None = None):
        self.art = articulation
        self.dof_names = dof_names
        self.hold_when_stopped = hold_when_stopped
        self._wheel_indices = [
            dof_names.index(name) for name in WHEEL_JOINTS if name in dof_names
        ]
        self._velocity_targets = (
            velocity_targets if velocity_targets is not None
            else np.zeros((1, len(dof_names)), dtype=np.float32))
        # The seam (see _resolve_motion). Kept as an attribute rather than a subclass hook
        # so a contact layer can be attached to an existing base without touching it.
        self.motion_filter = None
        self._yaw = 0.0
        self._xy: np.ndarray | None = None
        self._initialized = False
        # Released mode only: True while the physics owns the base (see the class
        # docstring). A released base starts released -- nothing has driven it yet.
        self._released = not hold_when_stopped
        # Vertical motion the DRIVE never asked for. `apply()` writes x and y but carries
        # z straight through from whatever the contact solver last produced, so any
        # bouncing of the wheel spheres against the floor is written back into the pose
        # every step -- and the chase camera hangs off `base_link`, so it lands in the
        # recorded video as a shake. Measured rather than argued about: `z_step_max` is
        # the largest single-step change, which is what the eye reads as jitter, and the
        # min/max span says whether the robot is also settling or climbing.
        self.z_min = float("inf")
        self.z_max = float("-inf")
        self.z_step_max = 0.0
        self._z_prev: float | None = None
        # Physics faults: physics steps on which the sim moved the robot more than
        # RESYNC_TOLERANCE_M sideways, or Z_STEP_FAULT_M vertically, while this class was
        # holding its velocity. See _judge.
        self.faults = 0
        self.first_fault: dict | None = None
        self.last_fault: dict | None = None
        # Legitimate resyncs: the sim moved the robot while nothing here controlled it.
        self.resyncs = 0
        # Windows in which a different amount of physics ran than the apply() that opened
        # them commanded -- more than half a physics step off. Their steps are still judged
        # (they ran under our velocity), but the integrated pose and the sim's then disagree
        # by the excess, so a nonzero count says the caller's dt is not Kit's update.
        self.step_anomalies = 0

        # --- per-physics-step judgement --------------------------------------------
        # apply() opens a WINDOW: "for the next `dt` seconds of physics the robot moves at
        # this velocity from this pinned start". A post-step callback checks every physics
        # step inside it against xy0 + v * elapsed, so a throw is caught on the physics step
        # it happens, however many steps Kit runs per update (1 in the runner; the G1 work
        # runs 3.33). The window stays open until it is judged -- by check_step() right
        # after the update, or by the next apply() -- and every step until then ran under
        # the velocity it wrote, so every one of them is judged against it.
        self._win_open = False
        self._win_budget_s = 0.0
        self._win_elapsed_s = 0.0
        self._win_steps = 0
        self._win_xy0 = np.zeros(2, dtype=np.float64)
        self._win_v = (0.0, 0.0)
        self._win_z0 = 0.0
        self._win_z_prev = 0.0
        self._win_fault: dict | None = None
        self._win_step_dt = 0.0
        # A post-step read failed inside this window: judge it at its end instead.
        self._win_unread = False
        # Physics steps seen, and the count at the last judgement, so steps nobody
        # commanded (boot, idle, a caller stepping Kit on purpose) are judged as external
        # moves. None if Kit's step stream is missing (offline tools), in which case every
        # window is assumed to be exactly one physics step.
        self._steps: int | None = 0
        self._steps_at_judge = 0
        # True when Kit offers the post-step stream, so every physics step is judged as it
        # happens; False on a count-only stream, where each window is judged at its end. A
        # single failed read does not clear it: only that window (`_win_unread`) falls back
        # to its end, with a one-time warning.
        self._per_step = False
        # Only meaningful without a step stream: a judgement has been made since the last
        # apply(), so the next one must not judge the same pose again.
        self._judged_since_apply = False
        self._warned: set[str] = set()
        self._step_sub = None
        try:
            from omni.physx import get_physx_interface

            physx = get_physx_interface()
            try:
                # pre_step=False: called right AFTER each physics step.
                self._step_sub = physx.subscribe_physics_on_step_events(
                    self._on_post_step, False, 0)
                self._per_step = True
            except (AttributeError, TypeError):
                self._step_sub = physx.subscribe_physics_step_events(self._on_count_step)
        except Exception:  # noqa: BLE001 -- no Kit (offline tools): assume 1 step/window
            self._steps = None

    def close(self) -> None:
        """Drop the physics-step subscription (a second base in one process would
        otherwise keep this one's callback alive)."""
        self._step_sub = None

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[base] WARNING: {message}", flush=True)

    def _window_tick(self, step_dt: float) -> bool:
        """Count one physics step; True if it falls inside an open window."""
        self._steps += 1
        if not self._win_open:
            return False
        self._win_steps += 1
        self._win_elapsed_s += float(step_dt)
        self._win_step_dt = float(step_dt)
        return True

    def _on_count_step(self, step_dt: float) -> None:
        self._window_tick(step_dt)

    def _on_post_step(self, step_dt: float, *_context) -> None:
        """Judge ONE physics step inside the open window. Never raises into PhysX."""
        if not self._window_tick(step_dt):
            return
        if self._win_fault is not None or not self._per_step:
            return
        valid = getattr(self.art, "is_physics_handle_valid", None)
        try:
            handle_ok = valid is None or bool(valid())
        except Exception:  # noqa: BLE001 -- never raise into PhysX's step
            handle_ok = False
        if not handle_ok:
            # Timeline stopped and played under an open window (control_terminal's
            # `sim stop|play`, the GUI buttons): the handle is re-initialised and the
            # window closed by sync_from_sim() before anyone drives again. Expected, so
            # no warning; if something judges first, it judges at the window's end.
            self._win_unread = True
            return
        try:
            positions, _ = self.art.get_world_poses()
            p = [float(v) for v in positions[0][:3]]
        except Exception as exc:  # noqa: BLE001 -- never raise into PhysX's step
            # This window is judged at its end instead (see _judge).
            self._win_unread = True
            self._warn_once("read", f"post-step pose read failed ({exc!r}); that window "
                                    "is judged at its end instead")
            return
        ex = self._win_xy0[0] + self._win_v[0] * self._win_elapsed_s
        ey = self._win_xy0[1] + self._win_v[1] * self._win_elapsed_s
        jump = math.hypot(p[0] - ex, p[1] - ey)
        dz = abs(p[2] - self._win_z_prev)
        self._win_z_prev = p[2]
        if jump > self.RESYNC_TOLERANCE_M or dz > self.Z_STEP_FAULT_M:
            self._win_fault = self._fault_record(jump, dz, (ex, ey), p, self._win_steps)

    def _fault_record(self, jump: float, dz: float, expected_xy, sim_xyz,
                      sub_step: int | None) -> dict:
        record = {
            "kind": "jump" if jump > self.RESYNC_TOLERANCE_M else "z_jump",
            "jump_m": round(jump, 4),
            "dz_m": round(dz, 4),
            "expected_xy": [round(float(v), 4) for v in expected_xy],
            "sim_xyz": [round(float(v), 4) for v in sim_xyz[:3]],
        }
        if sub_step is not None:
            # Which physics step of the window: 1 in the runner, up to ceil(dt/physics dt).
            record["physics_step"] = sub_step
        return record

    def _note_z(self, z: float) -> None:
        z = float(z)
        self.z_min = min(self.z_min, z)
        self.z_max = max(self.z_max, z)
        if self._z_prev is not None:
            self.z_step_max = max(self.z_step_max, abs(z - self._z_prev))
        self._z_prev = z

    def reset_stats(self) -> None:
        """Start the per-episode measurements over: z span/jitter, faults, resyncs.

        Separate from reset_to() because the settle that follows a reset is not part of
        the episode, and because one UI process runs many episodes -- without this the
        second run's `base_z_span_m` silently included the first run's.
        """
        self.z_min = float("inf")
        self.z_max = float("-inf")
        self.z_step_max = 0.0
        self._z_prev = None
        self.faults = 0
        self.first_fault = None
        self.last_fault = None
        self.resyncs = 0
        self.step_anomalies = 0

    def _close_window_at(self, t) -> None:
        """Forget any open window and take the CURRENT step count as judged: whatever
        ran before this moment belongs to whoever just set the pose (sync, reset)."""
        self._win_open = False
        self._win_fault = None
        self._win_unread = False
        if self._steps is not None:
            self._steps_at_judge = self._steps
        self._judged_since_apply = False

    def sync_from_sim(self) -> None:
        """Read the true pose out of the sim.

        Called once at episode start, and never again mid-episode: the integrated pose
        below IS the authority afterwards. Re-reading it every step would fight the
        teleport (the pose we just wrote is the pose we would read back).

        **That was true of yaw and stated here, but position was re-read every step
        anyway, and it cost 11%.** The wheels are commanded at the kinematically
        correct rate and carry no load, but traction is not zero, so each step the
        robot was dragged a little on top of the `vx * dt` the teleport had already
        applied -- and because the next step read that dragged position back as its
        starting point, the error compounded instead of cancelling. Measured over a
        full episode: 0.663 m/s mean against a 0.600 m/s cap, 1.11x.

        That is not a cosmetic discrepancy at this scale. Over the 32.4 m
        `hospital_down_hallway` episode it is ~3.5 m of extra travel, and the episode
        failed by plateauing 2.88 m from its goal. Integrating position the same way
        yaw already was makes the commanded speed exact.
        """
        positions, orientations = self.art.get_world_poses()
        w, _x, _y, z = orientations[0]
        self._yaw = 2.0 * math.atan2(z, w)
        self._xy = np.array([positions[0][0], positions[0][1]], dtype=np.float64)
        self._initialized = True
        self._close_window_at(positions[0])

    @property
    def yaw(self) -> float:
        return self._yaw

    def position(self) -> tuple[float, float, float]:
        positions, _ = self.art.get_world_poses()
        # Sampled here rather than inside `apply()`: this is called once per loop
        # iteration whatever the command was, so a robot held still by the guard or
        # standing through a synchronous decision is still watched.
        self._note_z(positions[0][2])
        return tuple(float(v) for v in positions[0])

    def physics_state(self) -> dict:
        """What the SIM says the body is doing: speed, roll, pitch, z.

        The runner's fault detector reads this after every step (run_navigation.py). The
        drive pins the orientation level every step (moving or held) and writes a planar
        velocity, so on a healthy step these are the commanded speed (<= 1.5 m/s, plus a
        clamped vz while falling: <= 2.63 m/s) and one step's worth of tilt (hospital: p99
        |w| 1.66 rad/s, roll/pitch within 0.2 deg; outdoor routes <= 6.8 deg).

        Contacts add short spikes on top, and they are not small. Over every per-step
        replay through this base (53 outdoor, 8 hospital, 3 office runs; scratch
        physfix/faults/spike_rules.txt): island-kerb push-throughs 12.00 m/s on one step
        and 11.19 on two, a barrier 4.54-4.77 on at most two, a chair climb 10.98 on at most
        two -- isolated, absorbed by the next write. Sustained only where the body is being
        driven INTO geometry: stair soffit 4 steps in a row (14.87 m/s), a reception desk 5,
        a kerb wedge under a traffic island 3 at 3.10 m/s (the upsway_far fault). See
        run_navigation.FAULT_SPEED_STEPS for the rule sized on these.
        """
        positions, orientations = self.art.get_world_poses()
        lin = np.asarray(self.art.get_linear_velocities(), dtype=np.float64).reshape(-1)
        w, x, y, z = (float(v) for v in orientations[0])
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        return {
            "speed_mps": float(np.linalg.norm(lin[:3])),
            "vz_mps": float(lin[2]),
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "z": float(positions[0][2]),
        }

    def _vertical_velocity(self) -> float:
        """The vz written with every pose. The drive never commands z.

        Chosen by measurement between two candidates (2026-10-05, outdoor stages, both
        with everything else in this file identical):

        * **0** -- the robot can only descend by what gravity adds inside one step
          (g*dt^2 = 2.7 mm), i.e. at most 0.16 m/s. Spawned 0.64 m above upsway's floor
          it hovered for ~4 s (gap to the floor p95 0.567 m); 0.93 m above pillars, gap
          p95 0.72 m and up to 0.93 m.
        * **the sim's vz clamped to [VZ_MIN_MPS, VZ_MAX_MPS]** -- the same drops land in
          ~0.3 s: gap p95 0.0087 m (upsway) and 0.017 m (pillars), z step max 36 mm,
          max read speed 2.16 m/s.

        On the routes themselves the two are the same to the millimetre (upsway, library,
        ramp_fountain, pillars at 1.5 m/s: z span within 2 mm, gap p95 within 1.4 mm,
        path/time speed 1.5001-1.5032 m/s; umbrellas' first 20 s: 0.0531 vs 0.0526 m), so
        the only difference is what happens when the robot is NOT on its floor, and there
        following the fall is the physical answer. The snap in ground_spawn_pose() makes
        that rare; this makes it harmless when it happens anyway.
        """
        lin = np.asarray(self.art.get_linear_velocities(), dtype=np.float64).reshape(-1)
        return float(min(self.VZ_MAX_MPS, max(self.VZ_MIN_MPS, lin[2])))

    def _write_velocity(self, vx_world: float, vy_world: float, omega: float) -> None:
        # Angular velocity is (0, 0, omega) and nothing else, every step: the orientation
        # is forced level each step, and a roll/pitch rate left alive under a forced-level
        # pose is the pump described in the module docstring.
        self.art.set_velocities(np.array(
            [[vx_world, vy_world, self._vertical_velocity(), 0.0, 0.0, omega]],
            dtype=np.float32))

    def _judge(self, t) -> None:
        """Decide what has happened since the last judgement; `t` is the sim's pose now.

        Under our velocity, one physics step moves the robot by v*dt <= 2.5 cm at the
        1.5 m/s cap, plus wheel drag and friction of a few mm. A step that moved it more
        than RESYNC_TOLERANCE_M (15 m/s) sideways, or Z_STEP_FAULT_M vertically, is not an
        external move, it is the contact solver throwing the body -- and following it is
        how the outdoor runs recorded km-scale drives: horizontal speed reached 15 m/s at
        t = 7.5 s on umbrellas, the old rule handed x/y to the sim, and the trace recorded
        the flight. So a jump on a step we controlled is counted as a FAULT and NOT
        followed; our integrated pose stays the authority and the runner ends the episode.

        Steps we did NOT control -- after sync_from_sim(), reset_to() or stop(), or after
        the window was judged when the caller stepped Kit again without commanding the
        base -- move the robot for somebody else's legitimate reason (a fall, an external
        teleport, a reset done around us), and there the sim still wins.

        Every physics step is judged by itself (_on_post_step), not by assuming one step
        per update: that assumption used to be built in, and with several physics steps
        per update (the G1 work runs 3.33) every throw would have read as a resync. A
        window that ran more or less physics than its apply() asked for is still judged
        step by step, and counted in `step_anomalies`. Without a step stream (offline
        tools) a window is taken as exactly one step.
        """
        if self._win_open:
            self._win_open = False
            self._judged_since_apply = True
            if self._steps is None:
                # No step stream: the window is assumed to be the one step it should be.
                jump = math.hypot(float(t[0]) - self._xy[0], float(t[1]) - self._xy[1])
                dz = abs(float(t[2]) - self._win_z0)
                if jump > self.RESYNC_TOLERANCE_M or dz > self.Z_STEP_FAULT_M:
                    self._record_fault(self._fault_record(jump, dz, self._xy, t, None))
                return
            self._steps_at_judge = self._steps
            if self._win_steps == 0:
                return                      # nothing ran since the apply (paused timeline)
            if abs(self._win_elapsed_s - self._win_budget_s) > 0.5 * self._win_step_dt:
                self.step_anomalies += 1
                self._warn_once(
                    "budget",
                    f"{self._win_steps} physics step(s) = {self._win_elapsed_s * 1000:.1f} ms "
                    f"ran in a window an apply() opened for {self._win_budget_s * 1000:.1f} ms; "
                    "every step is still judged, but the integrated pose drifts from the sim "
                    "by the difference -- pass apply() the dt Kit actually steps")
            fault = self._win_fault
            if fault is None and (not self._per_step or self._win_unread):
                # Count-only stream, or a read that failed: judge the window at its
                # end against the same expectation -- exact for one step, blind to a
                # throw undone inside it.
                ex = self._win_xy0[0] + self._win_v[0] * self._win_elapsed_s
                ey = self._win_xy0[1] + self._win_v[1] * self._win_elapsed_s
                jump = math.hypot(float(t[0]) - ex, float(t[1]) - ey)
                dz = abs(float(t[2]) - self._win_z0)
                if jump > self.RESYNC_TOLERANCE_M or dz > self.Z_STEP_FAULT_M:
                    fault = self._fault_record(jump, dz, (ex, ey), t, None)
            self._win_fault = None
            if fault is not None:
                self._record_fault(fault)
            return
        if self._steps is None:
            if self._judged_since_apply:
                return
        elif self._steps == self._steps_at_judge:
            return
        # Steps nobody controlled: a divergence is somebody else's legitimate move.
        jump = math.hypot(float(t[0]) - self._xy[0], float(t[1]) - self._xy[1])
        if jump > self.RESYNC_TOLERANCE_M:
            self.resyncs += 1
            self._xy[:] = (t[0], t[1])
        if self._steps is not None:
            self._steps_at_judge = self._steps
        self._judged_since_apply = True

    def _record_fault(self, fault: dict) -> None:
        self.faults += 1
        self.last_fault = fault
        if self.first_fault is None:
            self.first_fault = dict(fault)

    def check_step(self) -> None:
        """Judge the physics that just ran. Call it right after `kit.update()`.

        Judging used to happen only at the top of the NEXT apply(), one step late: by then
        the runner had already read the thrown pose for its success check, its path
        length, its trace and its final position -- a one-step 0.5 m throw that landed
        inside the goal radius latched SUCCESS with physics_faults = 0 (fake-Kit harness
        driving the real run_episode, 2026-10-05). Called here, the throw is a fault before
        anything reads it. apply() still judges whatever was not judged this way, so
        callers that never call this behave as before.
        """
        if not self._initialized or self._xy is None:
            return
        positions, _ = self.art.get_world_poses()
        self._judge(positions[0])

    def _release(self) -> None:
        """Released mode: stop the body once, then leave it to the physics."""
        if self._released:
            return
        positions, _ = self.art.get_world_poses()
        self._judge(positions[0])
        for idx in self._wheel_indices:
            self._velocity_targets[0][idx] = 0.0
        self.art.set_joint_velocity_targets(self._velocity_targets)
        self._write_velocity(0.0, 0.0, 0.0)
        self._released = True

    def _resolve_motion(self, dx: float, dy: float, dyaw: float,
                        dt: float, z: float) -> tuple[float, float, float]:
        """THE SEAM: the one place a commanded displacement becomes the next pose.

        (dx, dy) is this step's world-frame displacement and dyaw its rotation, as the
        command asks for them; what this returns is what the robot will actually do --
        apply() writes it as the velocity PhysX integrates across the step and adds it to
        the integrated pose, and nothing else in this class moves the robot. The contact
        layer (nav/sim/kinematic_contact.py, attached as `motion_filter` by the runner and
        every other driver of this class) goes here: it is called with the pose the step
        starts from, (x, y, yaw, z) -- z is the sim's, read once by apply() -- and returns
        a displacement, never a pose. A step it does not block comes back as the very
        objects it was given, so free space is bit-identical to no filter. Without one
        the command passes through unchanged: the bare kinematic base (NAV_CONTACT=0).
        """
        if self.motion_filter is None:
            return dx, dy, dyaw
        start = (float(self._xy[0]), float(self._xy[1]), float(self._yaw), float(z))
        rdx, rdy, rdyaw = self.motion_filter(start, dx, dy, dyaw, dt)
        return float(rdx), float(rdy), float(rdyaw)

    def apply(self, vx: float, vy: float, omega: float, dt: float) -> None:
        if not self._initialized:
            self.sync_from_sim()

        stopped = vx == 0.0 and vy == 0.0 and omega == 0.0
        if stopped and not self.hold_when_stopped:
            self._release()
            return
        if self._released:
            # The physics has owned the base since it stopped; start from where it is,
            # not from where it was when it was released.
            self.sync_from_sim()
            self._released = False

        # Spin the wheels at the visually correct rate even though they carry no load.
        for idx, speed in zip(self._wheel_indices, body_to_wheel_speeds(vx, vy, omega)):
            self._velocity_targets[0][idx] = speed
        self.art.set_joint_velocity_targets(self._velocity_targets)

        positions, _ = self.art.get_world_poses()
        t = positions[0]
        self._judge(t)

        yaw = self._yaw
        # Body-frame velocity rotated into world. Note vy is real here: on the
        # holonomic controller the robot genuinely translates sideways.
        dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        dx, dy, dyaw = self._resolve_motion(dx, dy, omega * dt, dt, float(t[2]))

        # Pin the START of the step to our integrated pose, level, at the yaw the step
        # starts from, and let PhysX integrate the commanded velocity across it. In steady
        # driving the teleport is then a no-op (the sim is already where we integrated
        # it), and the pose read after the step IS the integrated one: no step of lead, no
        # hop when the robot starts again from a stop. The motion is the same as writing
        # the post-step pose with the velocity on top (the counterfactual the fix was
        # measured with), minus those transients.
        #
        # A zero command takes the same path (hold_when_stopped): the pose is pinned and
        # the written velocity is zero. It used to write the zero velocity and NOT pin, so
        # a held robot was free to creep and rock: divergence accumulated over a long STOP
        # until the jump test read it as one step's throw (fake Kit: 1.2 mm/step of creep
        # faulted at 3.5 s with |v| = 0), and a robot settling on a slope stayed rolled
        # (pillars: 13-14 deg at the first decision's frame).
        #
        # Integrate x/y ourselves rather than reading the sim's value back and adding
        # to it. See sync_from_sim(): reading back folds wheel drag into the next
        # step's starting point, so the error compounds and the robot travels 1.11x
        # its commanded speed. _judge() is the only way the sim's x/y gets in.
        #
        # Z is carried through untouched, never recomputed. Forcing a target height
        # against the contact solver caused measurable drift before (0.0078 -> 0.0611 m
        # over 3 s) -- let gravity and the wheel contacts settle it.
        half = yaw / 2.0
        self.art.set_world_poses(
            positions=np.array([[self._xy[0], self._xy[1], t[2]]], dtype=np.float32),
            orientations=np.array([[math.cos(half), 0.0, 0.0, math.sin(half)]],
                                  dtype=np.float32),
        )
        if dt > 0.0:
            vxw, vyw, w = dx / dt, dy / dt, dyaw / dt
        else:
            vxw, vyw, w = 0.0, 0.0, 0.0
        self._write_velocity(vxw, vyw, w)

        # Open the window this apply() commands: `dt` of physics from the pinned start at
        # (vxw, vyw). _on_post_step judges each physics step of it. A zero-length command
        # (stop()) commands no physics, so it opens none: whatever runs after it -- the UI
        # idling between runs -- is not the drive's, and is followed if it moves the robot.
        self._win_open = dt > 0.0
        self._win_budget_s = float(dt)
        self._win_elapsed_s = 0.0
        self._win_steps = 0
        self._win_xy0 = self._xy.copy()
        self._win_v = (vxw, vyw)
        self._win_z0 = float(t[2])
        self._win_z_prev = float(t[2])
        self._win_fault = None
        self._win_unread = False
        self._judged_since_apply = False
        if self._steps is not None:
            self._steps_at_judge = self._steps

        self._yaw = yaw + dyaw
        self._xy += (dx, dy)

    def stop(self) -> None:
        self.apply(0.0, 0.0, 0.0, 0.0)

    def settle(self, kit, steps: int, dt: float, *,
               min_steps: int = SETTLE_MIN_STEPS) -> dict:
        """Hold a robot just placed by reset_to() until it is at rest: at most `steps`.

        The runner used to step Kit 30 times with nothing written, and on uneven tyre
        floors that was not enough: on outdoor_pillars (floors 6.300/6.272/6.306, a
        6.7 deg slope) the robot tipped onto its lower tyres and was still rocking at the
        end -- roll 14.0 deg, |v| 0.12 m/s -- so the first decision's camera frame was
        rolled ~13 deg, while every frame after it (the drive levels the body) was not.
        Left alone longer it does not come to rest either: it swings to ~2x the slope and
        tips over by step 120 (scenes/outdoor_small/probe2). Holding instead lets only z
        move: the highest tyre drops the 3 mm of SPAWN_CLEARANCE_M onto its floor and the
        robot is at rest, level, in the posture it will drive in.

        Stops as soon as the post-step state has been at rest (SETTLE_* above) for
        SETTLE_QUIET_STEPS steps in a row, after at least `min_steps`; `steps` is the cap.
        Returns what it saw, faults included -- the caller still ends with sync_from_sim()
        + reset_stats(), which would otherwise wipe a fault the settle itself caused.
        """
        quiet = 0
        done = 0
        z_prev = None
        state: dict = {}
        dz = float("nan")
        worst_tilt = 0.0
        for done in range(1, max(1, int(steps)) + 1):
            self.apply(0.0, 0.0, 0.0, dt)
            kit.update()
            self.check_step()
            state = self.physics_state()
            dz = abs(state["z"] - z_prev) if z_prev is not None else float("inf")
            z_prev = state["z"]
            tilt = max(abs(state["roll_deg"]), abs(state["pitch_deg"]))
            worst_tilt = max(worst_tilt, tilt)
            at_rest = (tilt <= SETTLE_TILT_DEG and state["speed_mps"] <= SETTLE_SPEED_MPS
                       and dz <= SETTLE_DZ_M)
            quiet = quiet + 1 if at_rest else 0
            if done >= min_steps and quiet >= SETTLE_QUIET_STEPS:
                break
        record = {
            "steps": done,
            "settled": quiet >= SETTLE_QUIET_STEPS,
            "tilt_deg": round(max(abs(state.get("roll_deg", 0.0)),
                                  abs(state.get("pitch_deg", 0.0))), 3),
            "worst_tilt_deg": round(worst_tilt, 3),
            "speed_mps": round(state.get("speed_mps", 0.0), 4),
            "dz_m": round(dz, 5) if dz == dz and dz != float("inf") else None,
            "z": round(state.get("z", 0.0), 4),
            "faults": self.faults,
            "first_fault": self.first_fault,
            "resyncs": self.resyncs,
        }
        return record

    def park_for_probe(self, kit) -> None:
        """Lift the robot PROBE_PARK_DZ_M straight up and step Kit once, so a following
        ground_spawn_pose() samples the floor, not the robot's own tyres. Call reset_to()
        right after the probe; nothing here is meant to survive it."""
        positions, orientations = self.art.get_world_poses()
        p = positions[0]
        self.art.set_world_poses(
            positions=np.array([[p[0], p[1], p[2] + PROBE_PARK_DZ_M]], dtype=np.float32),
            orientations=np.asarray(orientations, dtype=np.float32),
        )
        self.art.set_velocities(np.zeros((1, 6), dtype=np.float32))
        kit.update()
        # Whatever ran in that update was not the drive's: the next judgement must not
        # count the 50 m lift as anything.
        self._close_window_at(None)

    def reset_to(self, position: tuple[float, float, float], yaw: float) -> None:
        """Teleport back to a known pose — what the UI's Reset button does.

        Sets the integrated pose to match rather than re-reading it afterwards: the
        teleport and the internal state must agree, or the first drive step after a
        reset moves in the direction the robot was facing *before* it. The same now
        goes for x/y, which is integrated too -- leaving `_xy` stale here would make
        the first step teleport the robot straight back to where it was.

        Zeroes the wheel velocity targets too. Skipping that leaves the wheels spinning
        at whatever they were last commanded, which looks like the robot is still
        driving while it sits on the start line.

        **And zeroes the robot's actual velocities, root and joints.** A teleport moves
        the body and leaves its momentum alone, so this used to put a still-falling,
        still-spinning robot on the start line: (-0.43, 0.74, -0.12) m/s and
        (-2.24, -1.24, -0.42) rad/s at the reset of every outdoor run (it had fallen
        0.99 m during boot). Replayed with the velocities zeroed here and the same
        0.99 m drop kept, z span over the episode was 0.040 m instead of a blow-up. Pass a
        z from ground_spawn_pose() as well and there is no drop either.
        """
        # Wheel entries only: a private array has nothing else in it, and a shared one
        # (velocity_targets=) holds the caller's targets for the other joints.
        for idx in self._wheel_indices:
            self._velocity_targets[0][idx] = 0.0
        self.art.set_joint_velocity_targets(self._velocity_targets)

        half = yaw / 2.0
        self.art.set_world_poses(
            positions=np.array([list(position)], dtype=np.float32),
            orientations=np.array(
                [[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=np.float32
            ),
        )
        self.art.set_velocities(np.zeros((1, 6), dtype=np.float32))
        self.art.set_joint_velocities(np.zeros((1, len(self.dof_names)), dtype=np.float32))
        self._yaw = yaw
        self._xy = np.array([position[0], position[1]], dtype=np.float64)
        self._initialized = True
        self._close_window_at(position)
        self._released = not self.hold_when_stopped
