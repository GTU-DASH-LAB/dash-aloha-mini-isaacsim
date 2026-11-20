"""Contact response for the kinematic base: move-and-slide against the robot's real envelope.

WHY THIS EXISTS. The base is teleported (base_drive.KinematicBase), so PhysX never resolves a
contact against it: the root is pinned to the integrated pose every step and its velocity is
written, whatever the world says. Only two things touched the world at all -- the three wheel
spheres and the arm colliders -- and both lost. Replaying recorded hospital runs through the
FIXED base (2026-10-05) the arms (z 0.80-1.00 m) drove into the underside of the staircase
(17 steps over 3 m/s, |v| to 14.9 m/s), the lift carriage into a chair back (a wheel climbed to
z 0.2165 m, tilt 11 deg) and the arms into a reception desk; outdoors the wheel spheres pushed
through 12-15 cm island kerbs and either rode UNDER the island top or popped onto it. The
collision guard and the 2D lidar cannot see any of it: both are a plane at 0.30 m.

WHAT IT DOES. Before every teleport, each box of the robot's envelope is SWEPT along the step's
displacement (PhysX scene query, `sweep_box_all`). A hit closer than the displacement plus a
skin is a contact this step would make: the robot advances to the skin and the remaining motion
loses its component INTO each blocker (per blocker, as collision_guard.py does for its fan) and
keeps the rest, so it slides along a wall instead of stopping dead. The slid velocity is swept
once more, so a corner cannot be slid into. A turn is checked by OVERLAP at the end yaw with the
boxes grown by a rotation skin: a turn that would bring something new into that shell, or bring
something already in it deeper, is refused (translation is unaffected). Something already
inside the skin with no normal to slide on (spawned there, a person stepped in) may be moved
away from but never deeper into (the depth rule, _depth_ok). Nothing here writes to
PhysX; it only edits the displacement that KinematicBase.apply() then integrates, so every
existing invariant of the drive (pose and velocity written together, integrated x/y, the
jump/fault judge) is untouched.

INTEGRATION (base_drive.py and run_navigation.py are not this module's; this is the contract,
anchored on code text because the line numbers there move):
  * base_drive.py needs NO change. KinematicBase already has the seam: apply() calls
    `_resolve_motion(dx, dy, omega * dt, dt)`, which hands `start=(x, y, yaw)` to
    `self.motion_filter` when one is set. With no filter the base is what it was; with one, a
    step nothing blocks gets the caller's own (dx, dy, dyaw) objects back -- bit-identical by
    construction (measured: cruise, turn, arc and the OFF-paired replays below agree to the
    millimetre; the 12 normal replays had 0 blocked steps).
    Optional: append z (apply() has already read it as t[2]) as a 4th element of `start`,
    which saves the one pose read motion_filter() otherwise makes.
  * run_navigation.NavigationRunner.setup(), right after `self.base = KinematicBase(self.art,
    dof_names)` / `self.base.sync_from_sim()` (timeline playing, art initialised):
        from kinematic_contact import KinematicContact
        self.contact = (KinematicContact.from_stage(stage).attach(self.base)
                        if os.environ.get("NAV_CONTACT", "1") != "0" else None)
    `stage` is setup()'s own `usd_context.get_stage()`; the envelope is read from
    /World/Aloha's own meshes there, once. NAV_CONTACT=0 is the bare base, for A/B runs.
  * run_episode(), right after `self.base.sync_from_sim()` / `self.base.reset_stats()` (after
    the spawn snap and the held settle, before the first command), and the same in
    perform_reset(): `self.contact.reset_episode()`. It re-reads the link poses as the reset
    left them, zeroes the metric and forgets the clearance cache; missing it carries the last
    episode's contacts, and its free-space cache, into this one.
  * The control loop needs nothing: guard -> base.apply() already passes through the filter,
    and the recovery's reverse is swept like any other motion (the sweep is along the
    displacement, whichever way it points), so backing out cannot back into something.
  * EpisodeResult (episode.py) gets one field, `contact: dict | None = None`, filled with
    `self.contact.summary()` next to `physics_faults=` -- blocked / rotation / in-contact
    step counts, `first_contact` {t, prim, box, kind, distance_m}, spans, `max_run_s`,
    `physical_collision`, `by_prim`, `query_ms_per_step`. Print one line with it, like the
    spawn line. NEVER end an episode on it: DynaNav does not (below), and the robot that is
    held at a wall is exactly the robot the recovery exists for.
  * Recommended, stuck_recovery.py: a robot held by its ARMS (stair soffit, desk top) has
    nothing at the guard's 0.30 m plane, so `front_clearance_m() > stop_distance_m` and the
    stall is treated as a BALK after BALK_WINDOW_S = 5 s rather than a wedge after 3 s.
    Treat `contact.stats.run * dt >= STALL_WINDOW_S` (a contact still pressed) as wedge
    corroboration; the wedge reverse then fires on the evidence it was written for.
  * summarize_runs.py: columns `contact_steps`, `physical_collision`, `first_contact.prim`.
  * Other KinematicBase callers (scripts/control/control_terminal.py, nav/tools/
    replay_lidar_trace.py, check_guard_scan.py) attach the same way; control_terminal moves
    the arms and the lift, which REFRESH_EVERY covers.
  * Constants live HERE, each with its measurement. BASE_LINK_REST_Z_M comes from
    scripts/alohamini1_specs.py; DECK_SPLIT_Z_M must equal lidar_2d.COLUMN_Z_M[0] (same mesh
    slice), which nav/tools/check_kinematic_contact.py asserts.

THE ENVELOPE is the robot's own geometry at the current joint state, not a guessed box: every
gprim under /World/Aloha is bounded in its owning rigid body's frame once (UsdGeom.BBoxCache),
and on refresh() the link poses come from PhysX, so the lift and the arms are wherever the sim
holds them. Links are grouped into five boxes; at lift 0 / arms at rest, in base_link's frame
(`describe()`, nav_hospital, 2026-10-06; the base_link origin rests 0.0078 m above the floor):

    low       deck (base_link mesh below DECK_SPLIT_Z_M) + 3 wheel spheres (spin-invariant)
              x -0.211..+0.213, y -0.236..+0.150, z 0.037 (floor + MAX_STEP_M)..0.120
    column    base_link mesh above DECK_SPLIT_Z_M: x +-0.074, y -0.101..+0.154, z 0.121..1.084
    carriage  vertical_link (+ arm bases): x -0.214..+0.212, y -0.113..-0.012, z 0.589..0.825
    left_arm  left_link1..6:  x +0.126..+0.196, y -0.306..+0.012, z 0.771..1.005
    right_arm right_link1..6: x -0.188..-0.118, y -0.302..+0.016, z 0.772..1.006

Driving +x the robot is 0.460 m wide (arm tips at y -0.306 to the column's back at +0.154).

base_link has NO collider in the nav stages (fix_wheel_collision.py disabled its hull), which
is why the column and the deck must be modelled here: physically nothing stopped them.

THE FLOOR. The low box starts MAX_STEP_M above the floor the robot stands on (the base_link
origin is BASE_LINK_REST_Z_M above it), so a sill or floor seam lower than that never touches
it and the wheel spheres climb it in the physics as before. Hits on upward-facing surfaces
(normal_z >= WALKABLE_NORMAL_Z: ramps up to MAX_WALKABLE_SLOPE_DEG) are not blockers for the
low box either, and neither is an initial overlap with a prim the tyres are standing on --
but only where that prim stays below MAX_STEP_M + SUPPORT_LIFT_M inside the box
(_floor_prims): a kerb is one prim, and a tyre on its low end must not make its high end
floor. A kerb, a stair riser, a chair leg or a wall face is vertical, so it blocks.
Everything above the deck blocks whatever its normal. See MAX_STEP_M for the calibration.

IGNORED: the robot itself (/World/Aloha) and /World/People (people.py senses people
analytically; their PhysX capsules hold stale scene-query poses, see that module).

COLLISION METRIC, with DynaNav's semantics. DynaNav (TIC-VLA/DynaNav/benchmark.py) never ends
an episode on a collision -- the frame loop at :535-556 returns on goal reached and otherwise
runs to timeout -- and scores `collision = physical_collision or human_collision` (:180-184):
  * physical_collision: a ContactSensor on the robot's chassis_link (:1189-1192), created with
    min_threshold = 10 N (:130, :1216-1224), counted only when the reading stays >= 10 N for
    >= 100 consecutive frames (:131, :1276-1295; any frame below resets the count), at the
    benchmark's 30 frames per second (:426, :521) -- 3.33 s of continuous pushing;
  * human_collision: 2-D distance < 0.2 m to any character's SkelRoot (:690-701), people.py's
    job, not this module's.
Our analogue of "a contact with force" is a step on which the command pushed INTO an obstacle
and was clipped, or a turn into one was refused, or the box itself already overlapped one
(an obstacle merely inside the 2 cm skin is not a contact: sliding past a wall is not
colliding). A zero command never counts and ends a run, as a robot resting against a wall
reads no 10 N on DynaNav's sensor either. `contact_steps`, `max_run_s`, and
`physical_collision` = the longest run reaches DYNANAV_SUSTAINED_S in this caller's steps
(200 at 60 Hz). Like DynaNav, this module never ends an episode; the runner may, and should
say why if it does.

SCENE-QUERY SAFETY. `raycast_all` segfaults Kit on the outdoor stage (validate_episodes.py), and
so did an iterated `raycast_closest` ring with 30 m rays (scene/outdoor_small probe1/probe2,
crash inside libomni.physx.plugin's hashtable insert). The queries here are all short (one
step's displacement plus a skin, or an overlap of the robot's own size); `sweep_box_all`,
`overlap_box` and `raycast_closest` (five support rays of 0.145 m) ran 1.13 M times over
145 k filtered steps on all four nav stages (hospital 513 k, outdoor_small 276 k, warehouse
179 k, office 158 k; 636 k of them on the final module) with no crash and no PhysX error.

LIMITS, measured or by construction:
  * Vertical motion is not swept: z belongs to the sim (base_drive's clamped vz). Falling off
    a kerb edge, the low box dipped into the kerb top for 5 steps, at most 9.1 mm
    (ramp_fountain t = 58 s, island_6 kerb, identical with the module off); a lift raised
    into an overhang would likewise not be stopped.
  * Every collider is static to this module: a dynamic prop is stopped at like a wall, never
    pushed. nav_hospital has no rigid bodies outside the robot (scratch probe, 2026-10-06);
    the other stages were not probed for them.
  * Conservative where it must guess: the 5.5 cm step the tyres can climb is stopped at
    (MAX_STEP_M = 4.5 cm), and a 0.50 m doorway is refused while 0.52 m passes (SKIN_M).
  * Spawned or pushed INSIDE an obstacle with no slide normal, the depth rule lets the robot
    move sideways and out but never deeper; it does not push it out by itself.
  * People are ignored (IGNORE_PREFIXES); people.py senses them.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from alohamini1_specs import BASE_LINK_REST_Z_M  # noqa: E402

ROBOT_PRIM_PREFIX = "/World/Aloha"
BASE_LINK_PATH = "/World/Aloha/Geometry/base_link"
IGNORE_PREFIXES = (ROBOT_PRIM_PREFIX, "/World/People")

# --- calibration constants (each with its measurement; the full table is CALIBRATION, below) --
# Where base_link's mesh stops being the deck and becomes the lift column. base_link.STL sliced
# in 1-2 cm bands: up to z 0.12 the section spans x +-0.194, y -0.200..+0.150 (the deck, r 0.20);
# from 0.12 up it is x +-0.069, y -0.046..+0.147 and narrows further (the column). Same number as
# lidar_2d.COLUMN_Z_M[0], which was read off the same mesh.
DECK_SPLIT_Z_M = 0.12
# A floor feature lower than this, measured from the floor under the robot, is climbed by the
# wheel spheres in the physics and never reaches the envelope. Measured in Kit (synthetic steps on
# nav_hospital, 2026-10-05, bare base): the r 0.05 m sphere tyres climb 1.0-5.5 cm steps cleanly
# at 0.45 and 1.5 m/s (z rises by the step, tilt <= 4.7 deg, no |v| > 3 m/s), and from 7.0 cm up
# they do NOT climb -- the base is pushed through the step and rides 1.6-1.8 cm up inside it
# (7.0, 10, 13, 15 cm all alike). 4.5 cm sits inside the clean band, below the tyre radius (a step
# at or above the axle offers no upward contact normal; 5.5 cm climbed only on the solver's push)
# and above both things that must pass: the hospital's 2.6 cm floor-mesh step and a level deck
# over a 6.8 deg slope (0.320 m x tan 6.8 = 3.8 cm at the deck's far corner). With the final
# module on (Kit, 2026-10-06), 1.0-4.5 cm steps run identically to the bare base (disp 3.494-
# 3.502 m, same z rise, speed and tilt, 0 blocked) and 5.5-13 cm stop at the skin: disp 0.567-
# 0.568 m against 0.566 expected (face at 0.80 m - low box front 0.213 - skin - 1 mm backoff),
# 0 penetrating steps, where the bare base pushes through 7-15 cm on 116-392 steps.
MAX_STEP_M = 0.045
# Upward-facing surfaces the low box may meet without stopping: ramps up to this slope. The vz
# clamp in base_drive (VZ_MAX_MPS = 0.3) lets the robot climb atan(0.3 / v): 11.3 deg at the
# 1.5 m/s cap, 33.7 deg at 0.45 m/s -- the outdoor routes peak at 6.8 deg. Not binding on plain
# ramps: 5, 6.8, 10, 15 and 20 deg synthetic ramps (Kit, 2026-10-05) never reached the low box at
# all -- the front tyre lifts the base before the deck's leading edge meets the slope -- and ran
# bit-identical to the bare base. It matters where the terrain under the deck is uneven.
MAX_WALKABLE_SLOPE_DEG = 15.0
WALKABLE_NORMAL_Z = math.cos(math.radians(MAX_WALKABLE_SLOPE_DEG))
# Gap kept between the envelope and any obstacle when translating. Larger than one step at the
# speed cap would waste a doorway; smaller than a few mm would let PhysX's contact offset start
# generating contacts with the arm colliders. The robot is 0.460 m wide driving +x (arm tips at
# y -0.306 to the column's back at +0.154), so it needs 0.500 m. Measured (Kit, synthetic
# full-height doorways centred on the envelope, final module, 2026-10-06): 0.60, 0.55 and 0.52 m
# pass at 0.45 and 1.5 m/s exactly like the bare base (disp 1.995-1.998 m, 0 blocked); 0.50 and
# 0.45 m stop at the posts, 0.584 / 0.567 m in (the arm box / low box at the skin), pen 0. The
# bare base scrapes through 0.50 m (23 penetrating steps at 1.5 m/s) and drives its arms into
# the 0.45 m posts (36 arm-contact steps at 0.45 m/s). (An earlier module, 2026-10-05, held
# 0.52 m for 47 steps at 0.45 m/s, while its wheel bounds still depended on the spin angle and
# the low box reached y +0.160.)
SKIN_M = 0.02
# The shell checked for a turn. A turn at the controller's 1.2 rad/s cap swings the arm tips
# (radius 0.363 m) 7.3 mm per 1/60 s step, so 1 cm keeps the core box clear by construction.
ROT_SKIN_M = 0.01
# For the turn check only, the low box's bottom is raised this much more, to MAX_STEP_M + 1 cm =
# 5.5 cm: a turn swings the box's corners (r 0.320 m) over floor the support rays did not sample,
# and on a planar slope the uphill corner sits 0.320 x tan(a) above the floor under the tyres --
# 3.8 cm at 6.8 deg, so this clears planar slopes up to 9.8 deg. 5.5 cm is also the highest step
# the sphere tyres climbed cleanly in Kit (synthetic steps, nav_hospital, 2026-10-05: 5.5 cm
# climbed at 0.45 and 1.5 m/s, 7.0 cm pushed through), so whatever a turn ignores is something
# the physics climbs anyway. It was 3 cm (bottom at 7.5 cm): a turn could then swing the deck
# into a 5-7 cm obstacle the translation sweep would have stopped at (offline: a 6.5 cm bar
# beside the deck, check `lowbar`).
ROT_LOW_LIFT_M = 0.01
# A prim the tyres stand on (support_prims) is FLOOR for the low box only where it stays below
# MAX_STEP_M + this inside the box, i.e. 5.5 cm, the highest step the sphere tyres climbed cleanly
# (same measurement as ROT_LOW_LIFT_M). Support is decided per PRIM by five downward rays, and a
# kerb is one prim: on outdoor_small (20260905-115418 upsway window, Kit 2026-10-06) a deck corner
# stood over isl_curb piece_1303 where it is low, so the whole kerb became "floor" and the low box
# drove 2.6-5.6 mm into its high part with nothing flagged. The lifted overlap is what tells the
# two apart: uneven floor under the deck reaches a few cm into the box, a kerb reaches its top.
SUPPORT_LIFT_M = 0.01
# A sweep hit at or below this distance is an initial overlap (PhysX reports distance 0 and the
# normal as -direction), not a surface ahead.
INITIAL_OVERLAP_M = 1e-5
# A blocker whose normal is this close to vertical (|horizontal part| below this) gives no slide
# direction: the whole approach along the motion is removed instead.
MIN_HORIZONTAL_NORMAL = 0.2
# Re-read the link poses every this many resolve() calls. The nav runner never moves the arms or
# the lift, so this only matters to callers that do (control_terminal, a lift test); it is cheap.
REFRESH_EVERY = 30
# Shells swept, outermost first, for prims the outer shell already overlaps: the skin, half of
# it, the bare box, then the box SHRUNK (negative entries) so a prim the box already penetrates
# still yields a contact normal to slide on (see _blockers). The first entry stands for SKIN_M.
# The shrunk tiers were added after a hospital replay (down_hallway2 desk window, 2026-10-06):
# with only (skin, 1 cm, 0) a box parked on SM_WheelChair_01a had no normal at any tier, fell
# back to the depth rule, and crept 10.8 mm into it (below).
SHELL_TIERS_M = (None, 0.01, 0.0, -0.004, -0.012, -0.03)
# A sweep stops this far short of the shell's first contact, so the next step's shell starts
# clear of the obstacle instead of in initial overlap (which carries no normal).
BACKOFF_M = 1e-3
# THE DEPTH RULE, for something already inside the skin shell with no normal at any tier (a
# turn brought it there, the robot was placed there, a person stepped in, the lift moved): a
# move is allowed if it ends outside the box itself, or if it does not take the envelope DEEPER
# than the shallowest it has been since that contact began (a ratchet per box, see _depth_ok).
# Depth is continuous: the largest uniform shrink of the box-plus-skin (all six faces) that
# still overlaps, bisected to DEPTH_RES_M, capped at DEPTH_MAX_M; depth > SKIN_M means the box
# itself is penetrated by the difference. It replaced a 9-rung ladder whose rungs were
# 5 mm inside the skin and 1-4 cm past the box: a move that stayed on its rung counted as "not
# deeper", so in the desk window (Kit, nav_hospital, 2026-10-06) a turn walked the low box from
# inside the skin onto the box surface and the translations after it 10.8 mm into the
# wheelchair, all on one rung. DEPTH_TOL_M absorbs the bisection's own resolution only.
DEPTH_RES_M = 2.5e-4
DEPTH_TOL_M = 5e-4
DEPTH_MAX_M = SKIN_M + 0.05
# Free-space cache. A probe that finds nothing (but floor) within this margin of any box lets
# every following step skip the sweeps until the envelope could have moved margin - skin
# (translation + corner radius x rotation + |dz|). The sweeps cost ~0.1 ms per box per step,
# mostly in PhysX reporting the robot's own convex pieces, which every query of the envelope
# overlaps. Near obstacles a failed probe is retried only every CLEAR_PROBE_COOLDOWN steps.
CLEAR_MARGIN_M = 0.15
CLEAR_PROBE_COOLDOWN = 10
# DynaNav's physical_collision: >= 100 consecutive frames at 30 FPS (benchmark.py:131, :426).
DYNANAV_SUSTAINED_S = 100.0 / 30.0

# CALIBRATION. Kit 6.0.1, this module at md5 35625bd2, base_drive 3fda7115, 2026-10-06. Each
# row is a recorded or synthetic command trace driven through KinematicBase with the filter ON
# (and OFF = the bare base, same trace). pen = steps on which a box of the bare envelope
# overlaps a non-floor collider; armC = steps with a PhysX contact report on an arm collider
# (positive-separation contact-offset reports excluded); >3 = steps over 3 m/s; blk/rot =
# translation-clipped / turn-refused steps; max_run = longest contact run (the DynaNav metric).
# Harness: scratchpad physfix/contact/kc_harness2.py; offline logic: nav/tools/
# check_kinematic_contact.py (318 checks).
#
#   SYNTHETIC (nav_hospital, /World/ContactTest, at 0.45 and 1.5 m/s)
#   steps 1.0/2.0/2.6/3.5/4.5 cm   ON = OFF: disp 3.494-3.502 m, same z rise and tilt, blk 0
#   steps 5.5/7/10/13 cm           stop 0.567-0.568 m in (0.566 expected), pen 0
#                                  OFF: 5.5 climbs (pen 3-8); 7-15 cm pushed through, pen 116-392
#   ramps 5/6.8/10/15/20 deg       ON = OFF (disp, speed, tilt), blk 0
#   doors 0.60/0.55/0.52 m         ON = OFF, disp 1.995-1.998 m, blk 0
#   doors 0.50/0.45 m              stop at the posts, pen 0; OFF scrapes 0.50 (pen 23) and
#                                  drives its arms into 0.45 (armC 36/11, pen 76/22)
#   slab at arm / carriage height  (0.85 / 0.70 m) stop at the skin, pen 0, armC 0
#                                  OFF: armC 48-178, |v| up to 13.3 m/s
#   13 cm kerb at 45 deg           slides along it, 2.18 m, pen 0; OFF pushed through (pen 93-309)
#   wall at 45 deg, 0.6 m/s        slides 1.86 m, pen 0; OFF |v| 11.7 m/s, tilt 16 deg, armC 134
#   turn into a wall +-0.6/1.2 rad/s   refused at 15-19 deg of the 180 asked, pen 0
#   cruise 1.5 / 0.45 m/s, turn in place 0.6 rad/s, arc (0.8, 0.4): ON = OFF to the mm and
#                                  0.01 deg -- 8.995 m in 6 s (1.499 m/s), 2.695 m, 343.77 deg,
#                                  3.997 m / 183.35 deg; blk 0
#
#   HAZARDS (real command traces; OFF = bare base -> ON)
#   hospital forward_staircase [0:180 s]
#       OFF 102 steps > 3 m/s, vmax 13.2, tilt 8.5 ([160:180]: armC 280, pen 308)
#       ON  > 3: 0, pen 0, blk 302 / rot 265, left_arm on Section0, max_run 4.68 s
#   4B down_hallway chair, 0910-105400 [28:46]
#       OFF vmax 8.6, armC 236, pen 391
#       ON  pen 0, blk 319 (low box on SM_Chair_01a), max_run 5.30 s
#   down_hallway2 desk, 0902-110652 [196:218]
#       OFF vmax 5.5, armC 361, pen 402
#       ON  pen 0, armC 0, blk 368 / rot 179, max_run 2.35 s
#   upsway kerb, 0905-115418 [50:70]
#       OFF vmax 8.0, pen 249, climbs 10.6 cm onto the island
#       ON  pen 0, blk 398 / rot 190, max_run 4.53 s
#   upsway_far kerb, 0905-122644 [125:145]     OFF vmax 5.8, pen 70      ON pen 0, blk 101
#   upsway_far kerb, 1005-113221 [224:]        OFF pen 423               ON pen 0, blk 180 / rot 26
#   office hallway_turn (floor barrier)
#       OFF vmax 9.3, armC 277, pen 596
#       ON  pen 0, blk 725 / rot 198, max_run 6.13 s
#   warehouse_aisle6 (wet-floor sign)
#       OFF vmax 19.4, pen 693
#       ON  pen 0, blk 694 / rot 264, max_run 5.47 s
#   (the previous module, 5ebab3dc, let the kerb in through its support-prim rule: 115418 pen 307
#   at up to 5.6 mm, 113221 pen 20 at 2.3 mm -- the SUPPORT_LIFT_M fix)
#
#   NORMAL (real traces, 0 blocked, displacement = OFF / recorded): hospital vending_machine2
#   11.766 m, down_hallway 31.108 m (OFF 31.108), past_wheelchairs 12.475, down_hallway2 31.207,
#   exit_room 1.494; office nearest_elevator 13.708, passing_hallway 35.132; warehouse 17.180
#   and 25.139; outdoor umbrellas 14.462 (OFF 14.462), pillars 31.153, ramp_fountain [0:120 s]
#   24.938 m (pen 5: the kerb-edge fall under LIMITS, the same with the module off).
#
#   COST (self-time = perf_counter around resolve(), every scene query included)
#   free-space runs        0.08-0.10 ms/step; 95 % of steps answered by the clearance cache
#                          (down_hallway: 4607 of 4860 steps, 1265 queries in 81 s of driving)
#   staircase [0:180 s]    0.30 ms/step (14 368 queries)
#   hazard windows         0.35-1.75 ms/step while in contact (desk window worst, every step)
#   wall clock, interleaved OFF/ON/OFF/ON on a shared GPU0 (cost_clean, 2026-10-06):
#   down_hallway  OFF 20.0 / 22.9, ON 22.9 / 22.4 ms per step -- the same-mode spread (2.9 ms)
#                 is larger than any ON-OFF difference; the filter's 0.1 ms is not visible
#   staircase     OFF 22.4 / 22.1, ON 20.3 / 20.3 ms per step -- ON is CHEAPER: the bare
#                 base's arms in the stair cost PhysX more contact work than the filter costs
#   (OFF synthetic rows above were measured on base_drive 6caeb283 / 48e5f9ab, 2026-10-05; the
#   ON 1.0-4.5 cm steps and ramps on 3fda7115 reproduce them to the millimetre.)


@dataclass
class EnvBox:
    name: str
    center: np.ndarray      # base_link frame
    half: np.ndarray        # half extents, base_link frame axes
    low: bool = False       # the deck band: floor rules apply


@dataclass
class ContactResult:
    vx: float
    vy: float
    omega: float
    blocked: bool = False          # translation was changed
    rot_blocked: bool = False      # the turn was refused
    touching: bool = False         # something was already inside the skin shell
    in_contact: bool = False       # ...and inside the box itself (a real contact)
    prim: str = ""                 # nearest blocker (or touched prim)
    box: str = ""
    normal: tuple = (0.0, 0.0, 0.0)
    distance: float = float("inf")
    wx: float = 0.0                # the resolved velocity in the WORLD frame (what the
    wy: float = 0.0                # motion_filter seam integrates when `blocked`)


@dataclass
class ContactStats:
    steps: int = 0
    blocked_steps: int = 0
    rot_blocked_steps: int = 0
    touching_steps: int = 0        # something inside the skin shell at the start of the step
    in_contact_steps: int = 0      # ...inside the box itself
    contact_steps: int = 0         # blocked or rot_blocked or in_contact
    run: int = 0
    max_run: int = 0
    first_contact: dict | None = None
    by_prim: dict = field(default_factory=dict)
    by_box: dict = field(default_factory=dict)
    spans: list = field(default_factory=list)
    _span: list | None = None
    query_s: float = 0.0
    queries: int = 0
    cached_steps: int = 0          # steps answered by the clearance cache, no query at all
    clear_probes: int = 0
    # Low-box hits NOT treated as blockers, by reason and prim: what the floor rules let through.
    skip_support: dict = field(default_factory=dict)    # initial overlap with a support prim
    skip_walkable: dict = field(default_factory=dict)   # upward-facing surface


def _quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _group_of(link_path: str) -> str:
    name = link_path.rsplit("/", 1)[-1]
    if link_path == BASE_LINK_PATH:
        return "base"
    if name.startswith("wheel"):
        return "low"
    if name == "vertical_link":
        return "carriage"
    if name.startswith("left_link"):
        return "left_arm"
    if name.startswith("right_link"):
        return "right_arm"
    return name


class RobotEnvelope:
    """The robot's five boxes in base_link's frame, from its own meshes at the sim's joint state."""

    def __init__(self, stage, *, max_step_m: float = MAX_STEP_M,
                 base_link_path: str = BASE_LINK_PATH):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.base_link_path = base_link_path
        self.max_step_m = max_step_m
        root = stage.GetPrimAtPath(base_link_path)
        links = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        link_set = set(links)
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
                    UsdGeom.Tokens.guide]
        bbc = UsdGeom.BBoxCache(0.0, purposes, useExtentsHint=False)
        xfc = UsdGeom.XformCache()
        local: dict[str, list] = {}       # link -> [lo, hi] in the link's frame
        base_pts: list[np.ndarray] = []   # base_link mesh points, base_link frame
        # Wheels SPIN, so a box fixed in the wheel's frame is the wrong bound: the AABB of that
        # box rotated by the spin angle breathes by up to 41 % (the r 0.05 collision sphere's
        # bound read 0.100-0.137 m wide in base_link's frame at rest, probe_a). Refreshed every
        # REFRESH_EVERY steps, the low box's face then jumped 1-2 cm with the wheel angle --
        # into whatever the robot had stopped 2 cm short of, and only while the wheels kept
        # turning (a blocked command still spins them). So a wheel is bounded by the largest
        # distance of its geometry from the link origin (on the axle): spin-invariant.
        self.spin_r: dict[str, float] = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            owner = prim.GetParent()
            while owner and str(owner.GetPath()) not in link_set:
                owner = owner.GetParent()
            if not owner:
                continue
            lp = str(owner.GetPath())
            if _group_of(lp) == "low":
                m, _ = xfc.ComputeRelativeTransform(prim, owner)
                M = np.array(m, dtype=np.float64)
                if prim.IsA(UsdGeom.Sphere):
                    rad = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
                    r = float(np.linalg.norm(M[3, :3])) + rad * float(np.linalg.norm(M[:3, :3], axis=1).max())
                elif prim.IsA(UsdGeom.Mesh):
                    pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=np.float64)
                    r = float(np.linalg.norm(pts @ M[:3, :3] + M[3, :3], axis=1).max()) if pts.size else 0.0
                else:
                    rng = bbc.ComputeRelativeBound(prim, owner).ComputeAlignedRange()
                    r = 0.0 if rng.IsEmpty() else float(max(
                        np.linalg.norm([x, y, z]) for x in (rng.GetMin()[0], rng.GetMax()[0])
                        for y in (rng.GetMin()[1], rng.GetMax()[1]) for z in (rng.GetMin()[2], rng.GetMax()[2])))
                self.spin_r[lp] = max(self.spin_r.get(lp, 0.0), r)
            if lp == base_link_path and prim.IsA(UsdGeom.Mesh):
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=np.float64)
                if pts.size:
                    m, _ = xfc.ComputeRelativeTransform(prim, owner)
                    M = np.array(m, dtype=np.float64)          # row-vector convention
                    base_pts.append(pts @ M[:3, :3] + M[3, :3])
                continue
            rng = bbc.ComputeRelativeBound(prim, owner).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
            if lp in local:
                local[lp] = [np.minimum(local[lp][0], lo), np.maximum(local[lp][1], hi)]
            else:
                local[lp] = [lo, hi]
        self.links = [lp for lp in links if lp in local]
        self.local = local
        # base_link's own mesh, split into deck and column once: it is rigid with the frame.
        self.base_parts: dict[str, list] = {}
        if base_pts:
            pts = np.concatenate(base_pts)
            for name, mask in (("low", pts[:, 2] < DECK_SPLIT_Z_M),
                               ("column", pts[:, 2] >= DECK_SPLIT_Z_M)):
                if mask.any():
                    self.base_parts[name] = [pts[mask].min(0), pts[mask].max(0)]
        self.boxes: list[EnvBox] = []
        self.refresh_count = 0

    def refresh(self) -> list[EnvBox]:
        """Re-read every link pose from PhysX and rebuild the boxes in base_link's frame."""
        from omni.physx import get_physx_interface

        phx = get_physx_interface()
        tf = phx.get_rigidbody_transformation(self.base_link_path)
        Rb = _quat_xyzw_to_mat(tf["rotation"])
        pb = np.array(tf["position"], dtype=np.float64)
        groups: dict[str, list] = {k: list(v) for k, v in self.base_parts.items()}
        for lp in self.links:
            if lp == self.base_link_path:
                continue
            t = phx.get_rigidbody_transformation(lp)
            p = np.array(t["position"], dtype=np.float64)
            g = _group_of(lp)
            if lp in self.spin_r:
                c = (p - pb) @ Rb
                bl, bh = c - self.spin_r[lp], c + self.spin_r[lp]
            else:
                R = _quat_xyzw_to_mat(t["rotation"])
                lo, hi = self.local[lp]
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                    for z in (lo[2], hi[2])])
                body = ((corners @ R.T + p) - pb) @ Rb
                bl, bh = body.min(0), body.max(0)
            if g in groups:
                groups[g] = [np.minimum(groups[g][0], bl), np.maximum(groups[g][1], bh)]
            else:
                groups[g] = [bl, bh]
        boxes = []
        for name, (lo, hi) in groups.items():
            lo, hi = lo.copy(), hi.copy()
            low = name == "low"
            if low:
                # Floor + MAX_STEP up: base_link's origin rests BASE_LINK_REST_Z_M above the floor.
                lo[2] = max(lo[2], self.max_step_m - BASE_LINK_REST_Z_M)
                if hi[2] <= lo[2]:
                    continue
            boxes.append(EnvBox(name, (lo + hi) / 2.0, (hi - lo) / 2.0, low))
        self.boxes = boxes
        self.refresh_count += 1
        return boxes

    def describe(self) -> dict:
        return {b.name: {"min": [round(float(v), 4) for v in b.center - b.half],
                         "max": [round(float(v), 4) for v in b.center + b.half]}
                for b in self.boxes}


class KinematicContact:
    """Edit a velocity command so the envelope never enters an obstacle.

    resolve() takes the pose the step will START from (KinematicBase pins it there: integrated
    x/y and yaw, z as the sim has it) and a body-frame command, and returns the command to
    integrate. attach(base) installs the same thing on KinematicBase's `motion_filter` seam,
    which hands over a world-frame displacement instead (see motion_filter()).
    """

    def __init__(self, envelope: RobotEnvelope, *, skin_m: float = SKIN_M,
                 rot_skin_m: float = ROT_SKIN_M,
                 ignore_prefixes: tuple[str, ...] = IGNORE_PREFIXES,
                 refresh_every: int = REFRESH_EVERY, scene_query=None):
        self.env = envelope
        self.skin = skin_m
        self.rot_skin = rot_skin_m
        self.ignore = tuple(ignore_prefixes)
        self.refresh_every = refresh_every
        self._sq = scene_query
        self._since_refresh = 0
        self.stats = ContactStats()
        self.dt = 1.0 / 60.0
        # Clearance cache (see CLEAR_MARGIN_M): the pose at which every box grown by the margin
        # was found empty, and a cooldown before probing again after a probe that was not.
        self._clear: tuple | None = None
        self._probe_cooldown = 0
        self.last: ContactResult | None = None     # the latest step's answer, for the caller
        self._base = None
        # The depth rule's ratchet (see _depth_ok): box name -> (prims, shallowest depth since
        # the contact began). Entries not refreshed by a step are dropped at its end.
        self._touch_ref: dict[tuple, tuple] = {}
        self._touch_seen: set[str] = set()       # box names with a contact this step
        if not self.env.boxes:
            self.env.refresh()
        self._r_max = self._corner_radius()

    @classmethod
    def from_stage(cls, stage, **kw) -> "KinematicContact":
        env_kw = {k: kw.pop(k) for k in ("max_step_m",) if k in kw}
        return cls(RobotEnvelope(stage, **env_kw), **kw)

    # --- the KinematicBase seam -------------------------------------------------------------
    def attach(self, base) -> "KinematicContact":
        """Install on base_drive.KinematicBase's `motion_filter` seam; returns self."""
        self._base = base
        base.motion_filter = self.motion_filter
        return self

    def detach(self) -> None:
        if self._base is not None and getattr(self._base, "motion_filter", None) == self.motion_filter:
            self._base.motion_filter = None
        self._base = None

    def motion_filter(self, start, dx: float, dy: float, dyaw: float,
                      dt: float) -> tuple[float, float, float]:
        """KinematicBase._resolve_motion()'s contract: `start` = (x, y, yaw) the step starts
        from, (dx, dy) the WORLD-frame displacement and dyaw the rotation the command asks
        for; returns the displacement to integrate.

        z is not part of that contract, and the envelope's height above the floor is what
        separates a kerb from a floor seam, so it is read from the articulation -- the same
        read apply() made a few lines earlier. A 4th element in `start` is taken as z instead
        (one less tensor read, if base_drive ever passes t[2]).

        A step nothing blocked returns the caller's own (dx, dy, dyaw) objects: free space is
        bit-identical to running without the filter.
        """
        x, y, yaw = float(start[0]), float(start[1]), float(start[2])
        if dt <= 0.0:
            return dx, dy, dyaw
        if len(start) > 3:
            z = float(start[3])
        else:
            positions, _ = self._base.art.get_world_poses()
            z = float(positions[0][2])
        c, s = math.cos(yaw), math.sin(yaw)
        vx = (dx * c + dy * s) / dt
        vy = (-dx * s + dy * c) / dt
        r = self.resolve(x, y, z, yaw, vx, vy, dyaw / dt, dt)
        if r.blocked:
            dx, dy = r.wx * dt, r.wy * dt
        if r.rot_blocked:
            dyaw = 0.0
        return dx, dy, dyaw

    def reset_episode(self) -> None:
        """Call after every reset/settle, before the first command of an episode: re-reads the
        link poses (the arms and lift as the reset left them) and zeroes the metric."""
        self._refresh()
        self._since_refresh = 0
        self.reset_stats()

    # --- scene queries --------------------------------------------------------------------
    @property
    def sq(self):
        if self._sq is None:
            from omni.physx import get_physx_scene_query_interface

            self._sq = get_physx_scene_query_interface()
        return self._sq

    def _ignored(self, prim: str) -> bool:
        return prim.startswith(self.ignore)

    def _box_world(self, b: EnvBox, x, y, z, yaw, grow: float = 0.0, shrink: float = 0.0,
                   lift: float = 0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        cx = x + c * b.center[0] - s * b.center[1]
        cy = y + s * b.center[0] + c * b.center[1]
        hz = max(1e-3, float(b.half[2]) - lift / 2.0)
        cz = z + float(b.center[2]) + lift / 2.0
        hx = max(1e-3, float(b.half[0]) + grow - shrink)
        hy = max(1e-3, float(b.half[1]) + grow - shrink)
        return ((hx, hy, hz), (float(cx), float(cy), float(cz)),
                (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))

    def _sweep(self, b: EnvBox, pose, direction, dist, grow: float = 0.0):
        half, ctr, rot = self._box_world(b, *pose, grow=grow)
        hits = []

        def report(h) -> bool:
            prim = h.collision
            if not prim.startswith(self.ignore):
                n = h.normal
                hits.append((float(h.distance), prim, (float(n[0]), float(n[1]), float(n[2]))))
            return True

        self.stats.queries += 1
        self.sq.sweep_box_all(half, ctr, rot, (float(direction[0]), float(direction[1]), 0.0),
                              float(dist), report)
        return hits

    def _overlap(self, b: EnvBox, pose, grow: float = 0.0, shrink: float = 0.0,
                 lift: float = 0.0) -> set:
        half, ctr, rot = self._box_world(b, *pose, grow=grow, shrink=shrink, lift=lift)
        prims: set = set()

        def report(h) -> bool:
            prim = h.collision
            if not prim.startswith(self.ignore):
                prims.add(prim)
            return True

        self.stats.queries += 1
        self.sq.overlap_box(half, ctr, rot, report, False)
        return prims

    def support_prims(self, x, y, z, yaw: float = 0.0) -> set:
        """Prims straight under the robot's centre and the low box's corners: its floor."""
        out = set()
        floor_z = z - BASE_LINK_REST_Z_M
        low = next((b for b in self.env.boxes if b.low), None)
        pts = [(0.0, 0.0)]
        if low is not None:
            pts += [(float(low.center[0]) + sx * float(low.half[0]),
                     float(low.center[1]) + sy * float(low.half[1]))
                    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        c, s = math.cos(yaw), math.sin(yaw)
        for bx, by in pts:
            o = (x + c * bx - s * by, y + s * bx + c * by, floor_z + self.env.max_step_m)
            for _ in range(4):
                h = self.sq.raycast_closest(o, (0.0, 0.0, -1.0), self.env.max_step_m + 0.10)
                self.stats.queries += 1
                if not h or not h.get("hit"):
                    break
                prim = str(h.get("collision", ""))
                if not self._ignored(prim):
                    out.add(prim)
                    break
                d = float(h.get("distance", 0.0)) + 1e-3
                o = (o[0], o[1], o[2] - d)
        return out

    # --- the step -------------------------------------------------------------------------
    def _blockers(self, pose, d_hat, dist, support: set | None):
        """Sweep every box's SKIN SHELL (grown by skin_m in x/y) along d_hat for `dist`.

        Sweeping the shell rather than the box is what makes the gap a gap along the obstacle's
        NORMAL: a box swept toward a wall at 45 deg and stopped `skin` short along the motion is
        only skin * cos 45 from it.

        A shell that already overlaps something at the start gets no normal from PhysX (an
        initial overlap is reported at distance 0 with normal -direction), and it happens in
        practice: a robot parked 1 mm off a wall reads as overlapping it within a few steps
        (synthetic wall, 2026-10-05: stopped at step 80, initial overlap from step 89 on, while
        the pose did not change). So those prims are swept again with the shell grown by
        SHELL_TIERS_M[1], then with the bare box, then with the box shrunk, and a hit at an inner
        tier is a blocker at distance 0 -- no approach toward it, but its normal still lets the
        robot slide along it or leave. Only what overlaps even the most-shrunk box is left to the
        depth rule.

        Returns (blockers, touching, core, support): blockers are (distance, prim, box name, unit
        horizontal normal, raw normal); touching are (box, prim) overlapping the most-shrunk box;
        core are (box name, prim) the BARE box starts in overlap with (a real contact).
        """
        blockers, touching, core = [], [], []
        tiers = (self.skin,) + tuple(t for t in SHELL_TIERS_M[1:] if t < self.skin)
        for b in self.env.boxes:
            pending = None                     # prims still in initial overlap, by tier
            for k, grow in enumerate(tiers):
                initial = set()
                # A SHRUNK box starts |grow| clear of what the real box touches, so it must sweep
                # that much further to see what the real box meets within `dist` (up to
                # |grow| / MIN_HORIZONTAL_NORMAL along an oblique approach); each hit is then
                # judged by the real box's distance. Without this, a box parked 0.3 mm off a
                # wall drove into it: the 4 mm-shrunk sweep saw nothing within one step's
                # length, so there was neither a blocker nor a "touching" prim (offline jiggle).
                extra = (-grow) / MIN_HORIZONTAL_NORMAL if grow < 0.0 else 0.0
                for dd, prim, n in self._sweep(b, pose, d_hat, dist + extra, grow=grow):
                    if pending is not None and prim not in pending:
                        continue               # already handled at an outer tier
                    if dd <= INITIAL_OVERLAP_M:
                        if b.low:
                            if support is None:
                                support = self.support_prims(*pose)
                            if prim in support and prim in self._floor_prims(b, pose, {prim}, grow,
                                                                             support):
                                st = self.stats.skip_support
                                st[prim] = st.get(prim, 0) + 1
                                continue
                        initial.add(prim)
                        continue
                    if b.low and n[2] >= WALKABLE_NORMAL_Z:
                        st = self.stats.skip_walkable
                        st[prim] = st.get(prim, 0) + 1
                        continue
                    nh = math.hypot(n[0], n[1])
                    if nh < MIN_HORIZONTAL_NORMAL:
                        n_h = (-d_hat[0], -d_hat[1])
                    else:
                        n_h = (n[0] / nh, n[1] / nh)
                    if grow < 0.0:
                        approach = -(n_h[0] * d_hat[0] + n_h[1] * d_hat[1])
                        if approach <= 0.0:
                            continue           # moving along or away from it
                        if dd - (-grow) / max(approach, MIN_HORIZONTAL_NORMAL) >= dist:
                            continue           # the real box does not reach it this step
                    blockers.append((dd if k == 0 else 0.0, prim, b.name, n_h, n))
                if not initial:
                    break
                if grow <= 0.0:
                    core.extend((b.name, prim) for prim in initial if grow == 0.0)
                pending = initial
            else:
                touching.extend((b, prim) for prim in pending)
        return blockers, touching, core, support

    def _corner_radius(self) -> float:
        """Largest horizontal distance from the base_link origin to any box corner."""
        r = 0.0
        for b in self.env.boxes:
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    r = max(r, math.hypot(b.center[0] + sx * b.half[0], b.center[1] + sy * b.half[1]))
        return r

    def _refresh(self) -> None:
        old = [(b.name, b.center.copy(), b.half.copy()) for b in self.env.boxes]
        self.env.refresh()
        new = {b.name: b for b in self.env.boxes}
        moved = len(old) != len(new) or any(
            n not in new or float(np.max(np.abs(new[n].center - c))) > 1e-3
            or float(np.max(np.abs(new[n].half - h))) > 1e-3 for n, c, h in old)
        if moved:          # the arms or the lift moved: the clearance found before is void
            self._clear = None
            self._r_max = self._corner_radius()

    def _floor_prims(self, b: EnvBox, pose, prims: set, grow: float, support: set) -> set:
        """The prims among `prims` that are floor for low box b at `pose`: support prims that do
        not reach above MAX_STEP_M + SUPPORT_LIFT_M inside the box grown by `grow` + the skin.
        The extra skin is not a margin of safety but of measurement: the sweep that asks calls a
        prim 1-2 mm short of the shell "initial overlap" (PhysX's tolerance), the overlap query
        does not, so a lifted box of the sweep's own size can miss a kerb the sweep is touching
        (offline kerbcorner, diagonal approach: 12.5 mm into a 13 cm kerb)."""
        cand = prims & support
        if not cand:
            return set()
        return cand - self._overlap(b, pose, grow=max(grow, 0.0) + self.skin, lift=SUPPORT_LIFT_M)

    def _probe_clear(self, pose) -> bool:
        """True if no box grown by CLEAR_MARGIN_M (x/y) overlaps anything but the floor."""
        self.stats.clear_probes += 1
        support = None
        for b in self.env.boxes:
            prims = self._overlap(b, pose, grow=CLEAR_MARGIN_M)
            if prims and b.low:
                if support is None:
                    support = self.support_prims(*pose)
                prims -= self._floor_prims(b, pose, prims, CLEAR_MARGIN_M, support)
            if prims:
                return False
        return True

    def _overlap_shrunk(self, b: EnvBox, pose, s: float) -> set:
        """Overlap of the box grown by the skin on ALL six faces, then shrunk by s on all six:
        s = skin is the box itself. z counts too -- a slab meeting the box top is a vertical
        contact an x/y shrink would never resolve."""
        (hx, hy, hz), ctr, rot = self._box_world(b, *pose, grow=self.skin)
        half = (max(1e-4, hx - s), max(1e-4, hy - s), max(1e-4, hz + self.skin - s))
        prims: set = set()

        def report(h) -> bool:
            prim = h.collision
            if not prim.startswith(self.ignore):
                prims.add(prim)
            return True

        self.stats.queries += 1
        self.sq.overlap_box(half, ctr, rot, report, False)
        return prims

    def _depth(self, b: EnvBox, pose, prims: set) -> float:
        """How far `prims` reach into the skin shell, in metres: the largest shrink s (see
        _overlap_shrunk) at which they still overlap; -1 if the shell does not overlap them.
        > skin means they reach the box itself. Bisected to DEPTH_RES_M, capped at DEPTH_MAX_M."""
        if not self._overlap_shrunk(b, pose, 0.0) & prims:
            return -1.0
        lo, hi = 0.0, DEPTH_MAX_M
        if self._overlap_shrunk(b, pose, hi) & prims:
            return hi
        while hi - lo > DEPTH_RES_M:
            mid = 0.5 * (lo + hi)
            if self._overlap_shrunk(b, pose, mid) & prims:
                lo = mid
            else:
                hi = mid
        return lo

    def _depth_ok(self, b: EnvBox, start, end, prims: set, kind: str = "t") -> tuple[bool, float]:
        """The depth rule: may the envelope go from `start` to `end` with `prims` inside box b's
        skin shell? Yes if the end stays outside the box itself (depth < skin - DEPTH_TOL_M: the
        skin is a margin, not a contact), or if it is no deeper than the SHALLOWEST depth seen
        since this contact (this box, these prims) began: a ratchet, so a sequence of tiny moves
        cannot add up to a penetration. The first clause is what lets a turn AWAY from a wall
        proceed when its first degrees swing a box corner a fraction of a mm closer (offline
        seam check: a 1.2 rad/s turn away from a slab 1 cm off the arm, refused for 60 steps by
        a pure ratchet). Returns (allowed, depth at start)."""
        d0 = self._depth(b, start, prims)
        key = (kind, b.name)
        ref = self._touch_ref.get(key)
        if ref is None or ref[0] != prims:
            ref = (frozenset(prims), d0)
        else:
            ref = (ref[0], min(ref[1], d0))
        self._touch_ref[key] = ref
        self._touch_seen.add(b.name)
        d1 = self._depth(b, end, prims)
        return (d1 < self.skin - DEPTH_TOL_M) or (d1 <= ref[1] + DEPTH_TOL_M), d0

    def resolve(self, x: float, y: float, z: float, yaw: float, vx: float, vy: float,
                omega: float, dt: float) -> ContactResult:
        import time

        t0 = time.perf_counter()
        res = ContactResult(vx, vy, omega)
        self.last = res
        if dt <= 0.0:
            return res              # stop(): commands no physics, so it is not a step either
        self.dt = dt
        if vx == 0.0 and vy == 0.0 and omega == 0.0:
            self._account(res, moving=False)
            return res
        self._since_refresh += 1
        if self._since_refresh >= self.refresh_every or not self.env.boxes:
            self._refresh()
            self._since_refresh = 0

        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = vx * c - vy * s, vx * s + vy * c
        res.wx, res.wy = wx, wy
        pose = (x, y, z, yaw)
        # Free space: nothing within CLEAR_MARGIN_M of any box at a recent pose, and this step
        # cannot carry any point of the envelope out of that margin -> the command stands.
        motion = math.hypot(wx, wy) * dt + self._r_max * abs(omega) * dt
        budget = CLEAR_MARGIN_M - self.skin
        if self._clear is not None:
            cx, cy, cz, cyaw = self._clear
            moved = (math.hypot(x - cx, y - cy) + abs(z - cz)
                     + self._r_max * abs(math.remainder(yaw - cyaw, math.tau)))
            if moved + motion <= budget:
                self.stats.cached_steps += 1
                self.stats.query_s += time.perf_counter() - t0
                self._account(res, moving=True)
                return res
            self._clear = None
        if self._probe_cooldown > 0:
            self._probe_cooldown -= 1
        elif motion <= budget:
            if self._probe_clear(pose):
                self._clear = (x, y, z, yaw)
                self.stats.query_s += time.perf_counter() - t0
                self._account(res, moving=True)
                return res
            self._probe_cooldown = CLEAR_PROBE_COOLDOWN
        support = None
        speed = math.hypot(wx, wy)
        touching_all: list = []
        core_all: list = []
        self._touch_seen = set()
        if speed * dt > 1e-7:
            L = speed * dt
            d_hat = (wx / speed, wy / speed)
            blockers, touching, core, support = self._blockers(pose, d_hat, L + BACKOFF_M, support)
            touching_all += touching
            core_all += core
            if core:
                # The bare box starts in overlap: a real contact, whatever the motion does next.
                res.in_contact = True
                if not res.prim:
                    res.box, res.prim = core[0]
            hit = [bl for bl in blockers if bl[0] - BACKOFF_M < L]
            if hit:
                nearest = min(hit, key=lambda bl: bl[0])
                res.blocked = True
                res.prim, res.box, res.normal, res.distance = (nearest[1], nearest[2],
                                                               nearest[4], nearest[0])
                # Advance until the shell touches (less a backoff, so the next step's shell
                # starts clear of it), then slide the rest: per blocker, remove the component
                # pointing INTO it and keep the remainder.
                f = max(0.0, min(1.0, (nearest[0] - BACKOFF_M) / L))
                rx, ry = wx * (1.0 - f), wy * (1.0 - f)
                for bl in hit:
                    nx, ny = bl[3]
                    into = -(rx * nx + ry * ny)
                    if into > 0.0:
                        rx += into * nx
                        ry += into * ny
                nwx, nwy = wx * f + rx, wy * f + ry
                # The slide is a new direction: sweep it once, and stop short of whatever it meets.
                sp2 = math.hypot(nwx, nwy)
                if sp2 * dt > 1e-7 and (rx != 0.0 or ry != 0.0):
                    L2 = sp2 * dt
                    d2 = (nwx / sp2, nwy / sp2)
                    bl2, _t2, _c2, support = self._blockers(pose, d2, L2 + BACKOFF_M, support)
                    hit2 = [bl for bl in bl2 if bl[0] - BACKOFF_M < L2]
                    if hit2:
                        f2 = max(0.0, min(1.0, (min(bl[0] for bl in hit2) - BACKOFF_M) / L2))
                        nwx, nwy = nwx * f2, nwy * f2
                wx, wy = nwx, nwy
        # Something is inside the skin shell with no normal at any tier (a turn brought it
        # there, the robot was placed there, a person stepped in, the lift moved), or the bare
        # box itself starts on it: allow the move only if it does not take the envelope deeper
        # than this contact has been (_depth_ok's ratchet). The second group has normals and
        # was slid along already; checking it too is the backstop for a slide whose normal came
        # from a shrunk tier's corner rather than the real box's nearest face (offline jiggle:
        # 0.4-1 mm per step of creep past a post's edge without it).
        if touching_all or core_all:
            if touching_all:
                res.touching = True
                if not res.prim:
                    res.box, res.prim = touching_all[0][0].name, touching_all[0][1]
            end = (x + wx * dt, y + wy * dt, z, yaw)
            boxes = {b.name: b for b in self.env.boxes}
            by_box: dict = {}
            for b, prim in touching_all:
                by_box.setdefault(b.name, (b, set()))[1].add(prim)
            for name, prim in core_all:
                by_box.setdefault(name, (boxes[name], set()))[1].add(prim)
            for b, prims in by_box.values():
                ok, d0 = self._depth_ok(b, pose, end, prims, "t")
                res.in_contact = res.in_contact or d0 > self.skin
                if (wx != 0.0 or wy != 0.0) and not ok:
                    wx = wy = 0.0
                    res.blocked = True
        # The turn: overlap at the end yaw, shell grown by rot_skin. Refuse anything NEW in it,
        # and for anything that was already in it at the start (placed there, a person, a
        # 2 mm PhysX overlap) refuse a turn that brings it DEEPER -- the depth rule the
        # translation uses. Excluding those prims outright let a turn swing an arm straight
        # through a wall that started 1 cm off it (offline seam check, 2026-10-05).
        if omega != 0.0:
            end = (x + wx * dt, y + wy * dt, z, yaw + omega * dt)
            for b in self.env.boxes:
                lift = ROT_LOW_LIFT_M if b.low else 0.0
                # The low box is already lifted to MAX_STEP_M + ROT_LOW_LIFT_M (= the floor
                # rule's 5.5 cm, see SUPPORT_LIFT_M), so nothing it meets is floor: no support
                # exemption here. Subtracting support prims let a turn swing the deck into a
                # kerb whose low end a corner stood on.
                new = self._overlap(b, end, grow=self.rot_skin, lift=lift)
                if not new:
                    continue
                pre = self._overlap(b, pose, grow=self.rot_skin, lift=lift)
                kept = new & pre
                new -= pre
                if not new and kept and not self._depth_ok(b, pose, end, kept, "r")[0]:
                    new = kept
                if new:
                    res.rot_blocked = True
                    if not res.prim:
                        res.prim, res.box = sorted(new)[0], b.name
                    res.omega = 0.0
                    break
        # A contact the robot has left (or that this step did not look at) starts its ratchet
        # afresh next time.
        for key in [k for k in self._touch_ref if k[1] not in self._touch_seen]:
            del self._touch_ref[key]
        if res.blocked:
            res.wx, res.wy = wx, wy
            res.vx = wx * c + wy * s
            res.vy = -wx * s + wy * c
        # else: the command is returned bit-identical (res was built from it), so a robot in
        # free space integrates exactly what it was asked for -- the world->body round trip
        # is not the identity in floating point.
        self.stats.query_s += time.perf_counter() - t0
        self._account(res, moving=True)
        return res

    # --- metric ---------------------------------------------------------------------------
    def reset_stats(self) -> None:
        """Per episode. Also forgets the clearance cache: an episode starts somewhere new."""
        self.stats = ContactStats()
        self._clear = None
        self._probe_cooldown = 0
        self._touch_ref = {}

    def _account(self, res: ContactResult, moving: bool) -> None:
        st = self.stats
        st.steps += 1
        # A contact, for the metric, is a step the command pushed into something (and was
        # clipped) or the envelope itself overlapping something. Merely having an obstacle inside
        # the 2 cm skin is not one: a robot sliding past a wall at the skin is not colliding.
        contact = res.blocked or res.rot_blocked or res.in_contact
        if res.blocked:
            st.blocked_steps += 1
        if res.rot_blocked:
            st.rot_blocked_steps += 1
        if res.touching:
            st.touching_steps += 1
        if res.in_contact:
            st.in_contact_steps += 1
        t = st.steps * self.dt
        if contact:
            st.contact_steps += 1
            st.run += 1
            st.max_run = max(st.max_run, st.run)
            st.by_prim[res.prim] = st.by_prim.get(res.prim, 0) + 1
            st.by_box[res.box] = st.by_box.get(res.box, 0) + 1
            if st.first_contact is None:
                st.first_contact = {"t": round(t, 3), "prim": res.prim, "box": res.box,
                                    "kind": ("in_contact" if res.in_contact else
                                             "rotation" if res.rot_blocked and not res.blocked
                                             else "translation"),
                                    "distance_m": round(res.distance, 4)
                                    if math.isfinite(res.distance) else None}
            if st._span is None:
                st._span = [round(t, 3), round(t, 3), res.prim, res.box]
            else:
                st._span[1] = round(t, 3)
        else:
            # A held robot (zero command) that is not touching anything ends a run; a held
            # robot pressed against something cannot exist here, it never pushes.
            st.run = 0
            if st._span is not None:
                st.spans.append(st._span)
                st._span = None

    def sustained_steps(self) -> int:
        """DYNANAV_SUSTAINED_S in this caller's steps (200 at 60 Hz)."""
        return max(1, math.ceil(DYNANAV_SUSTAINED_S / self.dt - 1e-6))

    def summary(self) -> dict:
        st = self.stats
        spans = st.spans + ([st._span] if st._span is not None else [])
        max_run_s = st.max_run * self.dt
        return {
            "steps": st.steps,
            "contact_steps": st.contact_steps,
            "blocked_steps": st.blocked_steps,
            "rot_blocked_steps": st.rot_blocked_steps,
            "touching_steps": st.touching_steps,
            "in_contact_steps": st.in_contact_steps,
            "max_run_s": round(max_run_s, 3),
            "first_contact": st.first_contact,
            # DynaNav's physical_collision (benchmark.py:1276-1287): sustained >= 100 frames @30.
            # Compared in steps, not seconds: 200 * (1/60) and 100/30 are equal only by luck.
            "physical_collision": st.max_run >= self.sustained_steps(),
            "spans": spans[:50],
            "n_spans": len(spans),
            "by_prim": dict(sorted(st.by_prim.items(), key=lambda kv: -kv[1])[:8]),
            "by_box": st.by_box,
            "query_ms_per_step": round(1e3 * st.query_s / max(1, st.steps), 3),
            "queries": st.queries,
            "cached_steps": st.cached_steps,
            "clear_probes": st.clear_probes,
            "skip_support": dict(sorted(st.skip_support.items(), key=lambda kv: -kv[1])[:6]),
            "skip_walkable": dict(sorted(st.skip_walkable.items(), key=lambda kv: -kv[1])[:6]),
            "envelope": self.env.describe(),
            "max_step_m": self.env.max_step_m,
            "skin_m": self.skin,
        }
