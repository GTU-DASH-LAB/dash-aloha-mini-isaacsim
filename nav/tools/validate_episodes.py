#!/usr/bin/env python3
"""Can AlohaMini actually be placed in each DynaNav episode, with correct physics?

README
======

One process validates the episodes of ONE environment (one nav stage), all in one Kit
session, and writes one JSON file. It never talks to a policy server: it boots Kit the
way `run_navigation.NavigationRunner.setup()` does -- literally, it calls that method
with a stub policy -- so the stage, the kinematic base, the nav camera and the collision
guard are the very objects a real run drives. `base_drive.py` is imported, never copied,
so a fix there applies here on the next run.

Usage (always through the GPU slot wrapper, always Isaac's python.sh):

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \\
        nav/tools/validate_episodes.py --env hospital > /tmp/val_hospital.log 2>&1
    echo $?   # 0 = completed, no episode FAILED. Anything else is 1 through python.sh,
              # which maps every non-zero status to 1 -- so read the JSON's `status`:
              # "completed" (then `summary` says how many FAILed) or "crashed" (`fatal`).
              # The log's last line says the same: "[validate] STATUS completed ...".

    # a subset, from the ladder instead of the full table, to a chosen file:
    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \\
        nav/tools/validate_episodes.py --episodes nav/config/episodes.yaml \\
        --env outdoor_small --only outdoor_umbrellas --out /tmp/umbrellas.json

    --episodes PATH   episode table (default nav/config/episodes_full.yaml, all 85)
    --env NAME        hospital | office | warehouse | outdoor_small  -- one stage per
                      process; inferred from --only when every name shares one
    --only A,B        restrict to these episode names (comma list, repeatable)
    --out PATH        JSON report (default nav/results/validate/<env>_<stamp>.json)
    --frames-dir DIR  where each episode's camera_nav frame is saved as <name>.jpg
                      (default: next to --out, <stem>_frames/). LOOK at them.
                      Inside the repo the directory gets its own `*` .gitignore:
                      nav/results/.gitignore re-includes *.jpg for campaign evidence.
    --summarize J...  no Kit, plain python3 is fine: verdict tables of these reports and
                      every numeric metric's min..max per environment (how CALIBRATION
                      below was produced).

Do not pipe the run into head/tail (CLAUDE.md): redirect to a file and read $?.
Stage load is ~2-4 min (CDN-streamed environments); each episode then costs ~810
physics steps plus a few hundred scene queries, ~25-40 s wall.

WHAT EACH EPISODE GOES THROUGH, in this order, in one process:

  0. HEADING. The start yaw against the start->goal bearing (no Kit): the same formula as
     import_benchmark.off_bearing_deg, recomputed so any table is graded.
  1. RESTORE + SPAWN, exactly as run_episode spawns. The arm/lift joints go back to the
     positions and position targets setup() left them at, and root and joint velocities
     are zeroed (bench.sh runs one process per episode, so a real run never inherits the
     previous episode's motion; how far the joints had been knocked is recorded as
     `restore`). Then `start, snap = runner._spawn_pose(ep)` -- base_drive's
     ground_spawn_pose, which replaces DynaNav's z (the Nova Carter's) with where THIS
     robot rests on the floor under its tyres -- `base.reset_to(start, yaw)`, 30 physics
     steps, `sync_from_sim()`, `reset_stats()`, as run_episode does, and 30 more (60).
     Graded: the snap found a floor at all; the spawn gap (snapped z vs the floor under
     the centre + BASE_LINK_REST_Z_M = 0.0078 m); whether that floor IS the floor -- its
     height against the median of a 24-sample ring at 0.6/1.2 m around the start, and
     whether its prim appears on that ring (furniture is an island the floor around it
     does not share); then rest error, settle z span, xy drift, root speeds, tilt. Then 90
     more steps at rest (150 total -- the renderer's exposure settle) and the late z drift.
     DynaNav's own drop (episode z - snapped z) is reported as `dynanav_drop_m`, never
     graded: the runner removes it.
  2. CLEARANCE. A PhysX overlap box the size of the robot's measured envelope (read off
     the robot's own USD geometry at startup, base_link frame), from 0.10 m above the
     floor to the robot's top, at the settled pose, PLUS a containment test at a 3 x 3
     grid x 2 heights inside that box (an overlap box wholly inside a closed triangle mesh
     overlaps nothing). Plus a ring of horizontal rays from the robot's centre (72
     bearings x 6 heights, 0.10 m .. top): nearest obstacle distance, its prim, and the
     CLEARANCE = that distance minus the envelope's own extent along the same bearing.
  3. GOAL. Floor under the goal, goal z vs that floor, floor height start -> goal,
     straight-line distance; whether the goal point (0.30 m over its floor) is TRAPPED --
     inside a closed solid, or enclosed in a cavity or slot (see "RAYS AND CONTAINMENT");
     and whether the robot's footprint box fits at the goal or anywhere on three rings
     inside the success radius (12 bearings each), facing the start->goal bearing, at a
     point that can SEE the goal (two-sided line of sight at 0.30, 0.60 or 1.00 m) -- so a
     sample in the room behind a wall does not count.
  4. DRIVE PROBE. 4 s straight at 0.45 m/s, 4 s turning in place at 0.6 rad/s, 3 s
     straight, after `sync_from_sim()` + `reset_stats()` + `contact.reset_episode()`, with
     the collision guard applied and the contact layer attached exactly as run_navigation
     applies them (lidar step if NAV_LIDAR=1, `guard.check`,
     `base.apply(guard.vx, guard.vy, omega)`, one `kit.update()`), and the runner's own
     `_physics_fault()` after every step: the first fault ends the probe, as it ends a
     real episode. Measured: path, z span, max per-step z change, the ground error
     |z - (floor under the robot + 0.0078)| sampled every 3 steps, max |root linear /
     angular velocity|, how far the sim's x/y sits from the base's integrator (`_xy`,
     required -- no silent default), the base's own `faults` and `resyncs`, sim yaw vs
     integrated yaw after the turn, tilt, guard interventions and the share of translate
     steps it held, the contact layer's record (steps it clipped or refused, first prim),
     and summarize_runs.score's blow-up rule.
  5. CAMERA. One camera_nav frame (saved): mean/std luma (Rec.601), fraction of pixels
     near-black (< 10) and near-white (> 245), and whether the camera centre is inside
     geometry: a two-sided ray from the base_link origin to the camera, a 2 cm overlap
     sphere at it, and the trapped test, all ignoring the robot.
  6. LIDAR. One full revolution of a fresh SweepingLidar2D at the runner's mount
     (NAV_LIDAR_HEIGHT / NAV_LIDAR_FORWARD, default 0.30 m up, 0.20 m forward): the scan
     origin's height over the floor under it (vs 0.0078 + mount) and the tilt at the scan
     instant, external returns, nearest return and its clearance to the envelope.
     Self-hits are reported, not graded (see lidar below).
  7. PHYSX LOG. Every Kit log message at ERROR+ (any source) and WARN+ (PhysX) is tapped
     from inside SimulationApp's constructor and attributed to the episode it happened
     in (up to one frame late); messages logged while the stage loaded go to the
     report's `boot_log` and count against every episode. The tap is Kit's deferred
     GLOBAL_EVENT_ERROR_LOG, dispatched on the main thread -- NOT a carb logger callback,
     which deadlocked Kit the first time PhysX logged from a worker thread (see LogTap). Kit's stdout is 99% render noise outdoors (144k foliage
     instancing warnings in one run), and the one line that matters -- "PhysX error: ...
     increase PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity ... otherwise,
     the simulation will miss interactions" -- is invisible in it without this.

RAYS AND CONTAINMENT. Every scene query is an ITERATED `raycast_closest`, re-cast 1 mm
past each hit, robot prims skipped. Never `raycast_all`: on the outdoor stage it segfaults
Kit (exit 139 inside libomni.physx.plugin, 3 of 3 runs) on a ray raycast_closest answers,
and it reports one hit per shape, so a floor sharing a mesh with anything above it was
invisible. A hit at distance 0 is an initial overlap (origin inside a convex or primitive)
and is never taken for a surface. "Trapped" is two tests, because every collider on these
stages is a triangle mesh and an overlap query cannot see a volume wholly inside one:
  * inside a closed solid: per direction (+-x, +-y, +z), the first surface a two-sided
    cast meets is a BACK face -- the culled default cast skips it. >= 3 of 5 votes.
  * enclosed: <= 3 of 16 two-sided horizontal rays get 0.6 m out. Needed because most
    props are NOT closed solids, measured on the hospital stage: the bedside table is a
    hollow cabinet (its centre is in a compartment whose walls face into it, 0 rays
    escape), SideWall13 is two thin panel boxes with an air gap (2 rays escape, along the
    gap), the drinks machine an open shell (0-2). All three are FAIL cases that a
    back-face vote alone calls free. Every one of the 85 DynaNav goals and every camera
    centre at the 85 starts escapes on 16/16 (see CALIBRATION), so the bound has margin
    on both sides.

A NATIVE CRASH (segfault) skips every Python handler, so the JSON stays at status
"running". The JSON's `current_episode` / `current_phase` are rewritten before every
phase, so a "running" report whose `pid` is gone is a crash IN that episode and phase;
faulthandler also prints the Python stack of the crashing call to the log.

OUTPUT. Top level: `status` (running / completed / crashed -- and "running" with a dead
`pid` is a native crash, see above), `current_episode` / `current_phase`, `summary`,
`boot_s`, `boot_log`, `robot_envelope`, `robot_rest_joints` (the restore snapshot),
`thresholds`, `ray_casts`, GPU MiB, git sha. The JSON holds, per episode, `restore` (how
far the joints had drifted before it) and every number above under
`checks.<name>.metrics`,
and for each check a verdict OK / WARN / FAIL with `rules`: one line per rule that was
evaluated, carrying the metric, its value, the thresholds and the verdict it produced.
The episode verdict is the worst check; an episode that raised is ERROR with the
traceback. `summary` counts verdicts; `thresholds` is the THRESHOLDS table below as run;
`robot_envelope` is what step 2 measured. stdout ends with one line per episode.

THRESHOLDS -- every rule, and why it sits where it does
-------------------------------------------------------
[cal] is the min..max the 13 indoor episodes of nav/config/episodes.yaml measured: the
stages and starts every recorded ladder has driven end to end, 8 of which passed on the
8/13 arc-menu ladder (CLAUDE.md). First calibrated at 08:00 on 2026-10-05; re-measured at
09:21 after the spawn moved to the runner's snap and base_drive started writing velocities
(the drive numbers tightened ~2x; no bound needed to move). The full table is CALIBRATION
below. "ok"/"warn" are inclusive upper bounds unless the rule says "higher is better".
Where the healthy value is ~0 the bound is set by physics and says so; where it is not,
the OK bound sits at a stated multiple of the worst healthy value.

  heading
    off_bearing_abs_deg  |goal bearing - start yaw|, <= 30 OK, <= 90 WARN, else FAIL.
                         CLAUDE.md: keep episodes inside ~30 deg or the run is not testing
                         navigation (benchmark_example's ~170 deg starts looked exactly like
                         a broken policy). > 90 = the goal is behind the robot. [cal] 9 of
                         13 inside 30; hospital_forward_staircase 66.1 and hospital_exit_room
                         82.5 WARN. In the 85-episode table: 54 OK, 29 WARN, 2 FAIL
                         (warehouse_e66/e67, 101.8).
  spawn
    snap_floor_found     base_drive.ground_spawn_pose found no floor under any tyre or the
                         centre -> FAIL (the runner then spawns at DynaNav's raw z).
    floor_found          the validator's own floor under the snapped centre, within
                         [z - 10 m, z + 0.05 m] -> FAIL if none.
    start_embed_m        snapped z - (that floor + 0.0078); higher is better: >= -0.01 OK,
                         >= -0.03 WARN, else FAIL. Below rest height the wheel spheres
                         start INSIDE the floor and PhysX's separation impulse launches the
                         robot (the Office spawn: 7.7 m, tipped 181 deg). [cal] +0.0030 on
                         all 13 (base_drive.SPAWN_CLEARANCE_M).
    start_drop_m         the same number, <= 0.05 OK, <= 0.30 WARN, else FAIL. Since the
                         snap this is the slope's own few cm under the lower tyres (the snap
                         rests the HIGHEST tyre), not DynaNav's drop -- that is reported as
                         `dynanav_drop_m` and never graded: [cal] -0.0009 .. -0.0008 indoor;
                         outdoor 0.118 .. 2.183 m, all removed by the snap.
    floor_offset_abs_m   |spawn floor - median floor of a 24-sample ring at 0.6/1.2 m|,
                         <= 0.05 OK, <= 0.10 WARN, else FAIL. [cal] 0 .. 0.0001; outdoor
                         0.0012 .. 0.0147 on slopes up to 6.8 deg (a uniform slope is
                         symmetric about the centre, so the median stays at its height).
                         Mutations: on the 0.817 m bedside table 0.817 (FAIL); snapped onto
                         the table's inner shelf 0.096 and onto a wall trim 0.158.
    spawn_island         offset > 0.05 AND the spawn floor's prim is on none of the ring
                         samples -> FAIL: furniture or an object, not the floor. Catches the
                         inner shelf (0.096, under the offset FAIL bound) that the offset
                         alone would only WARN.
    rest_err_m           |z_rest - (floor under the settled robot + 0.0078)|, <= 0.005 OK,
                         <= 0.02 WARN. [cal] 0 .. 4e-5 m. Physics bound: 5 mm is far below
                         anything a camera notices, 2 cm is 40% of the wheel radius --
                         resting on something, or sunk into it.
    settle_tail_z_span_m z max-min over settle steps 31..60, <= 0.001 OK, <= 0.005 WARN.
                         [cal] 0 (below the 1e-6 rounding). Not still after 0.5 s = not
                         resting.
    late_z_drift_m       |z(step 150) - z(step 60)|, <= 0.001 OK, <= 0.005 WARN. [cal] 0.
                         The robot is NOT commanded during this rest -- longer than
                         run_episode's 30-step settle, but exactly what a real run does
                         whenever it stops (base.apply(0, 0, 0) writes zero velocity and
                         does not pin the pose), so a robot that slides here slides there.
    settle_xy_drift_m    xy displacement over the settle, <= 0.02 OK, <= 0.10 WARN.
                         [cal] 0 (the first calibration's one 0.0081 office kick is gone
                         with the velocity-writing reset_to). OK = 2.5x that old worst.
    end_lin_speed_mps    root |v| at settle end, <= 0.02 OK, <= 0.10 WARN.
                         [cal] 0.0004 .. 0.0044, so OK is ~5x the worst.
    end_ang_speed_radps  root |w| at settle end, <= 0.02 OK, <= 0.30 WARN.
                         [cal] 0.0003 .. 0.0031, OK ~6x the worst.
    tilt_deg             max(|roll|, |pitch|) at settle end, <= 3 OK, <= 10 WARN.
                         [cal] 0 .. 0.023 deg. Physics bound: 3 deg admits a 5% cross-
                         slope; 10 deg is twice an ADA ramp (4.8 deg).
  clearance
    footprint_overlap    any non-robot collider inside the envelope box (overlap query) or
                         containing a point of the 3 x 3 x 2 grid (back-face vote) -> FAIL,
                         unless every overlapping collider is the FLOOR's own prim (slope,
                         or a building authored as one mesh) -> WARN. [cal] none on all 13.
    min_clearance_m      higher is better: >= 0.10 OK, >= 0.0 WARN, else FAIL.
                         [cal] 0.28 .. 2.34 m (hospital_past_wheelchairs is the tight one);
                         both warehouse starts have nothing within the 3 m ring. 10 cm is
                         the guard's own "touching" margin (rays start 0.35 m out against a
                         0.375 m swing radius); < 0 is an obstacle inside the footprint.
  goal
    goal_floor_found     no walkable floor within [goal_z - 10, goal_z + 0.5] -> FAIL.
    goal_floor_gap_m     |goal_z - floor under goal|, <= 0.10 OK, <= 0.50 WARN.
                         [cal] 0.01. Beyond 10 cm DynaNav's goal z is not this floor --
                         usually the search found another level or a roof.
    goal_point_inside    the goal point, 0.30 m over its floor, is TRAPPED (inside a closed
                         solid, or <= 3/16 rays escape 0.6 m: see RAYS AND CONTAINMENT)
                         -> FAIL. DynaNav's goals sit in free space next to a landmark: all
                         85, on all four stages, escape on 16/16 -- the gurney goals of the
                         hospital_down_hallway pair included -- so a trapped goal means the
                         coordinates do not describe this stage. Mutations: goal inside the
                         drinks machine 0/16, the bedside cabinet 0/16, SideWall13's panel
                         gap 2/16 -- all FAIL. Not a claim that the radius is unreachable
                         (from a machine's face it often is reachable); a claim that the
                         episode is not what it says.
    goal_standable       footprint fits AT the goal -> OK; fits, IN SIGHT OF THE GOAL
                         (two-sided line of sight at 0.30 / 0.60 / 1.00 m), on a ring inside
                         the success radius -> WARN; nowhere in sight -> FAIL. Without the
                         sight test the room behind a wall counted (24/37 "standable" for a
                         goal in a wall). [cal] 11 OK, 2 WARN: the hospital_down_hallway
                         pair's goal is ON the red bed (SM_Gurney), and the footprint still
                         fits, in sight, at 18/37 points, nearest 0.49 m. A goal on a
                         landmark's face is normal.
    floor_dz_m           |floor(goal) - floor(start)|, <= 0.05 OK, else WARN. Not a fault
                         in the episode: the route needs slopes or ramps, which the
                         kinematic base and the horizontal guard fan were never verified on
                         (CLAUDE.md, outdoor). [cal] 0.000 on every indoor stage; outdoor
                         0.11 .. 0.29 m on 4 of 6.
  drive
    physics_fault        run_navigation._physics_fault (base jump, |v| > 3 m/s, tilt > 30
                         deg) after any step -> FAIL, and the probe ends there as a real
                         episode does. [cal] none. Mutations: robots spawned in a wall or in
                         the drinks machine fault within 0.08 - 0.45 s (|v| 3.3 - 8.1 m/s).
    blowup               path > 2 x max_speed_mps x elapsed -> FAIL. summarize_runs.score's
                         rule verbatim: 33 m for this 11 s probe against 3.15 m commanded.
                         [cal] path 2.08 .. 3.23 m.
    path_ratio           path / commanded path (3.15 m), <= 1.10 OK, <= 1.50 WARN.
                         [cal] 0.66 .. 1.026. Below 1 is the guard's slow band holding the
                         robot back near a wall (never penalised here); above 1 the robot
                         moved further than told, which a teleported base cannot do unless
                         something else is moving it.
    mobility             path_ratio < 0.30, or the guard held > 50% of the translate steps,
                         or the contact layer clipped/refused > 50% of the probe's steps
                         -> WARN: a start the robot cannot drive away from (pinned against a
                         wall) still has clearance >= 0. [cal] 0.66 and 0% at worst.
    ground_err_m         max |z - (floor under base + 0.0078)| sampled every 3 steps,
                         <= 0.010 OK, <= 0.05 WARN. [cal] 0.0013 .. 0.0020 m (first
                         calibration 0.0005 .. 0.0045; OK kept at ~2.2x that). A robot that
                         hovers, sinks or falls shows here first, and on a slope this stays
                         small where z_span legitimately grows.
    z_step_max_m         max |dz| between steps, <= 0.008 OK, <= 0.03 WARN.
                         [cal] 0.0013 .. 0.0018 m (first calibration up to 0.0044: the
                         rocking of a teleported body in wheel contact, before base_drive
                         wrote velocities). 3 cm/step is a 1.8 m/s vertical rate.
    max_lin_speed_mps    max root |v| reported by PhysX, <= 1.0 OK, <= 3.0 WARN.
                         [cal] 0.46 .. 0.57 (commanded 0.45). WARN at twice the 1.5 m/s cap.
    max_ang_speed_radps  max root |w|, <= 6.0 OK, <= 15 WARN. [cal] 0.91 .. 1.68 against a
                         commanded 0.6 (first calibration 1.82 .. 3.45, which set the bound).
    resyncs              base_drive's own count: the sim moved the robot > 0.25 m on a step
                         the base did NOT drive (a jump on a driven step is a FAULT, above):
                         0 OK, 1-2 WARN, more FAIL. [cal] 0.
    track_err_max_m      max distance sim x/y vs the integrator (base._xy, required),
                         <= 0.05 OK, <= 0.25 WARN. [cal] 0.0066 .. 0.0084 m. WARN stops at
                         the resync tolerance itself.
    yaw_err_deg          |sim yaw - integrated yaw| after the probe, <= 2 OK, <= 10 WARN.
                         [cal] 0.005 .. 0.18 deg.
    tilt_max_deg         max(|roll|, |pitch|) during the probe, <= 3 OK, <= 10 WARN.
                         [cal] 0.49 .. 0.67 deg.
  camera
    camera_inside        two-sided ray base_link origin -> camera centre, a 2 cm sphere at
                         the centre, or the centre TRAPPED -> FAIL. [cal] never; the camera
                         centre at all 85 starts escapes 16/16.
    luma_mean            higher is better: >= 40 OK, >= 10 WARN, else FAIL.
                         [cal] 78.6 .. 174.7 (darkest: an office corridor, then the
                         headlight-lit hospital corridor at 81.4). OK = half the darkest.
    luma_std             higher is better: >= 15 OK, >= 5 WARN. [cal] 38.9 .. 85.3. A
                         uniform frame is the camera looking at the inside of a wall.
    frac_black           <= 0.40 OK, <= 0.80 WARN. [cal] 0.0002 .. 0.191; OK ~2x.
    frac_white           <= 0.15 OK, <= 0.50 WARN. [cal] 0 .. 0.033 (the window-lit
                         vending-machine room); OK ~5x. Blown out. Facing the hospital's
                         windows from other starts reads 0.16 .. 0.37 (WARN) -- the frame is
                         real, the exposure is not settled for it.
  lidar
    height_err_m         |scan origin z - floor under it - (0.0078 + mount height)|,
                         <= 0.02 OK, <= 0.05 WARN. [cal] 0 .. 8e-5. The scan is a plane at
                         a fixed height above base_link: off by more than 2 cm it is no
                         longer the plane the guard and the menu were tuned on.
    tilt_deg             max(|roll|, |pitch|) at the scan instant, <= 3 OK, <= 10 WARN.
                         [cal] 0 .. 0.023. A tilted plane cuts the floor on one side.
    self_hits            REPORTED, NOT GRADED: [cal] 53 of 500 on all 13, but 53 is
                         lidar_2d's analytic column model (the lift column sliced at the
                         mount height), fixed by the mount offsets and blind to pose -- a
                         robot tilted 99.6 deg read 45 and passed the old <= 60 rule.
    min_clearance_m      nearest external return minus the envelope extent on its bearing;
                         higher is better: >= 0.10 OK, >= 0.0 WARN. [cal] 0.28 .. 2.29 m,
                         and it agrees with clearance.min_clearance_m to 1 cm wherever the
                         nearest thing crosses 0.30 m.
    external_returns     reported, never graded: [cal] 276 .. 447 in hospital/office and 0
                         at both warehouse starts -- the CORRECT answer where every wall is
                         beyond the C1's 12 m (CLAUDE.md measured 15.7 - 16.7 m).
  physx
    physx_errors         any PhysX ERROR during the episode -> FAIL. [cal] 0 on all 13 (and
                         0 Kit errors of any kind in the three indoor logs).
    boot_physx_errors    any PhysX ERROR while the stage loaded -> FAIL on every episode of
                         that stage: it is the scene they all run in. [cal] 0.
    physx_warnings       any PhysX WARN during the episode -> WARN. [cal] 0 (the one PhysX
                         warning indoors, "more than 4 velocity iterations ... TGS", is
                         logged once at load and is reported in boot_log, not graded).

CALIBRATION (2026-10-05 09:21, nav/config/episodes.yaml, fan guard, runner snap spawn)
-------------------------------------------------------------------------------------
13 indoor episodes, three processes, `--summarize` over their reports: 0 FAIL. Non-OK
verdicts: the two goal WARNs under goal_standable and the two heading WARNs above;
spawn, clearance, camera, lidar, drive and physx OK on all 13. Kit held 4.9 - 5.9 GiB of
GPU0 per indoor stage and booted in 15 - 20 s with the CDN cache warm; one episode costs
28 - 39 s wall (outdoor: 12.3 GiB, 38 s boot, 94 - 101 s per episode).

  metric (min .. max)        hospital (7)        office (4)          warehouse (2)
  spawn.start_gap_m          0.0030 .. 0.0031    0.0030              0.0030
  spawn.floor_offset_m       0 .. 0.0001         0                   0
  spawn.rest_err_m           0 .. 4e-5           0                   0
  spawn.settle_xy_drift_m    0                   0                   0
  spawn.end_lin_speed_mps    0.0004 .. 0.0044    0.0006 .. 0.0017    0.0005 .. 0.0008
  spawn.end_ang_speed_radps  0.0003 .. 0.0026    0.0004 .. 0.0031    0.0010 .. 0.0015
  spawn.tilt_deg             0 .. 0.023          0                   0
  clearance.min_clearance_m  0.28 .. 1.48        0.56 .. 2.34        none within 3 m
  goal.standable (at goal)   5 of 7              4 of 4              2 of 2
  drive.path_ratio           0.66 .. 1.017       0.74 .. 1.019       1.018 .. 1.026
  drive.ground_err_m         0.0019 .. 0.0020    0.0019 .. 0.0020    0.0013 .. 0.0015
  drive.z_step_max_m         0.0014 .. 0.0016    0.0015 .. 0.0018    0.0013 .. 0.0015
  drive.max_lin_speed_mps    0.55 .. 0.57        0.56 .. 0.57        0.46
  drive.max_ang_speed_radps  1.48 .. 1.68        1.45 .. 1.54        0.91
  drive.track_err_max_m      0.0083 .. 0.0084    0.0083 .. 0.0084    0.0066 .. 0.0069
  drive.yaw_err_deg          0.005 .. 0.073      0.018 .. 0.05       0.024 .. 0.18
  drive.tilt_max_deg         0.66 .. 0.67        0.65 .. 0.66        0.49 .. 0.54
  drive.resyncs / faults     0 / 0               0 / 0               0 / 0
  camera.luma_mean           81.4 .. 174.7       78.6 .. 130.3       103.2 .. 115.6
  camera.luma_std            38.9 .. 60.6        65.9 .. 85.3        51.9
  camera.frac_black          0.0002 .. 0.024     0.056 .. 0.191      0.014 .. 0.030
  camera.frac_white          0.0004 .. 0.033     0 .. 0.0002         0.0011 .. 0.0023
  lidar.height_err_m         0 .. 8e-5           0                   0
  lidar.external_returns     378 .. 447          276 .. 400          0

OUTDOOR READING (the 6 ladder episodes, same code, base_drive as of 09:27): 1 OK, 3 WARN,
2 FAIL. No PhysX error at boot or in any episode now that the runner raises
gpuFoundLostAggregatePairsCapacity (physics_setup.py), and the snap removed DynaNav's
0.12 - 2.18 m drops. What is left is the scene, not the thresholds:
  * outdoor_pillars FAIL: the start is on a 6.8 deg slope and the robot does not stay put
    uncommanded -- at step 60 still moving 0.16 m/s and tilted 11 deg, at step 150 tipped
    over (131 deg at the lidar scan); the drive probe then starts from that state
    (ground_err 0.18 m, yaw error 127 deg).
  * outdoor_upsway_far FAIL: drive ground_err 0.147 m with a z span of only 0.061 m -- the
    floor under the robot moved ~15 cm that the robot's z did not follow.
  * umbrellas / library / ramp_fountain WARN on goal floor_dz 0.11 - 0.29 m (and umbrellas'
    goal z 0.18 m off its floor): routes with slopes the base was never verified on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
# nav/tools/profile.py shadows the stdlib `profile`, which warp imports while Kit starts;
# Kit then fails to load omni.warp.core. Nothing here imports from nav/tools.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
for _p in (REPO / "scripts", REPO / "nav" / "ui", REPO / "nav" / "policy_server",
           REPO / "nav" / "sim"):
    sys.path.insert(0, str(_p))

import yaml  # noqa: E402

from episode import env_name, episodes_by_env  # noqa: E402

DEFAULT_EPISODES = REPO / "nav" / "config" / "episodes_full.yaml"
ROBOT_PREFIX = "/World/Aloha"
BASE_LINK = "/World/Aloha/Geometry/base_link"

SETTLE_STEPS = 60          # spawn metrics are taken over these
RUNNER_SETTLE_STEPS = 30   # run_episode's own settle: sync_from_sim + reset_stats after it
REST_TOTAL_STEPS = 150     # camera frame is taken after this many steps at rest
TAIL_STEPS = 30            # "still moving?" window at the end of the settle
FLOOR_ABOVE_M = 0.5        # a floor may sit at most this far ABOVE a DynaNav reference z
FLOOR_ABOVE_ROBOT_M = 0.05  # ...and this far above the ROBOT's own z (rest/ground errors)
FLOOR_BELOW_M = 10.0       # ...and is searched this far below it
FLOOR_MIN_NZ = 0.7         # walkable: surface normal within ~45 deg of vertical
CLEAR_Z_LO_M = 0.10        # envelope box / ring start this far above the floor
RING_BEARINGS = 72
RING_RANGE_M = 3.0
LOCAL_FLOOR_RADII_M = (0.6, 1.2)   # spawn-floor ring: just outside the footprint, and 2x
LOCAL_FLOOR_BEARINGS = 12
LOCAL_FLOOR_LEVEL_M = 0.05         # a ring sample "at the spawn level"
GOAL_RING_FRACTIONS = (0.33, 0.66, 0.95)   # of the success radius
GOAL_RING_BEARINGS = 12
GOAL_PROBE_H_M = 0.30      # goal point probed this far above its floor
GOAL_LOS_HEIGHTS_M = (0.30, 0.60, 1.00)    # a standable sample must SEE the goal at one
DRIVE_SEGMENTS = ((4.0, 0.45, 0.0), (4.0, 0.0, 0.6), (3.0, 0.45, 0.0))  # (s, vx, omega)
GROUND_EVERY = 3           # drive steps between floor samples
# Ray casting. Every query is an ITERATED raycast_closest, never raycast_all: on the
# outdoor stage raycast_all segfaults Kit inside libomni.physx.plugin on a ray that
# raycast_closest answers fine (dumpster enclosure, 3/3 reproductions), and it reports
# only one hit per shape, so a floor sharing a mesh with anything above it was invisible.
RAY_SKIP_M = 1e-3          # re-cast this far past each hit
RAY_ZERO_M = 1e-6          # a hit at distance <= this is an INITIAL OVERLAP (origin inside)
RAY_ZERO_STEP_MAX_M = 0.05  # stepping out of an initial overlap: 1 mm doubling to 5 cm
RAY_MAX_CASTS = 96         # per query, a hard bound whatever the scene holds
# Containment ("is this point INSIDE a solid?"). Every collider on these stages is a
# triangle mesh (probe: approximation "none"), and an overlap query only tests a volume
# against triangles -- a sphere wholly inside a closed mesh overlaps nothing. So the test
# is by face orientation instead: from inside a closed, outward-wound mesh the first
# surface a ray meets is a BACK face, which a two-sided cast reports and the default
# (culled) cast skips. Voted over 5 directions so one single-sided or mis-wound quad
# cannot fake it; a closed solid votes in all 5 that reach its walls within range.
CONTAIN_DIRS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0))
CONTAIN_RANGE_M = 8.0
CONTAIN_MIN_VOTES = 3
CONTAIN_GRID_HEIGHTS_M = (0.15, 0.60)   # footprint grid, above base_link
# ...but most props here are NOT closed solids, measured on the hospital stage: the
# bedside table is a hollow cabinet (its centre sits in an empty compartment whose walls
# face INTO it), SideWall13 is two thin panel boxes with an air gap (the "centre of the
# wall" is in the gap, panels 6.5 / 7.4 cm away facing it), and the drinks machine is an
# open shell. From inside each, every face the ray meets is a FRONT face, so the vote
# above stays silent. The second test is ENCLOSURE: two-sided horizontal rays from the
# point; if almost none get ESCAPE_RANGE_M out, the point is in a cavity or a slot no
# robot can occupy. A 0.6 m range lets a corridor or doorway escape along its length and
# across diagonals; an aligned slot narrower than ~0.46 m escapes on <= 2 bearings.
ESCAPE_BEARINGS = 16
ESCAPE_RANGE_M = 0.6
ESCAPE_MAX_ENCLOSED = 3

# (ok, warn). Upper bounds unless the rule is listed in HIGHER_IS_BETTER. Every value is
# justified in the module docstring next to what calibration measured.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "heading.off_bearing_abs_deg": (30.0, 90.0),
    "spawn.start_embed_m": (-0.01, -0.03),
    "spawn.start_drop_m": (0.05, 0.30),
    "spawn.floor_offset_abs_m": (0.05, 0.10),
    "spawn.rest_err_m": (0.005, 0.02),
    "spawn.settle_tail_z_span_m": (0.001, 0.005),
    "spawn.late_z_drift_m": (0.001, 0.005),
    "spawn.settle_xy_drift_m": (0.02, 0.10),
    # Held rest: the sim speed read after a held step is the contact solver's, not 0 (0.03-0.14 m/s
    # measured on outdoor_small, 0.137 on the 6.8 deg pillars slope). OK bound = the runner's own
    # base_drive.SETTLE_SPEED_MPS (0.25). Calibrated on ONE environment: re-measure indoors.
    "spawn.end_lin_speed_mps": (0.25, 0.50),
    "spawn.end_ang_speed_radps": (0.50, 1.00),   # held; worst measured 0.274 rad/s (pillars)
    "spawn.tilt_deg": (3.0, 10.0),
    "clearance.min_clearance_m": (0.10, 0.0),
    "goal.goal_floor_gap_m": (0.10, 0.50),
    "goal.floor_dz_m": (0.05, float("inf")),
    "drive.path_ratio": (1.10, 1.50),
    "drive.ground_err_m": (0.010, 0.05),
    "drive.z_step_max_m": (0.008, 0.03),
    "drive.max_lin_speed_mps": (1.0, 3.0),
    "drive.max_ang_speed_radps": (6.0, 15.0),
    "drive.resyncs": (0, 2),
    "drive.track_err_max_m": (0.05, 0.25),
    "drive.yaw_err_deg": (2.0, 10.0),
    "drive.tilt_max_deg": (3.0, 10.0),
    "camera.luma_mean": (40.0, 10.0),
    "camera.luma_std": (15.0, 5.0),
    "camera.frac_black": (0.40, 0.80),
    "camera.frac_white": (0.15, 0.50),
    "lidar.height_err_m": (0.02, 0.05),
    "lidar.tilt_deg": (3.0, 10.0),
    "lidar.min_clearance_m": (0.10, 0.0),
}
MOBILITY_MIN_PATH_RATIO = 0.30    # drive probe: below this the robot could barely move
MOBILITY_MAX_BLOCKED_FRAC = 0.50  # ...or the guard held more than half its translate steps
HIGHER_IS_BETTER = {
    "spawn.start_embed_m", "clearance.min_clearance_m", "camera.luma_mean",
    "camera.luma_std", "lidar.min_clearance_m",
}
ORDER = {"OK": 0, "WARN": 1, "FAIL": 2, "ERROR": 3}


# ----------------------------------------------------------------------------- grading
def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def grade(key: str, value) -> dict:
    """One threshold rule -> {metric, value, verdict, rule}. A missing or non-finite
    value FAILS: a measurement that could not be taken is not a pass."""
    ok, warn = THRESHOLDS[key]
    hib = key in HIGHER_IS_BETTER
    metric = key.split(".", 1)[1]
    if not _finite(value):
        return {"metric": metric, "value": value, "verdict": "FAIL",
                "rule": f"{metric}={value} is not a finite measurement -> FAIL"}
    if hib:
        verdict = "OK" if value >= ok else ("WARN" if value >= warn else "FAIL")
        rule = f"{metric}={value:.4g}: >= {ok:g} OK, >= {warn:g} WARN, else FAIL -> {verdict}"
    else:
        verdict = "OK" if value <= ok else ("WARN" if value <= warn else "FAIL")
        rule = f"{metric}={value:.4g}: <= {ok:g} OK, <= {warn:g} WARN, else FAIL -> {verdict}"
    return {"metric": metric, "value": value, "verdict": verdict, "rule": rule}


def flag(metric: str, verdict: str, rule: str, value=None) -> dict:
    return {"metric": metric, "value": value, "verdict": verdict, "rule": f"{rule} -> {verdict}"}


def worst(verdicts) -> str:
    verdicts = list(verdicts)
    return max(verdicts, key=lambda v: ORDER[v]) if verdicts else "OK"


def check(metrics: dict, rules: list[dict]) -> dict:
    return {"verdict": worst(r["verdict"] for r in rules), "rules": rules, "metrics": metrics}


# ------------------------------------------------------------------------ math helpers
def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def off_bearing_deg(start, goal, yaw_deg: float) -> float:
    """Goal bearing relative to the start heading, degrees, + = goal to the LEFT. The same
    formula as import_benchmark.off_bearing_deg (forward = +X, yaw a plain rotate-z),
    recomputed here so a table without that field is still graded."""
    bearing = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    return wrap_deg(bearing - float(yaw_deg))


class Hit(NamedTuple):
    dist: float                  # from the ray's ORIGINAL origin
    prim: str
    pos: tuple[float, float, float]
    normal: tuple[float, float, float]
    initial: bool                # initial overlap: the origin was inside this collider


def quat_rpy(q) -> tuple[float, float, float]:
    """(roll, pitch, yaw) radians from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_rotate(q, v) -> tuple[float, float, float]:
    """Rotate vector v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = (float(c) for c in q)
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w t + cross(q.xyz, t)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def r4(v, n: int = 4):
    return round(float(v), n) if _finite(v) else v


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_mib_of(pid: int) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            total += int(parts[1])
    return total or None


class _OfflinePolicy:
    """Stands in for PolicyClient so `NavigationRunner.setup()` runs unchanged without a
    policy server. Only `wait_until_ready` is reached during setup."""

    base = "http://offline:0"

    def wait_until_ready(self, *a, **k) -> dict:
        return {"mode": "validate_episodes (no policy)"}

    def __getattr__(self, name):
        raise RuntimeError(f"validate_episodes has no policy server (called .{name})")


class LogTap:
    """Kit log messages worth a verdict: ERROR+ from any source, WARN+ from PhysX.

    The outdoor stage's first run printed, between two otherwise ordinary lines,
    "PhysX error: The application needs to increase
    PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity ... otherwise, the
    simulation will miss interactions" -- i.e. contacts silently dropped -- in a 39 MB log
    whose other 144k warnings are foliage instancing noise. A check that only reads poses
    cannot see that, so the log is tapped and attributed to the episode it happened in.

    NOT `carb.logging.add_logger` with a Python callback -- that is what the first version
    did, and it DEADLOCKED Kit on exactly the stage it was written for. A carb logger runs
    on whichever thread logs; PhysX's GPU broadphase logs that capacity error from a worker
    thread mid-step, the callback then needs the GIL, and the main thread holds the GIL
    while it waits for that step: every thread asleep, 0% CPU, forever (measured: log
    frozen at 32.5 s right after the error, main thread parked in futex_do_wait). Kit's
    own telemetry extension says the same in its docstring ("known to create deadlocks with
    GIL because of being called from many threads"). The deferred event
    `omni.kit.app.GLOBAL_EVENT_ERROR_LOG` is the safe channel: Kit queues messages from any
    thread and dispatches them on the main thread during `kit.update()`. Probed before use:
    a `carb.log_error` raised on a Python worker thread arrived on the main thread, and
    WARN (level 0) is delivered as well as ERROR (level 1).

    The price is attribution lag of at most one frame -- a message logged in an episode's
    final step is dispatched in the next update, i.e. counted against the next episode.

    Installed from inside SimulationApp's constructor (see `_tap_simulation_app`) so the
    stage load is covered too: errors raised while the stage opens affect every episode.
    """

    def __init__(self):
        self.events: list[tuple[int, str, str]] = []
        self.off_main_thread = 0   # must stay 0; recorded so a Kit change would be visible
        self._sub = None

    def install(self) -> None:
        if self._sub is not None:
            return
        import threading

        import carb
        import carb.eventdispatcher
        import omni.kit.app

        self._main = threading.get_ident()
        self._get_ident = threading.get_ident
        self._err = carb.logging.LEVEL_ERROR
        self._warn = carb.logging.LEVEL_WARN
        self._sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_ERROR_LOG,
            on_event=self._on_log,
            observer_name="validate_episodes.LogTap",
        )

    def _on_log(self, e) -> None:
        try:
            if self._get_ident() != self._main:
                self.off_main_thread += 1
            lvl, source = int(e["level"]), str(e["source"])
            if lvl >= self._err or (lvl >= self._warn and "physx" in source.lower()):
                self.events.append((lvl, source, str(e["message"])[:400]))
        except Exception:  # noqa: BLE001 -- an observer must never raise into Kit
            pass

    def mark(self) -> int:
        return len(self.events)

    def summary(self, start: int, end: int | None = None) -> dict:
        import re

        evs = self.events[start:end]
        err = getattr(self, "_err", 1)
        uniq: dict[tuple, int] = {}
        for lvl, src, msg in evs:
            key = ("ERROR" if lvl >= err else "WARN", src, re.sub(r"\d+", "N", msg[:160]))
            uniq[key] = uniq.get(key, 0) + 1
        return {
            "physx_errors": sum(1 for lvl, src, _ in evs if lvl >= err and "physx" in src.lower()),
            "physx_warnings": sum(1 for lvl, src, _ in evs if lvl < err),
            "other_errors": sum(1 for lvl, src, _ in evs if lvl >= err
                                and "physx" not in src.lower()),
            "messages": [{"count": n, "level": k[0], "source": k[1], "message": k[2]}
                         for k, n in sorted(uniq.items(), key=lambda kv: -kv[1])[:12]],
            "observer_off_main_thread": self.off_main_thread,
        }


def _tap_simulation_app(tap: LogTap) -> None:
    """Make `from isaacsim import SimulationApp` (as run_navigation.setup() does) return a
    subclass that installs the log tap the moment Kit's framework exists, before the stage
    is opened. If setup() ever imports it another way the tap is installed right after
    setup instead, and only the stage-load window goes unwatched."""
    import isaacsim

    real = isaacsim.SimulationApp

    class _TappedSimulationApp(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            tap.install()

    isaacsim.SimulationApp = _TappedSimulationApp


# =============================================================================== main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate DynaNav episodes on one nav stage. "
                                 "See the module docstring for every check and threshold.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--env", default=None,
                    help="environment (stage) to validate: hospital, office, warehouse, "
                         "outdoor_small")
    ap.add_argument("--only", action="append", default=[],
                    help="episode names, comma separated; repeatable")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--gpu", type=int, default=0, help="Kit active/physics GPU (never "
                    "CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--summarize", nargs="+", type=Path, default=None, metavar="JSON",
                    help="no Kit: print every numeric metric's range per environment "
                         "across these reports, plus their verdict tables, and exit")
    return ap.parse_args()


def summarize(paths: list[Path]) -> int:
    """Range of every numeric metric per environment across reports -- the tool that
    produced the CALIBRATION table, kept so a scene agent can set its own numbers next
    to it with the same arithmetic. Plain Python, never starts Kit."""
    rows = []
    for path in paths:
        d = json.loads(path.read_text())
        print_table(d)
        for e in d.get("episodes", []):
            for name, c in e.get("checks", {}).items():
                for k, v in c["metrics"].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) \
                            and math.isfinite(v):
                        rows.append((d.get("env", "?"), f"{name}.{k}", float(v)))
    envs = sorted({r[0] for r in rows})
    print(f"\n{'metric':40} {'all':>22}   " + "   ".join(f"{e:>22}" for e in envs))
    for key in sorted({r[1] for r in rows}):
        def rng(vals):
            return f"{min(vals):.4g} .. {max(vals):.4g}" if vals else "-"
        allv = [v for _, k, v in rows if k == key]
        print(f"{key:40} {rng(allv):>22}   " + "   ".join(
            f"{rng([v for e2, k, v in rows if k == key and e2 == e]):>22}" for e in envs))
    return 0


def select_episodes(args) -> tuple[str, list]:
    grouped = episodes_by_env(args.episodes)
    only = [n.strip() for chunk in args.only for n in chunk.split(",") if n.strip()]
    if only:
        known = {ep.name: ep for eps in grouped.values() for ep in eps}
        missing = [n for n in only if n not in known]
        if missing:
            raise SystemExit(f"unknown episode(s) in {args.episodes}: {missing}")
        envs = {env_name(known[n].scene) for n in only}
        if args.env is None:
            if len(envs) != 1:
                raise SystemExit(f"--only spans {sorted(envs)}; one stage per process, "
                                 f"pass --env and run each separately")
            args.env = envs.pop()
        elif envs - {args.env}:
            raise SystemExit(f"--only names episodes outside --env {args.env}: "
                             f"{[n for n in only if env_name(known[n].scene) != args.env]}")
    if args.env is None:
        raise SystemExit(f"pass --env, one of {sorted(grouped)}")
    if args.env not in grouped:
        raise SystemExit(f"no episodes for env {args.env!r} in {args.episodes}; have "
                         f"{sorted(grouped)}")
    eps = grouped[args.env]
    if only:
        eps = [ep for ep in eps if ep.name in only]
    return args.env, eps


def main() -> int:
    args = parse_args()
    if args.summarize:
        return summarize(args.summarize)
    env, episodes = select_episodes(args)
    raw = (yaml.safe_load(args.episodes.read_text()) or {}).get("episodes", {})
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (REPO / "nav" / "results" / "validate" / f"{env}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.frames_dir or out.with_name(out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.resolve().is_relative_to(REPO):
        # nav/results/.gitignore keeps *.jpg on purpose (campaign evidence); a sweep of 85
        # 1080p frames is not that, so the frames directory ignores itself.
        (frames_dir / ".gitignore").write_text("*\n")
    print(f"[validate] {len(episodes)} episode(s) of env {env} from {args.episodes}",
          flush=True)
    print(f"[validate] report -> {out}\n[validate] frames -> {frames_dir}", flush=True)

    # ---- boot exactly like a real run ------------------------------------------------
    import run_navigation as rn

    t_boot = time.time()
    runner = rn.NavigationRunner(
        episode=episodes[0],
        controller_name="braking",
        scratch_dir=frames_dir / ".scratch",   # NavCameraSource mkdirs it; unused
        headless=True,
        sim_gpu=args.gpu,
    )
    runner.policy = _OfflinePolicy()
    report: dict = {
        "tool": "nav/tools/validate_episodes.py",
        "env": env,
        "episodes_file": str(args.episodes),
        "stage": str(runner.scene_path),
        "git": git_sha(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "nav_lidar": bool(rn.NAV_LIDAR),
        "thresholds": {k: list(v) for k, v in THRESHOLDS.items()},
        "higher_is_better": sorted(HIGHER_IS_BETTER),
        "episodes": [],
    }
    code = 1
    tap = LogTap()
    _tap_simulation_app(tap)
    report["status"] = "running"
    report["current_episode"] = None
    report["current_phase"] = "boot"
    _write(out, report)
    try:
        runner.setup()
        tap.install()            # no-op when the SimulationApp hook already did it
        # A native crash (segfault) skips every Python handler below, so the JSON would
        # stay at status "running" forever. faulthandler at least puts the Python stack of
        # the crashing call in the log; `current_episode`/`current_phase` in the JSON name
        # the culprit. Installed after Kit's own crash handler, and chains to it.
        import faulthandler

        faulthandler.enable(file=sys.stderr, all_threads=False)
        report["boot_s"] = round(time.time() - t_boot, 1)
        report["boot_log"] = tap.summary(0)
        boot_mark = tap.mark()
        print(f"[validate] stage ready in {report['boot_s']} s; boot log: "
              f"{report['boot_log']['physx_errors']} PhysX error(s), "
              f"{report['boot_log']['other_errors']} other error(s)", flush=True)
        ctx = Context(runner, rn)
        ctx.tap = tap
        ctx.boot_physx_errors = report["boot_log"]["physx_errors"]
        ctx.boot_mark = boot_mark

        def on_phase(phase: str) -> None:
            report["current_phase"] = phase
            _write(out, report)

        ctx.on_phase = on_phase
        report["robot_envelope"] = ctx.envelope
        report["robot_rest_joints"] = ctx.joint_snapshot_info
        report["gpu_mib_after_boot"] = gpu_mib_of(os.getpid())
        print(f"[validate] robot envelope (base_link frame): {ctx.envelope}", flush=True)
        for i, ep in enumerate(episodes, 1):
            t0 = time.time()
            print(f"\n[validate] ({i}/{len(episodes)}) {ep.name}", flush=True)
            report["current_episode"] = ep.name
            on_phase("start")
            try:
                rec = ctx.validate(ep, frames_dir)
            except Exception as exc:  # noqa: BLE001 -- recorded, the next episode runs
                rec = {"name": ep.name, "verdict": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
                print(rec["traceback"], flush=True)
            meta = raw.get(ep.name, {}) or {}
            rec["dynanav"] = {k: meta[k] for k in (
                "dynanav_episode", "num_people", "off_bearing_deg", "dz_m", "ours",
                "dynanav_scored", "dynanav_success", "dynanav_spl") if k in meta}
            rec["wall_s"] = round(time.time() - t0, 1)
            report["episodes"].append(rec)
            print(f"[validate] {ep.name}: {rec['verdict']}  " + "  ".join(
                f"{k}={v['verdict']}" for k, v in rec.get("checks", {}).items()), flush=True)
            _write(out, report)          # after every episode: a crash keeps what was done
        report["gpu_mib_peak_seen"] = gpu_mib_of(os.getpid())
        report["current_episode"] = None
        report["current_phase"] = "done"
        report["ray_casts"] = ctx.casts
        verdicts = [e["verdict"] for e in report["episodes"]]
        report["summary"] = {v: verdicts.count(v) for v in ORDER}
        report["finished"] = datetime.now().isoformat(timespec="seconds")
        report["status"] = "completed"
        _write(out, report)
        print_table(report)
        code = 2 if any(v in ("FAIL", "ERROR") for v in verdicts) else 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        report["fatal"] = traceback.format_exc()
        report["status"] = "crashed"
        _write(out, report)
        code = 1
    # python.sh turns EVERY non-zero status into 1 ("There was an error running python"),
    # so the exit code alone cannot tell "an episode failed" from "the harness died".
    # This line and the JSON's `status` can.
    print(f"[validate] STATUS {report['status']}  summary={report.get('summary')}  "
          f"exit {code}  ({out})", flush=True)
    sys.stdout.flush()
    # Last statement on purpose: nothing after SimulationApp.close() executes (CLAUDE.md),
    # and close() hands `exit_code` to os._exit on Kit's fast-shutdown path.
    if runner.kit is not None:
        runner.kit.close(exit_code=code)
    return code


def _write(path: Path, report: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, default=str))
    tmp.replace(path)


def print_table(report: dict) -> None:
    print("\n" + "=" * 106)
    print(f"{'episode':28} {'verdict':7} {'head':5} {'spawn':5} {'clear':5} {'goal':5} "
          f"{'drive':5} {'cam':5} {'lidar':5} {'physx':5}  worst rules")
    print("-" * 106)
    for e in report["episodes"]:
        c = e.get("checks", {})
        bad = [r["rule"] for ch in c.values() for r in ch["rules"]
               if r["verdict"] == "FAIL"]
        bad += [r["rule"] for ch in c.values() for r in ch["rules"]
                if r["verdict"] == "WARN"]
        print(f"{e['name']:28} {e['verdict']:7} " + " ".join(
            f"{c[k]['verdict'] if k in c else '-':5}"
            for k in ("heading", "spawn", "clearance", "goal", "drive", "camera", "lidar",
                      "physx"))
            + ("  " + " | ".join(bad[:2]) if bad else "")
            + (f"  {e['error']}" if e.get("error") else ""))
    print("=" * 106)
    if report.get("status") == "running":
        print("status 'running': if the pid", report.get("pid"), "is gone, the process "
              "CRASHED natively in episode", report.get("current_episode"), "phase",
              report.get("current_phase"))
    print("summary:", report.get("summary"))


# ============================================================================ context
class Context:
    """Everything one Kit session needs, built once after `setup()`."""

    def __init__(self, runner, rn):
        import numpy as np
        import omni.usd
        from omni.physx import get_physx_scene_query_interface

        from alohamini1_specs import BASE_LINK_REST_Z_M, NAV_CAMERA_PRIM_PATH
        from lidar_2d import SweepingLidar2D

        self.np = np
        self.runner = runner
        self.rn = rn
        self.kit = runner.kit
        self.art = runner.art
        self.base = runner.base
        self.camera = runner.camera
        self.guard = runner.guard
        self.lidar_live = runner.lidar     # None unless NAV_LIDAR=1, as in a real run
        self.dt = rn.PHYSICS_DT
        self.rest_z = BASE_LINK_REST_Z_M
        self.stage = omni.usd.get_context().get_stage()
        self.sq = get_physx_scene_query_interface()
        self.SweepingLidar2D = SweepingLidar2D
        self.cam_path = NAV_CAMERA_PRIM_PATH
        self.casts = 0                      # raycast_closest calls, for the report
        self.on_phase = lambda phase: None  # main() points this at the JSON writer
        self.envelope = self._robot_envelope()
        self.cam_local = self._camera_offset()
        self.envelope["camera_offset_m"] = [round(v, 4) for v in self.cam_local]
        # The robot's joint state as the stage left it, right after setup(): what every
        # bench.sh episode starts from, because each one is a fresh process. Restored
        # before every episode here (see restore_robot) -- without it an episode that
        # followed a violent one inherited its arm/lift motion: same start, yaw and stage
        # as the control, 12x the end angular speed, after the robot had been thrown at
        # 9 m/s in the episode before.
        for attr in ("_spawn_pose", "_physics_fault"):
            if not callable(getattr(runner, attr, None)):
                raise RuntimeError(f"run_navigation.NavigationRunner.{attr} is gone; this "
                                   "tool spawns and fault-checks THROUGH the runner and "
                                   "will not silently diverge from it -- update it")
        if not hasattr(self.base, "_xy"):
            raise RuntimeError("base_drive.KinematicBase has no _xy (the integrated x/y "
                               "the track/resync metrics compare against) -- update this tool")
        names = list(self.art.dof_names)
        wheels = set(getattr(self.base, "_wheel_indices", []))
        self.arm_idx = np.array([i for i in range(len(names)) if i not in wheels],
                                dtype=np.int64)
        self.arm_names = [names[i] for i in self.arm_idx]
        self.q0 = np.asarray(self.art.get_joint_positions(), dtype=np.float32).reshape(1, -1)
        act = self.art.get_applied_actions()
        tgt = getattr(act, "joint_positions", None) if act is not None else None
        self.target0 = (np.asarray(tgt, dtype=np.float32).reshape(1, -1)
                        if tgt is not None else None)
        # The robot's ROOT pose as the stage left it: a bench.sh process (one per episode)
        # runs ground_spawn_pose with the robot STANDING here, so for an episode whose start
        # is this pose the snap's rays go straight through the robot's own tyre spheres
        # (office_e26: base_drive 6caeb283 snaps 1.69 cm into the floor). Several episodes
        # share one process here, so every later episode used to snap with the robot parked
        # where the previous probe ended and could never show it (restore_robot() puts it back).
        _p0, _q0 = self.pose()
        self.authored_root = (_p0, quat_rpy(_q0)[2])
        self.joint_snapshot_info = {
            "joints": {n: round(float(self.q0[0, i]), 5) for i, n in zip(self.arm_idx,
                                                                          self.arm_names)},
            "position_targets_snapshotted": self.target0 is not None,
            "wheels_excluded": [names[i] for i in sorted(wheels)],
        }

    # ------------------------------------------------------------------- robot geometry
    def _robot_envelope(self) -> dict:
        """Axis-aligned bounds of the robot's own geometry in the base_link frame.

        Read off USD rather than assumed: every Gprim under /World/Aloha/Geometry
        (instance proxies included -- this asset is point-instanced, and a plain Mesh
        traverse undercounts it), each bound taken in world space and its corners pulled
        back into base_link's frame. Lights (the headlight's 0.3 m sphere) are not Gprims
        and stay out. Both transforms come from the same USD state, so the result is the
        robot's shape whether or not physics has written poses back.
        """
        from pxr import Gf, Usd, UsdGeom

        fallback = {"x_min": -0.21, "x_max": 0.21, "y_min": -0.31, "y_max": 0.31,
                    "z_min": -0.0078, "z_max": 1.20, "source": "fallback constants"}
        base = self.stage.GetPrimAtPath(BASE_LINK)
        geom = self.stage.GetPrimAtPath("/World/Aloha/Geometry")
        if not base.IsValid() or not geom.IsValid():
            return fallback
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False)
        inv = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).GetInverse()
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        n = 0
        for prim in Usd.PrimRange(geom, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            bound = cache.ComputeWorldBound(prim)
            rng = bound.GetRange()
            if rng.IsEmpty():
                continue
            m = bound.GetMatrix()
            a, b = rng.GetMin(), rng.GetMax()
            for cx in (a[0], b[0]):
                for cy in (a[1], b[1]):
                    for cz in (a[2], b[2]):
                        p = inv.Transform(m.Transform(Gf.Vec3d(cx, cy, cz)))
                        for k in range(3):
                            lo[k] = min(lo[k], p[k])
                            hi[k] = max(hi[k], p[k])
            n += 1
        ok = (n > 0 and lo[0] < 0 < hi[0] and lo[1] < 0 < hi[1]
              and 0.1 < hi[0] - lo[0] < 1.5 and 0.1 < hi[1] - lo[1] < 1.5
              and 0.3 < hi[2] < 2.5)
        if not ok:
            fallback["note"] = f"USD envelope implausible ({n} gprims, lo={lo}, hi={hi})"
            return fallback
        return {"x_min": round(lo[0], 4), "x_max": round(hi[0], 4),
                "y_min": round(lo[1], 4), "y_max": round(hi[1], 4),
                "z_min": round(lo[2], 4), "z_max": round(hi[2], 4),
                "gprims": n, "source": "USD bounds of /World/Aloha/Geometry"}

    def _camera_offset(self) -> tuple[float, float, float]:
        """camera_nav's translation in base_link's frame, as authored."""
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(self.cam_path)
        base = self.stage.GetPrimAtPath(BASE_LINK)
        if not prim.IsValid() or not base.IsValid():
            from alohamini1_specs import NAV_CAMERA_FORWARD_M, NAV_CAMERA_HEIGHT_M

            return (NAV_CAMERA_FORWARD_M, 0.0, NAV_CAMERA_HEIGHT_M)
        t = Usd.TimeCode.Default()
        rel = (UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)
               * UsdGeom.Xformable(base).ComputeLocalToWorldTransform(t).GetInverse())
        tr = rel.ExtractTranslation()
        return (float(tr[0]), float(tr[1]), float(tr[2]))

    def footprint_extent(self, bearing_body: float) -> float:
        """Distance from the base_link origin to the envelope's edge along a body bearing."""
        e = self.envelope
        c, s = math.cos(bearing_body), math.sin(bearing_body)
        xl = e["x_max"] if c > 0 else -e["x_min"]
        yl = e["y_max"] if s > 0 else -e["y_min"]
        tx = xl / abs(c) if abs(c) > 1e-9 else math.inf
        ty = yl / abs(s) if abs(s) > 1e-9 else math.inf
        return min(tx, ty)

    # -------------------------------------------------------------------- sim readouts
    def pose(self):
        p, q = self.art.get_world_poses()
        return tuple(float(v) for v in p[0]), tuple(float(v) for v in q[0])

    def speeds(self) -> tuple[float, float]:
        v = self.np.asarray(self.art.get_velocities())[0]
        return (float(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)),
                float(math.sqrt(v[3] ** 2 + v[4] ** 2 + v[5] ** 2)))

    def restore_robot(self) -> dict:
        """Put the joints back where setup() left them and stop everything.

        bench.sh runs one process per episode, so a real run always starts from the
        stage's own joint state. Here several episodes share a process, so the arms and
        lift are restored (positions AND the position targets they had after setup), and
        root and joint velocities zeroed. The wheels are continuous joints with velocity
        drives: their angle means nothing and their targets are base_drive's business, so
        they are left out. What the joints had drifted to is RETURNED, so an episode that
        follows a violent one shows how far it had been knocked.
        """
        np = self.np
        q = np.asarray(self.art.get_joint_positions(), dtype=np.float32).reshape(1, -1)
        dev = np.abs(q[0, self.arm_idx] - self.q0[0, self.arm_idx])
        k = int(np.argmax(dev)) if dev.size else 0
        self.art.set_joint_positions(self.q0[:, self.arm_idx], joint_indices=self.arm_idx)
        if self.target0 is not None:
            self.art.set_joint_position_targets(self.target0[:, self.arm_idx],
                                                joint_indices=self.arm_idx)
        self.art.set_velocities(np.zeros((1, 6), dtype=np.float32))
        self.art.set_joint_velocities(np.zeros((1, len(self.art.dof_names)), dtype=np.float32))
        # ...and the root: stand it at the stage-authored pose, 30 updates so PhysX's scene
        # queries see the tyre colliders there, exactly the state a fresh bench process has
        # when _spawn_pose runs. Without this an episode never starts with the robot at its
        # own spawn, which is where ground_spawn_pose's rays meet the robot's own colliders.
        self.base.reset_to(self.authored_root[0], self.authored_root[1])
        for _ in range(30):
            self.kit.update()
        return {"joint_dev_before_restore_rad": r4(float(dev.max()) if dev.size else 0.0, 5),
                "joint_dev_worst": self.arm_names[k] if dev.size else None}

    # ------------------------------------------------------------------- scene queries
    def ray_hits(self, origin, direction, dist, *, both_sides: bool = False,
                 max_hits: int = 32) -> list[Hit]:
        """Non-robot hits along a ray, nearest first.

        Iterated `raycast_closest`, re-cast RAY_SKIP_M past each hit -- never
        `raycast_all`, which segfaults Kit on the outdoor stage (exit 139 inside
        libomni.physx.plugin, 3 of 3 runs, on a ray raycast_closest handles fine) and
        reports one hit per shape, so a floor sharing a mesh with something above it was
        never seen. Robot prims (/World/Aloha) are skipped, as the guard and lidar skip
        them.

        `both_sides=False` is PhysX's default: back faces of triangle meshes are culled, so
        a ray from inside a closed mesh does not see its walls. `both_sides=True` reports
        them -- the difference between the two is the containment test.

        A hit at distance ~0 is an INITIAL OVERLAP: the origin was inside a solid
        primitive or convex. It is recorded once per prim with `initial=True` (never
        mistaken for a surface) and stepped out of in doubling increments, which also
        walks out of the robot's own convex links when a ray starts inside the robot.
        """
        o = [float(v) for v in origin]
        d = [float(v) for v in direction]
        hits: list[Hit] = []
        travelled = 0.0
        step = RAY_SKIP_M
        inside_once: set[str] = set()
        for _ in range(RAY_MAX_CASTS):
            remaining = float(dist) - travelled
            if remaining <= 1e-6:
                break
            r = self.sq.raycast_closest(tuple(o), tuple(d), remaining, bothSides=both_sides)
            self.casts += 1
            if not r or not r.get("hit"):
                break
            t = max(0.0, float(r.get("distance", 0.0)))
            prim = str(r.get("collision", ""))
            robot = prim.startswith(ROBOT_PREFIX)
            if t > RAY_ZERO_M:
                if not robot:
                    pos, nrm = r["position"], r["normal"]
                    hits.append(Hit(travelled + t, prim,
                                    (float(pos[0]), float(pos[1]), float(pos[2])),
                                    (float(nrm[0]), float(nrm[1]), float(nrm[2])), False))
                adv = t + RAY_SKIP_M
                step = RAY_SKIP_M
            else:
                if not robot and prim not in inside_once:
                    inside_once.add(prim)
                    hits.append(Hit(travelled, prim, (o[0], o[1], o[2]),
                                    (-d[0], -d[1], -d[2]), True))
                adv = step
                step = min(2.0 * step, RAY_ZERO_STEP_MAX_M)
            if len(hits) >= max_hits:
                break
            travelled += adv
            o = [o[k] + d[k] * adv for k in range(3)]
        return hits

    def first_hit(self, origin, direction, dist, both_sides: bool = False) -> Hit | None:
        hits = self.ray_hits(origin, direction, dist, both_sides=both_sides, max_hits=1)
        return hits[0] if hits else None

    def inside_solids(self, p) -> tuple[list[str], dict[str, int]]:
        """Which non-robot colliders CONTAIN point p: (prims with >= CONTAIN_MIN_VOTES
        votes, all votes).

        Per direction: the first surface a TWO-SIDED cast meets, against the first a
        culled (default) cast meets. Same surface -> it faces p, p is outside it along
        that direction. The two-sided hit is nearer, or the culled cast sees nothing ->
        that surface is a back face, p is behind it: one vote for that prim. An initial
        overlap (p inside a solid convex or the ground plane's half-space) is a vote too.
        A closed, outward-wound mesh votes in every direction whose wall lies within
        CONTAIN_RANGE_M; a single mis-wound quad votes in one.
        """
        votes: dict[str, int] = {}
        for d in CONTAIN_DIRS:
            hb = self.first_hit(p, d, CONTAIN_RANGE_M, both_sides=True)
            if hb is None:
                continue
            if not hb.initial:
                hf = self.first_hit(p, d, CONTAIN_RANGE_M, both_sides=False)
                if hf is not None and hf.dist <= hb.dist + 1e-4:
                    continue          # the nearest surface faces p: outside, this way
            votes[hb.prim] = votes.get(hb.prim, 0) + 1
        return sorted(k for k, n in votes.items() if n >= CONTAIN_MIN_VOTES), votes

    def trapped(self, p) -> dict:
        """Is point p somewhere no robot can be: inside a closed solid (inside_solids) OR
        enclosed in a cavity/slot (<= ESCAPE_MAX_ENCLOSED of ESCAPE_BEARINGS two-sided
        horizontal rays get ESCAPE_RANGE_M out). Robot prims never count."""
        inside, votes = self.inside_solids(p)
        escape, walls = 0, {}
        for k in range(ESCAPE_BEARINGS):
            b = 2.0 * math.pi * k / ESCAPE_BEARINGS
            h = self.first_hit(p, (math.cos(b), math.sin(b), 0.0), ESCAPE_RANGE_M,
                               both_sides=True)
            if h is None:
                escape += 1
            else:
                walls[h.prim] = walls.get(h.prim, 0) + 1
        enclosed = escape <= ESCAPE_MAX_ENCLOSED
        return {"trapped": bool(inside) or enclosed, "inside": inside,
                "back_face_votes": votes, "escape_rays": escape, "of": ESCAPE_BEARINGS,
                "enclosed": enclosed,
                "walls": dict(sorted(walls.items(), key=lambda kv: -kv[1])[:3])}

    def floor_under(self, x: float, y: float, z_ref: float,
                    above: float = FLOOR_ABOVE_M) -> dict | None:
        """The walkable surface under (x, y): the highest upward-facing hit at most `above`
        over z_ref. Anything higher is reported as `overhead` -- a table the point is
        under, or the ground itself if z_ref is buried. `above` is FLOOR_ABOVE_M against a
        DynaNav z, FLOOR_ABOVE_ROBOT_M against the robot's own z (rest and ground errors:
        a robot at rest is 7.8 mm over its floor, so 5 cm is already a different surface).
        Iterated casts see every surface down the column, including the ones that share a
        mesh with an overhead hit."""
        top = z_ref + above + 2.0
        hits = self.ray_hits((x, y, top), (0.0, 0.0, -1.0), 2.0 + above + FLOOR_BELOW_M)
        overhead = []
        for h in hits:
            if h.initial:
                continue              # the ray STARTED inside this; not a surface
            z = h.pos[2]
            if z > z_ref + above:
                overhead.append((round(z, 3), h.prim))
                continue
            if h.normal[2] < FLOOR_MIN_NZ:
                continue
            return {"z": z, "prim": h.prim, "normal_z": h.normal[2],
                    "slope_deg": math.degrees(math.acos(max(-1.0, min(1.0, h.normal[2])))),
                    "overhead": overhead[:3]}
        return None

    def local_floor(self, x: float, y: float, z_ref: float, prim: str) -> dict:
        """The walkable floor AROUND (x, y): a ring of samples just outside the footprint
        and at twice that. A spawn on furniture is a surface that the floor around it
        does not share -- the median of the ring is the floor, the spawn is above it, and
        its prim appears nowhere on the ring. On a uniform slope the ring is symmetric
        about the centre, so the median stays at the centre's height."""
        zs, same, at_level = [], 0, 0
        prims: dict[str, int] = {}
        for r in LOCAL_FLOOR_RADII_M:
            for k in range(LOCAL_FLOOR_BEARINGS):
                a = 2.0 * math.pi * k / LOCAL_FLOOR_BEARINGS
                fl = self.floor_under(x + r * math.cos(a), y + r * math.sin(a), z_ref)
                if fl is None:
                    continue
                zs.append(fl["z"])
                prims[fl["prim"]] = prims.get(fl["prim"], 0) + 1
                same += fl["prim"] == prim
                at_level += abs(fl["z"] - z_ref) <= LOCAL_FLOOR_LEVEL_M
        zs.sort()
        n = len(zs)
        median = (zs[n // 2] if n % 2 else 0.5 * (zs[n // 2 - 1] + zs[n // 2])) if n else None
        return {"samples": n, "of": len(LOCAL_FLOOR_RADII_M) * LOCAL_FLOOR_BEARINGS,
                "median_z": r4(median) if median is not None else None,
                "same_prim": same, "at_level": at_level,
                "prims": dict(sorted(prims.items(), key=lambda kv: -kv[1])[:4])}

    def overlap_box(self, center, half, yaw) -> list[str]:
        prims: list[str] = []

        def report(h) -> bool:
            prim = str(h.collision)
            if not prim.startswith(ROBOT_PREFIX) and prim not in prims:
                prims.append(prim)
            return True

        q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))   # (x, y, z, w)
        self.sq.overlap_box(tuple(float(v) for v in half), tuple(float(v) for v in center),
                            q, report, False)
        return prims

    def overlap_sphere(self, center, radius) -> list[str]:
        prims: list[str] = []

        def report(h) -> bool:
            prim = str(h.collision)
            if not prim.startswith(ROBOT_PREFIX) and prim not in prims:
                prims.append(prim)
            return True

        self.sq.overlap_sphere(float(radius), tuple(float(v) for v in center), report, False)
        return prims

    def footprint_box(self, x, y, floor_z, yaw):
        """(center, half extents) of the envelope box, CLEAR_Z_LO_M above the floor."""
        e = self.envelope
        cx_b = 0.5 * (e["x_min"] + e["x_max"])
        cy_b = 0.5 * (e["y_min"] + e["y_max"])
        z_lo = floor_z + CLEAR_Z_LO_M
        z_hi = floor_z + self.rest_z + e["z_max"]
        c, s = math.cos(yaw), math.sin(yaw)
        center = (x + c * cx_b - s * cy_b, y + s * cx_b + c * cy_b, 0.5 * (z_lo + z_hi))
        half = (0.5 * (e["x_max"] - e["x_min"]), 0.5 * (e["y_max"] - e["y_min"]),
                0.5 * (z_hi - z_lo))
        return center, half

    def footprint_inside(self, x, y, floor_z, yaw) -> dict[str, int]:
        """Containment over a 3 x 3 grid at two heights inside the envelope box: the part
        of "is the robot inside something" an overlap box cannot see, because a box wholly
        inside a closed triangle mesh crosses none of its triangles."""
        e = self.envelope
        m = 0.03
        xs = (e["x_min"] + m, 0.5 * (e["x_min"] + e["x_max"]), e["x_max"] - m)
        ys = (e["y_min"] + m, 0.5 * (e["y_min"] + e["y_max"]), e["y_max"] - m)
        c, s = math.cos(yaw), math.sin(yaw)
        found: dict[str, int] = {}
        for h in CONTAIN_GRID_HEIGHTS_M:
            for bx in xs:
                for by in ys:
                    pt = (x + c * bx - s * by, y + s * bx + c * by, floor_z + self.rest_z + h)
                    for prim in self.inside_solids(pt)[0]:
                        found[prim] = found.get(prim, 0) + 1
        return found

    def line_of_sight(self, a, b) -> bool:
        """Nothing but the robot between a and b, front OR back faces (two-sided, so a
        segment that starts inside a solid is blocked by the solid's wall too)."""
        seg = [b[k] - a[k] for k in range(3)]
        n = math.sqrt(sum(c * c for c in seg))
        if n < 1e-6:
            return True
        return self.first_hit(a, [c / n for c in seg], n - 1e-3, both_sides=True) is None

    def ring_heights(self) -> list[float]:
        top = self.rest_z + self.envelope["z_max"] - 0.05
        return [h for h in (0.10, 0.20, 0.35, 0.60, 0.90) if h < top] + [round(top, 3)]

    # ============================================================ one episode, 8 checks
    def validate(self, ep, frames_dir: Path) -> dict:
        rec: dict = {"name": ep.name, "start": list(ep.start), "goal": list(ep.goal),
                     "start_yaw_deg": ep.start_yaw_deg,
                     "success_threshold_m": ep.success_threshold_m, "checks": {}}
        checks = rec["checks"]
        log_mark = self.tap.mark() if getattr(self, "tap", None) else None

        # ---- 0. heading: does the robot start facing its goal? -----------------------
        checks["heading"] = self._heading(ep)

        # ---- 1. spawn, exactly as run_episode spawns ----------------------------------
        self.on_phase("restore")
        rec["restore"] = self.restore_robot()
        self.on_phase("spawn_pose")
        # run_episode: start, spawn = self._spawn_pose(ep); base.reset_to(start, yaw);
        # 30 x kit.update(); base.sync_from_sim(); base.reset_stats(). Through the runner's
        # own method, so a change to the snap applies here on the next run.
        start, snap = self.runner._spawn_pose(ep)
        self.on_phase("spawn_settle")
        self.base.reset_to(start, ep.start_yaw_rad)
        zs, xys, lin, ang, tilt = [], [], [], [], []
        for i in range(SETTLE_STEPS):
            # HELD, as run_episode does since base_drive.settle(): apply(0,0,0) + kit.update()
            self.base.apply(0.0, 0.0, 0.0, self.dt)
            self.kit.update()
            if i + 1 == RUNNER_SETTLE_STEPS:
                self.base.sync_from_sim()
                self.base.reset_stats()
            p, q = self.pose()
            r, pt, _ = quat_rpy(q)
            v, w = self.speeds()
            zs.append(p[2])
            xys.append(p[:2])
            lin.append(v)
            ang.append(w)
            tilt.append(math.degrees(max(abs(r), abs(pt))))
        self.on_phase("spawn_measure")
        p_end, q_end = self.pose()
        sz = float(start[2])
        start_floor = self.floor_under(start[0], start[1], sz, above=FLOOR_ABOVE_ROBOT_M)
        end_floor = self.floor_under(p_end[0], p_end[1], p_end[2], above=FLOOR_ABOVE_ROBOT_M)
        snap_z = snap.get("ground_z")
        local = (self.local_floor(start[0], start[1], snap_z, snap.get("ground_prim", ""))
                 if snap_z is not None else None)
        tail = zs[-TAIL_STEPS:]
        m = {
            "snap_ground_z": snap_z,
            "snap_ground_prim": snap.get("ground_prim"),
            "snap_tyre_ground_z": snap.get("tyre_ground_z"),
            "start_z": r4(sz),
            "dynanav_z": ep.start[2],
            # DynaNav's Nova Carter z minus where OUR robot was put: information about the
            # episode table, not a defect -- the runner removes it (base_drive.py).
            "dynanav_drop_m": r4(ep.start[2] - sz),
            "floor_z": r4(start_floor["z"]) if start_floor else None,
            "floor_prim": start_floor["prim"] if start_floor else None,
            "floor_slope_deg": r4(start_floor["slope_deg"], 2) if start_floor else None,
            "overhead_hits": start_floor["overhead"] if start_floor else [],
            "start_gap_m": (r4(sz - (start_floor["z"] + self.rest_z))
                            if start_floor else None),
            "local_floor": local,
            "floor_offset_m": (r4(snap_z - local["median_z"])
                               if local and local["median_z"] is not None else None),
            "z_rest": r4(p_end[2]),
            "rest_err_m": (r4(abs(p_end[2] - (end_floor["z"] + self.rest_z)), 5)
                           if end_floor else None),
            "settle_drop_m": r4(sz - p_end[2], 5),
            "settle_tail_z_span_m": r4(max(tail) - min(tail), 6),
            "settle_xy_drift_m": r4(math.dist(xys[-1], start[:2]), 5),
            "max_lin_speed_mps": r4(max(lin)),
            "max_ang_speed_radps": r4(max(ang)),
            "end_lin_speed_mps": r4(lin[-1], 5),
            "end_ang_speed_radps": r4(ang[-1], 5),
            "tilt_deg": r4(tilt[-1], 3),
            "nan_pose": not all(math.isfinite(v) for v in (*p_end, *q_end)),
        }
        rules = []
        if snap_z is None:
            rules.append(flag("snap_floor_found", "FAIL",
                              "base_drive.ground_spawn_pose found no floor under any tyre "
                              f"or the centre (cast from {ep.start[2] + 0.30:.2f} m down): "
                              "the runner spawns at DynaNav's raw z"))
        if start_floor is None:
            rules.append(flag("floor_found", "FAIL", "no walkable floor within "
                              f"[start_z - {FLOOR_BELOW_M} m, start_z + "
                              f"{FLOOR_ABOVE_ROBOT_M} m] under the spawn"))
        else:
            rules.append(grade("spawn.start_embed_m", m["start_gap_m"]))
            rules.append(grade("spawn.start_drop_m", m["start_gap_m"]))
        if snap_z is not None:
            off = m["floor_offset_m"]
            if off is None or local["samples"] < local["of"] // 4:
                rules.append(flag("floor_offset", "WARN",
                                  f"local floor undetermined: {local['samples']}/"
                                  f"{local['of']} ring samples found a floor", off))
            else:
                rules.append(grade("spawn.floor_offset_abs_m", abs(off)))
                if abs(off) > THRESHOLDS["spawn.floor_offset_abs_m"][0] \
                        and local["same_prim"] == 0:
                    rules.append(flag("spawn_island", "FAIL",
                                      f"spawned on {m['snap_ground_prim']} {off:+.3f} m from "
                                      f"the floor around it, and that prim appears on none "
                                      f"of the {local['samples']} ring samples (0.6/1.2 m): "
                                      "furniture or an object, not the floor", True))
        if m["nan_pose"]:
            rules.append(flag("nan_pose", "FAIL", "pose is NaN/inf after the settle"))
        rules.append(grade("spawn.rest_err_m", m["rest_err_m"]))
        rules.append(grade("spawn.settle_tail_z_span_m", m["settle_tail_z_span_m"]))
        rules.append(grade("spawn.settle_xy_drift_m", m["settle_xy_drift_m"]))
        rules.append(grade("spawn.end_lin_speed_mps", m["end_lin_speed_mps"]))
        rules.append(grade("spawn.end_ang_speed_radps", m["end_ang_speed_radps"]))
        rules.append(grade("spawn.tilt_deg", m["tilt_deg"]))
        spawn_m, spawn_rules = m, rules

        # ---- 2. clearance at the settled pose -------------------------------------------
        self.on_phase("clearance")
        floor_z = end_floor["z"] if end_floor else (start_floor["z"] if start_floor
                                                    else sz - self.rest_z)
        yaw = quat_rpy(q_end)[2]
        center, half = self.footprint_box(p_end[0], p_end[1], floor_z, yaw)
        inside = self.overlap_box(center, half, yaw)
        contained = self.footprint_inside(p_end[0], p_end[1], floor_z, yaw)
        floor_prims = {f["prim"] for f in (start_floor, end_floor) if f}
        best = (math.inf, None, None, None, math.inf)   # clearance, bearing, h, prim, raw
        for h in self.ring_heights():
            for k in range(RING_BEARINGS):
                b_body = 2.0 * math.pi * k / RING_BEARINGS
                b = yaw + b_body
                hit = self.first_hit((p_end[0], p_end[1], floor_z + h),
                                     (math.cos(b), math.sin(b), 0.0), RING_RANGE_M)
                if hit is None:
                    continue
                cl = hit.dist - self.footprint_extent(b_body)
                if cl < best[0]:
                    best = (cl, b_body, h, hit.prim, hit.dist)
        m = {
            "box_center": [r4(v) for v in center], "box_half": [r4(v) for v in half],
            "overlap_prims": inside[:10],
            "contained_in": contained,
            "min_clearance_m": r4(best[0]) if math.isfinite(best[0]) else None,
            "nearest_bearing_deg": r4(math.degrees(wrap(best[1])), 1) if best[1] is not None
            else None,
            "nearest_height_m": best[2], "nearest_prim": best[3],
            "nearest_dist_m": r4(best[4]) if math.isfinite(best[4]) else None,
            "ring_range_m": RING_RANGE_M,
        }
        rules = []
        non_floor = [p for p in inside if p not in floor_prims]
        non_floor += [p for p in contained if p not in non_floor]
        if non_floor:
            rules.append(flag("footprint_overlap", "FAIL",
                              f"{len(non_floor)} non-floor collider(s) inside the envelope "
                              f"box (overlap or containment): {non_floor[:3]}",
                              len(non_floor)))
        elif inside:
            rules.append(flag("footprint_overlap", "WARN",
                              f"only the floor's own collider overlaps the envelope box "
                              f"({inside[:2]}): slope or one-mesh building", len(inside)))
        else:
            rules.append(flag("footprint_overlap", "OK", "no collider in or around the "
                              "envelope box", 0))
        if m["min_clearance_m"] is None:
            rules.append(flag("min_clearance_m", "OK",
                              f"no obstacle within {RING_RANGE_M} m on any ring ray"))
        else:
            rules.append(grade("clearance.min_clearance_m", m["min_clearance_m"]))
        checks["spawn"] = check(spawn_m, spawn_rules)
        checks["clearance"] = check(m, rules)

        # ---- 3. goal --------------------------------------------------------------------
        self.on_phase("goal")
        checks["goal"] = self._goal(ep, start_floor)

        # ---- rest to 150 steps, then 5. camera and 6. lidar at the start pose ----------
        self.on_phase("rest")
        z60 = self.pose()[0][2]
        for _ in range(REST_TOTAL_STEPS - SETTLE_STEPS):
            self.base.apply(0.0, 0.0, 0.0, self.dt)   # a stopped robot is HELD in the runner
            self.kit.update()
        p150, q150 = self.pose()
        spawn_m["late_z_drift_m"] = r4(abs(p150[2] - z60), 6)
        spawn_rules.append(grade("spawn.late_z_drift_m", spawn_m["late_z_drift_m"]))
        checks["spawn"] = check(spawn_m, spawn_rules)
        self.on_phase("camera")
        checks["camera"] = self._camera(ep, frames_dir, p150, q150)
        self.on_phase("lidar")
        checks["lidar"] = self._lidar(p150, q150)

        # ---- 4. drive probe ---------------------------------------------------------------
        self.on_phase("drive")
        checks["drive"] = self._drive(ep)

        # ---- 7. what PhysX said while all of the above ran ------------------------------
        if log_mark is not None:
            checks["physx"] = self._physx_log(log_mark)

        rec["verdict"] = worst(c["verdict"] for c in checks.values())
        return rec

    # ------------------------------------------------------------------------ heading
    def _heading(self, ep) -> dict:
        off = off_bearing_deg(ep.start, ep.goal, ep.start_yaw_deg)
        m = {"goal_bearing_deg": r4(wrap_deg(off + ep.start_yaw_deg), 2),
             "start_yaw_deg": ep.start_yaw_deg, "off_bearing_deg": r4(off, 2)}
        return check(m, [grade("heading.off_bearing_abs_deg", abs(off))])

    # --------------------------------------------------------------------------- goal
    def _goal(self, ep, start_floor) -> dict:
        gx, gy, gz = ep.goal
        gfloor = self.floor_under(gx, gy, gz)
        approach = math.atan2(gy - ep.start[1], gx - ep.start[0])
        m = {"straight_m": r4(math.dist(ep.start[:2], ep.goal[:2]), 3),
             "straight_3d_m": r4(math.dist(ep.start, ep.goal), 3),
             "success_threshold_m": ep.success_threshold_m}
        rules = []
        if gfloor is None:
            rules.append(flag("goal_floor_found", "FAIL", "no walkable floor within "
                              f"[goal_z - {FLOOR_BELOW_M} m, goal_z + {FLOOR_ABOVE_M} m]"))
            return check(m, rules)
        gp = (gx, gy, gfloor["z"] + GOAL_PROBE_H_M)
        tr = self.trapped(gp)
        contained = tr["trapped"]
        m.update({
            "goal_floor_z": r4(gfloor["z"]), "goal_floor_prim": gfloor["prim"],
            "goal_floor_slope_deg": r4(gfloor["slope_deg"], 2),
            "goal_overhead_hits": gfloor["overhead"],
            "goal_floor_gap_m": r4(abs(gz - gfloor["z"])),
            "floor_dz_m": (r4(gfloor["z"] - start_floor["z"]) if start_floor else None),
            "goal_point_overlap": self.overlap_sphere(gp, 0.05),
            "goal_point_trapped": contained,
            "goal_point_probe": tr,
        })
        # Can the robot's footprint stand inside the success radius -- somewhere it can
        # also SEE the goal from? Line of sight at one of three heights rules out samples
        # in the room on the far side of a wall, and samples inside a solid (the two-sided
        # cast meets the solid's own wall), neither of which an overlap box can tell.
        samples = [(gx, gy, 0.0)]
        for f in GOAL_RING_FRACTIONS:
            r = f * ep.success_threshold_m
            for k in range(GOAL_RING_BEARINGS):
                a = 2.0 * math.pi * k / GOAL_RING_BEARINGS
                samples.append((gx + r * math.cos(a), gy + r * math.sin(a), r))
        free, hidden, blockers, nearest_free = 0, 0, {}, None
        goal_free = False
        for i, (sx, sy, r) in enumerate(samples):
            fl = self.floor_under(sx, sy, gfloor["z"])
            if fl is None or abs(fl["z"] - gfloor["z"]) > 0.5:
                blockers["<no floor / level change>"] = blockers.get(
                    "<no floor / level change>", 0) + 1
                continue
            c2, h2 = self.footprint_box(sx, sy, fl["z"], approach)
            hit = [p for p in self.overlap_box(c2, h2, approach) if p != fl["prim"]]
            if hit:
                for p in hit[:2]:
                    blockers[p] = blockers.get(p, 0) + 1
                continue
            if i == 0:
                visible = not contained
            else:
                visible = any(self.line_of_sight((sx, sy, fl["z"] + h),
                                                 (gx, gy, gfloor["z"] + h))
                              for h in GOAL_LOS_HEIGHTS_M)
            if not visible:
                hidden += 1
                continue
            free += 1
            goal_free = goal_free or i == 0
            nearest_free = r if nearest_free is None else min(nearest_free, r)
        m.update({"goal_standable": goal_free, "standable_samples": free,
                  "standable_but_hidden": hidden, "samples": len(samples),
                  "nearest_standable_m": nearest_free,
                  "standable_blockers": dict(sorted(blockers.items(),
                                                    key=lambda kv: -kv[1])[:5])})
        rules.append(grade("goal.goal_floor_gap_m", m["goal_floor_gap_m"]))
        if contained:
            what = (f"inside the closed solid {tr['inside'][:2]}" if tr["inside"] else
                    f"enclosed: {tr['escape_rays']}/{tr['of']} horizontal rays get "
                    f"{ESCAPE_RANGE_M} m out, walls {list(tr['walls'])[:2]}")
            rules.append(flag("goal_point_inside", "FAIL",
                              f"the goal point ({GOAL_PROBE_H_M} m over its floor) is {what}: "
                              "DynaNav's goal is not free space on this stage", True))
        else:
            rules.append(flag("goal_point_inside", "OK",
                              f"goal point in free space ({tr['escape_rays']}/{tr['of']} "
                              "rays escape)", False))
        if goal_free:
            rules.append(flag("goal_standable", "OK", "the footprint fits at the goal",
                              True))
        elif free:
            rules.append(flag("goal_standable", "WARN",
                              f"footprint blocked AT the goal ({m['goal_point_overlap'][:1]}"
                              f" / {list(m['standable_blockers'])[:2]}) but fits, in sight of "
                              f"the goal, at {free}/{len(samples)} points within the "
                              f"{ep.success_threshold_m} m radius, nearest "
                              f"{nearest_free:.2f} m", False))
        else:
            rules.append(flag("goal_standable", "FAIL",
                              f"the footprint fits NOWHERE in sight of the goal within the "
                              f"{ep.success_threshold_m} m radius ({hidden} fit but cannot "
                              f"see it; blockers {list(m['standable_blockers'])[:3]})",
                              False))
        if m["floor_dz_m"] is not None:
            rules.append(grade("goal.floor_dz_m", abs(m["floor_dz_m"])))
        return check(m, rules)

    def _physx_log(self, mark: int) -> dict:
        m = self.tap.summary(mark)
        m["boot_physx_errors"] = self.boot_physx_errors
        rules = []
        if m["physx_errors"]:
            rules.append(flag("physx_errors", "FAIL",
                              f"{m['physx_errors']} PhysX error(s) logged during this "
                              f"episode, e.g. {m['messages'][0]['message'][:120]!r}",
                              m["physx_errors"]))
        else:
            rules.append(flag("physx_errors", "OK", "no PhysX error during the episode", 0))
        if self.boot_physx_errors:
            rules.append(flag("boot_physx_errors", "FAIL",
                              f"{self.boot_physx_errors} PhysX error(s) while the stage "
                              "loaded -- they describe the scene every episode runs in "
                              "(see the report's boot_log)", self.boot_physx_errors))
        rules.append(flag("physx_warnings", "WARN" if m["physx_warnings"] else "OK",
                          f"{m['physx_warnings']} PhysX warning(s) during the episode",
                          m["physx_warnings"]))
        return check(m, rules)

    # ------------------------------------------------------------------------- camera
    def _camera(self, ep, frames_dir: Path, p, q) -> dict:
        np = self.np
        off = quat_rotate(q, self.cam_local)
        cam = (p[0] + off[0], p[1] + off[1], p[2] + off[2])
        seg = (cam[0] - p[0], cam[1] - p[1], cam[2] - p[2])
        seg_len = math.sqrt(sum(c * c for c in seg))
        # Two-sided: any surface between base_link and the lens, facing either way.
        ray = (self.first_hit(p, tuple(c / seg_len for c in seg), seg_len, both_sides=True)
               if seg_len > 1e-6 else None)
        sphere = self.overlap_sphere(cam, 0.02)
        tr = self.trapped(cam)
        m: dict = {"camera_world": [r4(v) for v in cam],
                   "occluder_on_mount_ray": ray.prim if ray else None,
                   "camera_inside_prims": sphere[:3],
                   "camera_trapped": tr["trapped"], "camera_probe": tr}
        rules = []
        if ray or sphere or tr["trapped"]:
            rules.append(flag("camera_inside", "FAIL",
                              f"camera centre inside / behind geometry: ray hit "
                              f"{m['occluder_on_mount_ray']}, sphere {sphere[:2]}, solid "
                              f"{tr['inside'][:2]}, {tr['escape_rays']}/{tr['of']} rays "
                              f"escape {ESCAPE_RANGE_M} m", True))
        else:
            rules.append(flag("camera_inside", "OK", "camera centre in free space", False))
        frame = self.camera.grab()
        if frame is None or getattr(frame, "size", 0) == 0:
            rules.append(flag("frame", "FAIL", "camera returned an empty frame"))
            return check(m, rules)
        f = frame.astype(np.float32)
        luma = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        m.update({"shape": list(frame.shape), "luma_mean": r4(float(luma.mean()), 2),
                  "luma_std": r4(float(luma.std()), 2),
                  "frac_black": r4(float((luma < 10).mean())),
                  "frac_white": r4(float((luma > 245).mean()))})
        try:
            from PIL import Image

            path = frames_dir / f"{ep.name}.jpg"
            Image.fromarray(frame).save(path, quality=90)
            m["frame"] = str(path)
        except Exception as exc:  # noqa: BLE001 -- the numbers still stand
            m["frame_error"] = str(exc)
        for k in ("luma_mean", "luma_std", "frac_black", "frac_white"):
            rules.append(grade(f"camera.{k}", m[k]))
        return check(m, rules)

    # -------------------------------------------------------------------------- lidar
    def _lidar(self, p, q) -> dict:
        np = self.np
        roll, pitch, yaw = quat_rpy(q)
        fwd = self.rn.NAV_LIDAR_FORWARD_M
        ox, oy = p[0] + fwd * math.cos(yaw), p[1] + fwd * math.sin(yaw)
        oz = p[2] + self.rn.NAV_LIDAR_HEIGHT_M
        fl = self.floor_under(ox, oy, p[2], above=FLOOR_ABOVE_ROBOT_M)
        expected = self.rest_z + self.rn.NAV_LIDAR_HEIGHT_M
        sw = self.SweepingLidar2D(mount_height_m=self.rn.NAV_LIDAR_HEIGHT_M,
                                  mount_forward_m=fwd)
        sw.reset()
        cast = sw.prime(p, yaw, 0.0)
        pts = sw.points_body(p, yaw)
        prims = sw.prims()
        self_hits = int(np.count_nonzero(getattr(sw, "_self", np.zeros(0, dtype=bool))))
        m: dict = {"mount_height_m": self.rn.NAV_LIDAR_HEIGHT_M,
                   "mount_forward_m": fwd,
                   "origin_world": [r4(ox), r4(oy), r4(oz)],
                   "origin_floor_z": r4(fl["z"]) if fl else None,
                   "origin_height_m": r4(oz - fl["z"]) if fl else None,
                   "expected_height_m": r4(expected),
                   "height_err_m": r4(abs(oz - fl["z"] - expected), 5) if fl else None,
                   "tilt_deg": r4(math.degrees(max(abs(roll), abs(pitch))), 3),
                   "rays": int(cast), "external_returns": int(len(pts)),
                   # Reported, not graded: 53 is lidar_2d's ANALYTIC column model (the
                   # lift column sliced at the mount height), fixed by the mount offsets
                   # and blind to pose -- a 99.6 deg tilt read 45 and passed.
                   "self_hits": self_hits}
        rules = [grade("lidar.height_err_m", m["height_err_m"]),
                 grade("lidar.tilt_deg", m["tilt_deg"])]
        if len(pts):
            d = np.hypot(pts[:, 0], pts[:, 1])
            b = np.arctan2(pts[:, 1], pts[:, 0])
            cl = np.array([d[i] - self.footprint_extent(float(b[i])) for i in range(len(d))])
            i = int(np.argmin(cl))
            j = int(np.argmin(d))
            m.update({"min_range_m": r4(float(d[j])), "min_range_prim": prims[j],
                      "min_clearance_m": r4(float(cl[i])),
                      "min_clearance_bearing_deg": r4(math.degrees(float(b[i])), 1),
                      "min_clearance_prim": prims[i]})
            rules.append(grade("lidar.min_clearance_m", m["min_clearance_m"]))
        else:
            rules.append(flag("external_returns", "OK", "no external return within the "
                              "C1's 12 m at the mount height -- open space, not a fault", 0))
        return check(m, rules)

    # -------------------------------------------------------------------------- drive
    def _drive(self, ep) -> dict:
        base, guard, lidar = self.base, self.guard, self.lidar_live
        tol = float(base.RESYNC_TOLERANCE_M)
        base.sync_from_sim()
        base.reset_stats()        # as run_episode does: faults/resyncs/z stats from here
        contact = getattr(self.runner, "contact", None)
        if contact is not None:
            contact.reset_episode()   # as run_episode: link poses, metric, clearance cache
        guard.interventions = 0
        if lidar is not None:
            lidar.reset()
            lidar.prime(base.position(), base.yaw, 0.0)
        p0, q0 = self.pose()
        prev = p0
        z_prev = p0[2]
        sim_yaw_prev = quat_rpy(q0)[2]
        sim_yaw_acc = 0.0
        int_yaw0 = base.yaw
        path = commanded = 0.0
        z_min = z_max = p0[2]
        z_step = track = ground = 0.0
        vmax = wmax = tilt_max = 0.0
        jumps: list[dict] = []
        blocked = translate_steps = 0
        guard_min = math.inf
        segs = []
        step = 0
        fault = None
        for dur, vx, om in DRIVE_SEGMENTS:
            n = int(round(dur / self.dt))
            seg_path0, seg_yaw0 = path, sim_yaw_acc
            for _ in range(n):
                t = step * self.dt
                position = base.position()
                xy = base._xy          # presence checked in __init__: no silent default
                d = math.hypot(position[0] - float(xy[0]), position[1] - float(xy[1]))
                track = max(track, d)
                if d > tol:
                    jumps.append({"t": round(t, 3), "offset_m": r4(d)})
                if lidar is not None:
                    lidar.step(position, base.yaw, self.dt, t)
                g = guard.check(position, base.yaw, vx, 0.0)
                if vx:
                    translate_steps += 1
                    blocked += bool(g.blocked)
                guard_min = min(guard_min, g.distance_m)
                base.apply(g.vx, g.vy, om, self.dt)
                self.kit.update()
                step += 1
                commanded += abs(vx) * self.dt
                # The runner's own fault detector, after every step, as run_episode calls
                # it: the first fault ends the probe, as it ends a real episode.
                fault = self.runner._physics_fault(step * self.dt)
                if fault is not None:
                    break
                p, q = self.pose()
                path += math.dist(p[:2], prev[:2])
                prev = p
                z_min, z_max = min(z_min, p[2]), max(z_max, p[2])
                z_step = max(z_step, abs(p[2] - z_prev))
                z_prev = p[2]
                r, pt, y = quat_rpy(q)
                sim_yaw_acc += wrap(y - sim_yaw_prev)
                sim_yaw_prev = y
                tilt_max = max(tilt_max, math.degrees(max(abs(r), abs(pt))))
                v, w = self.speeds()
                vmax, wmax = max(vmax, v), max(wmax, w)
                if step % GROUND_EVERY == 0:
                    fl = self.floor_under(p[0], p[1], p[2], above=FLOOR_ABOVE_ROBOT_M)
                    ground = max(ground, abs(p[2] - (fl["z"] + self.rest_z)) if fl
                                 else math.inf)
            segs.append({"duration_s": dur, "vx": vx, "omega": om,
                         "path_m": r4(path - seg_path0, 3),
                         "sim_yaw_deg": r4(math.degrees(sim_yaw_acc - seg_yaw0), 2)})
            if fault is not None:
                break
        base.stop()
        elapsed = step * self.dt
        cap = float(ep.max_speed_mps)
        p_end, _ = self.pose()
        yaw_err = math.degrees(abs(wrap((int_yaw0 + sim_yaw_acc) - base.yaw)))
        blocked_frac = blocked / translate_steps if translate_steps else 0.0
        m = {
            "elapsed_s": r4(elapsed, 3), "path_m": r4(path), "commanded_m": r4(commanded),
            "path_ratio": r4(path / commanded) if commanded > 0 else None,
            "blowup_bound_m": r4(2.0 * cap * elapsed, 2),
            "z_span_m": r4(z_max - z_min, 5), "z_step_max_m": r4(z_step, 6),
            "ground_err_m": r4(ground, 5),
            "max_lin_speed_mps": r4(vmax), "max_ang_speed_radps": r4(wmax),
            "track_err_max_m": r4(track, 4),
            "resyncs": int(base.resyncs), "base_faults": int(base.faults),
            "first_base_fault": base.first_fault,
            "offset_events": jumps[:10], "resync_tolerance_m": tol,
            "physics_fault": fault,
            "yaw_err_deg": r4(yaw_err, 3), "tilt_max_deg": r4(tilt_max, 3),
            "guard_interventions": guard.interventions, "guard_blocked_steps": blocked,
            "guard_blocked_frac": r4(blocked_frac, 3),
            "guard_nearest_m": r4(guard_min) if math.isfinite(guard_min) else None,
            "end_xy": [r4(p_end[0]), r4(p_end[1])],
            "end_offset_from_start_m": r4(math.dist(p_end[:2], ep.start[:2]), 3),
            "segments": segs,
            "guard_sensor": "lidar" if lidar is not None else "fan",
            "contact": (self.runner._contact_summary()
                        if callable(getattr(self.runner, "_contact_summary", None)) else None),
        }
        c_steps = int((m["contact"] or {}).get("contact_steps") or 0)
        c_frac = c_steps / step if step else 0.0
        rules = []
        if fault is not None:
            rules.append(flag("physics_fault", "FAIL",
                              f"run_navigation._physics_fault at t={fault['t']:.2f} s: "
                              f"{'+'.join(fault['kinds'])} (|v| {fault['speed_mps']:.2f} m/s, "
                              f"roll {fault['roll_deg']:.1f}, pitch {fault['pitch_deg']:.1f} "
                              "deg) -- a real episode ends here as a failure", fault))
        else:
            rules.append(flag("physics_fault", "OK", "no runner physics fault over the "
                              f"{elapsed:.1f} s probe", None))
        blown = path > 2.0 * cap * elapsed
        rules.append(flag("blowup", "FAIL" if blown else "OK",
                          f"path {path:.2f} m {'>' if blown else '<='} 2 x {cap:g} m/s x "
                          f"{elapsed:.1f} s = {2.0 * cap * elapsed:.1f} m "
                          "(summarize_runs.score)", blown))
        if m["path_ratio"] is not None:
            rules.append(grade("drive.path_ratio", m["path_ratio"]))
            stuck = m["path_ratio"] < MOBILITY_MIN_PATH_RATIO
            held = blocked_frac > MOBILITY_MAX_BLOCKED_FRAC
            pressed = c_frac > MOBILITY_MAX_BLOCKED_FRAC
            rules.append(flag("mobility", "WARN" if (stuck or held or pressed) else "OK",
                              f"path ratio {m['path_ratio']:.3g} (WARN < "
                              f"{MOBILITY_MIN_PATH_RATIO:g}), guard held "
                              f"{blocked}/{translate_steps} translate steps, contact clipped "
                              f"{c_steps}/{step} steps (WARN > "
                              f"{MOBILITY_MAX_BLOCKED_FRAC:.0%} of either)",
                              m["path_ratio"]))
        for k in ("ground_err_m", "z_step_max_m", "max_lin_speed_mps",
                  "max_ang_speed_radps", "resyncs", "track_err_max_m", "yaw_err_deg",
                  "tilt_max_deg"):
            rules.append(grade(f"drive.{k}", m[k]))
        return check(m, rules)


if __name__ == "__main__":
    raise SystemExit(main())
