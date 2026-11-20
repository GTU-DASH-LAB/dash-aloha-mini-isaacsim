#!/usr/bin/env python3
"""If AlohaMini drives the route a DynaNav episode asks for, does the physics hold?

README
======

`validate_episodes.py` checks each start pose and an 11 s drive probe next to it. It never
drives the ROUTE, and some physics failures only show up along one: the 13 cm traffic-island
curb on outdoor_umbrellas (CLAUDE.md, "Physics blow-ups: the fix") kicks the base to
2.5-2.8 m/s for single steps and leaves it riding 12-15 cm under the island's top surface.
This tool drives every episode's route with no policy. It takes the navmesh shortest path
from start to goal, follows it with the runner's own pure-pursuit controller at each
requested speed, and measures the physics along the way.

One process per environment (one nav stage), all its episodes in one Kit session, one JSON.

Usage (always through the GPU slot wrapper, always Isaac's python.sh, never piped):

    cd ~/robotics/dash-aloha-mini-isaacsim
    ISAAC_SLOTS=8 ISAAC_SLOT_MIN_FREE_MIB=8000 ISAAC_SLOT_TAG=route-hospital \\
      nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/route_drive.py \\
      --env hospital > /tmp/route_hospital.log 2>&1; echo "rc=$?"
    # outdoor_small needs ISAAC_SLOT_MIN_FREE_MIB=15000 (13 GiB per Kit). On a busy card
    # the wrapper gives up after ISAAC_SLOT_MEM_TIMEOUT_S (1800 s) with rc 75 and NOTHING
    # launched (an outdoor job waited 26 min on 2026-10-05): raise it, e.g. 5400.
    # Faster: split one env over processes with disjoint --only lists and --out files
    # (each Kit bakes its own navmesh; hospital ~6 GiB, outdoor ~13 GiB of GPU0 each).
    # Cost measured: indoor 34-60 ms per physics step (2-3.6x slower than real time),
    # outdoor 125-165 ms (7.5-10x): a 67 m outdoor route is 7.5 min at 1.5 m/s and
    # 22 min at 0.45.

    # the ladder names, a subset, two speeds, chosen output:
    ... nav/tools/route_drive.py --episodes nav/config/episodes.yaml \\
        --only hospital_down_hallway,hospital_vending_machine --speeds 1.5,0.6 \\
        --out nav/results/route_drive/cal_hospital.json

    --episodes PATH    episode table (default nav/config/episodes_full.yaml, all 85)
    --env NAME         hospital | office | warehouse | outdoor_small (inferred from --only)
    --only A,B         restrict to these episode names (comma list, repeatable)
    --out PATH         JSON (default nav/results/route_drive/<env>_<stamp>.json)
    --frames-dir DIR   camera_nav frames, 640x360, one every ~4 m of route plus the start
                       and the end (default <out stem>_frames/, git-ignored). LOOK at them.
    --speeds LIST      commanded cruise speeds, m/s, comma list (default 1.5, the policy's
                       cap). Each is clamped to the episode's own max_speed_mps, as the
                       runner's controller clamps it; the clamp is recorded.
    --lookahead-m F    pure-pursuit lookahead (default 1.0 = controllers.DYNANAV_LOOKAHEAD_M)
    --agent-radius-m F navmesh agent radius (default 0.40, never lower: the chassis is
                       0.31 m half-wide and swings 0.375 m turning in place)
    --settle-steps N   force base.settle with this cap (default: runner._settle(), i.e.
                       held until at rest, capped at run_navigation.SETTLE_STEPS, which is
                       what run_episode does)
    --navmesh-rigid-bodies obstacle|ignore
                       obstacle (default): rigid-body props block the route; ignore:
                       people.py's bake, which lets a route pass under a gurney
    --route-from JSON  drive a recorded run's xy `trace` instead of the navmesh path (one
                       episode only; --only must name it). Replays a policy's own route --
                       e.g. the one that rode onto outdoor_umbrellas' traffic island --
                       on the physics as it is on disk now.
    --guard MODE       observe (default): the runner's CollisionGuard is evaluated every step
                       with count=False and only reported. apply: its guarded velocity is
                       what gets driven, as in a real run.
    --continue-after-fault  keep driving after a runner fault (the runner stops there); every
                       fault is recorded, the verdict is still FAULT.
    --gpu N            Kit active/physics GPU (default 0; never CUDA_VISIBLE_DEVICES)

Exit status: python.sh maps every failure to 1 and Kit's fast shutdown can return 0 after a
Python exception, so READ THE JSON: `status` is "completed" (see `summary`) or "crashed"
(`fatal`). "running" with a dead `pid` is a native crash in `current_episode` /
`current_phase`. The log's last line says the same: "[route] STATUS ...".

HOW IT BOOTS. The same way validate_episodes.py does, by importing it rather than copying
it: `NavigationRunner.setup()` runs unchanged with the validator's stub policy, its log tap
records PhysX errors and warnings, and its `Context` supplies the scene queries. Every query
is an iterated `raycast_closest`; `raycast_all` segfaults Kit outdoors. Robot prims are
skipped. Spawn, drive and the fault test are imported as they are on disk when the run
starts: `runner._spawn_pose` (base_drive.ground_spawn_pose), `base.reset_to`,
`KinematicBase.apply` and `runner._physics_fault`. setup() also authors the session-layer
colliders for environment geometry that has none (scene_colliders.py, NAV_SCENE_COLLIDERS=0
skips it) and attaches the contact layer to the base (kinematic_contact.py, NAV_CONTACT=0
drives the bare base), so a route is driven against the same world, and with the same
contact response, as an episode. The md5 of base_drive.py, run_navigation.py,
physics_setup.py, kinematic_contact.py and scene_colliders.py goes into the JSON at the start
and at the end, since other workflows edit them.

THE NAVMESH comes from nav/sim/people.py's helpers. The settings are DynaNav's per-scene
bake (people.NAVMESH_PARAMS: height 1.8 m, step and slope per scene), with the agent radius
raised to --agent-radius-m (outdoor 0.20 -> 0.40; warehouse keeps its 0.50). Two changes
from people.py, both because this is a ROBOT route, not a pedestrian's:
  * rigid-body props are OBSTACLES (excludeRigidBodies=False). people.py excludes them so
    the robot is not baked in; that also dropped every dynamic prop, and the first
    calibration drove hospital_down_hallway's route under the SM_Gurney beside its goal
    (end frame: camera under the bed). The robot is excluded instead, by
    NavSchema.NavMeshExcludeAPI on every non-proxy prim under /World/Aloha (session
    layer). `navmesh.robot_hole_check` proves it: the closest navmesh point to the
    robot's authored pose is ~0, where a baked-in robot would leave a >= 0.7 m hole.
    --navmesh-rigid-bodies ignore restores people.py's behaviour.
  * the step height stays DynaNav's (0.40 m hospital/outdoor, 0.10 office, 0.01
    warehouse), so a route may climb a kerb the robot's 5 cm tyres cannot: that is
    exactly what the kerb measurements below are for.
Both navmesh caches are switched off before omni.anim.navigation.core starts (people.py
deviation 7: a cached bake reads 0.0 from every getter); `navmesh.radius_confirmed` says
the baked radius was read back. The warehouse NavMeshVolume is authored on the session
layer by PeopleSim._prepare_env (deviation 10), called only on a stage with no volume --
on the outdoor stage it would also deactivate the 200 baked characters the runner drives
among. The bake runs at the end of setup()'s stage load, before timeline.play(), through a
wrapper around physics_setup.raise_gpu_capacities (setup() imports it just before play); if
that hook never fires the bake runs after setup() and `navmesh.when` says so.
behavior.core is never enabled: there are no agents.

ROUTE. The closest navmesh point to the snapped spawn and to the floor under the goal; the
island ids of both; `query_shortest_path` between them; the route is the robot's spawn xy,
then the path's points (2-D; z is the physics' business). Verdicts before any driving:
  NO_PATH       either end has no navmesh point, the ends are on different islands, or
                no path is returned.
  PARTIAL_PATH  the path ends more than the episode's success_threshold_m (2-D) from the
                goal: a robot that drives it cannot score (people.py deviation 8). It is
                still driven, so its physics is measured too.
The drive ends within 0.5 m of the ROUTE's end; `closest_to_goal_m` is reported against the
success radius. A goal next to a landmark puts that last 0.5 m against it, so an envelope
overlap at the very end of a route (see `overlap_first[*].route_s_m`) is the goal's
geometry, not the corridor's.
--route-from RESULT_JSON drives a recorded run's xy `trace` instead (one episode; cut at
the first sample-to-sample jump > 2 m, i.e. before a blow-up's flight). That is how a
policy's own path -- e.g. one that rode onto a traffic island -- is replayed with the
fixed physics, independent of the commands that produced it.

DRIVE, per speed, from a fresh reset, as run_episode spawns, through the runner's and the
base's own methods as they are on disk (reset_like_runner): restore the arm/lift joints
(Context.restore_robot), `runner._spawn_pose(ep)`, `base.reset_to(start, yaw)`,
`runner._settle()` (held level until at rest, capped at run_navigation.SETTLE_STEPS;
`base.settle` / plain kit.update() steps on older code), `sync_from_sim()`,
`reset_stats()`, `runner.contact.reset_episode()`, and the counters
`_physics_fault` reads (`_overspeed_steps`, `_fall_steps`, `_fault_speed_mps` =
run_navigation.fault_speed_mps(cap)). Then every physics step does what the runner does:
`base.position()`, the command, `guard.check`, `base.apply(vx, vy=0, omega, PHYSICS_DT)`,
one `kit.update()`, then `runner._physics_fault` (which judges the step first). The command
is the runner's own controller (`controllers.make_controller("pursuit", v,
ep.max_yaw_rate_radps, lookahead_m=...)`), given the route ahead of the robot as body-frame
waypoints every 5 cm. The robot is projected onto the route, searching 0.5 m back to 3 m
forward of its last arc length so it never jumps to a parallel corridor. One addition the
runner has no need of: when the lookahead point is more than 45 deg off the nose the base
turns in place at max_yaw_rate_radps until it is within 15 deg (a start facing away from
its goal pivots; pure pursuit would otherwise circle at full speed). Ends: within 0.5 m of
the route's end; timeout = 2 x route length / v + 15 s; or the runner's physics fault,
which ends it as it ends an episode (--continue-after-fault records every fault instead).
Stepping is kit.update(), as the runner: PhysX simulate()+fetch_results() without the app
loop was measured by the outdoor replay (scratchpad/oreplay, upsway) NOT identical (0.57 m
max difference) and NOT faster (118 vs 121 s; outdoor is physics-bound), so there is no
fast mode. The JSON's `ms_per_step` splits wall time into update / control / scene queries.

MEASURED per drive (all off the SIM, after each step):
  faults            the runner's detector: kinds, time, |v|, roll, pitch, z of each fault.
  speed             max |v| (PhysX root linear velocity, what the fault rule reads), steps
                    over 3 m/s and the longest consecutive run of them (the spec's fixed
                    bar), the same against the runner's own fault speed for this episode
                    and its persistence (FAULT_SPEED_STEPS), and where each spike was (t,
                    route s, xy, vz, the floor prim under it). Below those bars, `spike_*`
                    (reported, not graded): steps more than 0.5 m/s above the commanded
                    speed, the longest run of them and the 8 fastest (one per 0.25 s, with
                    vz -- a kerb KICK reads vz > 0, a clamped DROP vz ~ -1.4 .. -2).
  tilt              max |roll| and |pitch|.
  z                 z span; the TYRE gap, graded: z - (highest support under the three
                    tyres + BASE_LINK_REST_Z_M), every 3rd step, each tyre's support the mean
                    of a 4-ray cross 5.5 cm from its axis (never one ray THROUGH the tyre
                    sphere, see tyre_gap). The drive holds the body level, so it rests on its
                    highest-supported tyre: ~0 on a slope, < 0 when a tyre is inside a kerb
                    or island top (`below_support_s`), > 0 hovering. Mean, p95 |gap|, min,
                    max. The CENTRE gap (floor under base_link) is reported, not graded: it
                    grows with slope. `buried` counts samples with a walkable surface 5-30 cm
                    ABOVE base_link: the robot riding under a kerb or island top.
  length            driven xy path / route arc length covered (pursuit cuts corners).
  cruise speed      over steps with |omega| < 0.05 rad/s at the full commanded speed after
                    15 steps of that: sim xy displacement / commanded, and PhysX |v| median.
                    The base pins x/y to its integrator every step, so the ratio is ~1 by
                    construction and checks the pinning (below 1 when the contact layer
                    clipped a step: the `contact` rule says so); PhysX |v| sits BELOW the command
                    (1.31 at 1.5, 0.27 at 0.45 m/s indoors): the sphere tyres drag.
  tracking          max |sim xy - base._xy|, resyncs, max cross-track error to the route.
  clearance         every 0.25 m of travel, 36 two-sided horizontal rays at each of the
                    validator's ring heights (0.10 m to the robot's top): nearest non-floor
                    hit minus the robot envelope along that bearing. Heights >= 0.30 m are
                    the BODY band (graded); 0.10/0.20 m is the kerb band (reported). Plus an
                    overlap query of the envelope box from 0.25 m above the floor to the
                    top: non-robot colliders INSIDE the chassis. A 0.30 m ring alone missed
                    a gurney deck the robot drove under. Below 0 the chassis is passing
                    THROUGH something: with NAV_CONTACT=0 nothing stops it; with the contact
                    layer on it is an overlap that layer allows -- a kerb under MAX_STEP_M
                    (low band), a spawn inside a collider it may only back out of, or
                    geometry with no collider (a skinned character, a guide prim).
  contact           the contact layer's record (runner._contact_summary()): steps it
                    clipped or refused, the longest pressed run and DynaNav's
                    physical_collision (>= 3.33 s), the first contact (t, prim named by its
                    environment source, envelope box) and steps per prim. None with
                    NAV_CONTACT=0.
  guard             steps the runner's guard would block / slow (fan of 7 rays, 0.6 m).
  frames            camera_nav at the start, every ~4 m of travel and at the end, 640x360.
  physx             PhysX ERROR/WARN lines logged during the drive (validator's LogTap).

VERDICTS (precedence ERROR > FAULT > NO_PATH > PARTIAL_PATH > WARN > OK; an episode is the
worst of its route and its drives):
  FAULT  the runner's physics fault fired; or no floor under any tyre at the start
         (`no_floor`: run_episode does not run such an episode, so it is not driven).
  WARN   any step over 3 m/s without a fault (spikes the persistence rule absorbed); tilt
         > 10 deg; tyre gap p95 > 0.02 m, max > 0.10 m (hovering) or min < -0.03 m (a tyre
         inside a surface); buried > 0; resyncs > 0; step anomalies > 0; track error > 0.05
         m; the contact layer clipped or refused any step (`contact`; the route runs into
         something the navmesh let through); cruise ratio outside [0.97, 1.03]; route end
         not reached in time; driven /
         covered outside [0.85, 1.10]; body clearance < 0 or the envelope box overlapping a
         collider (labelled when it happens only in the route's last END_ZONE_M, i.e. at
         the goal's own props); any PhysX ERROR during the drive; the spawn: the held
         settle LIFTED the robot > 5 mm off its snapped z (`spawn_embedded`: the snap
         under-read the floor), a spawn probe started inside a collider, a fault during
         the settle, or a settle that hit its cap without coming to rest.
  OK     none of the above.
Each drive lists every rule it evaluated with its value and the verdict it produced.

CALIBRATION (2026-10-05): see the bottom of this docstring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
# nav/tools/profile.py shadows the stdlib `profile`, which warp imports while Kit starts.
# Drop this directory from sys.path before anything else; validate_episodes is loaded by
# file path below, never by putting nav/tools back.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]

WATCHED = ("nav/sim/base_drive.py", "nav/sim/run_navigation.py", "nav/sim/physics_setup.py",
           "nav/sim/kinematic_contact.py", "nav/sim/scene_colliders.py")
DEFAULT_EPISODES = REPO / "nav" / "config" / "episodes_full.yaml"

ARRIVE_M = 0.5               # the drive ends within this of the route's end (2-D)
WAYPOINT_DS_M = 0.05         # body-frame waypoint spacing handed to the controller
WAYPOINT_MIN_POINTS = 8      # pad the tail with the end point so the controller's [2, n-3]
SEARCH_BACK_M = 0.5          # projection window around the last arc length
SEARCH_AHEAD_M = 3.0
PIVOT_ENTER_DEG = 45.0       # lookahead this far off the nose -> turn in place
PIVOT_EXIT_DEG = 15.0
TIMEOUT_PAD_S = 15.0
OVER_SPEED_MPS = 3.0         # the spec's fixed bar; the runner's own fault speed is separate
SPIKE_OVER_CMD_MPS = 0.5     # |v| this far above the commanded speed is a reported spike
SPIKE_GAP_S = 0.25           # spike_top keeps the highest step per window of this length
GROUND_EVERY = 3             # steps between floor samples
FLOOR_ABOVE_M = 0.05         # a floor is at most this above base_link (validate_episodes)
BURY_ABOVE_M = 0.30          # walkable surface in (FLOOR_ABOVE_M, this] above = buried
FLOOR_MIN_NZ = 0.7
TYRE_PROBE_M = 0.055         # tyre-support rays this far from the tyre axis (sphere r 0.05)
RING_EVERY_M = 0.25
RING_BEARINGS = 36
RING_BODY_MIN_H_M = 0.30     # ring heights >= this are the BODY band (graded); below: kerbs
RING_RANGE_M = 3.0
BOX_Z_LO_M = 0.25            # envelope overlap box starts this far above the floor (kerbs
                             # up to ~15 cm stay out of it; they are the speed/buried rules')
FRAME_EVERY_M = 4.0
END_ZONE_M = 1.0             # an overlap only this close to the route's end is labelled
FRAME_SIZE = (640, 360)
TRACE_EVERY = 15             # steps between trace samples (0.25 s)
CRUISE_OMEGA = 0.05          # rad/s: below this a step is "straight"
CRUISE_WARMUP = 15           # consecutive straight full-speed steps before one counts
# Rule bounds, (name: bound); see VERDICTS in the docstring.
WARN_TILT_DEG = 10.0
SPAWN_LIFT_WARN_M = 0.005    # settle z - snapped z above this: the spawn was embedded
SPAWN_CLEARANCE_NOTE = "~3 mm DOWN (base_drive.SPAWN_CLEARANCE_M)"
WARN_GAP_P95_M = 0.02
WARN_GAP_MAX_M = 0.10
WARN_GAP_MIN_M = -0.03
WARN_TRACK_M = 0.05
WARN_CRUISE = (0.97, 1.03)
WARN_DRIVEN = (0.85, 1.10)
BOOT_WATCHDOG_S = 900        # boot longer than this: dump the Python stacks and exit(1)
VERDICT_ORDER = ["OK", "WARN", "PARTIAL_PATH", "NO_PATH", "FAULT", "ERROR"]


def md5s() -> dict:
    out = {}
    for rel in WATCHED:
        try:
            out[rel] = hashlib.md5((REPO / rel).read_bytes()).hexdigest()
        except OSError as exc:
            out[rel] = f"unreadable: {exc}"
    return out


def worst(verdicts) -> str:
    vs = [v for v in verdicts if v]
    return max(vs, key=VERDICT_ORDER.index) if vs else "OK"


def r4(v, n: int = 4):
    return None if v is None else round(float(v), n)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------------ pure geometry
class Route:
    """A 2-D polyline with arc-length parametrisation. No Kit; unit-tested in --selftest."""

    def __init__(self, pts):
        import numpy as np

        clean = []
        for p in pts:
            p = (float(p[0]), float(p[1]))
            if not clean or math.dist(clean[-1], p) > 1e-3:
                clean.append(p)
        if len(clean) == 1:
            clean.append(clean[0])
        self.np = np
        self.p = np.array(clean, dtype=np.float64)
        d = np.diff(self.p, axis=0)
        self.seglen = np.hypot(d[:, 0], d[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.length = float(self.cum[-1])

    def points_at(self, s):
        """Route points at arc lengths s (array), clipped to [0, length]."""
        np = self.np
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        return np.stack([np.interp(s, self.cum, self.p[:, 0]),
                         np.interp(s, self.cum, self.p[:, 1])], axis=-1)

    def point_at(self, s: float):
        return self.points_at([s])[0]

    def project(self, xy, s_lo: float, s_hi: float) -> tuple[float, float]:
        """(s, distance) of the closest route point to xy with s in [s_lo, s_hi]."""
        np = self.np
        x = np.asarray(xy[:2], dtype=np.float64)
        s_lo, s_hi = max(0.0, s_lo), min(self.length, s_hi)
        i0 = max(0, int(np.searchsorted(self.cum, s_lo, side="right")) - 1)
        i1 = min(len(self.p) - 2, int(np.searchsorted(self.cum, s_hi, side="right")) - 1)
        a, b = self.p[i0:i1 + 1], self.p[i0 + 1:i1 + 2]
        L = self.seglen[i0:i1 + 1]
        ab = b - a
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(L > 0, np.einsum("ij,ij->i", x - a, ab) / (L * L), 0.0)
        s = np.clip(self.cum[i0:i1 + 1] + np.clip(t, 0.0, 1.0) * L, s_lo, s_hi)
        d = np.hypot(*(self.points_at(s) - x).T)
        k = int(np.argmin(d))
        return float(s[k]), float(d[k])

    def sample(self, s0: float, ds: float, horizon: float):
        np = self.np
        return self.points_at(s0 + np.arange(0.0, horizon + 0.5 * ds, ds))

    def heading_change_deg(self) -> float:
        """Total absolute turning along the route resampled every 1 m, a cheap 'how twisty'
        number (raw navmesh points jitter by mm, which summed to 274 deg on a straight
        18 m route)."""
        np = self.np
        q = self.points_at(np.append(np.arange(0.0, self.length, 1.0), self.length))
        d = np.diff(q, axis=0)
        d = d[np.hypot(d[:, 0], d[:, 1]) > 1e-6]
        h = np.arctan2(d[:, 1], d[:, 0])
        return float(np.degrees(np.sum(np.abs([wrap(b - a) for a, b in zip(h[:-1], h[1:])]))))


def body_waypoints(route: Route, s: float, xy, yaw: float, horizon: float):
    """Route ahead of arc length s, every WAYPOINT_DS_M, in the robot's body frame (x fwd)."""
    np = route.np
    pts = route.sample(s, WAYPOINT_DS_M, horizon)
    if len(pts) < WAYPOINT_MIN_POINTS:
        pts = np.vstack([pts, np.repeat(pts[-1:], WAYPOINT_MIN_POINTS - len(pts), axis=0)])
    rel = pts - np.asarray(xy[:2], dtype=np.float64)
    c, sn = math.cos(yaw), math.sin(yaw)
    return np.stack([c * rel[:, 0] + sn * rel[:, 1], -sn * rel[:, 0] + c * rel[:, 1]], axis=1)


def longest_run(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def top_spikes(events: list[dict], n: int = 8) -> list[dict]:
    """The n fastest spike steps, at most one per SPIKE_GAP_S window (a 5-step kick is one
    event, not five), fastest first."""
    out: list[dict] = []
    for e in sorted(events, key=lambda e: -e["speed_mps"]):
        if all(abs(e["t"] - o["t"]) >= SPIKE_GAP_S for o in out):
            out.append(e)
            if len(out) >= n:
                break
    return out


def selftest() -> int:
    """Plain python3, no Kit: the geometry and the controller hook-up."""
    import numpy as np

    sys.path.insert(0, str(REPO / "nav" / "sim"))
    from controllers import _lookahead_point, make_controller

    r = Route([(0, 0), (10, 0), (10, 10)])
    assert abs(r.length - 20.0) < 1e-9
    assert np.allclose(r.point_at(15.0), (10, 5))
    s, d = r.project((5, 1), 0, 20)
    assert abs(s - 5) < 1e-9 and abs(d - 1) < 1e-9
    # a window that excludes the near leg must not jump to it
    s, d = r.project((9, 9), 0, 3)
    assert s <= 3.0 + 1e-9
    wp = body_waypoints(r, 0.0, (0, 0), 0.0, 3.0)
    assert wp.shape[1] == 2 and abs(wp[1, 0] - WAYPOINT_DS_M) < 1e-9 and abs(wp[:, 1]).max() < 1e-9
    wp = body_waypoints(r, 0.0, (0, 0), math.pi / 2, 3.0)   # facing +y: route is to the right
    assert wp[-1, 1] < -2.9
    tail = body_waypoints(r, 19.99, (10, 9.99), math.pi / 2, 3.0)
    assert len(tail) >= WAYPOINT_MIN_POINTS
    assert longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    ev = [{"t": 1.0, "speed_mps": 2.0}, {"t": 1.1, "speed_mps": 2.5}, {"t": 3.0, "speed_mps": 2.2}]
    assert [e["t"] for e in top_spikes(ev)] == [1.1, 3.0]
    assert worst(["OK", "PARTIAL_PATH", "WARN"]) == "PARTIAL_PATH"
    assert worst(["NO_PATH", "FAULT"]) == "FAULT"
    # closed-loop kinematic simulation of the drive law on an L route, unicycle model
    ctrl = make_controller("pursuit", 1.5, 1.2, lookahead_m=1.0)
    ctrl.reset()
    x, y, yaw, sp, dt = 0.0, 0.0, 0.0, 0.0, 1 / 60
    driven, pivot, worst_xte = 0.0, False, 0.0
    for _ in range(int(60 / dt)):
        sp, xte = r.project((x, y), sp - SEARCH_BACK_M, sp + SEARCH_AHEAD_M)
        worst_xte = max(worst_xte, xte)
        if math.hypot(x - 10, y - 10) <= ARRIVE_M and sp >= r.length - ARRIVE_M - 1.0:
            break
        wp = body_waypoints(r, sp, (x, y), yaw, 3.0)
        cmd = ctrl(wp)
        xl, yl, _ = _lookahead_point(wp, 1.0)
        err = math.degrees(math.atan2(yl, xl))
        pivot = abs(err) > PIVOT_EXIT_DEG if pivot else abs(err) > PIVOT_ENTER_DEG
        vx, om = (0.0, math.copysign(1.2, err)) if pivot else (cmd.vx, cmd.omega)
        nx, ny = x + vx * math.cos(yaw) * dt, y + vx * math.sin(yaw) * dt
        driven += math.hypot(nx - x, ny - y)
        x, y, yaw = nx, ny, yaw + om * dt
    else:
        raise AssertionError("pursuit never reached the end of the L route")
    print(f"selftest: L route 20 m, driven {driven:.2f} m, worst cross-track {worst_xte:.2f} m")
    assert 18.0 < driven < 20.5 and worst_xte < 1.0
    print("selftest OK")
    return 0


def _jsonable(v):
    """A record from shared code, made JSON-safe (numpy scalars, tuples, NaN)."""
    try:
        return json.loads(json.dumps(v, default=lambda o: o.item() if hasattr(o, "item")
                                     else str(o)))
    except (TypeError, ValueError):
        return str(v)


def _ring_at(best) -> dict:
    cl, bdeg, hgt, prim, dist, s = best
    return {"bearing_deg": r4(bdeg, 1) if bdeg is not None else None, "height_m": hgt,
            "prim": prim, "dist_m": r4(dist, 3) if math.isfinite(dist) else None,
            "route_s_m": s}


# ------------------------------------------------------------------------------ navmesh
class NavMesh:
    """Bake + queries through nav/sim/people.py's helpers. Created before Kit starts."""

    def __init__(self, env: str, radius_m: float, rigid_bodies: str = "obstacle",
                 max_step_cm: float | None = None, max_slope_deg: float | None = None):
        import people as P

        self.P = P
        self.env = env
        self.radius_m = radius_m
        params = P.NAVMESH_PARAMS[env]
        cfg = {k: v for k, v in params.items() if k != "provenance"}
        cfg["agentMinRadius"] = max(float(cfg["agentMinRadius"]), 100.0 * radius_m)
        cfg["agentMaxRadius"] = max(50.0, cfg["agentMinRadius"])
        cfg["autoRebakeOnChanges"] = False
        # DynaNav's step height (40 cm outdoors) is a walker's; AlohaMini's tyres are 5 cm and
        # the contact layer refuses anything over kinematic_contact.MAX_STEP_M (4.5 cm). With
        # --agent-max-step-cm the bake asks whether a route exists for the ROBOT.
        if max_step_cm is not None:
            cfg["agentMaxStepHeight"] = float(max_step_cm)
        if max_slope_deg is not None:     # the contact layer passes ramps <= 20 deg (CLAUDE.md)
            cfg["agentMaxFloorSlope"] = float(max_slope_deg)
        # people.py bakes with excludeRigidBodies=True so the robot is not carved into the
        # mesh. For a ROBOT route that also drops every dynamic prop: the first calibration
        # drove hospital_down_hallway's route straight under the SM_Gurney its goal sits on
        # (end frame: camera under the bed frame). Here rigid props stay obstacles and the
        # robot is excluded explicitly with NavMeshExcludeAPI (see bake()).
        cfg["excludeRigidBodies"] = rigid_bodies == "ignore"
        self.cfg = cfg
        self.info: dict = {"settings_cm": cfg, "provenance": params["provenance"],
                           "max_step_overridden_cm": max_step_cm,
                           "max_slope_overridden_deg": max_slope_deg,
                           "radius_raised_from_cm": params["agentMinRadius"],
                           "rigid_bodies": rigid_bodies, "when": None, "error": None}
        self.ppl = None
        self.navmesh = None

    def _settings(self) -> None:
        import carb

        s = carb.settings.get_settings()
        for k in ("enabled", "enableUjitso"):
            s.set(f"{self.P.NAV_SETTINGS}/cache/{k}", False)
        for k, v in self.cfg.items():
            s.set(f"{self.P.NAV_SETTINGS}/config/{k}", v)

    def enable_extension(self) -> None:
        """Right after SimulationApp starts, before the stage opens: caches off first."""
        import omni.kit.app

        self._settings()
        app = omni.kit.app.get_app()
        if not app.get_extension_manager().set_extension_enabled_immediate(
                "omni.anim.navigation.core", True):
            raise RuntimeError("could not enable omni.anim.navigation.core")
        app.update()
        self.info["extension_enabled"] = "after SimulationApp(), before open_stage"

    def bake(self, stage, when: str) -> None:
        import NavSchema
        import omni.anim.navigation.core as nav

        t0 = time.time()
        self._settings()
        self.ppl = self.P.PeopleSim(stage, self.env, 0,
                                    log=lambda m: print(f"[route] navmesh {m}", flush=True))
        vols = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(NavSchema.NavMeshVolume)]
        self.info["traverse_s"] = round(time.time() - t0, 1)
        if not vols:
            # warehouse: the volume lives outside the referenced defaultPrim (deviation 10)
            prev = self.ppl._session()
            try:
                self.ppl._prepare_env()
            finally:
                stage.SetEditTarget(prev)
            vols = list(self.ppl.build_info.get("navmesh_volumes", []))
            self.info["authored_volume"] = self.ppl.build_info.get("authored_navmesh_volume")
        self.info["volumes"] = vols
        self.info["robot_excluded"] = self._exclude_robot(stage, NavSchema)
        self.ppl.inav = nav.acquire_interface()
        self.ppl._bake("route_bake")          # raises if the bake failed
        self.navmesh = self.ppl.navmesh
        self.info["bake"] = self.ppl.build_info.get("route_bake")
        self.info["warnings"] = self.ppl.build_info.get("warnings", [])
        self.info["when"] = when
        got = self.info["bake"].get("get_agent_min_radius")
        self.info["radius_confirmed"] = (isinstance(got, float)
                                         and abs(got - self.cfg["agentMinRadius"] / 100.0) < 1e-3)
        # Robot hole check: with the robot excluded, the navmesh is continuous where the
        # stage authors the robot. A hole there means the exclusion did not take.
        from pxr import Usd, UsdGeom

        bl = stage.GetPrimAtPath("/World/Aloha/Geometry/base_link")
        if bl and bl.IsValid():
            t = UsdGeom.Xformable(bl).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).ExtractTranslation()
            pt, _ = self.closest(t[0], t[1], t[2] - 0.0078)
            self.info["robot_hole_check"] = {
                "authored_xy": [r4(t[0], 3), r4(t[1], 3)],
                "closest_navmesh_m": r4(math.dist(pt[:2], (t[0], t[1])), 3) if pt else None}
        print(f"[route] navmesh baked ({when}) in {self.info['bake'].get('s')} s: "
              f"{self.info['bake']}  volumes={vols}  robot hole check "
              f"{self.info.get('robot_hole_check')}", flush=True)

    def _exclude_robot(self, stage, NavSchema) -> dict:
        """NavMeshExcludeAPI on every robot prim that can take an opinion (the root, its
        links, the instance roots -- not the instance proxies under them), session layer."""
        from pxr import Usd

        root = stage.GetPrimAtPath("/World/Aloha")
        if not root or not root.IsValid():
            return {"applied": 0, "note": "/World/Aloha not found"}
        n = 0
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for prim in Usd.PrimRange(root):
                if prim.IsInstanceProxy():
                    continue
                NavSchema.NavMeshExcludeAPI.Apply(prim)
                n += 1
        return {"applied": n, "root": "/World/Aloha", "layer": "session"}

    def closest(self, x, y, z):
        q = self.navmesh.query_closest_point(target=self.ppl._f3(x, y, z))
        pt, island = (q[0], q[1]) if isinstance(q, (tuple, list)) else (q, None)
        if pt is None:
            return None, None
        return (float(pt[0]), float(pt[1]), float(pt[2])), (None if island is None
                                                            else int(island))

    def path(self, a, b):
        return self.ppl._path(a, b)   # (points, length_m, end_gap_m) or None


# ------------------------------------------------------------------------------ the tool
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Drive each episode's navmesh route with the "
                                 "runner's controller and measure the physics. See the "
                                 "module docstring.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--env", default=None)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--speeds", default="1.5")
    ap.add_argument("--lookahead-m", type=float, default=1.0)
    ap.add_argument("--agent-radius-m", type=float, default=0.40)
    ap.add_argument("--navmesh-rigid-bodies", choices=("obstacle", "ignore"),
                    default="obstacle", help="obstacle (default): dynamic props block the "
                    "path, the robot is excluded by NavMeshExcludeAPI; ignore: people.py's "
                    "excludeRigidBodies=True")
    ap.add_argument("--settle-steps", type=int, default=None,
                    help="force base.settle with this cap; default runner._settle() (held "
                         "until at rest, capped at run_navigation.SETTLE_STEPS), as "
                         "run_episode")
    ap.add_argument("--route-from", type=Path, default=None, metavar="RESULT_JSON",
                    help="drive the xy trace of a recorded run (nav/results/*.json, its "
                         "`trace`) instead of the navmesh path; the trace is cut at the "
                         "first sample-to-sample jump > 2 m (a blow-up). One episode only.")
    ap.add_argument("--guard", choices=("observe", "apply"), default="observe")
    ap.add_argument("--continue-after-fault", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--agent-max-step-cm", type=float, default=None,
                    help="navmesh agentMaxStepHeight in cm (default: DynaNav's per-scene "
                         "value, people.NAVMESH_PARAMS -- 40 outdoors). 4.5 = what the "
                         "contact layer lets AlohaMini climb")
    ap.add_argument("--agent-max-slope-deg", type=float, default=None,
                    help="navmesh agentMaxFloorSlope in degrees (default DynaNav's, 45)")
    ap.add_argument("--plan-only", action="store_true",
                    help="bake and query each route, drive nothing (feasibility check)")
    ap.add_argument("--selftest", action="store_true", help="plain python3: unit checks, no Kit")
    a = ap.parse_args()
    a.speed_list = [float(v) for v in a.speeds.split(",") if v.strip()]
    if not a.speed_list or min(a.speed_list) <= 0:
        ap.error("--speeds needs positive values")
    if a.agent_radius_m < 0.40:
        ap.error("--agent-radius-m below 0.40 m lets the path squeeze a 0.31 m half-wide "
                 "chassis (0.375 m swing) through gaps it cannot pass")
    return a


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_episodes",
                                                  TOOLS / "validate_episodes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_episodes"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    md5_start = md5s()
    # A boot that hangs must not hold a GPU slot forever. Two of the six calibration launches
    # froze inside Kit's native `app.startup()` (simulation_app.py:576) before Kit printed a
    # line, 0% CPU, futex_wait -- with the user's inotify instances at the 128 limit. So:
    # `kill -USR1 <python3 pid>` prints every thread's Python stack at any time, and a boot
    # (Kit start + stage load) still running after BOOT_WATCHDOG_S prints it and EXITS (1).
    # The JSON then stays at status "running", phase "boot", with `boot_deadline_s` set:
    # relaunch the same command.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(BOOT_WATCHDOG_S, exit=True)
    ve = load_validator()          # also puts nav/sim, scripts, ... on sys.path
    env, episodes = ve.select_episodes(args)
    if args.route_from is not None and len(episodes) != 1:
        raise SystemExit("--route-from drives ONE recorded run: pass --only <its episode>")
    raw = (__import__("yaml").safe_load(args.episodes.read_text()) or {}).get("episodes", {})
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (REPO / "nav" / "results" / "route_drive" / f"{env}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.frames_dir or out.with_name(out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.resolve().is_relative_to(REPO):
        (frames_dir / ".gitignore").write_text("*\n")
    print(f"[route] {len(episodes)} episode(s) of env {env} from {args.episodes}; speeds "
          f"{args.speed_list}; report -> {out}; frames -> {frames_dir}", flush=True)

    nm = NavMesh(env, args.agent_radius_m, args.navmesh_rigid_bodies, args.agent_max_step_cm,
                 args.agent_max_slope_deg)
    report: dict = {
        "tool": "nav/tools/route_drive.py", "env": env, "episodes_file": str(args.episodes),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": ve.git_sha(), "started": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(), "md5_start": md5_start, "status": "running",
        "current_episode": None, "current_phase": "boot", "navmesh": nm.info,
        "boot_deadline_s": BOOT_WATCHDOG_S,
        "episodes": [],
    }
    ve._write(out, report)

    # ---- boot: the validator's way, plus two hooks for the navmesh ----------------------
    import isaacsim

    tap = ve.LogTap()
    ve._tap_simulation_app(tap)
    tapped = isaacsim.SimulationApp

    class _RouteSimulationApp(tapped):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            try:
                nm.enable_extension()
            except Exception:  # noqa: BLE001 -- reported; the bake will then fail loudly
                nm.info["error"] = traceback.format_exc()

    isaacsim.SimulationApp = _RouteSimulationApp

    import physics_setup

    hooked = callable(getattr(physics_setup, "raise_gpu_capacities", None))
    if hooked:
        _orig_raise = physics_setup.raise_gpu_capacities

        def _raise_then_bake(stage, *a, **k):
            lines = _orig_raise(stage, *a, **k)
            try:
                nm.bake(stage, "before_play (hook on physics_setup.raise_gpu_capacities)")
            except Exception:  # noqa: BLE001 -- retried after setup, reported either way
                nm.info["error"] = traceback.format_exc()
                print(nm.info["error"], flush=True)
            return lines

        physics_setup.raise_gpu_capacities = _raise_then_bake
    nm.info["hook_installed"] = hooked

    import run_navigation as rn

    report["fault_rule"] = {k: getattr(rn, k, None) for k in (
        "FAULT_SPEED_MPS", "FAULT_SPEED_STEPS", "FAULT_TILT_DEG", "PHYSICS_DT")}
    t_boot = time.time()
    runner = rn.NavigationRunner(episode=episodes[0], controller_name="pursuit",
                                 scratch_dir=frames_dir / ".scratch", headless=True,
                                 sim_gpu=args.gpu)
    runner.policy = ve._OfflinePolicy()
    report["stage"] = str(runner.scene_path)
    code = 1
    try:
        runner.setup()
        tap.install()
        faulthandler.cancel_dump_traceback_later()
        faulthandler.enable(file=sys.stderr, all_threads=False)
        if nm.navmesh is None:
            print("[route] navmesh not baked before play; baking now", flush=True)
            import omni.usd

            nm.info["error_before_play"] = nm.info.get("error")
            nm.info["error"] = None
            nm.bake(omni.usd.get_context().get_stage(), "after_play (hook did not bake)")
        report["boot_s"] = round(time.time() - t_boot, 1)
        report["boot_log"] = tap.summary(0)
        ctx = ve.Context(runner, rn)
        ctx.tap = tap
        report["robot_envelope"] = ctx.envelope
        report["gpu_mib_after_boot"] = ve.gpu_mib_of(os.getpid())
        report["base"] = {k: getattr(runner.base, k, None) for k in (
            "RESYNC_TOLERANCE_M", "VZ_MIN_MPS", "VZ_MAX_MPS")}
        drv = RouteDriver(ctx, runner, rn, nm, args, frames_dir)

        def on_phase(phase: str) -> None:
            report["current_phase"] = phase
            ve._write(out, report)

        drv.on_phase = on_phase
        print(f"[route] stage ready in {report['boot_s']} s; boot log "
              f"{report['boot_log']['physx_errors']} PhysX error(s)", flush=True)
        for i, ep in enumerate(episodes, 1):
            t0 = time.time()
            print(f"\n[route] ({i}/{len(episodes)}) {ep.name}", flush=True)
            report["current_episode"] = ep.name
            on_phase("route")
            try:
                rec = drv.episode(ep)
            except Exception as exc:  # noqa: BLE001 -- recorded, the next episode runs
                rec = {"name": ep.name, "verdict": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
                print(rec["traceback"], flush=True)
            meta = raw.get(ep.name, {}) or {}
            rec["dynanav"] = {k: meta[k] for k in (
                "dynanav_episode", "num_people", "off_bearing_deg", "ours", "dynanav_scored",
                "dynanav_success", "dynanav_spl") if k in meta}
            rec["wall_s"] = round(time.time() - t0, 1)
            report["episodes"].append(rec)
            print(f"[route] {ep.name}: {rec['verdict']}  " + "  ".join(
                f"v{d['speed_cmd_mps']:g}={d['verdict']}" for d in rec.get("drives", [])),
                flush=True)
            ve._write(out, report)
        verdicts = [e["verdict"] for e in report["episodes"]]
        report["summary"] = {v: verdicts.count(v) for v in VERDICT_ORDER}
        report["ray_casts"] = ctx.casts
        report["gpu_mib_end"] = ve.gpu_mib_of(os.getpid())
        report["current_episode"] = None
        report["current_phase"] = "done"
        report["status"] = "completed"
        # 2 = an episode raised (a harness problem); verdicts themselves never set it.
        # python.sh turns any non-zero into 1, so the JSON is still what to read.
        code = 2 if report["summary"].get("ERROR") else 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        report["fatal"] = traceback.format_exc()
        report["status"] = "crashed"
    report["md5_end"] = md5s()
    report["md5_changed_during_run"] = sorted(
        k for k in WATCHED if report["md5_end"].get(k) != md5_start.get(k))
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    ve._write(out, report)
    print_table(report)
    print(f"[route] STATUS {report['status']}  summary={report.get('summary')}  "
          f"md5_changed={report['md5_changed_during_run']}  ({out})", flush=True)
    sys.stdout.flush()
    # Last on purpose: nothing after SimulationApp.close() executes (CLAUDE.md).
    if runner.kit is not None:
        runner.kit.close(exit_code=code)
    return code


def _f(v, n: int = 2) -> str:
    """Table cell: a number rounded, or '-' for a value that was not measured."""
    if v is None:
        return "-"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return f"{round(float(v), n):g}"


def print_table(report: dict) -> None:
    print("\n" + "=" * 118)
    print(f"{'episode':30} {'verdict':12} {'v':>4} {'route':>6} {'drove':>6} {'vmax':>5} "
          f"{'>3':>3} {'tilt':>5} {'gap95':>6} {'bur':>4} {'clear':>6} {'goal':>6}  rules")
    print("-" * 118)
    for e in report.get("episodes", []):
        route = e.get("route", {})
        drives = e.get("drives") or [None]
        for d in drives:
            if d is None:
                print(f"{e['name']:30} {e['verdict']:12} {'-':>4} "
                      f"{_f(route.get('length_m'), 1):>6}  "
                      + (e.get("error") or route.get("rule", "")))
                continue
            m = d.get("metrics", {})
            bad = [r["rule"] for r in d.get("rules", []) if r["verdict"] != "OK"]
            if d.get("error"):
                bad.insert(0, d["error"])
            print(f"{e['name']:30} {d['verdict']:12} {_f(d.get('speed_cmd_mps')):>4} "
                  f"{_f(route.get('length_m'), 1):>6} {_f(m.get('driven_m'), 1):>6} "
                  f"{_f(m.get('max_speed_mps')):>5} {_f(m.get('overspeed_steps')):>3} "
                  f"{_f(m.get('tilt_max_deg'), 1):>5} {_f(m.get('gap_abs_p95_m'), 3):>6} "
                  f"{_f(m.get('buried_samples')):>4} "
                  f"{_f(m.get('route_min_clearance_m')):>6} "
                  f"{_f(m.get('closest_to_goal_m')):>6}  " + " | ".join(bad[:2]))
    print("=" * 118)
    if report.get("status") == "running":
        print("status 'running': if pid", report.get("pid"), "is gone it crashed natively in",
              report.get("current_episode"), report.get("current_phase"))
    print("summary:", report.get("summary"))


class RouteDriver:
    def __init__(self, ctx, runner, rn, nm: NavMesh, args, frames_dir: Path):
        import numpy as np

        from controllers import Command, _lookahead_point, make_controller

        self.np = np
        self.ctx, self.runner, self.rn, self.nm, self.args = ctx, runner, rn, nm, args
        self.base, self.kit, self.guard = runner.base, runner.kit, runner.guard
        self.dt = rn.PHYSICS_DT
        self.rest_z = ctx.rest_z
        self.frames_dir = frames_dir
        self.Command, self._lookahead_point, self.make_controller = (
            Command, _lookahead_point, make_controller)
        self.settle_steps = (args.settle_steps if args.settle_steps is not None
                             else int(getattr(rn, "SETTLE_STEPS", 30)))
        self.heights = ctx.ring_heights()
        from alohamini1_specs import WHEEL_ANGLES_RAD, WHEEL_MOUNT_RADIUS_M

        self.wheel_angles = list(WHEEL_ANGLES_RAD)
        self.wheel_r = float(WHEEL_MOUNT_RADIUS_M)
        self.on_phase = lambda phase: None

    # ------------------------------------------------------------------ scene samples
    def ground(self, x, y, z) -> dict:
        """Floor under (x, y) and any walkable surface 5-30 cm ABOVE base_link (buried)."""
        hits = self.ctx.ray_hits((x, y, z + 2.0), (0.0, 0.0, -1.0), 5.0)
        over = None
        for h in hits:              # nearest first = highest first
            if h.initial or h.normal[2] < FLOOR_MIN_NZ:
                continue
            hz = h.pos[2]
            if hz > z + BURY_ABOVE_M:
                continue            # a table, a canopy: overhead, not floor
            if hz > z + FLOOR_ABOVE_M:
                over = over or (hz, h.prim)
                continue
            return {"floor_z": hz, "prim": h.prim, "over_z": over[0] if over else None,
                    "over_prim": over[1] if over else None}
        return {"floor_z": None, "prim": None, "over_z": over[0] if over else None,
                "over_prim": over[1] if over else None}

    def support(self, x, y, z) -> float | None:
        """Highest walkable surface under (x, y) at most BURY_ABOVE_M above base_link: what
        a tyre there stands on, a kerb top included (which ground() would call `over`)."""
        for h in self.ctx.ray_hits((x, y, z + 2.0), (0.0, 0.0, -1.0), 5.0):
            if h.initial or h.normal[2] < FLOOR_MIN_NZ or h.pos[2] > z + BURY_ABOVE_M:
                continue
            return h.pos[2]
        return None

    def tyre_gap(self, x, y, z, yaw) -> float | None:
        """z - (highest support under the three tyres + BASE_LINK_REST_Z_M). The drive holds
        the body level, so it rests on its highest-supported tyre (base_drive's spawn snap
        uses the same rule): ~0 on a slope, < 0 when a tyre is inside a kerb or island top
        the body has not climbed, > 0 when the robot hovers.

        Each tyre's support is the MEAN of four rays TYRE_PROBE_M from its axis (in, out,
        and both sides), never one ray through the tyre itself: that ray walks out of the
        tyre sphere in steps of up to 5 cm (validate_episodes.Context.ray_hits) and lands
        below a ground PLANE, which then reads as an initial overlap and is skipped -- the
        first calibration found the floor under 5 of ~430 hospital samples, all 2.79 m
        down. The mean of a symmetric cross is exact on any plane, slope included."""
        sup = []
        for a in self.wheel_angles:
            tx = x + self.wheel_r * math.cos(yaw + a)
            ty = y + self.wheel_r * math.sin(yaw + a)
            ring = []
            for k in range(4):
                b = yaw + a + 0.5 * math.pi * k
                zz = self.support(tx + TYRE_PROBE_M * math.cos(b),
                                  ty + TYRE_PROBE_M * math.sin(b), z)
                if zz is not None:
                    ring.append(zz)
            if ring:
                sup.append(sum(ring) / len(ring) if len(ring) == 4 else max(ring))
        return (z - (max(sup) + self.rest_z)) if sup else None

    def ring(self, x, y, floor_z, yaw) -> dict:
        """Nearest non-floor hit per band, over the validator's ring heights (0.10 m to the
        robot's top): {"body": (clearance, bearing deg, height, prim, dist), "low": ...}.
        Two-sided, so a ring whose origin is inside a closed prop sees its walls."""
        out = {"body": (math.inf, None, None, None, math.inf),
               "low": (math.inf, None, None, None, math.inf)}
        for hgt in self.heights:
            band = "body" if hgt >= RING_BODY_MIN_H_M else "low"
            for k in range(RING_BEARINGS):
                b_body = 2.0 * math.pi * k / RING_BEARINGS
                b = yaw + b_body
                for h in self.ctx.ray_hits((x, y, floor_z + hgt),
                                           (math.cos(b), math.sin(b), 0.0), RING_RANGE_M,
                                           both_sides=True, max_hits=4):
                    if not h.initial and abs(h.normal[2]) >= FLOOR_MIN_NZ:
                        continue    # slope rising into the ring: floor, not obstacle
                    cl = h.dist - self.ctx.footprint_extent(b_body)
                    if cl < out[band][0]:
                        out[band] = (cl, math.degrees(wrap(b_body)), hgt, h.prim, h.dist)
                    break
        return out

    def body_overlap(self, x, y, floor_z, yaw, floor_prims) -> list[str]:
        """Non-robot, non-floor colliders inside the robot's envelope box from
        BOX_Z_LO_M above the floor to its top: the chassis is INSIDE something."""
        e = self.ctx.envelope
        cx = 0.5 * (e["x_min"] + e["x_max"])
        cy = 0.5 * (e["y_min"] + e["y_max"])
        z_lo = floor_z + BOX_Z_LO_M
        z_hi = floor_z + self.rest_z + e["z_max"]
        c, sn = math.cos(yaw), math.sin(yaw)
        center = (x + c * cx - sn * cy, y + sn * cx + c * cy, 0.5 * (z_lo + z_hi))
        half = (0.5 * (e["x_max"] - e["x_min"]), 0.5 * (e["y_max"] - e["y_min"]),
                0.5 * (z_hi - z_lo))
        # The floor's own prim is NOT exempt: at BOX_Z_LO_M up, a floor mesh can only reach
        # the box on a > 35 deg slope or where it is also a wall (a one-mesh building).
        return [q + (" (floor prim)" if q in floor_prims else "")
                for q in self.ctx.overlap_box(center, half, yaw)]

    def frame(self, tag: str) -> str | None:
        try:
            from PIL import Image

            img = self.runner.camera.grab()
            if img is None or getattr(img, "size", 0) == 0:
                return None
            path = self.frames_dir / f"{tag}.jpg"
            Image.fromarray(img).resize(FRAME_SIZE).save(path, quality=85)
            return str(path)
        except Exception as exc:  # noqa: BLE001 -- a missing frame is reported, not fatal
            return f"frame error: {exc}"

    # ------------------------------------------------------------------------ route
    def plan_from_trace(self, ep, start, rec: dict) -> tuple[dict, Route | None]:
        src = json.loads(Path(self.args.route_from).read_text())
        if src.get("episode") not in (None, ep.name):
            raise ValueError(f"--route-from is a run of {src.get('episode')!r}, not {ep.name!r}")
        tr = [(float(t[0]), float(t[1])) for t in src.get("trace") or []]
        cut = len(tr)
        for i in range(1, len(tr)):
            if math.dist(tr[i], tr[i - 1]) > 2.0:
                cut = i
                break
        tr = tr[:cut]
        if len(tr) < 2:
            rec.update(verdict="NO_PATH", rule=f"{self.args.route_from} has < 2 usable "
                       "trace points")
            return rec, None
        route = Route([tuple(start[:2])] + tr)
        end_gap = math.dist(tr[-1], ep.goal[:2])
        rec.update({"source": f"trace {self.args.route_from}", "trace_points": len(tr),
                    "trace_cut_at": cut if cut < len(src.get("trace") or []) else None,
                    "length_m": r4(route.length, 3), "end_gap_m": r4(end_gap, 3),
                    "success_threshold_m": ep.success_threshold_m,
                    "heading_change_deg": r4(route.heading_change_deg(), 1),
                    "verdict": "OK",
                    "rule": f"recorded trace, {len(tr)} points, ends {end_gap:.2f} m from the "
                            "goal (not graded: the run's own outcome)"})
        return rec, route

    def plan(self, ep) -> tuple[dict, Route | None]:
        start, snap = self.runner._spawn_pose(ep)
        gz = snap.get("ground_z")
        sz = gz if gz is not None else float(ep.start[2])
        gfl = self.ctx.floor_under(ep.goal[0], ep.goal[1], float(ep.goal[2]))
        goal_floor = gfl["z"] if gfl else float(ep.goal[2])
        s_pt, s_isl = self.nm.closest(start[0], start[1], sz)
        g_pt, g_isl = self.nm.closest(ep.goal[0], ep.goal[1], goal_floor)
        rec: dict = {"spawn": snap, "goal_floor_z": r4(goal_floor),
                     "start_nav": [r4(v) for v in s_pt] if s_pt else None,
                     "goal_nav": [r4(v) for v in g_pt] if g_pt else None,
                     "start_island": s_isl, "goal_island": g_isl,
                     "straight_line_m": r4(math.dist(ep.start[:2], ep.goal[:2]), 3)}
        if self.args.route_from is not None:
            return self.plan_from_trace(ep, start, rec)
        rec["source"] = "navmesh"
        if s_pt is None or g_pt is None:
            rec.update(verdict="NO_PATH", rule="no navmesh point near the "
                       + ("start" if s_pt is None else "goal"))
            return rec, None
        rec["start_off_navmesh_m"] = r4(math.dist(s_pt[:2], start[:2]), 3)
        rec["goal_off_navmesh_m"] = r4(math.dist(g_pt[:2], ep.goal[:2]), 3)
        if s_isl is not None and g_isl is not None and s_isl != g_isl:
            rec.update(verdict="NO_PATH", rule=f"start and goal on different navmesh "
                       f"islands ({s_isl} vs {g_isl})")
            return rec, None
        res = self.nm.path(s_pt, g_pt)
        if res is None:
            rec.update(verdict="NO_PATH", rule="query_shortest_path returned no path")
            return rec, None
        pts, nav_len, _ = res
        end_gap = math.hypot(pts[-1][0] - ep.goal[0], pts[-1][1] - ep.goal[1])
        route_pts = [tuple(start[:2])] + [p[:2] for p in pts]
        route = Route(route_pts)
        zs = [p[2] for p in pts]
        rec.update({
            "nav_points": len(pts), "nav_length_m": r4(nav_len, 3),
            "length_m": r4(route.length, 3), "end_gap_m": r4(end_gap, 3),
            "success_threshold_m": ep.success_threshold_m,
            "path_z_range_m": [r4(min(zs)), r4(max(zs))],
            "heading_change_deg": r4(route.heading_change_deg(), 1),
            "points": [[r4(p[0], 3), r4(p[1], 3), r4(p[2], 3)] for p in pts],
        })
        if end_gap > ep.success_threshold_m:
            rec.update(verdict="PARTIAL_PATH",
                       rule=f"path ends {end_gap:.2f} m from the goal > success radius "
                            f"{ep.success_threshold_m:g} m: driving it cannot score")
        else:
            rec.update(verdict="OK", rule=f"path ends {end_gap:.2f} m from the goal "
                                          f"(success radius {ep.success_threshold_m:g} m)")
        return rec, route

    def episode(self, ep) -> dict:
        rec: dict = {"name": ep.name, "start": list(ep.start), "goal": list(ep.goal),
                     "start_yaw_deg": ep.start_yaw_deg, "drives": []}
        self.on_phase("route")
        route_rec, route = self.plan(ep)
        rec["route"] = route_rec
        print(f"[route] {ep.name}: route {route_rec['verdict']} -- {route_rec['rule']}"
              + (f"; {route_rec['length_m']} m, "
                 f"{route_rec.get('nav_points', route_rec.get('trace_points'))} points"
                 if route else ""), flush=True)
        if route is None or self.args.plan_only:
            rec["verdict"] = route_rec["verdict"]
            return rec
        if (route_rec.get("spawn") or {}).get("ground_z", 0.0) is None:
            # run_episode does not run a start with no floor under any tyre (abort_reason
            # "no_floor"); driving it here would measure a fall nobody will ever run.
            rec["no_floor"] = True
            rec["verdict"] = "FAULT"
            rec["rule"] = ("no_floor: ground_spawn_pose found no floor under any tyre; the "
                           "runner does not run this episode (abort_reason no_floor)")
            print(f"[route] {ep.name}: {rec['rule']}", flush=True)
            return rec
        for v in self.args.speed_list:
            self.on_phase(f"drive v={v:g}")
            try:
                d = self.drive(ep, route, v)
            except Exception as exc:  # noqa: BLE001
                d = {"speed_requested_mps": v, "speed_cmd_mps": v, "verdict": "ERROR",
                     "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()}
                print(d["traceback"], flush=True)
            rec["drives"].append(d)
        rec["verdict"] = worst([route_rec["verdict"]] + [d["verdict"] for d in rec["drives"]])
        return rec

    # ------------------------------------------------------------------------ drive
    def reset_like_runner(self, ep) -> str:
        """run_episode's spawn, through the runner's and the base's own methods as they
        exist on disk: `_spawn_pose`, `reset_to`, `runner._settle()` (held until at rest,
        capped at run_navigation.SETTLE_STEPS; else `base.settle`; else free kit.update()
        steps; --settle-steps forces base.settle with that cap), `sync_from_sim`,
        `reset_stats`, and
        the per-episode counters `_physics_fault` reads. Those counters are set inline in
        run_episode, so they are mirrored here by NAME: `_overspeed_steps`, `_fall_steps`
        and `_fault_speed_mps` (= run_navigation.fault_speed_mps(cap) when it exists, else
        FAULT_SPEED_MPS)."""
        base, runner, rn = self.base, self.runner, self.rn
        start, snap = runner._spawn_pose(ep)
        self._last_start = start
        self._last_snap = snap
        base.reset_to(start, ep.start_yaw_rad)
        self._settle_record = None
        if callable(getattr(runner, "_settle", None)) and self.args.settle_steps is None:
            # the runner's own wrapper (base.settle with ITS cap, logged), as run_episode
            got = runner._settle()
            self._settle_record = got if isinstance(got, dict) else None
            how = f"runner._settle() (held, cap {getattr(rn, 'SETTLE_STEPS', '?')})"
        elif callable(getattr(base, "settle", None)):
            got = base.settle(self.kit, self.settle_steps, self.dt)
            # newer bases return what the settle saw (steps used, at rest, faults)
            self._settle_record = got if isinstance(got, dict) else None
            how = f"base.settle (held, cap {self.settle_steps})"
        else:
            for _ in range(self.settle_steps):
                self.kit.update()
            how = "kit.update() x N (base has no settle)"
        base.sync_from_sim()
        base.reset_stats()
        if getattr(runner, "contact", None) is not None:
            runner.contact.reset_episode()   # as run_episode: link poses, metric, cache
        runner._overspeed_steps = 0
        runner._fall_steps = 0
        fsm = getattr(rn, "fault_speed_mps", None)
        runner._fault_speed_mps = (float(fsm(ep.max_speed_mps)) if callable(fsm)
                                   else float(getattr(rn, "FAULT_SPEED_MPS", 3.0)))
        return how

    def drive(self, ep, route: Route, v_req: float) -> dict:
        np, ctx, base, runner = self.np, self.ctx, self.base, self.runner
        dt = self.dt
        v = min(v_req, float(ep.max_speed_mps))
        w_max = float(ep.max_yaw_rate_radps)
        tag = f"{ep.name}_v{v:g}"
        log_mark = ctx.tap.mark()

        # ---- reset exactly as run_episode spawns ------------------------------------
        restore = ctx.restore_robot()
        settle_how = self.reset_like_runner(ep)
        st0 = base.physics_state()
        start = self._last_start
        settle = {"steps": self.settle_steps, "how": settle_how, "restore": restore,
                  "speed_mps": r4(st0["speed_mps"]), "roll_deg": r4(st0["roll_deg"], 2),
                  "pitch_deg": r4(st0["pitch_deg"], 2), "z": r4(st0["z"]),
                  "z_minus_snap_m": r4(st0["z"] - start[2], 4),
                  "fault_speed_mps": r4(getattr(runner, "_fault_speed_mps", None), 3),
                  "base_record": _jsonable(self._settle_record),
                  "spawn": _jsonable({k: v for k, v in (self._last_snap or {}).items()
                                      if k != "start"})}

        ctrl = self.make_controller("pursuit", v, w_max, lookahead_m=self.args.lookahead_m)
        ctrl.reset()
        horizon = max(3.0, 2.0 * self.args.lookahead_m)
        timeout = 2.0 * route.length / v + TIMEOUT_PAD_S
        end_xy = route.p[-1]
        frames = []
        f = self.frame(f"{tag}_000_start")
        if f:
            frames.append({"s_m": 0.0, "t": 0.0, "path": f})

        s = 0.0
        p0, _ = ctx.pose()
        prev = p0
        driven = commanded = 0.0
        z_min = z_max = p0[2]
        vmax, vmax_at = 0.0, None
        over_flags: list[bool] = []          # > OVER_SPEED_MPS (the spec's fixed 3 m/s)
        fs_flags: list[bool] = []            # > the runner's own fault speed this episode
        fault_speed = float(getattr(runner, "_fault_speed_mps", OVER_SPEED_MPS))
        over_events: list[dict] = []
        # Spikes: |v| more than SPIKE_OVER_CMD_MPS above the commanded speed. Reported, not
        # graded (a clamped descent reads hypot(v, 2) legitimately); this is the kerb kick.
        spike_thr = v + SPIKE_OVER_CMD_MPS
        spike_flags: list[bool] = []
        spike_events: list[dict] = []
        gap_lo_at = gap_hi_at = None
        tilt_max, roll_max, pitch_max = 0.0, 0.0, 0.0
        gaps: list[float] = []
        tgaps: list[float] = []
        below_support: list[dict] = []
        buried: list[dict] = []
        ground_missing = 0
        last_floor = p0[2] - self.rest_z
        last_floor_prim = None
        track = xte_max = 0.0
        cruise_run = 0
        cruise_sim = cruise_cmd = 0.0
        cruise_speeds: list[float] = []
        ring_next = 0.0
        ring_n = ring_neg = ring_touch = low_neg = 0
        ring_best = {"body": (math.inf, None, None, None, math.inf, None),
                     "low": (math.inf, None, None, None, math.inf, None)}
        ring_worst: list[dict] = []
        overlap_rings = 0
        overlap_s_min = math.inf
        overlap_prims: dict[str, int] = {}
        overlap_first: list[dict] = []
        floor_prims: set[str] = set()
        tm = {"control": 0.0, "update": 0.0, "state": 0.0, "ground": 0.0, "ring": 0.0,
              "frame": 0.0}
        t_wall0 = time.perf_counter()
        frame_next = FRAME_EVERY_M
        pivot = False
        pivots = pivot_steps = 0
        guard_blocked = guard_slowed = 0
        faults: list[dict] = []
        base_faults = resyncs = 0
        closest_goal = math.dist(p0[:2], ep.goal[:2])
        trace = []
        step = 0
        outcome = "timeout"
        while True:
            t = step * dt
            _t0 = time.perf_counter()
            position = base.position()
            yaw = base.yaw
            s, xte = route.project(position, s - SEARCH_BACK_M, s + SEARCH_AHEAD_M)
            xte_max = max(xte_max, xte)
            if (math.hypot(position[0] - end_xy[0], position[1] - end_xy[1]) <= ARRIVE_M
                    and s >= route.length - ARRIVE_M - 1.0):
                outcome = "arrived"
                break
            if t >= timeout:
                break
            wp = body_waypoints(route, s, position, yaw, horizon)
            cmd = ctrl(wp)
            xl, yl, _ = self._lookahead_point(wp, self.args.lookahead_m)
            err = math.degrees(math.atan2(yl, xl))
            if pivot:
                pivot = abs(err) > PIVOT_EXIT_DEG
            elif abs(err) > PIVOT_ENTER_DEG:
                pivot = True
                pivots += 1
            if pivot:
                cmd = self.Command(0.0, 0.0, math.copysign(w_max, err))
                pivot_steps += 1
            g = self.guard.check(position, yaw, cmd.vx, cmd.vy, count=False)
            guard_blocked += bool(g.blocked)
            guard_slowed += (not g.blocked) and g.scale < 1.0
            vx, vy = (g.vx, g.vy) if self.args.guard == "apply" else (cmd.vx, cmd.vy)
            base.apply(vx, vy, cmd.omega, dt)
            _t1 = time.perf_counter()
            self.kit.update()
            _t2 = time.perf_counter()
            tm["control"] += _t1 - _t0
            tm["update"] += _t2 - _t1
            step += 1
            commanded += abs(vx) * dt

            fault = runner._physics_fault(step * dt)
            st = base.physics_state()
            p, _ = ctx.pose()
            tm["state"] += time.perf_counter() - _t2
            if fault is not None:
                fault["route_s_m"] = r4(s, 2)
                fault["xy"] = [r4(p[0], 3), r4(p[1], 3)]
                faults.append(fault)
                print(f"[route] {tag}: PHYSICS FAULT at t={fault['t']:.2f} s "
                      f"{'+'.join(fault['kinds'])} |v| {fault['speed_mps']:.2f} m/s "
                      f"s={s:.1f} m", flush=True)
                if not self.args.continue_after_fault:
                    outcome = "physics_fault"
                    break
                base_faults += int(base.faults)
                resyncs += int(base.resyncs)
                base.reset_stats()
                runner._overspeed_steps = 0
                runner._fall_steps = 0

            ds = math.dist(p[:2], prev[:2])
            driven += ds
            prev = p
            z_min, z_max = min(z_min, p[2]), max(z_max, p[2])
            spd = st["speed_mps"]
            over = spd > OVER_SPEED_MPS
            over_flags.append(over)
            fs_flags.append(spd > fault_speed)
            spike_flags.append(spd > spike_thr)
            if spd > spike_thr and len(spike_events) < 2000:
                spike_events.append({"t": r4(t + dt, 3), "speed_mps": r4(spd),
                                     "vz_mps": r4(st["vz_mps"]), "route_s_m": r4(s, 2),
                                     "xy": [r4(p[0], 3), r4(p[1], 3)], "z": r4(p[2]),
                                     "floor_prim": last_floor_prim})
            if spd > vmax:
                vmax, vmax_at = spd, {"t": r4(t + dt, 3), "route_s_m": r4(s, 2),
                                      "xy": [r4(p[0], 3), r4(p[1], 3)], "z": r4(p[2]),
                                      "vz_mps": r4(st["vz_mps"]), "floor_prim": last_floor_prim}
            if over and len(over_events) < 40:
                over_events.append({"t": r4(t + dt, 3), "speed_mps": r4(spd),
                                    "vz_mps": r4(st["vz_mps"]), "route_s_m": r4(s, 2),
                                    "xy": [r4(p[0], 3), r4(p[1], 3)],
                                    "floor_prim": last_floor_prim})
            roll_max = max(roll_max, abs(st["roll_deg"]))
            pitch_max = max(pitch_max, abs(st["pitch_deg"]))
            tilt_max = max(roll_max, pitch_max)
            xy_int = base._xy
            track = max(track, math.hypot(p[0] - float(xy_int[0]), p[1] - float(xy_int[1])))
            closest_goal = min(closest_goal, math.dist(p[:2], ep.goal[:2]))

            # cruise: straight, full speed, after a warm-up of the same
            if (not pivot and abs(cmd.omega) < CRUISE_OMEGA and vx >= 0.98 * v):
                cruise_run += 1
                if cruise_run > CRUISE_WARMUP:
                    cruise_sim += ds
                    cruise_cmd += vx * dt
                    cruise_speeds.append(spd)
            else:
                cruise_run = 0

            if step % GROUND_EVERY == 0:
                _tg = time.perf_counter()
                gr = self.ground(p[0], p[1], p[2])
                if gr["floor_z"] is None:
                    ground_missing += 1
                else:
                    last_floor, last_floor_prim = gr["floor_z"], gr["prim"]
                    floor_prims.add(gr["prim"])
                    gaps.append(p[2] - (gr["floor_z"] + self.rest_z))
                tg = self.tyre_gap(p[0], p[1], p[2], base.yaw)
                if tg is not None:
                    tgaps.append(tg)
                    here = {"t": r4(t + dt, 2), "route_s_m": r4(s, 2), "gap_m": r4(tg, 4),
                            "xy": [r4(p[0], 3), r4(p[1], 3)], "z": r4(p[2])}
                    if gap_lo_at is None or tg < gap_lo_at["gap_m"]:
                        gap_lo_at = here
                    if gap_hi_at is None or tg > gap_hi_at["gap_m"]:
                        gap_hi_at = here
                    if tg < WARN_GAP_MIN_M and len(below_support) < 2000:
                        below_support.append({"t": r4(t + dt, 2), "route_s_m": r4(s, 2),
                                              "gap_m": r4(tg, 3)})
                tm["ground"] += time.perf_counter() - _tg
                if gr["over_z"] is not None and len(buried) < 2000:
                    buried.append({"t": r4(t + dt, 2), "route_s_m": r4(s, 2),
                                   "above_m": r4(gr["over_z"] - p[2], 3),
                                   "prim": gr["over_prim"]})

            if driven >= ring_next:
                _tr = time.perf_counter()
                ring_next = driven + RING_EVERY_M
                rg = self.ring(p[0], p[1], last_floor, yaw)
                ring_n += 1
                for band, (cl, bdeg, hgt, prim, raw_d) in rg.items():
                    if not math.isfinite(cl):
                        continue
                    if cl < ring_best[band][0]:
                        ring_best[band] = (cl, bdeg, hgt, prim, raw_d, r4(s, 2))
                    if band == "low":
                        low_neg += cl < 0.0
                        continue
                    ring_neg += cl < 0.0
                    ring_touch += cl < 0.10
                    if cl < 0.0 and len(ring_worst) < 30:
                        ring_worst.append({"route_s_m": r4(s, 2), "clearance_m": r4(cl, 3),
                                           "bearing_deg": r4(bdeg, 1), "height_m": hgt,
                                           "prim": prim, "xy": [r4(p[0], 3), r4(p[1], 3)]})
                ov = self.body_overlap(p[0], p[1], last_floor, yaw, floor_prims)
                if ov:
                    overlap_rings += 1
                    overlap_s_min = min(overlap_s_min, s)
                    for q in ov:
                        overlap_prims[q] = overlap_prims.get(q, 0) + 1
                    if len(overlap_first) < 10:
                        overlap_first.append({"route_s_m": r4(s, 2), "t": r4(t + dt, 2),
                                              "xy": [r4(p[0], 3), r4(p[1], 3)],
                                              "prims": ov[:4]})
                tm["ring"] += time.perf_counter() - _tr
            if driven >= frame_next:
                _tf = time.perf_counter()
                frame_next += FRAME_EVERY_M
                fp = self.frame(f"{tag}_{int(round(driven)):03d}m")
                if fp:
                    frames.append({"s_m": r4(s, 2), "driven_m": r4(driven, 2),
                                   "t": r4(t + dt, 2), "path": fp})
                tm["frame"] += time.perf_counter() - _tf
            if step % TRACE_EVERY == 0:
                trace.append([r4(t + dt, 2), r4(p[0], 3), r4(p[1], 3), r4(p[2], 4),
                              r4(math.degrees(yaw), 1), r4(spd, 3), r4(s, 2)])

        elapsed = step * dt
        wall = time.perf_counter() - t_wall0
        base.stop()
        f = self.frame(f"{tag}_zzz_end")
        if f:
            frames.append({"s_m": r4(s, 2), "driven_m": r4(driven, 2), "t": r4(elapsed, 2),
                           "path": f, "end": True})
        base_faults += int(base.faults)
        resyncs += int(base.resyncs)
        g = np.array(gaps) if gaps else np.zeros(0)
        gt = np.array(tgaps) if tgaps else np.zeros(0)
        n_over = int(sum(over_flags))
        cruise_ratio = cruise_sim / cruise_cmd if cruise_cmd > 0 else None
        physx = ctx.tap.summary(log_mark)
        m = {
            "outcome": outcome, "elapsed_s": r4(elapsed, 2), "timeout_s": r4(timeout, 1),
            "steps": step, "route_m": r4(route.length, 3), "driven_m": r4(driven, 3),
            "driven_over_route": r4(driven / route.length, 4) if route.length > 0 else None,
            # against the arc length actually covered: the drive stops ARRIVE_M short of the
            # route's end, and pursuit cuts corners, so this is the honest length check
            "driven_over_progress": r4(driven / s, 4) if s > 0 else None,
            "commanded_m": r4(commanded, 3),
            "end_s_m": r4(s, 2), "closest_to_goal_m": r4(closest_goal, 3),
            "reached_success_radius": closest_goal <= ep.success_threshold_m,
            "max_speed_mps": r4(vmax), "max_speed_at": vmax_at,
            "overspeed_threshold_mps": OVER_SPEED_MPS, "overspeed_steps": n_over,
            "overspeed_longest_run": longest_run(over_flags), "overspeed_events": over_events,
            "spike_threshold_mps": r4(spike_thr, 3), "spike_steps": int(sum(spike_flags)),
            "spike_longest_run": longest_run(spike_flags),
            "spike_top": top_spikes(spike_events),
            "runner_fault_speed_mps": r4(fault_speed, 3),
            "runner_fault_speed_steps": int(sum(fs_flags)),
            "runner_fault_speed_longest_run": longest_run(fs_flags),
            "runner_fault_speed_persist_steps": getattr(
                runner, "fault_speed_steps", getattr(self.rn, "FAULT_SPEED_STEPS", None)),
            "base_step_anomalies": int(getattr(base, "step_anomalies", 0) or 0),
            "tilt_max_deg": r4(tilt_max, 3), "roll_max_deg": r4(roll_max, 3),
            "pitch_max_deg": r4(pitch_max, 3),
            "z_span_m": r4(z_max - z_min, 5),
            # graded: against the highest tyre support (slope-proof)
            "gap_samples": int(len(gt)), "ground_missing": ground_missing,
            "gap_mean_m": r4(float(gt.mean()), 5) if len(gt) else None,
            "gap_abs_p95_m": r4(float(np.percentile(np.abs(gt), 95)), 5) if len(gt) else None,
            "gap_min_m": r4(float(gt.min()), 5) if len(gt) else None,
            "gap_max_m": r4(float(gt.max()), 5) if len(gt) else None,
            "gap_min_at": gap_lo_at, "gap_max_at": gap_hi_at,
            "below_support_s": r4(len(below_support) * GROUND_EVERY * dt, 2),
            "below_support_first": below_support[:5],
            # reported: against the floor under base_link's centre (grows with slope)
            "centre_gap_mean_m": r4(float(g.mean()), 5) if len(g) else None,
            "centre_gap_abs_p95_m": (r4(float(np.percentile(np.abs(g), 95)), 5)
                                     if len(g) else None),
            "centre_gap_min_m": r4(float(g.min()), 5) if len(g) else None,
            "centre_gap_max_m": r4(float(g.max()), 5) if len(g) else None,
            "buried_samples": len(buried), "buried_first": buried[:5],
            "buried_prims": sorted({b["prim"] for b in buried})[:8],
            "buried_max_above_m": r4(max(b["above_m"] for b in buried), 3) if buried else None,
            "cruise_steps": len(cruise_speeds), "cruise_speed_ratio": r4(cruise_ratio, 5),
            "cruise_physx_speed_median_mps": (r4(float(np.median(cruise_speeds)), 4)
                                              if cruise_speeds else None),
            "track_err_max_m": r4(track, 4), "cross_track_max_m": r4(xte_max, 3),
            "resyncs": resyncs, "base_faults": base_faults,
            "pivots": pivots, "pivot_steps": pivot_steps,
            "guard_mode": self.args.guard, "guard_blocked_steps": guard_blocked,
            "guard_slowed_steps": guard_slowed,
            "rings": ring_n, "ring_heights_m": self.heights,
            "route_min_clearance_m": (r4(ring_best["body"][0], 3)
                                      if math.isfinite(ring_best["body"][0]) else None),
            "route_min_clearance_at": _ring_at(ring_best["body"]),
            "low_min_clearance_m": (r4(ring_best["low"][0], 3)
                                    if math.isfinite(ring_best["low"][0]) else None),
            "low_min_clearance_at": _ring_at(ring_best["low"]),
            "rings_through_obstacle": ring_neg, "rings_under_10cm": ring_touch,
            "low_rings_through_obstacle": low_neg,
            "through_obstacle_first": ring_worst[:10],
            "overlap_s_min_m": r4(overlap_s_min, 2) if overlap_rings else None,
            "overlap_rings": overlap_rings, "overlap_prims": dict(sorted(
                overlap_prims.items(), key=lambda kv: -kv[1])[:8]),
            "overlap_first": overlap_first,
            "wall_s": r4(wall, 1),
            "ms_per_step": {k: r4(1000.0 * v / max(step, 1), 2) for k, v in tm.items()},
            "faults": faults[:20], "fault_count": len(faults),
            "contact": (_jsonable(runner._contact_summary())
                        if callable(getattr(runner, "_contact_summary", None)) else None),
            "physx_errors": physx["physx_errors"], "physx_warnings": physx["physx_warnings"],
            "physx_messages": physx["messages"][:5],
        }
        m["spawn_lift_m"] = settle["z_minus_snap_m"]
        m["spawn_origin_inside"] = list((self._last_snap or {}).get("origin_inside") or [])
        sr = settle.get("base_record") or {}
        m["settle_faults"] = int(sr.get("faults") or 0) if isinstance(sr, dict) else 0
        m["settle_settled"] = sr.get("settled") if isinstance(sr, dict) else None
        rules = self.grade(m, ep)
        verdict = worst(r["verdict"] for r in rules)
        print(f"[route] {tag}: {verdict} {outcome} in {elapsed:.1f} s, drove {driven:.1f}/"
              f"{route.length:.1f} m, vmax {vmax:.2f} ({n_over} > {OVER_SPEED_MPS:g}, "
              f"longest {longest_run(over_flags)}), tilt "
              f"{tilt_max:.1f}, gap p95 {m['gap_abs_p95_m']}, buried {len(buried)}, "
              f"clear {m['route_min_clearance_m']} (low {m['low_min_clearance_m']}), overlap "
              f"{overlap_rings}/{ring_n}, cruise {m['cruise_speed_ratio']}, contact "
              f"{(m['contact'] or {}).get('contact_steps', 'off')}; {wall:.0f} s wall, "
              f"ms/step {m['ms_per_step']}", flush=True)
        return {"speed_requested_mps": v_req, "speed_cmd_mps": v,
                "speed_clamped": v < v_req, "settle": settle, "verdict": verdict,
                "rules": rules, "metrics": m, "frames": frames, "trace": trace,
                "trace_cols": ["t", "x", "y", "z", "yaw_deg", "speed_mps", "route_s_m"]}

    @staticmethod
    def grade(m: dict, ep) -> list[dict]:
        rules = []

        def rule(name, verdict, text, value=None):
            rules.append({"metric": name, "value": value, "verdict": verdict, "rule": text})

        if m["fault_count"]:
            f0 = m["faults"][0]
            rule("physics_fault", "FAULT",
                 f"runner fault at t={f0['t']:.2f} s, route s={f0.get('route_s_m')} m: "
                 f"{'+'.join(f0['kinds'])} (|v| {f0['speed_mps']:.2f} m/s, roll "
                 f"{f0['roll_deg']:.1f}, pitch {f0['pitch_deg']:.1f} deg)"
                 + (f"; {m['fault_count']} faults in all" if m["fault_count"] > 1 else ""),
                 m["fault_count"])
        else:
            rule("physics_fault", "OK", "no runner physics fault", 0)
        n = m["overspeed_steps"]
        rule("overspeed_steps", "WARN" if n else "OK",
             f"{n} step(s) over {m['overspeed_threshold_mps']:g} m/s, longest run "
             f"{m['overspeed_longest_run']} (max {m['max_speed_mps']} m/s); over the runner's "
             f"fault speed {m['runner_fault_speed_mps']} m/s: {m['runner_fault_speed_steps']} "
             f"step(s), longest {m['runner_fault_speed_longest_run']} of the "
             f"{m['runner_fault_speed_persist_steps']} that fault"
             + (" -- absorbed by the fault rule's persistence" if n and not m["fault_count"]
                else ""), n)
        lift = m.get("spawn_lift_m")
        if lift is not None and lift > SPAWN_LIFT_WARN_M:
            rule("spawn_embedded", "WARN", f"the held settle LIFTED the robot {lift * 1000:.1f} mm "
                 f"above its snapped spawn z (a correct snap settles {SPAWN_CLEARANCE_NOTE}): "
                 "ground_spawn_pose under-read the floor and the robot started inside it",
                 lift)
        if m.get("spawn_origin_inside"):
            rule("spawn_origin_inside", "WARN", "a spawn probe ray started inside "
                 f"{m['spawn_origin_inside'][:2]}: the start overlaps a collider",
                 len(m["spawn_origin_inside"]))
        if m.get("settle_faults"):
            rule("settle_fault", "WARN", f"{m['settle_faults']} base fault(s) during the held "
                 "settle before the drive (reset_stats() then wipes them in the runner)",
                 m["settle_faults"])
        if m.get("settle_settled") is False:
            rule("settle_at_rest", "WARN", "base.settle hit its step cap without coming to "
                 "rest", False)
        sa = m["base_step_anomalies"]
        if sa:
            rule("step_anomalies", "WARN", f"{sa} update(s) ran != 1 physics step between "
                 "apply() and its judgement: the jump test reads them as resyncs", sa)

        def bound(name, value, hi=None, lo=None, unit=""):
            if value is None:
                rule(name, "WARN", f"{name} not measured", None)
                return
            bad = (hi is not None and value > hi) or (lo is not None and value < lo)
            lim = " / ".join(x for x in (f"> {hi:g}" if hi is not None else "",
                                         f"< {lo:g}" if lo is not None else "") if x)
            rule(name, "WARN" if bad else "OK", f"{name}={value:g}{unit} (WARN {lim})", value)

        bound("tilt_max_deg", m["tilt_max_deg"], hi=WARN_TILT_DEG, unit=" deg")
        bound("gap_abs_p95_m", m["gap_abs_p95_m"], hi=WARN_GAP_P95_M, unit=" m")
        bound("gap_max_m", m["gap_max_m"], hi=WARN_GAP_MAX_M, unit=" m")
        bound("gap_min_m", m["gap_min_m"], lo=WARN_GAP_MIN_M, unit=" m")
        if m["below_support_s"]:
            rules[-1]["rule"] += (f"; a tyre was > {-WARN_GAP_MIN_M:g} m inside a walkable "
                                  f"surface for {m['below_support_s']} s")
        b = m["buried_samples"]
        rule("buried", "WARN" if b else "OK",
             f"{b} floor sample(s) with a walkable surface {FLOOR_ABOVE_M:g}-{BURY_ABOVE_M:g} m "
             f"above base_link (riding under it, up to {m['buried_max_above_m']} m)"
             + (f": {m['buried_prims'][:2]}" if b else ""), b)
        rule("resyncs", "WARN" if m["resyncs"] else "OK",
             f"{m['resyncs']} base resync(s)", m["resyncs"])
        bound("track_err_max_m", m["track_err_max_m"], hi=WARN_TRACK_M, unit=" m")
        if m["cruise_speed_ratio"] is None:
            rule("cruise_speed_ratio", "OK", "no straight full-speed stretch to measure", None)
        else:
            bound("cruise_speed_ratio", m["cruise_speed_ratio"], hi=WARN_CRUISE[1],
                  lo=WARN_CRUISE[0])
        if m["outcome"] == "arrived":
            bound("driven_over_progress", m["driven_over_progress"], hi=WARN_DRIVEN[1],
                  lo=WARN_DRIVEN[0])
        elif m["outcome"] == "timeout":
            rule("arrival", "WARN", f"route end not reached in {m['timeout_s']} s "
                 f"(stopped at s={m['end_s_m']} of {m['route_m']} m)", m["end_s_m"])
        cl = m["route_min_clearance_m"]
        if cl is None:
            rule("route_clearance", "OK", f"nothing COLLIDABLE within {RING_RANGE_M:g} m of "
                 f"the route at body height (>= {RING_BODY_MIN_H_M:g} m). Not evidence of free "
                 "space where geometry has no collider (skinned characters; every outdoor "
                 "prop with NAV_SCENE_COLLIDERS=0) -- the frames are the check there", None)
        else:
            at = m["route_min_clearance_at"]
            rule("route_clearance", "WARN" if cl < 0 else "OK",
                 f"min body clearance {cl:g} m at {at['height_m']} m ({at['prim']}); "
                 f"{m['rings_through_obstacle']} of {m['rings']} rings < 0 (chassis through "
                 "an obstacle)", cl)
        ov = m["overlap_rings"]
        rule("body_overlap", "WARN" if ov else "OK",
             f"envelope box ({BOX_Z_LO_M:g} m up to the top) overlapped a non-floor collider "
             f"at {ov} of {m['rings']} samples" + (f": {list(m['overlap_prims'])[:2]}"
                                                   if ov else "")
             + (f"; ONLY in the route's last {m['route_m'] - m['overlap_s_min_m']:.2f} m "
                f"(from s={m['overlap_s_min_m']} of {m['route_m']} m): the goal's own "
                "surroundings, not the corridor"
                if ov and m.get("overlap_s_min_m") is not None
                and m["overlap_s_min_m"] >= m["route_m"] - END_ZONE_M else ""), ov)
        c = m.get("contact")
        if c is None:
            rule("contact", "OK", "contact layer off (NAV_CONTACT=0): nothing stopped the base",
                 None)
        else:
            n = int(c.get("contact_steps") or 0)
            fc = c.get("first_contact") or {}
            rule("contact", "WARN" if n else "OK",
                 f"{n} step(s) clipped or refused by the contact layer"
                 + (f", longest pressed {c.get('max_run_s')} s"
                    + (" (DynaNav physical_collision)" if c.get("physical_collision") else "")
                    + f"; first at t={fc.get('t')} s, {fc.get('box')} box on {fc.get('prim')}"
                    if n else ""), n)
        rule("physx_errors", "WARN" if m["physx_errors"] else "OK",
             f"{m['physx_errors']} PhysX ERROR(s) during the drive", m["physx_errors"])
        return rules


# ---------------------------------------------------------------------- calibration
__doc__ += """
CALIBRATION
-----------
2026-10-05, nav/config/episodes.yaml names, --speeds 1.5,0.45, through isaac_slot (tag
route-tool). Shared code base_drive c6bd327c / run_navigation e9b91752 / physics_setup
8a8b7768 unless said; JSONs in nav/results/route_drive/ (frames in <stem>_frames/).

The tool against itself (indoors, where the answer is known):
  cal_hospital4.json (md5 unchanged start->end). hospital_down_hallway: navmesh route
  32.66 m, ends 0.00 m from the goal; v1.5 drove 32.17 m to s=32.18 (driven/progress
  0.9999), z span 1.9 mm, tyre gap mean = p95 = 0.62 mm, tilt 0.75 deg, 0 steps > 3 m/s,
  cruise sim/commanded 1.00001, track error 9 mm, 0 faults -> OK. v0.45 the same physics
  (z span 1.9 mm, cruise 0.99996) but WARN body_overlap: 1 of 127 samples, SM_Gurney_01b,
  at s=32.13 of 32.66 -- the goal stands at the gurney; the rule now labels an overlap
  confined to the last END_ZONE_M as the goal's surroundings. hospital_vending_machine:
  13.02 m, OK at both speeds (z span 1.4 mm, driven/progress 1.001).
  Repeatable: cal_hospital3.json (run_navigation e624c060) gives the same drive to the
  last digit (32.171 m, vmax 1.5005, z span 0.00192); so does cal_hospital5.json on
  base_drive 6caeb283 / run_navigation a431ed92 / physics_setup a46bde3b (settle through
  runner._settle(): 11 held steps, at rest) -- except the SPAWN: with the robot standing
  on its own start (the stage's authored pose is hospital_down_hallway's), the new
  footprint ground_spawn_pose read the tyre floors at -0.0169 (centre rays -2.78 m,
  walked through the robot's own tyre spheres into the ground plane) and spawned the
  robot 13.8 mm into the floor; the settle lifted it out with 0 faults. The same start
  with the robot elsewhere snaps correctly (-3.1 mm). Now the `spawn_embedded` rule.
  PhysX |v| on straight cruise is 1.309 at 1.5 and 0.266 at 0.45 m/s: BOTH 0.18-0.19 m/s
  under the command, i.e. one step of tyre friction (~11 m/s^2) after apply() writes
  the commanded velocity, while x/y stay pinned (displacement ratio 1.0000). So the
  runner's speed fault reads a post-friction |v|; a spike above it is contact.

Outdoor, the 13 cm traffic island on outdoor_umbrellas:
  cal_outdoor_nav2.json (base_drive changed to a246f68b DURING the run; the run used
  c6bd327c as loaded). The NAVMESH route (18.87 m, start on island_7's top) never meets a
  kerb: nearest kerb-band hit 2.27 m, tyre gap p95 0.5 mm / max 14 mm, z span 0.29 m (the
  route's own relief, path z 5.857-6.148), vmax 1.51, tilt <= 2.2 deg -> OK at both speeds.
  So the island is a POLICY-route hazard, not a shortest-route one.
  cal_outdoor_curb2.json, --route-from nav/results/20260901-063508_outdoor_umbrellas_
  braking.json (the policy trace that went onto the island, 67.1 m, md5 unchanged):
  at 1.5 m/s the base rides up onto island_7 (z 6.10 -> 6.13), at s=45.6 m drops through
  its top to the road (z 6.13 -> 5.98 in 0.25 s, vz -1.45, |v| 2.08 m/s = the drive's own
  clamped descent, the run's maximum) and then rides UNDER the island top for 16.4 s
  (tyre gap down to -0.153 m, 323 `buried` samples up to 0.146 m under it); at 0.45 m/s
  the same for 42.3 s (|v| max 1.42). 0 steps over 3 m/s, 0 over the runner's 3.033, no
  fault -- under the rule it ran with (3.033 x 30 steps) and under any 3-step rule, since
  nothing reached 3 m/s. Verdict WARN (gap, buried, track error 6 cm): the island,
  correctly found.
  cal_outdoor_upswayfar_trace.json, --route-from nav/results/20261005-113221_outdoor_
  upsway_far_braking.json (the run that FAULTED: speed 3.10 m/s x 3 steps, vz +1.33, at
  its last sample). Shared code at start base_drive 7e3452a5 / run_navigation a431ed92
  (FAULT_SPEED_STEPS back to 3, SETTLE_STEPS cap 120, settle-until-at-rest: 11 steps
  here); base_drive and physics_setup changed during the run, which used the loaded
  versions. 67.4 m trace, crossing island_7's kerbs at s=27-29 m and leaving it at 35.5:
    v1.5   |v| max 3.91 m/s (vz +1.74, s=29.05: a kick UP the kerb); 1 step > 3.033,
           longest run 1; 5 steps > 2.0 (cmd + 0.5), longest 2; tilt 6.3 deg; tyre 0.157 m
           inside the kerb at s=27.0; leaving the island 0.106 m above support (s=35.5,
           vz -1.54); 58 buried samples (island_7 curbs/top). No fault.
    v0.45  |v| max 4.67 (vz +1.95, s=29.04) and 4.10 (vz -1.61, s=27.18), each ONE step;
           10 steps > 0.95, longest 4; tilt 8.5; 171 buried samples. No fault.
  So on this route the kerb spikes are real and large (3.9-4.7 m/s) but single steps,
  and the current 3.033 m/s x 3-step rule absorbs them with 2 steps of margin. The
  run's own fault (3 steps at 3.10, at its LAST trace sample) did not reproduce: a
  replay stops ARRIVE_M = 0.5 m short of a trace's end and at constant speed, where the
  policy kept pushing into the wedge under its own speed profile; at that spot (s=64.9,
  z 5.829) the 0.45 replay read 1.67 m/s once. A replay is the route, not the commands.
  Its frames show what no rule here can: at s=28-32 m the trace drives INSIDE a store
  (dark interior, a mannequin at the lens) and at s=52 m into a doorway -- outdoor_small
  buildings have no colliders, so body clearance is None ("nothing collidable") along
  the whole route. Outdoors, look at the frames.
"""

if __name__ == "__main__":
    raise SystemExit(main())
