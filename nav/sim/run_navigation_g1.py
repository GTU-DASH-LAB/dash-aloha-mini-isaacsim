"""The navigation loop with a WALKING Unitree G1 instead of AlohaMini.

Same episode, same policy, same loop: `NavigationRunner.run_episode()` is inherited unchanged, so
the decision cadence, the async plan bookkeeping, the frame history, the waypoint history, the
recovery/replan protocol and the result record are the AlohaMini runner's own code. What is
swapped is everything that is a fact about the body:

  stage      the env's nav_<env>.usda opened read-only; in the SESSION layer only: /World/Aloha
             deactivated, the G1 referenced at the episode start, its D435 head camera (+ the
             headlight, AlohaMini camera_nav parity), a chase camera prim, and the physics scene
             at the G1's trained 200 Hz / TGS / CPU dynamics (locomotion.configure_physics_scene).
             raise_gpu_capacities() and the scene colliders are applied exactly as setup() does;
             neither touches timeStepsPerSecond. The stage file's md5 is checked before and after.
  base       locomotion.G1Base in CALLBACK mode: the runner's one kit.update() per 1/60 s step
             runs 3-4 physics sub-steps and the walk policy ticks at 50 Hz of sim time from a
             PHYSICS_PRE_STEP callback. Hold-on-stop and the command slew limit are the shipping
             defaults. physics_state()["terminal"] (fell / free_fall / no_floor /
             physics_dt_mismatch / handle lost) ends the episode, with abort_reason = that kind.
  controller "g1pursuit": the braking pure pursuit (lookahead 1.0 m, obeys plan speed), and when
             the lookahead point is more than 45 deg off the nose an in-place turn until it is
             within 15 deg -- the route survey's rule (24/27 routes, 0 falls). Speed cap 0.6 m/s
             (what the G1 tracks well; the menu speed 0.45 governs below it), yaw rate 0.8 rad/s.
  guard      nav/sim/g1/guard.G1Guard: floor-step probe + rings at 0.20/0.80/1.15 m, stop 0.8 m,
             slow 1.8 m, exact feasible-cone projection. Not CollisionGuard, whose single 0.30 m
             plane misses the 5 cm boxes a G1 falls over.
  recovery   StuckRecovery's triggers (wedge 3 s, balk 5 s), but the manoeuvre is a TURN IN PLACE
             toward the side with more clearance (G1Guard.recovery_turn): the walk policy was
             never trained to walk backwards and G1Base clamps a reverse request to a hold. A wedge
             turn ends when the guard sees room ahead (or after 150 deg), a balk turn after 35 deg;
             both end in the runner's forced replan, as for AlohaMini.
  camera     the G1's D435 head mount -- re-pitched to 12 deg down (`--camera p12`, primary) or the
             real 47.6 deg (`--camera real`). Its intrinsics, mount and LIVE pose (height, forward/
             left offset, pitch, roll, yaw vs the pelvis -- camera_geometry's convention, read in the
             same update the frame is grabbed in) ride along with every /predict as `camera`, with
             the arc subset this camera can show (+-0.60 dropped at 12 deg: 0-0.39 m of a 3 m arc
             is in frame). The chase camera is created lazily, when the run is being recorded.
  result     EpisodeResult + robot="g1", camera=<label>, g1={base stats, guard telemetry,
             controller, recovery turns, camera pose stats, speed tracking, cost, md5s}; same
             nav/results/<stamp>_<episode>_<controller>.json naming, controller "g1pursuit", so a
             G1 row can never be counted as an AlohaMini "braking" row by bench.sh's glob.

Run through the slot wrapper (GPU0, Kit pinned), and with a REAL policy also under the policy lock:

    ISAAC_SLOTS=8 ISAAC_SLOT_MIN_FREE_MIB=8000 ISAAC_SLOT_TAG=g1nav nav/tools/isaac_slot.sh \\
        ~/robotics/isaacsim-6.0.1/python.sh nav/sim/run_navigation_g1.py \\
        --episode hospital_vending_machine --policy stub --results-dir /tmp/x > log 2>&1
    nav/bench_g1.sh --only hospital_vending_machine --policy stub     # one process per episode

python.sh exits 0 whatever happens: read the "[g1nav] STATUS=" line and the result file.

SEAMS this file would rather have than wrap around (run_navigation.py is someone else's):
  * setup() is monolithic (Kit + stage + /World/Aloha articulation + KinematicBase + camera +
    guard + recovery), so it is re-implemented here in full; a split into
    _open_stage / _author_robot / _make_base / _make_sensors would let this reuse the rest.
  * run_episode() builds its controller with the module-level make_controller(name, cap, yaw) --
    "g1pursuit" is REGISTERED in controllers.CONTROLLERS for that (a `self._make_controller(ep)`
    hook would be cleaner);
  * it constructs `EpisodeResult(...)` by module global and hard-codes abort_reason
    "physics_fault" -- this file rebinds run_navigation.EpisodeResult (in this process only) to a
    subclass that adds the G1 fields and refines abort_reason (a `self._make_result(**kw)` hook);
  * it calls policy.predict() with a fixed argument list -- the camera rides in through the
    client's _post (a `self._predict_extra()` dict merged into the call would be cleaner).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

SIM = Path(__file__).resolve().parent
REPO = SIM.parents[1]
for _p in (SIM / "g1", SIM, REPO / "nav" / "policy_server", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import camera_geometry as CG  # noqa: E402  (nav/sim/g1)
import locomotion as L  # noqa: E402
from guard import G1Guard  # noqa: E402

import controllers  # noqa: E402
import run_navigation as RN  # noqa: E402
from client import POSE_KEYS as CG_POSE_KEYS, PolicyClient, camera_from_spec  # noqa: E402
from controllers import (  # noqa: E402
    _EPS,
    BrakingPursuitController,
    Command,
    _lookahead_point,
    is_stationary,
)
from episode import RESULTS_DIR, EpisodeResult, env_name, load_episode  # noqa: E402
from stuck_recovery import CLEAR_MARGIN_M, StuckRecovery  # noqa: E402

PHYSICS_DT = RN.PHYSICS_DT          # the runner's CONTROL step (one kit.update), 1/60 s

# ------------------------------------------------------------------------------- G1 parameters
G1_CONTROLLER = "g1pursuit"
G1_SPEED_CAP_MPS = 0.6              # the route survey's speed; the menu speed (0.45) governs below it
G1_YAW_RATE_RADPS = 0.8             # route survey --max-yaw-rate; trained box is +-1.0
G1_LOOKAHEAD_M = 1.0
G1_PIVOT_ENTER_DEG = 45.0           # route survey / route_drive: lookahead > 45 deg off -> turn in place
G1_PIVOT_EXIT_DEG = 15.0            # ... until within 15 deg
G1_ROBOT_TYPE = "humanoid robot"    # the policy's system prompt says "You are a {robot_type}"
G1_RECOVERY_WZ = 0.6                # G1Guard.recovery_turn default
G1_RECOVERY_MAX_TURN_DEG = 150.0    # a wedge turn ends when the guard sees room, or here
G1_BALK_TURN_DEG = 35.0             # a balk is a LOOK: half the 69 deg HFOV brings a cropped edge in
# The guard's SLOW band (0.8-1.8 m) scales speed linearly down to 0.15 x -- 0.07 m/s at the menu's
# 0.45 -- and the walk policy does not respond to that: achieved vx ~ 0.91 cmd - 0.063 (locomotion.py),
# i.e. it marks time. Measured in the first stub run (hospital_vending_machine, 2026-10-06): 16 s at
# 0.04 m/s beside a reception-desk corner that was 0.8-1.8 m off the nose at -30 deg, never blocked, never
# counted. So inside the slow band the G1 walks at no less than this (or the request, if lower); the STOP
# distance (0.8 m, sized for coasting from 0.8 m/s) is unchanged and still blocks, which is what hands a
# real wedge to the recovery.
G1_SLOW_BAND_MIN_MPS = 0.20
# Forward-speed calibration. The walk policy undershoots a forward command: achieved vx ~ 0.91 cmd - 0.063
# (locomotion.py's fit; g1/repair r4 measured cmd 0.5/0.8/1.0 -> 0.396/0.673/0.857 from pelvis POSITIONS),
# and this runner's own first Qwen runs (2026-10-11) walked 0.346 m/s on straight segments against a
# 0.436 command -- so the 0.45 m/s the menu asks for was walked at ~0.35, and hospital_down_hallway timed
# out 3.9 m short of a 32.4 m goal on a path no longer than the route. G1NavBase.apply() inverts the fit,
# so the robot WALKS the speed the plan and guard ask for; the policy command is capped at
# G1_VX_CMD_MAX (exercised without a fall in r4's 72 s mixed test). --no-speed-cal sends the raw request.
G1_VX_FIT = (0.91, -0.063)          # achieved = a * cmd + b
G1_VX_CMD_MAX = 0.8


def g1_vx_command(v: float) -> float:
    """Policy vx command that walks at `v` m/s (v <= 0 is passed through: a reverse request holds)."""
    if v <= 0.0:
        return v
    a, b = G1_VX_FIT
    return min(G1_VX_CMD_MAX, (v - b) / a)


G1_SETTLE_MIN_S = 1.5               # stand this long after a reset before the episode starts
G1_SETTLE_MAX_S = 3.0
CAMERAS = {"p12": "nav_camera", "real": "head_camera_real"}   # asset_spec.json keys
CHASE_PRIM = "/World/G1Chase"
CHASE_RES = (1280, 720)
FAULT_KIND_NAMES = {"fall": "fell", "free_fall": "free_fall", "no_floor": "no_floor",
                    "physics_dt_mismatch": "dt_mismatch", "physics_handle_lost": "handle_lost"}
# A sim error voids the run (physics_faults = 1, summarize_runs then flags it); a FALL or a free
# fall is the robot's own failure -- the trace up to it is a real walk -- so those keep
# physics_faults = 0 and say what happened in abort_reason + physics_fault["kinds"].
SIM_ERROR_KINDS = {"dt_mismatch", "handle_lost", "no_floor"}


def _md5(path: Path) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def g1_curvatures(cam_spec: dict, all_kappas=None) -> tuple[list[float], list[dict]]:
    """The menu arcs this camera can show while WALKING (mount pitch + the policy's ~3.3 deg lean,
    lens 1.089 m up, camera_geometry.WALK_*): arc_badge keep = at least 0.5 m of the 3 m arc in
    frame. At 12 deg: +-0.60 dropped (0.0 m visible), +-0.35 show 1.27 m. At 47.6 deg all seven
    keep their first ~1.4-1.7 m (the floor window is 0.46-2.4 m ahead)."""
    from arc_menu import CURVATURE_SETS
    kappas = list(all_kappas or CURVATURE_SETS["coarse"])
    lean = CG.WALK_PITCH_DEG - CG.MOUNT_PITCH_DEG
    ma = dict(cam_h=CG.WALK_CAM_HEIGHT_M, cam_x=CG.WALK_CAM_FORWARD_M, cam_y=CG.WALK_CAM_LEFT_M,
              pitch_deg=float(cam_spec["pitch_down_deg_rel_head"]) + lean, roll_deg=0.0, yaw_deg=0.0)
    rows = [CG.arc_badge(k, ma, fov_deg=float(cam_spec["hfov_deg"])) for k in kappas]
    return [r["kappa"] for r in rows if r["keep"]], rows


# =============================================================================== the base
class G1NavBase(L.G1Base):
    """G1Base plus the few KinematicBase methods run_episode()/perform_reset() call that a walking
    robot has no use for, and a WORLD pelvis-z record (G1Base's z_min/z_max are the pelvis height
    above the feet, a fall measure; base_z_span_m in a result means world z)."""

    step_anomalies = 0
    _per_step = False
    speed_cal = True                # G1_VX_FIT inversion in apply(); main() sets it from --no-speed-cal
    desired_vx = 0.0                # the last forward speed asked for (before calibration)

    def apply(self, vx: float, vy: float, omega: float, dt: float) -> None:
        self.desired_vx = float(vx)
        return super().apply(g1_vx_command(vx) if self.speed_cal else vx, vy, omega, dt)

    def reset_stats(self) -> None:
        super().reset_stats()
        self.zw_min, self.zw_max, self.zw_step_max, self._zw_prev = float("inf"), float("-inf"), 0.0, None

    def physics_state(self) -> dict:
        out = super().physics_state()
        z = out.get("z")
        if z is not None and not out.get("physics_handle_lost"):
            self.zw_min, self.zw_max = min(self.zw_min, z), max(self.zw_max, z)
            if self._zw_prev is not None:
                self.zw_step_max = max(self.zw_step_max, abs(z - self._zw_prev))
            self._zw_prev = z
        return out

    def park_for_probe(self, kit) -> None:      # floor_below() already skips /World/G1
        return None

    def check_step(self) -> None:               # nothing is teleported: no jump to judge
        return None

    def settle(self, kit, max_steps: int, dt: float) -> dict:
        """Stand (hold, zero command) for at least G1_SETTLE_MIN_S, then until 15 consecutive updates
        are calm (tilt < 8 deg, |v| < 0.25 m/s), capped at G1_SETTLE_MAX_S. The G1 steps in place at
        a zero command, so 'at rest' is a gait property, not zero velocity."""
        n_min = int(round(G1_SETTLE_MIN_S / dt))
        n_max = max(n_min, int(round(G1_SETTLE_MAX_S / dt)))
        calm, worst, ps, k = 0, 0.0, {}, 0
        for k in range(1, n_max + 1):
            self.apply(0.0, 0.0, 0.0, dt)
            kit.update()
            ps = self.physics_state()
            worst = max(worst, float(ps.get("tilt_deg", 0.0)))
            calm = calm + 1 if (ps.get("tilt_deg", 99) < 8.0 and ps.get("speed_mps", 9) < 0.25) else 0
            if ps.get("terminal"):
                break
            if k >= n_min and calm >= 15:
                break
        return {"steps": k, "settled": bool(calm >= 15 and not ps.get("terminal")),
                "tilt_deg": round(float(ps.get("tilt_deg", 0.0)), 3), "worst_tilt_deg": round(worst, 3),
                "speed_mps": round(float(ps.get("speed_mps", 0.0)), 4), "z": round(float(ps.get("z", 0.0)), 4),
                "faults": self.faults, "first_fault": self.first_fault}


# =============================================================================== controller
class G1PivotPursuit(BrakingPursuitController):
    """Braking pure pursuit + turn in place when the lookahead is > enter_deg off the nose, until it
    is within exit_deg (hysteresis). The inner pursuit runs on every plan (its heading filter stays
    current), and its output is replaced while pivoting. A STOP plan clears the pivot."""

    name = G1_CONTROLLER
    last = None                     # the instance run_episode built, for the result record

    def __init__(self, max_speed_mps: float, max_yaw_rate_radps: float, lookahead_m: float = G1_LOOKAHEAD_M,
                 pivot_enter_deg: float = G1_PIVOT_ENTER_DEG, pivot_exit_deg: float = G1_PIVOT_EXIT_DEG,
                 pivot_wz: float | None = None, **kw):
        super().__init__(max_speed_mps, max_yaw_rate_radps, lookahead_m=lookahead_m, **kw)
        self.enter = math.radians(pivot_enter_deg)
        self.exit = math.radians(pivot_exit_deg)
        self.pivot_wz = float(pivot_wz if pivot_wz is not None else max_yaw_rate_radps)
        G1PivotPursuit.last = self
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.pivoting = False
        self.pivots = 0
        self.pivot_decisions = 0
        self.decisions = 0
        self.stops = 0

    def describe(self) -> dict:
        return {"name": self.name, "v_max": self.v_max, "w_max": self.w_max, "lookahead_m": self.lookahead_m,
                "obey_plan_speed": self.obey_plan_speed, "speed_scaled_feedback": self.speed_scaled_feedback,
                "pivot_enter_deg": round(math.degrees(self.enter), 2), "pivot_exit_deg": round(math.degrees(self.exit), 2),
                "pivot_wz": self.pivot_wz, "decisions": self.decisions, "pivots": self.pivots,
                "pivot_decisions": self.pivot_decisions, "stop_decisions": self.stops}

    def __call__(self, waypoints: np.ndarray, guidance: np.ndarray | None = None) -> Command:
        cmd = super().__call__(waypoints, guidance)
        self.decisions += 1
        if is_stationary(waypoints):
            self.pivoting = False
            self.stops += 1
            return cmd
        x, y, dist = _lookahead_point(waypoints, self.lookahead_m)
        if dist < _EPS:
            self.pivoting = False
            return cmd
        bearing = math.atan2(y, x)
        if not self.pivoting and abs(bearing) > self.enter:
            self.pivoting = True
            self.pivots += 1
        elif self.pivoting and abs(bearing) < self.exit:
            self.pivoting = False
        if self.pivoting:
            self.pivot_decisions += 1
            return Command(0.0, 0.0, math.copysign(self.pivot_wz, bearing))
        return cmd


# =============================================================================== guard
class G1NavGuard(G1Guard):
    """G1Guard with a walking-speed floor inside the slow band (G1_SLOW_BAND_MIN_MPS) and a record of
    the time spent slowed -- a crawl the guard caused is otherwise invisible in a result (it is not an
    intervention: nothing blocked)."""

    def __init__(self, *a, slow_min_mps: float = G1_SLOW_BAND_MIN_MPS, **kw):
        super().__init__(*a, **kw)
        self.slow_min_mps = float(slow_min_mps)

    def reset_telemetry(self) -> None:
        super().reset_telemetry()
        self.slowed_checks = 0
        self.floored_checks = 0
        self.slow_spans: list = []
        self._slow_span = None
        self._t = 0.0

    def check(self, position, yaw, vx, vy, count: bool = True):
        r = super().check(position, yaw, vx, vy, count=count)
        if not count:                       # a measurement (recovery probe), not a drive
            return r
        self._t += PHYSICS_DT               # one drive check per control step (run_episode)
        if r.blocked or r.scale >= 1.0:
            if self._slow_span is not None:
                self.slow_spans.append(tuple(self._slow_span))
                self._slow_span = None
            return r
        self.slowed_checks += 1
        if self._slow_span is None:
            self._slow_span = [round(self._t, 2), round(self._t, 2), round(r.distance_m, 3), r.hit_prim]
        else:
            self._slow_span[1] = round(self._t, 2)
            if r.distance_m < self._slow_span[2]:
                self._slow_span[2], self._slow_span[3] = round(r.distance_m, 3), r.hit_prim
        want = math.hypot(vx, vy)
        got = math.hypot(r.vx, r.vy)
        floor = min(want, self.slow_min_mps)
        if got < floor and got > 1e-9:
            k = floor / got
            self.floored_checks += 1
            return dataclasses.replace(r, vx=r.vx * k, vy=r.vy * k)
        return r

    def telemetry(self) -> dict:
        spans = list(self.slow_spans) + ([tuple(self._slow_span)] if self._slow_span is not None else [])
        return {"slow_min_mps": self.slow_min_mps, "slowed_checks": self.slowed_checks,
                "slowed_s": round(self.slowed_checks * PHYSICS_DT, 2), "floored_checks": self.floored_checks,
                "slow_spans_over_1s": [list(sp) for sp in spans if sp[1] - sp[0] >= 1.0][:40]}


# =============================================================================== recovery
class G1TurnRecovery(StuckRecovery):
    """StuckRecovery's triggers, a TURN IN PLACE for the manoeuvre (module docstring)."""

    def __init__(self, guard: G1Guard, wz: float = G1_RECOVERY_WZ,
                 max_turn_deg: float = G1_RECOVERY_MAX_TURN_DEG, balk_turn_deg: float = G1_BALK_TURN_DEG):
        super().__init__(guard, contact=None)
        self.turn_wz_abs = abs(float(wz))
        self.max_turn = math.radians(max_turn_deg)
        self.balk_turn = math.radians(balk_turn_deg)
        # 2x the time the longest turn takes, + the slew ramp: a robot that cannot rotate (an arm
        # against a wall) gives up and re-decides rather than grinding out the episode.
        self._give_up_s = 2.0 * self.max_turn / self.turn_wz_abs + 1.0
        self._yaw_now = 0.0
        self._turn_wz = self.turn_wz_abs
        self._turned = 0.0
        self._prev_yaw = 0.0
        self._clear0: dict = {}
        self.turn_log: list[dict] = []

    def reset(self) -> None:
        super().reset()
        self.turn_log = []

    def update(self, sim_time, position, yaw, command):
        self._yaw_now = float(yaw)
        return super().update(sim_time, position, yaw, command)

    def front_clearance_m(self, position, yaw: float) -> float:
        """The guard's own fan (so recovery and guard agree on 'blocked'), with G1Guard's telemetry
        restored afterwards: a measurement must not count as interventions or ray hits."""
        g = self.guard
        saved = (dict(g.hits_by_source), g.rays_cast, g.blocked_checks, g.blocked_zero_translation,
                 g.floor_hits_skipped, g.self_hits_stepped, g.self_hits_skipped)
        try:
            return g.check(position, yaw, 1.0, 0.0, count=False).distance_m
        finally:
            (g.hits_by_source, g.rays_cast, g.blocked_checks, g.blocked_zero_translation,
             g.floor_hits_skipped, g.self_hits_stepped, g.self_hits_skipped) = saved

    def _engage(self, mode: str, sim_time: float, position, by_contact: bool = False):
        g = self.guard
        saved = (dict(g.hits_by_source), g.rays_cast, g.floor_hits_skipped, g.self_hits_stepped, g.self_hits_skipped)
        try:
            _, _, wz, clear = g.recovery_turn(position, self._yaw_now, self.turn_wz_abs)
        finally:
            (g.hits_by_source, g.rays_cast, g.floor_hits_skipped, g.self_hits_stepped, g.self_hits_skipped) = saved
        self._turn_wz = wz
        self._turned = 0.0
        self._prev_yaw = self._yaw_now
        self._clear0 = {int(k): (None if v == float("inf") else round(v, 3)) for k, v in clear.items()}
        return super()._engage(mode, sim_time, position, by_contact=by_contact)

    def _reverse(self):
        # Called by StuckRecovery._engage: the manoeuvre's command. Not a reverse for the G1.
        return Command(vx=0.0, vy=0.0, omega=self._turn_wz)

    def _continue(self, sim_time, position, yaw: float, command):
        self._turned += abs(_wrap(yaw - self._prev_yaw))
        self._prev_yaw = yaw
        self._backed_m = math.dist(self._start, (position[0], position[1]))
        clear = self.front_clearance_m(position, yaw)
        stuck = sim_time - self._engage_t > self._give_up_s
        if self._mode == "balk":
            done, why = self._turned >= self.balk_turn, "balk_turned"
        else:
            room = clear > self.guard.stop_distance_m + CLEAR_MARGIN_M
            done, why = (room or self._turned >= self.max_turn), ("room_ahead" if room else "max_turn")
        if stuck:
            self.blocked_behind += 1        # here: could not complete the turn
            why = "gave_up"
        if stuck or done:
            self._active = False
            self._release_t = sim_time
            self._release_pos = (position[0], position[1])
            self._release_mode = self._mode
            self.release_pending = True
            self._history.clear()
            self.turn_log.append({"t0": round(self._engage_t, 2), "t1": round(sim_time, 2), "mode": self._mode,
                                  "wz": round(self._turn_wz, 3), "turned_deg": round(math.degrees(self._turned), 1),
                                  "end": why, "front_clear_m": None if clear == float("inf") else round(clear, 3),
                                  "drift_m": round(self._backed_m, 3), "clearance_at_engage": self._clear0})
            return command, False
        return self._reverse(), True


# =============================================================================== cameras
def _make_head_camera_class():
    """Built after Kit is up: camera_source imports isaacsim.sensors at instantiation time."""
    from camera_source import NavCameraSource

    class _G1HeadCamera(NavCameraSource):
        """NavCameraSource on the G1 head camera, which also records the camera pose of every frame
        it writes (G1Locomotion.camera_pose(spec)["menu_args"]).

        The pose is the one the FRAME shows, not the one at grab time: a frame read after
        kit.update() k shows the pose from before that update (g1 r2's pixel test: 3.1 px with the
        pose snapshotted before the update, 14 px with the pose read at the grab). setup() wraps
        kit.update so snapshot() runs right before every update; grab_to_file() sends that
        snapshot and keeps the grab-time pose next to it as `lag0` for the record."""

        def __init__(self, loco, spec: dict, prim_path: str, scratch_dir):
            self.loco = loco
            self.spec = spec
            self.last: dict | None = None
            self._before_update: dict | None = None
            super().__init__(scratch_dir=scratch_dir, prim_path=prim_path,
                             resolution=tuple(spec["resolution_wh"]))

        def _menu_args(self) -> dict:
            return {k: float(v) for k, v in self.loco.camera_pose(self.spec)["menu_args"].items()}

        def snapshot(self) -> None:
            try:
                self._before_update = self._menu_args()
            except Exception:  # noqa: BLE001  (no pose yet, e.g. before initialize: grab falls back)
                self._before_update = None

        def grab_to_file(self, tag: str = ""):
            path = super().grab_to_file(tag)
            if path is None:
                return None
            now = self._menu_args()
            shown = self._before_update if self._before_update is not None else now
            self.last = {"frame": str(path), **shown, "pose_source": "before_update"
                         if self._before_update is not None else "grab_time", "lag0": now}
            return path

    return _G1HeadCamera


class G1ChaseCamera:
    """Third-person view behind the G1: world-frame camera 2.6 m behind, 1.6 m above the floor,
    looking at the pelvis + 0.85 m, yaw low-passed (the pelvis yaw wobbles with the gait). Re-aimed
    on every grab; the frame read is the one rendered by the previous update (one frame of lag)."""

    def __init__(self, loco, base, prim_path: str = CHASE_PRIM, resolution=CHASE_RES, alpha: float = 0.1):
        from isaacsim.sensors.camera import Camera

        self.loco, self.base, self.alpha = loco, base, alpha
        self._cam = Camera(prim_path=prim_path, resolution=tuple(resolution))
        self._cam.initialize()
        self._yaw = None
        self.aim()

    def aim(self) -> None:
        from pxr import Gf

        x, y, _, yaw = self.loco.pose()
        self._yaw = yaw if self._yaw is None else self._yaw + self.alpha * _wrap(yaw - self._yaw)
        fz = self.base.position()[2]
        # Pull the eye in front of whatever collider is between it and the robot (a turn swings it
        # through the hospital's glass wall otherwise, and the frame shows the parking lot).
        back = self._clear_back(x, y, fz + 1.3, self._yaw, 2.6)
        eye = (x - back * math.cos(self._yaw), y - back * math.sin(self._yaw), fz + 1.3 + 0.3 * back / 2.6)
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(x, y, fz + 0.85), Gf.Vec3d(0, 0, 1)).GetInverse()
        q = m.ExtractRotationQuat()
        im = q.GetImaginary()
        self._cam.set_world_pose(position=np.array(eye, dtype=float),
                                 orientation=np.array([q.GetReal(), im[0], im[1], im[2]]), camera_axes="usd")

    def _clear_back(self, x: float, y: float, z: float, yaw: float, dist: float) -> float:
        """Free distance behind the robot along -heading at height z, minus 0.25 m, in [0.8, dist]."""
        try:
            import carb
            from omni.physx import get_physx_scene_query_interface

            q = get_physx_scene_query_interface()
            d = (-math.cos(yaw), -math.sin(yaw), 0.0)
            o = [x, y, z]
            travelled = 0.0
            for _ in range(6):
                hit = q.raycast_closest(carb.Float3(*o), carb.Float3(*d), dist + 0.3 - travelled)
                if not hit or not hit.get("hit"):
                    return dist
                hd = float(hit.get("distance", 0.0))
                if str(hit.get("collision", "")).startswith("/World/G1"):
                    step = hd + 0.02
                    o = [o[i] + d[i] * step for i in range(3)]
                    travelled += step
                    continue
                return max(0.8, min(dist, travelled + hd - 0.25))
        except Exception:  # noqa: BLE001  (a camera placement must never cost an episode)
            pass
        return dist

    def grab_jpeg(self, quality: int = 85) -> bytes | None:
        import io

        from PIL import Image

        self.aim()
        rgba = self._cam.get_rgba()
        if rgba is None or np.asarray(rgba).size == 0:
            return None
        buf = io.BytesIO()
        Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


# =============================================================================== policy client
class G1PolicyClient(PolicyClient):
    """PolicyClient that adds the G1 `camera` to every /predict (at _post, so it works whether or
    not client.predict() has grown the keyword). A server that does not know the field ignores it
    (pydantic drops extras) -- and then draws AlohaMini's menu geometry, so check /health first."""

    def __init__(self, host: str, port: int, camera_provider=None, timeout: float = 120.0):
        super().__init__(host, port, timeout)
        self.camera_provider = camera_provider
        self.camera_sent = 0
        self.camera_missing = 0

    def _post(self, path, payload=None):
        if path == "/predict" and self.camera_provider is not None:
            payload = dict(payload or {})
            if payload.get("camera") is None:
                cam = self.camera_provider(payload.get("sim_time"))
                if cam is None:
                    self.camera_missing += 1
                else:
                    payload["camera"] = cam
                    self.camera_sent += 1
        return super()._post(path, payload)


# =============================================================================== result
@dataclass
class G1EpisodeResult(EpisodeResult):
    """EpisodeResult + the G1 fields (all defaulted, so every reader of EpisodeResult still loads
    it). Built by run_episode() through run_navigation.EpisodeResult, rebound to this class in a G1
    process (see the SEAMS note); __post_init__ hands it to the live runner to fill in."""

    robot: str = "g1"
    camera: str = ""
    g1: dict = field(default_factory=dict)

    def __post_init__(self):
        runner = G1EpisodeResult.__dict__.get("_runner_ref")
        if runner is not None:
            runner._finish_result(self)

    def summary(self) -> str:
        head = super().summary()
        if self.abort_reason in ("fell", "free_fall", "dt_mismatch", "handle_lost"):
            head = head.replace("[FAILED]", f"[{self.abort_reason.upper()}]", 1)
        g = self.g1 or {}
        sp = g.get("speed_tracking") or {}
        return (head + f"\n  robot            : g1, camera {self.camera}, "
                f"pivots {g.get('controller', {}).get('pivots')}, recovery turns {len(g.get('recovery_turns') or [])}, "
                f"tilt max {g.get('base', {}).get('tilt_max_deg')} deg, "
                f"walk speed {sp.get('achieved_mps')} / {sp.get('commanded_mps')} m/s, "
                f"wall/sim {g.get('cost', {}).get('wall_per_sim_s')}")

    def save(self, directory: Path | str | None = None) -> Path:
        path = super().save(directory or G1EpisodeResult.__dict__.get("_results_dir", RESULTS_DIR))
        G1EpisodeResult._saved = path
        return path


G1EpisodeResult._runner_ref = None
G1EpisodeResult._results_dir = RESULTS_DIR
G1EpisodeResult._saved = None


# =============================================================================== the runner
class G1NavigationRunner(RN.NavigationRunner):
    def __init__(self, episode, camera: str = "p12", policy: str = "qwen", policy_host: str = "127.0.0.1",
                 policy_port: int = 8766, stub_script: str = "exercise", record_root: Path | None = None,
                 headlight: bool = True, speed_cap: float = G1_SPEED_CAP_MPS, yaw_rate: float = G1_YAW_RATE_RADPS,
                 robot_type: str = G1_ROBOT_TYPE, timeout_s: float | None = None, **kw):
        # The episode as the G1 drives it: its own caps and robot type; goal, start, instruction,
        # success threshold untouched (summarize_runs scores by the episode's NAME).
        ep2 = dataclasses.replace(episode, max_speed_mps=float(speed_cap), max_yaw_rate_radps=float(yaw_rate),
                                  robot_type=robot_type,
                                  timeout_s=float(timeout_s) if timeout_s else episode.timeout_s)
        super().__init__(ep2, controller_name=G1_CONTROLLER, policy_host=policy_host,
                         policy_port=policy_port, **kw)
        self.episode_original = episode
        if camera not in CAMERAS:
            raise ValueError(f"--camera must be one of {sorted(CAMERAS)}")
        self.camera_key = camera
        self.policy_kind = policy
        self.headlight = headlight
        self.camera_rows: list[list] = []
        self.loco = None
        if policy == "stub":
            from nav_stub_policy import StubArcPolicy

            self.policy = StubArcPolicy(stub_script, run_root=record_root,
                                        camera_provider=self._camera_payload)
        else:
            self.policy = G1PolicyClient(policy_host, policy_port, camera_provider=self._camera_payload)
        self._stage_md5_before = None
        self._g1_setup: dict = {}
        self._speed = {"cmd_s": 0.0, "dist_m": 0.0, "steps": 0}
        self._prev_xy = None
        self._t_episode0 = None

    # ------------------------------------------------------------------ camera payload
    def _camera_payload(self, sim_time=None) -> dict | None:
        cam = getattr(self.camera, "last", None) if self.camera is not None else None
        if cam is None:
            return None
        # The server's camera keys are CLOSED (arc_menu.MENU_CAMERA_KEYS: an unknown key is a 422 on
        # every /predict), so the wire payload is exactly client.camera_from_spec's; the extra
        # bookkeeping (frame time, which pose was sent) stays in camera_rows.
        out = camera_from_spec(self.cam_spec, {k: round(cam[k], 5) for k in CG_POSE_KEYS},
                               name=self.camera_label, arcs=list(self.curvatures))
        t = None if sim_time is None else round(float(sim_time), 4)
        self.camera_rows.append([t, out["cam_h"], out["pitch_deg"], out["roll_deg"], out["yaw_deg"],
                                 out["cam_x"], out["cam_y"]])
        return out

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        if not self.scene_path.is_file():
            raise FileNotFoundError(f"nav scene not built: {self.scene_path}")
        from isaacsim import SimulationApp

        self.kit = SimulationApp({"headless": self.headless, "active_gpu": self.sim_gpu,
                                  "physics_gpu": self.sim_gpu, "multi_gpu": False})
        import omni.timeline
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        from physics_setup import raise_gpu_capacities

        t_setup = time.time()
        rec = self._g1_setup
        self._stage_md5_before = _md5(self.scene_path)
        ctx = omni.usd.get_context()
        ctx.open_stage(str(self.scene_path.resolve()))
        stage = ctx.get_stage()
        for _ in range(240):
            self.kit.update()
        stage.Load()
        for _ in range(30):
            self.kit.update()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        aloha = stage.GetPrimAtPath("/World/Aloha")
        rec["aloha_deactivated"] = bool(aloha.IsValid())
        if aloha.IsValid():
            aloha.SetActive(False)
        if not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]:
            UsdLux.DomeLight.Define(stage, "/World/G1Dome").CreateIntensityAttr(1000.0)
            rec["dome_added"] = True

        if RN.NAV_SCENE_COLLIDERS:
            import scene_colliders

            t_sc = time.time()
            self.scene_colliders = scene_colliders.add_session_colliders(
                stage, scene_colliders.env_episodes(env_name(self.episode.scene)))
            print(f"[g1nav] scene colliders: {self.scene_colliders.get('authored')} meshes in "
                  f"{time.time() - t_sc:.1f} s", flush=True)
        for line in raise_gpu_capacities(stage):
            print(f"[g1nav] physics: {line}", flush=True)

        self.loco = L.G1Locomotion("/World/G1")
        rec["physics_scene_changes"] = L.configure_physics_scene(stage, self.loco.physics_dt, cpu=True)
        for line in rec["physics_scene_changes"]:
            print(f"[g1nav] physics: {line}", flush=True)
        ep = self.episode
        rec["author"] = {k: v for k, v in self.loco.author(
            stage, (ep.start[0], ep.start[1], ep.start[2] + L.SPAWN_HEIGHT_M), ep.start_yaw_rad).items()
            if k != "colliders"}
        self.cam_spec = dict(self.loco.asset_spec[CAMERAS[self.camera_key]])
        self.camera_label = f"g1_d435_{self.camera_key}_{float(self.cam_spec['pitch_down_deg_rel_head']):g}deg"
        self.curvatures, rec["arc_visibility"] = g1_curvatures(self.cam_spec)
        cam_path = L.author_head_camera(stage, self.loco.prim_path, self.cam_spec, "g1_head_cam")
        rec["head_camera"] = {"prim": cam_path, "label": self.camera_label,
                              "mount_pitch_deg": self.cam_spec["pitch_down_deg_rel_head"],
                              "hfov_deg": self.cam_spec["hfov_deg"], "resolution": self.cam_spec["resolution_wh"],
                              "curvatures": self.curvatures}
        rec["headlight"] = (L.author_head_light(stage, self.loco.prim_path, self.cam_spec)
                            if self.headlight else None)
        ch = UsdGeom.Camera.Define(stage, CHASE_PRIM)
        ch.CreateFocalLengthAttr(16.0)
        ch.CreateHorizontalApertureAttr(20.955)
        ch.CreateVerticalApertureAttr(20.955 * 9 / 16)
        ch.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        self.kit.update()
        from isaacsim.core.simulation_manager import SimulationManager

        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        rec["initialize"] = self.loco.initialize()
        self.art = self.loco.art
        self.base = G1NavBase(self.loco)           # CALLBACK mode
        self.base.reset_to(tuple(ep.start), ep.start_yaw_rad)
        self.contact = None                         # the G1 is a dynamic body: PhysX is its contact model
        self.lidar = None
        HeadCam = _make_head_camera_class()
        self.camera = HeadCam(self.loco, self.cam_spec, cam_path, self.scratch_dir)
        # Snapshot the head pose before EVERY update (the inherited run_episode calls kit.update()
        # directly), so a frame is sent with the pose it actually shows -- see _G1HeadCamera.
        _kit_update = self.kit.update

        def _update_with_pose_snapshot(*a, **kw):
            if self.camera is not None:
                self.camera.snapshot()
            return _kit_update(*a, **kw)

        self.kit.update = _update_with_pose_snapshot
        for _ in range(30):                        # NavCameraSource.warmup, with the robot held
            self.base.apply(0.0, 0.0, 0.0, PHYSICS_DT)
            self.kit.update()
        self.base.physics_state()
        st = self.base.stats()
        rec["warmup_stats"] = {k: st[k] for k in ("step_dt_min_max", "dt_mismatch_steps", "pre_post_steps",
                                                  "faults", "first_fault", "tilt_max_deg")}
        # The 200 Hz contract, checked on the live stage rather than trusted from the authoring call:
        # every physics scene's timeStepsPerSecond and the step_dt PhysX actually hands the callbacks.
        rec["time_steps_per_second"] = {str(p.GetPath()): p.GetAttribute("physxScene:timeStepsPerSecond").Get()
                                        for p in L.physics_scene_prims(stage)}
        if st["dt_mismatch_steps"] or st["step_dt_min_max"] is None or \
                abs(st["step_dt_min_max"][0] - 0.005) > 1e-9 or abs(st["step_dt_min_max"][1] - 0.005) > 1e-9:
            raise RuntimeError(f"physics is not at the G1's 200 Hz: step_dt {st['step_dt_min_max']}, "
                               f"timeStepsPerSecond {rec['time_steps_per_second']}")
        print(f"[g1nav] physics 200 Hz confirmed: step_dt {st['step_dt_min_max']}, tsps "
              f"{rec['time_steps_per_second']}", flush=True)

        self.guard = G1NavGuard(robot_prim_prefix=self.loco.prim_path)
        self.recovery = G1TurnRecovery(self.guard)
        rec["setup_wall_s"] = round(time.time() - t_setup, 1)
        self._set(state="ready", message="waiting for policy server")
        info = self.policy.wait_until_ready()
        self._policy_info = info
        print(f"[g1nav] {RN.ready_message(info)}; camera {self.camera_label}, arcs {self.curvatures}", flush=True)
        self._set(state="idle", message=RN.ready_message(info))

    # ------------------------------------------------------------------ per-episode hooks
    def _spawn_pose(self, ep):
        pose, rec = L.g1_spawn_pose(tuple(ep.start), self.loco.prim_path)
        spawn = {**rec, "robot": "g1", "pelvis_spawn": [round(float(v), 4) for v in pose]}
        if rec["ground_z"] is None:
            print(f"[g1nav] spawn: NO FLOOR under {ep.name}'s start", flush=True)
        else:
            print(f"[g1nav] spawn: {ep.name} episode z {rec['episode_z']:.3f} -> floor {rec['ground_z']:.3f} "
                  f"({'/'.join(rec['ground_prim'].split('/')[-2:])}) -> pelvis z {rec['snapped_z']:.3f}", flush=True)
        # The EPISODE start goes to reset_to(): G1Base.reset_to snaps it itself (g1_spawn_pose).
        return tuple(ep.start), spawn

    def _settle(self) -> dict:
        rec = self.base.settle(self.kit, RN.SETTLE_STEPS, PHYSICS_DT)
        print(f"[g1nav] settle: {rec['steps']} held steps, {'calm' if rec['settled'] else 'NOT calm'}: tilt "
              f"{rec['tilt_deg']:.2f} deg (worst {rec['worst_tilt_deg']:.2f}), |v| {rec['speed_mps']:.3f} m/s"
              + (f"; {rec['faults']} fault(s): {rec['first_fault']}" if rec["faults"] else ""), flush=True)
        # Per-episode counters that run_episode() does not know to clear.
        self.guard.reset_telemetry()
        self.camera_rows = []
        self._speed = {"cmd_s": 0.0, "dist_m": 0.0, "steps": 0}
        self._prev_xy = None
        if G1PivotPursuit.last is not None:
            G1PivotPursuit.last.reset()
        self._t_episode0 = time.time()
        return rec

    def _update_chase(self) -> None:
        if not self.chase_wanted:
            return
        if self._chase is None:
            if self.chase_available is False:
                return
            try:
                self._chase = G1ChaseCamera(self.loco, self.base)
                self.chase_available = True
            except Exception as exc:  # noqa: BLE001  (a missing third-person view never costs an episode)
                self.chase_available = False
                print(f"[g1nav] chase camera unavailable: {exc!r}", flush=True)
                return
        frame = self._chase.grab_jpeg()
        if frame:
            self.chase_jpeg = frame

    def _physics_fault(self, sim_time: float) -> dict | None:
        """The G1's terminal signal, not AlohaMini's rules: G1Base.physics_state()["terminal"]."""
        ps = self.base.physics_state()
        # walking-speed bookkeeping (straight, not holding): achieved planar speed vs command
        pos = self.base.position()
        cmd = self.loco.cmd_target
        if self._prev_xy is not None and not self.base._holding and cmd[0] > 0.25 and abs(cmd[2]) < 0.15:
            self._speed["cmd_s"] += float(cmd[0]) * PHYSICS_DT
            self._speed["want_s"] = self._speed.get("want_s", 0.0) + float(self.base.desired_vx) * PHYSICS_DT
            self._speed["dist_m"] += math.dist(pos[:2], self._prev_xy)
            self._speed["steps"] += 1
        self._prev_xy = pos[:2]
        if not ps.get("terminal"):
            return None
        kinds = []
        for f in self.base.fault_log or ([self.base.last_fault] if self.base.last_fault else []):
            k = FAULT_KIND_NAMES.get(f.get("kind"), f.get("kind"))
            if k not in kinds:
                kinds.append(k)
        if not kinds:
            kinds = ["terminal"]
        fault = {"t": round(sim_time, 3), "kinds": kinds,
                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ps.items()},
                 "g1_fault": self.base.first_fault, "g1_fault_log": list(self.base.fault_log)[:8]}
        # run_episode prints these four; a lost handle leaves them at the last good values
        for k in ("speed_mps", "roll_deg", "pitch_deg", "z"):
            fault.setdefault(k, 0.0)
        return fault

    # ------------------------------------------------------------------ result
    def _finish_result(self, res: G1EpisodeResult) -> None:
        res.robot = "g1"
        res.camera = self.camera_label
        kinds = list((res.physics_fault or {}).get("kinds") or [])
        g1_kinds = [k for k in kinds if k in FAULT_KIND_NAMES.values()]
        if res.abort_reason == "physics_fault" and g1_kinds:
            res.abort_reason = g1_kinds[0]
            res.physics_faults = 1 if any(k in SIM_ERROR_KINDS for k in g1_kinds) else 0
        b = self.base
        if b.zw_max > b.zw_min:
            res.base_z_span_m = round(b.zw_max - b.zw_min, 6)
            res.base_z_step_max_m = round(b.zw_step_max, 6)
        ctrl = G1PivotPursuit.last
        rows = np.array([r[1:5] for r in self.camera_rows], dtype=float) if self.camera_rows else None

        def stat(col):
            if rows is None or not len(rows):
                return None
            a = rows[:, col]
            return {"mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}

        g = self.guard
        sp = self._speed
        wall = time.time() - self._t_episode0 if self._t_episode0 else None
        res.g1 = {
            "base": b.stats(),
            "world_z": {"min": None if b.zw_min == float("inf") else round(b.zw_min, 4),
                        "max": None if b.zw_max == float("-inf") else round(b.zw_max, 4),
                        "step_max": round(b.zw_step_max, 5)},
            "guard": {"class": type(g).__name__, "interventions": g.interventions, "hits_by_source": dict(g.hits_by_source),
                      "rays_cast": g.rays_cast, "blocked_checks": g.blocked_checks,
                      "blocked_zero_translation": g.blocked_zero_translation,
                      "floor_hits_skipped": g.floor_hits_skipped, "self_hits_stepped": g.self_hits_stepped,
                      "self_hits_skipped": g.self_hits_skipped, "stop_m": g.stop_distance_m, "slow_m": g.slow_distance_m,
                      "rings_m": list(g.ring_heights_m), "step_max_m": g.step_max_m, "step_down_max_m": g.step_down_max_m,
                      **(g.telemetry() if hasattr(g, "telemetry") else {})},
            "controller": ctrl.describe() if ctrl is not None else None,
            "recovery_turns": list(getattr(self.recovery, "turn_log", [])),
            "camera": {"label": self.camera_label, "mount_pitch_deg": self.cam_spec["pitch_down_deg_rel_head"],
                       "curvatures": list(self.curvatures), "decisions": len(self.camera_rows),
                       "cam_h_m": stat(0), "pitch_deg": stat(1), "roll_deg": stat(2), "yaw_vs_pelvis_deg": stat(3),
                       "rows": self.camera_rows,
                       "sent": getattr(self.policy, "camera_sent", None),
                       "missing": getattr(self.policy, "camera_missing", None)},
            "speed_tracking": {"steps": sp["steps"],
                               "commanded_mps": round(sp["cmd_s"] / (sp["steps"] * PHYSICS_DT), 3) if sp["steps"] else None,
                               "achieved_mps": round(sp["dist_m"] / (sp["steps"] * PHYSICS_DT), 3) if sp["steps"] else None,
                               "desired_mps": round(sp.get("want_s", 0.0) / (sp["steps"] * PHYSICS_DT), 3) if sp["steps"] else None,
                               "speed_cal": {"on": bool(self.base.speed_cal), "fit": list(G1_VX_FIT),
                                             "cmd_max": G1_VX_CMD_MAX},
                               "rule": "control steps with clamped cmd vx > 0.25, |wz| < 0.15, not holding; "
                                       "commanded = the policy command (after calibration), desired = the request"},
            "cost": {"wall_s": None if wall is None else round(wall, 1),
                     "wall_per_sim_s": (round(wall / res.elapsed_s, 3) if wall and res.elapsed_s else None),
                     "physx_step_ms_mean": b.stats().get("physx_step_ms_mean")},
            "episode_overrides": {"max_speed_mps": self.episode.max_speed_mps,
                                  "max_yaw_rate_radps": self.episode.max_yaw_rate_radps,
                                  "robot_type": self.episode.robot_type, "timeout_s": self.episode.timeout_s,
                                  "timeout_s_original": self.episode_original.timeout_s},
            "policy_kind": self.policy_kind,
            # menus (menu_<step>.jpg), decisions.jsonl and chase_<step>.jpg of this run; None = not recorded
            "record_dir": None if self.record_dir is None else str(self.record_dir),
            "chase_every_steps": RN.CHASE_EVERY,
            "stub": ({"script": self.policy.script_name, "calls": self.policy.calls, "replans": list(self.policy.replans),
                      "draw_errors": self.policy.draw_errors, "camera_missing": self.policy.camera_missing}
                     if self.policy_kind == "stub" else None),
            "setup": self._g1_setup,
            "stage_file": {"path": str(self.scene_path), "md5_before": self._stage_md5_before,
                           "md5_after": _md5(self.scene_path)},
        }
        res.g1["stage_file"]["unchanged"] = res.g1["stage_file"]["md5_after"] == self._stage_md5_before

    def run_config(self) -> dict:
        cfg = super().run_config()
        files = ("nav/sim/run_navigation_g1.py", "nav/sim/run_navigation.py", "nav/sim/g1/locomotion.py",
                 "nav/sim/g1/guard.py", "nav/sim/g1/camera_geometry.py", "nav/sim/g1/walk_policy.py",
                 "nav/sim/g1/nav_stub_policy.py", "nav/sim/stuck_recovery.py", "nav/sim/controllers.py",
                 "nav/sim/collision_guard.py", "nav/policy_server/client.py")
        cfg["robot"] = "g1"
        cfg["fault_rules"] = {
            "terminal": "G1Base.physics_state()['terminal']",
            "fall": {"tilt_deg": L.FALL_TILT_DEG, "pelvis_above_feet_m": L.FALL_ABOVE_FEET_M},
            "free_fall": {"vz_mps": L.FREE_FALL_VZ_MPS, "hold_s": L.FREE_FALL_HOLD_S},
            "physics_dt_s": self.loco.physics_dt if self.loco else None,
            "aloha_rules": "not applied (tilt 30 / speed / vz clamp are KinematicBase's)",
            "judged": "end_of_update"}
        cfg["g1"] = {"camera": self.camera_label, "curvatures": list(self.curvatures),
                     "controller": G1_CONTROLLER, "speed_cap_mps": self.episode.max_speed_mps,
                     "yaw_rate_radps": self.episode.max_yaw_rate_radps,
                     "speed_cal": {"on": bool(G1NavBase.speed_cal), "fit": list(G1_VX_FIT), "cmd_max": G1_VX_CMD_MAX},
                     "code_md5": {f: _md5(REPO / f) for f in files}}
        return cfg


# =============================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description="Run one DynaNav episode with the walking G1.")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--camera", choices=sorted(CAMERAS), default="p12",
                    help="p12 = the D435 head mount re-pitched to 12 deg down (primary); real = 47.6 deg")
    ap.add_argument("--policy", choices=["qwen", "stub"], default="qwen")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=8766)
    ap.add_argument("--stub-script", default="exercise", help="stub: a built-in script name or a JSON file")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="where the result JSON goes (default nav/results; stub runs should NOT go there)")
    ap.add_argument("--record-root", type=Path, default=None,
                    help="stub: run directories (menus, decisions.jsonl, chase frames) go under here")
    ap.add_argument("--scratch-dir", type=Path, default=Path("/tmp/g1nav-frames"))
    ap.add_argument("--speed-cap", type=float, default=G1_SPEED_CAP_MPS)
    ap.add_argument("--yaw-rate", type=float, default=G1_YAW_RATE_RADPS)
    ap.add_argument("--lookahead", type=float, default=G1_LOOKAHEAD_M)
    ap.add_argument("--pivot-deg", type=float, default=G1_PIVOT_ENTER_DEG)
    ap.add_argument("--pivot-exit-deg", type=float, default=G1_PIVOT_EXIT_DEG)
    ap.add_argument("--timeout", type=float, default=None, help="override the episode's timeout (sim s)")
    ap.add_argument("--robot-type", default=G1_ROBOT_TYPE)
    ap.add_argument("--no-headlight", action="store_true")
    ap.add_argument("--no-speed-cal", action="store_true",
                    help="send the raw forward speed (no G1_VX_FIT inversion; the pre-2026-10-11 behaviour)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.policy == "stub" and args.results_dir is None:
        raise SystemExit("--policy stub needs --results-dir: a stub run must not land in nav/results")
    # The three process-local rebinds the SEAMS note lists.
    controllers.CONTROLLERS[G1_CONTROLLER] = functools.partial(
        G1PivotPursuit, lookahead_m=args.lookahead, pivot_enter_deg=args.pivot_deg,
        pivot_exit_deg=args.pivot_exit_deg)
    RN.EpisodeResult = G1EpisodeResult
    if args.results_dir is not None:
        G1EpisodeResult._results_dir = Path(args.results_dir)

    G1NavBase.speed_cal = not args.no_speed_cal
    ep = load_episode(args.episode)
    record_root = args.record_root or (Path(args.results_dir) / "runs" if args.results_dir else None)
    runner = G1NavigationRunner(ep, camera=args.camera, policy=args.policy, policy_host=args.policy_host,
                                policy_port=args.policy_port, stub_script=args.stub_script,
                                record_root=record_root, headlight=not args.no_headlight,
                                speed_cap=args.speed_cap, yaw_rate=args.yaw_rate, robot_type=args.robot_type,
                                timeout_s=args.timeout, scratch_dir=args.scratch_dir, sim_gpu=args.gpu)
    G1EpisodeResult._runner_ref = runner
    status, success = "error", False
    try:
        runner.setup()
        result = runner.run_episode(args.instruction)
        print("\n" + result.summary(), flush=True)
        status, success = "ok", bool(result.success)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        status = f"error: {exc!r}"
    finally:
        print(f"[g1nav] STATUS={'OK' if status == 'ok' else 'FAILED'} success={success} "
              f"result={G1EpisodeResult._saved} ({status})", flush=True)
        sys.stdout.flush()
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:  # noqa: BLE001
            pass
        runner.close()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
