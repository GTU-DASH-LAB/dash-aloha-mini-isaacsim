"""Turning TIC-VLA's waypoint plan into base velocity commands.

TIC-VLA emits a (T, 2) array of body-frame FLU displacements: +x forward, +y left.
It does NOT emit velocities, so something has to close the gap between "here is a
path" and "here is what the wheels do". That something is a controller, and which
controller is correct depends on the robot's kinematics -- which is exactly where
AlohaMini and DynaNav's Nova Carter part company.

  Nova Carter is DIFFERENTIAL DRIVE. It cannot move sideways. A (dx, dy) path must
  collapse to (v, omega), and lateral offset is expressed by turning. DynaNav does
  this with pure pursuit.

  AlohaMini is a 3-wheel HOLONOMIC omni base. It can translate in any direction
  without rotating at all. Collapsing (dx, dy) to (v, omega) throws away a degree of
  freedom the robot actually has.

So this module ships both:

  PursuitController   -- a faithful port of DynaNav's pure pursuit, same gains. This
                         is the parity baseline: it answers "does AlohaMini reproduce
                         the published behaviour when driven the published way?"
  HolonomicController -- uses the omni base properly.

Keeping both is the point. If only the holonomic one existed, any difference from
DynaNav's published numbers would be unattributable -- policy? robot? controller?
Running both over the same episodes isolates the controller as the variable.

A trap worth stating explicitly, because it is not obvious and it shaped the
holonomic design: **full holonomy is dangerous for a vision-language policy.** The
policy sees the world through a forward-facing camera and was trained on a robot
whose camera always points where it is going. If AlohaMini crabs sideways while
facing a fixed direction, the next observation no longer corresponds to the
direction of travel, and the policy plans from a view that contradicts its own
motion model. So HolonomicController still slews yaw toward the path -- it just
does not have to *wait* for the rotation before making progress, which is the actual
win over differential drive.

**That trap is real and `yaw_align` does not defuse it. Measured, not argued.**
Same episode, same policy, controller the only variable:

    controller   rightward yaw by t=21 s   closest approach to goal
    holonomic     4.9 / 2.1 deg            6.06 / 6.98 m
    pursuit      11.8 / 12.5 deg           5.77 / 6.39 m

Four full runs, all on the benchmark Aisle-05 instruction. Yaw is measured over the
approach window (t = 0-21 s) only -- what the robot accumulates after it has passed
the aisle mouth is wandering, not steering. Pursuit turns ~2.7x as far (mean 12.2 vs
3.5 deg) and does it consistently; one holonomic run turned the WRONG way, reaching
105.9 deg. Note the closest-approach gain is modest (mean 6.08 vs 6.52 m), which is
the honest shape of this result: the deficit is 25 deg and the controller is worth
about 9 of them.

The mechanism: TIC-VLA expresses "the target is off to your right" as a small
lateral offset, because on the differential-drive robot it was trained on, lateral
offset can ONLY be satisfied by rotating. HolonomicController satisfies the same
offset by translating, so the offset is discharged as sideways drift, the heading
error is driven back to ~0 before the next replan, and the camera barely rotates.
The next frame therefore looks nearly the same, the policy asks for the same small offset
again, and the loop that was supposed to converge converges far more slowly. On
identical plans pursuit produces ~1.7x the yaw rate (1.7 vs 1.0 deg/s at a 1.2 deg
plan; 27.4 vs 16.0 deg/s at 20 deg).

Prefer `pursuit` unless you are specifically measuring the omni base. The lateral
DOF is still worth having; `collision_guard.py` uses it to slide along obstacles,
where there is no policy in the loop to confuse.

**What pure pursuit does NOT fix**, so nobody re-runs this experiment: on the
warehouse episode the policy asked for a right turn greater than 5 deg in 2 of 129
calls, mean +1.1 deg, while the bearing it needed ran from -24.9 to -80 deg. Fixing
the controller recovers the turn the policy asks for. It cannot invent one.

That sentence is true and it is also where the third controller comes from. "The
turn the policy asks for" was measured on the ACTION HEAD, and the action head is
only 3 s long. The policy has a second output that reaches 9 s and asks for
19 deg -- see `guidance.py`, which decodes it. `GuidedPursuitController` steers on
that instead. It is not a better tracker; it is the same tracker reading a channel
that had been thrown away as display text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from guidance import guidance_heading

# DynaNav's tuned values, read off behavior/nova_carter_test_ticvla.py:1372-1375.
# Do not "round these off" -- they are the published operating point.
DYNANAV_LOOKAHEAD_M = 1.0
DYNANAV_K_ANGULAR = 0.8
DYNANAV_ALPHA_FILTER = 0.35
_EPS = 1e-3

# The action head emits exactly `action_horizon_steps: int = 30` waypoints
# (ticvla/training/config.py:37) at 10 Hz. So every plan is 3.0 s of motion, always,
# and the arc length of one is therefore a SPEED the policy is requesting.
ACTION_HORIZON_S = 3.0


def plan_speed(waypoints: np.ndarray) -> float:
    """How fast the policy is asking to go, m/s. Arc length over a fixed 3.0 s.

    This channel was being thrown away, and warehouse_aisle6 is what that costs.
    That episode closed 92% of a 34.3 m gap, reached 2.80 m -- 1.3 m short of the
    1.5 m success threshold -- and then sailed straight past to 22.0 m. Reading the
    plans back shows the policy braking hard exactly where it should:

        t (s)   dist to goal   plan reach
        101.5       4.14 m       1.03 m
        103.0       3.37 m       0.71 m
        105.5       2.80 m       0.65 m     <- closest approach
        107.0       3.10 m       0.52 m
        109.0       4.32 m       0.35 m

    A 0.65 m plan over 3.0 s is a request for 0.22 m/s. We drove 0.6 -- nearly 3x
    the commanded speed -- because pure pursuit sets `v_cmd = min(v_max, v_kappa)`
    and neither term knows how long the plan is. The robot could not stop where the
    policy was trying to stop it.

    DynaNav has the same blind spot and it costs them nothing, which is worth
    understanding before copying them: their episode TERMINATES the moment the robot
    is within 1.5 m, so overshooting after that point is unscored. Ours terminates
    on the same rule -- but only if it gets inside 1.5 m, and at 2.80 m it never
    did. Parity with DynaNav's controller is not the same as parity with DynaNav's
    scoring harness.
    """
    if len(waypoints) < 2:
        return 0.0
    inc = np.diff(waypoints, axis=0)
    return float(np.sum(np.hypot(inc[:, 0], inc[:, 1])) / ACTION_HORIZON_S)


@dataclass
class Command:
    """Body-frame velocity command. vy is always 0 for differential drive."""

    vx: float
    vy: float
    omega: float

    def is_stopped(self, tol: float = 1e-4) -> bool:
        return abs(self.vx) < tol and abs(self.vy) < tol and abs(self.omega) < tol


def _lookahead_point(waypoints: np.ndarray, lookahead_m: float) -> tuple[float, float, float]:
    """Pick the point ~`lookahead_m` along the path, DynaNav's way.

    Returns (x, y, distance_from_robot).
    """
    n = len(waypoints)
    inc = np.diff(waypoints, axis=0)
    seg = np.hypot(inc[:, 0], inc[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    j = int(np.searchsorted(arc, lookahead_m, side="left"))
    # DynaNav clips to [2, T-3]: never steer at the first couple of points (they sit
    # almost on top of the robot, so their heading is numerically meaningless) and
    # never at the very last (the tail of a generated chunk is the least reliable).
    j = int(np.clip(j, min(2, n - 1), max(n - 3, 0)))

    x, y = float(waypoints[j, 0]), float(waypoints[j, 1])
    return x, y, float(math.hypot(x, y))


class PursuitController:
    """Pure pursuit, ported from DynaNav. Differential-drive: vy is always 0."""

    name = "pursuit"

    def __init__(
        self,
        max_speed_mps: float,
        max_yaw_rate_radps: float,
        lookahead_m: float = DYNANAV_LOOKAHEAD_M,
        k_angular: float = DYNANAV_K_ANGULAR,
        alpha_filter: float = DYNANAV_ALPHA_FILTER,
        obey_plan_speed: bool = False,
    ):
        self.v_max = max_speed_mps
        self.w_max = max_yaw_rate_radps
        self.lookahead_m = lookahead_m
        self.k_angular = k_angular
        self.alpha_filter = alpha_filter
        # Off here, on in GuidedPursuitController. This class is the DynaNav parity
        # baseline and DynaNav does not read plan length, so turning it on by default
        # would quietly stop this being a baseline. See `plan_speed`.
        self.obey_plan_speed = obey_plan_speed
        self._yaw_err_filt: float | None = None

    def reset(self) -> None:
        self._yaw_err_filt = None

    def __call__(self, waypoints: np.ndarray, guidance: np.ndarray | None = None) -> Command:
        # `guidance` is accepted and ignored. This is the DynaNav parity baseline and
        # DynaNav does not read the text head, so using it here would quietly stop
        # this controller being the thing it exists to be.
        if len(waypoints) < 2:
            return Command(0.0, 0.0, 0.0)

        x_l, y_l, dist = _lookahead_point(waypoints, self.lookahead_m)
        if dist < _EPS:
            # The plan is degenerate (all points on top of the robot). This is what a
            # "stop" instruction produces -- see plan.md Phase 9, where "Stop. Do not
            # move." collapsed path length to ~0.4 against ~1.3 for everything else.
            return Command(0.0, 0.0, 0.0)

        yaw_err = math.atan2(y_l, x_l)

        # Low-pass the heading error. Without this the command chatters, because each
        # replan produces a fresh plan whose first points differ slightly.
        if self._yaw_err_filt is None:
            self._yaw_err_filt = yaw_err
        wrapped = math.atan2(
            math.sin(yaw_err - self._yaw_err_filt), math.cos(yaw_err - self._yaw_err_filt)
        )
        self._yaw_err_filt += self.alpha_filter * wrapped

        kappa = 2.0 * y_l / (dist * dist)  # pure-pursuit curvature
        v_kappa = self.w_max / (abs(kappa) + _EPS)  # slow down for sharp turns
        v_cap = min(self.v_max, plan_speed(waypoints)) if self.obey_plan_speed else self.v_max
        v_cmd = float(np.clip(min(v_cap, v_kappa), 0.0, self.v_max))

        w_ff = 0.5 * v_cmd * kappa
        w_fb = self.k_angular * self._yaw_err_filt
        w_cmd = float(np.clip(w_ff + w_fb, -self.w_max, self.w_max))

        return Command(vx=v_cmd, vy=0.0, omega=w_cmd)


class HolonomicController:
    """Tracks the waypoint path directly, using the omni base's lateral freedom.

    Differences from pure pursuit, and why:

    1. The velocity vector points AT the lookahead point instead of only along +x.
       A differential-drive robot has to rotate before a lateral offset becomes
       progress; this one does not, so a sidestep costs nothing.
    2. Yaw is slewed toward the path heading rather than being the mechanism of
       travel. `yaw_align` sets how hard: 1.0 keeps the camera pinned to the
       direction of travel (closest to what the policy expects), 0.0 lets the robot
       crab freely (fastest, but the camera stops agreeing with the motion -- see
       this module's docstring).
    3. Speed is not reduced for curvature. Pure pursuit slows in turns because
       turning IS how it makes lateral progress; here the two are decoupled.
    """

    name = "holonomic"

    def __init__(
        self,
        max_speed_mps: float,
        max_yaw_rate_radps: float,
        lookahead_m: float = DYNANAV_LOOKAHEAD_M,
        yaw_align: float = 0.8,
        alpha_filter: float = DYNANAV_ALPHA_FILTER,
    ):
        self.v_max = max_speed_mps
        self.w_max = max_yaw_rate_radps
        self.lookahead_m = lookahead_m
        self.yaw_align = yaw_align
        self.alpha_filter = alpha_filter
        self._heading_filt: float | None = None

    def reset(self) -> None:
        self._heading_filt = None

    def __call__(self, waypoints: np.ndarray, guidance: np.ndarray | None = None) -> Command:
        if len(waypoints) < 2:
            return Command(0.0, 0.0, 0.0)

        x_l, y_l, dist = _lookahead_point(waypoints, self.lookahead_m)
        if dist < _EPS:
            return Command(0.0, 0.0, 0.0)

        heading = math.atan2(y_l, x_l)
        if self._heading_filt is None:
            self._heading_filt = heading
        wrapped = math.atan2(
            math.sin(heading - self._heading_filt), math.cos(heading - self._heading_filt)
        )
        self._heading_filt += self.alpha_filter * wrapped

        # Drive straight at the lookahead point. Unit direction * full speed.
        vx = self.v_max * (x_l / dist)
        vy = self.v_max * (y_l / dist)

        # Turn to face the way we are going, at a rate proportional to how far off we
        # are -- but travel is already happening regardless of whether the turn has
        # finished. That is the whole advantage of the omni base.
        omega = float(np.clip(self.yaw_align * self._heading_filt, -self.w_max, self.w_max))

        return Command(vx=float(vx), vy=float(vy), omega=omega)


class GuidedPursuitController(PursuitController):
    """Pure pursuit steered by the 9 s text head instead of the 3 s action head.

    Same tracker, same gains, one substitution: the heading reference comes from
    the guidance waypoint at `guidance_horizon_s` (see `guidance.py`) rather than
    from a 1 m lookahead into a plan that is only ~1.6 m long.

    Speed is still set by the action head, and that split is the design:

      near field (action head)  -- how fast to go. `v_kappa` exists to slow the
                                   robot when the path immediately in front of it
                                   bends. That is a question about the next metre,
                                   and the action head is the dense, reliable
                                   authority on the next metre.
      far field (text head)     -- which way to point. That is a question about
                                   the next six seconds, and the action head
                                   cannot answer it because it stops at three.

    Falls back to exact `PursuitController` behaviour whenever guidance is absent
    -- no `<answer>`, or the -100 "no guidance" sentinel, which the model emits
    legitimately and often. The fallback shares the same filter state, so a
    dropout is a change of reference, not a reset.

    Honest about what this is NOT: it does not make the robot track a better path,
    and it cannot help on episodes that are already straight. It only matters when
    the action head's 3 s window truncates a turn that the policy has in fact
    planned. On the Aisle-05 start frame that is the difference between steering
    -6.5 deg and steering -18.9 deg, against -24.9 deg needed.
    """

    name = "guided"

    def __init__(
        self,
        *args,
        guidance_horizon_s: float = 6.0,
        obey_plan_speed: bool = True,
        **kwargs,
    ):
        super().__init__(*args, obey_plan_speed=obey_plan_speed, **kwargs)
        self.guidance_horizon_s = guidance_horizon_s
        # Set every call so the runner can log how often guidance was actually
        # available. A "guided" run where this is mostly False is a pursuit run,
        # and reporting it as guided would be a measurement error.
        self.last_used_guidance = False

    def reset(self) -> None:
        super().reset()
        self.last_used_guidance = False

    def __call__(self, waypoints: np.ndarray, guidance: np.ndarray | None = None) -> Command:
        heading = guidance_heading(guidance, self.guidance_horizon_s)
        if heading is None:
            self.last_used_guidance = False
            return super().__call__(waypoints)

        if len(waypoints) < 2:
            return Command(0.0, 0.0, 0.0)
        x_l, y_l, dist = _lookahead_point(waypoints, self.lookahead_m)
        if dist < _EPS:
            # The action head says "stop". Guidance does not override a stop: the
            # only instruction that produces a degenerate plan is one that asked
            # the robot to hold still, and driving anyway would disobey it.
            return Command(0.0, 0.0, 0.0)

        self.last_used_guidance = True

        if self._yaw_err_filt is None:
            self._yaw_err_filt = heading
        wrapped = math.atan2(
            math.sin(heading - self._yaw_err_filt), math.cos(heading - self._yaw_err_filt)
        )
        self._yaw_err_filt += self.alpha_filter * wrapped

        kappa = 2.0 * y_l / (dist * dist)  # near-field curvature, action head
        v_kappa = self.w_max / (abs(kappa) + _EPS)
        v_cap = self.v_max
        if self.obey_plan_speed:
            # Honour the deceleration the policy is asking for. See `plan_speed`:
            # the request is the plan's arc length over its fixed 3.0 s horizon, and
            # ignoring it is what let aisle6 sail past its goal at 2.80 m.
            v_cap = min(v_cap, plan_speed(waypoints))
        v_cmd = float(np.clip(min(v_cap, v_kappa), 0.0, self.v_max))

        w_ff = 0.5 * v_cmd * kappa
        w_fb = self.k_angular * self._yaw_err_filt
        w_cmd = float(np.clip(w_ff + w_fb, -self.w_max, self.w_max))

        return Command(vx=v_cmd, vy=0.0, omega=w_cmd)


CONTROLLERS = {
    PursuitController.name: PursuitController,
    HolonomicController.name: HolonomicController,
    GuidedPursuitController.name: GuidedPursuitController,
}


def make_controller(name: str, max_speed_mps: float, max_yaw_rate_radps: float, **kwargs):
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; have: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](max_speed_mps, max_yaw_rate_radps, **kwargs)
