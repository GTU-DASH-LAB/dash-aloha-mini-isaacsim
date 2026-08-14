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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# DynaNav's tuned values, read off behavior/nova_carter_test_ticvla.py:1372-1375.
# Do not "round these off" -- they are the published operating point.
DYNANAV_LOOKAHEAD_M = 1.0
DYNANAV_K_ANGULAR = 0.8
DYNANAV_ALPHA_FILTER = 0.35
_EPS = 1e-3


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
    ):
        self.v_max = max_speed_mps
        self.w_max = max_yaw_rate_radps
        self.lookahead_m = lookahead_m
        self.k_angular = k_angular
        self.alpha_filter = alpha_filter
        self._yaw_err_filt: float | None = None

    def reset(self) -> None:
        self._yaw_err_filt = None

    def __call__(self, waypoints: np.ndarray) -> Command:
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
        v_cmd = float(np.clip(min(self.v_max, v_kappa), 0.0, self.v_max))

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

    def __call__(self, waypoints: np.ndarray) -> Command:
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


CONTROLLERS = {
    PursuitController.name: PursuitController,
    HolonomicController.name: HolonomicController,
}


def make_controller(name: str, max_speed_mps: float, max_yaw_rate_radps: float, **kwargs):
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; have: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](max_speed_mps, max_yaw_rate_radps, **kwargs)
