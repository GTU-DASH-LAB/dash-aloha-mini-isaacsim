"""How far the robot has turned and driven since the instruction was given.

The arc-menu selector is told the WORDS of its recent choices ("hard left, hard left, ...")
and nothing else about its own motion. That is not enough to know when a commanded turn is
finished, and the failure is concrete: on the hospital tour (2026-11-26, step 3, "Turn left
and drive to the foot of the staircase.") it chose hard left on 52 of 52 decisions and the
robot turned 1001 degrees in 73 s -- almost three circles around the same 3 m patch of floor
-- while its own description said "the most open, walkable floor is straight ahead" on a
third of those decisions. `STAGE_RULE` tells it to do "the earliest move not yet done", and
nothing it was shown could ever make "turn left" done.

A real robot has this number for free (wheel odometry / IMU yaw), so handing it over is not
privileged information the way a goal bearing would be. This module keeps it:

    turned_deg      signed yaw change since `reset()`, unwrapped (+ = left, as yaw)
    net_m           straight-line distance from where the instruction was given
    path_m          distance driven since then
    since_s         sim seconds since then
    recent_turn_deg signed yaw change over the last `window_s` -- the spin detector's input.
                    The cumulative number cannot serve: after one accidental circle it stays
                    at 360 for the rest of the instruction, and a detector keyed on it would
                    forbid every later turn that way.

Fed every physics step (`update`), because yaw is unwrapped from successive differences and
a difference taken across a decision period (0.5 s at up to 1.2 rad/s) is still well inside
+-180 deg -- but per-step costs nothing and leaves no aliasing question to argue about.
"""
from __future__ import annotations

import math
from collections import deque


class TurnOdometer:
    def __init__(self, window_s: float = 30.0, sample_s: float = 0.5):
        self.window_s = float(window_s)
        self.sample_s = float(sample_s)
        self.reset(0.0, 0.0, 0.0, 0.0)

    def reset(self, t: float, x: float, y: float, yaw: float) -> None:
        self.t0, self.x0, self.y0 = float(t), float(x), float(y)
        self.t, self.x, self.y, self.yaw = float(t), float(x), float(y), float(yaw)
        self.turned = 0.0          # radians, unwrapped
        self.path = 0.0
        # (t, turned) samples every `sample_s`, enough to read the windowed turn back.
        self._hist: deque[tuple[float, float]] = deque([(self.t, 0.0)])

    def update(self, t: float, x: float, y: float, yaw: float) -> None:
        d = math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw))
        self.turned += d
        self.path += math.hypot(x - self.x, y - self.y)
        self.t, self.x, self.y, self.yaw = float(t), float(x), float(y), float(yaw)
        if t - self._hist[-1][0] >= self.sample_s:
            self._hist.append((self.t, self.turned))
        # Keep one sample at or before the window's start so the window is always covered.
        while len(self._hist) > 2 and self._hist[1][0] <= t - self.window_s:
            self._hist.popleft()

    def recent_turn(self) -> tuple[float, float]:
        """(signed radians turned over the window, seconds the window actually covers)."""
        t_old, turned_old = self._hist[0]
        return self.turned - turned_old, self.t - t_old

    def payload(self) -> dict:
        """What /predict carries as `odometry`. Rounded: it is read by a language model."""
        rt, rs = self.recent_turn()
        return {
            "turned_deg": round(math.degrees(self.turned), 1),
            "net_m": round(math.hypot(self.x - self.x0, self.y - self.y0), 2),
            "path_m": round(self.path, 2),
            "since_s": round(self.t - self.t0, 1),
            "recent_turn_deg": round(math.degrees(rt), 1),
            "recent_s": round(rs, 1),
        }
