#!/usr/bin/env python3
"""A `Base` with no hardware behind it. For the hour before you trust the loop.

    PrintBase()     integrates the commands it is given and prints them. Nothing moves.

This is the default in `robot/run_robot.py`, deliberately: the first run of any new
integration should answer "does the camera work, does the policy answer, do the commands
point where I expect" without a robot in the room contributing its own failure modes.

It also serves a second purpose that is easy to miss. Its `odometry()` is PERFECT --
exactly the commanded velocity, integrated -- so a run against it is the closest thing to
the simulator you can get without Isaac. If the robot behaves differently against your
real base, the difference is your odometry or your wheels, and you now know which half to
look at.
"""

from __future__ import annotations

import math
import time


class PrintBase:
    """Integrates commands into a pose. Optionally prints them."""

    def __init__(self, verbose: bool = True, print_every_s: float = 1.0):
        self.verbose = verbose
        self.print_every_s = print_every_s
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.commands = 0
        self.stops = 0
        self._last_t = time.monotonic()
        self._last_print = 0.0

    def drive(self, vx: float, vy: float, omega: float) -> None:
        now = time.monotonic()
        dt = min(now - self._last_t, 0.5)   # a long stall must not teleport the pose
        self._last_t = now
        self.commands += 1

        # Integrated in the body frame at the START of the tick, i.e. the same
        # first-order integration the simulator's kinematic base uses. At 20 Hz and
        # 0.25 m/s the error against an exact arc is well under a millimetre per tick,
        # and this pose is only ever read as a difference.
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (vx * cos_y - vy * sin_y) * dt
        self.y += (vx * sin_y + vy * cos_y) * dt
        self.yaw = (self.yaw + omega * dt + math.pi) % (2 * math.pi) - math.pi

        if self.verbose and now - self._last_print >= self.print_every_s:
            self._last_print = now
            print(f"[base] vx={vx:+.2f} omega={omega:+.2f} | "
                  f"x={self.x:+.2f} y={self.y:+.2f} yaw={math.degrees(self.yaw):+.0f}deg",
                  flush=True)

    def stop(self) -> None:
        self.stops += 1
        self.drive(0.0, 0.0, 0.0)

    def odometry(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)
