#!/usr/bin/env python3
"""Adapts your `RangeSensor` to the shape `nav/sim/collision_guard.py` already reads.

`CollisionGuard` was written for the simulator but is not coupled to it: it imports Isaac
lazily, inside the raycast path only, and its `scan=` mode never reaches that path. So on
a real robot the guard is not reimplemented -- it is the same file, the same fan geometry
and the same intervention counter that produced every guard number in `nav/results`,
reading your sensor instead of a physics scene.

What it wants is an object with `points_body(position, yaw)` and `prims()`. That is a
simulator-shaped interface, and the two halves of the mismatch are worth naming:

  - `points_body` takes a world pose because the simulator's lidar accumulates returns in
    the world frame across a sweep. A real sensor hands you body-frame points already, so
    the arguments are accepted and ignored. They are not dropped from the signature,
    because then this would no longer be the interface the guard calls.
  - `prims()` exists so a blocked guard can name the thing it stopped for -- a USD path,
    in the simulator. Here there is nothing to name, so every point reports the same
    label, and a guard message says "blocked by scan at 0.62 m" instead of naming a desk.
"""

from __future__ import annotations

import math

import numpy as np

# What `prims()` reports. Chosen to read correctly in the guard's own message, which
# formats it as `hit_prim.split('/')[-1]`.
_LABEL = "scan"


class ScanBuffer:
    """Holds the most recent sweep and presents it the way the guard expects.

    Holds rather than polls: the control loop reads the sensor once per tick and pushes
    the result in, so the guard and the policy call see the SAME sweep. Letting each
    consumer poll independently would let the guard clip a velocity against one sweep
    while the server drops arcs against another, and the two disagreeing is the kind of
    fault that shows up as "it stopped for no reason" once an hour.
    """

    def __init__(self, chassis_radius_m: float = 0.35, max_range_m: float = 12.0):
        self.chassis_radius_m = chassis_radius_m
        self.max_range_m = max_range_m
        self._points = np.zeros((0, 2), dtype=float)
        self.dropped_self_hits = 0
        self.updates = 0

    def update(self, points: list[tuple[float, float]] | None) -> None:
        """Push one sweep of body-frame (x_forward, y_left) metres. None keeps the last."""
        if points is None:
            return
        self.updates += 1
        if not points:
            self._points = np.zeros((0, 2), dtype=float)
            return

        arr = np.asarray(points, dtype=float).reshape(-1, 2)
        dist = np.hypot(arr[:, 0], arr[:, 1])
        # Self-hits are dropped HERE and only here, so both consumers get the same
        # filtering. Nothing downstream can tell a return off your own bumper from a wall
        # 20 cm away, and a robot guarding against itself simply never moves -- which
        # presents as the policy refusing to drive, not as a sensor problem.
        keep = (dist > self.chassis_radius_m) & (dist <= self.max_range_m)
        # NaN and inf are what a real driver reports for "no return on this bearing", and
        # they survive every comparison above as False-ish in ways that are easy to get
        # wrong; drop them explicitly.
        keep &= np.isfinite(dist)
        self.dropped_self_hits += int(np.count_nonzero(dist <= self.chassis_radius_m))
        self._points = arr[keep]

    # --- the interface CollisionGuard calls ---------------------------------------
    def points_body(self, position=None, yaw: float = 0.0) -> np.ndarray:
        """Body-frame points. `position`/`yaw` are the simulator's; ignored here."""
        return self._points

    def prims(self) -> list[str]:
        return [_LABEL] * len(self._points)

    # --- what the policy server wants ---------------------------------------------
    def scan_points(self) -> list[list[float]] | None:
        """The same sweep, as the `scan_points` field of a /predict call.

        None rather than [] when nothing has been received: an empty list is a positive
        claim that the robot is surrounded by free space, and the server would use it to
        mark every arc drivable. Never having heard from a sensor is a different fact.
        """
        if self.updates == 0:
            return None
        return self._points.tolist()

    def forward_clearance_m(self, half_angle_deg: float = 20.0) -> float:
        """Nearest return within a wedge straight ahead. For logging and for bring-up."""
        if len(self._points) == 0:
            return float("inf")
        rel = np.arctan2(self._points[:, 1], self._points[:, 0])
        keep = np.abs(rel) <= math.radians(half_angle_deg)
        if not np.any(keep):
            return float("inf")
        pts = self._points[keep]
        return float(np.min(np.hypot(pts[:, 0], pts[:, 1])))
