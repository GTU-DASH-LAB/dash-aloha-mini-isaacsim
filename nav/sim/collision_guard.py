"""Raycast guard: stop the base before it drives through a wall.

Why this has to exist at all, and why it is a guard rather than physics:

AlohaMini's base is driven KINEMATICALLY -- ../../CLAUDE.md documents the whole
investigation (a genuine collision bug was found and fixed, per-wheel sphere
colliders and a friction material were added, velocity-drive damping was retuned)
after which a single wheel spinning in isolation still produced ~200x less
translation than expected. The pre-approved fallback was to teleport the root pose
with set_world_poses() each step.

The consequence is easy to state and easy to forget: **a teleported body does not
collide.** PhysX is not integrating forces on it, so it will pass straight through
walls, shelving and pedestrians without either body reacting. A navigation demo that
looked great because the robot ghosted through a rack would be worthless.

So obstacles are handled by explicitly querying the physics scene before each move.
This is genuinely weaker than contact physics and should be described that way: it
stops the robot, it does not push back, and it only sees what the ray fan samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    distance_m: float
    vx: float  # body-frame velocity to actually apply -- already guarded
    vy: float
    hit_prim: str = ""
    scale: float = 1.0  # speed factor from the slow band, for telemetry only


class CollisionGuard:
    """A fan of forward raycasts, sampled in the direction of travel.

    The fan matters: a single centre ray misses a shelf leg the chassis is about to
    clip, because the ray passes cleanly to one side of it.

    Blocking is DIRECTIONAL: a hit cancels the part of the commanded velocity pointing
    into it and leaves the rest. The first version scaled the whole velocity to zero
    whenever any ray -- including one aimed 35 degrees off the direction of travel --
    came back short, which made the guard strictly less physical than the contact it
    substitutes for. A real wall stops you along its normal and lets you slide; that one
    stopped you dead because something was near your shoulder.

    It deadlocked for real. At the Aisle 05 entrance the robot hugs a rack endcap, an
    outer ray sits in the racking, and translation goes to zero -- while the plan says
    "straight ahead", so the controller's omega (yaw_align * heading_error) is also ~0.
    Neither driving nor turning, with the centre path clear: 2734 of 4201 steps stopped,
    two thirds of the episode spent frozen beside an aisle it could have driven into.
    """

    def __init__(
        self,
        stop_distance_m: float = 0.6,
        slow_distance_m: float = 1.5,
        fan_half_angle_deg: float = 35.0,
        fan_rays: int = 7,
        ray_height_m: float = 0.30,
        chassis_radius_m: float = 0.35,
        robot_prim_prefix: str = "/World/Aloha",
        scan=None,
    ):
        # `scan` is an optional SweepingLidar2D. With it the guard reads the sensor's
        # rolling buffer instead of casting its own fan; without it nothing changes at
        # all. Kept as a switch rather than a replacement because every number this repo
        # has recorded -- 2734 frozen steps, 956 -> 195 interventions, the whole
        # 13-episode ladder -- was measured on the fan, and swapping the input silently
        # would make the next ladder incomparable to all of them.
        #
        # WHAT THE SENSOR BUYS IS DENSITY, and it is worth stating as a number because it
        # is the entire argument. Seven rays across +-35 degrees are 11.7 degrees apart:
        # at the 0.6 m stop distance that is a 12 cm gap between adjacent rays, and 30 cm
        # at 1.5 m. A chair leg, a table leg or a forklift tine fits through it cleanly.
        # The C1's 0.72 degrees closes the same gaps to 0.8 cm and 1.9 cm.
        #
        # WHAT IT COSTS IS FRESHNESS, and that is not free. The fan is cast now; the
        # buffer holds one revolution, so a bearing can be 100 ms old, which at the
        # 1.5 m/s cap is 15 cm of travel. That is why `stop_distance_m` should be raised
        # when driving on the scan -- the distance has to absorb the latency, and a guard
        # tuned on instantaneous rays is optimistic by exactly that margin.
        self.scan = scan
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = slow_distance_m
        self.fan_half_angle = math.radians(fan_half_angle_deg)
        self.fan_rays = fan_rays
        self.ray_height_m = ray_height_m
        # Rays start just outside the chassis, not at the robot's centre: an origin
        # inside the robot's own column collider gives implementation-defined results
        # (some report no hit, some report distance 0 forever). 0.35 clears the
        # measured 0.375 m turning swing radius closely enough to still see anything
        # the robot is about to touch.
        self.chassis_radius_m = chassis_radius_m
        self.robot_prim_prefix = robot_prim_prefix
        self.interventions = 0
        self._query = None

    def _scene_query(self):
        if self._query is None:
            from omni.physx import get_physx_scene_query_interface

            self._query = get_physx_scene_query_interface()
        return self._query

    def _sample_fan(self, position, travel) -> list[tuple[float, float, str]]:
        """The original seven rays. (world bearing, distance from CENTRE, prim)."""
        query = self._scene_query()
        out = []
        for i in range(self.fan_rays):
            frac = (i / (self.fan_rays - 1)) * 2.0 - 1.0 if self.fan_rays > 1 else 0.0
            angle = travel + frac * self.fan_half_angle
            direction = (math.cos(angle), math.sin(angle), 0.0)
            origin = (
                position[0] + direction[0] * self.chassis_radius_m,
                position[1] + direction[1] * self.chassis_radius_m,
                position[2] + self.ray_height_m,
            )
            hit = query.raycast_closest(origin, direction, self.slow_distance_m)
            if not hit or not hit.get("hit"):
                continue
            prim = str(hit.get("collision", ""))
            # Never guard against ourselves. The arms and column carry colliders
            # (self-collision is enabled on this articulation), and a fan ray angled
            # outward can clip them.
            if prim.startswith(self.robot_prim_prefix):
                continue
            # Rays start at the chassis edge, so add the radius back to make the distance
            # centre-relative -- which is the frame the thresholds are written in.
            out.append((angle, float(hit.get("distance", float("inf")))
                        + self.chassis_radius_m, prim))
        return out

    def _sample_scan(self, position, yaw, travel) -> list[tuple[float, float, str]]:
        """The same question asked of the lidar buffer instead of a fresh fan.

        Two differences from `_sample_fan`, and both are in the sensor's favour rather
        than hidden by it. The returns are ~16x denser across the same wedge. And they are
        already centre-relative with no radius to add back, because the unit sits at the
        robot's centre and measures from there -- where the fan has to start outside the
        chassis to avoid casting from inside the robot's own column collider.

        Self-hits are already dropped by `points_body`, so the "never guard against
        ourselves" rule is enforced in exactly one place for both sources.
        """
        pts = self.scan.points_body(position, yaw)
        if pts.size == 0:
            return []
        import numpy as np

        dist = np.hypot(pts[:, 0], pts[:, 1])
        # Bearing relative to travel, wrapped, so the wedge test is a plain comparison.
        rel = np.arctan2(pts[:, 1], pts[:, 0]) + yaw - travel
        rel = (rel + math.pi) % (2.0 * math.pi) - math.pi
        keep = (dist <= self.slow_distance_m) & (np.abs(rel) <= self.fan_half_angle)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            return []
        prims = self.scan.prims()
        return [(float(rel[i] + travel), float(dist[i]), prims[i]) for i in idx]

    def check(
        self,
        position: tuple[float, float, float],
        yaw: float,
        vx: float,
        vy: float,
        count: bool = True,
    ) -> GuardResult:
        """Sample toward the direction of TRAVEL, which is not necessarily yaw.

        On the holonomic controller the robot can move sideways while facing
        elsewhere, so casting along the heading would guard the wrong direction
        entirely -- it would happily strafe into a wall it was not looking at.

        `count=False` runs the same fan without touching `interventions`, for callers
        asking "what is over there?" rather than "clip this velocity" -- see
        `stuck_recovery.py`, which probes clearance every step. Counting those would
        add hundreds of interventions to a run that never had one, and that number is
        read as evidence about how cluttered the route was.
        """
        speed = math.hypot(vx, vy)
        if speed < 1e-6:
            return GuardResult(False, float("inf"), vx, vy)

        # Body-frame travel direction -> world.
        travel = math.atan2(vy, vx) + yaw

        samples = (self._sample_scan(position, yaw, travel) if self.scan is not None
                   else self._sample_fan(position, travel))

        nearest = float("inf")
        nearest_prim = ""
        blockers: list[float] = []  # world bearings of returns that came back too short
        for angle, distance, prim in samples:
            if distance <= self.stop_distance_m:
                blockers.append(angle)
            if distance < nearest:
                nearest = distance
                nearest_prim = prim

        if nearest >= self.slow_distance_m:
            return GuardResult(False, nearest, vx, vy, nearest_prim)

        if not blockers:
            # Ramp linearly between stop and slow distance rather than dropping to
            # zero at a threshold -- a hard cutoff makes the robot judder in and out
            # of the guard band as the fan flickers on and off a thin obstacle.
            span = self.slow_distance_m - self.stop_distance_m
            scale = max(0.15, (nearest - self.stop_distance_m) / span)
            return GuardResult(False, nearest, vx * scale, vy * scale, nearest_prim, scale)

        # --- blocked: keep whatever motion is not INTO an obstacle -----------------
        if count:
            self.interventions += 1

        # Work in world for the projection, because the ray bearings are world angles.
        wx = speed * math.cos(travel)
        wy = speed * math.sin(travel)
        for angle in blockers:
            nx, ny = math.cos(angle), math.sin(angle)
            approach = wx * nx + wy * ny
            if approach > 0.0:  # only cancel motion TOWARD the hit, never away from it
                wx -= approach * nx
                wy -= approach * ny
        # Applied per blocker rather than to the average bearing: in a corner, two hits
        # from different sides must each remove their own component. Averaging leaves a
        # velocity that still drives into one of them. Successive projection converges
        # to ~0 for a genuine dead end, which is the right answer there.

        # Slide at half speed, not at full tilt along a wall we are already touching.
        moving = math.hypot(wx, wy)
        if moving > 1e-6:
            slide_speed = min(moving, 0.5 * speed)
            wx *= slide_speed / moving
            wy *= slide_speed / moving

        # World -> body.
        c, s = math.cos(yaw), math.sin(yaw)
        return GuardResult(
            True, nearest, wx * c + wy * s, -wx * s + wy * c, nearest_prim, 0.0
        )
