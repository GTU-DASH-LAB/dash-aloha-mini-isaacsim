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
    ):
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

    def check(
        self,
        position: tuple[float, float, float],
        yaw: float,
        vx: float,
        vy: float,
    ) -> GuardResult:
        """Sample toward the direction of TRAVEL, which is not necessarily yaw.

        On the holonomic controller the robot can move sideways while facing
        elsewhere, so casting along the heading would guard the wrong direction
        entirely -- it would happily strafe into a wall it was not looking at.
        """
        speed = math.hypot(vx, vy)
        if speed < 1e-6:
            return GuardResult(False, float("inf"), vx, vy)

        # Body-frame travel direction -> world.
        travel = math.atan2(vy, vx) + yaw
        query = self._scene_query()

        nearest = float("inf")
        nearest_prim = ""
        blockers: list[float] = []  # world bearings of rays that came back too short
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
            distance = float(hit.get("distance", float("inf"))) + self.chassis_radius_m
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
