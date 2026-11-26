"""Collision guard for the walking G1: a floor-step probe plus three horizontal ray rings, and G1 distances.

Why the AlohaMini guard (nav/sim/collision_guard.py, one 7-ray fan at 0.30 m) is not enough here --
measured, not argued:

  * g1_minimal.usd has THREE colliders: the two foot boxes and a torso box 0.65-0.94 m above the floor.
    Legs, pelvis, arms and head collide with nothing (g1/policy, g1/asset).
  * TRIP SWEEP (nav/tools/g1_obstacle_course.py, walking straight at vx 0.5 into a box with a vertical
    face, flat ground, 2026-10-05): 0.03 m -> steps over it; 0.05 / 0.08 / 0.12 / 0.20 / 0.40 m -> a
    foot catches the face and the robot FALLS 4.40-4.42 s after starting (tilt 40-43 deg or pelvis
    < 0.40 m above the feet) with torso_contact_steps 0 every time. A fall is final for this robot.
    So anything 5 cm tall in the path is a fall, and a 0.30 m ray ring would miss a 5-25 cm obstacle.
  * a bar at head height (z 1.05-1.40 m): the robot walks straight through it (the head and camera have
    no collider) -- the camera image then shows the inside of whatever it drove into.
  * a chest-height slab: only the torso box touches it (304 sub-steps, no fall) -- contact, not safety.
  * second sweep (course2/course3): 0.04 m falls at 0.5 and 0.8 m/s and is cleared at 1.0 (speed-dependent
    foot clearance), so the up-step limit stays at 3 cm. Stepping DOWN is much easier: 0.13 m at 0.5/0.8/1.0
    and 0.20 m at 0.8/1.0 m/s walked off without a fall. A 5 cm pole: torso contact at 0.8, FALL at 1.0.

So the G1 guard samples, toward the direction of travel (fan of `fan_rays` over +-fan_half_angle):

  STEP PROBE   downward rays at 0.10 m spacing from the chassis edge out to the slow distance; the floor
               height under each sample is compared with the previous sample along the same ray (the first
               sample is the reference -- not the floor under the robot, see _sample_fan). A rise of more
               than G1_STEP_MAX_M or a drop of more than G1_STEP_DOWN_MAX_M is an obstacle at that distance:
               a box, a kerb, a stair, a drop-off. A slope is a gradual change (6.8 deg = 1.2 cm per 0.1 m;
               rises pass up to ~19 deg) and is not flagged. A downward ray that STARTS inside something
               (distance 0), or finds no floor within 1.1 m, is an obstacle. NOTE the probe also sees the
               open SIDE of a raised walkway: on the course's 1.6 m wide ramp the outer rays find a >0.135 m
               drop ~1.2-1.6 m out and the guard slows (scale 0.35-0.85, offline_narrow_ramp.py); on a
               3.6 m wide ramp, nothing.
  RINGS        horizontal rays at G1_RING_HEIGHTS_M above the floor estimate: 0.20 m (thin vertical things
               the step probe can fall between -- chair and table legs, poles), 0.80 m (torso box, arms),
               1.15 m (head and camera). A hit with a walkable normal (n_z >= 0.7) is floor (a ramp the
               ray runs into), not an obstacle.

Ring hits on the robot's own prims are stepped past, not dropped (a foot box in front of a ray would
otherwise hide what is behind it); a downward probe that lands on the robot is skipped instead.

Distances (centre-relative, the frame CollisionGuard's thresholds are written in):
  stop  G1_STOP_DISTANCE_M: foot reach ahead of the pelvis mid-stride (~0.30 m) + coast after a zero
        command (0.13 m from 0.5 m/s, 0.23-0.24 m from 0.8 m/s, g1/check) + the slew limit's braking delay
        (<= 0.25 s, locomotion.DEFAULT_SLEW vx_down) + margin. Checked by the guard passes of
        nav/tools/g1_obstacle_course.py (min gap between the foot colliders and the obstacle face).
  slow  G1_SLOW_DISTANCE_M: CollisionGuard's linear speed ramp (min scale 0.15) starts here; the G1 needs
        ~0.9 s to stop, so the ramp has to begin earlier than AlohaMini's 1.5 m.

Blocking stays DIRECTIONAL (motion into a blocker is removed, the rest is kept at half speed), but the
removal is the exact projection onto the feasible cone (G1Guard.project_feasible), not CollisionGuard's
one-pass sequential projection, which leaves a back-left sidestep in front of a wall. A downward probe
that lands on the robot's own foot is skipped, not stepped through (see _floor_z).
`position` is G1Base.position(): (x, y, floor estimate = pelvis z - 0.63 m, +-1.5 cm while walking).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SIM = Path(__file__).resolve().parents[1]
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from collision_guard import CollisionGuard, GuardResult  # noqa: E402,F401

G1_STEP_MAX_M = 0.035             # trip sweep: 0.03 m stepped over, 0.05 m fell; 0.035 so a 3 cm sill
                                  # is not flagged by float noise at exactly the measured height
G1_STEP_DOWN_MAX_M = 0.135        # drops (kerb_down lanes, no guard): 0.13 m walked down at 0.5/0.8/1.0 m/s,
                                  # 0.20 m at 0.8/1.0; 0.135 = the drop measured safe at every speed
G1_STEP_SPACING_M = 0.10
G1_STEP_PROBE_FROM_M = 0.60       # downward probes start this far above the floor estimate
G1_RING_HEIGHTS_M = (0.20, 0.80, 1.15)
G1_STOP_DISTANCE_M = 0.80
G1_SLOW_DISTANCE_M = 1.80
G1_CHASSIS_RADIUS_M = 0.25        # rays start here; torso box half-extent 0.11 m, hands ~0.25 m sideways
G1_FAN_RAYS = 7
WALKABLE_NORMAL_Z = 0.7


class G1Guard(CollisionGuard):
    """CollisionGuard with the G1 sampling above. `scan` (a SweepingLidar2D) is merged in as one more
    source of returns if given; the step probe and rings are always cast."""

    def __init__(self, ring_heights_m=G1_RING_HEIGHTS_M, stop_distance_m: float = G1_STOP_DISTANCE_M,
                 slow_distance_m: float = G1_SLOW_DISTANCE_M, fan_half_angle_deg: float = 35.0,
                 fan_rays: int = G1_FAN_RAYS, chassis_radius_m: float = G1_CHASSIS_RADIUS_M,
                 robot_prim_prefix: str = "/World/G1", scan=None, step_probe: bool = True,
                 step_max_m: float = G1_STEP_MAX_M, step_down_max_m: float = G1_STEP_DOWN_MAX_M,
                 step_spacing_m: float = G1_STEP_SPACING_M,
                 walkable_normal_z: float = WALKABLE_NORMAL_Z):
        super().__init__(stop_distance_m=stop_distance_m, slow_distance_m=slow_distance_m,
                         fan_half_angle_deg=fan_half_angle_deg, fan_rays=fan_rays,
                         ray_height_m=ring_heights_m[0] if ring_heights_m else 0.0,
                         chassis_radius_m=chassis_radius_m, robot_prim_prefix=robot_prim_prefix, scan=None)
        self.ring_heights_m = tuple(float(h) for h in ring_heights_m)
        self.step_probe = bool(step_probe)
        self.step_max_m = float(step_max_m)
        self.step_down_max_m = float(step_down_max_m)
        self.step_spacing_m = float(step_spacing_m)
        self.walkable_normal_z = float(walkable_normal_z)
        self.g1_scan = scan
        self._yaw = 0.0
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.interventions = 0
        self.hits_by_source = {"step": 0, **{f"ring{h:.2f}": 0 for h in self.ring_heights_m}, "scan": 0}
        self.floor_hits_skipped = 0
        self.self_hits_stepped = 0
        self.self_hits_skipped = 0
        self.rays_cast = 0
        self.blocked_checks = 0
        self.blocked_zero_translation = 0

    # ------------------------------------------------------------------ ray helpers
    def _cast(self, origin, direction, max_dist, skip_walkable: bool):
        """(distance from origin, prim, hit z, normal z) of the first hit not on the robot, or None.
        skip_walkable: a hit with a walkable normal ends the ray with None (it is floor)."""
        q = self._scene_query()
        o = list(origin)
        travelled = 0.0
        for _ in range(8):
            self.rays_cast += 1
            hit = q.raycast_closest(tuple(o), direction, max_dist - travelled)
            if not hit or not hit.get("hit"):
                return None
            d = float(hit.get("distance", 0.0))
            prim = str(hit.get("collision", ""))
            if prim.startswith(self.robot_prim_prefix):
                self.self_hits_stepped += 1
                step = d + 0.02
                o = [o[i] + direction[i] * step for i in range(3)]
                travelled += step
                if travelled >= max_dist:
                    return None
                continue
            n = hit.get("normal")
            nz = float(n[2]) if n is not None else 0.0
            if skip_walkable and nz >= self.walkable_normal_z:
                self.floor_hits_skipped += 1
                return None
            return travelled + d, prim, float(hit["position"][2]), nz
        return None

    def _floor_z(self, x, y, z_from, depth):
        """(floor z, prim, started inside something, landed on the robot).

        A downward ray that meets the robot's own foot is NOT stepped through: the foot box is 5 cm thick
        and sits on the floor, so stepping past it in 2 cm increments ends below the floor surface, where
        the ground reports an initial overlap (distance 0) or nothing -- either read as an obstacle. Caught
        offline (scratchpad g1/repair2/offline_guard.py, 'foot_in_front_only' was flagged as a step).
        The caller skips such a sample; the foot moves on within a stride and the next check samples it."""
        self.rays_cast += 1
        hit = self._scene_query().raycast_closest((x, y, z_from), (0.0, 0.0, -1.0), depth)
        if not hit or not hit.get("hit"):
            return None, "", False, False
        prim = str(hit.get("collision", ""))
        if prim.startswith(self.robot_prim_prefix):
            self.self_hits_skipped += 1
            return None, prim, False, True
        d = float(hit.get("distance", 0.0))
        return float(hit["position"][2]), prim, d < 1e-4, False

    # ------------------------------------------------------------------ sampling
    def _sample_fan(self, position, travel) -> list[tuple[float, float, str]]:
        out = []
        if self.g1_scan is not None:
            self.scan = self.g1_scan
            try:
                pts = self._sample_scan(position, self._yaw, travel)
            finally:
                self.scan = None
            self.hits_by_source["scan"] += sum(1 for _, d, _ in pts if d <= self.stop_distance_m)
            out.extend(pts)
        px, py, pz = position
        dirs = []
        for i in range(self.fan_rays):
            frac = (i / (self.fan_rays - 1)) * 2.0 - 1.0 if self.fan_rays > 1 else 0.0
            dirs.append(travel + frac * self.fan_half_angle)

        if self.step_probe:
            # Heights are compared only between neighbouring samples 0.10 m apart; the first sample of each
            # ray is the reference. NOT the floor under the robot: that ray often lands on a foot, and the
            # fallback (pelvis z - 0.63 m, +-1.5 cm while walking) plus 0.25 m of a 6.8 deg slope (0.030 m)
            # crossed the threshold -- the Kit course stopped the robot ON the ramp (guard@0.8, 33 step
            # hits) while the offline test, with an exact floor estimate, passed.
            z_top = pz + G1_STEP_PROBE_FROM_M
            n_s = max(1, int(math.floor((self.slow_distance_m - self.chassis_radius_m) / self.step_spacing_m)) + 1)
            for ang in dirs:
                c, s = math.cos(ang), math.sin(ang)
                z_prev = None
                for k in range(n_s):
                    d = self.chassis_radius_m + k * self.step_spacing_m
                    hz, prim, inside, on_robot = self._floor_z(px + c * d, py + s * d, z_top,
                                                               G1_STEP_PROBE_FROM_M + 0.5)
                    if on_robot:
                        z_prev = None                  # own foot under this sample: restart the reference
                        continue
                    if inside or hz is None:
                        # inside something taller than the probe start, or no floor at all within 1.1 m
                        if d <= self.stop_distance_m:
                            self.hits_by_source["step"] += 1
                        out.append((ang, d, prim or "<no floor>"))
                        break
                    if z_prev is not None:
                        dz = hz - z_prev
                        if dz > self.step_max_m or -dz > self.step_down_max_m:
                            if d <= self.stop_distance_m:
                                self.hits_by_source["step"] += 1
                            out.append((ang, d, prim))
                            break
                    z_prev = hz

        for h in self.ring_heights_m:
            for ang in dirs:
                c, s = math.cos(ang), math.sin(ang)
                origin = (px + c * self.chassis_radius_m, py + s * self.chassis_radius_m, pz + h)
                r = self._cast(origin, (c, s, 0.0), self.slow_distance_m, skip_walkable=True)
                if r is None:
                    continue
                d = r[0] + self.chassis_radius_m
                if d <= self.stop_distance_m:
                    self.hits_by_source[f"ring{h:.2f}"] += 1
                out.append((ang, d, r[1]))
        return out

    @staticmethod
    def project_feasible(wx: float, wy: float, blockers) -> tuple[float, float]:
        """Closest velocity to (wx, wy) that does not move toward ANY blocker bearing (v . n_i <= 0 for all i).

        CollisionGuard.check projects the blockers out one after another, in a single pass. That is not
        the projection onto their intersection: for a wall dead ahead (7 blockers over +-35 deg) it leaves
        a residual pointing back-left, and after the half-speed cap the guarded command is
        (vx -0.143, vy +0.205) from a 0.5 m/s request (offline_guard.py). On AlohaMini that is a slide;
        on the G1 the vx < 0 is clamped (no backward walking was trained) and what is left is a 0.2 m/s
        sidestep that nothing sampled -- the fan points along the REQUESTED travel -- and a sidestep into a
        5 cm box is a fall. In 2-D the exact projection onto the cone is one of: the request itself, its
        projection onto one blocker's boundary line, or zero -- the feasible candidate nearest the request
        wins. A blocker at one side still leaves the slide along it (the Aisle-05 lesson in CLAUDE.md:
        block directions, not speed); a wall dead ahead gives zero translation."""
        ns = [(math.cos(a), math.sin(a)) for a in blockers]

        def feasible(x, y):
            return all(x * nx + y * ny <= 1e-9 for nx, ny in ns)

        if feasible(wx, wy):
            return wx, wy
        best, best_d = (0.0, 0.0), math.hypot(wx, wy)
        for nx, ny in ns:
            a = wx * nx + wy * ny
            if a <= 0.0:
                continue
            cx, cy = wx - a * nx, wy - a * ny
            if feasible(cx, cy):
                d = math.hypot(wx - cx, wy - cy)
                if d < best_d:
                    best, best_d = (cx, cy), d
        return best

    # ------------------------------------------------------------------ recovery (turn in place)
    def clearance(self, position, yaw, rel_bearings_deg=(-90, -60, -30, 0, 30, 60, 90)) -> dict:
        """Nearest return (m from the centre, inf if none within the slow distance) along each body-relative
        bearing, with the same step probe + rings but ONE ray per bearing. Read-only: no telemetry counted."""
        saved = (self.fan_rays, dict(self.hits_by_source), self.floor_hits_skipped, self.self_hits_stepped,
                 self.self_hits_skipped, self.rays_cast)
        out = {}
        try:
            self.fan_rays = 1
            for b in rel_bearings_deg:
                samples = self._sample_fan(position, yaw + math.radians(b))
                out[b] = min((d for _, d, _ in samples), default=float("inf"))
        finally:
            (self.fan_rays, self.hits_by_source, self.floor_hits_skipped, self.self_hits_stepped,
             self.self_hits_skipped, self.rays_cast) = saved
        return out

    def recovery_turn(self, position, yaw, wz: float = 0.6) -> tuple[float, float, float, dict]:
        """Command for a G1 that is wedged (guard returns zero translation): it cannot walk backwards (vx < 0 was
        never trained and G1Base clamps it), so AlohaMini's reverse-out recovery does not exist for it. Turn in
        place toward the side with more free space instead: score = sum of the clearances at +-30/60/90 deg,
        each clipped at the slow distance; ties turn left. Returns (vx 0, vy 0, wz, clearances). The caller
        keeps turning until check() no longer blocks straight ahead; the slew limit (wz 2 rad/s^2) ramps the
        turn in and out."""
        c = self.clearance(position, yaw)
        cap = self.slow_distance_m
        left = sum(min(c[b], cap) for b in (30, 60, 90))
        right = sum(min(c[b], cap) for b in (-30, -60, -90))
        return 0.0, 0.0, (abs(wz) if left >= right else -abs(wz)), c

    def check(self, position, yaw, vx, vy, count: bool = True) -> GuardResult:
        """CollisionGuard.check with the G1 sampling above and an exact projection when blocked
        (project_feasible); the slow band and the half-speed slide cap are the parent's."""
        self._yaw = yaw
        speed = math.hypot(vx, vy)
        if speed < 1e-6:
            return GuardResult(False, float("inf"), vx, vy)
        travel = math.atan2(vy, vx) + yaw
        samples = self._sample_fan(position, travel)
        nearest, nearest_prim, blockers = float("inf"), "", []
        for angle, distance, prim in samples:
            if distance <= self.stop_distance_m:
                blockers.append(angle)
            if distance < nearest:
                nearest, nearest_prim = distance, prim
        if nearest >= self.slow_distance_m:
            return GuardResult(False, nearest, vx, vy, nearest_prim)
        if not blockers:
            span = self.slow_distance_m - self.stop_distance_m
            scale = max(0.15, (nearest - self.stop_distance_m) / span)
            return GuardResult(False, nearest, vx * scale, vy * scale, nearest_prim, scale)
        if count:
            self.interventions += 1
        self.blocked_checks += 1
        wx, wy = self.project_feasible(speed * math.cos(travel), speed * math.sin(travel), blockers)
        moving = math.hypot(wx, wy)
        if moving > 1e-6:
            slide = min(moving, 0.5 * speed)
            wx, wy = wx * slide / moving, wy * slide / moving
        else:
            wx, wy = 0.0, 0.0
            self.blocked_zero_translation += 1
        c, s = math.cos(yaw), math.sin(yaw)
        return GuardResult(True, nearest, wx * c + wy * s, -wx * s + wy * c, nearest_prim, 0.0)
