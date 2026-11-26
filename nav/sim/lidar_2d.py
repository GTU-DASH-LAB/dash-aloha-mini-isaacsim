"""A synthetic 2D lidar modelled on a specific device you can actually buy.

    nav/sim/lidar_2d.py

WHY A NAMED DEVICE AND NOT "SOME RAYCASTS". PhysX's `raycast_closest` is a ray, not a
sensor -- the sensor is whatever we build out of it, and nothing stops us from building
one that cannot be bought. That is the trap: rays are free in simulation, so it costs
nothing to cast 4000 of them over 30 degrees of elevation and end up with a policy
trained against a 3D lidar nobody on this project is going to purchase. Pinning the
parameters to a real part number is what makes the sim-to-real claim mean anything.

THE DEVICE: Slamtec RPLIDAR C1 (C1M1-R2), from its datasheet --

    range               0.05 m .. 12 m          (DTOF, not triangulation)
    angular resolution  0.72 deg                -> 500 points per revolution
    scan rate           10 Hz typical           (8-12 Hz)
    sample rate         5000 points/s           (500 x 10 Hz, consistent)
    plane               one, horizontal

Chosen over the LDROBOT LD19 (<=1 deg, 4500 Hz) on resolution and on Slamtec's
first-party ROS 2 driver listing C1 explicitly, and over the RPLIDAR A1M8 because the
A1 is triangulation-based -- its effective range degrades on dark and oblique surfaces,
which in a hospital corridor is exactly where the map needs to close.

WHY 10 Hz IS ENOUGH, which is the question a rotating sensor always raises. The policy
decides at 2 Hz in sim time (one decision every 30 physics steps of 1/60 s, i.e. 0.5 s;
about 1 Hz of wall clock at the sim's ~0.5x realtime). A real C1 completes ~5
revolutions in that time, so the freshest full revolution handed to any decision is at
most 100 ms old -- 15 cm of motion at the 1.5 m/s cap, below the 0.13 m position error
the odometry already carries. A snapshot scan is therefore a faithful model of a 10 Hz
device read by a 2 Hz consumer, and modelling the spin would add a distortion smaller
than the localisation error it rides on. This stops being true if the control rate rises
or the robot speeds up; the assumption is stated here so it can be rechecked rather than
inherited.

WHAT IS DELIBERATELY NOT MODELLED, and why each is safe to omit *for now*:

  * RANGE NOISE. The PhysX query returns ground truth, and NVIDIA's PhysX SDK Lidar
    documentation says the same of theirs. That is a feature while the open question is
    whether an occupancy map helps at all -- sensor noise would be a second variable in
    an experiment that already fights a +/-2-of-6-episode noise floor. `noise_sigma_m`
    exists -- on the snapshot `Lidar2D` ONLY; `SweepingLidar2D`, the class the control
    loop runs, has no noise at all -- so noise can be switched on deliberately, as its own
    robustness study, rather than being present by accident from the start.
  * DROPOUTS ON DARK / SPECULAR SURFACES. A real DTOF unit loses returns on some
    materials, and the datasheet's range is 12 m only at 70% reflectivity: 6 m on black
    (10%). Ours returns everything to 12 m. Harmless for today's consumers -- the guard
    reads within 1.5 m and the arc filter within ~3.4 m -- but not for anything ranging a
    distant target.
  * SCAN GEOMETRY. The real plane is flat to 0-1.5 deg and the head turns CLOCKWISE; ours
    is perfectly level and sweeps counter-clockwise. Not reversed in the default, because
    that would change the beam phase against every recorded c1 run.
  * GLASS. This is the one gap that is not merely optimistic, it is qualitative: a real
    lidar sees straight through glass and reports the wall behind it, while PhysX
    reports whatever collider the glass door carries. Two of the hospital episodes have
    glazed doors. Any map built here will be RIGHT about those doors in a way the real
    sensor would be WRONG about, so do not read a hospital result as a prediction of
    real-robot behaviour at a glass door.

SELF-OCCLUSION IS MODELLED, AND THAT IS A DEPARTURE FROM THE GUARD. `collision_guard.py`
skips any hit on the robot's own prims, which is correct for a guard -- you do not brake
for your own arm. It is wrong for a lidar: a real unit bolted to this robot really is
blinded by the lift column and the two arms, and a scan that pretends otherwise would
promise a 360 degree field of view that the hardware cannot deliver. Self-hits are
returned in `self_hit` so the blind sector is visible in the data instead of being
quietly deleted, and `check_lidar_2d.py` measures how big that sector is at a candidate
mount point. Note this is a real constraint on the purchase, not a simulation detail:
on AlohaMini a genuine 360 degree scan needs the unit on a mast clearing the arms.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LidarSpec:
    """The parts of a datasheet that change what the sensor returns."""

    name: str
    min_range_m: float
    max_range_m: float
    points_per_rev: int
    scan_rate_hz: float

    @property
    def angular_resolution_deg(self) -> float:
        return 360.0 / self.points_per_rev


# Slamtec RPLIDAR C1 (C1M1-R2). 0.72 deg resolution is quoted directly; 500 points is
# that resolution over a full turn, and it reproduces the quoted 5000 samples/s at the
# quoted 10 Hz exactly -- the three numbers agree, which is the check that the datasheet
# was read rather than paraphrased.
RPLIDAR_C1 = LidarSpec(
    name="Slamtec RPLIDAR C1",
    min_range_m=0.05,
    max_range_m=12.0,
    points_per_rev=500,
    scan_rate_hz=10.0,
)

# The LD19 is kept so the cheaper part can be A/B'd against the C1 in simulation before
# anything is ordered -- the whole point of pinning to real devices.
#
# 450, not 360. Its datasheet quotes resolution as "<=1 deg" and sampling as 4500 Hz,
# and those are not the same statement: 360 points/rev satisfies the first and
# contradicts the second, while 450 satisfies both (0.8 deg, 4500 Hz at 10 Hz). A bound
# is not a value, and taking the round number would have understated the cheaper sensor
# by 25% in the one comparison this entry exists to make.
LDROBOT_LD19 = LidarSpec(
    name="LDROBOT LD19 (D300)",
    min_range_m=0.02,
    max_range_m=12.0,
    points_per_rev=450,
    scan_rate_hz=10.0,
)


# THE LIFT COLUMN, AS THE SCAN PLANE SEES IT. base_link's mesh is a deck (top at ~0.11 m)
# plus the lift column, which rises to 1.084 m. The nav stages give base_link NO collider
# -- its convex hull sealed over the wheels, see `fix_wheel_collision.py` -- so PhysX rays
# pass straight through the column. Before this was modelled the lidar sat at (0, 0, 0.30),
# i.e. INSIDE the column, and "0% blind" described the colliders rather than the robot: no
# unit can be mounted there. It is modelled in the SENSOR rather than as a USD collider on
# purpose: a new rigid shape on base_link is a new contact with the floor and the arms,
# which is how the wheel bug started, and an occluder only the beam can see has no such
# side effect. The occluder is the real mesh sliced at the mount height (base_link.STL is
# authored in the base_link frame, metres). Measured at z = 0.30: mount x = +0.10 -> 21.4%
# blind (worst arc 77 deg), +0.15 -> 14.2% / 51, +0.20 -> 10.6% / 38, +0.25 -> 8.4% / 30,
# all dead astern; x = 0 -> 100%.
BASE_LINK_STL = (Path(__file__).resolve().parents[2]
                 / "assets/upstream_alohamini1/meshes/base_link.STL")
COLUMN_Z_M = (0.12, 1.084)       # above the base_link origin: deck top .. column top


@functools.lru_cache(maxsize=8)
def _column_section(z: float) -> np.ndarray:
    """(M, 2, 2) segments where the plane z cuts base_link.STL, base_link frame."""
    raw = BASE_LINK_STL.read_bytes()
    count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tri = np.frombuffer(raw, dtype=rec, count=count, offset=84)["v"].astype(np.float64)
    zs = tri[:, :, 2] - z
    segs = []
    for t, d in zip(tri, zs):
        if d.min() > 0 or d.max() < 0:
            continue
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (d[a] > 0) != (d[b] > 0) and d[a] != d[b]:
                u = d[a] / (d[a] - d[b])
                pts.append(t[a, :2] + u * (t[b, :2] - t[a, :2]))
        if len(pts) == 2:
            segs.append(pts)
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)


def column_occlusion(bearings_body: np.ndarray, mount_forward_m: float,
                     mount_height_m: float) -> np.ndarray:
    """Per beam, the range at which the lift column blocks it; inf where it does not.

    Ray-segment intersection of each beam from (mount_forward_m, 0) against the mesh
    section at the mount height. A mount inside the column hits its inner wall within a
    few centimetres on every beam -- below the C1's 0.05 m minimum on most of them, so the
    real unit would report nothing there at all, which is the honest answer.
    """
    out = np.full(len(bearings_body), np.inf)
    if not COLUMN_Z_M[0] <= mount_height_m <= COLUMN_Z_M[1] or not BASE_LINK_STL.is_file():
        return out
    S = _column_section(round(float(mount_height_m), 4))
    if len(S) == 0:
        return out
    p, q = S[:, 0, :], S[:, 1, :]
    e = q - p                                       # (M, 2)
    o = np.array([mount_forward_m, 0.0])
    for i, b in enumerate(bearings_body):
        d = np.array([math.cos(float(b)), math.sin(float(b))])
        den = d[0] * e[:, 1] - d[1] * e[:, 0]        # cross(d, e)
        ok = np.abs(den) > 1e-12
        w = p - o
        t = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1), -1)
        hit = ok & (t >= 0) & (u >= 0) & (u <= 1)
        if hit.any():
            out[i] = float(t[hit].min())
    return out


@dataclass
class Scan:
    """One revolution, in the sensor's own frame.

    `ranges` reports `max_range_m` where nothing was struck, which is what a real unit
    reports for a max-range return -- but that value is ambiguous on its own (a hit AT
    12 m reads the same), so `hit` carries the distinction. Occupancy carving needs it:
    a miss marks the whole ray free, while a hit marks it free up to the strike and
    occupied at it.
    """

    ranges: np.ndarray        # (N,) float32, metres, sensor frame
    hit: np.ndarray           # (N,) bool, True where something was struck
    self_hit: np.ndarray      # (N,) bool, True where the strike was the robot itself
    bearings_body: np.ndarray  # (N,) float32, radians, 0 = robot forward, CCW positive
    yaw: float                # robot heading when the scan was taken
    origin: tuple[float, float, float]

    @property
    def blind_fraction(self) -> float:
        """Share of the revolution the robot's own body blocks."""
        return float(self.self_hit.mean())

    def world_points(self) -> np.ndarray:
        """(M, 2) world XY of the rays that struck something outside the robot.

        Self-hits are dropped here on purpose: they are real returns, but they are the
        robot, and carving them into a map would stamp a permanent obstacle onto the
        robot's own footprint that travels with it.
        """
        keep = self.hit & ~self.self_hit
        world = self.bearings_body[keep] + self.yaw
        r = self.ranges[keep]
        return np.stack([self.origin[0] + r * np.cos(world),
                         self.origin[1] + r * np.sin(world)], axis=1)


class Lidar2D:
    """A planar scan built from PhysX scene queries.

    Planar in the strict sense, and that is the whole point: every ray has a zero z
    component and every origin sits at the same height, so what comes back is one
    horizontal slice. Adding an elevation loop here would turn it into a 3D unit and
    break the parity with the part number in the module docstring -- don't.

    Consequence of the single plane, and it is the same one the real device has: an
    obstacle whose geometry does not cross the mount height is invisible. A pallet on
    the floor, a table top overhead, a forklift's forks. This is why the guard's own
    short-range fan stays in place rather than being replaced by this scan.
    """

    def __init__(
        self,
        spec: LidarSpec = RPLIDAR_C1,
        mount_height_m: float = 1.30,
        mount_forward_m: float = 0.0,
        robot_prim_prefix: str = "/World/Aloha",
        noise_sigma_m: float = 0.0,
        seed: int = 0,
        model_column: bool = True,
    ):
        # 1.30 m is PROVISIONAL and is a mast height, not a measurement: the forward
        # camera had to sit at z~1.21 to clear the column's top corner, so the column is
        # at least that tall and a scan plane below it is blocked behind the robot.
        # `check_lidar_2d.py` measures the blind sector at candidate heights against the
        # robot's actual colliders; set this from that output, not from this comment.
        self.spec = spec
        self.mount_height_m = mount_height_m
        self.mount_forward_m = mount_forward_m
        self.robot_prim_prefix = robot_prim_prefix
        self.noise_sigma_m = noise_sigma_m
        self._rng = np.random.default_rng(seed)
        self._query = None

        n = spec.points_per_rev
        self.bearings_body = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False
                                         ).astype(np.float32)
        self._occ = (column_occlusion(self.bearings_body, mount_forward_m, mount_height_m)
                     if model_column else np.full(n, np.inf))
        self.scans = 0

    def _scene_query(self):
        if self._query is None:
            from omni.physx import get_physx_scene_query_interface

            self._query = get_physx_scene_query_interface()
        return self._query

    def scan(self, position: tuple[float, float, float], yaw: float) -> Scan:
        """One full revolution from the robot's current pose.

        `position` is the robot base, as `base_drive` reports it; the sensor sits
        `mount_height_m` above it and `mount_forward_m` along the robot's heading.
        """
        query = self._scene_query()
        n = self.spec.points_per_rev
        ranges = np.full(n, self.spec.max_range_m, dtype=np.float32)
        hit = np.zeros(n, dtype=bool)
        self_hit = np.zeros(n, dtype=bool)

        origin = (
            position[0] + self.mount_forward_m * math.cos(yaw),
            position[1] + self.mount_forward_m * math.sin(yaw),
            position[2] + self.mount_height_m,
        )

        for i in range(n):
            angle = float(self.bearings_body[i]) + yaw
            direction = (math.cos(angle), math.sin(angle), 0.0)
            occ = float(self._occ[i])
            result = query.raycast_closest(origin, direction,
                                           min(self.spec.max_range_m, occ))
            if (not result or not result.get("hit")) and occ < self.spec.max_range_m:
                # Nothing before the column: the beam ends on the robot itself.
                ranges[i], hit[i], self_hit[i] = occ, True, True
                continue
            if not result or not result.get("hit"):
                continue
            distance = float(result.get("distance", self.spec.max_range_m))
            # Below the blind zone the real unit returns nothing at all, so neither do
            # we -- reporting the true short distance would hand the map a wall the
            # hardware could never have seen.
            if distance < self.spec.min_range_m:
                continue
            ranges[i] = distance
            hit[i] = True
            if str(result.get("collision", "")).startswith(self.robot_prim_prefix):
                self_hit[i] = True

        if self.noise_sigma_m > 0.0:
            # Applied only where something was struck: a max-range non-return has no
            # measurement to perturb, and jittering it would invent returns.
            jitter = self._rng.normal(0.0, self.noise_sigma_m, n).astype(np.float32)
            ranges = np.where(hit, np.clip(ranges + jitter, self.spec.min_range_m,
                                           self.spec.max_range_m), ranges)

        self.scans += 1
        return Scan(ranges=ranges, hit=hit, self_hit=self_hit,
                    bearings_body=self.bearings_body, yaw=yaw, origin=origin)


class SweepingLidar2D:
    """The same device read at CONTROL rate, where the spin stops being ignorable.

    `Lidar2D.scan()` returns a whole revolution at one instant, and the module docstring
    argues that is faithful for a consumer running at 1 Hz. The collision guard is not
    that consumer: it runs every physics step, ~60 Hz, and a snapshot there would hand it
    a fresh 360 degree view six times per revolution -- a sensor six times better than the
    one we are pinning the design to. That is the same failure as casting an elevation
    loop and calling it a 2D unit, just in time instead of space.

    So the beam is modelled. A C1 emits 5000 points/s in a fixed rotation, which at 60 Hz
    is ~83 points per step covering ~60 degrees; a given bearing is refreshed once per
    revolution, every 100 ms. Rays cost 2-3 us (measured), so casting the arc the beam
    would really have swept costs ~0.28 ms/step against an 8.65 ms bare step -- the honest
    model is affordable and there is no reason to take the optimistic one.

    RETURNS ARE STORED IN WORLD COORDINATES, WITH A TIMESTAMP, and that is what makes the
    staleness a latency rather than a smear. A point measured 90 ms ago is still a true
    statement about where a wall is; what has changed is where the ROBOT is, and the
    odometry knows that exactly. Reading the buffer back into the current body frame is
    therefore ordinary motion compensation -- the same de-skewing a real stack does with
    a worse pose estimate than ours. What survives is the genuine cost of a spinning
    sensor: an obstacle that appeared 50 ms ago in a bearing the beam has not reached yet
    is simply not in the buffer.
    """

    def __init__(
        self,
        spec: LidarSpec = RPLIDAR_C1,
        mount_height_m: float = 0.30,
        mount_forward_m: float = 0.0,
        robot_prim_prefix: str = "/World/Aloha",
        model_column: bool = True,
    ):
        # Where it can actually go. The lift column fills the centre from 0.12 to 1.084 m
        # (see COLUMN_Z_M), so a mount in that band must sit ahead of it; at 0.30 m and
        # +0.20 m forward `check_lidar_2d.py` measures 10.6% blind in one 38 deg arc dead
        # astern. Higher is worse below the column top, at lift 0: vertical_link blinds
        # 0.59-0.75 m (130-155 deg arcs), the arm bases and link1s 0.75-0.85 m, the arm
        # links 0.87-0.90 m, and the band rises with the lift by up to 0.60 m. The earlier
        # "0% blind everywhere but 0.80 m" counted colliders only, from inside the column.
        # The defaults here stay (0.30, 0.0) for callers that pass both; the runner uses
        # NAV_LIDAR_FORWARD.
        self.spec = spec
        self.mount_height_m = mount_height_m
        self.mount_forward_m = mount_forward_m
        self.robot_prim_prefix = robot_prim_prefix

        n = spec.points_per_rev
        self.bearings_body = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False
                                         ).astype(np.float32)
        self._wx = np.zeros(n, dtype=np.float64)
        self._wy = np.zeros(n, dtype=np.float64)
        self._stamp = np.full(n, -np.inf)      # -inf = this bearing has never been swept
        self._hit = np.zeros(n, dtype=bool)
        self._self = np.zeros(n, dtype=bool)
        self._prim = [""] * n
        self._cursor = 0
        self._carry = 0.0
        self._query = None
        # Fixed in the body frame, so computed once. `model_column=False` reproduces what
        # runs before 2026-10 saw (the sensor looked through the column); the replay tool
        # uses it for those runs' labels and nothing else should.
        self.model_column = model_column
        self._occ = (column_occlusion(self.bearings_body, mount_forward_m, mount_height_m)
                     if model_column else np.full(n, np.inf))
        self.rays_cast = 0
        self.revolutions = 0.0

    def reset(self) -> None:
        """Forget every return. Call on every teleport -- a new episode, a Reset.

        The buffer holds WORLD points, so after a teleport the previous episode's returns
        are re-expressed around the new pose as obstacles that are not there, and their
        stamps sit in a sim clock that has since restarted at 0, which makes any age
        filter accept them (now - stamp < 0). Clearing is half the fix; the caller primes
        it with one full revolution so the first decision has a scan at all -- otherwise
        decision 0 of every episode is taken unfiltered.
        """
        self._stamp[:] = -np.inf
        self._hit[:] = False
        self._self[:] = False
        self._prim = [""] * self.spec.points_per_rev
        self._cursor = 0
        self._carry = 0.0

    def prime(self, position: tuple[float, float, float], yaw: float, now: float) -> int:
        """One full revolution at rest, so the buffer is complete before the first read."""
        return self.step(position, yaw, 1.0 / self.spec.scan_rate_hz, now)

    def _scene_query(self):
        if self._query is None:
            from omni.physx import get_physx_scene_query_interface

            self._query = get_physx_scene_query_interface()
        return self._query

    def step(self, position: tuple[float, float, float], yaw: float, dt: float,
             now: float) -> int:
        """Advance the beam by however far it turned in `dt`. Returns rays cast.

        The fractional remainder is carried rather than rounded away: at 60 Hz the beam
        covers 83.33 points per step, and dropping the third of a point every time would
        run the sensor 0.4% slow forever -- small, but it is the kind of error that is
        invisible in one episode and shows up as a phase drift over a long one.
        """
        self._carry += self.spec.points_per_rev * self.spec.scan_rate_hz * dt
        count = int(self._carry)
        self._carry -= count
        if count <= 0:
            return 0
        count = min(count, self.spec.points_per_rev)   # never re-sweep past a full turn

        query = self._scene_query()
        n = self.spec.points_per_rev
        ox = position[0] + self.mount_forward_m * math.cos(yaw)
        oy = position[1] + self.mount_forward_m * math.sin(yaw)
        origin = (ox, oy, position[2] + self.mount_height_m)

        for _ in range(count):
            i = self._cursor
            self._cursor = (i + 1) % n
            angle = float(self.bearings_body[i]) + yaw
            dx, dy = math.cos(angle), math.sin(angle)
            occ = float(self._occ[i])
            result = query.raycast_closest(origin, (dx, dy, 0.0),
                                           min(self.spec.max_range_m, occ))
            self._stamp[i] = now
            if (not result or not result.get("hit")) and occ < self.spec.max_range_m:
                # The beam ends on the lift column: a self-return, kept as one so the blind
                # sector shows up in the data, and excluded from every consumer like the
                # arm hits are.
                self._hit[i] = True
                self._self[i] = True
                self._prim[i] = self.robot_prim_prefix + "/Geometry/base_link#column"
                self._wx[i] = ox + occ * dx
                self._wy[i] = oy + occ * dy
                continue
            if not result or not result.get("hit"):
                self._hit[i] = False
                self._self[i] = False
                continue
            distance = float(result.get("distance", self.spec.max_range_m))
            if distance < self.spec.min_range_m:
                self._hit[i] = False
                self._self[i] = False
                continue
            prim = str(result.get("collision", ""))
            self._hit[i] = True
            self._self[i] = prim.startswith(self.robot_prim_prefix)
            self._prim[i] = prim
            self._wx[i] = ox + distance * dx
            self._wy[i] = oy + distance * dy

        self.rays_cast += count
        self.revolutions += count / n
        return count

    def _keep(self, now: float | None, max_age_s: float | None) -> np.ndarray:
        keep = self._hit & ~self._self & np.isfinite(self._stamp)
        if max_age_s is not None:
            if now is None:
                raise ValueError("max_age_s needs a timestamp")
            keep &= (now - self._stamp) <= max_age_s
        return keep

    def points_body(self, position: tuple[float, float, float], yaw: float,
                    now: float | None = None,
                    max_age_s: float | None = None) -> np.ndarray:
        """(M, 2) body-frame XY of external returns, motion-compensated to now.

        Self-hits are dropped, as in `Scan.world_points`: they are the robot, and to a
        consumer asking "what is in my way" the robot's own arm is never the answer.

        No age filter by default, and that is not laziness -- every bearing is overwritten
        exactly once per revolution, so the buffer cannot hold anything older than 100 ms
        once the beam has been round. Filtering would only ever discard the freshest
        picture available of a bearing the beam has not returned to yet.
        """
        keep = self._keep(now, max_age_s)
        if not keep.any():
            return np.zeros((0, 2), dtype=np.float64)
        rx = self._wx[keep] - position[0]
        ry = self._wy[keep] - position[1]
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.stack([rx * c - ry * s, rx * s + ry * c], axis=1)

    def prims(self, now: float | None = None,
              max_age_s: float | None = None) -> list[str]:
        """Prim path per row of `points_body`, same order. For telemetry only."""
        keep = self._keep(now, max_age_s)
        return [self._prim[i] for i in np.flatnonzero(keep)]

    def max_age_s(self, now: float) -> float:
        """Age of the STALEST bearing in the buffer, seconds.

        Reported rather than assumed to be one revolution: it is one revolution only once
        the beam has been round once, and it is what the stop distance has to absorb.
        """
        seen = self._stamp[np.isfinite(self._stamp)]
        return float(now - seen.min()) if seen.size else float("inf")
