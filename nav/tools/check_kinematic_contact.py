#!/usr/bin/env python3
"""Offline checks of nav/sim/kinematic_contact.py: no Isaac, no GPU, a fake PhysX scene query.

The fake world is axis-aligned boxes plus an infinite floor at z = 0 (prim
/World/Ground). Its `sweep_box_all` marches the yaw-rotated query box along the direction
in 0.25 mm steps and reports, per obstacle, the first distance at which the two overlap
(separating-axis test in the plane, interval test in z) and the face normal of the
obstacle on the axis that closed last; an obstacle already overlapping at distance 0 is
reported at 0 with normal -direction, which is what PhysX does. `overlap_box` is the same
SAT test. Every query also reports a handful of robot shapes (/World/Aloha/...) at
distance 0, because a real sweep of the envelope always starts inside the robot.

The envelope is the measured one (probe_a, nav_hospital, 2026-10-05) so the geometry of
the checks is the robot's, not a toy's. What is checked, each as an assertion:

  open        any command in free space comes back bit-identical, 0 contact steps
  wall        head-on at 0.45 and 1.5 m/s: stops at the skin (never closer, within one step
              of it), never penetrates; the guard metric counts it
  slide45     45 deg into a wall: the normal component dies, the tangential one survives
              (|v_t| = v cos 45 within 1%), the robot travels along the wall
  corner      two walls: comes to rest, never penetrates either
  low_step    a 3 cm step (under MAX_STEP_M) is not a blocker; an 8 cm one is
  lowbar      a 6.5 cm bar beside the deck: a turn toward it is refused before the deck
              reaches it (either direction); a 4 cm bar never refuses a turn
  slab        a slab at 0.85-0.90 m (arm height) blocks; one at 1.20 m does not
  turn        arm tip near a wall: the turn is refused before the core box reaches the wall;
              the same turn in free space is never refused
  escape      spawned 3 cm INTO a wall: moving deeper is refused, backing out and sliding
              are allowed
  dynanav     physical_collision needs 200 consecutive contact steps at 60 Hz (100 frames
              at 30 FPS, benchmark.py:131/:426): 199 -> False, 200 -> True
  seam        the REAL base_drive.KinematicBase.apply() with this attached as its
              `motion_filter`, on a fake articulation that integrates exactly what was
              written: free space is bit-identical to no filter (pose AND written velocity,
              step by step), a wall stops at the skin where the bare base drives through it,
              a refused turn leaves yaw alone, stop() (dt = 0) is not counted as a step,
              and z comes from the articulation (or from a 4th element of `start`)
  kerbcorner  ONE kerb prim, low under a deck corner and 13 cm high ahead (the outdoor
              isl_curb case): the tyres standing on its low end must not make its high end
              floor, and a 2 cm sill under the same corner must still not stop the robot
  creep       a box already 0.5 mm into a wall slides along it and never sinks further
  jiggle      3000 random commands pushing at a wall, a corner and a pillar: the core never
              sinks in, and backing out is never refused
  constants   second copies of a measured number still agree with their source:
              DECK_SPLIT_Z_M == lidar_2d.COLUMN_Z_M[0], DYNANAV_SUSTAINED_S == 100 / 30

Exit status 0 only if every assertion passes.  Usage: python3 nav/tools/check_kinematic_contact.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("nav/tools")]
sys.path.insert(0, str(REPO / "scripts"))
# KC_SIM_DIR: import a different copy of the module (the mutation test points it at a mutant).
sys.path.insert(0, os.environ.get("KC_SIM_DIR", str(REPO / "nav/sim")))
import kinematic_contact as KC  # noqa: E402
from alohamini1_specs import BASE_LINK_REST_Z_M  # noqa: E402

DT = 1.0 / 60.0
FAILS: list[str] = []
CHECKS = [0]


def check(cond: bool, msg: str) -> None:
    CHECKS[0] += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)


# --- the robot's envelope (probe_a, base_link frame, lift 0, arms at rest) -------------------
MEASURED = {
    "low": ((-0.2224, -0.2288, -0.0081), (0.2238, 0.1598, 0.1197)),
    "column": ((-0.0738, -0.1013, 0.121), (0.0734, 0.1542, 1.0836)),
    "carriage": ((-0.2139, -0.1132, 0.5886), (0.212, -0.0124, 0.8249)),
    "left_arm": ((0.1256, -0.3061, 0.771), (0.1957, 0.0121, 1.0052)),
    "right_arm": ((-0.1882, -0.3024, 0.7717), (-0.1181, 0.0158, 1.0059)),
}


class StubEnvelope:
    def __init__(self, max_step_m=KC.MAX_STEP_M):
        self.max_step_m = max_step_m
        self.boxes = []
        for name, (lo, hi) in MEASURED.items():
            lo, hi = np.array(lo, float), np.array(hi, float)
            if name == "low":
                lo[2] = max(lo[2], max_step_m - BASE_LINK_REST_Z_M)
            self.boxes.append(KC.EnvBox(name, (lo + hi) / 2, (hi - lo) / 2, name == "low"))

    def refresh(self):
        return self.boxes

    def describe(self):
        return {}


# --- fake scene query -------------------------------------------------------------------------
def _yaw_of(rot):
    return 2.0 * math.atan2(rot[2], rot[3])


def _sat_overlap(half, ctr, yaw, lo, hi, eps=0.0):
    """Yaw-rotated query box vs AABB. Returns per-axis penetration (or None if separated)."""
    if ctr[2] + half[2] <= lo[2] + eps or ctr[2] - half[2] >= hi[2] - eps:
        return None
    bc = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
    bh = ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)
    c, s = math.cos(yaw), math.sin(yaw)
    ax = [(1.0, 0.0), (0.0, 1.0), (c, s), (-s, c)]
    pens = []
    for a in ax:
        ra = abs(half[0] * (c * a[0] + s * a[1])) + abs(half[1] * (-s * a[0] + c * a[1]))
        rb = abs(bh[0] * a[0]) + abs(bh[1] * a[1])
        d = abs((ctr[0] - bc[0]) * a[0] + (ctr[1] - bc[1]) * a[1])
        pen = ra + rb - d
        if pen <= eps:
            return None
        pens.append(pen)
    return pens


class FakeSQ:
    """`sticky`: report an INITIAL overlap whenever the gap is under this (PhysX does, in Kit: a
    robot parked 1 mm off the synthetic wall read as overlapping it within 9 steps)."""

    def __init__(self, boxes: dict[str, tuple], sticky: float = 0.0):
        # name -> list of (lo, hi) parts: one prim may be several boxes (a kerb that is low where
        # a deck corner stands on it and high in front -- one USD prim, like isl_curb pieces).
        def parts(v):
            return [v] if np.ndim(v[0]) == 1 else list(v)
        self.boxes = {k: [(np.array(lo, float), np.array(hi, float)) for lo, hi in parts(v)]
                      for k, v in boxes.items()}
        self.calls = 0
        self.sticky = sticky

    def _robot_hits(self, report, distance=True):
        for i in range(4):
            h = SimpleNamespace(collision=f"/World/Aloha/Geometry/base_link/wheel{i}/collision_sphere",
                                distance=0.0, normal=(0.0, 0.0, 1.0))
            report(h)

    def sweep_box_all(self, half, ctr, rot, d, dist, report, bothSides=False):
        self.calls += 1
        self._robot_hits(report)
        yaw = _yaw_of(rot)
        for name, plist in self.boxes.items():
            if any(_sat_overlap(half, ctr, yaw, lo, hi, eps=-self.sticky) is not None for lo, hi in plist):
                report(SimpleNamespace(collision=name, distance=0.0, normal=(-d[0], -d[1], -d[2])))
                continue
            best = None
            for lo, hi in plist:
                h = self._march(half, ctr, yaw, lo, hi, d, dist)
                if h is not None and (best is None or h[0] < best[0]):
                    best = h
            if best is not None:
                report(SimpleNamespace(collision=name, distance=best[0], normal=best[1]))
        return True

    @staticmethod
    def _march(half, ctr, yaw, lo, hi, d, dist):
        """First distance along d at which the query box meets one AABB part, and that part's
        face normal (the world axis along which the box entered); None within dist."""
        t, step = 0.0, 2.5e-4
        while t <= dist:
            t += step
            p = (ctr[0] + d[0] * t, ctr[1] + d[1] * t, ctr[2] + d[2] * t)
            if _sat_overlap(half, p, yaw, lo, hi) is not None:
                bc = (lo + hi) / 2
                prev = (p[0] - d[0] * step, p[1] - d[1] * step)
                gx = abs(prev[0] - bc[0]) - (hi[0] - lo[0]) / 2
                gy = abs(prev[1] - bc[1]) - (hi[1] - lo[1]) / 2
                if gx >= gy:
                    return t, (math.copysign(1.0, prev[0] - bc[0]), 0.0, 0.0)
                return t, (0.0, math.copysign(1.0, prev[1] - bc[1]), 0.0)
        return None

    def overlap_box(self, half, ctr, rot, report, anyHit=False):
        self.calls += 1
        self._robot_hits(report)
        yaw = _yaw_of(rot)
        for name, plist in self.boxes.items():
            if any(_sat_overlap(half, ctr, yaw, lo, hi) is not None for lo, hi in plist):
                report(SimpleNamespace(collision=name, distance=0.0, normal=(0, 0, 1)))
        return 1

    def raycast_closest(self, o, d, dist, bothSides=False):
        self.calls += 1
        best = None
        for name, lo, hi in ((n_, lo_, hi_) for n_, pl in self.boxes.items() for lo_, hi_ in pl):
            if lo[0] <= o[0] <= hi[0] and lo[1] <= o[1] <= hi[1] and hi[2] <= o[2] and o[2] - hi[2] <= dist:
                t = o[2] - hi[2]
                if best is None or t < best[0]:
                    best = (t, name)
        if (best is None or o[2] < best[0]) and 0.0 <= o[2] <= dist:
            if best is None or o[2] < best[0]:
                best = (o[2], "/World/Ground")
        if best is None:
            return {"hit": False}
        return {"hit": True, "distance": best[0], "collision": best[1],
                "position": (o[0], o[1], o[2] - best[0]), "normal": (0.0, 0.0, 1.0)}


def make(boxes, sticky=0.0, **kw):
    return KC.KinematicContact(StubEnvelope(), scene_query=FakeSQ(boxes, sticky=sticky), **kw)


Z = BASE_LINK_REST_Z_M   # base_link origin height on a z = 0 floor


def core_penetrates(kc, x, y, yaw, grow=0.0):
    for b in kc.env.boxes:
        half, ctr, rot = kc._box_world(b, x, y, Z, yaw, grow=grow)
        for name, lo, hi in ((n_, lo_, hi_) for n_, pl in kc.sq.boxes.items() for lo_, hi_ in pl):
            if _sat_overlap(half, ctr, _yaw_of(rot), lo, hi, eps=1e-6) is not None:
                return f"{b.name}/{name}"
    return None


SKIN_VIOLATIONS = [0]


def drive(kc, x, y, yaw, cmd, steps, skin_check=False):
    """Integrate resolve()'s output. skin_check: count steps on which an obstacle came inside
    the translation skin (core boxes grown by SKIN_M less 1.5 mm) -- 0 for a correct slide."""
    vx, vy, om = cmd
    pen = None
    SKIN_VIOLATIONS[0] = 0
    for _ in range(steps):
        r = kc.resolve(x, y, Z, yaw, vx, vy, om, DT)
        c, s = math.cos(yaw), math.sin(yaw)
        x += (r.vx * c - r.vy * s) * DT
        y += (r.vx * s + r.vy * c) * DT
        yaw += r.omega * DT
        pen = pen or core_penetrates(kc, x, y, yaw)
        if skin_check and core_penetrates(kc, x, y, yaw, grow=kc.skin - 1.5e-3):
            SKIN_VIOLATIONS[0] += 1
    return x, y, yaw, pen


class FakeArt:
    """A perfect sim for base_drive.KinematicBase: integrates exactly the velocity written."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, z=Z):
        self.p = np.array([[x, y, z]], dtype=np.float64)
        self.q = np.array([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], dtype=np.float64)
        self.v = np.zeros((1, 6))
        self.pose_reads = 0
        self.writes: list = []

    def get_world_poses(self):
        self.pose_reads += 1
        return self.p.copy(), self.q.copy()

    def set_world_poses(self, positions, orientations):
        self.p = np.asarray(positions, dtype=np.float64).copy()
        self.q = np.asarray(orientations, dtype=np.float64).copy()

    def set_velocities(self, v):
        self.v = np.asarray(v, dtype=np.float64).copy()
        self.writes.append(self.v.copy())

    def get_linear_velocities(self):
        return self.v[:, :3].copy() * 0.0

    def set_joint_velocity_targets(self, *_a, **_k):
        pass

    def step(self, dt):
        w, _x, _y, z = self.q[0]
        yaw = 2.0 * math.atan2(z, w) + self.v[0, 5] * dt
        self.p[0, :2] += self.v[0, :2] * dt
        self.q = np.array([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def seam_drive(kc, cmd, steps, x=0.0, yaw=0.0):
    """Drive the real KinematicBase through `steps` steps; kc None = no filter."""
    import base_drive as BD

    art = FakeArt(x=x, yaw=yaw)
    base = BD.KinematicBase(art, ["wheel1", "wheel2", "wheel3"])
    base.sync_from_sim()
    if kc is not None:
        kc.attach(base)
        kc.reset_episode()
    poses = []
    for _ in range(steps):
        base.apply(*cmd, DT)
        art.step(DT)
        base.check_step()
        poses.append((float(base._xy[0]), float(base._xy[1]), float(base.yaw)))
    return base, art, poses


def check_seam(WALL, front) -> None:
    sys.path.insert(0, str(REPO / "nav/sim"))
    try:
        import base_drive as BD
    except Exception as exc:  # noqa: BLE001
        check(False, f"seam: base_drive does not import offline ({exc!r})")
        return
    if not hasattr(BD.KinematicBase, "_resolve_motion"):
        check(False, "seam: KinematicBase has no _resolve_motion/motion_filter seam")
        return
    print("seam: the real KinematicBase.apply() with the filter attached")
    rng = np.random.default_rng(1)
    for _ in range(5):
        cmd = (rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 0.5), rng.uniform(-1.2, 1.2))
        yaw0 = rng.uniform(-3, 3)
        b0, a0, p0 = seam_drive(None, cmd, 120, yaw=yaw0)
        kc = make({})
        b1, a1, p1 = seam_drive(kc, cmd, 120, yaw=yaw0)
        check(p0 == p1, f"seam open {cmd}: pose differs from no filter")
        check(all(np.array_equal(u, v) for u, v in zip(a0.writes, a1.writes)),
              f"seam open {cmd}: written velocity differs from no filter")
        check(kc.summary()["steps"] == 120 and kc.summary()["contact_steps"] == 0,
              f"seam open {cmd}: steps {kc.summary()['steps']} contacts {kc.summary()['contact_steps']}")
    for v in (0.45, 1.5):
        n = int(2.0 / v / DT)
        _b, _a, bare = seam_drive(None, (v, 0.0, 0.0), n)
        check(bare[-1][0] + front > 1.2, f"seam wall@{v}: the bare base did not reach through the wall "
                                         f"({bare[-1][0] + front:.3f}) -- the check is not testing anything")
        kc = make(WALL)
        base, art, poses = seam_drive(kc, (v, 0.0, 0.0), n)
        gap = 1.0 - (poses[-1][0] + front)
        check(kc.skin - 0.003 <= gap <= kc.skin + v * DT + 1e-6,
              f"seam wall@{v}: gap {gap:.4f} (skin {kc.skin})")
        check(core_penetrates(kc, poses[-1][0], poses[-1][1], poses[-1][2]) is None, f"seam wall@{v}: penetrated")
        check(abs(float(art.p[0, 0]) - poses[-1][0]) < 1e-6,
              f"seam wall@{v}: sim x {float(art.p[0, 0]):.4f} != integrated {poses[-1][0]:.4f}")
        sm = kc.summary()
        check(sm["blocked_steps"] > 0 and sm["first_contact"]["box"] in ("low", "carriage", "left_arm",
                                                                          "right_arm", "column"),
              f"seam wall@{v}: metric {sm['blocked_steps']} blocked, first {sm['first_contact']}")
        check(base.faults == 0, f"seam wall@{v}: base faults {base.faults}")
    # A refused turn: a slab at arm height only (above the carriage, 0.825 m), its face 1 cm
    # off the left arm's tip -- INSIDE the rotation shell before the turn starts, which is
    # exactly the case "refuse only what is new in the shell" let through. Turning +z swings
    # the arm's outer corner (0.196, -0.306) toward +x.
    arm_x = MEASURED["left_arm"][1][0]
    slab = {"armslab": ((arm_x + 0.01, -3.0, Z + 0.85), (arm_x + 0.3, 3.0, 2.0))}
    kc = make(slab)
    base, _art, poses = seam_drive(kc, (0.0, 0.0, 1.2), 90)
    check(kc.summary()["rot_blocked_steps"] > 0 and abs(poses[-1][2]) < 1.2 * 90 * DT - 0.05,
          f"seam turn: refused {kc.summary()['rot_blocked_steps']} steps, yaw {poses[-1][2]:.3f}")
    worst = None
    for pz in poses:
        worst = worst or core_penetrates(kc, *pz)
    check(worst is None, f"seam turn: penetrated ({worst})")
    # ...and the same start may turn AWAY from it (-z swings that corner toward -x).
    kc = make(slab)
    base, _art, poses = seam_drive(kc, (0.0, 0.0, -1.2), 60)
    check(kc.summary()["rot_blocked_steps"] == 0 and abs(poses[-1][2] + 1.2 * 60 * DT) < 1e-9,
          f"seam turn away: refused {kc.summary()['rot_blocked_steps']} steps, yaw {poses[-1][2]:.3f}")
    # stop(): dt = 0 commands no physics and is not a step; held zero commands are steps
    kc = make({})
    base, art, _ = seam_drive(kc, (0.0, 0.0, 0.0), 10)
    base.stop()
    check(kc.summary()["steps"] == 10, f"seam stop: {kc.summary()['steps']} steps counted for 10 + stop()")
    kc.resolve(0.0, 0.0, Z, 0.0, 0.0, 0.0, 0.0, 0.0)
    check(kc.summary()["steps"] == 10, f"resolve(dt=0) counted as a step ({kc.summary()['steps']})")
    # z: read from the articulation, or taken from a 4th element without a read
    kc = make(WALL)
    art = FakeArt(x=0.5, z=Z + 0.5)
    base = BD.KinematicBase(art, ["wheel1", "wheel2", "wheel3"])
    base.sync_from_sim()
    kc.attach(base)
    reads = art.pose_reads
    kc.motion_filter((0.5, 0.0, 0.0), 0.02, 0.0, 0.0, DT)
    check(art.pose_reads == reads + 1, f"seam z: {art.pose_reads - reads} pose reads for a 3-tuple start")
    reads = art.pose_reads
    out = kc.motion_filter((1.0 - front - 0.005, 0.0, 0.0, Z), 0.02, 0.0, 0.0, DT)
    check(art.pose_reads == reads and out[0] < 0.02, f"seam z: 4-tuple start read {art.pose_reads - reads} "
                                                     f"poses, dx {out[0]:.4f}")
    kc.detach()
    check(base.motion_filter is None, "seam: detach() left the filter attached")


def main() -> int:
    WALL = {"wall": ((1.0, -3.0, 0.0), (1.2, 3.0, 2.0))}
    front = MEASURED["low"][1][0]   # +x extent of the envelope (the wheels/deck, 0.2238)
    front = max(front, MEASURED["left_arm"][1][0], MEASURED["carriage"][1][0])

    print("open: free space returns the command untouched")
    kc = make({})
    rng = np.random.default_rng(0)
    for _ in range(200):
        vx, vy, om = rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 0.5), rng.uniform(-1.2, 1.2)
        r = kc.resolve(0.0, 0.0, Z, rng.uniform(-3, 3), vx, vy, om, DT)
        check((r.vx, r.vy, r.omega) == (vx, vy, om), f"open space changed {(vx, vy, om)} -> {(r.vx, r.vy, r.omega)}")
    check(kc.summary()["contact_steps"] == 0, "open space counted contacts")

    for v in (0.45, 1.5):
        print(f"wall: head-on at {v} m/s")
        kc = make(WALL)
        x, y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (v, 0.0, 0.0), int(2.0 / v / DT), skin_check=True)
        check(SKIN_VIOLATIONS[0] == 0, f"wall@{v}: inside the skin on {SKIN_VIOLATIONS[0]} steps")
        gap = 1.0 - (x + front)
        check(pen is None, f"wall@{v}: core penetrated ({pen})")
        check(KC.SKIN_M - 1e-3 <= gap <= KC.SKIN_M + v * DT + 2e-3,
              f"wall@{v}: stopped {gap:.4f} m from the wall, expected the skin {KC.SKIN_M}")
        sm = kc.summary()
        check(sm["blocked_steps"] > 0 and sm["first_contact"]["prim"] == "wall",
              f"wall@{v}: metric missed it {sm['first_contact']}")

    print("slide45: 45 deg into a wall keeps the tangential half")
    kc = make(WALL)
    v = 0.6
    x, y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (v * math.sqrt(0.5), v * math.sqrt(0.5), 0.0), 120, skin_check=True)
    check(SKIN_VIOLATIONS[0] == 0, f"slide45: inside the skin on {SKIN_VIOLATIONS[0]} steps")
    r = kc.resolve(x, y, Z, yaw, v * math.sqrt(0.5), v * math.sqrt(0.5), 0.0, DT)
    check(pen is None, f"slide45: core penetrated ({pen})")
    check(abs(r.vx) < 1e-6, f"slide45: normal component survived: vx {r.vx:.5f}")
    check(abs(r.vy - v * math.sqrt(0.5)) < 0.01 * v, f"slide45: tangential {r.vy:.4f} != {v * math.sqrt(0.5):.4f}")
    y0 = y
    x, y, yaw, pen = drive(kc, x, y, yaw, (v * math.sqrt(0.5), v * math.sqrt(0.5), 0.0), 60)
    check(y - y0 > 0.99 * v * math.sqrt(0.5) * 1.0, f"slide45: slid only {y - y0:.3f} m in 1 s")

    print("sticky: PhysX-like initial overlap 2 mm short of contact must not stall a slide")
    for name, boxes, cmd, axis in (
            ("wall45", WALL, (0.6 * math.sqrt(0.5), 0.6 * math.sqrt(0.5), 0.0), 1),
            ("wall_shallow", WALL, (0.3, 0.55, 0.0), 1)):
        kc = make(boxes, sticky=2e-3)
        x, y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, cmd, 300, skin_check=True)
        check(pen is None, f"sticky {name}: core penetrated ({pen})")
        r = kc.resolve(x, y, Z, yaw, *cmd, DT)
        check(abs(r.vy - cmd[1]) < 0.01 * abs(cmd[1]) and abs(r.vx) < 1e-6,
              f"sticky {name}: slide stalled, resolved ({r.vx:.4f}, {r.vy:.4f}) for {cmd[:2]}")
        check(kc.summary()["in_contact_steps"] == 0, f"sticky {name}: counted a real contact")

    print("corner: two walls stop it, no penetration, the skin holds on both")
    for name, cmd in (("diag", (0.6, 0.6, 0.0)), ("shallow", (0.6, 0.15, 0.0)), ("steep", (0.15, 0.6, 0.0))):
        kc = make({"wall": WALL["wall"], "wall2": ((-3.0, 0.6, 0.0), (3.0, 0.8, 2.0))})
        x, y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, cmd, 400, skin_check=True)
        check(pen is None, f"corner {name}: core penetrated ({pen})")
        check(SKIN_VIOLATIONS[0] == 0, f"corner {name}: inside the skin on {SKIN_VIOLATIONS[0]} steps")
        r = kc.resolve(x, y, Z, yaw, *cmd, DT)
        check(math.hypot(r.vx, r.vy) < 1e-6, f"corner {name}: still moving {r.vx:.4f},{r.vy:.4f}")
    print("pillar: sliding along a wall into a post sticking out of it")
    kc = make({"wall": WALL["wall"], "post": ((0.90, 0.45, 0.0), (1.0, 0.60, 2.0))})
    x, y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.5, 0.5, 0.0), 300, skin_check=True)
    check(pen is None, f"pillar: core penetrated ({pen})")
    check(SKIN_VIOLATIONS[0] == 0, f"pillar: inside the skin on {SKIN_VIOLATIONS[0]} steps")

    print("low_step: under MAX_STEP climbs, over it blocks")
    for h, blocks in ((0.030, False), (0.080, True)):
        kc = make({"step": ((1.0, -2.0, 0.0), (3.0, 2.0, h))})
        x, _y, _yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.45, 0.0, 0.0), 300)
        reached = x > 0.99 * 0.45 * 300 * DT          # travelled the whole command
        check(reached != blocks, f"step {h}: blocks={blocks} expected, reached={reached} (x {x:.3f})")
        if blocks:
            check(pen is None and 1.0 - (x + MEASURED['low'][1][0]) >= KC.SKIN_M - 1e-3,
                  f"step {h}: stopped too close or inside (x {x:.3f}, {pen})")

    print("slab: arm-height slab blocks, one above the robot does not")
    for z0, blocks in ((0.85, True), (1.20, False)):
        kc = make({"slab": ((1.0, -2.0, z0), (1.8, 2.0, z0 + 0.05))})
        x, _y, _yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.45, 0.0, 0.0), 300)
        reached = x > 0.99 * 0.45 * 300 * DT
        check(reached != blocks, f"slab {z0}: blocks={blocks} expected, reached={reached} (x {x:.3f})")
        check(pen is None, f"slab {z0}: core penetrated ({pen})")
        if blocks:
            arm_front = MEASURED["left_arm"][1][0]
            check(abs(1.0 - (x + arm_front) - KC.SKIN_M) < 0.45 * DT + 2e-3,
                  f"slab {z0}: the ARM should stop at the skin, gap {1.0 - (x + arm_front):.4f}")

    print("turn: arm tip near a wall cannot turn into it; free space always turns")
    tip = math.hypot(0.1957, 0.3061)
    wall_y = -tip - 0.004
    kc = make({"side": ((-2.0, wall_y - 0.2, 0.0), (2.0, wall_y, 2.0))})
    for om in (0.6, -0.6, 1.2):
        kc.reset_stats()
        _x, _y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.0, 0.0, om), int(math.pi / abs(om) / DT))
        check(pen is None, f"turn {om}: core penetrated ({pen})")
        check(kc.summary()["rot_blocked_steps"] > 0, f"turn {om}: never refused")
    # Starts OUTSIDE the clearance margin (the deck's +y face is 0.16 m off the wall), and only
    # the turn brings a corner (r 0.32 m) to it: the free-space cache must not carry it there.
    wall_y = MEASURED["low"][1][1] + KC.CLEAR_MARGIN_M + 0.01
    for om in (0.6, 1.2):
        kc = make({"deckside": ((-2.0, wall_y, 0.0), (2.0, wall_y + 0.2, 2.0))})
        _x, _y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.0, 0.0, om), int(math.pi / om / DT))
        check(pen is None, f"turn from clear space {om}: core penetrated ({pen})")
        check(kc.summary()["rot_blocked_steps"] > 0 and kc.summary()["cached_steps"] > 0,
              f"turn from clear space {om}: {kc.summary()['rot_blocked_steps']} refused, "
              f"{kc.summary()['cached_steps']} cached")
    kc = make({})
    _x, _y, yaw, _ = drive(kc, 0.0, 0.0, 0.0, (0.0, 0.0, 1.2), 300)
    check(abs(yaw - 1.2 * 300 * DT) < 1e-9 and kc.summary()["rot_blocked_steps"] == 0, "free turn was refused")

    print("lowbar: a 6.5 cm bar beside the deck -- a turn must not swing the deck into it")
    bar_y = MEASURED["low"][1][1] + 0.03
    for om in (0.6, -0.6, 1.2):
        kc = make({"lowbar": ((-1.0, bar_y, 0.0), (1.0, bar_y + 0.1, 0.065))})
        _x, _y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.0, 0.0, om), int(math.pi / 2 / abs(om) / DT))
        check(pen is None, f"lowbar turn {om}: core penetrated ({pen})")
        check(kc.summary()["rot_blocked_steps"] > 0, f"lowbar turn {om}: never refused")
    kc = make({"lowbar": ((-1.0, bar_y, 0.0), (1.0, bar_y + 0.1, 0.040))})
    _x, _y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.0, 0.0, 1.2), int(math.pi / 2 / 1.2 / DT))
    check(kc.summary()["rot_blocked_steps"] == 0, "lowbar 4 cm (under MAX_STEP): the turn was refused")

    print("escape: spawned 3 cm inside a wall")
    x0 = 1.0 - front + 0.03
    kc = make(WALL)
    r = kc.resolve(x0, 0.0, Z, 0.0, 0.45, 0.0, 0.0, DT)
    check(r.vx <= 1e-9 and (r.touching or r.in_contact),
          f"escape: allowed deeper (vx {r.vx:.4f}, touching {r.touching}, in_contact {r.in_contact})")
    r = kc.resolve(x0, 0.0, Z, 0.0, -0.45, 0.0, 0.0, DT)
    check(abs(r.vx + 0.45) < 1e-9, f"escape: backing out refused (vx {r.vx:.4f})")
    r = kc.resolve(x0, 0.0, Z, 0.0, 0.0, 0.45, 0.0, DT)
    check(abs(r.vy - 0.45) < 1e-9, f"escape: sliding refused (vy {r.vy:.4f})")

    print("kerbcorner: one kerb prim, low under a deck corner and 13 cm high in front, is not floor")
    # outdoor_small, 20260905-115418 window (Kit 2026-10-06): a corner ray found isl_curb
    # piece_1303 where it is low, the whole prim became "support", and the low box drove into
    # its high part with nothing flagged. Same shape here: the high part already 1 cm off the
    # box (inside the 2 cm skin), so every tier starts in initial overlap.
    lo_x, lo_y = MEASURED["low"][1][0], MEASURED["low"][1][1]
    kerb = {"kerb": [((lo_x - 0.03, lo_y - 0.06, 0.0), (lo_x + 0.08, lo_y + 0.14, 0.02)),
                     ((lo_x + 0.01, -0.6, 0.0), (lo_x + 0.6, 0.6, 0.13))]}
    for cmd in ((0.3, 0.0, 0.0), (0.25, 0.15, 0.0), (0.2, 0.0, 0.6)):
        kc = make(kerb, sticky=0.002)
        x, y, yaw, pen = drive(kc, 0.0, 0.0, 0.0, cmd, 60)
        check(pen is None, f"kerbcorner {cmd}: core penetrated ({pen})")
        check(x < 0.01 + 1e-6, f"kerbcorner {cmd}: advanced {x:.4f} m toward a 13 cm kerb 1 cm away")
    # ...while the same low part alone (2 cm, the floor) is never a reason to stop.
    kc = make({"sill": kerb["kerb"][0]}, sticky=0.002)
    x, _y, _yaw, pen = drive(kc, 0.0, 0.0, 0.0, (0.3, 0.0, 0.0), 60)
    check(abs(x - 0.3 * 60 * DT) < 1e-6 and kc.summary()["contact_steps"] == 0,
          f"kerbcorner: a 2 cm sill under the corner stopped the robot (x {x:.4f})")

    print("creep: a box already 0.5 mm into a wall may slide along it, never sink further")
    # The Kit case exactly: PhysX's overlap tolerance let a turn end with the low box a hair
    # inside SM_WheelChair_01a; the old ladder then counted every later step as the same rung
    # and the robot sank 10.8 mm in 1-2 mm steps. Here: into the wall at 10 deg off tangent.
    x0 = 1.0 - front + 0.0005
    for sticky in (0.0, 0.002):
        kc = make(WALL, sticky=sticky)
        x, y, yaw, _pen = drive(kc, x0, 0.0, 0.0, (0.3 * math.sin(math.radians(10)),
                                                    0.3 * math.cos(math.radians(10)), 0.0), 120)
        sank = x + front - 1.0
        check(sank <= 0.0005 + 1e-6, f"creep sticky={sticky}: sank to {1e3 * sank:.2f} mm (started 0.50)")
        check(y > 0.9 * 0.3 * math.cos(math.radians(10)) * 120 * DT,
              f"creep sticky={sticky}: did not slide along the wall (y {y:.3f})")

    print("jiggle: 3000 random commands pushing at a wall, a corner and a pillar -- no creep, no trap")
    # The hospital desk window (Kit, 2026-10-06) crept 10.8 mm into a wheelchair through moves
    # each of which stayed on one rung of the old 9-rung depth ladder. Random small commands,
    # biased into the obstacle, are the offline version of that: the box may touch, never sink.
    rng = np.random.default_rng(20261006)
    scenes = {
        "wall": ({"wall": ((1.0 - front + 0.003, -3.0, 0.0), (1.3, 3.0, 2.0))}, (1.0, 0.0)),
        "corner": ({"wx": ((1.0 - front + 0.003, -3.0, 0.0), (1.3, 3.0, 2.0)),
                    "wy": ((-3.0, 0.40, 0.0), (3.0, 0.7, 2.0))}, (1.0, 1.0)),
        "pillar": ({"post": ((0.24, -0.05, 0.0), (0.34, 0.05, 2.0))}, (1.0, 0.0)),
        "armslab": ({"slab": ((MEASURED["left_arm"][1][0] + 0.004, -3.0, Z + 0.85),
                              (MEASURED["left_arm"][1][0] + 0.3, 3.0, 2.0))}, (1.0, 0.0)),
    }
    for name, (boxes, bias) in scenes.items():
        kc = make(boxes, sticky=0.002)
        x = y = yaw = 0.0
        worst = None
        for _ in range(3000):
            vx = float(rng.uniform(-0.2, 0.6)) * bias[0]
            vy = float(rng.uniform(-0.2, 0.6)) * bias[1] if bias[1] else float(rng.uniform(-0.3, 0.3))
            om = float(rng.uniform(-1.2, 1.2))
            r = kc.resolve(x, y, Z, yaw, vx, vy, om, DT)
            c, s = math.cos(yaw), math.sin(yaw)
            x += (r.vx * c - r.vy * s) * DT
            y += (r.vx * s + r.vy * c) * DT
            yaw += r.omega * DT
            worst = worst or core_penetrates(kc, x, y, yaw, grow=-(KC.DEPTH_TOL_M + 1e-4)
                                             if hasattr(KC, "DEPTH_TOL_M") else -6e-4)
        check(worst is None, f"jiggle {name}: the core sank into {worst}")
        # Not trapped: backing straight out of the obstacle is always allowed.
        away = (-0.3 * bias[0], -0.3 * bias[1] if bias[1] else 0.0)
        r = kc.resolve(x, y, Z, yaw, *(np.array([[math.cos(yaw), math.sin(yaw)],
                                                  [-math.sin(yaw), math.cos(yaw)]]) @ away), 0.0, DT)
        check(math.hypot(r.vx, r.vy) > 0.1, f"jiggle {name}: trapped, backing out refused "
                                              f"({r.vx:.3f}, {r.vy:.3f})")

    print("dynanav: physical_collision needs 3.33 s of continuous contact")
    for n, want in ((199, False), (200, True)):
        kc = make({})
        for _ in range(n):
            kc._account(KC.ContactResult(0.0, 0.0, 0.0, blocked=True, prim="p", box="low"), True)
        kc._account(KC.ContactResult(0.3, 0.0, 0.0), True)
        check(kc.summary()["physical_collision"] is want, f"dynanav: {n} steps -> {kc.summary()['physical_collision']}")

    check_seam(WALL, front)

    print("constants: second copies of a measured number must still agree with their source")
    # DECK_SPLIT_Z_M and lidar_2d.COLUMN_Z_M[0] were both read off base_link.STL (deck top /
    # column base). Loaded by file path so a KC_SIM_DIR mutant does not need its own copy.
    import importlib.util
    spec = importlib.util.spec_from_file_location("_lidar_2d_ref", REPO / "nav/sim/lidar_2d.py")
    lidar = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lidar          # its dataclasses look themselves up there
    spec.loader.exec_module(lidar)
    check(KC.DECK_SPLIT_Z_M == lidar.COLUMN_Z_M[0],
          f"DECK_SPLIT_Z_M {KC.DECK_SPLIT_Z_M} != lidar_2d.COLUMN_Z_M[0] {lidar.COLUMN_Z_M[0]}")
    check(abs(KC.DYNANAV_SUSTAINED_S - 100.0 / 30.0) < 1e-12, "DYNANAV_SUSTAINED_S is not 100 frames @ 30 FPS")

    print(f"\n{CHECKS[0] - len(FAILS)}/{CHECKS[0]} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
