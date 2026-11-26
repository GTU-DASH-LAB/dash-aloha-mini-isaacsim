"""Draw a menu of drivable arcs onto the robot's own camera image.

Why this exists. Two models, trained completely differently, failed in the same place:
they reason correctly about where to go and then emit a trajectory that ignores it. The
common factor is not their training, it is that both were asked for *metric coordinates*
-- and neither has any idea how far 2.07 m is in the picture in front of it. Asked for a
number it cannot ground, a model falls back on what it can do: copy a number out of the
prompt, or repeat a memorised straight line. Both were measured; see CLAUDE.md.

So stop asking. Generate the candidate paths ourselves, where the geometry is exact and
free, project them onto the floor of the image, label them, and ask the model to pick one.
The spatial reasoning moves *into the image*, which is where a VLM's spatial competence
actually lives: it no longer has to imagine where (1.8, -0.6) is, it can see the curve
drawn on the floor. And the failure mode we measured becomes structurally impossible --
there is no number to copy when the answer is an integer.

The geometry is a level-camera flat-floor pinhole projection:

    fx = (W/2) / tan(fov_h / 2),  cx = W/2, cy = H/2, square pixels
    a ground point (x forward, y left) seen from a lens c metres ahead of the rotation
    centre and h metres above the floor maps to
    u = cx - fx * y / (x - c)        v = cy + fy * h / (x - c)

Nothing here is learned or fitted, and there are two geometries, chosen once per process
by QVLA_MENU_GEOM:
  legacy   c = 0, h = 0.346 -- the lens at the rotation centre. Every result recorded up to
           2026-10 was drawn this way, including the paper's. It is wrong by the camera's
           0.25 m forward mount and 0.008 m ride height, so each drawn point sits ~0.27 /
           0.29 / 0.32 m further down the floor than the driven path at 1 / 2 / 3 m and
           curved arcs look 0.8-4 deg straighter than they are.
  mounted  c = NAV_CAMERA_FORWARD_M, h = NAV_CAMERA_HEIGHT_M + BASE_LINK_REST_Z_M, from
           scripts/alohamini1_specs.py -- what camera_nav actually is.
`legacy` stays the default so probes and videos of old runs reproduce bit for bit; a new
measurement should run `mounted` and say so (it is on /health as `menu_geom`).

A THIRD SOURCE, per request: `MenuCamera`. One server drives more than one body -- AlohaMini's
level camera_nav and the G1's head D435, pitched 12 deg down on its mount and bobbing 11-16 deg
while walking -- so a /predict request may carry the camera its frame was taken with:
intrinsics, mount and the LIVE attitude at frame time, plus an optional subset of the menu's
arcs. That camera is a full pinhole with pitch/roll/yaw (same convention as
nav/sim/g1/camera_geometry.py, checked against it by nav/tools/check_menu_camera.py). A request
WITHOUT one goes through the exact code path above, untouched, so every recorded menu still
reproduces byte for byte.

Label assignment is deliberately NOT left-to-right. Numbering the arcs in spatial order
lets a model score well by answering "1" for left and "K" for right without ever looking
at the image, which is exactly the confound that would make this whole approach look like
it works when it does not. Callers pass their own permutation; the probe randomises it.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from alohamini1_specs import (  # noqa: E402
    BASE_LINK_REST_Z_M, NAV_CAMERA_FORWARD_M, NAV_CAMERA_HEIGHT_M,
)

# The nav camera: level, looking along +x. See the module docstring for the two geometries.
MENU_GEOM = os.environ.get("QVLA_MENU_GEOM", "legacy").strip().lower()
_GEOMS = {
    "legacy": (0.346, 0.0),
    "mounted": (NAV_CAMERA_HEIGHT_M + BASE_LINK_REST_Z_M, NAV_CAMERA_FORWARD_M),
}
if MENU_GEOM not in _GEOMS:
    # Loudly: a typo falling back to `legacy` would label a run with the geometry it was
    # meant to replace.
    raise SystemExit(f"QVLA_MENU_GEOM must be one of {sorted(_GEOMS)}, got {MENU_GEOM!r}")
CAM_HEIGHT_M, CAM_FORWARD_M = _GEOMS[MENU_GEOM]
CAM_FOV_DEG = 90.1
CAM_W, CAM_H = 1920, 1080

# HOW MANY ARCS THE MENU MAY HOLD, and why the answer is not "as many as you like".
#
# The arcs fan out from one point, so their endpoints crowd together near straight ahead
# in the IMAGE even where they are far apart on the floor. What limits the menu is
# therefore not the geometry, it is whether two labels can be told apart after the
# model's own downsample: frames are capped at QVLA_MAX_PIXELS (200704 px = 448x448),
# which is 0.31x, so a 96 px label arrives about 30 px across.
#
# Measured, at 3 m. CORRECTED 2026-10: the first version of this table said the disc was
# ~115 px and that `fine` had no collisions. The disc `render_menu` draws is 2 * 96 * 0.72
# = 138 px, and with `badge_xy` as written (unchanged since this table was added) `fine`
# has two colliding pairs, the +-0.45/+-0.60 badges 115 px apart -- visible in recorded
# warehouse_aisle6 videos of the hard_study 'all' arm. 158 px is its smallest gap that does
# NOT collide. Under the `mounted` geometry those pairs close to 75 px.
#
#     set       n    smallest gap (legacy)    colliding pairs    smallest gap (mounted)
#     coarse    7          219 px                   0                  240 px
#     fine     11          115 px                   2                   75 px
#     (13)     13          108 px                   4   <- rejected
#
# So `coarse`, the shipping set, separates under both geometries; `fine` needs the outer
# badges walked back along their arcs before it can be trusted, and 13 is not a finer menu,
# it is the same menu with pairs of labels drawn on top of each other. The extra
# resolution goes where there was a real gap -- between "gentle" (0.15) and "medium"
# (0.35), a 26-to-60 degree jump with nothing in between. Nothing is added inside +-0.15:
# those endpoints are 108 px apart and would collide, and a 3 m arc at 0.075 differs from
# straight by 0.33 m, which the next replan three seconds later can express anyway.
CURVATURE_SETS: dict[str, tuple[float, ...]] = {
    "coarse": (-0.60, -0.35, -0.15, 0.0, 0.15, 0.35, 0.60),
    "fine":   (-0.60, -0.45, -0.35, -0.25, -0.15, 0.0, 0.15, 0.25, 0.35, 0.45, 0.60),
}
# Public, and reported on /health, so a ladder can refuse to drive against a server that
# loaded a different menu than the one the campaign is about to record. Checking the NAME
# rather than the arc count keeps the sizes in this table and nowhere else.
ARC_SET = os.environ.get("QVLA_MENU_ARCS", "coarse").strip().lower()
if ARC_SET not in CURVATURE_SETS:
    # Loudly, for the reason QVLA_THINK_LEVEL is: a typo that fell back to the default
    # would run the seven-arc menu and label the results as the eleven-arc one.
    raise SystemExit(
        f"QVLA_MENU_ARCS must be one of {sorted(CURVATURE_SETS)}, got {ARC_SET!r}")
DEFAULT_CURVATURES = CURVATURE_SETS[ARC_SET]
# 3.0 m matches the horizon both policies were compared on, so a chosen arc is directly
# comparable to a predicted trajectory.
DEFAULT_LENGTH_M = 3.0

# Colours are per-arc and fixed by index so the same arc looks the same across frames,
# but they are never mentioned in the prompt -- the model must use position, not hue.
# Eleven entries because the `fine` set needs eleven; the index wraps, so a shorter set
# simply uses the first few and keeps the colours it always had.
_PALETTE = [
    (255, 87, 51), (255, 189, 51), (163, 255, 51), (51, 255, 128),
    (51, 214, 255), (122, 51, 255), (255, 51, 195),
    (255, 140, 90), (200, 255, 90), (90, 255, 220), (190, 120, 255),
]

# --------------------------------------------------------------------------------------
# TURNING IN PLACE, and why it is a separate thing from every curvature above
#
# Every arc in the menu is 3 m of FORWARD travel. The tightest of them, kappa 0.60, sweeps
# 103 degrees -- but it ends 1.63 m in front of where it started, because a curvature is a
# shape you drive along, not a heading you adopt. So when the way ahead is blocked at 1 m,
# the model has seven paths that all hit the obstacle and one label that means "stop". It
# has no way at all to say "I need to be facing somewhere else before any of these is
# usable", because that sentence does not exist in the menu.
#
# That is not a prompt problem and it cannot be fixed with one. The selector prompt has
# said "do not answer STOP merely because the way ahead is tight" since the first ladder,
# and the first ladder still answered STOP on 42 of 65 decisions from 5.5 m out while its
# OWN describe call named an open side on the same frames. It answered the only label that
# did not drive into something.
#
# Measured on the synchronous ladder, four failures against three successes:
#
#     frozen*  guard  recoveries          frozen*  guard  recoveries
#     16-35%   0-299     1-4                  0%    0-61       0
#     ^ the four failures                     ^ the three successes
#     *share of the trace spent inside a 0.5 m circle
#
# Perfect separation on two variables, and neither of them is "drove to the wrong place".
# Every failure is the robot stuck facing a direction it cannot navigate out of.
#
# A pivot is the missing verb. It has no curvature -- 1/kappa is zero, the radius is zero,
# it is not on the same axis as the arcs at all -- so it cannot be drawn on the floor and
# it cannot be expressed as waypoints either: a pure rotation is every waypoint at the
# origin, which `plan_speed` reads as STOP. It needs its own drawn symbol and its own
# channel back to the runner, and that is what the rest of this section builds.
#
# The turn is small on purpose: the robot re-decides the moment a pivot finishes, so a
# bigger turn is expressed by choosing it AGAIN on a fresh view rather than by committing
# to a sweep taken blind. What the right size is, is an open question and therefore a knob.
#
# 30 degrees was the first guess and it cut collisions five-fold (guard 956 -> 195 over a
# 13-episode ladder) without moving success. The suspicion behind lowering it: a 30 degree
# step is coarse next to the arcs it competes with -- the whole menu spans about 60 degrees
# of heading change -- so a turn aimed at an opening can carry the robot's heading PAST it,
# and the correction is another turn in the other direction. 15 degrees costs one extra
# decision to cover the same angle and cannot overshoot by more than half as much.
#
# Kept as a constant read once at import, not a per-call argument, because the runner sizes
# the pivot's step deadline from the SAME number and the two disagreeing would leave the
# robot re-deciding mid-rotation.
PIVOT_DEG = float(os.environ.get("QVLA_PIVOT_DEG", "15"))
PIVOT_RAD = math.radians(PIVOT_DEG)


@dataclass(frozen=True)
class Arc:
    """One constant-curvature candidate path in the robot's body frame (FLU)."""

    kappa: float          # 1/m; positive turns LEFT, matching y-positive-is-left
    points: np.ndarray    # (N, 2) of (x_forward, y_left), starting at the robot

    @property
    def end(self) -> tuple[float, float]:
        return float(self.points[-1, 0]), float(self.points[-1, 1])

    @property
    def heading_deg(self) -> float:
        return math.degrees(math.atan2(self.points[-1, 1], self.points[-1, 0]))


def make_arcs(curvatures=DEFAULT_CURVATURES, length_m: float = DEFAULT_LENGTH_M,
              n_points: int = 40) -> list[Arc]:
    """Constant-curvature arcs, the paths a differential-drive base can actually follow.

    A straight line and a circle are the only trajectories a fixed (v, omega) produces, so
    every candidate here is executable by definition -- which is the point. The model is
    never given the option to choose something the robot cannot drive.
    """
    out = []
    s = np.linspace(0.0, length_m, n_points)
    for k in curvatures:
        if abs(k) < 1e-9:
            pts = np.stack([s, np.zeros_like(s)], axis=1)
        else:
            r = 1.0 / k
            theta = k * s
            pts = np.stack([r * np.sin(theta), r * (1.0 - np.cos(theta))], axis=1)
        out.append(Arc(kappa=float(k), points=pts))
    return out


CHASSIS_RADIUS_M = 0.35


def arc_clearance(arcs: list[Arc], points, radius_m: float = CHASSIS_RADIUS_M,
                  length_m: float = DEFAULT_LENGTH_M) -> list[float]:
    """How far along each arc the chassis gets before it touches a lidar return.

    A SWEPT-DISC TEST, not a bearing test, and the difference is the entire value of the
    thing. Asking "is the bearing at the end of arc 3 clear" answers a question about a
    ray; the robot is 0.7 m wide and follows a curve, so what has to be clear is a 0.35 m
    tube along the path. A gap the beam slips through between two rack legs is not a
    corridor, and the bearing test calls it one.

    This is what the camera structurally cannot supply. `gate_lidar_blindside.py` measured
    the other candidate channel -- openings outside the 90 degree frame -- and it carries
    nothing: the goal is outside the frame on 96% of failure moments, the scan contains it
    54% of the time, and the blind arc is open 65% of the time anyway, so the lift is
    -11pp. That channel reports that a room is a room. THIS one answers a question the
    model demonstrably gets wrong: asked to pick a drawn arc with an obstacle in view, it
    chose one that runs into the wall on 89% of blocked frames, and 0 of 18 choices
    negated when the frame was mirrored. Perception was never the problem there -- forced
    to choose between a blocked and an open frame it scored 70/70 -- so the fix does not
    need better eyes, it needs the geometry the eyes cannot measure.

    Returns metres per arc, capped at the arc's own length: `length_m` means "clear all
    the way", NOT "clear forever", and a caller comparing against a threshold above
    `length_m` would reject every arc on a completely empty floor.
    """
    if points is None or len(points) == 0:
        return [length_m] * len(arcs)
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    # Blocked means "brings the chassis CLOSER than it already is", once something is
    # already inside the disc. Every arc starts at the robot centre, so with a fixed 0.35 m
    # disc any return within 0.35 m -- a wall to one side, or BEHIND -- blocked every arc at
    # sample 0 and zeroed the whole menu, and `drivable` then kept index 0, the hardest
    # right turn, even with that wall on the right. Measured: wall 0.30 m behind ->
    # [0]*7; wall 0.30 m to the right -> [0]*7, kept kappa -0.60, i.e. into it. Shrinking
    # the threshold to just under the current nearest return keeps the open side open
    # (right wall at 0.30 -> [0.31, 0.38, 0.54, 3, 3, 3, 3]) and is IDENTICAL to the fixed
    # disc whenever the nearest return is beyond 0.37 m, so the normal regime -- and every
    # recorded number -- is unchanged.
    d0 = float(np.hypot(P[:, 0], P[:, 1]).min())
    thr = min(radius_m, d0 - 0.02)
    out = []
    for arc in arcs:
        A = arc.points
        # (n_arc, n_points) -- 7 x 40 x 500 for the coarse menu and a full revolution,
        # which is 140k distances and takes microseconds. No need to be clever.
        near = ((A[:, None, 0] - P[None, :, 0]) ** 2
                + (A[:, None, 1] - P[None, :, 1]) ** 2) < thr ** 2
        idx = np.flatnonzero(near.any(axis=1))
        # The first blocked sample, not the count: an arc clear for 2 m and blocked after
        # is drivable for 2 m, and taking the whole arc as blocked would throw away the
        # part the robot can actually use before its next decision.
        out.append(length_m if idx.size == 0
                   else float(length_m * idx[0] / (len(A) - 1)))
    return out


def drivable(arcs: list[Arc], labels: list[int], clearance: list[float],
             min_clear_m: float) -> tuple[list[Arc], list[int], list[int]]:
    """Split the menu into what may be offered and what may not.

    Returns (kept arcs, kept labels, dropped labels). The LABELS ARE NOT REALLOCATED and
    that is deliberate on two counts. `stop_label()` derives STOP's number from the arc
    count, so renumbering a shrunken menu would move STOP -- silently turning "stop" into
    "turn right", which is the exact failure that function's docstring warns about. And
    the permutation is drawn from the same RNG stream every earlier run used, so leaving
    it untouched keeps the shuffle bit-for-bit comparable with every arm already on disk.
    The menu simply has holes in it, which is what a menu of drivable paths should look
    like.

    An empty menu is never returned. If nothing clears the threshold the single roomiest
    arc is kept, because the alternative is a decision where the only answers are STOP and
    a pivot -- and STOP-as-"everything is blocked" is a measured pathology on this stack,
    42 of 65 decisions on one episode, not a hypothetical.
    """
    keep = [i for i, c in enumerate(clearance) if c >= min_clear_m]
    if not keep:
        # Ties go to the STRAIGHTEST arc. `max` alone returns the first index, which is
        # kappa -0.60, so an all-equal menu always collapsed to the hardest right turn.
        keep = [max(range(len(clearance)),
                    key=lambda i: (clearance[i], -abs(arcs[i].kappa)))]
    kept = set(keep)
    return ([arcs[i] for i in keep], [labels[i] for i in keep],
            [labels[i] for i in range(len(arcs)) if i not in kept])


def project(x_forward: float, y_left: float, w: int = CAM_W, h: int = CAM_H,
            fov_deg: float = CAM_FOV_DEG, cam_h: float = CAM_HEIGHT_M,
            cam_x: float = CAM_FORWARD_M) -> tuple[float, float] | None:
    """Ground point in the ROBOT frame (rotation centre) -> pixel.

    None when the point is at or behind the camera plane. Points between the robot and
    the bottom of the frame project below `h`; callers draw through them and PIL clips.
    """
    depth = x_forward - cam_x                   # distance in front of the LENS
    if depth <= 0.05:                           # behind, or on, the image plane
        return None
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    u = w / 2.0 - fx * (y_left / depth)
    v = h / 2.0 + fx * (cam_h / depth)          # square pixels, so fy == fx
    return u, v


# --------------------------------------------------------------------------------------
# A CAMERA PER REQUEST
# --------------------------------------------------------------------------------------
# `project()` above is the server's own camera: level, 90.1 deg, 1920x1080, mount fixed per
# process by QVLA_MENU_GEOM. It is right for AlohaMini's camera_nav and wrong for anything
# else, and the G1 is the case that forced the issue: its head D435 is 69 deg wide, sits
# ~1.09 m up and looks 12 deg down on its mount, and the walking gait swings that to 11-16 deg
# and rolls it a few degrees. The level formula puts every arc 150-250 px too high on that
# camera (camera_geometry.py), and constants measured "while walking" are 253 px off the live
# pose in the worst frame (g1/check, 64 frames). So the menu has to be drawn with the pose of
# the frame it is drawn ON, which only the caller knows -- hence a camera in the request.
#
# The convention is camera_geometry.project_ground()'s, on purpose, so the G1 runner can pass
# G1Locomotion.camera_pose()["menu_args"] through unchanged:
#
#     R_cam = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ M_LEVEL          (heading frame: x fwd, y left, z up)
#     pitch + = looking DOWN, roll + = image right side DOWN, yaw + = looking LEFT of the heading
#     the lens at (cam_x forward, cam_y left, cam_h up) from the floor point under the rotation centre
#
# With B = Rz Ry Rx, B's columns are the camera's own forward / left / up axes, and a floor
# point d = (x - cam_x, y - cam_y, -cam_h) maps to
#
#     depth = fwd . d     u = cx - fx * (left . d) / depth     v = cy - fy * (up . d) / depth
#
# which at zero attitude, cam_y = 0, fx = fy = (W/2)/tan(hfov/2), (cx, cy) = (W/2, H/2) is
# exactly `project()`. Written out here rather than imported from camera_geometry so the check
# tool compares two independent implementations, and so the server does not depend on the G1
# package.
#
# THE KEYS ARE CLOSED. An unknown key is a ValueError, not ignored: `pitch` for `pitch_deg`
# would otherwise draw a LEVEL menu on a pitched camera with no error anywhere -- the
# second-copy-of-a-constant failure this repo keeps paying for.
MENU_CAMERA_KEYS = (
    "name", "width", "height", "fx", "fy", "cx", "cy", "hfov_deg",
    "cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg",
    "arcs", "min_visible_m", "prompt", "robot",
)
# The live-pose keys, the same six camera_geometry.MENU_ARG_KEYS names.
MENU_POSE_KEYS = ("cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg")


def _finite(name: str, v, lo: float | None = None, hi: float | None = None) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"camera.{name} must be a number, got {v!r}") from None
    if not math.isfinite(f):
        raise ValueError(f"camera.{name} must be finite, got {v!r}")
    if (lo is not None and f < lo) or (hi is not None and f > hi):
        raise ValueError(f"camera.{name} must be in [{lo}, {hi}], got {f}")
    return f


@dataclass(frozen=True)
class MenuCamera:
    """The camera a /predict frame was taken with, as far as drawing the menu needs.

    `width`/`height` are the pixels the intrinsics refer to; when the frame actually drawn on
    has another size the intrinsics are scaled to it (`for_image`). Either `fx` (optionally
    `fy`, `cx`, `cy`) or `hfov_deg` gives the focal length, never both -- two statements of
    one number are a disagreement waiting to happen. Square pixels and a centred principal
    point are the defaults.
    """

    cam_h: float                       # lens height above the floor, m
    cam_x: float = 0.0                 # lens forward of the rotation centre, m
    cam_y: float = 0.0                 # lens left of the rotation centre, m
    pitch_deg: float = 0.0             # + = looking down
    roll_deg: float = 0.0              # + = image right side down
    yaw_deg: float = 0.0               # + = looking left of the robot's heading
    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    hfov_deg: float | None = None
    name: str = ""
    # Curvatures to offer, a subset of the server's set; None = all of them.
    arcs: tuple[float, ...] | None = None
    # Drop an arc from THIS frame's menu when less than this much of it is in frame.
    min_visible_m: float | None = None
    # Describe this camera (and `robot`) in the prompts instead of the server's own.
    prompt: bool = False
    robot: str = ""

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_request(cls, d: dict, curvatures=None) -> MenuCamera:
        """Validate a request's `camera` dict. Raises ValueError naming the bad key."""
        if not isinstance(d, dict):
            raise ValueError(f"camera must be an object, got {type(d).__name__}")
        unknown = sorted(set(d) - set(MENU_CAMERA_KEYS))
        if unknown:
            raise ValueError(f"camera has unknown keys {unknown}; allowed: {list(MENU_CAMERA_KEYS)}")
        if d.get("cam_h") is None:
            raise ValueError("camera.cam_h (lens height above the floor, m) is required")
        has_fx, has_fov = d.get("fx") is not None, d.get("hfov_deg") is not None
        if has_fx == has_fov:
            raise ValueError("camera needs exactly one of fx (pixels) or hfov_deg")
        kw: dict = {"cam_h": _finite("cam_h", d["cam_h"], 0.01, 10.0)}
        kw["cam_x"] = _finite("cam_x", d.get("cam_x", 0.0), -5.0, 5.0)
        kw["cam_y"] = _finite("cam_y", d.get("cam_y", 0.0), -5.0, 5.0)
        # Under 90 deg each: a camera turned past that does not look along the floor ahead,
        # and an arc menu drawn for it is meaningless rather than merely skewed.
        for k in ("pitch_deg", "roll_deg", "yaw_deg"):
            kw[k] = _finite(k, d.get(k, 0.0), -89.0, 89.0)
        for k in ("width", "height"):
            if d.get(k) is not None:
                v = _finite(k, d[k], 1, 20000)
                if v != int(v):
                    raise ValueError(f"camera.{k} must be a whole number of pixels, got {v}")
                kw[k] = int(v)
        if has_fov:
            kw["hfov_deg"] = _finite("hfov_deg", d["hfov_deg"], 1.0, 179.0)
            for k in ("fy", "cx", "cy"):
                if d.get(k) is not None:
                    raise ValueError(f"camera.{k} needs fx; with hfov_deg the pixels are square "
                                     f"and the principal point is the image centre")
        else:
            kw["fx"] = _finite("fx", d["fx"], 1.0, 1e6)
            for k in ("fy", "cx", "cy"):
                if d.get(k) is not None:
                    kw[k] = _finite(k, d[k], -1e6, 1e6)
            if kw.get("fy") is not None and kw["fy"] <= 0:
                raise ValueError(f"camera.fy must be positive, got {kw['fy']}")
        if d.get("name") is not None:
            kw["name"] = str(d["name"])[:64]
        if d.get("robot") is not None:
            robot = " ".join(str(d["robot"]).split())
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 \-]{0,39}", robot):
                raise ValueError(f"camera.robot must be a short plain noun phrase, got {d['robot']!r}")
            kw["robot"] = robot
        if d.get("prompt") is not None:
            if not isinstance(d["prompt"], bool):
                raise ValueError(f"camera.prompt must be true/false, got {d['prompt']!r}")
            kw["prompt"] = d["prompt"]
        if d.get("min_visible_m") is not None:
            kw["min_visible_m"] = _finite("min_visible_m", d["min_visible_m"], 0.0, 100.0)
        if d.get("arcs") is not None:
            have = tuple(float(k) for k in (DEFAULT_CURVATURES if curvatures is None else curvatures))
            req = d["arcs"]
            if not isinstance(req, (list, tuple)) or not req:
                raise ValueError(f"camera.arcs must be a non-empty list of curvatures, got {req!r}")
            want = set()
            for k in req:
                kf = _finite("arcs[]", k)
                match = [h for h in have if abs(h - kf) < 1e-6]
                if not match:
                    # Membership, not "nearest": the plan is built from the chosen kappa, and
                    # a curvature the server does not draw is one nobody has looked at.
                    raise ValueError(f"camera.arcs: {kf} is not on this server's menu {list(have)}")
                want.add(match[0])
            kw["arcs"] = tuple(h for h in have if h in want)      # the server's order
        return cls(**kw)

    def for_image(self, w: int, h: int) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) in the pixels of a `w` x `h` frame."""
        rw = self.width if self.width is not None else w
        rh = self.height if self.height is not None else h
        if self.fx is not None:
            fx = self.fx
            fy = self.fy if self.fy is not None else fx
            cx = self.cx if self.cx is not None else rw / 2.0
            cy = self.cy if self.cy is not None else rh / 2.0
        else:
            fx = fy = (rw / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
            cx, cy = rw / 2.0, rh / 2.0
        sx, sy = w / float(rw), h / float(rh)
        return fx * sx, fy * sy, cx * sx, cy * sy

    def axes(self) -> np.ndarray:
        """(3, 3): columns are the camera's forward, left and up axes in the heading frame."""
        p, r, y = (math.radians(self.pitch_deg), math.radians(self.roll_deg),
                   math.radians(self.yaw_deg))
        cz, sz, cyy, syy, cr, sr = (math.cos(y), math.sin(y), math.cos(p), math.sin(p),
                                    math.cos(r), math.sin(r))
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[cyy, 0.0, syy], [0.0, 1.0, 0.0], [-syy, 0.0, cyy]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return rz @ ry @ rx

    def projector(self, w: int, h: int):
        """A `project`-shaped function (x_forward, y_left) -> (u, v) | None for a w x h frame.

        Same depth cut as `project()` -- None at or behind 5 cm in front of the lens -- and
        the same habit of returning off-frame pixels for the caller to clip."""
        fx, fy, cx, cy = self.for_image(w, h)
        B = self.axes()
        f, l, u = B[:, 0], B[:, 1], B[:, 2]
        cam = np.array([self.cam_x, self.cam_y, 0.0])
        hz = self.cam_h

        def proj(x_forward: float, y_left: float) -> tuple[float, float] | None:
            d = (x_forward - cam[0], y_left - cam[1], -hz)
            depth = f[0] * d[0] + f[1] * d[1] + f[2] * d[2]
            if depth <= 0.05:
                return None
            return (float(cx - fx * (l[0] * d[0] + l[1] * d[1] + l[2] * d[2]) / depth),
                    float(cy - fy * (u[0] * d[0] + u[1] * d[1] + u[2] * d[2]) / depth))
        return proj

    def project(self, x_forward: float, y_left: float, w: int, h: int
                ) -> tuple[float, float] | None:
        """One floor point -> pixel. Build a `projector` once instead for many points."""
        return self.projector(w, h)(x_forward, y_left)

    def visible_m(self, arcs: list[Arc], w: int, h: int, n: int = 301) -> list[float]:
        """Metres of each arc that land inside a w x h frame, sampled every length/(n-1)."""
        proj = self.projector(w, h)
        out = []
        for arc in arcs:
            length = float(np.sum(np.hypot(*np.diff(arc.points, axis=0).T)))
            pts = make_arcs((arc.kappa,), length_m=length, n_points=n)[0].points
            inside = 0
            for x, y in pts:
                p = proj(float(x), float(y))
                if p is not None and 0.0 <= p[0] < w and 0.0 <= p[1] < h:
                    inside += 1
            out.append(round(length * inside / n, 3))
        return out

    def record(self, w: int | None = None, h: int | None = None) -> dict:
        """The camera as a JSON-able dict, with the intrinsics resolved for the drawn frame."""
        rec = {"source": "request", "model": "pinhole", "name": self.name,
               **{k: round(float(getattr(self, k)), 5) for k in MENU_POSE_KEYS},
               "width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
               "cx": self.cx, "cy": self.cy, "hfov_deg": self.hfov_deg,
               "arcs": list(self.arcs) if self.arcs is not None else None,
               "min_visible_m": self.min_visible_m, "prompt": self.prompt,
               "robot": self.robot or None}
        if w is not None and h is not None:
            fx, fy, cx, cy = self.for_image(w, h)
            rec["drawn_wh"] = [int(w), int(h)]
            rec["drawn_intrinsics"] = [round(fx, 4), round(fy, 4), round(cx, 4), round(cy, 4)]
        return rec

    @classmethod
    def from_record(cls, rec: dict) -> MenuCamera:
        """Inverse of `record()` for replay tools (the video's chosen-arc highlight)."""
        keys = set(MENU_CAMERA_KEYS)
        d = {k: v for k, v in rec.items() if k in keys and v is not None}
        if d.get("arcs") is not None:
            d["arcs"] = list(d["arcs"])
        return cls.from_request(d, curvatures=d.get("arcs") or DEFAULT_CURVATURES)


def default_camera_record() -> dict:
    """What a request WITHOUT a camera is drawn with, in the same shape as `MenuCamera.record`.

    Recorded on every such decision so a row says which camera drew it without the reader
    having to know what QVLA_MENU_GEOM meant on the day."""
    return {"source": "server", "model": "level", "geom": MENU_GEOM,
            "hfov_deg": CAM_FOV_DEG, "width": CAM_W, "height": CAM_H,
            "cam_h": round(CAM_HEIGHT_M, 5), "cam_x": round(CAM_FORWARD_M, 5), "cam_y": 0.0,
            "pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0, "arcs": None}


def camera_keep(arcs: list[Arc], camera: MenuCamera, w: int, h: int
                ) -> tuple[list[int], list[float]]:
    """Which of `arcs` a camera's menu offers, and how many metres of EVERY arc are in frame.

    Returns (kept indices into `arcs`, visible metres per arc). The subset (`camera.arcs`)
    goes first; `min_visible_m` then drops arcs this frame cannot show -- a badge clamped to
    the border over 0.2 m of curve is a choice the model cannot see. As with `drivable`,
    never empty: if nothing is visible enough, the most visible arc stays (ties to the
    straightest). Indices, not labels, so the caller keeps the shuffle untouched.
    """
    idx = list(range(len(arcs))) if camera.arcs is None else [
        i for i, a in enumerate(arcs) if any(abs(a.kappa - k) < 1e-9 for k in camera.arcs)]
    visible = camera.visible_m(arcs, w, h)
    if camera.min_visible_m is not None and idx:
        seen = [i for i in idx if visible[i] >= camera.min_visible_m]
        idx = seen or [max(idx, key=lambda i: (visible[i], -abs(arcs[i].kappa)))]
    return idx, visible


def badge_xy(px: list[tuple[float, float]], w: float, h: float,
             r: float) -> tuple[float, float]:
    """Where an arc's number badge goes, given the arc's projected pixels.

    At the far end of the arc, which is where the paths are furthest apart and a number is
    least likely to sit on top of a neighbour's. "Far end" means the last point still
    INSIDE the frame, not the arc's true end: the outer arcs leave the 90 degree view
    before 3 m, and clamping their true end to the border stacks their badges in the corner
    where they read as a pair rather than as two paths.

    Shared with the video tool rather than duplicated there. The video redraws the chosen
    arc over the menu, which lands a thick white stroke straight through that arc's own
    badge and makes the one number a viewer most needs the hardest to read -- so it redraws
    the badge on top afterwards, and it has to land in exactly the same place to do that.
    """
    inside = [p for p in px if r < p[0] < w - r and r < p[1] < h - r]
    ux, uy = inside[-1] if inside else px[-1]
    return min(max(ux, r + 4), w - r - 4), min(max(uy, r + 4), h - r - 4)


# Where the two pivot glyphs sit, as fractions of the frame. ABOVE THE HORIZON on purpose,
# and that is a geometric guarantee rather than a layout preference: `project` puts every
# ground point at v = h/2 + fx*cam_h/x, so with a level camera the whole floor -- and
# therefore every arc and every arc badge -- lives strictly below h/2. Putting the pivots at
# 0.28*h means they can never collide with a path, on any frame, without needing a
# collision test. It also says something true about them: a pivot is not a place on the
# floor you could drive to.
_PIVOT_XY = ((0.115, 0.28), (0.885, 0.28))   # (left glyph, right glyph)
_PIVOT_COLOUR = (255, 255, 255)

# How much dark to put UNDER the white glyph, as extra total stroke width -- so half of it
# shows on each side. This was 6, and 6 is the number that made the turn option disappear.
#
# The glyph is white because it must not read as one of the coloured floor arcs, but the
# pivots sit at 0.28h, which in an indoor scene is ceiling and upper wall, which in these
# scenes is WHITE. On the frame this was caught on, the right glyph over a dark vending
# machine read perfectly and the left glyph over a plain wall collapsed into a faint
# outline of its own halo. Both were drawn identically; only the background differed.
#
# The arithmetic says it had to: 6 px of extra width is 3 px of dark on each side, and the
# server hands the model a 0.31x downsample, so that halo arrives 0.9 px wide and JPEG
# takes what bilinear leaves. 16 gives 8 px a side, ~2.5 px to the model -- thick enough to
# survive both. Judge any change to this on the DOWNSAMPLED image over a white wall, which
# is the only place the failure is visible at all.
_HALO = (20, 20, 20)
_HALO_PAD = 16


def _rotation_glyph(draw, cx: float, cy: float, r: float, clockwise: bool,
                    width: int, colour=_PIVOT_COLOUR) -> None:
    """A ring with a gap at the top and an arrowhead, drawn at (cx, cy).

    White, where every arc is saturated: the glyph has to read as "not one of the coloured
    paths on the floor" at the 0.31x downsample the server actually feeds the model, and
    hue is the only channel wide enough to carry that at ~30 px. The prompt still never
    mentions a colour -- this separates the two KINDS of action, it does not identify one.

    Every white stroke is laid over a dark one so the shape reads on a white ceiling as a
    dark line drawing and on a dark wall as a white one; see `_HALO_PAD`.

    PIL's arc angles start at 3 o'clock and increase clockwise on screen (y grows
    downward), so `arc(300, 240)` is the ring minus a 60 degree gap centred on 12 o'clock,
    and the arrowhead goes on whichever end of that gap the sweep finishes at.
    """
    box = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(box, 300, 240, fill=_HALO, width=width + _HALO_PAD)
    draw.arc(box, 300, 240, fill=colour, width=width)

    theta = math.radians(240.0 if clockwise else 300.0)
    px, py = cx + r * math.cos(theta), cy + r * math.sin(theta)
    # Tangent for INCREASING angle is (-sin, cos); a counter-clockwise glyph finishes by
    # travelling the other way, so its head points along the negation.
    tx, ty = -math.sin(theta), math.cos(theta)
    if not clockwise:
        tx, ty = -tx, -ty
    nx, ny = math.cos(theta), math.sin(theta)          # radial, for the head's base
    # Long and narrow, and the ratio is the whole legibility of the direction cue. At
    # 0.62 x 0.34 the head was almost as wide as it was long, which put its APEX level
    # with one of its base corners: at the 0.31x downsample that reads as a plain triangle
    # with no direction in it at all, and the only thing separating "turn left" from "turn
    # right" in the picture is which way this points.
    a, b = r * 0.85, r * 0.26
    head = [(px + tx * a, py + ty * a),
            (px + nx * b, py + ny * b),
            (px - nx * b, py - ny * b)]
    # The head's halo is a second, larger triangle underneath rather than `outline=`: an
    # outline straddles the edge, so half of any width thick enough to survive the
    # downsample is eaten out of a head that is only 0.52r wide at its base. Pushing each
    # vertex out along its ray from the centroid moves the EDGES out by less than the
    # vertices, which is why the push is 1.6x the pad rather than the pad.
    gx, gy = sum(p[0] for p in head) / 3.0, sum(p[1] for p in head) / 3.0
    grown = []
    for hx, hy in head:
        dx, dy = hx - gx, hy - gy
        d = math.hypot(dx, dy) or 1.0
        grown.append((hx + dx / d * _HALO_PAD * 1.6, hy + dy / d * _HALO_PAD * 1.6))
    draw.polygon(grown, fill=_HALO)
    draw.polygon(head, fill=colour)


def render_menu(image_path: str, out_path: str, arcs: list[Arc], labels: list[int],
                width: int = 13, font_size: int = 96,
                pivot_labels: tuple[int, int] | None = None,
                camera: MenuCamera | None = None,
                colours: list[int] | None = None) -> str:
    """Draw the arcs and their labels onto a copy of the frame. Returns `out_path`.

    Each arc gets a dark outline under its colour: hallway floors in these scenes are
    light grey and a thin bright line on light grey is exactly the sort of thing that
    survives inspection at full size and vanishes once the image is downsampled to the
    200704-pixel budget the server feeds the model.

    The defaults are sized for that budget, not for this file's own preview: 1920x1080 into
    200704 pixels is a 0.31x scale, so a 96 px label arrives as ~30 px and a 13 px stroke as
    ~4 px. Judge any change to them on the downsampled image, never on the full-size one.

    `camera` is the request's own camera (see `MenuCamera`); None draws with `project()`
    exactly as before. `colours` gives each arc's palette index -- callers that draw a
    SUBSET pass the arcs' indices in the full menu so an arc keeps its colour whatever else
    was dropped; None numbers them by position, as before.
    """
    from PIL import Image, ImageDraw, ImageFont

    if len(labels) != len(arcs):
        raise ValueError(f"{len(labels)} labels for {len(arcs)} arcs")
    if colours is not None and len(colours) != len(arcs):
        raise ValueError(f"{len(colours)} colours for {len(arcs)} arcs")
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    # Two separate branches rather than a camera object for the default case too: the
    # no-camera path must stay the literal expression every recorded menu was drawn with,
    # so a replay is byte-identical rather than identical-up-to-rounding.
    cam_proj = camera.projector(w, h) if camera is not None else None

    for i, (arc, label) in enumerate(zip(arcs, labels)):
        if cam_proj is None:
            px = [p for p in (project(x, y, w, h) for x, y in arc.points) if p is not None]
        else:
            px = [p for p in (cam_proj(float(x), float(y)) for x, y in arc.points)
                  if p is not None]
        if len(px) < 2:
            continue
        colour = _PALETTE[(i if colours is None else colours[i]) % len(_PALETTE)]
        draw.line(px, fill=(20, 20, 20), width=width + 6, joint="curve")
        draw.line(px, fill=colour, width=width, joint="curve")

        r = font_size * 0.72
        ux, uy = badge_xy(px, w, h, r)
        draw.ellipse([ux - r, uy - r, ux + r, uy + r], fill=(20, 20, 20), outline=colour,
                     width=6)
        draw.text((ux, uy), str(label), fill=colour, font=font, anchor="mm")

    if pivot_labels is not None:
        # The left glyph turns the robot to its LEFT and sits on the left of the frame.
        # That pairing is not a shuffle-able detail like the numbers are -- a symbol for
        # "turn left" drawn on the right is a lie about the image, and the whole reason
        # the arcs are drawn rather than listed is that the model reads geometry off the
        # picture. The NUMBERS are still shuffled with the arcs' numbers, which is what
        # stops "answer the leftmost tag" from scoring.
        # Sized against the DOWNSAMPLED image, like everything else here: the ring has
        # to clear the number's disc so the two do not merge into one blob at ~30 px, and
        # the arrowhead needs room outside it. Still comfortably above the horizon --
        # `project` puts every floor point below h/2, and the glyph's furthest reach is
        # about 1.3r from a centre at 0.28h.
        # With a request `camera` that guarantee is NOT geometric any more: a camera pitched
        # down puts floor above h/2. For the G1 nav mount (12 deg + 0-5 deg of gait) the 3 m
        # arcs still end well below the glyphs; for the real 47.6 deg mount they leave the
        # top of the frame and the glyphs would sit on them. check_menu_camera.py measures
        # both; keep pivots off for a camera that looks at the floor.
        r = font_size * 1.05
        for (fx, fy), label, cw in zip(_PIVOT_XY, pivot_labels, (False, True)):
            cx, cy = fx * w, fy * h
            _rotation_glyph(draw, cx, cy, r, clockwise=cw, width=width)
            # Sized from the text, not from `r`. This was a flat `r * 0.52` -- a 105 px
            # disc, which fits one digit and not two, and the pivots take the LAST two
            # labels, so with seven arcs they were 8 and 9 and with eleven they are 12
            # and 13. The 0.52 disc did not grow with the menu and the digits spilled
            # over its rim onto the wall behind, where dark-on-dark is unreadable.
            # Floored at the old value so a one-digit menu renders exactly as before.
            tb = draw.textbbox((0, 0), str(label), font=font, anchor="mm")
            dr = max(r * 0.52, max(tb[2] - tb[0], tb[3] - tb[1]) / 2.0 + 14)
            draw.ellipse([cx - dr, cy - dr, cx + dr, cy + dr],
                         fill=_HALO, outline=_PIVOT_COLOUR, width=6)
            draw.text((cx, cy), str(label), fill=_PIVOT_COLOUR, font=font, anchor="mm")

    im.save(out_path, quality=92)
    return out_path


# --------------------------------------------------------------------------------------
# Driving on it
#
# Everything above is the experiment; everything below is what the policy server runs.
#
# The prompts are the SIDED variant of `nav/tools/probe_arc_repair.py` -- the only one of
# four framings that avoided walls at 100% while still holding straight down an open
# corridor 92% of the time, and the only one whose choice negated when the frame was
# mirrored (83%, against 0% for asking in one call). Two calls, both on the frozen model:
# ask what is in the way and which side is open, then hand that answer to the menu.
#
# The measured text lives in the probe files and is deliberately duplicated rather than
# imported: those files are the record of what was actually run, and a shared string would
# let an edit here silently invalidate a result recorded there. If these two drift apart,
# the numbers in CLAUDE.md describe the probe, not the server.
#
# The one addition the probes had no need for is STOP. Every benchmark instruction names a
# destination ("stop at the nearest elevator"), and a menu of seven forward arcs cannot
# express arrival: without it the robot is structurally incapable of finishing an episode
# and every run ends in an overshoot.
# --------------------------------------------------------------------------------------

FREE_SPACE_SYSTEM = (
    f"You are looking through the forward camera of a small wheeled robot. The camera is "
    f"{CAM_HEIGHT_M:.2f} m above the floor, level, with a {CAM_FOV_DEG:.0f} degree "
    f"horizontal field of view. Answer briefly and literally, describing only what is "
    f"visible in this image.")

DESCRIBE_SIDED = (
    "Answer in two short sentences. First: is there a wall, a door, a doorframe or a large "
    "object straight ahead of the robot within about 3 metres? Name it, or say the way "
    "ahead is open. Second: say which direction has the most open, walkable floor the "
    "robot could drive along -- far left, left, straight ahead, right, or far right.")

# Appended to the describe question when QVLA_KERB_PROMPT=1 (server_qwen.py). The question
# above asks for walls, doors and large objects, and a 13 cm kerb is none of those: on the
# contact ladder (2026-10-11) the robot was held at island kerbs for 130-183 s on library and
# ramp_fountain while the describe call, looking at the kerb filling the lower third of the
# frame right after a reverse (library menu_00002430), answered "the way ahead is open" on 6
# of 8 re-decisions. The wheels are 5 cm and the contact layer's step limit is 4.5 cm.
DESCRIBE_KERB = (
    " Count a kerb, a curb, a step, a raised edge or a low wall as a wall: this robot's "
    "wheels are 5 cm tall and it cannot drive over anything higher than about 4 cm, so if "
    "one crosses the way ahead, name it and do not call the way ahead open.")

# --------------------------------------------------------------------------------------
# WHY THE SUPERLATIVE ABOVE IS THE CEILING ON THIS WHOLE POLICY
# --------------------------------------------------------------------------------------
# `DESCRIBE_SIDED` asks for an ARGMAX over open floor, and `STAGE_RULE` below tells the
# select call to follow it. Cross-tabulating the two over 840 decisions from four
# reference ladders shows what that combination actually built:
#
#   describe said     n     chose straight   mean kappa
#   far left         36           11%           +0.533
#   left             38           21%           +0.326
#   straight        668           94%           -0.008
#   right            96            6%           -0.373
#   far right         2            0%           -0.600
#
# The selector is a lookup table on one word. That would be fine if the word tracked the
# task, and it does not: holding the word fixed, the correlation between the bearing error
# to the goal and the chosen curvature is +0.042 for `straight` (n=668), +0.09 and +0.11
# for `left` and `right`. And the word itself does not point at the goal either -- the
# `right` bucket has the goal a mean 6.2 degrees to the LEFT. So the goal-to-action channel
# carries essentially no information, and the closed loop reduces to "follow the corridor
# you are in". Whether an episode succeeds is then a property of the MAP -- does the
# corridor's medial axis pass within 1.5 m of the goal -- rather than of the policy.
#
# That single fact explains the campaign. All thirteen episodes have been solved at least
# once and none is solved reliably; every intervention tried so far (11 arcs instead of 7,
# 15-degree pivots, a second frame) changed the ACTION space and moved nothing, because the
# bottleneck is the OBSERVATION. A better actuator cannot steer to a goal that is not in
# the input.
#
# Two things are wrong with the question and they are separate:
#
#   1. A SUPERLATIVE DESTROYS THE FEASIBLE SET. In a corridor, straight is always more open
#      than the side hallway you are supposed to take, so the correct turn is never named.
#      What the decision needs is which directions are DRIVABLE -- a set -- so that the goal
#      can pick within it. An argmax over safety pre-empts the choice that belongs to the
#      task.
#   2. THERE IS NO GOAL DIRECTION ANYWHERE. `DESCRIBE_TARGET` asks how FAR the target is and
#      never which WAY, so when the target is not visible -- 142 of 243 frames on
#      `warehouse_aisle6` -- free space is the only directional signal in the system.
#
# These two constants address exactly those two things and nothing else, and they are
# selected by `QVLA_GOAL_DIR=1` rather than replacing the originals, because every number in
# `nav/results/hard_study/` was measured under the wording above and an in-place edit would
# leave the reference arm unreproducible -- the same mistake the shared label RNG made.
#
# THEY DO NOT WORK, AND THEY ARE KEPT BECAUSE THE IDEA IS TOO GOOD TO NOT BE RE-TRIED.
# `probe_goal_direction.py` asks both questions of the same pixels, on the exact decisions
# where the reference policy said `straight` while the goal was 40+ degrees off the nose and
# the run went on to fail:
#
#   on 20 such moments               old question   new question
#   named a side at all                 5/20            3/20
#   named the RIGHT side                  --            2/20
#   control (goal really ahead):
#   said straight ahead                18/20           19/20
#
# So HEADING collapses to `straight ahead` on 17 of the 20 moments that needed a turn. The
# control holds, which rules out the cheap explanation -- this is not a variant that has
# merely failed to acquire a turning bias.
#
# The OPEN SET half fails differently and the failure is easy to mistake for a win: it
# names the needed side on 15/20 against the superlative's 2/20, but 10 of those 20 answers
# list ALL FIVE directions, which is recall of 100% at a precision of zero. Scored only on
# the answers that could have been wrong -- a real subset, mean 2.6 of 5 directions -- it is
# 4/7, which is what a set that size scores by construction. The headline is an artifact of
# the word "EVERY" in the question.
#
# WHY, and this is the finding that matters, because it is not about wording at all. The
# describe call is asked which way the task needs to go, and on these frames it genuinely
# cannot know. Across all 844 reference decisions the model reports `TARGET: not visible`
# on 65% of them, and the rate rises monotonically with the size of the turn required:
#
#   bearing error to goal      0-15 deg   15-40   40-90   90-180
#   target not visible            42%      55%     81%      92%
#
# A turn is needed exactly when the goal is behind a wall. So a policy that decides from the
# current frame alone is structurally incapable of these episodes, and no phrasing repairs
# that -- `straight ahead` was the model being honest. What is missing is STATE: which
# clause of the instruction is in progress, whether the doorway it names has been passed
# yet, and how far and how much the robot has turned since it was told. Two frames three
# seconds apart is not that, which is also why the `memory` arm moved nothing.
DESCRIBE_OPEN_SET = (
    "Answer in two short sentences. First: is there a wall, a door, a doorframe or a large "
    "object straight ahead of the robot within about 3 metres? Name it, or say the way "
    "ahead is open. Second: list EVERY direction in which the robot could drive at least a "
    "few metres over open, walkable floor -- any of far left, left, straight ahead, right, "
    "far right -- not only the most open one. A side corridor, an open doorway or a gap "
    "between shelves counts as open even when the way straight ahead is wider.")

# The goal-direction question, and the reason it is safe to ask. Naming where a thing is is
# perception, which this model is good at and which was measured separately: forced choice
# between a blocked and an open frame scored 70/70, and the free text named the obstacle
# with a distance on 6/6 frames. It is emphatically NOT `bearing_to_goal` -- that is scoring
# only and is never fed to the policy. The model is being asked what it can see, exactly as
# it is already asked for the distance.
#
# The TARGET line keeps its wording and its position character for character, because
# `parse_target_distance` anchors on it and losing it is a silent regression in the speed
# channel rather than a failure. The HEADING line is asked for in the SAME sentence so that
# "finish your answer with these two lines" is still the last thing the model reads --
# appending a second request after the TARGET instruction is what would break it.
#
# The `not visible` branch is the half that matters. It is where a corridor-follower is
# indistinguishable from a navigator, and it is the state the failing episodes spend most
# of their decisions in, so the question has to keep producing a direction when the answer
# is a guess from the layout rather than a sighting.
DESCRIBE_TARGET_DIR = (
    " Third: the robot has been told \"{instruction}\". Finish your answer with exactly "
    "these two lines:\n"
    "TARGET: <number> m   (roughly how far away the place or object that instruction names "
    "is, or exactly 'TARGET: not visible' if you cannot see it)\n"
    "HEADING: <far left | left | straight ahead | right | far right>   (which way the robot "
    "must go NEXT to carry out that instruction -- toward the target if you can see it, and "
    "otherwise the way the room or corridor leads toward where it must be)")

# --------------------------------------------------------------------------------------
# GIVING THE DESCRIBE CALL A MEMORY
# --------------------------------------------------------------------------------------
# The menu format has always shown the model ONE frame, and the reason was good: the
# question is "where is the floor open right now", the arcs are drawn on that frame, and
# the other formats only carry four frames because they have to infer their own motion.
#
# What the campaign then showed is a failure that one frame cannot see. The episodes that
# lose do not crash -- they OVERSHOOT and mill. warehouse drove 28.6 m of path to end
# 13.1 m from a goal it started 16.5 m from; hospital_forward_staircase ends 19-24 m out in
# every arm ever run. From a single frame, "the way ahead is open, drive straight" is the
# correct description of a robot that passed its goal thirty seconds ago and is still
# accelerating away from it. There is a text history line, but it lists CHOICES ("straight,
# straight, left"), not consequences -- it can say what the robot did and never that the
# scene stopped getting closer.
#
# Two frames make that visible without asking the model to track anything: same place
# growing larger means progress, same place shrinking means it is behind the robot now.
#
# The older frame goes FIRST and the prompt says which is which, because the arcs are drawn
# on the newest one and a model that describes the wrong image is worse than a model with
# no memory at all. The second sentence of the measured question is untouched -- the memory
# is an extra sentence, not a rewrite of the part that was scored.
FREE_SPACE_SYSTEM_2 = (
    f"You are looking through the forward camera of a small wheeled robot. The camera is "
    f"{CAM_HEIGHT_M:.2f} m above the floor, level, with a {CAM_FOV_DEG:.0f} degree "
    f"horizontal field of view. You are given two frames from that camera: the FIRST is "
    f"where the robot was a few seconds ago, the SECOND is where it is now. Describe the "
    f"SECOND image. Use the first only to judge whether the robot is getting closer to "
    f"things or has driven past them. Answer briefly and literally.")

DESCRIBE_MEMORY = (
    " Also: comparing the two frames, say in a few words whether the robot is approaching "
    "the place the instruction names, has already driven past it, or is not making "
    "progress at all.")

# --------------------------------------------------------------------------------------
# The camera sentence, when a request brings its own camera
# --------------------------------------------------------------------------------------
# Every system prompt above opens by telling the model what it is looking through: "a small
# wheeled robot ... 0.35 m above the floor, level, 90 degree". That is true of camera_nav and
# false of a G1's head camera (1.1 m, tilted ~15 deg down, 69 deg). It is changed ONLY when
# the request's camera asks for it (`prompt: true`) -- the default keeps every model call
# word for word what was measured, and drawing the menu correctly does not depend on it.
# With `prompt` off, or no camera at all, these return the original strings exactly.
_WHEELED = "small wheeled robot"


def _camera_parts(camera: MenuCamera | None) -> tuple[str, str]:
    """(robot noun phrase with its article, the "The camera is ..." sentence)."""
    if camera is None or not camera.prompt:
        return (f"a {_WHEELED}",
                f"The camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a "
                f"{CAM_FOV_DEG:.0f} degree horizontal field of view.")
    robot = camera.robot or _WHEELED
    article = "an" if robot[0].lower() in "aeiou" else "a"
    tilt = int(round(camera.pitch_deg))
    attitude = ("level" if tilt == 0 else
                f"tilted {abs(tilt)} degrees {'down' if tilt > 0 else 'up'}")
    if camera.hfov_deg is not None:
        hfov = camera.hfov_deg
    else:
        ref_w = camera.width if camera.width is not None else (
            2.0 * camera.cx if camera.cx is not None else None)
        hfov = (math.degrees(2.0 * math.atan((ref_w / 2.0) / camera.fx))
                if ref_w is not None else None)
    fov = f", with a {hfov:.0f} degree horizontal field of view" if hfov is not None else ""
    return (f"{article} {robot}",
            f"The camera is {camera.cam_h:.2f} m above the floor, {attitude}{fov}.")


def free_space_system(camera: MenuCamera | None = None, two_frames: bool = False) -> str:
    """FREE_SPACE_SYSTEM / FREE_SPACE_SYSTEM_2, with the request camera's sentence if asked."""
    if camera is None or not camera.prompt:
        return FREE_SPACE_SYSTEM_2 if two_frames else FREE_SPACE_SYSTEM
    robot, sentence = _camera_parts(camera)
    if two_frames:
        return (f"You are looking through the forward camera of {robot}. {sentence} You are "
                f"given two frames from that camera: the FIRST is where the robot was a few "
                f"seconds ago, the SECOND is where it is now. Describe the SECOND image. Use "
                f"the first only to judge whether the robot is getting closer to things or "
                f"has driven past them. Answer briefly and literally.")
    return (f"You are looking through the forward camera of {robot}. {sentence} Answer "
            f"briefly and literally, describing only what is visible in this image.")

# The two sentences above are verbatim what was measured; this is appended, not merged, so
# the validated part of the question is still being asked in the same words.
#
# It exists because the first ladder failed in a way neither the probes nor I predicted.
# Every episode ran the same shape: a clean approach closing 70-90% of the distance, then
# the robot parking at 4-7 m and milling there for the rest of the run, travelling 1.6x to
# 3.0x the straight-line distance and timing out just outside the 1.5 m threshold. Not too
# slow -- a perfect run needed 0.19-0.30 m/s and it moved at 0.51-0.64. Not a missing stop
# either: STOP was chosen on 3-20% of calls.
#
# The cause is in the menu's geometry. The shortest arc is 3 m and the plan runs 7 m, so
# once the target is 3 m away EVERY option drives past it. There is no "go forward one
# metre" to choose, only "drive 7 m" or "stop dead", and the robot oscillates between them.
#
# The fix is a metric channel, and this call is where it is cheapest: it already names
# objects with distances at 6/6 ("a large red double door", "approximately 0.5 to 1 metre
# away"). Asking how far the target is costs no extra call and no extra latency.
#
# Note what this does NOT do. The model still chooses direction only; the menu is unchanged
# because the menu is what measured 100%. Distance comes from this sentence, not from a
# choice, so it is an estimate the model was never scored on -- which is why it only scales
# speed and never stops the robot. Stopping stays the model's own labelled decision.
#
# The trailing TARGET line is a parse anchor, not decoration. The first sentence already
# reports distances -- "a wall approximately 0.5 to 1 metre away" -- so a regex looking for
# metres anywhere in the reply reads the OBSTACLE's distance and calls it the target's,
# which is the one confusion that would make the robot creep toward every wall it passes.
# Asking for the number in a fixed place is cheaper and more reliable than any amount of
# sentence-splitting on our side.
DESCRIBE_TARGET = (
    " Third: the robot has been told \"{instruction}\". Finish your answer with a line of "
    "the form 'TARGET: <number> m' saying roughly how far away the place or object that "
    "instruction names is, or exactly 'TARGET: not visible' if you cannot see it.")

# --------------------------------------------------------------------------------------
# What to ask right after the robot has reversed out of a wedge
# --------------------------------------------------------------------------------------
# `stuck_recovery.py` gets the robot out of a dead end and hands control straight back to
# a policy that has no idea any of it happened. The camera is looking at the same desk it
# was looking at before, only from further away, so the natural answer is the answer that
# wedged it -- and if it answers STOP instead, the robot stands at 1.4 m and the recovery
# will not fire again, because at 1.4 m nothing is close enough to be a wedge. The
# manoeuvre alone turns a permanent stall into a permanent stall one metre back.
#
# So the retreat has to be told to the model, and these two sentences are how. They are
# additions, not rewrites: the measured questions are still asked in the words they were
# measured in, with this in front of them.
#
# This is worth spending prompt on HERE and was not worth spending on TIC-VLA, and the
# difference is measured rather than stylistic. Language moves TIC-VLA's action expert by
# about 9 degrees and its `<answer>` by 0.6; it moves this menu by 103 degrees, and "which
# side is open" is the exact question the SIDED chain already scores 90% on. Telling the
# describe call that straight ahead is the direction already known to be blocked feeds the
# channel with the authority, not the one without it.
#
# Inserted BETWEEN the sided question and the target question, not appended to the end.
# It bears on the first two sentences -- what is ahead, and which side is open -- and
# `DESCRIBE_TARGET` ends with "Finish your answer with a line of the form 'TARGET: <n> m'".
# Put anything after that and the last thing the model reads is no longer the instruction
# about how to finish, which is the one part of this answer that is PARSED. Losing the
# TARGET line does not fail: `parse_target_distance` returns None, the speed ramp falls
# back to cruise, and the robot simply stops slowing down near its goal -- a silent
# regression in the exact channel that stopped it overshooting.
DESCRIBE_AFTER_RECOVERY = (
    " One thing you cannot see in this image: a moment ago the robot could not go on, "
    "because something was directly in front of it, and it has just reversed away from "
    "that thing. It is still facing it. So straight ahead is the one direction already "
    "known to be blocked -- say which side of it the robot can drive along instead.")

# The menu call gets the same fact in its own words. It is not redundant with the sentence
# above: call 2 sees only the description text, so anything call 1 declines to pass on is
# gone by the time a path is actually chosen -- which is the whole reason CHAIN scored 72%
# and SIDED 100% on the same frames.
SELECT_AFTER_RECOVERY = (
    "The robot has just reversed away from something that was blocking it, and it is "
    "still facing that thing. Do not choose a path that carries straight on into it. "
    "Choose a path that leads around it, toward whichever side the description above "
    "says is open.")

# --------------------------------------------------------------------------------------
# ...and after a BALK, which is a different situation and needs different words
# --------------------------------------------------------------------------------------
# A wedge is "something is in my way". A balk is "I stopped, and I should not have". The
# robot was standing still with clear floor in front of it, which on the first full ladder
# was the single largest failure: 42 of 65 decisions on `office_nearest_elevator` answered
# STOP from 5.5 m out, with the model's own description of the same frames naming an open
# side AND saying the target was not visible.
#
# So do not tell it to go round an obstacle -- there is no obstacle, and saying there is
# would be false. Tell it the two things that are true and that the picture cannot show:
# it has just backed up to get a wider view, and standing still is not an option because
# it has not arrived. The second half is a fact about the harness, not encouragement:
# arrival ends the episode, so a robot still being asked has not arrived.
DESCRIBE_AFTER_BALK = (
    " One thing you cannot see in this image: the robot had stopped moving for several "
    "seconds even though the floor ahead of it was clear, and it has just backed up a "
    "little to widen its view. It has NOT reached its destination. Look carefully for the "
    "place or object the instruction names -- it may be off to one side, far away, or "
    "partly hidden -- and say where it is.")

SELECT_AFTER_BALK = (
    "The robot stopped a moment ago without having arrived anywhere, and has backed up "
    "slightly to see more. Standing still is what already failed, so keep moving: pick "
    "the path that makes the most progress toward what the instruction asks for, using "
    "the description above to decide which way that is.")

# --------------------------------------------------------------------------------------
# The history line
# --------------------------------------------------------------------------------------
# One frame of a stationary robot and one frame of a moving robot are the same picture, so
# without this the model re-answers each decision as if it were the first. That is exactly
# how a run ends up choosing "straight" eleven times against a wall: nothing in the input
# says the last ten "straight"s achieved nothing.
#
# It goes to the SELECT call and not to the describe call, and the split is deliberate.
# Call 1's job is to report what is in the image, and its wording is the wording the 100% /
# 92% / 90% / 83% numbers were measured under; a paragraph about the past would change the
# question being asked there. What the robot already tried bears on WHICH PATH to take, not
# on what the floor looks like, and call 2 is the call we are free to compose.
def history_note(recent: list[str], stalled_s: float) -> str:
    """One sentence of what was recently chosen and whether it worked. "" when new.

    `recent` is oldest-first direction words. The stall reading comes from the runner,
    which is the only place that knows where the robot actually is -- the server sees the
    choices it made and has no way to know whether any of them moved anything.
    """
    if not recent and stalled_s < 1.0:
        return ""
    parts = []
    if recent:
        parts.append("Recently you chose, oldest first: " + ", ".join(recent) + ".")
    if stalled_s >= 1.0:
        parts.append(f"The robot has not moved for about {stalled_s:.0f} seconds, so "
                     f"whatever it has been choosing is not working.")
    return " ".join(parts)


# THE ODOMETRY NOTE AND THE SPIN GUARD (QVLA_ODOMETRY=1; off by default, so every number
# recorded before 2026-11-26 is unchanged).
#
# What it fixes, measured: hospital tour step 3, "Turn left and drive to the foot of the
# staircase." -- 52 of 52 decisions hard left, 1001 degrees turned in 73 s, three circles
# over the same patch of floor. `history_note` showed the model "hard left, hard left, ..."
# and nothing in that line says whether the robot has turned 20 degrees or 700, so under
# `STAGE_RULE` ("do only the earliest move not yet done") the turn could never be done.
#
# Two layers, for the reason this repo keeps re-learning (STOP-means-blocked, the kerb
# sentence): a rule in the prompt is honoured most of the time and a structural gate all of
# the time. The sentence gives the model the number a real robot has from its wheels and
# says when a turn counts as finished; the guard is the backstop for when it does not hold
# -- once the robot has swept a full circle one way within `window` seconds, the sharp arcs
# that way are taken off the menu, exactly as the lidar filter takes off arcs into a wall,
# without renumbering anything (`drivable` has why).
TURN_DONE_DEG = 75.0     # past this, a commanded turn that way is called done
SPIN_DEG = 270.0         # swept one way within the window -> the guard fires
SPIN_MIN_KAPPA = 0.30    # the guard drops arcs at least this sharp toward the spin


def odometry_note(odo: dict | None, target_m: float | None = None,
                  last_seen: dict | None = None) -> tuple[str, int]:
    """(sentence for the select prompt, spin side: +1 left, -1 right, 0 none).

    `odo` is `TurnOdometer.payload()` as the runner sent it; None or empty -> ("", 0), so a
    caller that sends nothing gets exactly the old prompt. `target_m` is the describe call's
    TARGET distance (None = not visible). While a turn is not yet done and the target is in
    view, the note says a turn word means "turn toward it", not "take the sharpest arc":
    the same tour step opened with the staircase 2 degrees off the nose, 12 m away, the
    description calling the floor straight ahead open -- and answered hard left, because
    the instruction said "left".

    `last_seen` is the server's memory of the last decision whose description saw the
    target: {"turned_deg", "since_s", "target_m"} from the odometry of THAT decision. When
    the target is out of view now, the note says how far the robot has turned since, i.e.
    which way to look for it. On the same step the staircase left the frame at 25 degrees
    of turn and the robot kept turning left to 1001: from 93 degrees on it was behind the
    robot's right shoulder, and nothing in the prompt said so.
    """
    if not odo:
        return "", 0
    turned = float(odo.get("turned_deg", 0.0))
    net = float(odo.get("net_m", 0.0))
    since = float(odo.get("since_s", 0.0))
    recent = float(odo.get("recent_turn_deg", 0.0))
    recent_s = float(odo.get("recent_s", 0.0))
    side = "left" if turned > 0 else "right"
    parts = []
    if abs(turned) < 20.0 and since < 3.0:
        pass                                           # nothing has happened yet
    elif abs(turned) < 20.0:
        parts.append(f"Since this instruction was given {since:.0f} s ago the robot has kept "
                     f"roughly the same heading and is {net:.1f} m from where it started.")
    else:
        parts.append(f"Since this instruction was given {since:.0f} s ago the robot has "
                     f"turned {abs(turned):.0f} degrees to its {side} in total and is "
                     f"{net:.1f} m from where it started.")
    if target_m is not None and abs(turned) < TURN_DONE_DEG:
        parts.append(f"The place the instruction names is in view, about {target_m:.0f} m "
                     f"away. A turn the instruction asks for only has to bring the robot "
                     f"onto a path that leads there: if a drawn path already leads toward "
                     f"it on open floor, take that path rather than the sharpest turn.")
    spin = (1 if recent > 0 else -1) if abs(recent) >= SPIN_DEG else 0
    if target_m is None and last_seen:
        # + = it was to the LEFT. Wrapped: after 276 degrees of left turn a target seen at
        # 25 is 109 degrees to the LEFT, not 251 to the right -- the short way round.
        delta = (float(last_seen.get("turned_deg", 0.0)) - turned + 180.0) % 360.0 - 180.0
        ago = since - float(last_seen.get("since_s", since))
        # Not when it points the way the robot is circling: the guard below has just said
        # that way only repeats the circle, and two sentences pulling opposite ways is a
        # prompt arguing with itself.
        if abs(delta) >= 30.0 and not (spin and (delta > 0) == (spin > 0)):
            w = "left" if delta > 0 else "right"
            parts.append(f"The place the instruction names was last in view {ago:.0f} s ago, "
                         f"about {float(last_seen.get('target_m', 0)):.0f} m away, when the "
                         f"robot was facing about {abs(delta):.0f} degrees further to its {w} "
                         f"than now: it is that way, so head back toward the {w} rather than "
                         f"away from it.")
    if spin:
        s = "left" if spin > 0 else "right"
        parts.append(f"In the last {recent_s:.0f} s it has turned {abs(recent):.0f} degrees "
                     f"to its {s} -- a full circle -- without getting anywhere, so turning "
                     f"{s} again would only repeat the circle. The sharp {s} paths are off "
                     f"the menu for now: take the path toward the open floor or the target "
                     f"the description names, usually the straight one.")
    elif abs(turned) >= TURN_DONE_DEG:
        parts.append(f"A turn of about 90 degrees is a complete turn, so any turn to the "
                     f"{side} the instruction asked for is already done (unless it asked to "
                     f"turn around): do the NEXT part of the instruction now, and do not "
                     f"keep turning {side} just because the instruction says so.")
    return " ".join(parts), spin


def spin_filter(arcs: list[Arc], labels: list[int], spin: int,
                min_kappa: float = SPIN_MIN_KAPPA
                ) -> tuple[list[Arc], list[int], list[int]]:
    """Drop arcs turning toward `spin` at |kappa| >= min_kappa. (arcs, labels, dropped).

    Same contract as `drivable`: labels are never renumbered and the menu is never left
    empty -- with nothing else left, the straightest remaining arc stays.
    """
    if not spin:
        return list(arcs), list(labels), []
    keep = [i for i, a in enumerate(arcs) if not (a.kappa * spin >= min_kappa)]
    if not keep:
        keep = [min(range(len(arcs)), key=lambda i: abs(arcs[i].kappa))]
    kept = set(keep)
    return ([arcs[i] for i in keep], [labels[i] for i in keep],
            [labels[i] for i in range(len(arcs)) if i not in kept])


def direction_word(kappa: float) -> str:
    """A curvature as words, for the history line. Negative kappa turns right."""
    for edge, word in ((-0.5, "hard right"), (-0.25, "right"), (-0.05, "slight right"),
                       (0.05, "straight"), (0.25, "slight left"), (0.5, "left")):
        if kappa <= edge:
            return word
    return "hard left"


def stop_label(n_arcs: int, pivots: bool) -> int:
    """The number STOP carries on a menu of `n_arcs` arcs, with or without the turns.

    Derived, never written down twice. STOP sits after everything selectable, so it moves
    when the menu grows, and an off-by-one here does not raise -- it silently turns "stop"
    into "turn right", on a run that otherwise looks entirely normal.
    """
    return n_arcs + (2 if pivots else 0) + 1


def allocate_labels(rng, n_arcs: int, pivots: bool
                    ) -> tuple[list[int], tuple[int, int] | None]:
    """One shuffle across every selectable label: (arc labels, (left, right) or None).

    Arcs and pivots are numbered from the SAME permutation. A pivot that is always
    numbered 8 is answerable without looking at the image, which is precisely the confound
    the arc shuffle exists to remove -- and it would be the easier one to fall for, since
    there are only two of them.

    With `pivots` false the permutation is over `n_arcs` alone, so the random stream, and
    therefore every label in every earlier run, is bit-for-bit unchanged.

    The two pivot numbers come back as (left glyph, right glyph) and `render_menu` reads
    them in that order. The SIDE is fixed while the numbers shuffle: a "turn left" symbol
    drawn on the right of the frame would be a lie about the image, and geometry read off
    the picture is the whole reason this format works.
    """
    perm = (rng.permutation(n_arcs + (2 if pivots else 0)) + 1).tolist()
    return perm[:n_arcs], (tuple(perm[n_arcs:]) if pivots else None)


def pivot_word(pivot_rad: float) -> str:
    """A pivot as words, for the history line.

    Deliberately not folded into `direction_word`'s scale. "hard left" and "turned left in
    place" are different events and the history exists to let the model notice a pattern in
    what it has been doing -- three pivots in a row is the signature of a robot spinning
    where it stands, and it is unreadable if those entries say "hard left" like a drive.
    """
    return ("turned left in place" if pivot_rad > 0 else "turned right in place")


CRUISE_MPS = 0.7      # what the trained policy asks for on these episodes
CREEP_MPS = 0.2       # slow enough that one replan period advances well under a metre
SLOW_FROM_M = 4.0     # where the approach starts easing off

# Below this, the model is asked to pick its own approach speed instead of being handed
# the ramp's. 3.0 m and not 4.0 (where `SLOW_FROM_M` starts easing) because the two answer
# different questions: the ramp is a safety net that must cover every case, and this is a
# discretionary channel that should only open where the model has something to see. At
# 4 m the target is often a smear at the end of a corridor and "how fast should I close
# the last bit" is not yet a question about anything.
SPEED_CHOICE_FROM_M = 3.0
# The scale the model writes on. Ten steps, not a float in m/s: the metric channel is the
# one this model cannot ground -- that is the whole finding behind the menu format -- so
# asking for 0.35 would be asking for the number it has already been measured unable to
# produce. A fraction of a speed IT DOES NOT HAVE TO KNOW is a choice, not an estimate.
SPEED_LEVELS = 10

# "3", "3.5 m", "0.5 to 1 metre", "2-3 meters". The optional second number is a range, and
# the pair is kept so the caller can decide which end to believe.
_TARGET_RE = re.compile(
    r"TARGET\s*[:\-]?\s*(?:(not\s*visible|none|unknown)"
    r"|(\d+(?:\.\d+)?)\s*(?:(?:to|-|–|and)\s*(\d+(?:\.\d+)?)\s*)?"
    r"(?:m\b|met(?:re|er)s?\b)?)", re.IGNORECASE)


def parse_target_distance(reply: str) -> float | None:
    """Metres to the instruction's target, or None for 'cannot see it' / no answer.

    None means "no estimate", and every caller must treat it as cruise, not as zero. The
    model is not scored on this number -- there is no ground truth for it in the benchmark
    -- so the failure that has to be cheap is the missing answer, not the wrong one.

    A range collapses to its NEAR end. Slowing a metre early costs a second; braking a
    metre late is the overshoot this whole channel exists to remove.
    """
    m = _TARGET_RE.search(reply)
    if m is None or m.group(1):
        return None
    lo = float(m.group(2))
    return min(lo, float(m.group(3))) if m.group(3) else lo


# Anchored on the HEADING label rather than searched for anywhere, and the reason is the
# same one that gave the TARGET line its own label: by the time this runs, the reply already
# contains the words "left" and "right" in the free-space sentence, so a bare direction
# search reads the OPEN SET and calls it the goal. The two channels exist precisely to
# disagree -- open floor is not where the goal is, and if it were, none of this would be
# needed.
#
# Longest-first alternation, because Python's `|` is first-match, not longest-match: with
# "left" before "far left" every "far left" parses as "left" and the widest turn on the
# menu becomes unreachable.
_HEADING_RE = re.compile(
    r"HEADING\s*[:\-]?\s*\**\s*"
    r"(far\s*left|far\s*right|hard\s*left|hard\s*right|left|right|"
    r"straight\s*ahead|straight|ahead|forward)", re.IGNORECASE)

_HEADING_WORDS = {"far left": "far left", "hard left": "far left",
                  "far right": "far right", "hard right": "far right",
                  "left": "left", "right": "right",
                  "straight ahead": "straight ahead", "straight": "straight ahead",
                  "ahead": "straight ahead", "forward": "straight ahead"}


def parse_heading_word(reply: str) -> str | None:
    """The direction the describe call says the task needs next, or None if it did not say.

    None is a real answer and must stay cheap. It means the select call runs on free space
    alone -- exactly the reference policy -- rather than on a direction invented to fill the
    field. A fabricated heading is worse than no heading: it steers with full authority
    (the cross-tab above shows one word moving mean kappa across the whole menu span) on no
    evidence at all.
    """
    m = _HEADING_RE.search(reply)
    return _HEADING_WORDS[" ".join(m.group(1).lower().split())] if m else None


def speed_for(target_m: float | None, cruise: float = CRUISE_MPS) -> float:
    """Plan speed from the model's estimate of how far the target is.

    Linear rather than a threshold, and floored rather than zeroed. A cliff would turn one
    bad estimate into a stall, and a zero would duplicate the STOP label with a number the
    model was never scored on. Ramping means a wrong estimate costs a slower approach,
    which the next call undoes 3 s later, and the robot converges on the target instead of
    flying past it and circling back.

    `cruise` is a parameter and not a second reading of the module constant because the
    server already owns that number as `QVLA_MENU_SPEED`. Two constants holding one value
    is the failure mode this repo has hit three times -- each copy stayed plausible, each
    produced a wrong number instead of an error.
    """
    if target_m is None:
        return cruise
    return float(min(cruise, max(CREEP_MPS, target_m / SLOW_FROM_M * cruise)))


def parse_choice_speed(reply: str) -> tuple[int | None, int | None]:
    """(path label, speed level 0-10 or None) from a reply that may hold one or two ints.

    Lenient in one direction only. The FIRST integer is always the path, because that is
    the answer that was measured and the one the robot cannot do without; a missing second
    integer returns None and the caller falls back to the ramp. The reverse leniency --
    guessing which of two numbers is which -- is what would turn "7" into speed 7 at
    cruise 0 and park the robot, so it is not offered.

    A second number outside 0..10 is dropped rather than clamped. In range it is a choice;
    out of range it is the model having written something else entirely -- a distance, a
    label it changed its mind about -- and clamping would silently convert that into a
    speed command.
    """
    nums = re.findall(r"\d+", reply)
    if not nums:
        return None, None
    choice = int(nums[0])
    if len(nums) < 2:
        return choice, None
    k = int(nums[1])
    return choice, (k if 0 <= k <= SPEED_LEVELS else None)


def speed_from_level(level: int, cruise: float = CRUISE_MPS) -> float:
    """A 0-10 answer as m/s, floored at the creep speed.

    Level 0 does NOT mean stop, and the floor is the whole reason this is a function. STOP
    is a menu label -- a decision the model is scored on, that ends the episode's driving
    and that the recovery machinery can reason about. A speed of zero would be a SECOND way
    to express the same thing, reachable by an off-by-one or a stray token, indistinguish-
    able afterwards from a chosen stop, and invisible to `stuck_recovery` (which sees a
    robot standing still with clear floor ahead and would spend a reverse manoeuvre undoing
    a command the model meant). This repo has paid for a collapsed-to-zero speed channel
    three times; keeping the two channels disjoint costs one `max()`.
    """
    return float(max(CREEP_MPS, min(cruise, cruise * level / SPEED_LEVELS)))


# A benchmark instruction is usually a SEQUENCE, and this selector had no notion of which
# part of it it was on. `hospital_exit_room` says "Exit the room AND TURN RIGHT to enter
# the hallway. Continue straight ahead...", and on its first two decisions the model's own
# describe call said "the way ahead is open, with a doorway visible in the distance, the
# most open walkable floor is straight ahead, TARGET: 10 m" -- and the select call answered
# a right-hand arc anyway, kappa -0.35 then -0.60. Those were the ONLY two decisions in
# that 160-decision episode where describe and select disagreed. By decision three the
# robot faced the room's cabinets, the target was never visible again, and it spent the
# episode grinding along furniture: 257 guard interventions, 40 m of path, 6.66 m closest.
# The turn belonged after the doorway and was spent before it.
#
# The prompt is the right channel for this and that is measured, not assumed: an
# instruction moves THIS plan 103 degrees, against ~9 for TIC-VLA's action expert.
#
# Wording chosen by `probe_instruction_stage.py`, holding the frame and the description the
# model itself produced on it and varying only the question. Three candidates all recovered
# exactly what deleting the direction word recovers (mean kappa -0.350 -> -0.075) while
# still answering HARD RIGHT, 6 of 6, when the description was flipped to say the open
# floor was on the right -- so this is not "always go straight". The longer wording was
# rejected on the control frames, where it replied "Based on the..." instead of a digit on
# 3 of 9 calls; in the live loop that is a decision with no plan behind it. Two sentences
# is the version that fixes the failure and keeps the answer contract.
STAGE_RULE = ("The instruction may list several moves in order; do only the earliest one "
              "not yet done. Where it names a direction the description does not call "
              "open, follow the description.")

# THE SECOND SENTENCE OF `STAGE_RULE` IS WHAT MAKES THIS A CORRIDOR-FOLLOWER, and it is
# worth being precise about, because it was added for a real failure and it did fix that
# failure. `hospital_exit_room` turned right before the doorway instead of after it, so the
# rule was written to stop a commanded direction being spent early. It does that by making
# free space OUTRANK the instruction whenever the two disagree -- and since
# `DESCRIBE_SIDED` reports a superlative, and a corridor is always more open than the side
# hallway leading off it, the two disagree at exactly the junctions where the turn is
# needed. The measured consequence is in the cross-tab beside `DESCRIBE_OPEN_SET`: 668 of
# 840 decisions ran under "straight", and 94% of those went straight.
#
# The right fix for turning early is a STAGE condition -- turn once you are through the
# doorway -- not a blanket override, and the first sentence already says that. So this
# version keeps sentence one verbatim and replaces sentence two with a lexicographic rule:
# feasibility filters, the task orders. That is the standard shape for a constrained
# choice, and it is what the original wording collapsed into a single argmax.
#
# On length: `probe_instruction_stage.py` rejected a longer candidate because the model
# answered "Based on the..." instead of a digit on 3 of 9 calls. That constraint is gone --
# `SELECT_PREFILL` now ends the prompt mid-sentence, so the next token cannot be "Based",
# and the prose rate it was protecting against went from 5.9% of the campaign to a
# structural zero. Three sentences are affordable now; nine would still not be.
STAGE_RULE_GOAL = (
    "The instruction may list several moves in order; do only the earliest one not yet "
    "done. The description ends with a HEADING, which is the direction the task needs to "
    "go next, and it separately lists which directions are open: choose the drawn path "
    "that goes closest to the HEADING among the paths that stay on open floor. Only when "
    "no open path leads that way should you go elsewhere, and then pick the one that keeps "
    "the HEADING reachable rather than the one with the most room.")

# THE ANSWER IS FORCED, not requested. `select_system` already ends with "Answer with the
# number and nothing else", and 392 of 6673 decisions across the campaign ignored it and
# wrote prose instead -- "Based on the navigation instruction to", "Looking at the image
# and the" -- truncated mid-sentence by the 8-token answer budget, naming no label at all.
# `menu_plan` then returns None and the robot reuses the plan it already had.
#
# The rate is what makes this structural rather than cosmetic. It is 0.3% on a clean
# prompt and 9.2% once the robot is stalled, 20-22% on the decision right after a recovery
# manoeuvre: the prompts that carry an extra paragraph are the prompts the model
# summarises instead of answering, so the decision channel goes missing exactly in the
# situations the extra paragraph was added to handle. And it latches -- no answer means
# reusing the plan that was already not working, which keeps the robot stalled, which
# keeps the note in the prompt.
#
# Ending the prompt mid-sentence makes the point structurally: after "The number is " the
# next token cannot be "Based"; it has to be a digit. This is the same device the waypoint
# formats have always used (`ANSWER_PREFIX = "<answer>("`) and `think_then_answer` already
# takes a `prefill` argument for it -- the menu's select call was simply the one caller
# that never passed one.
#
# NO DIGIT MAY APPEAR IN EITHER STRING. The prefill is prepended to the reply the parser
# sees, and `parse_choice_speed` takes the FIRST integer as the path label, so a digit
# here would become the robot's steering choice.
SELECT_PREFILL = "The number is "
SELECT_PREFILL_SPEED = "The two numbers are "


def select_system(stop_label: int, stop_allowed: bool = True,
                  speed_choice: bool = False,
                  pivot_labels: tuple[int, int] | None = None,
                  goal_dir: bool = False,
                  camera: MenuCamera | None = None) -> str:
    """The arc-selection system prompt, with `stop_label` reserved for stopping.

    `camera` changes only the opening camera sentence, and only when its `prompt` flag is
    set (see `_camera_parts`); otherwise the text is exactly what it always was.

    `goal_dir=True` swaps `STAGE_RULE` for `STAGE_RULE_GOAL` and rewrites the path-judging
    sentence, and the two go together: the rule tells the model to rank open paths by the
    HEADING, and leaving "prefer the path with the most open floor" standing underneath it
    would be a prompt arguing with itself over the exact question at issue. It is off by
    default so that every arm already on disk keeps running under the words it was measured
    under.

    `stop_allowed=False` takes that choice away for one decision, and it is used on the
    calls immediately after a recovery manoeuvre -- either kind, a wedge or a balk. The
    justification is that BOTH reasons this prompt gives for answering `stop_label` are
    known to be false at that instant. The robot cannot have arrived -- the control loop in
    `run_navigation.py` breaks the moment it is inside the success threshold, so a robot
    still being driven has not got there. And "every drawn path runs into something" was
    the situation the reverse just undid; there is a metre more room now than there was
    when that was last true. After a balk the second reason is even weaker: the front was
    measured CLEAR, which is what made it a balk rather than a wedge.

    `speed_choice=True` asks for a second number, the approach speed, and is set only when
    the describe call has just reported the target within `SPEED_CHOICE_FROM_M`. It is an
    independent flag rather than a third mode because it crosses the other one: a robot can
    balk two metres from its goal, and that decision needs both the no-stop rule and the
    speed question.

    Note this removes an ANSWER, not a capability: the next decision, one generation
    later, has the full menu back. A robot that genuinely should stop will stop then, a
    couple of metres of driving later, which is the price of not having it stand in front
    of a desk for the rest of the episode.
    """
    # THE STOP RULE CHANGES WHEN PIVOTS EXIST, and this is the point of adding them.
    # Without a pivot, "every drawn path runs into something" is a true statement with
    # exactly one legal answer, so STOP has to accept it -- and the first ladder duly
    # answered STOP on 42 of 65 decisions from 5.5 m out while its own describe call was
    # naming an open side on those same frames. With a pivot on the menu that statement no
    # longer implies stopping: the robot can face the open side instead. So the second
    # reason is not softened here, it is DELETED and redirected. Leaving both in would keep
    # the cheaper answer available for the situation the expensive one was added for.
    if pivot_labels is None:
        stop_rule = f"""There is one extra choice, {stop_label}, which is not drawn on the \
image. Answer {stop_label} to stop and stay still, and only for one of two reasons: the \
robot has arrived at the place the instruction names, or every drawn path runs into \
something. Stopping at the right place is part of the task. Do not answer {stop_label} \
merely because the way ahead is tight."""
    else:
        stop_rule = f"""There is one more choice, {stop_label}, which is not drawn on the \
image. Answer {stop_label} ONLY when the robot has arrived at the place the instruction \
names. Stopping at the right place is part of the task. Blocked is not arrived: if the way \
ahead is tight or every drawn path runs into something, turn in place instead of \
answering {stop_label}."""
    if not stop_allowed:
        stop_rule = """Stopping is not one of your choices here. The robot has not arrived \
anywhere, and it has just reversed a little after failing to make any progress, so \
standing still is the one thing already known not to work. Pick the drawn path with the \
most open floor along it, even if none of them is good."""

    # The pivot paragraph. Written to answer WHEN, not just what: an action the model has
    # to infer the use of is an action it will not reach for, and the failure being fixed
    # is precisely one of not reaching for a way out. The last sentence is the brake --
    # turning is free of collision risk and therefore always the safe answer, and an
    # always-safe answer with no cost attached is how a robot spends an episode pirouetting
    # instead of arriving.
    pivot_rule = ""
    if pivot_labels is not None:
        left, right = pivot_labels
        pivot_rule = f"""Two more choices are drawn as white circular arrows in the top \
corners: {left} on the left and {right} on the right. They are not paths. They turn the \
robot {PIVOT_DEG:.0f} degrees in place -- {left} to its left, {right} to its right -- \
without moving it anywhere, and then you get to look again from the new direction.

Turn in place when the robot needs to be facing somewhere else before any drawn path is \
worth taking: the description says the way ahead is blocked and names an open side, or the \
instruction says to turn and no drawn path curves far enough to do it. Every drawn path \
carries the robot three metres forward, so when something is close in front, all of them \
run into it and turning is the only way out. Turning costs no ground and cannot hit \
anything. But do not turn when a drawn path already leads onto open floor in the direction \
you want -- a turn spends a decision without covering any distance, and repeating it \
leaves the robot spinning where it stands."""

    # The speed question is asked ONLY near the target, and the prompt says so, because
    # "how fast" is a different question at 12 m and at 2 m. Far away it has one sensible
    # answer -- go -- and asking it anyway spends tokens and adds a number that can go
    # wrong for no gain. Close in, it is the actual difficulty: the arcs are all 3 m long,
    # so with the target 2 m off the choice is not WHERE to drive but HOW MUCH of the
    # chosen path to use before looking again.
    # "the number you chose" rather than "of the chosen path" once pivots are on the menu:
    # the pivot paragraph has just finished saying "they are not paths", and an answer
    # contract that then asks for a path is an invitation to rule the turns out.
    what = "you chose" if pivot_labels is not None else "of the chosen path"
    answer_rule = (f"""Answer with TWO whole numbers separated by a space: first the \
number {what}, then how fast to drive it, from 0 (barely creeping) to \
{SPEED_LEVELS} (full speed). You are close to what the instruction names, so choose the \
speed deliberately: slow down to place the robot accurately, keep the speed up if there is \
still ground to cover. A low number never means stopping -- it means moving gently. \
Write nothing but the two numbers.""" if speed_choice else
                   f"Answer with the number {what} and nothing else.")
    robot, cam_sentence = _camera_parts(camera)
    return f"""You are the navigation system of {robot}, looking through its \
forward camera. {cam_sentence}

Several candidate paths have been drawn onto the floor of the image. Each one is a real \
route the robot can drive, starting at the robot and curving away from it, and each ends \
in a numbered circle. The numbers are arbitrary tags, not an order: they do not run left \
to right and carry no meaning beyond identifying a path.

Your job is to choose the one option that best carries out the navigation instruction. \
{STAGE_RULE_GOAL if goal_dir else STAGE_RULE} \
Judge each path by where it actually goes on the floor in the image. {
"A path is usable when it stays on open, walkable floor and does not run into a wall, a \
door frame or an object; among the usable ones, the one that goes where the task needs to \
go wins, and going straight because it is roomier is how a robot drives past its turning."
if goal_dir else
"Prefer a path that stays on open, walkable floor and does not run into a wall, a door \
frame or an object."}
{f"{chr(10)}{pivot_rule}{chr(10)}" if pivot_rule else ""}
{stop_rule}

{answer_rule}"""


# --------------------------------------------------------------------------------------
# The TEXT-ONLY ablation of the menu (QVLA_SELECTOR=text)
# --------------------------------------------------------------------------------------
# What it tests: is it the DRAWING that makes the menu work, or merely the fact that the
# model chooses among a few fixed curvatures? Everything else is held fixed -- the describe
# call, the same seven curvatures, the same shuffled numbers per decision (same random
# stream), STOP, the speed question, the history and recovery notes, the plan builder and
# the controller. The select call still sees the camera image, but as it came off the
# camera, with nothing drawn on it; the paths are given as words instead. The words carry
# the same geometry the drawing does (direction, how far it turns, where it ends), so the
# one thing taken away is where each path lies relative to what is in the picture.
def option_text(kappa: float, arc_len_m: float = DEFAULT_LENGTH_M) -> str:
    """One menu arc as words: direction, total turn and end point over its length."""
    if abs(kappa) < 1e-9:
        return f"straight -- goes straight ahead for {arc_len_m:.0f} metres"
    side = "left" if kappa > 0 else "right"
    k = abs(kappa)
    x = math.sin(k * arc_len_m) / k
    y = (1.0 - math.cos(k * arc_len_m)) / k
    return (f"{direction_word(kappa)} -- curves to the {side}, turning "
            f"{math.degrees(k * arc_len_m):.0f} degrees over {arc_len_m:.0f} metres and "
            f"ending {x:.1f} m ahead and {y:.1f} m to the {side}")


def text_options(kappas: list[float], labels: list[int]) -> str:
    """The menu as a list, in label order. The labels are shuffled per decision, so label
    order is a random order of directions -- the text counterpart of numbers that 'do not
    run left to right' on the drawn menu."""
    return "\n".join(f"{lab}: {option_text(k)}"
                     for lab, k in sorted(zip(labels, kappas), key=lambda r: r[0]))


def select_system_text(stop_label: int, options: str, stop_allowed: bool = True,
                       speed_choice: bool = False) -> str:
    """`select_system` for the text-only ablation: same rules, same answer contract, the
    paths listed in `options` instead of drawn. Only the no-pivot, no-heading, default-
    camera case exists, which is the shipping profile; the server refuses the others."""
    if stop_allowed:
        stop_rule = f"""There is one extra choice, {stop_label}, which is not a path. \
Answer {stop_label} to stop and stay still, and only for one of two reasons: the robot has \
arrived at the place the instruction names, or every path runs into something. Stopping at \
the right place is part of the task. Do not answer {stop_label} merely because the way \
ahead is tight."""
    else:
        stop_rule = """Stopping is not one of your choices here. The robot has not arrived \
anywhere, and it has just reversed a little after failing to make any progress, so \
standing still is the one thing already known not to work. Pick the path with the most \
open floor along it, even if none of them is good."""
    answer_rule = (f"""Answer with TWO whole numbers separated by a space: first the \
number of the chosen path, then how fast to drive it, from 0 (barely creeping) to \
{SPEED_LEVELS} (full speed). You are close to what the instruction names, so choose the \
speed deliberately: slow down to place the robot accurately, keep the speed up if there is \
still ground to cover. A low number never means stopping -- it means moving gently. \
Write nothing but the two numbers.""" if speed_choice else
                   "Answer with the number of the chosen path and nothing else.")
    robot, cam_sentence = _camera_parts(None)
    return f"""You are the navigation system of {robot}, looking through its \
forward camera. {cam_sentence}

The robot can drive one of the candidate paths listed below. Each one is a real route the \
robot can drive, starting at the robot and curving away from it over the floor you see in \
the image; nothing is drawn on the image. The numbers are arbitrary tags, not an order: \
they do not run left to right and carry no meaning beyond identifying a path.

{options}

Your job is to choose the one option that best carries out the navigation instruction. \
{STAGE_RULE} Judge each path by where it would take the robot on the floor in the image. \
Prefer a path that stays on open, walkable floor and does not run into a wall, a door \
frame or an object.

{stop_rule}

{answer_rule}"""


def plan_from_kappa(kappa: float, speed_mps: float, n_waypoints: int, dt: float,
                    arc_len_m: float = DEFAULT_LENGTH_M) -> np.ndarray:
    """A chosen curvature -> an (N, 2) FLU plan of cumulative positions at `dt` spacing.

    Waypoint i is where the robot should be at t = (i+1)*dt, which is the convention the
    servers and controllers already use -- the origin is not a waypoint.

    The drawn arc is 3 m long because that is as far as the projection stays legible; past
    about 4 m the candidates converge toward the horizon and their labels collide. The plan
    has to cover a full horizon, which is longer. Holding the curvature for the whole
    horizon is the wrong reading of the model's answer: kappa = 0.60 held for 7 m is 240
    degrees, a pirouette nobody chose. So the plan drives the arc the model actually looked
    at and then continues straight along the heading it ends on, which is what "take the
    path that curves left" means to anyone who has ever driven -- turn onto it, then carry
    on. The curvature is re-chosen from a fresh image every replan, so a longer turn is
    expressed by choosing it again, not by extrapolating one choice further than it was
    made.
    """
    s = np.arange(1, n_waypoints + 1) * dt * speed_mps
    on_arc = np.minimum(s, arc_len_m)
    beyond = np.maximum(s - arc_len_m, 0.0)
    theta = kappa * on_arc                       # heading, frozen once past the arc
    if abs(kappa) < 1e-9:
        x, y = on_arc, np.zeros_like(on_arc)
    else:
        r = 1.0 / kappa
        x, y = r * np.sin(theta), r * (1.0 - np.cos(theta))
    return np.stack([x + beyond * np.cos(theta), y + beyond * np.sin(theta)], axis=-1)


def leftmost_label(arcs: list[Arc], labels: list[int]) -> int:
    """Label of the arc that turns hardest left -- ground truth for 'turn left'."""
    return labels[max(range(len(arcs)), key=lambda i: arcs[i].kappa)]


def rightmost_label(arcs: list[Arc], labels: list[int]) -> int:
    return labels[min(range(len(arcs)), key=lambda i: arcs[i].kappa)]


def straight_label(arcs: list[Arc], labels: list[int]) -> int:
    return labels[min(range(len(arcs)), key=lambda i: abs(arcs[i].kappa))]
