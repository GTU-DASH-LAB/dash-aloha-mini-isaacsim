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

The geometry is a level-camera flat-floor pinhole projection, exact for this rig:

    fx = (W/2) / tan(fov_h / 2),  cx = W/2, cy = H/2, square pixels
    a ground point (x forward, y left) at camera height h maps to
    u = cx - fx * y / x        v = cy + fy * h / x

Nothing here is learned or fitted. Both parameters come from the sim camera the frames
were rendered with (`server_qwen.py`: 0.346 m, 90.1 deg), so if the numbers change, change
them in one place.

Label assignment is deliberately NOT left-to-right. Numbering the arcs in spatial order
lets a model score well by answering "1" for left and "K" for right without ever looking
at the image, which is exactly the confound that would make this whole approach look like
it works when it does not. Callers pass their own permutation; the probe randomises it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The nav camera, from `nav/policy_server/server_qwen.py`'s prompt block. Level, forward.
CAM_HEIGHT_M = 0.346
CAM_FOV_DEG = 90.1
CAM_W, CAM_H = 1920, 1080

# Seven is a legible menu: enough to express "hard left" through "hard right" with a
# distinguishable straight-ahead, few enough that the labels stay readable at 1920x1080
# and the model is choosing rather than searching.
DEFAULT_CURVATURES = (-0.60, -0.35, -0.15, 0.0, 0.15, 0.35, 0.60)
# 3.0 m matches the horizon both policies were compared on, so a chosen arc is directly
# comparable to a predicted trajectory.
DEFAULT_LENGTH_M = 3.0

# Colours are per-arc and fixed by index so the same arc looks the same across frames,
# but they are never mentioned in the prompt -- the model must use position, not hue.
_PALETTE = [
    (255, 87, 51), (255, 189, 51), (163, 255, 51), (51, 255, 128),
    (51, 214, 255), (122, 51, 255), (255, 51, 195),
]


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


def project(x_forward: float, y_left: float, w: int = CAM_W, h: int = CAM_H,
            fov_deg: float = CAM_FOV_DEG,
            cam_h: float = CAM_HEIGHT_M) -> tuple[float, float] | None:
    """Ground point -> pixel. None when the point is at or behind the camera plane."""
    if x_forward <= 0.05:                       # behind, or on, the image plane
        return None
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    u = w / 2.0 - fx * (y_left / x_forward)
    v = h / 2.0 + fx * (cam_h / x_forward)      # square pixels, so fy == fx
    return u, v


def render_menu(image_path: str, out_path: str, arcs: list[Arc], labels: list[int],
                width: int = 13, font_size: int = 96) -> str:
    """Draw the arcs and their labels onto a copy of the frame. Returns `out_path`.

    Each arc gets a dark outline under its colour: hallway floors in these scenes are
    light grey and a thin bright line on light grey is exactly the sort of thing that
    survives inspection at full size and vanishes once the image is downsampled to the
    200704-pixel budget the server feeds the model.

    The defaults are sized for that budget, not for this file's own preview: 1920x1080 into
    200704 pixels is a 0.31x scale, so a 96 px label arrives as ~30 px and a 13 px stroke as
    ~4 px. Judge any change to them on the downsampled image, never on the full-size one.
    """
    from PIL import Image, ImageDraw, ImageFont

    if len(labels) != len(arcs):
        raise ValueError(f"{len(labels)} labels for {len(arcs)} arcs")
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i, (arc, label) in enumerate(zip(arcs, labels)):
        px = [p for p in (project(x, y, w, h) for x, y in arc.points) if p is not None]
        if len(px) < 2:
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        draw.line(px, fill=(20, 20, 20), width=width + 6, joint="curve")
        draw.line(px, fill=colour, width=width, joint="curve")

        # Label at the far end of the arc, which is where the paths are furthest apart and
        # a number is least likely to sit on top of a neighbour's. "Far end" means the last
        # point still INSIDE the frame, not the arc's true end: the outer arcs leave the
        # 90 degree view before 3 m, and clamping their true end to the border stacks their
        # badges in the corner where they read as a pair rather than as two paths.
        r = font_size * 0.72
        inside = [p for p in px if r < p[0] < w - r and r < p[1] < h - r]
        ux, uy = inside[-1] if inside else px[-1]
        ux = min(max(ux, r + 4), w - r - 4)
        uy = min(max(uy, r + 4), h - r - 4)
        draw.ellipse([ux - r, uy - r, ux + r, uy + r], fill=(20, 20, 20), outline=colour,
                     width=6)
        draw.text((ux, uy), str(label), fill=colour, font=font, anchor="mm")

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


def select_system(stop_label: int) -> str:
    """The arc-selection system prompt, with `stop_label` reserved for stopping."""
    return f"""You are the navigation system of a small wheeled robot, looking through its \
forward camera. The camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a \
{CAM_FOV_DEG:.0f} degree horizontal field of view.

Several candidate paths have been drawn onto the floor of the image. Each one is a real \
route the robot can drive, starting at the robot and curving away from it, and each ends \
in a numbered circle. The numbers are arbitrary tags, not an order: they do not run left \
to right and carry no meaning beyond identifying a path.

Your job is to choose the one path that best carries out the navigation instruction. \
Judge each path by where it actually goes on the floor in the image. Prefer a path that \
stays on open, walkable floor and does not run into a wall, a door frame or an object.

There is one extra choice, {stop_label}, which is not drawn on the image. Answer \
{stop_label} to stop and stay still, and only for one of two reasons: the robot has \
arrived at the place the instruction names, or every drawn path runs into something. \
Stopping at the right place is part of the task. Do not answer {stop_label} merely \
because the way ahead is tight.

Answer with the number of the chosen path and nothing else."""


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
