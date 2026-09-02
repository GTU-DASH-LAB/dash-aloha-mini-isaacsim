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
import os
import re
from dataclasses import dataclass

import numpy as np

# The nav camera, from `nav/policy_server/server_qwen.py`'s prompt block. Level, forward.
CAM_HEIGHT_M = 0.346
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
# Measured, at 3 m, against a label disc ~115 px wide at full resolution:
#
#     set       n    smallest gap between adjacent labels    colliding pairs
#     coarse    7          219 px                                  0
#     fine     11          158 px                                  0
#     (13)     13          108 px                                  4   <- rejected
#
# So 11 is the densest menu whose labels still separate, and 13 is not a finer menu, it
# is the same menu with two pairs of labels drawn on top of each other. The extra
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
                pivot_labels: tuple[int, int] | None = None) -> str:
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
                  pivot_labels: tuple[int, int] | None = None) -> str:
    """The arc-selection system prompt, with `stop_label` reserved for stopping.

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
    return f"""You are the navigation system of a small wheeled robot, looking through its \
forward camera. The camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a \
{CAM_FOV_DEG:.0f} degree horizontal field of view.

Several candidate paths have been drawn onto the floor of the image. Each one is a real \
route the robot can drive, starting at the robot and curving away from it, and each ends \
in a numbered circle. The numbers are arbitrary tags, not an order: they do not run left \
to right and carry no meaning beyond identifying a path.

Your job is to choose the one option that best carries out the navigation instruction. \
{STAGE_RULE} \
Judge each path by where it actually goes on the floor in the image. Prefer a path that \
stays on open, walkable floor and does not run into a wall, a door frame or an object.
{f"{chr(10)}{pivot_rule}{chr(10)}" if pivot_rule else ""}
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
