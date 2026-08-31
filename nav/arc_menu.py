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
import re
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

        r = font_size * 0.72
        ux, uy = badge_xy(px, w, h, r)
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


def select_system(stop_label: int, stop_allowed: bool = True,
                  speed_choice: bool = False) -> str:
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
    stop_rule = f"""There is one extra choice, {stop_label}, which is not drawn on the \
image. Answer {stop_label} to stop and stay still, and only for one of two reasons: the \
robot has arrived at the place the instruction names, or every drawn path runs into \
something. Stopping at the right place is part of the task. Do not answer {stop_label} \
merely because the way ahead is tight.""" if stop_allowed else """Stopping is not one of \
your choices here. The robot has not arrived anywhere, and it has just reversed a little \
after failing to make any progress, so standing still is the one thing already known not \
to work. Pick the drawn path with the most open floor along it, even if none of them is \
good."""

    # The speed question is asked ONLY near the target, and the prompt says so, because
    # "how fast" is a different question at 12 m and at 2 m. Far away it has one sensible
    # answer -- go -- and asking it anyway spends tokens and adds a number that can go
    # wrong for no gain. Close in, it is the actual difficulty: the arcs are all 3 m long,
    # so with the target 2 m off the choice is not WHERE to drive but HOW MUCH of the
    # chosen path to use before looking again.
    answer_rule = (f"""Answer with TWO whole numbers separated by a space: first the \
number of the chosen path, then how fast to drive it, from 0 (barely creeping) to \
{SPEED_LEVELS} (full speed). You are close to what the instruction names, so choose the \
speed deliberately: slow down to place the robot accurately, keep the speed up if there is \
still ground to cover. A low number never means stopping -- it means moving gently. \
Write nothing but the two numbers.""" if speed_choice else
                   "Answer with the number of the chosen path and nothing else.")
    return f"""You are the navigation system of a small wheeled robot, looking through its \
forward camera. The camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a \
{CAM_FOV_DEG:.0f} degree horizontal field of view.

Several candidate paths have been drawn onto the floor of the image. Each one is a real \
route the robot can drive, starting at the robot and curving away from it, and each ends \
in a numbered circle. The numbers are arbitrary tags, not an order: they do not run left \
to right and carry no meaning beyond identifying a path.

Your job is to choose the one path that best carries out the navigation instruction. \
{STAGE_RULE} \
Judge each path by where it actually goes on the floor in the image. Prefer a path that \
stays on open, walkable floor and does not run into a wall, a door frame or an object.

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
