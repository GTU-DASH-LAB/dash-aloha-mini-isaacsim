"""Q-VLA-direct: Qwen3.8-27B emits the waypoint plan itself, with no action expert.

The premise, and why it is worth testing
----------------------------------------
TIC-VLA splits navigation into a slow VLM that produces a KV cache and a fast 10.27 M
`ActionExpert` that turns that cache into 30 waypoints. Swapping the VLM therefore means
surgery: the expert's two input projections are shaped to InternVL3-1B (896 / 128) and
Qwen3.8-27B is 5120 / 1024, so 3.15 M parameters have to be re-initialised and stage-2
retrained before a single episode can run.

This server asks whether that is necessary at all. If the VLM can simply *write down* the
plan, the expert, the coupling, the re-initialisation and the retraining all disappear,
and what is left is a prompt. That is a real experiment with a real chance of failing --
a stock VLM has no metric scale from a monocular image -- but it is cheap to run and the
benchmark can score it against the 6/13 the trained stack gets.

The contract is deliberately unchanged
--------------------------------------
Same `/health`, `/reset`, `/predict`, same request and response models as `server.py`, so
`client.py`, `run_navigation.py` and every controller work against this server without
edits. Swapping the two is a matter of which launch script ran.

Three things this file has to get right, none of them about prompting
---------------------------------------------------------------------
1. **Decode is the entire latency budget, so the output format is a latency decision.**
   Measured on this hardware (`nav/tools/probe_qvla_latency.py`): decode is ~95% of a
   27B call and flat across a 10x change in prompt size. Emitting all 30 waypoints as
   "(0.06, 0.01), ..." costs 364 tokens; six control points cost 76. Densifying six
   points back to 30 was validated against 24 REAL action-expert plans pulled from the
   live TIC-VLA server: max position error 0.13 m and mean heading error at the 1 s
   lookahead 0.63 deg, against a 1.5 m success threshold and the ~25 deg heading deficit
   that actually loses episodes -- roughly 10x and 40x of margin. A 3 s wheeled-base
   trajectory at 0.73 m/s simply has no high-frequency content to lose.

2. **The plan is stale by one generation and must be re-expressed, not replayed.**
   TIC-VLA gets away with a stale KV cache because the action expert re-runs every call
   on the *current* frame. Here there is no fast head: the plan itself ages. A plan
   written 1.7 s ago is in a body frame the robot has since left, so it is rotated and
   translated into the current frame, and the waypoints whose scheduled time has passed
   are dropped. The plan therefore gets SHORTER as it ages, which is the truth --
   padding it back to 30 would fabricate motion the model never planned.

3. **A parse failure must not become a navigation failure.** A malformed generation
   reuses the previous plan (already the semantics of a stale cache) and increments a
   counter that `/health` exposes. A benchmark whose policy silently emits zeros scores
   a number that means nothing.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.interpolate import PchipInterpolator

# Running this file puts its own directory on sys.path (that is how `qwen_load` resolves)
# but not `nav/`, where the shared code lives. The `menu` format needs it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arc_menu import (  # noqa: E402
    CREEP_MPS, DESCRIBE_AFTER_RECOVERY, DESCRIBE_SIDED, DESCRIBE_TARGET,
    FREE_SPACE_SYSTEM, SELECT_AFTER_RECOVERY, make_arcs,
    parse_target_distance, plan_from_kappa, render_menu, speed_for,
    select_system,
)

MODEL_PATH = os.environ.get(
    "QVLA_MODEL", "/home/gtu-dsa/robotics/models/Qwen3.8-27B-NVFP4"
)
# 200704 px = 448x448. On this vision tower a frame costs exactly
# pixels / (patch_size * spatial_merge)^2 = pixels / 1024 tokens, so this is 196
# tokens per frame -- the same budget InternVL3-1B feeds today. Native 1920x1080 would
# be 2059 tokens per frame and 4.1 s more prefill for no measured benefit.
MAX_PIXELS = int(os.environ.get("QVLA_MAX_PIXELS", 200704))
# Wall-clock cap on one generation: a safety net against a hung decode, NOT a freshness
# gate. Freshness is enforced downstream by slicing the plan by `time_delay`, which is
# exact where a timeout is a guess. Deliberately loose, because the FIRST call pays
# DeepGEMM's JIT warmup -- measured at 20.0 s against 2.7 s steady-state -- and a timeout
# tight enough to be a quality gate would reject that one and start the episode with no
# plan at all.
GEN_TIMEOUT_S = float(os.environ.get("QVLA_GEN_TIMEOUT_S", 45.0))

# THE HORIZON MUST OUTLAST A GENERATION, and 3.0 s did not.
#
# This started as 30 points over 3.0 s to match the action head exactly. That parity is
# wrong here, and measurably so. TIC-VLA's action expert re-runs on the CURRENT frame
# every control tick, so its waypoints are never stale -- only the KV cache behind them
# is. Take the expert away and the waypoints themselves become the thing that ages, and
# they age against a 27B generation that takes 2.7-5 s.
#
# What that cost, measured on `office_nearest_elevator`: the server slices the cached plan
# by staleness (`full[ceil(time_delay/DT):]`), so at 2.9 s exactly ONE point survives;
# `plan_speed` returns 0.0 for a plan shorter than two points; and `braking` does
# `v_cap = min(v_cap, plan_speed(...))`. The robot is commanded to a full stop. 129 of 141
# calls ran at staleness >= 3.0 s. The robot moved 0.27 m in 70 s and timed out. The model
# was never at fault -- feeding one frame seven different `time_delay` values returns the
# SAME plan reported as 1.353, 0.887, 0.420, then 0.000 m/s.
#
# So the horizon is set by inference latency, not by parity: 10.0 s leaves 5 s of usable
# plan even at the worst staleness observed. This is what the original instruction asked
# for -- "the next 100 actions, so the control loop is fast" -- and 100 points at 10 Hz is
# exactly that. DT is unchanged at 0.1 s, so the control rate is the same 10 Hz.
HORIZON_S = 10.0
N_WAYPOINTS = 100
# Ten control points, one per second. The model writes these; PCHIP densifies to
# N_WAYPOINTS. Spacing them 1.0 s apart rather than packing more in keeps the count the
# model has to hold in its head roughly where it was at 6.
CTRL_TIMES = tuple(float(i) for i in range(1, 11))
DT = HORIZON_S / N_WAYPOINTS          # 0.1 s -- 10 Hz, unchanged

# Reject a hallucinated scale, but DERIVE the threshold, because a hardcoded one is a
# second copy of the horizon hiding in the parser. This was a flat 6.0 m -- 3 s at the
# 1.5 m/s cap is 4.5 m, plus a third for slack -- and lengthening the horizon without it
# cost a whole restart: the model wrote a perfectly good `(0.70, 0.00) ... (7.00, 0.00)`,
# 0.7 m/s held for 10 s, and the parser threw it away for exceeding 6.0. The result was
# indistinguishable from the failure being fixed -- `has_plan: false`, every waypoint
# zero, robot stopped -- and only `parse_failures: 1` told the two apart. 2.0 m per second
# of horizon reproduces the original 6.0 exactly at 3 s.
MAX_REACH_M = 2.0 * HORIZON_S

app = FastAPI(title="Q-VLA direct policy server")

_state: dict = {
    "model": None, "proc": None, "predictions": 0, "generations": 0,
    "parse_failures": 0, "empty_plans": 0, "gen_errors": 0,
    # The plan currently in force, in the body frame of the moment generation STARTED.
    "plan": None, "plan_gen_step": None, "reasoning": None,
    "gen_thread": None, "gen_step": None,
    "think": os.environ.get("QVLA_THINK", "0") == "1",
    # Where this episode's menus and decisions are being written; set by /reset, None
    # until a caller names a run.
    "run_dir": None,
    # Bumped by /reset and /replan. A worker captures it when it starts and drops its
    # plan if it no longer matches, which is what makes "throw the cached plan away"
    # mean anything: a generation in flight was started on the OLD view, so without
    # this it finishes a second later and installs exactly the plan that was discarded.
    # Between episodes it is the same bug wearing a different hat -- the last plan of
    # one run landing in the first step of the next.
    "epoch": 0,
    "stale_discards": 0,
}
_lock = threading.Lock()
# Serialises GPU decode for `/raw` alone. `_lock` cannot do this job: it is held for
# microseconds around `_state` updates while generation runs outside it, by design, so
# that a background plan never blocks the control loop's status reads.
_raw_gate = threading.Lock()


class PredictRequest(BaseModel):
    """Mirrors server.py's request, plus one field.

    `yaw_delta_rad` is new and optional. server.py does not need it because TIC-VLA's
    action expert re-runs on the current frame every call, so only translation had to be
    declared. Here the plan itself is stale, and rotating it back requires the heading
    change since generation. Defaulting to 0.0 keeps every existing caller valid -- it
    just means "assume the robot has not turned", which is exactly what the old contract
    implied anyway.
    """

    image_paths: list[str] = Field(..., min_length=1)
    instruction: str
    robot_state: list[float] = Field(default_factory=lambda: [0.0] * 6)
    current_step: int | None = None
    time_delay: float = 0.0
    previous_waypoints_text: str = ""
    delayed_image_paths: list[str] | None = None
    robot_type: str = "wheeled robot"
    yaw_delta_rad: float = 0.0
    # True for the handful of decisions taken just after the robot reversed out of a
    # wedge. It is a fact about the last few seconds that the camera cannot show -- the
    # view is the same obstacle from a metre further back -- so it has to be told. See
    # `DESCRIBE_AFTER_RECOVERY` in arc_menu.py for why it is spent on the prompt here
    # when the same idea was worth nothing on TIC-VLA. Defaults False, so every existing
    # caller keeps its exact behaviour.
    recovered: bool = False


class PredictResponse(BaseModel):
    waypoints: list[list[float]]
    reasoning: str | None
    num_waypoints: int
    latency_s: float
    kv_cache_available: bool          # here: "a plan exists", same meaning to the caller
    vlm_generation_start_step: int | None


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
# The system and instruction framing are copied from TIC-VLA's own training format
# (`ticvla/data/vlm_data.py:_build_messages`) rather than reworded. That format is the
# one the benchmark's episodes and `previous_waypoints_text` were written against, and
# `nav/README.md` already records what happens when a prompt in this stack is rephrased
# to read better to a human: it lands off the distribution the rest of the pipeline
# assumes. What is added below is only what a model that was never fine-tuned on this
# task cannot know.

# Ask for the plan in a y-is-RIGHT-positive frame, then negate y on the way back in.
#
# This exists because of a measured, one-sided failure and not because either convention
# is nicer. Under the natural FLU wording (y positive = left), Qwen3.8-27B writes a clean
# right-turn arc when told "Turn right." -- (0.32, -0.01), (0.64, -0.03), (0.96, -0.06),
# ... a proper parabola -- and writes literal 0.00 six times for "Turn left.", in five
# different scenes, and again when told "y must be POSITIVE and grow along the list", and
# again when told "answer with the exact mirror image of a right turn". It never emits a
# positive y at all. That is not a comprehension failure and not obstacle avoidance; the
# geometry it produces for the direction it CAN express is correct.
#
# The hypothesis it tested was narrow: the model can only put a MINUS sign in that slot,
# so the steerable side should follow the wording and flipping should buy left turns at
# the cost of right ones.
#
# MEASURED, AND THE ANSWER IS NO -- keep this at 0. Flipping does not move the dead side,
# it kills the live one: with y-positive-is-RIGHT, "Turn right." goes to 0.00 as well, in
# all five scenes. Both directions dead. So the constraint is not "only minus is
# writable". What fits both runs is that the model holds its own FLU convention (right is
# negative y, which is correct) and will only produce a nonzero y when the prompt AGREES
# with it; contradict it and it stops steering entirely rather than following the wording.
#
# The consequence is the useful part: a sign convention is not the bug, so no rewording of
# it is the fix. Getting a left turn out of this model requires an output format that does
# not ask it to emit a signed lateral offset at all -- a direction word plus an unsigned
# magnitude, say. Kept, not deleted, so the next person does not re-derive it: this is
# 20 generations of evidence for one line of config.
FLIP_Y = os.environ.get("QVLA_FLIP_Y", "0") == "1"

_SYSTEM = (
    "You are a {robot_type} assigned to perform navigation tasks.\n"
    "You are provided with a video consisting of visual observations, including "
    "historical and current frames.\n"
)

# The scale paragraph is the single most important thing here. A monocular VLM cannot
# recover metric scale from an image -- but it does not have to, because
# `previous_waypoints_text` hands it its own displacement over the last seconds in
# metres. That is a measured ruler laid over the scene, and saying so explicitly is the
# difference between "predict metres" (impossible) and "continue at roughly the speed you
# just travelled" (easy).
_TASK = """
The four images are consecutive frames from your forward camera, oldest first, about 3 seconds apart. The last one is NOW.

Your camera: mounted {cam_h:.2f} m above the floor, pointing straight ahead and level, horizontal field of view {cam_fov:.0f} degrees. The horizontal centre of the image is straight ahead; the left edge is about {half_fov:.0f} degrees to your left, the right edge {half_fov:.0f} degrees to your right.

Scale: use the waypoint list above as your ruler. Those numbers are how far you actually moved, in metres, over the last seconds. Your speed never exceeds 1.5 m/s.

Predict where you should be at {times} seconds from now, as {n} waypoints (x, y):
- x is metres FORWARD, positive; y is metres {pos_side}, positive.
- Each waypoint is the CUMULATIVE offset from where you are right now, not from the previous waypoint. So x must increase along the list while you are still moving forward.
- Steer by making y negative to go {neg_side} and positive to go {pos_side}. A gentle course correction is a few centimetres of y; a real turn is tens of centimetres.
- Do not drive into walls, furniture, people or shelving. Steer around obstacles and leave clearance.
- If you have arrived at the target the instruction names, STOP: return waypoints that barely change, for example (0.05, 0.00) repeated. Stopping at the right place is part of the task, not the end of it. Do not drive past the target.

Reply with the waypoints only, inside answer tags, and nothing else. Write exactly {n} of them:
<answer>{slots}</answer>
"""

# The output format the FLIP_Y measurement points at. Under `pairs` the model must write a
# signed lateral offset and it will not write a positive one -- not for any wording, and
# not for either sign convention. `arc` removes the signed number from the model's job
# entirely: it writes a DIRECTION WORD, an UNSIGNED magnitude in degrees, and the six
# distances it already writes correctly. We apply the sign and build the curve.
#
# The split is deliberate about which channel it touches. Speed is the part that measured
# GOOD -- Q-VLA braked to 0.01 m/s at the same call TIC-VLA braked to 0.18 -- so the six
# cumulative path lengths stay exactly as expressive as the six x values were. Only the
# lateral channel changes representation. If left turns appear under `arc` and nothing
# else moves, the cause is isolated to the signed-offset encoding and nothing else.
#
# `menu` is a third thing and does not belong to that comparison at all: the model stops
# writing a trajectory in any form and instead PICKS one off a menu drawn on its own
# camera image (`nav/arc_menu.py`). It exists because `pairs` and `arc` were both measured
# and both fail in the same place -- the model reasons correctly about where to go and
# then writes a number it cannot ground, since nothing in a monocular frame tells it how
# far 2.07 m is. Selecting a label removes the ungrounded number rather than re-encoding
# it, and it measures 100% on following a direction word against 43% chance. See CLAUDE.md.
_ARC_FORMAT = os.environ.get("QVLA_FORMAT", "pairs").lower()
if _ARC_FORMAT not in ("pairs", "arc", "menu"):
    # Loudly, not by falling back. A typo here would silently run the format we already
    # know is one-sided and label the result as the new one -- an hour of GPU time spent
    # re-measuring a known negative.
    raise SystemExit(f"QVLA_FORMAT must be 'pairs', 'arc' or 'menu', got {_ARC_FORMAT!r}")

# How fast the chosen arc is driven. Unlike `pairs` and `arc`, `menu` has no speed channel
# at all: every candidate is the same 3 m long, so the model chooses a DIRECTION and
# nothing else, and the speed has to come from here. 0.7 m/s is the operating point the
# trained policy asks for on these episodes (measured mean 0.731 on office_nearest_
# elevator), so the two are driven at the same pace and the comparison is about steering.
#
# The consequence is worth stating rather than discovering: `plan_speed` is constant under
# this format, so the `braking` controller's min(cap, plan) never engages and the only way
# this policy can slow down is to stop outright. That is what the STOP label is for.
MENU_SPEED_MPS = float(os.environ.get("QVLA_MENU_SPEED", "0.7"))
MENU_DIR = Path(os.environ.get("QVLA_MENU_DIR", "/tmp/qvla-menus"))
# Menus are keyed by the control step, and the step counter restarts with every episode, so
# a flat directory has the second episode overwriting the first at the same numbers. With
# a run label from /reset each episode gets its own folder plus a JSONL of what the model
# saw and chose -- which together are a complete, replayable record of the policy's side of
# the run. `nav/tools/make_run_video.py` turns one into a video.
RECORD = os.environ.get("QVLA_RECORD", "1") not in ("0", "", "false", "no")

_TASK_ARC = """
The four images are consecutive frames from your forward camera, oldest first, about 3 seconds apart. The last one is NOW.

Your camera: mounted {cam_h:.2f} m above the floor, pointing straight ahead and level, horizontal field of view {cam_fov:.0f} degrees. The horizontal centre of the image is straight ahead; the left edge is about {half_fov:.0f} degrees to your left, the right edge {half_fov:.0f} degrees to your right.

Scale: use the waypoint list above as your ruler. Those numbers are how far you actually moved, in metres, over the last seconds. Your speed never exceeds 1.5 m/s, so no distance below can be more than {max_dist:.1f}.

Describe the path you should drive over the next {horizon:.0f} seconds, as a turn plus a distance profile.

First the turn, as one word and one number:
- LEFT, RIGHT, or STRAIGHT.
- Then how many degrees you will have turned by the end of the 3 seconds, as a POSITIVE number from 0 to 90. Never write a minus sign. The word carries the direction; the number is only the size. A gentle course correction is 5 to 15 degrees; a corridor turn is 45 to 90. Write STRAIGHT 0 if you are not turning.

Then the distance, as {n} numbers: how far you will have travelled ALONG THE PATH at {times} from now, in metres, cumulative from where you are right now. These must never decrease. Slow down by making the gaps between them shrink; hold a steady pace by keeping the gaps equal.
- These {n} numbers describe THIS scene at THIS moment. Read them off what you can see: how much clear floor is ahead of you, and how far away the target is. Do not fall back on a stock profile.
- Do not drive into walls, furniture, people or shelving. Turn away from obstacles and leave clearance.
- If you have arrived at the target the instruction names, STOP: write STRAIGHT 0 and distances that barely change. Stopping at the right place is part of the task, not the end of it. Do not drive past the target.

Reply with the turn and the distances only, inside answer tags, and nothing else. The shape is one word, one number, a vertical bar, then {n} numbers separated by commas:

<answer>DIRECTION DEGREES | {dslots}</answer>

DIRECTION, DEGREES and d1 to d{n} are placeholders. Replace every one of them with your own value. Do not copy them, and do not copy any number that appears anywhere in these instructions.
"""

_THINK_TASK = (
    "\nBefore the answer, give at most two short sentences of reasoning inside "
    "<think></think> tags: what you see, and where you are going.\n"
)

# Written into the assistant's turn before generation starts, so the first token the model
# is free to choose is already inside a coordinate. See `_generate` for why the polite
# version of this instruction is not enough. It is prepended back onto the decoded text
# before parsing -- without that the leading "(" is missing and the first pair is lost.
#
# `arc` stops one token earlier. The prefill has to leave the direction word free -- that
# word IS the prediction -- so it can only force the tag, not the first character inside it.
ANSWER_PREFIX = "<answer>" if _ARC_FORMAT == "arc" else "<answer>("


# Read once, at import, so a benchmark run cannot change decoding halfway through and so
# the value is visible in one place rather than at two call sites in `_generate`. Unset
# means greedy, which stays the default: sampling is the experiment, not the baseline.
_TEMPERATURE = float(os.environ.get("QVLA_TEMPERATURE", "0") or 0)
_SAMPLING: dict = (
    {"do_sample": True, "temperature": _TEMPERATURE, "top_p": 0.9}
    if _TEMPERATURE > 0 else {"do_sample": False}
)


def _answer_tokens() -> int:
    """Decode budget for the coordinate list alone, derived from the control-point count.

    `pairs` writes about 14 tokens per point -- "(0.35, 0.00), " -- so a flat 110 was sized
    for six of them and would have truncated ten mid-list, turning a horizon change into a
    parse failure with no visible connection to its cause. `arc` is cheaper per point (one
    bare number) but sizing both off the same count costs nothing and cannot drift apart.

    A function rather than a constant because the budget-forcing second pass in `_generate`
    needs exactly this number too, and a second copy of it is the bug class this file has
    already paid for three times over. See CLAUDE.md.
    """
    return 20 * len(CTRL_TIMES) + 30


def build_messages(req: PredictRequest, image_paths: list[str], think: bool) -> list[dict]:
    prev = req.previous_waypoints_text.strip()
    if not prev:
        # DynaNav raises on an empty string and it is right to: "no history yet" has its
        # own wording in the training data, and the empty string is not it. See
        # nav/README.md -- defaulting this away is what made every plan a fresh straight
        # line back when the TIC-VLA server did it.
        prev = "From 0.0s to current timestamp time is 0.0s. No waypoints available."

    # A bootstrap sentence was added here and REMOVED, because it was measured and did
    # nothing. The theory was that "No waypoints available." reads as prose to this model
    # (TIC-VLA's action expert never reads it -- it consumes a KV cache), so being handed
    # no "ruler" right after being told to use one made it write zeros. Telling it plainly
    # that standing still at t=0 is expected and the task begins now left the standstill
    # case at 0.000 m/s in 3 of 3, unchanged.
    #
    # It failed because the premise was too small. With thinking off, the plan is not a
    # response to the history OR the image -- it is a number copied out of the prompt; see
    # the perception probe in CLAUDE.md. Zeros at the start are one face of that, not a
    # missing-ruler problem, and no wording fixes it. Left as a comment so the next reader
    # does not re-derive an attractive theory the evidence has already closed.

    if _ARC_FORMAT == "arc":
        task = _TASK_ARC.format(
            cam_h=0.346, cam_fov=90.1, half_fov=45.0,
            # Units on every entry, and "and" before the last, so the timestamps do not
            # read as a bare six-number list. `pairs` prints them plain; here that would
            # sit in the prompt looking exactly like the answer being asked for, and a
            # steady 1.0 m/s plan has those same six cumulative distances -- so a parser
            # guard could not tell a copy from a correct answer. Fix it upstream instead.
            times=", ".join(f"{t:.1f} s" for t in CTRL_TIMES[:-1]) + f" and {CTRL_TIMES[-1]:.1f} s",
            n=len(CTRL_TIMES),
            # Generated, not typed: the answer template used to carry a real six-number
            # profile, and the model copied it verbatim in 2 of 8 calls. Placeholders
            # leave nothing to copy -- and generating them means a change to CTRL_TIMES
            # cannot leave the example showing a different count than the text asks for.
            dslots=", ".join(f"d{i}" for i in range(1, len(CTRL_TIMES) + 1)),
            # Both derived, for the reason MAX_REACH_M is: a horizon written as a literal
            # in the prompt is a copy of the constant that will not be updated with it,
            # and here it would tell the model a ceiling the parser no longer enforces.
            horizon=HORIZON_S,
            max_dist=1.5 * HORIZON_S,
        )
    else:
        task = _TASK.format(
            cam_h=0.346, cam_fov=90.1, half_fov=45.0,
            times=", ".join(f"{t:.1f}" for t in CTRL_TIMES), n=len(CTRL_TIMES),
            pos_side="RIGHT" if FLIP_Y else "LEFT",
            neg_side="left" if FLIP_Y else "right",
            # Generated for the same reason the arc template is, and it turned out to
            # matter just as much here. The template used to be six literal `(x, y)`
            # pairs next to a sentence naming 0.7 m/s over 3 s, and probing ONE frame
            # with three different histories returned `(0.35, 0.00) ... (2.10, 0.00)`
            # nine times out of nine -- 0.7 x [1..6], the prompt's own arithmetic read
            # back. Both the number and the pair-count now come from CTRL_TIMES.
            slots=", ".join("(x, y)" for _ in CTRL_TIMES),
        )
    if think:
        task += _THINK_TASK

    user_text = f"The navigation instruction is: {req.instruction}\n{prev}\n{task}"
    return [
        {"role": "system", "content": _SYSTEM.format(robot_type=req.robot_type)},
        {"role": "user", "content":
            [{"type": "image", "image": str(p)} for p in image_paths]
            + [{"type": "text", "text": user_text}]},
    ]


# --------------------------------------------------------------------------------------
# Parsing and densification
# --------------------------------------------------------------------------------------
# `[-+]?`, not `-?`. An explicit plus is exactly what a model writes when the prompt draws
# its attention to the sign -- and this one says "make y negative to go right and positive
# to go LEFT", which invites `+0.15` for precisely the turn that was recorded as impossible.
# `(0.32, +0.01)` failed the whole pair match, `parse_control_points` returned None, and the
# runner saw no plan: the same all-zeros symptom as the model refusing to steer. Every probe
# that concluded "it never emits a positive y" read that symptom, and none of them read the
# raw text. `float("+0.15")` has always worked; only the regex disagreed.
_NUM = r"[-+]?\d+(?:\.\d+)?"
# The third component is optional and discarded. TIC-VLA's training format is
# (x, y, theta) and theta there is literally atan2(y, x) of the same pair -- redundant --
# so a model that emits one out of habit is not wrong, just verbose. Rejecting those
# would throw away a perfectly good plan over a number that carries no information.
_PAIR = re.compile(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*(?:,\s*{_NUM}\s*)?\)")


_ARC_HEAD = re.compile(r"\b(LEFT|RIGHT|STRAIGHT)\b[^0-9\-]*(\d+(?:\.\d+)?)?", re.I)


def parse_arc(text: str) -> np.ndarray | None:
    """Pull `WORD degrees | d1, d2, ...` out of a generation. Returns (K, 2) or None.

    The whole point of this format is that the model never writes a signed number, so the
    sign is applied here, once, from the direction word. Everything downstream is FLU and
    stays FLU: left is +y, which is the convention the model was measured to already hold.

    The curve is a constant-curvature arc, which is the honest reading of "you will have
    turned D degrees by the end". Two numbers cannot describe more than that, and pretending
    otherwise -- easing the turn in, say -- would be us inventing a manoeuvre the model did
    not write, the same failure mode PCHIP was chosen over a natural cubic to avoid.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    body = m.group(1) if m else text

    h = _ARC_HEAD.search(body)
    if h is None:
        return None
    word = h.group(1).upper()
    deg = float(h.group(2)) if h.group(2) else 0.0
    # Unsigned by construction: a minus sign cannot survive the regex, and clamping at 90
    # matches the prompt. 90 deg in 3 s is already 30 deg/s -- past that is a misread.
    deg = min(abs(deg), 90.0)

    # Distances come from AFTER the header, so the degree number can never be read as the
    # first distance. Without that slice, "LEFT 30 | 0.35, ..." parses 30 as a 30 m step.
    dists = [float(v) for v in re.findall(_NUM, body[h.end():])]
    if len(dists) < 2:
        return None
    s = np.maximum.accumulate(np.clip(np.array(dists[:len(CTRL_TIMES)]), 0.0, None))
    if s[-1] > MAX_REACH_M:
        return None

    total = float(s[-1])
    psi = np.deg2rad(deg) * (1.0 if word == "LEFT" else -1.0 if word == "RIGHT" else 0.0)
    if total < 1e-6 or abs(psi) < 1e-6:
        return np.stack([s, np.zeros_like(s)], axis=-1)   # straight, and the kappa->0 limit
    kappa = psi / total                   # constant curvature: heading(s) = kappa * s
    return np.stack([np.sin(kappa * s) / kappa,
                     (1.0 - np.cos(kappa * s)) / kappa], axis=-1)


def parse_control_points(text: str) -> np.ndarray | None:
    """Pull (x, y) pairs out of a generation. Returns (K, 2) or None.

    Deliberately forgiving about everything except the numbers. The model is not
    fine-tuned on this format, so it will sometimes wrap the answer in prose, close the
    tag late, or emit a third component out of habit (TIC-VLA's own training format is
    (x, y, theta), and theta there is literally atan2(y, x) -- redundant, so dropping a
    third number loses nothing).
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    body = m.group(1) if m else text
    pairs = _PAIR.findall(body)
    if len(pairs) < 2:
        return None
    pts = np.array([[float(a), float(b)] for a, b in pairs], dtype=np.float64)
    if FLIP_Y:
        # The model was asked in a y-is-right-positive frame; everything downstream of
        # here -- densify, reframe, the controllers, the traces -- is FLU. Undo it once,
        # here, so the convention cannot leak past this function. See FLIP_Y above.
        pts[:, 1] = -pts[:, 1]

    # A plan that goes backwards is a misread, not a manoeuvre: this base does not
    # reverse and no training trajectory does either. Clamp rather than reject, so one
    # bad component does not throw away an otherwise usable plan.
    pts[:, 0] = np.maximum.accumulate(np.clip(pts[:, 0], 0.0, None))
    if pts[-1, 0] > MAX_REACH_M or np.abs(pts[:, 1]).max() > MAX_REACH_M:
        return None
    return pts


def densify(ctrl: np.ndarray) -> np.ndarray:
    """(K, 2) control points -> (N_WAYPOINTS, 2) at 10 Hz.

    PCHIP rather than a natural cubic because it cannot overshoot: a cubic through six
    points can curl outside their convex hull and invent a swerve the model did not
    write. The origin is prepended as a free, exact control point -- the plan provably
    starts where the robot is.

    Validated against 24 real action-expert plans: 0.13 m max position error, 0.63 deg
    mean heading error at the 1 s lookahead. See the module docstring.
    """
    k = len(ctrl)
    tc = np.concatenate([[0.0], np.array(CTRL_TIMES[:k]) if k <= len(CTRL_TIMES)
                         else np.linspace(CTRL_TIMES[0], CTRL_TIMES[-1], k)])
    cc = np.vstack([[0.0, 0.0], ctrl])
    t = np.arange(1, N_WAYPOINTS + 1) * DT
    return np.stack([PchipInterpolator(tc, cc[:, d])(t) for d in (0, 1)], axis=-1)


def reframe(plan: np.ndarray, dx: float, dy: float, dyaw: float,
            age_s: float) -> np.ndarray:
    """Express a plan written `age_s` ago in the CURRENT body frame, and drop what is spent.

    Two corrections, and both are needed:
      - spatial: the plan's origin was the robot's pose at generation. Subtract the
        translation since, then rotate by -dyaw. This is the same 2D form of
        `R_start.T @ delta_world` the runner already uses for dx, dy.
      - temporal: waypoint i was for t = (i+1)*0.1 s after generation. Points whose time
        has passed are behind the robot and are dropped. The plan therefore shortens as
        it ages, which is honest -- a 3 s plan really does run out 3 s later.
    """
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    rel = plan - np.array([dx, dy])
    out = np.stack([c * rel[:, 0] - s * rel[:, 1],
                    s * rel[:, 0] + c * rel[:, 1]], axis=-1)
    keep = int(np.ceil(max(0.0, age_s) / DT))
    return out[keep:]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _load_model() -> None:
    if _state["model"] is not None:
        return
    from qwen_load import load_qwen

    t0 = time.time()
    print(f"[qvla-server] loading {MODEL_PATH}", flush=True)
    # QVLA_MAX_MEMORY="0:19,1:22" caps per device. Without a cap, device_map="auto" takes
    # whatever is free at load time and will crowd out a neighbour -- Isaac Sim holds
    # ~25.4 GiB of GPU0 during a benchmark run.
    mm = os.environ.get("QVLA_MAX_MEMORY", "").strip()
    max_memory = ({int(k): float(v) for k, v in
                   (kv.split(":") for kv in mm.split(",") if kv)} if mm else None)
    # load_qwen carries the two fixes without which this checkpoint is either unloadable
    # or silently broken -- see qwen_load.py. The second one matters here specifically:
    # a model whose gate_proj lost its FP8 scale still runs at full speed and emits
    # gibberish, which a navigation benchmark would score as "the policy drove badly".
    proc, model = load_qwen(MODEL_PATH, max_memory=max_memory, max_pixels=MAX_PIXELS)
    _state["proc"], _state["model"] = proc, model
    print(f"[qvla-server] ready in {time.time() - t0:.1f}s", flush=True)


def _generate(messages: list[dict], image_paths: list[str], max_new: int) -> str:
    """Generate, with the answer's opening bracket already written for the model.

    Asking politely for "the waypoints only" does not work on a model that was never
    fine-tuned on this format: it complies with the *format* and still prefaces it with
    a paragraph of reasoning, which costs decode -- the entire latency budget -- and,
    worse, gives the parser numbers to find that were never meant as waypoints.

    Putting `<answer>(` at the end of the prompt makes the point structurally instead of
    rhetorically. The next token cannot be prose; it has to be a digit. Measured against
    the polite version: the reply goes from a paragraph plus coordinates to coordinates
    alone, and there is no path by which a stray number from the reasoning reaches the
    plan, because there is no reasoning.

    `enable_thinking=False` is separate and also needed: Qwen3.5 is a thinking model and
    its template turns reasoning on by default -- a stock checkpoint answers "what is the
    capital of France" with a <think> block. QVLA_THINK=1 opts back in, in the two-sentence
    form the prompt asks for, so the value of reasoning here can be measured rather than
    assumed.
    """
    from PIL import Image

    proc, model = _state["proc"], _state["model"]
    kwargs = {} if _state["think"] else {"enable_thinking": False}
    try:
        text = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs)
    except TypeError:
        # Older processors do not take the flag; losing the saving beats not running.
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # With QVLA_THINK the model needs room to open its own <think> block first, so the
    # answer cannot be forced from the very first token.
    prefix = "" if _state["think"] else ANSWER_PREFIX
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = proc(text=[text + prefix], images=imgs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Greedy by default. The claim this comment used to make -- that sampling "buys
        # nothing when the output is a list of coordinates" -- was never measured, and the
        # evidence now points the other way: greedy decoding into a heavily templated
        # answer collapses onto a clean arithmetic sequence, `(0.70, 0.00) … (7.00, 0.00)`,
        # identical across five unrelated scenes and unmoved by the model's own reasoning
        # that it should steer left. A collapsed mode is exactly what a temperature is for.
        # QVLA_TEMPERATURE opts in so the question can be settled instead of asserted.
        out = model.generate(**inputs, max_new_tokens=max_new, **_SAMPLING)
    gen = proc.batch_decode(out[:, inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True)[0]
    if not _state["think"]:
        return prefix + gen

    # BUDGET FORCING. Asking this model to think and then answer gets the thinking and not
    # the answer: at a 1024-token cap it ran to the cap and emitted no `</think>` in 5 of 6
    # generations, so the parser saw nothing and the robot stopped. The reasoning itself
    # was good -- correct scene ("a black console table on the right, windows with vertical
    # blinds"), correct geometry off the stated 90 degree FOV, and a correct decision
    # ("move forward at ~1.0 m/s and slightly left to align with the vending machine").
    # It simply never converted the decision into coordinates, because nothing made it.
    #
    # So make it. Close the block ourselves and re-enter with the same `<answer>(` prefill
    # the no-think path relies on, then decode only the answer. The model is not asked to
    # decide anything at this point -- it has already decided, in text that is now in its
    # own context -- it is asked to write down what it just concluded.
    #
    # Why this is not the same as raising the cap: the failure is not that the budget is
    # too small, it is that the model does not stop on its own. A bigger budget buys longer
    # deliberation and the same missing answer, at 54 s a call instead of 8 s.
    if ANSWER_PREFIX in gen:
        return prefix + gen                        # it closed by itself; nothing to force
    closed = text + prefix + gen + "</think>\n" + ANSWER_PREFIX
    inputs = proc(text=[closed], images=imgs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=_answer_tokens(), **_SAMPLING)
    answer = proc.batch_decode(out[:, inputs["input_ids"].shape[-1]:],
                               skip_special_tokens=True)[0]
    return prefix + gen + "</think>\n" + ANSWER_PREFIX + answer


def free_generate(image_paths: list[str], system: str, user: str, max_new: int,
                  prefill: str = "", think: bool = False) -> tuple[str, float]:
    """One generation from an arbitrary system/user prompt. Returns (text, seconds).

    Shared by `/raw` and by the `menu` format, which is the reason it is a function rather
    than the body of the endpoint: the two must run the model identically, or the numbers
    the probes recorded through `/raw` would not describe what the benchmark then drives.
    """
    from PIL import Image

    proc, model = _state["proc"], _state["model"]
    content: list[dict] = [{"type": "image", "image": p} for p in image_paths]
    content.append({"type": "text", "text": user})
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content}]
    kwargs = {} if think else {"enable_thinking": False}
    try:
        text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False, **kwargs)
    except TypeError:
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    t0 = time.perf_counter()
    with _raw_gate:
        inputs = proc(text=[text + prefill], images=imgs or None,
                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, **_SAMPLING)
        gen = proc.batch_decode(out[:, inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True)[0]
    return prefill + gen, time.perf_counter() - t0


_MENU_ARCS = make_arcs()
_STOP_LABEL = len(_MENU_ARCS) + 1
_MENU_SYSTEM = select_system(_STOP_LABEL)
# Same menu, same labels, minus the one answer that cannot be right immediately after a
# reverse. Built once at import next to its sibling rather than per call, so the two can
# only ever differ in the way `select_system` says they differ.
_MENU_SYSTEM_MOVE = select_system(_STOP_LABEL, stop_allowed=False)
# Seeded, so a benchmark run is reproducible, and shuffled per call rather than fixed,
# because a stable numbering is the one confound that makes this whole approach look like
# it works when it does not: number the arcs left to right and a model can answer "1" for
# left without ever looking at the image. Every measurement behind this format was taken
# under a shuffle and the policy has to run under one too.
_menu_rng = np.random.default_rng(int(os.environ.get("QVLA_MENU_SEED", "0")))


def menu_plan(image_paths: list[str], instruction: str,
              step: int | None,
              recovered: bool = False) -> tuple[np.ndarray | None, str]:
    """The `menu` format's whole pipeline: two generations, one curvature, one plan.

    Returns (plan or None, a reasoning string for the run log). None means the reply named
    no label on the menu, which the caller counts as a parse failure and answers by reusing
    the previous plan -- the same fallback every other format has.

    Only the newest frame is used. The other formats hand the model four frames at 3 s
    spacing because they have to infer their own motion; here the question is about where
    the floor is open right now, and the arcs are drawn on one image.
    """
    out_dir = _state["run_dir"] or MENU_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    newest = image_paths[-1]

    labels = (_menu_rng.permutation(len(_MENU_ARCS)) + 1).tolist()
    menu = str(out_dir / f"menu_{step if step is not None else 0:08d}.jpg")
    render_menu(newest, menu, _MENU_ARCS, labels)
    by_label = {int(lab): _MENU_ARCS[i].kappa for i, lab in enumerate(labels)}

    # Call 1 is the whole reason this variant works. Asked to choose in one call, the model
    # drove into the wall 78% of the time; asked to describe first, 28%; asked to describe
    # AND name the open side, 0%. The middle number is the informative one -- a description
    # that says "a wall about 1 m away" reports that the way is blocked and never says
    # which way is open, so the second call turns and guesses where.
    # Call 1 also carries the DISTANCE channel now, which is why the instruction reaches it.
    # The menu cannot supply one: its shortest arc is 3 m and the plan runs 7 m, so with the
    # target 3 m off EVERY label overshoots it and the model's only options are "drive 7 m"
    # or "stop dead". The first ladder failed exactly there -- a clean approach closing
    # 70-90% of the gap, then milling at 4-7 m for the rest of the episode.
    #
    # Say plainly what this costs: DESCRIBE_SIDED's numbers (100% / 92% / 90% / 83%, and
    # 6/6 on naming obstacles with a distance) were measured WITHOUT this sentence and
    # without the instruction in front of the model. The two measured sentences are asked
    # in the same words, but the question they sit in is no longer the question that was
    # scored, and the arc choices below could move. The re-run is the check.
    # Order matters and the reason is in `DESCRIBE_AFTER_RECOVERY`: the recovery note goes
    # between the two questions, so the TARGET line stays the last thing asked for.
    describe = (DESCRIBE_SIDED
                + (DESCRIBE_AFTER_RECOVERY if recovered else "")
                + DESCRIBE_TARGET.format(instruction=instruction))
    # 110 tokens was sized for two sentences plus the TARGET line. The recovery note asks
    # for no extra output -- it changes what the model looks at, not what it writes -- so
    # the budget is deliberately not raised with it.
    seen, _ = free_generate([newest], FREE_SPACE_SYSTEM, describe, 110)
    seen = seen.strip().replace("\n", " ")
    target_m = parse_target_distance(seen)
    speed = speed_for(target_m, MENU_SPEED_MPS)

    user = (f"What the robot can see ahead of it: {seen}\n\n"
            + (f"{SELECT_AFTER_RECOVERY}\n\n" if recovered else "")
            + f"Navigation instruction: {instruction}\n\n"
            + f"Which numbered path do you take?")
    reply, _ = free_generate(
        [menu], _MENU_SYSTEM_MOVE if recovered else _MENU_SYSTEM, user, 8)

    m = re.search(r"\d+", reply)
    choice = int(m.group()) if m else None
    note = (("[after reverse] " if recovered else "")
            + f"[free space] {seen}\n[menu] {labels} -> {reply.strip()[:20]!r}")
    if recovered and choice == _STOP_LABEL:
        # Answered the one label the system prompt just took away. Honoured anyway, and
        # flagged instead: the alternative is to overrule the model with a direction we
        # invented, and the off-menu branch below would reuse the previous plan, which on
        # this path IS the stop. So both roads end here; only one of them says so in the
        # log. If this line shows up often the prompt is not landing and that is the
        # finding, not something to paper over in the parser.
        note = note + " [STOP despite the no-stop prompt]"

    if choice == _STOP_LABEL:
        # Cumulative positions all at the origin. `_lookahead_point` returns a distance
        # below `_EPS` for this and every controller answers Command(0, 0, 0) -- the same
        # path a "Stop. Do not move." plan already takes, not a special case bolted on.
        plan, kappa, note = np.zeros((N_WAYPOINTS, 2)), None, note + " -> STOP"
    elif choice not in by_label:
        plan, kappa, note = None, None, note + " -> unparsed"
    else:
        kappa = by_label[choice]
        # The distance only scales the plan; the SHAPE is still the label the model chose.
        # Steering stays a choice off the drawn menu -- the thing that was measured -- and
        # the estimate, which was not, can do nothing worse than pick the wrong speed.
        plan = plan_from_kappa(kappa, speed, N_WAYPOINTS, DT)
        note = note + (f" -> kappa {kappa:+.2f} @{speed:.2f}m/s"
                       + (f" (target {target_m:.1f} m)" if target_m is not None else ""))

    _record(step, menu, instruction, labels, choice, kappa, seen, reply.strip(),
            target_m, speed, recovered)
    return plan, note


def _record(step: int | None, menu: str, instruction: str, labels: list[int],
            choice: int | None, kappa: float | None, seen: str, reply: str,
            target_m: float | None = None, speed: float | None = None,
            recovered: bool = False) -> None:
    """Append one decision to the run's JSONL, next to the menu image it was made on.

    Everything needed to redraw the decision later and nothing that has to be recomputed
    to do it: the shuffle is stored, so the video can show which curve carried which
    number without re-deriving a permutation that was random. Best-effort by design --
    losing a video frame is not worth failing a benchmark episode over, so any error here
    is swallowed after being printed.
    """
    run_dir = _state["run_dir"]
    if not RECORD or run_dir is None:
        return
    try:
        import json
        with open(run_dir / "decisions.jsonl", "a") as fh:
            fh.write(json.dumps({
                "step": step, "menu": Path(menu).name, "instruction": instruction,
                "labels": labels, "choice": choice, "kappa": kappa,
                "stop": choice == _STOP_LABEL, "free_space": seen, "reply": reply,
                # Kept apart on purpose: `target_m` is what the model SAID and `speed` is
                # what we DID with it. Storing only the speed would make a bad estimate and
                # a bad mapping look identical afterwards, and they have different fixes.
                # `speed` defaults to None and is resolved HERE, not in the signature: a
                # default argument is bound once at def time, so `= MENU_SPEED_MPS` would
                # have frozen the launch value and gone on recording it after /menu_speed
                # changed the real one -- a wrong number in the record with nothing wrong
                # in the driving, which is the hardest kind to notice later.
                "target_m": target_m,
                "speed_mps": MENU_SPEED_MPS if speed is None else speed,
                # Which prompt this decision was actually taken under. Without it the
                # recovery decisions are indistinguishable from the rest in the record,
                # and they are the ones worth reading -- they are taken under a menu with
                # no STOP on it, so counting stops across a run would otherwise mix two
                # different question sets and report the average of them.
                "recovered": recovered,
            }) + "\n")
    except Exception as exc:
        print(f"[qvla-server] could not record decision: {exc}", flush=True)


def _generation_worker(messages: list[dict], image_paths: list[str],
                       step: int | None, max_new: int, instruction: str = "",
                       recovered: bool = False, epoch: int = 0) -> None:
    """Runs on a background thread so the control loop never waits on decode.

    This is the same two-rate idea as TIC-VLA's `predict_async`, one level up: there the
    fast half was the action expert, here it is simply "keep driving the plan you have".
    A 3 s plan against a ~2-3 s generation leaves the loop a plan at all times.

    `menu` spends two generations instead of one and lands at ~2.1 s, which is inside that
    budget only because the horizon was already raised to 10 s. It is also why `menu` is
    given the raw instruction rather than `messages`: it builds its own prompts around a
    rendered image that does not exist until this thread runs.
    """
    try:
        if _ARC_FORMAT == "menu":
            plan, text = menu_plan(image_paths, instruction, step, recovered)
        else:
            text = _generate(messages, image_paths, max_new)
            ctrl = parse_arc(text) if _ARC_FORMAT == "arc" else parse_control_points(text)
            plan = None if ctrl is None else densify(ctrl)
        with _lock:
            _state["generations"] += 1
            _state["reasoning"] = text
            if _state["epoch"] != epoch:
                # The plan was thrown away while this was decoding -- a new episode, or a
                # reverse manoeuvre that made the view this was built on wrong. Dropping
                # it is the point of the discard; installing it now would be the same
                # stale plan arriving two seconds late, which is worse than none at all
                # because the caller cannot tell it apart from a fresh one.
                # NOT counted as a parse failure: nothing failed, and that counter is read
                # as "how much of this run scored the fallback instead of the model".
                _state["stale_discards"] += 1
            elif plan is None:
                _state["parse_failures"] += 1
            else:
                _state["plan"] = plan
                _state["plan_gen_step"] = step
    except Exception as exc:                       # never let a worker kill the server
        # Counted apart from parse_failures on purpose. Those two look identical from the
        # outside -- no plan came back -- and have nothing in common: a parse failure is
        # the model writing something unusable, this is the stack being broken. Merging
        # them once read as "the prompt is bad" when the real answer was HF_HUB_OFFLINE=1
        # blocking an FP8 kernel download at the first forward pass.
        # /health carries one line, which is right for a status endpoint and useless for a
        # diagnosis -- a bare `KeyError: 'weight_packed'` says nothing about which layer or
        # which call path raised it. Print the traceback to stderr as well: the server still
        # does not die, and the log has the one thing the summary cannot carry.
        traceback.print_exc()
        with _lock:
            _state["gen_errors"] += 1
            _state["reasoning"] = f"[generation failed] {type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _state["gen_thread"] = None


@app.get("/health")
def health() -> dict:
    with _lock:
        return {
            "ok": _state["model"] is not None,
            "model": MODEL_PATH,
            "mode": ("qvla-arc-menu (selects a drawn path)" if _ARC_FORMAT == "menu"
                     else "qvla-direct (no action expert)"),
            # Reported so a probe can never misattribute a result to the wrong format.
            # The three are not comparable: `arc` cannot express an S-curve at all, and
            # `menu` expresses speed only as a scalar the describe call estimates -- the
            # drawn choice is pure geometry, so there is no braking WITHIN a menu plan.
            "format": _ARC_FORMAT,
            **({"menu_speed_mps": MENU_SPEED_MPS, "menu_creep_mps": CREEP_MPS,
                "stop_label": _STOP_LABEL} if _ARC_FORMAT == "menu" else {}),
            "max_pixels": MAX_PIXELS,
            "predictions": _state["predictions"],
            "generations": _state["generations"],
            # Worth reading after every run. A high failure rate means the benchmark
            # scored the fallback plan, not the model.
            "parse_failures": _state["parse_failures"],
            # Not the same thing: this one means the stack broke, not the model.
            "gen_errors": _state["gen_errors"],
            # Plans that finished decoding after a /replan or /reset threw their view
            # away. A few per run is the recovery working; a lot means the loop is
            # discarding faster than the model can generate and is driving on nothing.
            "stale_discards": _state["stale_discards"],
            "empty_plans": _state["empty_plans"],
            "last_reasoning": _state["reasoning"],
            "has_plan": _state["plan"] is not None,
        }


class SpeedRequest(BaseModel):
    """Cruise speed, in m/s, for menu plans where the target is not in sight."""

    cruise_mps: float


@app.post("/menu_speed")
def menu_speed(req: SpeedRequest) -> dict:
    """Retune the cruise speed without reloading 28.75 GiB of weights.

    Speed is the one menu parameter worth changing while watching a run, because what it
    really sets is how far the robot travels BLIND: a generation takes about 3 s, so at
    0.70 m/s it covers 2.1 m of a 3.0 m arc before it can revise anything, and the whole
    overshoot failure lives in that gap. Trying a value is a 30-second question and a
    server restart is a 35-second answer to it plus a cold first generation.

    Every decision already records the speed it used, so a run whose speed changed halfway
    is still readable afterwards -- but it is not one measurement, and `summarize_runs.py`
    cannot know that. Change it between episodes, not during a scored one.

    The floor is not settable. `CREEP_MPS` is the bottom of the approach ramp and is
    already under what the tightest episode's timeout allows; making it adjustable from
    here would let a stall be dialled in by accident and read as a policy failure.
    """
    global MENU_SPEED_MPS
    v = float(req.cruise_mps)
    if not CREEP_MPS <= v <= 1.5:
        raise HTTPException(
            status_code=400,
            detail=f"cruise must be between the creep floor {CREEP_MPS} and the episodes' "
                   f"own 1.5 m/s cap; got {v}")
    was, MENU_SPEED_MPS = MENU_SPEED_MPS, v
    print(f"[qvla-server] menu cruise {was:.2f} -> {v:.2f} m/s "
          f"({v * 3.0:.1f} m travelled per ~3 s generation)", flush=True)
    return {"ok": True, "was": was, "cruise_mps": v,
            "metres_per_generation": round(v * 3.0, 2)}


class ResetRequest(BaseModel):
    """`run` names the episode about to start. Optional, and every field defaults, so the
    older callers that post a bare `{}` keep working unchanged."""

    run: str = ""


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> dict:
    run_dir = None
    if RECORD and req.run:
        from datetime import datetime
        # Timestamped, so re-running an episode adds a recording rather than silently
        # overwriting the one you were about to compare against.
        run_dir = MENU_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{req.run}"
        run_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        _state.update(plan=None, plan_gen_step=None, reasoning=None, gen_step=None,
                      run_dir=run_dir, epoch=_state["epoch"] + 1)
    return {"ok": True, "run_dir": str(run_dir) if run_dir else None}


@app.post("/replan")
def replan() -> dict:
    """Drop the cached plan and nothing else. Mid-episode; `/reset` is between episodes.

    They look interchangeable and are not, which is the reason this exists as its own
    route rather than as a bare `POST /reset` from the runner. `/reset` also rebuilds
    `run_dir`, and with no `run` name that means setting it to None -- so calling it in
    the middle of an episode would silently stop recording the menus and decisions for
    the rest of the run, and the trace would be missing exactly the decisions worth
    reading. Same call, same three lines of shared effect, one field of difference that
    costs the evidence.

    Used by `stuck_recovery.py` the instant the robot finishes reversing out of a wedge.
    The cached plan at that moment was generated on the view from INSIDE the wedge and
    almost always says STOP; re-serving it hands the robot straight back into the thing
    it just backed out of. Bumping the epoch also invalidates any generation already in
    flight, which is what makes this different from waiting for the next one -- that
    generation was started on the old view too.
    """
    with _lock:
        _state.update(plan=None, plan_gen_step=None, epoch=_state["epoch"] + 1)
        return {"ok": True, "epoch": _state["epoch"]}


class RawRequest(BaseModel):
    """Free-form generation against the loaded model. A probe endpoint, not a policy one.

    It exists because the interesting question stopped being "what waypoints does the
    model write" -- that channel is measured and dead -- and became "what can this model
    answer about a picture". Asking that through `/predict` is impossible: `/predict` owns
    the prompt, forces `<answer>(` as a prefill and runs the reply through a waypoint
    parser. `/raw` hands the prompt over and returns the text.

    Loading a second copy to avoid touching the server is not an option: the 27B FP8
    weights hold 31.7 of GPU1's 32.6 GiB.
    """

    image_paths: list[str] = Field(default_factory=list)
    system: str = ""
    user: str
    prefill: str = ""
    max_new_tokens: int = 64
    think: bool = False


@app.post("/raw")
def raw(req: RawRequest) -> dict:
    """Generate freely. Refuses rather than racing a benchmark for the same GPU."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    missing = [p for p in req.image_paths if not Path(p).is_file()]
    if missing:
        raise HTTPException(status_code=400, detail=f"image paths not found: {missing[:3]}")
    # `_lock` guards `_state`, not the GPU -- `_generation_worker` decodes outside it. Two
    # concurrent `model.generate` calls on one FP8 model is not a race worth debugging
    # later from a weird number, so decline while a plan is being generated.
    with _lock:
        if _state["gen_thread"] is not None:
            raise HTTPException(status_code=409, detail="a plan generation is in flight")

    text, dt = free_generate(req.image_paths, req.system, req.user, req.max_new_tokens,
                             prefill=req.prefill, think=req.think)
    return {"text": text, "latency_s": round(dt, 3)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    missing = [p for p in req.image_paths if not Path(p).is_file()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"image paths not found (is the scratch dir shared?): {missing[:3]}")

    t0 = time.perf_counter()
    think = _state["think"]
    # With thinking on this is the budget for the THINKING pass only; `_generate` then
    # spends `_answer_tokens()` more on the forced answer. 300 is where the reasoning is
    # still going somewhere useful -- the scene description and the steering decision are
    # both settled well inside it, and what comes after is the model relitigating its own
    # distance estimate for another 700 tokens without changing the answer.
    max_new = int(os.environ.get(
        "QVLA_MAX_NEW_TOKENS", 300 if think else _answer_tokens()))
    # `menu` writes its own prompts on a thread, around an image that has not been rendered
    # yet, so there is nothing to build here and building it anyway would only bill the
    # tokeniser for a prompt nobody sends.
    messages = [] if _ARC_FORMAT == "menu" else build_messages(req, req.image_paths, think)

    started_step = None
    with _lock:
        busy = _state["gen_thread"] is not None
        have_plan = _state["plan"] is not None

    if not busy:
        with _lock:
            epoch = _state["epoch"]
        th = threading.Thread(
            target=_generation_worker,
            args=(messages, list(req.image_paths), req.current_step, max_new,
                  req.instruction, req.recovered, epoch), daemon=True)
        with _lock:
            _state["gen_thread"] = th
        started_step = req.current_step
        th.start()
        if not have_plan:
            # The very first call has nothing to be stale: acting before the model has
            # ever looked at the scene would be driving on nothing. Blocking here is
            # correct and unavoidable, and it is exactly what server.py does.
            th.join(timeout=GEN_TIMEOUT_S)

    with _lock:
        plan = _state["plan"]
        reasoning = _state["reasoning"]
        _state["predictions"] += 1

    if plan is None:
        # Still nothing after a blocking wait. Standing still is the only honest answer;
        # a fabricated straight line would score as navigation.
        with _lock:
            _state["empty_plans"] += 1
        wp = [[0.0, 0.0]] * N_WAYPOINTS
    else:
        dx, dy = (req.robot_state + [0.0] * 6)[4:6]
        full = reframe(plan, float(dx), float(dy), float(req.yaw_delta_rad), 0.0)
        live = full[int(np.ceil(max(0.0, float(req.time_delay)) / DT)):]
        if len(live) == 0:
            # The plan has been fully consumed and the next one is not ready. Hold its
            # last point -- in the CURRENT frame, so it is still a real place -- rather
            # than inventing more of a plan the model never wrote.
            live = full[-1:]
        wp = live.astype(float).tolist()

    return PredictResponse(
        waypoints=wp,
        reasoning=reasoning,
        num_waypoints=len(wp),
        latency_s=round(time.perf_counter() - t0, 3),
        kv_cache_available=plan is not None,
        vlm_generation_start_step=started_step,
    )


@app.on_event("startup")
def _startup() -> None:
    _load_model()


if __name__ == "__main__":
    import argparse
    import uvicorn

    # argparse rather than reading the env directly, because this server is normally run
    # ALONGSIDE server.py and the whole point of the second process is a second port.
    # Silently ignoring an unrecognised --port meant loading 30 GB of weights for three
    # minutes and then failing to bind on top of the TIC-VLA server. An unknown argument
    # must be an error here, not a shrug.
    ap = argparse.ArgumentParser(description="Q-VLA-direct policy server")
    ap.add_argument("--host", default=os.environ.get("NAV_POLICY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("NAV_POLICY_PORT", "8765")))
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
