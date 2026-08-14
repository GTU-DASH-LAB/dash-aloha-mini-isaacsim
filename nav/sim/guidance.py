"""The long-horizon channel TIC-VLA emits as text, and how to read it.

TIC-VLA has TWO output heads and they cover different amounts of time.

  action head  -- 30 waypoints, `action_horizon_steps: int = 30` in
                  ticvla/training/config.py, at 10 Hz. Exactly 3.0 s of motion,
                  and on this robot about 1.6 m of it. This is what the
                  controllers drive on.
  text head    -- the `<answer>` block, three triples. These are NOT (x, y, z)
                  position-and-height, which is what they look like. Read the
                  label off the training data (ticvla/data/vlm_data.py:508-528):

                      for idx in [29, 59, 89]:      # 3s, 6s, 9s at 10 Hz
                          x, y = float(offset[0]), float(offset[1])
                          theta = math.atan2(y, x + 1e-3)
                          guidance_waypoint_list.extend([x, y, theta])

                  So the triple is (x, y, THETA) and the three of them sit at
                  3 s, 6 s and 9 s. **Three times the action head's horizon.**

That difference is the entire reason this module exists. Measured on the
Aisle-05 start frame, one policy call, both heads decoded:

    source                     horizon   heading
    action head (driven on)      3.0 s    -6.5 deg
    guidance <answer>            3   s    -8.0 deg
    guidance <answer>            6   s   -18.9 deg
    guidance <answer>            9   s   -17.8 deg
    what the goal needed           --    -24.9 deg

The policy is not failing to see the turn. It reports a 19 deg right turn in
text while the channel we steer on has only committed 6.5 deg of it, because at
3 s the turn genuinely has barely started. Replanning every 3 s on a 3 s plan
therefore re-enters the same shallow arc forever: the robot never accumulates
the yaw that would make the next frame look different.

One thing to be clear about, because it is easy to over-read: **theta carries no
information that (x, y) does not.** It is literally `atan2(y, x)` of the same
pair -- decoded values agree to the rounding in the text (-18.9 vs -18.6 deg).
The win here is not a new steering quantity. It is the HORIZON. Anyone reaching
for "feed the model theta instead of dx/dy" is reaching for the wrong axis; the
action head is `action_dim=2` (ticvla/models/ticvla.py:258) and no theta exists
in it to feed.

**The sentinel.** vlm_data.py:519 initialises guidance to `torch.ones(9) * -100`
and only fills it when `len(future_offsets) > 89`, i.e. when at least 9 s of
future actually exists. So the model is trained to emit
`(-100.00, -100.00, -100.00)` triples meaning "no guidance available", and it
does, verbatim, in real runs. Treating -100 as a coordinate would command a
hard-left turn into a wall. `parse_guidance` returns None for it.
"""

from __future__ import annotations

import math
import re

import numpy as np

# The horizons the three triples correspond to, in seconds. Fixed by the
# training-time indices [29, 59, 89] at 10 Hz; not a tunable.
GUIDANCE_HORIZONS_S = (3.0, 6.0, 9.0)

# vlm_data.py fills the tensor with -100 when there is no 9 s future to describe.
# Compared loosely because the value arrives as text, having been through a float
# and a formatter.
_SENTINEL = -100.0
_SENTINEL_TOL = 1.0

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_guidance(reasoning: str | None) -> np.ndarray | None:
    """Decode the `<answer>` block into a (3, 3) array of (x, y, theta).

    Returns None when there is no usable guidance -- no `<answer>`, too few
    numbers, or the -100 "no guidance" sentinel. Callers must handle None by
    falling back to the action head; guidance is advisory, and a policy that
    declines to give it is behaving correctly.
    """
    if not reasoning:
        return None

    match = _ANSWER_RE.search(reasoning)
    if match is None:
        return None

    nums = [float(v) for v in _NUM_RE.findall(match.group(1))]
    if len(nums) < 9:
        return None

    triples = np.asarray(nums[:9], dtype=float).reshape(3, 3)

    # Any sentinel anywhere disqualifies the whole block. The tensor is filled
    # or not filled as a unit, so a partial hit means something else is wrong
    # and guessing which rows are real is not worth the failure mode.
    if np.any(np.abs(triples - _SENTINEL) < _SENTINEL_TOL):
        return None

    # Recompute theta rather than trusting the printed value. It is redundant by
    # construction, so disagreement means the text was mangled -- and a wrong
    # heading is the one error here that steers the robot rather than stopping it.
    triples[:, 2] = np.arctan2(triples[:, 1], triples[:, 0] + 1e-3)
    return triples


def guidance_heading(
    guidance: np.ndarray | None, horizon_s: float = 6.0
) -> float | None:
    """Body-frame heading, in radians, that the guidance asks for. None if absent.

    `horizon_s` picks which of the three triples to steer at, and 6 s is the
    default for a reason worth stating: 3 s duplicates the action head (it is the
    same instant, and measured -8.0 vs -6.5 deg it says nothing new), while 9 s
    is far enough out that on a 3 s replan cycle the robot re-plans twice before
    reaching it, so committing to it hands the policy less say over its own path,
    not more. 6 s is one replan ahead -- the furthest point still belonging to the
    maneuver currently being executed.
    """
    if guidance is None:
        return None

    idx = int(np.argmin([abs(h - horizon_s) for h in GUIDANCE_HORIZONS_S]))
    x, y = float(guidance[idx, 0]), float(guidance[idx, 1])
    if math.hypot(x, y) < 1e-3:
        # A guidance point sitting on the robot has no direction. This is what a
        # "stop" instruction produces, and atan2(0, 0) would invent one.
        return None
    return math.atan2(y, x)
