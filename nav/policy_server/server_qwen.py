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
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.interpolate import PchipInterpolator

MODEL_PATH = os.environ.get(
    "QVLA_MODEL", "/home/gtu-dsa/robotics/models/Qwen3.8-27B-NVFP4"
)
# 200704 px = 448x448. On this vision tower a frame costs exactly
# pixels / (patch_size * spatial_merge)^2 = pixels / 1024 tokens, so this is 196
# tokens per frame -- the same budget InternVL3-1B feeds today. Native 1920x1080 would
# be 2059 tokens per frame and 4.1 s more prefill for no measured benefit.
MAX_PIXELS = int(os.environ.get("QVLA_MAX_PIXELS", 200704))
# Wall-clock cap on one generation. Past this the plan is older than the horizon it
# describes and driving on it is driving on fiction.
GEN_TIMEOUT_S = float(os.environ.get("QVLA_GEN_TIMEOUT_S", 20.0))

HORIZON_S = 3.0                       # the action head's own horizon: 30 pts at 10 Hz
N_WAYPOINTS = 30
CTRL_TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)     # what the model is asked to write
DT = HORIZON_S / N_WAYPOINTS

app = FastAPI(title="Q-VLA direct policy server")

_state: dict = {
    "model": None, "proc": None, "predictions": 0, "generations": 0,
    "parse_failures": 0, "empty_plans": 0, "gen_errors": 0,
    # The plan currently in force, in the body frame of the moment generation STARTED.
    "plan": None, "plan_gen_step": None, "reasoning": None,
    "gen_thread": None, "gen_step": None,
    "think": os.environ.get("QVLA_THINK", "0") == "1",
}
_lock = threading.Lock()


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

Scale: use the waypoint list above as your ruler. Those numbers are how far you actually moved, in metres, over the last seconds. Your speed is normally about 0.7 m/s and never above 1.5 m/s, so over the next 3 seconds you will cover roughly 2 metres unless you are slowing down or stopping.

Predict where you should be at {times} seconds from now, as {n} waypoints (x, y):
- x is metres FORWARD, positive; y is metres {pos_side}, positive.
- Each waypoint is the CUMULATIVE offset from where you are right now, not from the previous waypoint. So x must increase along the list while you are still moving forward.
- Steer by making y negative to go {neg_side} and positive to go {pos_side}. A gentle course correction is a few centimetres of y; a real turn is tens of centimetres.
- Do not drive into walls, furniture, people or shelving. Steer around obstacles and leave clearance.
- If you have arrived at the target the instruction names, STOP: return waypoints that barely change, for example (0.05, 0.00) repeated. Stopping at the right place is part of the task, not the end of it. Do not drive past the target.

Reply with the waypoints only, inside answer tags, and nothing else:
<answer>(x, y), (x, y), (x, y), (x, y), (x, y), (x, y)</answer>
"""

_THINK_TASK = (
    "\nBefore the answer, give at most two short sentences of reasoning inside "
    "<think></think> tags: what you see, and where you are going.\n"
)

# Written into the assistant's turn before generation starts, so the first token the model
# is free to choose is already inside a coordinate. See `_generate` for why the polite
# version of this instruction is not enough. It is prepended back onto the decoded text
# before parsing -- without that the leading "(" is missing and the first pair is lost.
ANSWER_PREFIX = "<answer>("


def build_messages(req: PredictRequest, image_paths: list[str], think: bool) -> list[dict]:
    prev = req.previous_waypoints_text.strip()
    if not prev:
        # DynaNav raises on an empty string and it is right to: "no history yet" has its
        # own wording in the training data, and the empty string is not it. See
        # nav/README.md -- defaulting this away is what made every plan a fresh straight
        # line back when the TIC-VLA server did it.
        prev = "From 0.0s to current timestamp time is 0.0s. No waypoints available."

    task = _TASK.format(
        cam_h=0.346, cam_fov=90.1, half_fov=45.0,
        times=", ".join(f"{t:.1f}" for t in CTRL_TIMES), n=len(CTRL_TIMES),
        pos_side="RIGHT" if FLIP_Y else "LEFT",
        neg_side="left" if FLIP_Y else "right",
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
_NUM = r"-?\d+(?:\.\d+)?"
# The third component is optional and discarded. TIC-VLA's training format is
# (x, y, theta) and theta there is literally atan2(y, x) of the same pair -- redundant --
# so a model that emits one out of habit is not wrong, just verbose. Rejecting those
# would throw away a perfectly good plan over a number that carries no information.
_PAIR = re.compile(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*(?:,\s*{_NUM}\s*)?\)")


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
    # 3 s at the 1.5 m/s cap is 4.5 m. Anything past that is a hallucinated scale.
    if pts[-1, 0] > 6.0 or np.abs(pts[:, 1]).max() > 6.0:
        return None
    return pts


def densify(ctrl: np.ndarray) -> np.ndarray:
    """(K, 2) control points -> (30, 2) at 10 Hz.

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
        # Greedy. TIC-VLA samples at temperature 0.7, but sampling buys nothing when the
        # output is a list of coordinates, and nav/README.md already records that run-to-
        # run variance on this stack comes from async cache timing rather than sampling.
        # One less source of noise in a benchmark that is scored on 3 repeats.
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = proc.batch_decode(out[:, inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True)[0]
    return prefix + gen


def _generation_worker(messages: list[dict], image_paths: list[str],
                       step: int | None, max_new: int) -> None:
    """Runs on a background thread so the control loop never waits on decode.

    This is the same two-rate idea as TIC-VLA's `predict_async`, one level up: there the
    fast half was the action expert, here it is simply "keep driving the plan you have".
    A 3 s plan against a ~2-3 s generation leaves the loop a plan at all times.
    """
    try:
        text = _generate(messages, image_paths, max_new)
        ctrl = parse_control_points(text)
        with _lock:
            _state["generations"] += 1
            _state["reasoning"] = text
            if ctrl is None:
                _state["parse_failures"] += 1
            else:
                _state["plan"] = densify(ctrl)
                _state["plan_gen_step"] = step
    except Exception as exc:                       # never let a worker kill the server
        # Counted apart from parse_failures on purpose. Those two look identical from the
        # outside -- no plan came back -- and have nothing in common: a parse failure is
        # the model writing something unusable, this is the stack being broken. Merging
        # them once read as "the prompt is bad" when the real answer was HF_HUB_OFFLINE=1
        # blocking an FP8 kernel download at the first forward pass.
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
            "mode": "qvla-direct (no action expert)",
            "max_pixels": MAX_PIXELS,
            "predictions": _state["predictions"],
            "generations": _state["generations"],
            # Worth reading after every run. A high failure rate means the benchmark
            # scored the fallback plan, not the model.
            "parse_failures": _state["parse_failures"],
            # Not the same thing: this one means the stack broke, not the model.
            "gen_errors": _state["gen_errors"],
            "empty_plans": _state["empty_plans"],
            "last_reasoning": _state["reasoning"],
            "has_plan": _state["plan"] is not None,
        }


@app.post("/reset")
def reset() -> dict:
    with _lock:
        _state.update(plan=None, plan_gen_step=None, reasoning=None, gen_step=None)
    return {"ok": True}


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
    max_new = int(os.environ.get("QVLA_MAX_NEW_TOKENS", 220 if think else 110))
    messages = build_messages(req, req.image_paths, think)

    started_step = None
    with _lock:
        busy = _state["gen_thread"] is not None
        have_plan = _state["plan"] is not None

    if not busy:
        th = threading.Thread(
            target=_generation_worker,
            args=(messages, list(req.image_paths), req.current_step, max_new), daemon=True)
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
