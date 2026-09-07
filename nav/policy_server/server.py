"""TIC-VLA inference server -- the "brain" half of the navigation stack.

Runs in ~/envs/tic-vla (Python 3.11, torch 2.8.0+cu128) on GPU1, completely separate
from the Isaac Sim 6.0.1 process that drives AlohaMini. See ../plan.md "finding 1" for
why this separation is forced rather than chosen; briefly:

  * TIC-VLA needs Python 3.11 (Isaac Sim 5.0's NumPy 1.x ABI), AlohaMini needs 3.12.
  * The DynaNav office scene alone leaves Isaac Sim holding 25.4 GiB of GPU0's
    31.35 GiB, so the checkpoint cannot share that card anyway.

Frames cross the process boundary as *file paths*, not bytes: both processes are on one
machine, and TICVLA.predict() already takes `image_paths: list[str]`. That keeps a
448x448 render out of JSON entirely.

The two-process split lands on the paper's own slow/fast seam almost exactly. TIC-VLA
is "Think-in-Control": a ~1 s VLM generation that produces a KV cache, and a millisecond
action expert that consumes whatever cache is currently on hand. `/predict` calls
`predict_async`, so the slow half runs on a background thread INSIDE this process and
the HTTP call only pays for the fast half. The sim never has to thread anything: it
issues a normal blocking request and gets an answer back quickly, while the VLM keeps
churning on GPU1 between calls.

Usage:
    nav/policy_server/launch.sh
    # then, from anywhere:
    curl localhost:8765/health
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# TIC-VLA's DynaNav modules use bare imports of their siblings (`from vision_utils
# import ...`, `from ticvla_vlm import ...`), so the DynaNav directory itself has to be
# on sys.path -- importing it as a package does not work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

DYNANAV_ROOT = str(paths.dynanav_root())
sys.path.insert(0, DYNANAV_ROOT)

from ticvla import TICVLA  # noqa: E402

BASE_MODEL_PATH = os.environ.get(
    "TICVLA_BASE_MODEL_PATH", str(paths.model_root() / "InternVL3-1B")
)
CHECKPOINT_PATH = os.environ.get(
    "TICVLA_CHECKPOINT_PATH", str(paths.model_root() / "TIC-VLA-model.ckpt")
)
# launch.sh pins CUDA_VISIBLE_DEVICES=1, so cuda:0 here is physically GPU1. That is
# safe *only* because this process never starts Kit -- see
# memory/gtu-workstation-gpu-asymmetry.md, which is emphatic that Isaac Sim itself must
# never be pinned this way.
DEVICE = os.environ.get("NAV_POLICY_DEVICE", "cuda:0")

app = FastAPI(title="TIC-VLA policy server")

_state: dict[str, Any] = {"model": None, "loaded_at": None, "predictions": 0}


class PredictRequest(BaseModel):
    """Mirrors TICVLA.predict()'s signature (DynaNav/ticvla.py:548)."""

    image_paths: list[str] = Field(..., min_length=1)
    instruction: str
    robot_state: list[float] = Field(
        default_factory=lambda: [0.0] * 6,
        description="5 or 6 floats. The 6-form is unpacked as [vx, vy, _, yaw_speed, dx, dy].",
    )
    current_step: int | None = None
    time_delay: float = 0.0
    previous_waypoints_text: str = ""
    delayed_image_paths: list[str] | None = None
    robot_type: str = "wheeled robot"


class PredictResponse(BaseModel):
    waypoints: list[list[float]]  # (T, 2) -- (dx, dy) in body frame, FLU
    reasoning: str | None
    num_waypoints: int
    latency_s: float
    kv_cache_available: bool
    # Set only on the call where a NEW background generation was kicked off; None on
    # every other call. The caller needs it because `time_delay`/`dx,dy` must be
    # measured against the generation that produced the cache currently IN USE, which
    # is the one before the one now running -- DynaNav's own wording is "second-to-last
    # inference start frame". The server cannot answer that on its own: the model
    # overwrites `_kv_cache_generation_step` the moment a new generation starts, so by
    # the time a cache is being consumed its own start step is gone. DynaNav keeps the
    # bookkeeping in the behaviour script for exactly this reason and so do we.
    vlm_generation_start_step: int | None


def _load_model() -> None:
    if _state["model"] is not None:
        return
    t0 = time.time()
    print(f"[policy-server] loading base VLM from {BASE_MODEL_PATH}", flush=True)
    model = TICVLA(model_path=BASE_MODEL_PATH, device=DEVICE)

    # Checkpoint layout matches DynaNav's own loader
    # (behavior/nova_carter_test_ticvla.py:417-420): a Lightning-style dict whose keys
    # are prefixed "model.".
    print(f"[policy-server] loading checkpoint {CHECKPOINT_PATH}", flush=True)
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")["state_dict"]
    state_dict = {k[len("model."):]: v for k, v in raw.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _state["model"] = model
    _state["loaded_at"] = time.time()
    print(f"[policy-server] ready in {time.time() - t0:.1f}s on {DEVICE}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    model = _state["model"]
    return {
        "ok": model is not None,
        "device": str(DEVICE),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "base_model": BASE_MODEL_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "num_action_chunks": getattr(model, "num_action_chunks", None),
        "predictions_served": _state["predictions"],
    }


@app.post("/reset")
def reset() -> dict[str, Any]:
    """Clear per-episode state (KV cache, waypoint history) between episodes."""
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    model.reset_episode_state()
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    missing = [p for p in req.image_paths if not Path(p).is_file()]
    if missing:
        # Worth being loud: the sim process writes these, so a missing path means the
        # two processes disagree about the scratch directory.
        raise HTTPException(
            status_code=400,
            detail=f"image paths not found (is the scratch dir shared?): {missing[:3]}",
        )

    t0 = time.perf_counter()
    with torch.no_grad():
        # `predict_async`, NOT `predict` -- this is the paper's headline mechanism and
        # the two entry points differ in where the ~1 s VLM generation happens.
        # `predict` (DynaNav/ticvla.py:548) runs it inline, so every call pays for a
        # 200-token autoregressive decode and the caller is stopped for the whole
        # thing. `predict_async` (:891) instead polls the previous generation, kicks a
        # new one onto a background thread if none is running (it SKIPS when one
        # already is, so calling every step is safe), and then runs only the fast
        # half: one 448px vision-encoder pass over the current frame plus the 3-layer
        # cross-attention action expert, against whatever KV cache is on hand. The
        # cache it uses is therefore STALE by roughly one generation -- that is the
        # design, not a defect, and `time_delay`/`dx,dy` in the request are how the
        # caller declares how stale so the action head can compensate.
        #
        # The very first call after a reset still blocks: there is no cache to be
        # stale, and acting before the model has ever looked at the scene would be
        # driving on nothing. Correct, and unavoidable.
        #
        # `current_robot_pose` is deliberately not passed. That argument only exists so
        # the model can stash a pose to compute dx/dy from later; we compute dx/dy on
        # the sim side, where the pose actually lives, and hand it in via `robot_state`.
        reasoning, waypoints, gen_step, kv_ok, _gen_pose = model.predict_async(
            image_paths=req.image_paths,
            instruction=req.instruction,
            robot_state=torch.tensor(req.robot_state, dtype=torch.float32),
            current_step=req.current_step,
            time_delay=req.time_delay,
            previous_waypoints_text=req.previous_waypoints_text,
            delayed_image_paths=req.delayed_image_paths,
            robot_type=req.robot_type,
        )
    latency = time.perf_counter() - t0

    # (1, T, 2) -> [[dx, dy], ...]
    wp = waypoints[0].float().cpu().numpy().tolist()
    _state["predictions"] += 1

    return PredictResponse(
        waypoints=wp,
        reasoning=reasoning,
        num_waypoints=len(wp),
        latency_s=round(latency, 3),
        kv_cache_available=bool(kv_ok),
        vlm_generation_start_step=int(gen_step) if gen_step is not None else None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("NAV_POLICY_HOST", "127.0.0.1"),
        port=int(os.environ.get("NAV_POLICY_PORT", "8765")),
        log_level="info",
    )
