"""A candidate VLM behind the same `/raw` endpoint `server_qwen.py` serves.

The point of this file is that the arc-menu policy does not need a Qwen. It needs one
function -- "here is a system prompt, a user prompt and an image; give me text back" --
and every probe in `nav/tools/` already reaches that function over HTTP. So a swap
experiment does not have to touch the policy, the prompts, the menu renderer or the
scoring: it has to serve `/raw` with the same request and response schema on a different
port, and then the probes are the benchmark, run unmodified.

What that buys is the thing an A/B usually fails to control. `probe_arc_selection.py` and
`probe_arc_repair.py` own the prompts, render the menus themselves from the same frames
under the same seed, and shuffle the labels the same way. Point them at 8766 and they
measure Qwen3.8-27B-FP8; point them at 8767 and they measure the candidate. The only
thing that differs between the two numbers is the model.

Three parity details are load-bearing and are NOT the probes' business, so they live here:

  PIXELS   `QVLA_MAX_PIXELS` defaults to 200704 (448x448) in `server_qwen.py`, and a
           candidate left at its own default would see a different picture. A survey that
           varies resolution alongside the model is measuring neither. Same cap here,
           applied through whichever knob the processor exposes.
  DECODING greedy, `do_sample=False`, matching `_SAMPLING` with `QVLA_TEMPERATURE` unset.
           The token budget comes from the probe, so it is already shared.
  DEVICE   one card, chosen by `CUDA_VISIBLE_DEVICES`, because a model split across both
           GPUs runs different kernels than the same model on one -- measured on the FP8
           baseline, where spanning devices routes linear layers off DeepGEMM. That makes
           a two-card timing uncomparable to a one-card timing, and every number this
           server produces is meant to be compared. Setting the variable is safe here for
           exactly the reason it is safe in `server_qwen.py`: this process never starts
           Kit.

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/gtu-dsa/envs/qvla/bin/python \\
        nav/policy_server/server_vlm.py --candidate qwen3vl-4b --port 8767
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

# Same default as `server_qwen.py:97`. Named here rather than imported because importing
# that module loads 30 GB of Qwen-specific machinery to read one integer.
MAX_PIXELS = int(os.environ.get("QVLA_MAX_PIXELS", 200704))

_state: dict = {"model": None, "proc": None, "name": None, "shape": None, "calls": 0}
_lock = threading.Lock()

app = FastAPI(title="nav candidate VLM server")


# --------------------------------------------------------------------------- loading

def load(model_dir: str, name: str) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.perf_counter()
    # `max_pixels` is Qwen's knob and `size` is everyone else's, and a processor that does
    # not take one raises TypeError rather than ignoring it -- which is the good outcome,
    # because a silently ignored resolution cap is the failure this whole block exists to
    # avoid. Try the specific knob, fall back to the generic one, and SAY which landed.
    proc, knob = None, "none"
    for kwargs, label in (({"max_pixels": MAX_PIXELS}, "max_pixels"),
                          ({"size": {"longest_edge": int(MAX_PIXELS ** 0.5)}},
                           "size.longest_edge"),
                          ({}, "none")):
        try:
            proc = AutoProcessor.from_pretrained(model_dir, **kwargs)
            knob = label
            break
        except (TypeError, ValueError):
            continue
    if proc is None:
        raise RuntimeError(f"no AutoProcessor accepted {model_dir}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.memory_allocated() / 1024 ** 3
    _state.update(model=model, proc=proc, name=name, pixel_knob=knob,
                  params=n_params, weights_gib=round(mem, 2),
                  load_s=round(time.perf_counter() - t0, 1))
    print(f"[{name}] {n_params / 1e9:.2f} B params, {mem:.2f} GiB on card, "
          f"pixel cap via {knob}, ready in {_state['load_s']} s", flush=True)


# ------------------------------------------------------------------- prompt plumbing

def _messages(image_paths: list[str], system: str, user: str, sys_in_user: bool) -> list:
    """Build the chat messages. `sys_in_user` folds the system prompt into the user turn.

    Not every one of these five templates accepts a `system` role -- SmolVLM's raises, and
    a survey that quietly dropped the system prompt for one candidate would be comparing a
    model that was told the camera geometry against models that were not. So the fallback
    keeps the text and moves it, rather than losing it, and `/health` reports which form
    was used so the difference is visible next to the score instead of hidden in it.
    """
    content = [{"type": "image", "image": p} for p in image_paths]
    text = f"{system}\n\n{user}" if (sys_in_user and system) else user
    content.append({"type": "text", "text": text})
    msgs = []
    if system and not sys_in_user:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append({"role": "user", "content": content})
    return msgs


def _template(image_paths: list[str], system: str, user: str) -> str:
    """Template the prompt, discovering once whether this model takes a system role."""
    proc = _state["proc"]
    forms = [False, True] if _state.get("sys_in_user") is None else [_state["sys_in_user"]]
    last: Exception | None = None
    for sys_in_user in forms:
        try:
            out = proc.apply_chat_template(
                _messages(image_paths, system, user, sys_in_user),
                add_generation_prompt=True, tokenize=False)
            if _state.get("sys_in_user") is None:
                _state["sys_in_user"] = sys_in_user
                if sys_in_user:
                    print("[note] template rejected a system role; folded into the user "
                          "turn", flush=True)
            return out
        except Exception as e:  # noqa: BLE001 -- templates raise many different types
            last = e
    raise RuntimeError(f"no message form templated cleanly: {last}")


def free_generate(image_paths: list[str], system: str, user: str, max_new: int,
                  prefill: str = "") -> tuple[str, float]:
    """One generation. Returns (new text, seconds) -- the same contract as `server_qwen`."""
    from PIL import Image

    proc, model = _state["proc"], _state["model"]
    t0 = time.perf_counter()
    prompt = _template(image_paths, system, user) + prefill
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = proc(text=[prompt], images=imgs or None, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = proc.batch_decode(out[:, inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True)[0]
    return prefill + gen, time.perf_counter() - t0


# ------------------------------------------------------------------------- endpoints

class RawRequest(BaseModel):
    """Field-for-field `server_qwen.RawRequest`, minus the channels no probe uses.

    `images_b64` and `think` are absent on purpose rather than accepted and ignored: a
    request carrying either would be answered differently by the two servers, and a probe
    that silently got a different answer from one of them is the kind of bug this repo has
    already paid for. Pydantic rejects unknown fields loudly instead.
    """

    image_paths: list[str] = Field(default_factory=list)
    system: str = ""
    user: str
    prefill: str = ""
    max_new_tokens: int = 64


@app.get("/health")
def health() -> dict:
    return {
        "ok": _state["model"] is not None,
        "candidate": _state["name"],
        "params_b": round(_state.get("params", 0) / 1e9, 2),
        "weights_gib": _state.get("weights_gib"),
        "load_s": _state.get("load_s"),
        "pixel_cap": MAX_PIXELS,
        "pixel_knob": _state.get("pixel_knob"),
        "system_role_supported": (None if _state.get("sys_in_user") is None
                                  else not _state["sys_in_user"]),
        "calls": _state["calls"],
    }


@app.post("/reset")
def reset() -> dict:
    """A no-op that exists so a probe written against `server_qwen` runs here unchanged.

    On that server `/reset` clears the cached plan, and the rule in CLAUDE.md is that
    every probe must call it or it measures one generation N times. There is no plan cache
    here -- `/raw` always generates -- so there is nothing to clear, and the honest thing
    is to answer rather than 404 and make the caller special-case which server it is
    talking to.
    """
    return {"ok": True, "run_dir": None}


@app.post("/raw")
def raw(req: RawRequest) -> dict:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    missing = [p for p in req.image_paths if not Path(p).is_file()]
    if missing:
        raise HTTPException(status_code=400, detail=f"image paths not found: {missing[:3]}")
    # One generation at a time, for the reason `server_qwen` declines a concurrent one:
    # two `model.generate` calls on one card produce a timing that describes neither.
    with _lock:
        text, dt = free_generate(req.image_paths, req.system, req.user,
                                 req.max_new_tokens, prefill=req.prefill)
        _state["calls"] += 1
    return {"text": text, "latency_s": round(dt, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="short name under paths.model_root, e.g. qwen3vl-4b")
    ap.add_argument("--model-dir", default=None, help="override the resolved path")
    ap.add_argument("--host", default="127.0.0.1")
    # NOT 8766. Sharing the baseline's port would either fail to bind or, worse, answer
    # after the baseline died and let a candidate's numbers be filed under Qwen.
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args()

    model_dir = a.model_dir or str(paths.model_root() / a.candidate)
    if not Path(model_dir).is_dir():
        print(f"no such model directory: {model_dir}", file=sys.stderr)
        return 2
    load(model_dir, a.candidate)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
