"""Does a four-frame prefill fit TIC-VLA's 3.0 s plan horizon on a 27B VLM?

This is the gate that decides Q-VLA, and it is deliberately separate from
`check_vlm_swap.py --time-it`, which times a TEXT prompt and says so. Text timing answers
the wrong question. TIC-VLA is a video model: DynaNav passes **four** 1920x1080 frames at
3 s spacing every single call, and on a vision tower with `patch_size 16` /
`spatial_merge_size 2` those frames dominate the forward pass. Decode speed barely
matters by comparison, which is why the "a 27B is ~7x slower per token" estimate in
`qwen_swap_plan.md` was answering a question nobody asked.

What is measured, and why each half separately:

  PREFILL  -- time to the first token, i.e. encoding four frames and the prompt. This is
              the part that scales with image count and resolution, and the part a bigger
              VLM makes worse. It cannot be hidden by streaming.
  DECODE   -- the remaining `max_new_tokens - 1` tokens. TIC-VLA generates 200
              (`ticvla/models/ticvla.py:579`). Qwen3.8-27B is a hybrid with 48 of 64
              layers linear-attention, so this should be far cheaper than a dense 27B.

The budget is 3.0 s of SIM time -- 30 waypoints at 10 Hz is exactly the action head's
horizon. The sim runs at roughly 0.5x realtime, so the wall-clock allowance is about
double that, but only while the sim stays slow. `nav/README.md` records staleness already
at 1.5-2.0 s with InternVL3-1B.

Memory: the model is placed with an explicit `max_memory` cap rather than plain
`device_map="auto"`, because Isaac Sim may be holding GPU0 -- a DynaNav scene alone was
measured at 25.4 GiB. Check `nvidia-smi` and set the caps to what is actually free.
This never starts Kit, so it is safe to touch both GPUs; do NOT set CUDA_VISIBLE_DEVICES
for anything that does.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/probe_qvla_latency.py \\
        --model /path/to/Qwen3.8-27B-FP8 --gpu0-gib 18 --gpu1-gib 16
"""

from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
from pathlib import Path

# 30 waypoints at 10 Hz. A generation slower than this hands the controller a plan that
# expired before it was consumed.
ACTION_HORIZON_S = 3.0
# ticvla/models/ticvla.py:579 -- generation_config = dict(max_new_tokens=200, ...)
TICVLA_MAX_NEW_TOKENS = 200
# DynaNav samples [-9, -6, -3, 0] s, oldest first.
N_FRAMES = 4

INSTRUCTION = (
    "Go straight ahead and stop at the front of the vending machine at front left."
)


def pick_frames(frame_dir: str, n: int) -> list[Path]:
    """Four frames spaced like DynaNav's 3 s sampling, oldest first."""
    fs = sorted(Path(p) for p in glob.glob(f"{frame_dir}/*.jpg"))
    if len(fs) < n:
        print(f"ERROR: need {n} frames in {frame_dir}, found {len(fs)}", file=sys.stderr)
        raise SystemExit(2)
    # A replan is ~1 frame, and the run replans faster than once per 3 s, so a fixed
    # stride is only an approximation of the real spacing. It does not matter here --
    # the cost is set by frame COUNT and resolution, not by which frames they are.
    stride = max(1, len(fs) // 40)
    return fs[-1 - 3 * stride :: stride][-n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--max-new-tokens", type=int, default=TICVLA_MAX_NEW_TOKENS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu0-gib", type=float, default=18.0)
    ap.add_argument("--gpu1-gib", type=float, default=16.0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    frames = pick_frames(args.frame_dir, N_FRAMES)
    print(f"frames    : {', '.join(f.name for f in frames)}")
    imgs = [Image.open(f).convert("RGB") for f in frames]
    print(f"resolution: {imgs[0].size[0]}x{imgs[0].size[1]}")

    max_memory = {0: f"{args.gpu0_gib}GiB", 1: f"{args.gpu1_gib}GiB"}
    print(f"max_memory: {max_memory}\n")

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory)
    model.eval()
    print(f"load      : {time.time() - t0:.1f} s")
    print(f"placement : {sorted(set(str(d) for d in model.hf_device_map.values()))}\n")

    messages = [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in imgs]
                 + [{"type": "text", "text": INSTRUCTION}]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)

    n_tok = int(inputs["input_ids"].shape[-1])
    print(f"prompt    : {n_tok} tokens for {N_FRAMES} frames + instruction")
    print(f"            ~{n_tok // N_FRAMES} tokens per frame\n")

    def timed(max_new: int) -> float:
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        return time.time() - t

    print("warm-up...")
    timed(4)

    prefill = [timed(1) for _ in range(args.repeats)]
    full = [timed(args.max_new_tokens) for _ in range(args.repeats)]
    p, f = statistics.fmean(prefill), statistics.fmean(full)
    decode = f - p

    print()
    print("=" * 74)
    print(f"{'PREFILL (4 frames -> first token)':44}{p:8.2f} s   "
          f"({min(prefill):.2f}-{max(prefill):.2f})")
    print(f"{'DECODE  (' + str(args.max_new_tokens - 1) + ' more tokens)':44}"
          f"{decode:8.2f} s   ({(args.max_new_tokens - 1) / decode:.0f} tok/s)")
    print(f"{'TOTAL   (one TIC-VLA call)':44}{f:8.2f} s   "
          f"({min(full):.2f}-{max(full):.2f})")
    print("=" * 74)

    print(f"\nbudget: {ACTION_HORIZON_S:.1f} s of SIM time; the sim runs ~0.5x realtime so "
          f"the\n        wall-clock allowance is roughly {2 * ACTION_HORIZON_S:.1f} s "
          "while it stays slow.")
    if f < ACTION_HORIZON_S:
        print(f"\nVERDICT: {f:.2f} s -- fits the horizon outright.")
    elif f < 2 * ACTION_HORIZON_S:
        print(f"\nVERDICT: {f:.2f} s -- over the 3.0 s horizon but inside the wall-clock\n"
              "         allowance the current sim speed buys. Workable, and fragile: it\n"
              "         breaks the moment the sim runs faster.")
    else:
        print(f"\nVERDICT: {f:.2f} s -- EXCEEDS even the wall-clock allowance. The robot\n"
              "         would drive on plans older than the horizon the action head was\n"
              "         trained for. Fall back to Qwen3.8-9B-Distill.")
    print(f"\nprefill is {100 * p / f:.0f}% of the call"
          + (" -- resolution/frame count is the lever, not model size."
             if p > decode else " -- decode dominates, unusually."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
