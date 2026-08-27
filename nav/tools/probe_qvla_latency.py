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
# ticvla/models/ticvla.py:579 caps generation at 200, but the cap is not the cost: the
# model emits an EOS long before it. Measured against the LIVE policy server on real nav
# frames -- six generations, 131/133/134/135/135/135 tokens by Qwen's tokenizer, mean
# 134. The distribution is tight because the output is a fixed shape: a <think> paragraph
# then one <answer> line of three (x, y, theta) triples.
#
# This matters more than any other constant here. Decode is ~97% of a 27B call and scales
# linearly in tokens, so budgeting the 200-token cap instead of the 134-token reality
# overstates the whole call by 1.5x.
TICVLA_MAX_NEW_TOKENS = 134
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
    # On this vision tower a frame costs exactly pixels / (patch_size * spatial_merge)^2
    # = pixels / (16*2)^2 = pixels / 1024 tokens. So max_pixels IS a token budget:
    #   200704 -> 196 tok/frame, which is 448x448 -- what InternVL3-1B feeds today
    #   524288 -> 512 tok/frame
    #  1048576 -> 1024 tok/frame
    #        0 -> native 1920x1080, ~2025 tok/frame
    ap.add_argument("--max-pixels", type=int, nargs="+",
                    default=[200704, 524288, 1048576, 0],
                    help="token budget per frame x1024; 0 means native resolution")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # --- work around an upstream bug in transformers 5.16.1 -------------------------
    # quantizers/quantizer_finegrained_fp8.py:update_tp_plan does
    #
    #     impl = getattr(config, "_experts_implementation", None)
    #     layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
    #     ... {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
    #
    # and `_impl_tp_layer_overrides` has exactly one key, 'deepgemm_megamoe'. A DENSE
    # model has no `_experts_implementation`, so the lookup returns None and the
    # comprehension raises AttributeError -- but only after the branch above has filled
    # base_model_tp_plan, which it does for anything whose config class name contains
    # "Qwen3". Qwen3_5Config does. So every dense Qwen3-family FP8 checkpoint hits this.
    #
    # Registering None -> {} makes the rewrite an identity for the dense path and leaves
    # the real 'deepgemm_megamoe' path untouched. Patched here rather than in
    # site-packages because a `uv pip install --upgrade` reverts that silently --
    # see memory/lelab-local-patches for what that costs.
    #
    # None of this affects the measurement: device_map="auto" is PIPELINE parallel, so
    # the tensor-parallel plan is never consulted.
    from transformers.integrations.finegrained_fp8 import FP8Experts
    FP8Experts._impl_tp_layer_overrides.setdefault(None, {})

    frames = pick_frames(args.frame_dir, N_FRAMES)
    print(f"frames    : {', '.join(f.name for f in frames)}")
    imgs = [Image.open(f).convert("RGB") for f in frames]
    print(f"resolution: {imgs[0].size[0]}x{imgs[0].size[1]}")

    # A zero budget must DROP the device, not cap it at 0GiB: the point of a single-GPU
    # run is that the model cannot span two cards, and spanning is exactly what changes
    # which FP8 kernels transformers selects (DeepGEMM's cached kernels are bound to one
    # CUDA context, so a split silently falls back to Triton/grouped_mm). Leaving the key
    # in with "0GiB" would let accelerate reintroduce the very penalty being measured.
    max_memory = {i: f"{g}GiB" for i, g in ((0, args.gpu0_gib), (1, args.gpu1_gib))
                  if g > 0 and i < torch.cuda.device_count()}
    print(f"max_memory: {max_memory}"
          + ("   [single GPU -- DeepGEMM path]" if len(max_memory) == 1 else "") + "\n")

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    # dtype="auto", not bfloat16: this checkpoint carries a quantization_config
    # (quant_method fp8, e4m3, dynamic activations) and forcing a dtype fights it.
    # Qwen leaves the vision blocks out of the FP8 conversion, so they load at their
    # stored precision either way.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="auto", max_memory=max_memory)
    model.eval()
    print(f"load      : {time.time() - t0:.1f} s")
    print(f"placement : {sorted(set(str(d) for d in model.hf_device_map.values()))}\n")

    def measure(max_pixels: int | None):
        """One resolution: build the 4-frame prompt, time prefill and full generation.

        Resolution is the one lever that is independent of which FP8 kernel path the
        placement happens to select, so it is swept rather than fixed. The processor
        resizes to fit `max_pixels`; None means the frames' native 1920x1080.
        """
        p_kwargs = {} if max_pixels is None else {"max_pixels": max_pixels}
        pr = AutoProcessor.from_pretrained(args.model, **p_kwargs)
        messages = [{"role": "user", "content":
                     [{"type": "image", "image": im} for im in imgs]
                     + [{"type": "text", "text": INSTRUCTION}]}]
        inputs = pr.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        n = int(inputs["input_ids"].shape[-1])

        def timed(max_new: int) -> float:
            torch.cuda.synchronize()
            t = time.time()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            torch.cuda.synchronize()
            return time.time() - t

        timed(4)                                     # warm-up at this shape
        pre = statistics.fmean(timed(1) for _ in range(args.repeats))
        ful = statistics.fmean(timed(args.max_new_tokens) for _ in range(args.repeats))
        return n, pre, ful

    rows = []
    for mp in args.max_pixels:
        mp = mp or None
        label = "native 1920x1080" if mp is None else f"{mp // 1024} tok/frame"
        print(f"measuring {label} ...", flush=True)
        try:
            rows.append((label,) + measure(mp))
        except Exception as exc:                     # one bad shape must not kill the sweep
            print(f"  SKIPPED {label}: {type(exc).__name__}: {exc}", flush=True)
    if not rows:
        print("no resolution measured successfully", file=sys.stderr)
        return 1

    print()
    print("=" * 78)
    print(f"{'resolution':22}{'tokens':>9}{'/frame':>8}{'prefill':>10}"
          f"{'decode':>10}{'tok/s':>8}{'TOTAL':>9}")
    for label, n, pre, ful in rows:
        dec = ful - pre
        print(f"{label:22}{n:>9}{n // N_FRAMES:>8}{pre:>9.2f}s{dec:>9.2f}s"
              f"{(args.max_new_tokens - 1) / dec:>8.0f}{ful:>8.2f}s")
    print("=" * 78)
    # Verdict on the CHEAPEST configuration measured: the question is whether 27B can be
    # made to fit at all, so the fastest legitimate setting is the one that answers it.
    best = min(rows, key=lambda r: r[3])
    label, p, f = best[0], best[2], best[3]
    decode = f - p
    print(f"verdict below is for the cheapest setting measured: {label}")

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
