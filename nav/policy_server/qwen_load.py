"""Load a Qwen3.8 checkpoint without silently getting a broken model.

Two upstream problems sit between `from_pretrained` and a working Qwen3.8-27B-FP8 on
transformers 5.16.1. Both announce themselves as warnings, or not at all, and one of them
produces a model that loads, runs at full speed, reports no error, and emits pure
gibberish. Anything in this repo that loads Qwen goes through here so the fixes cannot be
applied to one caller and forgotten in another.

1. `modules_to_not_convert` is PREFIX-matched, and the list has MoE leftovers
--------------------------------------------------------------------------------------
Qwen3.8-27B-FP8's `quantization_config.modules_to_not_convert` has 882 entries, among
them `model.language_model.layers.N.mlp.gate` and `...mlp.shared_expert_gate` for every
layer. Those are MoE router modules; this checkpoint is dense and has neither. They would
be harmless except that the skip list is matched by prefix, and `...mlp.gate` is a prefix
of `...mlp.gate_proj` -- which is a real, large linear layer in every one of the 64
blocks.

The consequence: `gate_proj` is excluded from FP8 conversion, so it stays an ordinary
`nn.Linear`. Its e4m3 weights are then loaded into it *raw*, and its `weight_scale_inv` is
discarded as an unexpected key. The gate half of every SwiGLU is off by a per-block scale.

What that looks like, measured on this machine before the fix:

    Q: What is the capital of France?
    A: 'althocie不成这儿ardy礼拜 forth喜怒哀perationarkedptos爵iletucarКТ混混混混混混'

and after:

    A: 'We need answer simple question. Need final concise.</think>The capital of France is Paris.'

The only warning transformers gives is a load report listing
`layers.{0..63}.mlp.gate_proj.weight_scale_inv` as UNEXPECTED, next to a note saying
UNEXPECTED "can be ignored when loading from different task/architecture". Here it could
not be ignored at all. **Read the load report.**

Note this does not touch timing: same shapes, same FLOPs, same token counts, so latency
measured on the broken model is still valid. Quality measured on it is worthless.

2. `update_tp_plan` crashes on any dense Qwen3-family FP8 checkpoint
--------------------------------------------------------------------------------------
`quantizer_finegrained_fp8.py:update_tp_plan` fills `base_model_tp_plan` for anything
whose config class name contains "Qwen3" (Qwen3_5Config does), then looks up
`FP8Experts._impl_tp_layer_overrides[config._experts_implementation]`. A dense model has
no `_experts_implementation`, the dict has exactly one key, and the lookup returns None
just before `.get` is called on it. Registering None -> {} makes the rewrite an identity
for the dense path and leaves the real 'deepgemm_megamoe' path alone.

Patched at runtime rather than in site-packages, which a `uv pip install --upgrade`
reverts silently -- see memory/lelab-local-patches for what that has cost before.
"""

from __future__ import annotations

import re

# MoE router modules that do not exist in a dense checkpoint, and whose names prefix real
# linear layers. Anchored to the end of the entry so a genuine module named e.g.
# `mlp.gate_proj` in some future config is never dropped.
_SPURIOUS_SKIP = re.compile(r"\.mlp\.(gate|shared_expert_gate)$")


def patch_transformers_fp8() -> None:
    """Make dense Qwen3-family FP8 checkpoints loadable at all."""
    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
    except ImportError:
        return
    FP8Experts._impl_tp_layer_overrides.setdefault(None, {})


def fixed_config(model_path: str, verbose: bool = True):
    """Return a config whose FP8 skip-list cannot swallow `mlp.gate_proj`.

    Returns None when there is nothing to fix, so callers can pass it straight through to
    `from_pretrained(config=...)` only when it matters.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    qc = getattr(cfg, "quantization_config", None)
    if qc is None:
        return None
    is_dict = isinstance(qc, dict)
    # `.get`, not `[...]`: the key is FP8's. NVFP4 quantises the same dense checkpoint
    # and ships no skip-list at all, so subscripting raised KeyError and the server died
    # in startup before it ever reached the weights. The object branch below was already
    # tolerant; only the dict branch was not.
    skip = list((qc.get("modules_to_not_convert") if is_dict
                 else getattr(qc, "modules_to_not_convert", None)) or [])
    if not skip:
        return None

    keep = [s for s in skip if not _SPURIOUS_SKIP.search(s)]
    dropped = len(skip) - len(keep)
    if dropped == 0:
        return None
    if verbose:
        print(f"[qwen-load] dropped {dropped} MoE-only entries from "
              f"modules_to_not_convert; without this, mlp.gate_proj is prefix-matched "
              f"by mlp.gate, loses its FP8 scale, and the model emits gibberish",
              flush=True)
    if is_dict:
        qc["modules_to_not_convert"] = keep
    else:
        qc.modules_to_not_convert = keep
    return cfg


def load_qwen(model_path: str, max_memory: dict | None = None,
              max_pixels: int | None = None, verbose: bool = True):
    """(processor, model), correctly quantized, with the placement reported.

    `max_memory` entries of 0 are DROPPED rather than capped, because a device left in
    the map can still be spilled onto -- and a model that spans devices is routed off
    DeepGEMM onto Triton, which changes decode throughput by more than most of the things
    one would be benchmarking.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    patch_transformers_fp8()
    cfg = fixed_config(model_path, verbose)

    p_kwargs = {} if max_pixels is None else {"max_pixels": max_pixels}
    proc = AutoProcessor.from_pretrained(model_path, **p_kwargs)

    mm = None
    if max_memory:
        mm = {i: f"{g}GiB" for i, g in max_memory.items()
              if float(g) > 0 and i < torch.cuda.device_count()}
    kwargs = {"dtype": "auto", "device_map": "auto", "max_memory": mm}
    if cfg is not None:
        kwargs["config"] = cfg
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()

    # Read the placement off the parameters, not off `hf_device_map`. The map is
    # accelerate's PLAN and it is only attached when accelerate actually dispatches --
    # with one visible GPU the whole model can load straight onto it and the attribute
    # never appears, which crashed this line. Parameter devices are the ground truth the
    # DeepGEMM/Triton warning below actually depends on, so prefer them outright and keep
    # the map only as a fallback for a model with no parameters of its own.
    devs = sorted({str(p.device) for p in model.parameters()}) or sorted(
        {str(d) for d in getattr(model, "hf_device_map", {}).values()})
    if verbose:
        print(f"[qwen-load] placement={devs}"
              + ("  [single GPU -- DeepGEMM path]" if len(devs) == 1 else ""), flush=True)
        if len(devs) > 1:
            print("[qwen-load] WARNING: model spans multiple devices; FP8 layers fall "
                  "back from DeepGEMM to Triton and decode timings are not comparable "
                  "to a single-GPU run", flush=True)
    return proc, model
