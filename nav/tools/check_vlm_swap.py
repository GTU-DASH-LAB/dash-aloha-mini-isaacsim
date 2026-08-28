"""Can TIC-VLA's VLM be replaced with a different one, and what would it cost?

The VLM is not a plug-in component, and the reason is two `nn.Linear` layers inside the
action expert whose shapes are pinned to whatever VLM produced the KV cache:

    action_expert.down_proj      (hidden_dim, VLM hidden_size)
    action_expert.kv_cache_proj  (hidden_dim, num_key_value_heads * head_dim)

`ticvla/models/ticvla.py` names them in the constructor -- `input_dim` is documented as
"Dimension of image embeddings (hidden_size from VLM)" and `kv_cache_feat_dim` as
"num_heads * head_dim (typically 2 * 64 = 128 for InternVL3-1B)". In the released
checkpoint they are trained weights of shape (512, 896) and (512, 128). Swap the VLM and
both are the wrong shape, so they cannot be loaded -- they have to be re-initialised and
re-trained. A randomly initialised projection into a trained cross-attention stack does
not degrade gracefully; it emits noise. Benchmarking that measures nothing.

This script answers, for a candidate VLM, the four questions that decide the swap:

  1. SHAPE    -- what would mismatch, and how many parameters have to be retrained;
  2. KV CACHE -- does `past_key_values[-1]` even yield a (key, value) pair? On a hybrid
                 linear-attention/SSM model most layers carry a recurrent state instead,
                 and ticvla.py:108 unpacks the last layer unconditionally;
  3. MODALITY -- is it a vision-language model at all? A text-only Qwen3 cannot take the
                 camera frame, which ends the discussion before shapes matter;
  4. LATENCY  -- how long one KV-cache generation takes on this hardware. This is the one
                 that can kill a swap outright regardless of retraining. The action head
                 plans exactly 3.0 s ahead (30 waypoints at 10 Hz), and `nav/README.md`
                 records staleness already sitting at 1.5-2.0 s with InternVL3-1B. A VLM
                 that needs longer than ~3 s hands the controller a plan that has expired
                 before it is used, and no amount of model quality compensates.

Usage (config-only, no weights needed -- answers 1 and 2):

    /home/gtu-dsa/envs/tic-vla/bin/python nav/tools/check_vlm_swap.py \\
        --candidate Qwen/Qwen3-VL-8B-Instruct

Add --time-it once the weights are on disk to answer 3. That loads the model, so give it
a card with room: `CUDA_VISIBLE_DEVICES=1` is safe here because this never starts Kit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CKPT = Path("/home/gtu-dsa/robotics/models/TIC-VLA-model.ckpt")
BASELINE = Path("/home/gtu-dsa/robotics/models/InternVL3-1B")

# The budget a generation has to beat: a plan that outlives its own replacement.
#
# This was 3.0 s, TIC-VLA's action-head horizon, and copying that number here was wrong
# once the action head was removed. With the expert in place the waypoints are refreshed
# every control tick and only the KV cache ages, so 3.0 s is a cache-freshness target.
# Without it the waypoints ARE what ages, and a generation slower than the horizon leaves
# the controller nothing to follow -- measured as a permanent stop, 129 of 141 calls.
# `server_qwen.py` now plans 10.0 s ahead for exactly this reason; the check has to move
# with it or it reports EXCEEDS for a configuration that is fine.
ACTION_HORIZON_S = 10.0


def kv_feat_dim(cfg: dict) -> tuple[int, str]:
    """num_key_value_heads * head_dim, with the fallbacks HF configs actually use."""
    text = cfg.get("text_config") or cfg.get("llm_config") or cfg
    hidden = text.get("hidden_size")
    n_kv = (text.get("num_key_value_heads")
            or text.get("num_kv_heads")
            or text.get("num_attention_heads"))
    head_dim = text.get("head_dim")
    if head_dim is None and hidden and text.get("num_attention_heads"):
        head_dim = hidden // text["num_attention_heads"]
    if not (n_kv and head_dim):
        return 0, "could not determine (no num_key_value_heads / head_dim in config)"
    return n_kv * head_dim, f"{n_kv} kv heads x {head_dim} head_dim"


def load_config(ref: str) -> dict:
    """Local directory first, then AutoConfig, then the raw config.json off the hub.

    The third path is not redundant. A candidate new enough to be interesting is often
    too new for the installed transformers -- Qwen3.8-27B declares `transformers_version:
    5.8.0.dev0` and AutoConfig on 4.57 dies with "does not recognize this architecture
    `qwen3_5`". That is a *loader* limitation; the shapes this script reports are plain
    JSON and need no architecture support at all, so refusing to answer would be wrong.
    """
    local = Path(ref) / "config.json"
    if local.is_file():
        return json.loads(local.read_text())
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(ref, trust_remote_code=True).to_dict()
    except Exception as exc:
        try:
            import urllib.request
            url = f"https://huggingface.co/{ref}/resolve/main/config.json"
            with urllib.request.urlopen(url, timeout=30) as r:
                cfg = json.loads(r.read())
            print(f"note: transformers could not load {ref!r} ({type(exc).__name__}); "
                  "read config.json off the hub instead.\n", file=sys.stderr)
            return cfg
        except Exception:
            pass
        print(f"ERROR: could not read a config for {ref!r}: {exc}\n"
              "       Pass a local directory, or download the model first.", file=sys.stderr)
        raise SystemExit(2)


def kv_cache_reachable(cfg: dict) -> tuple[bool, str]:
    """Does `past_key_values[-1]` yield a (key, value) pair on this architecture?

    TIC-VLA reads the LAST layer's cache -- `ticvla/models/ticvla.py:108` does
    `_, last_layer_value = past_key_values[-1]` and asserts 4-D on the next line. On a
    hybrid (linear-attention / SSM layers interleaved with full attention) most layers
    carry a recurrent state instead of a KV pair, so that unpack fails unless the final
    layer happens to be a full-attention one. Qwen3.8-27B passes only by arithmetic:
    16 full-attention layers at interval 4 in 64 layers puts one exactly at index 63.
    """
    text = cfg.get("text_config") or cfg.get("llm_config") or cfg
    types = text.get("layer_types")
    if not types:
        return True, "dense (no layer_types) -- every layer has a KV pair"
    kinds = sorted(set(types))
    if len(kinds) == 1:
        return True, f"uniform {kinds[0]}"
    full = [i for i, t in enumerate(types) if "full" in t]
    if not full:
        return False, f"no full-attention layer at all; layer_types={kinds}"
    counts = {k: types.count(k) for k in kinds}
    ok = full[-1] == len(types) - 1
    return ok, (f"HYBRID {counts}; last full-attention layer is index {full[-1]} "
                f"of {len(types)} -> last layer is {types[-1]}")


def is_vision_language(cfg: dict) -> tuple[bool, str]:
    for key in ("vision_config", "vision_tower_config", "visual"):
        if cfg.get(key):
            return True, f"has {key}"
    arch = " ".join(cfg.get("architectures") or [])
    if any(t in arch.lower() for t in ("vl", "vision", "image")):
        return True, f"architecture {arch}"
    return False, f"no vision config; architectures={cfg.get('architectures')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="local model dir or HF repo id of the replacement VLM")
    ap.add_argument("--time-it", action="store_true",
                    help="also load the model and time one generation (needs weights)")
    ap.add_argument("--gen-tokens", type=int, default=64)
    args = ap.parse_args()

    # --- what the trained checkpoint actually pins -----------------------------
    import torch
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    dp = sd["model.action_expert.down_proj.weight"].shape        # (hidden_dim, vlm_hidden)
    kp = sd["model.action_expert.kv_cache_proj.weight"].shape    # (hidden_dim, kv_feat)
    trained_hidden, trained_kv = int(dp[1]), int(kp[1])

    base = load_config(str(BASELINE))
    base_kv, base_kv_how = kv_feat_dim(base)
    base_hidden = (base.get("text_config") or base.get("llm_config") or base)["hidden_size"]

    cand = load_config(args.candidate)
    cand_kv, cand_kv_how = kv_feat_dim(cand)
    cand_hidden = (cand.get("text_config") or cand.get("llm_config") or cand)["hidden_size"]

    print("=" * 78)
    print("1. SHAPE")
    print("=" * 78)
    print(f"{'':22} {'hidden_size':>12} {'kv_feat_dim':>12}")
    print(f"{'trained checkpoint':22} {trained_hidden:>12} {trained_kv:>12}")
    print(f"{'InternVL3-1B (base)':22} {base_hidden:>12} {base_kv:>12}   ({base_kv_how})")
    print(f"{args.candidate[:22]:22} {cand_hidden:>12} {cand_kv:>12}   ({cand_kv_how})")
    print()

    ok_hidden = cand_hidden == trained_hidden
    ok_kv = cand_kv == trained_kv
    if ok_hidden and ok_kv:
        print("  Both projections match. This is the ONLY case where the released action")
        print("  expert loads as-is -- and it still assumes the candidate's KV cache means")
        print("  the same thing per channel, which sharing a width does not guarantee.")
    else:
        out_dim = int(dp[0])
        retrain = 0
        if not ok_hidden:
            retrain += out_dim * cand_hidden + out_dim
            print(f"  MISMATCH down_proj     : trained for {trained_hidden}, candidate is {cand_hidden}")
        if not ok_kv:
            retrain += out_dim * cand_kv + out_dim
            print(f"  MISMATCH kv_cache_proj : trained for {trained_kv}, candidate is {cand_kv}")
        ae = sum(v.numel() for k, v in sd.items()
                 if "action_expert" in k and hasattr(v, "numel"))
        print()
        print(f"  -> {retrain/1e6:.2f} M parameters must be re-initialised and retrained.")
        print(f"     They sit at the INPUT of a {ae/1e6:.2f} M action expert, so the whole")
        print("     head has to be retrained with them (stage 2, configs/train_action.yaml).")

    print()
    print("=" * 78)
    print("2. KV CACHE REACHABILITY")
    print("=" * 78)
    reachable, how = kv_cache_reachable(cand)
    print(f"  past_key_values[-1] unpacks: {'YES' if reachable else 'NO'}")
    print(f"  {how}")
    if not reachable:
        print("  The action expert is fed the LAST layer's value tensor. On this model")
        print("  that layer carries a recurrent state, not a (key, value) pair, so")
        print("  ticvla.py:108 raises before any shape question arises. A port must")
        print("  select the last FULL-attention layer by index from layer_types.")
    elif "HYBRID" in how:
        print("  Works, but by arithmetic rather than by design -- select the last")
        print("  full-attention layer explicitly, or a sibling with a different layer")
        print("  count will fail at the dim()!=4 check with an unhelpful message.")

    print()
    print("=" * 78)
    print("3. MODALITY")
    print("=" * 78)
    vl, why = is_vision_language(cand)
    print(f"  vision-language: {'YES' if vl else 'NO'}   ({why})")
    if not vl:
        print("  A text-only model cannot consume camera_nav frames. TIC-VLA's own system")
        print("  prompt describes 'a video consisting of visual observations' and DynaNav")
        print("  passes four frames per call. This ends the swap regardless of shapes.")

    print()
    print("=" * 78)
    print("4. LATENCY")
    print("=" * 78)
    if not args.time_it:
        print("  skipped (pass --time-it with the weights on disk)")
        print(f"  Budget: one generation must stay well under {ACTION_HORIZON_S:.1f} s of SIM")
        print("  time. Staleness here already runs 1.5-2.0 s with InternVL3-1B, and the sim")
        print("  runs at ~0.5x realtime, so the wall-clock budget is roughly double that --")
        print("  but only while the sim stays slow. Speeding the sim up shrinks it.")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(args.candidate, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.candidate, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model.eval()
        print(f"  load: {time.time() - t0:.1f} s")
        prompt = "Describe the scene ahead and plan a path." * 8
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():                      # warm-up, so the timing is steady state
            model.generate(**ids, max_new_tokens=8, do_sample=False)
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model.generate(**ids, max_new_tokens=args.gen_tokens, do_sample=False)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        mean = sum(times) / len(times)
        print(f"  generation ({args.gen_tokens} tokens): {mean:.2f} s mean over 3 "
              f"({min(times):.2f}-{max(times):.2f})")
        print(f"  VERDICT: {'within' if mean < ACTION_HORIZON_S else 'EXCEEDS'} the "
              f"{ACTION_HORIZON_S:.1f} s action horizon")
        print("  Note this is text-only prefill; the real call also encodes four frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
