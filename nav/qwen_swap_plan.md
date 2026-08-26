# Replacing TIC-VLA's VLM — what it actually costs

Branch `nav/qwen-vlm-swap`. Written before doing the work, because the cheap version of
this idea does not exist and it is better to find that out on paper than after a
download.

## The VLM is not a plug-in part

Two `nn.Linear` layers inside the action expert are shaped by whatever VLM produced the
KV cache. From the released checkpoint:

| tensor | shape | pinned to |
|---|---|---|
| `action_expert.down_proj.weight` | **(512, 896)** | InternVL3-1B `hidden_size` = 896 |
| `action_expert.kv_cache_proj.weight` | **(512, 128)** | 2 kv heads × 64 head_dim = 128 |

`ticvla/models/ticvla.py` says so in the constructor — `input_dim` is "Dimension of image
embeddings (hidden_size from VLM)", `kv_cache_feat_dim` is "num_heads * head_dim
(typically 2 * 64 = 128 for InternVL3-1B)". Any other VLM changes both. The weights then
cannot be loaded, and a randomly initialised projection feeding a trained cross-attention
stack does not degrade gracefully — it emits noise. **Benchmarking that measures
nothing**, which is why this branch does not contain a "just swap it and run" commit.

`nav/tools/check_vlm_swap.py` reports this for any candidate, plus whether it is actually
vision-language and how long one generation takes. Self-test against InternVL3-1B
reproduces 896 / 128.

## And the checkpoint carries a fine-tuned VLM, not a stock one

Of the checkpoint's 1886.66 M parameters, **1876.39 M (99.46%) are the VLM** and only
10.27 M are the action expert. That VLM is stage-1 output (`configs/train_vlm.yaml`),
fine-tuned on navigation chain-of-thought. Pairing a *stock* Qwen with a retrained action
head skips that stage, so the honest swap is both stages, not one:

1. **Stage 1** — fine-tune the candidate VLM on the nav CoT data. This is the expensive
   half: it trains the 1.88 B-parameter side.
2. **Stage 2** — freeze it, retrain the 10.27 M action expert (`configs/train_action.yaml`).
   Cheap by comparison.

## Hardware puts a ceiling on the candidate, and it is well under 27B

Two RTX 5090s, 31.35 GiB each, **no NVLink and no P2P** — the root `CLAUDE.md` says
prefer two independent single-GPU experiments over DDP for VLA fine-tuning. bf16 weights
alone:

| candidate | bf16 weights | fits one card? | stage-1 fine-tune here? |
|---|---|---|---|
| InternVL3-1B (current) | ~2 GB | yes | yes |
| Qwen3-VL **8B** class | ~16 GB | yes | plausible with LoRA + gradient checkpointing |
| Qwen3-VL **27–32B** | ~54–64 GB | **no** | no — needs FSDP across PCIe with no P2P |

4-bit quantisation gets a 27B *inference* footprint to ~14 GB, so **running** one is
possible. Fine-tuning through a 4-bit base for a navigation task is a research gamble,
not a step. Note also `memory/pi05-memory-is-activations-not-params` — LoRA does not
reduce activation memory, which is what actually OOMs.

## Latency is the argument that does not go away

TIC-VLA exists *because* VLM inference is slow. The action head plans exactly 3.0 s ahead
(30 waypoints at 10 Hz) and `nav/README.md` records staleness already at **1.5–2.0 s**
with InternVL3-1B at ~1.7 s per generation.

Generation is memory-bandwidth-bound. Rough scaling on one 5090 (~1.8 TB/s): 1B at bf16
is ~2 GB of weight traffic per token; a 4-bit 27B is ~14 GB, i.e. **~7× slower per
token**, before the heavier prefill over four frames. A 1.7 s generation becomes roughly
**7–12 s** — two to four times the entire plan horizon. Async inference means the robot
will not freeze; it means it drives on plans that expired several seconds ago, far outside
the horizon the action head was trained for.

Making the VLM 27× larger attacks the premise of the paper it implements.

## And the failure we just measured is not a language failure

Worth stating before spending a week on this. The `hospital_vending_machine2` diagnosis
(commit `c73b470`) found the model already reads the instruction correctly — 35.6° of
steering authority between "turn left" and "turn right" — and already sees the landmark,
centred and 0.7° off the nose. What failed was the *geometry* (a traversable plan bending
around a machine set against a wall) and inconsistent arrival recognition. Both live in
the 10.27 M action expert, which stays the same size whichever VLM feeds it. A larger VLM
is not aimed at the measured defect.

## Data

`handsomeYun/TIC-VLA` on HuggingFace, CC-BY-4.0, public and ungated — the three sources
`train_action.yaml` names, plus the checkpoint we already have.

| file | size |
|---|---|
| `SCAND/SCAND_data.zip` | 195.66 GB |
| `GND/GND_json.zip` | 107.38 GB |
| `SCAND/SCAND_json.zip` | 90.12 GB |
| `DynaNav/DynaNav_data.zip` | 77.48 GB |
| `GND/GND_data.zip` | 39.80 GB |
| `DynaNav/DynaNav_json.zip` | 25.90 GB |
| **total** | **538.29 GB** |

Stage 2 points only at the three `_json` dirs = 223.40 GB compressed.

**Download speed matters and the obvious measurement is wrong.** A single-connection
`curl` gets **1.35 MB/s** from HuggingFace here; `aria2c -x16` gets **~10.5 MiB/s**, the
same parallel-beats-serial effect `memory/gtu-network-cdn-speeds` records for NVIDIA's
CDN. So: `_json` only ≈ **5.6 h**, everything ≈ **13.6 h**.

Beware a trap when measuring it: `aria2c -x16` writes segments at file offsets, so the
file is **sparse** and `ls -la` reports apparent size. That read as 270 MB/s on the first
attempt. `du` without `--apparent-size` showed 1021 M against an apparent 23 G. Use
aria2's own `DL:` counter or `du`.

Disk is fine — 1.2 TB free.

## What this branch does

- `nav/tools/check_vlm_swap.py` — shape / modality / latency check for any candidate.
- this document.

It deliberately does **not** contain a VLM swap, because a swap without stage 1 + stage 2
produces a model that cannot be meaningfully benchmarked.

## Recommended order

1. Name the exact candidate and run `check_vlm_swap.py` on it. An 8B-class **vision**-
   language Qwen3-VL is the only size this hardware supports end to end.
2. Pull `DynaNav_json` first (25.9 GB, ~36 min) and confirm stage 2 trains against the
   *current* VLM. Reproducing the released action expert is the control — without it, a
   bad result after the swap cannot be attributed to the swap.
3. Only then stage 1 on the candidate, then stage 2, then the 13-episode ladder with
   3 repeats per episode (single runs are noise — `predict()` is deterministic to 0.00°
   sd but a run is not, because async cache timing varies).

## The cheaper experiment, if the goal is "better reasoning"

A large Qwen does not have to be *inside* TIC-VLA. Run it as a slow outer layer — re-read
the scene every few seconds and rewrite the instruction or pick a subgoal — and let
TIC-VLA keep driving at 10 Hz. No retraining, no shape problem, no latency problem (the
outer loop is allowed to be slow), and it is benchmarkable in days. If the hypothesis is
"a stronger language model would navigate better", this tests it without rebuilding the
policy.
