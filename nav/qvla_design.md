# Q-VLA — TIC-VLA with Qwen3.8-27B as the slow model

Branch `nav/qwen-vlm-swap`. This is the design; `qwen_swap_plan.md` is the feasibility
study that preceded it and whose candidate table it supersedes.

Q-VLA keeps TIC-VLA's two-rate structure exactly — a slow vision-language model that
produces a KV cache, and a fast 3-layer cross-attention action expert that emits `(30, 2)`
waypoints at 10 Hz — and replaces only the slow half. The name is Fouad's.

## The two numbers the whole design turns on

The action expert is coupled to its VLM through two trained `nn.Linear` layers whose
shapes are the VLM's, not the head's. Measured off the released checkpoint and off
Qwen3.8-27B's own `config.json`:

| | InternVL3-1B (what the ckpt was trained against) | **Qwen3.8-27B** |
|---|---|---|
| `hidden_size` → `down_proj` | 896 | **5120** |
| kv heads × head_dim → `kv_cache_proj` | 2 × 64 = **128** | 4 × 256 = **1024** |
| layers | — | 64 |
| vision-language | yes | **yes** (`vision_config`, `out_hidden_size` 5120) |

Both mismatch, so `nav/tools/check_vlm_swap.py` reports **3.15 M parameters to
re-initialise** at the input of a 10.27 M action expert. Roughly a third of the head is
new, and the rest cannot be kept around it — a randomly initialised projection feeding a
trained cross-attention stack emits noise. **The action expert is retrained in full.**

## Qwen3.8-27B is a hybrid, and that nearly broke the design

`layer_types` is 48 `linear_attention` + 16 `full_attention`, `full_attention_interval:
4` — a GatedDeltaNet-style hybrid, not a dense transformer. This matters because
`ticvla/models/ticvla.py:108` does

```python
_, last_layer_value = past_key_values[-1]
```

and asserts 4-D `(B, num_heads, seq_len, head_dim)` on the next line. Linear-attention
layers hold a recurrent/conv state, not a `(key, value)` pair, so on most hybrids `[-1]`
is not unpackable at all.

**It survives here by arithmetic:** full-attention layers are `[3, 7, …, 59, 63]` and the
model has 64 layers, so the *last* layer is index 63 and is full attention. `[-1]` lands
on a real KV pair.

Do not rely on that. Q-VLA selects the last full-attention layer **by index, from
`layer_types`**, because `full_attention_interval` dividing `num_hidden_layers` is a
property of this checkpoint and not of the architecture — a 62-layer sibling would end on
`linear_attention` and fail at the `dim() != 4` check with a message about shapes that
says nothing about why.

## The four surgery points, each verified by grep

TIC-VLA is vendored **read-only** at `/home/gtu-dsa/robotics/TIC-VLA`, so none of this is
edited in place. Q-VLA subclasses and overrides from `nav/qvla/`.

| # | where | what is wrong | fix |
|---|---|---|---|
| 1 | `ticvla.py:250-252` | `kv_num_heads = 2` / `kv_head_dim = 64`, hardcoded, comment says "Fixed value for InternVL3-1B model state (**NOT from VLM config**)" | read `num_key_value_heads` × `head_dim` from `text_config` → 1024 |
| 2 | `ticvla.py:256, 452, 466, 470` | `self.vlm.config.llm_config.hidden_size` — `llm_config` is InternVL's key | Qwen uses `text_config`; four call sites, all the same |
| 3 | `ticvla.py:108` | `past_key_values[-1]` assumes every layer has a KV pair | select last `full_attention` index from `layer_types` |
| 4 | `vlm.py` (all 206 lines) | `img_context_token_id` / `<IMG_CONTEXT>` (`:84-86`), `self.vlm.chat(...)` (`:139`), `num_patches_list` tiling (`:119, 146`) — none exist on Qwen | rewrite against the processor API and `image_token_id` 248056 |

Point 4 is the bulk of the work. `vlm.py` is not a thin wrapper; it is an InternVL
adapter, and `.chat()` is a custom method that ships in InternVL's remote code.

## Two environment facts that force the layout

**`transformers` must jump 4.57.6 → 5.8.0.dev0.** Qwen3.8's config declares it, and 4.57
cannot even *read* the config: `AutoConfig` raises "does not recognize this architecture
`qwen3_5`". A major-version jump will not leave InternVL's `trust_remote_code` path
working. So Q-VLA gets **its own venv, `~/envs/qvla`**, and `~/envs/tic-vla` is not
touched — the same rule the root `CLAUDE.md` applies to `isaac5`/`isaac6`, for the same
reason.

**Two GPUs at 31.35 GiB, no NVLink, no P2P**, and Isaac Sim already holds ~25.4 GiB of
GPU0 for a DynaNav scene. That leaves GPU1 for the policy, exactly as the InternVL server
runs today.

## Which build of Qwen3.8-27B, and why FP8 cannot be the closed-loop one

Sizes from the HF API; the VRAM column is weights only, before KV cache and the 4-frame
vision prefill.

| build | on disk | fits GPU1 alone (31.35 GiB)? | HF-loadable? |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` bf16 | 55.59 GB | no | yes |
| `Qwen/Qwen3.8-27B-FP8` | **30.89 GB** | **no — 0.5 GB headroom** | yes |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 21.04 GB | yes (~10 GB spare) | yes |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.44 GB | yes (~8 GB spare) | yes |
| `empero-ai/Qwen3.8-27B-Ridge-GGUF` | 13.53 GB | yes | **no — see below** |

**GGUF is out, including the Ridge build**, and this is worth stating plainly because
Ridge is the lightest option and the one the linked write-up recommends. Ridge is real
and well-targeted — it keeps the GatedDeltaNet state layers at higher precision and
pushes the compression into the feed-forward layers, lands at 11.7 GB, and runs vision
through a separate `mmproj`. But Q-VLA does not consume the model's *text*. It consumes
`past_key_values` as a tensor. llama.cpp exposes no HF-style per-layer KV cache to Python,
so a GGUF build cannot feed the action expert at all. Ridge is a good way to *chat* with
this model; it is not a backbone.

**FP8 is the one currently downloading, and it does not fit GPU1.** 30.89 GB of weights
against 33.66 GB of card leaves nothing for the vision tower, the KV cache, or four
frames of prefill. It has a real use — spread across both GPUs with `device_map="auto"`
it fits comfortably (pipeline-parallel inference only moves activations over PCIe, and
needs no P2P), which makes it the **quality reference** for offline work. It cannot be
the closed-loop build, because closed-loop means Isaac Sim owns GPU0.

**So: FP8 for offline / quality ceiling, a 4-bit build on GPU1 for the benchmark.**
Between the two 4-bit candidates, NVFP4 is the one aimed at this hardware — sm_120 has
FP4 tensor cores, and Blackwell executes NVFP4 natively where AWQ-INT4 dequantises. That
is the second download to queue.

## Training: stage 1 is not affordable, and skipping it is the honest experiment

The released checkpoint is 1886.66 M parameters of which **1876.39 M (99.46%) are the
VLM**, and that VLM is stage-1 output — fine-tuned on navigation chain-of-thought, not
stock InternVL3-1B. The faithful port would redo both stages.

Stage 1 at 27B is not possible here: 55.59 GB of bf16 weights before optimiser state or
activations, on 2×31.35 GiB with no P2P, and `memory/pi05-memory-is-activations-not-params`
records that LoRA does not reduce the activation memory that actually OOMs.

Q-VLA therefore **freezes the quantized Qwen and trains only the action expert** —
stage 2 alone, `configs/train_action.yaml`, 10.27 M trainable parameters of which 3.15 M
are new. This is a real narrowing of scope and it is stated rather than hidden. It is
also the cleaner test of the hypothesis: the question is "does a much stronger VLM
navigate better", and holding the VLM stock isolates that from "does more nav-specific
fine-tuning help".

**The control comes first.** Before the swap is benchmarked, stage 2 must be reproduced
against the *current* InternVL VLM. Without that, a bad Q-VLA number cannot be attributed
to the swap rather than to our training loop.

## Latency — the earlier estimate was built on a wrong premise

`qwen_swap_plan.md` argued a 27B would generate in 7–12 s against a 3.0 s action horizon,
by scaling InternVL3-1B's 1.7 s by the weight-traffic ratio of a **dense** 27B. Qwen3.8-27B
is not dense: 48 of its 64 layers are linear attention, which is the architecture whose
whole point is cheap decode. Published single-card NVFP4 numbers for this model are in the
~140 tok/s range. That estimate does not transfer and should not be quoted.

What has *not* gone away is **prefill**. TIC-VLA passes four frames, and Qwen's vision
tower at `patch_size 16` / `spatial_merge_size 2` turns a 1920×1080 frame into ~2000
tokens, so ~8000 vision tokens per call before a single output token. That is where the
time will go, and it is measurable rather than arguable: `check_vlm_swap.py --time-it`
answers it once the weights are on disk, and it should be run **before** any training.

The budget is unchanged. One generation must stay well inside 3.0 s of *sim* time; the
sim runs at ~0.5× realtime so the wall-clock allowance is roughly double, and
`nav/README.md` records staleness already at 1.5–2.0 s with the 1B model.

**If prefill blows the budget, the fallback is `empero-ai/Qwen3.8-9B-Distill`** — same
family, same tokenizer, a third of the size — not a return to InternVL.

## Data

The dataset is `handsomeYun/TIC-VLA`, 538.29 GB, public, CC-BY-4.0. Two measurements
worth keeping:

- **The zips are `stored`, not deflated — 1.00× expansion**, measured on the completed
  `DynaNav_json.zip` (288,602 entries, 25.83 GB compressed, 25.83 GB uncompressed) by
  reading the central directory rather than extracting. So unzipping costs the archive
  size *again*: 538 GB of zips plus 538 GB extracted is 1076 GB against ~936 GB free.
  **Extract each zip then delete it**, which caps the peak at 538 GB + the largest single
  archive.
- **`_json` contains no images** — 151,183 `.txt` and 137,419 `.json`, paths like
  `DynaNav_json/office_6_spot_50s_70s/rgb_01812.json`. The frames live in the `_data`
  archives, so stage 2 needs both and the earlier "223 GB of `_json` is enough" reading
  was wrong.

## Order of work

1. `check_vlm_swap.py --time-it` on the FP8 build across both GPUs, **before** anything
   is written. If four-frame prefill cannot fit the horizon, the 27B is over and the 9B
   distill is the design.
2. `~/envs/qvla` with transformers ≥ 5.8, torch `+cu128` (sm_120), verified with a real
   matmul rather than `is_available()`.
3. Reproduce stage 2 against InternVL. This is the control.
4. `nav/qvla/` — the four surgery points, as subclasses over the read-only vendor tree.
5. Stage 2 with the frozen 4-bit Qwen.
6. The 13-episode ladder, **3 repeats per episode**. Single runs are noise: `predict()`
   is deterministic to 0.00° sd, but a run is not, because async cache timing varies.
