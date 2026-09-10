# Can a small VLM drive the arc menu? — survey, selection, and what the swap has to pass

The navigation policy on this branch is a frozen **Qwen3.8-27B-FP8** that never sees a
trajectory, never gets fine-tuned, and answers with an integer. `nav/arc_menu.py` draws
candidate paths on the robot's own camera frame, and the model picks one. That is the
whole policy, and it scores **8/13** on the ladder against TIC-VLA's 6/13.

A 27B holding **28.75 GiB** is a strange thing to be at the bottom of a robot. This
document asks whether something an order of magnitude smaller can do the same job, picks
five candidates, and defines the test.

---

## 1. What a candidate actually has to do

This is the part that decides the shortlist, and it is much narrower than "be a good VLM".

The arc-menu policy needs exactly one function:

```
(system prompt, user prompt, one image) -> text
```

No KV cache. No action head. No hidden states. That is a **much** weaker requirement than
the TIC-VLA swap analysed in [`nav/qwen_swap_plan.md`](qwen_swap_plan.md), where the
candidate had to yield a `past_key_values[-1]` of the right shape to feed a trained
`down_proj` — which is why a GGUF build was disqualified there and is not disqualified
here. Here the model is a black box that reads a picture and writes a number.

So the gates are:

| gate | why it bites |
|---|---|
| **loads on `transformers` 5.16.1** | the policy-server venv (`~/envs/qvla`, py3.12). Not negotiable without a second venv, and a second venv means a second torch build. |
| **takes an interleaved image + text chat template** | the probes pass a system prompt and one image. A captioner with a fixed prompt cannot be asked a navigation question. |
| **follows "answer with the number and nothing else"** | the answer is parsed. A model that narrates is not wrong, it is unusable at 8 tokens. |
| **decides inside the control period** | the SIDED chain is two calls. The 27B pays 2.10 s for both. |

Note what is *not* a gate: benchmark scores on DocVQA, MMMU or OCR. Those are the numbers
every source below reports and they measure reading documents, not judging whether a
drawn arc lands on floor or on wall. They are used here only to rank candidates for
*inclusion*, never as the answer. The answer comes from §5.

---

## 2. The landscape (web research, September 2026)

Third-party figures. Treated as data for choosing what to download, not as results.

**General-purpose small VLMs**

- **Qwen3-VL** — 2B / 4B / 8B / 30B+, Apache-2.0. Repeatedly ranked the strongest
  small open model of 2026: **MMMU 69.6, DocVQA 96.1** at 8B, DocVQA 95.3 at 4B, both
  clearing every Gemma 3 size *including* 27B on document reading. 256K context. Runs in
  ~6 GB at Q4 (4B) / ~12 GB (8B). ([HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
  [codersera](https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/),
  [tinyweights](https://tinyweights.dev/posts/best-local-vision-language-models-2026/))
- **InternVL3.5** — 1B…241B incl. MoE. Their own table: **2B avg 50.7, 4B 57.4, 8B 60.3**,
  against InternVL3-2B's 32.4 and MiniCPM-V-4's 33.5. A separate throughput study puts
  **InternVL3.5-2B at the highest throughput and lowest per-token latency** of the models
  it tested, across concurrency levels. ([arXiv 2508.18265](https://arxiv.org/html/2508.18265v1))
- **Gemma 3** — 1B/4B/12B/27B. COCOcap 116, DocVQA 85.6, MMMU 56.1 at 27B. Weaker per
  parameter than the two above, but the **widest runtime support** of anything here.
  ([codersera](https://codersera.com/blog/gemma-3-vs-qwen-3-in-depth-comparison-of-two-leading-open-source-llms/))
- **Ovis2.5** — 2B and 9B. **OpenCompass 73.9 at 2B**, 78.3 at 9B; claimed SOTA under 40B.
  The strongest score-per-parameter in the survey.
  ([tech report](https://huggingface.co/papers/2508.11737))
- **MiniCPM-V 4.5** — 8.7B, OpenCompass avg 77.0, ~6 GB, strong OCR/hi-res/video.
- **SmolVLM2** — 256M / 500M / 2.2B. Not competitive on quality; it is the *floor*, SOTA
  for memory footprint, and the 256M runs under 1 GB.
  ([HF blog](https://huggingface.co/blog/smolvlm))
- **Moondream 3**, **FastVLM** (Apple, CVPR 2025 — FastViTHD, **85× faster time-to-first
  token** than LLaVA-OneVision-0.5B). Both interesting for latency specifically.
  ([FastVLM](https://github.com/apple/ml-fastvlm))

**Depth and spatial reasoning** — the user asked for this explicitly, and the answer is a
negative worth stating plainly.

- **SpatialBot** (ICRA 2025) is the real thing in this space: it beats GPT-4o on
  depth-understanding tasks. But it works by **taking a depth image as a second input
  channel**, trained on SpatialQA/SpatialQA-E.
  ([arXiv 2406.13642](https://arxiv.org/abs/2406.13642))
- Same shape for **SpatialVLM**, **RoboRefer**, **SD-VLM**: extra depth channel, or an
  RGB-D transformer.
- The benchmarks (**SPATIALAB**, SpatialRGPT-Bench, MV-RoboBench, Ego3D-Bench,
  OmniSpatial) agree that 25+ SOTA VLMs sit well below human baselines on spatial
  reasoning. ([arXiv 2601.09954](https://arxiv.org/pdf/2601.09954))

**Why none of them is a candidate.** `camera_nav` is a monocular RGB camera and the arc
menu is drawn in image space; there is no depth map in the pipeline to hand a depth-input
model, and producing one would add a monocular-depth network to the loop and change the
input distribution for every candidate at once — a different experiment. Separately, every
one of these is pinned to `trust_remote_code` against a `transformers` 4.3x-era API and
fails §3's gate outright. **The practical depth channel in this survey is Qwen3-VL, whose
release notes claim enhanced spatial comprehension as a headline feature** — measured
here in §5's `avoid wall` and `open side` columns, which is what depth perception cashes
out as for this policy.

---

## 3. The gate that decided the shortlist, and it is not quality

Ranking by benchmark score picks a list that mostly does not load. Checked directly
against the installed stack — `AutoConfig`'s `model_type` against
`transformers.models.auto.CONFIG_MAPPING_NAMES` on **transformers 5.16.1**:

| model | `model_type` | native on 5.16.1? |
|---|---|---|
| `Qwen/Qwen3-VL-{2,4,8}B-Instruct` | `qwen3_vl` | **yes** |
| `google/gemma-3-{4,12}b-it` | `gemma3` | **yes** (gated repo — needs the HF token) |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `smolvlm` | **yes** |
| `OpenGVLab/InternVL3_5-4B` | `internvl_chat` | **no** |
| `OpenGVLab/InternVL3_5-4B-**HF**` | `internvl` | **yes** |
| `AIDC-AI/Ovis2.5-2B` | `ovis2_5` | no (`ovis2` exists; `ovis2_5` does not) |
| `openbmb/MiniCPM-V-4_5` | `minicpmv` | no |
| `allenai/Molmo-7B-D-0924` | `molmo` | no |
| `moondream/moondream3-preview` | `moondream3` | no |
| `apple/FastVLM-1.5B` | `llava_qwen2` | no |
| `RussRobin/SpatialBot-3B` | — | gated |

Three things this table is worth keeping for:

- **The `-HF` suffix is the whole difference for InternVL3.5.** `OpenGVLab/InternVL3_5-4B`
  is OpenGVLab's own repo format and needs `trust_remote_code`; `…-4B-HF` is the converted
  checkpoint and loads natively. Same weights, same benchmark numbers, one loads.
- **`ovis2` being present does not mean Ovis2.5 loads.** The `.5` is a different
  `model_type`. A name-level check would have passed it and it would have failed at load
  — the same shape of error as grepping for a constant's name instead of its value.
- **Ovis2.5 has the best score-per-parameter in §2 and is not in the five.** That is the
  cost of the gate, stated rather than hidden: the selection below is the best *loadable*
  set, not the best set.

---

## 4. The five, and why these five

| # | candidate | params | lineage | what it is here to answer |
|---|---|---|---|---|
| 1 | `Qwen/Qwen3-VL-8B-Instruct` | 8 B | Qwen | **the ceiling.** Strongest small model in every source. If nothing beats the 27B, this is the one that had the best chance. |
| 2 | `Qwen/Qwen3-VL-4B-Instruct` | 4 B | Qwen | **size, isolated.** Same family and same prompt handling as the baseline, ~7× fewer parameters. Any gap here is size and not lineage. |
| 3 | `OpenGVLab/InternVL3_5-4B-HF` | 4 B | InternVL | **lineage, isolated.** Held at 4 B against #2. Also the family TIC-VLA already uses (at 1 B), so a win here is directly reusable. |
| 4 | `google/gemma-3-4b-it` | 4 B | Gemma | **a third lineage at the same size,** and the weakest-on-paper of the three. Its job is to say whether the 4 B tier's result is a property of the tier or of one vendor. |
| 5 | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 2.2 B | SmolVLM | **the floor.** Fastest and smallest. If it scores at chance, that bounds how far down this task can be pushed; if it does not, that is the headline. |

The shape is deliberate: **three different lineages held at 4 B**, with an 8 B ceiling and
a 2.2 B floor bracketing them. That answers "does size matter here or does family matter
here", which a list of five different sizes cannot.

What is deliberately *not* here: a second Qwen at 2 B (three of five slots on one family
buys less than a third lineage), and any quantised build (the baseline's FP8 is a memory
decision on a 27 B; at 4 B bf16 fits in 8 GiB and quantising would add a second variable).

---

## 5. The test

**The probes are the benchmark, unmodified.** `nav/tools/probe_arc_selection.py` and
`nav/tools/probe_arc_repair.py` reach the model over HTTP at `--port`. They own the
prompts, render the menus themselves from the same 881-frame capture under the same seed,
and shuffle the arc labels the same way. Point them at **8766** and they measure
Qwen3.8-27B-FP8; point them at **8767** and they measure a candidate.
`nav/policy_server/server_vlm.py` serves `/raw` with the same schema on 8767. Nothing else
changes, so the only difference between the two numbers is the model.

Three parity details live in the server rather than the probes, because getting one wrong
would mean measuring something else:

- **pixels** — `QVLA_MAX_PIXELS = 200704` (448×448), the baseline's cap. A candidate left
  at its own default would be reading a different picture.
- **decoding** — greedy, matching `_SAMPLING` with `QVLA_TEMPERATURE` unset.
- **one card** — a model split across both GPUs runs different kernels than the same model
  on one. Measured on the FP8 baseline, where spanning devices routes linear layers off
  DeepGEMM. Every candidate is loaded on GPU0 with `CUDA_VISIBLE_DEVICES=0`, which is safe
  here for the same reason it is safe in `server_qwen.py`: this process never starts Kit.
  GPU1 stays with the running 27 B, so the baseline does not have to be unloaded to be
  compared against.

### The four columns that matter

From `probe_arc_repair.py`, and the baseline's own numbers are the target to beat — **all
of them re-measured on 2026-09-10, because the labels this probe used had drifted**:

| variant | avoid wall | keep straight | open side | mirror | latency |
|---|---|---|---|---|---|
| DIGIT — one call, bare digit | 8% | 100% | 25% | 0% | 0.32 s |
| **SIDED** — describe *and name the open side*, then choose | **67%** | **100%** | **69%** | **50%** | 2.30 s |

**This is a different frame set, not a regression.** `CLAUDE.md` records SIDED at
100/92/90/83, measured 2026-08-31; `/tmp/alohamini-nav-frames` was regenerated on
2026-09-05 from a different splice of episodes and the hand labels were never rewritten,
so for five days the probe scored choices against descriptions of pictures that no longer
existed. See `nav/config/probe_frames.json`, which now carries a SHA-16 of every frame it
labelled and refuses to run on a mismatch. The new six blocked frames span three
environments and include a staircase taken nose-on; they are simply harder than the old
set. **A number measured against different ground truth is not comparable to one measured
against this ground truth, in either direction** — the only baseline the candidates are
scored against is the one in the table above, taken in the same session, on the same
frames, through the same probes.

Chance is 43% on `open side` and ~0% on `mirror`. **`mirror` is the column to read first**
— it asks whether the choice negates when the frame is mirrored, needs no human labels,
and cannot be scored by a positional prior. The original obstacle probe scored 0/18 on it
while looking fine elsewhere.

And from `probe_arc_selection.py`: left / right / straight at **100%** each against 43%
chance, mean κ separating left from right by **+1.190** of a 1.20 menu span, median
latency **0.32 s**. That probe renders its own scenes from instructions and does not touch
the hand labels, so it was unaffected by the drift — and it reproduced the recorded
+1.190 exactly, which is the check that says the harness itself is sound.

### What "beats Qwen" would mean

Three separate ways a candidate can win, and they are not the same win:

1. **Quality** — a higher `mirror` or `open side` at the same variant. This is the one
   that would change the policy.
2. **Latency at parity** — the same scores materially faster. At 2.10 s per SIDED decision
   the 27 B is a large part of the control period; a 4 B at a few hundred ms is a
   different robot even at identical quality.
3. **Footprint at parity** — 28.75 GiB → ~8 GiB frees GPU1 entirely, which is the
   difference between needing two cards and needing one.

A candidate that ties on quality and wins on 2 and 3 is a **win**, and the report has to
say so rather than reporting a single scalar.

### Reading the result honestly

- **n is small and this stack is noisy.** Three clean prior ladders scored 2/13, 8/13,
  8/13 on the *same* configuration. A few points of difference on a 24-scene probe is not
  a finding. The probes are the right instrument precisely because they are deterministic
  and controlled where a ladder is not.
- **A model that refuses the format is not a model that cannot navigate.** If a candidate
  narrates instead of answering with a digit, that is a parse failure, and it has to be
  reported as its own column rather than folded into a low score.
- **Do not average the four columns.** `keep straight` at 100% with `open side` at 20% is
  the exact signature of a model that always picks the centre arc — a positional prior,
  not comprehension. The baseline's own DIGIT row is that failure.

---

## 6. The result

All five loaded, all five ran both probes, **0 parse failures and 0 unparsed answers
anywhere** — 6 models × 2 probes × 288 calls. One dependency was missing (`num2words`,
required by SmolVLM's processor and not by any other); nothing else needed a patch, a
`trust_remote_code`, or a prompt change. Measured 2026-09-10, candidates on GPU0 at
`:8767`, the 27 B on GPU1 at `:8766`, same frames, same seed, same 448×448 cap, greedy.

### 6.1 Does the WORD reach the CURVE? (`probe_arc_selection`)

This is the job the arc policy actually runs at 100% and the one the whole architecture
rests on. Chance for a side-correct pick is 43%.

| model | GiB | left | right | straight | L−R mean κ | latency |
|---|---|---|---|---|---|---|
| **`baseline-qwen27b`** | 28.75 | **100%** | **100%** | **100%** | **+1.190** | 0.32 s |
| `qwen3vl-8b` | 16.33 | **100%** | **100%** | **100%** | +0.744 | **0.09 s** |
| `qwen3vl-4b` | 8.27 | **100%** | 96% | **100%** | +0.554 | **0.06 s** |
| `internvl3.5-4b` | 8.82 | 88% | 54% | 33% | +0.298 | 0.21 s |
| `smolvlm2-2.2b` | 4.19 | 71% | 42% | 4% | +0.129 | 0.20 s |
| `gemma3-4b` | 8.01 | 25% | 79% | 38% | +0.065 | 0.11 s |

**Two candidates pass and they are both Qwen3-VL.** The 8 B matches the baseline's
100/100/100 at **3.6× the speed and 1.76× less memory**; the 4 B gives up four points on
`right` for **5.3× the speed and 3.5× less memory**. Both keep the sign separation well
clear of zero — the menu spans 1.20, so +0.744 and +0.554 are real, if softer than the
27 B's +1.190, which is nearly the whole span.

The other three fail, and each fails differently. `gemma3-4b` is the flattest: +0.065 of
separation with `right` at 79% and `left` at 25% is not a weak left/right map, it is a
right-ish prior that scores whenever the answer happens to be right. `smolvlm2-2.2b`
answers the **same label for every instruction** on most scenes — its per-scene rows read
`neutr=5 left=5 right=5 strai=5` — so its 71% on `left` is what a fixed answer scores
against a shuffled menu, not comprehension. `internvl3.5-4b` separates the sides (+0.298)
but scores `straight` at 33%, which is the instruction that should be easiest.

### 6.2 Can it see free space? (`probe_arc_repair`, SIDED)

| model | params | GiB | avoid wall | keep straight | open side | mirror | latency |
|---|---|---|---|---|---|---|---|
| **`baseline-qwen27b`** | 27.0 B | 28.75 | **67%** | **100%** | **69%** | **50%** | 2.30 s |
| `smolvlm2-2.2b` | 2.25 B | 4.19 | 89% | 11% | 61% | 56% | 0.54 s |
| `gemma3-4b` | 4.3 B | 8.01 | 53% | 53% | 61% | 39% | 0.89 s |
| `internvl3.5-4b` | 4.73 B | 8.82 | 3% | 100% | 44% | 22% | 0.84 s |
| `qwen3vl-8b` | 8.77 B | 16.33 | 39% | 89% | 36% | 28% | 0.36 s |
| `qwen3vl-4b` | 4.44 B | 8.27 | 22% | 94% | 36% | 22% | 0.41 s |

**Nothing beats the baseline here, and the ways of failing are worth more than the
ranking.** Three distinct shapes, none of them "a bit worse":

- **Centre-arc prior** — `internvl3.5-4b` (100% keep, 3% avoid) and `qwen3vl-4b` (94/22).
  It picks the middle arc and scores `keep straight` perfectly by doing nothing. This is
  the baseline's own DIGIT failure, and it is why these columns must never be averaged.
- **Edge-arc prior** — `smolvlm2-2.2b` (89% avoid, **11% keep**). Read the first two
  columns alone and it beats the 27 B on wall avoidance; read the third and it is a model
  that swerves at everything, including a clear corridor. **This is exactly what the
  open-corridor frames were put in for**, and without them SmolVLM2 would have been
  written up as the surprise winner of this survey. Its 56% `mirror` beating the
  baseline's 50% is the same illusion: a model that always turns *away* from image mass
  negates when the image is mirrored without seeing anything.
- **Near chance across the board** — `gemma3-4b` (53/53/61/39). No prior, no signal.

`avoid` and `keep` trade against each other along one axis — where a model sits on it is a
bias, and only `open side` and `mirror` say whether anything was read off the picture. On
those two the baseline leads every candidate by 8–33 pp and 6–28 pp, with the single
exception of SmolVLM2's mirror, explained above.

### 6.3 So is there anything better than Qwen? — yes, on one of the two jobs

**Split the question, because the answer splits.**

| the job | winner | margin |
|---|---|---|
| turn an instruction into an arc (`selection`, DIGIT path) | **`qwen3vl-4b`** / `qwen3vl-8b` | same accuracy, **5.3× / 3.6× faster**, **3.5× / 1.76× lighter** |
| read free space off the frame (`repair`, SIDED path) | **the 27 B, unbeaten** | +8…33 pp on `open side`, +6…28 pp on `mirror` |

The 4 B answers a menu pick in **0.06 s against 0.32 s**, in **8.27 GiB against 28.75** —
which frees a whole card, since the 27 B needs all of GPU1 and the 4 B does not. On the
selection probe that costs four points on one instruction out of six.

**Why this is not a small finding for this stack:** the repair probe's own verdict, on
every model including the baseline, is *"filter the menu geometrically before it is drawn,
and leave the model the job it does at 100%: choosing a direction from the instruction."*
That filter already exists — `SweepingLidar2D` feeds a per-arc clearance filter on the
VLM's menu (see `CLAUDE.md`). In the configuration this repo actually runs, the model is
asked to do the job the 4 B does at parity and not the job only the 27 B can do.

**What this does NOT establish.** Both probes are open-loop, deterministic, 144 and 288
calls, and they measure a decision, not an episode. Nothing here has driven a robot. The
13-episode ladder is the test that would settle it, and the honest prior is that this
stack is noisy: three clean ladders of the *same* configuration scored 2/13, 8/13, 8/13.
A swap that looks free at 0.06 s could still lose episodes to the softer κ separation
(+0.554 against +1.190) once a controller integrates those choices over 30 m. **Run the
ladder before believing the table.**

Two smaller results worth keeping:

- **Size is not the axis.** The 8 B is better than the 4 B on selection (+0.744 vs
  +0.554) and no better on free space (36% side, both). The 2.2 B and the three 4 Bs span
  the entire range of outcomes between them. What separates the passes from the failures
  here is the *lineage* — both passes are Qwen3-VL — not the parameter count, which is
  the question holding three lineages at 4 B was designed to answer.
- **On-disk size is not the inference footprint.** SmolVLM2 ships fp32: 8.4 GiB on disk,
  larger than any 4 B here, and 4.19 GiB once loaded in bf16.
