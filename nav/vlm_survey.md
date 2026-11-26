# Can a small VLM drive the arc menu? — survey, selection, and what the swap has to pass

The navigation policy on this branch is a frozen **Qwen3.8-27B-FP8** that never sees a
trajectory, never gets fine-tuned, and answers with an integer. `nav/arc_menu.py` draws
candidate paths on the robot's own camera frame, and the model picks one. That is the
whole policy, and it scores **8/13** on the ladder against TIC-VLA's 6/13.

A 27B holding **28.75 GiB** is a strange thing to be at the bottom of a robot. This
document asks whether something an order of magnitude smaller can do the same job, picks
five candidates, and defines the test.

**The answer, up front: no — and the interesting part is that it splits.** `qwen3vl-4b`
matches the 27 B at turning an instruction into an arc, at 5.3× the speed in 3.5× less
memory, and is far worse at seeing free space (§6). Put on the real ladder it scores
**4/19 against 10/19**, with guard interventions up **10.6×** through exactly the channel
the open-loop probe flagged (§7). A 288-call probe costing four minutes predicted a
four-hour ladder, which is the reusable result here.

**§8 and §9 then take the four published ObjectNav systems as far as each one ports** —
VLFM's BLIP-2 value function as weights, WMNav's, OpenFMNav's and L3MVN's contributions as
prompts, all on the same twelve labelled frames and the same four columns. None of them
replaces the baseline. One thing survives and is free: **naming the goal in the instruction
doubled obstacle avoidance** (§9.1).

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

**Why this looked like a bigger finding than it is — and the correction that decides it.**
The repair probe's own verdict, on every model including the baseline, is *"filter the menu
geometrically before it is drawn, and leave the model the job it does at 100%: choosing a
direction from the instruction."* That filter is written — `SweepingLidar2D` feeds a
per-arc clearance filter on the VLM's menu — and **it is switched off in the configuration
every number in this repo was measured under.** `nav/config/profiles/baseline.yaml` pins
`NAV_LIDAR: "0"`, so `run_navigation.py:622` never constructs the sensor and `:957` sends
`scan_points=None`; with no scan the server has nothing to compute clearance from and the
menu is drawn whole. That is deliberate — it is the arm that keeps every earlier ladder
number comparable — but it inverts the reading. The shipping policy does **not** hand the
geometry to a filter and keep only the instruction for the model. It asks the model for
both, which makes the SIDED column load-bearing, and SIDED is the column where all five
candidates lose. So the honest prediction going into the ladder is not "the swap is free":
it is that the 4 B should lose episodes, in proportion to how often an episode needs free
space read off the frame rather than a direction read off a sentence.

**What this does NOT establish.** Both probes are open-loop, deterministic, 144 and 288
calls, and they measure a decision, not an episode. Nothing here has driven a robot. The
13-episode ladder is the test that would settle it, and the honest prior is that this
stack is noisy: three clean ladders of the *same* configuration scored 2/13, 8/13, 8/13.
A swap that looks free at 0.06 s could still lose episodes to the softer κ separation
(+0.554 against +1.190) once a controller integrates those choices over 30 m. **Run the
ladder before believing the table.** — It was run; §7 is the result, and it is 4/19
against the baseline's 10/19.

Two smaller results worth keeping:

- **Size is not the axis.** The 8 B is better than the 4 B on selection (+0.744 vs
  +0.554) and no better on free space (36% side, both). The 2.2 B and the three 4 Bs span
  the entire range of outcomes between them. What separates the passes from the failures
  here is the *lineage* — both passes are Qwen3-VL — not the parameter count, which is
  the question holding three lineages at 4 B was designed to answer.
- **On-disk size is not the inference footprint.** SmolVLM2 ships fp32: 8.4 GiB on disk,
  larger than any 4 B here, and 4.19 GiB once loaded in bf16.

---

## 7. The ladder — `qwen3vl-4b` in the closed loop

The table above says run the ladder before believing it, so the ladder was run: the full
19 episodes, `braking`, the pinned baseline profile with **exactly one variable changed**
(`nav/tools/profile.py check` confirmed the live `/health` deviated from
`nav/config/profiles/baseline.yaml` in `model` and nothing else). The candidate is the
open-loop winner, `qwen3vl-4b`. Server counters over the ladder: **4910 predictions, 4910
generations, 0 parse failures, 0 generation errors.**

| | 27 B baseline | `qwen3vl-4b` |
|---|---|---|
| passed | **10 / 19** | **4 / 19** |
| indoor | **9 / 13** | 3 / 13 |
| outdoor | 1 / 6 | 1 / 6 |
| guard interventions | **1 369** | **14 474** (10.6×) |
| mean gap closed (14 rows clean on both arms) | **85%** | 56% |
| physics blow-ups | 0 | **5** |
| ladder wall time | 2.96 h | 4.06 h |

Episode by episode, the newest run of each arm (baseline ladder `20260904-232005 ..
20260905-022414`, candidate `20260910`):

| episode | 27 B | `qwen3vl-4b` | flip | guard 27 B | guard 4 B |
|---|---|---|---|---|---|
| `office_nearest_elevator` | ✓ 1.50 m | 6.04 m | **LOST** | 0 | 0 |
| `hospital_down_hallway` | ✓ 1.50 m | 28.12 m | **LOST** | 0 | 838 |
| `hospital_down_hallway2` | ✓ 1.50 m | 24.49 m | **LOST** | 0 | 283 |
| `office_passing_hallway` | ✓ 1.50 m | 5.02 m ✱ | **LOST** | 4 | 270 |
| `hospital_vending_machine` | 1.89 m | **✓ 1.50 m** | WON | 0 | 0 |
| `hospital_vending_machine2` | ✓ 1.50 m | ✓ 1.50 m | = | 0 | 0 |
| `office_hallway_turn` | 2.83 m | 2.12 m | = | 321 | 777 |
| `office_hallway_turn2` | ✓ 1.49 m | 2.41 m | **LOST** | 23 | 410 |
| `hospital_past_wheelchairs` | ✓ 1.50 m | ✓ 1.49 m | = | 0 | 0 |
| `hospital_forward_staircase` | ✓ 1.50 m | 18.08 m | **LOST** | 9 | 914 |
| `hospital_exit_room` | 2.28 m | 17.35 m | = | 562 | 822 |
| `warehouse` | ✓ 1.50 m | 14.65 m | **LOST** | 0 | 748 |
| `warehouse_aisle6` | 3.68 m | 19.25 m | = | 450 | **9 410** |
| `outdoor_umbrellas` | ✓ 3.99 m | 11.57 m ✱ | **LOST** | 0 | 0 |
| `outdoor_pillars` | 6.06 m | 29.02 m ✱ | = | 0 | 0 |
| `outdoor_library` | 14.29 m | **✓ 4.00 m** | WON | 0 | 0 |
| `outdoor_upsway` | 2.86 m | 3.69 m | = | 0 | 2 |
| `outdoor_upsway_far` | 9.43 m | 2.27 m ✱ | = | 0 | 0 |
| `outdoor_ramp_fountain` | 14.17 m | 27.60 m ✱ | = | 0 | 0 |

✱ = physics blow-up; every distance on that row is void (see below).

### 7.1 Read the guard column, not the pass count

**The pass count alone does not settle this, and saying it does would be the same error
this document has warned about twice.** Three clean prior ladders of one configuration
scored 2/13, 8/13 and 8/13 indoor. The candidate's 3/13 sits *inside* that spread. Two
episodes flipped its way, eight against; n=1 per arm, and `predict()` being deterministic
does not make a run deterministic.

What is outside the noise is the number that does not pass through a threshold at all.
**Guard interventions went 1 369 → 14 474.** The guard is a raycast that fires when the
robot is about to drive into something; it does not know what a goal is, it cannot be
gamed by stopping early, and it counts events rather than episodes, so 19 runs give it
thousands of samples instead of 19. A 10.6× rise says the robot is being steered into
obstacles an order of magnitude more often. `warehouse_aisle6` alone fires **9 410**
times against 450.

And that is exactly the channel §6.2 measured. SIDED `avoid wall` is **67% → 22%**: on
the open-loop probe the 4 B drove the straight arc into a visible wall on 78% of blocked
frames. §6.3, corrected, predicted this would be load-bearing rather than absorbed by a
geometric filter, because the shipping profile pins `NAV_LIDAR: "0"` and the filter never
runs. **The 288-call probe predicted the 19-episode result through the right variable** —
which is the most useful thing in this document, since the probe costs four minutes and
the ladder costs four hours.

The continuous score agrees and is not threshold-bound: mean gap closed **85% → 56%**
over the fourteen rows clean on both arms.

### 7.2 The five blow-ups are vertical, and they are not scored as navigation

Five candidate rows record impossible paths — `office_passing_hallway` 17 348 m,
`outdoor_umbrellas` 20 046 m, `outdoor_pillars` 34 599 m, `outdoor_ramp_fountain`
19 781 m, and `outdoor_upsway_far` **194 378 m**. `summarize_runs.score()` flags all five
against what the speed cap physically permits, so their distance aggregates are excluded
above; the pass denominator keeps them, because the robot really did not arrive.

The divergence is in **z**, not in the plane. `base_z_span_m` on those rows is 11.7 km,
15.3 km, 19.8 km, 23.4 km and **972 km**, with single steps up to 492 m, against
0.003–0.03 m on every clean row. So this is the kinematic base's known vertical drift
(CLAUDE.md: teleporting the root has no contact response, and refreshing z every step
caused drift), not a steering failure — the same class as the deferred `warehouse_aisle6`
152 995 m case.

**What is new is the rate: 5 in 19, against 1 in 57 across three 27 B ladders.** Four of
the five are outdoors, where the guard's planar fan at 0.30 m cannot see a kerb, a ramp
or a step at all. The plausible chain is that the candidate reaches such geometry far
more often — which is what the guard column says about the geometry the fan *can* see —
and a kinematic base driven onto it climbs. That chain is a hypothesis, stated as one:
nothing here isolates it, and the blown rows themselves have guard 0 precisely because
the obstacle was below the rays.

Note the arithmetic of it: `outdoor_upsway_far` reached **2.27 m** on a 2.0 m threshold —
27 cm short — and then flew. Whether that row is "nearly a pass" is unanswerable, which
is why it is scored as a fail and its distances are void rather than quoted.

### 7.3 The honest answer to the question this branch asked

**Is there a lightweight VLM that beats Qwen here? On the whole job, no.**

- **Turning a word into an arc**, `qwen3vl-4b` matches the 27 B at 100/96/100 for **5.3×
  the speed and 3.5× less memory**. That result stands, and it is worth having: it means
  the *instruction* half of this policy does not need 28.75 GiB.
- **Reading free space off the frame**, it does not come close — and in the configuration
  that actually ships, that half decides the ladder. 10/19 → 4/19, guard ×10.6.

So the swap is not free and the survey's own §6.3 caveat was the right one. What the result
actually supports:

1. **`qwen3vl-8b` on the same ladder.** It is better than the 4 B on both probes (avoid
   39% vs 22%, κ +0.744 vs +0.554) at 16.33 GiB, still half the baseline and still
   single-card. It was not run here because the 4 B was the open-loop winner and the
   ladder costs four hours; the 4 B's result makes the 8 B the interesting row rather
   than a redundant one.
2. **Nothing that needs a second sensor.** An earlier draft of this section put
   `NAV_LIDAR=1` first, on the argument that the geometric filter does the exact job the
   4 B fails. Fouad has ruled that line out, so the free-space deficit has to be closed
   from the camera or not at all — which is the question §8 goes and tests against the
   published alternative.

Not supported by anything measured: prompt changes. §6.2's failure is a perceptual one on
identical pixels and identical wording to the model that gets it right.

---

## 8. The four ObjectNav systems, and the one piece of them that runs here

Fouad asked for four more:

| System | SR | SPL | What it uses |
| --- | --- | --- | --- |
| WMNav | 58.1 | 31.2 | VLM world model + curiosity value map |
| OpenFMNav | 54.9 | 24.4 | LLM + VLM detector |
| VLFM | 52.5 | 30.4 | BLIP-2 + value map |
| L3MVN | 50.4 | 23.1 | LLM + semantic map |

**None of the four runs against this stack as published, for the same three reasons each
time.** They are Habitat agents: a discrete action space (`move_forward` 0.25 m, turn ±30°,
`stop`), episodes out of HM3D/MP3D, and a loop that is `habitat.Env.step()`. They are all
RGB-**D**: every one of them builds its map by projecting a depth image, and `camera_nav`
publishes RGB only. And they are a **different task** — ObjectNav succeeds by stopping
within 1 m of *any* instance of a category, where our episodes name one coordinate with a
per-episode `success_threshold_m`. A system that wins by finding any chair is not
measurable on an episode that names one place.

VLFM is the hardest of the four to port even so, because its low-level executor is a
*trained* PointNav (VER/DDPPO) checkpoint whose observation space is depth plus a goal
vector. That is weights, not a prompt, and it cannot be handed something else.

**One warning about the numbers in that table.** Our baseline is 10/19 = 52.6%, which lands
between VLFM's 52.5 and OpenFMNav's 54.9. That is a coincidence between two different
tasks. Nothing below compares to those SRs and neither should anything else.

### 8.1 What is portable as a MODEL, and it is exactly one thing

(Prompts port too, and §9 runs three sets of them. This section is about weights.)

VLFM's **semantic value function** is RGB-only and separable from the rest of it: a BLIP-2
ITM cosine similarity between the current frame and a prompt naming the target, painted
into a top-down map through a cone mask over the camera's FOV, with frontiers then ranked
by that value. Depth enters only to build the occupancy map the frontiers come from and to
trim the occluded part of the cone. Strip those and what is left is a pure function
`(RGB, text) -> float` — no Habitat, no depth, no map, no PointNav head.

`Salesforce/blip2-itm-vit-g`, 4.4 GB on disk, **1.17 B parameters**, loads in 2.6 s. That is
its own result: it is 24× smaller than the 27 B baseline.

`nav/tools/probe_value_map.py` measures it on our frames with the arc probes' own rules.
Three readings, in increasing distance from VLFM as published:

- **FRAME** — VLFM unmodified. One value for one whole frame; every blocked frame paired
  against every open frame, asking whether the open one scores higher.
- **SECTOR** — the adaptation, and the only invented part. VLFM gets directional resolution
  by *turning*: score, yaw 30°, score again. With one frame per moment we crop instead — a
  640 px window, **30.0° wide, one Habitat turn step** — centred on each of the seven arcs'
  badge columns, which is the pixel each arc's number was actually drawn at. The argmax
  window names an arc and that arc's κ is scored with `probe_arc_obstacles.py`'s metrics.
- **MIRROR** — the label-free control. Flip the frame and a scorer that reads the image must
  negate its chosen κ. The badge columns are symmetric about the optical axis, so a fixed
  window index returns the *same* κ both ways: a positional prior scores 0% here, not 43%.
  It is cheap enough to run on frames nobody labelled, so it is the only well-powered
  column in the section — 56 frames rather than 12.

The prompt keeps VLFM's sentence frame and swaps the noun, because our task has no object
category: `"Seems like there is a clear path ahead."` against
`"Seems like there is a wall blocking the way ahead."` Both BLIP-2 heads and both the
positive score and the contrast are reported — four readouts off the same forward passes,
all printed rather than the best one quoted.

### 8.2 The result

Both columns were measured on the **same twelve frames** (see §8.4 — the old ones no longer
existed and the set was rebuilt, so the 27 B was re-run rather than quoted).

| | 27 B arc selector | BLIP-2 value function |
| --- | --- | --- |
| ranks a blocked frame below an open one | **91%** (64/70 forced choice) | **75%** (27/36 pairs) |
| avoided the straight arc on blocked frames | 11% (n=36) | **92%** (n=12) |
| stayed straight on open frames | **100%** (n=36) | 33% (n=12) |
| turned toward the labelled open side | 31% (n=36) | 42% (n=12) — chance is 43% |
| choice negates when the image is flipped | **0/18 = 0%** | **47/56 = 84%** |
| picks the same window both ways (a prior) | — | 1/56 = 2% |

Across the four readouts the frame-level number spans 58–75% and the mirror sweep spans
48–84%; `itc / clear` — the cosine head with the positive prompt, which is what VLFM
actually calls — is the best on the 56-frame sweep and is the column quoted. It is chosen
on the sweep and not on the six labelled frames on purpose: `itm / clear-wall` scored 6/6
there and 70% over 56, which is what picking a winner off n=6 buys you.

### 8.3 They fail in opposite directions, and that is the finding

The 27 B **sees the wall and drives into it**. It names the obstacle in free text on 4 of 6
blocked frames ("a large, black, reflective panel"), separates blocked from open at 91%,
and then picks the straight arc anyway on 89% of blocked frames, does not change its answer
when the image is mirrored (0/18), and drives at the wall 100% of the time when told to.
The percept exists and never reaches the choice.

The value function **reads the image and swerves at the wrong thing**. 84% of flips negate
its choice and only 2% return the same window — so it is genuinely scoring pixels, which is
the one thing the 27 B provably is not doing. But it swerves off 92% of blocked straights
*and* 67% of open ones, and finds the labelled open side 42% of the time against 43%
chance. That is the always-swerves model the OPEN frames exist to catch. It is measuring
something real and that something is not free floor — which is consistent with what the
function is for: BLIP-2 ITM was trained to score *semantic* agreement between a picture and
a caption, and VLFM uses it to decide which frontier looks like it leads to a sofa, not
which bearing is drivable. The drivability question is answered in VLFM by the occupancy
map, and the occupancy map is built from depth.

The two failures do not compose. Adding a "swerve toward salience" signal to a policy that
always goes straight gives a policy that swerves toward salience.

**So: no, none of the four beats the baseline here, and the reason is not that they are
weak.** VLFM's one portable component is a semantic scorer being asked a geometric
question. The other three cannot be run *as systems* without Habitat and depth — but that
is a statement about their maps and their loops, not about their prompts, and §9 runs the
prompts.

### 8.4 The labelled frames had rotted, again, and fingerprints were not enough

`probe_value_map.py` refused on its first run: eight of the twelve labelled frames had
changed SHA. `/tmp/alohamini-nav-frames` had been written into twice more on 2026-09-10 and
had ended up holding three episodes layered by index — 0–400, 401–600, and a surviving
601–880 — so the hospital atrium frames the manifest described were now an outdoor strip
mall and a grassland with the camera tumbling through a physics blow-up.

The fingerprints did their job; they refused instead of scoring against pictures that no
longer existed, which is the exact failure they were added for. But refusing is not
running. So:

- the twelve frames now live in **`nav/config/probe_frames/`**, in git, next to the labels
  that describe them, and `frame_dir` is read out of the manifest instead of defaulting to
  `/tmp` — which also closes a second gap, where the SHA check could pass against one
  directory while `render_menu` drew on another;
- the set was rebuilt inside the one intact block, **3 frames open-left and 3 open-right**
  so a constant side bias scores exactly 50%, with 680/700/800/880 carried over
  bit-identical and keeping their original descriptions;
- and every 27 B number in §8.2 was **re-measured**, not quoted. It reproduced: avoid 11%,
  keep straight 100%, mirror 0/18, against 22/100/20/0 and 8/100/0/0 on the two earlier
  captures. That the new labels reproduce the known behaviour is the best evidence
  available that they are sound.

**One honest limit on the blocked set.** On 790 and 880 the obstacle fills a side rather
than the centre, and the 27 B's free text called both "clear" — arguably correctly. They
are kept because they were labelled before any model answer was seen, and moving a label
after reading the scores is how a benchmark stops being one. It does mean the "avoided the
straight arc" column is measured against a set where two of six frames have a defensible
straight.

**What was not tested**, and should not be read into §8.2: VLFM's occupancy map, its
frontier detection, its PointNav executor, or its value function on the task it was built
for. The claim here is narrow — BLIP-2 ITM does not supply the free-space channel §6.2
found missing — and it is the claim that decides whether porting the rest is worth four
hours.

## 9. WMNav, OpenFMNav and L3MVN, run — their prompts, our frames

§8 stopped at "three of the four cannot be run here," which is true of the *systems* and
was doing too much work. WMNav, OpenFMNav and L3MVN each contribute a **prompt** and a
scoring loop around a frozen model, and a prompt ports perfectly: it is text, it costs
nothing to move, and our policy server already runs a model for it to talk to. The Habitat
loop and the depth-built map are what does not port.

So `nav/tools/probe_objectnav_prompts.py` runs their prompts against the **same twelve
labelled frames** in `nav/config/probe_frames/`, scored with `probe_arc_obstacles.py`'s
four columns, on the same 27 B server, in one sitting. Every prompt is copied out of the
repository that published it — including WMNav's doubled `{{'action': ...}}` braces and
its "acheives" typo, because a tidied prompt is a different prompt.

All three are object-goal navigators whose prompts say NAVIGATE TO THE NEAREST *{goal}*,
and our frames name no object category. The goal noun is substituted as **`"open floor"`**,
once, everywhere, and stated here rather than buried.

### 9.1 The result

| arm | avoid wall | keep straight | open side | mirror | no answer | s/call |
| --- | --- | --- | --- | --- | --- | --- |
| *chance* | *57%* | *43%* | *43%* | *0%* | | |
| **qvla (ours)** — bare digit | 11% | **100%** | 31% | 0/18 = **0%** | 0/72 | **0.32** |
| **WMNav `action`**, verbatim | **63%** | 94% | **67%** | 3/12 = 25% | **8/72** | 23.4 |
| WMNav `action`, goal→neutral | 34% | 94% | 54% | 2/17 = 12% | 1/72 | 18.6 |
| **WMNav `predicting`** (value map) | **92%** | 83% | **92%** | 5/6 = **83%** | 0/24 | 21.2 |
| **OpenFMNav** `discover`→`scoring` | 91% | **18%** | 55% | 5/5 = 100% | 2/24 | 26.5 |

The menu arms are 36 blocked decisions, 36 open and 18 mirror pairs (12 frames × 2 flips ×
3 label shufflings). The two sector arms score one decision per frame per flip, so 12 / 12
/ 6 — **and that small n is the whole story of §9.2.**

Three things are real and survive everything below.

**WMNav's `action` prompt repairs the obstacle blindness that §6.2 found, using nothing but
wording.** Same model, same pixels, same shuffled labels: our bare-digit prompt avoids the
wall on 11% of blocked frames and WMNav's on 63%, and the mirror control goes 0/18 → 3/12.
This is not a surprise so much as an independent confirmation — WMNav's prompt makes the
model *talk before answering* ("First, tell me what you see… Lastly, explain which action
acheives that best"), which is precisely the CHAIN/SIDED repair §6.2 measured at 72–100%.
Worth noting their wording lands between our THINK (33%) and our CHAIN (72%).

**The goal noun is load-bearing, and it is carrying more than the scaffolding.** Swapping
only WMNav's goal sentence for our neutral "Drive safely." — leaving the arrow
description, the talk-then-answer structure, the `{'action': N}` format and the
closed-doors note untouched — **halves the avoidance, 63% → 34%.** Being told to navigate
*to open floor* makes the model look at the floor. That is a cheap, portable finding and it
is the opposite of the usual advice that a neutral instruction is the fair one.

**It is expensive, and 11% of the time it does not answer at all.** 23.4 s per decision
against our 0.32 s, ~70×, and **8 of 72** verbatim replies were still deliberating when a
700-token budget ran out. That is not a truncation artifact to be tuned away — it was
already raised from 320 after 320 cut every reply off mid-sentence — it is WMNav's prompt
being costly on a 27 B. Their published agent talks to Gemini, which has no such budget.
Note also that **action 0 ("turn around") was never chosen, 0 times in 144 decisions.**

### 9.2 The value map looks like the best thing in this survey, and the sweep says it is not

WMNav's `predicting` prompt — 0–10 per direction, with an explicitly geometric criterion
("If there is no visible way to move to other areas … assign a score of 0", where a way out
is "a turn in the corner, an open door, a hallway") — is the only arm anywhere in this
document that clears chance on **all four columns at once**: avoid 92%, keep 83%, open side
92%, mirror 83%. Nothing else, ours included, has done that.

That is 12 / 12 / **6** decisions. §8.2 recorded exactly this shape once already —
`itm/clear-wall` scored 6/6 on six labelled frames and 70% over 56 — so the mirror control
was re-run on **40 unlabelled frames** from the same intact capture block, which needs no
hand labels and so is cheap to power properly:

| WMNav `predicting`, negation under mirroring | rate |
| --- | --- |
| 6 labelled pairs | 5/6 = **83%** |
| **40 unlabelled frames** | 12/40 = **30%** |
| …of which returned a *bit-identical* answer both ways | 9/40 = 22% |

**So it does not hold.** For comparison on the same control, BLIP-2's `itc/clear` negates
**84% over 56 frames**. The 1.17 B model that §8 concluded was swerving at the wrong thing
is four times better at *reading the image at all* than the 27 B running WMNav's value
prompt. Read the two results together: BLIP-2 reliably sees pixels and scores the wrong
property; WMNav's prompt asks for the right property and the 27 B mostly does not see it.

The 22% identical column names the mechanism. The prompt saturates: on open scenes it
returns 10 for every direction, which averages to "straight" no matter which way the
picture faces. That is the same illness as the §6.2 open-set prompt that named all five
directions at recall 100% and precision 0 — a scorer that likes everything has ranked
nothing.

**One methodological note that changed a number by 25 points.** A plain argmax over a
saturated 0–10 score list returns the *first* maximum, which is one end of the menu, which
is a constant positional bias — the exact artifact the mirror control exists to catch.
Averaging the tied sectors' curvatures instead moved this arm from 67% to 92% avoid. Ties
are not a detail when the scale has eleven values and seven sectors.

### 9.3 OpenFMNav is instructed to delete the thing we are asking about

OpenFMNav is two stages and both were run, in order, on the same sector crops: `discover`
(a VLM names what is in view) then `scoring` (an LLM rates each area 0–1 for containing the
goal, from those names alone). Its `scoring` stage never sees an image — by design, and
that is fine. What decides the outcome is `discover`, whose **rule (1) is**:

> "…you should only include objects in the house, and avoid things that are part of the
> house, like ceiling, wall, floor, window etc"

and whose rule (4) drops doors as "common everywhere". Wall, floor, door: that is the
entire vocabulary our question is asked in. This is not a limitation the probe imposes —
it is correct for OpenFMNav, whose question is which *room* a sofa is likely to be in.

Measured over 168 discover calls, rather than argued from the prompt text:

- **119 / 168 = 71%** of sectors have the model name a wall, floor, ceiling or door in its
  own reasoning **and then explicitly exclude it** ("The floor, walls, ceiling, and door are
  structural parts of the building, so they should be excluded");
- of **310** surviving object mentions, **5** contain a structural noun at all and only
  **3** name the structure itself (`door` ×2, `window`) — the other two are fixtures,
  `door handle` and `ceiling light`;
- **50 / 168** sectors come back with an empty list entirely.

The stage is not blind to the obstacle. It sees it and is told to throw it away. Downstream
the consequence is visible in the table: OpenFMNav swerves off 91% of blocked straights and
**82% of open ones too** (keep straight 18%) — the always-swerves signature the OPEN frames
exist to catch. Its 100% mirror score is 5 pairs and should not be read as more than
"the object lists differ when the picture is flipped".

### 9.4 L3MVN cannot express the question, and its own code is where that is clearest

`construct_dist` (`main_llm_zeroshot.py:471`) was ported unchanged — the trailing `", and"`,
the `-loss` mean NLL from `scoring_fxn`, the `F.softmax`, GPT2-large, and their six-entry
`category_to_id`. Their line 754 is what a frontier's score actually is:

```python
frontier_score_list[e].append(new_dist[category_to_id.index(cname)])
```

Run that with our goal and their own code raises: `category_to_id.index('open floor')` →
**`ValueError`**. The score is *indexed by goal category*, and ours is not a category. Run
the distribution anyway, on the object lists OpenFMNav's stage produced:

- **23 of 84** sectors never reach the LM at all — their `if len(objs_list)>0` guard sends
  an empty frontier to a geometric fallback, not to the LLM;
- across the 61 that do, the argmax is only ever `bed` (26), `chair` (26) or `toilet` (9);
- the distribution is close to flat: the top probability averages **0.214** against 0.167
  for uniform, and the whole six-way range (max − min) averages **0.101**, at most 0.163.
  One sector whose object list literally contains `chair` still ranks `bed` above it,
  0.205 vs 0.198.

There is no bearing in that output and no image in the function that produced it. Two
sectors containing the same objects get the same number whether one is a wall and the other
a corridor. **That is not a defect** — it is what a co-occurrence prior is, and in L3MVN it
is correct, because the thing that knows *where* anything is is the semantic map,
`local_map[e][se_cn+4, fmb[0]:fmb[1], fmb[2]:fmb[3]]`, a depth projection. The LM ranks
frontiers the map has already found. Remove the depth and the half of L3MVN that answers
our question is the half that was removed.

### 9.5 What to actually take from §8 and §9

Four systems, run as far as each one ports:

| | what ported | verdict here |
| --- | --- | --- |
| VLFM | BLIP-2 ITM value function (a model) | reads pixels (84% mirror / 56 frames), scores the wrong property |
| WMNav | `action` and `predicting` (prompts) | `action` repairs obstacle blindness at ~70× the latency; `predicting` does not survive its own sweep |
| OpenFMNav | `discover` + `scoring` (prompts) | stage 1 is instructed to discard walls and floors |
| L3MVN | `construct_dist` (a function) | takes no image; goal category lookup raises on our task |

**None replaces the baseline, and one thing is worth keeping.** WMNav's goal-noun finding
is free: *naming what you are looking for in the instruction doubled obstacle avoidance*
(34% → 63%) with the scaffolding held fixed. Our shipped prompt says "Drive safely."
Whether "Stay on open floor" buys the same on the ladder is a one-variable experiment that
costs one run, and it is the only follow-up in §9 worth booking.

The honest negative is that the two value maps failed in the two available ways —
BLIP-2 scores pixels reliably for the wrong property, WMNav's prompt names the right
property and the model mostly cannot see it — and neither supplies the free-space channel
§6.2 found missing. That channel is still missing.

**Reproduce:**

```
/home/gtu-dsa/envs/qvla/bin/python nav/tools/probe_objectnav_prompts.py --port 8766 \
    --perms 3 --value-sweep 40 --json-out /tmp/objectnav.json
```
