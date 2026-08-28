# TIC-VLA architecture — block diagrams

Two views of the same network, each in English and Turkish. Every dimension, parameter
count and connection was **read out of the source**, not estimated, and checked against
the 50 action-expert tensors in the shipped `TIC-VLA-model.ckpt`.

| | English | Türkçe |
|---|---|---|
| **Diagrams** — drawn, layer by layer | [`visual.en.html`](visual.en.html) | [`visual.tr.html`](visual.tr.html) |
| **Full reference** — every shape and loss in tables | [`reference.en.html`](reference.en.html) | [`reference.tr.html`](reference.tr.html) |

Open any of them straight off disk:

```bash
xdg-open docs/ticvla-architecture/visual.tr.html
```

They are self-contained — no build step, no network, no external assets. The header bar
switches between the two views, between EN and TR, and between light and dark.

## Which one to read

**`visual.*`** is three drawn figures: the whole dataflow end to end, the inside of a
single cross-attention block, and the two training stages side by side. Read this first
— the shape of the design is visible in it.

**`reference.*`** is the same content as prose and tables: the signal path block by
block, the parameter budget, the ActionExpert tensor by tensor, both loss functions with
their real code, and the delay mechanism. Read this when you need a specific number.

## What the diagrams say

The three facts worth carrying away, all of which the figures are arranged to show:

- **The hand-off is one layer, values only.** `past_key_values[-1]`, keys discarded,
  `(B, 2, L, 64)` → `(B, L, 128)` — 1/24th of the cache and half of that layer, then
  `.detach()`ed. Everything the 1.9 B VLM knows reaches the controller through that.
- **There are three independent gradient barriers**, not one: `requires_grad = False` on
  the VLM, `.detach()` on the cache, and `extract_feature` under `no_grad`. So
  `TICVLA.forward` still returns a `language_loss` that stage 2 never reads, and with the
  VLM frozen it is a zero tensor computed and discarded on every batch.
- **No action error ever reaches the VLM.** There is no joint objective anywhere in the
  codebase — no weighted sum, no alternating schedule. Stage 1 fits text, stage 2 fits
  trajectories, and nothing fits the two together.

That last point is why these diagrams exist. Whatever steering competence TIC-VLA has
was learned by **10.27 M parameters reading a frozen 128-dimensional summary** — not by
the 1.9 B model that produced it. It is the structural counterpart to the Q-VLA-direct
result in [`../../CLAUDE.md`](../../CLAUDE.md): asked to write a signed lateral offset in
text, a 27 B model would not do it, and TIC-VLA does not solve that problem so much as
route around it — the lateral sign is never asked of a language model at all.

## Provenance

Read from the vendored, read-only checkout at `~/robotics/TIC-VLA`:

```
ticvla/models/ticvla.py          the two-branch forward, the KV hand-off, ActionExpert
ticvla/models/vlm.py             the InternVL wrapper and the freezing loop
ticvla/utils/vision.py           tiling, pixel_shuffle, the 448×448 transform
ticvla/data/policy_data.py       the delay sampling and the action targets
ticvla/data/vlm_data.py          the chat template and the <answer> triples
ticvla/training/{train,config}.py  both losses, both optimiser configs
models/InternVL3-1B/config.json  every vision and language dimension
models/TIC-VLA-model.ckpt        the parameter counts, verified not estimated
```

## Regenerating

The pages are hand-authored HTML, checked in as-is. Two scratch scripts built them from
the published artifacts and are not part of the repo: one translated the English source
string by string (asserting every source string was found, so a miss fails loudly rather
than stranding English in the Turkish page), the other wrapped the artifact fragments
into standalone documents — adding the `<!doctype>`/`<head>`/`<body>` skeleton the
artifact host normally supplies, plus `<meta charset>`, without which the Turkish
characters are at the mercy of the browser's encoding guess.

To edit, edit the HTML directly. If you change a figure, keep the four files in step —
the SVG text labels are translated too, not just the prose.
