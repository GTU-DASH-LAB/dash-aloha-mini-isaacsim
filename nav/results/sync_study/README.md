# sync_study — read this before using any number in here

**Every arm in this directory ran before the label-RNG reseed, and none of these numbers
may be used as a baseline.**

The arc-menu server drew its per-call menu label permutation from a generator shared with
other randomised state. Arms that were meant to differ only in the variable under test
therefore also differed in their label ordering, and arms that were meant to be identical
were not. `analysis.txt` reaches the right conclusion from the wrong evidence when it
opens with "nothing is distinguishable from anything else" — that is what a confounded
comparison looks like, and it is also what a real null result looks like.

The boundary is exact, and it is on disk:

| | |
|---|---|
| Reseed | `2026-09-02 10:59:35`, recorded in the name of `../hard_study/ref_s0.prereseed-20260902-105935` |
| This directory | every arm ran `2026-09-01 22:06` → `2026-09-02 08:27` — **all of it before** |
| `../hard_study/` | `ref_s0` onward, from `11:45` — after, and therefore usable |

## So why is it still here

Because the reasoning survives the confound even where the numbers do not.
`analysis.txt` is 181 lines of what was checked and how, and its method — per-episode
tables rather than aggregate scores, closest-approach beside success, a determinism
re-run as its own arm — is what made the confound findable at all. The scoreboards are
kept as history, not as evidence.

Anything to be compared against lives in
[`nav/config/profiles/baseline.yaml`](../../config/profiles/baseline.yaml), which records
its own measurements next to the arms it rejected.

## What is not in git

The 126 videos and 142 result JSONs behind these summaries, ~2.2 GB, stay on the
workstation. See [`../.gitignore`](../.gitignore). The three `pivot_menu_sample/*.jpg`
are kept because a rendered arc menu is not reconstructable from a number.
