"""Where do two identical runs stop being identical? Answer it stage by stage.

    python3 nav/tools/check_determinism.py                       # ref_s0 vs refrep_s0
    python3 nav/tools/check_determinism.py --arms all_s0 all_s1  # any other pair

`hard_study_table.py` reports THAT the determinism check failed -- two runs of one
configuration, one seed, disagreeing on 2 of 6 episodes with a 17 m spread in closest
approach. That verdict is the licence for every arm comparison above it and it is worth
nothing on its own, because "not reproducible" has at least four causes that need four
different responses:

    the labels were drawn differently        -> a seeding bug, ours, fixable
    the robot started somewhere else         -> the runner's reset, ours, fixable
    the rendered image differs               -> the renderer, not ours
    everything above matches, the text does  -> the model amplifying noise, unfixable

This walks the chain in that order and prints the first link that broke, per episode.
The image column is what separates the third case from the fourth: a mean absolute
difference near one grey level with nothing above 40 is sub-perceptual noise, and a
frame that genuinely shows something else is tens of levels with whole regions moving.

MATCHING RUNS TO THEIR MENU DIRECTORIES IS THE PART THAT GOES WRONG. Two prefix traps
already produced two fictitious "structural outliers" here, each of which survived long
enough to get an explanation written for it:

  - `nav/results/<arm>/*_warehouse_*.json` also matches `warehouse_aisle6_braking.json`,
    so `warehouse` was compared against a different episode's frames -- 30 grey levels of
    difference, which reads exactly like a scene change.
  - `/tmp/qvla-menus/` accumulates a directory per run for weeks, and different episodes
    have different totals, so indexing from the end (`sorted(...)[-4:]`) lines up
    different runs for different episodes.

Hence exact equality on the episode name in both places, and the menu directory chosen
by timestamp: a run's menu directory is stamped when it STARTS and its result file when
it ENDS, so the right directory is the newest one that starts before the result's stamp.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "nav/results/hard_study"
MENUS = Path("/tmp/qvla-menus")

# A menu directory is named "<15-char stamp>_<episode>": "20260902-110026_warehouse".
_STAMP = len("20260902-110026")


def runs(arm: str) -> dict[str, str]:
    """{episode: result stamp} for one ladder, keyed on the EXACT episode name.

    The controller suffix is stripped with `rsplit` rather than matched, because every
    episode name in this benchmark contains underscores and several are prefixes of
    each other.
    """
    out: dict[str, str] = {}
    for f in (STUDY / arm).glob("*.json"):
        if f.stem == "arm":
            continue
        stamp, rest = f.stem.split("_", 1)
        out[rest.rsplit("_", 1)[0]] = stamp
    return out


def menu_dir(episode: str, stamp: str) -> Path | None:
    """The menu directory belonging to one run: newest start before that run's end."""
    cands = [d for d in MENUS.glob("*")
             if d.name[_STAMP + 1:] == episode and d.name[:_STAMP] <= stamp]
    return max(cands, key=lambda d: d.name) if cands else None


def decisions(d: Path) -> list[dict]:
    f = d / "decisions.jsonl"
    return [json.loads(l) for l in f.open() if l.strip()] if f.exists() else []


def frame_delta(a: Path, b: Path) -> tuple[float, float, float]:
    """Mean absolute grey-level difference of frame 0, and two tail fractions.

    Frame 0 specifically: it is rendered before the robot has moved, so anything that
    differs there cannot be a consequence of an earlier decision and is the renderer
    talking. Greyscale because a per-channel breakdown has never changed a reading here,
    and the JPEG the model is served has already thrown away most of the chroma.
    """
    im = lambda p: np.asarray(Image.open(p / "menu_00000000.jpg").convert("L"), float)
    d = np.abs(im(a) - im(b))
    return d.mean(), (d > 8).mean() * 100, (d > 40).mean() * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs=2, default=["ref_s0", "refrep_s0"],
                    metavar=("A", "B"), help="two ladder directories under hard_study/")
    args = ap.parse_args()

    a_runs, b_runs = runs(args.arms[0]), runs(args.arms[1])
    shared = [e for e in a_runs if e in b_runs]
    if not shared:
        print(f"no episode is present in both {args.arms[0]} and {args.arms[1]}")
        return 1

    print(f"{args.arms[0]}  vs  {args.arms[1]}   -- frame 0 is rendered before the "
          f"robot moves\n")
    print(f"{'episode':<30}{'mean|d|':>9}{'>8 lv':>8}{'>40 lv':>7}"
          f"{'labels':>8}{'text':>6}{'choice':>8}{'decisions':>12}")
    print("-" * 88)
    for ep in shared:
        da, db = menu_dir(ep, a_runs[ep]), menu_dir(ep, b_runs[ep])
        if not (da and db):
            print(f"{ep:<30}  no menu directory on disk -- /tmp/qvla-menus was cleared")
            continue
        mean, hi, vhi = frame_delta(da, db)
        ra, rb = decisions(da), decisions(db)
        n = min(len(ra), len(rb))
        # First index where a stage disagrees. Reported per stage rather than as one
        # "first divergence" because the stages are nested: the text can differ for ten
        # decisions while the choice it feeds stays the same, and that gap is the
        # measurement -- it says how much wording the selector absorbs before it flips.
        first = lambda k: next((i for i in range(n) if ra[i].get(k) != rb[i].get(k)), None)
        s = lambda v: "same" if v is None else f"@{v}"
        print(f"{ep:<30}{mean:>9.2f}{hi:>7.2f}%{vhi:>6.2f}%"
              f"{s(first('labels')):>8}{s(first('free_space')):>6}"
              f"{s(first('choice')):>8}{f'{len(ra)}/{len(rb)}':>12}")
    print("-" * 88)
    print("  labels  first decision whose shuffled menu differs -- 'same' means the "
          "per-episode\n          reseeding holds and the two runs were offered "
          "identical menus throughout")
    print("  text    first decision whose free-space description differs")
    print("  choice  first decision whose selected label differs")
    print("\n  A mean|d| near 1 with 0.00% above 40 levels is sub-perceptual render "
          "noise:\n  the two frames look identical and are not bit-identical. Tens of "
          "levels, or any\n  real mass above 40, is a different scene -- check the "
          "start pose before blaming\n  the renderer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
