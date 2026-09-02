"""Does asking for a goal DIRECTION recover the turns the free-space superlative loses?

    nav/tools/probe_goal_direction.py            # the A/B, both controls
    nav/tools/probe_goal_direction.py --n 20     # fewer moments, same shape

WHAT THIS IS TESTING, and why it needs its own probe rather than a ladder. The cross-tab
over 840 reference decisions says the selector is a lookup table on one word from the
describe call:

    describe said     n     chose straight   mean kappa
    far left         36           11%           +0.533
    left             38           21%           +0.326
    straight        668           94%           -0.008
    right            96            6%           -0.373
    far right         2            0%           -0.600

and that the word carries no information about the goal: with the word held fixed, the
correlation between the bearing error to the goal and the chosen curvature is +0.042 on
the 668 `straight` decisions. So the policy follows corridors, and `DESCRIBE_OPEN_SET` +
`DESCRIBE_TARGET_DIR` are the proposed fix -- report which directions are drivable rather
than which is MOST open, and say which way the task needs to go.

That is a hypothesis about the model, and it costs an 18-run ladder to test the slow way.
It costs four minutes to test directly: take the exact frames where the reference policy
went wrong, ask both questions of the same pixels, and see whether the new one names the
side the goal is actually on. If it answers `straight ahead` there too, the rewrite is
prose and the ladder would only have measured noise.

THE MOMENTS ARE CHOSEN BY THE FAILURE, not sampled at random: decisions where the
reference describe call said `straight`, where the goal was more than `--min-err` degrees
off the nose, and where the run went on to fail. That is the population the fix has to
move, and a probe averaged over easy frames would hide a null result inside frames where
straight was correct anyway.

TWO CONTROLS, and neither is optional:

  * THE STRAIGHT CONTROL. The same question on moments where the goal really is ahead
    (|bearing error| < `--ctrl-err`). A variant that answers `left` or `right` everywhere
    scores perfectly on the failure set and is worse than what it replaces -- it has
    traded a straight bias for a turning bias, not acquired a goal. This control is what
    separates the two, and it is the same role the open-corridor frames play in
    `probe_arc_repair.py`.
  * THE OLD QUESTION ON THE SAME PIXELS. Asked here rather than read out of the decision
    log, because the log's answers were produced on the RAW camera frame and this probe
    runs on the rendered menu (the raw frames are deleted per step; only menus survive a
    run). Re-asking makes the comparison within-frame, so the arcs drawn on the image are
    a constant across both arms and cannot be the difference.

Chance is not 50%. `HEADING` has five values and the score is the SIDE, so a model
guessing uniformly over {far left, left, straight, right, far right} is right about 40% of
the time on the failure set. The number to beat is that, and the straight control's number
to beat is the old question's own.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))
sys.path.insert(0, str(REPO / "nav/sim"))
from arc_menu import (DESCRIBE_OPEN_SET, DESCRIBE_SIDED, DESCRIBE_TARGET,  # noqa: E402
                      DESCRIBE_TARGET_DIR, FREE_SPACE_SYSTEM, parse_heading_word)
from summarize_runs import load_config, score  # noqa: E402

STUDY = REPO / "nav/results/hard_study"
MENUS = Path(os.environ.get("QVLA_MENU_DIR", "/tmp/qvla-menus"))
PORT = os.environ.get("NAV_POLICY_PORT", "8766")
_STAMP = len("20260902-110026")

# Signed side of each answer, left positive, to match the bearing-error convention. Used
# only for the SIGN -- this is not a magnitude comparison, because the describe call is
# choosing among five words and the bearing error is continuous.
_SIDE = {"far left": +2, "left": +1, "straight ahead": 0, "right": -1, "far right": -2}

# The old question's answer lives inside a sentence, so it is parsed the way the analysis
# parsed it: on the phrase the question itself dictates. Longest alternative first --
# Python's `|` is first-match, so `left` ahead of `far left` silently collapses the two.
_OLD_RE = re.compile(
    r"most open,? (?:and )?walkable floor[^.]*?\b"
    r"(far left|far right|left|right|straight ahead|straight)\b", re.IGNORECASE)


def _old_side(text: str) -> str | None:
    m = _OLD_RE.search(text)
    if not m:
        return None
    w = " ".join(m.group(1).lower().split())
    return "straight ahead" if w == "straight" else w


def _menu_dir(episode: str, stamp: str) -> Path | None:
    """The menu directory belonging to one run: newest start at or before that run's end.

    Exact-match the episode name rather than globbing a prefix. `*_warehouse_*` also
    catches `warehouse_aisle6`, which is how an earlier analysis scored one episode's
    choices against another episode's frames and produced a confident wrong number.
    """
    cands = [d for d in MENUS.glob("*")
             if d.name[_STAMP + 1:] == episode and d.name[:_STAMP] <= stamp]
    return max(cands, key=lambda d: d.name) if cands else None


def moments(cfg: dict, min_err: float, ctrl_err: float) -> tuple[list[dict], list[dict]]:
    """Decisions from FAILED reference runs, split into the failure set and the control.

    Both sets are drawn from the same runs and the same describe answer (`straight`), so
    the only thing separating them is where the goal actually was. That matters: a control
    taken from the successful runs would differ in scene, episode mix and phase of the
    approach all at once, and any difference in the result could be any of them.
    """
    fail, ctrl = [], []
    for arm in ("ref_s0", "ref_s1", "ref_s2", "refrep_s0"):
        for f in sorted((STUDY / arm).glob("*.json")):
            if f.stem == "arm":
                continue
            r = score(f, cfg)
            if r is None or r["success"]:
                continue                       # the fix is for the runs that lost
            stamp, rest = f.stem.split("_", 1)
            ep = rest.rsplit("_", 1)[0]
            d = _menu_dir(ep, stamp)
            if d is None:
                continue
            gx, gy = cfg[ep]["goal"][:2]
            trace = json.loads(f.read_text())["trace"]
            # The trace is sampled on its own clock, so a decision's `step` is mapped onto
            # it by FRACTION of the run. The denominator comes from the decisions file
            # itself rather than from the result -- `score()` reports no step count, and a
            # second constant standing in for one is how this repo has three times produced
            # a plausible wrong number instead of an error.
            decs = [json.loads(l) for l in (d / "decisions.jsonl").open() if l.strip()]
            steps = max((x["step"] for x in decs), default=0)
            for x in decs:
                if x.get("stop") or _old_side(x.get("free_space", "")) != "straight ahead":
                    continue
                img = d / f"menu_{x['step']:08d}.jpg"
                if not img.is_file():
                    continue
                i = min(len(trace) - 1, max(0, int(x["step"] / max(steps, 1)
                                                   * (len(trace) - 1))))
                err = math.degrees(math.atan2(gy - trace[i][1], gx - trace[i][0])
                                   - trace[i][2])
                err = (err + 180) % 360 - 180
                rec = {"episode": ep, "arm": arm, "step": x["step"], "img": str(img),
                       "instruction": x["instruction"], "err": err}
                (fail if abs(err) >= min_err else
                 ctrl if abs(err) <= ctrl_err else []).append(rec)
    return fail, ctrl


def ask(images: list[str], system: str, user: str, max_new: int) -> str:
    body = {"image_paths": images, "system": system, "user": user,
            "max_new_tokens": max_new, "think": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/raw",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["text"]


def run(rows: list[dict], label: str) -> dict:
    """Ask both questions of every moment. Returns the counts, prints one line each."""
    print(f"\n{label}  ({len(rows)} moments)\n")
    print(f"  {'episode':<28}{'err':>7}   {'OLD (most open)':<16}{'NEW (heading)':<16}")
    print("  " + "-" * 70)
    n_old_turn = n_new_turn = n_new_right = n_new_none = 0
    for m in rows:
        old = _old_side(ask([m["img"]], FREE_SPACE_SYSTEM,
                            DESCRIBE_SIDED + DESCRIBE_TARGET.format(
                                instruction=m["instruction"]), 110))
        new = parse_heading_word(ask([m["img"]], FREE_SPACE_SYSTEM,
                                     DESCRIBE_OPEN_SET + DESCRIBE_TARGET_DIR.format(
                                         instruction=m["instruction"]), 150))
        n_old_turn += old not in (None, "straight ahead")
        n_new_turn += new not in (None, "straight ahead")
        n_new_none += new is None
        # On the failure set "right" means the sign matches; on the control it means the
        # answer is `straight ahead`, and `run` cannot tell which set it has -- so both are
        # counted and the caller reads the column it asked for.
        n_new_right += (_SIDE.get(new, 0) > 0) == (m["err"] > 0) and new is not None \
            and new != "straight ahead"
        print(f"  {m['episode']:<28}{m['err']:>+7.0f}   {str(old):<16}{str(new):<16}")
    return {"n": len(rows), "old_turn": n_old_turn, "new_turn": n_new_turn,
            "new_side_ok": n_new_right, "new_none": n_new_none}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="moments per set")
    ap.add_argument("--min-err", type=float, default=40.0)
    ap.add_argument("--ctrl-err", type=float, default=15.0)
    args = ap.parse_args()

    cfg = load_config()
    fail, ctrl = moments(cfg, args.min_err, args.ctrl_err)
    if not fail:
        print("no qualifying moments -- are there failed ref runs with menus on disk?")
        return 1
    # Even stride rather than the first N: decisions cluster by episode and by phase, and
    # the first 24 of a sorted list is one episode's opening.
    pick = lambda v: v[::max(1, len(v) // args.n)][:args.n]
    fail, ctrl = pick(fail), pick(ctrl)

    print(__doc__.split("\n\n")[0])
    print(f"\nfailure set: describe said 'straight', goal >= {args.min_err:.0f} deg off, "
          f"run FAILED")
    print(f"control set: same, but goal <= {args.ctrl_err:.0f} deg off -- straight is "
          f"CORRECT there")

    f = run(fail, "FAILURE SET -- straight was wrong")
    c = run(ctrl, "CONTROL SET -- straight was right")

    print("\n\nVERDICT\n")
    print(f"  {'':<34}{'old question':>14}{'new question':>14}")
    print("  " + "-" * 62)
    print(f"  {'failure set: named a side at all':<34}"
          f"{f['old_turn']}/{f['n']:<13}{f['new_turn']}/{f['n']}")
    print(f"  {'failure set: named the RIGHT side':<34}{'--':>14}"
          f"{f['new_side_ok']}/{f['n']}")
    print(f"  {'control set: said straight ahead':<34}"
          f"{c['n'] - c['old_turn']}/{c['n']:<13}{c['n'] - c['new_turn']}/{c['n']}")
    if f["new_none"] or c["new_none"]:
        print(f"\n  no HEADING parsed: {f['new_none']} of {f['n']} failure, "
              f"{c['new_none']} of {c['n']} control  (budget truncation shows up here)")
    print("\n  Chance on the failure set is ~40% -- five words, scored on the side.")
    print("  A variant that beats it there AND holds the control has a goal channel.")
    print("  One that beats it there and loses the control has a turning bias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
