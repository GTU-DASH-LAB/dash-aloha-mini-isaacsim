#!/usr/bin/env python3
"""Replay the decisions that produced NO answer, and test whether a prefill fixes them.

    nav/tools/probe_select_prefill.py --limit 30

THE DEFECT. 392 of 6673 menu decisions across the whole campaign -- 5.9% -- named no
label at all. The reply was not a wrong number, it was prose: "Based on the navigation
instruction to", "Looking at the image and the", cut off mid-sentence because the select
call's answer budget is 8 tokens at medium. `menu_plan` returns None, and the caller
reuses the previous plan.

It is not spread evenly, and that is the part that matters:

    clean prompt (no history note, not stalled)      0.3%     7 / 2363
    history note present, robot still moving         3.9%     8 /  204
    robot STALLED, history says "not working"        9.2%   377 / 4106
    the decision right after a wedge recovery       20.5%     9 /   44
    the decision right after a balk recovery        22.4%    17 /   76

The model stops answering thirty times more often exactly when the robot is in trouble --
because those are the prompts that grew an extra paragraph, and a longer prompt is a
stronger pull toward summarising it. So the decision channel is lost precisely in the
situations the recovery machinery exists to handle, and it latches: no answer -> reuse the
plan that was already not working -> still stuck -> the note grows -> no answer.

THE FIX is structural, not rhetorical, and it is the one this file already uses elsewhere.
The waypoint formats end the prompt with `ANSWER_PREFIX` ("<answer>(") so the next token
cannot be prose; it has to be a digit. `think_then_answer` takes a `prefill` argument for
exactly this and the menu's select call is the one caller that never passes one. Asking
politely -- "Answer with the number and nothing else" -- is already in the system prompt
and is what these 392 replies ignored.

WHAT THIS PROBE MEASURES. The same frames, the same system prompt, the same user turn,
the same 8-token budget, twice: once as it ran, once with the prefill. Anything other than
a large gap means the prefill is not the fix and the answer budget or the wording is.

MEASURED, 24 decisions that had produced no answer, 2026-09-01:

    as it ran      14/24 answered   (58%)
    with prefill   24/24 answered  (100%)

The bare column is 58% rather than 0% because decoding is sampled -- replaying a lost
decision recovers it about half the time by luck, which is precisely why the failure was
never obvious. The prefill column has no exceptions and no regressions: every reply was
the bare number, and where both answered they agreed on 8 of 14.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))

from arc_menu import (  # noqa: E402
    SELECT_PREFILL, SELECT_PREFILL_SPEED, parse_choice_speed, select_system, stop_label,
)

MENUS = Path("/tmp/qvla-menus")
# Rebuilt here rather than imported, because importing them means importing the server,
# which means importing torch. Kept verbatim; a paraphrase would make this probe measure a
# prompt the robot never ran.
SELECT_AFTER_RECOVERY_KEY = "wedge"


def raw(port: int, system: str, user: str, image: str, prefill: str,
        max_new: int) -> str:
    body = json.dumps({
        "image_paths": [image], "system": system, "user": user,
        "prefill": prefill, "max_new_tokens": max_new,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/raw", data=body,
                                 headers={"Content-Type": "application/json"})
    # 409 means a plan generation is in flight, which is normal against a live benchmark
    # and is not an error -- wait for the GPU rather than recording a miss that never
    # reached the model.
    for _ in range(240):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())["text"]
        except urllib.error.HTTPError as exc:
            if exc.code != 409:
                raise
            import time
            time.sleep(2.0)
    raise TimeoutError("the server stayed busy for 8 minutes")


def failing_decisions(limit: int) -> list[tuple[Path, dict]]:
    """Every decision that named no label, newest runs first, at most `limit`."""
    out: list[tuple[Path, dict]] = []
    runs = sorted(MENUS.glob("2026*_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        f = run / "decisions.jsonl"
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("choice") is None and (run / rec.get("menu", "")).is_file():
                out.append((run, rec))
                if len(out) >= limit:
                    return out
    return out


def rebuild(rec: dict) -> tuple[str, str, bool]:
    """(system, user, ask_speed) exactly as `menu_plan` built them for this decision."""
    labels = rec["labels"]
    pivots = tuple(rec["pivot_labels"]) if rec.get("pivot_labels") else None
    ask_speed = rec.get("speed_source") == "model" or rec.get("speed_level") is not None
    kind = rec.get("kind") or ""
    system = select_system(stop_label(len(labels), pivots is not None),
                           stop_allowed=not kind, speed_choice=ask_speed,
                           pivot_labels=pivots)
    # The recovery sentence between history and instruction is part of the user turn in
    # `menu_plan`. It is omitted here rather than guessed: `kind` is recorded, the exact
    # sentence is not, and a probe that invents one measures a prompt nobody ran. Its
    # absence makes this test HARDER on the prefill, not easier -- the recovery decisions
    # are where the failure rate is worst.
    hist = rec.get("history") or ""
    user = (f"What the robot can see ahead of it: {rec['free_space']}\n\n"
            + (f"{hist}\n\n" if hist else "")
            + f"Navigation instruction: {rec['instruction']}\n\n"
            + ("Which numbered path do you take, and how fast?" if ask_speed
               else "Which numbered path do you take?"))
    return system, user, ask_speed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--answer-tokens", type=int, default=8,
                    help="the medium level's budget, which is what these failed under")
    args = ap.parse_args()

    cases = failing_decisions(args.limit)
    if not cases:
        print("!! no unparsed decisions found under /tmp/qvla-menus")
        return 1
    print(f"-- replaying {len(cases)} decisions that produced no answer, "
          f"{args.answer_tokens}-token budget, with and without the prefill\n")

    score = Counter()
    for i, (run, rec) in enumerate(cases, 1):
        system, user, ask_speed = rebuild(rec)
        image = str(run / rec["menu"])
        prefill = SELECT_PREFILL_SPEED if ask_speed else SELECT_PREFILL
        bare = raw(args.port, system, user, image, "", args.answer_tokens)
        with_p = raw(args.port, system, user, image, prefill, args.answer_tokens)
        c_bare, _ = parse_choice_speed(bare)
        # `/raw` returns `prefill + generated`, which is exactly the string `menu_plan`
        # hands the parser. Do NOT prepend the prefill again here: it is harmless to the
        # parse (the prefill holds no digit, by construction) but it prints a doubled
        # sentence that reads as the model having repeated the prompt back.
        c_with, _ = parse_choice_speed(with_p)
        score["bare"] += c_bare is not None
        score["prefill"] += c_with is not None
        flag = {(False, True): "FIXED", (False, False): "still lost",
                (True, True): "both", (True, False): "REGRESSED"}[
                    (c_bare is not None, c_with is not None)]
        print(f"{i:>3}. {run.name[9:40]:<31} step {rec['step']:>5}  {flag:<10} "
              f"bare={c_bare!s:<5} prefill={c_with!s:<5} {with_p[:46]!r}")

    n = len(cases)
    print(f"\n   as it ran      {score['bare']:>3}/{n} answered  "
          f"({100 * score['bare'] / n:.0f}%)")
    print(f"   with prefill   {score['prefill']:>3}/{n} answered  "
          f"({100 * score['prefill'] / n:.0f}%)")
    return 0 if score["prefill"] > score["bare"] else 1


if __name__ == "__main__":
    sys.exit(main())
