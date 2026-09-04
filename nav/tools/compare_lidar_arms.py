#!/usr/bin/env python3
"""Pair the two arms of the lidar ladder episode by episode.

    python3 nav/tools/compare_lidar_arms.py
    python3 nav/tools/compare_lidar_arms.py --a fan --b c1@0.30 --controller braking

WHY PAIRED AND NOT POOLED. "9 of 19 against 8 of 19" is not an answer to "can we now do
things we could not before" -- it is consistent with one episode flipping each way, with
nine flipping each way, and with nothing changing at all. The question is which episodes
moved and in which direction, and only the join answers it. ../../CLAUDE.md already
carries the general form of this: three failures on one ladder had three different causes
and averaging them would have described none of them.

READ `flip` FIRST AND `closed` SECOND. Success is a hard threshold at 1.5 m, so an
episode can move a long way and not flip, or flip on 8 cm -- that has happened here for
real, `hospital_down_hallway2` stopping at 1.58 m against a twin that scored spl 1.00.
The fraction of the gap closed is the continuous version and it moves when the threshold
does not.

AND DO NOT READ ONE LADDER AS A MEASUREMENT OF THE SENSOR. `predict()` is deterministic
on identical inputs but a RUN is not: which KV cache is ready at which step depends on
wall-clock generation timing, and the same episode with the same sentence has been
observed to succeed twice and fail once. n=1 per arm is the noise floor of this stack,
so a one- or two-episode difference in either direction is not a result. What this tool
can settle is a LARGE difference, and the direction of a consistent small one across many
episodes -- which is why the totals line prints both arms' `closed` mean.

IMPORTABLE ON PURPOSE. `report_lidar_arms.py` renders the same comparison as HTML for
email, and the pairing rules here -- the 25-minute clustering, the clean-ladder filter,
the refusal to backfill `lidar: ""` -- are exactly the kind of thing CLAUDE.md records as
failing silently when they exist in two copies. So the logic lives in `pair_arms()` and
both front ends call it; nothing below the fold re-derives a rule.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

from summarize_runs import load_config, score  # noqa: E402

RESULTS = REPO / "nav" / "results"


def prior_ladders(controller: str) -> list[dict[str, bool]]:
    """Per-episode outcomes of each clean single-pass ladder from before the sensor.

    CLEAN means every episode appears exactly once in the cluster. Ladders where an
    episode was re-run are excluded outright rather than reduced to their last result:
    a re-run is usually a config being changed mid-ladder, and CLAUDE.md is explicit
    that a directory holding two policies distinguishable only by timestamp is how this
    stack talks itself into believing something. Clustered on a 25-minute gap, which is
    comfortably longer than the slowest episode and far shorter than the gap between
    sessions.

    These are a RELIABILITY PRIOR, not a baseline. They span real config changes -- the
    wedge-recovery fix landed between them, which is why one of the three scores 2/13
    against 8/13 for the other two -- so an episode's rate says how often it has ever
    worked, not how often the current code would work.
    """
    runs = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("controller") != controller:
            continue
        if "menu" not in (d.get("policy") or ""):
            continue
        if d.get("lidar", ""):
            continue          # anything labelled is from the sensor era, not before it
        runs.append((p.name[:15], d["episode"], bool(d.get("success"))))
    if not runs:
        return []

    def when(s: str) -> datetime.datetime:
        return datetime.datetime.strptime(s, "%Y%m%d-%H%M%S")

    groups, cur = [], [runs[0]]
    for r in runs[1:]:
        if (when(r[0]) - when(cur[-1][0])).total_seconds() > 25 * 60:
            groups.append(cur)
            cur = [r]
        else:
            cur.append(r)
    groups.append(cur)
    return [{e: ok for _, e, ok in g} for g in groups
            if len({x[1] for x in g}) == len(g) >= 13]


@dataclass
class Comparison:
    """Everything a front end needs, with no scoring rule left for it to reinvent."""

    a_label: str
    b_label: str
    controller: str
    order: list[str]                       # paired episodes, in ladder order
    arm_a: dict[str, dict]
    arm_b: dict[str, dict]
    history: list[dict[str, bool]] = field(default_factory=list)
    never: list[str] = field(default_factory=list)   # never passed on any prior ladder
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)

    def flip(self, ep: str) -> str:
        """WON / LOST / = for one episode. The first column anyone should read."""
        a, b = self.arm_a[ep]["success"], self.arm_b[ep]["success"]
        return "WON" if b and not a else "LOST" if a and not b else "="

    def was(self, ep: str) -> tuple[int, int] | None:
        """(passes, ladders) on the pre-sensor ladders, or None if never run then."""
        seen = [d for d in self.history if ep in d]
        return (sum(1 for d in seen if d[ep]), len(seen)) if seen else None

    def totals(self) -> dict:
        n = len(self.order) or 1
        return {
            "n": len(self.order),
            "a_pass": sum(1 for e in self.order if self.arm_a[e]["success"]),
            "b_pass": sum(1 for e in self.order if self.arm_b[e]["success"]),
            "a_closed": sum(self.arm_a[e]["closed_frac"] for e in self.order) / n * 100,
            "b_closed": sum(self.arm_b[e]["closed_frac"] for e in self.order) / n * 100,
            "a_guard": sum(self.arm_a[e]["guard"] for e in self.order),
            "b_guard": sum(self.arm_b[e]["guard"] for e in self.order),
            "a_path": sum(self.arm_a[e]["path_m"] for e in self.order),
            "b_path": sum(self.arm_b[e]["path_m"] for e in self.order),
            "a_spl": sum(self.arm_a[e]["spl"] for e in self.order) / n,
            "b_spl": sum(self.arm_b[e]["spl"] for e in self.order) / n,
        }

    def won(self) -> list[str]:
        return [e for e in self.order if self.flip(e) == "WON"]

    def lost(self) -> list[str]:
        return [e for e in self.order if self.flip(e) == "LOST"]

    def prior_scores(self) -> list[str]:
        return [f"{sum(d.values())}/{len(d)}" for d in self.history]


def pair_arms(a: str = "fan", b: str = "c1@0.30", controller: str = "braking",
              history: bool = True) -> Comparison:
    cfg = load_config()
    # Sorted by filename, which is the timestamp, so the LAST match per episode is the
    # newest run of that arm. Both arms are re-run for a comparison; an older stray run
    # of the same label would otherwise be paired against a fresh one from the other.
    rows = [r for r in (score(p, cfg) for p in sorted(RESULTS.glob("*.json"))) if r]
    rows = [r for r in rows if r["controller"] == controller]

    arm_a = {r["episode"]: r for r in rows if r["lidar"] == a}
    arm_b = {r["episode"]: r for r in rows if r["lidar"] == b}
    both = [e for e in arm_a if e in arm_b]
    if not both:
        raise SystemExit(
            f"no episode has a run under BOTH '{a}' and '{b}' with controller "
            f"{controller}.\n"
            f"  '{a}' has {len(arm_a)} episodes, '{b}' has {len(arm_b)}.\n"
            "  Runs recorded before the `lidar` field existed carry '' -- they were the "
            "fan, but\n  nothing in the file says so, so they are not matched by --a fan."
        )
    # Ladder order, so the table reads easiest-first like the ladder itself rather than
    # alphabetically. Episodes present in only one arm are listed separately, never
    # silently dropped -- a missing run is usually a crash and is the thing worth seeing.
    order = [e for e in cfg if e in both] + [e for e in both if e not in cfg]

    hist = prior_ladders(controller) if history else []
    cmp = Comparison(a, b, controller, order, arm_a, arm_b, hist,
                     only_a=sorted(set(arm_a) - set(arm_b)),
                     only_b=sorted(set(arm_b) - set(arm_a)))
    # An episode that never passed before is the only place a single run can carry real
    # information: everywhere else the prior already contains both outcomes, so one flip
    # is inside the noise this stack is known to have.
    cmp.never = [e for e in order
                 if (w := cmp.was(e)) is not None and w[0] == 0]
    return cmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="fan", help="reference arm's `lidar` label")
    ap.add_argument("--b", default="c1@0.30", help="treatment arm's `lidar` label")
    ap.add_argument("--controller", default="braking")
    ap.add_argument("--history", action="store_true",
                    help="add a column of how often each episode passed on the ladders "
                         "recorded BEFORE the sensor existed")
    args = ap.parse_args()

    c = pair_arms(args.a, args.b, args.controller, history=args.history)

    hcol = f"{'was':>7}" if args.history else ""
    print(f"\n  {'episode':<26}{hcol}{'A ' + c.a_label:>14}{'B ' + c.b_label:>14}   "
          f"flip   {'closed A':>9}{'closed B':>9}{'  d':>7}")
    print("  " + "-" * (92 + len(hcol)))

    for ep in c.order:
        a, b = c.arm_a[ep], c.arm_b[ep]
        ta = f"{'YES' if a['success'] else 'no':<4}{a['closest_m']:6.2f}m"
        tb = f"{'YES' if b['success'] else 'no':<4}{b['closest_m']:6.2f}m"
        flip = {"WON": " WON", "LOST": "LOST"}.get(c.flip(ep), "  =")
        h = ""
        if args.history:
            w = c.was(ep)
            h = f"{f'{w[0]}/{w[1]}' if w else '-':>7}"
        ca, cb = a["closed_frac"] * 100, b["closed_frac"] * 100
        print(f"  {ep:<26}{h}{ta:>14}{tb:>14}   {flip}   "
              f"{ca:8.0f}%{cb:8.0f}%{cb - ca:+7.0f}")

    t = c.totals()
    print("  " + "-" * 92)
    print(f"  {'TOTAL':<26}{t['a_pass']:>9} /{t['n']:3d}{t['b_pass']:>9} /{t['n']:3d}"
          f"   {t['b_pass'] - t['a_pass']:+4d}   {t['a_closed']:8.0f}%"
          f"{t['b_closed']:8.0f}%{t['b_closed'] - t['a_closed']:+7.0f}")

    print(f"\n  won  ({len(c.won())}): {', '.join(c.won()) or '-'}")
    print(f"  lost ({len(c.lost())}): {', '.join(c.lost()) or '-'}")

    if args.history:
        print(f"\n  `was` is the pass rate over {len(c.history)} clean single-pass "
              f"ladders recorded before the")
        print(f"  sensor existed, scoring {', '.join(c.prior_scores())}. That spread IS "
              f"the noise floor -- those")
        print("  ladders span a config change, and even the two best differ episode by "
              "episode.")
        print("  So a one- or two-episode move here is not a result and must not be "
              "reported as one.")
        if c.never:
            print(f"\n  NEVER PASSED BEFORE: {', '.join(c.never)}")
            for e in c.never:
                aa, bb = c.arm_a[e]["success"], c.arm_b[e]["success"]
                mark = ("BOTH arms" if aa and bb else
                        f"only {c.b_label}" if bb else
                        f"only {c.a_label}" if aa else "neither")
                print(f"    {e:<26} now: {mark}")
            print("  This row is the only place one ladder can say something a repeat "
                  "could not:")
            print("  everywhere else the prior already contains both outcomes, so a flip "
                  "is noise.")

    if c.only_a or c.only_b:
        print("\n  NOT PAIRED -- these are excluded from every number above, and a "
              "missing run is\n  usually a crash rather than a result:")
        if c.only_a:
            print(f"    only in {c.a_label}: {', '.join(c.only_a)}")
        if c.only_b:
            print(f"    only in {c.b_label}: {', '.join(c.only_b)}")

    # Guard interventions are the one number that speaks about the SENSOR rather than
    # about the policy, since the guard is the half of this change that does not go
    # through a VLM. A denser scan should raise the count -- it sees more -- and a large
    # fall would mean the robot stopped meeting obstacles, which on a fixed set of
    # episodes means it stopped getting to them.
    print(f"\n  guard interventions   {c.a_label} {t['a_guard']}   "
          f"{c.b_label} {t['b_guard']}")
    print("  (the guard is the half of this change that does not pass through the VLM;")
    print("   more returns should mean MORE interventions, and a large drop means the "
          "robot")
    print("   stopped reaching the obstacles rather than that it stopped hitting them)")

    print(f"\n  path driven           {c.a_label} {t['a_path']:.0f} m   "
          f"{c.b_label} {t['b_path']:.0f} m")
    print("  (read next to `closed`, never alone -- a short path is a tidy route or a "
          "run")
    print("   that parked, and warehouse_aisle6 has produced both)")

    if not math.isclose(len(c.order), len(set(c.arm_a) & set(c.arm_b))):
        print("\n  note: some paired episodes are missing from episodes.yaml and were "
              "appended after the ladder order.")


if __name__ == "__main__":
    main()
