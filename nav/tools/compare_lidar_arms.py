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
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

from summarize_runs import load_config, score  # noqa: E402

RESULTS = REPO / "nav" / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="fan", help="reference arm's `lidar` label")
ap.add_argument("--b", default="c1@0.30", help="treatment arm's `lidar` label")
ap.add_argument("--controller", default="braking")
args = ap.parse_args()

cfg = load_config()
# Sorted by filename, which is the timestamp, so the LAST match per episode is the newest
# run of that arm. Both arms are re-run for a comparison; an older stray run of the same
# label would otherwise be paired against a fresh one from the other.
rows = [r for r in (score(p, cfg) for p in sorted(RESULTS.glob("*.json"))) if r]
rows = [r for r in rows if r["controller"] == args.controller]

arm_a = {r["episode"]: r for r in rows if r["lidar"] == args.a}
arm_b = {r["episode"]: r for r in rows if r["lidar"] == args.b}

both = [e for e in arm_a if e in arm_b]
if not both:
    raise SystemExit(
        f"no episode has a run under BOTH '{args.a}' and '{args.b}' with controller "
        f"{args.controller}.\n"
        f"  '{args.a}' has {len(arm_a)} episodes, '{args.b}' has {len(arm_b)}.\n"
        "  Runs recorded before the `lidar` field existed carry '' -- they were the fan, "
        "but\n  nothing in the file says so, so they are not matched by --a fan."
    )
# Ladder order, so the table reads easiest-first like the ladder itself rather than
# alphabetically. Episodes present in only one arm are listed under the table, never
# silently dropped -- a missing run is usually a crash and is the thing worth seeing.
order = [e for e in cfg if e in both]
order += [e for e in both if e not in cfg]

print(f"\n  {'episode':<26}{'A ' + args.a:>14}{'B ' + args.b:>14}   flip   "
      f"{'closed A':>9}{'closed B':>9}{'  d':>7}")
print("  " + "-" * 92)

flips_won, flips_lost = [], []
closed_a, closed_b = [], []
for ep in order:
    a, b = arm_a[ep], arm_b[ep]
    ta = f"{'YES' if a['success'] else 'no':<4}{a['closest_m']:6.2f}m"
    tb = f"{'YES' if b['success'] else 'no':<4}{b['closest_m']:6.2f}m"
    if a["success"] and not b["success"]:
        flip, flips_lost = "LOST", flips_lost + [ep]
    elif b["success"] and not a["success"]:
        flip, flips_won = " WON", flips_won + [ep]
    else:
        flip = "  ="
    ca, cb = a["closed_frac"] * 100, b["closed_frac"] * 100
    closed_a.append(ca)
    closed_b.append(cb)
    print(f"  {ep:<26}{ta:>14}{tb:>14}   {flip}   {ca:8.0f}%{cb:8.0f}%{cb - ca:+7.0f}")

print("  " + "-" * 92)
na = sum(1 for e in order if arm_a[e]["success"])
nb = sum(1 for e in order if arm_b[e]["success"])
ma = sum(closed_a) / len(closed_a)
mb = sum(closed_b) / len(closed_b)
print(f"  {'TOTAL':<26}{na:>9} /{len(order):3d}{nb:>9} /{len(order):3d}"
      f"   {nb - na:+4d}   {ma:8.0f}%{mb:8.0f}%{mb - ma:+7.0f}")

print(f"\n  won  ({len(flips_won)}): {', '.join(flips_won) or '-'}")
print(f"  lost ({len(flips_lost)}): {', '.join(flips_lost) or '-'}")

only_a = sorted(set(arm_a) - set(arm_b))
only_b = sorted(set(arm_b) - set(arm_a))
if only_a or only_b:
    print(f"\n  NOT PAIRED -- these are excluded from every number above, and a missing "
          f"run is\n  usually a crash rather than a result:")
    if only_a:
        print(f"    only in {args.a}: {', '.join(only_a)}")
    if only_b:
        print(f"    only in {args.b}: {', '.join(only_b)}")

# Guard interventions are the one number that speaks about the SENSOR rather than about
# the policy, since the guard is the half of this change that does not go through a VLM.
# A denser scan should raise the count -- it sees more -- and a large fall would mean the
# robot stopped meeting obstacles, which on a fixed set of episodes means it stopped
# getting to them.
ga = sum(arm_a[e]["guard"] for e in order)
gb = sum(arm_b[e]["guard"] for e in order)
print(f"\n  guard interventions   {args.a} {ga}   {args.b} {gb}")
print("  (the guard is the half of this change that does not pass through the VLM;")
print("   more returns should mean MORE interventions, and a large drop means the robot")
print("   stopped reaching the obstacles rather than that it stopped hitting them)")

pa = sum(arm_a[e]["path_m"] for e in order)
pb = sum(arm_b[e]["path_m"] for e in order)
print(f"\n  path driven           {args.a} {pa:.0f} m   {args.b} {pb:.0f} m")
print("  (read next to `closed`, never alone -- a short path is a tidy route or a run")
print("   that parked, and warehouse_aisle6 has produced both)")

if not math.isclose(len(order), len(both)):
    print("\n  note: some paired episodes are missing from episodes.yaml and were "
          "appended after the ladder order.")
