#!/usr/bin/env python3
"""Prove a run actually used bounded-async planning, before a campaign is built on it.

    nav/tools/check_bounded_mode.py --episode office_hallway_turn2 --period 3.0

WHY THIS EXISTS. `NAV_PLAN_MODE` is read by the sim process and `wait_inflight` is
honoured by the policy server, and the two are separate processes started by separate
scripts. Every way that pairing can fail produces a COMPLETE, PLAUSIBLE result file:

  * the runner never exports the variable          -> a silently asynchronous ladder
  * the server predates `wait_inflight`            -> the field is ignored, silently
  * a `wait_fresh` left set somewhere wins over it -> a synchronous ladder wearing a
                                                      bounded label

None of those raise. All three yield thirteen numbers that would be written up as
"bounded async at 3 s" and compared against ladders that really were something else.
That is the same failure `run_level_ladder.sh` guards against for the thinking level,
and it deserves the same treatment: read the fingerprint back off the artifacts rather
than trusting the environment that was supposed to produce it.

THE FINGERPRINT, and it separates all three regimes without ambiguity. Under bounded
async the simulation only advances between calls, in whole planning periods, and the
plan handed back at call k is the generation started at call k-1. So:

    decision spacing   exactly the period, zero variance   (async: whatever fits)
    time_delay         exactly the period on every call    (sync: 0.0; async: varies)
                       except the first, which has nothing behind it
    plan_mode          "bounded" in the result file
    bounded_calls      nonzero on the server, one per decision

and `bounded_stalls` is the verdict on the mode rather than a check on it: zero means a
generation always fitted inside a period and the bound never had to engage, which is the
outcome bounded async exists to produce.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "nav/results"

# Sim steps at 1/60 s land a hair off a round number of seconds, and `plans` rounds to
# 2 decimals besides. 20 ms is far tighter than the difference between any two regimes
# here (3.0 s vs 0.0 s vs 1.5-2.5 s) and far looser than that quantisation.
TOL_S = 0.02


def newest_result(episode: str, controller: str) -> Path | None:
    runs = sorted(RESULTS.glob(f"*_{episode}_{controller}.json"),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def health(port: int) -> dict:
    """Server counters, or {} -- an unreachable server is reported, not fatal.

    The result file carries the sim side of the fingerprint on its own; /health adds the
    server side. Losing the second half weakens the check but does not invalidate the
    first, and refusing to verify at all because a status read failed is worse.
    """
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"   (could not read /health on :{port} -- {exc})")
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="office_hallway_turn2")
    ap.add_argument("--controller", default="braking")
    ap.add_argument("--period", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    path = newest_result(args.episode, args.controller)
    if path is None:
        print(f"!! no result file for {args.episode}/{args.controller} -- "
              f"the preflight episode did not produce a measurement")
        return 1
    raw = json.loads(path.read_text())
    print(f"-- checking {path.name}")

    fails: list[str] = []

    mode = raw.get("plan_mode")
    if mode != "bounded":
        fails.append(f"plan_mode={mode!r}, expected 'bounded' -- the runner did not see "
                     f"NAV_PLAN_MODE=bounded")
    period = float(raw.get("plan_period_s") or 0.0)
    if abs(period - args.period) > TOL_S:
        fails.append(f"plan_period_s={period}, expected {args.period}")

    plans = raw.get("plans") or []
    if len(plans) < 3:
        fails.append(f"only {len(plans)} decisions recorded -- too few to fingerprint")
        print("\n".join(f"!! {f}" for f in fails))
        return 1

    # Decision spacing. Sim time advances only between calls, so under any fixed-period
    # mode this is the period with no variance at all; under async it is whatever the
    # episode's own `replan_every_steps` gives and drifts with generation timing.
    times = [p[0] for p in plans]
    gaps = [round(b - a, 3) for a, b in zip(times, times[1:])]
    bad_gaps = [g for g in gaps if abs(g - args.period) > TOL_S]
    if bad_gaps:
        fails.append(f"{len(bad_gaps)} of {len(gaps)} decision gaps are not "
                     f"{args.period} s (saw {sorted(set(bad_gaps))[:6]})")

    # Staleness. This is the half that separates bounded from synchronous, which share
    # a period and a spacing and differ entirely in whether the robot drove during it.
    delays = [p[6] for p in plans]
    if abs(delays[0]) > TOL_S:
        fails.append(f"first decision reports time_delay={delays[0]}, expected 0.0 -- "
                     f"nothing has been generated yet at that point")
    bad_delays = [d for d in delays[1:] if abs(d - args.period) > TOL_S]
    if bad_delays:
        fails.append(f"{len(bad_delays)} of {len(delays) - 1} decisions report a "
                     f"staleness other than {args.period} s "
                     f"(saw {sorted(set(bad_delays))[:6]}) -- "
                     f"all-zero means synchronous, varying means asynchronous")

    print(f"   plan_mode        {mode}")
    print(f"   plan_period_s    {period}")
    print(f"   decisions        {len(plans)}")
    print(f"   spacing          {min(gaps):.2f}-{max(gaps):.2f} s")
    print(f"   staleness        {min(delays):.2f}-{max(delays):.2f} s "
          f"(first {delays[0]:.2f})")
    print(f"   think_wall_s     {raw.get('think_wall_s')} of wall_s {raw.get('wall_s')}")

    info = health(args.port)
    if info:
        calls = info.get("bounded_calls")
        stalls = info.get("bounded_stalls")
        stall_s = info.get("bounded_stall_s")
        print(f"   server           predictions={info.get('predictions')} "
              f"generations={info.get('generations')} bounded_calls={calls}")
        if calls is not None:
            # Only meaningful once the server is one that keeps these counters; a server
            # that predates them answers None, and printing "None stalls" next to a real
            # episode reads as "measured, and it was nothing".
            print(f"   bound engaged    {stalls} stalls, {stall_s} s total")
        if not calls:
            fails.append("server reports bounded_calls=0 -- it never saw "
                         "wait_inflight, so the run was asynchronous")
        if info.get("sync_calls"):
            fails.append(f"server reports sync_calls={info['sync_calls']} -- something "
                         f"is still sending wait_fresh, which outranks wait_inflight")
        if stalls == 0:
            print("   VERDICT          the bound never engaged: every generation fitted "
                  "inside a period,\n                    so this was free async with a "
                  "safety guarantee.")
        elif calls:
            print(f"   VERDICT          the bound engaged on {stalls}/{calls} decisions "
                  f"({100.0 * stalls / calls:.0f}%);\n                    the period is "
                  f"shorter than a generation and the robot paid {stall_s} s for it.")

    if fails:
        print()
        for f in fails:
            print(f"!! {f}")
        return 1
    print("\n-- BOUNDED ASYNC CONFIRMED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
