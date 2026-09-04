#!/usr/bin/env python3
"""Prove a closed-loop run really offered, served and executed in-place turns.

    nav/tools/check_pivot_run.py --episode office_hallway_turn2

WHY THIS EXISTS, and it is the same argument as `check_bounded_mode.py`. The pivot
crosses three process boundaries -- an env var read by the policy server, a float on an
HTTP response, a rotation performed by the simulator -- and each of them fails silently:

  * the server never sees QVLA_MENU_PIVOTS   -> a plain menu, a normal-looking ladder
  * the runner predates `pivot_rad`          -> the field is ignored and every turn the
                                                model chose is executed as a STOP
  * the sim never accumulates the yaw        -> the deadline expires and the robot
                                                re-decides from the same direction

All three produce thirteen complete episodes. The first two are indistinguishable from
"the model was offered the turn and never wanted it", which is a REAL possible finding and
therefore the one a broken wire is most likely to be mistaken for.

THE FINGERPRINT is an agreement, not a threshold. The server counts turns it HANDED OUT
and the runner counts turns it PERFORMED, they are incremented in different processes from
different code, and under a working pipeline they are equal. Unequal means the field is
being dropped or served twice; equal and zero means the model genuinely did not choose one,
which is a finding about the prompt and not about the plumbing.
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


def newest_result(episode: str, controller: str) -> Path | None:
    runs = sorted(RESULTS.glob(f"*_{episode}_{controller}.json"),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def health(port: int) -> dict:
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
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--expect-pivots", type=int, default=0,
                    help="minimum turns the model must have chosen; 0 means "
                         "'do not require any', which is the right default for a "
                         "preflight that is checking wiring rather than behaviour")
    args = ap.parse_args()

    path = newest_result(args.episode, args.controller)
    if path is None:
        print(f"!! no result file for {args.episode}/{args.controller}")
        return 1
    raw = json.loads(path.read_text())
    print(f"-- checking {path.name}")

    fails: list[str] = []
    info = health(args.port)

    # --- the menu really carried the turns --------------------------------------------
    if not info:
        fails.append("no /health -- cannot tell whether the turns were on the menu at all")
    else:
        if not info.get("menu_pivots"):
            fails.append("server reports menu_pivots=false -- QVLA_MENU_PIVOTS never "
                         "reached it, so the model was never offered a turn")
        # Derived from what the server says it drew, NOT hardcoded. This was `!= 10`, which
        # was right for exactly one menu (seven arcs plus two turns plus STOP) and became a
        # false alarm the moment the arc set grew -- a check that fails when the thing it
        # checks is merely CONFIGURED differently teaches people to ignore it.
        n_arcs = info.get("n_arcs")
        want = (n_arcs + 3) if isinstance(n_arcs, int) else None
        arcs_plus = info.get("stop_label")
        if want is not None and arcs_plus != want:
            fails.append(f"server reports stop_label={arcs_plus} with n_arcs={n_arcs}, "
                         f"expected {want} ({n_arcs} arcs + 2 turns + 1)")
        print(f"   menu             pivots={info.get('menu_pivots')} "
              f"angle={info.get('pivot_deg')} deg arcs={n_arcs} "
              f"frames={info.get('menu_frames')} stop_label={arcs_plus}")

    # --- the turns crossed the wire ---------------------------------------------------
    # `pivots` is absent, not zero, on a runner that predates the field. The two mean
    # opposite things -- "this code cannot execute a turn" against "no turn was chosen" --
    # and a `.get(..., 0)` would erase the distinction this whole tool exists to draw.
    if "pivots" not in raw:
        fails.append("the result file has no `pivots` field -- this run was recorded by "
                     "a runner that cannot execute an in-place turn")
        executed = None
    else:
        executed = int(raw["pivots"])
    served = info.get("pivots") if info else None

    print(f"   turns            served={served} executed={executed} "
          f"of {len(raw.get('plans') or [])} decisions")

    if served is not None and executed is not None and served != executed:
        fails.append(f"the server handed out {served} turns and the robot performed "
                     f"{executed} -- the pivot_rad field is being dropped or re-served")
    if executed is not None and executed < args.expect_pivots:
        fails.append(f"only {executed} turns were taken, expected at least "
                     f"{args.expect_pivots}")

    print(f"   result           {'SUCCESS' if raw.get('success') else 'FAIL'} "
          f"at {raw.get('final_distance_m'):.2f} m, guard "
          f"{raw.get('guard_interventions')}, recoveries {raw.get('recoveries')}")

    if fails:
        print()
        for f in fails:
            print(f"!! {f}")
        return 1
    if executed:
        print(f"\n-- PIVOTS CONFIRMED END TO END ({executed} turns chosen, served and "
              f"performed)")
    else:
        print("\n-- WIRING OK, but the model chose no turn in this episode. That is a "
              "finding about the\n   prompt, not a fault: the plumbing is proven by "
              "menu_pivots and by the counters agreeing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
