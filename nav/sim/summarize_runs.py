#!/usr/bin/env python3
"""Score saved runs against DynaNav's own numbers for the same episodes.

Our result is only meaningful next to the reference one. "Final distance 6.4 m" is
not a grade; "6.4 m where DynaNav finished at 1.49 m with spl 1.00" is. So this
reads `nav/results/*.json`, joins each run to the DynaNav figures carried in
`episodes.yaml`, and prints both.

Two scoring rules worth stating, because both were got wrong once here:

  Rank by SPL, not by navigation error. Every DynaNav success records ~1.49 m of
  final error, because the episode terminates the instant the robot crosses the
  1.5 m threshold. Sorting successes by error therefore sorts by nothing.

  Report CLOSEST APPROACH alongside final distance. A robot that reaches 2.8 m and
  then drives away ends at 22 m, and those two numbers describe completely
  different failures -- one is a stopping problem, the other is a steering problem,
  and they have opposite fixes. Final distance alone cannot tell them apart. This
  is not hypothetical: warehouse_aisle6 did exactly that.

Usage:
    python3 nav/sim/summarize_runs.py                       # every run, grouped
    python3 nav/sim/summarize_runs.py --controller guided   # one controller
    python3 nav/sim/summarize_runs.py --latest hospital_vending --oneline
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "nav/results"
CONFIG = REPO / "nav/config/episodes.yaml"
SUCCESS_THRESHOLD_M = 1.5


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text()).get("episodes", {})


def reference(spec: dict | None) -> dict:
    """DynaNav's numbers, parsed out of the generated comment block if present.

    They live in a comment rather than a field on purpose: they describe a
    different implementation's run and must never be mistaken for something our
    runner produced. Absent for hand-written episodes, which is the honest answer
    for warehouse -- DynaNav never scored it.
    """
    if not spec:
        return {}
    return {
        k: spec[k]
        for k in ("dynanav_spl", "dynanav_path_m", "dynanav_duration_s")
        if k in spec
    }


def score(path: Path, cfg: dict) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    spec = cfg.get(d.get("episode"))
    goal = (spec or {}).get("goal") or d.get("goal")
    trace = d.get("trace") or []

    closest = float("nan")
    if goal and trace:
        closest = min(math.dist(p[:2], goal[:2]) for p in trace)

    final = d.get("final_distance_m")
    if final is None and goal and trace:
        final = math.dist(trace[-1][:2], goal[:2])

    initial = d.get("initial_distance_m") or float("nan")
    path_len = d.get("path_length_m") or 0.0
    success = bool(d.get("success")) or (
        closest == closest and closest <= SUCCESS_THRESHOLD_M
    )

    # SPL as the benchmark defines it: success weighted by how much the robot
    # overdrove the shortest path. Zero for a failure, by construction.
    spl = 0.0
    if success and path_len > 1e-6:
        spl = initial / max(path_len, initial)

    # How much of the gap the robot actually closed. This is the number that
    # separates "went nowhere" from "went almost all the way", and neither final
    # distance nor SPL shows it.
    closed = (
        (initial - closest) / initial if initial == initial and initial > 1e-6 else 0.0
    )

    plans = d.get("plans") or []
    asked = [p[1] for p in plans]
    guide = [p[4] for p in plans if len(p) > 4 and p[4] is not None]

    return {
        "file": path.name,
        "episode": d.get("episode", "?"),
        "controller": d.get("controller", "?"),
        "success": success,
        "initial_m": initial,
        "closest_m": closest,
        "final_m": final if final is not None else float("nan"),
        "path_m": path_len,
        "spl": spl,
        "closed_frac": closed,
        "calls": d.get("policy_calls", 0),
        "guard": d.get("guard_interventions", 0),
        "elapsed_s": d.get("elapsed_s") or 0.0,
        "asked_mean": (sum(asked) / len(asked)) if asked else float("nan"),
        "guide_mean": (sum(guide) / len(guide)) if guide else float("nan"),
        "guide_frac": (len(guide) / len(plans)) if plans else 0.0,
        **reference(spec),
    }


def fmt(v: float, spec: str = "6.2f") -> str:
    return "     -" if v != v else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controller", help="only runs with this controller")
    ap.add_argument("--latest", metavar="EPISODE", help="only the newest run of it")
    ap.add_argument("--oneline", action="store_true", help="one terse line, for scripts")
    args = ap.parse_args()

    cfg = load_config()
    rows = [r for r in (score(p, cfg) for p in sorted(RESULTS.glob("*.json"))) if r]
    if args.controller:
        rows = [r for r in rows if r["controller"] == args.controller]
    if args.latest:
        rows = [r for r in rows if r["episode"] == args.latest]
        rows = rows[-1:]

    if not rows:
        print("no matching runs" if not args.oneline else "NO-RUN")
        return

    if args.oneline:
        r = rows[-1]
        verdict = "SUCCESS" if r["success"] else "FAIL"
        print(
            f"{verdict} {r['episode']}/{r['controller']}: "
            f"{r['initial_m']:.1f} -> closest {fmt(r['closest_m'], '.2f')} m "
            f"(final {fmt(r['final_m'], '.1f')}), closed {r['closed_frac'] * 100:.0f}%, "
            f"spl {r['spl']:.2f}, path {r['path_m']:.1f} m, guard {r['guard']}"
        )
        return

    hdr = (
        f"{'episode':<24}{'ctrl':<10}{'ok':<4}{'init':>7}{'closest':>8}{'final':>7}"
        f"{'closed':>8}{'spl':>6}{'path':>7}{'guard':>6}{'asked':>7}{'guide':>7}"
        f"{'  DynaNav':<18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ref = (
            f"  spl {r['dynanav_spl']:.2f} {r['dynanav_path_m']:.0f}m"
            if "dynanav_spl" in r
            else "  (never scored)"
        )
        print(
            f"{r['episode']:<24}{r['controller']:<10}"
            f"{'YES' if r['success'] else 'no':<4}"
            f"{fmt(r['initial_m'], '7.1f')}{fmt(r['closest_m'], '8.2f')}"
            f"{fmt(r['final_m'], '7.1f')}{r['closed_frac'] * 100:7.0f}%"
            f"{r['spl']:6.2f}{r['path_m']:7.1f}{r['guard']:6d}"
            f"{fmt(r['asked_mean'], '7.1f')}{fmt(r['guide_mean'], '7.1f')}{ref:<18}"
        )

    done = [r for r in rows if r["success"]]
    print(
        f"\n{len(done)}/{len(rows)} succeeded"
        f"  |  mean closed {sum(r['closed_frac'] for r in rows) / len(rows) * 100:.0f}%"
        f"  |  mean spl {sum(r['spl'] for r in rows) / len(rows):.2f}"
    )
    print(
        "asked = mean heading the ACTION head requested (deg, + is left); "
        "guide = same from the 9s TEXT head."
    )


if __name__ == "__main__":
    main()
