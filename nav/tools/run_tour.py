"""Long-horizon, human-in-the-loop demo: several instructions in ONE session, each handed
over only after the robot has arrived at the previous place, and each starting where the
last one stopped (no teleport between them).

    NAV_EPISODES_CONFIG=episodes_tour.yaml NAV_POLICY_PORT=8766 \
        <isaacsim python.sh> nav/tools/run_tour.py [--prefix hospital_tour_]

The policy server must already be running (QVLA: nav/policy_server/launch_qwen.sh medium).
Episodes are taken from the table in NAV_EPISODES_CONFIG whose names start with --prefix,
in name order. Every step is an ordinary EpisodeResult saved to nav/results/, with its own
menus/decisions/chase run directory, so nav/tools/make_run_video.py works per step; a step
that times out is reported and the next instruction is still given from where it stopped,
as a person would. The step list goes to nav/results/tours/<stamp>_<prefix>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
# Not this directory first: nav/tools/profile.py shadows the stdlib `profile`, which Kit's
# warp imports (CLAUDE.md, P1 fixes).
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(REPO / "nav" / "sim"))
sys.path.insert(0, str(REPO / "nav"))

from episode import load_episodes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="hospital_tour_")
    ap.add_argument("--controller", default="braking")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=int(os.environ.get("NAV_POLICY_PORT", "8766")))
    a = ap.parse_args()

    eps = load_episodes()
    names = sorted(n for n in eps if n.startswith(a.prefix))
    # The human's correction for a step that did not arrive: the table's `correction`, else
    # the same instruction repeated once. Read off the raw YAML because Episode has no field
    # for it -- it is a property of the demo, not of an episode.
    from episode import CONFIG_PATH
    raw = (yaml.safe_load(Path(CONFIG_PATH).read_text()) or {}).get("episodes", {})
    if not names:
        print(f"no episodes starting with {a.prefix!r} in NAV_EPISODES_CONFIG", file=sys.stderr)
        return 2

    from run_navigation import NavigationRunner

    runner = NavigationRunner(episode=eps[names[0]], controller_name=a.controller,
                              policy_host=a.policy_host, policy_port=a.policy_port,
                              headless=True)
    runner.setup()
    steps = []
    try:
        for k, name in enumerate(names):
            ep = eps[name]
            correction = (raw.get(name) or {}).get("correction") or ep.instruction
            for attempt, instruction in enumerate([ep.instruction, correction]):
                if attempt and steps[-1]["success"]:
                    break
                who = "HUMAN CORRECTS" if attempt else "TOUR STEP"
                print(f"\n=== {who} {k + 1}/{len(names)}: {instruction}", flush=True)
                t0 = time.time()
                res = runner.run_episode(instruction, None, name,
                                         continue_from_here=(k > 0 or attempt > 0))
                row = {
                    "step": k + 1, "attempt": attempt + 1, "episode": name,
                    "instruction": instruction, "correction": bool(attempt),
                    "success": bool(res.success), "elapsed_s": round(res.elapsed_s, 1),
                    "final_distance_m": round(res.final_distance_m, 2),
                    "path_length_m": round(res.path_length_m, 2),
                    "record_dir": str(runner.record_dir) if runner.record_dir else None,
                    "wall_s": round(time.time() - t0, 1),
                }
                steps.append(row)
                print(f"=== TOUR STEP {k + 1}{' (corrected)' if attempt else ''} "
                      f"{'ARRIVED' if res.success else 'NOT ARRIVED'}: "
                      f"{row['final_distance_m']} m from the goal after {row['elapsed_s']} s "
                      f"({row['path_length_m']} m driven)", flush=True)
    finally:
        out = REPO / "nav" / "results" / "tours"
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{a.prefix.rstrip('_')}.json"
        f.write_text(json.dumps({"prefix": a.prefix, "steps": steps}, indent=1))
        done = {s["step"] for s in steps if s["success"]}
        print(f"\nTOUR_DONE {len(done)}/{len(names)} steps arrived, "
              f"{sum(s['correction'] for s in steps)} human correction(s) -> {f}", flush=True)
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
