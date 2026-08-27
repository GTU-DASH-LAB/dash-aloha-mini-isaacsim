"""Ask both policies the same questions, from a real run, and score the answers.

The question this exists to settle: **is the 10.27 M action expert doing work that a
large VLM cannot do in text?** `server_qwen.py` replaces it with six coordinates written
out as words. That is either roughly as good, in which case the whole slow/fast
architecture is unnecessary here, or it is not, in which case it should be visible in
the numbers rather than argued about.

Two things this is careful about, because the obvious version of it is misleading:

  * **Every call must be its OWN generation.** Both servers are asynchronous by design --
    `/predict` returns the plan it has, and starts a new one only if no generation is in
    flight. Walking a list of moments and calling `/predict` on each therefore compares
    one plan against itself N times, and prints N identical rows that look like a
    consistent model. `/reset` before each call is what forces the blocking path (both
    servers block when they have no plan), so each row is a fresh answer to its own frame.
  * **The states are real.** Frames, `previous_waypoints_text` and `robot_state` are
    rebuilt from a finished run by `probe_stop_decision.reconstruct` -- the same code
    path, so a difference between the two policies cannot be a difference between two
    reconstructions.

Scoring. `bearing_to_goal` is in the trace and is never sent to either policy, so
|asked - bearing| is a fair score for "did this plan point at the goal". It is a weak
score on its own -- a policy avoiding an obstacle *should* differ from the bearing -- so
the summary reports the TIC-VLA column beside it as the reference, not as truth.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/compare_qvla_ticvla.py \\
        --run nav/results/20260827-133558_office_hallway_turn_braking.json \\
        --frame-base 800 --n 8
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "policy_server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import PolicyClient  # noqa: E402
from probe_stop_decision import Moment, reconstruct  # noqa: E402

# The lookahead the `pursuit`/`braking` controllers actually steer on: 1.0 s into a plan
# that is 30 waypoints over 3.0 s. Heading at the far end is reported too, but nothing
# steers on it, so it is context rather than score.
LOOKAHEAD_IDX = 9
HORIZON_S = 3.0


def plan_stats(wp: list[list[float]]) -> tuple[float, float, float]:
    """(heading at 1.0 s, heading at the end, arc length) in degrees and metres."""
    xs = [p[0] for p in wp]
    ys = [p[1] for p in wp]
    k = min(LOOKAHEAD_IDX, len(wp) - 1)
    head1 = math.degrees(math.atan2(ys[k], xs[k]))
    headN = math.degrees(math.atan2(ys[-1], xs[-1]))
    arc = math.hypot(xs[0], ys[0]) + sum(
        math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j]) for j in range(len(xs) - 1))
    return head1, headN, arc


def ask(client: PolicyClient, mom: Moment, instruction: str) -> dict:
    # Reset first: with a plan in hand both servers answer from it and start a background
    # generation, so without this every moment after the first is scored on the previous
    # moment's answer. See the module docstring.
    client.reset()

    # robot_state[4:6] is the displacement SINCE THE PLAN IN HAND WAS GENERATED -- it is
    # the delay-compensation channel, and both servers apply it. Here the plan is being
    # generated right now, so that displacement is zero by construction, and the run's own
    # value describes motion that has not happened in this replay. Passing it through
    # translated every plan backwards by ~1 m and turned a dead-straight plan into a
    # heading of 177 deg, which reads exactly like a model driving in reverse.
    # It must be zeroed together with time_delay, and for both policies, or the comparison
    # is between two different questions. vx/vy/omega stay: those describe motion now.
    state = list(mom.robot_state)
    state[4] = state[5] = 0.0

    t0 = time.perf_counter()
    out = client.predict(
        image_paths=[str(f) for f in mom.frames],
        instruction=instruction,
        robot_state=state,
        current_step=0,
        time_delay=0.0,          # a fresh generation is not stale
        previous_waypoints_text=mom.prev_text,
    )
    wall = time.perf_counter() - t0
    head1, headN, arc = plan_stats(out["waypoints"])
    return {"head1": head1, "headN": headN, "arc": arc, "wall": wall,
            "n": len(out["waypoints"]), "reasoning": out.get("reasoning"),
            "waypoints": out["waypoints"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frame-base", type=int, required=True,
                    help="frame index of this run's FIRST policy call")
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--n", type=int, default=8, help="moments, spread across the run")
    ap.add_argument("--at", type=int, nargs="*", help="explicit plan indices instead")
    ap.add_argument("--ticvla-port", type=int, default=8765)
    ap.add_argument("--qvla-port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", help="write the raw answers here as json")
    ap.add_argument("--skip-ticvla", action="store_true")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text())
    instruction = run["instruction"]
    n_plans = len(run["plans"])

    if args.at:
        idxs = [i for i in args.at if 0 < i < n_plans]
    else:
        # Spread over the run rather than clustered: the interesting behaviour is at the
        # ends (setting off, and deciding to stop), and a middle-only sample sees neither.
        lo, hi = 4, n_plans - 2
        idxs = sorted({round(lo + (hi - lo) * k / max(1, args.n - 1))
                       for k in range(args.n)})

    moments = []
    for i in idxs:
        try:
            moments.append(reconstruct(run, i, args.frame_base, args.frame_dir))
        except FileNotFoundError as exc:
            print(f"skipping call {i}: {exc}", file=sys.stderr)
    if not moments:
        print("ERROR: no usable moments", file=sys.stderr)
        return 2

    print(f"run         : {Path(args.run).name}")
    print(f"instruction : {instruction}")
    print(f"moments     : {len(moments)} of {n_plans} calls "
          f"(t={moments[0].t:.1f}..{moments[-1].t:.1f}s)\n")

    tic = None if args.skip_ticvla else PolicyClient(args.host, args.ticvla_port)
    qvla = PolicyClient(args.host, args.qvla_port)
    if tic is not None:
        tic.wait_until_ready()
    qvla.wait_until_ready()

    hdr = (f"{'call':>5}{'t':>7}{'goal':>8} | {'TIC head@1s':>12}{'arc':>7}{'m/s':>6}"
           f" | {'Q-VLA head@1s':>14}{'arc':>7}{'m/s':>6}{'lat':>7}{'n':>4}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for mom in moments:
        t_ans = None if tic is None else ask(tic, mom, instruction)
        q_ans = ask(qvla, mom, instruction)
        rows.append({"call": mom.index, "t": mom.t, "goal": mom.goal_rel_deg,
                     "run_asked": mom.asked_deg, "ticvla": t_ans, "qvla": q_ans})
        tcol = ("      --      " if t_ans is None else
                f"{t_ans['head1']:>+11.2f}d{t_ans['arc']:>7.2f}"
                f"{t_ans['arc'] / HORIZON_S:>6.2f}")
        print(f"{mom.index:>5}{mom.t:>7.1f}{mom.goal_rel_deg:>+8.1f} | {tcol}"
              f" | {q_ans['head1']:>+13.2f}d{q_ans['arc']:>7.2f}"
              f"{q_ans['arc'] / HORIZON_S:>6.2f}{q_ans['wall']:>7.1f}{q_ans['n']:>4}")

    # ---- summary -------------------------------------------------------------------
    def err(rows_, key):
        return [abs(((r[key]["head1"] - r["goal"] + 180) % 360) - 180) for r in rows_]

    print()
    qe = err(rows, "qvla")
    print(f"Q-VLA  |asked - bearing to goal|  mean {statistics.fmean(qe):6.1f} deg   "
          f"median {statistics.median(qe):6.1f}")
    if tic is not None:
        te = err(rows, "ticvla")
        print(f"TIC-VLA|asked - bearing to goal|  mean {statistics.fmean(te):6.1f} deg   "
              f"median {statistics.median(te):6.1f}")
        agree = [abs(((r["qvla"]["head1"] - r["ticvla"]["head1"] + 180) % 360) - 180)
                 for r in rows]
        print(f"disagreement between the two    mean {statistics.fmean(agree):6.1f} deg   "
              f"max {max(agree):6.1f}")
        print("\nThe bearing column is scoring only and neither policy ever saw it. It is")
        print("not ground truth -- avoiding an obstacle means departing from it on")
        print("purpose -- so read the two error columns against each other, not against 0.")

    qa = [r["qvla"]["arc"] for r in rows]
    print(f"\nQ-VLA implied speed   mean {statistics.fmean(qa) / HORIZON_S:.2f} m/s "
          f"(range {min(qa) / HORIZON_S:.2f}..{max(qa) / HORIZON_S:.2f})")
    if tic is not None:
        ta = [r["ticvla"]["arc"] for r in rows]
        print(f"TIC-VLA implied speed mean {statistics.fmean(ta) / HORIZON_S:.2f} m/s "
              f"(range {min(ta) / HORIZON_S:.2f}..{max(ta) / HORIZON_S:.2f})")
    lat = [r["qvla"]["wall"] for r in rows]
    print(f"Q-VLA wall per call   mean {statistics.fmean(lat):.1f} s "
          f"(range {min(lat):.1f}..{max(lat):.1f}) -- BLOCKING, not the deployed path")

    health = qvla.health()
    print(f"\nQ-VLA health: generations={health.get('generations')} "
          f"parse_failures={health.get('parse_failures')} "
          f"gen_errors={health.get('gen_errors')} "
          f"empty_plans={health.get('empty_plans')}")
    if health.get("gen_errors", 0):
        # Loud, and separate from a parse failure: this is the stack failing, so every
        # number above is about nothing. Not a result to interpret.
        print(f"  ERROR: {health['gen_errors']} generations raised. The table above is "
              "not a measurement of the model.")
        print(f"  last: {health.get('last_reasoning')}")
    elif health.get("generations", 0) < len(rows):
        print("  WARNING: fewer generations than moments -- some rows reused a plan.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"run": args.run, "instruction": instruction, "rows": rows}, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
