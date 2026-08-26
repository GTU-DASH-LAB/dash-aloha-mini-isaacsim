"""Replay one recorded moment of a run and ask the policy the same question again.

`check_policy_sanity.py` answers "does language matter at all?" with a synthetic
single-frame prompt. This answers a narrower and more useful question: at THIS state,
in a real episode that failed, what would the policy have said if the sentence had been
different? Everything else is reconstructed from the run's own trace, so the sentence is
the only variable.

It exists because of a wrong diagnosis that was very hard to argue with. The two hospital
vending-machine episodes share a scene, a start pose and a goal, and differ only in the
instruction; one passes 3/4 and the other 0/3. Every failing run bends left at ~4.5 m out
while the goal sits dead ahead, and the failing sentence says "at front left". That story
is false — see CLAUDE.md. Swapping left for right moves the answer 1.0 deg, and the
sentence that never mentions the vending machine asks for the same left turn. Replaying
the state is what settled it; arguing about the sentence was not going to.

What gets rebuilt from the run, identically for every arm of the comparison:

  * four frames at [-9, -6, -3, 0] s, oldest first, as DynaNav samples them;
  * previous_waypoints_text in WaypointHistory's exact format;
  * robot_state (vx, vy, omega, dx, dy) and time_delay from the run's own numbers.

Frames live in the scratch dir and are overwritten by later runs, so this only works
while the run being probed is still the most recent one on disk. `--frame-base` is the
frame index of that run's FIRST policy call: the counter is process-wide and spans every
episode in the session, so it is `sum(policy_calls)` of the runs that preceded it.

Usage:
    /home/gtu-dsa/envs/tic-vla/bin/python nav/tools/probe_stop_decision.py \\
        --run nav/results/<stamp>_<episode>_braking.json \\
        --frame-base 140 --plan-index 27
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "policy_server"))
from client import PolicyClient  # noqa: E402

# The last two are the controls that matter. A directional pair that does not separate
# means the model is not reading the phrase as a direction; an instruction naming no
# object at all says how much of the answer is the scene rather than the sentence.
DEFAULT_INSTRUCTIONS = [
    "Go straight ahead and stop at the front of the vending machine at front left.",
    "Go straight ahead and stop at the vending machine at front.",
    "Go straight ahead and stop at the front of the vending machine at front right.",
    "Go straight ahead.",
    "Turn right.",
    "Stop. Do not move.",
]


def body_delta(p_from, yaw_from, p_to) -> tuple[float, float]:
    """World displacement expressed in the body frame at `p_from` (FLU: +x fwd, +y left)."""
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    c, s = math.cos(yaw_from), math.sin(yaw_from)
    return c * dx + s * dy, -s * dx + c * dy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an EpisodeResult json from nav/results/")
    ap.add_argument("--frame-base", type=int, required=True,
                    help="frame index of this run's FIRST policy call (see module docstring)")
    ap.add_argument("--plan-index", type=int, required=True,
                    help="which policy call to replay")
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--instructions", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=1,
                    help="predict() is effectively greedy; >1 only to re-confirm that")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text())
    trace, plans = run["trace"], run["plans"]
    i = args.plan_index
    if not 0 < i < len(plans):
        print(f"ERROR: --plan-index must be in 1..{len(plans) - 1}", file=sys.stderr)
        return 1
    t_now, asked, _reach, goal_rel = plans[i][:4]
    stale = plans[i][6] if len(plans[i]) > 6 and plans[i][6] is not None else 0.0

    # Plans are one per replan; the trace is sampled alongside them, so index i is the
    # same moment in both. A 3 s frame step is 3 / (t_now / i) plans.
    per_plan_s = t_now / i
    step = max(1, round(3.0 / per_plan_s))
    frame_dir = Path(args.frame_dir)
    idxs = [i - 3 * step, i - 2 * step, i - step, i]
    frames = [frame_dir / f"nav_{args.frame_base + k:06d}.jpg" for k in idxs if k >= 0]
    missing = [f for f in frames if not f.is_file()]
    if missing:
        # A later run overwrites the scratch dir. Guessing at substitutes would produce
        # a plausible number for a state that never happened.
        print(f"ERROR: {len(missing)} frame(s) gone, e.g. {missing[0]}.\n"
              "       The scratch dir only holds the most recent run.", file=sys.stderr)
        return 2

    # previous_waypoints_text: one sample per 1.0 s of sim time, body frame at the start
    # of each second, first (0,0,0) filtered out -- WaypointHistory's own contract.
    per_sample = max(1, round(1.0 / per_plan_s))
    parts = []
    for k in range(per_sample, i + 1, per_sample):
        dx, dy = body_delta(trace[k - per_sample], trace[k - per_sample][2], trace[k])
        if abs(dx) >= 1e-6 or abs(dy) >= 1e-6:
            parts.append(f"({dx:.2f}, {dy:.2f}, {0.0:.2f})")
    prev_text = (
        f"From 0.0s to current timestamp time is {t_now:.1f}s. "
        f"(a list of waypoints 1s in between): {', '.join(parts)}\n"
        "Each waypoint (x, y, z) is the displacement over the previous 1.0s. "
        "x is forward, y is left, z is up."
    )

    vx, vy = body_delta(trace[i - 1], trace[i - 1][2], trace[i])
    vx, vy = vx / per_plan_s, vy / per_plan_s
    omega = ((trace[i][2] - trace[i - 1][2] + math.pi) % (2 * math.pi) - math.pi) / per_plan_s
    back = max(0, i - int(round(stale / per_plan_s)))
    dx, dy = body_delta(trace[back], trace[back][2], trace[i])
    robot_state = [vx, vy, 0.0, omega, dx, dy]

    print(f"run          : {Path(args.run).name}")
    print(f"replaying    : call {i} at t={t_now:.1f}s, frame {frames[-1].name} "
          f"(+{len(frames) - 1} history)")
    print(f"the run asked: {asked:+.2f} deg   (goal was {goal_rel:+.1f} deg off the nose, "
          f"scoring only -- never sent)")
    print(f"robot_state  : vx={vx:+.2f} vy={vy:+.2f} omega={omega:+.2f} "
          f"dx={dx:+.2f} dy={dy:+.2f}   time_delay={stale:.2f}s")
    print(f"history      : {len(parts)} waypoints\n")

    client = PolicyClient(args.host, args.port)
    info = client.wait_until_ready()
    print(f"policy server ready on {info.get('device')}\n")

    results: dict[str, list[tuple[float, float]]] = {}
    for instr in (args.instructions or DEFAULT_INSTRUCTIONS):
        rows = []
        for r in range(args.repeats):
            # Reset so a cache built under the previous sentence cannot leak into this
            # one -- that would either fake a dependence on language or mask a real one.
            client.reset()
            out = client.predict(
                image_paths=[str(f) for f in frames],
                instruction=instr,
                robot_state=robot_state,
                current_step=0,
                time_delay=stale,
                previous_waypoints_text=prev_text,
            )
            wp = out["waypoints"]
            xs, ys = [p[0] for p in wp], [p[1] for p in wp]
            heading = math.degrees(math.atan2(ys[-1], xs[-1]))
            reach = sum(math.hypot(xs[k + 1] - xs[k], ys[k + 1] - ys[k])
                        for k in range(len(xs) - 1))
            rows.append((heading, reach))
            print(f"  heading={heading:+7.2f} deg   reach={reach:5.3f} m   {instr}")
        results[instr] = rows

    print("\n" + "=" * 92)
    print(f"{'mean heading':>13} {'sd':>6} {'mean reach':>11}   instruction")
    for instr, rows in results.items():
        h = [r[0] for r in rows]
        print(f"{statistics.fmean(h):>+12.2f}d "
              f"{(statistics.pstdev(h) if len(h) > 1 else 0.0):>6.2f} "
              f"{statistics.fmean(r[1] for r in rows):>10.3f}m   {instr}")
    spread = max(statistics.fmean(r[0] for r in v) for v in results.values()) - \
        min(statistics.fmean(r[0] for r in v) for v in results.values())
    print(f"\nspread across instructions: {spread:.2f} deg")
    print("Reach is the stop signal: 30 waypoints over a fixed 3.0 s, so reach/3 is the")
    print("speed the policy is asking for. A collapse below ~0.6 m is it deciding to stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
