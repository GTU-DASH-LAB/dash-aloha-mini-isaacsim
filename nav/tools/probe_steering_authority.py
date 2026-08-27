"""Can this policy steer BOTH ways? Asked across several scenes, not one.

`probe_stop_decision.py` measures how much a *sentence* moves the answer at one moment.
This measures something narrower and, for a text-output policy, more dangerous: whether
the plan can express a turn to each side at all.

It exists because Q-VLA-direct failed in a way that a single-moment probe reads as
success. Told "Turn right.", Qwen3.8-27B writes a clean arc -- (0.32, -0.01),
(0.64, -0.03), (0.96, -0.06), ... a proper parabola with growing curvature. Told
"Turn left." it writes literal 0.00 six times. One frame looks like "the model steers,
just weakly". Five frames show it never emits a positive y at all, which is a different
problem with a different fix: the geometry is right, the SIGN is unreachable.

Two failure modes this separates, which is the whole point:

  * **A dead side.** Zero lateral offset for one direction across every scene. That is
    an encoding problem -- try `QVLA_FLIP_Y=1`, which asks in the mirrored frame and
    negates on the way back, and see whether the dead side follows the wording.
  * **An obstacle.** Zero lateral offset for one direction in SOME scenes. That is the
    policy correctly refusing to drive into a wall, and it is not a bug. Distinguishing
    the two needs several scenes; one frame cannot do it, and reading one frame as the
    first when it is the second is how a working policy gets "fixed" into a broken one.

Both directions are asked at the same moment with everything else identical, so the
comparison is within-frame and no reconstruction difference can explain a gap.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/probe_steering_authority.py \\
        --run nav/results/<stamp>_<episode>_braking.json --frame-base 800 --port 8766
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "policy_server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import PolicyClient  # noqa: E402
from probe_stop_decision import reconstruct  # noqa: E402

LOOKAHEAD_IDX = 9        # 1.0 s into a 3.0 s / 30-waypoint plan: what `pursuit` steers on
DEAD_DEG = 0.5           # below this the side is not being expressed at all


def ask(client: PolicyClient, mom, instruction: str) -> tuple[float, float, str]:
    """(heading at 1.0 s in degrees, arc length, raw text). Always a fresh generation."""
    client.reset()
    state = list(mom.robot_state)
    # Zero the delay-compensation channel: reset() means the plan is generated now, so
    # the displacement since generation is zero. See compare_qvla_ticvla.ask.
    state[4] = state[5] = 0.0
    out = client.predict(
        image_paths=[str(f) for f in mom.frames], instruction=instruction,
        robot_state=state, current_step=0, time_delay=0.0,
        previous_waypoints_text=mom.prev_text)
    wp = out["waypoints"]
    xs, ys = [p[0] for p in wp], [p[1] for p in wp]
    k = min(LOOKAHEAD_IDX, len(wp) - 1)
    arc = math.hypot(xs[0], ys[0]) + sum(
        math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j]) for j in range(len(xs) - 1))
    return math.degrees(math.atan2(ys[k], xs[k])), arc, out.get("reasoning") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frame-base", type=int, required=True)
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--at", type=int, nargs="*", default=[10, 25, 40, 58, 79])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--left", default="Turn left.")
    ap.add_argument("--right", default="Turn right.")
    ap.add_argument("--raw", action="store_true", help="print the generations too")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text())
    client = PolicyClient(args.host, args.port)
    client.wait_until_ready()

    print(f"run   : {Path(args.run).name}")
    print(f"asking: {args.left!r}  vs  {args.right!r}\n")
    print(f"{'call':>5}{'LEFT head@1s':>15}{'arc':>7} | {'RIGHT head@1s':>15}{'arc':>7}"
          f" | {'separation':>11}")
    print("-" * 64)

    lefts, rights, seps = [], [], []
    for i in args.at:
        try:
            mom = reconstruct(run, i, args.frame_base, args.frame_dir)
        except FileNotFoundError as exc:
            print(f"skipping call {i}: {exc}", file=sys.stderr)
            continue
        lh, la, lraw = ask(client, mom, args.left)
        rh, ra, rraw = ask(client, mom, args.right)
        lefts.append(lh)
        rights.append(rh)
        seps.append(lh - rh)
        print(f"{i:>5}{lh:>+14.2f}d{la:>7.2f} | {rh:>+14.2f}d{ra:>7.2f} | {lh - rh:>+10.2f}d")
        if args.raw:
            print(f"      L: {lraw.strip()}")
            print(f"      R: {rraw.strip()}")

    if not seps:
        print("ERROR: no usable moments", file=sys.stderr)
        return 2

    # A correctly-signed policy turns left POSITIVE and right NEGATIVE, so separation is
    # positive and large. Near zero means the words are not reaching the plan; a side
    # pinned at 0.00 everywhere means that side cannot be expressed.
    dead_l = sum(1 for h in lefts if abs(h) < DEAD_DEG)
    dead_r = sum(1 for h in rights if abs(h) < DEAD_DEG)
    print(f"\nseparation (left - right)  mean {statistics.fmean(seps):+.2f} deg   "
          f"min {min(seps):+.2f}   max {max(seps):+.2f}")
    print(f"left  turns below {DEAD_DEG} deg: {dead_l}/{len(lefts)}")
    print(f"right turns below {DEAD_DEG} deg: {dead_r}/{len(rights)}")

    if dead_l == len(lefts) or dead_r == len(rights):
        side = "LEFT" if dead_l == len(lefts) else "RIGHT"
        print(f"\n{side} is dead in EVERY scene. That is not obstacle avoidance -- a wall "
              "does not\nfollow the robot through an episode. It is the sign being "
              "unreachable in this output\nformat. Re-run with QVLA_FLIP_Y=1: if the dead "
              "side follows the wording, the encoding\nis the bug and rewording the "
              "prompt will not fix it.")
    elif dead_l or dead_r:
        print("\nOne side is dead in SOME scenes only, which is what avoiding an obstacle "
              "looks like.\nCheck those frames before treating it as a defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
