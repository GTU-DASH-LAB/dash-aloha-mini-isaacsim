"""Does the controller turn as much as the plan asked, or more?

Reported from watching a run: the robot pivots on the spot down a corridor that is
straight and that the model is steering straight down. The model was not the cause --
over 85 decisions of `hospital_down_hallway` its median chosen curvature was 0.000 1/m
and its mean signed curvature -0.004, i.e. straight more than half the time and
symmetric otherwise, while the robot turned 136 deg per metre travelled. Ten times what
the chosen arcs add up to has to come from the controller.

The mechanism is one missing factor in `PursuitController`:

    w_ff = 0.5 * v_cmd * kappa        # scales with speed
    w_fb = k_angular * yaw_err_filt   # does NOT

Yaw rate is degrees per SECOND; path shape is degrees per METRE. A term with no `v` in
it therefore bends the path harder the slower you go. For a gentle arc the lookahead
error is about `kappa/2` rad, so

    commanded  = kappa * (0.5 * v + 0.4)      required = kappa * v
    ratio      = 0.5 + k_angular / (2 * v)

which is 1.00 at v = 0.8 m/s and grows without bound as v falls. DynaNav's Nova Carter
planned at ~0.73 m/s, so their constants were right where they sit; ours are not once
cruise drops to 0.45 to stop the robot overshooting its target.

This measures that ratio on the REAL objects -- `make_arcs` curvatures through
`plan_from_kappa` through `_lookahead_point` -- rather than through the small-angle
algebra above, because the algebra is what suggested the fix and cannot also be the
evidence for it.

Usage:
    python3 nav/tools/check_turn_rate.py
    python3 nav/tools/check_turn_rate.py --speeds 0.2,0.45,0.7,0.8,1.5
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from arc_menu import DEFAULT_CURVATURES, plan_from_kappa  # noqa: E402
from controllers import make_controller  # noqa: E402

N_WAYPOINTS, DT = 100, 0.1          # the server's plan format
MAX_YAW = 1.2                        # episodes.yaml max_yaw_rate_radps
TOL = 0.10                           # 10% is the most a shape error should ever be


def ratio_at(controller_name: str, speed: float, kappa: float) -> tuple[float, float]:
    """(commanded deg/m, required deg/m) for one arc driven at one speed.

    `speed` is passed as the controller's own v_max as well as being baked into the
    plan. Setting only the plan speed looks right and is not: `pursuit` has
    `obey_plan_speed=False`, so it never reads the plan's speed and would have driven
    every row at 1.5 m/s while the table printed a speed column that did nothing.
    """
    plan = plan_from_kappa(kappa, speed, N_WAYPOINTS, DT)
    c = make_controller(controller_name, speed, MAX_YAW)
    cmd = c(np.asarray(plan, dtype=float))
    if cmd.vx < 1e-9:
        return float("nan"), math.degrees(abs(kappa))
    return math.degrees(abs(cmd.omega) / cmd.vx), math.degrees(abs(kappa))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="braking")
    ap.add_argument("--speeds", default="0.20,0.35,0.45,0.60,0.70,0.80,1.00,1.50")
    args = ap.parse_args()
    speeds = [float(s) for s in args.speeds.split(",")]

    # Curvature is signed and the law is odd in kappa, so the positive half says
    # everything; straight is excluded because 0/0 is not a ratio.
    kappas = [k for k in DEFAULT_CURVATURES if k > 1e-9]

    print(f"controller {args.controller!r}, {len(kappas)} arcs, yaw cap {MAX_YAW} rad/s")
    print("\nhow far the robot BENDS per metre, against what the chosen arc asked for")
    print(f"\n{'speed':>6}  " + "  ".join(f"k={k:+.2f}" for k in kappas) + "   worst")
    print(f"{'m/s':>6}  " + "  ".join("      " for _ in kappas) + "   ratio")
    rows = []
    for v in speeds:
        cells, worst = [], 0.0
        for k in kappas:
            got, want = ratio_at(args.controller, v, k)
            r = got / want if want > 1e-9 else float("nan")
            cells.append(f"{r:6.2f}")
            worst = max(worst, r)
        rows.append((v, worst))
        flag = "  <-- over-turns" if worst > 1.0 + TOL else ""
        print(f"{v:6.2f}  " + "  ".join(cells) + f"   {worst:5.2f}{flag}")

    print("\n1.00 means the robot draws exactly the arc the model picked. Above 1 it "
          "cuts a\ntighter curve than it was told to, which is what pivoting in a "
          "straight corridor is:\nthe menu keeps answering 'straight', the odd curved "
          "answer gets driven several times\nharder than it asked, and the robot "
          "sweeps back and forth across a corridor it should\nbe going down.")

    bad = [(v, r) for v, r in rows if r > 1.0 + TOL]
    print("\n" + "=" * 78)
    if bad:
        for v, r in bad:
            print(f"  FAIL  at {v:.2f} m/s the path bends {r:.2f}x what the plan asked")
        print(f"\nSpeed and turn rate are not independent here, so 'slow it down' and "
              f"'stop it\nspinning' pull against each other -- which is the bug, not a "
              f"trade-off to tune.")
    else:
        print("  PASS - path shape is the same at every speed, so cruise speed can be "
              "set for\n  overshoot alone and it will not change how sharply the robot "
              "turns.")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
