"""Does Q-VLA-direct's text->plan path survive contact with real trajectories?

`server_qwen.py` replaces TIC-VLA's 10.27 M action expert with six numbers written out
in text. Three things stand between those numbers and a waypoint the controller can
steer on, and each can be checked WITHOUT the 27B loaded, which is the point of this
file -- these are where the bugs live, not in the prompt.

  1. **Densification loss.** Six control points become 30 waypoints. Measured against
     real action-expert plans pulled from the live TIC-VLA server, so the question is
     not "is the spline accurate" but "does it lose anything the controller uses".
  2. **Reframing.** The plan is written in the body frame of the moment generation
     started and consumed up to ~2 s later, after the robot has translated and turned.
     A sign error here is invisible in a unit test and catastrophic in an episode: it
     steers the robot by exactly twice the heading change, in the wrong direction. So
     it is checked against an independently constructed ground truth rather than
     against itself.
  3. **Parse robustness.** The model is not fine-tuned on this format, so it will wrap
     the answer in prose, emit a stray third component, or hallucinate a scale. Every
     one of those must degrade to "reuse the last plan", never to a plausible-looking
     wrong plan -- `nav/README.md` records what a plausible wrong number costs.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/check_qvla_direct.py \\
        --plans /path/to/real_plans.npy      # (N, 30, 2), from the TIC-VLA server
    # ...or with no --plans, against synthetic arcs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_server"))
from server_qwen import (  # noqa: E402
    CTRL_TIMES, DT, N_WAYPOINTS, densify, parse_control_points, reframe,
)

# What the numbers have to be compared against. Both are measured facts from
# nav/README.md, not thresholds picked to make this pass.
SUCCESS_RADIUS_M = 1.5      # the benchmark's scored threshold
LOSING_HEADING_DEG = 25.0   # the deficit that actually loses warehouse_aisle6


def synth(n: int = 24) -> np.ndarray:
    """Arcs in the range the real plans occupy: 0.73 m/s mean, gentle curvature."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        v, k = rng.uniform(0.3, 1.2), rng.uniform(-0.6, 0.6)
        x, y, th, pts = 0.0, 0.0, 0.0, []
        for _ in range(N_WAYPOINTS):
            th += k * v * DT
            x += v * DT * math.cos(th)
            y += v * DT * math.sin(th)
            pts.append((x, y))
        out.append(pts)
    return np.array(out)


def emit(plan: np.ndarray) -> str:
    """Render a plan the way the model is asked to: six control points, two decimals."""
    idx = [int(round(t / DT)) - 1 for t in CTRL_TIMES]
    return ("<answer>" + ", ".join(f"({plan[i][0]:.2f}, {plan[i][1]:.2f})" for i in idx)
            + "</answer>")


def heading(p: np.ndarray, i: int) -> float:
    return math.degrees(math.atan2(p[i][1], p[i][0]))


def check_roundtrip(P: np.ndarray) -> bool:
    pos, h1, h3, arc = [], [], [], []
    for p in P:
        ctrl = parse_control_points(emit(p))
        assert ctrl is not None, "our own emission failed to parse"
        r = densify(ctrl)
        pos.append(np.linalg.norm(r - p, axis=-1).max())
        h1.append(abs(heading(r, 9) - heading(p, 9)))
        h3.append(abs(heading(r, 29) - heading(p, 29)))
        arc.append(abs(np.linalg.norm(np.diff(np.vstack([[0, 0], r]), axis=0), axis=-1).sum()
                       - np.linalg.norm(np.diff(np.vstack([[0, 0], p]), axis=0), axis=-1).sum()))
    print(f"  max position error   {max(pos):.4f} m   "
          f"({100 * max(pos) / SUCCESS_RADIUS_M:.1f}% of the 1.5 m success radius)")
    print(f"  heading err @1.0s    {np.mean(h1):.2f} deg mean, {max(h1):.2f} max   "
          f"({100 * max(h1) / LOSING_HEADING_DEG:.1f}% of the 25 deg that loses episodes)")
    print(f"  heading err @3.0s    {np.mean(h3):.2f} deg mean, {max(h3):.2f} max")
    print(f"  arc-length error     {np.mean(arc):.4f} m mean   "
          "(this is the requested SPEED, which `braking` obeys)")
    ok = max(pos) < 0.25 and max(h1) < 5.0
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def check_reframe(P: np.ndarray) -> bool:
    """Drive partway along each plan, then check the reframed remainder is still right.

    The ground truth goes through WORLD coordinates, which is the whole point. An
    earlier version of this check built its expectation from the same rotate-and-
    subtract expression `reframe` uses; it reported 0.00e+00 error and proved only that
    numpy is deterministic. A sign convention cannot be validated against itself.

    So: put the plan in the world from a generation pose with a deliberately non-zero
    heading (a bug in the rotation is invisible at gyaw=0), drive the robot to a later
    point on it, and derive what each remaining waypoint must be in the new body frame
    from the plain world->body transform. `reframe` is then fed exactly what the runner
    would send it -- dx, dy expressed in the GENERATION frame, per
    run_navigation.py:518-522 -- and has to agree.
    """
    worst_pos, worst_head, worst_len = 0.0, 0.0, 0
    for p in P:
        for gyaw in (0.0, 0.7, -2.4, math.pi):          # must hold for any heading
            g = np.array([3.1, -4.2])                    # arbitrary generation position
            world = g + p @ _rot(gyaw).T                 # plan in world coordinates

            for age in (0.5, 1.0, 1.7, 2.5):
                i = int(round(age / DT)) - 1
                c_pos = world[i]                         # robot drove along its own plan
                j = max(0, i - 1)
                d = world[i] - world[j]
                c_yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-9 else gyaw

                # What the runner sends: world displacement rotated into the GENERATION
                # frame, and the heading change since generation.
                dxy = _rot(-gyaw) @ (c_pos - g)
                dyaw = c_yaw - gyaw

                k = int(np.ceil(age / DT))
                live = reframe(p, dxy[0], dxy[1], dyaw, 0.0)[k:]
                # Ground truth, built the other way round: world -> current body frame.
                expect = (world - c_pos) @ _rot(-c_yaw).T
                expect = expect[k:]

                if len(live) != len(expect):
                    print(f"  length mismatch at age {age}: {len(live)} vs {len(expect)}")
                    return False
                if len(live) == 0:
                    continue
                worst_pos = max(worst_pos, np.abs(live - expect).max())
                worst_len = max(worst_len, len(live))
                gt = math.degrees(math.atan2(expect[0][1], expect[0][0]))
                gv = math.degrees(math.atan2(live[0][1], live[0][0]))
                worst_head = max(worst_head, abs(gt - gv))

    print(f"  max coordinate error {worst_pos:.2e} m   (float noise, not zero: the two "
          "sides are computed differently)")
    print(f"  max heading error    {worst_head:.2e} deg")
    print(f"  longest remainder    {worst_len} of {N_WAYPOINTS} waypoints")
    # A flipped rotation sign shows up as ~2*dyaw -- tens of degrees, not 1e-12.
    ok = worst_pos < 1e-9 and worst_head < 1e-9
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_shrinkage(P: np.ndarray) -> bool:
    """A stale plan must run out, and the runner must still get something to steer on."""
    print(f"  {'age':>6}{'waypoints left':>17}{'sim seconds left':>19}")
    ok = True
    for age in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        n = len(reframe(P[0], 0.0, 0.0, 0.0, age))
        print(f"  {age:>5.1f}s{n:>17}{n * DT:>18.1f}s")
        if age <= 2.0 and n < 5:
            ok = False
    print(f"  -> {'PASS' if ok else 'FAIL'}  "
          "(nav/README.md measures staleness at 1.70 s mean, 2.00 s max)")
    return ok


def check_parsing() -> bool:
    good = "<answer>(0.35, 0.01), (0.70, 0.03), (1.05, 0.07), (1.40, 0.12), (1.75, 0.19), (2.10, 0.27)</answer>"
    cases: list[tuple[str, str, bool]] = [
        ("clean", good, True),
        ("prose around it",
         "Sure! Here is the plan.\n" + good + "\nLet me know if you need more.", True),
        ("no tags", good.replace("<answer>", "").replace("</answer>", ""), True),
        ("stray theta, TIC-VLA habit",
         "<answer>(0.35, 0.01, 0.03), (0.70, 0.03, 0.04), (1.05, 0.07, 0.07)</answer>", True),
        ("hallucinated scale (metres->tens)",
         "<answer>(35.0, 1.0), (70.0, 3.0), (105.0, 7.0)</answer>", False),
        ("refusal", "I cannot determine waypoints from these images.", False),
        ("one point only", "<answer>(0.35, 0.01)</answer>", False),
        ("empty", "", False),
    ]
    ok = True
    for name, text, want in cases:
        got = parse_control_points(text) is not None
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} {name:34} parsed={got!s:5} expected={want}")

    # Monotonicity: this base does not reverse, so a backwards x must be clamped, and
    # the clamp must not silently pass a plan through unchanged.
    back = parse_control_points("<answer>(0.35, 0.0), (0.20, 0.0), (0.90, 0.0)</answer>")
    mono = back is not None and bool(np.all(np.diff(back[:, 0]) >= 0))
    print(f"  {'ok ' if mono else 'BAD'} backwards x clamped to monotone         "
          f"{None if back is None else back[:, 0].tolist()}")
    ok = ok and mono
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", help="(N, 30, 2) .npy of real action-expert plans")
    args = ap.parse_args()

    if args.plans and Path(args.plans).is_file():
        P = np.load(args.plans)
        src = f"{len(P)} REAL action-expert plans from {args.plans}"
    else:
        P = synth()
        src = f"{len(P)} synthetic arcs (no --plans given)"
    print(f"source: {src}\n")

    results = []
    for title, fn in (
        ("1. SIX CONTROL POINTS -> 30 WAYPOINTS", lambda: check_roundtrip(P)),
        ("2. REFRAMING A STALE PLAN", lambda: check_reframe(P)),
        ("3. A STALE PLAN RUNS OUT", lambda: check_shrinkage(P)),
        ("4. PARSING AN UNTRAINED MODEL'S OUTPUT", check_parsing),
    ):
        print(title)
        results.append(fn())
        print()

    print("=" * 70)
    print("ALL PASS" if all(results) else "FAILURES ABOVE")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
