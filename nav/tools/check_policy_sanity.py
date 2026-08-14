"""Phase 9 gate: does TIC-VLA's output actually depend on the instruction?

Why this exists (../plan.md, finding 3). This machine's own DynaNav results show the
policy passing the "go straight ahead" hospital episode while driving 34.5 m in the
*wrong* direction on the office episode -- starting 13.74 m from the goal and finishing
40 m away. A robot that drives forward and stops near a goal therefore proves nothing:
a policy that ignored the prompt entirely would produce the same demo.

So before wiring anything to AlohaMini, hold the image fixed and vary only the
instruction. If the waypoints do not change, "intelligent navigation" is not a claim
this stack can make, and we should say so rather than ship a convincing-looking video.

The experiment controls for sampling noise: predict() uses do_sample=True
(temperature 0.1, ticvla.py:585), so a single sample per instruction cannot separate
"responds to language" from "is slightly random". Each instruction is therefore run
--repeats times, and BETWEEN-instruction spread is compared against WITHIN-instruction
spread. Only a between/within ratio comfortably above 1 means the language mattered.

Usage:
    /home/gtu-dsa/envs/tic-vla/bin/python nav/tools/check_policy_sanity.py \
        --frame /path/to/front_frame.jpg --repeats 3
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

# Deliberately contradictory: if the model is listening, these cannot all map to the
# same path. The last one is the strongest test -- "stop" should collapse the path
# length, which is a different axis from heading.
DEFAULT_INSTRUCTIONS = [
    "Go straight ahead down the hallway.",
    "Turn left immediately and go through the door on your left.",
    "Turn right immediately and head toward the right-hand wall.",
    "Stop. Do not move.",
]


def summarize(waypoints: list[list[float]]) -> dict[str, float]:
    """Reduce a (T,2) body-frame FLU path to a few interpretable scalars."""
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    # Net heading of the endpoint: +ve = left (FLU), -ve = right.
    heading = math.degrees(math.atan2(ys[-1], xs[-1]))
    length = sum(
        math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1)
    )
    return {
        "end_x": round(xs[-1], 3),
        "end_y": round(ys[-1], 3),
        "heading_deg": round(heading, 2),
        "max_abs_lateral": round(max(abs(v) for v in ys), 3),
        "path_length": round(length, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="A single RGB frame the policy will see.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--robot-type", default="wheeled robot")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--instructions", nargs="*", default=None)
    args = ap.parse_args()

    frame = Path(args.frame)
    if not frame.is_file():
        print(f"ERROR: no such frame: {frame}", file=sys.stderr)
        return 1

    instructions = args.instructions or DEFAULT_INSTRUCTIONS
    client = PolicyClient(args.host, args.port)
    info = client.wait_until_ready()
    print(f"policy server ready: {info['num_action_chunks']} chunks on {info['device']}\n")

    results: dict[str, list[dict]] = {}
    for instr in instructions:
        rows = []
        for r in range(args.repeats):
            # Reset between samples so a stale KV cache from the previous instruction
            # cannot leak across the comparison -- that would fake a dependence on
            # language, or mask a real one.
            client.reset()
            out = client.predict(
                image_paths=[str(frame)],
                instruction=instr,
                robot_state=[0.0] * 6,
                current_step=r,
                robot_type=args.robot_type,
            )
            s = summarize(out["waypoints"])
            s["latency_s"] = out["latency_s"]
            rows.append(s)
            print(f"  [{instr[:38]:38s}] rep{r}  "
                  f"end=({s['end_x']:+.2f},{s['end_y']:+.2f})  "
                  f"heading={s['heading_deg']:+7.2f}deg  len={s['path_length']:.2f}  "
                  f"({out['latency_s']:.2f}s)")
        results[instr] = rows
        print()

    # --- between-instruction vs within-instruction spread ---------------------
    def spread(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    per_instr_mean_heading = {k: statistics.fmean(r["heading_deg"] for r in v)
                              for k, v in results.items()}
    per_instr_mean_len = {k: statistics.fmean(r["path_length"] for r in v)
                          for k, v in results.items()}
    within = statistics.fmean(
        spread([r["heading_deg"] for r in v]) for v in results.values()
    )
    between = spread(list(per_instr_mean_heading.values()))
    ratio = (between / within) if within > 1e-9 else float("inf")

    print("=" * 74)
    print("mean heading by instruction (FLU: +left / -right)")
    for k, v in per_instr_mean_heading.items():
        print(f"  {v:+8.2f} deg   len {per_instr_mean_len[k]:6.2f}   {k}")
    print()
    print(f"  between-instruction spread : {between:.3f} deg")
    print(f"  within-instruction spread  : {within:.3f} deg  (sampling noise)")
    print(f"  ratio                      : {ratio:.2f}")
    print()

    if ratio > 3.0:
        verdict = "PASS - waypoints depend on the instruction well beyond sampling noise"
    elif ratio > 1.5:
        verdict = "WEAK - some dependence on the instruction, but close to sampling noise"
    else:
        verdict = ("FAIL - the instruction barely changes the output. Any navigation "
                   "demo built on this is showing motion, not language understanding.")
    print(f"VERDICT: {verdict}")
    print("=" * 74)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"frame": str(frame), "repeats": args.repeats, "results": results,
             "between_spread_deg": between, "within_spread_deg": within,
             "ratio": ratio, "verdict": verdict}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
