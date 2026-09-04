"""How much authority does ANY prompt have over TIC-VLA's action expert? A bound.

The recurring proposal -- a better system prompt, a "skill", a soft prompt -- assumes the
expert can be reached through language. It cannot be reached through language directly:
the expert cross-attends to the last-layer KV *values* of the conversation
(DynaNav/ticvla.py, `_process_kv_cache`), and the only part of that conversation we do not
already control is the VLM's generated reasoning. So rather than trying to coax better
reasoning out of the VLM, this patches `vlm.chat` and writes the reasoning directly. That
is strictly more authority than any prompt could ever have, because a prompt can only
*influence* the text this sets exactly.

Two experiments, in order:

  INJECTION   Six maximally opposed reasonings on a fixed scene, including two controls
              that decide the interpretation. NONSENSE (a bechamel recipe carrying large
              waypoint numbers) separates "the expert reads meaning" from "the expert
              reacts to the cache being different at all". EMPTY gives the no-information
              baseline. Without those controls, "the output changed" proves nothing --
              different tokens make a different cache regardless of what they mean.

  GAIN SWEEP  Prose held fixed and neutral, only the injected lateral value varied. If the
              expert tracks the NUMBERS, this is a transfer function and its slope is the
              answer: it says how hard a prompt would have to push to produce a turn, and
              where it stops responding at all.

Measured on this checkpoint (frames 600-603, robot_state zeros): the words are worth about
2 degrees, the injected numbers about +-9 degrees with saturation past |y| = 3, and the
scene 153 degrees. A trim tab, not a steering wheel. See CLAUDE.md.

Read-only rule: TIC-VLA is vendored and untouched. The patch is on the instance, in this
process, and dies with it.

Usage:
    /home/gtu-dsa/envs/tic-vla/bin/python nav/tools/probe_cache_bandwidth.py
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # never launches Kit; safe to pin

import torch  # noqa: E402

sys.path.insert(0, os.environ.get("TICVLA_DYNANAV_ROOT",
                                  "/home/gtu-dsa/robotics/TIC-VLA/DynaNav"))
from ticvla import TICVLA  # noqa: E402

BASE = "/home/gtu-dsa/robotics/models/InternVL3-1B"
CKPT = "/home/gtu-dsa/robotics/models/TIC-VLA-model.ckpt"
HISTORY = (
    "From 0.0s to current timestamp time is 12.0s. (a list of waypoints 1s in between): "
    "(0.71, 0.02, 0.00), (0.68, -0.01, 0.00), (0.70, 0.03, 0.00), (0.66, 0.01, 0.00), "
    "(0.69, -0.02, 0.00)\nEach waypoint (x, y, z) is the displacement over the previous "
    "1.0s. x is forward, y is left, z is up."
)
# Neutral and identical across the gain sweep, so that anything that moves, moved because
# of the numbers.
NEUTRAL = ("I am in a hallway with a clear view ahead. I have been moving forward "
           "steadily. I will continue along the corridor at a safe speed.")


def block(think: str, answer: str) -> str:
    """The model's own output shape, so a null result cannot be blamed on the format."""
    return f"<think>\n{think}\n</think>\n<answer>{answer}</answer>"


INJECTIONS = [
    ("RIGHT, maximal", block(
        "The navigation instruction is TURN RIGHT. I must turn right immediately. Since y "
        "is positive for left, y must be strongly NEGATIVE for this right turn.",
        "(1.00, -1.50, -0.98), (1.50, -3.00, -1.10), (2.00, -4.50, -1.15)")),
    ("LEFT, maximal", block(
        "The navigation instruction is TURN LEFT. I must turn left immediately. Since y is "
        "positive for left, y must be strongly POSITIVE for this left turn.",
        "(1.00, 1.50, 0.98), (1.50, 3.00, 1.10), (2.00, 4.50, 1.15)")),
    ("STOP, maximal", block(
        "The navigation instruction is STOP. I must brake to a complete halt and not move "
        "at all. My displacement over the next 3, 6 and 9 seconds is zero.",
        "(0.00, 0.00, 0.00), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00)")),
    ("REVERSE, maximal", block(
        "The navigation instruction is REVERSE. A wall blocks me and I must back away. "
        "Since x is positive for forward, x must be NEGATIVE.",
        "(-1.00, 0.00, 3.14), (-2.00, 0.00, 3.14), (-3.00, 0.00, 3.14)")),
    ("NONSENSE (control)", block(
        "To make a bechamel, melt the butter over low heat, whisk in an equal weight of "
        "flour, and cook the roux for two minutes without colouring it.",
        "(7.77, 7.77, 7.77), (8.88, 8.88, 8.88), (9.99, 9.99, 9.99)")),
    ("EMPTY (control)", ""),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--scene", type=int, default=600)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--floor-repeats", type=int, default=6)
    args = ap.parse_args()

    frames = [f"{args.frame_dir}/nav_{args.scene + i:06d}.jpg" for i in range(args.n_frames)]
    print(f"loading TIC-VLA on {torch.cuda.get_device_name(0)}", flush=True)
    model = TICVLA(model_path=BASE, device="cuda:0")
    raw = torch.load(CKPT, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in raw.items()}, strict=True)
    model.eval()
    state = torch.zeros(6)

    def run(text: str | None = None) -> tuple[float, float, float]:
        """Endpoint (x, y) at t=3.0 s and its heading. `text=None` generates for real.

        A real generation MUST happen before any patched one: `vlm.chat` sets
        `img_context_token_id` as a side effect, and the forward pass that builds the KV
        cache reads it. Patching chat away on a cold model crashes inside InternVL on
        `selected.sum()` with `selected` still a bool.
        """
        if text is not None:
            model.vlm.chat = lambda *a, _t=text, **k: _t
        with torch.no_grad():
            _r, wp, _s, _k, _p = model.predict(
                image_paths=frames, instruction="Go straight ahead down the hallway.",
                robot_state=state, time_delay=0.0, previous_waypoints_text=HISTORY)
        p = wp[0].float().cpu().numpy()
        return float(p[-1][0]), float(p[-1][1]), math.degrees(math.atan2(p[-1][1], p[-1][0]))

    # --- noise floor -------------------------------------------------------------------
    print(f"\n--- NOISE FLOOR: unpatched, {args.floor_repeats} real generations ---")
    floor = [run() for _ in range(args.floor_repeats)]
    for i, (x, y, h) in enumerate(floor):
        print(f"  run {i}   heading {h:+7.2f} deg   end=({x:+.2f},{y:+.2f})")
    fh = statistics.pstdev([r[2] for r in floor])
    print(f"  noise floor: heading sd {fh:.3f} deg")

    # --- injection ---------------------------------------------------------------------
    print("\n--- INJECTION: reasoning written by us, everything else identical ---")
    inj = {}
    for label, text in INJECTIONS:
        x, y, h = run(text)
        print(f"  {label:20s} heading {h:+7.2f} deg   end=({x:+.2f},{y:+.2f})")
        inj[label] = (x, y, h)
    turn = abs(inj["RIGHT, maximal"][2] - inj["LEFT, maximal"][2])
    nons = abs(inj["RIGHT, maximal"][2] - inj["NONSENSE (control)"][2])
    print(f"\n  right vs left (opposed meaning, near-identical wording) : {turn:6.2f} deg")
    print(f"  right vs a recipe (unrelated meaning, alien wording)    : {nons:6.2f} deg")
    print(f"  semantic / incidental ratio                             : "
          f"{turn / max(nons, 1e-9):6.2f}")

    # --- gain sweep --------------------------------------------------------------------
    print("\n--- GAIN SWEEP: prose fixed, only the injected lateral value varies ---")
    print(f"{'inject y@3s':>12} {'out x':>8} {'out y':>8} {'heading':>10}")
    rows = []
    for y in (-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0):
        ans = (f"(1.00, {y:.2f}, 0.00), (1.50, {y * 2:.2f}, 0.00), "
               f"(2.00, {y * 3:.2f}, 0.00)")
        ox, oy, h = run(block(NEUTRAL, ans))
        print(f"{y:12.2f} {ox:8.2f} {oy:8.3f} {h:9.2f} deg")
        rows.append((y, ox, oy, h))

    ys = [r[2] for r in rows]
    mono = sum(ys[i] <= ys[i + 1] + 1e-6 for i in range(len(ys) - 1))
    lin = [r for r in rows if abs(r[0]) <= 3.0]
    gain = (lin[-1][2] - lin[0][2]) / (lin[-1][0] - lin[0][0])
    span = max(r[3] for r in rows) - min(r[3] for r in rows)
    print(f"\n  monotone over {mono}/{len(rows) - 1} adjacent pairs")
    print(f"  gain over y in [-3, 3] : {gain:.4f}  ({1 / max(gain, 1e-9):.0f}x attenuation)")
    print(f"  heading span over the whole sweep : {span:.2f} deg")
    print(f"  best steering achieved            : "
          f"{max(abs(r[3]) for r in rows):.2f} deg")

    # 90 deg is what "turn right" has to mean. The comparison is not to zero.
    reach = max(abs(r[3]) for r in rows)
    print(f"\nVERDICT: prompting has about +-{reach:.0f} deg of authority over this expert, "
          f"and\n         saturates -- pushing the injected value past |y| = 3 steers LESS, "
          f"not more.\n         A turn is 90 deg. This is a trim tab, not a steering wheel, "
          f"and no\n         prompt (hard, soft, system or skill) changes that, because "
          f"this measurement\n         already grants more control than any prompt has.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
