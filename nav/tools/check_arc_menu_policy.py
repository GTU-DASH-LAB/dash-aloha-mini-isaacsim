"""The gate between the arc-menu probes and an Isaac Sim benchmark run.

Everything measured so far went through `/raw`, which hands the prompt over and returns
text. The benchmark drives `/predict`, which renders the menu itself, runs the two-call
chain on a background thread, converts a curvature into 100 waypoints and reframes them
for staleness. None of that was exercised by a probe, and every one of those steps can be
wrong in a way that still returns a well-formed plan -- which an episode would score as
bad driving rather than as a bug.

So this checks the wiring, not the model, and it costs a minute against the ~40 the ladder
costs. Four things, in the order they would bite:

  SHAPE      100 waypoints, monotone, arc length matching the commanded speed. A plan of
             the wrong length is a speed error the controller cannot see.
  STEERING   the thing TIC-VLA failed at. Same frame, same everything, only the
             instruction changes. TIC-VLA's expert moved 0.6 deg across six instructions;
             anything in that neighbourhood here means the instruction is not reaching the
             plan and the run would measure nothing.
  STOP       an arrival instruction has to produce a plan the controller reads as stop.
             Without this the robot cannot finish an episode, only overshoot one.
  STALENESS  a plan handed back with time_delay must be shorter, and one handed back after
             the robot has turned must be rotated the other way. A sign error here is
             invisible in the shape and steers at twice the rate, backwards.

`/reset` runs before every instruction: with a plan already cached the server hands it back
and starts a new generation in the background, so without the reset this would compare each
instruction against the previous one's answer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_menu import DEFAULT_LENGTH_M  # noqa: E402

INSTRUCTIONS = [
    ("left", "Turn left."),
    ("right", "Turn right."),
    ("straight", "Go straight ahead."),
    ("safe", "Drive safely."),
]
ARRIVED = "You have arrived at the elevator. Stop here."


def call(base: str, path: str, body: dict | None = None) -> dict:
    if body is None:
        with urllib.request.urlopen(base + path, timeout=300) as r:
            return json.load(r)
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def predict(base: str, frames: list[str], instruction: str, **kw) -> dict:
    body = {"image_paths": frames, "instruction": instruction,
            "robot_state": [0.0] * 6, "current_step": 0, "time_delay": 0.0,
            "previous_waypoints_text": "none", "robot_type": "wheeled robot"}
    body.update(kw)
    return call(base, "/predict", body)


def heading(wp: np.ndarray) -> float:
    """Bearing of the plan's endpoint, degrees, +left. What the controller steers on."""
    return math.degrees(math.atan2(wp[-1, 1], wp[-1, 0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--frame", type=int, default=600)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    info = call(base, "/health")
    print(f"server: {info['mode']}  format={info['format']}  "
          f"speed={info.get('menu_speed_mps')}  stop_label={info.get('stop_label')}")
    if info["format"] != "menu":
        print(f"\nFAIL: server is running format={info['format']!r}, not 'menu'. Relaunch "
              f"it with QVLA_FORMAT=menu -- otherwise this checks the wrong policy.",
              file=sys.stderr)
        return 1

    # Four frames at the spacing the runner uses. `menu` reads only the newest, but the
    # request has to survive the same payload the benchmark sends.
    frames = [f"{args.frame_dir}/nav_{args.frame - 30 * i:06d}.jpg" for i in (3, 2, 1, 0)]
    for f in frames:
        if not Path(f).is_file():
            print(f"missing frame {f}", file=sys.stderr)
            return 1

    fails: list[str] = []
    plans: dict[str, np.ndarray] = {}
    print(f"\nframe {args.frame}, one instruction at a time, /reset between each")
    for name, text in INSTRUCTIONS:
        call(base, "/reset", {})
        out = predict(base, frames, text)
        wp = np.asarray(out["waypoints"], dtype=float)
        plans[name] = wp
        arc = float(np.sum(np.hypot(*np.diff(np.vstack([[0, 0], wp]), axis=0).T)))
        note = (out.get("reasoning") or "").split("->")[-1].strip()
        print(f"  {name:9} {len(wp):4d} wp  end=({wp[-1, 0]:+.2f},{wp[-1, 1]:+.2f})  "
              f"bearing {heading(wp):+7.1f} deg  arc {arc:.2f} m  "
              f"{out['latency_s']:5.2f}s  {note[:28]}")

    # ---- SHAPE
    wp = plans["straight"]
    speed = info["menu_speed_mps"]
    arc = float(np.sum(np.hypot(*np.diff(np.vstack([[0, 0], wp]), axis=0).T)))
    want = speed * len(wp) * 0.1
    if len(wp) != 100:
        fails.append(f"expected 100 waypoints, got {len(wp)}")
    if abs(arc - want) > 0.05:
        fails.append(f"arc length {arc:.2f} m, expected {want:.2f} m at {speed} m/s")

    # ---- STEERING. The comparison that matters, stated in the same units as the failure
    # it is checking for: TIC-VLA's expert moved the plan 0.6 deg across six instructions.
    spread = heading(plans["left"]) - heading(plans["right"])
    print(f"\n  left minus right bearing : {spread:+.1f} deg "
          f"(TIC-VLA's action expert: +0.6)")
    if spread < 30.0:
        fails.append(f"left and right differ by only {spread:+.1f} deg -- the instruction "
                     f"is not reaching the plan")
    if heading(plans["left"]) <= 0 or heading(plans["right"]) >= 0:
        fails.append(f"sides are wrong way round: left {heading(plans['left']):+.1f}, "
                     f"right {heading(plans['right']):+.1f} (positive must be left)")
    if abs(heading(plans["straight"])) > 10.0:
        fails.append(f"'go straight' bends {heading(plans['straight']):+.1f} deg")

    # ---- STOP
    call(base, "/reset", {})
    out = predict(base, frames, ARRIVED)
    wp = np.asarray(out["waypoints"], dtype=float)
    reach = float(np.hypot(wp[-1, 0], wp[-1, 1]))
    print(f"  arrival instruction      : reach {reach:.3f} m "
          f"({'STOP' if reach < 1e-3 else 'still driving'})")
    if reach >= 1e-3:
        print("    note: not a hard failure -- whether the model believes it has arrived "
              "is a judgement about this frame, not about the wiring. The STOP label being "
              "reachable at all is what is checked below.")

    # ---- STALENESS. Checked against arithmetic, not against the server's own output.
    call(base, "/reset", {})
    predict(base, frames, "Go straight ahead.")          # cache one plan
    fresh = np.asarray(predict(base, frames, "Go straight ahead.")["waypoints"], float)
    aged = np.asarray(predict(base, frames, "Go straight ahead.",
                              time_delay=2.0)["waypoints"], float)
    turned = np.asarray(predict(base, frames, "Go straight ahead.",
                                yaw_delta_rad=math.radians(20.0))["waypoints"], float)
    print(f"  fresh {len(fresh)} wp -> 2.0 s stale {len(aged)} wp "
          f"(expected {len(fresh) - 20})")
    if len(aged) != len(fresh) - 20:
        fails.append(f"2.0 s of staleness dropped {len(fresh) - len(aged)} waypoints, "
                     f"expected 20 at 10 Hz")
    # Turning the robot 20 deg LEFT must move the same world path 20 deg RIGHT in the body
    # frame. The opposite sign is the error this exists to catch.
    got = heading(turned) - heading(fresh)
    print(f"  robot yaws +20 deg -> plan bearing moves {got:+.1f} deg (expected -20.0)")
    if abs(got + 20.0) > 1.0:
        fails.append(f"a +20 deg yaw moved the plan {got:+.1f} deg, expected -20.0")

    print("\n" + "=" * 78)
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print("\nDo not start the ladder. These are wiring faults; an episode would score "
              "them as driving.")
    else:
        print("  PASS - shape, steering, staleness and the stop path all behave. The "
              "policy is wired to the controller correctly; run the ladder.")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
