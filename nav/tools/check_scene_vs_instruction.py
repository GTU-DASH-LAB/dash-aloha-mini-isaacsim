"""The control that turns "the policy ignores my instruction" into a measurement.

`check_policy_sanity.py` varies the instruction and asks whether the plan moves. That is
half a test. A policy whose output never moved for any reason would fail it too, and that
is a completely different diagnosis -- broken policy, not deaf-to-language policy. So vary
the two axes separately against the same scalar and compare:

    SCENE axis        instruction pinned, N different sets of frames
    INSTRUCTION axis  frames pinned, N instructions including contradictory ones

Scene-spread >> instruction-spread means the KV cache is carrying perception and dropping
language. That is a claim worth making. Either spread alone is not.

It also answers the obvious objection to probing a SCAND/GND-trained policy with our own
Isaac Sim renders -- "your frames are out of distribution, of course it ignores you". If
the scene axis produces large, structured variation on those same frames, the model is
plainly reading them, and the language channel is failing for its own reasons.

Pick the pinned scene from the SCENE axis output, not arbitrarily. Some frames put the
policy in a degenerate regime (frames 150-153 of our nav cache plan +90 deg over 0.84 m --
a sideways shuffle), and an instruction sweep run there measures the degeneracy, not the
instruction. Pick a scene whose pinned-instruction plan is forward and full speed.

Usage:
    /home/gtu-dsa/envs/tic-vla/bin/python nav/tools/check_scene_vs_instruction.py \
        --frame-dir /tmp/alohamini-nav-frames --pinned-scene 600 --port 8765
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

# The history text is held fixed with the scene: it is part of the observation, and
# letting it vary would put a second uncontrolled variable on the instruction axis.
HISTORY = (
    "From 0.0s to current timestamp time is 12.0s. (a list of waypoints 1s in between): "
    "(0.71, 0.02, 0.00), (0.68, -0.01, 0.00), (0.70, 0.03, 0.00), (0.66, 0.01, 0.00), "
    "(0.69, -0.02, 0.00)\nEach waypoint (x, y, z) is the displacement over the previous "
    "1.0s. x is forward, y is left, z is up."
)

# Contradictory on purpose, and spanning two different axes of control: heading (left /
# right / reverse) and speed (stop). A policy can be deaf on one and not the other.
INSTRUCTIONS = [
    "Go straight ahead down the hallway.",
    "Stop immediately. Do not move at all.",
    "Turn right.",
    "Turn left.",
    "Turn hard right, 90 degrees, now.",
    "Go backward. Reverse.",
    "Follow the person in front of you.",
    "Enter the door on your left.",
]


def post(base: str, route: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(
        base + route, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def plan(base: str, frames: list[str], instruction: str) -> tuple[float, float, bool]:
    """Heading (deg, FLU +left) and path length (m) of one freshly generated plan.

    `/reset` first, always. `/predict` re-serves a cached plan when one exists, so a sweep
    without it measures one generation N times and calls it agreement.
    """
    post(base, "/reset")
    out = post(base, "/predict", {
        "image_paths": frames,
        "instruction": instruction,
        "robot_state": [0.0] * 6,
        "previous_waypoints_text": HISTORY,
    })
    w = out["waypoints"]
    heading = math.degrees(math.atan2(w[-1][1], w[-1][0]))
    length = sum(
        math.hypot(w[i + 1][0] - w[i][0], w[i + 1][1] - w[i][1]) for i in range(len(w) - 1)
    )
    # TIC-VLA emits the CE ignore_index as text when its guidance waypoint was padding;
    # see CLAUDE.md. Worth surfacing here because it silently poisons any <answer> parser.
    return heading, length, "-100.00" in (out.get("reasoning") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--prefix", default="nav_")
    ap.add_argument("--n-frames", type=int, default=4, help="Frames per observation.")
    ap.add_argument("--pinned-scene", type=int, default=600,
                    help="Start index of the scene held fixed on the instruction axis.")
    ap.add_argument("--scenes", type=int, nargs="*", default=[0, 100, 200, 300, 400, 500, 600, 700])
    ap.add_argument("--pinned-instruction", default=INSTRUCTIONS[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    d = Path(args.frame_dir)

    def frames(start: int) -> list[str]:
        return [str(d / f"{args.prefix}{start + i:06d}.jpg") for i in range(args.n_frames)]

    missing = [p for s in args.scenes + [args.pinned_scene] for p in frames(s)
               if not Path(p).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} frames missing, first: {missing[0]}", file=sys.stderr)
        return 1

    with urllib.request.urlopen(base + "/health", timeout=30) as r:  # GET, not POST
        info = json.load(r)
    print(f"policy server ready: {info.get('num_action_chunks')} chunks on "
          f"{info.get('device')}\n")

    sentinels = 0
    print("--- SCENE axis (instruction pinned) ---")
    sh, sl, scene_rows = [], [], []
    for s in args.scenes:
        h, l, sent = plan(base, frames(s), args.pinned_instruction)
        sentinels += sent
        print(f"  frames {s:5d}   heading {h:+8.2f} deg   len {l:5.2f} m")
        sh.append(h); sl.append(l); scene_rows.append({"scene": s, "heading": h, "len": l})

    print(f"\n--- INSTRUCTION axis (scene pinned at {args.pinned_scene}) ---")
    ih, il, instr_rows = [], [], []
    pinned = frames(args.pinned_scene)
    for ins in INSTRUCTIONS:
        h, l, sent = plan(base, pinned, ins)
        sentinels += sent
        print(f"  {ins[:38]:38s} heading {h:+8.2f} deg   len {l:5.2f} m")
        ih.append(h); il.append(l); instr_rows.append({"instruction": ins, "heading": h, "len": l})

    s_h, i_h = statistics.pstdev(sh), statistics.pstdev(ih)
    s_l, i_l = statistics.pstdev(sl), statistics.pstdev(il)
    n = len(args.scenes) + len(INSTRUCTIONS)

    print("\n" + "=" * 74)
    print(f"{'axis':20}{'heading sd':>12}{'heading range':>16}{'len sd':>10}{'len range':>12}")
    print(f"{'scene':20}{s_h:12.2f}{max(sh) - min(sh):15.1f} {s_l:9.3f}{max(sl) - min(sl):11.2f}")
    print(f"{'instruction':20}{i_h:12.2f}{max(ih) - min(ih):15.1f} {i_l:9.3f}{max(il) - min(il):11.2f}")
    ratio = s_h / max(i_h, 1e-9)
    print(f"\n  scene / instruction heading-spread ratio : {ratio:.2f}")
    print(f"  slowest plan on the instruction axis     : {min(il):.2f} m over 3 s "
          f"= {min(il) / 3:.2f} m/s")
    if sentinels:
        print(f"  <answer> was the -100 padding sentinel   : {sentinels}/{n} generations")

    # A policy that reads the scene and not the instruction is the interesting failure and
    # deserves its own verdict, distinct from one that reads neither.
    if s_h < 5.0:
        verdict = ("INCONCLUSIVE - the scene barely moves the plan either, so this is not "
                   "a language finding. Check the server, the frames, and robot_state.")
    elif i_h >= 15.0:
        verdict = "PASS - the instruction moves the heading by a drivable amount."
    else:
        verdict = (f"DEAF TO LANGUAGE - the scene moves the heading {ratio:.0f}x more than "
                   f"contradictory instructions do ({s_h:.1f} vs {i_h:.1f} deg sd). The "
                   f"observation channel works; the language channel does not reach the "
                   f"numbers.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 74)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "scene_axis": scene_rows, "instruction_axis": instr_rows,
            "scene_heading_sd": s_h, "instruction_heading_sd": i_h,
            "ratio": ratio, "sentinels": sentinels, "verdict": verdict,
        }, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
