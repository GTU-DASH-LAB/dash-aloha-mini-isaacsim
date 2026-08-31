"""Blind, or just not looking? Splitting the obstacle failure into its two possible causes.

`probe_arc_obstacles.py` returned an unambiguous FAIL: told only "Drive safely", the model
picked the straight arc into a wall on 89% of blocked frames, and its choice did not change
when the image was mirrored (0/18). But that measures the CHOICE, and a choice has two
inputs. Either

  (a) the model cannot tell a wall 2 m ahead from open floor -- a perception failure, which
      no prompt fixes and which costs depth plus a reactive layer underneath to repair, or
  (b) it sees the wall perfectly well and still answers with the straight arc -- a decision
      failure, where the percept exists but never reaches the output, and which a two-step
      chain (describe, then choose) or a differently-posed question could fix.

The same visible symptom, two completely different price tags. So ask about free space
directly, with no arcs drawn at all -- raw frames, nothing but the question.

The scored instrument is a two-alternative forced choice, not a yes/no. A first pass here
asked "can the robot drive straight forward 3 m?" one frame at a time and got 100% recall
with 50% specificity off 18 NOs in 24 answers -- which is most of what an always-say-NO
model scores, and therefore mostly unreadable. Showing a blocked frame and an open frame
together and asking which one is clear removes that escape: a constant answer scores 50%
by construction, and answering by position rather than content shows up directly in the
share of "1" replies, which is reported.

Every blocked frame is paired with every open frame, and every pair is asked in both
orders, so each frame's contribution is balanced against the other set rather than against
a threshold I chose.

The free-text question is diagnostic only, never scored. Its first version ("describe what
is between the robot and 3 metres ahead") is kept in mind as a lesson: it put the word
"floor" in the model's mouth and got twelve descriptions of tile patterns and not one
mention of the wall the robot was pointed at. It measured my phrasing. This version names
the categories it wants and offers "clear" as an explicit out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arc_menu import CAM_FOV_DEG, CAM_HEIGHT_M  # noqa: E402
from probe_arc_obstacles import BLOCKED, OPEN  # noqa: E402

# Deliberately NOT the arc-menu system prompt, which is why `probe_arc_selection.ask` is
# not reused: that prompt talks about numbered paths drawn on the floor, and borrowing it
# for a frame with no arcs on it would test how the model handles a contradictory framing
# rather than what it can see. This states the camera and nothing else, so a wrong answer
# here is about the picture.
SYSTEM = (f"You are looking through the forward camera of a small wheeled robot. The "
          f"camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a "
          f"{CAM_FOV_DEG:.0f} degree horizontal field of view. Answer briefly and "
          f"literally, describing only what is visible in this image.")

DESCRIBE = ("Is there a wall, a door, a doorframe or a large object standing in the "
            "robot's way, straight ahead of it and within about 3 metres? If there is, "
            "name it and estimate how far away it is. If the way straight ahead is open "
            "for 3 metres, answer 'clear'.")

CHOOSE = ("These two pictures were taken by the same robot in two different places. In "
          "exactly one of them, the robot can drive straight forward 3 metres without "
          "hitting a wall, a door or an object; in the other, something is in the way. "
          "Which picture shows the open way? Answer with one digit, 1 or 2.")


def ask(base: str, images: list[str], question: str, max_new: int = 60) -> tuple[str, float]:
    body = {"image_paths": images, "system": SYSTEM, "user": question,
            "max_new_tokens": max_new}
    req = urllib.request.Request(base + "/raw", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["text"], out["latency_s"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    fr = lambda f: f"{args.frame_dir}/nav_{f:06d}.jpg"  # noqa: E731
    blocked = [f for f, _s in BLOCKED]

    print("free-text, diagnostic only (b = blocked ahead, o = open corridor)")
    descs: list[dict] = []
    for fid, kind in [(f, "b") for f in blocked] + [(f, "o") for f in OPEN]:
        text, _ = ask(base, [fr(fid)], DESCRIBE)
        text = text.strip().replace("\n", " ")
        descs.append({"frame": fid, "kind": kind, "text": text[:200]})
        print(f"  f{fid:<4d} [{kind}] {text[:104]}", flush=True)

    print(f"\nforced choice: {len(blocked)}x{len(OPEN)} pairs, both orders = "
          f"{len(blocked) * len(OPEN) * 2} calls")
    trials: list[dict] = []
    for b, o in itertools.product(blocked, OPEN):
        for open_pos in (1, 2):
            imgs = [fr(o), fr(b)] if open_pos == 1 else [fr(b), fr(o)]
            reply, lat = ask(base, imgs, CHOOSE, max_new=4)
            said = 1 if "1" in reply else (2 if "2" in reply else None)
            trials.append({"blocked": b, "open": o, "open_pos": open_pos,
                           "said": said, "correct": said == open_pos, "latency_s": lat})

    got = [t for t in trials if t["said"] is not None]
    correct = sum(t["correct"] for t in got)
    acc = correct / len(got)
    chose1 = sum(t["said"] == 1 for t in got) / len(got)
    # Accuracy split by where the open frame sat: a model reading the pictures scores the
    # same either way, one answering by position scores ~100% on one half and ~0% on the
    # other, and the average of those two is a perfectly ordinary-looking 50%.
    a1 = [t["correct"] for t in got if t["open_pos"] == 1]
    a2 = [t["correct"] for t in got if t["open_pos"] == 2]

    print("\n" + "=" * 78)
    print(f"  forced-choice accuracy      : {correct}/{len(got)} = {acc * 100:5.0f}%"
          f"   (chance 50%)")
    print(f"    when the open frame was 1 : {statistics.fmean(a1) * 100:5.0f}%  (n={len(a1)})")
    print(f"    when the open frame was 2 : {statistics.fmean(a2) * 100:5.0f}%  (n={len(a2)})")
    print(f"  answered '1'                : {chose1 * 100:5.0f}%   (50% = no position bias)")
    print(f"  unparsed                    : {len(trials) - len(got)}")
    print(f"  latency                     : "
          f"{statistics.median(t['latency_s'] for t in got):.2f}s median")
    named = sum(1 for d in descs if d["kind"] == "b"
                and any(w in d["text"].lower() for w in ("wall", "door", "frame")))
    print(f"  free text named an obstacle on a blocked frame: {named}/{len(blocked)}")

    balanced = min(statistics.fmean(a1), statistics.fmean(a2))
    if acc > 0.75 and balanced > 0.6:
        verdict = ("PERCEPTION IS FINE -- it can tell a blocked way from an open one, and "
                   "still picks the arc that drives into the wall. The percept exists but "
                   "never reaches the choice. Fix the query before reaching for depth: "
                   "chain describe-then-choose, or score the arcs one at a time.")
    elif acc < 0.6 or balanced < 0.4:
        verdict = ("PERCEPTION IS THE BOTTLENECK -- it cannot reliably separate a blocked "
                   "way from an open one in a single monocular frame, so no prompt fixes "
                   "the arc choice. Free space has to come from geometry (depth, or a "
                   "monocular depth model), and the menu must be pre-filtered before the "
                   "model ever sees it.")
    else:
        verdict = (f"MARGINAL -- {acc * 100:.0f}% overall, worst-order "
                   f"{balanced * 100:.0f}%. Real signal, not enough of it to carry a "
                   f"safety behaviour on its own.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"descriptions": descs, "trials": trials}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
