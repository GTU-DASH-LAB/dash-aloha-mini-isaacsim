"""Can a frozen Qwen pick the right arc off the floor? The whole architecture rests on it.

Everything measured so far says the reasoning is fine and the emission is deaf. The arc
menu is the fix: generate drivable paths ourselves, draw them on the robot's own camera
image, and ask for an integer instead of a coordinate. That removes the failure we
measured -- there is no number to copy when the answer is a label -- but it substitutes a
new assumption, and this file exists to test it rather than believe it:

    can the model map the WORD "left" to the CURVE that is visually on the left?

VLMs are known to be unreliable at left/right. If this is at chance, the architecture is
dead and we learn it in half an hour instead of after building a controller around it.

Two controls, both necessary:

  LABEL PERMUTATION  Numbering the arcs left-to-right lets a model score perfectly by
                     answering "1" for left and "K" for right without looking at the
                     image. Labels are therefore shuffled per trial, and ground truth
                     follows the shuffle. Without this the experiment cannot fail.

  NEUTRAL BASELINE   A model that always picks the centre arc scores 100% on "go straight"
                     and 0% on the turns, which is a positional prior, not comprehension.
                     Asking with an instruction that names no direction gives the default
                     pick, and the turns are then judged by how far they MOVE it.

Scored on side, not on exact label: "turn left" is satisfied by any left-curving arc. The
exact-match rate is reported alongside, but side is the honest metric -- how hard to turn
is a question about the corridor, and the model is not being asked that here.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/probe_arc_selection.py --scenes 8 --perms 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arc_menu import (  # noqa: E402
    CAM_FOV_DEG, CAM_HEIGHT_M, make_arcs, render_menu,
)

# The system prompt is the part Fouad wanted a "skill" to be, and here it can actually
# act: the model's output IS the decision, with no frozen regressor in between to
# attenuate it. Everything it needs to ground the picture is stated, and nothing about
# which label is which -- that is the thing being measured.
SYSTEM = f"""You are the navigation system of a small wheeled robot, looking through its \
forward camera. The camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a \
{CAM_FOV_DEG:.0f} degree horizontal field of view.

Several candidate paths have been drawn onto the floor of the image. Each one is a real \
route the robot can drive, starting at the robot and curving away from it, and each ends \
in a numbered circle. The numbers are arbitrary tags, not an order: they do not run left \
to right and carry no meaning beyond identifying a path.

Your job is to choose the one path that best carries out the navigation instruction. \
Judge each path by where it actually goes on the floor in the image. Prefer a path that \
stays on open, walkable floor and does not run into a wall, a door frame or an object.

Answer with the number of the chosen path and nothing else."""

INSTRUCTIONS = [
    # label,        text,                                   expected side (None = no claim)
    ("neutral", "Drive safely.", None),
    ("left", "Turn left.", +1),
    ("right", "Turn right.", -1),
    ("straight", "Go straight ahead.", 0),
    ("left-long", "Take the path that curves to your left.", +1),
    ("right-long", "Take the path that curves to your right.", -1),
]


def ask(base: str, image: str, instruction: str, max_new: int = 8) -> tuple[str, float]:
    body = {
        "image_paths": [image],
        "system": SYSTEM,
        "user": f"Navigation instruction: {instruction}\n\nWhich numbered path do you take?",
        "max_new_tokens": max_new,
    }
    req = urllib.request.Request(base + "/raw", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["text"], out["latency_s"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--perms", type=int, default=3, help="Label shufflings per scene.")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/arc-menus")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    rng = random.Random(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    arcs = make_arcs()
    k = len(arcs)

    trials: list[dict] = []
    print(f"{k} arcs, {args.scenes} scenes x {args.perms} label permutations x "
          f"{len(INSTRUCTIONS)} instructions = "
          f"{args.scenes * args.perms * len(INSTRUCTIONS)} calls\n")

    for s in range(args.scenes):
        frame = f"{args.frame_dir}/nav_{s * args.stride:06d}.jpg"
        if not Path(frame).is_file():
            print(f"  skipping missing {frame}")
            continue
        for p in range(args.perms):
            labels = list(range(1, k + 1))
            rng.shuffle(labels)
            menu = f"{args.out_dir}/menu_{s * args.stride:06d}_p{p}.jpg"
            render_menu(frame, menu, arcs, labels)
            # label -> the arc it was pinned to, so a reply is read through the shuffle
            by_label = {lab: arcs[i] for i, lab in enumerate(labels)}
            for name, text, side in INSTRUCTIONS:
                reply, lat = ask(base, menu, text)
                m = re.search(r"\d+", reply)
                choice = int(m.group()) if m else None
                arc = by_label.get(choice)
                trials.append({
                    "scene": s * args.stride, "perm": p, "instruction": name,
                    "labels": labels, "reply": reply.strip()[:40], "choice": choice,
                    "kappa": None if arc is None else arc.kappa,
                    "expect_side": side, "latency_s": lat,
                })
            print(f"  scene {s * args.stride:5d} perm {p}  " + "  ".join(
                f"{t['instruction'][:5]}={t['choice']}({t['kappa']:+.2f})"
                if t["kappa"] is not None else f"{t['instruction'][:5]}=?"
                for t in trials[-len(INSTRUCTIONS):]), flush=True)

    if not trials:
        print("no trials ran", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    neutral = [t["kappa"] for t in trials
               if t["instruction"] == "neutral" and t["kappa"] is not None]
    base_k = statistics.fmean(neutral) if neutral else 0.0
    print(f"{'instruction':12}{'n':>4}{'unparsed':>9}{'mean kappa':>12}"
          f"{'vs neutral':>12}{'side ok':>9}{'exact':>8}")
    summary = {}
    for name, _text, side in INSTRUCTIONS:
        rows = [t for t in trials if t["instruction"] == name]
        got = [t for t in rows if t["kappa"] is not None]
        bad = len(rows) - len(got)
        mk = statistics.fmean(t["kappa"] for t in got) if got else float("nan")
        if side is None:
            ok = ex = float("nan")
        elif side == 0:
            ok = sum(abs(t["kappa"]) <= 0.15 for t in got) / max(len(got), 1)
            ex = sum(abs(t["kappa"]) < 1e-9 for t in got) / max(len(got), 1)
        else:
            ok = sum(t["kappa"] * side > 0 for t in got) / max(len(got), 1)
            ex = sum(abs(t["kappa"]) > 0.5 and t["kappa"] * side > 0
                     for t in got) / max(len(got), 1)
        print(f"{name:12}{len(rows):4d}{bad:9d}{mk:12.3f}{mk - base_k:12.3f}"
              f"{ok * 100:8.0f}%{ex * 100:7.0f}%")
        summary[name] = {"n": len(rows), "unparsed": bad, "mean_kappa": mk,
                         "side_ok": ok, "exact": ex}

    # Chance is not 50%: 3 of 7 arcs curve left, 3 curve right, 1 is straight.
    chance = 3.0 / len(arcs)
    left = summary["left"]["mean_kappa"]
    right = summary["right"]["mean_kappa"]
    print(f"\n  chance for a side-correct pick : {chance * 100:.0f}%")
    print(f"  left minus right, mean kappa   : {left - right:+.3f} "
          f"(1/m; the full menu spans {max(a.kappa for a in arcs) - min(a.kappa for a in arcs):.2f})")
    lat = [t["latency_s"] for t in trials]
    print(f"  latency                        : {statistics.median(lat):.2f}s median, "
          f"{max(lat):.2f}s max")

    sep = left - right
    if summary["left"]["side_ok"] > 0.8 and summary["right"]["side_ok"] > 0.8:
        verdict = ("PASS - the model grounds left and right in the image. The selector "
                   "architecture stands; build the controller on it.")
    elif sep > 0.2:
        verdict = (f"PARTIAL - left and right separate by {sep:+.2f} 1/m but the sides are "
                   f"not clean. Worth prompt work, which is legitimate here because the "
                   f"model's output IS the decision.")
    else:
        verdict = ("FAIL - left and right do not separate. Selecting from a drawn menu "
                   "does not rescue this model's spatial grounding, and the arc "
                   "architecture needs a different question or a different model.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"trials": trials, "summary": summary, "chance": chance}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
