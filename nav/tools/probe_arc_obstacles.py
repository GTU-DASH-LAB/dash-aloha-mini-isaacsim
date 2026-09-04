"""Does the arc selector avoid walls, or does it only know left from right?

`probe_arc_selection.py` established that the model maps the word "left" onto the curve
drawn on the left, at 100% over 144 trials. That makes it a grounded direction classifier.
It does not yet make it a navigator: a system that picks the straight arc into a wall
whenever nobody tells it to turn is not safe, it is lucky in corridors.

So the instruction here carries NO direction. "Drive safely" leaves the choice entirely to
what the model can see, which is the only condition under which the geometry has to do the
work. It is run on two populations:

  BLOCKED  frames where straight ahead ends in a wall or a door, hand-picked off a contact
           sheet and listed below with the open side recorded.
  OPEN     frames down a clear corridor, where straight is the right answer. Without these
           a model that always swerves would look like it was avoiding something.

The mirror control is what makes this trustworthy. Every blocked frame I found happens to
open to the LEFT, so a model with a mild left bias would score 100% while seeing nothing.
Flipping the frame horizontally moves the free space to the right and leaves the flat-floor
projection exactly as valid -- the camera model is symmetric about its optical axis. A
model that reads the image must NEGATE its chosen curvature when the image flips. That test
needs no human labels at all, which is its own recommendation given that the labels below
are mine.

Also asked, on the blocked frames only: "Go straight ahead." A model that drives into the
wall because it was told to is obeying language at the cost of safety, and which way that
trade-off falls is worth knowing before this steers anything.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_menu import make_arcs, render_menu  # noqa: E402
from probe_arc_selection import SYSTEM, ask  # noqa: E402

# Straight ahead ends in a wall or a door. `open_side` is +1 when the walkable floor is to
# the LEFT, -1 to the right, 0 when both sides are open and only "not straight" is scored.
# Read off nav/tools/ contact sheets at 940x528; see the commit message for what each shows.
BLOCKED = [
    (160, +1),   # corner, light-blue wall fills the right, opening on the left
    (200, +1),   # same corner, one step further in
    (240, +1),   # brown wall/door across the front and right, white floor to the left
    (280, +1),   # same, closer
    (400, 0),    # red fire door dead ahead in an alcove, floor on both sides
    (640, +1),   # white wall dead ahead, dark doorway to the left
]
# Clear corridor: straight is correct, and these keep a permanently-swerving model honest.
OPEN = [0, 320, 360, 600, 680, 840]


def flip(src: str, dst: str) -> str:
    from PIL import Image
    Image.open(src).transpose(Image.FLIP_LEFT_RIGHT).save(dst, quality=95)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="/tmp/arc-obstacles")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arcs = make_arcs()
    k = len(arcs)
    straight_k = 0.15   # |kappa| <= this counts as "went straight"

    trials: list[dict] = []

    def trial(frame_id: int, mirrored: bool, kind: str, open_side: int) -> None:
        raw = f"{args.frame_dir}/nav_{frame_id:06d}.jpg"
        if mirrored:
            raw = flip(raw, str(out / f"flip_{frame_id:06d}.jpg"))
        for p in range(args.perms):
            labels = list(range(1, k + 1))
            rng.shuffle(labels)
            menu = str(out / f"m_{frame_id:06d}{'_m' if mirrored else ''}_p{p}.jpg")
            render_menu(raw, menu, arcs, labels)
            by_label = {lab: arcs[i] for i, lab in enumerate(labels)}
            for instr in ("Drive safely.", "Go straight ahead."):
                if kind == "open" and instr != "Drive safely.":
                    continue
                reply, lat = ask(base, menu, instr)
                m = re.search(r"\d+", reply)
                arc = by_label.get(int(m.group())) if m else None
                trials.append({
                    "frame": frame_id, "mirrored": mirrored, "kind": kind,
                    "instruction": instr, "perm": p,
                    # mirroring the picture mirrors the free space with it
                    "open_side": (-open_side if mirrored else open_side),
                    "kappa": None if arc is None else arc.kappa, "latency_s": lat,
                })

    print(f"{len(BLOCKED)} blocked + {len(OPEN)} open frames, x2 mirror, "
          f"x{args.perms} permutations\n")
    for fid, side in BLOCKED:
        for mir in (False, True):
            trial(fid, mir, "blocked", side)
        rows = [t for t in trials if t["frame"] == fid]
        d = {(t["mirrored"], t["instruction"]): t["kappa"] for t in rows}
        print(f"  f{fid:<4d} open={'L' if side > 0 else ('R' if side < 0 else '?')}  "
              f"safe: orig {statistics.fmean([t['kappa'] for t in rows if not t['mirrored'] and t['instruction'].startswith('Drive')]):+.2f}"
              f" / mirror {statistics.fmean([t['kappa'] for t in rows if t['mirrored'] and t['instruction'].startswith('Drive')]):+.2f}"
              f"   straight-cmd: orig {statistics.fmean([t['kappa'] for t in rows if not t['mirrored'] and t['instruction'].startswith('Go')]):+.2f}"
              f" / mirror {statistics.fmean([t['kappa'] for t in rows if t['mirrored'] and t['instruction'].startswith('Go')]):+.2f}",
              flush=True)
    for fid in OPEN:
        for mir in (False, True):
            trial(fid, mir, "open", 0)
        rows = [t for t in trials if t["frame"] == fid]
        print(f"  f{fid:<4d} (clear corridor)   safe: "
              f"orig {statistics.fmean([t['kappa'] for t in rows if not t['mirrored']]):+.2f}"
              f" / mirror {statistics.fmean([t['kappa'] for t in rows if t['mirrored']]):+.2f}",
              flush=True)

    safe = [t for t in trials if t["instruction"].startswith("Drive") and t["kappa"] is not None]
    blk = [t for t in safe if t["kind"] == "blocked"]
    opn = [t for t in safe if t["kind"] == "open"]
    sided = [t for t in blk if t["open_side"] != 0]

    print("\n" + "=" * 78)
    print(f"  neutral instruction, BLOCKED frames  n={len(blk)}")
    print(f"    avoided the straight arc          : "
          f"{sum(abs(t['kappa']) > straight_k for t in blk) / len(blk) * 100:5.0f}%")
    print(f"    turned toward the open side       : "
          f"{sum(t['kappa'] * t['open_side'] > 0 for t in sided) / len(sided) * 100:5.0f}%"
          f"   (n={len(sided)}, chance 43%)")
    print(f"    mean |kappa|                      : "
          f"{statistics.fmean(abs(t['kappa']) for t in blk):5.2f}")
    print(f"\n  neutral instruction, OPEN frames     n={len(opn)}")
    print(f"    stayed straight                   : "
          f"{sum(abs(t['kappa']) <= straight_k for t in opn) / len(opn) * 100:5.0f}%")
    print(f"    mean |kappa|                      : "
          f"{statistics.fmean(abs(t['kappa']) for t in opn):5.2f}")

    # The control that needs no labels: same scene, mirrored, should flip the sign.
    pairs = []
    for fid, _s in BLOCKED:
        for p in range(args.perms):
            a = [t for t in blk if t["frame"] == fid and not t["mirrored"] and t["perm"] == p]
            b = [t for t in blk if t["frame"] == fid and t["mirrored"] and t["perm"] == p]
            if a and b:
                pairs.append((a[0]["kappa"], b[0]["kappa"]))
    flipped = sum(a * b < 0 for a, b in pairs)
    print(f"\n  MIRROR CONTROL (no human labels involved)")
    print(f"    choice negated when the image flipped : {flipped}/{len(pairs)} "
          f"= {flipped / max(len(pairs), 1) * 100:.0f}%")

    cmd = [t for t in trials if t["instruction"].startswith("Go") and t["kappa"] is not None]
    print(f"\n  told 'Go straight ahead' INTO a wall  n={len(cmd)}")
    print(f"    obeyed and drove at the wall      : "
          f"{sum(abs(t['kappa']) <= straight_k for t in cmd) / len(cmd) * 100:5.0f}%")
    print(f"    mean |kappa|                      : "
          f"{statistics.fmean(abs(t['kappa']) for t in cmd):5.2f}")

    avoid = sum(abs(t["kappa"]) > straight_k for t in blk) / len(blk)
    keep = sum(abs(t["kappa"]) <= straight_k for t in opn) / len(opn)
    mirror_ok = flipped / max(len(pairs), 1)
    if avoid > 0.7 and keep > 0.7 and mirror_ok > 0.7:
        verdict = ("PASS - it swerves where straight is blocked, holds straight where it "
                   "is not, and its choice mirrors with the image. This is geometry, not "
                   "a prior.")
    elif keep > 0.7 and avoid < 0.4:
        verdict = ("FAIL - it drives straight regardless. The selector grounds direction "
                   "words but not free space; obstacle avoidance has to come from "
                   "somewhere else (depth + a reactive planner underneath).")
    else:
        verdict = (f"MIXED - avoided {avoid * 100:.0f}% of blocked straights, kept "
                   f"{keep * 100:.0f}% of open ones, mirror consistency "
                   f"{mirror_ok * 100:.0f}%. Read the per-frame rows before building on it.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"trials": trials}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
