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

The mirror control is what makes this trustworthy. A model with a mild left bias would
score well on a left-opening frame while seeing nothing. Flipping the frame horizontally
moves the free space to the right and leaves the flat-floor projection exactly as valid --
the camera model is symmetric about its optical axis. A model that reads the image must
NEGATE its chosen curvature when the image flips. That test needs no human labels at all,
which is its own recommendation given that the labels are hand-made.

**The labels live in `nav/config/probe_frames.json`, fingerprinted, and this file refuses
to run if the frames have changed.** They used to be a literal list here, and that cost a
whole benchmark run: the capture under `/tmp/alohamini-nav-frames` was regenerated on
2026-09-05 from a different set of episodes, five days after the labels were written, so
frame 160 stopped being "corner, light-blue wall fills the right" and became a staircase
in a different building. Nothing raised. The probe kept scoring choices against
descriptions of pictures that no longer existed, and the 27B's own DIGIT row fell from
22/100/20/0 to 8/100/0/0 -- which reads exactly like a model regression and was a ground
truth regression. `keep straight` stayed at 100% throughout, because that column only asks
whether straight was picked and a straight-preferring model scores it whatever the frame
shows; so the one column that still looked healthy was the one that could not fail.

A hand label is a claim about a specific image, and `/tmp` is not a place where a specific
image stays put. The manifest carries a SHA of every frame it labelled, and a mismatch is
a hard error rather than a quiet re-scoring against the wrong picture.

IT DRIFTED AGAIN, AND FINGERPRINTS WERE NOT ENOUGH. On 2026-09-12 the check refused: eight
of the twelve frames had changed. Two further captures had been written into the same
directory on 09-10 and it had ended up holding three episodes layered by index -- 0..400,
401..600 and a surviving 601..880 -- so the hospital atrium was now an outdoor strip mall
and a grassland with the camera tumbling through a physics blow-up. The SHA did its job:
it refused instead of scoring. But refusing is not running, and a benchmark that cannot run
is not much better than one that lies.

So the twelve frames now live in `nav/config/probe_frames/`, in git, next to the labels
that describe them, and `frame_dir` is read from the manifest rather than defaulted to
`/tmp` -- which also closes the gap where the SHA check passed against one directory while
`render_menu` drew on another. The fingerprints stay as the second lock. A capture under
`/tmp` is still useful for the things that need no labels (see `--mirror-sweep` in
`probe_value_map.py`); it is just no longer the place the ground truth is kept.

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

MANIFEST = Path(__file__).resolve().parent.parent / "config" / "probe_frames.json"


def load_frames(manifest: Path = MANIFEST
                ) -> tuple[list[tuple[int, int]], list[int], Path]:
    """Read the labelled frames and verify each one is still the picture that was labelled.

    Returns `(BLOCKED, OPEN, frame_dir)` in the shapes the rest of this file and
    `probe_arc_repair` expect. Raises rather than degrading: a wrong label produces a
    plausible number, and a plausible number is the failure this repo keeps paying for.
    Refusing to run is cheap.

    `frame_dir` comes back with the labels instead of being a separate `--frame-dir`
    default, because the two drifting apart is a silent failure rather than a loud one:
    the SHA check would pass against the manifest's copy while `render_menu` drew on
    whatever `/tmp` happened to hold. A relative `frame_dir` resolves against the manifest,
    so the pixels travel with the labels in git.
    """
    import hashlib

    doc = json.loads(manifest.read_text())
    frame_dir = Path(doc["frame_dir"])
    if not frame_dir.is_absolute():
        frame_dir = manifest.parent / frame_dir
    drift: list[str] = []
    for entry in doc["blocked"] + doc["open"]:
        f = frame_dir / f"nav_{entry['frame']:06d}.jpg"
        if not f.is_file():
            drift.append(f"{f} is missing")
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if got != entry["sha16"]:
            drift.append(f"nav_{entry['frame']:06d}.jpg changed "
                         f"({entry['sha16']} -> {got}); it was labelled "
                         f"{entry['desc']!r}")
    if drift:
        raise SystemExit(
            f"\n{manifest} was labelled against a different capture:\n  "
            + "\n  ".join(drift)
            + f"\n\nThe labels are claims about specific images. Re-label against the "
              f"current {frame_dir} and rewrite the manifest -- do NOT run the probe, "
              f"the obstacle columns would score against pictures that no longer exist.\n")
    return ([(e["frame"], e["open_side"]) for e in doc["blocked"]],
            [e["frame"] for e in doc["open"]], frame_dir)


BLOCKED, OPEN, FRAME_DIR = load_frames()


def flip(src: str, dst: str) -> str:
    from PIL import Image
    Image.open(src).transpose(Image.FLIP_LEFT_RIGHT).save(dst, quality=95)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default=str(FRAME_DIR),
                    help="Defaults to the directory the manifest labelled, so the "
                         "SHA check and the pixels can never come from two places.")
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
