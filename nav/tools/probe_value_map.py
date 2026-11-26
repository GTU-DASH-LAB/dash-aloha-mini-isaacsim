"""VLFM's value function, ported off Habitat and measured on our frames.

Fouad asked for four zero-shot ObjectNav systems -- WMNav 58.1 SR, OpenFMNav 54.9, VLFM
52.5, L3MVN 50.4. None of the four runs against this stack as published, and the reason is
the same three times over, so it is worth writing down before the one thing that *is*
portable:

  HABITAT-BOUND  All four are Habitat agents. Their action space is discrete (move_forward
                 0.25 m, turn +-30 deg, stop), their episodes come from HM3D/MP3D scene
                 datasets, and their loop is `habitat.Env.step()`. Ours is a continuous
                 (v, omega) base in Isaac Sim driving a curvature menu. There is no adapter
                 that is not a rewrite of the agent.
  DEPTH-BOUND    All four build a top-down map by projecting a depth image. `camera_nav`
                 publishes RGB only. VLFM additionally ends in a *trained* PointNav
                 (VER/DDPPO) policy whose observation space is depth + goal vector -- it is
                 a checkpoint, not a prompt, and it cannot be fed something else.
  A DIFFERENT TASK  ObjectNav is "find a chair": success is stopping within 1 m of ANY
                 instance of a category. Ours is instruction-following to a specific
                 coordinate with a per-episode `success_threshold_m`. A system that wins by
                 finding any chair is not measurable on an episode that names one place.

The SR numbers invite a fourth mistake. Our baseline is 10/19 = 52.6%, which lands between
VLFM's 52.5 and OpenFMNav's 54.9. That is a coincidence between two different tasks and
means nothing; nothing in this file should be read as comparing to those numbers.

WHAT IS PORTABLE, AND IT IS ONE THING.

VLFM's semantic value function is RGB-only and separable from everything else in it. It is
a BLIP-2 ITM cosine similarity between the current RGB frame and a prompt naming the
target ("Seems like there is a target_object ahead."), painted into a top-down value map
through a cone mask over the camera's FOV; frontiers are then ranked by that value. Depth
enters only to build the occupancy map the frontiers come from and to trim the occluded
part of the cone. Strip those two and what is left is a pure function

    (RGB image, text) -> float

which needs no Habitat, no depth, no map and no PointNav head. That function is what this
file measures, on our frames, against the numbers already on disk.

WHY THE EXISTING PROBES CANNOT BE REUSED UNMODIFIED.

`probe_arc_selection.py` and `probe_arc_obstacles.py` shuffle the arc labels per trial and
tell the model the numbers "are arbitrary tags ... carry no meaning beyond identifying a
path". The label-to-arc mapping therefore lives only in the pixels, which is the control
that makes those probes falsifiable -- and it also means any scorer that does not read a
drawn digit out of the image cannot answer them at all. VLFM's value function is a
BEARING SCORER, not a menu picker: it returns a number for a view, and has no mechanism
for emitting "4". So it is probed on bearings, and then scored with the SAME rules, the
SAME frames and the SAME labels as the arc probes, so the two are read off one ruler.

THREE MEASUREMENTS, IN INCREASING DISTANCE FROM VLFM AS PUBLISHED.

  FRAME     VLFM unmodified: one value for one whole frame. Every BLOCKED frame is paired
            with every OPEN frame and we ask whether the open one scores higher. This is
            the same question `probe_free_space.py` put to the 27B as a two-alternative
            forced choice, where it scored 100% (70/70) -- so there is a number to beat,
            measured on these exact frames.
  SECTOR    The adaptation, and the only place this file invents anything. VLFM gets
            directional resolution by TURNING: it scores a frame, steps the yaw 30 deg,
            scores again. With one frame per moment we crop instead -- a window
            ~30 deg wide (one Habitat turn step) centred on each of the seven arcs' badge
            columns, which is the pixel each arc's number was actually drawn at. The
            argmax window names an arc, and that arc's curvature is scored with
            `probe_arc_obstacles.py`'s metrics verbatim.
  MIRROR    The control that needs no human labels, and the one the 27B failed 0/18. Flip
            the frame; the free space moves to the other side and the flat-floor
            projection stays exactly as valid, so a scorer that reads the image must
            negate its chosen curvature. The badge columns are symmetric about the optical
            axis (87.7 <-> 1832.3, 404.9 <-> 1515.1, 740.7 <-> 1179.3, 960 fixed), so
            cropping the flipped frame at the same columns pairs arc i with arc k-1-i for
            free. A scorer with a positional prior scores 0% here by construction, not 43%.
            It is also cheap enough to run on frames nobody labelled -- `--mirror-sweep`
            walks the whole capture -- so this one number is the only well-powered thing
            in the file.

ON n. The value function is deterministic and label-free, so the permutation axis that
gave the arc probes n=36 on twelve frames does not exist here: one frame is one decision.
The labelled columns are n=12 and are quoted as such. Read `--mirror-sweep` for power.

PROMPTS. VLFM's prompt names the target object, and our task has no object category, so
the noun is replaced by what the labelled frames actually record -- free space:

    positive  "Seems like there is a clear path ahead."
    negative  "Seems like there is a wall blocking the way ahead."

Both are scored, and so is their difference. A contrast between a matched pair cancels the
per-image offset that ITM similarity carries, which is worth having because the absolute
scale of a cosine is not comparable across images and the argmax over sectors is. Both
BLIP-2 heads are reported: `itc` is the cosine similarity VLFM uses, `itm` is the fused
match head. Four readouts, all from the same forward passes, none of them chosen after
seeing the answer -- which is why they are all printed rather than the best one quoted.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/probe_value_map.py --mirror-sweep 88
"""

from __future__ import annotations

import sys
from pathlib import Path

# THIS BLOCK MUST RUN BEFORE torch IS IMPORTED, AND IT IS NOT SUPERSTITION.
# A script's own directory is sys.path[0], this script lives in `nav/tools/`, and
# `nav/tools/profile.py` is the config tool -- so `profile` resolves to it instead of the
# stdlib module of that name. Nothing here imports `profile`; `cProfile` does, at
# `run.__doc__ = _pyprofile.run.__doc__`, and `torch._dynamo.convert_frame` imports
# cProfile, and `transformers.masking_utils` imports torch._dynamo. The failure surfaces
# nine frames deep as `ModuleNotFoundError: Could not import module
# 'Blip2ForImageTextRetrieval'`, which points at the model and not at the shadow.
# So: drop the tools directory, force the real cProfile into sys.modules, put it back.
# Renaming `nav/tools/profile.py` would fix this everywhere at once, but it is a command
# documented in `baseline.yaml` and `robot/README.md` and is not this file's to rename.
_TOOLS = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _TOOLS:
    sys.path.pop(0)
import cProfile  # noqa: E402, F401  -- imported for its side effect on sys.modules

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, _TOOLS)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arc_menu import (  # noqa: E402
    CAM_FOV_DEG, CAM_H, CAM_W, badge_xy, make_arcs, project,
)
from probe_arc_obstacles import BLOCKED, FRAME_DIR, OPEN  # noqa: E402  (checks the SHAs)

MODEL_DIR = "/home/gtu-dsa/robotics/models/blip2-itm-vit-g"

# VLFM's own phrasing, with the object category swapped for free space. The sentence frame
# ("Seems like there is a ... ahead.") is kept exactly, because BLIP-2 ITM is sensitive to
# it and changing the frame as well as the noun would make this a different prompt rather
# than the same prompt about a different thing.
PROMPT_POS = "Seems like there is a clear path ahead."
PROMPT_NEG = "Seems like there is a wall blocking the way ahead."

STRAIGHT_K = 0.15          # |kappa| <= this counts as "went straight", as in the arc probes
BADGE_R = 96 * 0.72        # render_menu's default font_size * 0.72, so the columns match


def badge_columns(arcs) -> list[float]:
    """The image column each arc's number badge is drawn at, in the unflipped frame.

    Taken from `arc_menu.badge_xy` rather than recomputed, so a sector is centred on the
    pixel the VLM was looking at when it answered with that arc's label. The outer arcs
    leave the 90 degree view before 3 m and `badge_xy` already handles that by taking the
    last point still inside the frame; reimplementing the projection here would silently
    drift from the menu the comparison is against.
    """
    cols = []
    for arc in arcs:
        px = [p for p in (project(x, y) for x, y in arc.points) if p is not None]
        ux, _uy = badge_xy(px, CAM_W, CAM_H, BADGE_R)
        cols.append(ux)
    return cols


class ValueFunction:
    """BLIP-2 ITM as a scalar value on (image, text). VLFM's `BLIP2ITMClient`, locally.

    Holds both heads. `itc` is the contrastive cosine similarity -- image queries against
    the text embedding, max over the 32 query tokens -- and is what VLFM calls with
    `cosine`. `itm` is the fused cross-attention match head, returned as P(match). They
    are different functions and there is no reason to assume they rank the same way, so
    both are carried through to the tables instead of one being picked here.
    """

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cuda:0",
                 dtype: str = "float16") -> None:
        import torch
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor

        self.torch = torch
        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_dir)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            model_dir, dtype=getattr(torch, dtype)).to(device).eval()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def score(self, images: list, text: str, head: str) -> list[float]:
        """One scalar per image for one prompt. `head` is "itc" or "itm"."""
        torch = self.torch
        inputs = self.processor(images=images, text=[text] * len(images),
                                padding=True, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            out = self.model(**inputs, use_image_text_matching_head=(head == "itm"))
        logits = out.logits_per_image.float()
        if head == "itm":
            # (B, 2) logits over [no-match, match]; the value is P(match).
            return logits.softmax(dim=1)[:, 1].tolist()
        # (B, T) cosine similarities; T == 1 here because the text is repeated per image.
        return logits.diagonal().tolist() if logits.shape[1] == len(images) else \
            logits[:, 0].tolist()


def crops(im, cols: list[float], width_px: int) -> list:
    """One window per arc, `width_px` wide and full height, centred on each badge column.

    Full height on purpose. The label that separates a blocked frame from an open one in
    this capture is usually a wall or a staircase standing ABOVE the floor, so a window
    cropped down to the floor band would remove the evidence and then report that the
    value function cannot see it. Every window is squashed identically by the processor's
    resize, which is what keeps the argmax across windows fair.
    """
    half = width_px / 2.0
    out = []
    for c in cols:
        left = int(round(min(max(c - half, 0), im.width - width_px)))
        out.append(im.crop((left, 0, left + width_px, im.height)))
    return out


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.0f}%" if d else "    -"


MODES = [("itc", "pos"), ("itc", "contrast"), ("itm", "pos"), ("itm", "contrast")]
MODE_LABEL = {("itc", "pos"): "itc / clear",
              ("itc", "contrast"): "itc / clear-wall",
              ("itm", "pos"): "itm / clear",
              ("itm", "contrast"): "itm / clear-wall"}


def sector_values(vf: ValueFunction, windows: list) -> dict[tuple[str, str], list[float]]:
    """Every readout for one frame's windows, from one pair of forward passes per head."""
    vals: dict[tuple[str, str], list[float]] = {}
    for head in ("itc", "itm"):
        pos = vf.score(windows, PROMPT_POS, head)
        neg = vf.score(windows, PROMPT_NEG, head)
        vals[(head, "pos")] = pos
        vals[(head, "contrast")] = [p - n for p, n in zip(pos, neg)]
    return vals


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default=str(FRAME_DIR),
                    help="Defaults to the directory the manifest labelled, so the "
                         "SHA check and the pixels can never come from two places.")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="cuda:0",
                    help="GPU0 by default; the 27B baseline server owns GPU1.")
    ap.add_argument("--sector-w", type=int, default=640,
                    help="Window width in px. 640 of 1920 at 90.1 deg HFOV is 30.4 deg, "
                         "one Habitat turn step -- the unit VLFM would have turned by.")
    ap.add_argument("--mirror-sweep", type=int, default=0,
                    help="Also run the label-free mirror control on every Nth frame of "
                         "--sweep-dir. 0 disables.")
    ap.add_argument("--sweep-dir", default="/tmp/alohamini-nav-frames",
                    help="Where the sweep reads from. NOT the manifest's directory and "
                         "not fingerprinted, on purpose: the mirror control uses no hand "
                         "labels, so it is the one measurement a drifted capture cannot "
                         "corrupt, and it is worth running on every frame there is.")
    ap.add_argument("--sweep-range", default="601:880",
                    help="LO:HI index bounds for the sweep. The default is the indoor "
                         "block the twelve labelled frames were cut from. It matters: "
                         "/tmp currently holds three captures layered by index, and "
                         "frames 190-600 are empty grassland with the camera tumbling "
                         "through a physics blow-up, where 'which way is open' has no "
                         "answer and a mirror score would be noise dressed as a result.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    arcs = make_arcs()
    cols = badge_columns(arcs)
    kappas = [a.kappa for a in arcs]
    deg_per_px = CAM_FOV_DEG / CAM_W

    print(f"BLIP-2 ITM value function, {args.model_dir}")
    t0 = time.time()
    vf = ValueFunction(args.model_dir, args.device)
    print(f"  loaded {vf.n_params / 1e9:.2f} B params on {args.device} "
          f"in {time.time() - t0:.1f} s")
    print(f"  prompt +  {PROMPT_POS!r}")
    print(f"  prompt -  {PROMPT_NEG!r}")
    print(f"  {len(arcs)} sectors, {args.sector_w} px = "
          f"{args.sector_w * deg_per_px:.1f} deg each, centred at "
          + " ".join(f"{c:.0f}" for c in cols))
    print()

    frame_dir = Path(args.frame_dir)

    def load(fid: int, mirrored: bool):
        im = Image.open(frame_dir / f"nav_{fid:06d}.jpg").convert("RGB")
        return im.transpose(Image.FLIP_LEFT_RIGHT) if mirrored else im

    # ----------------------------------------------------------------- FRAME level
    # VLFM unmodified: one value for one whole frame, no cropping, no arcs.
    whole: dict[int, dict[tuple[str, str], float]] = {}
    fids = [f for f, _ in BLOCKED] + list(OPEN)
    ims = [load(f, False) for f in fids]
    full = sector_values(vf, ims)
    for i, f in enumerate(fids):
        whole[f] = {m: full[m][i] for m in MODES}

    print("=" * 78)
    print("  FRAME LEVEL -- VLFM as published: one value per whole frame")
    print("  every blocked frame paired with every open frame; does the open one score "
          "higher?")
    print(f"  {'':<18}" + "".join(f"{MODE_LABEL[m]:>17}" for m in MODES))
    pairs = list(itertools.product([f for f, _ in BLOCKED], OPEN))
    frame_rate: dict[tuple[str, str], float] = {}
    row = []
    for m in MODES:
        won = sum(whole[o][m] > whole[b][m] for b, o in pairs)
        frame_rate[m] = won / len(pairs)
        row.append(f"{won:>6}/{len(pairs):<3} {won / len(pairs) * 100:3.0f}%")
    print(f"  {'open scores higher':<18}" + "".join(f"{r:>17}" for r in row))
    print(f"  {'':<18}" + f"{'chance 50%; the 27B answered this as a forced choice at 100% (70/70)':>68}")
    print()
    for f, side in BLOCKED:
        print(f"    blocked f{f:<4d} open={'L' if side > 0 else 'R'}  "
              + "".join(f"{whole[f][m]:>17.4f}" for m in MODES))
    for f in OPEN:
        print(f"    open    f{f:<4d}          "
              + "".join(f"{whole[f][m]:>17.4f}" for m in MODES))
    print()

    # ---------------------------------------------------------------- SECTOR level
    trials: list[dict] = []
    for fid, side in [(f, s) for f, s in BLOCKED] + [(f, 0) for f in OPEN]:
        kind = "blocked" if any(fid == f for f, _ in BLOCKED) else "open"
        for mirrored in (False, True):
            vals = sector_values(vf, crops(load(fid, mirrored), cols, args.sector_w))
            for m in MODES:
                i = max(range(len(arcs)), key=lambda j: vals[m][j])
                trials.append({
                    "frame": fid, "kind": kind, "mirrored": mirrored,
                    "head": m[0], "readout": m[1],
                    # mirroring the picture mirrors the free space with it
                    "open_side": (-side if mirrored else side),
                    "arc": i, "kappa": kappas[i],
                    "values": [round(v, 5) for v in vals[m]],
                })

    print("=" * 78)
    print(f"  SECTOR LEVEL -- argmax over {len(arcs)} windows, scored with "
          f"probe_arc_obstacles.py's rules")
    print(f"  {'':<34}" + "".join(f"{MODE_LABEL[m]:>17}" for m in MODES))

    def sector_row(label: str, pick, pool, chance: str = "") -> dict[tuple[str, str], float]:
        got: dict[tuple[str, str], float] = {}
        cells = []
        for m in MODES:
            rows = [t for t in trials if (t["head"], t["readout"]) == m and pool(t)]
            n = len(rows)
            hit = sum(1 for t in rows if pick(t))
            got[m] = hit / n if n else 0.0
            cells.append(f"{hit:>6}/{n:<3} {hit / n * 100:3.0f}%" if n else "     -")
        print(f"  {label:<34}" + "".join(f"{c:>17}" for c in cells)
              + (f"   {chance}" if chance else ""))
        return got

    blk = lambda t: t["kind"] == "blocked"                            # noqa: E731
    opn = lambda t: t["kind"] == "open"                               # noqa: E731
    avoid = sector_row("blocked: avoided the straight arc",
                       lambda t: abs(t["kappa"]) > STRAIGHT_K, blk, "chance 57%")
    toward = sector_row("blocked: turned toward the open side",
                        lambda t: t["kappa"] * t["open_side"] > 0,
                        lambda t: blk(t) and t["open_side"] != 0, "chance 43%")
    keep = sector_row("open: stayed straight",
                      lambda t: abs(t["kappa"]) <= STRAIGHT_K, opn, "chance 43%")
    print()
    for m in MODES:
        rows = [t for t in trials if (t["head"], t["readout"]) == m]
        print(f"    {MODE_LABEL[m]:<18} mean |kappa| blocked "
              f"{statistics.fmean(abs(t['kappa']) for t in rows if blk(t)):.2f}"
              f"   open {statistics.fmean(abs(t['kappa']) for t in rows if opn(t)):.2f}"
              f"   arcs chosen "
              + "".join("x" if any(t["arc"] == j for t in rows) else "." for j in range(len(arcs))))
    print()

    # ---------------------------------------------------------------- MIRROR control
    print("=" * 78)
    print("  MIRROR CONTROL -- no human labels involved")
    print(f"  {'':<34}" + "".join(f"{MODE_LABEL[m]:>17}" for m in MODES))
    mirror: dict[tuple[str, str], float] = {}
    for label, pool in (("labelled frames (blocked)", blk),
                        ("labelled frames (open)", opn)):
        cells = []
        for m in MODES:
            rows = [t for t in trials if (t["head"], t["readout"]) == m and pool(t)]
            byf: dict[int, dict[bool, float]] = {}
            for t in rows:
                byf.setdefault(t["frame"], {})[t["mirrored"]] = t["kappa"]
            got = [(v[False], v[True]) for v in byf.values() if len(v) == 2]
            flipped = sum(a * b < 0 for a, b in got)
            if pool is blk:
                mirror[m] = flipped / len(got) if got else 0.0
            cells.append(f"{flipped:>6}/{len(got):<3} {flipped / len(got) * 100:3.0f}%"
                         if got else "     -")
        print(f"  {label:<34}" + "".join(f"{c:>17}" for c in cells))
    print(f"  {'':<34}"
          + "   a positional prior scores 0% here, not 43%: the columns are symmetric")
    print(f"  {'':<34}"
          + "   about the optical axis, so a fixed window index gives the SAME kappa twice")
    print(f"  {'':<34}   the 27B scored 0/18 on this control")

    sweep: dict[tuple[str, str], tuple[int, int]] = {}
    if args.mirror_sweep:
        step = args.mirror_sweep
        sweep_dir = Path(args.sweep_dir)
        lo, hi = (int(x) for x in args.sweep_range.split(":"))
        ids = sorted(int(p.stem.split("_")[1]) for p in sweep_dir.glob("nav_*.jpg"))
        ids = [i for i in ids if lo <= i <= hi][::step]
        load = lambda fid, m: (  # noqa: E731  -- rebound to the sweep's directory
            Image.open(sweep_dir / f"nav_{fid:06d}.jpg").convert("RGB")
            .transpose(Image.FLIP_LEFT_RIGHT) if m else
            Image.open(sweep_dir / f"nav_{fid:06d}.jpg").convert("RGB"))
        print(f"\n  sweeping every {step}th frame of {sweep_dir} in [{lo}, {hi}], "
              f"{len(ids)} frames, no labels used")
        cells = []
        t0 = time.time()
        per: dict[tuple[str, str], list[tuple[float, float]]] = {m: [] for m in MODES}
        for fid in ids:
            a = sector_values(vf, crops(load(fid, False), cols, args.sector_w))
            b = sector_values(vf, crops(load(fid, True), cols, args.sector_w))
            for m in MODES:
                ka = kappas[max(range(len(arcs)), key=lambda j: a[m][j])]
                kb = kappas[max(range(len(arcs)), key=lambda j: b[m][j])]
                per[m].append((ka, kb))
        for m in MODES:
            got = per[m]
            flipped = sum(x * y < 0 for x, y in got)
            same = sum(x == y for x, y in got)
            sweep[m] = (flipped, len(got))
            cells.append(f"{flipped:>6}/{len(got):<3} {flipped / len(got) * 100:3.0f}%")
        print(f"  {'kappa negated when flipped':<34}" + "".join(f"{c:>17}" for c in cells))
        cells = []
        for m in MODES:
            same = sum(x == y for x, y in per[m])
            cells.append(f"{same:>6}/{len(per[m]):<3} {same / len(per[m]) * 100:3.0f}%")
        print(f"  {'identical arc both ways (a prior)':<34}"
              + "".join(f"{c:>17}" for c in cells))
        print(f"  {len(ids)} frames x 2 flips x 4 readouts in {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------- VERDICT
    print("\n" + "=" * 78)
    # Selected on the SWEEP when it ran, not on the labelled column, and the difference is
    # not cosmetic: the labelled mirror column is six frames, and six frames will hand you
    # a 6/6 by chance often enough that picking the winner off it is picking noise. The
    # first run of this file did exactly that -- `itm / clear-wall` scored 6/6 on the
    # labelled blocked frames and 70% over 56 unlabelled ones, while `itc / clear` scored
    # 4/6 and 84%. The sweep is the well-powered number and is the one that decides.
    if sweep:
        best = max(MODES, key=lambda m: (sweep[m][0] / max(sweep[m][1], 1), toward[m]))
        basis = f"{sweep[best][1]}-frame mirror sweep"
    else:
        best = max(MODES, key=lambda m: (mirror[m], toward[m]))
        basis = f"mirror on {len(BLOCKED)} labelled frames -- underpowered, run --mirror-sweep"
    print(f"  best readout: {MODE_LABEL[best]}   (chosen on the {basis})")
    a, k, fr = avoid[best], keep[best], frame_rate[best]
    mi = sweep[best][0] / sweep[best][1] if sweep else mirror[best]
    if mi > 0.7 and a > 0.7 and k > 0.7:
        verdict = ("PASS -- it swerves where straight is blocked, holds straight where it "
                   "is not, and its choice mirrors with the image.")
    elif mi > 0.7 and a > 0.7 and k < 0.5:
        verdict = (f"MIRRORS BUT DOES NOT NAVIGATE -- the value tracks the image "
                   f"({mi * 100:.0f}% of flips negate the choice), so it is reading the "
                   f"picture and not answering from a positional prior. But it swerves "
                   f"off {a * 100:.0f}% of blocked straights AND {(1 - k) * 100:.0f}% of "
                   f"open ones, and finds the labelled open side only "
                   f"{toward[best] * 100:.0f}% of the time against 43% chance. This is "
                   f"the always-swerves model the OPEN set exists to catch: it is "
                   f"scoring something real in the image, and that something is not free "
                   f"floor.")
    elif mi > 0.7:
        verdict = (f"MIRRORS BUT DOES NOT NAVIGATE -- the value moves with the image "
                   f"({mi * 100:.0f}%), so it is reading the picture, but avoid "
                   f"{a * 100:.0f}% / keep {k * 100:.0f}% says what it moves toward is "
                   f"not free space.")
    elif fr > 0.7:
        verdict = (f"SEPARATES FRAMES, NOT BEARINGS -- {fr * 100:.0f}% of blocked/open "
                   f"frame pairs rank correctly, but the per-sector choice does not "
                   f"mirror ({mi * 100:.0f}%). A whole-frame value is not a steering "
                   f"signal; VLFM gets the bearing from the frontier map, which is the "
                   f"depth-built half we cannot port.")
    else:
        verdict = (f"NO SIGNAL -- frame pairs {fr * 100:.0f}%, mirror {mi * 100:.0f}%, "
                   f"avoid {a * 100:.0f}%, keep {k * 100:.0f}%. Read the per-frame values "
                   f"before building on any of it.")
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model_dir": args.model_dir, "sector_w": args.sector_w,
            "sector_deg": args.sector_w * deg_per_px, "columns": cols,
            "kappas": kappas, "prompt_pos": PROMPT_POS, "prompt_neg": PROMPT_NEG,
            "whole_frame": {str(k2): {f"{h}/{r}": v for (h, r), v in d.items()}
                            for k2, d in whole.items()},
            "frame_pair_rate": {f"{h}/{r}": v for (h, r), v in frame_rate.items()},
            "avoid": {f"{h}/{r}": v for (h, r), v in avoid.items()},
            "toward_open": {f"{h}/{r}": v for (h, r), v in toward.items()},
            "keep_straight": {f"{h}/{r}": v for (h, r), v in keep.items()},
            "mirror": {f"{h}/{r}": v for (h, r), v in mirror.items()},
            "mirror_sweep": {f"{h}/{r}": v for (h, r), v in sweep.items()},
            "trials": trials,
        }, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
