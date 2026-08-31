"""Turn a recorded arc-menu run into a video of what the model saw and chose.

A benchmark row says an episode closed 87% of its gap with 506 collision-guard stops. It
does not say whether the model kept picking a curve into a wall, whether it lost the
corridor at one specific corner, or whether it was steering fine and the controller was
scraping. Those have different fixes and the trace alone cannot separate them, because the
trace records where the robot went and not what it was looking at when it decided to.

The server writes both, per episode: the rendered menu it made each decision on, and a
JSONL line holding the label shuffle, the choice, the free-space sentence and the
instruction. This assembles them, drawing the CHOSEN path back over the menu so a viewer
can see the decision rather than decode a number, and prints the model's own words under
each frame.

One frame per decision, not per control step. A decision takes ~2.9 s and the loop runs at
60 Hz, so a per-step video would be 174 copies of each frame with nothing changing; the
interesting event is the choice. Frames are held for `--hold` seconds each so the text is
readable, which means the video is NOT real time -- the sim clock is printed on every
frame precisely so nobody reads pacing off it.

Usage:
    python3 nav/tools/make_run_video.py --latest
    python3 nav/tools/make_run_video.py --all --out-dir nav/results/videos
    python3 nav/tools/make_run_video.py /tmp/qvla-menus/20260831-1443_warehouse
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_menu import _PALETTE, badge_xy, make_arcs, project  # noqa: E402

W, H = 1280, 720          # the menu, downscaled from 1920x1080
CAPTION_H = 300           # room for three wrapped lines plus a header
# The third-person panel, when the run recorded one. Same height as the menu so the two
# sit on one baseline; 16:9 keeps the chase camera's own aspect, so nothing is stretched.
CHASE_W = 1280
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def draw_frame(rec: dict, menu_path: Path, arcs, index: int, total: int,
               chase_path: Path | None = None):
    """One video frame: the menu, the chosen path drawn back over it, and the reasoning.

    With `chase_path` the third-person view is placed beside it, same instant, same
    width. The two answer different questions and neither is sufficient: the menu says
    what the model was looking at and what it picked, and only the outside view shows
    whether the robot then actually went there -- a scraped wall, a pivot in place, a
    reverse out of a wedge are all invisible from the camera doing the deciding, because
    that camera moves with them.
    """
    from PIL import Image, ImageDraw

    im = Image.open(menu_path).convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)

    kappa, stop = rec.get("kappa"), rec.get("stop")
    if kappa is not None:
        # Match on kappa rather than on the label: the label is a shuffled tag and the
        # curvature is the actual decision, so this stays correct even if the menu is
        # ever rendered with a different permutation than the one recorded.
        i = min(range(len(arcs)), key=lambda j: abs(arcs[j].kappa - kappa))
        # project() works in the ORIGINAL 1920x1080 the geometry was computed for; the
        # menu was drawn there too, so scaling the pixels afterwards keeps the redrawn
        # curve exactly on top of the one already in the image.
        px = [(u * W / 1920.0, v * H / 1080.0)
              for u, v in (p for p in (project(x, y) for x, y in arcs[i].points)
                           if p is not None)]
        if len(px) >= 2:
            colour_i = _PALETTE[i % len(_PALETTE)]
            d.line(px, fill=(255, 255, 255), width=18, joint="curve")
            d.line(px, fill=colour_i, width=8, joint="curve")
            # ...and put the badge back on top. The highlight is 18 px of white drawn
            # straight through the chosen arc's own number, so without this the ONE label
            # a viewer needs to read is the only illegible one on the frame -- reported
            # from a real frame where the chosen "7" had the stroke through it.
            r = 96 * 0.72 * H / 1080.0
            bx, by = badge_xy(px, W, H, r)
            d.ellipse([bx - r, by - r, bx + r, by + r], fill=(20, 20, 20),
                      outline=colour_i, width=5)
            d.text((bx, by), str(rec.get("choice")), fill=colour_i,
                   font=_font(FONT_B, int(96 * H / 1080.0)), anchor="mm")

    # Say WHY there is no such number on the picture. The stop label is deliberately not
    # drawn -- it is the one option that is not a path -- and a viewer watching the video
    # has no way to know that, so an unlabelled choice reads as the model inventing a
    # label it could not have seen.
    tag = (f"STOP  (label {rec['choice']} - not drawn, it is not a path)" if stop else
           f"path {rec['choice']}   kappa {kappa:+.2f} 1/m" if kappa is not None
           else "UNREADABLE REPLY -- previous plan reused")
    colour = ((255, 90, 90) if stop else (120, 255, 160) if kappa is not None
              else (255, 200, 80))
    d.rectangle([0, 0, W, 54], fill=(15, 15, 18))
    # Commanded speed on every frame, because "the robot looks too fast" is a judgement a
    # viewer should be able to check against a number rather than estimate off playback
    # that is not real time in the first place. `target` is the model's own distance
    # estimate; blank when it said it could not see the target, which is also when the
    # speed is at cruise and worth being able to tell apart from a near-target cruise.
    v, tgt = rec.get("speed_mps"), rec.get("target_m")
    speed = "" if v is None else (f"   {v:.2f} m/s" + (f"  target {tgt:.1f} m" if tgt
                                                      is not None else "  target unseen"))
    # Decisions taken right after a reverse were asked a DIFFERENT question -- the menu
    # they saw had no STOP on it and the description was told the way ahead is the way
    # that just failed. Without a mark, those frames look like ordinary ones that happen
    # never to stop, which is the wrong conclusion to let a viewer draw from a video.
    after = "   [after reversing out]" if rec.get("recovered") else ""
    # The banner is one line over a panel of fixed width and both halves grow: the left
    # gains a target distance and a recovery mark, the right gains a long STOP
    # explanation. Laid out at fixed sizes they overlapped for real -- the recovery mark
    # printed straight through "path 4", so the two things this frame exists to show were
    # the two that were illegible. Measure and shrink instead; the marker earns its place
    # by being kept last, so what degrades first is a word and not a number.
    # Nothing here is ever DROPPED to make room, only shrunk. A layout that discards the
    # recovery mark when the line is long discards it on exactly the frames where the
    # right-hand tag is longest, which is the STOP tag -- i.e. it would go missing on
    # "stopped anyway despite the no-stop prompt", the one combination worth seeing.
    short = "   [recovered]" if after else ""
    for size, mark in ((26, after), (26, short), (23, short), (20, short), (17, short)):
        left = f"decision {index + 1}/{total}   step {rec.get('step')}{speed}{mark}"
        left_font, tag_font = _font(FONT, size), _font(FONT_B, size + 2)
        if (d.textlength(left, font=left_font)
                + d.textlength(tag, font=tag_font) <= W - 56):
            break
    d.text((16, 27), left, fill=(255, 200, 80) if after else (190, 190, 200),
           font=left_font, anchor="lm")
    d.text((W - 16, 27), tag, fill=colour, font=tag_font, anchor="rm")

    # `chase_path` not None means the RUN recorded a third-person view, so the panel is
    # reserved whether or not this particular frame survived. Sizing the canvas off
    # whether one file happens to exist would give a video whose dimensions change
    # mid-stream, which ffmpeg rejects outright -- one lost jpeg would cost the video.
    chase = None
    if chase_path is not None and chase_path.is_file():
        try:
            chase = Image.open(chase_path).convert("RGB").resize(
                (CHASE_W, H), Image.LANCZOS)
        except Exception:
            chase = None      # truncated jpeg from a run killed mid-write

    total_w = W + (CHASE_W if chase_path is not None else 0)
    out = Image.new("RGB", (total_w, H + CAPTION_H), (15, 15, 18))
    out.paste(im, (0, 0))
    c = ImageDraw.Draw(out)
    if chase_path is not None:
        if chase is not None:
            out.paste(chase, (W, 0))
        else:
            c.text((W + CHASE_W // 2, H // 2), "third-person frame missing",
                   fill=(90, 95, 115), font=_font(FONT, 30), anchor="mm")
        # A seam and a label, because two 16:9 renders of the same room side by side are
        # genuinely ambiguous about which one the model is answering on.
        c.rectangle([W - 2, 0, W + 1, H], fill=(15, 15, 18))
        for x, text, col in ((16, "what the model sees  (its own camera)", (150, 200, 255)),
                             (W + 16, "third person  (not shown to the model)",
                              (255, 190, 120))):
            c.text((x, H - 18), text, fill=col, font=_font(FONT_B, 20), anchor="ls")

    y = H + 22
    for label, body, col, size in (
        ("instruction", rec.get("instruction", ""), (255, 255, 255), 25),
        ("sees", rec.get("free_space", ""), (185, 195, 215), 23),
        ("menu", f"labels {rec.get('labels')}  ->  answered "
                 f"{rec.get('reply', '')!r}", (140, 145, 165), 21),
    ):
        c.text((22, y), label, fill=(105, 110, 130), font=_font(FONT_B, 17))
        for line in _wrap(str(body), _font(FONT, size), total_w - 132 - 22)[:3]:
            c.text((132, y), line, fill=col, font=_font(FONT, size))
            y += size + 7
        y += 12
    return out


def _wrap(text: str, font, px: int) -> list[str]:
    """Wrap to a PIXEL width by measuring, not to a character count by guessing.

    A character estimate has to assume an average glyph width, and DejaVu's is not the
    one an estimate picks: the model's descriptions ran off the right edge of the frame
    at a width that looked safe on paper. Measuring costs nothing here and cannot be
    wrong about the font it is actually drawing with.
    """
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and font.getlength(trial) > px:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def build(run_dir: Path, out_dir: Path, hold: float, keep_frames: bool) -> Path | None:
    recs = [json.loads(l) for l in
            (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    if not recs:
        print(f"  {run_dir.name}: no decisions recorded, skipping")
        return None

    arcs = make_arcs()
    tmp = Path(tempfile.mkdtemp(prefix="qvla-vid-"))
    # Decided once for the whole run, not per frame: see draw_frame on why the canvas
    # cannot change size mid-video. Runs recorded before the chase camera was wired in
    # have none, and still build exactly as they did.
    has_chase = any(run_dir.glob("chase_*.jpg"))
    n = 0
    for i, rec in enumerate(recs):
        menu = run_dir / rec["menu"]
        if not menu.is_file():
            continue          # a decision whose image was cleaned up; skip, do not fake it
        # Paired on the filename rather than on a second index. Both are written from the
        # same step number, so the two views cannot drift apart without the names saying so.
        chase = run_dir / rec["menu"].replace("menu_", "chase_") if has_chase else None
        draw_frame(rec, menu, arcs, i, len(recs), chase).save(tmp / f"f{n:05d}.png")
        n += 1
    if n == 0:
        print(f"  {run_dir.name}: {len(recs)} decisions but no menu images survived")
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_dir.name}.mp4"
    # yuv420p and the even-dimension crop are what make this play in a browser and in
    # QuickTime rather than only in mpv.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{1.0 / hold:.4f}",
           "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", "30", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {run_dir.name}: ffmpeg failed\n{r.stderr[:400]}", file=sys.stderr)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    if keep_frames:
        print(f"  frames kept in {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    stops = sum(bool(x.get("stop")) for x in recs)
    bad = sum(x.get("kappa") is None and not x.get("stop") for x in recs)
    print(f"  {out}  ({n} decisions, {n * hold:.0f}s, {stops} stop, {bad} unreadable)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--menu-root", type=Path, default=Path("/tmp/qvla-menus"))
    ap.add_argument("--latest", action="store_true", help="the most recent run only")
    ap.add_argument("--all", action="store_true", help="every recorded run")
    ap.add_argument("--out-dir", type=Path, default=Path("nav/results/videos"))
    ap.add_argument("--hold", type=float, default=1.4,
                    help="seconds each decision is held on screen")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    runs = list(args.run_dirs)
    if args.latest or args.all:
        found = sorted(p for p in args.menu_root.glob("*/")
                       if (p / "decisions.jsonl").is_file())
        runs += found[-1:] if args.latest else found
    if not runs:
        print(f"no recorded runs under {args.menu_root} -- the server records only when "
              f"/reset is given a run name, which run_navigation.py does per episode.",
              file=sys.stderr)
        return 1

    print(f"{len(runs)} run(s) -> {args.out_dir}")
    made = [build(r, args.out_dir, args.hold, args.keep_frames) for r in runs]
    return 0 if any(m for m in made) else 1


if __name__ == "__main__":
    raise SystemExit(main())
