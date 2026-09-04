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

TWO TIMELINES, chosen by what the run actually recorded.

A run that captured a third-person frame every couple of physics steps gets a REAL-TIME
video: one video frame per captured frame, 30 fps, so the robot's motion is motion. The
decision panel beside it holds the last choice made at or before that instant, which is
literally what the robot was driving on. This is the readable one -- a reverse out of a
wedge, a pivot in place, a wall scrape are all continuous events and a slideshow cannot
show any of them.

A run that only captured on decisions -- everything recorded before per-frame capture
existed -- gets the old one-frame-per-decision video, each held `--hold` seconds. That is
not real time and the sim clock is printed on every frame precisely so nobody reads pacing
off it.

The switch is `len(chase frames) > 2 * len(decisions)` and not a flag, because it is a
question about the files on disk rather than about what the operator wants: asking for a
30 fps video of 13 frames would produce half a second of video, and asking for a held
slideshow of 3000 frames would produce an hour of it.

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
# ...and more when the run has reasoning to show. Decided once per run and passed down,
# never per frame: the canvas cannot change size mid-video, ffmpeg rejects it outright.
CAPTION_H_THINK = 420
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


def decision_panel(rec: dict, menu_path: Path, arcs, index: int, total: int):
    """The W x H left panel: the menu, the chosen path drawn back over it, and a header.

    Separated from the composition below because it is the EXPENSIVE half and the one that
    does not change between video frames. On a real-time video a decision covers ~90
    frames; redrawing the arc overlay, the badge and the shrink-to-fit header ninety times
    to produce ninety identical panels would dominate the whole build. Rendered once per
    decision and pasted, it is a memcpy.
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
    # Who chose the speed, in one word, on every frame that had a choice. Without it a
    # slow approach and a slow ramp look identical, and they are the two hypotheses the
    # whole speed channel exists to separate.
    src = "*" if rec.get("speed_source") == "model" else ""
    speed = "" if v is None else (f"   {v:.2f} m/s{src}"
                                  + (f"  target {tgt:.1f} m" if tgt is not None
                                     else "  target unseen"))
    # Decisions taken right after a reverse were asked a DIFFERENT question -- the menu
    # they saw had no STOP on it and the description was told what had just happened.
    # Without a mark, those frames look like ordinary ones that happen never to stop,
    # which is the wrong conclusion to let a viewer draw from a video.
    #
    # WHICH reverse, because the two mean opposite things to someone watching: after a
    # wedge there is an obstacle in the picture and it is the one that stopped the robot;
    # after a balk there is nothing in the picture at all and that IS the problem. A
    # viewer told "reversed" looks for an obstacle, and on a balk frame will find one that
    # was never involved.
    kind = rec.get("kind") or ("wedge" if rec.get("recovered") else "")
    st = rec.get("stalled_s") or 0.0
    stall = f"   stalled {st:.0f}s" if st >= 1.0 else ""
    after = {"wedge": "   [after reversing out of a wedge]",
             "balk": "   [after backing off -- it had stalled]"}.get(kind, "") + stall
    # The banner is one line over a panel of fixed width and both halves grow: the left
    # gains a target distance and a recovery mark, the right gains a long STOP
    # explanation. Laid out at fixed sizes they overlapped for real -- the recovery mark
    # printed straight through "path 4", so the two things this frame exists to show were
    # the two that were illegible. Measure and shrink instead; the marker earns its place
    # by being kept last, so what degrades first is a word and not a number.
    # Nothing here is ever DROPPED to make room, only ABBREVIATED, and the abbreviation
    # keeps both facts. A layout that discards the recovery mark when the line is long
    # discards it on exactly the frames where the right-hand tag is longest, which is the
    # STOP tag -- i.e. it would go missing on "stopped anyway despite the no-stop prompt",
    # the one combination worth seeing. The stall seconds ride inside the short form for
    # the same reason: on a balk frame they are the evidence, not an annotation, and the
    # long form is long enough that it is the short one that will actually be drawn.
    mid = f"   [after a {kind}]{stall}" if after else ""
    short = (f"   [{kind}{stall.replace('   stalled ', ' ')}]" if after else "")
    for size, mark in ((26, after), (26, mid), (26, short), (23, short), (20, short),
                       (17, short)):
        left = f"decision {index + 1}/{total}   step {rec.get('step')}{speed}{mark}"
        left_font, tag_font = _font(FONT, size), _font(FONT_B, size + 2)
        if (d.textlength(left, font=left_font)
                + d.textlength(tag, font=tag_font) <= W - 56):
            break
    d.text((16, 27), left, fill=(255, 200, 80) if after else (190, 190, 200),
           font=left_font, anchor="lm")
    d.text((W - 16, 27), tag, fill=colour, font=tag_font, anchor="rm")
    return im


def decision_canvas(rec: dict, panel, wide: bool, caption_h: int):
    """The full canvas with everything EXCEPT the third-person picture drawn on it.

    Also cached per decision. The seam, the two panel labels and the caption all depend
    only on the decision, so on a real-time video they are drawn once per ~90 frames and
    the per-frame cost collapses to one copy plus one paste.

    `wide` says the RUN recorded a third-person view, so the panel is reserved whether or
    not any given frame survived. Sizing the canvas off whether one file happens to exist
    would give a video whose dimensions change mid-stream, which ffmpeg rejects outright --
    one lost jpeg would cost the whole video.
    """
    from PIL import Image, ImageDraw

    total_w = W + (CHASE_W if wide else 0)
    out = Image.new("RGB", (total_w, H + caption_h), (15, 15, 18))
    out.paste(panel, (0, 0))
    c = ImageDraw.Draw(out)

    y = H + 22
    rows = [("instruction", rec.get("instruction", ""), (255, 255, 255), 25),
            ("sees", rec.get("free_space", ""), (185, 195, 215), 23),
            ("menu", f"labels {rec.get('labels')}  ->  answered "
                     f"{rec.get('reply', '')!r}", (140, 145, 165), 21)]
    if rec.get("history"):
        # What the robot already tried. It is in the caption rather than folded into
        # "sees" because it is the one line that did NOT come from the picture, and a
        # viewer comparing the description against the frame needs to know which is which.
        rows.insert(2, ("tried", rec["history"], (200, 175, 140), 21))
    think = " || ".join(t for t in (rec.get("think_describe"), rec.get("think_select"))
                        if t)
    if think:
        rows.append(("thinks", think, (150, 180, 220), 20))
    for label, body, col, size in rows:
        c.text((22, y), label, fill=(105, 110, 130), font=_font(FONT_B, 17))
        for line in _wrap(str(body), _font(FONT, size), total_w - 132 - 22)[:3]:
            c.text((132, y), line, fill=col, font=_font(FONT, size))
            y += size + 7
        y += 12
    return out


def paste_chase(canvas, chase_path: Path | None):
    """Paste one third-person frame onto a cached canvas and label both panels.

    Mutates and returns `canvas`, so callers on the real-time path must hand it a copy.
    The labels are drawn AFTER the paste and not baked into the cache because they sit at
    the bottom of the picture area, inside the region the chase frame covers.
    """
    from PIL import Image, ImageDraw

    c = ImageDraw.Draw(canvas)
    chase = None
    if chase_path is not None and chase_path.is_file():
        try:
            chase = Image.open(chase_path).convert("RGB").resize(
                (CHASE_W, H), Image.LANCZOS)
        except Exception:
            chase = None      # truncated jpeg from a run killed mid-write
    if chase is not None:
        canvas.paste(chase, (W, 0))
    else:
        c.rectangle([W, 0, W + CHASE_W, H], fill=(15, 15, 18))
        c.text((W + CHASE_W // 2, H // 2), "third-person frame missing",
               fill=(90, 95, 115), font=_font(FONT, 30), anchor="mm")
    # A seam and a label, because two 16:9 renders of the same room side by side are
    # genuinely ambiguous about which one the model is answering on.
    c.rectangle([W - 2, 0, W + 1, H], fill=(15, 15, 18))
    for x, text, col in ((16, "what the model sees  (its own camera)", (150, 200, 255)),
                         (W + 16, "third person  (not shown to the model)",
                          (255, 190, 120))):
        c.text((x, H - 18), text, fill=col, font=_font(FONT_B, 20), anchor="ls")
    return canvas


def draw_frame(rec: dict, menu_path: Path, arcs, index: int, total: int,
               chase_path: Path | None = None, caption_h: int = CAPTION_H):
    """One complete frame, rendered from scratch. The slideshow path uses this."""
    canvas = decision_canvas(
        rec, decision_panel(rec, menu_path, arcs, index, total),
        chase_path is not None, caption_h)
    return canvas if chase_path is None else paste_chase(canvas, chase_path)


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


def _encode(frames, size: tuple[int, int], fps: float, out: Path) -> bool:
    """Pipe RGB frames straight into ffmpeg. Returns whether it worked.

    Raw over stdin rather than a directory of PNGs, and the reason is the real-time
    timeline: a 100 s episode at 30 fps is 3000 frames, and 3000 PNGs of a 2560x1140
    canvas is about 6 GB written and read back for one 40 MB video. Piping writes nothing
    to disk and removes the temp directory as a failure mode entirely.

    `frames` is consumed lazily, so peak memory is one canvas regardless of length.
    """
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-framerate", f"{fps:.4f}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "veryfast",
           # yuv420p needs even dimensions; the pad is what makes this play in a browser
           # and in QuickTime rather than only in mpv.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", "30", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for im in frames:
            p.stdin.write(im.tobytes())
    except BrokenPipeError:
        pass                  # ffmpeg died; its stderr below is the real message
    p.stdin.close()
    err = p.stderr.read().decode()[:400]
    if p.wait() != 0:
        print(f"  ffmpeg failed\n{err}", file=sys.stderr)
        return False
    return True


def _steps(recs: list[dict]) -> list[int]:
    """Decision steps, with None treated as 0 so a run missing them still orders."""
    return [int(r.get("step") or 0) for r in recs]


def _timeline_frames(run_dir: Path, recs: list[dict], arcs, chases: list[Path],
                     caption_h: int, keep_frames: bool):
    """One frame per captured third-person frame, paired with the decision in force.

    The pairing is "the last decision at or before this step", which is not an
    approximation -- it is exactly what the robot was driving on at that instant. A
    decision generated at step 900 governs every step from 900 until the next one lands,
    so a viewer watching the robot at step 947 is watching that decision being executed.

    Frames before the FIRST decision are dropped rather than paired with it. The robot is
    driving on nothing there (the first call blocks until a plan exists), and pairing them
    forward would show a choice being executed several seconds before it was made.
    """
    from PIL import Image

    steps = _steps(recs)
    cached_i, canvas = -1, None
    kept = 0
    for path in chases:
        try:
            s = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        # The last decision at or before this step.
        i = -1
        for j, ds in enumerate(steps):
            if ds <= s:
                i = j
            else:
                break
        if i < 0:
            continue
        if i != cached_i:
            menu = run_dir / recs[i]["menu"]
            if not menu.is_file():
                continue      # image cleaned up; skip, do not fake it
            canvas = decision_canvas(
                recs[i], decision_panel(recs[i], menu, arcs, i, len(recs)),
                True, caption_h)
            cached_i = i
        frame = paste_chase(canvas.copy(), path)
        if keep_frames and kept < 8:
            frame.save(Path(tempfile.gettempdir()) / f"qvla-frame-{kept}.png")
            kept += 1
        yield frame
    del Image


def build(run_dir: Path, out_dir: Path, hold: float, keep_frames: bool,
          fps: float = 30.0) -> Path | None:
    recs = [json.loads(l) for l in
            (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    if not recs:
        print(f"  {run_dir.name}: no decisions recorded, skipping")
        return None

    arcs = make_arcs()
    # Decided once for the whole run, not per frame: see `decision_canvas` on why the
    # canvas cannot change size mid-video. Runs recorded before the chase camera was wired
    # in have none, and still build exactly as they did.
    chases = sorted(run_dir.glob("chase_*.jpg"))
    caption_h = (CAPTION_H_THINK
                 if any(r.get("think_describe") or r.get("think_select") for r in recs)
                 else CAPTION_H)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_dir.name}.mp4"

    if len(chases) > 2 * len(recs):
        # Real time. See the module docstring for why the switch is a property of the
        # files rather than a flag.
        size = (W + CHASE_W, H + caption_h)
        ok = _encode(_timeline_frames(run_dir, recs, arcs, chases, caption_h,
                                      keep_frames), size, fps, out)
        if not ok:
            return None
        secs = len(chases) / fps
        note = f"{len(chases)} frames, {secs:.0f}s real time"
    else:
        def slideshow():
            for i, rec in enumerate(recs):
                menu = run_dir / rec["menu"]
                if not menu.is_file():
                    continue  # image was cleaned up; skip, do not fake it
                # Paired on the filename rather than on a second index. Both are written
                # from the same step, so the two views cannot drift apart silently.
                chase = (run_dir / rec["menu"].replace("menu_", "chase_")
                         if chases else None)
                yield draw_frame(rec, menu, arcs, i, len(recs), chase, caption_h)

        first = next(slideshow(), None)
        if first is None:
            print(f"  {run_dir.name}: {len(recs)} decisions but no menu images survived")
            return None
        if not _encode(slideshow(), first.size, 1.0 / hold, out):
            return None
        note = f"{len(recs)} decisions, {len(recs) * hold:.0f}s held"

    stops = sum(bool(x.get("stop")) for x in recs)
    bad = sum(x.get("kappa") is None and not x.get("stop") for x in recs)
    balks = sum(1 for x in recs if x.get("kind") == "balk")
    print(f"  {out}  ({note}, {stops} stop, {bad} unreadable, {balks} after a balk)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--menu-root", type=Path, default=Path("/tmp/qvla-menus"))
    ap.add_argument("--latest", action="store_true", help="the most recent run only")
    ap.add_argument("--all", action="store_true", help="every recorded run")
    ap.add_argument("--out-dir", type=Path, default=Path("nav/results/videos"))
    ap.add_argument("--hold", type=float, default=1.4,
                    help="slideshow path only: seconds each decision is held on screen")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="real-time path only: playback rate. The runner captures every "
                         "NAV_CHASE_EVERY physics steps, so the default of 30 is 1x sim "
                         "speed at the default capture rate of every 2 steps.")
    ap.add_argument("--keep-frames", action="store_true",
                    help="also write the first few composed frames to /tmp for inspection")
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
    made = [build(r, args.out_dir, args.hold, args.keep_frames, args.fps) for r in runs]
    return 0 if any(m for m in made) else 1


if __name__ == "__main__":
    raise SystemExit(main())
