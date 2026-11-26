"""Compose and encode the frames nav/tools/g1_render_check.py saved, AS RENDERED: no gamma, no brightness,
no contrast, no denoise. The only pixel operations are a resize (Lanczos) into the panel and the text
labels drawn on a black strip OUTSIDE the camera panels.

Layout (1920 x 760):  [ head camera, 12 deg (the nav camera), 1280x720 ] [ real 47.6 deg mount 640x360 ]
                                                                         [ chase view 640x360          ]
                      [ 40 px label strip: time, command, measured body speed, tilt, frame luminance  ]

Then it verifies the encode did not change brightness: decodes frames back out of the mp4 and compares
the mean luminance of the nav panel with the same panel composed from the source jpg (reported in
<frames_dir>/encode_check.json).

    python3 nav/tools/g1_encode_video.py --run <render_check out dir> --out nav/results/g1/<name>.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--crf", type=int, default=18)
ap.add_argument("--nav-label", default="G1 head camera: D435 mount, re-pitched to 12 deg down (nav camera)")
ap.add_argument("--real-label", default="same D435, real mount: 47.6 deg down")
ap.add_argument("--real-label2", default="", help="second line under the real-mount label")
ap.add_argument("--chase-label", default="chase view (not seen by the policy)")
args = ap.parse_args()
RUN = Path(args.run)
rep = json.loads((RUN / "report.json").read_text())
rows = {r["i"]: r for r in rep.get("video_rows", [])}
nav_dir, real_dir, chase_dir = RUN / "frames" / "nav", RUN / "frames" / "real", RUN / "frames" / "chase"
frames = sorted(p.stem for p in nav_dir.glob("*.jpg"))
if not frames:
    raise SystemExit(f"no frames in {nav_dir}")


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_LABEL, F_STRIP = font(22), font(20)
W, H_PANELS, H_STRIP = 1920, 720, 40
LUMA_ROWS = slice(60, 720)        # nav panel rows below its label box: what the luminance check compares
intensity = rep.get("video_light_intensity")


def lum(a: np.ndarray) -> float:
    a = a.astype(np.float32)
    return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())


def label(draw: ImageDraw.ImageDraw, xy, text):
    # text on a solid box so it never alters the scene pixels' statistics in an unknown way
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=F_LABEL)
    draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=F_LABEL)


def compose(stem: str) -> tuple[Image.Image, np.ndarray]:
    canvas = Image.new("RGB", (W, H_PANELS + H_STRIP), (0, 0, 0))
    nav = Image.open(nav_dir / f"{stem}.jpg").convert("RGB").resize((1280, 720), Image.LANCZOS)
    nav_arr = np.asarray(nav)[LUMA_ROWS]
    canvas.paste(nav, (0, 0))
    for d, y in ((real_dir, 0), (chase_dir, 360)):
        p = d / f"{stem}.jpg"
        if p.exists():
            canvas.paste(Image.open(p).convert("RGB").resize((640, 360), Image.LANCZOS), (1280, y))
    dr = ImageDraw.Draw(canvas)
    label(dr, (14, 10), args.nav_label)
    if (real_dir / f"{stem}.jpg").exists():
        label(dr, (1290, 10), args.real_label)
        if args.real_label2:
            label(dr, (1290, 44), args.real_label2)
    label(dr, (1290, 370), args.chase_label)
    r = rows.get(int(stem))
    if r:
        lr = f"  real {r['lum_real']:.0f}" if r.get("lum_real") is not None else ""
        seg = f"[{r['seg']}]  " if r.get("seg") else ""
        txt = (f"{seg}t {r['t']:5.1f} s   cmd vx {r['cmd'][0]:+.2f}  vy {r['cmd'][1]:+.2f}  wz {r['cmd'][2]:+.2f}   "
               f"body v {r['v_b'][0]:.2f} m/s   tilt {r['tilt']:.1f} deg   head cam h {r['menu_args']['cam_h']:.3f} m "
               f"pitch {r['menu_args']['pitch_deg']:.1f} deg   mean luma nav {r.get('lum_nav') or 0:.0f}{lr}/255")
        dr.text((14, H_PANELS + 9), txt, fill=(230, 230, 230), font=F_STRIP)
    return canvas, nav_arr


with tempfile.TemporaryDirectory(dir=str(RUN)) as tmp:
    src_lums = []
    for k, stem in enumerate(frames):
        img, nav_arr = compose(stem)
        img.save(Path(tmp) / f"{k:05d}.png")
        src_lums.append(lum(nav_arr))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}", "-i", str(Path(tmp) / "%05d.png"),
           "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-pix_fmt", "yuv420p", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"ffmpeg failed ({res.returncode}): {res.stderr[-2000:]}")
    # decode a few frames back and compare the nav panel's luminance with the composed source
    check = []
    for k in np.linspace(0, len(frames) - 1, 6).astype(int).tolist():
        png = Path(tmp) / f"dec_{k:05d}.png"
        r2 = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-vf", f"select=eq(n\\,{k})",
                             "-vframes", "1", str(png)], capture_output=True, text=True)
        if r2.returncode == 0 and png.exists():
            dec = np.asarray(Image.open(png).convert("RGB"))[:720, :1280][LUMA_ROWS]
            check.append({"frame": k, "source_nav_luma": round(src_lums[k], 2), "decoded_nav_luma": round(lum(dec), 2)})
    rep_out = {"video": str(out), "frames": len(frames), "fps": args.fps, "crf": args.crf, "ffmpeg": " ".join(cmd[:-1]) + " <out>",
               "filters": "none (scale+labels in PIL, yuv420p conversion in libx264)", "luma_check": check,
               "max_abs_luma_diff": max((abs(c["source_nav_luma"] - c["decoded_nav_luma"]) for c in check), default=None),
               "headlight_intensity": intensity}
    (RUN / "encode_check.json").write_text(json.dumps(rep_out, indent=1))
    print(json.dumps(rep_out, indent=1))
