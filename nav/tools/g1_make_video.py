"""Compose the panels written by nav/tools/try_g1_walk.py into one labelled H.264 video.

    python3 nav/tools/g1_make_video.py <run_dir> <out.mp4> [--fps 10]

Plain Python + PIL + ffmpeg, no Kit: the layout can change without another simulation run.
Layout: with the real-mount camera recorded, a 2x2 grid (1920x1080):
    [ head cam 12 deg (nav camera) | head cam 47.6 deg (real mount) ]
    [ third-person chase           | telemetry                      ]
otherwise one row [ head cam 12 deg | chase ] over a telemetry strip (1920x880).
Every panel says what it is, and the 12 deg view says it is a deviation from the real robot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

LABELS = {
    "nav": "G1 OWN HEAD CAMERA (D435 mount) re-pitched 12° down -- nav camera; deviation: real mount is 47.6°",
    "real": "G1 OWN HEAD CAMERA (D435 mount) as on the real robot: 47.6° down",
    "chase": "third-person chase view (not a robot sensor)",
}


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def label(img: Image.Image, text: str, color=(255, 255, 255)) -> Image.Image:
    d = ImageDraw.Draw(img, "RGBA")
    f = font(17, bold=True)
    w = d.textlength(text, font=f)
    d.rectangle([0, 0, min(img.width, w + 16), 28], fill=(0, 0, 0, 170))
    d.text((8, 4), text, font=f, fill=color)
    return img


def telemetry(meta: dict, report: dict, size) -> Image.Image:
    img = Image.new("RGB", size, (18, 20, 24))
    d = ImageDraw.Draw(img)
    f, fb = font(22), font(24, bold=True)
    cmd = meta["cmd"]
    vb, wb = meta["v_b"], meta["w_b"]
    lines = [
        (f"t = {meta['t']:5.1f} s     segment: {meta['seg']}", fb),
        (f"command  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f} m/s   wz {cmd[2]:+.2f} rad/s", f),
        (f"measured vx {vb[0]:+.2f}  vy {vb[1]:+.2f} m/s   wz {wb[2]:+.2f} rad/s  (pelvis frame)", f),
        (f"pelvis height {meta['pelvis_h']:.3f} m   roll {meta['roll_deg']:+.1f}  pitch {meta['pitch_deg']:+.1f}  tilt {meta['tilt_deg']:.1f} deg", f),
        (f"head cam (12 deg mount): {meta['cam_h']:.3f} m above floor, view {meta['cam_pitch']:.1f} deg down, roll {meta['cam_roll']:+.1f}", f),
    ]
    cn = meta.get("contact_n")
    if cn:
        lines.append((f"contact N:  L foot {cn.get('left_ankle_roll_link', 0):6.0f}   R foot "
                      f"{cn.get('right_ankle_roll_link', 0):6.0f}   torso {cn.get('torso_link', 0):5.1f}", f))
    if meta.get("goal_dist_m") is not None:
        lines.append((f"goal distance {meta['goal_dist_m']:.2f} m    clear ahead (chest height) "
                      f"{meta.get('fwd_clear_m') if meta.get('fwd_clear_m') is not None else '>30'} m", f))
    lines.append(("FALLEN" if meta.get("fallen") else "upright", fb))
    ep = report.get("episode")
    foot = (f"{ep['name']}: {ep['instruction'][:70]}" if ep else
            "flat Grid environment -- Isaac Lab Isaac-Velocity-Flat-G1-v0 policy, 200 Hz physics, 50 Hz policy")
    y = 18
    for text, ff in lines:
        d.text((24, y), text, font=ff, fill=(235, 235, 235) if ff is f else (255, 210, 90))
        y += 40
    d.text((24, size[1] - 40), foot, font=font(18), fill=(160, 170, 180))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    run = Path(a.run_dir)
    tel = json.loads((run / "telemetry.json").read_text())
    report = json.loads((run / "report.json").read_text())
    frames = tel["frames"]
    comp = run / "composed"
    comp.mkdir(exist_ok=True)
    has_real = (run / "frames" / "real_00000.jpg").exists()
    has_chase = (run / "frames" / "chase_00000.jpg").exists()
    blank = Image.new("RGB", (960, 540), (0, 0, 0))

    def panel(name, i):
        p = run / "frames" / f"{name}_{i:05d}.jpg"
        im = Image.open(p).convert("RGB") if p.exists() else blank.copy()
        return label(im, LABELS[name], (255, 230, 120) if name == "nav" else (255, 255, 255))

    n = 0
    for m in frames:
        i = m["i"]
        if has_real:
            canvas = Image.new("RGB", (1920, 1080))
            canvas.paste(panel("nav", i), (0, 0))
            canvas.paste(panel("real", i), (960, 0))
            canvas.paste(panel("chase", i) if has_chase else blank, (0, 540))
            canvas.paste(telemetry(m, report, (960, 540)), (960, 540))
        else:
            canvas = Image.new("RGB", (1920, 880))
            canvas.paste(panel("nav", i), (0, 0))
            canvas.paste(panel("chase", i) if has_chase else blank, (960, 0))
            canvas.paste(telemetry(m, report, (1920, 340)), (0, 540))
        canvas.save(comp / f"c_{i:05d}.jpg", quality=90)
        n += 1
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(comp / "c_%05d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", a.out]
    r = subprocess.run(cmd)
    print(f"[g1video] {n} frames -> {a.out} rc={r.returncode}")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
