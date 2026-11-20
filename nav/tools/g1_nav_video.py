"""Video of one G1 navigation run (nav/sim/run_navigation_g1.py): what the G1's head camera showed the
policy with the menu drawn on it, next to the third-person chase view, in real time.

    python3 nav/tools/g1_nav_video.py --result <results>/<stamp>_<episode>_g1pursuit.json [--out x.mp4]

The run directory (menu_<step>.jpg + decisions.jsonl from the server or the stub, chase_<step>.jpg
from the runner, all keyed on the runner's step) is read from the result's g1.record_dir. One video
frame per chase frame, at 60 / chase_every fps (real time); the menu panel holds the newest decision
at or before that step, which is what the robot was driving on. AS RENDERED: the only pixel
operations are a Lanczos resize into the panels; the text sits in a strip below them.

Layout 1920 x 760: [ chase 960x540 ][ head camera + menu 960x540 ] over a 220 px text strip (episode,
instruction, sim time, distance to goal, the decision, the live camera pose it was drawn with).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL = (960, 540)
STRIP_H = 220
STEP_DT = 1.0 / 60.0


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _step_of(name: str) -> int | None:
    m = re.search(r"_(\d{8})\.jpg$", name)
    return int(m.group(1)) if m else None


def _decision_text(rec: dict) -> str:
    if rec is None:
        return "no decision yet"
    if rec.get("stop"):
        what = "STOP"
    elif rec.get("kind") == "heading" or rec.get("rel_deg") is not None:
        what = f"heading {rec.get('rel_deg'):+.0f} deg off the nose"
    elif rec.get("kappa") is not None:
        k = float(rec["kappa"])
        what = f"arc kappa {k:+.2f} ({'straight' if abs(k) < 1e-6 else 'left' if k > 0 else 'right'})"
    else:
        what = "unreadable"
    if rec.get("choice") is not None:
        what = f"label {rec['choice']}: " + what
    return what


def build(result_path: Path, out: Path | None, crf: int = 20) -> Path:
    d = json.loads(result_path.read_text())
    g = d.get("g1") or {}
    run_dir = Path(g.get("record_dir") or "")
    if not run_dir.is_dir():
        raise SystemExit(f"no record dir in {result_path} (g1.record_dir = {g.get('record_dir')!r})")
    every = int(g.get("chase_every_steps") or 2)
    fps = 1.0 / (every * STEP_DT)
    recs = []
    dec_file = run_dir / "decisions.jsonl"
    if dec_file.is_file():
        for line in dec_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                s = r.get("step")
                if s is None and r.get("menu"):
                    s = _step_of(r["menu"])
                r["_step"] = s if s is not None else 0
                recs.append(r)
    recs.sort(key=lambda r: r["_step"])
    chases = sorted((p for p in run_dir.glob("chase_*.jpg")), key=lambda p: _step_of(p.name) or 0)
    if not chases:
        raise SystemExit(f"no chase frames in {run_dir}")
    goal = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
        from episode import all_episode_specs

        goal = (all_episode_specs().get(d["episode"]) or {}).get("goal")
    except Exception:  # noqa: BLE001
        pass
    trace = d.get("trace") or []
    out = out or (result_path.parent / "videos" / (result_path.stem + ".mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    f_big, f_small = _font(26), _font(22)
    ri, menu_img, menu_for = -1, None, None
    verdict = ("SUCCESS" if d.get("success") else (d.get("abort_reason") or
               ("TIMEOUT" if d.get("timed_out") else "FAILED")).upper())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = 0
        for cp in chases:
            step = _step_of(cp.name) or 0
            while ri + 1 < len(recs) and recs[ri + 1]["_step"] <= step:
                ri += 1
            rec = recs[ri] if ri >= 0 else None
            if rec is not None and rec.get("menu") and menu_for != rec.get("menu"):
                mp = run_dir / rec["menu"]
                if mp.is_file():
                    menu_img = Image.open(mp).convert("RGB").resize(PANEL, Image.LANCZOS)
                    menu_for = rec["menu"]
            canvas = Image.new("RGB", (2 * PANEL[0], PANEL[1] + STRIP_H), (0, 0, 0))
            canvas.paste(Image.open(cp).convert("RGB").resize(PANEL, Image.LANCZOS), (0, 0))
            if menu_img is not None:
                canvas.paste(menu_img, (PANEL[0], 0))
            dr = ImageDraw.Draw(canvas)
            t = step * STEP_DT
            dist = ""
            if goal and trace:
                k = min(len(trace) - 1, step // 30)
                dist = f"   distance to goal {math.dist(trace[k][:2], goal[:2]):5.2f} m"
            cam = (rec or {}).get("camera") or {}
            y0 = PANEL[1] + 12
            dr.text((16, y0), f"Unitree G1 (walking policy) -- {d['episode']} -- {d.get('policy', '')} -- "
                    f"camera {d.get('camera', '')} -- result {verdict}", fill=(255, 255, 255), font=f_big)
            dr.text((16, y0 + 40), f"\"{d.get('instruction', '')[:150]}\"", fill=(200, 200, 200), font=f_small)
            dr.text((16, y0 + 80), f"t = {t:6.2f} s sim{dist}   decision: {_decision_text(rec)}",
                    fill=(255, 230, 120), font=f_big)
            if cam:
                dr.text((16, y0 + 120), f"menu drawn with the live head pose: h {cam.get('cam_h', 0):.3f} m, pitch "
                        f"{cam.get('pitch_deg', 0):.1f} deg, roll {cam.get('roll_deg', 0):+.1f}, yaw vs pelvis "
                        f"{cam.get('yaw_deg', 0):+.1f}", fill=(180, 220, 255), font=f_small)
            fs = (rec or {}).get("free_space")
            if fs:
                dr.text((16, y0 + 158), str(fs)[:170], fill=(200, 255, 200), font=f_small)
            dr.text((16, PANEL[1] - 36), "chase view (not seen by the policy)", fill=(255, 255, 255), font=f_small)
            dr.text((PANEL[0] + 16, PANEL[1] - 36), "G1 head camera (what the policy sees) + menu",
                    fill=(255, 255, 255), font=f_small)
            canvas.save(tmp / f"{n:06d}.jpg", quality=92)
            n += 1
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.4f}", "-i", str(tmp / "%06d.jpg"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg failed: {r.stderr[-800:]}")
    print(f"{n} frames at {fps:.1f} fps ({n / fps:.1f} s), {len(recs)} decisions -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()
    build(a.result, a.out, a.crf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
