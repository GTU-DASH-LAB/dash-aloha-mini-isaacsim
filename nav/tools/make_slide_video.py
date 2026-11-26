"""Slide-sized QVLA run video: menu (chosen path outlined) | chase view, and underneath, in large
text, what the model said it sees and the path it chose. Real time, one frame per chase frame.

    python3 nav/tools/make_slide_video.py <run_dir> <out.mp4>

Built on make_run_video.py (decision_panel draws the chosen arc; _encode pipes to
ffmpeg); only the layout differs: 1280x720, text sized for a video shown at half a slide.
"""
import json
import sys
from pathlib import Path

NAV = Path("/home/gtu-dsa/robotics/dash-aloha-mini-isaacsim/nav")
sys.path.insert(0, str(NAV / "tools"))
sys.path.insert(0, str(NAV))
import make_run_video as M            # noqa: E402
from arc_menu import _PALETTE, make_arcs, direction_word   # noqa: E402
from PIL import Image, ImageDraw      # noqa: E402

CW, CH = 1280, 720
# Subfolder of a run dir holding alternative menu images with the same names (e.g.
# "attention", written by tools/attention_for_run.py); None = the menus as recorded.
OVERLAY: str | None = None
OVERLAY_LABEL = "where the model looked (attention, 16 attention layers)"
PW, PH = 640, 360
FONT, FONT_B = M.FONT, M.FONT_B


def strip(rec, arcs, t):
    im = Image.new("RGB", (CW, CH - PH), (15, 17, 24))
    d = ImageDraw.Draw(im)
    lab = M._font(FONT_B, 22)
    body = M._font(FONT, 30)
    y = 22
    d.text((24, y + 4), "SEES", fill=(140, 150, 175), font=lab)
    for line in M._wrap(str(rec.get("free_space", "")), body, CW - 150 - 24)[:4]:
        d.text((150, y), line, fill=(235, 238, 245), font=body)
        y += 40
    y = max(y + 14, 200)
    d.text((24, y + 6), "CHOSE", fill=(140, 150, 175), font=lab)
    if rec.get("stop"):
        txt, col = "STOP", (235, 90, 90)
    elif rec.get("kappa") is not None:
        k = float(rec["kappa"])
        i = min(range(len(arcs)), key=lambda j: abs(arcs[j].kappa - k))
        col = tuple(_PALETTE[i % len(_PALETTE)])
        sp = rec.get("speed_mps")
        txt = (f"path {rec.get('choice')}: {direction_word(k)}"
               + (f"  ·  {float(sp):.2f} m/s" if sp is not None else ""))
    else:
        txt, col = "no readable answer, previous plan kept", (200, 200, 200)
    d.text((150, y), txt, fill=col, font=M._font(FONT_B, 34))
    d.text((CW - 24, CH - PH - 18), f"t = {t:5.1f} s", fill=(120, 128, 150),
           font=M._font(FONT, 22), anchor="rs")
    return im


def frames(run_dir: Path):
    recs = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    # A step that CONTINUES a tour (run_episode(continue_from_here=True)) can open with a
    # generation that started in the previous step and landed after the reset: it carries the
    # previous step's (large) step number. It is not this step's decision; drop it, or every
    # frame before that number is skipped.
    while len(recs) > 1 and int(recs[0].get("step") or 0) > int(recs[1].get("step") or 0):
        recs.pop(0)
    arcs = make_arcs()
    steps = [int(r.get("step") or 0) for r in recs]
    chases = sorted(run_dir.glob("chase_*.jpg"))
    cached, top_menu, bottom = -1, None, None
    for path in chases:
        s = int(path.stem.split("_")[1])
        i = -1
        for j, ds in enumerate(steps):
            if ds <= s:
                i = j
            else:
                break
        if i < 0:
            continue
        if i != cached:
            menu = run_dir / recs[i]["menu"]
            if OVERLAY and (run_dir / OVERLAY / recs[i]["menu"]).is_file():
                menu = run_dir / OVERLAY / recs[i]["menu"]
            if not menu.is_file():
                continue
            top_menu = M.decision_panel(recs[i], menu, arcs, i, len(recs)).resize((PW, PH), Image.LANCZOS)
            cached = i
        canvas = Image.new("RGB", (CW, CH), (15, 17, 24))
        canvas.paste(top_menu, (0, 0))
        try:
            canvas.paste(Image.open(path).convert("RGB").resize((PW, PH), Image.LANCZOS), (PW, 0))
        except Exception:
            pass
        d = ImageDraw.Draw(canvas)
        d.rectangle([PW - 2, 0, PW + 1, PH], fill=(15, 17, 24))
        f = M._font(FONT_B, 17)
        for x, t, c in ((10, OVERLAY_LABEL if OVERLAY else "what the model sees", (150, 200, 255)),
                        (PW + 10, "camera behind the robot (not shown to the model)", (255, 190, 120))):
            d.rectangle([x - 4, PH - 30, x + f.getlength(t) + 6, PH - 6], fill=(15, 17, 24))
            d.text((x, PH - 10), t, fill=c, font=f, anchor="ls")
        canvas.paste(strip(recs[i], arcs, s / 60.0), (0, PH))
        yield canvas


if __name__ == "__main__":
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    ok = M._encode(frames(run), (CW, CH), 30.0, out)
    print("ok" if ok else "FAILED", out)
