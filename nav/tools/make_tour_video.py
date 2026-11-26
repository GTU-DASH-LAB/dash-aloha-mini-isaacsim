"""One continuous video of a long-horizon tour (nav/tools/run_tour.py): before each step a
title card with the human's instruction, then the step in real time -- menu (chosen path
outlined) | third-person view, the model's description and choice underneath, and the
instruction in force on a banner at the top.

    python3 nav/tools/make_tour_video.py nav/results/tours/<stamp>_hospital_tour.json <out.mp4>
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_slide_video as R       # noqa: E402  (frames(), layout constants)
M = R.M

BANNER = 64
CW, CH = R.CW, R.CH + BANNER
NAVY, RED, PINK = (15, 37, 80), (200, 16, 46), (235, 163, 173)


def title_card(k, n, instruction, result):
    im = Image.new("RGB", (CW, CH), NAVY)
    d = ImageDraw.Draw(im)
    d.text((80, 150), f"STEP {k} OF {n}", fill=PINK, font=M._font(M.FONT_B, 34))
    d.text((80, 215), "Human (correction):" if result.get("correction") else "Human:",
           fill=(255, 190, 120) if result.get("correction") else (201, 211, 230),
           font=M._font(M.FONT, 34))
    y = 270
    for line in M._wrap(f"“{instruction}”", M._font(M.FONT_B, 48), CW - 160)[:4]:
        d.text((80, y), line, fill=(255, 255, 255), font=M._font(M.FONT_B, 48))
        y += 62
    d.text((80, CH - 110), "The robot did not arrive; the human gives a new instruction from where it stopped."
           if result.get("correction") else "The robot continues from where the previous step ended.",
           fill=(142, 160, 196), font=M._font(M.FONT, 26))
    return im


def banner(frame, k, n, instruction):
    out = Image.new("RGB", (CW, CH), (15, 17, 24))
    out.paste(frame, (0, BANNER))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, CW, BANNER], fill=NAVY)
    d.text((20, BANNER // 2), f"STEP {k}/{n}", fill=PINK, font=M._font(M.FONT_B, 24), anchor="lm")
    f = M._font(M.FONT_B, 24)
    text = f"“{instruction}”"
    while f.getlength(text) > CW - 170 and len(text) > 20:
        text = text[:-4] + "…”"
    d.text((150, BANNER // 2), text, fill=(255, 255, 255), font=f, anchor="lm")
    return out


SPEED = 3.0      # driving plays this many times faster; cards and holds are stretched to match


def frames(tour):
    steps = tour["steps"]
    n = len({s["step"] for s in steps})       # corrections are extra rows of the same step
    for s in steps:
        card = title_card(s["step"], n, s["instruction"], s)
        for _ in range(int(105 * SPEED)):   # 3.5 s on screen at any SPEED
            yield card
        rd = Path(s["record_dir"]) if s.get("record_dir") else None
        if rd is None or not (rd / "decisions.jsonl").is_file():
            continue
        last = None
        # A long straight drive (step 5's hallway, 200 s) plays twice as fast again, so the
        # demo stays near two minutes; the banner says which speed is showing.
        skip = 2 if s.get("elapsed_s", 0) > 100 else 1
        for idx, fr in enumerate(R.frames(rd)):
            if idx % skip:
                continue
            last = banner(fr, s["step"], n, s["instruction"])
            ImageDraw.Draw(last).text((CW - 16, BANNER - 8), f"video {SPEED * skip:g}×",
                                      fill=(142, 160, 196), font=M._font(M.FONT, 16), anchor="rs")
            yield last
        if last is not None:                # hold the arrival for 1.5 s
            d = ImageDraw.Draw(last)
            tag = "ARRIVED" if s["success"] else f"STOPPED {s['final_distance_m']} m SHORT"
            col = (60, 200, 110) if s["success"] else (235, 140, 60)
            d.rectangle([CW - 330, BANNER + 14, CW - 14, BANNER + 64], fill=(15, 17, 24))
            d.text((CW - 24, BANNER + 39), tag, fill=col, font=M._font(M.FONT_B, 28), anchor="rm")
            for _ in range(int(60 * SPEED)):   # 2 s on screen
                yield last


if __name__ == "__main__":
    if "--attention" in sys.argv:          # menus overlaid by tools/attention_for_run.py
        sys.argv.remove("--attention")
        R.OVERLAY = "attention"
    tour = json.loads(Path(sys.argv[1]).read_text())
    ok = M._encode(frames(tour), (CW, CH), 30.0 * SPEED, Path(sys.argv[2]))
    print("ok" if ok else "FAILED", sys.argv[2])
