#!/usr/bin/env python3
"""Attention map for every decision of a recorded arc-menu run, as overlaid menu images.

    nav/tools/attention_for_run.py <tour json | run dir> [...] [--port 8766] [--layers all]

For each decision in each run dir's decisions.jsonl, the select call is rebuilt exactly as
`server_qwen.menu_plan` built it -- same menu image, same description, history line,
odometry sentence, recovery note, instruction, system prompt and prefill -- and sent to
/explain with the answer the robot actually gave as the target. Written next to the menus:

    <run dir>/attention/att_<step>.json     the /explain result (16 layers x 180 tokens)
    <run dir>/attention/menu_<step>.jpg     the menu with the attention heat drawn over it

`make_tour_video.py --attention` then renders the tour with these in place of the menus.
The reconstruction is checked on every decision: the model's most likely first token must be
the recorded label. A mismatch means the prompt was not rebuilt faithfully, and is counted.

Attention here is the answer position's attention over the 180 image tokens, averaged over
heads and over the chosen full-attention layers (16 of the model's 64 layers have one; the
other 48 are linear attention and have no map). Normalised per decision: the colours say
where THIS decision looked most, not how much it looked at the image overall -- that
number, the image share, is in the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))

from arc_menu import (  # noqa: E402
    SELECT_AFTER_BALK, SELECT_AFTER_RECOVERY, SELECT_PREFILL, SELECT_PREFILL_SPEED,
    SPEED_CHOICE_FROM_M, select_system, stop_label,
)

STOP = stop_label(7, False)
LAYERS = {"all": slice(None), "early": slice(0, 5), "mid": slice(5, 11), "late": slice(11, None)}


def explain(port: int, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/explain", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def rebuild(rec: dict) -> tuple[str, str, str]:
    """(system, user, prefill) exactly as menu_plan composed them for this decision."""
    kind = rec.get("kind") or ""
    ask_speed = rec.get("target_m") is not None and rec["target_m"] <= SPEED_CHOICE_FROM_M
    system = select_system(STOP, stop_allowed=not kind, speed_choice=ask_speed)
    hist, odo = rec.get("history") or "", rec.get("odometry_note") or ""
    user = (f"What the robot can see ahead of it: {rec['free_space']}\n\n"
            + (f"The direction the task needs to go next -- HEADING: {rec['heading']}\n\n"
               if rec.get("heading") else "")
            + (f"{hist}\n\n" if hist else "")
            + (f"{odo}\n\n" if odo else "")
            + {"wedge": f"{SELECT_AFTER_RECOVERY}\n\n", "balk": f"{SELECT_AFTER_BALK}\n\n"}.get(kind, "")
            + f"Navigation instruction: {rec['instruction']}\n\n"
            + ("Which numbered path do you take, and how fast?" if ask_speed
               else "Which numbered path do you take?"))
    return system, user, (SELECT_PREFILL_SPEED if ask_speed else SELECT_PREFILL)


def overlay(menu: Path, att: np.ndarray, grid: list[int], out: Path) -> None:
    import matplotlib
    from PIL import Image
    img = Image.open(menu).convert("RGB")
    W, H = img.size
    gh, gw = grid
    m = att.reshape(gh, gw)
    m = m / (m.max() or 1.0)
    v = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                   float) / 255
    col = matplotlib.colormaps["inferno"](v)[..., :3]
    a = (v ** 0.9)[..., None] * 0.85                     # low attention leaves the image as is
    base = np.asarray(img, float) / 255
    Image.fromarray(np.clip((base * (1 - a) * 0.75 + col * a) * 255, 0, 255).astype(np.uint8)
                    ).save(out, quality=92)


def run_dirs(args: list[str]) -> list[Path]:
    out = []
    for a in args:
        p = Path(a)
        if p.suffix == ".json":
            out += [Path(s["record_dir"]) for s in json.loads(p.read_text())["steps"]
                    if s.get("record_dir")]
        else:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--layers", default="all", choices=list(LAYERS))
    ap.add_argument("--redraw", action="store_true",
                    help="only redraw overlays from the saved att_*.json (no server)")
    a = ap.parse_args()
    total = match = 0
    for rd in run_dirs(a.runs):
        recs = [json.loads(l) for l in (rd / "decisions.jsonl").read_text().splitlines() if l.strip()]
        while len(recs) > 1 and int(recs[0].get("step") or 0) > int(recs[1].get("step") or 0):
            recs.pop(0)                  # a previous tour step's generation (make_slide_video)
        od = rd / "attention"
        od.mkdir(exist_ok=True)
        for rec in recs:
            if rec.get("choice") is None or not (rd / rec["menu"]).is_file():
                continue
            saved = od / f"att_{int(rec['step']):08d}.json"
            if a.redraw:
                if saved.is_file():
                    r = json.loads(saved.read_text())
                    overlay(rd / rec["menu"], np.asarray(r["attention"])[LAYERS[a.layers]].mean(0),
                            r["grid"], od / rec["menu"])
                continue
            system, user, prefill = rebuild(rec)
            r = explain(a.port, {"image_paths": [str((rd / rec["menu"]).resolve())],
                                 "system": system, "user": user, "prefill": prefill,
                                 "target": str(rec["choice"]), "occlusion": 0, "top_k": 5})
            top = r["alternatives"][0]["token"].strip()
            total += 1
            match += top == str(rec["choice"])
            r["step"], r["menu"] = rec["step"], rec["menu"]
            (od / f"att_{int(rec['step']):08d}.json").write_text(json.dumps(r))
            att = np.asarray(r["attention"])[LAYERS[a.layers]].mean(0)
            overlay(rd / rec["menu"], att, r["grid"], od / rec["menu"])
            print(f"{rd.name} step {rec['step']:5d} choice {rec['choice']} p={np.exp(r['logp']):.2f} "
                  f"top={top!r} image share {np.mean(r['image_mass']):.2f} {r['seconds']}s", flush=True)
    print(f"\n{total} decisions explained; the model's top answer matched the recorded one on "
          f"{match} ({100 * match / max(total, 1):.0f}%)")


if __name__ == "__main__":
    main()
