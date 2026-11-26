#!/usr/bin/env python3
"""Where does the arc-menu model look? Attention and occlusion maps through /explain.

    nav/tools/explain_attention.py <experiment.json> <out_dir> [--port 8766] [--occlusion 1]

experiment.json:
    {"image": "<menu jpg>",            # what the select call was shown (arcs drawn)
     "labels": [7, 2, 4, 6, 5, 1, 3],   # the decision's shuffle, kappa -0.60 .. +0.60
     "description": "...",             # the describe call's text, held FIXED across prompts
     "history": "",                    # the history line, if any
     "prompts": {"A": "Turn left and ...", "B": "..."}}

For every prompt the select call is rebuilt exactly as `menu_plan` builds it (same system
prompt, same prefill), the model's own answer is taken, and /explain returns:
  - attention of the answer position over the 18 x 10 image tokens in each of the 16
    full-attention layers (heads averaged), and how much of it lands on the image at all;
  - the probability of every label (the answer distribution);
  - with --occlusion k: the drop in log p(answer) when each k x k block is greyed out.
One figure per prompt plus a side-by-side sheet; the raw JSON is kept next to them.

Holding the description fixed is deliberate: it isolates what the INSTRUCTION does to the
select call. In the live loop the description would change with the instruction too.
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

from arc_menu import SELECT_PREFILL, direction_word, select_system, stop_label  # noqa: E402

KAPPAS = [-0.6, -0.35, -0.15, 0.0, 0.15, 0.35, 0.6]
STOP = stop_label(len(KAPPAS), False)


def explain(port: int, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/explain",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def label_word(tok: str, labels: list[int]) -> str:
    t = tok.strip()
    if not t.isdigit():
        return repr(tok)
    n = int(t)
    if n == STOP:
        return f"{n} STOP"
    if n in labels:
        return f"{n} {direction_word(KAPPAS[labels.index(n)])}"
    return f"{n} (not on menu)"


def overlay(ax, img, grid, values, title, cmap, vmax=None, signed=False):
    import matplotlib.pyplot as plt  # noqa: F401
    gh, gw = grid
    m = np.asarray(values, float).reshape(gh, gw)
    W, H = img.size
    ax.imshow(img)
    if signed:
        v = vmax or (np.abs(m).max() or 1.0)
        ax.imshow(m, cmap=cmap, alpha=0.55, extent=(0, W, H, 0), vmin=-v, vmax=v,
                  interpolation="bilinear")
    else:
        ax.imshow(m, cmap=cmap, alpha=0.55, extent=(0, W, H, 0), vmin=0,
                  vmax=vmax or m.max(), interpolation="bilinear")
    for r in range(gh + 1):
        ax.axhline(r * H / gh, color="white", lw=0.15, alpha=0.4)
    for c in range(gw + 1):
        ax.axvline(c * W / gw, color="white", lw=0.15, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--occlusion", type=int, default=1, help="block size in tokens, 0 = off")
    ap.add_argument("--layers", default="all", help="'all', 'early', 'mid', 'late'")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ex = json.loads(Path(a.experiment).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(ex["image"]).convert("RGB")
    system = select_system(STOP, stop_allowed=True, speed_choice=False)
    results = {}
    for name, instr in ex["prompts"].items():
        hist = ex.get("history") or ""
        extra = ex.get("extra", {}).get(name, "")
        user = (f"What the robot can see ahead of it: {ex['description']}\n\n"
                + (f"{hist}\n\n" if hist else "") + (f"{extra}\n\n" if extra else "")
                + f"Navigation instruction: {instr}\n\nWhich numbered path do you take?")
        r = explain(a.port, {"image_paths": [str(Path(ex["image"]).resolve())],
                             "system": system, "user": user, "prefill": SELECT_PREFILL,
                             "occlusion": a.occlusion, "top_k": 12})
        r["instruction"], r["user"] = instr, user
        results[name] = r
        (out / f"explain_{name}.json").write_text(json.dumps(r, indent=1))
        print(f"[{name}] {instr!r} -> {label_word(r['target'], ex['labels'])}  "
              f"p={np.exp(r['logp']):.3f}  image mass {np.mean(r['image_mass']):.3f}  "
              f"{r['seconds']} s", flush=True)

    n = len(results)
    rows = 3 if a.occlusion else 2
    fig, axes = plt.subplots(rows, n, figsize=(6.2 * n, 3.6 * rows + 0.6), squeeze=False)
    sel = {"all": slice(None), "early": slice(0, 5), "mid": slice(5, 11), "late": slice(11, None)}
    for j, (name, r) in enumerate(results.items()):
        att = np.mean(np.asarray(r["attention"])[sel[a.layers]], 0)
        overlay(axes[0][j], img, r["grid"], att,
                f"{name}: “{r['instruction'][:60]}”\nattention ({a.layers} full-attn layers)"
                f" → answer {label_word(r['target'], ex['labels'])}", "inferno")
        if a.occlusion:
            occ = r["occlusion"]
            drop = np.asarray(occ["drop"], float).reshape(occ["rows"], occ["cols"])
            drop = np.kron(drop, np.ones((occ["block"], occ["block"])))[:r["grid"][0], :r["grid"][1]]
            overlay(axes[1][j], img, r["grid"], drop.flatten(),
                    "occlusion: red = hiding this LOWERS p(answer)", "RdBu_r", signed=True)
        ax = axes[rows - 1][j]
        alts = [x for x in r["alternatives"] if x["token"].strip().isdigit()]
        ax.barh([label_word(x["token"], ex["labels"]) for x in alts][::-1],
                [x["p"] for x in alts][::-1], color="#c8102e")
        ax.set_xlim(0, 1)
        ax.set_title("answer probabilities", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sheet.png", dpi=110)
    print("sheet:", out / "sheet.png")


if __name__ == "__main__":
    main()
