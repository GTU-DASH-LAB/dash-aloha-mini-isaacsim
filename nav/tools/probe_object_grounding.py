#!/usr/bin/env python3
"""Does the arc-menu model find the object the instruction names? Two measurements.

    nav/tools/probe_object_grounding.py <experiment.json> <out_dir> [--port 8766]

experiment.json is explain_attention.py's format plus "objects": {"staircase": "B", ...},
mapping each object word to the prompt (key of "prompts") that names it.

1. WORD ATTENTION. /explain with `query_text` = the object word: the attention FROM that
   word's tokens in the instruction TO the 180 image tokens, per full-attention layer. The
   answer-position map (explain_attention.py) shows where the model reads when it WRITES the
   number -- mostly the number badges. Multimodal LLMs are reported to move object
   information from image tokens into the question's own tokens in the lower and middle
   layers, and the last position then reads from text (Zhang et al. 2024, "Cross-modal
   information flow in multimodal large language models"). If that holds here, the object
   lights up from the WORD, not from the answer.
2. POINTING. /raw asks for the object's bounding box; it is drawn on the frame. If the model
   can box the target, a grounding step can pick the path toward it geometrically.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))
from arc_menu import SELECT_PREFILL, select_system, stop_label  # noqa: E402

STOP = stop_label(7, False)
GROUPS = {"early (layers 3-19)": slice(0, 5), "middle (23-43)": slice(5, 11),
          "late (47-63)": slice(11, 16)}


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ex = json.loads(Path(a.experiment).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(ex["image"]).convert("RGB")
    W, H = img.size
    base = np.asarray(img, float) / 255
    system = select_system(STOP, stop_allowed=True, speed_choice=False)

    def heat(ax, m, gh, gw, title):
        m = np.clip(m, 0, None).reshape(gh, gw)
        v = np.asarray(Image.fromarray((m / (m.max() or 1) * 255).astype(np.uint8))
                       .resize((W, H), Image.BICUBIC), float) / 255
        col = matplotlib.colormaps["inferno"](v)[..., :3]
        al = (v ** 0.9)[..., None] * 0.85
        ax.imshow(base * (1 - al) * 0.75 + col * al)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    rows = list(ex["objects"].items())
    fig, axes = plt.subplots(len(rows), 5, figsize=(24, 3.3 * len(rows) + 0.4), squeeze=False)
    report = {}
    for i, (obj, key) in enumerate(rows):
        instr = ex["prompts"][key]
        user = (f"What the robot can see ahead of it: {ex['description']}\n\n"
                f"Navigation instruction: {instr}\n\nWhich numbered path do you take?")
        body = {"image_paths": [ex["image"]], "system": system, "user": user,
                "prefill": SELECT_PREFILL, "occlusion": 0}
        ans = post(a.port, "/explain", body)
        word = post(a.port, "/explain", {**body, "query_text": obj})
        gh, gw = word["grid"]
        heat(axes[i][0], np.asarray(ans["attention"]).mean(0), gh, gw,
             f"“{instr}”\nfrom the ANSWER token (all layers) → {ans['target']!r}")
        wa = np.asarray(word["attention"])
        for j, (g, sl) in enumerate(GROUPS.items()):
            heat(axes[i][j + 1], wa[sl].mean(0), gh, gw,
                 f"from the WORD “{obj}” {word['query_tokens']}\n{g}")
        # Pointing: ask for a box. Qwen-VL boxes are reported on a 0-1000 grid; absolute
        # pixels of the 576x320 model input are the older convention, so both are checked.
        raw = post(a.port, "/raw", {"image_paths": [ex["image"]], "system": "",
                                    "user": f"Find the {obj} in this image. Reply only with JSON: "
                                            f'{{"bbox_2d": [x1, y1, x2, y2]}}',
                                    "max_new_tokens": 60})
        # The numbers INSIDE the list after "bbox_2d" -- the key itself contains a digit.
        mlist = re.search(r"bbox_2d\D*\[([^\]]*)\]", raw["text"])
        nums = ([float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", mlist.group(1))][:4]
                if mlist else [])
        ax = axes[i][4]
        ax.imshow(img)
        ax.axis("off")
        box = None
        if len(nums) == 4:
            if max(nums) <= 1000:
                box = [nums[0] / 1000 * W, nums[1] / 1000 * H, nums[2] / 1000 * W, nums[3] / 1000 * H]
            mi = word["model_input"]
            for b, c, lab in ((box, "lime", "0-1000"),
                              ([nums[0] * W / mi[0], nums[1] * H / mi[1], nums[2] * W / mi[0],
                                nums[3] * H / mi[1]], "cyan", "pixels")):
                if b and b[2] <= W * 1.05 and b[3] <= H * 1.05:
                    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                               ec=c, lw=2.5, label=lab))
            ax.legend(loc="lower right", fontsize=7)
        ax.set_title(f"asked to box the {obj}: {nums}", fontsize=9)
        report[obj] = {"answer": ans["target"], "answer_p": float(np.exp(ans["logp"])),
                       "query_tokens": word["query_tokens"], "box_reply": raw["text"],
                       "word_attention": word["attention"], "grid": word["grid"],
                       "answer_attention": ans["attention"]}
        print(f"{obj}: answer {ans['target']!r} p={np.exp(ans['logp']):.2f}; word tokens "
              f"{word['query_tokens']}; box reply {raw['text'].strip()!r}", flush=True)
    fig.tight_layout()
    fig.savefig(out / "grounding.png", dpi=72)
    (out / "grounding.json").write_text(json.dumps(report))
    print("figure:", out / "grounding.png")


if __name__ == "__main__":
    main()
