"""Put the survey's candidates and the 27B baseline in one table, and say what won.

Three columns of this table are three different kinds of win and they must not be
collapsed into one number: a candidate can win on **quality** (higher `mirror` or `open
side`), on **latency at parity** (same scores, materially faster), or on **footprint at
parity** (8 GiB instead of 28.75, which is the difference between needing two cards and
needing one). A single scalar ranking would hide two of the three.

Two reading rules are enforced rather than left to the reader:

  DO NOT AVERAGE THE COLUMNS.  `keep straight` at 100% with `open side` near chance is the
  signature of a model that always picks the centre arc -- a positional prior, not
  comprehension, and the baseline's own DIGIT row is exactly that. An average scores it
  well. So the summary reports the columns separately and flags the pattern by name.

  UNPARSED IS ITS OWN COLUMN.  A model that narrates instead of answering with a digit has
  not failed at navigation, it has failed at the format, and those have different fixes.
  Folding it into a low score would report the wrong one.

Usage:
    python3 nav/tools/compare_vlm_candidates.py
    python3 nav/tools/compare_vlm_candidates.py --results nav/results/vlm_survey --md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display order: the baseline first, then ascending size, so the table reads as "what it
# costs now" followed by "what each step down costs you".
ORDER = ["baseline-qwen27b", "smolvlm2-2.2b", "qwen3vl-4b", "gemma3-4b",
         "internvl3.5-4b", "qwen3vl-8b"]

# What the arc-menu policy actually runs. DIGIT is the cheap path used when the
# instruction carries the direction; SIDED is what "drive safely" costs. Reporting only
# one of them would answer half the question -- see nav/vlm_survey.md section 5.
VARIANTS = ["DIGIT", "SIDED"]

CHANCE = {"side": 0.43, "mirror": 0.0}


def load(results: Path, tag: str) -> dict | None:
    rep = results / f"{tag}__repair.json"
    sel = results / f"{tag}__selection.json"
    if not rep.is_file():
        return None
    row: dict = {"tag": tag, "repair": json.loads(rep.read_text())}
    if sel.is_file():
        row["selection"] = json.loads(sel.read_text())
    h = results / f"{tag}__health.json"
    if not h.is_file() and tag == "baseline-qwen27b":
        h = results / "baseline__health.json"
    if h.is_file():
        try:
            row["health"] = json.loads(h.read_text())
        except json.JSONDecodeError:
            pass
    return row


def stats(row: dict, variant: str) -> dict | None:
    return row["repair"].get(f"{variant}_stats")


def fmt_pct(x: float | None) -> str:
    return "  --" if x is None else f"{x * 100:3.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="nav/results/vlm_survey")
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    args = ap.parse_args()

    results = Path(args.results)
    rows = [r for r in (load(results, t) for t in ORDER) if r]
    if not rows:
        print(f"no results under {results}")
        return 1

    for variant in VARIANTS:
        print(f"\n{'=' * 92}\n{variant}   "
              f"(chance: open side {CHANCE['side'] * 100:.0f}%, mirror ~0%)\n{'=' * 92}")
        head = (f"{'model':18}{'params':>8}{'GiB':>7}{'avoid':>7}{'keep':>7}"
                f"{'side':>7}{'mirror':>8}{'latency':>10}{'unparsed':>10}")
        print(head)
        print("-" * 92)
        for r in rows:
            s = stats(r, variant)
            if not s:
                continue
            h = r.get("health", {})
            pb = h.get("params_b")
            gib = h.get("weights_gib")
            # The baseline's health payload is server_qwen's, which carries neither field
            # -- it is a different schema on the same route. Two servers matching on
            # routes is not two servers matching on schema; fill it from what is measured
            # elsewhere rather than printing a blank the reader has to look up.
            if r["tag"] == "baseline-qwen27b":
                pb, gib = 27.0, 28.75
            print(f"{r['tag']:18}"
                  f"{(f'{pb:.1f}B' if pb else '--'):>8}"
                  f"{(f'{gib:.1f}' if gib else '--'):>7}"
                  f"{fmt_pct(s['avoid']):>7}{fmt_pct(s['keep']):>7}"
                  f"{fmt_pct(s['side']):>7}{fmt_pct(s['mirror']):>8}"
                  f"{s['lat']:9.2f}s{s['unparsed']:10d}")

    # --------------------------------------------------------------- selection probe
    print(f"\n{'=' * 92}\nprobe_arc_selection -- does the WORD reach the CURVE?"
          f"\n{'=' * 92}")
    print(f"{'model':18}{'left':>8}{'right':>8}{'straight':>10}{'L-R kappa':>11}"
          f"{'unparsed':>10}{'latency':>10}")
    print("-" * 92)
    for r in rows:
        sel = r.get("selection")
        if not sel:
            continue
        by = sel.get("summary", {})
        cells = [by.get(k, {}).get("side_ok") for k in ("left", "right", "straight")]
        # left-minus-right mean kappa is the control the percentages cannot give: it says
        # how far the two instructions pull the choice apart on a menu spanning 1.20,
        # which distinguishes a model that separates them weakly from one that does not.
        lk, rk = by.get("left", {}).get("mean_kappa"), by.get("right", {}).get("mean_kappa")
        sep = (lk - rk) if isinstance(lk, float) and isinstance(rk, float) else None
        bad = sum(by.get(k, {}).get("unparsed", 0) for k in by)
        lats = sorted(t["latency_s"] for t in sel.get("trials", []) if "latency_s" in t)
        lat = lats[len(lats) // 2] if lats else None
        print(f"{r['tag']:18}"
              + "".join(f"{fmt_pct(c):>8}" for c in cells[:2])
              + f"{fmt_pct(cells[2]):>10}"
              + (f"{sep:+11.3f}" if sep is not None else f"{'--':>11}")
              + f"{bad:10d}"
              + (f"{lat:9.2f}s" if lat is not None else f"{'--':>10}"))

    # ------------------------------------------------------------------- the verdict
    base = next((r for r in rows if r["tag"] == "baseline-qwen27b"), None)
    print(f"\n{'=' * 92}\nverdict\n{'=' * 92}")
    if not base:
        print("no baseline row -- nothing to beat. Re-run with the 27B up on :8766.")
        return 0

    b = stats(base, "SIDED")
    if not b:
        print("baseline has no SIDED row")
        return 0
    print(f"baseline SIDED: avoid {fmt_pct(b['avoid'])} keep {fmt_pct(b['keep'])} "
          f"side {fmt_pct(b['side'])} mirror {fmt_pct(b['mirror'])} "
          f"{b['lat']:.2f}s  28.75 GiB\n")

    for r in rows:
        if r["tag"] == base["tag"]:
            continue
        s = stats(r, "SIDED")
        if not s:
            print(f"  {r['tag']:18} no SIDED result")
            continue
        gib = (r.get("health") or {}).get("weights_gib") or 0
        # "Comparable" is deliberately generous at 5 points: this probe is 24-48 scenes
        # and three clean prior ladders of the SAME configuration scored 2/13, 8/13, 8/13.
        # A tighter threshold would report noise as a regression.
        quality = all(s[k] >= b[k] - 0.05 for k in ("avoid", "keep", "side", "mirror"))
        # A quality WIN has to survive the parity gate too. Without it, a model that
        # beats the baseline on mirror by 6 points while losing `keep` by 89 reads as
        # "QUALITY" -- which is the don't-average-the-columns error wearing a different
        # hat. SmolVLM2 is exactly that case, so the guard is not hypothetical.
        better = quality and any(s[k] > b[k] + 0.05 for k in ("side", "mirror"))
        faster = s["lat"] < b["lat"] * 0.8
        lighter = 0 < gib < 28.75 * 0.5

        wins = []
        if better:
            wins.append("QUALITY")
        if quality and faster:
            wins.append(f"LATENCY ({b['lat'] / max(s['lat'], 1e-6):.1f}x)")
        if quality and lighter:
            wins.append(f"FOOTPRINT ({28.75 / gib:.1f}x)")

        # Name the positional priors explicitly. They are the failures an averaged score
        # rewards, and there are TWO of them -- opposite answers, identical emptiness.
        # `keep` at 100% with `side` near chance is the centre-arc prior, and it is what
        # DIGIT does on the baseline itself. Its mirror image scores `avoid` near 100%
        # with `keep` near zero, which looks like two wins on this table and is a model
        # that swerves at everything. The open-corridor frames exist to catch it.
        notes = []
        if s["keep"] > 0.9 and s["side"] < CHANCE["side"] + 0.1:
            notes.append("centre-arc prior, not comprehension")
        if s["avoid"] > 0.8 and s["keep"] < 0.3:
            notes.append("edge-arc prior -- swerves at everything, incl. clear corridors")
        if s["unparsed"] > 0:
            notes.append(f"{s['unparsed']} unparsed -- format, not navigation")
        note = ("  <- " + "; ".join(notes)) if notes else ""

        if wins:
            verdict = " + ".join(wins)
        elif quality:
            verdict = "comparable"
        else:
            # Say WHICH columns fell, so "worse" is a reading rather than a label.
            lost = [f"{k} {(s[k] - b[k]) * 100:+.0f}pp"
                    for k in ("avoid", "keep", "side", "mirror") if s[k] < b[k] - 0.05]
            verdict = "worse (" + ", ".join(lost) + ")"
        print(f"  {r['tag']:18} {verdict}{note}")

    if args.md:
        print("\n<!-- markdown -->\n")
        print("| model | params | GiB | avoid wall | keep straight | open side | mirror "
              "| latency | unparsed |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            s = stats(r, "SIDED")
            if not s:
                continue
            h = r.get("health", {})
            pb = 27.0 if r["tag"] == "baseline-qwen27b" else h.get("params_b")
            gib = 28.75 if r["tag"] == "baseline-qwen27b" else h.get("weights_gib")
            print(f"| `{r['tag']}` | {pb or '--'} B | {gib or '--'} | "
                  f"{fmt_pct(s['avoid']).strip()} | {fmt_pct(s['keep']).strip()} | "
                  f"{fmt_pct(s['side']).strip()} | {fmt_pct(s['mirror']).strip()} | "
                  f"{s['lat']:.2f} s | {s['unparsed']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
