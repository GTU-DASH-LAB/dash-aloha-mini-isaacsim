#!/usr/bin/env python3
"""Score the camera-ready campaign (nav/tools/run_camera_ready.sh) for the MIDI 2026 paper.

    python3 nav/tools/camera_ready_table.py                 # prefix `cr`
    python3 nav/tools/camera_ready_table.py --prefix smoke  # the smoke test
    python3 nav/tools/camera_ready_table.py --json out.json # also write every row

Runs are selected by the tag the campaign wrote into every result (config.env.QVLA_RUN_TAG,
`[medium_]<prefix>_<arm>_r<round>`), never by date or by "newest file". Scoring is
summarize_runs.score(), so success, SPL, closest approach and the blown-up test are the
same numbers every other table in this repo uses.

Reported per arm: success over the 13 indoor episodes per round (mean, s.d., range), SPL,
mean closest approach, guard interventions per ladder, the DN-11 subset, and the episodes
on which the contact layer recorded a DynaNav physical collision. Arms are compared per
episode: each episode's success rate over the rounds, paired across arms. The p-value is an
exact sign-flip test over the 2^13 sign patterns of the per-episode differences, and the
95% interval is a bootstrap over episodes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))
from summarize_runs import load_config, score  # noqa: E402

TAG_RE = re.compile(r"(?:^|_)(?P<prefix>[a-z0-9]+)_(?P<arm>qvla|ticvla|straight)_r(?P<r>\d+)$")
ARM_ORDER = ["ticvla", "qvla", "straight"]


def collect(prefix: str) -> dict[str, dict[int, dict[str, dict]]]:
    """arm -> round -> episode -> scored row. A rerun of the same episode in the same
    round (a crash retried by hand) keeps the LAST file, and says so."""
    cfg = load_config()
    out: dict = defaultdict(lambda: defaultdict(dict))
    for path in sorted((REPO / "nav" / "results").glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tag = ((d.get("config") or {}).get("env") or {}).get("QVLA_RUN_TAG", "")
        m = TAG_RE.search(tag or "")
        if not m or m["prefix"] != prefix:
            continue
        row = score(path, cfg)
        if row is None:
            continue
        row["sha"] = ((d.get("config") or {}).get("git") or {}).get("sha", "?")
        arm, r, ep = m["arm"], int(m["r"]), row["episode"]
        if ep in out[arm][r]:
            print(f"   note: {arm} r{r} {ep} has two results; keeping {path.name}",
                  file=sys.stderr)
        out[arm][r][ep] = row
    return out


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip test of mean(diffs) == 0."""
    obs = abs(sum(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        hits += abs(sum(s * x for s, x in zip(signs, diffs))) >= obs - 1e-12
    return hits / total


def boot_ci(diffs: list[float], n: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def msd(xs: list[float]) -> str:
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="cr")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    data = collect(a.prefix)
    if not data:
        sys.exit(f"no results tagged {a.prefix}_<arm>_r<n>")
    cfg = load_config()
    indoor = [e for e, s in cfg.items() if e in
              {ep for arm in data.values() for rnd in arm.values() for ep in rnd}]
    dn11 = sorted(e for e in indoor if (cfg.get(e) or {}).get("dynanav_spl") is not None)

    print(f"== camera-ready campaign `{a.prefix}`")
    shas = {row["sha"] for arm in data.values() for rnd in arm.values() for row in rnd.values()}
    print(f"   commits: {sorted(shas)}   episodes seen: {len(indoor)}   DN-11 subset: {len(dn11)}")
    print()
    hdr = (f"{'arm':9} {'rounds':>6} {'SR/ep':>12} {'range':>7} {'SPL':>12} "
           f"{'closest m':>12} {'guard':>14} {'DN-11 SR':>12} {'DN-11 SPL':>11} "
           f"{'phys.coll':>9} {'blown':>5}")
    print(hdr)
    per_ep: dict[str, dict[str, float]] = {}
    for arm in [x for x in ARM_ORDER if x in data] + [x for x in data if x not in ARM_ORDER]:
        rounds = data[arm]
        sr, spl, closest, guard, dsr, dspl, coll, blown = [], [], [], [], [], [], [], 0
        for r, eps in sorted(rounds.items()):
            rows = list(eps.values())
            sr.append(sum(x["success"] for x in rows))
            spl.append(st.mean(x["spl"] for x in rows))
            ok = [x["closest_m"] for x in rows if not x["blown_up"]]
            closest.append(st.mean(ok) if ok else float("nan"))
            guard.append(sum(x["guard"] for x in rows))
            d = [eps[e] for e in dn11 if e in eps]
            dsr.append(sum(x["success"] for x in d))
            dspl.append(st.mean(x["spl"] for x in d) if d else float("nan"))
            coll.append(sum(bool(x["physical_collision"]) for x in rows))
            blown += sum(x["blown_up"] for x in rows)
        n_ep = max(len(eps) for eps in rounds.values())
        print(f"{arm:9} {len(rounds):>6} {msd(sr):>9}/{n_ep:<2} {min(sr)}-{max(sr):<5} "
              f"{msd(spl):>12} {msd(closest):>12} {msd(guard):>14} {msd(dsr):>12} "
              f"{msd(dspl):>11} {msd(coll):>9} {blown:>5}")
        per_ep[arm] = {}
        for e in indoor:
            got = [float(rounds[r][e]["success"]) for r in rounds if e in rounds[r]]
            per_ep[arm][e] = st.mean(got) if got else float("nan")
    print()

    print("per-episode success rate over rounds:")
    arms = [x for x in ARM_ORDER if x in per_ep]
    print(f"   {'episode':28} " + " ".join(f"{x:>8}" for x in arms))
    for e in indoor:
        print(f"   {e:28} " + " ".join(f"{per_ep[x].get(e, float('nan')):8.2f}" for x in arms))
    print()

    for x, y in [("qvla", "ticvla"), ("qvla", "straight"), ("ticvla", "straight")]:
        if x in per_ep and y in per_ep:
            both = [e for e in indoor if per_ep[x][e] == per_ep[x][e]
                    and per_ep[y][e] == per_ep[y][e]]       # an episode missing in either arm is left out
            diffs = [per_ep[x][e] - per_ep[y][e] for e in both]
            lo, hi = boot_ci(diffs)
            print(f"{x} - {y}: mean per-episode success difference {st.mean(diffs):+.3f} "
                  f"over {len(both)} episodes, 95% bootstrap CI "
                  f"[{lo:+.3f}, {hi:+.3f}], exact sign-flip p = {sign_flip_p(diffs):.4f}, "
                  f"episodes better/worse/tied {sum(d > 0 for d in diffs)}/"
                  f"{sum(d < 0 for d in diffs)}/{sum(d == 0 for d in diffs)}")

    if a.json:
        a.json.write_text(json.dumps({arm: {str(r): eps for r, eps in rounds.items()}
                                      for arm, rounds in data.items()}, indent=1,
                                     default=str))
        print(f"\nrows written to {a.json}")


if __name__ == "__main__":
    main()
