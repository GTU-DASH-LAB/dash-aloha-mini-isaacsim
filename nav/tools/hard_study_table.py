"""Score the hard-episode study: five menus x three label seeds on the six unsolved runs.

    python3 nav/tools/hard_study_table.py            # the matrix and the scoreboard
    python3 nav/tools/hard_study_table.py --verbose  # + every run, one line each

WHAT THIS TABLE IS FOR, and it is not the same job as `sync_study_table.py`. That one
compares planning regimes across all thirteen episodes and has to be careful about which
columns may be compared at all. This one holds the regime fixed -- synchronous, 3 s,
medium, every arm -- and varies only the menu, on the six episodes that stayed broken.
Everything here is therefore comparable, and the only real question is whether a
difference is bigger than the seed.

WHY THE SEEDS ARE SHOWN INDIVIDUALLY rather than averaged into a rate. With three samples
a mean is a rounding of "one, two or three", and the pattern across seeds carries the part
that matters: `YYY` is a policy that solves the episode, `Y..` is a policy that solved it
once. Six arms of single runs already produced ten episodes out of thirteen that flipped
at least once, and reading those single runs as rates is what made the campaign look like
it was measuring configuration when it was measuring the label permutation.

THE REPRODUCIBILITY LINE at the bottom is the check that licenses everything above it.
Arm `ref` at seed 0 is the same configuration, same seed, same episodes as the archived
`p3.0_medium` condition. If it does not reproduce, the seed is not the only thing moving
between runs, and no comparison in this file -- or in the campaign before it -- separates
a policy change from run-to-run drift. It is printed whether it agrees or not.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "nav/results"
STUDY = RESULTS / "hard_study"

# Same import, same reason as in `sync_study_table.py`: `success` is recomputed from the
# trace against the threshold rather than trusted from the file, and a second copy of a
# scoring rule is how two tables of the same runs come to disagree.
sys.path.insert(0, str(REPO / "nav/sim"))
from summarize_runs import load_config, score  # noqa: E402

_CFG = load_config()

# The order arms are presented in, which is the order they answer questions in and not the
# order they finish in. Arms absent from disk are skipped rather than shown empty: a study
# stopped after two arms should read as two arms, not as five with three blanks that look
# like failures.
ARM_ORDER = ["ref", "all", "pivot15", "memory", "fine11"]
ARM_WHAT = {
    "ref":     "coarse 7, no turns, 1 frame",
    "all":     "fine 11, 15 deg turns, 2 frames",
    "pivot15": "coarse 7, 15 deg turns, 1 frame",
    "memory":  "coarse 7, no turns, 2 frames",
    "fine11":  "fine 11, no turns, 1 frame",
}
_DIR = re.compile(r"^(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)$")


def _runs(d: Path) -> dict[str, dict]:
    """Every scored run in one ladder directory, keyed by episode.

    Keyed rather than listed because a directory holds at most one run per episode and
    the caller always wants it by name. A directory with two files for one episode means
    the archive picked up a neighbour's result, so the newer one wins and that is stated
    rather than silently resolved.
    """
    out: dict[str, dict] = {}
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if f.name == "arm.json":
            continue
        r = score(f, _CFG)
        if r is None:
            continue
        raw = json.loads(f.read_text())
        out[r["episode"]] = r | {"pivots": raw.get("pivots"),
                                 "recoveries": raw.get("recoveries")}
    return out


def collect() -> tuple[dict[str, dict[int, dict]], list[str]]:
    """{arm: {seed: {episode: run}}}, plus the episode list in config order."""
    arms: dict[str, dict[int, dict]] = {}
    for d in sorted(STUDY.iterdir()) if STUDY.is_dir() else []:
        m = _DIR.match(d.name) if d.is_dir() else None
        if not m:
            continue                      # .superseded-*, comparison.txt, stray files
        runs = _runs(d)
        if runs:
            arms.setdefault(m["arm"], {})[int(m["seed"])] = runs
    seen = {e for seeds in arms.values() for r in seeds.values() for e in r}
    order = [e for e in _CFG if e in seen] + sorted(seen - set(_CFG))
    return arms, order


def matrix(arms: dict[str, dict[int, dict]], eps: list[str]) -> None:
    names = [a for a in ARM_ORDER if a in arms] + \
            [a for a in sorted(arms) if a not in ARM_ORDER]
    seeds = sorted({s for v in arms.values() for s in v})
    w = max((len(e) for e in eps), default=10) + 1
    # Wide enough for the widest ARM NAME as well as for one character per seed. Sizing it
    # on the seeds alone ran `all` straight into `pivot15` in the header.
    cell = max(len(seeds), 4, max((len(a) for a in names), default=4)) + 2

    print("PER-EPISODE, ONE CHARACTER PER SEED".center(w + cell * len(names)))
    print(f"  seeds {seeds}   Y = reached the goal, . = did not, ? = no run on disk\n")
    print(f"{'episode':<{w}}" + "".join(f"{a:>{cell}}" for a in names))
    print("-" * (w + cell * len(names)))
    for ep in eps:
        row = f"{ep:<{w}}"
        for a in names:
            s = "".join("Y" if (r := arms[a].get(sd, {}).get(ep)) and r["success"]
                        else ("?" if r is None else ".") for sd in seeds)
            row += f"{s:>{cell}}"
        print(row)
    print("-" * (w + cell * len(names)))
    tot = f"{'solved / runs':<{w}}"
    for a in names:
        rs = [r for sd in seeds for r in arms[a].get(sd, {}).values()]
        tot += f"{sum(r['success'] for r in rs)}/{len(rs)}".rjust(cell)
    print(tot + "\n")
    for a in names:
        print(f"  {a:<9} {ARM_WHAT.get(a, '')}")


def scoreboard(arms: dict[str, dict[int, dict]]) -> None:
    names = [a for a in ARM_ORDER if a in arms] + \
            [a for a in sorted(arms) if a not in ARM_ORDER]
    print("\n\nARM SCOREBOARD\n")
    print(f"{'arm':<10}{'solved':>9}{'closest':>10}{'closed':>9}{'spl':>8}"
          f"{'guard':>8}{'turns':>8}")
    print("-" * 62)
    for a in names:
        rs = [r for sd in sorted(arms[a]) for r in arms[a][sd].values()]
        if not rs:
            continue
        fin = [r["closest_m"] for r in rs if r["closest_m"] == r["closest_m"]]
        # `pivots` is None on an arm that never offered a turn, and that is printed as a
        # dash rather than 0. "The action did not exist" and "it existed and was never
        # chosen" are different findings, and the second is the interesting one.
        piv = [r["pivots"] for r in rs if r["pivots"] is not None]
        print(f"{a:<10}{sum(r['success'] for r in rs):>4}/{len(rs):<4}"
              f"{sum(fin) / len(fin) if fin else math.nan:>10.2f}"
              f"{sum(r['closed_frac'] for r in rs) / len(rs):>8.0%} "
              f"{sum(r['spl'] for r in rs) / len(rs):>8.3f}"
              f"{sum(r['guard'] for r in rs):>8}"
              f"{(str(sum(piv)) if piv else '--'):>8}")
    print("\n  closest = mean closest approach to the goal, metres (lower is better)")
    print("  closed  = mean fraction of the starting gap the robot actually closed")
    print("  turns   = in-place turns chosen, summed; -- means the arm had none to offer")


def reproducibility(arms: dict[str, dict[int, dict]], eps: list[str]) -> None:
    """Does `ref` at seed 0 reproduce the archived p3.0_medium run of the same episodes?"""
    ref = arms.get("ref", {}).get(0)
    if not ref:
        return
    old = max(STUDY.parent.glob("sync_study/p3.0_medium"), default=None)
    if old is None or not old.is_dir():
        print("\n\n(no archived p3.0_medium to check reproducibility against)")
        return
    prev = _runs(old)
    rows = [(e, prev[e]["success"], ref[e]["success"])
            for e in eps if e in prev and e in ref]
    agree = sum(a == b for _, a, b in rows)
    print(f"\n\nREPRODUCIBILITY  --  ref/seed 0 against the archived p3.0_medium\n")
    print(f"  same configuration, same seed, same six episodes: {agree}/{len(rows)} agree")
    for e, a, b in rows:
        if a != b:
            print(f"    {e:<30} p3.0_medium {'Y' if a else '.'}  "
                  f"ref/s0 {'Y' if b else '.'}   <- differs")
    if agree == len(rows):
        print("  Nothing outside the label seed is moving between runs, so a difference"
              "\n  between arms below is a difference in the policy.")
    else:
        print("  SOMETHING OUTSIDE THE SEED IS MOVING. Until that is explained, an arm"
              "\n  that differs by fewer episodes than this line does has not been shown"
              "\n  to differ at all.")


def verbose(arms: dict[str, dict[int, dict]], eps: list[str]) -> None:
    print("\n\nEVERY RUN\n")
    print(f"{'arm/seed':<12}{'episode':<30}{'':<4}{'closest':>8}{'path':>8}"
          f"{'guard':>7}{'turns':>7}{'rec':>5}")
    print("-" * 81)
    for a in [x for x in ARM_ORDER if x in arms] + \
             [x for x in sorted(arms) if x not in ARM_ORDER]:
        for sd in sorted(arms[a]):
            for ep in eps:
                r = arms[a][sd].get(ep)
                if not r:
                    continue
                print(f"{a + '/s' + str(sd):<12}{ep:<30}"
                      f"{'Y' if r['success'] else '.':<4}"
                      f"{r['closest_m']:>8.2f}{r['path_m']:>8.1f}{r['guard']:>7}"
                      f"{(r['pivots'] if r['pivots'] is not None else '--')!s:>7}"
                      f"{(r['recoveries'] if r['recoveries'] is not None else 0)!s:>5}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    arms, eps = collect()
    if not arms:
        print(f"no results yet in {STUDY}")
        return 0
    matrix(arms, eps)
    scoreboard(arms)
    reproducibility(arms, eps)
    if args.verbose:
        verbose(arms, eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
