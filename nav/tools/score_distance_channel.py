"""Was the model's distance estimate any good, and did it change what the robot did?

The distance channel was added on a mechanism argument: the shortest arc on the menu is
3 m and the plan runs 7 m, so inside 3 m every label overshoots the target and the robot
can only drive past it or stop dead. Seven of seven failures in the first ladder showed
that shape -- `final` well beyond `closest`, one episode reaching 1.92 m against a 1.5 m
threshold and ending 12.9 m out.

An argument is not a result, and the benchmark's own columns cannot check this one. A
success rate says whether episodes passed; it cannot say whether the estimate that was
supposed to cause the passing was ever right, and a channel that fires at random would
still move the number in a 13-episode ladder. So this joins the two halves the run already
writes down:

  decisions.jsonl   what the model SAID (`target_m`) and what we DID with it (`speed_mps`)
  results/*.json    where the robot actually WAS, via `trace`, against `episode.goal`

Both are recorded at the same 30-step cadence, so decision `step` indexes `trace` directly
at `step // 30`.

One thing to be careful reading. `target_m` is the model's estimate of the distance to the
OBJECT the instruction names; the scored distance is to a benchmark waypoint that is near
that object but is not it. A constant offset between them is expected and is not error.
What would condemn the channel is no relationship at all -- an estimate uncorrelated with
where the robot actually is, which is what "it answers plausibly and means nothing" looks
like. Hence the correlation and the near/far split are the verdict, not the mean error.

Usage:
    python3 nav/tools/score_distance_channel.py --latest-ladder
    python3 nav/tools/score_distance_channel.py /tmp/qvla-menus/20260831-1658_office*
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nav.sim.episode import load_episodes  # noqa: E402
from arc_menu import CREEP_MPS, SLOW_FROM_M  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _run_stamp(run_dir: Path) -> datetime | None:
    """The `20260831-165858` prefix `/reset` stamped onto the directory."""
    try:
        return datetime.strptime(run_dir.name.split("_")[0], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def match_result(run_dir: Path, controller: str) -> Path | None:
    """The results file this run produced.

    Matched on episode name AND on being written after the run started, then earliest of
    those. `--latest` semantics burned this project once already: `summarize_runs.py
    --latest` answered with the newest matching file and scored a 13-episode ladder off
    files up to two weeks old, printing a plausible 6/13 for a run that never drove a
    metre. A file that predates the run cannot be its output, so say so rather than
    guessing.
    """
    episode = run_dir.name.split("_", 1)[1] if "_" in run_dir.name else run_dir.name
    started = _run_stamp(run_dir)
    cands = sorted(RESULTS.glob(f"*_{episode}_{controller}.json"))
    for p in cands:
        # `20260831-165949_office_nearest_elevator_braking.json` -- the stamp already
        # carries its own dash, so it is the first underscore-field whole.
        try:
            when = datetime.strptime(p.name.split("_")[0], "%Y%m%d-%H%M%S")
        except (ValueError, IndexError):
            continue
        if started is None or when >= started:
            return p
    return None


def score(run_dir: Path, controller: str) -> dict | None:
    dec_path = run_dir / "decisions.jsonl"
    if not dec_path.is_file():
        return None
    decs = [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()]
    res_path = match_result(run_dir, controller)
    if res_path is None:
        print(f"  {run_dir.name}: no results file written after this run -- skipping "
              f"rather than scoring it against an older one", file=sys.stderr)
        return None
    res = json.loads(res_path.read_text())
    trace = res["trace"]
    episodes = load_episodes()
    if res["episode"] not in episodes:
        return None
    gx, gy = episodes[res["episode"]].goal[:2]

    rows = []
    for d in decs:
        step = d.get("step")
        if step is None:
            continue
        i = step // 30
        if i >= len(trace):
            continue
        x, y = trace[i][0], trace[i][1]
        rows.append((math.dist((x, y), (gx, gy)), d.get("target_m"), d.get("speed_mps"),
                     bool(d.get("stop"))))
    if not rows:
        return None

    fired = [(t, e, s) for t, e, s, _ in rows if e is not None]
    slowed = [r for r in rows if r[2] is not None and r[2] < 0.7 - 1e-9 and not r[3]]
    near = [(t, e) for t, e, _ in fired if t <= SLOW_FROM_M]
    far = [(t, e) for t, e, _ in fired if t > SLOW_FROM_M]
    return {
        "episode": res["episode"], "success": res["success"],
        "final": res["final_distance_m"], "initial": res["initial_distance_m"],
        "n": len(rows), "fired": len(fired), "slowed": len(slowed),
        "stops": sum(1 for r in rows if r[3]),
        "corr": _pearson([t for t, _, _ in fired], [e for _, e, _ in fired]),
        "near_med": _median([e for _, e in near]), "near_n": len(near),
        "far_med": _median([e for _, e in far]), "far_n": len(far),
        "closest": min(t for t, _, _, _ in rows),
    }


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def _pearson(a: list[float], b: list[float]) -> float | None:
    """Correlation, or None when it is not defined -- never a divide-by-zero verdict.

    This project has printed `ratio: inf` as a PASS off exactly that. If every estimate is
    the same number the correlation has no value, and saying so is the honest answer.
    """
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if va < 1e-9 or vb < 1e-9:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--menu-root", type=Path, default=Path("/tmp/qvla-menus"))
    ap.add_argument("--latest-ladder", action="store_true",
                    help="every recorded run under --menu-root")
    ap.add_argument("--controller", default="braking")
    args = ap.parse_args()

    runs = list(args.run_dirs)
    if args.latest_ladder or not runs:
        runs = sorted(p for p in args.menu_root.glob("*/")
                      if (p / "decisions.jsonl").is_file())
    if not runs:
        print(f"no recorded runs under {args.menu_root}", file=sys.stderr)
        return 1

    # `closest*` is the minimum over TRACE samples, which are written every 30 steps --
    # the harness checks the threshold every step, so its own `closest` is smaller. Read
    # this column for shape, and the benchmark's for the verdict; they are not the same
    # measurement and quoting one as the other is how a run gets misreported.
    print(f"{'episode':27} {'ok':3} {'closest*':>8} {'final':>6} {'fired':>9} {'slowed':>7} "
          f"{'stop':>5} {'corr':>6} {'est<=4m':>8} {'est>4m':>7}")
    scored = []
    for r in runs:
        s = score(r, args.controller)
        if s is None:
            continue
        scored.append(s)
        c = "-" if s["corr"] is None else f"{s['corr']:+.2f}"
        nm = "-" if s["near_med"] is None else f"{s['near_med']:.1f}"
        fm = "-" if s["far_med"] is None else f"{s['far_med']:.1f}"
        print(f"{s['episode']:27} {'Y' if s['success'] else 'n':3} {s['closest']:8.2f} "
              f"{s['final']:6.1f} {s['fired']:4d}/{s['n']:<4d} {s['slowed']:7d} "
              f"{s['stops']:5d} {c:>6} {nm:>8} {fm:>7}")

    if not scored:
        print("nothing scored", file=sys.stderr)
        return 1
    tot_n = sum(s["n"] for s in scored)
    tot_f = sum(s["fired"] for s in scored)
    tot_s = sum(s["slowed"] for s in scored)
    corrs = [s["corr"] for s in scored if s["corr"] is not None]
    print(f"\n{len(scored)} episodes, {sum(s['success'] for s in scored)} succeeded")
    print(f"channel fired on {tot_f}/{tot_n} decisions ({100*tot_f/tot_n:.0f}%); "
          f"the ramp actually slowed the plan on {tot_s} ({100*tot_s/tot_n:.0f}%)")
    if corrs:
        print(f"estimate vs true distance to goal: mean r = {sum(corrs)/len(corrs):+.2f} "
              f"over {len(corrs)} episodes where it is defined")
    print(f"\nRead `est<=4m` against `est>4m`: those are the median ESTIMATE when the robot "
          f"was truly inside and outside {SLOW_FROM_M:.0f} m. If the second is not clearly "
          f"larger,\nthe estimate is not tracking distance and the ramp is firing on noise "
          f"-- in which case any change in the success rate is not this channel working.\n"
          f"A floor of {CREEP_MPS} m/s is 0.24 m/s under what the tightest episode "
          f"(hospital_vending_machine, 13.0 m in 54 s) needs, so a persistent underestimate "
          f"shows up as a timeout, not as a stall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
