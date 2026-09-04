"""Score the synchronous-planning study: four sync conditions against the async baseline.

    python3 nav/tools/sync_study_table.py            # scoreboard + per-episode matrix
    python3 nav/tools/sync_study_table.py --verbose  # + the diagnostics behind them

THE QUESTION THIS ANSWERS. The async ladder scored 7/13 at medium, 2/13 at high and 2/13
at very_high, which reads as "thinking harder makes it worse" and is not what was being
measured. `/predict` served a cached plan and regenerated in the background, so the robot
drove blind for the length of every generation and that window grew with the budget.
NAV_PLAN_PERIOD_S stops the robot for each decision, removing the confound, and this
table asks the original question again on clean ground.

WHAT MAY AND MAY NOT BE COMPARED, because three of these columns are traps:

  SUCCESS and CLOSEST are comparable everywhere. A 1.5 m threshold is a 1.5 m threshold
  whatever the planning regime, and closest approach is the metric that survived every
  previous confusion on this stack -- it is what separated a robot that drove to the
  wrong place from one that drove nowhere.

  PATH LENGTH IS NOT COMPARABLE, and reading it as though it were has already cost one
  wrong conclusion here: `warehouse_aisle6` went 20.9 m -> 103.8 m and that 5x was a fix
  working, not a regression. The short run was short because the robot parked for 82% of
  the episode. A path length mixes distance travelled with time spent frozen, and the two
  regimes freeze for entirely different reasons.

  WALL TIME IS NOT COMPARABLE EITHER, and under synchronous planning it is barely even
  interesting: `think_wall_s` is time the robot spent standing still, by design. It is
  reported as a fraction so the cost of a thinking budget is visible, not so two
  conditions can be ranked on it.

  SPL is DynaNav's own metric and is comparable, with the caveat it always had: it is 0
  for any failure, so a mean SPL over 13 episodes is mostly a restatement of the success
  count. It is printed next to that count rather than instead of it.

The async baseline is recovered from `nav/results/ladder_progress_<level>.log` -- the
file `on_episode.sh` wrote as that ladder ran -- and each entry is matched to the newest
result JSON saved at or before its timestamp. That indirection exists because the result
filenames carry a timestamp and an episode name and nothing that says which ladder they
came from; the progress log is the only record that does.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "nav/results"
STUDY = RESULTS / "sync_study"

# `closest_m` and `spl` are NOT fields in the result JSON -- they are derived from the
# trace against the episode's goal, and `success` is recomputed from the threshold rather
# than trusted from the file. Import that instead of reimplementing it: a second copy of
# a scoring rule is how two tables of the same runs come to disagree, and this repo has
# already paid for second copies of a constant three separate times.
sys.path.insert(0, str(REPO / "nav/sim"))
from summarize_runs import load_config, score  # noqa: E402

# (label, period_s, level, mode). Kept in one list so the ordering of the scoreboard, the
# matrix columns and the diagnostics cannot drift apart. Period 0.0 with mode "async" is the baseline regime,
# recovered from the progress logs rather than from a study directory. `mode` is part of
# the key and not decoration: "bounded" and "sync" share a period and differ entirely in
# what the robot does during it, so a (period, level) pair does not identify a condition.
CONDITIONS: list[tuple[str, float, str, str]] = [
    ("async medium", 0.0, "medium", "async"),
    ("async high", 0.0, "high", "async"),
    ("async very_high", 0.0, "very_high", "async"),
    ("bound3 medium", 3.0, "medium", "bounded"),
    ("bound3 high", 3.0, "high", "bounded"),
    ("sync3 medium", 3.0, "medium", "sync"),
    ("sync3 high", 3.0, "high", "sync"),
    ("sync1 medium", 1.0, "medium", "sync"),
    ("sync1 high", 1.0, "high", "sync"),
    # Same regime as `sync3 *` and directly comparable to it: the ONLY difference is that
    # the menu carries two in-place turns. `mode` here names the condition's directory,
    # not the planning regime -- these runs record plan_mode "sync" like their control,
    # which is the point of them.
    ("pivot3 medium", 3.0, "medium", "pivot"),
    ("pivot3 high", 3.0, "high", "pivot"),
]


_CFG = load_config()


def _load(path: Path) -> dict | None:
    """One run, scored by summarize_runs and carrying the raw fields that scoring drops.

    `score()` answers the benchmark questions (did it arrive, how close, spl). The raw
    JSON carries the ones this study added -- the plan period, the wall time spent
    standing still, the vertical jitter, and the per-plan staleness that proves the run
    was synchronous. Both are needed and they are merged under distinct keys.
    """
    scored = score(path, _CFG)
    if scored is None:
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    plans = raw.get("plans") or []
    return scored | {
        "plan_mode": raw.get("plan_mode"),
        "plan_period_s": raw.get("plan_period_s"),
        "think_wall_s": raw.get("think_wall_s"),
        "wall_s": raw.get("wall_s"),
        "base_z_step_max_m": raw.get("base_z_step_max_m"),
        "base_z_span_m": raw.get("base_z_span_m"),
        # None, not 0, on every run recorded before in-place turns existed. The two mean
        # opposite things -- "the action was not available" against "it was and the model
        # never chose it" -- and the second is a finding while the first is just history.
        "pivots": raw.get("pivots"),
        # Index 6 of a plan tuple is `time_delay`. The length guard is not paranoia:
        # runs recorded before the async switch have shorter tuples.
        "time_delays": [p[6] for p in plans if len(p) > 6 and p[6] is not None],
    }


def _async_runs(level: str) -> dict[str, dict]:
    """Recover one async ladder's runs from the progress log it wrote as it went.

    Each progress line is `<iso-time>\t<episode>\t<controller>\t<verdict>`, appended
    immediately after the run's JSON was saved. So the run is the newest
    `*_<episode>_<controller>.json` whose own timestamp is not after the progress line's
    -- newest-at-or-before, not merely newest, or a later re-run of the same episode
    under a different condition answers for this one.
    """
    progress = RESULTS / f"ladder_progress_{level}.log"
    if not progress.is_file():
        return {}
    runs: dict[str, dict] = {}
    for line in progress.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        stamp, ep, ctrl, _verdict = parts[0], parts[1], parts[2], parts[3]
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        best: tuple[datetime, Path] | None = None
        for cand in RESULTS.glob(f"*_{ep}_{ctrl}.json"):
            try:
                made = datetime.strptime(cand.name[:15], "%Y%m%d-%H%M%S")
            except ValueError:
                continue
            made = made.replace(tzinfo=when.tzinfo)
            if made <= when and (best is None or made > best[0]):
                best = (made, cand)
        if best and (data := _load(best[1])):
            runs[ep] = data
    return runs


def _sync_runs(period: float, level: str, mode: str) -> dict[str, dict]:
    """Read one synchronous condition out of its own archive directory.

    No timestamp matching needed here: `run_sync_study.sh` copies exactly this
    condition's results into its own folder, so the directory IS the label. Where an
    episode has more than one file -- a re-run after a crash -- the newest wins.
    """
    d = STUDY / (f"p{period:.1f}_{level}" if mode == "sync"
                 else f"{mode}{period:.1f}_{level}")
    if not d.is_dir():
        return {}
    runs: dict[str, dict] = {}
    for path in sorted(d.glob("*.json")):
        data = _load(path)
        if not data or "episode" not in data:
            continue
        runs[data["episode"]] = data
    return runs


def _runs(period: float, level: str, mode: str) -> dict[str, dict]:
    return _async_runs(level) if mode == "async" else _sync_runs(period, level, mode)


def _fmt(value, unit: str = "", spec: str = "{:.2f}", dash: str = "-") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return spec.format(value) + unit
    return f"{value}{unit}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _real(v) -> bool:
    """True for a usable number -- not None, not NaN. `closest_m` is NaN for a run whose
    episode has no goal in the config, and NaN propagates through a mean silently."""
    return isinstance(v, (int, float)) and v == v


def scoreboard(data: dict[str, dict[str, dict]]) -> list[str]:
    out = [
        "SCOREBOARD",
        "",
        f"{'condition':<16} {'n':>3} {'success':>8} {'closest':>9} {'spl':>6} "
        f"{'guard':>7} {'calls':>7} {'think':>7} {'turns':>7}",
        "-" * 82,
    ]
    for label, _period, _level, _mode in CONDITIONS:
        runs = data[label]
        if not runs:
            out.append(f"{label:<16} {'-- not run --':>40}")
            continue
        rs = list(runs.values())
        ok = sum(bool(r.get("success")) for r in rs)
        # `closest_m` is NaN when the episode has no goal in the config or an empty
        # trace. Dropped rather than propagated: one NaN would turn the whole column
        # into NaN and silently delete the condition from the comparison.
        closest = _mean([r["closest_m"] for r in rs if _real(r.get("closest_m"))])
        spl = _mean([r.get("spl") or 0.0 for r in rs])
        guard = _mean([float(r.get("guard") or 0) for r in rs])
        calls = _mean([float(r.get("calls") or 0) for r in rs])
        # Fraction of the wall clock the robot spent standing still waiting for a
        # decision. Zero under async by construction -- nothing ever waited.
        think = [r["think_wall_s"] / r["wall_s"] for r in rs
                 if r.get("think_wall_s") and r.get("wall_s")]
        # Blank, not 0, where the action did not exist. A column of zeros next to a
        # column of dashes says something true; a column of zeros everywhere would read
        # as "nobody ever wanted to turn", which is a claim about the model and not
        # about which code recorded the run.
        turns = _mean([float(r["pivots"]) for r in rs if r.get("pivots") is not None])
        out.append(
            f"{label:<16} {len(rs):>3} {f'{ok}/{len(rs)}':>8} "
            f"{_fmt(closest, ' m'):>9} {_fmt(spl, '', '{:.2f}'):>6} "
            f"{_fmt(guard, '', '{:.0f}'):>7} {_fmt(calls, '', '{:.0f}'):>7} "
            f"{_fmt(_mean(think) * 100 if _mean(think) is not None else None, '%', '{:.0f}'):>7} "
            f"{_fmt(turns, '', '{:.1f}'):>7}")
    out += [
        "",
        "closest = mean closest approach to the goal (threshold 1.5 m); lower is better.",
        "guard   = mean collision-guard interventions per episode.",
        "calls   = mean policy calls per episode; it is a DECISION COUNT, and the period",
        "          sets it directly, so it is context for the other columns and not a",
        "          score. think = share of wall time the robot stood still deciding.",
        "turns   = mean in-place turns per episode; blank where the menu did not offer",
        "          them. Read it next to success: an unchanged score with turns near 0",
        "          means the prompt failed, and with turns high means the action did.",
        "Path length and wall time are deliberately absent: neither is comparable across",
        "these regimes. See this file's docstring.",
    ]
    return out


def matrix(data: dict[str, dict[str, dict]], episodes: list[str]) -> list[str]:
    heads = [lbl for lbl, _, _, _ in CONDITIONS if data[lbl]]
    out = ["", "PER EPISODE -- closest approach in metres, * marks a success", ""]
    out.append(f"{'episode':<32}" + "".join(f"{h.replace(' ', ''):>16}" for h in heads))
    out.append("-" * (32 + 16 * len(heads)))
    for ep in episodes:
        row = f"{ep:<32}"
        for h in heads:
            r = data[h].get(ep)
            if not r:
                row += f"{'-':>16}"
                continue
            d = r.get("closest_m") if _real(r.get("closest_m")) else None
            mark = "*" if r.get("success") else " "
            row += f"{_fmt(d) + mark:>16}"
        out.append(row)
    return out


def diagnostics(data: dict[str, dict[str, dict]]) -> list[str]:
    """The evidence that the synchronous runs really were synchronous, plus the jitter.

    The first half is not decoration. A run is labelled synchronous because an env var
    was set, and that is a claim about a flag, not about behaviour -- the same class of
    mistake as scoring a ladder against a policy server that had died. `plan_period_s`
    is written by the runner only when it actually took the synchronous branch, and
    `time_delay` is what the robot told the model about how stale its view was: under
    working synchronous planning every one of them is zero, because the robot had not
    moved. A nonzero max here means some episode silently fell back to async.
    """
    out = ["", "DIAGNOSTICS", ""]
    for label, period, level, mode in CONDITIONS:
        runs = data[label]
        if not runs or mode == "async":
            continue
        rs = list(runs.values())
        modes = sorted({r.get("plan_mode") or "?" for r in rs})
        periods = sorted({x for r in rs if (x := r.get("plan_period_s")) is not None})
        stale = [t for r in rs for t in (r.get("time_delays") or [])]
        out.append(
            f"  {label:<16} mode={modes} period={periods}"
            f"  max time_delay={max(stale) if stale else 0.0:.3f} s"
            f"  ({len(rs)} runs)")
    out += [
        "",
        "  The two regimes leave DIFFERENT fingerprints, and each is the proof that the",
        "  run did what its label says:",
        "    sync     max time_delay exactly 0.000 s. The robot did not move while the",
        "             model thought, so it had nothing stale to declare. Anything above",
        "             zero means an episode silently fell back to driving blind.",
        "    bounded  max time_delay at or just under the period, and NEVER far above it.",
        "             The robot did drive, so a real delay is expected and honest; a max",
        "             well past the period means the bound is not holding and the run is",
        "             async wearing the wrong label.",
        ""]

    jitter = [(r.get("base_z_step_max_m"), r.get("base_z_span_m"))
              for runs in data.values() for r in runs.values()
              if r.get("base_z_step_max_m")]
    if jitter:
        step = max(s for s, _ in jitter)
        span = max(sp for _, sp in jitter)
        out += [
            f"  vertical jitter: worst single-step {step * 1000:.1f} mm, worst span "
            f"{span * 1000:.1f} mm, over {len(jitter)} runs",
            "  The base drive writes x and y but carries z through from the contact",
            "  solver, so the wheel spheres bouncing on the floor land in the pose every",
            "  step -- and camera_chase hangs off base_link, so it lands in the video.",
            "  Cosmetic: nothing in navigation reads z, and the collision guard casts a",
            "  horizontal fan.",
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true",
                    help="also print the synchronicity and jitter diagnostics")
    args = ap.parse_args()

    data = {label: _runs(period, level, mode)
            for label, period, level, mode in CONDITIONS}

    # Episode order comes from the config, which is ranked easiest-first, so the matrix
    # reads down the ladder the way the ladder ran. Anything found in a result but not in
    # the config is appended rather than dropped.
    import yaml
    cfg = yaml.safe_load((REPO / "nav/config/episodes.yaml").read_text())["episodes"]
    seen = {ep for runs in data.values() for ep in runs}
    episodes = [n for n in cfg if n in seen] + sorted(seen - set(cfg))

    lines = ["SYNCHRONOUS-PLANNING STUDY", ""]
    if not seen:
        lines.append("No runs found yet.")
        print("\n".join(lines))
        return 0
    lines += scoreboard(data)
    lines += matrix(data, episodes)
    if args.verbose:
        lines += diagnostics(data)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
