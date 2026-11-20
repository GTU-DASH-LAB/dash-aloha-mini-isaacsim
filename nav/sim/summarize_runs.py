#!/usr/bin/env python3
"""Score saved runs against DynaNav's own numbers for the same episodes.

Our result is only meaningful next to the reference one. "Final distance 6.4 m" is
not a grade; "6.4 m where DynaNav finished at 1.49 m with spl 1.00" is. So this
reads `nav/results/*.json`, joins each run to the DynaNav figures carried in
`episodes.yaml`, and prints both.

Two scoring rules worth stating, because both were got wrong once here:

  Rank by SPL, not by navigation error. Every DynaNav success records ~1.49 m of
  final error, because the episode terminates the instant the robot crosses the
  1.5 m threshold. Sorting successes by error therefore sorts by nothing.

  Report CLOSEST APPROACH alongside final distance. A robot that reaches 2.8 m and
  then drives away ends at 22 m, and those two numbers describe completely
  different failures -- one is a stopping problem, the other is a steering problem,
  and they have opposite fixes. Final distance alone cannot tell them apart. This
  is not hypothetical: warehouse_aisle6 did exactly that.

Usage:
    python3 nav/sim/summarize_runs.py                       # every run, grouped
    python3 nav/sim/summarize_runs.py --controller guided   # one controller
    python3 nav/sim/summarize_runs.py --latest hospital_vending --oneline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode import CONFIG_PATH, all_episode_specs, env_name  # noqa: E402  (pure data, no Isaac)

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "nav/results"
# The selected table (episode.CONFIG_PATH: episodes.yaml, or NAV_EPISODES_CONFIG). Rows are
# scored against EVERY table, though -- see load_config().
CONFIG = CONFIG_PATH

# The most the base's z can legitimately span in one episode, in metres. A ground robot's
# height changes only with the floor under it, so the bound is the floor's relief plus
# what the start used to add, with margin -- measured, not guessed (2026-10-05):
#
#   * floor relief: along the six outdoor episodes' straight start->goal lines, 0.11-0.34 m
#     (scratch terrain_lines.json); under the 53 recorded outdoor traces, floor tracked by
#     raycast at every trace sample, at most 1.26 m (outdoor_library reaches a lower plaza).
#     Indoor floors are flat: 199 of 237 indoor runs span < 0.1 m.
#   * the start drop: before 2026-10 the robot spawned at DynaNav's z, up to 2.21 m above
#     the terrain (pillars), and part of that fall landed inside the recorded span.
#
# So a clean pre-fix outdoor run can reach ~3.5 m, a post-fix one ~1.3 m. 5.0 m clears
# the first by 1.4x, and the smallest real event on record is 8.0 m: the nine
# warehouse_aisle6 runs at 8.0-8.4 m, on a flat floor where 8 sibling runs span 3-17 mm,
# with single steps of 0.20-0.49 m (12-29 m/s -- more than a free fall from 8.4 m can
# reach). Every aisle6 trace, clean or not, crosses the footprints of 9 m SM_PillarA_9M
# columns the kinematic base drives through (samples ~20 and ~52); some crossings throw
# it. Four of those nine latched success; that stays on the row, but the row is flagged,
# because the run spent part of its time not on the floor.
Z_SPAN_BOUND_M = 5.0
# The same bound for the FLAT indoor floors (hospital, office, warehouse), where there is no
# relief to allow for. 5 m let the indoor pump through: 15 indoor rows span 2.08-2.94 m, a
# cluster no flat floor can produce, and replaying the fixed office_hallway_turn path
# through the old base reproduces it (2.08 m, |v| 11.6 m/s, started by backing out of a
# barrier). Measured over the 245 indoor rows that record z (2026-10-05): every row whose
# trace is clean spans <= 0.29 m (199 under 0.1 m); the next is 0.62 m, a row whose trace
# also steps 1.29 m in one 0.5 s sample. 0.5 m sits in that gap.
Z_SPAN_BOUND_INDOOR_M = 0.5
# Trace samples are 30 physics steps = 0.5 s apart (run_navigation.py), and the drive
# integrates x/y at no more than the speed cap, so one sample can legitimately move at
# most cap x 0.5 s -- 0.75 m at 1.5 m/s. The bound allows 1.5x that, 1.125 m (the replays'
# onset rule, scratch oreplay/derive2.py). The z and path tests above missed real throws
# that this one sees: 20260901-052020_outdoor_library (13 samples over the bound, up to
# 3.16 m), 20260905-034509_outdoor_umbrellas (3.12 m), 20260905-105942_outdoor_pillars
# (2.59 m), 20260905-115418_outdoor_upsway (4.03 m), and indoor rows up to 7.53 m.
TRACE_SAMPLE_S = 0.5
TRACE_STEP_SLACK = 1.5


def load_config() -> dict:
    """Every episode of every table in nav/config/, each spec with its own table's
    defaults underneath (episode.all_episode_specs). A row is scored by its own episode's
    radius, cap and environment whichever table it was run from -- read from episodes.yaml
    alone, the 85 episodes_full.yaml names had no spec, and outdoor_e5x rows were held to
    the indoor z-span bound and the default arrival radius."""
    return all_episode_specs()


def load_defaults() -> dict:
    return yaml.safe_load(CONFIG.read_text()).get("defaults", {})


def success_threshold(spec: dict | None) -> float:
    """The episode's OWN arrival radius, which is not 1.5 m for six of the nineteen.

    This used to be a module constant `SUCCESS_THRESHOLD_M = 1.5`, and that was a second
    copy of `episodes.yaml` -- the exact failure class CLAUDE.md records, where the copy
    carries the number and not the identifier so nothing links them. `run_navigation.py`
    latches arrival on `ep.success_threshold_m`, and the outdoor set overrides it in both
    directions: 4.0 m for `outdoor_umbrellas`/`_pillars`/`_library`/`_ramp_fountain`,
    but 1.0 m for `outdoor_upsway` and 2.0 m for `outdoor_upsway_far`.

    The tighter overrides are what made this a bug rather than an inconsistency. Scoring
    `closest <= 1.5` on an episode whose real radius is 1.0 m credits a success the
    runner correctly denied, and it does it silently -- a run that touched 1.2 m and
    drove away would have read as a pass here and a fail in the harness that produced it.
    """
    return float((spec or {}).get("success_threshold_m",
                                  load_defaults().get("success_threshold_m", 1.5)))


def reference(spec: dict | None) -> dict:
    """DynaNav's numbers, parsed out of the generated comment block if present.

    They live in a comment rather than a field on purpose: they describe a
    different implementation's run and must never be mistaken for something our
    runner produced. Absent for hand-written episodes, which is the honest answer
    for warehouse -- DynaNav never scored it.
    """
    if not spec:
        return {}
    return {
        k: spec[k]
        for k in ("dynanav_spl", "dynanav_path_m", "dynanav_duration_s")
        if k in spec
    }


def score(path: Path, cfg: dict) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    spec = cfg.get(d.get("episode"))
    goal = (spec or {}).get("goal") or d.get("goal")
    trace = d.get("trace") or []

    closest = float("nan")
    if goal and trace:
        closest = min(math.dist(p[:2], goal[:2]) for p in trace)

    final = d.get("final_distance_m")
    if final is None and goal and trace:
        final = math.dist(trace[-1][:2], goal[:2])

    initial = d.get("initial_distance_m") or float("nan")
    path_len = d.get("path_length_m") or 0.0
    threshold = success_threshold(spec)

    # How much of the gap the robot actually closed. This is the number that
    # separates "went nowhere" from "went almost all the way", and neither final
    # distance nor SPL shows it.
    closed = (
        (initial - closest) / initial if initial == initial and initial > 1e-6 else 0.0
    )

    # --- did the physics let go? -----------------------------------------------------
    # A kinematic base is teleported, so nothing in the sim bounds how far it can end up
    # from where it was: when the integration diverges the robot is flung, and the trace
    # records the flight as if it were a drive. Measured on outdoor_pillars, arm B: the
    # last samples step 85 m EACH, ending 17.9 km out, against 0.7 m maximum on the very
    # same episode's other arm.
    #
    # This is not a cosmetic outlier. Such a run's `path_m` is meaningless, and so are
    # `closest_m` and `closed_frac` -- closest approach becomes a property of the
    # explosion's trajectory rather than of any navigation. Summed into an aggregate it
    # does not merely add noise, it INVERTS the answer: including this one episode made
    # the lidar arm look like it drove 19952 m against 1063, when excluding it the arm
    # drove 713 m against 970, and it moved the mean-closed delta from -3.7 pp to -0.3.
    #
    # The test is against what the speed cap physically permits over the run's own
    # duration, with 2x of slack so no legitimate run can trip it. The margin here is
    # 32x, so there is no borderline case to tune.
    elapsed = d.get("elapsed_s") or 0.0
    cap = float((spec or {}).get("max_speed_mps", load_defaults().get("max_speed_mps", 1.5)))
    reach_bound = 2.0 * cap * elapsed
    # The path test alone missed one: 20260905-013046_outdoor_upsway drove 209 m in 250 s,
    # inside the bound, while its base went 414 m up. A ground robot's height can only
    # change with the floor under it, so z span is the second, independent test, and runs
    # written since 2026-10 carry the runner's own fault record as a third.
    z_span = d.get("base_z_span_m") or 0.0
    faults = d.get("physics_faults") or 0
    max_step = max((math.dist(a[:2], b[:2]) for a, b in zip(trace, trace[1:])),
                   default=float("nan"))
    # The environment from the episode's scene; for a row whose episode is in no table any
    # more, from its name -- every outdoor episode is named outdoor_*.
    outdoor = (env_name(spec["scene"]).startswith("outdoor") if spec and spec.get("scene")
               else str(d.get("episode", "")).startswith("outdoor"))
    z_bound = Z_SPAN_BOUND_M if outdoor else Z_SPAN_BOUND_INDOOR_M
    step_bound = TRACE_STEP_SLACK * cap * TRACE_SAMPLE_S
    blown_why = [why for why, hit in (
        ("path", elapsed > 0 and path_len > reach_bound > 0),
        ("z_span", z_span > z_bound),
        ("trace_step", max_step == max_step and max_step > step_bound),
        ("fault", faults > 0),
    ) if hit]
    blown_up = bool(blown_why)

    # The runner's own verdict, or -- for rows written before it latched arrival every
    # step -- the trace's closest approach. Not the trace on a blown-up row: that closest
    # is a point on the solver's throw, and crediting it would score a flight past the
    # goal as an arrival (0 rows on disk do that today; the guard is for the next one).
    success = bool(d.get("success")) or (
        not blown_up and closest == closest and closest <= threshold)

    # SPL as the benchmark defines it: success weighted by how much the robot
    # overdrove the shortest path. Zero for a failure, by construction.
    spl = 0.0
    if success and path_len > 1e-6:
        spl = initial / max(path_len, initial)

    plans = d.get("plans") or []
    asked = [p[1] for p in plans]
    guide = [p[4] for p in plans if len(p) > 4 and p[4] is not None]
    # Staleness of the KV cache each plan was built on. Absent from runs recorded
    # before the async switch, where it was 0.0 by construction -- hence the length
    # guard rather than a bare index. Worth watching because the action head only
    # plans 3.0 s ahead: a mean creeping toward that means the robot is being steered
    # by plans that have nearly expired, which is a real regression even if the
    # success flag has not moved yet.
    stale = [p[6] for p in plans if len(p) > 6 and p[6] is not None]
    # The contact layer's record (EpisodeResult.contact). None on the bare base and on
    # every row before 2026-10-06 -- "not modelled", which is reported as such and never
    # as zero: those runs drove through whatever was in the way with no record of it.
    contact = d.get("contact")

    return {
        "file": path.name,
        "episode": d.get("episode", "?"),
        "controller": d.get("controller", "?"),
        # "" for every run written before EpisodeResult grew this field. Reported as
        # "unrecorded" rather than guessed: the whole point is to stop a table implying
        # that thirteen numbers came from one brain when nothing on disk says so.
        "policy": d.get("policy", "") or "unrecorded",
        "success": success,
        "initial_m": initial,
        "closest_m": closest,
        "final_m": final if final is not None else float("nan"),
        # Carried so a table can say WHY a 3.99 m closest approach is a pass. Without it
        # the outdoor rows read as a scoring error next to the 1.5 m indoor ones.
        "threshold_m": threshold,
        "path_m": path_len,
        # True when the trace is a flight rather than a drive. Every distance-derived
        # field on this row is void when it is set -- see the comment above.
        "blown_up": blown_up,
        # Which tests fired: "path", "z_span", "trace_step", "fault".
        "blown_why": blown_why,
        "max_step_m": max_step,
        "z_span_m": z_span,
        "physics_faults": faults,
        "fault_kinds": list((d.get("physics_fault") or {}).get("kinds") or []),
        "abort_reason": d.get("abort_reason", ""),
        "spl": spl,
        "closed_frac": closed,
        "calls": d.get("policy_calls", 0),
        "guard": d.get("guard_interventions", 0),
        # Contact columns: steps the command pushed into something (clipped or refused, or
        # the envelope overlapping), DynaNav's physical_collision (a contact held >= 3.33 s,
        # benchmark.py:1276-1295), and the first prim met. None/"" when not modelled.
        "contact_steps": (contact or {}).get("contact_steps") if contact else None,
        "physical_collision": bool(contact.get("physical_collision")) if contact else None,
        "contact_max_run_s": (contact or {}).get("max_run_s") if contact else None,
        "first_contact_prim": (((contact or {}).get("first_contact") or {}).get("prim", "")
                               if contact else ""),
        # "" on every run recorded before the sensor existed. Those were all the
        # 7-ray fan, but the field says so nowhere, so it stays "" rather than being
        # backfilled -- a filter must not silently claim provenance it does not have.
        "lidar": d.get("lidar", ""),
        "elapsed_s": d.get("elapsed_s") or 0.0,
        "asked_mean": (sum(asked) / len(asked)) if asked else float("nan"),
        "guide_mean": (sum(guide) / len(guide)) if guide else float("nan"),
        "guide_frac": (len(guide) / len(plans)) if plans else 0.0,
        "stale_mean": (sum(stale) / len(stale)) if stale else float("nan"),
        "stale_max": max(stale) if stale else float("nan"),
        **reference(spec),
    }


def fmt(v: float, spec: str = "6.2f") -> str:
    return "     -" if v != v else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controller", help="only runs with this controller")
    ap.add_argument("--policy", help="only runs whose policy label contains this "
                                     "substring, e.g. 'menu' or 'InternVL'")
    ap.add_argument("--lidar", help="only runs whose sensor label matches exactly, "
                                    "e.g. 'fan' or 'c1@0.30'. The two arms of the lidar "
                                    "ladder share everything else, so without this a "
                                    "mixed directory scores them as one experiment.")
    ap.add_argument("--latest", metavar="EPISODE", help="only the newest run of it")
    ap.add_argument("--oneline", action="store_true", help="one terse line, for scripts")
    args = ap.parse_args()

    cfg = load_config()
    rows = [r for r in (score(p, cfg) for p in sorted(RESULTS.glob("*.json"))) if r]
    if args.controller:
        rows = [r for r in rows if r["controller"] == args.controller]
    if args.policy:
        rows = [r for r in rows if args.policy in r["policy"]]
    if args.lidar is not None:
        rows = [r for r in rows if r["lidar"] == args.lidar]
    if args.latest:
        rows = [r for r in rows if r["episode"] == args.latest]
        rows = rows[-1:]

    if not rows:
        print("no matching runs" if not args.oneline else "NO-RUN")
        return

    if args.oneline:
        r = rows[-1]
        verdict = "SUCCESS" if r["success"] else "FAIL"
        print(
            f"{verdict} {r['episode']}/{r['controller']}/{r['policy']}: "
            f"{r['initial_m']:.1f} -> closest {fmt(r['closest_m'], '.2f')} m "
            f"(final {fmt(r['final_m'], '.1f')}), closed {r['closed_frac'] * 100:.0f}%, "
            f"spl {r['spl']:.2f}, path {r['path_m']:.1f} m, guard {r['guard']}"
            + (f", contact {r['contact_steps']} steps"
               + (f" (first {r['first_contact_prim'].split('/')[-1]})"
                  if r["first_contact_prim"] else "")
               + (" PHYSICAL COLLISION" if r["physical_collision"] else "")
               if r["contact_steps"] is not None else ", contact n/a")
            # The verdict stays the runner's (a physics fault ends a run as a FAIL, and
            # bench.sh counts it as one; an older row that latched success and was thrown
            # LATER keeps its SUCCESS), and the bracket says the distances on this row
            # are void and why, because a throw is not a navigation result.
            + ("  [PHYSICS: "
               + (f"fault {'+'.join(r['fault_kinds']) or '?'}" if r["physics_faults"]
                  else f"blown up ({'+'.join(r['blown_why'])})")
               + f", z span {r['z_span_m']:.2f} m, max step {fmt(r['max_step_m'], '.2f')} m]"
               if r["blown_up"] else "")
        )
        return

    hdr = (
        f"{'episode':<24}{'ctrl':<10}{'ok':<4}{'init':>7}{'closest':>8}{'final':>7}"
        f"{'closed':>8}{'spl':>6}{'path':>7}{'guard':>6}{'cont':>6}{'coll':>5}"
        f"{'asked':>7}{'guide':>7}{'stale':>7}{'  DynaNav':<18}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ref = (
            f"  spl {r['dynanav_spl']:.2f} {r['dynanav_path_m']:.0f}m"
            if "dynanav_spl" in r
            else "  (never scored)"
        )
        print(
            f"{r['episode']:<24}{r['controller']:<10}"
            f"{'YES' if r['success'] else 'no':<4}"
            f"{fmt(r['initial_m'], '7.1f')}{fmt(r['closest_m'], '8.2f')}"
            f"{fmt(r['final_m'], '7.1f')}{r['closed_frac'] * 100:7.0f}%"
            f"{r['spl']:6.2f}{r['path_m']:7.1f}{r['guard']:6d}"
            f"{'     -' if r['contact_steps'] is None else format(r['contact_steps'], '6d')}"
            f"{'    -' if r['physical_collision'] is None else ('  YES' if r['physical_collision'] else '   no')}"
            f"{fmt(r['asked_mean'], '7.1f')}{fmt(r['guide_mean'], '7.1f')}"
            f"{fmt(r['stale_mean'], '7.2f')}{ref:<18}"
        )

    done = [r for r in rows if r["success"]]
    # Named before the score, because a mixed table is not a result and the reader has
    # to see that before reading the fraction under it.
    seen = sorted({r["policy"] for r in rows})
    print(f"\npolicy: {', '.join(seen)}"
          + ("   <- MIXED; this total spans different policies and is not one result"
             if len(seen) > 1 else ""))
    print(
        f"{len(done)}/{len(rows)} succeeded"
        f"  |  mean closed {sum(r['closed_frac'] for r in rows) / len(rows) * 100:.0f}%"
        f"  |  mean spl {sum(r['spl'] for r in rows) / len(rows):.2f}"
    )
    print(
        "cont = contact steps (the envelope pushed into something and was clipped, "
        "'-' = not modelled: bare base or a run before 2026-10-06); coll = DynaNav's "
        "physical_collision (a contact held >= 3.33 s). "
        "asked = mean heading the ACTION head requested (deg, + is left); "
        "guide = same from the 9s TEXT head; stale = mean age of the KV cache each "
        "plan was built on (s; the action head only plans 3.0 s ahead, and '-' means "
        "a run recorded before inference went async)."
    )


if __name__ == "__main__":
    main()
