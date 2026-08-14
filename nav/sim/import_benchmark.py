#!/usr/bin/env python3
"""Generate our episode list from DynaNav's benchmark, ordered easiest-first.

Why this is a script and not a hand-written YAML file: the first version of
`episodes.yaml` was hand-picked by reading `benchmark_full.yaml` and choosing
episodes that looked representative. Two of the four choices were bad, and both
in ways that were invisible without cross-referencing DynaNav's own results:

  * `hospital` was episode_6, which DynaNav FAILS (17.86 m, spl 0.00). We spent
    runs holding ourselves to a target the reference implementation misses too.
  * `warehouse` was episode_61, which DynaNav has never scored at all. Its
    reference run covers 49 of the 85 defined episodes -- all hospital and
    office, zero warehouse, zero outdoor. There is no published number for the
    Aisle-05 task, so a failure there is not a regression against anything.

The selection criterion has to be DynaNav's measured result, not a human reading
the instruction text. So: join `benchmark_full.yaml` (the definitions, which have
`start_yaw`) against `benchmark_*_results_latest.json` (the outcomes, which have
`success`, `spl` and `path_length`), keep what DynaNav actually completes, and
sort by how hard it is likely to be FOR US specifically.

That last part is the interesting bit. Our difficulty ordering is not DynaNav's,
because our known weakness is not theirs:

  TURN SIZE DOMINATES. Measured over several runs, the policy commits to far
  less yaw than a task needs, and the deficit is what kills episodes -- see
  `guidance.py`. So `|initial bearing to goal|` is weighted hardest.
  DETOUR RATIO next. path_length / straight_line_distance ~ 1.0 means the route
  is a straight shot; 1.5 means it bends around something, i.e. more turning.
  LENGTH last, and mildly. Long episodes are not harder per metre, they just
  offer more chances to drift, and drift compounds.
  SCENE PRIOR as a tie-break: DynaNav manages hospital 19/25 = 76.0% and office
  10/24 = 41.7%, so a hospital episode is the better bet at equal geometry.

Usage:
    python3 nav/sim/import_benchmark.py                 # print the ranked table
    python3 nav/sim/import_benchmark.py --write         # regenerate episodes.yaml
    python3 nav/sim/import_benchmark.py --top 10        # only the 10 easiest
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml

DYNANAV_ROOT = Path(
    os.environ.get("TICVLA_DYNANAV_ROOT", "/home/gtu-dsa/robotics/TIC-VLA/DynaNav")
)
CONFIG = DYNANAV_ROOT / "configs/benchmark_full.yaml"
REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "nav/config/episodes.yaml"
# Episodes DynaNav never scored, so the generator cannot produce them, but which we
# still want to run. Appended verbatim after the ladder. See that file for why.
MANUAL = REPO / "nav/config/episodes_manual.yaml"

# DynaNav's per-scene success rate on its own reference run. Used only as a
# tie-break, and stated as data rather than folded into a magic constant.
SCENE_PRIOR = {"hospital": 0.760, "office": 0.417, "warehouse": 0.0, "outdoor": 0.0}


def find_results() -> Path:
    """Newest full-benchmark result file. The `full` run is the only one that
    covers more than a handful of episodes; `local` and `example` runs are smoke
    tests and ranking against them would rank against noise."""
    candidates = sorted(
        (DYNANAV_ROOT / "benchmark_results").glob("*/benchmark_full_results_latest.json")
    )
    if not candidates:
        raise SystemExit(f"no full benchmark results under {DYNANAV_ROOT}/benchmark_results")
    return candidates[-1]


def scene_name(scene_url: str) -> str:
    stem = scene_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".usd").lower()
    for known in SCENE_PRIOR:
        if known in stem:
            return known
    return stem


def slugify(instruction: str, scene: str, used: set[str]) -> str:
    """A readable name, because `episode_39` tells you nothing at the CLI."""
    stop = {
        "the", "a", "an", "to", "at", "in", "on", "of", "and", "then", "go",
        "walk", "move", "proceed", "straight", "ahead", "stop", "front",
        "your", "is", "it", "through", "toward", "towards", "until", "next",
    }
    words = [
        w.strip(".,()").lower()
        for w in instruction.split()
        if w.strip(".,()").lower() not in stop and w.strip(".,()").isalpha()
    ]
    base = f"{scene}_{'_'.join(words[:2]) or 'episode'}"
    name, n = base, 2
    while name in used:
        name, n = f"{base}{n}", n + 1
    used.add(name)
    return name


def difficulty(turn_deg: float, detour: float, dist_m: float, scene: str) -> float:
    """Lower is easier. The weights encode our failure mode, not DynaNav's.

    Turn is divided by 10 deg so that a 25 deg turn -- the Aisle-05 deficit we
    could not close -- scores 2.5, dominating everything else in the sum. A
    perfectly straight 30 m corridor scores well under 1.0, which is the intended
    message: length is cheap, turning is not.
    """
    return (
        abs(turn_deg) / 10.0
        + 2.0 * max(0.0, detour - 1.0)
        + dist_m / 40.0
        + (1.0 - SCENE_PRIOR.get(scene, 0.0))
    )


def build() -> list[dict]:
    cfg = yaml.safe_load(CONFIG.read_text())
    results = json.loads(find_results().read_text())["episodes"]

    rows = []
    for res in results:
        if not res.get("success"):
            continue
        name = res["episode"]
        spec = cfg.get(name)
        if spec is None:
            continue

        start, goal = res["start"], res["goal"]
        # `start_yaw` in benchmark_full.yaml is DEGREES, despite sitting next to
        # metre-valued coordinates. Reading it as radians once produced a -5156 deg
        # bearing and a completely wrong difficulty ranking.
        yaw = math.radians(float(spec.get("start_yaw", 0.0)))
        dist = math.dist(start[:2], goal[:2])
        bearing = math.atan2(goal[1] - start[1], goal[0] - start[0]) - yaw
        turn = math.degrees((bearing + math.pi) % (2 * math.pi) - math.pi)
        detour = (res["path_length"] / dist) if dist > 1e-6 else 99.0
        scene = scene_name(res["scene"])

        rows.append(
            {
                "episode": name,
                "scene_name": scene,
                "scene": res["scene"],
                "start": start,
                "start_yaw_deg": float(spec.get("start_yaw", 0.0)),
                "goal": goal,
                "instruction": res["instruction"],
                "dynanav_timeout": res["timeout"],
                "dynanav_spl": round(res["spl"], 3),
                "dynanav_path_m": round(res["path_length"], 2),
                "dynanav_duration_s": round(res["duration"], 1),
                "straight_m": round(dist, 2),
                "turn_deg": round(turn, 1),
                "detour": round(detour, 2),
                "difficulty": round(difficulty(turn, detour, dist, scene), 2),
            }
        )

    rows.sort(key=lambda r: r["difficulty"])
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Collapse episodes that differ only in their human traffic.

    The 29 completions are really 9 tasks. DynaNav is a DYNAMIC navigation
    benchmark, so it reruns the same start/goal/instruction under different
    pedestrian seeds -- episodes 1 through 5 are all the same vending machine.
    Those reruns measure something real for DynaNav and nothing at all for us:
    our scenes are static geometry with no simulated humans in them, so five
    identical runs would cost five times the wall clock and produce one data
    point. Collapsed here rather than in the runner so the ranked table and the
    generated YAML agree about what is being attempted.

    Keeps the variant DynaNav did best on, and records how many there were, so
    "DynaNav managed this 5 times out of 5" stays visible where it matters.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (tuple(r["start"]), tuple(r["goal"]), r["instruction"])
        groups.setdefault(key, []).append(r)

    out = []
    for members in groups.values():
        best = max(members, key=lambda r: (r["dynanav_spl"], -r["dynanav_path_m"]))
        best = dict(best)
        best["dynanav_variants"] = len(members)
        best["dynanav_also"] = [m["episode"] for m in members if m is not None]
        out.append(best)
    out.sort(key=lambda r: r["difficulty"])
    return out


def to_yaml(rows: list[dict]) -> str:
    used: set[str] = set()
    out = [
        "# GENERATED by nav/sim/import_benchmark.py -- do not hand-edit.",
        "#",
        "# Every episode here is one DynaNav COMPLETES on this machine (spl > 0).",
        "# Ordered easiest-first for us, which is not the same as easiest for",
        "# DynaNav: the ranking weights turn size hardest, because that is where our",
        "# runs fail. See the module docstring for the weights and why.",
        "#",
        "# `dynanav_*` fields are the reference numbers to beat, carried alongside so",
        "# a result can be read without opening DynaNav's JSON. Rank candidates by",
        "# spl, never by navigation error: every DynaNav success records ~1.49 m",
        "# because the episode terminates the instant it crosses the 1.5 m threshold.",
        "",
        "defaults:",
        "  success_threshold_m: 1.5   # DynaNav's own criterion (benchmark_full.yaml)",
        '  robot_type: "wheeled robot"  # goes into TIC-VLA\'s system prompt verbatim',
        "  # DynaNav's own Nova Carter cap, and that matters more than it looks: the",
        "  # policy sees its own past motion through previous_waypoints_text, and it",
        "  # plans at the speed it was trained at (measured mean 0.731 m/s asked on",
        "  # office_nearest_elevator). A 0.6 cap clipped nearly every plan, which also",
        "  # threw away the braking the `braking` controller exists to obey -- min(cap,",
        "  # plan) engaged in 11 of 141 calls. Cap above the policy's usual ask and the",
        "  # plan speed is what actually governs.",
        "  max_speed_mps: 1.5",
        "  max_yaw_rate_radps: 1.2",
        "  replan_every_steps: 30     # physics steps between policy calls (60 Hz -> 0.5 s)",
        "",
        "episodes:",
    ]
    for r in rows:
        name = slugify(r["instruction"], r["scene_name"], used)
        # Budget on distance at a conservative average speed, with headroom for the
        # policy's non-straight path, and never less than DynaNav allowed. The cap now
        # matches Nova Carter's, but the *average* is well under it -- the policy
        # brakes, and the guard slows the robot near geometry -- so DynaNav's own
        # timeout would still cut some episodes off mid-approach. A generous timeout
        # costs nothing: the episode ends on arrival, not on the clock.
        timeout = max(r["dynanav_timeout"], int(r["dynanav_path_m"] / 0.35) + 20)
        out += [
            f"  {name}:",
            f"    # {r['episode']}  |  DynaNav: spl {r['dynanav_spl']}, "
            f"path {r['dynanav_path_m']} m, {r['dynanav_duration_s']} s",
            f"    # ours: turn {r['turn_deg']:+.1f} deg, straight {r['straight_m']} m, "
            f"detour {r['detour']}x, difficulty {r['difficulty']}",
            f'    scene: "{r["scene"]}"',
            f"    start: {r['start']}",
            f"    start_yaw_deg: {r['start_yaw_deg']}",
            f"    goal: {r['goal']}",
            f'    instruction: "{r["instruction"]}"',
            f"    timeout_s: {timeout}",
            f"    dynanav_spl: {r['dynanav_spl']}",
            f"    dynanav_path_m: {r['dynanav_path_m']}",
            f"    dynanav_duration_s: {r['dynanav_duration_s']}",
            "",
        ]

    if MANUAL.exists():
        body = [ln for ln in MANUAL.read_text().splitlines() if not ln.startswith("#")]
        out += [
            "  # ---- appended verbatim from episodes_manual.yaml ----",
            "  # Not in DynaNav's scored set, so they carry no reference numbers.",
            *[ln for ln in body if ln.strip()],
            "",
        ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help=f"overwrite {OUT}")
    ap.add_argument("--top", type=int, default=0, help="keep only the N easiest")
    ap.add_argument(
        "--all-variants",
        action="store_true",
        help="keep DynaNav's pedestrian-seed reruns as separate episodes",
    )
    args = ap.parse_args()

    rows = build()
    if not args.all_variants:
        rows = dedupe(rows)
    if args.top:
        rows = rows[: args.top]

    print(f"{len(rows)} DynaNav-completed episodes, easiest first\n")
    hdr = (
        f"{'#':>3} {'episode':>11} {'x':>3} {'scene':>9} {'turn':>7} {'dist':>6} "
        f"{'detour':>7} {'diff':>6}  instruction"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        print(
            f"{i:3d} {r['episode']:>11} {r.get('dynanav_variants', 1):3d} "
            f"{r['scene_name']:>9} {r['turn_deg']:+7.1f} "
            f"{r['straight_m']:6.1f} {r['detour']:7.2f} {r['difficulty']:6.2f}  "
            f"{r['instruction'][:52]}"
        )
    print("\n'x' = how many pedestrian-seed variants DynaNav completed for that task.")

    if args.write:
        OUT.write_text(to_yaml(rows))
        print(f"\nwrote {len(rows)} episodes to {OUT}")


if __name__ == "__main__":
    main()
