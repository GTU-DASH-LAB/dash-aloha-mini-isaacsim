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
    python3 nav/sim/import_benchmark.py --write --out /tmp/x.yaml   # ladder elsewhere
    python3 nav/sim/import_benchmark.py --all           # ALL 85 -> episodes_full.yaml
    python3 nav/sim/import_benchmark.py --all --out /tmp/full.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

DYNANAV_ROOT = paths.dynanav_root()
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


def portable_scene(scene: str) -> str:
    """Write the DynaNav root back out as a variable rather than as this machine's path.

    DynaNav's results JSON records whatever absolute path the benchmark ran against, so
    the generated episode table used to carry four copies of one person's home directory
    -- which is a file that cannot be checked out anywhere else and run. `episode._expand`
    resolves the variable back through `nav/paths.py`, so the local scenes now follow the
    submodule wherever it lands. Omniverse URLs contain no local root and pass through.
    """
    root = str(DYNANAV_ROOT).rstrip("/")
    if scene.startswith(root + "/"):
        return "${TICVLA_DYNANAV_ROOT}" + scene[len(root):]
    return scene


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


# The `defaults:` block both generated tables open with. One copy, so the full table
# (`--all`) cannot drift from the ladder on the speed cap or the arrival radius.
DEFAULTS_BLOCK = [
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
]


def timeout_budget(dynanav_timeout: float, path_m: float) -> int:
    """Seconds of sim time an episode gets: DynaNav's own allowance, or a 0.35 m/s
    average over the reference path plus 20 s, whichever is longer. See `to_yaml`."""
    return max(dynanav_timeout, int(path_m / 0.35) + 20)


def ladder_names(rows: list[dict]) -> list[tuple[str, dict]]:
    """The name `to_yaml` gives each ladder row, in order.

    Factored out so `--all` can say which full-table episode is which ladder episode
    without a second copy of the naming rule. `slugify` disambiguates by order of
    appearance (`..._hallway`, `..._hallway2`), so the names are only reproducible by
    walking the rows in exactly this order.
    """
    used: set[str] = set()
    return [(slugify(r["instruction"], r["scene_name"], used), r) for r in rows]


def to_yaml(rows: list[dict]) -> str:
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
        *DEFAULTS_BLOCK,
        "",
        "episodes:",
    ]
    for name, r in ladder_names(rows):
        # Budget on distance at a conservative average speed, with headroom for the
        # policy's non-straight path, and never less than DynaNav allowed. The cap now
        # matches Nova Carter's, but the *average* is well under it -- the policy
        # brakes, and the guard slows the robot near geometry -- so DynaNav's own
        # timeout would still cut some episodes off mid-approach. A generous timeout
        # costs nothing: the episode ends on arrival, not on the clock.
        timeout = timeout_budget(r["dynanav_timeout"], r["dynanav_path_m"])
        out += [
            f"  {name}:",
            f"    # {r['episode']}  |  DynaNav: spl {r['dynanav_spl']}, "
            f"path {r['dynanav_path_m']} m, {r['dynanav_duration_s']} s",
            f"    # ours: turn {r['turn_deg']:+.1f} deg, straight {r['straight_m']} m, "
            f"detour {r['detour']}x, difficulty {r['difficulty']}",
            f'    scene: "{portable_scene(r["scene"])}"',
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


# ======================================================================================
# --all: EVERY benchmark episode, not just the ones DynaNav completed
# ======================================================================================
#
# The ladder above answers "which episodes should we attempt first". The full table
# answers a different question -- "can our robot even be placed in each of DynaNav's 85
# episodes, with correct physics?" -- and for that a failed or never-scored episode is
# exactly as interesting as a passed one. So nothing is filtered, ranked or deduped here:
# one entry per `episode_N`, in DynaNav's own order, named `<scene>_eNN` so the number
# can be grepped back to `benchmark_full.yaml` without a lookup table.
#
# It is written to its OWN file and never to episodes.yaml: the ladder, every result on
# disk and `bench.sh` are keyed on the ladder's names, and renaming them would orphan all
# of it. `ours:` on each entry records the ladder name when one exists.

FULL_OUT = REPO / "nav/config/episodes_full.yaml"
# The average speed `timeout_budget` assumes, and the detour allowed for an episode with
# no successful DynaNav path to budget against: the mean detour over DynaNav's 29
# completions is ~1.0 and the worst is 1.14 (hospital_forward_staircase / _exit_room),
# so 1.25 covers every route DynaNav itself has driven.
NO_PATH_DETOUR = 1.25
# Same-point tolerance when matching coordinates between tables. Every coordinate in both
# files is written to 2 decimals, so anything closer than 1.1 cm is the same number.
SAME_POINT_M = 0.011


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def off_bearing_deg(start, goal, yaw_deg: float) -> float:
    """Goal bearing relative to the start heading, degrees, positive = goal to the LEFT.

    `start_yaw` in benchmark_full.yaml is DEGREES (see `build`), and is applied by
    DynaNav as a plain rotate-z with forward = +X -- the same convention as
    `base_drive.reset_to`. Three of DynaNav's four smoke episodes spawn ~170 deg off
    their goal, which looks exactly like a broken policy, so this rides on every entry.
    """
    bearing = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    return _wrap_deg(bearing - float(yaw_deg))


def _norm_text(s: str) -> str:
    """Instruction identity ignoring case and ALL whitespace -- episodes_manual.yaml
    deliberately splits DynaNav's 'slopeon' into 'slope on', and that is still the same
    instruction."""
    return "".join(s.lower().split())


def _same_xy(a, b) -> bool:
    return math.dist(a[:2], b[:2]) < SAME_POINT_M


def episode_number(key: str) -> int:
    return int(key.rsplit("_", 1)[-1])


def load_benchmark() -> tuple[dict, dict[str, dict]]:
    """(top-level settings, {episode_N: spec}) straight out of benchmark_full.yaml."""
    cfg = yaml.safe_load(CONFIG.read_text())
    top = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    eps = {k: v for k, v in cfg.items() if isinstance(v, dict) and k.startswith("episode_")}
    return top, dict(sorted(eps.items(), key=lambda kv: episode_number(kv[0])))


def match_ladder(bench: dict[str, dict]) -> dict[str, str]:
    """{episode_N: name in episodes.yaml} for every ladder episode we can pin down.

    Generated rows carry their DynaNav episode exactly (`build` knows it). Manual rows
    are matched on `notes` ("DynaNav episode_52") when they say, otherwise on identical
    start, yaw, goal and instruction, tie-broken by timeout -- DynaNav reruns a task
    under five crowd sizes with different timeouts, and the manual entries copied the
    timeout of the one they were taken from.
    """
    import re

    existing = yaml.safe_load(OUT.read_text()).get("episodes", {}) or {}
    out: dict[str, str] = {}
    for name, r in ladder_names(dedupe(build())):
        spec = existing.get(name)
        if spec and _same_xy(spec["start"], r["start"]) and _same_xy(spec["goal"], r["goal"]):
            out[r["episode"]] = name
    claimed = set(out.values())
    for name, spec in existing.items():
        if name in claimed:
            continue
        m = re.search(r"episode_(\d+)", str(spec.get("notes", "")))
        if m and f"episode_{m.group(1)}" in bench:
            out[f"episode_{m.group(1)}"] = name
            continue
        cands = [
            k for k, b in bench.items()
            if _same_xy(b["start"], spec["start"]) and _same_xy(b["goal"], spec["goal"])
            and abs(_wrap_deg(float(b.get("start_yaw", 0.0)) - float(spec["start_yaw_deg"])))
            < 0.01
            and _norm_text(b["instruction"]) == _norm_text(spec["instruction"])
        ]
        if len(cands) > 1:
            timed = [k for k in cands if float(bench[k]["timeout"]) == float(spec["timeout_s"])]
            cands = timed or cands
        if cands:
            out[cands[0]] = name
    return out


def build_full() -> list[dict]:
    """One row per benchmark_full.yaml episode, all 85, with everything a scene agent
    needs to decide whether the episode is runnable and what to compare it against."""
    top, bench = load_benchmark()
    results = {r["episode"]: r for r in json.loads(find_results().read_text())["episodes"]}
    existing = yaml.safe_load(OUT.read_text()).get("episodes", {}) or {}
    ours = match_ladder(bench)
    default_thr = float(top.get("success_threshold", 1.5))

    rows = []
    for key, spec in bench.items():
        n = episode_number(key)
        scene = scene_name(spec["scene"])
        start, goal = list(spec["start"]), list(spec["goal"])
        yaw = float(spec.get("start_yaw", 0.0))
        res = results.get(key)
        mine = ours.get(key)
        mine_spec = existing.get(mine) if mine else None

        # Same TASK: identical start, yaw, goal and instruction -- DynaNav's crowd-size
        # reruns. Same ROUTE: identical start and goal, whatever the wording; a route is
        # what a path length and a time budget are properties of.
        twins, route = [], []
        for k2, s2 in bench.items():
            if k2 == key or not (_same_xy(s2["start"], start) and _same_xy(s2["goal"], goal)):
                continue
            route.append(k2)
            if (abs(_wrap_deg(float(s2.get("start_yaw", 0.0)) - yaw)) < 0.01
                    and _norm_text(s2["instruction"]) == _norm_text(spec["instruction"])):
                twins.append(episode_number(k2))

        # --- arrival radius ------------------------------------------------------
        # DynaNav sets it PER EPISODE on the outdoor set (4 / 1 / 2 m), and those values
        # win: they are the benchmark's own definition of arriving. Only where DynaNav is
        # silent can a hand-set value of ours for the same start/goal apply, and only
        # then the top-level 1.5 m. Note DynaNav's own twins disagree -- episode_55 is
        # 1 m and episode_56 (same start/goal) 4 m -- which is why a twin's value is
        # never borrowed over the episode's own.
        conflict = ""
        if "success_threshold" in spec:
            thr, thr_src = float(spec["success_threshold"]), "dynanav per-episode"
        else:
            hand = [
                (nm, float(sp["success_threshold_m"])) for nm, sp in existing.items()
                if "success_threshold_m" in sp
                and _same_xy(sp["start"], start) and _same_xy(sp["goal"], goal)
            ]
            if hand:
                thr, thr_src = hand[0][1], f"episodes.yaml:{hand[0][0]} (same start/goal)"
            else:
                thr, thr_src = default_thr, "dynanav top-level success_threshold"
        if mine_spec and "success_threshold_m" in mine_spec:
            if abs(float(mine_spec["success_threshold_m"]) - thr) > 1e-9:
                conflict = (f"episodes.yaml:{mine} uses {mine_spec['success_threshold_m']} m")

        # --- time budget ---------------------------------------------------------
        # An episode already on the ladder keeps the ladder's timeout, so a run of
        # `hospital_e07` is comparable with every recorded run of hospital_down_hallway.
        # Otherwise `timeout_budget` over the best SUCCESSFUL DynaNav path on this route
        # (a failed run's path is a wander, not a route -- episode_22 drove 72 m), and
        # only failing that over the straight line with NO_PATH_DETOUR.
        dist = math.dist(start[:2], goal[:2])
        ok_paths = [results[k]["path_length"] for k in [key, *route]
                    if k in results and results[k].get("success")]
        if mine_spec:
            timeout, to_src = int(mine_spec["timeout_s"]), f"episodes.yaml:{mine}"
        elif ok_paths:
            timeout = timeout_budget(int(spec["timeout"]), min(ok_paths))
            to_src = f"0.35 m/s over best DynaNav path {min(ok_paths):.2f} m"
        else:
            timeout = timeout_budget(int(spec["timeout"]), dist * NO_PATH_DETOUR)
            to_src = f"0.35 m/s over {NO_PATH_DETOUR}x straight line (no DynaNav path)"

        # The ladder's wording wins only where it is the SAME instruction up to
        # whitespace -- i.e. the one deliberate 'slopeon' -> 'slope on' fix.
        instruction = spec["instruction"]
        if mine_spec and mine_spec["instruction"] != instruction \
                and _norm_text(mine_spec["instruction"]) == _norm_text(instruction):
            instruction = mine_spec["instruction"]

        rows.append({
            "name": f"{scene}_e{n:02d}",
            "n": n,
            "scene_name": scene,
            "scene": portable_scene(spec["scene"]),
            "start": start,
            "start_yaw_deg": yaw,
            "goal": goal,
            "instruction": instruction,
            "dynanav_instruction": spec["instruction"],
            "timeout_s": timeout,
            "timeout_source": to_src,
            "success_threshold_m": thr,
            "threshold_source": thr_src,
            "threshold_conflict": conflict,
            "dynanav_timeout_s": spec["timeout"],
            "num_people": int(spec.get("num_people", 0)),
            "off_bearing_deg": round(off_bearing_deg(start, goal, yaw), 1),
            "straight_m": round(dist, 2),
            "dz_m": round(float(goal[2]) - float(start[2]), 2),
            "twins": twins,
            "ours": mine or "",
            "result": res,
        })
    return rows


def _num(v) -> str:
    """A YAML number exactly as the source wrote it (8 stays 8, 0.01 stays 0.01)."""
    return repr(v) if isinstance(v, float) else str(v)


def to_full_yaml(rows: list[dict]) -> str:
    q = json.dumps  # a JSON string is a valid YAML double-quoted scalar, quotes included
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["scene_name"]] = counts.get(r["scene_name"], 0) + 1
    scored = sum(1 for r in rows if r["result"])
    passed = sum(1 for r in rows if r["result"] and r["result"].get("success"))
    out = [
        "# GENERATED by nav/sim/import_benchmark.py --all -- do not hand-edit.",
        "#",
        f"# EVERY episode of DynaNav's benchmark_full.yaml: {len(rows)} episodes, "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".",
        f"# DynaNav's own reference run scored {scored} of them and passed {passed}; the",
        "# rest carry `dynanav_scored: false` and NO reference number -- a failure there is",
        "# not a regression against anything.",
        "#",
        "# Names are <scene>_eNN with NN = DynaNav's episode number. `ours:` names the same",
        "# episode in episodes.yaml (the ranked ladder) when it is on it; that entry's",
        "# timeout and wording are reused so runs of either name stay comparable.",
        "#",
        "# Loads with episode.load_episodes(config_path=...). Fields beyond the Episode",
        "# schema (dynanav_*, off_bearing_deg, num_people, ...) are bookkeeping that",
        "# load_episodes ignores. Two of them qualify EVERY number from this table:",
        "#   num_people  DynaNav spawns that many animated pedestrians; our stages are",
        "#               static, so any episode with num_people > 0 is EASIER here.",
        "#   off_bearing_deg  goal bearing relative to the start heading (+ = left). Beyond",
        "#               ~30 deg the robot starts facing away from its goal.",
        "",
        *DEFAULTS_BLOCK,
        "",
        "episodes:",
    ]
    current = None
    for r in rows:
        if r["scene_name"] != current:
            current = r["scene_name"]
            out += [f"  # ---- {current} ({counts[current]} episodes) "
                    + "-" * max(0, 60 - len(current)), ""]
        res = r["result"]
        if res:
            verdict = "SUCCESS" if res.get("success") else "FAILED"
            ref = (f"DynaNav {verdict}: spl {res['spl']:.2f}, path {res['path_length']:.2f} m, "
                   f"nav error {res['navigation_error']:.2f} m, {res['duration']:.1f} s")
            note = (f"DynaNav episode_{r['n']}: {verdict.lower()} (spl {res['spl']:.2f}, "
                    f"nav error {res['navigation_error']:.2f} m)")
        else:
            ref = "DynaNav: never scored"
            note = f"DynaNav episode_{r['n']}: never scored by DynaNav, no reference"
        if r["num_people"]:
            note += f"; {r['num_people']} pedestrians in DynaNav, none in our static stage"
        out += [
            f"  {r['name']}:",
            f"    # episode_{r['n']}  |  {ref}"
            + (f"  |  ours: {r['ours']}" if r["ours"] else ""),
            f"    scene: {q(r['scene'])}",
            f"    start: [{', '.join(_num(v) for v in r['start'])}]",
            f"    start_yaw_deg: {_num(r['start_yaw_deg'])}",
            f"    goal: [{', '.join(_num(v) for v in r['goal'])}]",
            f"    instruction: {q(r['instruction'])}",
            f"    timeout_s: {r['timeout_s']}",
            f"    success_threshold_m: {_num(r['success_threshold_m'])}",
            f"    notes: {q(note)}",
            f"    dynanav_episode: {r['n']}",
            f"    dynanav_timeout_s: {_num(r['dynanav_timeout_s'])}",
            f"    num_people: {r['num_people']}",
            f"    off_bearing_deg: {r['off_bearing_deg']}",
            f"    straight_m: {r['straight_m']}",
            f"    dz_m: {r['dz_m']}",
            f"    threshold_source: {q(r['threshold_source'])}",
            f"    timeout_source: {q(r['timeout_source'])}",
            f"    dynanav_scored: {'true' if res else 'false'}",
        ]
        if res:
            out += [
                f"    dynanav_success: {'true' if res.get('success') else 'false'}",
                f"    dynanav_spl: {round(res['spl'], 3)}",
                f"    dynanav_path_m: {round(res['path_length'], 2)}",
                f"    dynanav_duration_s: {round(res['duration'], 1)}",
                f"    dynanav_nav_error_m: {round(res['navigation_error'], 2)}",
            ]
        if r["twins"]:
            out.append(f"    dynanav_twins: {r['twins']}   # same start/yaw/goal/instruction")
        if r["ours"]:
            out.append(f"    ours: {r['ours']}")
        if r["instruction"] != r["dynanav_instruction"]:
            out.append(f"    dynanav_instruction: {q(r['dynanav_instruction'])}")
        if r["threshold_conflict"]:
            out.append(f"    threshold_conflict: {q(r['threshold_conflict'])}")
        out.append("")
    return "\n".join(out)


def main_full(out_path: Path) -> None:
    if out_path.resolve() == OUT.resolve():
        raise SystemExit(f"--all never writes {OUT}: the ladder's names key every result on "
                         f"disk. Pick another --out.")
    rows = build_full()
    print(f"{len(rows)} benchmark_full.yaml episodes\n")
    hdr = (f"{'name':>13} {'ours':>26} {'ref':>8} {'thr':>4} {'tmo':>4} {'off':>7} "
           f"{'dist':>6} {'dz':>5} {'ppl':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        res = r["result"]
        ref = ("pass" if res.get("success") else "fail") if res else "-"
        print(f"{r['name']:>13} {r['ours'] or '-':>26} {ref:>8} "
              f"{r['success_threshold_m']:4g} {r['timeout_s']:4d} {r['off_bearing_deg']:+7.1f} "
              f"{r['straight_m']:6.1f} {r['dz_m']:+5.2f} {r['num_people']:4d}")
    conflicts = [r for r in rows if r["threshold_conflict"]]
    for r in conflicts:
        print(f"\nTHRESHOLD CONFLICT {r['name']}: table uses {r['success_threshold_m']} m "
              f"({r['threshold_source']}); {r['threshold_conflict']}")
    out_path.write_text(to_full_yaml(rows))
    print(f"\nwrote {len(rows)} episodes to {out_path}")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help=f"overwrite {OUT} (or --out)")
    ap.add_argument(
        "--all",
        action="store_true",
        help=f"write EVERY benchmark_full.yaml episode to --out (default {FULL_OUT}); "
        "never touches episodes.yaml",
    )
    ap.add_argument("--out", type=Path, default=None,
                    help="output path for --write / --all")
    ap.add_argument("--top", type=int, default=0, help="keep only the N easiest")
    ap.add_argument(
        "--all-variants",
        action="store_true",
        help="keep DynaNav's pedestrian-seed reruns as separate episodes",
    )
    args = ap.parse_args()

    if args.all:
        main_full(args.out or FULL_OUT)
        return

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
        out = args.out or OUT
        out.write_text(to_yaml(rows))
        print(f"\nwrote {len(rows)} episodes to {out}")


if __name__ == "__main__":
    main()
