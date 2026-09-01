"""Episode definitions loaded from ../config/episodes.yaml.

Pure data + validation, no Isaac Sim imports, so this is importable from the UI
process and from tests without paying a 90 s Kit startup.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "episodes.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass(frozen=True)
class Episode:
    name: str
    scene: str
    start: tuple[float, float, float]
    start_yaw_deg: float
    goal: tuple[float, float, float]
    instruction: str
    timeout_s: float
    success_threshold_m: float
    robot_type: str
    max_speed_mps: float
    max_yaw_rate_radps: float
    replan_every_steps: int
    notes: str = ""

    @property
    def start_yaw_rad(self) -> float:
        return math.radians(self.start_yaw_deg)

    @property
    def straight_line_distance_m(self) -> float:
        """Distance from start to goal, ignoring obstacles.

        Worth logging next to the final distance: DynaNav's office run on this machine
        *started* 13.74 m from the goal and finished 40 m away, so "how far from the
        goal did it end" only means something against where it began.
        """
        return math.dist(self.start[:2], self.goal[:2])


def env_name(scene: str) -> str:
    """The ENVIRONMENT an episode lives in, e.g. "hospital", "office", "warehouse".

    Nav stages are built per environment, not per episode, and this is the key. The
    start pose used to be baked into each stage, so eleven benchmark episodes meant
    eleven Isaac Sim build chains -- four processes each, minutes apiece -- to produce
    stages that differed only in where one prim sat. Six of them were the same
    `hospital.usd`.

    They can share, because the runner teleports to `episode.start` before every run
    anyway (`KinematicBase.reset_to`, which the UI's Reset button has always used).
    The baked pose was never load-bearing; it was just the pose the stage happened to
    open at.

    Derived from the scene URL rather than the episode name so that two episodes
    naming the same USD always share a stage, whatever they are called.
    """
    stem = _expand(scene).rstrip("/").rsplit("/", 1)[-1]
    for suffix in (".usd", ".usda", ".usdc", ".usdz"):
        stem = stem.removesuffix(suffix)
    return stem.replace("full_", "").lower()


def episodes_by_env(
    config_path: Path | str = CONFIG_PATH,
) -> dict[str, list[Episode]]:
    """Every episode grouped by the environment it runs in, config order preserved.

    This is the unit the rest of the system works in now. A built stage serves an
    ENVIRONMENT, so "which episodes can I run right now" is exactly "which episodes
    share the loaded stage" -- the UI asks that question, `bench.sh` asks it to decide
    what to build, and both used to answer it by string-matching episode names.
    """
    grouped: dict[str, list[Episode]] = {}
    for ep in load_episodes(config_path).values():
        grouped.setdefault(env_name(ep.scene), []).append(ep)
    return grouped


def representative_episode(env: str, config_path: Path | str = CONFIG_PATH) -> Episode:
    """Any episode from `env` -- the first one, for build-time defaults.

    The stage still gets authored with a robot at *some* pose, and it should be a
    real one: the office environment is 1063 m across and the origin is not inside
    the navigable area, where a spawn overlapping building geometry made PhysX throw
    the articulation 7.7 m and tip it 181 degrees. Every DynaNav start is known-good
    free space. Which one it is no longer matters, because the runner teleports to
    the episode's own start before each run, but "somewhere valid" still does.
    """
    grouped = episodes_by_env(config_path)
    if env not in grouped:
        raise KeyError(f"unknown environment {env!r}; have: {sorted(grouped)}")
    return grouped[env][0]


def _expand(scene: str) -> str:
    """Resolve ${TICVLA_DYNANAV_ROOT} in scene paths.

    DynaNav's own config uses this variable, so we honour it rather than hard-coding
    a path that would only be right on this machine.
    """
    default_root = "/home/gtu-dsa/robotics/TIC-VLA/DynaNav"
    env = dict(os.environ)
    env.setdefault("TICVLA_DYNANAV_ROOT", default_root)
    for key, value in env.items():
        scene = scene.replace("${" + key + "}", value)
    return scene


def load_episodes(config_path: Path | str = CONFIG_PATH) -> dict[str, Episode]:
    raw = yaml.safe_load(Path(config_path).read_text())
    defaults = raw.get("defaults", {})
    episodes: dict[str, Episode] = {}

    for name, spec in raw["episodes"].items():
        episodes[name] = Episode(
            name=name,
            scene=_expand(spec["scene"]),
            start=tuple(spec["start"]),
            start_yaw_deg=float(spec["start_yaw_deg"]),
            goal=tuple(spec["goal"]),
            instruction=spec["instruction"],
            timeout_s=float(spec["timeout_s"]),
            success_threshold_m=float(
                spec.get("success_threshold_m", defaults["success_threshold_m"])
            ),
            robot_type=spec.get("robot_type", defaults["robot_type"]),
            max_speed_mps=float(spec.get("max_speed_mps", defaults["max_speed_mps"])),
            max_yaw_rate_radps=float(
                spec.get("max_yaw_rate_radps", defaults["max_yaw_rate_radps"])
            ),
            replan_every_steps=int(
                spec.get("replan_every_steps", defaults["replan_every_steps"])
            ),
            notes=spec.get("notes", ""),
        )
    return episodes


def load_episode(name: str, config_path: Path | str = CONFIG_PATH) -> Episode:
    episodes = load_episodes(config_path)
    if name not in episodes:
        raise KeyError(f"unknown episode {name!r}; have: {sorted(episodes)}")
    return episodes[name]


@dataclass
class EpisodeResult:
    """What a run produced. Mirrors DynaNav's benchmark bookkeeping."""

    episode: str
    instruction: str
    success: bool
    final_distance_m: float
    initial_distance_m: float
    elapsed_s: float   # SIM time -- what the episode timeout budgets
    wall_s: float      # includes the ~1.0-1.5 s each synchronous policy call blocks
    steps: int
    policy_calls: int
    path_length_m: float
    guard_interventions: int
    timed_out: bool
    controller: str
    # Which policy answered, straight off its own `/health`: "<model> [<format>] @<port>".
    # Filenames carry the episode and the controller and nothing about the brain, so a
    # directory of runs from TIC-VLA, Q-VLA-direct and the arc-menu selector was
    # distinguishable only by guessing from timestamps -- and this harness has already
    # once scored a ladder off two-week-old files and printed a plausible 6/13. Defaulted
    # so every run recorded before this field existed still loads; "" means unknown, which
    # is the truth about those and must not be read as any particular policy.
    policy: str = ""
    # How many times the robot had to back out of a wedge, and how many of those found
    # the way behind blocked too. Defaulted so runs recorded before the recovery
    # existed still load; 0 on those means "not measured", not "never happened".
    recoveries: int = 0
    recoveries_blocked_behind: int = 0
    # How many of `recoveries` were BALKS -- the robot standing still with clear floor
    # ahead of it -- rather than wedges. A subset, not a second total, and worth its own
    # field because the two are different failures with different fixes: a wedge is a
    # steering problem and a balk is the policy answering STOP without having arrived,
    # which on the first full ladder was the single largest cause of failure. Defaulted,
    # so runs recorded before the balk trigger existed still load; 0 there means "not
    # measured", and specifically not "the robot never stalled".
    balks: int = 0
    # How many of those ended in a forced re-decision, and how many could not. A run with
    # `recoveries > 0` and `recovery_replans == 0` is a robot that reversed and was then
    # handed the same plan back -- the manoeuvre without the point of it.
    recovery_replans: int = 0
    recovery_replans_failed: int = 0
    # The planning regime this run was measured under. 0.0 means ASYNCHRONOUS -- the
    # model thought on a background thread while the robot kept driving the previous
    # plan -- and anything else is the fixed period, in simulated seconds, that the robot
    # drove between full stops. Defaulted so runs recorded before the mode existed still
    # load, and 0.0 is the truth about those.
    #
    # This is not bookkeeping. The two regimes are different experiments and their
    # numbers must never be pooled: under async, a longer thinking budget buys a longer
    # BLIND window, so a thinking-level comparison measures how far the robot travels
    # with its eyes shut. Under sync it measures the decision. A results directory
    # holding both, distinguishable only by timestamp, is how a stack talks itself into
    # believing more thinking makes a policy worse.
    # Which planning regime produced this run: "async" (the robot drives blind for a
    # whole generation), "bounded" (blind for at most one period), or "sync" (never
    # blind, the robot stops for every decision). Three different experiments, and the
    # single most important field for telling them apart afterwards -- `plan_period_s`
    # alone cannot, because bounded and sync share a period and differ entirely in what
    # happens during it.
    plan_mode: str = "async"
    plan_period_s: float = 0.0
    # Wall seconds the robot spent stopped, waiting for decisions. Zero under async by
    # construction, since nothing waits there. Reported next to `wall_s` because it is
    # the whole cost of synchronous planning and `elapsed_s` cannot show it: sim time
    # does not advance while the loop is blocked.
    think_wall_s: float = 0.0
    # Vertical motion of the base over the episode: the largest single-step change in z,
    # and the total span. The drive never commands z -- it carries through whatever the
    # contact solver produced -- so anything here is the wheel spheres working against
    # the floor, and the chase camera is parented to `base_link`, so it is also exactly
    # what a viewer sees as the robot shaking. A step figure near the float32 resolution
    # of the scene's coordinates is nothing; a millimetre-scale one at 60 Hz is a visible
    # buzz. Defaulted, so runs recorded before this was measured still load.
    base_z_step_max_m: float = 0.0
    base_z_span_m: float = 0.0
    # (x, y, yaw_rad). Yaw is in here because position alone cannot distinguish a
    # robot that chose to drive the wrong way from one that never turned at all.
    trace: list[tuple[float, float, float]] = field(default_factory=list)
    # (sim_time, plan_heading_deg, reach_m, bearing_to_goal_deg, guidance_deg,
    # plan_speed_mps) per policy call. `guidance_deg` is None when the model emitted
    # the -100 sentinel. `plan_speed_mps` is the plan's arc length over its fixed
    # 3.0 s horizon -- i.e. the speed the policy is asking for, which the `guided`
    # controller obeys and `pursuit` ignores.
    #
    # The trace records what the ROBOT did; this records what the POLICY ASKED for,
    # and the two failures look identical from the trace alone. A robot that drives
    # straight past its goal may be obeying a straight plan (a perception problem) or
    # ignoring a turning one (a controller problem) -- opposite fixes again. Plan
    # heading is the body-frame direction of the lookahead point, positive left, so it
    # is directly comparable to the bearing the robot would have needed.
    #
    # `guidance_deg` is the SAME quantity read off the policy's other head, the 6 s
    # text guidance. It is logged on every run regardless of controller, because the
    # interesting comparison is plan_heading vs guidance_deg vs bearing_to_goal on one
    # row: when the first two disagree, the 3 s action head is truncating a turn the
    # policy has actually planned, and that is a horizon problem, not a perception one.
    plans: list[tuple[float, float, float, float, float | None, float, float]] = field(
        default_factory=list
    )

    def summary(self) -> str:
        verdict = "SUCCESS" if self.success else ("TIMEOUT" if self.timed_out else "FAILED")
        # Closing the gap matters more than the raw final number -- a run that ends
        # 3 m out having started 20 m out did something; one that ends 3 m out having
        # started 3 m out did not.
        closed = self.initial_distance_m - self.final_distance_m
        return (
            f"[{verdict}] {self.episode} ({self.controller})\n"
            f"  distance to goal : {self.initial_distance_m:.2f} m -> "
            f"{self.final_distance_m:.2f} m  (closed {closed:+.2f} m)\n"
            f"  path travelled   : {self.path_length_m:.2f} m in {self.elapsed_s:.1f} s sim "
            f"({self.wall_s:.1f} s wall, {self.steps} steps, "
            f"{self.policy_calls} policy calls)\n"
            f"  guard stops      : {self.guard_interventions}"
        )

    def save(self, directory: Path | str = RESULTS_DIR) -> Path:
        """Write the run to JSON, trace included.

        The trace is the only artifact that answers "where did it actually go?", and
        it used to live in memory until the process exited. Two runs were compared on
        their final distance alone, which cannot distinguish a robot that drove to the
        wrong place from one that drove nowhere at all -- and those two failures have
        opposite fixes. Cheap to keep: a 70 s episode is ~140 points.
        """
        import dataclasses
        import json
        from datetime import datetime

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = directory / f"{stamp}_{self.episode}_{self.controller}.json"
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return path
