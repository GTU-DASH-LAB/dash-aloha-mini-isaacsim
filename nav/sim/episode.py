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
    plans: list[tuple[float, float, float, float, float | None, float]] = field(
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
