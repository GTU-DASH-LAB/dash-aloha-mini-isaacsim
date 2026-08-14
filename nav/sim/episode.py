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
    trace: list[tuple[float, float]] = field(default_factory=list)

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
