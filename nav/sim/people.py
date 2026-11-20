"""DynaNav-style pedestrians on our nav stages, in Isaac Sim 6.0.1.

DynaNav (Isaac Sim 5.0) spawns `num_people` stock characters with IRA 0.7.19, gives each a
fixed list of 10 commands drawn from ONE global seed (666), and lets omni.anim.people 0.7.7
walk them along navmesh shortest paths. The people have no physics; "human collision" is a
binary latch, robot chassis origin within 0.2 m (2-D) of a character root. This module
reproduces that on the 6.0.1 stack and says, below, exactly where it cannot.

Evidence for every claim here: scratchpad/people/design.md (research, file:line citations
into DynaNav and both Isaac Sim versions) and the probe JSON copied to
nav/results/people/. `DN` = /home/gtu-dsa/robotics/TIC-VLA/DynaNav.

SAME AS DYNANAV (reproduced exactly)
  * Seed scheme: one global seed, default 666 (DN/configs/benchmark_full.yaml:3), so every
    episode of a scene with the same num_people has identical people, and a smaller crowd
    is a PREFIX of a larger one.
  * Command lists: DynaNav's OWN generator, imported at runtime from
    DN/generate_commands.py (generate_character_commands, DN/generate_commands.py:84-160,
    called the way DN/benchmark.py:1380-1387 calls it: num_commands=10, seed=666). It is
    imported, not copied: DynaNav is (c) 2026 UCLA Mobility Lab under an Academic Software
    License (TIC-VLA/LICENSE.md) that forbids transferring derivatives, so this repo carries
    no copy of its code or landmark table. Without the local DynaNav checkout, setup()
    raises. The global `random` state DynaNav's generator reseeds is saved and restored.
  * Names `Character`, `Character_01`, ... and asset order: character i gets
    `dynanav_asset_order(seed)[i % 21]`, the IRA 0.7.19 shuffle
    (isaacsim.replicator.agent.core simulation.py:206-217: np.random.seed(seed) +
    choice(replace=False) over the lexicographic CDN folder list minus biped_demo). For seed
    666 it equals the order DynaNav's own run referenced (dn5: Character..Character_09).
    Assets come from the 6.0 CDN; all 21 folders exist there (DynaNav's run pulled them
    from the 4.5 tree).
  * Hospital spawn points: DynaNav's own (dn5 run, 5.0 + IRA, seed 666), 10 of them. Yaw 0.
  * Command semantics (omni.anim.people character_behavior.py:493-557): strictly
    sequential; GoTo walks the navmesh shortest path; a GoTo with no path completes at once
    (skipped); after the last command the character idles forever (number_of_loop = 0).
  * Navmesh parameters: DynaNav's EFFECTIVE bake, not benchmark.py's settings (those were
    5.0 keys the 5.0 nav core overrode on stage open). Hospital measured 1.80/0.40/0.40 m/45
    deg; office/warehouse/outdoor inferred from each scene layer's navmeshSettings.
  * Scoring: `human_collision` is DynaNav's rule exactly (DN/benchmark.py:649-702):
    2-D distance from the robot base origin to each person's root < 0.2 m, latched for the
    episode.

DEVIATIONS (6.0.1 forces them, or a choice made here; all logged in metrics())
  1. omni.anim.people is gone and IRA 1.6.8 rejects 0.7.x configs and opens its own stage.
     People are omni.anim.behavior.core behavior agents (CreateBehaviorAgentCommand with
     the 6.0 HumanMotionLibrary) sequenced from Python in step().
  2. Gait/speed: behavior-agent locomotion instead of 5.0's AnimationGraph clips.
     set_speed(1.1) to match DynaNav's measured median 1.10 m/s; the measured per-person
     median is reported, it is not guaranteed.
  3. LookAround has no clip in the 6.0 motion library: LookAround d = plain idle for d s.
  4. Avoidance: 6.0's C++ crowd sim (avoids rigid Mesh bodies incl. the AlohaMini, passes
     ~0.7 m from its centre) instead of DynaNav's Python side-step rule (robot r = 2 m,
     10 s prediction). Same intent, different distances.
  5. People are PHYSICAL in 6.0.1: each agent gets a kinematic capsule (r ~0.14 m) at play.
     Unfiltered, one walking person knocked the parked AlohaMini 3.97 m and tipped it 108
     deg (design.md 3.6.1). robot_interaction="ghost" (default) restores DynaNav's ghosts:
     UsdPhysics.FilteredPairsAPI on every capsule, targets = every RigidBodyAPI prim under
     the robot, authored after the capsules exist (re-applied after any stop/play). The
     capsule stays a PhysX scene-query target (DynaNav's people were invisible to everything)
     -- BUT PhysX raycasts are NOT a reliable way to sense the walkers. In runs G/H/V two of
     ten walkers were hit by raycast_closest only at their SPAWN point, never at their live
     position, although get_rigidbody_transformation() had every capsule exactly on its agent
     (ghost filter on or off): a stale scene-query pose for some kinematic capsules. Sense people
     with ray_distances_2d() (analytic, live behavior-API pose; matches PhysX to 3 cm on every
     non-stale ray, 156 of 156) and DROP PhysX hits for which is_person_prim() is true.
  6. Positions come from the behavior API (`get_world_translation()`), never USD: in 6.0.1
     the USD transform stays at the spawn point (Fabric-only writes). DynaNav read USD,
     which in 5.0 did follow the walker, so the latch measures the same thing.
     `get_linear_velocity()` returns 0, so velocities are differenced positions.
  7. Navmesh: 6.0.1 key names (agentMinHeight / agentMinRadius / ...; the 5.0 names are
     silently ignored) and BOTH navmesh caches (disk + UJITSO) are DISABLED, because a cached
     bake reports 0.0 from every getter and a results file would claim "agent radius 0.0".
     _bake() flags `cache_hit_suspected` if the getters still read 0.0.
  8. GoTo with a PARTIAL path (6.0.1 query_shortest_path returned a path ending short of
     an unreachable landmark, and move_to(landmark) then "SUCCEEDED" instantly without
     moving): the agent walks to the reachable end of the path, as omni.anim.people walked
     the returned path points. Logged as `partial`.
  9. Spawn points for scenes/counts DynaNav's table does not cover (office, warehouse, and
     hospital beyond 10 people) are drawn with DynaNav's METHOD on the 6.0.1 navmesh
     (`inav.set_random_seed(name, seed + i)` + `query_random_point`, as IRA's randomizer.py
     did) -- different points from DynaNav's, logged per person as spawn_source.
 10. Warehouse: our nav_warehouse.usda has no NavMeshVolume (full_warehouse.usd keeps it
     outside its defaultPrim, so the reference drops it); it is authored on the session
     layer at DynaNav's coordinates. If any bake fails, build() RAISES rather than run a
     peopleless episode labelled num_people=N.
 11. Outdoor: the 200 people baked into outdoor_small.usd (static in 6.0.1, no script) are
     deactivated on the session layer and the first num_people are re-spawned as live
     agents at their baked xyz with their baked assets (that is what DynaNav ran: IRA skips
     existing characters). Untested on GPU (cost of 100-200 agents not measured).
 12. spawn="near" (not DynaNav): seeded navmesh points in an annulus around a centre, for
     demos/tests that need people in view of the robot.

LIFECYCLE (same Kit process as the runner; nothing goes to GPU1)
    people.enable_extensions(app)                     # right after SimulationApp(), BEFORE open_stage
    ppl = people.setup(stage, "hospital", episode_name="hospital_e02", app=app)  # BEFORE play
    timeline.play()
    ppl.attach()                                      # optional: takes the one-off ghost-filter stall here
    ...each physics step, after kit.update():
        ppl.step(dt=1/60, robot_xy=(x, y), robot_yaw=yaw)   # attaches lazily, sequences
    ppl.positions()   # [(name, x, y, z, vx, vy)]  live, metres, world frame
    ppl.metrics()     # dict for EpisodeResult
    ppl.reset()       # episode boundary in a long-lived process: everyone back to spawn
    ppl.teardown()

Nothing here ever saves a layer. Every edit goes to the stage's SESSION layer.

MEASURED ON nav_hospital.usda (Isaac Sim 6.0.1, shared GPU0, nav camera 1920x1080 + chase + overview
rendered every update; scratchpad runs A-G, copies in nav/results/people/try_people_*.json)
    people  Kit update   GPU0 memory (after play -> steady)   one-off attach stall (ghost filter)
       0     28 ms         6.0 GiB flat                        -
       3     38-40 ms      6.3 -> 7.4 GiB                      1.7 s
      10     69 ms         7.1 -> 10.8 GiB                     7.9 s
      40    123-128 ms     6.9/7.5 -> 21.7/22.4 GiB            60-64 s
  40 people with NO cameras is the same (21.7 GiB, 123 ms): the cost is the characters (skinning /
  animation / crowd), not render products. This module's own per-step Python cost is ~0.2-0.4 ms (p50, 40
  people); a "mean" of tens of ms in a results file is the single attach stall divided by the step count.
  Plan for ~2.5-4 ms and ~0.4 GiB of GPU0 per person.
  Determinism: same seed, two separate processes, 3 people, 40 s, robot driving in both -> person positions
  identical to 0.0000 m (every sample). reset() inside a process is reproducible but NOT equal to a fresh
  process (2-4 m apart after ~10 s): use one process per episode for bit-exact crowds.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(__file__).resolve().parent / "people_assets"
EPISODES_FULL = REPO / "nav" / "config" / "episodes_full.yaml"

CDN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"
CHARACTERS_URL = f"{CDN}/People/Characters"
MOTION_LIBRARY_URL = f"{CDN}/People/MotionLibrary/HumanMotionLibrary.usd"

DYNANAV_SEED = 666                  # DN/configs/benchmark_full.yaml:3
DYNANAV_NUM_COMMANDS = 10           # DN/benchmark.py:1383
HUMAN_COLLISION_RADIUS_M = 0.2      # DN/benchmark.py:694-701
DEFAULT_SPEED_MPS = 1.1             # DynaNav walkers, measured median (dn5): 0.0183 m/update at 60 Hz
CAPSULE_RADIUS_M = 0.15             # 6.0.1 BodyCollider radius measured 0.134-0.147 m (probe3)
CAPSULE_TOP_M = 1.65                # BodyCollider spans the floor to ~1.65 m (probe3)
PEOPLE_ROOT = "/World/People"       # PeopleSim(root=...) default; PhysX hits under it are suspect, see below
NAV_SETTINGS = "/exts/omni.anim.navigation.core/navMesh"

# Folder -> usd file, the 21 character folders of People/Characters on the CDN (S3 lists
# keys in byte order; IRA took omni.client.list order and pruned biped_demo, which the 6.0
# tree no longer has). File names read off the 6.0 S3 listing, 2026-10-05.
CHARACTER_FOLDERS: dict[str, str] = {
    "F_Business_02": "F_Business_02.usd",
    "F_Medical_01": "F_Medical_01.usd",
    "M_Medical_01": "M_Medical_01.usd",
    "female_adult_police_01_new": "female_adult_police_01_new.usd",
    "female_adult_police_02": "female_adult_police_02.usd",
    "female_adult_police_03_new": "female_adult_police_03_new.usd",
    "male_adult_construction_01_new": "male_adult_construction_01_new.usd",
    "male_adult_construction_03": "male_adult_construction_03.usd",
    "male_adult_construction_05_new": "male_adult_construction_05_new.usd",
    "male_adult_police_04": "male_adult_police_04.usd",
    "original_female_adult_business_02": "female_adult_business_02.usd",
    "original_female_adult_medical_01": "female_adult_medical_01.usd",
    "original_female_adult_police_01": "female_adult_police_01.usd",
    "original_female_adult_police_02": "female_adult_police_02.usd",
    "original_female_adult_police_03": "female_adult_police_03.usd",
    "original_male_adult_construction_01": "male_adult_construction_01.usd",
    "original_male_adult_construction_02": "male_adult_construction_02.usd",
    "original_male_adult_construction_03": "male_adult_construction_03.usd",
    "original_male_adult_construction_05": "male_adult_construction_05.usd",
    "original_male_adult_medical_01": "male_adult_medical_01.usd",
    "original_male_adult_police_04": "male_adult_police_04.usd",
}
# np.random.seed(666); np.random.choice(sorted folders, 21, replace=False) -- computed with
# numpy 2.5.2 (legacy RandomState, version-stable), and the first 10 equal the assets
# DynaNav's own 5.0 run referenced for Character..Character_09 (dn5).
DYNANAV_ASSET_ORDER_666 = [
    "original_male_adult_construction_02", "original_male_adult_construction_03",
    "original_female_adult_business_02", "original_male_adult_police_04",
    "original_male_adult_construction_01", "original_male_adult_medical_01",
    "female_adult_police_03_new", "male_adult_construction_03", "male_adult_construction_05_new",
    "F_Business_02", "F_Medical_01", "female_adult_police_01_new", "original_female_adult_medical_01",
    "original_male_adult_construction_05", "female_adult_police_02", "original_female_adult_police_03",
    "male_adult_police_04", "male_adult_construction_01_new", "original_female_adult_police_02",
    "M_Medical_01", "original_female_adult_police_01",
]

# our env name (episode.env_name) -> DynaNav's scene_type (DN/benchmark.py:_detect_scene_type)
SCENE_TYPE = {"hospital": "hospital", "office": "office", "warehouse": "warehouse",
              "outdoor_small": "outdoor", "outdoor": "outdoor"}

# DynaNav's EFFECTIVE navmesh bake per scene, in 6.0.1 keys and cm (design.md 1.1).
NAVMESH_PARAMS: dict[str, dict] = {
    "hospital": {"agentMinHeight": 180.0, "agentMinRadius": 40.0, "agentMaxStepHeight": 40.0,
                 "agentMaxFloorSlope": 45.0,
                 "provenance": "measured: DynaNav's own 5.0 bake on hospital (dn5)"},
    "office": {"agentMinHeight": 180.0, "agentMinRadius": 40.0, "agentMaxStepHeight": 10.0,
               "agentMaxFloorSlope": 45.0,
               "provenance": "inferred: office.usd navmeshSettings (r 0.4, step 0.1, slope 45) + 5.0 defaults"},
    "warehouse": {"agentMinHeight": 180.0, "agentMinRadius": 50.0, "agentMaxStepHeight": 1.0,
                  "agentMaxFloorSlope": 1.0,
                  "provenance": "inferred: full_warehouse.usd navmeshSettings (r 0.5, step 0.01, slope 1.0)"},
    "outdoor_small": {"agentMinHeight": 180.0, "agentMinRadius": 20.0, "agentMaxStepHeight": 40.0,
                      "agentMaxFloorSlope": 45.0,
                      "provenance": "inferred: outdoor_small.usd navmeshSettings (r 0.2, step 0.4, slope 45)"},
}
# full_warehouse.usd /Navmesh/NavMeshVolume -- outside defaultPrim /Root, so our reference
# drops it. Raw values read off the layer (pxr). /Root is identity, so world = layer coords.
WAREHOUSE_NAVMESH_VOLUME = {"translate": (-9.999999999999998, 3.593179408144213, 0.0),
                            "scale": (36.114853, 60.0, 5.0)}

# DynaNav's own spawn points (dn5: Isaac Sim 5.0 + IRA 0.7.19, seed 666, usd_at_play).
DYNANAV_SPAWN_TABLES: dict[str, list[tuple[float, float, float]]] = {
    "hospital": [(15.243, 29.244, 0.0), (-27.299, 9.783, 0.0), (-22.362, 7.393, 0.0), (5.374, 9.615, 0.0),
                 (-29.912, 19.516, 0.0), (-40.33, 14.808, 0.0), (-4.895, 6.079, 0.0), (3.896, 11.468, 0.0),
                 (-5.634, 11.606, 0.0), (9.502, 1.96, 0.0)],
}


# ---------------------------------------------------------------------------------- pure
def character_names(n: int, base: str = "Character") -> list[str]:
    """DynaNav / IRA naming: Character, Character_01, Character_02, ..."""
    return ([base] + [f"{base}_{i:02d}" for i in range(1, n)]) if n > 0 else []


DN = Path("/home/gtu-dsa/robotics/TIC-VLA/DynaNav")


def _dynanav_generator():
    """DynaNav's generate_commands module, loaded from the local checkout by file path."""
    import importlib.util

    src = DN / "generate_commands.py"
    if not src.is_file():
        raise RuntimeError(f"people: DynaNav's command generator not found at {src}; it is imported, "
                           "never copied (UCLA Academic Software License)")
    spec = importlib.util.spec_from_file_location("dynanav_generate_commands", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_character_commands(scene_type: str, num_characters: int,
                                num_commands: int = DYNANAV_NUM_COMMANDS,
                                seed: int | None = DYNANAV_SEED,
                                base_name: str = "Character") -> list[str]:
    """DynaNav's command lines for a crowd, e.g. "Character GoTo -5.31 -1.36 0.01 _".

    Calls DN/generate_commands.generate_character_commands exactly as DN/benchmark.py:1380
    does. It reseeds the GLOBAL `random` and writes a file; both side effects are contained
    (state restored, file in a temp dir).
    """
    import contextlib
    import io
    import tempfile

    if num_characters <= 0:
        return []
    mod = _dynanav_generator()
    state = random.getstate()
    try:
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            text = mod.generate_character_commands(targets=scene_type, num_characters=num_characters,
                                                   num_commands=num_commands, seed=seed,
                                                   filename=str(Path(td) / "cmds.txt"), base_name=base_name)
    finally:
        random.setstate(state)
    return [ln for ln in text.splitlines() if ln.strip()]


def parse_commands(lines: list[str]) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for line in lines:
        parts = line.split()
        if parts:
            out.setdefault(parts[0], []).append(parts[1:])
    return out


def dynanav_asset_order(seed: int = DYNANAV_SEED) -> list[str]:
    """IRA 0.7.19 character shuffle (simulation.py:206-217), as folder names."""
    if seed == DYNANAV_SEED:
        return list(DYNANAV_ASSET_ORDER_666)
    import numpy as np  # RandomState == the legacy global np.random.seed path, no global side effect

    folders = sorted(CHARACTER_FOLDERS)
    return [str(f) for f in np.random.RandomState(seed).choice(folders, size=len(folders), replace=False)]


def asset_url(folder_or_rel: str) -> str:
    rel = folder_or_rel if folder_or_rel.endswith(".usd") else f"{folder_or_rel}/{CHARACTER_FOLDERS[folder_or_rel]}"
    return f"{CHARACTERS_URL}/{rel}"


def episode_num_people(episode_name: str | None, config: Path = EPISODES_FULL) -> int | None:
    """DynaNav's num_people for an episode, from episodes_full.yaml (key or its `ours:` name)."""
    if not episode_name:
        return None
    import yaml

    eps = (yaml.safe_load(config.read_text()) or {}).get("episodes", {})
    if episode_name in eps:
        return int(eps[episode_name].get("num_people", 0))
    for e in eps.values():
        if isinstance(e, dict) and e.get("ours") == episode_name:
            return int(e.get("num_people", 0))
    return None


# ------------------------------------------------------------------------------ runtime
@dataclass
class Person:
    name: str
    asset: str                       # 6.0 CDN url
    spawn: tuple[float, float, float]
    spawn_source: str
    commands: list[list[str]]
    prim_path: str
    skel: str | None = None
    agent: object = None
    # sequencer
    ci: int = -1
    task: object = None
    t_end: float | None = None
    cmd: str = ""
    cmd_t0: float = 0.0
    goto_target: tuple[float, float, float] | None = None
    radius: float = CAPSULE_RADIUS_M     # BodyCollider radius, read from USD at attach when available
    # state
    xyz: tuple[float, float, float] | None = None
    vel: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    # metrics
    path_m: float = 0.0
    moving_speeds: list[float] = field(default_factory=list)
    min_dist_robot_m: float = float("inf")
    # issued = GoTo commands started; no_path = skipped (no navmesh path); partial = a path that
    # ended short of the landmark (a FLAG counted at issue: that GoTo's outcome is counted again
    # below, against the reachable end it was given); reached = finished within 0.5 m of the goal
    # it was given; short = succeeded but > 0.5 m from it; failed = the task did not SUCCEED.
    goto: dict[str, int] = field(default_factory=lambda: {"issued": 0, "reached": 0, "short": 0,
                                                          "no_path": 0, "partial": 0, "failed": 0,
                                                          "instant_success": 0})
    log: list[tuple[float, str]] = field(default_factory=list)


class PeopleSim:
    """One crowd on one opened stage. See the module docstring for the lifecycle."""

    def __init__(self, stage, env: str, num_people: int, *, seed: int = DYNANAV_SEED,
                 spawn: str = "dynanav", center: tuple[float, float, float] | None = None,
                 spawn_radius_m: tuple[float, float] = (1.5, 6.0), spawn_heading_rad: float | None = None,
                 spawn_sector_rad: float = math.pi, spawn_spacing_m: float = 1.0,
                 robot_prefix: str = "/World/Aloha", robot_base_path: str = "/World/Aloha/Geometry/base_link",
                 robot_half_extents_m: tuple[float, float] = (0.21, 0.31),
                 robot_interaction: str = "ghost", speed_mps: float = DEFAULT_SPEED_MPS,
                 root: str = "/World/People", dt: float = 1.0 / 60.0, app=None,
                 log: Callable[[str], None] | None = None):
        if env not in NAVMESH_PARAMS:
            raise ValueError(f"people: unknown env {env!r}; known: {sorted(NAVMESH_PARAMS)}")
        if spawn not in ("dynanav", "sample", "near"):
            raise ValueError("spawn must be dynanav | sample | near")
        if spawn == "near" and center is None:
            raise ValueError("spawn='near' needs center=(x, y, z)")
        if robot_interaction not in ("ghost", "solid"):
            raise ValueError("robot_interaction must be ghost | solid")
        self.stage = stage
        self.env = env
        self.num_people = int(num_people)
        self.seed = seed
        self.spawn_mode = spawn
        self.center = center
        self.spawn_radius_m = spawn_radius_m
        self.spawn_heading_rad = spawn_heading_rad
        self.spawn_sector_rad = spawn_sector_rad
        self.spawn_spacing_m = spawn_spacing_m
        self.robot_prefix = robot_prefix
        self.robot_base_path = robot_base_path
        self.robot_half_extents_m = robot_half_extents_m
        self.robot_interaction = robot_interaction
        self.speed_mps = speed_mps
        self.root = root
        self.dt = dt
        self.log = log or (lambda m: print(f"[people] {m}", flush=True))
        self._app = app
        self.people: list[Person] = []
        self.navmesh = None
        self.t = 0.0
        self.built = False
        self.attached = False
        self._attach_wait = 0
        self._needs_start = True
        # True once the command lists have started since attach / the last reset: reset() only
        # touches agents that have actually run (see reset()).
        self._agents_ran = False
        self.build_info: dict = {"errors": []}
        self.ghost_info: dict = {}
        self._prof = {"api_s": 0.0, "advance_s": 0.0, "path_s": 0.0, "path_calls": 0, "steps": 0}
        self._reset_metrics()

    # ------------------------------------------------------------------ helpers
    def _update(self, n: int = 1) -> None:
        if self._app is None:
            import omni.kit.app

            self._app = omni.kit.app.get_app()
        for _ in range(n):
            self._app.update()

    def _session(self):
        from pxr import Usd

        prev = self.stage.GetEditTarget()
        self.stage.SetEditTarget(Usd.EditTarget(self.stage.GetSessionLayer()))
        return prev

    def _reset_metrics(self) -> None:
        self.m = {"steps": 0, "min_dist_m": float("inf"), "min_dist_t": None, "min_dist_person": None,
                  "human_collision": False, "human_collision_t": None, "human_collision_person": None,
                  "human_collision_people": [], "footprint_contact_steps": 0, "footprint_contact_people": [],
                  "steps_within": {"0.5": 0, "1.0": 0, "2.0": 0}, "step_ms": []}

    # -------------------------------------------------------------------- build
    def build(self) -> PeopleSim:
        """BEFORE timeline.play(): navmesh settings, spawn points, agents, bake."""
        if self.num_people <= 0:
            self.log("num_people = 0: nothing to build")
            self.built = True
            return self
        import carb
        import omni.kit.app

        t_build = time.time()
        em = omni.kit.app.get_app().get_extension_manager()
        for ext in ("omni.anim.navigation.core", "omni.anim.behavior.core"):
            if not em.set_extension_enabled_immediate(ext, True):
                raise RuntimeError(f"people: could not enable {ext}")
        self._update()
        import NavSchema  # noqa: F401 -- registered by omni.anim.navigation.core
        import omni.anim.navigation.core as nav

        s = carb.settings.get_settings()
        params = NAVMESH_PARAMS[self.env]
        cfg = {k: v for k, v in params.items() if k != "provenance"}
        cfg["agentMaxRadius"] = max(50.0, cfg["agentMinRadius"])
        cfg["autoRebakeOnChanges"] = False
        cfg["excludeRigidBodies"] = True
        for k, v in cfg.items():
            s.set(f"{NAV_SETTINGS}/config/{k}", v)
        # deviation 7. TWO caches exist (libomni.anim.navigation.core strings): the disk cache
        # (cache/enabled) and UJITSO (cache/enableUjitso, default true). Run A set only the
        # first and still got "0.06 s bake, every getter 0.0" -- the UJITSO hit signature.
        s.set(f"{NAV_SETTINGS}/cache/enabled", False)
        s.set(f"{NAV_SETTINGS}/cache/enableUjitso", False)
        self.build_info["navmesh_settings_cm"] = cfg
        self.build_info["navmesh_provenance"] = params["provenance"]

        prev = self._session()
        try:
            self._prepare_env()
            self.inav = nav.acquire_interface()
            # Agents must exist BEFORE the navmesh is baked (behavior core
            # tests/base_unit_test.py:748-776: "create_human_agent() must be called before
            # create_default_navmesh()"). A pre-bake for spawn sampling followed by agent
            # creation crashed Kit inside the behavior plugin at the first update after play
            # (try_people run A, exit 139). So: create every agent at a provisional point
            # (its final one when the DynaNav table covers it), bake ONCE, sample the
            # remaining spawn points on that navmesh, and move those agents' prims through
            # USD before play.
            table = self._dynanav_table() if self.spawn_mode == "dynanav" else []
            prov = [(tuple(table[i]), "dynanav_table") if i < len(table) else (self._provisional(i), "provisional")
                    for i in range(self.num_people)]
            self._create_agents(prov)
            self._bake("bake")
            if any(src == "provisional" for _, src in prov):
                spawns = self._spawn_points()
                self._move_agents(spawns)
            self._validate()
        finally:
            self.stage.SetEditTarget(prev)
        self.build_info["build_s"] = round(time.time() - t_build, 2)
        self.built = True
        self.log(f"built {len(self.people)} people in {self.build_info['build_s']} s "
                 f"(spawn={self.spawn_mode}, interaction={self.robot_interaction})")
        return self

    def _dynanav_table(self) -> list[tuple[float, float, float]]:
        if self.env.startswith("outdoor"):
            doc = json.loads((ASSET_DIR / "outdoor_small_baked.json").read_text())
            return [tuple(c["xyz"]) for c in doc["characters"]]
        return DYNANAV_SPAWN_TABLES.get(self.env, [])

    def _prepare_env(self) -> None:
        import NavSchema
        from pxr import Gf, Sdf, UsdGeom

        vols = [str(p.GetPath()) for p in self.stage.Traverse() if p.IsA(NavSchema.NavMeshVolume)]
        if not vols and self.env == "warehouse":
            path = f"{self.root}_NavMeshVolume"   # deviation 10
            vol = NavSchema.NavMeshVolume.Define(self.stage, path)
            prim = vol.GetPrim()
            try:
                NavSchema.NavMeshAreaAPI.Apply(prim)
                prim.CreateAttribute("nav:area", Sdf.ValueTypeNames.String).Set("Walkable")
            except Exception as e:  # noqa: BLE001
                self.build_info["errors"].append(f"NavMeshAreaAPI: {e!r}")
            UsdGeom.Boundable(prim).CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(*WAREHOUSE_NAVMESH_VOLUME["translate"]))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
            xf.AddScaleOp().Set(Gf.Vec3f(*WAREHOUSE_NAVMESH_VOLUME["scale"]))
            vols = [path]
            self.build_info["authored_navmesh_volume"] = path
        if not vols:
            raise RuntimeError(f"people: no NavMeshVolume in the {self.env} stage -- cannot bake")
        self.build_info["navmesh_volumes"] = vols
        if self.env.startswith("outdoor"):   # deviation 11
            baked = self.stage.GetPrimAtPath("/World/Environment/Characters")
            if baked and baked.IsValid():
                baked.SetActive(False)
                self.build_info["deactivated"] = ["/World/Environment/Characters"]

    def _bake(self, what: str) -> None:
        t0 = time.time()
        ok = self.inav.start_navmesh_baking_and_wait()
        self.navmesh = self.inav.get_navmesh()
        rec = {"ok": bool(ok), "s": round(time.time() - t0, 2), "has_navmesh": self.navmesh is not None}
        if self.navmesh is not None:
            for fn in ("get_agent_min_height", "get_agent_min_radius", "get_agent_max_step_height",
                       "get_agent_max_slope"):
                try:
                    rec[fn] = round(float(getattr(self.navmesh, fn)()), 3)
                except Exception as e:  # noqa: BLE001
                    rec[fn] = repr(e)
        if self.navmesh is not None and all(isinstance(rec.get(k), float) and rec[k] == 0.0 for k in rec
                                            if k.startswith("get_agent")):
            # A freshly baked mesh reports its agent parameters (hospital: 1.8/0.4/0.4/45);
            # all four 0.0 means a cached mesh was served and nothing here proves which
            # settings it was baked with.
            rec["cache_hit_suspected"] = True
            self.build_info.setdefault("warnings", []).append(
                f"{what}: all agent getters 0.0 -- navmesh probably served from a cache; the settings "
                "written (navmesh_settings_cm) are not confirmed by the baked mesh")
        self.build_info[what] = rec
        self.log(f"{what}: {rec}")
        if not ok or self.navmesh is None:
            raise RuntimeError(f"people: navmesh {what} failed on {self.env} ({rec}); refusing to run a "
                               f"peopleless episode labelled num_people={self.num_people}")

    def _f3(self, x, y, z):
        import carb

        return carb.Float3(float(x), float(y), float(z))

    def _closest(self, x, y, z):
        try:
            q = self.navmesh.query_closest_point(target=self._f3(x, y, z))
            q = q[0] if isinstance(q, (tuple, list)) else q
            return None if q is None else (float(q[0]), float(q[1]), float(q[2]))
        except Exception:  # noqa: BLE001
            return None

    def _path(self, a, b):
        """(points, length_m, end_gap_m) of the navmesh shortest path, or None."""
        t0 = time.perf_counter()
        p = self.navmesh.query_shortest_path(start_pos=self._f3(*a), end_pos=self._f3(*b))
        self._prof["path_s"] += time.perf_counter() - t0
        self._prof["path_calls"] += 1
        if p is None:
            return None
        pts = [(float(q[0]), float(q[1]), float(q[2])) for q in p.get_points()]
        if not pts:
            return None
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        gap = math.hypot(pts[-1][0] - b[0], pts[-1][1] - b[1])
        return pts, length, gap

    def _spawn_points(self) -> list[tuple[tuple[float, float, float], str]]:
        n = self.num_people
        out: list[tuple[tuple[float, float, float], str]] = []
        if self.spawn_mode == "dynanav":
            table = self._dynanav_table()
            out = [(tuple(table[i]), "dynanav_table") for i in range(min(n, len(table)))]
            if len(out) < n:
                self.build_info["errors"].append(
                    f"dynanav spawn table for {self.env} has {len(table)} points; people {len(out)}..{n - 1} "
                    "sampled with IRA's method on the 6.0.1 navmesh (deviation 9)")
        if self.spawn_mode == "near":
            return self._near_points()
        # IRA 0.7.19 (randomizer.py:285-292, 430-494; simulation.py:801-808): ONE stream named after
        # the randomizer class ("Character"), seeded once with global_seed + agent_id (the first
        # randomizer has agent_id -1), sequential query_random_point() draws for i = 0..n-1 (the
        # existing/table characters consume their draw too). IRA rejects a draw within 2 m of
        # `_existing_pos + agent_positions`, but its `_existing_pos.append` is commented out, so it
        # never separated characters from each other. The previous per-character streams
        # (set_random_seed(name_i, seed + i)) put 7 of 30 sampled hospital people within 1 m of
        # (5.1, 18.9), min pair 3.4 cm (people/check probe_z). A minimum separation (min_sep_m,
        # default 1.0 m; 0 = IRA's effective behaviour) keeps capsules from spawning inside each other.
        stream = "Character"
        self.inav.set_random_seed(stream, int(self.seed) - 1)
        taken = [xyz for xyz, _ in out]
        min_sep = float(getattr(self, "min_sep_m", 1.0))
        rejected = 0
        for i in range(n):
            for _attempt in range(1000):
                q = self.navmesh.query_random_point(stream)
                if isinstance(q, (tuple, list)) and q and not isinstance(q[0], (int, float)):
                    q = q[0]
                xyz = (float(q[0]), float(q[1]), float(q[2]))
                if i < len(out):          # a table point: the draw is consumed, the table wins
                    break
                if all(math.hypot(xyz[0] - t[0], xyz[1] - t[1]) >= min_sep for t in taken):
                    break
                rejected += 1
            if i >= len(out):
                out.append((xyz, f"navmesh_random(stream '{stream}', seed {int(self.seed) - 1})"))
                taken.append(xyz)
        self.build_info["spawn_sampler"] = {"stream": stream, "seed": int(self.seed) - 1,
                                            "min_sep_m": min_sep, "rejected_draws": rejected}
        return out

    def _near_points(self) -> list[tuple[tuple[float, float, float], str]]:
        cx, cy, cz = self.center
        r0, r1 = self.spawn_radius_m
        rng = random.Random(self.seed)
        out: list[tuple[tuple[float, float, float], str]] = []
        tries = 0
        while len(out) < self.num_people and tries < 400 * max(1, self.num_people):
            tries += 1
            r = math.sqrt(rng.uniform(r0 * r0, r1 * r1))
            th = rng.uniform(-self.spawn_sector_rad, self.spawn_sector_rad) + (self.spawn_heading_rad or 0.0)
            x, y = cx + r * math.cos(th), cy + r * math.sin(th)
            q = self._closest(x, y, cz)
            if q is None or math.hypot(q[0] - x, q[1] - y) > 0.3 or abs(q[2] - cz) > 0.5:
                continue
            if math.hypot(q[0] - cx, q[1] - cy) < r0 - 0.05:
                continue
            if any(math.hypot(q[0] - p[0], q[1] - p[1]) < self.spawn_spacing_m for p, _ in out):
                continue
            path = self._path((cx, cy, cz), q)
            if path is None or path[2] > 0.3:   # same connected floor as the centre
                continue
            out.append((q, "near"))
        if len(out) < self.num_people:
            raise RuntimeError(f"people: found {len(out)}/{self.num_people} 'near' spawn points in {tries} tries")
        self.build_info["near_tries"] = tries
        return out

    def _provisional(self, i: int) -> tuple[float, float, float]:
        """Where an agent waits, before play, until the navmesh exists to sample its spawn."""
        bx, by, bz = self.center if self.center is not None else (0.0, 0.0, 0.0)
        return (bx + 0.6 * (i % 10), by + 0.6 * (i // 10), bz)

    def _move_agents(self, spawns) -> None:
        """Before play: set each provisional agent's prim translate to its sampled spawn."""
        from pxr import Gf, UsdGeom

        for p, (xyz, src) in zip(self.people, spawns):
            if p.spawn_source != "provisional":
                continue
            prim = self.stage.GetPrimAtPath(p.prim_path)
            ops = [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                   if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            if ops:
                ops[0].Set(Gf.Vec3d(*xyz))
            else:
                UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*xyz))
            p.spawn, p.spawn_source = tuple(xyz), src
        self._update(2)

    def _create_agents(self, spawns) -> None:
        import NavSchema
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom

        UsdGeom.Scope.Define(self.stage, self.root)
        names = character_names(self.num_people)
        lines = generate_character_commands(SCENE_TYPE[self.env], self.num_people, DYNANAV_NUM_COMMANDS, self.seed)
        cmds = parse_commands(lines)
        self.command_lines = lines
        if self.env.startswith("outdoor") and self.spawn_mode == "dynanav":
            baked = json.loads((ASSET_DIR / "outdoor_small_baked.json").read_text())["characters"]
            assets = [asset_url(c["asset"]) for c in baked]
        else:
            order = dynanav_asset_order(self.seed)
            assets = [asset_url(order[i % len(order)]) for i in range(self.num_people)]
        t0 = time.time()
        for i, name in enumerate(names):
            (x, y, z), src = spawns[i]
            path = f"{self.root}/{name}"
            omni.kit.commands.execute(
                "CreateBehaviorAgentCommand", character_prim_path=Sdf.Path(path), character_asset_url=assets[i],
                motion_library_prim_path=Sdf.Path(f"{self.root}/HumanMotionLibrary"),
                motion_library_asset_url=MOTION_LIBRARY_URL, motion_library_skeleton_rig="Human",
                position=Gf.Vec3d(x, y, z))
            omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=Sdf.Path(path), api=NavSchema.NavMeshExcludeAPI)
            self._update()
            self.people.append(Person(name=name, asset=assets[i], spawn=(x, y, z), spawn_source=src,
                                      commands=cmds.get(name, []), prim_path=path))
        ctx = omni.usd.get_context()
        for k in range(1500):   # payloads stream from the CDN
            self._update()
            if k % 50 == 49 and not ctx.get_stage_loading_status()[2]:
                break
        for p in self.people:
            sk = next((q for q in Usd.PrimRange(self.stage.GetPrimAtPath(p.prim_path))
                       if q.GetTypeName() == "SkelRoot"), None)
            p.skel = str(sk.GetPath()) if sk else None
        missing = [p.name for p in self.people if p.skel is None]
        self.build_info["agents_create_s"] = round(time.time() - t0, 1)
        self.build_info["skelroots_found"] = len(self.people) - len(missing)
        if missing:
            raise RuntimeError(f"people: no SkelRoot for {missing} (asset failed to load?)")

    def _validate(self) -> None:
        """Spawn-on-navmesh and per-GoTo reachability, up front (DynaNav skipped them silently)."""
        rep = {}
        for p in self.people:
            q = self._closest(*p.spawn)
            snap = None if q is None else round(math.hypot(q[0] - p.spawn[0], q[1] - p.spawn[1]), 3)
            gotos = []
            for c in p.commands:
                if c[0] == "GoTo":
                    tgt = (float(c[1]), float(c[2]), float(c[3]))
                    r = self._path(p.spawn, tgt)
                    gotos.append("none" if r is None else ("ok" if r[2] <= 0.5 else f"partial {r[2]:.1f} m"))
            rep[p.name] = {"spawn": [round(v, 3) for v in p.spawn], "spawn_source": p.spawn_source,
                           "spawn_to_navmesh_m": snap, "goto_from_spawn": gotos}
        self.build_info["validate"] = rep

    # ------------------------------------------------------------------- attach
    def _robot_bodies(self) -> list[str]:
        from pxr import Usd, UsdPhysics

        root = self.stage.GetPrimAtPath(self.robot_prefix)
        if not root or not root.IsValid():
            return []
        return [str(q.GetPath()) for q in Usd.PrimRange(root) if q.HasAPI(UsdPhysics.RigidBodyAPI)]

    def _try_attach(self) -> bool:
        import omni.anim.behavior.core as bh

        iface = bh.acquire_interface()
        for p in self.people:
            if p.agent is None and p.skel:
                p.agent = iface.get_agent(p.skel)
        self._attach_wait += 1
        live = sum(p.agent is not None for p in self.people)
        if live < len(self.people) and self._attach_wait < 900:
            return False
        if live == 0:
            raise RuntimeError(f"people: 0/{len(self.people)} behavior agents came alive after "
                               f"{self._attach_wait} updates (is there a navmesh?)")
        if live < len(self.people):
            self.build_info["errors"].append(
                f"only {live}/{len(self.people)} agents alive: "
                f"{[p.name for p in self.people if p.agent is None]}")
        for p in self.people:
            if p.agent is not None:
                p.agent.set_speed(self.speed_mps)
            p.radius = self._capsule_radius(p)
        if self.robot_interaction == "ghost":
            self._apply_ghost_filter()
        self.attached = True
        self.build_info["attach_updates"] = self._attach_wait
        self.log(f"attached: {live}/{len(self.people)} agents live after {self._attach_wait} updates; "
                 f"ghost={self.ghost_info or 'off'}")
        return True

    def _capsule_radius(self, p: Person) -> float:
        try:
            from pxr import UsdGeom

            prim = self.stage.GetPrimAtPath(f"{p.skel}/Physics/BodyCollider")
            r = UsdGeom.Capsule(prim).GetRadiusAttr().Get()
            return float(r) if r else CAPSULE_RADIUS_M
        except Exception:  # noqa: BLE001
            return CAPSULE_RADIUS_M

    def _apply_ghost_filter(self) -> None:
        """Deviation 5: no contact with the robot, still visible to scene queries.

        Authored as raw Sdf specs on the session layer inside ONE Sdf.ChangeBlock. The earlier
        UsdPhysics API version (Apply + one AddTarget per robot body) sent omni.physx
        n_people * (1 + n_bodies) separate change notices; the first app.update() after it grew
        GPU0 by ~0.37 GiB per person (10 people 6.5 -> 10.3 GiB, 40 people -> 22 GiB) and the
        authoring stalled 8 s / 64 s. Batched (people/check P4, P8): 0.2 s, GPU0 flat, and a walker
        teleported onto the parked AlohaMini still moves it 0.000 m.
        """
        from pxr import Sdf

        bodies = self._robot_bodies()
        layer = self.stage.GetSessionLayer()
        ok, missing = 0, []
        t0 = time.time()
        with Sdf.ChangeBlock():
            for p in self.people:
                path = f"{p.skel}/Physics/BodyCollider" if p.skel else None
                if path is None or not self.stage.GetPrimAtPath(path).IsValid():
                    missing.append(p.name)
                    continue
                spec = Sdf.CreatePrimInLayer(layer, path)
                lo = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
                items = list(lo.prependedItems)
                if "PhysicsFilteredPairsAPI" not in items:
                    items.append("PhysicsFilteredPairsAPI")
                lo.prependedItems = items
                spec.SetInfo("apiSchemas", lo)
                rel = spec.relationships.get("physics:filteredPairs") or \
                    Sdf.RelationshipSpec(spec, "physics:filteredPairs", False)
                rel.targetPathList.explicitItems = [Sdf.Path(b) for b in bodies]
                ok += 1
        self.ghost_info = {"robot_bodies": len(bodies), "capsules_filtered": ok, "capsules_missing": missing,
                           "author_s": round(time.time() - t0, 3), "impl": "sdf_changeblock"}
        if not bodies or missing:
            self.build_info["errors"].append(f"ghost filter incomplete: {self.ghost_info}")

    def attach(self, max_updates: int = 900) -> PeopleSim:
        """AFTER timeline.play(), BEFORE the first episode step: wait for the agents, apply the
        ghost filter, start no clock. The filter authoring is a one-off stall (1.7 s / 3 people,
        7.9 s / 10, 60-64 s / 40) that would otherwise land inside the first step() of the first
        episode. step() still attaches lazily if this was not called.

        DETERMINISM IS CONDITIONAL ON WHERE THIS IS CALLED. Same seed, 3 people, 12 s, robot parked:
        lazy attach with cameras (run A) == lazy attach without cameras (Z2) to 2 mm, but attaching
        BEFORE the first update of the loop (Z, Z3) moves the two walkers that interact (within
        ~1.7 m of each other at spawn) by 0.26 / 0.48 m, and the one that does not by 0.0001 m. So
        pick one call sequence and keep it for every run you intend to compare."""
        if not self.people or self.attached:
            return self
        import omni.timeline

        for _ in range(max_updates):
            if not omni.timeline.get_timeline_interface().is_playing():
                raise RuntimeError("people.attach(): the timeline is not playing")
            if self._try_attach():
                return self
            self._update()
        raise RuntimeError(f"people.attach(): agents not alive after {max_updates} updates")

    # --------------------------------------------------------------- sequencer
    def _start_next(self, p: Person) -> None:
        a = p.agent
        while True:
            p.ci += 1
            if p.ci >= len(p.commands):          # number_of_loop = 0: idle forever
                p.task, p.t_end, p.cmd, p.goto_target = a.idle(), None, "DONE", None
                p.log.append((round(self.t, 2), "DONE"))
                return
            c = p.commands[p.ci]
            p.cmd_t0 = self.t
            if c[0] == "GoTo":
                tgt = (float(c[1]), float(c[2]), float(c[3]))
                q = a.get_world_translation()
                cur = (float(q[0]), float(q[1]), float(q[2]))
                r = self._path(cur, tgt)
                p.goto["issued"] += 1
                if r is None:                    # omni.anim.people: no path -> completes at once
                    p.goto["no_path"] += 1
                    p.log.append((round(self.t, 2), f"GoTo {tgt[0]:.2f},{tgt[1]:.2f} SKIPPED no path"))
                    continue
                pts, length, gap = r
                # ALWAYS walk to the navmesh end of the path, never the raw landmark. A landmark
                # only 0.215 m off the 6.0.1 navmesh (hospital (-33.15, 3.12), 12 of 136 hospital
                # GoTos) makes agent.move_to(raw) report SUCCEEDED after one update without moving,
                # while move_to(pts[-1]) walks it (people/check probe_goto). omni.anim.people walked
                # the returned path points, i.e. the snapped end, too.
                goal = pts[-1]
                note = f"path {length:.1f} m"
                if gap > 0.5:                    # deviation 8
                    p.goto["partial"] += 1
                    note += f", PARTIAL: reachable end {gap:.1f} m short"
                p.task = a.move_to(target=self._f3(*goal))
                p.t_end, p.cmd, p.goto_target = None, "GoTo", goal
                p.log.append((round(self.t, 2), f"GoTo {tgt[0]:.2f},{tgt[1]:.2f} ({note})"))
                return
            d = float(c[1])                      # Idle d / LookAround d (deviation 3)
            p.task, p.t_end, p.cmd, p.goto_target = a.idle(), self.t + d, c[0], None
            p.log.append((round(self.t, 2), f"{c[0]} {d}"))
            return

    def _advance(self, p: Person) -> None:
        import omni.anim.behavior.core as bh

        a = p.agent
        if p.t_end is not None:
            if self.t >= p.t_end:
                a.cancel_task(p.task)
                self._start_next(p)
        elif p.cmd == "GoTo":
            st = a.get_task_status(p.task)
            if st != bh.BehaviorTaskStatus.RUNNING:
                g = p.goto_target
                gap = math.hypot(p.xyz[0] - g[0], p.xyz[1] - g[1]) if (g and p.xyz) else float("nan")
                name = str(st).split(".")[-1]
                if name != "SUCCEEDED":
                    p.goto["failed"] += 1
                elif gap > 0.5 and self.t - p.cmd_t0 < 0.1:
                    # "succeeded" within a few updates, far from the goal: the agent never walked.
                    # Count it as what it is instead of hiding it under 'short'.
                    p.goto["instant_success"] = p.goto.get("instant_success", 0) + 1
                elif gap <= 0.5:
                    p.goto["reached"] += 1
                else:
                    p.goto["short"] += 1
                p.log.append((round(self.t, 2), f"  -> {name} after {self.t - p.cmd_t0:.2f} s, {gap:.2f} m from goal"))
                self._start_next(p)

    # --------------------------------------------------------------------- step
    def step(self, dt: float | None = None, robot_xy: tuple[float, float] | None = None,
             robot_yaw: float | None = None) -> None:
        """Once per physics step, AFTER kit.update(). Sim time only, never wall time."""
        if not self.people:
            return
        import omni.timeline

        ts = time.perf_counter()
        if not omni.timeline.get_timeline_interface().is_playing():
            if self.attached:   # capsules and agents are re-created at the next play
                self.attached, self._attach_wait, self._needs_start = False, 0, True
                for p in self.people:
                    p.agent = None
            return
        # The clock starts when the command lists start (episode t = 0, as in DynaNav where
        # characters and robot both begin at play); waiting for agents costs no episode time.
        if not self.attached:
            if not self._try_attach():
                return
        if self._needs_start:
            self._restart_commands()
        else:
            self.t += self.dt if dt is None else dt
        if robot_xy is None or robot_yaw is None:
            rx, ry, ryaw = self._robot_pose_usd()
            robot_xy = robot_xy or (rx, ry)
            robot_yaw = ryaw if robot_yaw is None else robot_yaw
        step_dt = self.dt if dt is None else dt
        self.m["steps"] += 1
        cy, sy = math.cos(robot_yaw), math.sin(robot_yaw)
        hx, hy = self.robot_half_extents_m
        for p in self.people:
            a = p.agent
            if a is None:
                continue
            t_api = time.perf_counter()
            q = a.get_world_translation()
            xyz = (float(q[0]), float(q[1]), float(q[2]))
            if p.xyz is not None:
                dx, dy = xyz[0] - p.xyz[0], xyz[1] - p.xyz[1]
                p.vel = (dx / step_dt, dy / step_dt)
                ds = math.hypot(dx, dy)
                p.path_m += ds
                if ds / step_dt > 0.2:
                    p.moving_speeds.append(ds / step_dt)
            p.xyz = xyz
            try:
                f = a.get_facing_direction()
                p.yaw = math.atan2(float(f[1]), float(f[0]))
            except Exception:  # noqa: BLE001
                pass
            self._prof["api_s"] += time.perf_counter() - t_api
            # metrics against the robot base origin (DynaNav's chassis_link analogue)
            ddx, ddy = xyz[0] - robot_xy[0], xyz[1] - robot_xy[1]
            d = math.hypot(ddx, ddy)
            p.min_dist_robot_m = min(p.min_dist_robot_m, d)
            if d < self.m["min_dist_m"]:
                self.m.update(min_dist_m=d, min_dist_t=round(self.t, 3), min_dist_person=p.name)
            if d < HUMAN_COLLISION_RADIUS_M:
                if not self.m["human_collision"]:
                    self.m.update(human_collision=True, human_collision_t=round(self.t, 3),
                                  human_collision_person=p.name)
                if p.name not in self.m["human_collision_people"]:
                    self.m["human_collision_people"].append(p.name)
            bx, by = cy * ddx + sy * ddy, -sy * ddx + cy * ddy     # robot frame
            if abs(bx) <= hx + CAPSULE_RADIUS_M and abs(by) <= hy + CAPSULE_RADIUS_M:
                self.m["footprint_contact_steps"] += 1
                if p.name not in self.m["footprint_contact_people"]:
                    self.m["footprint_contact_people"].append(p.name)
            t_adv = time.perf_counter()
            self._advance(p)
            self._prof["advance_s"] += time.perf_counter() - t_adv
        self._prof["steps"] += 1
        dmin = min((math.hypot(p.xyz[0] - robot_xy[0], p.xyz[1] - robot_xy[1])
                    for p in self.people if p.xyz), default=float("inf"))
        for k in self.m["steps_within"]:
            if dmin < float(k):
                self.m["steps_within"][k] += 1
        self.m["step_ms"].append((time.perf_counter() - ts) * 1000.0)

    def _restart_commands(self) -> None:
        for p in self.people:
            if p.agent is not None:
                p.ci, p.task, p.t_end, p.cmd, p.goto_target = -1, None, None, "", None
                self._start_next(p)
        self._needs_start = False
        self._agents_ran = True

    def _robot_pose_usd(self) -> tuple[float, float, float]:
        import omni.usd
        from pxr import Gf

        m = omni.usd.get_world_transform_matrix(self.stage.GetPrimAtPath(self.robot_base_path))
        t = m.ExtractTranslation()
        fwd = m.TransformDir(Gf.Vec3d(1, 0, 0))
        return float(t[0]), float(t[1]), math.atan2(float(fwd[1]), float(fwd[0]))

    # ----------------------------------------------------------------- queries
    def positions(self) -> list[tuple[str, float, float, float, float, float]]:
        """[(name, x, y, z, vx, vy)] from the behavior API, world frame, metres."""
        return [(p.name, *p.xyz, *p.vel) for p in self.people if p.xyz is not None]

    def snapshot(self) -> list[dict]:
        return [{"name": p.name, "xyz": [round(v, 3) for v in p.xyz] if p.xyz else None,
                 "vel": [round(v, 3) for v in p.vel], "yaw": round(p.yaw, 3), "cmd_index": p.ci, "cmd": p.cmd}
                for p in self.people]

    def ray_distances_2d(self, origin_xy, angles, max_range: float = 12.0, z: float = 0.30):
        """Distance along each horizontal ray (world bearing, radians) to the nearest person.

        An analytic ray-vs-vertical-capsule test at height z (a circle of the person's capsule
        radius around the live position from the behavior API). Returns (dist, who): dist[i] is
        inf where nothing is hit within max_range, who[i] the person index or -1.

        WHY THIS EXISTS. PhysX scene queries are NOT a reliable way to sense these people:
        in run G/H two of ten walkers (Character, Character_04) were hit by raycast_closest at
        their SPAWN POINT for the whole run and not at their live position, while
        get_rigidbody_transformation() put every capsule exactly at the agent (gap 0.00 m, all
        ten, t = 1/10/20 s), with the ghost filter on or off. The query structure holds a stale
        pose for some kinematic capsules. So: a guard / lidar should take people from here and
        DROP any PhysX hit whose prim is under `root` (a phantom at a spawn point is a false
        obstacle; a missing live hit is a blind spot).
        """
        import numpy as np

        ang = np.asarray(angles, dtype=float)
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        best = np.full(len(ang), np.inf)
        who = np.full(len(ang), -1, dtype=int)
        for k, p in enumerate(self.people):
            if p.xyz is None or not (0.0 <= z - p.xyz[2] <= CAPSULE_TOP_M):
                continue
            c = np.array([p.xyz[0] - ox, p.xyz[1] - oy])
            cc = float(c @ c) - p.radius * p.radius
            b = d @ c
            disc = b * b - cc
            ok = disc >= 0.0
            root = np.sqrt(np.where(ok, disc, 0.0))
            t_in, t_out = b - root, b + root
            t = np.where(t_in >= 0.0, t_in, 0.0)          # origin inside the circle: distance 0
            ok &= t_out >= 0.0
            ok &= t <= max_range
            upd = ok & (t < best)
            best[upd] = t[upd]
            who[upd] = k
        return best, who

    def is_person_prim(self, prim_path: str) -> bool:
        """True for a PhysX hit that belongs to one of these people (to be dropped, see above)."""
        return str(prim_path).startswith(self.root + "/")

    def ray_visibility(self, z: float = 0.30, standoff_m: float = 1.0) -> dict:
        """raycast_closest at height z from 4 sides of each person, standoff_m away.

        Answers "do the guard / lidar see this person": a hit counts only when the first
        thing hit lies under that person's own prim.
        """
        import omni.physx

        sq = omni.physx.get_physx_scene_query_interface()
        out = {}
        for p in self.people:
            if p.xyz is None:
                continue
            hits = []
            for k in range(4):
                th = k * math.pi / 2
                d = (math.cos(th), math.sin(th))
                o = (p.xyz[0] - d[0] * standoff_m, p.xyz[1] - d[1] * standoff_m, p.xyz[2] + z)
                h = sq.raycast_closest(self._f3(*o), self._f3(d[0], d[1], 0.0), standoff_m + 1.0)
                if h and h.get("hit"):
                    prim = str(h.get("collision", "") or h.get("rigidBody", ""))
                    hits.append({"own": prim.startswith(p.prim_path + "/"), "prim": prim,
                                 "dist": round(float(h["distance"]), 3)})
                else:
                    hits.append({"own": False, "prim": None, "dist": None})
            rec = {"own_hits": sum(h["own"] for h in hits), "rays": hits}
            # the same four rays, analytically (live behavior-API pose). Where PhysX hit this person's
            # own capsule the two distances must agree; where PhysX's query structure is stale the
            # analytic one still hits at standoff_m - radius.
            errs, ana = [], []
            for k in range(4):
                th = k * math.pi / 2
                o = (p.xyz[0] - math.cos(th) * standoff_m, p.xyz[1] - math.sin(th) * standoff_m)
                dist, _ = self.ray_distances_2d(o, [th], standoff_m + 1.0, z=p.xyz[2] + z)
                ana.append(None if not math.isfinite(float(dist[0])) else round(float(dist[0]), 4))
                if hits[k]["own"] and ana[-1] is not None:
                    errs.append(round(abs(ana[-1] - hits[k]["dist"]), 4))
            rec["analytic_dist_m"] = ana
            rec["analytic_vs_physx_err_m"] = errs
            if math.hypot(p.xyz[0] - p.spawn[0], p.xyz[1] - p.spawn[1]) > 1.5:
                # A capsule that did NOT follow its agent is still hit where the agent started
                # (run G, Character_04: ring distance == distance to its spawn point for 15 s).
                n_sp = 0
                for k in range(4):
                    th = k * math.pi / 2
                    d = (math.cos(th), math.sin(th))
                    o = (p.spawn[0] - d[0] * standoff_m, p.spawn[1] - d[1] * standoff_m, p.spawn[2] + z)
                    h = sq.raycast_closest(self._f3(*o), self._f3(d[0], d[1], 0.0), standoff_m + 1.0)
                    if h and h.get("hit"):
                        prim = str(h.get("collision", "") or h.get("rigidBody", ""))
                        n_sp += prim.startswith(p.prim_path + "/")
                rec["own_hits_at_spawn"] = n_sp
            out[p.name] = rec
        return out

    def capsule_report(self) -> dict:
        """Per person: the PhysX pose of <SkelRoot>/Physics/BodyCollider next to the behavior
        agent's pose, plus the USD attributes that decide whether the capsule is simulated.

        Written after run G: two of ten walkers had a capsule that stayed at the spawn point
        while the agent walked away (ray_visibility 'own_hits_at_spawn'), with or without the ghost
        filter. This reads the actor's pose from PhysX directly instead of inferring it from rays.
        """
        import omni.physx
        from pxr import UsdGeom

        px = omni.physx.get_physx_interface()
        out = {}
        for p in self.people:
            path = f"{p.skel}/Physics/BodyCollider" if p.skel else None
            rec: dict = {"agent_xyz": [round(v, 3) for v in p.xyz] if p.xyz else None}
            if path:
                try:
                    t = px.get_rigidbody_transformation(path)
                    rec["physx"] = {k: (repr(v) if not isinstance(v, (bool, int, float)) else v)
                                    for k, v in t.items()} if isinstance(t, dict) else repr(t)
                except Exception as e:  # noqa: BLE001
                    rec["physx_err"] = repr(e)
                prim = self.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    rec["usd_attrs"] = {n: (repr(prim.GetAttribute(n).Get()) if prim.GetAttribute(n).IsValid() else None)
                                        for n in ("physics:rigidBodyEnabled", "physics:kinematicEnabled",
                                                  "physics:collisionEnabled", "physics:simulationOwner",
                                                  "physxRigidBody:disableGravity")}
                    rec["usd_xformOps"] = [str(x) for x in (UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get() or [])]
                    rec["usd_apis"] = [str(a) for a in prim.GetAppliedSchemas()]
            out[p.name] = rec
        return out

    def metrics(self) -> dict:
        per = {}
        for p in self.people:
            per[p.name] = {
                "asset": p.asset.rsplit("/", 2)[-2], "spawn": [round(v, 3) for v in p.spawn],
                "spawn_source": p.spawn_source, "path_m": round(p.path_m, 2),
                "median_walk_speed_mps": round(statistics.median(p.moving_speeds), 3) if p.moving_speeds else None,
                "min_dist_robot_m": round(p.min_dist_robot_m, 3) if p.min_dist_robot_m < 1e9 else None,
                "goto": dict(p.goto), "cmd_index": p.ci, "log": p.log}
        sm = self.m["step_ms"]
        steps = max(1, self.m["steps"])
        return {
            "env": self.env, "num_people": self.num_people, "seed": self.seed, "spawn": self.spawn_mode,
            "robot_interaction": self.robot_interaction, "speed_set_mps": self.speed_mps,
            "sim_time_s": round(self.t, 3),
            "human_collision": self.m["human_collision"], "human_collision_t": self.m["human_collision_t"],
            "human_collision_person": self.m["human_collision_person"],
            "human_collision_people": self.m["human_collision_people"],
            "human_collision_rule": "DynaNav benchmark.py:649-702: 2-D robot base origin within 0.2 m of a person root",
            "min_dist_m": round(self.m["min_dist_m"], 3) if self.m["min_dist_m"] < 1e9 else None,
            "min_dist_t": self.m["min_dist_t"], "min_dist_person": self.m["min_dist_person"],
            "frac_within": {k: round(v / steps, 4) for k, v in self.m["steps_within"].items()},
            "footprint_contact_steps": self.m["footprint_contact_steps"],
            "footprint_contact_people": self.m["footprint_contact_people"],
            "people_step_ms_mean": round(sum(sm) / len(sm), 3) if sm else None,
            "people_step_ms_p50": round(sorted(sm)[len(sm) // 2], 3) if sm else None,
            "people_step_ms_max": round(max(sm), 1) if sm else None,
            "people_step_profile_ms_per_step": {
                "behavior_api_reads": round(1000 * self._prof["api_s"] / max(1, self._prof["steps"]), 3),
                "sequencer_advance_incl_path": round(1000 * self._prof["advance_s"] / max(1, self._prof["steps"]), 3),
                "navmesh_path_queries_total_s": round(self._prof["path_s"], 3),
                "navmesh_path_queries": self._prof["path_calls"]},
            "ghost": self.ghost_info, "build": self.build_info, "people": per,
        }

    # ---------------------------------------------------------- reset/teardown
    def reset(self) -> None:
        """Episode boundary in a long-lived process: everyone back to spawn, lists restart.

        Agents that have not run since attach (or since the last reset) are left alone: they are
        at their spawn, at the speed attach() set, with no task. Calling agent.reset() on 30 such
        agents in the frame before their first update crashed Kit in libomni.anim.behavior.core
        (exit 139, office 30 walkers, 3 of 3 runner episodes; the same crowd ran clean when reset
        came after 20 s of walking). The runner resets at every episode start, i.e. right after
        attach on its first episode, which is exactly that frame.
        """
        if self.attached and not self._agents_ran:
            self.log("reset: agents have not run since attach/last reset -- left at spawn")
        elif self.attached:
            for p in self.people:
                if p.agent is None:
                    continue
                if p.task is not None:
                    try:
                        p.agent.cancel_task(p.task)
                    except Exception:  # noqa: BLE001
                        pass
                p.agent.reset(target=self._f3(*p.spawn), facing=self._f3(1.0, 0.0, 0.0))  # DynaNav yaw 0
                # agent.reset() restores the behavior default speed (1.5 m/s): walkers ran at
                # 1.49-1.54 m/s after every reset (people/check P1); set it again.
                p.agent.set_speed(self.speed_mps)
        self._agents_ran = False
        for p in self.people:
            p.xyz, p.vel, p.path_m, p.moving_speeds = None, (0.0, 0.0), 0.0, []
            p.min_dist_robot_m = float("inf")
            p.goto = {k: 0 for k in p.goto}
            p.log = []
        self.t = 0.0
        self._needs_start = True
        self._reset_metrics()

    def teardown(self, remove_prims: bool = False) -> None:
        """Stop sequencing and disable the agents. Prims are removed only with the timeline
        stopped (removing a live agent under a playing behavior system is untested)."""
        for p in self.people:
            if p.agent is not None:
                try:
                    if p.task is not None:
                        p.agent.cancel_task(p.task)
                    p.agent.set_enabled(False)
                except Exception:  # noqa: BLE001
                    pass
        if remove_prims:
            import omni.timeline

            if omni.timeline.get_timeline_interface().is_playing():
                self.log("teardown: timeline playing, prims left in place")
            else:
                prev = self._session()
                try:
                    self.stage.RemovePrim(self.root)
                finally:
                    self.stage.SetEditTarget(prev)
        self.people = []
        self.attached = False


PEOPLE_EXTENSIONS = ("omni.anim.navigation.core", "omni.anim.behavior.core")


def enable_extensions(app=None) -> None:
    """Enable the navigation + behavior extensions. Call BEFORE opening the stage.

    navstage1 (worked) enabled them before open_stage; try_people run A enabled them after
    the stage was open AND pre-baked a navmesh before creating agents, and Kit crashed in
    libomni.anim.behavior.core at the first update after play. build() enables them too
    (idempotent), but the runner should call this first.
    """
    import carb
    import omni.kit.app

    # The navmesh caches are switched off BEFORE the extension starts: setting them after it
    # is enabled (people.build() in runs A/B/D) still gave "0.06 s bake, getters 0.0". See
    # nav/tools/probe_navmesh_cache.py for the measurement (--when before|after).
    st = carb.settings.get_settings()
    for k in ("enabled", "enableUjitso"):
        st.set(f"{NAV_SETTINGS}/cache/{k}", False)
    app = app or omni.kit.app.get_app()
    em = app.get_extension_manager()
    for ext in PEOPLE_EXTENSIONS:
        if not em.set_extension_enabled_immediate(ext, True):
            raise RuntimeError(f"people: could not enable {ext}")
    app.update()


def setup(stage, env_name: str, episode_name: str | None = None, num_people: int | None = None,
          seed: int = DYNANAV_SEED, **kw) -> PeopleSim:
    """Build a crowd BEFORE timeline.play(). num_people defaults to DynaNav's count for the
    episode (episodes_full.yaml); 0 builds nothing and every call on the handle is a no-op."""
    if num_people is None:
        num_people = episode_num_people(episode_name)
        if num_people is None:
            raise ValueError(f"people: no num_people for episode {episode_name!r}; pass num_people=")
    return PeopleSim(stage, env_name, num_people, seed=seed, **kw).build()
