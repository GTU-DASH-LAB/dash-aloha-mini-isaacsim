"""Pedestrians in the AlohaMini navigation runner: the glue between run_navigation.py and people.py.

OFF unless NAV_PEOPLE=1, so every number recorded without it is unchanged. With it, an episode runs
DynaNav's own crowd for that episode (episodes_full.yaml `num_people`, DynaNav's seed, spawn points
and command lists -- see people.py for what is reproduced exactly and every deviation 6.0.1 forces).

WHAT THE RUNNER CALLS, IN ORDER (run_navigation.NavigationRunner):
  PeopleLayer.maybe(episode)         None unless NAV_PEOPLE=1 and the episode has people
  .before_stage(kit)                 after SimulationApp(...), BEFORE ctx.open_stage(): enables the
                                     navigation + behavior extensions (people.enable_extensions --
                                     enabling them after the stage is open crashed Kit, people.py)
  .before_play(stage)                after the stage and scene colliders, BEFORE timeline.play():
                                     builds the crowd (people.setup); robot_interaction="ghost"
  .after_play()                      after play and the articulation exists: waits for the agents and
                                     applies the ghost filter (one Sdf.ChangeBlock, ~0.2 s)
  .install_guard(guard)              the CollisionGuard's fan sees people ANALYTICALLY (below)
  .episode_start()                   every run_episode, before the first step: everyone back to
                                     spawn, command lists restart at t = 0 (DynaNav: people and robot
                                     start together)
  .step(dt, robot_xy, robot_yaw)     every control step, AFTER kit.update() and the fault check
  .summary()                         -> EpisodeResult.people (metrics incl. DynaNav human_collision)

SENSING. PhysX scene queries do not reliably see the 6.0.1 walkers: in people/prototype runs G/H
two of ten capsules were hit by raycast_closest only at their SPAWN point for the whole run. So the
guard drops every PhysX hit under the people root (a phantom at a spawn point would be a false
obstacle) and adds people.ray_distances_2d() along the same fan bearings, at the same height, in
the same centre-relative frame the guard's thresholds are written in. The lidar (NAV_LIDAR=1) is
NOT patched: the shipping profile runs it off; a lidar run with people would see phantoms and miss
walkers -- install_guard() records that.

CONTACT. kinematic_contact.py ignores /World/People and the capsules are ghost-filtered against the
robot's bodies, so the robot and people pass through each other physically, as DynaNav's people
(no physics APIs at all) did. A pass-through is what DynaNav scores as `human_collision`: robot base
origin within 0.2 m (2-D) of a person root, latched -- people.py computes it every step.
"""

from __future__ import annotations

import math
import os
import time

ENV_FLAG = "NAV_PEOPLE"
ENV_NUM = "NAV_PEOPLE_NUM"          # optional: override the episode's num_people (0 = none)


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip() == "1"


class PeopleLayer:
    """One crowd for one runner process; reset at every episode."""

    def __init__(self, episode, num_people: int):
        self.episode_name = episode.name
        from episode import env_name

        self.env = env_name(episode.scene)
        self.num_people = int(num_people)
        self.sim = None                 # people.PeopleSim after before_play()
        self.info: dict = {"env": self.env, "num_people": self.num_people, "enabled_by": ENV_FLAG,
                           "guard": None, "lidar": "not patched (phantoms / blind spots if NAV_LIDAR=1)",
                           "errors": []}
        self._t_setup = 0.0

    # ------------------------------------------------------------------ construction
    @classmethod
    def maybe(cls, episode) -> "PeopleLayer | None":
        if not enabled():
            return None
        n = os.environ.get(ENV_NUM)
        if n is not None and n.strip() != "":
            num = int(n)
        else:
            import people

            num = people.episode_num_people(episode.name)
            if num is None:
                raise ValueError(f"{ENV_FLAG}=1 but {episode.name!r} has no num_people in "
                                 f"episodes_full.yaml; set {ENV_NUM}")
        if num <= 0:
            print(f"[people] {episode.name}: num_people = 0 -- running without people", flush=True)
            return None
        return cls(episode, num)

    def before_stage(self, kit) -> None:
        import people

        t0 = time.time()
        people.enable_extensions()
        self._t_setup += time.time() - t0

    def before_play(self, stage) -> None:
        import people

        t0 = time.time()
        self.sim = people.setup(stage, self.env, self.episode_name, num_people=self.num_people,
                                robot_interaction="ghost",
                                log=lambda m: print(f"[people] {m}", flush=True))
        self._t_setup += time.time() - t0
        print(f"[people] built {self.num_people} people for {self.episode_name} ({self.env}) "
              f"in {time.time() - t0:.1f} s", flush=True)

    def after_play(self) -> None:
        if self.sim is None:
            return
        t0 = time.time()
        self.sim.attach()
        self._t_setup += time.time() - t0
        self.info["ghost"] = dict(getattr(self.sim, "ghost_info", {}) or {})
        print(f"[people] attached in {time.time() - t0:.1f} s: {self.info['ghost']}", flush=True)

    # ------------------------------------------------------------------ sensing
    def install_guard(self, guard) -> None:
        """Wrap this guard INSTANCE's fan sampler (collision_guard.py is not edited)."""
        if self.sim is None or getattr(guard, "_people_wrapped", False):
            return
        sim = self.sim
        orig = guard._sample_fan
        stats = {"phantom_hits_dropped": 0, "people_hits_added": 0}

        def fan_with_people(position, travel):
            hits = orig(position, travel)
            kept = [h for h in hits if not sim.is_person_prim(h[2])]
            stats["phantom_hits_dropped"] += len(hits) - len(kept)
            n = guard.fan_rays
            angles = [travel + (((i / (n - 1)) * 2.0 - 1.0) if n > 1 else 0.0) * guard.fan_half_angle
                      for i in range(n)]
            # The fan casts slow_distance_m from the chassis EDGE and reports centre-relative
            # distances; the analytic rays start at the centre, so the same reach is r + slow.
            reach = guard.slow_distance_m + guard.chassis_radius_m
            dist, who = sim.ray_distances_2d(position[:2], angles, max_range=reach,
                                             z=float(position[2]) + guard.ray_height_m)
            for a, d, k in zip(angles, dist, who):
                if k >= 0 and math.isfinite(float(d)):
                    kept.append((a, float(d), f"{sim.root}/{sim.people[int(k)].name}"))
                    stats["people_hits_added"] += 1
            return kept

        guard._sample_fan = fan_with_people
        guard._people_wrapped = True
        self.info["guard"] = stats

    # ------------------------------------------------------------------ per episode / per step
    def episode_start(self) -> None:
        if self.sim is not None:
            self.sim.reset()

    def step(self, dt: float, robot_xy, robot_yaw: float | None = None) -> None:
        if self.sim is None:
            return
        try:
            self.sim.step(dt, robot_xy=(float(robot_xy[0]), float(robot_xy[1])), robot_yaw=robot_yaw)
        except Exception as exc:  # noqa: BLE001  (a crowd failure must not void the navigation run)
            if len(self.info["errors"]) < 8:
                self.info["errors"].append(repr(exc))
                print(f"[people] step error: {exc!r}", flush=True)

    def summary(self) -> dict | None:
        if self.sim is None:
            return None
        m = self.sim.metrics()
        m["layer"] = {**self.info, "setup_s": round(self._t_setup, 2)}
        return m
