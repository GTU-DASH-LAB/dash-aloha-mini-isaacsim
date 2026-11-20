"""Offline (no Kit) test of nav/sim/people.py's command sequencer, scoring and reset.

PeopleSim talks to Isaac Sim only through a handful of objects (behavior agents, the
navmesh, the timeline). This stubs exactly those with a kinematic fake -- agents that walk
straight at 1.1 m/s toward their move_to target -- and drives PeopleSim.step() for a few
hundred simulated seconds, so the logic that decides WHEN things happen can be checked in
milliseconds instead of a 3-minute Kit run:

  * commands run strictly in order; Idle/LookAround last their duration in SIM time;
  * a GoTo with no path completes at once (skipped); a PARTIAL path walks to its reachable end;
  * after the last command the person idles forever;
  * the DynaNav latch fires at < 0.2 m (2-D, base origin to person root) and not at 0.25 m;
  * min-distance / time-within bookkeeping;
  * reset() restarts every list from t = 0 and clears the metrics;
  * metrics() serialises (json) and carries the profile block.

It does NOT test anything about omni.anim.behavior.core itself -- that needs Kit.

    python3 nav/tools/check_people_sequencer.py        # exit 0 = all passed
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

# ------------------------------------------------------------------ stubs for the omni modules
DT = 1.0 / 60.0


class F3:
    def __init__(self, x, y, z):
        self.v = (x, y, z)

    def __getitem__(self, i):
        return self.v[i]


carb = types.ModuleType("carb")
carb.Float3 = F3
omni = types.ModuleType("omni")
tl = types.ModuleType("omni.timeline")


class _TL:
    playing = True

    def is_playing(self):
        return self.playing


_tl = _TL()
tl.get_timeline_interface = lambda: _tl


class Status:
    RUNNING = "BehaviorTaskStatus.RUNNING"
    SUCCEEDED = "BehaviorTaskStatus.SUCCEEDED"
    FAILED = "BehaviorTaskStatus.FAILED"


class _Enum(str):
    pass


bh = types.ModuleType("omni.anim.behavior.core")
bh.BehaviorTaskStatus = types.SimpleNamespace(RUNNING=_Enum(Status.RUNNING), SUCCEEDED=_Enum(Status.SUCCEEDED),
                                              FAILED=_Enum(Status.FAILED))


class FakeAgent:
    def __init__(self, xyz):
        self.p = list(xyz)
        self.target = None
        self.speed = 1.1
        self.tasks = {}
        self.n = 0
        self.face = (1.0, 0.0, 0.0)

    def set_speed(self, v):
        self.speed = v

    def get_world_translation(self):
        return F3(*self.p)

    def get_facing_direction(self):
        return F3(*self.face)

    def idle(self):
        self.target = None
        return self._new("idle")

    def move_to(self, target):
        self.target = (target[0], target[1])
        return self._new("move")

    def _new(self, kind):
        self.n += 1
        self.tasks[self.n] = kind
        return self.n

    def cancel_task(self, t):
        self.tasks[t] = "cancelled"

    def get_task_status(self, t):
        return bh.BehaviorTaskStatus.RUNNING if self.tasks.get(t) == "move" and self.target else \
            bh.BehaviorTaskStatus.SUCCEEDED

    def reset(self, target, facing):
        self.p = [target[0], target[1], target[2]]
        self.target = None
        return True

    def tick(self, dt):
        if self.target is None:
            return
        dx, dy = self.target[0] - self.p[0], self.target[1] - self.p[1]
        d = math.hypot(dx, dy)
        step = self.speed * dt
        if d <= step:
            self.p[0], self.p[1] = self.target
            self.target = None
        else:
            self.p[0] += dx / d * step
            self.p[1] += dy / d * step


class FakeIface:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, skel):
        return self.agents.get(skel)


AGENTS: dict[str, FakeAgent] = {}
bh.acquire_interface = lambda: FakeIface(AGENTS)
for name, mod in {"carb": carb, "omni": omni, "omni.timeline": tl, "omni.anim": types.ModuleType("omni.anim"),
                  "omni.anim.behavior": types.ModuleType("omni.anim.behavior"),
                  "omni.anim.behavior.core": bh}.items():
    sys.modules[name] = mod
omni.timeline = tl

import people  # noqa: E402


# ------------------------------------------------------------------ fake navmesh
class FakePath:
    def __init__(self, pts):
        self.pts = pts

    def get_points(self):
        return self.pts


class FakeNavmesh:
    """Reachable everywhere except x > 100 (no path) and y > 50 (path ends at y = 50)."""

    def query_shortest_path(self, start_pos, end_pos):
        a, b = start_pos, end_pos
        if b[0] > 100:
            return None
        end = (b[0], min(b[1], 50.0), b[2])
        return FakePath([(a[0], a[1], a[2]), end])


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def make(commands: dict[str, list[list[str]]], spawns: dict[str, tuple]):
    AGENTS.clear()
    sim = people.PeopleSim(None, "hospital", len(commands), robot_interaction="solid", log=lambda m: None)
    sim.navmesh = FakeNavmesh()
    for name, cmds in commands.items():
        sk = f"/World/People/{name}/ManRoot/x"
        AGENTS[sk] = FakeAgent(spawns[name])
        sim.people.append(people.Person(name=name, asset="a/b/c.usd", spawn=spawns[name], spawn_source="test",
                                        commands=cmds, prim_path=f"/World/People/{name}", skel=sk))
    sim.built = True
    return sim


def run(sim, seconds, robot=lambda t: (1000.0, 1000.0)):
    n = int(round(seconds * 60))
    for k in range(n):
        for a in AGENTS.values():
            a.tick(DT)
        sim.step(dt=DT, robot_xy=robot(k * DT), robot_yaw=0.0)


# 1. ordered execution, skip, partial, done-forever ---------------------------------------
cmds = {"Character": [["Idle", "2.0"], ["GoTo", "5.0", "0.0", "0.0", "_"], ["LookAround", "1.5"],
                      ["GoTo", "200.0", "0.0", "0.0", "_"],          # no path -> skipped
                      ["GoTo", "5.0", "80.0", "0.0", "_"],           # path ends at y=50 -> partial
                      ["Idle", "1.0"]]}
sim = make(cmds, {"Character": (0.0, 0.0, 0.0)})
run(sim, 120.0)
p = sim.people[0]
log = [m for _, m in p.log]
# `partial` is counted when the GoTo is issued and is a FLAG, not an outcome: that GoTo's outcome
# (reached, against the reachable end it was given) is counted too, so reached == 2 here.
check("goto counters: issued 3, reached 2 (one of them partial), no_path 1", p.goto["issued"] == 3 and
      p.goto["no_path"] == 1 and p.goto["partial"] == 1 and p.goto["reached"] == 2 and p.goto["short"] == 0,
      str(p.goto))
check("sequence ends in DONE and stays there", log[-1] == "DONE" and p.cmd == "DONE", log[-1])
t_goto = next(t for t, m in p.log if m.startswith("GoTo 5.00,0.00"))
check("Idle 2.0 lasts 2.0 s of sim time", abs(t_goto - 2.0) < 2 * DT, f"GoTo issued at {t_goto}")
t_arrive = next(t for t, m in p.log if "SUCCEEDED" in m)
check("5 m walk at 1.1 m/s takes ~4.5 s", abs((t_arrive - t_goto) - 5.0 / 1.1) < 0.1, f"{t_arrive - t_goto:.2f} s")
check("partial GoTo walked to the reachable end (y=50)", abs(AGENTS[p.skel].p[1] - 50.0) < 1e-6,
      f"final y {AGENTS[p.skel].p[1]:.3f}")
check("median walking speed ~1.1 m/s", abs(sim.metrics()["people"]["Character"]["median_walk_speed_mps"] - 1.1) < 0.02)

# 2. DynaNav latch: < 0.2 m fires, 0.25 m does not -------------------------------------------
sim = make({"Character": [["Idle", "30.0"]]}, {"Character": (3.0, 0.0, 0.0)})
run(sim, 5.0, robot=lambda t: (3.25, 0.0))
check("no latch at 0.25 m", not sim.m["human_collision"] and abs(sim.m["min_dist_m"] - 0.25) < 1e-6)
sim = make({"Character": [["Idle", "30.0"]]}, {"Character": (3.0, 0.0, 0.0)})
run(sim, 5.0, robot=lambda t: (3.0 + 0.5 * max(0.0, 1.0 - t) if t < 1.0 else 3.19, 0.0))
check("latch at 0.19 m", sim.m["human_collision"] and sim.m["human_collision_person"] == "Character")
# robot x = 3 + 0.5 * (1 - t): the distance is 0.2 m at t = 0.6 s
check("latch time is when it first crossed 0.2 m (0.6 s)", sim.m["human_collision_t"] is not None and
      0.6 <= sim.m["human_collision_t"] <= 0.65, str(sim.m["human_collision_t"]))

# 3. frac_within / min distance --------------------------------------------------------------
sim = make({"Character": [["Idle", "30.0"]]}, {"Character": (0.0, 0.0, 0.0)})
run(sim, 10.0, robot=lambda t: (1.5 if t < 5.0 else 6.0, 0.0))
fw = sim.metrics()["frac_within"]
check("frac_within 2.0 m ~ 50% of steps, 1.0 m = 0", abs(fw["2.0"] - 0.5) < 0.01 and fw["1.0"] == 0.0, str(fw))

# 4. reset ------------------------------------------------------------------------------------
sim = make(cmds, {"Character": (0.0, 0.0, 0.0)})
run(sim, 30.0, robot=lambda t: (0.1, 0.0))
check("before reset: latched and moved", sim.m["human_collision"] and sim.people[0].path_m > 1.0)
sim.attached = True
sim.reset()
check("reset clears metrics and restarts", not sim.m["human_collision"] and sim.t == 0.0 and sim._needs_start and
      sim.people[0].path_m == 0.0 and sim.people[0].goto["reached"] == 0)
run(sim, 3.0)
check("after reset the list restarts at command 0", sim.people[0].log[0][1] == "Idle 2.0" and
      sim.people[0].log[0][0] == 0.0, str(sim.people[0].log[:2]))

# 5. stop/play re-attach ----------------------------------------------------------------------
sim = make({"Character": [["Idle", "5.0"]]}, {"Character": (0.0, 0.0, 0.0)})
run(sim, 1.0)
_tl.playing = False
sim.step(dt=DT, robot_xy=(9.0, 9.0), robot_yaw=0.0)
check("timeline stop detaches agents", not sim.attached and sim.people[0].agent is None)
_tl.playing = True
run(sim, 1.0)
check("timeline play re-attaches and restarts the lists", sim.attached and sim._needs_start is False and
      sim.people[0].ci == 0)

# 7. analytic ray-vs-people test (the sensing path that does not depend on PhysX queries) ---------
import numpy as np  # noqa: E402

sim = make({"Character": [["Idle", "1.0"]], "Character_01": [["Idle", "1.0"]]},
           {"Character": (5.0, 0.0, 0.0), "Character_01": (9.0, 0.5, 0.0)})
for q in sim.people:
    q.xyz = q.spawn
r = people.CAPSULE_RADIUS_M
dist, who = sim.ray_distances_2d((0.0, 0.0), [0.0], 12.0, z=0.30)
check("ray straight at a person: distance = centre distance - radius", abs(dist[0] - (5.0 - r)) < 1e-9 and who[0] == 0,
      f"{dist[0]:.4f}")
dist, who = sim.ray_distances_2d((0.0, 0.0), [math.atan2(r * 0.99, 5.0), math.atan2(r * 1.05, 5.0)], 12.0, z=0.30)
check("grazing ray inside the radius hits, just outside misses", math.isfinite(dist[0]) and not math.isfinite(dist[1]))
dist, who = sim.ray_distances_2d((0.0, 0.0), [0.0], 12.0, z=1.70)
check("above the 1.65 m capsule top: no hit", not math.isfinite(dist[0]))
dist, who = sim.ray_distances_2d((0.0, 0.0), [0.0], 12.0, z=-0.1)
check("below the floor: no hit", not math.isfinite(dist[0]))
dist, who = sim.ray_distances_2d((5.05, 0.0), [0.0, math.pi], 12.0, z=0.30)
check("origin inside a capsule: distance 0", dist[0] == 0.0 and dist[1] == 0.0)
dist, who = sim.ray_distances_2d((0.0, 0.0), [0.0], 4.0, z=0.30)
check("max_range respected", not math.isfinite(dist[0]))
ang = np.arctan2(0.5, 9.0)
dist, who = sim.ray_distances_2d((0.0, 0.0), [0.0, ang, math.pi / 2], 12.0, z=0.30)
check("nearest of two wins; second person hit on its own bearing; nothing at 90 deg",
      who[0] == 0 and who[1] == 1 and who[2] == -1 and abs(dist[1] - (math.hypot(9.0, 0.5) - r)) < 1e-3,
      f"{dist}")
n = 500
ang = np.linspace(-math.pi, math.pi, n, endpoint=False)
dist, who = sim.ray_distances_2d((0.0, 0.0), ang, 12.0, z=0.30)
check("a 500-ray revolution works vectorised", dist.shape == (n,) and int((who >= 0).sum()) >= 2)
check("is_person_prim", sim.is_person_prim("/World/People/Character/Physics/BodyCollider") and
      not sim.is_person_prim("/World/Aloha/base_link"))

# 8. attach() waits for agents that appear late, then attaches without starting the clock ---------
sim = make({"Character": [["Idle", "5.0"]]}, {"Character": (0.0, 0.0, 0.0)})
sk = sim.people[0].skel
late = AGENTS.pop(sk)
sim.people[0].agent = None
calls = {"n": 0}


class _App:
    def update(self):
        calls["n"] += 1
        if calls["n"] == 7:
            AGENTS[sk] = late


sim._app = _App()
sim.attach()
check("attach() polled until the agent appeared and did not start the clock",
      sim.attached and calls["n"] == 7 and sim._needs_start and sim.t == 0.0 and sim.people[0].ci == -1,
      f"{calls['n']} updates")
sim.attach()
check("attach() is idempotent", calls["n"] == 7)

# 6. metrics are JSON-serialisable and carry the profile ---------------------------------------
m = sim.metrics()
s = json.dumps(m)
check("metrics() is JSON and has the step profile", "people_step_profile_ms_per_step" in m and len(s) > 100,
      str(m["people_step_profile_ms_per_step"]))
sys.exit(0 if ok else 1)
