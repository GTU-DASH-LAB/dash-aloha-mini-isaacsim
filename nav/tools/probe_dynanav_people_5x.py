"""Ground truth for DynaNav's pedestrians: run DynaNav's OWN people pipeline in Isaac Sim 5.0.

Run with Isaac Sim **5.0**'s python.sh (Python 3.11; IRA 0.7.19 + omni.anim.people 0.7.7 are only
there), through the GPU0 slot wrapper:

  ISAAC_SLOT_TAG=people-dn5 ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh \
      ~/robotics/isaacsim-5.0.0/python.sh nav/tools/probe_dynanav_people_5x.py --out <dir>

It mirrors TIC-VLA/DynaNav/benchmark.py step for step -- same extension list (:956-961), same
settings (:966-985), the same IRA 0.7.19 config template (:448-510, minus the replicator writer and
the robot section), same command generator (generate_commands.py, seed 666), SimulationManager
set-up, 200 warm-up updates with the timeline stopped, then play at 30 tcps -- on the real
hospital scene, and measures what the people.py design has to match:

  1. Which navmesh agent parameters were actually BAKED (DynaNav sets the 5.0 non-persistent
     keys before IRA opens a stage whose root layer carries its own `navmeshSettings`).
  2. Whether `omni.usd.get_world_transform_matrix(SkelRoot)` -- the exact read DynaNav's
     human-collision check uses (benchmark.py:688-690) -- follows the walking character, or
     stays at the spawn point because the anim graph writes Fabric (use_usdrt=true) only.
     Compared against the anim-graph character's own position (what omni.anim.people and IRA's
     AgentManager use) and against Fabric (usdrt).
  3. Spawn points, walking speed, command progress, physics APIs under /World/Characters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--num-people", type=int, default=10)
ap.add_argument("--frames", type=int, default=900, help="played frames at 30 tcps")
ap.add_argument("--scene", default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/"
                                    "Isaac/Environments/Hospital/hospital.usd")
ap.add_argument("--scene-type", default="hospital")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
DN = Path("/home/gtu-dsa/robotics/TIC-VLA/DynaNav")

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
                     "renderer": "RayTracedLighting", "width": 640, "height": 480})

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402

report: dict = {"args": vars(args), "errors": [], "samples": []}


def log(m):
    print(f"[dn5] {m}", flush=True)


def save():
    (OUT / "probe_dynanav_people_5x.json").write_text(json.dumps(report, indent=1, default=str))


app = omni.kit.app.get_app()
em = app.get_extension_manager()
# benchmark.py:955-961, verbatim
for ext in ["omni.kit.viewport.window", "omni.kit.manipulator.prim", "omni.kit.property.usd",
            "omni.kit.scripting", "omni.anim.timeline", "omni.anim.graph.core",
            "omni.anim.retarget.core", "omni.anim.navigation.core",
            "omni.anim.people", "omni.isaac.sensor", "isaacsim.replicator.agent.core",
            "omni.kit.mesh.raycast", "omni.physx", "omni.physx.tensors"]:
    report[f"enable_{ext}"] = bool(em.set_extension_enabled_immediate(ext, True))
app.update()

# benchmark.py:966-985, verbatim (minus log level / debug print)
s = carb.settings.get_settings()
s.set("/persistent/scripting/enablePythonScripting", True)
s.set("/persistent/scripting/allowPythonScripting", True)
s.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
s.set("/exts/omni.anim.navigation.core/navMesh/config/agentHeight", 180)
s.set("/exts/omni.anim.navigation.core/navMesh/config/agentRadius", 40)
s.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight", 20)
s.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope", 45.0)
app.update()
NAVK = ["agentHeight", "agentRadius", "agentMaxStepHeight", "agentMaxFloorSlope", "excludeRigidBodies",
        "autoRebakeOnChanges"]
report["nav_settings_before_open"] = {k: s.get(f"/exts/omni.anim.navigation.core/navMesh/config/{k}") for k in NAVK}
report["people_settings"] = {k: s.get(f"/exts/omni.anim.people/{k}") for k in
                             ["navigation_settings/navmesh_enabled", "navigation_settings/dynamic_avoidance_enabled",
                              "behavior_settings/number_of_loop", "command_settings/command_file_path"]}
report["animgraph_settings"] = {"use_usdrt": s.get("/exts/omni.anim.graph.core/use_usdrt"),
                                "forceUpdateUSD": s.get("/animGraph/forceUpdateUSD")}

# Commands: DynaNav's own generator (benchmark.py:1380-1387), seed 666 (benchmark_full.yaml:3)
sys.path.insert(0, str(DN))
from generate_commands import generate_character_commands  # noqa: E402

cmd_file = OUT / "benchmark_character_commands.txt"
generate_character_commands(targets=args.scene_type, num_characters=args.num_people, num_commands=10,
                            seed=666, filename=str(cmd_file), base_name="Character")
report["commands"] = cmd_file.read_text().splitlines()

cfg = {"isaacsim.replicator.agent": {
    "version": "0.7.19",
    "global": {"seed": 666, "simulation_length": int(args.frames)},
    "scene": {"asset_path": args.scene},
    "sensor": {"camera_num": 0},
    "character": {"command_file": str(cmd_file), "num": args.num_people},
}}
cfg_path = OUT / "benchmark_config_probe.yaml"
cfg_path.write_text(yaml.dump(cfg))

from isaacsim.replicator.agent.core.simulation import SimulationManager  # noqa: E402

sm = SimulationManager()
if not sm.load_config_file(str(cfg_path)):
    report["errors"].append("load_config_file failed")
done = {"v": False}


def _done(_e):
    done["v"] = True


sub = sm.register_set_up_simulation_done_callback(_done)
t0 = time.time()
sm.set_up_simulation_from_config_file()
for k in range(20000):
    app.update()
    if done["v"]:
        break
report["setup_frames"], report["setup_s"] = k + 1, round(time.time() - t0, 1)
log(f"setup done={done['v']} after {k + 1} updates, {report['setup_s']} s")
for _ in range(200):  # benchmark.py:449 / :1007-1009 -- timeline stopped
    app.update()

stage = omni.usd.get_context().get_stage()
report["stage_url"] = omni.usd.get_context().get_stage_url()
report["root_layer_navmeshSettings"] = {k: (str(v)) for k, v in
                                        dict(stage.GetRootLayer().customLayerData.get("navmeshSettings", {})).items()}
report["nav_settings_after_open"] = {k: s.get(f"/exts/omni.anim.navigation.core/navMesh/config/{k}") for k in NAVK}
import omni.anim.navigation.core as nav  # noqa: E402

nm = nav.acquire_interface().get_navmesh()
if nm is not None:
    for fn in ["get_agent_height", "get_agent_radius", "get_agent_max_step_height", "get_agent_max_slope"]:
        try:
            report[f"navmesh_{fn}"] = getattr(nm, fn)()
        except Exception as e:  # noqa: BLE001
            report[f"navmesh_{fn}"] = f"ERR {e!r}"
log(f"navmesh: { {k: v for k, v in report.items() if k.startswith('navmesh_')} }")

from pxr import Usd, UsdPhysics  # noqa: E402

chars = []
root = stage.GetPrimAtPath("/World/Characters")
phys = {"collision": 0, "rigid": 0}
if root.IsValid():
    for c in root.GetChildren():
        sk = next((p for p in Usd.PrimRange(c) if p.GetTypeName() == "SkelRoot"), None)
        chars.append((c.GetName(), str(sk.GetPath()) if sk else None))
    for p in Usd.PrimRange(root):
        phys["collision"] += int(p.HasAPI(UsdPhysics.CollisionAPI))
        phys["rigid"] += int(p.HasAPI(UsdPhysics.RigidBodyAPI))
report["characters"] = chars
report["characters_physics_api_counts"] = phys
report["character_assets"] = {}
for name, sk in chars:
    p = stage.GetPrimAtPath(f"/World/Characters/{name}")
    refs = []
    for spec in p.GetPrimStack():
        for r in spec.referenceList.GetAddedOrExplicitItems():
            refs.append(r.assetPath)
    report["character_assets"][name] = refs[:2]
save()

import omni.anim.graph.core as ag  # noqa: E402

try:
    from usdrt import Usd as RtUsd  # noqa: E402

    rt_stage = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
except Exception as e:  # noqa: BLE001
    rt_stage = None
    report["errors"].append(f"usdrt attach: {e!r}")


def usd_pos(path):
    t = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path)).ExtractTranslation()
    return [round(t[0], 3), round(t[1], 3), round(t[2], 3)]


def fabric_pos(path):
    if rt_stage is None:
        return None
    try:
        p = rt_stage.GetPrimAtPath(path)
        for name in ("_worldPosition", "omni:fabric:worldMatrix", "xformOp:translate"):
            a = p.GetAttribute(name)
            if a and a.IsValid():
                v = a.Get()
                if v is None:
                    continue
                if name == "omni:fabric:worldMatrix":
                    v = v.ExtractTranslation()
                return {name: [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)]}
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"
    return None


def ag_pos(path):
    try:
        c = ag.get_character(path)
        if c is None:
            return None
        pos, rot = carb.Float3(0, 0, 0), carb.Float4(0, 0, 0, 0)
        c.get_world_transform(pos, rot)
        return [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


timeline = omni.timeline.get_timeline_interface()
timeline.set_time_codes_per_second(30)  # benchmark.py:521-523
timeline.commit_silently()
app.update()
report["usd_at_play"] = {n: usd_pos(sk) for n, sk in chars if sk}
timeline.play()
t_steps = []
for f in range(args.frames):
    ts = time.time()
    app.update()
    t_steps.append(time.time() - ts)
    if f % 15 == 0 or f == args.frames - 1:
        row = {"f": f, "tl": round(timeline.get_current_time(), 3), "chars": {}}
        for n, sk in chars[: min(len(chars), 10)]:
            if sk:
                row["chars"][n] = {"usd": usd_pos(sk), "anim_graph": ag_pos(sk), "fabric": fabric_pos(sk)}
        report["samples"].append(row)
        if f % 150 == 0:
            log(json.dumps(row)[:700])
            save()

# summary: per character, how far did each reading move from its play-time value
summ = {}
for n, sk in chars[:10]:
    if not sk:
        continue
    first = report["samples"][0]["chars"].get(n, {})
    last = report["samples"][-1]["chars"].get(n, {})
    d = {}
    for key in ("usd", "anim_graph"):
        a, b = first.get(key), last.get(key)
        if isinstance(a, list) and isinstance(b, list):
            d[f"{key}_moved_m"] = round(math.hypot(b[0] - a[0], b[1] - a[1]), 3)
    ag_path = [r["chars"][n]["anim_graph"] for r in report["samples"] if isinstance(r["chars"].get(n, {}).get("anim_graph"), list)]
    d["anim_graph_path_m"] = round(sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(ag_path, ag_path[1:])), 3)
    summ[n] = d
report["summary_moved"] = summ
report["step_ms"] = {"mean": round(1000 * sum(t_steps) / len(t_steps), 2), "max": round(1000 * max(t_steps), 2)}
report["timeline_end_s"] = round(timeline.get_current_time(), 3)
save()
log(f"summary: {json.dumps(summ)}")
log(f"step_ms={report['step_ms']} timeline_end={report['timeline_end_s']} errors={report['errors']}")
timeline.stop()
app.update()
kit.close()
