"""WHICH part of a nav stage makes one PhysX step with the walking G1 cost 12-70 ms instead of ~1.6 ms?

Measured before this tool (nav/tools/g1_physx_cost.py, idle card, 0 other Kit): per 5 ms physics sub-step
hospital 1.6 ms, office 3.1, warehouse 12.0, outdoor_small 67. Ruled out on warehouse: broadphase type
(MBP/ABP/SAP), GPU dynamics, USD writes, scene queries, contact reports, collisionEnabled=False culling,
REMOVING CollisionAPI from all but the 22 colliders within 5 m (PhysX then holds 22 trimeshes and still
takes 17.6 ms), rendering (viewport updates off: 14.0 ms) and kit.update() itself (IPhysx.update_simulation
called directly, no Kit frame at all: 14.1 ms per call). One physics scene in the whole stage.

So the cost is inside update_simulation but not in collision. This tool bisects the ENVIRONMENT: the G1 stands
on a probe floor (a static 40x40 m cube whose top is the episode's floor height, authored in the session
layer, so it keeps standing when the environment's own floor is switched off), and for each candidate set of
environment prims made INACTIVE (session layer) it measures update_simulation() wall time over a short
stand + walk, stepping physics directly (no rendering). Greedy descent: at each level deactivate every child
of the current node one at a time; the child whose removal brings the step closest to the "environment
off" floor is descended into, until no child explains more than --explain of the excess.

    ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \
        nav/tools/g1_physx_bisect.py --env warehouse --out <dir>

report.json in <out>; the exit code is not the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]   # nav/tools/profile.py shadows stdlib
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--env", choices=["hospital", "office", "warehouse", "outdoor_small"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episode", default="")
ap.add_argument("--root", default="/World/Environment")
ap.add_argument("--steps", type=int, default=300, help="physics sub-steps per measurement (first 60 not timed)")
ap.add_argument("--max-depth", type=int, default=4)
ap.add_argument("--max-children", type=int, default=40, help="children tried per level (largest prim counts first)")
ap.add_argument("--mode", choices=["isolate", "bisect", "exts", "schemas", "notices", "instance"], default="isolate")
ap.add_argument("--exts", default="isaacsim.robot.surface_gripper,isaacsim.sensors.physics.nodes,"
                "isaacsim.sensors.experimental.physics,isaacsim.sensors.physx,isaacsim.sensors.experimental.rtx,"
                "isaacsim.sensors.rtx,omni.usd.metrics.assembler.physics,isaacsim.replicator.behavior,"
                "isaacsim.cortex.behaviors,omni.behavior.scripting.core,omni.physics.stageupdate",
                help="--mode exts: extensions disabled CUMULATIVELY, one per measurement (NO robot, full stage)")
ap.add_argument("--explain", type=float, default=0.5, help="descend only if one child removes >= this share of the excess")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"env": args.env, "args": vars(args), "pid": os.getpid(), "measurements": [], "path": []}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


from gpu_poll import GpuPoll  # noqa: E402

poll = GpuPoll(1.0, t0=T0)
poll.start()
rc = 1
try:
    import numpy as np
    import omni.physx
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    import locomotion as L
    from episode import episodes_by_env
    from isaacsim.core.simulation_manager import SimulationManager

    eps = episodes_by_env()[args.env]
    ep = next(e for e in eps if e.name == args.episode) if args.episode else eps[0]
    report["episode"] = {"name": ep.name, "start": list(ep.start), "yaw_deg": ep.start_yaw_deg}
    ctx = omni.usd.get_context()
    scene_file = (REPO / "assets" / "usd" / f"nav_{args.env}.usda").resolve()
    report["stage_file"] = {"path": str(scene_file), "md5_before": hashlib.md5(scene_file.read_bytes()).hexdigest()}
    ctx.open_stage(str(scene_file))
    stage = ctx.get_stage()
    for _ in range(240):
        kit.update()
    stage.Load()
    for _ in range(30):
        kit.update()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    if stage.GetPrimAtPath("/World/Aloha").IsValid():
        stage.GetPrimAtPath("/World/Aloha").SetActive(False)
    if not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]:
        UsdLux.DomeLight.Define(stage, "/World/G1Dome").CreateIntensityAttr(1000.0)

    # floor height under the start, from the environment itself (physics must be parsed: play once)
    L.configure_physics_scene(stage, 0.005, cpu=True, broadphase="MBP")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(3):
        kit.update()
    f = L.floor_below(ep.start[0], ep.start[1], ep.start[2] + L.SPAWN_PROBE_ABOVE_M)
    floor_z = f[0] if f else float(ep.start[2])
    report["floor"] = {"z": floor_z, "prim": f[1] if f else None}
    timeline.stop()
    for _ in range(3):
        kit.update()
    # probe floor: static cube, top at floor_z, 40x40 m, session layer
    cube = UsdGeom.Cube.Define(stage, "/World/G1ProbeFloor")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(ep.start[0], ep.start[1], floor_z - 0.5))
    xf.AddScaleOp().Set(Gf.Vec3f(40.0, 40.0, 1.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()

    loco = L.G1Locomotion("/World/G1")
    report["author"] = {"articulation_root": loco.author(stage, (ep.start[0], ep.start[1], floor_z + L.SPAWN_HEIGHT_M),
                                                         ep.start_yaw_rad)["articulation_root"]}
    physx_iface = omni.physx.get_physx_interface()

    root = stage.GetPrimAtPath(args.root)
    if not root.IsValid():
        raise RuntimeError(f"{args.root} not in stage")

    def count(prim) -> int:
        return sum(1 for _ in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))

    def children(path: str) -> list[tuple[str, int]]:
        # the previous measurement may have deactivated `path` itself (or the root): inactive prims have no
        # composed children, so clear those opinions first (timeline is restarted by the next measure anyway)
        timeline.stop()
        for _ in range(2):
            kit.update()
        for ps in base_holder.get("off", []):
            spec = stage.GetSessionLayer().GetPrimAtPath(ps)
            if spec is not None and spec.HasInfo("active"):
                spec.ClearInfo("active")
        base_holder["off"] = []
        p = stage.GetPrimAtPath(path)
        out = [(str(c.GetPath()), count(c)) for c in p.GetChildren()]
        out.sort(key=lambda kv: -kv[1])
        return out

    base_holder: dict = {"base": None}

    env_mesh_colliders = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                          if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh) and not p.IsInstanceProxy()]
    report["env_mesh_colliders"] = len(env_mesh_colliders)
    stripped: list[str] = []

    def set_strip(on: bool) -> int:
        sl = stage.GetSessionLayer()
        for ps in stripped:
            spec = sl.GetPrimAtPath(ps)
            if spec is not None and spec.HasInfo("apiSchemas"):
                spec.ClearInfo("apiSchemas")
        stripped.clear()
        if on:
            for ps in env_mesh_colliders:
                stage.GetPrimAtPath(ps).RemoveAPI(UsdPhysics.CollisionAPI)
                stripped.append(ps)
        return len(stripped)

    def measure(off: list[str], label: str, strip_all: bool = False, callbacks: bool = True, robot: bool = True,
                fixed_time: bool = False, profile: bool = False) -> dict:
        t_m = time.time()
        if base_holder["base"] is not None:
            base_holder["base"].close()
            base_holder["base"] = None
        timeline.stop()
        for _ in range(3):
            kit.update()
        # clear the previous measurement's session-layer `active` opinions, then set this one's
        for ps in base_holder.get("off", []):
            spec = stage.GetSessionLayer().GetPrimAtPath(ps)
            if spec is not None and spec.HasInfo("active"):
                spec.ClearInfo("active")
        off_all = list(off) + ([] if robot else ["/World/G1"])
        for ps in off_all:
            stage.GetPrimAtPath(ps).SetActive(False)
        base_holder["off"] = off_all
        n_stripped = set_strip(strip_all)
        L.configure_physics_scene(stage, 0.005, cpu=True, broadphase="MBP")
        timeline.play()
        kit.update()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        base = None
        if robot:
            loco.initialize()
            if callbacks:
                base = L.G1Base(loco)
                base_holder["base"] = base
                base.reset_to(tuple(ep.start), ep.start_yaw_rad)
            else:
                pose, _rec = L.g1_spawn_pose(tuple(ep.start), loco.prim_path)
                loco.reset_to(pose, ep.start_yaw_rad)
            kit.update()
            if base is not None:
                base.reset_stats()
        t_cur = float(timeline.get_current_time())
        times = []
        fell = None
        prof = None
        for k in range(args.steps):
            if base is not None:
                base.apply(0.0 if k < args.steps // 3 else 0.5, 0.0, 0.0, 0.005)
            if profile and k == 60:
                import cProfile

                prof = cProfile.Profile()
                prof.enable()
            a = time.perf_counter()
            physx_iface.update_simulation(0.005, t_cur)
            times.append(time.perf_counter() - a)
            if not fixed_time:
                t_cur += 0.005
            if k % 12 == 0:
                physx_iface.update_transformations(False, True, True, False)
                if base is not None:
                    ps_ = base.physics_state()
                    if ps_["fallen"] and fell is None:
                        fell = base.fall_time_s
        prof_top = None
        if prof is not None:
            prof.disable()
            import io
            import pstats

            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(25)
            prof_top = buf.getvalue()[-6000:]
        tt = np.array(times[60:]) * 1000.0
        try:
            st = omni.physx.get_physxunittests_interface().get_physics_stats()
            st = {k: st[k] for k in ("numStaticRigids", "numDynamicRigids", "numTriMeshShapes", "numConvexShapes",
                                     "numArticulations", "numConstraints") if k in st}
        except Exception as e:  # noqa: BLE001
            st = repr(e)
        m = {"label": label, "off": off, "strip_all": strip_all, "stripped": n_stripped, "callbacks": callbacks,
             "robot": robot, "step_ms_mean": round(float(tt.mean()), 3),
             "step_ms_median": round(float(np.median(tt)), 3), "fell": fell, "physics_stats": st,
             "wall_s": round(time.time() - t_m, 1), "pre_steps": None if base is None else base.physics_steps,
             "fixed_time": fixed_time, "python_profile_top": prof_top}
        report["measurements"].append(m)
        print(f"[bisect] {label}: {m['step_ms_median']} ms median, fell {fell}, stats {st}", flush=True)
        save()
        return m

    if args.mode == "instance":
        # g1_physx_prims.py: the per-step cost tracks the number of composed NON-proxy prims (hospital 1953 ->
        # 1.7 ms, office 4670 -> 3.0, warehouse 26346 -> 16.2, outdoor 83713 -> 80.8; ~0.6-1.0 us per prim per
        # step), and the warehouse uses no instancing at all. Make every referenced sub-asset instanceable in
        # the SESSION layer (file untouched) and measure again.
        def make_instanceable() -> dict:
            n_set, n_desc = 0, 0
            it = iter(Usd.PrimRange(root))
            for pr in it:
                if pr == root:
                    continue
                if pr.IsInstance():
                    it.PruneChildren()
                    continue
                if pr.HasAuthoredReferences() or pr.HasAuthoredPayloads():
                    nd = sum(1 for _ in Usd.PrimRange(pr)) - 1
                    if nd >= 2:
                        pr.SetInstanceable(True)
                        n_set += 1
                        n_desc += nd
                        it.PruneChildren()
            return {"made_instanceable": n_set, "their_descendants": n_desc}

        before = sum(1 for _ in stage.Traverse())
        measure([], "NO robot, full stage (as authored)", robot=False)
        measure([], "G1 walking, full stage (as authored)")
        timeline.stop()
        for _ in range(3):
            kit.update()
        rec = make_instanceable()
        for _ in range(30):
            kit.update()
        rec["prims_no_proxies_before"] = before
        rec["prims_no_proxies_after"] = sum(1 for _ in stage.Traverse())
        rec["prototypes_after"] = len(stage.GetPrototypes())
        report["instancing"] = rec
        print(f"[bisect] instancing {rec}", flush=True)
        save()
        m1 = measure([], "NO robot, referenced sub-assets INSTANCEABLE (session layer)", robot=False)
        m2 = measure([], "G1 walking, referenced sub-assets INSTANCEABLE (session layer)")
        report["status"] = "ok"
        raise SystemExit(0)

    if args.mode == "notices":
        # Is something changing the stage while physics steps? Count USD ObjectsChanged notices (and the paths
        # they touch) during direct stepping, with and without kit.update() between steps.
        from pxr import Tf

        seen = {"n": 0, "resync": 0, "info": 0, "examples": {}}

        def on_changed(notice, sender):
            seen["n"] += 1
            rs, inf = notice.GetResyncedPaths(), notice.GetChangedInfoOnlyPaths()
            seen["resync"] += len(rs)
            seen["info"] += len(inf)
            for pth in list(rs)[:3] + list(inf)[:3]:
                k = str(pth)
                seen["examples"][k] = seen["examples"].get(k, 0) + 1

        listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, on_changed, stage)
        for label, kwargs in (("NO robot, full stage", {"robot": False}), ("G1 walking, full stage", {})):
            seen.update({"n": 0, "resync": 0, "info": 0, "examples": {}})
            m = measure([], label + " [notices counted]", **kwargs)
            top = sorted(seen["examples"].items(), key=lambda kv: -kv[1])[:20]
            m["usd_notices"] = {"notices": seen["n"], "resynced_paths": seen["resync"], "info_paths": seen["info"],
                                "top_paths": top}
            print(f"[bisect] notices during '{label}': {m['usd_notices']}", flush=True)
            save()
        # the same stage after a long settle (MDL compiles etc. finished?)
        for _ in range(1200):
            kit.update()
        m = measure([], "NO robot, full stage, after 1200 more kit.update()", robot=False)
        listener.Revoke()
        report["status"] = "ok"
        raise SystemExit(0)

    if args.mode == "schemas":
        # which applied API schemas do the environment prims carry, and does removing the PHYSICS ones (not just
        # UsdPhysics.CollisionAPI) remove the per-step cost?
        hist: dict = {}
        phys_prims: list[tuple[str, list[str]]] = []
        for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            names = list(pr.GetAppliedSchemas())
            for n_ in names:
                hist[n_] = hist.get(n_, 0) + 1
            ph = [n_ for n_ in names if n_.startswith(("Physics", "Physx", "Newton", "PhysxSchema"))]
            if ph and not pr.IsInstanceProxy():
                phys_prims.append((str(pr.GetPath()), ph))
        report["schema_hist"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
        report["physics_schema_prims"] = len(phys_prims)
        rel_hist: dict = {}
        for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            for r_ in pr.GetRelationships():
                if "physics" in r_.GetName().lower():
                    rel_hist[r_.GetName()] = rel_hist.get(r_.GetName(), 0) + 1
        report["physics_relationship_hist"] = rel_hist
        print(f"[bisect] schema hist {json.dumps(report['schema_hist'])[:800]} rels {rel_hist}", flush=True)
        save()
        measure([], "NO robot, full stage", robot=False)
        sl = stage.GetSessionLayer()
        removed = 0
        for ps, ph in phys_prims:
            pr = stage.GetPrimAtPath(ps)
            for n_ in ph:
                if pr.RemoveAppliedSchema(n_):
                    removed += 1
        report["physics_schemas_removed"] = removed
        m = measure([], f"NO robot, ALL Physics*/Physx* API schemas removed from {len(phys_prims)} env prims "
                        f"({removed} removals)", robot=False)
        # materials bound for physics: unbind by blocking material:binding:physics relationships
        n_rel = 0
        for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if pr.IsInstanceProxy():
                continue
            for r_ in pr.GetRelationships():
                if r_.GetName().startswith("material:binding:physics"):
                    r_.SetTargets([])
                    n_rel += 1
        report["physics_material_bindings_cleared"] = n_rel
        measure([], f"NO robot, + {n_rel} physics material bindings cleared", robot=False)
        measure([args.root], f"NO robot, {args.root} OFF (floor)", robot=False)
        report["status"] = "ok"
        raise SystemExit(0)

    if args.mode == "exts":
        import omni.kit.app

        em = omni.kit.app.get_app().get_extension_manager()
        report["exts"] = []
        measure([], "NO robot, full stage, all extensions", robot=False)
        for name in [e for e in args.exts.split(",") if e]:
            was = em.is_extension_enabled(name)
            ok = None
            if was:
                try:
                    timeline.stop()
                    kit.update()
                    ok = em.set_extension_enabled_immediate(name, False)
                    for _ in range(3):
                        kit.update()
                except Exception as e:  # noqa: BLE001
                    ok = repr(e)
            now = em.is_extension_enabled(name)
            m = measure([], f"NO robot, full stage, cumulative OFF: {name} (was {was}, now {now})", robot=False)
            report["exts"].append({"ext": name, "was_enabled": was, "disable_result": ok, "enabled_after": now,
                                   "step_ms_median": m["step_ms_median"]})
            save()
        measure([args.root], f"NO robot, {args.root} OFF (floor)", robot=False)
        report["status"] = "ok"
        raise SystemExit(0)

    if args.mode == "isolate":
        kids_all = children(args.root)
        report["top_children"] = len(kids_all)
        evens = [ps for i, (ps, _n) in enumerate(kids_all) if i % 2 == 0]
        # time-sampled attributes in the environment (omni.physx update_simulation takes a currentTime "used for
        # time sampled transformations")
        n_ts_attr = n_ts_xform = n_ts_prims = 0
        ts_examples = []
        for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            hit = False
            for a in pr.GetAttributes():
                if a.GetNumTimeSamples() > 0:
                    n_ts_attr += 1
                    hit = True
                    if a.GetName().startswith("xformOp"):
                        n_ts_xform += 1
                    if len(ts_examples) < 12:
                        ts_examples.append([str(a.GetPath()), a.GetNumTimeSamples()])
            n_ts_prims += int(hit)
        report["time_samples"] = {"attrs": n_ts_attr, "xform_attrs": n_ts_xform, "prims": n_ts_prims,
                                  "examples": ts_examples}
        print(f"[bisect] time samples {report['time_samples']}", flush=True)
        save()
        measure([], "NO robot, full stage", robot=False)
        measure([], "NO robot, full stage, currentTime held constant", robot=False, fixed_time=True)
        measure([], "NO robot, full stage, cProfile", robot=False, profile=True)
        measure([args.root], f"NO robot, {args.root} OFF", robot=False)
        quarter = [ps for i, (ps, _n) in enumerate(kids_all) if i % 4 != 0]
        measure(quarter, f"NO robot, 3 of 4 top-level children OFF ({len(quarter)} of {len(kids_all)})", robot=False)
        measure(evens, f"NO robot, every other top-level child OFF ({len(evens)} of {len(kids_all)})", robot=False)
        measure([], "NO robot, ALL env mesh colliders stripped", robot=False, strip_all=True)
        measure([], "G1 walking, full stage")
        report["status"] = "ok"
        raise SystemExit(0)

    full = measure([], "full stage")
    none_ = measure([args.root], f"{args.root} OFF")
    report["full_ms"], report["env_off_ms"] = full["step_ms_median"], none_["step_ms_median"]
    excess = full["step_ms_median"] - none_["step_ms_median"]
    node = args.root
    for depth in range(args.max_depth):
        kids = children(node)[: args.max_children]
        if not kids:
            break
        level = []
        for ps, n in kids:
            m = measure([ps], f"d{depth} OFF {ps} ({n} prims)")
            level.append((full["step_ms_median"] - m["step_ms_median"], ps, n))
        level.sort(reverse=True)
        best_gain, best, n = level[0]
        report["path"].append({"node": node, "best_child": best, "prims": n, "gain_ms": round(best_gain, 3),
                               "excess_ms": round(excess, 3), "share": round(best_gain / max(excess, 1e-9), 3),
                               "level": [(round(g, 3), p, c) for g, p, c in level]})
        save()
        if best_gain < args.explain * excess:
            break
        node = best
    if base_holder["base"] is not None:
        base_holder["base"].close()
    report["status"] = "ok"
    rc = 0
except SystemExit:
    rc = 0
except Exception:  # noqa: BLE001
    import traceback

    report["status"] = "error"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    poll.stop_flag = True
    report["gpu_load"] = poll.window(0.0, poll.now())
    try:
        report["stage_file"]["md5_after"] = hashlib.md5(Path(report["stage_file"]["path"]).read_bytes()).hexdigest()
        report["stage_file"]["unchanged"] = report["stage_file"]["md5_after"] == report["stage_file"]["md5_before"]
    except Exception as e:  # noqa: BLE001
        report.setdefault("stage_file", {})["md5_after"] = repr(e)
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[bisect] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
