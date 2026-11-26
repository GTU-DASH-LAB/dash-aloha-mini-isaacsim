"""Where does the PhysX time go with the walking G1 on a nav stage, and which scene setting cuts it?

The G1 needs 200 Hz physics (3.33 sub-steps per 1/60 s control step). With CPU PhysX (MBP), the
configuration the walk was validated with, the start survey measured per physics sub-step:
hospital 5.4 ms, office 6.4, warehouse 15-16, outdoor_small 67-72 -- 14.4 wall s per control second
outdoors, ~36 min for one 150 s episode. This tool opens ONE nav stage (in memory, session layer
only, file md5 checked) with the G1 at an episode start and measures, per VARIANT of the physics
scene, the same 1 s stand + 3 s walk at vx 0.5 the runner would drive (one kit.update per 1/60 s,
G1Base in callback mode):

  ms per kit.update, ms per PhysX sub-step (G1Base pre->post callback), the omni.physx profile zones
  (IPhysxBenchmarks, summed per zone), PhysX's own body/shape counts (get_physics_stats) and scene
  statistics (active dynamic bodies, contact pairs), the G1's walk (distance, fell, tilt), and the
  GPU0 load over the same window (gpu_poll.GpuPoll).

VARIANTS (comma list, run in order, timeline stopped and replayed between them):
  cpu_mbp          CPU dynamics, MBP broadphase (the validated configuration)
  cpu_abp, cpu_sap CPU dynamics, other CPU broadphases
  gpu              GPU dynamics + GPU broadphase, readback ON (numpy tensor API, same as AlohaMini)
  cpu_static_env   cpu_mbp with every DYNAMIC environment rigid body made static
                   (physics:rigidBodyEnabled = False, session layer) and environment articulation
                   roots deactivated -- isolates what the environment's own dynamics cost
  gpu_static_env   the same on GPU
  cpu_nousd        cpu_mbp with /physics/updateToUsd and updateVelocitiesToUsd off (cost probe only:
                   rendering may then not follow the robot)

    ISAAC_SLOT_MIN_FREE_MIB=15000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \
        nav/tools/g1_physx_cost.py --env outdoor_small --out <dir>

report.json in <out>; the exit code is not the verdict (kit.close() ends the process).
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

CTL_DT = 1.0 / 60.0
VARIANTS = {
    "cpu_mbp": {"gpu": False, "bp": "MBP"},
    "cpu_abp": {"gpu": False, "bp": "ABP"},
    "cpu_sap": {"gpu": False, "bp": "SAP"},
    "gpu": {"gpu": True, "bp": "GPU"},
    "cpu_static_env": {"gpu": False, "bp": "MBP", "static_env": True},
    "gpu_static_env": {"gpu": True, "bp": "GPU", "static_env": True},
    "cpu_nousd": {"gpu": False, "bp": "MBP", "settings": {"/physics/updateToUsd": False,
                                                          "/physics/updateVelocitiesToUsd": False}},
    # diagnostics for "what in PhysX Update scales with the stage" (round 2: broadphase type, GPU dynamics and
    # USD writes all left warehouse at 16-21 ms per sub-step)
    "cpu_nosq": {"gpu": False, "bp": "MBP", "scene": {"EnableSceneQuerySupport": False}},
    "cpu_nocr": {"gpu": False, "bp": "MBP", "no_contact_report": True},
    "cpu_cull5": {"gpu": False, "bp": "MBP", "cull_m": 5.0},
    "cpu_cull20": {"gpu": False, "bp": "MBP", "cull_m": 20.0},
    "gpu_cull20": {"gpu": True, "bp": "GPU", "cull_m": 20.0},
    # round 2b: cull (collisionEnabled False) left PhysX Update at 14.3-14.6 ms with 22 colliders enabled and
    # ZERO contact pairs, so the cost is not collision. strip = RemoveAPI(CollisionAPI) in the session layer,
    # so omni.physx creates no PxActor/PxShape at all for those prims. Non-mesh colliders (the ground plane,
    # whose bbox is empty -> infinite "distance") are never culled or stripped.
    "cpu_strip5": {"gpu": False, "bp": "MBP", "strip_m": 5.0},
    "cpu_strip20": {"gpu": False, "bp": "MBP", "strip_m": 20.0},
    "cpu_strip50": {"gpu": False, "bp": "MBP", "strip_m": 50.0},
    # round 3: stripping colliders to 22 left warehouse PhysX Update at 17.6 ms per sub-step (hospital: 2 ms), and the
    # outdoor survey ran at 72 ms per sub-step on an almost idle machine (GPU0 19%, load 3.0). So it is neither
    # collision nor contention with other Kits. These two separate "PhysX" from "everything else kit.update() does":
    #   cpu_norender  the active viewport's updates_enabled = False (kit.update() still runs, nothing renders)
    #   cpu_direct    NO kit.update() at all: IPhysx.update_simulation(0.005, t) per sub-step +
    #                 update_transformations once per 1/60 s (what omni.physx's own update loop calls)
    # round 3b (g1_physx_bisect.py): the per-step cost follows the stage's composed NON-proxy prim count, and making
    # every referenced environment sub-asset instanceable in the SESSION layer cut a bare PhysX step 76.6 -> 1.7 ms
    # outdoors, 12.1 -> 3.6 ms in the warehouse. This variant checks it with the G1 walking on the real floor.
    "cpu_instanced": {"gpu": False, "bp": "MBP", "instanced": True},
    # g1_instancing_render_check.py: without a fresh stage.Load() the new prototypes' payloads stay UNLOADED
    # (outdoor buildings vanished from the render); with it the outdoor render matches. Cost with payloads loaded:
    "cpu_instanced_loaded": {"gpu": False, "bp": "MBP", "instanced": True, "load_payloads": True},
    "cpu_norender": {"gpu": False, "bp": "MBP", "norender": True},
    "cpu_direct": {"gpu": False, "bp": "MBP", "direct": True},
}

ap = argparse.ArgumentParser()
ap.add_argument("--env", choices=["hospital", "office", "warehouse", "outdoor_small"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variants", default="cpu_mbp,gpu,cpu_static_env,cpu_abp,gpu_static_env")
ap.add_argument("--episode", default="", help="start pose (default: first episode of the env)")
ap.add_argument("--stand-s", type=float, default=1.0)
ap.add_argument("--walk-s", type=float, default=3.0)
ap.add_argument("--physx-threads", type=int, default=None,
                help="--/persistent/physics/numThreads=N at Kit launch (read when omni.physx starts, not per variant)")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
_launch = {"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False}
if args.physx_threads is not None:
    _launch["extra_args"] = [f"--/persistent/physics/numThreads={int(args.physx_threads)}"]
kit = SimulationApp(_launch)
report: dict = {"env": args.env, "args": vars(args), "pid": os.getpid(), "variants": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


from gpu_poll import GpuPoll  # noqa: E402

poll = GpuPoll(1.0, t0=T0)
poll.start()
rc = 1
try:
    import carb
    import numpy as np
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

    import locomotion as L
    from episode import episodes_by_env
    from isaacsim.core.simulation_manager import SimulationManager

    eps = episodes_by_env()[args.env]
    ep = next(e for e in eps if e.name == args.episode) if args.episode else eps[0]
    report["episode"] = {"name": ep.name, "start": list(ep.start), "yaw_deg": ep.start_yaw_deg}
    settings = carb.settings.get_settings()
    report["carb"] = {k: settings.get(k) for k in ("/persistent/physics/numThreads", "/physics/updateToUsd",
                                                   "/physics/updateVelocitiesToUsd", "/physics/fabricEnabled",
                                                   "/physics/suppressReadback",
                                                   "/plugins/carb.tasking.plugin/threadCount")}
    report["omni_physx_fabric_enabled"] = omni.kit.app.get_app().get_extension_manager().is_extension_enabled("omni.physx.fabric")

    ctx = omni.usd.get_context()
    scene_file = (REPO / "assets" / "usd" / f"nav_{args.env}.usda").resolve()
    report["stage_file"] = {"path": str(scene_file), "md5_before": hashlib.md5(scene_file.read_bytes()).hexdigest()}
    t = time.time()
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
    report["stage_load_s"] = round(time.time() - t, 1)

    # ---- environment census (USD side): dynamic rigid bodies and articulations outside the robots
    t = time.time()
    dyn_bodies, kin_bodies, off_bodies, art_roots, unauthorable = [], 0, 0, [], 0
    all_scenes: list[str] = []           # UsdPhysics.Scene prims at ANY depth (L.physics_scene_prims looks 2 deep)
    n_prims = 0
    coll_paths: list[tuple[str, bool]] = []
    n_coll = {"total": 0, "mesh_none": 0, "convexHull": 0, "convexDecomposition": 0, "meshSimplification": 0,
              "sdf": 0, "boundingCube": 0, "boundingSphere": 0, "other_mesh_approx": 0, "non_mesh": 0}
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for p in it:
        n_prims += 1
        ps = str(p.GetPath())
        if ps.startswith("/World/G1") or ps.startswith("/World/Aloha"):
            it.PruneChildren()
            continue
        if p.IsA(UsdPhysics.Scene):
            all_scenes.append(ps)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(ps)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            n_coll["total"] += 1
            coll_paths.append((ps, p.IsInstanceProxy()))
            if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                a = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() or "none"
                key = "mesh_none" if a == "none" else (a if a in n_coll else "other_mesh_approx")
                n_coll[key] += 1
            else:
                n_coll["non_mesh"] += 1
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(p)
            if rb.GetRigidBodyEnabledAttr().Get() is False:
                off_bodies += 1
            elif rb.GetKinematicEnabledAttr().Get():
                kin_bodies += 1
            else:
                dyn_bodies.append(ps)
                unauthorable += int(p.IsInstanceProxy())
    in_art = [b for b in dyn_bodies if any(b == r or b.startswith(r + "/") for r in art_roots)]
    free_dyn = [b for b in dyn_bodies if b not in set(in_art)]
    report["census_usd"] = {"dynamic_bodies": len(dyn_bodies), "dynamic_bodies_in_articulations": len(in_art),
                            "dynamic_bodies_free": len(free_dyn), "kinematic_bodies": kin_bodies,
                            "rigid_body_disabled": off_bodies, "articulation_roots": art_roots,
                            "dynamic_instance_proxies": unauthorable, "colliders": n_coll,
                            "free_dynamic_examples": free_dyn[:15], "census_s": round(time.time() - t, 1),
                            "collider_instance_proxies": sum(1 for _, px in coll_paths if px),
                            "physics_scenes_any_depth": all_scenes, "prims_traversed": n_prims}
    # 2-D distance of every environment collider's world AABB from the start (for the cull variants)
    t = time.time()
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                            useExtentsHint=True)
    coll_dist: list[tuple[str, bool, float]] = []
    for ps, px in coll_paths:
        try:
            r_ = bbc.ComputeWorldBound(stage.GetPrimAtPath(ps)).ComputeAlignedRange()
            lo, hi = r_.GetMin(), r_.GetMax()
            dx = max(lo[0] - ep.start[0], 0.0, ep.start[0] - hi[0])
            dy = max(lo[1] - ep.start[1], 0.0, ep.start[1] - hi[1])
            coll_dist.append((ps, px, math.hypot(dx, dy)))
        except Exception:  # noqa: BLE001
            coll_dist.append((ps, px, float("nan")))
    dists = np.array([d for _, _, d in coll_dist if d == d])
    report["census_usd"]["collider_distance_from_start_m"] = {
        "n": int(len(dists)), "within_5": int((dists <= 5).sum()), "within_20": int((dists <= 20).sum()),
        "within_50": int((dists <= 50).sum()), "bbox_s": round(time.time() - t, 1)}
    print(f"[cost] census {json.dumps(report['census_usd'])[:600]}", flush=True)
    save()

    loco = L.G1Locomotion("/World/G1")
    report["author"] = loco.author(stage, (ep.start[0], ep.start[1], ep.start[2] + L.SPAWN_HEIGHT_M), ep.start_yaw_rad)
    timeline = omni.timeline.get_timeline_interface()
    physx_iface = omni.physx.get_physx_interface()
    bench = omni.physx.get_physx_benchmarks_interface()
    stats_iface = omni.physx.get_physx_statistics_interface()
    stage_id = omni.usd.get_context().get_stage_id()

    def physics_stats() -> dict:
        try:
            return omni.physx.get_physxunittests_interface().get_physics_stats()
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def scene_stats() -> dict:
        out = {}
        for sp in L.physics_scene_prims(stage):
            st = omni.physx.bindings._physx.PhysicsSceneStats()
            ok = False
            try:
                ok = stats_iface.get_physx_scene_statistics(stage_id, PhysicsSchemaTools.sdfPathToInt(sp.GetPath()), st)
            except Exception as e:  # noqa: BLE001
                out[str(sp.GetPath())] = repr(e)
                continue
            if ok:
                out[str(sp.GetPath())] = {k: getattr(st, k) for k in dir(st) if k.startswith("nb_")}
        return out

    def set_static_env(on: bool) -> dict:
        """on: every free dynamic environment body static, environment articulations inactive (session
        layer). off: remove exactly those session opinions again."""
        n_set = n_fail = 0
        for b in free_dyn:
            p = stage.GetPrimAtPath(b)
            if not p.IsValid() or p.IsInstanceProxy():
                n_fail += 1
                continue
            if on:
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr().Set(False)
                n_set += 1
            else:
                a = UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr()
                if a and a.IsValid() and stage.GetSessionLayer().GetAttributeAtPath(a.GetPath()):
                    a.Clear()
        for r in art_roots:
            p = stage.GetPrimAtPath(r)
            if p:
                p.SetActive(not on)
        return {"bodies_made_static": n_set, "not_authorable": n_fail,
                "articulations_deactivated": len(art_roots) if on else 0}

    culled: list[str] = []

    def set_cull(radius):
        """collisionEnabled = False (session layer) on every environment collider whose world AABB is more than
        `radius` m (2-D) from the start; None restores. Instance proxies cannot be authored: counted, left on."""
        nonlocal_culled = culled
        for ps in nonlocal_culled:
            a = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(ps)).GetCollisionEnabledAttr()
            if a and a.IsValid() and stage.GetSessionLayer().GetAttributeAtPath(a.GetPath()):
                a.Clear()
        nonlocal_culled.clear()
        if radius is None:
            return None
        skipped = 0
        for ps, px, d in coll_dist:
            if d == d and d > radius:
                if px:
                    skipped += 1
                    continue
                if not stage.GetPrimAtPath(ps).IsA(UsdGeom.Mesh):
                    continue                    # ground plane etc.: empty bbox, never "far"
                UsdPhysics.CollisionAPI(stage.GetPrimAtPath(ps)).CreateCollisionEnabledAttr().Set(False)
                nonlocal_culled.append(ps)
        return {"radius_m": radius, "colliders_disabled": len(nonlocal_culled), "proxies_left_on": skipped,
                "colliders_kept": len(coll_dist) - len(nonlocal_culled)}

    stripped: list[str] = []

    def set_strip(radius):
        """RemoveAPI(CollisionAPI) in the session layer on every environment MESH collider farther than `radius`
        (2-D AABB distance from the start); None restores by clearing the session layer's apiSchemas edit."""
        sl = stage.GetSessionLayer()
        for ps in stripped:
            spec = sl.GetPrimAtPath(ps)
            if spec is not None and spec.HasInfo("apiSchemas"):
                spec.ClearInfo("apiSchemas")
        stripped.clear()
        if radius is None:
            return None
        skipped = 0
        with Usd.EditContext(stage, sl):
            for ps, px, d in coll_dist:
                if d == d and d > radius:
                    p = stage.GetPrimAtPath(ps)
                    if px:
                        skipped += 1
                        continue
                    if not p.IsA(UsdGeom.Mesh):
                        continue
                    p.RemoveAPI(UsdPhysics.CollisionAPI)
                    stripped.append(ps)
        still = sum(1 for ps in stripped if stage.GetPrimAtPath(ps).HasAPI(UsdPhysics.CollisionAPI))
        return {"radius_m": radius, "colliders_stripped": len(stripped), "proxies_left_on": skipped,
                "strip_failed_still_has_api": still, "colliders_kept": len(coll_dist) - len(stripped)}

    def set_no_contact_report(on: bool):
        n = 0
        for name in (*L.FOOT_LINKS, L.TORSO_LINK):
            p = stage.GetPrimAtPath(f"/World/G1/{name}")
            if not p.IsValid():
                continue
            if on and p.HasAPI(PhysxSchema.PhysxContactReportAPI):
                p.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
                n += 1
            elif not on and not p.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
        return n

    inst_paths: list[str] = []

    def set_instanced(on: bool) -> dict | None:
        """instanceable=True (session layer) on every referenced/payloaded sub-asset under /World/Environment that
        has >= 2 descendants and is not inside an instance already; off clears exactly those opinions."""
        sl = stage.GetSessionLayer()
        for ps in inst_paths:
            spec = sl.GetPrimAtPath(ps)
            if spec is not None and spec.HasInfo("instanceable"):
                spec.ClearInfo("instanceable")
        inst_paths.clear()
        if not on:
            return None
        env = stage.GetPrimAtPath("/World/Environment")
        before = sum(1 for _ in stage.Traverse())
        it = iter(Usd.PrimRange(env))
        for pr in it:
            if pr == env:
                continue
            if pr.IsInstance():
                it.PruneChildren()
                continue
            if (pr.HasAuthoredReferences() or pr.HasAuthoredPayloads()) and sum(1 for _ in Usd.PrimRange(pr)) > 2:
                pr.SetInstanceable(True)
                inst_paths.append(str(pr.GetPath()))
                it.PruneChildren()
        return {"made_instanceable": len(inst_paths), "prims_no_proxies_before": before,
                "prims_no_proxies_after": sum(1 for _ in stage.Traverse()), "prototypes": len(stage.GetPrototypes())}

    base = None
    prev_settings: dict = {}
    for vname in [v for v in args.variants.split(",") if v]:
        spec = VARIANTS[vname]
        rec: dict = {"spec": spec}
        try:
            if base is not None:
                base.close()
                base = None
            timeline.stop()
            for _ in range(3):
                kit.update()
            for k, v in prev_settings.items():
                settings.set(k, v)
            prev_settings = {}
            for k, v in spec.get("settings", {}).items():
                prev_settings[k] = settings.get(k)
                settings.set(k, v)
            rec["static_env"] = set_static_env(bool(spec.get("static_env")))
            rec["cull"] = set_cull(spec.get("cull_m"))
            rec["strip"] = set_strip(spec.get("strip_m"))
            rec["instanced"] = set_instanced(bool(spec.get("instanced")))
            if spec.get("load_payloads"):
                stage.Load()
                for _ in range(10):
                    kit.update()
                rec["instanced"]["payloads_loaded"] = True
                rec["instanced"]["prims_no_proxies_after_load"] = sum(1 for _ in stage.Traverse())
                rec["instanced"]["prototypes_after_load"] = len(stage.GetPrototypes())
            rec["contact_report_removed"] = set_no_contact_report(bool(spec.get("no_contact_report")))
            rec["scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True, broadphase=spec["bp"])
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for sp in L.physics_scene_prims(stage):
                    api = PhysxSchema.PhysxSceneAPI.Apply(sp)
                    api.CreateEnableGPUDynamicsAttr().Set(bool(spec["gpu"]))
                    api.CreateBroadphaseTypeAttr().Set(spec["bp"])
                    api.CreateEnableSceneQuerySupportAttr().Set(bool(spec.get("scene", {}).get("EnableSceneQuerySupport", True)))
            settings.set_bool("/physics/suppressReadback", False)
            from physics_setup import raise_gpu_capacities

            rec["gpu_capacity_changes"] = raise_gpu_capacities(stage)
            timeline.play()
            kit.update()
            if SimulationManager.get_physics_sim_view() is None:
                SimulationManager.initialize_physics()
            rec["init"] = loco.initialize()
            rec["sim_device"] = str(SimulationManager.get_physics_sim_device())
            base = L.G1Base(loco)
            base.reset_to(tuple(ep.start), ep.start_yaw_rad)
            for _ in range(3):
                kit.update()
            base.reset_stats()
            zones: dict = {}

            def on_prof(stats_list, zones=zones):
                for s_ in stats_list:
                    z = zones.setdefault(s_.zone_name, [0.0, 0])
                    z[0] += float(s_.ms)
                    z[1] += 1
            bench.enable_profile(True)
            sub = bench.subscribe_profile_stats_events(on_prof)
            vp_api = None
            if spec.get("norender"):
                try:
                    from omni.kit.viewport.utility import get_active_viewport

                    vp_api = get_active_viewport()
                    rec["norender"] = {"viewport": None if vp_api is None else str(getattr(vp_api, "id", "?")),
                                       "updates_enabled_before": None if vp_api is None else bool(vp_api.updates_enabled)}
                    if vp_api is not None:
                        vp_api.updates_enabled = False
                except Exception as e:  # noqa: BLE001
                    rec["norender"] = {"error": repr(e)}
            direct = bool(spec.get("direct"))
            t_cur = float(timeline.get_current_time())
            carry, sim_wall, n_direct = 0.0, 0.0, 0
            tw0 = poll.now()
            upd_wall, n_upd, rows = 0.0, 0, []
            steps0 = base.physics_steps
            fell = None
            for name, secs, cmd in (("stand", args.stand_s, (0.0, 0.0, 0.0)), ("walk", args.walk_s, (0.5, 0.0, 0.0))):
                for k in range(int(round(secs / CTL_DT))):
                    base.apply(*cmd, CTL_DT)
                    a = time.perf_counter()
                    if direct:
                        carry += CTL_DT
                        n_sub_now = int(math.floor(carry / loco.physics_dt + 1e-6))
                        carry -= n_sub_now * loco.physics_dt
                        for _ in range(n_sub_now):
                            b_ = time.perf_counter()
                            physx_iface.update_simulation(loco.physics_dt, t_cur)
                            sim_wall += time.perf_counter() - b_
                            n_direct += 1
                            t_cur += loco.physics_dt
                        physx_iface.update_transformations(False, True, True, False)
                    else:
                        kit.update()
                    upd_wall += time.perf_counter() - a
                    n_upd += 1
                    ps = base.physics_state()
                    if k % 6 == 0:
                        x, y, z, _ = loco.pose()
                        rows.append((name, x, y, ps["tilt_deg"]))
                    if ps["fallen"] and fell is None:
                        fell = {"t": base.fall_time_s, "why": base.last_fault}
                        break
                if fell:
                    break
            sub = None
            bench.enable_profile(False)
            if vp_api is not None:
                vp_api.updates_enabled = True
                rec["norender"]["restored"] = True
            if direct:
                rec["direct"] = {"update_simulation_calls": n_direct,
                                 "update_simulation_ms_mean": round(1000 * sim_wall / max(n_direct, 1), 3),
                                 "pre_step_callbacks_seen": base.physics_steps - steps0}
                for _ in range(3):
                    kit.update()
            walk = [r_ for r_ in rows if r_[0] == "walk"]
            st = base.stats()
            n_sub = base.physics_steps - steps0
            rec.update({
                "updates": n_upd, "substeps": n_sub, "ms_per_update": round(1000 * upd_wall / max(n_upd, 1), 2),
                "wall_per_control_second": round(upd_wall / max(n_upd * CTL_DT, 1e-9), 3),
                "physx_step_ms_mean": st["physx_step_ms_mean"], "policy_hz": st["policy_hz"],
                "profile_zones_ms_per_substep": {z: round(v[0] / max(n_sub, 1), 3) for z, v in
                                                 sorted(zones.items(), key=lambda kv: -kv[1][0])[:25]},
                "profile_zone_calls": {z: v[1] for z, v in sorted(zones.items(), key=lambda kv: -kv[1][0])[:25]},
                "walk_distance_m": round(math.hypot(walk[-1][1] - walk[0][1], walk[-1][2] - walk[0][2]), 3) if len(walk) > 1 else None,
                "tilt_max_deg": round(max(r_[3] for r_ in rows), 2) if rows else None, "fell": fell,
                "gpu_load": poll.window(tw0, poll.now()),
            })
            rec["physics_stats"] = physics_stats()
            try:
                rec["scene_stats"] = scene_stats()
            except Exception as e:  # noqa: BLE001
                rec["scene_stats"] = repr(e)
        except Exception:  # noqa: BLE001
            import traceback

            rec["error"] = traceback.format_exc()
            print(rec["error"], flush=True)
        key = vname if vname not in report["variants"] else f"{vname}#{sum(k.split('#')[0] == vname for k in report['variants']) + 1}"
        report["variants"][key] = rec
        print(f"[cost] {vname}: ms/update {rec.get('ms_per_update')} physx/substep {rec.get('physx_step_ms_mean')} "
              f"wall/ctl-s {rec.get('wall_per_control_second')} walk {rec.get('walk_distance_m')} fell {rec.get('fell')} "
              f"gpu {rec.get('gpu_load')}", flush=True)
        save()
    if base is not None:
        base.close()
    report["stage_file"]["root_layer_dirty"] = bool(stage.GetRootLayer().dirty)
    report["status"] = "ok" if all("error" not in v for v in report["variants"].values()) else "partial"
    rc = 0
except Exception:  # noqa: BLE001
    import traceback

    report["status"] = "error"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    poll.stop_flag = True
    try:
        report["stage_file"]["md5_after"] = hashlib.md5(Path(report["stage_file"]["path"]).read_bytes()).hexdigest()
        report["stage_file"]["unchanged"] = report["stage_file"]["md5_after"] == report["stage_file"]["md5_before"]
    except Exception as e:  # noqa: BLE001
        report.setdefault("stage_file", {})["md5_after"] = repr(e)
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[cost] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
