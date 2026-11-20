"""Does one PhysX step (IPhysx.update_simulation) cost time per USD PRIM, physics or not?

g1_physx_bisect.py on nav_warehouse.usda, NO robot, physics stepped directly (no kit.update, no rendering):
full stage 11-21 ms per step, /World/Environment inactive 0.26-0.40 ms -- and still 15.9 ms after REMOVING
EVERY Physics*/Physx* API schema from all 3473 environment prims (PhysX then holds 1 static actor and
nothing else). Turning off 1/2 or 3/4 of the 3148 top-level children: 5.2 / 2.4 ms. Not Python (cProfile:
0.29 s of Python in 3.6 s), not time samples (none in the stage; a constant currentTime changes nothing), not
any of 9 sensor/behavior extensions (disabled one by one, cumulative).

Part 1 (synthetic): a new stage with a PhysicsScene and a ground plane, then N extra prims that have NO
physics at all, N in --counts, of --kinds (xform = UsdGeom.Xform, mesh = 8-vertex cube UsdGeom.Mesh,
nested = Xform chains 10 deep). update_simulation(0.005) wall time per N.
Part 2 (nav stages): open each nav_<env>.usda (in memory; md5 checked), count composed prims (with instance
proxies), meshes, instances, and time update_simulation with /World/Aloha inactive and no robot.

    ISAAC_SLOT_MIN_FREE_MIB=15000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \
        nav/tools/g1_physx_prims.py --out <dir>

report.json in <out>; the exit code is not the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
ap.add_argument("--out", required=True)
ap.add_argument("--counts", default="0,2500,5000,10000,20000,40000")
ap.add_argument("--kinds", default="xform,mesh,nested")
ap.add_argument("--envs", default="hospital,office,warehouse,outdoor_small")
ap.add_argument("--steps", type=int, default=200)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"args": vars(args), "pid": os.getpid(), "synthetic": [], "stages": []}


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
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    import locomotion as L

    physx_iface = omni.physx.get_physx_interface()
    timeline = omni.timeline.get_timeline_interface()
    ctx = omni.usd.get_context()

    def time_steps(stage) -> dict:
        L.configure_physics_scene(stage, 0.005, cpu=True, broadphase="MBP")
        timeline.play()
        for _ in range(3):
            kit.update()
        t_cur = float(timeline.get_current_time())
        tt = []
        for k in range(args.steps):
            a = time.perf_counter()
            physx_iface.update_simulation(0.005, t_cur)
            tt.append(time.perf_counter() - a)
            t_cur += 0.005
        timeline.stop()
        for _ in range(3):
            kit.update()
        x = np.array(tt[30:]) * 1000.0
        try:
            st = omni.physx.get_physxunittests_interface().get_physics_stats()
            st = {k: st[k] for k in ("numStaticRigids", "numDynamicRigids", "numTriMeshShapes") if k in st}
        except Exception as e:  # noqa: BLE001
            st = repr(e)
        return {"step_ms_median": round(float(np.median(x)), 3), "step_ms_mean": round(float(x.mean()), 3),
                "physics_stats": st}

    def census(stage) -> dict:
        n = meshes = inst = protos = 0
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            n += 1
            if p.IsA(UsdGeom.Mesh):
                meshes += 1
            if p.IsInstance():
                inst += 1
        n_noproxy = sum(1 for _ in stage.Traverse())
        return {"prims_with_instance_proxies": n, "prims_no_proxies": n_noproxy, "meshes": meshes,
                "instances": inst, "prototypes": len(stage.GetPrototypes())}

    # ---------------------------------------------------------------- part 1: synthetic
    counts = [int(c) for c in args.counts.split(",") if c]
    kinds = [k for k in args.kinds.split(",") if k]
    pts = Vt.Vec3fArray([Gf.Vec3f(x, y, z) for x in (0, 0.1) for y in (0, 0.1) for z in (0, 0.1)])
    for kind in kinds:
        for n in counts:
            ctx.new_stage()
            for _ in range(3):
                kit.update()
            stage = ctx.get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
            gp = UsdGeom.Plane.Define(stage, "/World/Ground")
            gp.CreateAxisAttr("Z")
            UsdPhysics.CollisionAPI.Apply(gp.GetPrim())
            t_a = time.time()
            layer = stage.GetRootLayer()
            with Sdf.ChangeBlock():
                junk = Sdf.CreatePrimInLayer(layer, "/World/Junk")
                junk.specifier = Sdf.SpecifierDef
                junk.typeName = "Xform"
                parent = None
                for i in range(n):
                    if kind == "nested":
                        path = f"/World/Junk/n{i // 10}" if i % 10 == 0 else f"{parent}/c{i % 10}"
                        parent = path
                        typ = "Xform"
                    elif kind == "xform":
                        path, typ = f"/World/Junk/x{i}", "Xform"
                    else:
                        path, typ = f"/World/Junk/m{i}", "Mesh"
                    spec = Sdf.CreatePrimInLayer(layer, path)
                    spec.specifier = Sdf.SpecifierDef
                    spec.typeName = typ
                    if typ == "Mesh":
                        a = Sdf.AttributeSpec(spec, "points", Sdf.ValueTypeNames.Point3fArray)
                        a.default = pts
                        a = Sdf.AttributeSpec(spec, "faceVertexCounts", Sdf.ValueTypeNames.IntArray)
                        a.default = Vt.IntArray([4] * 6)
                        a = Sdf.AttributeSpec(spec, "faceVertexIndices", Sdf.ValueTypeNames.IntArray)
                        a.default = Vt.IntArray([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
                        a = Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                        a.default = Gf.Vec3d((i % 200) * 0.3, (i // 200) * 0.3, 0.05)
                        a = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                        a.default = Vt.TokenArray(["xformOp:translate"])
            author_s = round(time.time() - t_a, 1)
            for _ in range(3):
                kit.update()
            r = {"kind": kind, "n_extra_prims": n, "author_s": author_s, **time_steps(stage)}
            report["synthetic"].append(r)
            print(f"[prims] synthetic {r}", flush=True)
            save()

    # ---------------------------------------------------------------- part 2: nav stages
    for env in [e for e in args.envs.split(",") if e]:
        f = (REPO / "assets" / "usd" / f"nav_{env}.usda").resolve()
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        t_o = time.time()
        ctx.open_stage(str(f))
        stage = ctx.get_stage()
        for _ in range(240):
            kit.update()
        stage.Load()
        for _ in range(30):
            kit.update()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        if stage.GetPrimAtPath("/World/Aloha").IsValid():
            stage.GetPrimAtPath("/World/Aloha").SetActive(False)
        r = {"env": env, "open_s": round(time.time() - t_o, 1), **census(stage), **time_steps(stage)}
        r["md5_unchanged"] = hashlib.md5(f.read_bytes()).hexdigest() == md5
        r["root_layer_dirty"] = bool(stage.GetRootLayer().dirty)
        report["stages"].append(r)
        print(f"[prims] stage {r}", flush=True)
        save()
    report["status"] = "ok"
    rc = 0
except Exception:  # noqa: BLE001
    import traceback

    report["status"] = "error"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    poll.stop_flag = True
    report["gpu_load"] = poll.window(0.0, poll.now())
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[prims] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
