"""Does making the environment's referenced sub-assets instanceable (session layer) change what a camera SEES?

g1_physx_cost.py --variants cpu_instanced: PhysX step per 5 ms sub-step with the walking G1, instanced vs as
authored -- warehouse 4.8 vs 16-21 ms, outdoor_small 1.3 vs 82 ms; identical walk distance and tilt; identical
shape counts outdoors, warehouse 67 triangle-mesh colliders become box colliders. The camera image is the
policy's input, so before instancing could be used it has to render the same.

Per env: a world-fixed camera with the G1 head camera's optics (69.0 x 42.27 deg, 1920x1080) at the episode
start, 1.09 m above the floor, pitched 15 deg down (the walking head pose), plus a co-mounted DiskLight like
the G1 headlight; and a wide view from 3 m behind / 2.5 m up. Render each with the stage as authored, then with
every referenced/payloaded sub-asset under /World/Environment instanceable=True (session layer), then as
authored again (to measure the renderer's own frame-to-frame noise). Frames saved; mean abs pixel difference
and mean luminance reported.

    ISAAC_SLOT_MIN_FREE_MIB=15000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \
        nav/tools/g1_instancing_render_check.py --out <dir>
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
ap.add_argument("--out", required=True)
ap.add_argument("--envs", default="warehouse,outdoor_small")
ap.add_argument("--settle", type=int, default=120)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"args": vars(args), "pid": os.getpid(), "envs": []}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


rc = 1
try:
    import numpy as np
    import omni.usd
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    import locomotion as L
    from episode import episodes_by_env
    from isaacsim.sensors.camera import Camera

    ctx = omni.usd.get_context()

    def lum(a):
        a = a.astype(np.float32)
        return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())

    def look_quat(eye, target):
        """USD camera looks down -Z with +Y up: build the rotation whose -Z points eye->target, +Y ~ world up."""
        f = np.array(target, float) - np.array(eye, float)
        f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        m = np.stack([r, u, -f], axis=1)          # columns = camera x, y, z in world
        return L.matrix_to_quat_wxyz(m)

    def place(prim, eye, target):
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, eye)))
        q = look_quat(eye, target)
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))

    for env in [e for e in args.envs.split(",") if e]:
        ep = episodes_by_env()[env][0]
        f = (REPO / "assets" / "usd" / f"nav_{env}.usda").resolve()
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
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
        x, y, z = ep.start
        yaw = ep.start_yaw_rad
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        eye = np.array([x, y, z + 1.09])
        target = eye + 5.0 * fwd + np.array([0, 0, -5.0 * math.tan(math.radians(15.0))])
        head = UsdGeom.Camera.Define(stage, "/World/InstCheckHead")
        # G1 head camera optics from the asset spec (D435i colour 69.0 x 42.27 deg)
        spec = L.load_specs()[1]["nav_camera"]
        head.CreateFocalLengthAttr(float(spec["focal_length_mm"]))
        head.CreateHorizontalApertureAttr(float(spec["horizontal_aperture_mm"]))
        head.CreateVerticalApertureAttr(float(spec["vertical_aperture_mm"]))
        head.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        place(head.GetPrim(), eye, target)
        light = UsdLux.DiskLight.Define(stage, "/World/InstCheckLight")
        light.CreateIntensityAttr(float(L.G1_HEADLIGHT_INTENSITY))
        light.CreateRadiusAttr(float(L.G1_HEADLIGHT_RADIUS_M))
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))
        light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        place(light.GetPrim(), eye + 0.06 * (target - eye) / np.linalg.norm(target - eye), target)
        wide = UsdGeom.Camera.Define(stage, "/World/InstCheckWide")
        wide.CreateFocalLengthAttr(12.0)
        wide.CreateHorizontalApertureAttr(20.955)
        wide.CreateVerticalApertureAttr(20.955 * 9 / 16)
        wide.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        weye = np.array([x, y, z]) - 3.0 * fwd + np.array([0, 0, 2.5])
        place(wide.GetPrim(), weye, np.array([x, y, z]) + 4.0 * fwd)
        cams = {"head": Camera(prim_path="/World/InstCheckHead", resolution=(1920, 1080)),
                "wide": Camera(prim_path="/World/InstCheckWide", resolution=(1280, 720))}
        for c in cams.values():
            c.initialize()

        inst_paths: list[str] = []

        def set_instanced(on: bool) -> dict:
            sl = stage.GetSessionLayer()
            for ps in inst_paths:
                sp = sl.GetPrimAtPath(ps)
                if sp is not None and sp.HasInfo("instanceable"):
                    sp.ClearInfo("instanceable")
            inst_paths.clear()
            if on:
                envp = stage.GetPrimAtPath("/World/Environment")
                it = iter(Usd.PrimRange(envp))
                for pr in it:
                    if pr == envp:
                        continue
                    if pr.IsInstance():
                        it.PruneChildren()
                        continue
                    if (pr.HasAuthoredReferences() or pr.HasAuthoredPayloads()) and sum(1 for _ in Usd.PrimRange(pr)) > 2:
                        pr.SetInstanceable(True)
                        inst_paths.append(str(pr.GetPath()))
                        it.PruneChildren()
            return {"made_instanceable": len(inst_paths), "prims_no_proxies": sum(1 for _ in stage.Traverse())}

        frames: dict = {}
        rec: dict = {"env": env, "episode": ep.name, "modes": {}}
        import omni.timeline

        tl = omni.timeline.get_timeline_interface()
        for mode in ("authored", "instanced", "instanced_loaded", "authored_again"):
            tl.stop()                       # instanceable edits while stopped; the Camera annotators need a playing
            for _ in range(3):              # timeline to deliver frames (first attempt returned None while stopped)
                kit.update()
            info = set_instanced(mode in ("instanced", "instanced_loaded"))
            if mode == "instanced_loaded":
                # attempt 2 showed buildings/walls MISSING from the instanced render: new prototypes may carry
                # payloads that the stage's load rules leave unloaded -> load everything again
                stage.Load()
                info["payload_load_called"] = True
            info["prototypes"] = len(stage.GetPrototypes())
            tl.play()
            for _ in range(args.settle):
                kit.update()
            frames[mode] = {}
            for name, c in cams.items():
                a = c.get_rgba()
                img = None if a is None or np.asarray(a).size == 0 else np.asarray(a)[:, :, :3].astype(np.uint8)
                frames[mode][name] = img
                if img is not None:
                    Image.fromarray(img).save(OUT / f"{env}_{name}_{mode}.jpg", quality=92)
            rec["modes"][mode] = {**info, "luma": {n: (None if im is None else round(lum(im), 2)) for n, im in frames[mode].items()}}
            print(f"[instcheck] {env} {mode} {rec['modes'][mode]}", flush=True)

        def mad(a, b):
            return None if a is None or b is None else round(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), 3)

        rec["mean_abs_diff"] = {n: {"authored_vs_instanced": mad(frames["authored"][n], frames["instanced"][n]),
                                    "authored_vs_instanced_loaded": mad(frames["authored"][n], frames["instanced_loaded"][n]),
                                    "authored_vs_authored_again": mad(frames["authored"][n], frames["authored_again"][n]),
                                    "frac_px_diff_gt_16": (None if frames["authored"][n] is None or frames["instanced"][n] is None else
                                                           round(float((np.abs(frames["authored"][n].astype(np.int16) -
                                                                               frames["instanced"][n].astype(np.int16)).max(axis=2) > 16).mean()), 5))}
                                for n in cams}
        tl.stop()
        for _ in range(3):
            kit.update()
        set_instanced(False)
        rec["md5_unchanged"] = hashlib.md5(f.read_bytes()).hexdigest() == md5
        report["envs"].append(rec)
        print(f"[instcheck] {env} diff {rec['mean_abs_diff']}", flush=True)
        save()
        for c in cams.values():
            pass   # keep Camera objects alive until the stage is replaced (destroying them mid-run is not needed)
    report["status"] = "ok"
    rc = 0
except Exception:  # noqa: BLE001
    import traceback

    report["status"] = "error"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[instcheck] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    kit.close()
    os._exit(rc)
