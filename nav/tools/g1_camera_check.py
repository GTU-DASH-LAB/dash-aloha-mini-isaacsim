"""The G1 head camera, checked on IMAGE CONTENT and rendered for the videos.

--stage flat  (Kit: a cube floor, dome light, a grid of red floor markers at known world points)
  latency    walk + arc + spin-in-place, grab the head camera after EVERY kit.update() the way the
             runner grabs (isaacsim.sensors.camera.Camera.get_rgba()), detect each marker's red blob
             near where camera_geometry.project_ground(**menu_args) puts it, and compare the detected
             centroid with the prediction made from the pose read (a) after the same update, (b) one
             update earlier, (c) two updates earlier. The hypothesis with the smallest residual is the
             frame latency in updates. Until this, "rendered pose == PhysX + mount" had only been
             checked prim pose against prim pose (g1/check), never against pixels.
  flatvideo  a 30 s walk (stand, straight, arcs, stop with hold) with the head camera + headlight and
             a chase view, frames at 10 fps for nav/tools/g1_encode_video.py.

--stage hospital  (assets/usd/nav_hospital.usda in memory, session layer, md5 checked)
  cost       10 s of walking each with NO camera, the head camera, head + chase, every figure next to
             GPU0 utilisation and the number of OTHER Kit processes on GPU0 over the same window.
  reallight  the real 47.6 deg D435 mount gets its OWN headlight (locomotion.author_head_light with the
             real mount's pitch); intensity swept with only that light on, mean luminance against the
             AlohaMini camera_nav + headlight values measured at the same four poses by
             nav/tools/g1_render_check.py (nav/results/g1/g1_render_check_hospital.json). Also reported:
             the real-mount frame lit by the NAV headlight, which is what over-exposed it in the
             round-1 video (the near floor at 0.5-1.6 m under a light calibrated for 3-10 m).
  hospvideo  the hospital_down_hallway dogleg route twice in this process: run A nav headlight only
             (frames/nav + frames/chase), run B real-mount light only (frames/real) -- one light per
             camera, so neither view is lit by the other's lamp. Same route and controller; frame i
             of both runs is the same sim time.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/g1_camera_check.py --stage flat --out <dir>

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
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["flat", "hospital"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--skip", default="", help="comma list of phases to skip")
ap.add_argument("--cost-seconds", type=float, default=10.0)
ap.add_argument("--video-seconds", type=float, default=36.0)
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--real-multipliers", default="0.02,0.04,0.07,0.12,0.2,0.35,0.6,1.0")
ap.add_argument("--real-light", type=float, default=None,
                help="real-mount headlight intensity when the reallight sweep is skipped (round 2 picked 182250 = 0.2 x nav)")
args = ap.parse_args()
OUT = Path(args.out)
for d in ("frames/nav", "frames/real", "frames/chase", "lat", "ab"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
SKIP = set(args.skip.split(",")) if args.skip else set()

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"pid": os.getpid(), "args": vars(args)}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


from gpu_poll import GpuPoll  # noqa: E402

poll = GpuPoll(1.0, t0=T0)
poll.start()
rc = 1
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

    import camera_geometry as CG
    import locomotion as L
    from controllers import make_controller
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.sensors.camera import Camera

    ctx = omni.usd.get_context()
    loco = L.G1Locomotion("/World/G1")
    nav_spec, real_spec = loco.asset_spec["nav_camera"], loco.asset_spec["head_camera_real"]
    CTL = 1.0 / 60.0
    markers: list[tuple[float, float]] = []
    MARKER_TOP_Z = 0.01

    if args.stage == "flat":
        ctx.new_stage()
        for _ in range(5):
            kit.update()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        g = UsdGeom.Cube.Define(stage, "/World/Ground")
        g.CreateSizeAttr(1.0)
        UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
        UsdGeom.Xformable(g).AddScaleOp().Set(Gf.Vec3f(200, 200, 1))
        g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.55)]))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        # red floor markers, 0.15 x 0.15 x 0.01 m, on a 0.6 m grid ahead of and around the start (no colliders)
        k = 0
        for xi in np.arange(1.2, 9.01, 0.6):
            for yi in np.arange(-3.0, 3.01, 0.6):
                m = UsdGeom.Cube.Define(stage, f"/World/Markers/m{k:03d}")
                m.CreateSizeAttr(1.0)
                UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(float(xi), float(yi), MARKER_TOP_Z / 2))
                UsdGeom.Xformable(m).AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, MARKER_TOP_Z))
                m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.02, 0.02)]))
                markers.append((float(xi), float(yi)))
                k += 1
        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000.0)
        start, start_yaw = (0.0, 0.0, 0.0), 0.0
        scene = None
    else:
        from episode import load_episode
        from physics_setup import raise_gpu_capacities

        ep = load_episode("hospital_down_hallway")
        scene = REPO / "assets" / "usd" / "nav_hospital.usda"
        report["md5_before"] = hashlib.md5(scene.read_bytes()).hexdigest()
        ctx.open_stage(str(scene))
        stage = ctx.get_stage()
        for _ in range(240):
            kit.update()
        stage.Load()
        for _ in range(30):
            kit.update()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        stage.GetPrimAtPath("/World/Aloha").SetActive(False)
        report["gpu_capacity_changes"] = raise_gpu_capacities(stage)
        start, start_yaw = tuple(ep.start), ep.start_yaw_rad

    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True)
    report["author"] = loco.author(stage, (start[0], start[1], start[2] + L.SPAWN_HEIGHT_M), start_yaw)
    nav_path = L.author_head_camera(stage, loco.prim_path, nav_spec, "g1_head_cam")
    real_path = L.author_head_camera(stage, loco.prim_path, real_spec, "g1_head_cam_real")
    nav_light = UsdLux.DiskLight(stage.GetPrimAtPath(L.author_head_light(stage, loco.prim_path, nav_spec)))
    real_light = UsdLux.DiskLight(stage.GetPrimAtPath(
        L.author_head_light(stage, loco.prim_path, real_spec, name="g1_head_cam_real_headlight", intensity=0.0)))
    chase = UsdGeom.Camera.Define(stage, "/World/G1Chase")
    chase.CreateFocalLengthAttr(16.0)
    chase.CreateHorizontalApertureAttr(20.955)
    chase.CreateVerticalApertureAttr(20.955 * 9 / 16)
    chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    if args.stage == "flat":
        for _ in range(60):
            kit.update()
        stage.Load()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["init"] = loco.initialize()
    base = L.G1Base(loco)                 # callback mode, shipping defaults (hold on, slew limit on)

    def lum(a):
        a = a.astype(np.float32)
        return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())

    def grab(cam):
        rgba = cam.get_rgba()
        return None if rgba is None or np.asarray(rgba).size == 0 else np.asarray(rgba)[:, :, :3].astype(np.uint8)

    def step(cmd, measure=None):
        base.apply(*cmd, CTL)
        a = time.perf_counter()
        kit.update()
        if measure is not None:
            measure["s"] += time.perf_counter() - a
            measure["n"] += 1
        return base.physics_state()

    def hold_updates(n):
        for _ in range(n):
            step((0.0, 0.0, 0.0))

    chase_yaw = {"v": start_yaw}

    def aim_chase():
        x, y, _, yaw_ = loco.pose()
        chase_yaw["v"] += 0.15 * math.atan2(math.sin(yaw_ - chase_yaw["v"]), math.cos(yaw_ - chase_yaw["v"]))
        cy = chase_yaw["v"]
        fz = loco.floor_z
        eye = (x - 2.6 * math.cos(cy), y - 2.6 * math.sin(cy), fz + 1.6)
        mtx = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(x, y, fz + 0.85), Gf.Vec3d(0, 0, 1)).GetInverse()
        qq = mtx.ExtractRotationQuat()
        im = qq.GetImaginary()
        cam("chase").set_world_pose(position=np.array(eye, dtype=float),
                                 orientation=np.array([qq.GetReal(), im[0], im[1], im[2]]), camera_axes="usd")

    def snapshot_pose() -> dict:
        cp = loco.camera_pose(nav_spec)
        x, y, _, yaw = loco.pose()
        return {"menu": dict(cp["menu_args"]), "x": x, "y": y, "yaw": yaw}

    def predict(pose: dict, mx: float, my: float):
        dx, dy = mx - pose["x"], my - pose["y"]
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        xf, yl = c * dx + s * dy, -s * dx + c * dy
        m = dict(pose["menu"])
        m["cam_h"] = m["cam_h"] - MARKER_TOP_Z          # the marker's top face, not the floor
        if xf - m["cam_x"] > 5.0:
            return None
        return CG.project_ground(xf, yl, **m)

    cams: dict = {}

    def cam(name: str):
        """Cameras are created lazily (a Camera renders every update from creation on) and once."""
        if name not in cams:
            path, res = {"head": (nav_path, nav_spec["resolution_wh"]), "real": (real_path, real_spec["resolution_wh"]),
                         "chase": ("/World/G1Chase", (1280, 720))}[name]
            c = Camera(prim_path=path, resolution=tuple(res))
            c.initialize()
            cams[name] = c
        return cams[name]

    if args.stage == "flat":
        cam("head")
    base.reset_to(start, start_yaw)
    base.reset_stats()
    hold_updates(90)

    # ================================================================== flat: latency on pixels
    if args.stage == "flat" and "latency" not in SKIP:
        sched = [("stand", 1.0, (0.0, 0.0, 0.0)), ("straight", 3.0, (0.5, 0.0, 0.0)), ("arc_left", 3.0, (0.5, 0.0, 0.6)),
                 ("spin_right", 2.5, (0.0, 0.0, -0.8)), ("arc_right", 3.0, (0.4, 0.0, -0.5))]
        hist: list[dict] = []
        res = {h: [] for h in ("lag0", "lag1", "lag2")}
        per_seg = {}
        n_frames = n_det = 0
        saved = 0
        WIN = 70
        for seg, secs, cmd in sched:
            seg_res = {h: [] for h in res}
            for k in range(int(round(secs / CTL))):
                step(cmd)
                fr = grab(cam("head"))
                hist.append(snapshot_pose())
                hist = hist[-3:]
                if fr is None or len(hist) < 3:
                    continue
                n_frames += 1
                r = fr[..., 0].astype(np.int32)
                g_ = fr[..., 1].astype(np.int32)
                b = fr[..., 2].astype(np.int32)
                red = (r > 90) & (r > 2 * g_) & (r > 2 * b)
                dets = []
                for mx, my in markers:
                    p0 = predict(hist[-1], mx, my)
                    if p0 is None:
                        continue
                    u0, v0 = int(round(p0[0])), int(round(p0[1]))
                    if not (WIN <= u0 < CG.W - WIN and WIN <= v0 < CG.H - WIN):
                        continue
                    win = red[v0 - WIN:v0 + WIN, u0 - WIN:u0 + WIN]
                    ys, xs = np.nonzero(win)
                    if len(xs) < 25:
                        continue
                    cu, cv = u0 - WIN + xs.mean(), v0 - WIN + ys.mean()
                    # the blob must be one marker: its spread cannot exceed the window
                    if xs.min() == 0 or ys.min() == 0 or xs.max() == 2 * WIN - 1 or ys.max() == 2 * WIN - 1:
                        continue
                    row = {}
                    for h, pose in (("lag0", hist[-1]), ("lag1", hist[-2]), ("lag2", hist[-3])):
                        p = predict(pose, mx, my)
                        if p is not None:
                            e = math.hypot(cu - p[0], cv - p[1])
                            res[h].append(e)
                            seg_res[h].append(e)
                            row[h] = round(e, 2)
                    dets.append(row)
                n_det += len(dets)
                if saved < 6 and k == int(round(secs / CTL)) // 2 and dets:
                    Image.fromarray(fr).resize((960, 540)).save(OUT / "lat" / f"{seg}.jpg", quality=90)
                    saved += 1
            per_seg[seg] = {h: (None if not v else {"median_px": round(float(np.median(v)), 2),
                                                    "p90_px": round(float(np.percentile(v, 90)), 2), "n": len(v)})
                            for h, v in seg_res.items()}
            print(f"[camera] latency {seg}: {per_seg[seg]}", flush=True)
        summ = {h: (None if not v else {"median_px": round(float(np.median(v)), 2), "p90_px": round(float(np.percentile(v, 90)), 2),
                                        "mean_px": round(float(np.mean(v)), 2), "n": len(v)}) for h, v in res.items()}
        best = min((h for h in summ if summ[h]), key=lambda h: summ[h]["median_px"])
        report["latency"] = {"frames": n_frames, "marker_detections": n_det, "by_hypothesis": summ, "by_segment": per_seg,
                             "best": best, "frame_latency_updates": int(best[-1]),
                             "note": "lagN = prediction from the pose read N kit.updates before the grab"}
        save()
        print(f"[camera] latency: best {best} {summ}", flush=True)

    # ================================================================== flat: video
    if args.stage == "flat" and "flatvideo" not in SKIP:
        cam("chase")
        base.reset_to(start, start_yaw)
        base.reset_stats()
        aim_chase()
        hold_updates(120)
        sched = [("stand", 2.0, (0.0, 0.0, 0.0)), ("straight", 6.0, (0.6, 0.0, 0.0)), ("arc_left", 5.0, (0.5, 0.0, 0.5)),
                 ("straight", 4.0, (0.8, 0.0, 0.0)), ("arc_right", 5.0, (0.5, 0.0, -0.5)), ("stop_hold", 6.0, (0.0, 0.0, 0.0)),
                 # round-2 fix: a reverse request is clamped to (0,0,0) BEFORE the hold decision, so it holds
                 ("reverse_request_holds", 4.0, (-0.3, 0.0, 0.0))]
        every = max(1, int(round(1.0 / (args.fps * CTL))))
        rows, i, t = [], 0, 0.0
        m = {"s": 0.0, "n": 0}
        tw0 = poll.now()
        fell = None
        for seg, secs, cmd in sched:
            for k in range(int(round(secs / CTL))):
                ps = step(cmd, m)
                t += CTL
                if int(round(t / CTL)) % 3 == 0:
                    aim_chase()
                if ps["fallen"] and fell is None:
                    fell = base.last_fault
                if int(round(t / CTL)) % every:
                    continue
                fn, fc = grab(cam("head")), grab(cam("chase"))
                if fn is not None:
                    Image.fromarray(fn).save(OUT / "frames" / "nav" / f"{i:05d}.jpg", quality=92)
                if fc is not None:
                    Image.fromarray(fc).save(OUT / "frames" / "chase" / f"{i:05d}.jpg", quality=90)
                st = loco.state()
                cp = loco.camera_pose(nav_spec)
                rows.append({"i": i, "t": round(t, 2), "seg": seg, "cmd": [round(v, 3) for v in cmd],
                             "pos": [round(float(v), 3) for v in st["pos"]], "v_b": [round(float(v), 3) for v in st["lin_b"]],
                             "tilt": round(st["tilt_deg"], 2), "lum_nav": None if fn is None else round(lum(fn), 2),
                             "menu_args": {k_: round(v, 4) for k_, v in cp["menu_args"].items()}})
                i += 1
        report["video_rows"] = rows
        report["video_light_intensity"] = nav_light.GetIntensityAttr().Get()
        report["flatvideo"] = {"frames": i, "fell": fell, "stats": base.stats(),
                               "ms_per_update_nav_chase": round(1000 * m["s"] / max(m["n"], 1), 2),
                               "gpu_load_same_window": poll.window(tw0, poll.now())}
        save()
        print(f"[camera] flatvideo: {report['flatvideo']}", flush=True)

    # ================================================================== hospital: cost, 0/1/2 cameras
    path_pts = None
    if args.stage == "hospital":
        x0, y0 = start[0], start[1]
        path_pts = [(x0, y0), (x0 + 0.35, y0 - 12.0), (x0 - 0.25, y0 - 24.0)]

    def plan_in_robot_frame(px, py, yaw, n=30, dt=0.1, v=0.7):
        pts = np.array(path_pts)
        seg = pts[1:] - pts[:-1]
        slen = np.hypot(seg[:, 0], seg[:, 1])
        best, bs, bt = 1e9, 0, 0.0
        for i_ in range(len(seg)):
            t_ = float(np.clip(((px - pts[i_, 0]) * seg[i_, 0] + (py - pts[i_, 1]) * seg[i_, 1]) / slen[i_] ** 2, 0, 1))
            q = pts[i_] + t_ * seg[i_]
            d = math.hypot(px - q[0], py - q[1])
            if d < best:
                best, bs, bt = d, i_, t_
        out, s_i, s_t = [], bs, bt * slen[bs]
        for _ in range(n):
            s_t += v * dt
            while s_i < len(seg) - 1 and s_t > slen[s_i]:
                s_t -= slen[s_i]
                s_i += 1
            s_t = min(s_t, slen[s_i])
            wp = pts[s_i] + seg[s_i] * (s_t / slen[s_i])
            dx, dy = wp[0] - px, wp[1] - py
            out.append((math.cos(-yaw) * dx - math.sin(-yaw) * dy, math.sin(-yaw) * dx + math.cos(-yaw) * dy))
        return np.array(out)

    def route(seconds, on_update=None, measure=None, measure_from_s=3.0):
        ctrl = make_controller("braking", 1.0, 1.0)
        ctrl.reset()
        wp = None
        for k in range(int(round(seconds / CTL))):
            x, y, _ = base.position()
            if k < int(round(1.5 / CTL)):
                cmd = (0.0, 0.0, 0.0)
            else:
                if wp is None or k % 30 == 0:
                    wp = plan_in_robot_frame(x, y, base.yaw)
                c = ctrl(wp)
                cmd = (c.vx, c.vy, c.omega)
            ps = step(cmd, measure if k >= int(round(measure_from_s / CTL)) else None)
            if on_update is not None:
                on_update(k, k * CTL, cmd, ps)
            if ps["fallen"] or ps["physics_handle_lost"]:
                return {"stopped_at_s": round(k * CTL, 3), "fault": base.last_fault}
        return None

    if args.stage == "hospital" and "cost" not in SKIP:
        # in this order because a Camera cannot be un-created: none, then the head camera (read every
        # 0.5 s, the runner's decision period), then head + chase
        costs = {}
        for label in ("no_camera", "head_cam", "head_plus_chase"):
            if label == "head_cam":
                cam("head")
            if label == "head_plus_chase":
                cam("chase")
            base.reset_to(start, start_yaw)
            base.reset_stats()
            hold_updates(60)
            m = {"s": 0.0, "n": 0}
            gt = {"s": 0.0, "n": 0}
            tw0 = poll.now()

            def on_u(k, t, cmd, ps, label=label):
                if label == "head_plus_chase" and k % 3 == 0:
                    aim_chase()
                if label != "no_camera" and k % 30 == 0:
                    a = time.perf_counter()
                    grab(cam("head"))
                    if label == "head_plus_chase":
                        grab(cam("chase"))
                    gt["s"] += time.perf_counter() - a
                    gt["n"] += 1
            w0, s0 = time.perf_counter(), loco.sim_steps
            res = route(args.cost_seconds + 3.0, on_update=on_u, measure=m)
            wall, sim = time.perf_counter() - w0, (loco.sim_steps - s0) * loco.physics_dt
            costs[label] = {"walk_result": res, "ms_per_update": round(1000 * m["s"] / max(m["n"], 1), 2),
                            "wall_per_sim_s_incl_grabs": round(wall / sim, 3),
                            "grab_ms": round(1000 * gt["s"] / max(gt["n"], 1), 2) if gt["n"] else None,
                            "physx_step_ms_mean": base.stats()["physx_step_ms_mean"],
                            "gpu_load_same_window": poll.window(tw0, poll.now())}
            report["cost"] = costs
            save()
            print(f"[camera] cost {label}: {costs[label]}", flush=True)

    # ================================================================== hospital: real-mount light
    if args.stage == "hospital" and "reallight" not in SKIP:
        ref = json.loads((REPO / "nav" / "results" / "g1" / "g1_render_check_hospital.json").read_text())["light_ab"]
        cam("head")
        cam("real")
        from episode import load_episode

        mults = [float(v) for v in args.real_multipliers.split(",")]
        table = {}
        for name in ("hospital_down_hallway", "hallway+10m", "hospital_vending_machine", "hospital_exit_room"):
            if name == "hallway+10m":
                e = load_episode("hospital_down_hallway")
                pose = ((e.start[0] + 10 * math.cos(e.start_yaw_rad), e.start[1] + 10 * math.sin(e.start_yaw_rad), e.start[2]),
                        e.start_yaw_rad)
            else:
                e = load_episode(name)
                pose = (tuple(e.start), e.start_yaw_rad)
            base.reset_to(*pose)
            base.reset_stats()
            hold_updates(60)
            row = {"aloha_target": ref[name]["aloha_cam_aloha_light"], "g1_nav_lit_x2.25_round1": ref[name].get("g1_x2.25")}
            for label, ni, ri, cname in [("real_no_light", 0.0, 0.0, "real"),
                                         ("real_lit_by_nav_light", L.G1_HEADLIGHT_INTENSITY, 0.0, "real"),
                                         ("nav_lit_by_nav_light", L.G1_HEADLIGHT_INTENSITY, 0.0, "head")] + \
                                        [(f"real_own_x{mm:g}", 0.0, mm * L.G1_HEADLIGHT_INTENSITY, "real") for mm in mults]:
                nav_light.GetIntensityAttr().Set(float(ni))
                real_light.GetIntensityAttr().Set(float(ri))
                hold_updates(150)
                fr = grab(cam(cname))
                row[label] = None if fr is None else round(lum(fr), 2)
                if fr is not None:
                    row[label + "_sat_frac"] = round(float((fr.min(axis=2) >= 250).mean()), 4)
                    if label in ("real_no_light", "real_lit_by_nav_light", "nav_lit_by_nav_light") or label.endswith(f"x{mults[len(mults) // 2]:g}"):
                        Image.fromarray(fr).resize((960, 540)).save(OUT / "ab" / f"{name}_{label}.jpg", quality=90)
            row["ratio_to_aloha"] = {f"x{mm:g}": (None if row.get(f"real_own_x{mm:g}") is None
                                                  else round(row[f"real_own_x{mm:g}"] / row["aloha_target"], 3)) for mm in mults}
            table[name] = row
            report["real_light"] = {"table": table}
            save()
            print(f"[camera] real light {name}: {row}", flush=True)
        scores = {}
        for mm in mults:
            rs = [table[n]["ratio_to_aloha"][f"x{mm:g}"] for n in table if table[n]["ratio_to_aloha"].get(f"x{mm:g}")]
            if rs:
                scores[mm] = {"mean_abs_log_ratio": round(float(np.mean([abs(math.log(r)) for r in rs])), 4),
                              "within_20pct": int(sum(0.8 <= r <= 1.2 for r in rs)), "ratios": rs}
        best = min(scores, key=lambda k_: scores[k_]["mean_abs_log_ratio"])
        report["real_light"]["pick"] = {"multiplier_of_nav_light": best, "intensity": best * L.G1_HEADLIGHT_INTENSITY,
                                        "scores": {f"x{k_:g}": v for k_, v in scores.items()}}
        nav_light.GetIntensityAttr().Set(float(L.G1_HEADLIGHT_INTENSITY))
        real_light.GetIntensityAttr().Set(0.0)
        save()
        print(f"[camera] real light pick: {report['real_light']['pick']}", flush=True)

    # ================================================================== hospital: two-run video
    if args.stage == "hospital" and "hospvideo" not in SKIP:
        for c_ in ("head", "real", "chase"):
            cam(c_)
        real_i = float(report.get("real_light", {}).get("pick", {}).get(
            "intensity", args.real_light if args.real_light is not None else 0.12 * L.G1_HEADLIGHT_INTENSITY))
        report["real_light_used"] = real_i
        every = max(1, int(round(1.0 / (args.fps * CTL))))
        runs = {}
        rows_by_run = {}
        for run_name, ni, ri in (("A_nav_light", L.G1_HEADLIGHT_INTENSITY, 0.0), ("B_real_light", 0.0, real_i)):
            nav_light.GetIntensityAttr().Set(float(ni))
            real_light.GetIntensityAttr().Set(float(ri))
            base.reset_to(start, start_yaw)
            base.reset_stats()
            chase_yaw["v"] = start_yaw
            aim_chase()
            hold_updates(120)
            rows, cnt = [], {"i": 0}
            m = {"s": 0.0, "n": 0}
            tw0 = poll.now()

            def on_u(k, t, cmd, ps, run_name=run_name, rows=rows, cnt=cnt):
                if k % 3 == 0:
                    aim_chase()
                if k % every:
                    return
                i = cnt["i"]
                st = loco.state()
                row = {"i": i, "t": round(t, 2), "cmd": [round(v, 3) for v in cmd],
                       "pos": [round(float(v), 3) for v in st["pos"]], "yaw_deg": round(math.degrees(st["yaw"]), 2),
                       "v_b": [round(float(v), 3) for v in st["lin_b"]], "tilt": round(st["tilt_deg"], 2)}
                if run_name.startswith("A"):
                    fn, fc = grab(cam("head")), grab(cam("chase"))
                    if fn is not None:
                        Image.fromarray(fn).save(OUT / "frames" / "nav" / f"{i:05d}.jpg", quality=92)
                        row["lum_nav"] = round(lum(fn), 2)
                    if fc is not None:
                        Image.fromarray(fc).save(OUT / "frames" / "chase" / f"{i:05d}.jpg", quality=90)
                    cp = loco.camera_pose(nav_spec)
                    row["menu_args"] = {k_: round(v, 4) for k_, v in cp["menu_args"].items()}
                else:
                    fr = grab(cam("real"))
                    if fr is not None:
                        Image.fromarray(fr).resize((960, 540)).save(OUT / "frames" / "real" / f"{i:05d}.jpg", quality=90)
                        row["lum_real"] = round(lum(fr), 2)
                        row["real_sat_frac"] = round(float((fr.min(axis=2) >= 250).mean()), 4)
                rows.append(row)
                cnt["i"] += 1
            res = route(args.video_seconds, on_update=on_u, measure=m)
            runs[run_name] = {"walk_result": res, "frames": cnt["i"], "stats": base.stats(),
                              "ms_per_update": round(1000 * m["s"] / max(m["n"], 1), 2),
                              "nav_light": ni, "real_light": ri, "gpu_load_same_window": poll.window(tw0, poll.now())}
            rows_by_run[run_name] = rows
            print(f"[camera] video run {run_name}: {runs[run_name]}", flush=True)
        A, B = rows_by_run["A_nav_light"], rows_by_run["B_real_light"]
        n = min(len(A), len(B))
        pos_diff = [math.dist(A[i]["pos"][:2], B[i]["pos"][:2]) for i in range(n)]
        merged = []
        for i in range(n):
            r_ = dict(A[i])
            r_["lum_real"] = B[i].get("lum_real")
            r_["real_sat_frac"] = B[i].get("real_sat_frac")
            merged.append(r_)
        report["video_rows"] = merged
        report["video_light_intensity"] = L.G1_HEADLIGHT_INTENSITY
        report["hospvideo"] = {"runs": runs, "frames_aligned": n,
                               "run_A_vs_B_pelvis_xy_diff_m": {"max": round(max(pos_diff), 3) if pos_diff else None,
                                                               "end": round(pos_diff[-1], 3) if pos_diff else None},
                               "lum_nav": [min(r["lum_nav"] for r in A if "lum_nav" in r), max(r["lum_nav"] for r in A if "lum_nav" in r)],
                               "lum_real": [min(r["lum_real"] for r in B if "lum_real" in r), max(r["lum_real"] for r in B if "lum_real" in r)],
                               "real_sat_frac_max": max(r.get("real_sat_frac", 0) for r in B)}
        save()
        print(f"[camera] hospvideo: {report['hospvideo']}", flush=True)

    if scene is not None:
        report["md5_unchanged"] = hashlib.md5(scene.read_bytes()).hexdigest() == report["md5_before"]
        report["root_layer_dirty"] = bool(stage.GetRootLayer().dirty)
    report["gpu_whole_run"] = poll.window(0, poll.now())
    base.close()
    report["status"] = "ok"
    rc = 0
except Exception:  # noqa: BLE001
    import traceback

    report["status"] = "error"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    poll.stop_flag = True
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[camera] status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
