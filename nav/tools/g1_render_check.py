"""The walking G1 on the hospital stage, rendered the way the nav runner will render it, for three jobs:

  1. COST   10 s of walking with only the 1920x1080 head camera, then (phase 4) with the head camera,
            the real-mount comparison camera and a chase camera. Every figure is reported next to the
            GPU0 utilisation, the number of compute processes on GPU0 and the load average sampled
            once a second over the same window: a cost number without the card's load is not a number.
  2. LIGHT  calibrate the G1 headlight (locomotion.author_head_light: radius G1_HEADLIGHT_RADIUS_M,
            G1_HEADLIGHT_FORWARD_M ahead of the lens, not visible to primary rays) against AlohaMini's
            camera_nav optics + AlohaMini's headlight at the SAME poses: mean luminance of a 1920x1080
            frame, each light alone, plus the no-light baseline of both cameras. Candidates are
            multiples of --base-intensity; the pick is the candidate whose G1/AlohaMini ratio is
            closest to 1 (mean |log ratio| over the poses), accepted if within +-20% at >= 2 poses.
  3. VIDEO  a closed-loop walk down the hospital_down_hallway corridor (braking pursuit along a
            dogleg polyline, the route the prototype used) with the picked intensity: head camera
            (12 deg, the nav camera) saved at native 1920x1080 at 10 fps, the real 47.6 deg mount and
            a chase view alongside. NO gamma or brightness processing anywhere; nav/tools/
            g1_encode_video.py composes and encodes the frames as they are.

Stage: assets/usd/nav_hospital.usda opened IN MEMORY (edits in the session layer, file md5 checked
unchanged), /World/Aloha deactivated. G1Base in CALLBACK mode (hold_on_stop on), one kit.update()
per 1/60 s. The AlohaMini comparison camera's Camera object is DESTROYED after the calibration --
which also exercises the prim-deletion fix on a real nav stage (the G1 must keep walking).

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/g1_render_check.py --out <dir>

Report: <out>/report.json (the process exit code is NOT the verdict).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]   # nav/tools/profile.py shadows stdlib
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--base-intensity", type=float, default=405000.0)
ap.add_argument("--multipliers", default="0.75,1,1.5,2.25,3.4,5")
ap.add_argument("--poses", default="hospital_down_hallway,hallway+10m,hospital_vending_machine,hospital_exit_room")
ap.add_argument("--settle", type=int, default=150, help="updates per lighting setting before the grab")
ap.add_argument("--cost-seconds", type=float, default=10.0)
ap.add_argument("--video-seconds", type=float, default=36.0)
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--skip", default="", help="comma list of phases to skip: cost,light,video")
args = ap.parse_args()
OUT = Path(args.out)
for d in ("frames/nav", "frames/real", "frames/chase", "ab"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
SKIP = set(filter(None, args.skip.split(",")))

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"pid": os.getpid(), "args": vars(args)}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


class GpuPoll(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop_flag = [], False

    def run(self):
        while not self.stop_flag:
            try:
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                cards = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                idx, util = {}, {}
                for line in cards.strip().splitlines():
                    i, u, ut, mu = [c.strip() for c in line.split(",")]
                    idx[u] = int(i)
                    util[int(i)] = (int(ut), int(mu))
                mine, n0 = {}, 0
                for line in apps.strip().splitlines():
                    pid, uuid, mib = [c.strip() for c in line.split(",")]
                    if int(pid) == os.getpid():
                        mine[idx.get(uuid)] = int(mib)
                    if idx.get(uuid) == 0:
                        n0 += 1
                self.samples.append({"t": round(time.time() - T0, 1), "mine_gpu0": mine.get(0, 0), "mine_gpu1": mine.get(1, 0),
                                     "gpu0_util": util[0][0], "gpu0_used": util[0][1], "procs_gpu0": n0,
                                     "load1": round(os.getloadavg()[0], 1)})
            except Exception as e:  # noqa: BLE001
                self.samples.append({"error": repr(e)})
            time.sleep(1.0)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if "gpu0_util" in x and t0 <= x["t"] <= t1]
        if not s:
            return {"n": 0}
        return {"n": len(s), "gpu0_util_mean": round(sum(x["gpu0_util"] for x in s) / len(s), 1),
                "gpu0_util_min_max": [min(x["gpu0_util"] for x in s), max(x["gpu0_util"] for x in s)],
                "procs_gpu0_min_max": [min(x["procs_gpu0"] for x in s), max(x["procs_gpu0"] for x in s)],
                "gpu0_card_used_mib_max": max(x["gpu0_used"] for x in s),
                "mine_gpu0_max_mib": max(x["mine_gpu0"] for x in s), "mine_gpu1_max_mib": max(x["mine_gpu1"] for x in s),
                "load1_mean": round(sum(x["load1"] for x in s) / len(s), 1)}


poll = GpuPoll()
poll.start()
rc = 1
try:
    import numpy as np
    import carb
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux

    import locomotion as L
    from camera_geometry import camera_angles
    from controllers import make_controller
    from episode import load_episode
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.sensors.camera import Camera
    from physics_setup import raise_gpu_capacities

    ep = load_episode("hospital_down_hallway")
    scene = REPO / "assets" / "usd" / "nav_hospital.usda"
    md5_before = hashlib.md5(scene.read_bytes()).hexdigest()
    ctx = omni.usd.get_context()
    ctx.open_stage(str(scene))
    stage = ctx.get_stage()
    for _ in range(240):
        kit.update()
    stage.Load()
    for _ in range(30):
        kit.update()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    stage.GetPrimAtPath("/World/Aloha").SetActive(False)
    report["stage_boot_s"] = round(time.time() - T0, 1)

    loco = L.G1Locomotion("/World/G1")
    nav_spec, real_spec = loco.asset_spec["nav_camera"], loco.asset_spec["head_camera_real"]
    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True)
    report["gpu_capacity_changes"] = raise_gpu_capacities(stage)
    report["author"] = loco.author(stage, (ep.start[0], ep.start[1], ep.start[2] + L.SPAWN_HEIGHT_M), ep.start_yaw_rad)
    nav_path = L.author_head_camera(stage, loco.prim_path, nav_spec, "g1_head_cam")
    real_path = L.author_head_camera(stage, loco.prim_path, real_spec, "g1_head_cam_real")
    light_path = L.author_head_light(stage, loco.prim_path, nav_spec)
    g1_light = UsdLux.DiskLight(stage.GetPrimAtPath(light_path))
    report["headlight"] = {"prim": light_path, "radius_m": L.G1_HEADLIGHT_RADIUS_M, "forward_m": L.G1_HEADLIGHT_FORWARD_M,
                           "visibleInPrimaryRay": stage.GetPrimAtPath(light_path).GetAttribute("visibleInPrimaryRay").Get(),
                           "translate_in_head_link": list(UsdGeom.Xformable(g1_light).GetOrderedXformOps()[0].Get())}
    # AlohaMini camera_nav optics + AlohaMini headlight (scripts/alohamini1_specs.py), as static world prims
    al = UsdGeom.Camera.Define(stage, "/World/CheckAlohaCam")
    al.CreateFocalLengthAttr(2.8734347820281982)
    al.CreateHorizontalApertureAttr(5.760000228881836)
    al.CreateVerticalApertureAttr(3.5999999046325684)
    al.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    al_light = UsdLux.DiskLight.Define(stage, "/World/CheckAlohaLight")
    al_light.CreateIntensityAttr(0.0)
    al_light.CreateRadiusAttr(float(L.NAV_HEADLIGHT_RADIUS_M))
    al_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))
    for prim in (al.GetPrim(), al_light.GetPrim()):
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, Gf.Vec3d(0, 0, 0)))
    chase = UsdGeom.Camera.Define(stage, "/World/G1Chase")
    chase.CreateFocalLengthAttr(16.0)
    chase.CreateHorizontalApertureAttr(20.955)
    chase.CreateVerticalApertureAttr(20.955 * 9 / 16)
    chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["init"] = loco.initialize()
    base = L.G1Base(loco)                       # CALLBACK mode, hold_on_stop on
    st_ = carb.settings.get_settings()
    report["render_settings"] = {k: st_.get(k) for k in ("/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/rendermode",
                                                         "/rtx/post/histogram/enabled", "/rtx/post/tonemap/op")}
    CTL = 1.0 / 60.0

    def lum(a):
        a = a.astype(np.float32)
        return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())

    def grab(cam):
        rgba = cam.get_rgba()
        return None if rgba is None or np.asarray(rgba).size == 0 else np.asarray(rgba)[:, :, :3].astype(np.uint8)

    def updates(n, cmd=(0.0, 0.0, 0.0), measure=None):
        for _ in range(n):
            base.apply(*cmd, CTL)
            a = time.perf_counter()
            kit.update()
            if measure is not None:
                measure["upd_s"] += time.perf_counter() - a
                measure["n"] += 1
            ps = base.physics_state()
            if ps["fallen"] or ps["physics_handle_lost"]:
                return ps
        return None

    head = Camera(prim_path=nav_path, resolution=tuple(nav_spec["resolution_wh"]))
    head.initialize()
    x0, y0 = ep.start[0], ep.start[1]
    path_pts = [(x0, y0), (x0 + 0.35, y0 - 12.0), (x0 - 0.25, y0 - 24.0)]

    def plan_in_robot_frame(px, py, yaw, n=30, dt=0.1, v=0.7):
        pts = np.array(path_pts)
        seg = pts[1:] - pts[:-1]
        slen = np.hypot(seg[:, 0], seg[:, 1])
        best, bs, bt = 1e9, 0, 0.0
        for i in range(len(seg)):
            t_ = float(np.clip(((px - pts[i, 0]) * seg[i, 0] + (py - pts[i, 1]) * seg[i, 1]) / slen[i] ** 2, 0, 1))
            q = pts[i] + t_ * seg[i]
            d = math.hypot(px - q[0], py - q[1])
            if d < best:
                best, bs, bt = d, i, t_
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
        return np.array(out), best

    def walk(seconds, on_update=None, measure=None):
        """Closed-loop braking pursuit along path_pts from wherever the robot is. Returns the fall/fault state."""
        ctrl = make_controller("braking", 1.0, 1.0)
        ctrl.reset()
        wp = None
        n = int(round(seconds / CTL))
        for k in range(n):
            x, y, _ = base.position()
            yaw = base.yaw
            if k < int(round(1.5 / CTL)):
                cmd = (0.0, 0.0, 0.0)
            else:
                if wp is None or k % 30 == 0:
                    wp, _ = plan_in_robot_frame(x, y, yaw)
                c = ctrl(wp)
                cmd = (c.vx, c.vy, c.omega)
            base.apply(*cmd, CTL)
            a = time.perf_counter()
            kit.update()
            du = time.perf_counter() - a
            if measure is not None and k >= int(round(3.0 / CTL)):
                measure["upd_s"] += du
                measure["n"] += 1
            ps = base.physics_state()
            if on_update is not None:
                on_update(k, k * CTL, cmd, ps)
            if ps["fallen"] or ps["physics_handle_lost"]:
                return {"stopped_at_s": round(k * CTL, 3), "state": ps, "fault": base.last_fault}
        return None

    # ============================================================== 1) cost, head camera only
    if "cost" not in SKIP:
        base.reset_to(tuple(ep.start), ep.start_yaw_rad)
        base.reset_stats()
        updates(60)
        m = {"upd_s": 0.0, "n": 0}
        t_a, w0, s0 = time.time() - T0, time.perf_counter(), loco.sim_steps
        grab_t = {"s": 0.0, "n": 0}

        def grab_every_decision(k, t, cmd, ps):
            if k % 30 == 0:
                a = time.perf_counter()
                grab(head)
                grab_t["s"] += time.perf_counter() - a
                grab_t["n"] += 1
        res = walk(args.cost_seconds + 3.0, on_update=grab_every_decision, measure=m)
        wall, sim = time.perf_counter() - w0, (loco.sim_steps - s0) * loco.physics_dt
        report["cost_nav_only"] = {"walk_result": res, "ms_per_update_walking": round(1000 * m["upd_s"] / max(m["n"], 1), 2),
                                   "wall_per_sim_s": round(wall / sim, 3), "sim_s": round(sim, 2),
                                   "grab_ms_per_decision": round(1000 * grab_t["s"] / max(grab_t["n"], 1), 2),
                                   "gpu_load_same_window": poll.window(t_a, time.time() - T0)}
        save()
        print(f"[render] cost nav only: {report['cost_nav_only']}", flush=True)

    # ============================================================== 2) headlight calibration A/B
    pick = None
    if "light" not in SKIP:
        alc = Camera(prim_path="/World/CheckAlohaCam", resolution=(1920, 1080))
        alc.initialize()
        mults = [float(v) for v in args.multipliers.split(",")]
        poses = []
        for name in args.poses.split(","):
            if name == "hallway+10m":
                yaw = ep.start_yaw_rad
                poses.append((name, (ep.start[0] + 10 * math.cos(yaw), ep.start[1] + 10 * math.sin(yaw), ep.start[2]), yaw))
            else:
                e = load_episode(name)
                poses.append((name, tuple(e.start), e.start_yaw_rad))
        table = {}
        t_a = time.time() - T0
        for name, start, yaw in poses:
            base.reset_to(start, yaw)
            base.reset_stats()
            updates(60)
            px, py, _, pyaw = loco.pose()
            c, s_ = math.cos(pyaw), math.sin(pyaw)
            rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
            q = L.matrix_to_quat_wxyz(rz @ L.camera_mount_matrix(0.0))
            eye = (px + 0.25 * c, py + 0.25 * s_, loco.floor_z + 0.354)
            for prim in (al.GetPrim(), al_light.GetPrim()):
                ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                ops[0].Set(Gf.Vec3d(*eye))
                ops[1].Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
            row = {"pelvis": [round(px, 3), round(py, 3)], "yaw_deg": round(math.degrees(pyaw), 1)}
            settings = [("g1_no_light", 0.0, 0.0, head), ("aloha_no_light", 0.0, 0.0, alc),
                        ("aloha_cam_aloha_light", 0.0, float(L.NAV_HEADLIGHT_INTENSITY), alc)]
            settings += [(f"g1_x{mm:g}", mm * args.base_intensity, 0.0, head) for mm in mults]
            for label, gi, ai, cam in settings:
                g1_light.GetIntensityAttr().Set(gi)
                al_light.GetIntensityAttr().Set(ai)
                ps = updates(args.settle)
                fr = grab(cam)
                row[label] = None if fr is None else round(lum(fr), 2)
                if fr is not None and (label in ("g1_no_light", "aloha_no_light", "aloha_cam_aloha_light") or label == "g1_x1"
                                       or label == f"g1_x{mults[len(mults) // 2]:g}"):
                    Image.fromarray(fr).resize((960, 540)).save(OUT / "ab" / f"{name}_{label}.jpg", quality=90)
                if ps is not None:
                    row["fault"] = base.last_fault
                    break
            al_ref = row.get("aloha_cam_aloha_light")
            row["ratio_g1_over_aloha"] = {f"x{mm:g}": (None if row.get(f"g1_x{mm:g}") is None or not al_ref
                                                       else round(row[f"g1_x{mm:g}"] / al_ref, 3)) for mm in mults}
            table[name] = row
            report["light_ab"] = table
            save()
            print(f"[render] light A/B {name}: {row}", flush=True)
        g1_light.GetIntensityAttr().Set(0.0)
        al_light.GetIntensityAttr().Set(0.0)
        scores = {}
        for mm in mults:
            rs = [table[n]["ratio_g1_over_aloha"].get(f"x{mm:g}") for n in table]
            rs = [r for r in rs if r]
            if rs:
                scores[mm] = {"mean_abs_log_ratio": round(float(np.mean([abs(math.log(r)) for r in rs])), 4),
                              "poses_within_20pct": int(sum(0.8 <= r <= 1.2 for r in rs)), "ratios": rs}
        best = min(scores, key=lambda k_: scores[k_]["mean_abs_log_ratio"])
        pick = best * args.base_intensity
        report["light_pick"] = {"multiplier": best, "intensity": pick, "scores": {f"x{k_:g}": v for k_, v in scores.items()},
                                "accepted_within_20pct_at_2_poses": scores[best]["poses_within_20pct"] >= 2,
                                "gpu_load_same_window": poll.window(t_a, time.time() - T0)}
        save()
        print(f"[render] light pick: {report['light_pick']}", flush=True)
        # destroy the comparison camera: the G1 must keep walking (prim-deletion fix, on a real nav stage)
        alc.destroy()
        kit.update()
        report["after_alc_destroy"] = {"handle_ok": loco.handle_ok(),
                                       "deletions_ignored": getattr(loco.art, "_g1_deletions_ignored", 0),
                                       "deletions_forwarded": getattr(loco.art, "_g1_deletions_forwarded", 0)}
        save()

    # ============================================================== 3) video walk, 3 cameras
    if "video" not in SKIP:
        g1_light.GetIntensityAttr().Set(float(pick if pick is not None else L.G1_HEADLIGHT_INTENSITY))
        report["video_light_intensity"] = g1_light.GetIntensityAttr().Get()
        real = Camera(prim_path=real_path, resolution=tuple(real_spec["resolution_wh"]))
        real.initialize()
        chase_cam = Camera(prim_path="/World/G1Chase", resolution=(1280, 720))
        chase_cam.initialize()
        base.reset_to(tuple(ep.start), ep.start_yaw_rad)
        base.reset_stats()
        chase_yaw = {"v": ep.start_yaw_rad}

        def aim_chase():
            x, y, _, yaw_ = loco.pose()
            chase_yaw["v"] += 0.15 * math.atan2(math.sin(yaw_ - chase_yaw["v"]), math.cos(yaw_ - chase_yaw["v"]))
            cy = chase_yaw["v"]
            fz = loco.floor_z
            eye = (x - 2.6 * math.cos(cy), y - 2.6 * math.sin(cy), fz + 1.6)
            mtx = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(x, y, fz + 0.85), Gf.Vec3d(0, 0, 1)).GetInverse()
            qq = mtx.ExtractRotationQuat()
            im = qq.GetImaginary()
            chase_cam.set_world_pose(position=np.array(eye, dtype=float),
                                     orientation=np.array([qq.GetReal(), im[0], im[1], im[2]]), camera_axes="usd")
        aim_chase()
        updates(int(round(2.0 / CTL)))            # settle the new render products + light
        every = max(1, int(round(1.0 / (args.fps * CTL))))
        rows, frame_i = [], {"i": 0}
        sat = []
        m = {"upd_s": 0.0, "n": 0}
        t_a, w0, s0 = time.time() - T0, time.perf_counter(), loco.sim_steps
        io_t = {"grab": 0.0, "save": 0.0}

        def on_update(k, t, cmd, ps):
            if k % 3 == 0:
                aim_chase()
            if k % every:
                return
            a = time.perf_counter()
            fn, fr, fc = grab(head), grab(real), grab(chase_cam)
            io_t["grab"] += time.perf_counter() - a
            a = time.perf_counter()
            i = frame_i["i"]
            if fn is not None:
                Image.fromarray(fn).save(OUT / "frames" / "nav" / f"{i:05d}.jpg", quality=92)
            if fr is not None:
                Image.fromarray(fr).resize((960, 540)).save(OUT / "frames" / "real" / f"{i:05d}.jpg", quality=90)
            if fc is not None:
                Image.fromarray(fc).save(OUT / "frames" / "chase" / f"{i:05d}.jpg", quality=90)
                sat.append(float((fc.min(axis=2) >= 250).mean()))
            io_t["save"] += time.perf_counter() - a
            frame_i["i"] += 1
            # camera pose: rendered prim vs PhysX + mount, in the menu convention
            cp = loco.camera_pose(nav_spec)
            cpos, cq = head.get_world_pose(camera_axes="usd")
            r_usd = L.quat_to_matrix_wxyz(np.asarray(cq, dtype=float))
            c_, s_ = math.cos(-cp["pelvis_yaw"]), math.sin(-cp["pelvis_yaw"])
            rz_inv = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]])
            p_r, r_r, y_r = camera_angles(rz_inv @ r_usd)
            st = loco.state()
            rows.append({"i": i, "t": round(t, 2), "cmd": [round(v, 3) for v in cmd],
                         "pos": [round(float(v), 3) for v in st["pos"]], "yaw_deg": round(math.degrees(st["yaw"]), 2),
                         "v_b": [round(float(v), 3) for v in st["lin_b"]], "tilt": round(st["tilt_deg"], 2),
                         "lum_nav": None if fn is None else round(lum(fn), 2),
                         "menu_args": {k_: round(v, 4) for k_, v in cp["menu_args"].items()},
                         "render_minus_physx": {"eye_m": round(float(np.linalg.norm(np.asarray(cpos) - cp["eye"])), 5),
                                                "pitch_deg": round(p_r - cp["pitch_down_deg"], 4),
                                                "roll_deg": round(r_r - cp["roll_deg"], 4),
                                                "yaw_deg": round(y_r - cp["yaw_offset_deg"], 4)}})

        res = walk(args.video_seconds, on_update=on_update, measure=m)
        wall, sim = time.perf_counter() - w0, (loco.sim_steps - s0) * loco.physics_dt
        walking = [r for r in rows if r["t"] >= 4.0]

        def agg(vals):
            a = np.array(vals, dtype=float)
            return None if not len(a) else {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
                                            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}
        p0, p1 = rows[0]["pos"] if rows else None, rows[-1]["pos"] if rows else None
        report["video"] = {"walk_result": res, "frames": frame_i["i"], "fps": args.fps, "sim_s": round(sim, 2),
                           "wall_per_sim_s": round(wall / sim, 3),
                           "ms_per_update_walking_3cams": round(1000 * m["upd_s"] / max(m["n"], 1), 2),
                           "grab_ms_per_frame_3cams": round(1000 * io_t["grab"] / max(frame_i["i"], 1), 2),
                           "jpeg_save_ms_per_frame": round(1000 * io_t["save"] / max(frame_i["i"], 1), 2),
                           "gpu_load_same_window": poll.window(t_a, time.time() - T0),
                           "path_m": None if not rows else round(float(sum(math.dist(rows[i]["pos"][:2], rows[i + 1]["pos"][:2])
                                                                           for i in range(len(rows) - 1))), 2),
                           "start_end_xy": [p0, p1], "stats": base.stats(),
                           "lum_nav_walking": agg([r["lum_nav"] for r in walking if r["lum_nav"] is not None]),
                           "chase_saturated_px_frac": agg(sat),
                           "cam_walking": {k_: agg([r["menu_args"][k_] for r in walking]) for k_ in
                                           ("cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg")},
                           "render_minus_physx_absmax": {k_: max(abs(r["render_minus_physx"][k_]) for r in rows) if rows else None
                                                         for k_ in ("eye_m", "pitch_deg", "roll_deg", "yaw_deg")}}
        report["video_rows"] = rows
        save()
        print(f"[render] video: {({k_: v for k_, v in report['video'].items() if k_ not in ('stats',)})}", flush=True)

    report["md5_unchanged"] = hashlib.md5(scene.read_bytes()).hexdigest() == md5_before
    report["root_layer_dirty"] = bool(stage.GetRootLayer().dirty)
    report["gpu_whole_run"] = poll.window(0, time.time() - T0)
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
    print(f"[render] status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
