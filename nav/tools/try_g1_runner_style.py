"""The G1 driven the way nav/sim/run_navigation.py drives a robot: ONE `kit.update()` per 1/60 s
control step, `base.apply(vx, vy, omega, dt)` before it, `base.physics_state()` after it -- through
nav/sim/g1/locomotion.py:G1Base in CALLBACK mode (the scene runs at 200 Hz, the policy ticks from a
PHYSICS_PRE_STEP callback). try_g1_walk.py steps PhysX itself; this script is the test that the
runner's loop does not have to.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/try_g1_runner_style.py \
        --mode cadence --out <dir>

modes (all write <out>/report.json; the process exit code is NOT the verdict, see report["status"]):
  cadence   flat Grid. Physics sub-steps per update, policy ticks per sim second, sim-clock drift
            against the 1/60 s control clock, tracking of a short vx/wz schedule, wall cost per sim
            second with and without a camera (--camera none|nav|nav+chase).
  envelope  flat Grid. One trial per command (reset, stand 1.5 s, hold the command 4 s, no clamping
            outside the trained box so out-of-range commands are TESTED): achieved (vx, vy, wz) and
            fall/no fall. The map of what the policy actually delivers for combined commands.
  stress    flat Grid. Long random command sequences changing every 0.1-1.5 s (random box / corners
            / arcs through PursuitController), N seeds: falls, time to fall.
  route     assets/usd/nav_hospital.usda opened IN MEMORY (session layer only; /World/Aloha
            deactivated; the file md5 is verified unchanged). Closed-loop PursuitController down the
            corridor from hospital_down_hallway's start, 1920x1080 head camera grabbed every 0.5 s
            (the runner's decision period): cost, and the camera pose the RENDERER used versus the
            one computed from PhysX (height, pitch, roll, head yaw against pelvis yaw).
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
# nav/tools/profile.py shadows the stdlib `profile`, which Kit's warp imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

CDN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"
GRID_ENV = f"{CDN}/Environments/Grid/default_environment.usd"
CTL_DT = 1.0 / 60.0

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True,
                help="cadence | envelope | stress | route, '+'-joined to run several in ONE Kit process "
                     "(e.g. envelope+stress; route runs on the hospital stage, the others on the flat Grid)")
ap.add_argument("--out", required=True)
ap.add_argument("--camera", choices=["none", "nav", "nav+chase"], default="none")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--stress-seconds", type=float, default=60.0)
ap.add_argument("--route-seconds", type=float, default=50.0)
ap.add_argument("--chase-from", type=float, default=0.0,
                help="route: create the 1280x720 chase Camera object this many control-seconds into the walk "
                     "(0 = never) to measure what the runner's chase view adds")
ap.add_argument("--survey", action="store_true", help="route: afterwards run the start survey over this stage's episodes")
ap.add_argument("--episode", default="hospital_down_hallway")
ap.add_argument("--save-every", type=float, default=5.0, help="route: save a full-res nav frame every N s")
ap.add_argument("--headlight", choices=["on", "off"], default="on",
                help="G1 head headlight (AlohaMini camera_nav parity). route mode also runs an on/off exposure A/B at the start")
args = ap.parse_args()
MODES = set(args.mode.split("+"))
bad = MODES - {"cadence", "envelope", "stress", "route"}
if bad or ("route" in MODES and len(MODES) > 1):
    raise SystemExit(f"bad --mode {args.mode!r}: route cannot be mixed with the flat modes ({bad or ''})")

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T_LAUNCH = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"mode": args.mode, "args": vars(args), "pid": os.getpid()}


def save_report():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


class GpuPoll(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag, self.pid = period, [], False, os.getpid()

    def read(self) -> dict:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10).stdout
            idx = {u.split(",")[1].strip(): int(u.split(",")[0]) for u in uu.strip().splitlines()}
            mine = {}
            for line in out.strip().splitlines():
                pid, uuid, mib = [c.strip() for c in line.split(",")]
                if int(pid) == self.pid:
                    mine[f"gpu{idx.get(uuid, uuid)}"] = int(mib)
            return mine
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def run(self):
        while not self.stop_flag:
            self.samples.append((round(time.time() - T_LAUNCH, 1), self.read()))
            time.sleep(self.period)

    def summary(self) -> dict:
        per = {}
        for _, d in self.samples:
            for k, v in d.items():
                if isinstance(v, int):
                    per.setdefault(k, []).append(v)
        return {k: {"max_mib": max(v), "last_mib": v[-1], "n": len(v)} for k, v in per.items()}


gpu = GpuPoll(1.0)
gpu.start()
rc = 1
try:
    import numpy as np
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux

    import locomotion as L

    kapp = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    loco = L.G1Locomotion("/World/G1")
    nav_spec = loco.asset_spec["nav_camera"]

    # ------------------------------------------------------------------ stage
    stage_file_md5 = None
    if "route" not in MODES:
        ctx.new_stage()
        for _ in range(5):
            kit.update()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        env = stage.DefinePrim("/World/Environment", "Xform")
        env.GetReferences().AddReference(GRID_ENV)
        for _ in range(120):
            kit.update()
        stage.Load()
        for _ in range(20):
            kit.update()
        start_xy, start_yaw, start_z = (0.0, 0.0), 0.0, 0.0
        episode = None
    else:
        from episode import load_episode

        ep = load_episode(args.episode)
        episode = {"name": ep.name, "start": list(ep.start), "start_yaw_deg": math.degrees(ep.start_yaw_rad),
                   "goal": list(ep.goal), "timeout_s": ep.timeout_s}
        report["episode"] = episode
        scene_file = (REPO / "assets" / "usd" / "nav_hospital.usda").resolve()
        stage_file_md5 = hashlib.md5(scene_file.read_bytes()).hexdigest()
        report["stage_file"] = {"path": str(scene_file), "md5_before": stage_file_md5}
        ctx.open_stage(str(scene_file))
        stage = ctx.get_stage()
        for _ in range(240):
            kit.update()
        stage.Load()
        for _ in range(30):
            kit.update()
        start_xy, start_yaw, start_z = (ep.start[0], ep.start[1]), ep.start_yaw_rad, ep.start[2]
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    if "route" in MODES:
        aloha = stage.GetPrimAtPath("/World/Aloha")
        if aloha.IsValid():
            aloha.SetActive(False)
    if not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]:
        UsdLux.DomeLight.Define(stage, "/World/G1Dome").CreateIntensityAttr(1000.0)
        UsdLux.DistantLight.Define(stage, "/World/G1Sun").CreateIntensityAttr(3000.0)

    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True)
    if "route" in MODES:
        from physics_setup import raise_gpu_capacities

        report["gpu_capacity_changes"] = raise_gpu_capacities(stage)

    spawn_guess = (start_xy[0], start_xy[1], start_z + L.SPAWN_HEIGHT_M)
    report["author"] = loco.author(stage, spawn_guess, start_yaw)
    if "route" in MODES and args.camera == "none":
        args.camera = "nav"
    cam_path = L.author_head_camera(stage, loco.prim_path, nav_spec, "g1_head_cam") if args.camera != "none" else None
    light_path = L.author_head_light(stage, loco.prim_path, nav_spec) if (args.camera != "none" and args.headlight == "on") else None
    report["headlight_prim"] = light_path
    chase_path = None
    if args.camera == "nav+chase" or args.chase_from > 0:
        ch = UsdGeom.Camera.Define(stage, "/World/G1Chase")
        ch.CreateFocalLengthAttr(16.0)
        ch.CreateHorizontalApertureAttr(20.955)
        ch.CreateVerticalApertureAttr(20.955 * 9 / 16)
        ch.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        chase_path = "/World/G1Chase"

    # ------------------------------------------------------------------ play + init (runner order)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["physics_view_ok"] = SimulationManager.get_physics_sim_view() is not None
    report["init"] = loco.initialize()
    base = L.G1Base(loco)                                  # CALLBACK mode: the runner's kit.update() drives physics
    report["drive_mode"] = "callback" if base.callback_mode else "explicit"
    cams = {}
    if cam_path:
        from isaacsim.sensors.camera import Camera

        cams["nav"] = Camera(prim_path=cam_path, resolution=tuple(nav_spec["resolution_wh"]))
        if chase_path and args.chase_from <= 0:
            cams["chase"] = Camera(prim_path=chase_path, resolution=(1280, 720))
        for c in cams.values():
            c.initialize()
    for _ in range(10):
        kit.update()

    floor = L.floor_below(start_xy[0], start_xy[1], start_z + L.SPAWN_PROBE_ABOVE_M, skip_prefix=loco.prim_path)
    report["floor"] = {"z": floor[0] if floor else None, "prim": floor[1] if floor else None}

    # ------------------------------------------------------------------ shared run helpers
    def read_state():
        st = loco.state()
        return st

    def run_updates(cmd_fn, seconds, rows=None, row_dt=0.1, on_update=None, stop_on_fall=True):
        """Runner-style loop for `seconds` of CONTROL time. cmd_fn(t) -> (vx, vy, wz)."""
        n = int(round(seconds / CTL_DT))
        sub_hist: dict[int, int] = {}
        ticks0, steps0 = loco.ticks, base.physics_steps
        t_wall = time.perf_counter()
        t_upd = 0.0
        next_row = 0.0
        for k in range(n):
            t = k * CTL_DT
            cmd = cmd_fn(t)
            allow = getattr(cmd_fn, "allow_untrained", False)
            if allow:
                loco.set_command(*cmd, allow_untrained=True)
            else:
                base.apply(*cmd, CTL_DT)
            s0 = base.physics_steps
            a = time.perf_counter()
            kit.update()
            t_upd += time.perf_counter() - a
            d = base.physics_steps - s0
            sub_hist[d] = sub_hist.get(d, 0) + 1
            ps = base.physics_state()
            if on_update:
                on_update(k, t, ps)
            if rows is not None and t >= next_row - 1e-9:
                next_row += row_dt
                st = loco.state()
                rows.append({"t": round(t, 3), "cmd": list(map(float, loco.cmd)),
                             "pos": [round(float(v), 4) for v in st["pos"]], "yaw_deg": round(math.degrees(st["yaw"]), 3),
                             "v_b": [round(float(v), 4) for v in st["lin_b"]], "w_b": [round(float(v), 4) for v in st["ang_b"]],
                             "pelvis_h": round(st["height"], 4), "above_feet": round(st["above_feet"], 4),
                             "tilt": round(st["tilt_deg"], 3), "roll": round(st["roll_deg"], 3), "pitch": round(st["pitch_deg"], 3)})
            if ps["fallen"] and stop_on_fall:
                return {"fell_at_s": round(t + CTL_DT, 3), "why": base.last_fault["why"] if base.last_fault else None,
                        "sub_hist": sub_hist, "updates": k + 1, "wall_s": time.perf_counter() - t_wall,
                        "update_wall_s": t_upd, "ticks": loco.ticks - ticks0, "phys_steps": base.physics_steps - steps0}
        return {"fell_at_s": None, "sub_hist": sub_hist, "updates": n, "wall_s": time.perf_counter() - t_wall,
                "update_wall_s": t_upd, "ticks": loco.ticks - ticks0, "phys_steps": base.physics_steps - steps0}

    def reset_flat():
        base.reset_to((start_xy[0], start_xy[1], start_z), start_yaw)
        base.reset_stats()
        loco.set_command(0, 0, 0)
        for _ in range(3):
            kit.update()

    def steady(rows, t0, t1):
        rs = [r for r in rows if t0 <= r["t"] <= t1]
        if not rs:
            return None
        v = np.array([r["v_b"] for r in rs])
        w = np.array([r["w_b"][2] for r in rs])
        p0, p1 = np.array(rs[0]["pos"][:2]), np.array(rs[-1]["pos"][:2])
        y0, y1 = math.radians(rs[0]["yaw_deg"]), math.radians(rs[-1]["yaw_deg"])
        dt = rs[-1]["t"] - rs[0]["t"]
        ym = y0 + math.atan2(math.sin(y1 - y0), math.cos(y1 - y0)) / 2
        d = p1 - p0
        disp = None
        if dt > 0.3:
            disp = [round((math.cos(ym) * d[0] + math.sin(ym) * d[1]) / dt, 3),
                    round((-math.sin(ym) * d[0] + math.cos(ym) * d[1]) / dt, 3),
                    round(math.atan2(math.sin(y1 - y0), math.cos(y1 - y0)) / dt, 3)]
        return {"vx": round(float(v[:, 0].mean()), 3), "vy": round(float(v[:, 1].mean()), 3), "wz": round(float(w.mean()), 3),
                "disp_vx_vy_wz": disp, "above_feet_min": round(min(r["above_feet"] for r in rs), 3),
                "above_feet_mean": round(float(np.mean([r["above_feet"] for r in rs])), 3),
                "tilt_max": round(max(r["tilt"] for r in rs), 2), "roll_absmax": round(max(abs(r["roll"]) for r in rs), 2),
                "pitch_absmax": round(max(abs(r["pitch"]) for r in rs), 2)}

    def set_cmd_fn(cmd, allow_untrained=False):
        def fn(t):
            return cmd
        fn.allow_untrained = allow_untrained
        return fn

    gpu_setup = gpu.read()
    report["gpu_after_setup"] = gpu_setup

    # ====================================================================== MODES
    if "cadence" in MODES:
        reset_flat()
        sched = [("stand", 2.0, (0, 0, 0)), ("vx0.5", 6.0, (0.5, 0, 0)), ("wz+0.6", 3.0, (0, 0, 0.6)),
                 ("vx0.8", 4.0, (0.8, 0, 0)), ("stop", 2.0, (0, 0, 0))]
        rows, per = [], {}
        t_off = 0.0
        seg_t = {}
        for name, secs, cmd in sched:
            seg_rows_start = len(rows)

            def fn(t, cmd=cmd):
                return cmd
            fn.allow_untrained = False
            # time axis continues across segments
            def on_u(k, t, ps, off=t_off):
                pass
            res = run_updates(lambda t, c=cmd: c, secs, rows=None)
            per[name] = {k: v for k, v in res.items() if k != "sub_hist"}
            per[name]["sub_steps_per_update"] = res["sub_hist"]
            t_off += secs
        # separate pass with rows for tracking (re-run of vx0.5 and vx0.8 from rest)
        reset_flat()
        run_updates(lambda t: (0, 0, 0), 2.0)
        rows = []
        res = run_updates(lambda t: (0.5, 0, 0), 6.0, rows=rows)
        track = steady(rows, 2.0, 6.0)
        total_updates = sum(v["updates"] for v in per.values())
        total_phys = sum(v["phys_steps"] for v in per.values())
        total_ticks = sum(v["ticks"] for v in per.values())
        wall = sum(v["wall_s"] for v in per.values())
        report["cadence"] = {
            "updates": total_updates, "control_seconds": round(total_updates * CTL_DT, 3),
            "physics_substeps": total_phys, "physics_seconds": round(total_phys * loco.physics_dt, 4),
            "sim_clock_drift_s": round(total_phys * loco.physics_dt - total_updates * CTL_DT, 4),
            "substeps_per_update_mean": round(total_phys / total_updates, 4), "ideal": round(CTL_DT / loco.physics_dt, 4),
            "policy_ticks": total_ticks, "ticks_per_sim_second": round(total_ticks / (total_phys * loco.physics_dt), 3),
            "segments": per, "vx0.5_tracking_runner_style": track, "vx0.5_fell": res["fell_at_s"],
            "wall_per_control_second": round(wall / (total_updates * CTL_DT), 3),
            "update_wall_per_control_second": round(sum(v["update_wall_s"] for v in per.values()) / (total_updates * CTL_DT), 3),
            "camera": args.camera, "falls": [k for k, v in per.items() if v["fell_at_s"] is not None],
        }

    if "envelope" in MODES:
        trials = []
        for vx in (0.0, 0.3, 0.6, 1.0):
            for vy in (-0.5, 0.0, 0.5):
                for wz in (-1.0, 0.0, 1.0):
                    trials.append((f"in_vx{vx}_vy{vy}_wz{wz}", (vx, vy, wz), False))
        for vx in (-0.2, -0.4, -0.6):
            trials.append((f"OOD_vx{vx}", (vx, 0.0, 0.0), True))
        for wz in (-1.5, 1.5):
            trials.append((f"OOD_wz{wz}", (0.0, 0.0, wz), True))
        trials.append(("OOD_vx1.3", (1.3, 0.0, 0.0), True))
        trials.append(("OOD_vx0.6_wz1.5", (0.6, 0.0, 1.5), True))
        results = {}
        for name, cmd, ood in trials:
            reset_flat()
            res0 = run_updates(set_cmd_fn((0, 0, 0)), 1.5)
            rows = []
            res = run_updates(set_cmd_fn(cmd, allow_untrained=ood), 4.0, rows=rows)
            results[name] = {"cmd": list(cmd), "stand_fell": res0["fell_at_s"], "fell_at_s": res["fell_at_s"],
                             "why": res.get("why"), "steady_last2s": steady(rows, 2.0, 4.0)}
            st = results[name]["steady_last2s"]
            print(f"[envelope] {name:28s} fell={res['fell_at_s']} "
                  f"{'' if st is None else (st['vx'], st['vy'], st['wz'])}", flush=True)
            save_report()
        report["envelope"] = results
        report["envelope_summary"] = {
            "in_range_trials": sum(1 for n, _, o in trials if not o),
            "in_range_falls": [n for n, _, o in trials if not o and results[n]["fell_at_s"] is not None],
            "ood_falls": [n for n, _, o in trials if o and results[n]["fell_at_s"] is not None],
            "ood_trials": [n for n, _, o in trials if o],
        }

    if "stress" in MODES:
        from controllers import make_controller

        def arc_plan(kappa, v, n=30, dt=0.1):
            s = np.arange(1, n + 1) * v * dt
            if abs(kappa) < 1e-6:
                return np.stack([s, np.zeros_like(s)], 1)
            return np.stack([np.sin(kappa * s) / kappa, (1 - np.cos(kappa * s)) / kappa], 1)

        results = []
        kappas = (-0.60, -0.35, -0.15, 0.0, 0.15, 0.35, 0.60)
        for gen in ("random_box", "box_corners", "pursuit_arcs"):
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * (1 + ["random_box", "box_corners", "pursuit_arcs"].index(gen)) + seed)
                # a command sequence long enough for stress-seconds, held 0.1-1.5 s each
                seq, tt = [], 0.0
                ctrl = make_controller("braking", 1.0, 1.0)
                ctrl.reset()
                while tt < args.stress_seconds + 2.0:
                    hold = float(rng.uniform(0.1, 1.5)) if gen != "pursuit_arcs" else 0.5
                    if gen == "random_box":
                        cmd = (float(rng.uniform(0, 1)), float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-1, 1)))
                    elif gen == "box_corners":
                        cmd = (float(rng.choice([0.0, 1.0])), float(rng.choice([-0.5, 0.0, 0.5])), float(rng.choice([-1.0, 0.0, 1.0])))
                    else:
                        v = float(rng.uniform(0.3, 1.0))
                        c = ctrl(arc_plan(float(rng.choice(kappas)), v))
                        cmd = (c.vx, c.vy, c.omega)
                    seq.append((tt, tt + hold, cmd))
                    tt += hold
                idx = {"i": 0}

                def cmd_fn(t, seq=seq, idx=idx):
                    while idx["i"] < len(seq) - 1 and t >= seq[idx["i"]][1]:
                        idx["i"] += 1
                    return seq[idx["i"]][2]
                reset_flat()
                run_updates(set_cmd_fn((0, 0, 0)), 1.5)
                rows = []
                res = run_updates(cmd_fn, args.stress_seconds, rows=rows)
                tr = steady(rows, 1.0, args.stress_seconds)
                r = {"gen": gen, "seed": seed, "fell_at_s": res["fell_at_s"], "why": res.get("why"),
                     "seconds_run": round(res["updates"] * CTL_DT, 2), "n_cmd_changes": len(seq),
                     "mean_speed_vx": None if tr is None else tr["vx"], "tilt_max": None if tr is None else tr["tilt_max"],
                     "above_feet_min": None if tr is None else tr["above_feet_min"],
                     "path_m": round(float(sum(math.dist(rows[i]["pos"][:2], rows[i + 1]["pos"][:2]) for i in range(len(rows) - 1))), 2),
                     "wall_per_sim_s": round(res["wall_s"] / (res["updates"] * CTL_DT), 3)}
                results.append(r)
                print(f"[stress] {gen} seed {seed}: {r}", flush=True)
                report["stress"] = results
                save_report()
        report["stress_summary"] = {
            "runs": len(results), "falls": sum(1 for r in results if r["fell_at_s"] is not None),
            "sim_seconds": round(sum(r["seconds_run"] for r in results), 1),
            "by_gen": {g: {"runs": sum(1 for r in results if r["gen"] == g),
                           "falls": sum(1 for r in results if r["gen"] == g and r["fell_at_s"] is not None)}
                       for g in ("random_box", "box_corners", "pursuit_arcs")}}

    if "route" in MODES:
        from controllers import make_controller

        ep_start = (ep.start[0], ep.start[1], ep.start[2])
        base.reset_to(ep_start, ep.start_yaw_rad)
        base.reset_stats()
        report["spawn"] = base.spawn
        loco.set_command(0, 0, 0)
        for _ in range(30):
            kit.update()
        floor_z = loco.floor_z

        def lum(frame):
            f = frame.astype(np.float32)
            return round(float((0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).mean()), 2)

        def grab_nav():
            return np.asarray(cams["nav"].get_rgba())[:, :, :3].astype(np.uint8)

        if cams and light_path:
            # EXPOSURE A/B on the standing robot (physics keeps running, the policy keeps it upright):
            # headlight off vs on, each given 150 updates to settle (alohamini1_specs.py: gains from
            # exposure settings looked real at 40-60 steps and vanished at 150), then the same scene through
            # render_only() (no physics step) to see whether the render PATH changes the exposure.
            from pxr import UsdLux

            inten = UsdLux.DiskLight(stage.GetPrimAtPath(light_path)).GetIntensityAttr()
            ab = {}
            inten.Set(0.0)
            for _ in range(150):
                base.apply(0, 0, 0, CTL_DT)
                kit.update()
            f_off = grab_nav()
            ab["headlight_off_kitupdate_150"] = lum(f_off)
            inten.Set(float(L.NAV_HEADLIGHT_INTENSITY))
            for n_ in (1, 10, 60, 150):
                for _ in range(n_ - (0 if n_ == 1 else {10: 1, 60: 10, 150: 60}[n_])):
                    base.apply(0, 0, 0, CTL_DT)
                    kit.update()
                ab[f"headlight_on_kitupdate_{n_}"] = lum(grab_nav())
            Image.fromarray(grab_nav()).save(OUT / "ab_headlight_on.jpg", quality=92)
            Image.fromarray(f_off).save(OUT / "ab_headlight_off.jpg", quality=92)
            for n_ in (1, 10, 60):
                for _ in range(n_ if n_ == 1 else (n_ - (1 if n_ == 10 else 10))):
                    L.render_only(kapp, loco.physics_dt, 0.0)
                ab[f"headlight_on_render_only_{n_}"] = lum(grab_nav())
            for _ in range(60):
                base.apply(0, 0, 0, CTL_DT)
                kit.update()
            ab["headlight_on_after_back_to_kitupdate_60"] = lum(grab_nav())
            report["exposure_ab"] = ab
            print(f"[route] exposure A/B (mean luminance 0-255): {ab}", flush=True)
            save_report()
        # closed-loop pursuit along start -> (x0 + 0.35, y0 - 12) -> (x0 - 0.25, y0 - 26): a gentle dogleg
        x0, y0 = ep.start[0], ep.start[1]
        path = [(x0, y0), (x0 + 0.35, y0 - 12.0), (x0 - 0.25, y0 - 24.0)]
        ctrl = make_controller("braking", 1.0, 1.0)
        ctrl.reset()

        def plan_in_robot_frame(px, py, yaw, n=30, dt=0.1, v=0.7):
            """Waypoints along the polyline ahead of the robot, in the robot FLU frame."""
            # nearest point on the polyline, then walk forward v*dt per point
            pts = np.array(path)
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

        phase = {"nav": [0, 0.0], "nav+chase": [0, 0.0]}
        chase_yaw = {"v": ep.start_yaw_rad}

        def aim_chase(pos, yaw_):
            """World-frame chase view 2.6 m behind, 1.35 m up, yaw low-passed (the pelvis yaw wobbles)."""
            chase_yaw["v"] += 0.15 * math.atan2(math.sin(yaw_ - chase_yaw["v"]), math.cos(yaw_ - chase_yaw["v"]))
            cy = chase_yaw["v"]
            eye = (pos[0] - 2.6 * math.cos(cy), pos[1] - 2.6 * math.sin(cy), floor_z + 1.35)
            m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(pos[0], pos[1], floor_z + 0.75), Gf.Vec3d(0, 0, 1)).GetInverse()
            qq = m.ExtractRotationQuat()
            im = qq.GetImaginary()
            cams["chase"].set_world_pose(position=np.array(eye, dtype=float),
                                         orientation=np.array([qq.GetReal(), im[0], im[1], im[2]]), camera_axes="usd")

        chase_k = int(round(args.chase_from / CTL_DT)) if args.chase_from > 0 else None
        rows, cam_rows, lum_rows = [], [], []
        save_dir = OUT / "frames"
        save_dir.mkdir(exist_ok=True)
        decision_every = int(round(0.5 / CTL_DT))                    # replan_every_steps = 30
        plan_cache = {"cmd": (0.0, 0.0, 0.0), "wp": None}
        timing = {"update_s": 0.0, "grab_s": 0.0, "policy_s": 0.0, "measure_s": 0.0}
        sub_hist: dict[int, int] = {}
        fell_at, why = None, None
        n_ctl = int(round(args.route_seconds / CTL_DT))
        prev_phys = None
        cross_track = []
        t_loop = time.perf_counter()
        prev_pose_eye = None
        for k in range(n_ctl):
            t = k * CTL_DT
            x, y, z_floor = base.position()
            yaw = base.yaw
            if k < int(round(2.0 / CTL_DT)):
                cmd = (0.0, 0.0, 0.0)                                  # stand 2 s first, like the walk tests
            else:
                if (k % decision_every) == 0 or plan_cache["wp"] is None:
                    wp, off = plan_in_robot_frame(x, y, yaw)
                    plan_cache["wp"] = wp
                    cross_track.append(off)
                c = ctrl(plan_cache["wp"])
                cmd = (c.vx, c.vy, c.omega)
            if chase_k is not None and k == chase_k and "chase" not in cams:
                from isaacsim.sensors.camera import Camera

                cams["chase"] = Camera(prim_path=chase_path, resolution=(1280, 720))
                cams["chase"].initialize()
                aim_chase(base.loco.pose()[:3], yaw)
                report["chase_created_at_s"] = round(t, 2)
            if "chase" in cams and k % 3 == 0:
                aim_chase(base.loco.pose()[:3], yaw)
            base.apply(*cmd, CTL_DT)
            s0 = base.physics_steps
            a = time.perf_counter()
            kit.update()
            du = time.perf_counter() - a
            timing["update_s"] += du
            if k >= int(round(4.0 / CTL_DT)):                       # walking only
                ph = phase["nav+chase" if "chase" in cams else "nav"]
                ph[0] += 1
                ph[1] += du
            d = base.physics_steps - s0
            sub_hist[d] = sub_hist.get(d, 0) + 1
            ps = base.physics_state()
            if k % decision_every == 0 and cams:
                a = time.perf_counter()
                rgba = cams["nav"].get_rgba()
                frame = np.asarray(rgba)[:, :, :3].astype(np.uint8)
                timing["grab_s"] += time.perf_counter() - a
                lum_rows.append(lum(frame))
                m0 = time.perf_counter()
                cpos, cq = cams["nav"].get_world_pose(camera_axes="usd")
                rq = L.quat_to_matrix_wxyz(np.asarray(cq, dtype=float))
                view = -rq[:, 2]
                right = rq[:, 0]
                mine = loco.camera_pose(nav_spec)
                st = loco.state()
                e_usd = np.asarray(cpos, dtype=float)
                cam_rows.append({
                    "t": round(t, 3), "eye_err_m": round(float(np.linalg.norm(e_usd - mine["eye"])), 4),
                    "h_usd": round(float(e_usd[2] - floor_z), 4), "h_physx": round(mine["height"], 4),
                    "pitch_usd": round(math.degrees(math.asin(max(-1, min(1, -view[2])))), 3),
                    "pitch_physx": round(mine["pitch_down_deg"], 3),
                    "roll_usd": round(math.degrees(math.asin(max(-1, min(1, right[2])))), 3),
                    "roll_physx": round(mine["roll_deg"], 3),
                    "view_yaw_usd_deg": round(math.degrees(math.atan2(view[1], view[0])), 3),
                    "pelvis_yaw_deg": round(math.degrees(st["yaw"]), 3),
                    "head_minus_pelvis_yaw_usd_deg": round(math.degrees(math.atan2(math.sin(math.atan2(view[1], view[0]) - st["yaw"]),
                                                                                    math.cos(math.atan2(view[1], view[0]) - st["yaw"]))), 3),
                    "fwd_of_pelvis_physx": round(mine["forward_of_pelvis"], 4), "left_of_pelvis_physx": round(mine["left_of_pelvis"], 4),
                    "torso_joint_deg": round(math.degrees(float(np.asarray(loco.art.get_joint_positions()).reshape(-1)[
                        list(loco.art.dof_names).index("torso_joint")])), 3)})
                timing["measure_s"] += time.perf_counter() - m0
                if args.save_every > 0 and (k % int(round(args.save_every / CTL_DT)) == 0):
                    Image.fromarray(frame).save(save_dir / f"nav_t{t:05.1f}.jpg", quality=92)
            if k % 6 == 0:
                st = loco.state()
                rows.append({"t": round(t, 3), "cmd": [round(v, 3) for v in cmd], "pos": [round(float(v), 3) for v in st["pos"]],
                             "yaw_deg": round(math.degrees(st["yaw"]), 2), "v_b": [round(float(v), 3) for v in st["lin_b"]],
                             "above_feet": round(st["above_feet"], 3), "tilt": round(st["tilt_deg"], 2)})
            if ps["fallen"]:
                fell_at, why = round(t + CTL_DT, 3), base.last_fault["why"]
                print(f"[route] FELL at {fell_at}: {why}", flush=True)
                break
            if k % 300 == 0:
                print(f"[route] t={t:5.1f} cmd=({cmd[0]:.2f},{cmd[2]:+.2f}) pos=({x:.2f},{y:.2f}) wall={time.perf_counter() - t_loop:.1f}s", flush=True)
        wall = time.perf_counter() - t_loop
        sim_s = (k + 1) * CTL_DT
        cr = cam_rows
        def agg(key, sl=slice(None)):
            a = np.array([c[key] for c in cr[sl]], dtype=float)
            return None if not len(a) else {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
                                            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}
        walking = [c for c in cr if c["t"] >= 4.0]
        def aggw(key):
            a = np.array([c[key] for c in walking], dtype=float)
            return None if not len(a) else {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
                                            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}
        report["route"] = {
            "fell_at_s": fell_at, "why": why, "sim_seconds": round(sim_s, 2), "wall_s": round(wall, 2),
            "wall_per_sim_s": round(wall / sim_s, 3), "timing_s": {k_: round(v, 2) for k_, v in timing.items()},
            "update_ms_per_control_step": round(1000 * timing["update_s"] / (k + 1), 2),
            "grab_ms_per_decision": round(1000 * timing["grab_s"] / max(1, len(cam_rows)), 2),
            "sub_steps_per_update": sub_hist, "decisions": len(cam_rows),
            "final_pos": rows[-1]["pos"] if rows else None, "path_m": round(float(sum(math.dist(rows[i]["pos"][:2], rows[i + 1]["pos"][:2]) for i in range(len(rows) - 1))), 2),
            "mean_speed_mps": None, "cross_track_at_decisions_m": {"mean": round(float(np.mean(cross_track)), 3) if cross_track else None,
                                                                    "max": round(float(np.max(cross_track)), 3) if cross_track else None},
            "camera_truth_after_4s": {key: aggw(key) for key in (
                "eye_err_m", "h_usd", "h_physx", "pitch_usd", "pitch_physx", "roll_usd", "roll_physx",
                "head_minus_pelvis_yaw_usd_deg", "torso_joint_deg", "fwd_of_pelvis_physx", "left_of_pelvis_physx")} if walking else None,
            "camera_truth_first_decision": cr[0] if cr else None,
            "update_ms_per_control_step_walking": {k_: (None if v[0] == 0 else round(1000 * v[1] / v[0], 2)) for k_, v in phase.items()},
            "update_counts_walking": {k_: v[0] for k_, v in phase.items()},
            "frame_mean_luminance": {"n": len(lum_rows), "mean": round(float(np.mean(lum_rows)), 2) if lum_rows else None,
                                     "min": min(lum_rows) if lum_rows else None, "max": max(lum_rows) if lum_rows else None},
        }
        if rows and len(rows) > 2:
            dt_ = rows[-1]["t"] - rows[8]["t"] if len(rows) > 8 else 1.0
            report["route"]["mean_speed_mps"] = round(float(math.dist(rows[8]["pos"][:2], rows[-1]["pos"][:2]) / max(dt_, 1e-6)), 3) if len(rows) > 8 else None
        report["route_cam_rows"] = cam_rows
        report["route_rows"] = rows
        root_dirty = stage.GetRootLayer().dirty
        report["stage_file"]["root_layer_dirty"] = bool(root_dirty)

        # ---- start survey over every episode of this stage's environment (g1_start_survey.py's loop,
        # here with the cameras already alive, so its wall numbers include rendering)
        if args.survey:
            import carb
            from omni.physx import get_physx_scene_query_interface
            from episode import episodes_by_env

            qry = get_physx_scene_query_interface()

            def ray(origin, direction, dist, skip="/World/G1"):
                o = np.array(origin, dtype=float)
                dv = np.array(direction, dtype=float)
                for _ in range(30):
                    hit = qry.raycast_closest(carb.Float3(*o.tolist()), carb.Float3(*dv.tolist()), dist)
                    if not hit or not hit.get("hit"):
                        return None, ""
                    prim = str(hit.get("collision", ""))
                    hp = np.array([float(hit["position"][i]) for i in range(3)])
                    if not prim.startswith(skip):
                        return float(np.linalg.norm(hp - np.array(origin, dtype=float))), prim
                    o = (o + 0.03 * dv) if float(hit.get("distance", 0.0)) < 1e-4 else (hp + 0.01 * dv)
                return None, ""

            def clearances(x, y, yaw_, fz):
                out = {}
                dd, pp = ray((x, y, fz + 0.30), (0, 0, 1), 6.0)
                out["headroom_m"] = None if dd is None else round(dd + 0.30, 3)
                out["headroom_prim"] = pp.split("/")[-1] if pp else ""
                for hh in (1.0, 0.30):
                    for nm, a_ in (("fwd", 0.0), ("left", math.pi / 2), ("right", -math.pi / 2), ("back", math.pi)):
                        d2, p2 = ray((x + 0.35 * math.cos(yaw_ + a_), y + 0.35 * math.sin(yaw_ + a_), fz + hh),
                                     (math.cos(yaw_ + a_), math.sin(yaw_ + a_), 0.0), 12.0)
                        out[f"{nm}@{hh:.2f}"] = (None if d2 is None else round(d2 + 0.35, 3), p2.split("/")[-1] if p2 else "")
                return out

            def collider_extents():
                out = {}
                pelvis = stage.GetPrimAtPath(loco.root_path)
                inv = UsdGeom.Xformable(pelvis).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
                for path_ in report["author"]["colliders"]:
                    prim = stage.GetPrimAtPath(path_)
                    cube = UsdGeom.Cube(prim)
                    if not cube:
                        continue
                    half = float(cube.GetSizeAttr().Get()) / 2.0
                    m_ = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    pts = [inv.Transform(m_.Transform(Gf.Vec3d(sx * half, sy * half, sz * half)))
                           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                    arr = np.array([[p_[0], p_[1], p_[2]] for p_ in pts])
                    out[path_.replace("/World/G1/", "")] = {"min_xyz": arr.min(0).round(3).tolist(), "max_xyz": arr.max(0).round(3).tolist()}
                return out

            eps_here = [e for e in episodes_by_env()["hospital"]]
            survey = []
            tot_u = tot_w = 0
            for e_ in eps_here:
                r_: dict = {"episode": e_.name, "start": list(e_.start), "yaw_deg": e_.start_yaw_deg}
                base.reset_to(tuple(e_.start), e_.start_yaw_rad)
                base.reset_stats()
                loco.set_command(0, 0, 0)
                r_["spawn"] = base.spawn
                for _ in range(3):
                    kit.update()
                if not survey:
                    try:
                        report["collider_extents_pelvis_frame_m"] = collider_extents()
                    except Exception as ex:  # noqa: BLE001
                        report["collider_extents_pelvis_frame_m"] = repr(ex)
                r_["clearance"] = clearances(e_.start[0], e_.start[1], e_.start_yaw_rad, loco.floor_z)
                rows_s = []
                fell_s = None
                for phase_name, cmd_s, secs in (("stand", (0, 0, 0), 2.0), ("walk", (0.5, 0.0, 0.0), 4.0)):
                    n_s = int(round(secs / CTL_DT))
                    for kk in range(n_s):
                        base.apply(*cmd_s, CTL_DT)
                        a = time.perf_counter()
                        kit.update()
                        tot_w += time.perf_counter() - a
                        tot_u += 1
                        pss = base.physics_state()
                        if phase_name == "walk" and kk % 6 == 0:
                            st_ = loco.state()
                            rows_s.append((kk * CTL_DT, [float(v) for v in st_["pos"]], float(st_["above_feet"]), float(st_["tilt_deg"])))
                        if pss["fallen"]:
                            fell_s = {"phase": phase_name, "t": round((kk + 1) * CTL_DT, 2), "why": base.last_fault["why"]}
                            break
                    if fell_s:
                        break
                r_["fell"] = fell_s
                if rows_s:
                    p0_, p1_ = np.array(rows_s[0][1][:2]), np.array(rows_s[-1][1][:2])
                    r_["walk"] = {"distance_m": round(float(np.linalg.norm(p1_ - p0_)), 3),
                                  "mean_speed_mps": round(float(np.linalg.norm(p1_ - p0_) / max(rows_s[-1][0] - rows_s[0][0], 1e-6)), 3),
                                  "above_feet_min": round(min(x_[2] for x_ in rows_s), 3), "tilt_max": round(max(x_[3] for x_ in rows_s), 2)}
                r_["torso_contact_steps"] = base.torso_contact_steps
                survey.append(r_)
                print(f"[survey] {e_.name}: fell={fell_s} headroom={r_['clearance'].get('headroom_m')} walk={r_.get('walk')}", flush=True)
                report["survey"] = survey
                save_report()
            report["survey_cost"] = {"updates": tot_u, "ms_per_update": round(1000 * tot_w / max(tot_u, 1), 2),
                                     "cameras": sorted(cams)}

    report["gpu_mem"] = gpu.summary()
    report["launch_to_end_s"] = round(time.time() - T_LAUNCH, 1)
    report["status"] = "ok"
    rc = 0
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    report["status"] = f"error: {e!r}"
    rc = 1
finally:
    gpu.stop_flag = True
    try:
        if report.get("stage_file"):
            report["stage_file"]["md5_after"] = hashlib.md5(Path(report["stage_file"]["path"]).read_bytes()).hexdigest()
            report["stage_file"]["unchanged"] = report["stage_file"]["md5_after"] == report["stage_file"]["md5_before"]
    except Exception as e:  # noqa: BLE001
        report.setdefault("stage_file", {})["md5_after"] = repr(e)
    save_report()
    print(f"[runner_style] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    print(f"[runner_style] STATUS={'OK' if rc == 0 else 'FAILED'}", flush=True)
    sys.stdout.flush()
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    sys.exit(rc)
