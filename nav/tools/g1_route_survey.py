"""Can the walking G1 (current nav/sim/g1/locomotion.py) walk the first 10 m of every DynaNav route?

Replaces nav/tools/g1_start_survey.py as the pre-integration survey. That one was a SPAWN/STAND survey
(2 s stand + 4 s straight at vx 0.5 = ~1.4 m per pose, on pre-round-2 code: no slew limit, no hold, older
stats); g1/check rightly said "27/27 stand and walk" overstated it. This one drives each pose the way the
runner will: one kit.update() per 1/60 s, G1Base in CALLBACK mode with the shipping defaults (slew limit,
hold on stop), the runner's pure-pursuit controller, and G1Guard APPLIED (the guard the integration plan
names), along the episode's own navmesh route.

ROUTES. route_drive.py (the scenes/physics-fix tool) bakes each episode's navmesh shortest path and writes
the points into nav/results/route_drive/*.json; the newest file holding an episode wins. Those routes were
baked for AlohaMini with agent radius 0.40 m (the G1's arms reach ~0.30 m from the pelvis axis, its torso
box ~0.15 m), DynaNav's per-scene step/slope. A pose group (several episodes sharing a start) uses the
first episode in the group that has a route. A pose with NO route on disk falls back to the straight line
from the start toward its own goal, labelled `route_source: straight_to_goal` -- the guard stops it at
whatever is in the way, which is reported, not hidden.

PER POSE, in one Kit session per environment (the stage opened IN MEMORY, /World/Aloha deactivated, every
edit in the session layer, the file's md5 checked unchanged):
  1. G1Base.reset_to(episode start, yaw) -- the floor raycast snap (g1_spawn_pose); no_floor reported.
  2. stand --stand-s with a zero command (hold on).
  3. follow the route's first --route-m metres (clipped to the route length) with
     controllers.make_controller("pursuit", --speed, --max-yaw-rate, lookahead 1.0 m), body-frame waypoints
     every 5 cm over a 3 m horizon. Like route_drive: when the lookahead point is more than 45 deg off the
     nose the G1 turns IN PLACE (vx 0, wz +-max-yaw-rate) until within 15 deg -- a trained command (vx 0 is
     inside the box). Every command goes through G1Guard.check (--guard apply) before G1Base.apply.
     Ends: within 0.3 m of the route segment's end; or a G1Base terminal state (fall, free fall, no floor,
     physics_dt_mismatch, handle lost); or a timeout of 2.5 x length / speed + 15 s.
  4. stop: --hold-s of zero command -- the hold band (max distance from the latched pose, G1Base's own
     figure, and the tool's per-update one) and the displacement over the last 2 s.
  5. reverse request: --reverse-s of vx -0.3 -- must HOLD (reverse_requests_held, max distance from where
     the request started).
Recorded per pose: route source and length, s reached, path length, mean speed, max cross-track error,
fell / free fall / faults, tilt max, vz min, torso contacts, guard interventions and which ring fired,
pivots, wall time per sim second and PhysX ms per sub-step with the GPU0 load over the same window.

--video-episode NAME saves head-camera (D435 re-pitched 12 deg + headlight) and chase frames at --fps for
that pose, which is run LAST (a Camera renders every update from creation on, so every other pose is
measured without cameras); encode with nav/tools/g1_encode_video.py --run <out>.

    ISAAC_SLOT_MIN_FREE_MIB=8000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \
        nav/tools/g1_route_survey.py --env hospital --out <dir>

report.json in <out>; the exit code is NOT the verdict (kit.close() ends the process).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# nav/tools/profile.py shadows the stdlib `profile`, which Kit's warp imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

CTL = 1.0 / 60.0
ap = argparse.ArgumentParser()
ap.add_argument("--env", choices=["hospital", "office", "warehouse", "outdoor_small"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episodes-file", default=str(REPO / "nav" / "config" / "episodes_full.yaml"))
ap.add_argument("--only", default="", help="comma list of episode names (pose-group leaders or members)")
ap.add_argument("--route-m", type=float, default=10.0)
ap.add_argument("--speed", type=float, default=0.6)
ap.add_argument("--max-yaw-rate", type=float, default=0.8)
ap.add_argument("--stand-s", type=float, default=2.0)
ap.add_argument("--hold-s", type=float, default=4.0)
ap.add_argument("--reverse-s", type=float, default=2.0)
ap.add_argument("--guard", choices=["apply", "observe", "off"], default="apply")
ap.add_argument("--video-episode", default="")
ap.add_argument("--fps", type=float, default=10.0)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- routes (no Kit needed)
import numpy as np  # noqa: E402


def load_routes() -> dict:
    """episode name -> (file, route dict with 'points'), newest route_drive file wins."""
    out = {}
    for f in sorted(glob.glob(str(REPO / "nav" / "results" / "route_drive" / "*.json")), key=os.path.getmtime):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        eps = d.get("episodes") or []
        items = list(eps.items()) if isinstance(eps, dict) else [(e.get("name"), e) for e in eps]
        for name, e in items:
            r = (e or {}).get("route") or {}
            if isinstance(r, dict) and isinstance(r.get("points"), list) and len(r["points"]) >= 2:
                out[name] = (os.path.relpath(f, REPO), r)
    return out


class Route:
    """2-D polyline with arc length (the same parametrisation as route_drive.Route)."""

    def __init__(self, pts):
        clean = []
        for p in pts:
            p = (float(p[0]), float(p[1]))
            if not clean or math.dist(clean[-1], p) > 1e-3:
                clean.append(p)
        if len(clean) == 1:
            clean.append(clean[0])
        self.p = np.array(clean)
        d = np.diff(self.p, axis=0)
        self.seglen = np.hypot(d[:, 0], d[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seglen)])
        self.length = float(self.cum[-1])

    def points_at(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return np.stack([np.interp(s, self.cum, self.p[:, 0]), np.interp(s, self.cum, self.p[:, 1])], axis=-1)

    def project(self, xy, s_lo, s_hi):
        s_lo, s_hi = max(0.0, s_lo), min(self.length, s_hi)
        ss = np.linspace(s_lo, s_hi, max(2, int((s_hi - s_lo) / 0.02) + 1))
        q = self.points_at(ss)
        d = np.hypot(q[:, 0] - xy[0], q[:, 1] - xy[1])
        k = int(np.argmin(d))
        return float(ss[k]), float(d[k])

    def heading_change_deg(self, s_end) -> float:
        q = self.points_at(np.append(np.arange(0.0, s_end, 1.0), s_end))
        dd = np.diff(q, axis=0)
        dd = dd[np.hypot(dd[:, 0], dd[:, 1]) > 1e-6]
        if len(dd) < 2:
            return 0.0
        h = np.arctan2(dd[:, 1], dd[:, 0])
        return float(np.degrees(np.abs(np.angle(np.exp(1j * np.diff(h)))).sum()))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"env": args.env, "args": vars(args), "pid": os.getpid(), "results": []}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


from gpu_poll import GpuPoll  # noqa: E402

poll = GpuPoll(1.0, t0=T0)
poll.start()
rc = 1
try:
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux

    import guard as G
    import locomotion as L
    from controllers import make_controller
    from episode import episodes_by_env

    report["code_md5"] = {p: hashlib.md5((REPO / p).read_bytes()).hexdigest()
                          for p in ("nav/sim/g1/locomotion.py", "nav/sim/g1/guard.py", "nav/sim/g1/walk_policy.py",
                                    "nav/sim/controllers.py", "nav/sim/collision_guard.py")}
    eps_all = episodes_by_env(config_path=args.episodes_file)[args.env]
    groups: dict = {}
    order = []
    for e in eps_all:
        key = (round(e.start[0], 3), round(e.start[1], 3), round(e.start[2], 3), round(e.start_yaw_deg, 2))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)
    poses = [groups[k] for k in order]
    if args.only:
        want = set(args.only.split(","))
        poses = [g for g in poses if any(e.name in want for e in g)]
    if args.video_episode:
        poses.sort(key=lambda g: any(e.name == args.video_episode for e in g))     # video pose last
    routes = load_routes()
    report["routes_on_disk"] = len(routes)

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
    aloha = stage.GetPrimAtPath("/World/Aloha")
    if aloha.IsValid():
        aloha.SetActive(False)
    if not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]:
        UsdLux.DomeLight.Define(stage, "/World/G1Dome").CreateIntensityAttr(1000.0)

    loco = L.G1Locomotion("/World/G1")
    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True)
    from physics_setup import raise_gpu_capacities

    report["gpu_capacity_changes"] = raise_gpu_capacities(stage)
    e0 = poses[0][0]
    report["author"] = loco.author(stage, (e0.start[0], e0.start[1], e0.start[2] + L.SPAWN_HEIGHT_M), e0.start_yaw_rad)
    nav_spec = loco.asset_spec["nav_camera"]        # the G1's own D435 mount re-pitched to 12 deg (asset_spec.json)
    video = bool(args.video_episode)
    if video:
        nav_path = L.author_head_camera(stage, loco.prim_path, nav_spec, "g1_head_cam")
        L.author_head_light(stage, loco.prim_path, nav_spec)
        chase = UsdGeom.Camera.Define(stage, "/World/G1Chase")
        chase.CreateFocalLengthAttr(16.0)
        chase.CreateHorizontalApertureAttr(20.955)
        chase.CreateVerticalApertureAttr(20.955 * 9 / 16)
        chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        (OUT / "frames" / "nav").mkdir(parents=True, exist_ok=True)
        (OUT / "frames" / "chase").mkdir(parents=True, exist_ok=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["init"] = loco.initialize()
    base = L.G1Base(loco)                       # callback mode, shipping defaults
    gd = G.G1Guard()
    report["guard"] = {"mode": args.guard, "rings": gd.ring_heights_m, "stop": gd.stop_distance_m, "slow": gd.slow_distance_m,
                       "step_down_max_m": gd.step_down_max_m}
    for _ in range(10):
        kit.update()

    cams: dict = {}

    def cam(name):
        if name not in cams:
            from isaacsim.sensors.camera import Camera

            path, res = {"head": (nav_path, nav_spec["resolution_wh"]), "chase": ("/World/G1Chase", (1280, 720))}[name]
            c = Camera(prim_path=path, resolution=tuple(res))
            c.initialize()
            cams[name] = c
        return cams[name]

    chase_yaw = {"v": 0.0}

    def aim_chase():
        x, y, _, yaw_ = loco.pose()
        chase_yaw["v"] += 0.15 * wrap(yaw_ - chase_yaw["v"])
        cy = chase_yaw["v"]
        fz = base.position()[2]
        eye = (x - 2.6 * math.cos(cy), y - 2.6 * math.sin(cy), fz + 1.6)
        mtx = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(x, y, fz + 0.85), Gf.Vec3d(0, 0, 1)).GetInverse()
        qq = mtx.ExtractRotationQuat()
        im = qq.GetImaginary()
        cam("chase").set_world_pose(position=np.array(eye, dtype=float),
                                    orientation=np.array([qq.GetReal(), im[0], im[1], im[2]]), camera_axes="usd")

    def lum(a):
        a = a.astype(np.float32)
        return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())

    def grab(c):
        rgba = c.get_rgba()
        return None if rgba is None or np.asarray(rgba).size == 0 else np.asarray(rgba)[:, :, :3].astype(np.uint8)

    vstate = {"on": False, "i": 0, "k": 0, "rows": [], "seg": ""}
    every = max(1, int(round(1.0 / (args.fps * CTL))))

    def video_tick(cmd):
        if not vstate["on"]:
            return
        vstate["k"] += 1
        if vstate["k"] % 3 == 0:
            aim_chase()
        if vstate["k"] % every:
            return
        from PIL import Image

        i = vstate["i"]
        fn, fc = grab(cam("head")), grab(cam("chase"))
        if fn is not None:
            Image.fromarray(fn).save(OUT / "frames" / "nav" / f"{i:05d}.jpg", quality=92)
        if fc is not None:
            Image.fromarray(fc).save(OUT / "frames" / "chase" / f"{i:05d}.jpg", quality=90)
        st = loco.state()
        cp = loco.camera_pose(nav_spec)
        vstate["rows"].append({"i": i, "t": round(base.sim_time() - vstate.get("t0", 0.0), 2), "seg": vstate["seg"], "cmd": [round(v, 3) for v in cmd],
                               "pos": [round(float(v), 3) for v in st["pos"]], "v_b": [round(float(v), 3) for v in st["lin_b"]],
                               "tilt": round(st["tilt_deg"], 2), "lum_nav": None if fn is None else round(lum(fn), 2),
                               "menu_args": {k_: round(v, 4) for k_, v in cp["menu_args"].items()}})
        vstate["i"] += 1

    timing = {"upd_s": 0.0, "n_upd": 0}

    def step(cmd):
        base.apply(*cmd, CTL)
        a = time.perf_counter()
        kit.update()
        timing["upd_s"] += time.perf_counter() - a
        timing["n_upd"] += 1
        ps = base.physics_state()
        video_tick(cmd)
        return ps

    for grp in poses:
        ep = grp[0]
        names = [e.name for e in grp]
        r: dict = {"episode": ep.name, "pose_group": names, "start": list(ep.start), "yaw_deg": ep.start_yaw_deg,
                   "goal": list(ep.goal)}
        try:
            src = next((n for n in names if n in routes), None)
            if src is not None:
                rfile, rt = routes[src]
                pts = [p[:2] for p in rt["points"]]
                # some navmesh routes start at the closest navmesh point, not at the start (office_e36/e37: 1.9 m
                # off); begin the route AT the robot so cross-track and arc length mean what they say
                connector = math.dist(pts[0], ep.start[:2])
                if connector > 0.3:
                    pts = [list(ep.start[:2])] + pts
                route = Route(pts)
                r["route_source"] = {"kind": "navmesh (route_drive)", "episode": src, "file": rfile,
                                     "route_length_m": round(route.length, 2), "route_rule": rt.get("rule"),
                                     "start_connector_m": round(connector, 3) if connector > 0.3 else 0.0}
            else:
                route = Route([ep.start[:2], ep.goal[:2]])
                r["route_source"] = {"kind": "straight_to_goal (no navmesh route on disk)", "route_length_m": round(route.length, 2)}
            s_end = min(args.route_m, route.length)
            r["route_segment_m"] = round(s_end, 2)
            r["route_segment_heading_change_deg"] = round(route.heading_change_deg(s_end), 1)
            is_video = video and args.video_episode in names
            if is_video:
                chase_yaw["v"] = ep.start_yaw_rad
                cam("head")
                cam("chase")
            base.reset_to(tuple(ep.start), ep.start_yaw_rad)
            base.reset_stats()
            gd.reset_telemetry()
            r["spawn"] = dict(base.spawn)
            for _ in range(3):
                step((0.0, 0.0, 0.0))
            vstate.update(on=is_video, seg="stand", t0=base.sim_time())
            twall0, tsim0, tpoll0 = time.perf_counter(), base.sim_time(), poll.now()
            upd0 = dict(timing)
            # ---- stand
            for _ in range(int(round(args.stand_s / CTL))):
                ps = step((0.0, 0.0, 0.0))
                if ps["terminal"]:
                    break
            r["stand"] = {"hold_max_from_latch_m": base.stats()["hold_last_window_max_m"], "terminal": ps["terminal"]}
            # ---- route
            vstate["seg"] = "route"
            ctrl = make_controller("pursuit", args.speed, args.max_yaw_rate, lookahead_m=1.0)
            ctrl.reset()
            s_cur, xt_max, path, pivots, pivoting = 0.0, 0.0, 0.0, 0, False
            x_prev, y_prev, _ = base.position()
            t_route0 = base.sim_time()
            timeout = 2.5 * s_end / args.speed + 15.0
            end_reason = "timeout"
            guard_blocked_steps = 0
            n = 0
            while not ps["terminal"]:
                x, y, _ = base.position()
                yaw = base.yaw
                s, xt = route.project((x, y), s_cur - 0.5, s_cur + 3.0)
                s_cur = max(s_cur, s)
                xt_max = max(xt_max, xt)
                if s_cur >= s_end - 0.3:
                    end_reason = "reached"
                    break
                if base.sim_time() - t_route0 > timeout:
                    break
                la = route.points_at(min(s_cur + 1.0, s_end))
                bearing = wrap(math.atan2(la[1] - y, la[0] - x) - yaw)
                if not pivoting and abs(bearing) > math.radians(45.0):
                    pivoting, pivots = True, pivots + 1
                if pivoting and abs(bearing) < math.radians(15.0):
                    pivoting = False
                    ctrl.reset()
                if pivoting:
                    cmd = (0.0, 0.0, math.copysign(args.max_yaw_rate, bearing))
                else:
                    ss = s_cur + np.arange(0.05, 3.0 + 1e-9, 0.05)
                    pts = route.points_at(np.minimum(ss, s_end)) - np.array([x, y])
                    c, sn = math.cos(yaw), math.sin(yaw)
                    wp = np.stack([c * pts[:, 0] + sn * pts[:, 1], -sn * pts[:, 0] + c * pts[:, 1]], axis=1)
                    cc = ctrl(wp)
                    cmd = (cc.vx, cc.vy, cc.omega)
                if args.guard != "off" and (abs(cmd[0]) > 1e-6 or abs(cmd[1]) > 1e-6):
                    g = gd.check(base.position(), yaw, cmd[0], cmd[1], count=(args.guard == "apply"))
                    if args.guard == "apply":
                        if math.hypot(g.vx, g.vy) < 1e-6 < math.hypot(cmd[0], cmd[1]):
                            guard_blocked_steps += 1
                        cmd = (g.vx, g.vy, cmd[2])
                ps = step(cmd)
                n += 1
                x2, y2, _ = base.position()
                path += math.hypot(x2 - x_prev, y2 - y_prev)
                x_prev, y_prev = x2, y2
            t_route = base.sim_time() - t_route0
            if ps["terminal"]:
                end_reason = "terminal"
            st_route = base.stats()
            r["route"] = {"end": end_reason, "s_reached_m": round(s_cur, 2), "of_m": round(s_end, 2),
                          "reached": end_reason == "reached", "sim_s": round(t_route, 2), "path_m": round(path, 2),
                          "mean_speed_mps": round(path / max(t_route, 1e-6), 3), "cross_track_max_m": round(xt_max, 3),
                          "pivots": pivots, "guard_interventions": gd.interventions,
                          "guard_hits_by_source": dict(gd.hits_by_source), "guard_blocked_steps": guard_blocked_steps,
                          "tilt_max_deg": st_route["tilt_max_deg"], "vz_min_mps": st_route["vz_min_mps"],
                          "torso_contact_steps": st_route["torso_contact_steps"], "fell": st_route["fell"],
                          "free_fall": st_route["free_fall"], "faults": [f["kind"] for f in st_route["fault_log"]]}
            # ---- stop (hold)
            vstate["seg"] = "stop_hold"
            xs = []
            if not ps["terminal"]:
                x_s, y_s, _ = base.position()
                for _ in range(int(round(args.hold_s / CTL))):
                    ps = step((0.0, 0.0, 0.0))
                    xs.append(base.position()[:2])
                    if ps["terminal"]:
                        break
                st_h = base.stats()
                A = np.array(xs)
                n2 = int(round(2.0 / CTL))
                r["stop_hold"] = {"hold_last_window_max_m": st_h["hold_last_window_max_m"],
                                  "max_from_stop_cmd_pose_m": round(float(np.hypot(A[:, 0] - x_s, A[:, 1] - y_s).max()), 4),
                                  "last2s_displacement_m": round(float(np.hypot(*(A[-1] - A[-n2]))), 4) if len(A) >= n2 else None}
            # ---- reverse request (must hold)
            vstate["seg"] = "reverse_request"
            if not ps["terminal"]:
                x_r, y_r, _ = base.position()
                held0 = base.reverse_requests_held
                xs = []
                for _ in range(int(round(args.reverse_s / CTL))):
                    ps = step((-0.3, 0.0, 0.0))
                    xs.append(base.position()[:2])
                    if ps["terminal"]:
                        break
                A = np.array(xs)
                st_r = base.stats()
                r["reverse_request"] = {"requests_held": st_r["reverse_requests_held"] - held0, "updates": len(xs),
                                        "max_from_request_pose_m": round(float(np.hypot(A[:, 0] - x_r, A[:, 1] - y_r).max()), 4),
                                        "hold_last_window_max_m": st_r["hold_last_window_max_m"]}
            vstate["on"] = False
            st = base.stats()
            sim_s = base.sim_time() - tsim0
            wall = time.perf_counter() - twall0
            n_upd = timing["n_upd"] - upd0["n_upd"]
            r["g1_stats"] = st
            r["fell"] = st["fell"]
            r["terminal"] = bool(ps["terminal"])
            r["cost"] = {"sim_s": round(sim_s, 2), "wall_s": round(wall, 2), "wall_per_sim_s": round(wall / max(sim_s, 1e-6), 3),
                         "ms_per_update": round(1000 * (timing["upd_s"] - upd0["upd_s"]) / max(n_upd, 1), 2),
                         "physx_step_ms_mean": st["physx_step_ms_mean"], "gpu_load": poll.window(tpoll0, poll.now()),
                         "cameras_live": bool(cams)}
            if is_video:
                report["video_rows"] = vstate["rows"]
                report["video_episode"] = ep.name
                report["video_light_intensity"] = L.G1_HEADLIGHT_INTENSITY
        except Exception:  # noqa: BLE001
            import traceback

            r["error"] = traceback.format_exc()
            print(r["error"], flush=True)
        report["results"].append(r)
        rr = r.get("route", {})
        print(f"[route-survey] {ep.name}: end={rr.get('end')} s={rr.get('s_reached_m')}/{rr.get('of_m')} fell={r.get('fell')} "
              f"tilt={rr.get('tilt_max_deg')} guard={rr.get('guard_interventions')} torso={rr.get('torso_contact_steps')} "
              f"hold={r.get('stop_hold', {}).get('hold_last_window_max_m')} rev={r.get('reverse_request', {}).get('max_from_request_pose_m')} "
              f"cost={r.get('cost', {}).get('wall_per_sim_s')}", flush=True)
        save()

    res = report["results"]
    ok = [x for x in res if "error" not in x]
    report["summary"] = {
        "poses": len(res), "errors": len(res) - len(ok),
        "reached": sum(1 for x in ok if x.get("route", {}).get("reached")),
        "falls": sum(1 for x in ok if x.get("fell")),
        "free_falls": sum(1 for x in ok if x.get("g1_stats", {}).get("free_fall")),
        "no_floor": sum(1 for x in ok if x.get("g1_stats", {}).get("no_floor")),
        "torso_contact_poses": sum(1 for x in ok if x.get("route", {}).get("torso_contact_steps", 0) > 0),
        "tilt_max_deg": max((x["route"]["tilt_max_deg"] for x in ok if "route" in x), default=None),
        "hold_last_window_max_m": max((x["stop_hold"]["hold_last_window_max_m"] for x in ok if "stop_hold" in x), default=None),
        "reverse_max_from_request_pose_m": max((x["reverse_request"]["max_from_request_pose_m"] for x in ok
                                                if "reverse_request" in x), default=None),
        "route_m_walked": round(sum(x.get("route", {}).get("path_m", 0.0) for x in ok), 1),
        "wall_per_sim_s_median": (float(np.median([x["cost"]["wall_per_sim_s"] for x in ok if "cost" in x and not x["cost"]["cameras_live"]]))
                                  if any("cost" in x and not x["cost"]["cameras_live"] for x in ok) else None),
    }
    report["gpu_whole_run"] = poll.window(0, poll.now())
    report["stage_file"]["root_layer_dirty"] = bool(stage.GetRootLayer().dirty)
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
    try:
        sf = report.get("stage_file", {})
        if sf.get("path"):
            sf["md5_after"] = hashlib.md5(Path(sf["path"]).read_bytes()).hexdigest()
            sf["unchanged"] = sf["md5_after"] == sf["md5_before"]
    except Exception as e:  # noqa: BLE001
        report.setdefault("stage_file", {})["md5_after"] = repr(e)
    report["wall_total_s"] = round(time.time() - T0, 1)
    save()
    print(f"[route-survey] status={report.get('status')} summary={report.get('summary')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
