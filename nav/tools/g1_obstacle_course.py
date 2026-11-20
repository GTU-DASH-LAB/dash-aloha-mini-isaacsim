"""Obstacle course for the walking G1: what it trips on, and whether G1Guard stops it in time.

Flat ground (cube, no material -> the robot's own 0.8/0.6 'max' material decides friction), lanes
4 m apart, the robot spawned 1.5 m in front of each obstacle face, facing it, G1Base in CALLBACK mode
with one kit.update() per 1/60 s (the runner's way, slew limit and hold on as shipped).

Lanes: open | box_<h> for h in BOX_HEIGHTS (0.40 m deep, 1.6 m wide, vertical face) | chest_slab
(z 0.70-1.30: only the torso box can touch it) | head_bar (z 1.05-1.40: NOTHING on the robot can touch
it -- the head has no collider -- so without the guard the camera drives through it) | wall (0-2 m) |
ramp (6.8 deg, the outdoor pillars slope: the guard must NOT stop here).

Passes:
  noguard  constant vx command straight at the obstacle, 7 s. Trip sweep: which heights fell, when,
           and the torso-contact counter next to it (blind below 0.65 m by construction).
  guard    the same command through nav/sim/g1/guard.G1Guard (rings G1_RING_HEIGHTS_M, stop/slow
           distances from guard.py) -> fell?, contact?, how close the FEET got to the face
           (foot collider boxes from the USD, posed by PhysX every update), when it stopped, which
           ring fired, and on the ramp / open lane: did it keep walking.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/g1_obstacle_course.py --out <dir>

report.json in <out>; the exit code is not the verdict (kit.close() ends the process).
"""

from __future__ import annotations

import argparse
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
ap.add_argument("--out", required=True)
ap.add_argument("--box-heights", default="0.03,0.05,0.08,0.12,0.20,0.40")
ap.add_argument("--seconds", type=float, default=7.0)
ap.add_argument("--passes", default="noguard@0.5,noguard@0.8,guard@0.8,guard@1.0")
ap.add_argument("--stop", type=float, default=None, help="override guard.G1_STOP_DISTANCE_M")
ap.add_argument("--slow", type=float, default=None, help="override guard.G1_SLOW_DISTANCE_M")
ap.add_argument("--step-down", type=float, default=None, help="override guard.G1_STEP_DOWN_MAX_M")
ap.add_argument("--only-lanes", default="", help="comma list of lane names to run (default: all); all lanes are "
                                                "still authored so lane positions do not move")
ap.add_argument("--extra-lanes", default="sill_0.01,kerb_down_0.13,pole",
                help="also: sill_0.01 (1 cm door sill, must NOT stop the robot), kerb_down_0.13 (a 13 cm drop), "
                     "pole (5 cm thick vertical pole)")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
BOX_HEIGHTS = [float(h) for h in args.box_heights.split(",") if h]
FACE_AHEAD = 1.5
LANE_DY = 4.0

from isaacsim import SimulationApp  # noqa: E402

T0 = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"pid": os.getpid(), "args": vars(args), "runs": []}


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
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    import guard as G
    import locomotion as L
    from isaacsim.core.simulation_manager import SimulationManager

    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        kit.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    def box(path, center, size, pitch_deg=0.0):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        if pitch_deg:
            xf.AddRotateYOp().Set(float(pitch_deg))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        return path

    box("/World/Ground", (0, 0, -0.5), (400, 400, 1))
    lanes = [("open", None)] + [(f"box_{h:.2f}", ("box", h)) for h in BOX_HEIGHTS] + \
            [("chest_slab", ("slab", 0.70, 1.30)), ("head_bar", ("slab", 1.05, 1.40)), ("wall", ("slab", 0.0, 2.0)),
             ("ramp", ("ramp", 6.8))]
    extra = [e for e in args.extra_lanes.split(",") if e]
    if "sill_0.01" in extra:
        lanes.append(("sill_0.01", ("box", 0.01)))
    for e in extra:
        if e.startswith("kerb_down_"):          # kerb_down_<h>: start on a platform h high, drop at the face
            lanes.append((e, ("drop", float(e.split("_")[-1]))))
    if "pole" in extra:
        lanes.append(("pole", ("pole", 0.05)))
    if "ramp_wide" in extra:
        # the 1.6 m ramp is a raised wedge whose open SIDES drop 0.1-0.9 m: the step probe's outer rays see that
        # drop at ~1.2 m and the guard slows down (correctly, for that geometry). 3.6 m wide = terrain-like.
        lanes.append(("ramp_wide", ("ramp", 6.8, 3.6)))
    lane_y = {}
    for i, (name, obs) in enumerate(lanes):
        y = (i - len(lanes) / 2.0) * LANE_DY
        lane_y[name] = y
        fx = FACE_AHEAD
        if obs is None:
            continue
        if obs[0] == "box":
            h = obs[1]
            box(f"/World/Obs_{i}", (fx + 0.2, y, h / 2.0), (0.4, 1.6, h))
        elif obs[0] == "slab":
            z0, z1 = obs[1], obs[2]
            box(f"/World/Obs_{i}", (fx + 0.1, y, (z0 + z1) / 2.0), (0.2, 1.6, z1 - z0))
        elif obs[0] == "drop":
            # a raised platform the robot starts on, ending in a vertical drop at the face
            hh = obs[1]
            box(f"/World/Obs_{i}", (fx - 3.0, y, hh / 2.0), (6.0, 1.6, hh))
        elif obs[0] == "pole":
            box(f"/World/Obs_{i}", (fx + obs[1] / 2.0, y, 1.0), (obs[1], obs[1], 2.0))
        elif obs[0] == "ramp":
            ang = math.radians(obs[1])
            length, thick = 8.0, 0.4
            width = obs[2] if len(obs) > 2 else 1.6
            # top surface passes through (fx, 0) and rises toward +x: centre = face + half length along the slope,
            # minus half the thickness along the slope normal
            cx = fx + 0.5 * length * math.cos(ang) + 0.5 * thick * math.sin(ang)
            cz = 0.5 * length * math.sin(ang) - 0.5 * thick * math.cos(ang)
            box(f"/World/Obs_{i}", (cx, y, cz), (length, width, thick), pitch_deg=-obs[1])
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(500.0)
    report["lanes"] = {n: {"y": lane_y[n], "obstacle": o} for n, o in lanes}

    loco = L.G1Locomotion("/World/G1")
    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True)
    report["author"] = loco.author(stage, (0.0, lane_y["open"], L.SPAWN_HEIGHT_M), 0.0)
    for _ in range(60):
        kit.update()
    stage.Load()

    # foot collider corners in their LINK frame (USD, before play), posed later by PhysX link transforms
    def corners_in_link(link):
        cube = stage.GetPrimAtPath(f"/World/G1/{link}/Cube")
        lp = stage.GetPrimAtPath(f"/World/G1/{link}")
        mc = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ml = UsdGeom.Xformable(lp).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel = mc * ml.GetInverse()
        if cube.IsA(UsdGeom.Mesh):           # g1_minimal's "Cube" colliders are Meshes, not UsdGeom.Cube
            pts = [Gf.Vec3d(*map(float, q)) for q in UsdGeom.Mesh(cube).GetPointsAttr().Get()]
        else:
            half = float(UsdGeom.Cube(cube).GetSizeAttr().Get()) / 2.0
            pts = [Gf.Vec3d(sx * half, sy * half, sz * half) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return np.array([[*rel.Transform(q)] for q in pts])
    foot_corners = {f: corners_in_link(f) for f in L.FOOT_LINKS}
    report["foot_box_corners_link_frame"] = {f: c.round(4).tolist() for f, c in foot_corners.items()}

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["init"] = loco.initialize()
    base = L.G1Base(loco)
    foot_idx = [loco.body_names.index(f) for f in L.FOOT_LINKS]
    CTL = 1.0 / 60.0

    def foot_front_x() -> float:
        lt = loco.link_poses()
        xs = []
        for f, i in zip(L.FOOT_LINKS, foot_idx):
            p = lt[i, :3]
            qx, qy, qz, qw = lt[i, 3:]
            R = L.quat_to_matrix_wxyz((qw, qx, qy, qz))
            xs.append(float((p + foot_corners[f] @ R.T)[:, 0].max()))
        return max(xs)

    gd = G.G1Guard(stop_distance_m=args.stop if args.stop is not None else G.G1_STOP_DISTANCE_M,
                   slow_distance_m=args.slow if args.slow is not None else G.G1_SLOW_DISTANCE_M,
                   step_down_max_m=args.step_down if args.step_down is not None else G.G1_STEP_DOWN_MAX_M)
    report["guard"] = {"rings": gd.ring_heights_m, "step_probe": gd.step_probe, "step_max_m": gd.step_max_m, "step_down_max_m": gd.step_down_max_m,
                       "step_spacing_m": gd.step_spacing_m, "stop": gd.stop_distance_m, "slow": gd.slow_distance_m,
                       "fan_rays": gd.fan_rays, "chassis_radius": gd.chassis_radius_m}

    for pspec in [p for p in args.passes.split(",") if p]:
        mode, v = pspec.split("@")
        v = float(v)
        only = [x for x in args.only_lanes.split(",") if x]
        for name, obs in lanes:
            if only and name not in only:
                continue
            if mode == "noguard" and v > 0.6 and obs is not None and obs[0] == "box" and obs[1] > 0.13:
                continue            # higher boxes at 0.8: the 0.5 pass already shows them
            y = lane_y[name]
            z_start = obs[1] if (obs is not None and obs[0] == "drop") else 0.0
            base.reset_to((0.0, y, z_start), 0.0)
            base.reset_stats()
            gd.reset_telemetry()
            guard_s = {"s": 0.0, "n": 0}
            for _ in range(int(round(1.0 / CTL))):              # stand 1 s
                base.apply(0.0, 0.0, 0.0, CTL)
                kit.update()
            t, fell, stop_t, rows = 0.0, None, None, []
            min_clear, max_px, max_dy = float("inf"), -1e9, 0.0
            z0 = loco.pose()[2]
            twall = time.perf_counter()
            for k in range(int(round(args.seconds / CTL))):
                vx, vy = v, 0.0
                if mode == "guard":
                    a_ = time.perf_counter()
                    r = gd.check(base.position(), base.yaw, v, 0.0)
                    guard_s["s"] += time.perf_counter() - a_
                    guard_s["n"] += 1
                    vx, vy = r.vx, r.vy
                base.apply(vx, vy, 0.0, CTL)
                kit.update()
                t += CTL
                ps = base.physics_state()
                px, py, pz, _ = loco.pose()
                ff = foot_front_x()
                max_px = max(max_px, px)
                max_dy = max(max_dy, abs(py - y))
                min_clear = min(min_clear, FACE_AHEAD - ff)
                spd = float(np.linalg.norm(loco.state()["lin_w"][:2]))
                if mode == "guard" and stop_t is None and t > 1.5 and spd < 0.05:
                    stop_t = round(t, 2)
                if k % 6 == 0:
                    rows.append([round(t, 2), round(px, 3), round(pz, 3), round(ff, 3), round(vx, 3), round(spd, 3),
                                 round(ps["tilt_deg"], 1), round(vy, 3), round(py - y, 3)])
                if ps["fallen"] and fell is None:
                    fell = {"t": round(t, 2), "why": (base.last_fault or {}).get("why")}
                    break
            st = base.stats()
            rec = {"pass": pspec, "lane": name, "obstacle": obs, "fell": fell,
                   "torso_contact_steps": st["torso_contact_steps"], "torso_contact_max_n": st["torso_contact_max_n"],
                   "max_pelvis_x": round(max_px, 3), "end_pelvis_x": round(loco.pose()[0], 3) if loco.handle_ok() else None,
                   "pelvis_dz_end": round(loco.pose()[2] - z0, 3),
                   "max_abs_lateral_drift_m": round(max_dy, 3),
                   "min_face_minus_foot_front_m": round(min_clear, 3) if obs is not None else None,
                   "stop_t": stop_t, "tilt_max": st["tilt_max_deg"], "slew_limited_ticks": st["slew_limited_ticks"],
                   "wall_per_sim_s": round((time.perf_counter() - twall) / max(t, 1e-6), 3), "trace_every_0.1s": rows,
                   "trace_columns": ["t", "pelvis_x", "pelvis_z", "foot_front_x", "cmd_vx", "speed", "tilt_deg", "cmd_vy",
                                     "pelvis_y_minus_lane"]}
            if mode == "guard":
                rec.update({"guard_interventions": gd.interventions, "guard_hits_by_source": dict(gd.hits_by_source),
                            "guard_floor_hits_skipped": gd.floor_hits_skipped,
                            "guard_self_hits_skipped": getattr(gd, "self_hits_skipped", None),
                            "guard_blocked_checks": getattr(gd, "blocked_checks", None),
                            "guard_blocked_zero_translation": getattr(gd, "blocked_zero_translation", None),
                            "guard_ms_per_check": round(1000 * guard_s["s"] / max(guard_s["n"], 1), 3),
                            "guard_rays_per_check": round(gd.rays_cast / max(guard_s["n"], 1), 1)})
            report["runs"].append(rec)
            print(f"[course] {pspec:12s} {name:11s} fell={fell} torso={rec['torso_contact_steps']} maxx={rec['max_pelvis_x']} "
                  f"clear={rec['min_face_minus_foot_front_m']} stop_t={stop_t} dz={rec['pelvis_dz_end']} dy={rec['max_abs_lateral_drift_m']} "
                  f"hits={rec.get('guard_hits_by_source')} ms={rec.get('guard_ms_per_check')}",
                  flush=True)
            save()
    base.close()
    report["gpu_whole_run"] = poll.window(0, poll.now())
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
    print(f"[course] status={report.get('status')}", flush=True)
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    os._exit(rc)
