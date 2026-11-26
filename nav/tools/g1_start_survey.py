"""Does the walking G1 fit and stand at every DynaNav start, and what does PhysX cost there?

A SPAWN/STAND survey, not a walking one: 2 s stand + 4 s straight at vx 0.5 (~1.4 m) per pose. Its
round-1 result ("27/27 poses stand and walk") means every pose has a floor and the robot stands up and
starts walking -- nothing about turning or routes (g1/check round 2). The pre-integration WALKING survey
is nav/tools/g1_route_survey.py (first 10 m of each navmesh route, pursuit + G1Guard, current code).

One nav stage per process (the stage file is opened IN MEMORY, every edit in the session layer, the
file's md5 verified unchanged, /World/Aloha deactivated). For each episode of that environment:

  * G1Base.reset_to(episode start, yaw): floor raycast snap (g1_spawn_pose), what the snap found;
  * clearances from the start: headroom (ray up from 0.30 m above the floor), and forward / left /
    right / back at 1.0 m (chest) and 0.30 m (the collision guard's ray height);
  * stand 2 s, then walk straight (vx 0.5, wz 0) for 4 s, the runner's way (one kit.update per 1/60 s,
    G1Base in CALLBACK mode): fell or not, pelvis height above the feet, tilt, distance covered, and
    whether the torso box touched anything (G1Base.stats(): counted on every physics sub-step from a
    PHYSICS_POST_STEP callback -- before 2026-10-05 repair round 1 this counter could not increment in
    callback mode, so earlier "torso contacts 0" results carry no information);
  * wall time per kit.update() and per physics sub-step on this stage, GPU memory of this process, and
    the GPU0 utilisation / compute-process count / load average over the same window.

--episodes-file nav/config/episodes_full.yaml --distinct-poses surveys every distinct (start, yaw)
of the 85-episode benchmark (27 poses: hospital 6, office 6, outdoor_small 8, warehouse 7) instead
of the 19-episode ladder file.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/g1_start_survey.py \
        --env hospital --out <dir>

The report is <out>/report.json (the process exit code is NOT the verdict: kit.close() ends the
process). Nothing here saves a stage.
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
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
sys.path.insert(0, str(REPO / "nav" / "sim" / "g1"))
sys.path.insert(1, str(REPO / "nav" / "sim"))

CTL_DT = 1.0 / 60.0
ap = argparse.ArgumentParser()
ap.add_argument("--env", choices=["hospital", "office", "warehouse", "outdoor_small"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stand-s", type=float, default=2.0)
ap.add_argument("--walk-s", type=float, default=4.0)
ap.add_argument("--episodes", default="", help="comma list; default every episode of the env")
ap.add_argument("--episodes-file", default="", help="episode config (default nav/config/episodes.yaml)")
ap.add_argument("--distinct-poses", action="store_true", help="one survey per distinct (start, yaw)")
ap.add_argument("--broadphase", default="MBP", help="CPU broadphase for the physics scene (MBP = what the walk was validated with; ABP/SAP to compare cost)")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T_LAUNCH = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
report: dict = {"env": args.env, "args": vars(args), "pid": os.getpid()}


def save_report():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


class GpuPoll(threading.Thread):
    """THIS pid's memory on GPU0/GPU1 plus GPU0 utilisation, compute-process count and load average,
    once a second: a cost figure is reported next to the card's load over the same window."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag, self.pid = period, [], False, os.getpid()

    def read(self) -> dict:
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
                if int(pid) == self.pid:
                    mine[idx.get(uuid)] = int(mib)
                if idx.get(uuid) == 0:
                    n0 += 1
            return {"t": round(time.time() - T_LAUNCH, 1), "mine_gpu0": mine.get(0, 0), "mine_gpu1": mine.get(1, 0),
                    "gpu0_util": util[0][0], "gpu0_used": util[0][1], "procs_gpu0": n0, "load1": round(os.getloadavg()[0], 1)}
        except Exception as e:  # noqa: BLE001
            return {"t": round(time.time() - T_LAUNCH, 1), "error": repr(e)}

    def run(self):
        while not self.stop_flag:
            self.samples.append(self.read())
            time.sleep(self.period)

    def window(self, t0: float = 0.0, t1: float = 1e12) -> dict:
        s_ = [x for x in self.samples if "gpu0_util" in x and t0 <= x["t"] <= t1]
        if not s_:
            return {"n": 0}
        return {"n": len(s_), "gpu0_util_mean": round(sum(x["gpu0_util"] for x in s_) / len(s_), 1),
                "procs_gpu0_min_max": [min(x["procs_gpu0"] for x in s_), max(x["procs_gpu0"] for x in s_)],
                "gpu0_card_used_mib_max": max(x["gpu0_used"] for x in s_),
                "mine_gpu0_max_mib": max(x["mine_gpu0"] for x in s_), "mine_gpu1_max_mib": max(x["mine_gpu1"] for x in s_),
                "load1_mean": round(sum(x["load1"] for x in s_) / len(s_), 1)}

    def summary(self) -> dict:
        return self.window()


gpu = GpuPoll(1.0)
gpu.start()
rc = 1
try:
    import carb
    import numpy as np
    import omni.timeline
    import omni.usd
    from omni.physx import get_physx_scene_query_interface
    from pxr import Usd, UsdLux

    import locomotion as L
    from episode import episodes_by_env

    eps = (episodes_by_env(config_path=args.episodes_file) if args.episodes_file else episodes_by_env())[args.env]
    if args.episodes:
        want = set(args.episodes.split(","))
        eps = [e for e in eps if e.name in want]
    if args.distinct_poses:
        seen, uniq, shared = set(), [], {}
        for e in eps:
            key = (round(e.start[0], 3), round(e.start[1], 3), round(e.start[2], 3), round(e.start_yaw_deg, 2))
            shared.setdefault(key, []).append(e.name)
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        report["pose_groups"] = {shared[k][0]: shared[k] for k in shared}
        eps = uniq
    report["episodes"] = [e.name for e in eps]

    ctx = omni.usd.get_context()
    loco = L.G1Locomotion("/World/G1")
    scene_file = (REPO / "assets" / "usd" / f"nav_{args.env}.usda").resolve()
    md5 = hashlib.md5(scene_file.read_bytes()).hexdigest()
    report["stage_file"] = {"path": str(scene_file), "md5_before": md5}
    t0 = time.time()
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
    report["stage_load_s"] = round(time.time() - t0, 1)
    if not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]:
        UsdLux.DomeLight.Define(stage, "/World/G1Dome").CreateIntensityAttr(1000.0)

    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True, broadphase=args.broadphase)
    from physics_setup import raise_gpu_capacities

    report["gpu_capacity_changes"] = raise_gpu_capacities(stage)
    e0 = eps[0]
    report["author"] = loco.author(stage, (e0.start[0], e0.start[1], e0.start[2] + L.SPAWN_HEIGHT_M), e0.start_yaw_rad)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    kit.update()
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["init"] = loco.initialize()
    base = L.G1Base(loco)
    for _ in range(10):
        kit.update()
    report["gpu_after_setup"] = gpu.read()

    q = get_physx_scene_query_interface()

    def collider_extents():
        """The G1's collision shapes (torso box, two foot boxes) as 8-corner extents in the PELVIS
        frame (x forward, y left, z up), from the USD at the reset pose. Everything else on the
        robot -- legs, arms, pelvis, head -- has no collider and passes through anything."""
        from pxr import Gf, UsdGeom

        out = {}
        pelvis = stage.GetPrimAtPath(loco.root_path)
        pm = UsdGeom.Xformable(pelvis).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inv = pm.GetInverse()
        for path in report["author"]["colliders"]:
            prim = stage.GetPrimAtPath(path)
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2.0
                local = [Gf.Vec3d(sx * half, sy * half, sz * half) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            elif prim.IsA(UsdGeom.Mesh):
                local = [Gf.Vec3d(*map(float, p_)) for p_ in (UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])]
            else:
                out[path.replace("/World/G1/", "")] = f"collider type {prim.GetTypeName()} not handled"
                continue
            pts = [inv.Transform(m.Transform(v_)) for v_ in local]
            a = np.array([[p[0], p[1], p[2]] for p in pts])
            out[path.replace("/World/G1/", "")] = {"min_xyz": a.min(0).round(3).tolist(), "max_xyz": a.max(0).round(3).tolist()}
        return out

    base.reset_to(tuple(eps[0].start), eps[0].start_yaw_rad)
    for _ in range(3):
        kit.update()
    try:
        report["collider_extents_pelvis_frame_m"] = collider_extents()
    except Exception as e:  # noqa: BLE001
        report["collider_extents_pelvis_frame_m"] = repr(e)
    save_report()

    def ray(origin, direction, dist, skip="/World/G1"):
        o = np.array(origin, dtype=float)
        d = np.array(direction, dtype=float)
        for _ in range(30):
            hit = q.raycast_closest(carb.Float3(*o.tolist()), carb.Float3(*d.tolist()), dist)
            if not hit or not hit.get("hit"):
                return None, ""
            prim = str(hit.get("collision", ""))
            hp = np.array([float(hit["position"][i]) for i in range(3)])
            if not prim.startswith(skip):
                return float(np.linalg.norm(hp - np.array(origin, dtype=float))), prim
            o = (o + 0.03 * d) if float(hit.get("distance", 0.0)) < 1e-4 else (hp + 0.01 * d)
        return None, ""

    def clearances(x, y, yaw, floor_z):
        out = {}
        d, p = ray((x, y, floor_z + 0.30), (0, 0, 1), 6.0)
        out["headroom_m"] = None if d is None else round(d + 0.30, 3)
        out["headroom_prim"] = p.split("/")[-1] if p else ""
        for h in (1.0, 0.30):
            for name, a in (("fwd", 0.0), ("left", math.pi / 2), ("right", -math.pi / 2), ("back", math.pi)):
                dd, pp = ray((x + 0.35 * math.cos(yaw + a), y + 0.35 * math.sin(yaw + a), floor_z + h),
                             (math.cos(yaw + a), math.sin(yaw + a), 0.0), 12.0)
                out[f"{name}@{h:.2f}"] = (None if dd is None else round(dd + 0.35, 3), pp.split("/")[-1] if pp else "")
        return out

    def run_updates(cmd, seconds, rows=None):
        n = int(round(seconds / CTL_DT))
        t_upd, steps0 = 0.0, base.physics_steps
        for k in range(n):
            base.apply(*cmd, CTL_DT)
            a = time.perf_counter()
            kit.update()
            t_upd += time.perf_counter() - a
            ps = base.physics_state()
            if rows is not None and k % 6 == 0:
                st = loco.state()
                rows.append((round(k * CTL_DT, 2), [float(v) for v in st["pos"]], float(st["lin_b"][0]),
                             float(st["above_feet"]), float(st["tilt_deg"])))
            if ps["fallen"]:
                return {"fell_at_s": round((k + 1) * CTL_DT, 2), "why": base.last_fault["why"], "update_wall_s": t_upd,
                        "updates": k + 1, "substeps": base.physics_steps - steps0}
        return {"fell_at_s": None, "update_wall_s": t_upd, "updates": n, "substeps": base.physics_steps - steps0}

    results = []
    tot_upd = tot_wall = tot_sub = 0
    t_survey0 = time.time() - T_LAUNCH
    for ep in eps:
        r: dict = {"episode": ep.name, "start": list(ep.start), "yaw_deg": ep.start_yaw_deg}
        try:
            base.reset_to(tuple(ep.start), ep.start_yaw_rad)
            base.reset_stats()
            loco.set_command(0, 0, 0)
            r["spawn"] = base.spawn
            fz = loco.floor_z
            r["clearance"] = clearances(ep.start[0], ep.start[1], ep.start_yaw_rad, fz)
            for _ in range(3):
                kit.update()
            st0 = run_updates((0, 0, 0), args.stand_s)
            r["stand"] = {k: v for k, v in st0.items() if k not in ("update_wall_s",)}
            r["stand_state"] = {k: round(float(v), 3) for k, v in
                                loco.state().items() if k in ("above_feet", "tilt_deg", "height")}
            rows = []
            st1 = run_updates((0.5, 0.0, 0.0), args.walk_s, rows=rows)
            r["walk"] = {k: v for k, v in st1.items() if k not in ("update_wall_s",)}
            if rows:
                p0, p1 = np.array(rows[0][1][:2]), np.array(rows[-1][1][:2])
                dt = rows[-1][0] - rows[0][0]
                r["walk"].update({"distance_m": round(float(np.linalg.norm(p1 - p0)), 3),
                                  "mean_speed_mps": round(float(np.linalg.norm(p1 - p0) / max(dt, 1e-6)), 3),
                                  "above_feet_min": round(min(x[3] for x in rows), 3), "tilt_max": round(max(x[4] for x in rows), 2),
                                  "z_end": round(rows[-1][1][2], 3), "floor_z": round(fz, 3)})
            for s in (st0, st1):
                tot_upd += s["updates"]
                tot_wall += s["update_wall_s"]
                tot_sub += s["substeps"]
            r["torso_contact_steps"] = base.torso_contact_steps
            r["g1_stats"] = base.stats()
            r["fell"] = bool(st0["fell_at_s"] is not None or st1["fell_at_s"] is not None)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            r["error"] = repr(e)
        results.append(r)
        print(f"[survey] {ep.name}: fell={r.get('fell')} spawn={r.get('spawn', {}).get('ground_z')} "
              f"headroom={r.get('clearance', {}).get('headroom_m')} walk={r.get('walk', {}).get('distance_m')}", flush=True)
        report["results"] = results
        save_report()

    phx = [x.get("g1_stats", {}).get("physx_step_ms_mean") for x in results]
    phx = [v for v in phx if v is not None]
    report["cost"] = {"physx_step_ms_mean_over_poses": round(sum(phx) / len(phx), 3) if phx else None,
                      "updates": tot_upd, "control_seconds": round(tot_upd * CTL_DT, 2),
                      "update_wall_s": round(tot_wall, 2), "wall_per_control_second": round(tot_wall / max(tot_upd * CTL_DT, 1e-9), 3),
                      "ms_per_update": round(1000 * tot_wall / max(tot_upd, 1), 2),
                      "ms_per_physics_substep": round(1000 * tot_wall / max(tot_sub, 1), 3), "substeps": tot_sub}
    report["gpu_mem"] = gpu.summary()
    report["cost"]["gpu_load_whole_survey"] = gpu.window(t_survey0, time.time() - T_LAUNCH)
    report["stage_file"]["root_layer_dirty"] = bool(stage.GetRootLayer().dirty)
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
        report["stage_file"]["md5_after"] = hashlib.md5(Path(report["stage_file"]["path"]).read_bytes()).hexdigest()
        report["stage_file"]["unchanged"] = report["stage_file"]["md5_after"] == report["stage_file"]["md5_before"]
    except Exception as e:  # noqa: BLE001
        report.setdefault("stage_file", {})["md5_after"] = repr(e)
    save_report()
    print(f"[survey] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    print(f"[survey] STATUS={'OK' if rc == 0 else 'FAILED'}", flush=True)
    sys.stdout.flush()
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    sys.exit(rc)
