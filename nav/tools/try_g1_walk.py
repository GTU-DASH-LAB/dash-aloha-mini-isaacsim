"""Walk the Unitree G1 on Isaac Lab's pretrained flat policy in plain Isaac Sim 6.0.1, through
nav/sim/g1/locomotion.py, and record what it does and what its own head camera sees.

    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/try_g1_walk.py \
        --scenario flat --out <dir>
    nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh nav/tools/try_g1_walk.py \
        --scenario hospital --episode hospital_down_hallway --out <dir>

flat      Grid default_environment, the command schedule the task fixes (stand 2 s, vx 0.5 6 s,
          wz +0.6 4 s, wz -0.6 4 s, vy +0.3 3 s, vx 0.8 5 s, stop 3 s), then an EXTENSION the
          task did not ask for, labelled as such: vx 1.0 for 5 s (the top of the trained range,
          which is what an episode timeout has to be checked against) and stop 2 s.
hospital  assets/usd/nav_hospital.usda opened IN MEMORY (every edit in the session layer,
          /World/Aloha deactivated, root layer verified clean and the file's md5 verified
          unchanged), G1 snapped onto the floor at the episode start, a scripted corridor walk.

Physics: 200 Hz (5 ms) synchronous PhysX steps, policy every 4 (50 Hz), rendering ONLY every
0.1 s (render_only(): app.update with playSimulations off). Panels are written as JPEGs plus a
per-frame telemetry JSON; nav/tools/g1_make_video.py composes and encodes them, so layout
changes never need another Kit run.
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

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", choices=["flat", "hospital"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episode", default="hospital_down_hallway")
ap.add_argument("--fps", type=float, default=10.0, help="render + record rate")
ap.add_argument("--no-real-cam", action="store_true", help="skip the 47.6 deg real-mount camera")
ap.add_argument("--no-chase", action="store_true")
ap.add_argument("--no-extension", action="store_true", help="flat: skip the vx 1.0 extension")
ap.add_argument("--full-res-every", type=float, default=2.0, help="save a full-res nav frame every N s")
args = ap.parse_args()

OUT = Path(args.out)
(OUT / "frames").mkdir(parents=True, exist_ok=True)
(OUT / "fullres").mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

T_LAUNCH = time.time()
kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})

report: dict = {"scenario": args.scenario, "args": vars(args), "pid": os.getpid()}


def save_report():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


# ----------------------------------------------------------------------------- GPU memory poll
class GpuPoll(threading.Thread):
    """This process's own GPU memory (nvidia-smi per-process), not the card total, so other
    Kit instances on GPU0 do not contaminate it."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples: list[tuple[float, dict]] = []
        self.stop_flag = False
        self.pid = os.getpid()

    def read(self) -> dict:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=10).stdout
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
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    import locomotion as L
    from isaacsim.sensors.camera import Camera

    import omni.kit.app

    kapp = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    loco = L.G1Locomotion("/World/G1")
    aspec = loco.asset_spec
    nav_spec, real_spec = aspec["nav_camera"], aspec["head_camera_real"]

    # ------------------------------------------------------------------ stage
    t0 = time.time()
    episode = None
    stage_file_md5 = None
    if args.scenario == "flat":
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
        start_xy, start_yaw, start_z = (0.0, 0.0), 0.0, 0.0
        for _ in range(120):
            kit.update()
        stage.Load()
        for _ in range(20):
            kit.update()
        # visual distance markers along +X (no physics), like the asset probe: judge camera pitch by eye
        for i, (x, y, col) in enumerate([(2.0, 0.6, (1, 0, 0)), (4.0, -0.6, (0, 0.8, 0)),
                                         (8.0, 0.8, (0, 0.3, 1)), (12.0, -0.9, (1, 0.8, 0))]):
            c = UsdGeom.Cube.Define(stage, f"/World/Markers/m{i}_{int(x)}m")
            c.CreateSizeAttr(1.0)
            c.CreateDisplayColorAttr([Gf.Vec3f(*col)])
            xf = UsdGeom.Xformable(c)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.5))
            xf.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 1.0))
    else:
        from episode import load_episode

        ep = load_episode(args.episode)
        episode = {"name": ep.name, "start": list(ep.start), "start_yaw_deg": math.degrees(ep.start_yaw_rad),
                   "goal": list(ep.goal), "instruction": ep.instruction, "timeout_s": ep.timeout_s}
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
    # From here on, EVERY edit goes to the session layer: the opened file is never dirtied.
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    report["stage_load_s"] = round(time.time() - t0, 1)
    if args.scenario == "hospital":
        aloha = stage.GetPrimAtPath("/World/Aloha")
        if aloha.IsValid():
            aloha.SetActive(False)
            report["aloha_deactivated_in_session_layer"] = True

    lights = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    report["lights"] = lights[:20]
    if not lights:
        UsdLux.DomeLight.Define(stage, "/World/G1Dome").CreateIntensityAttr(1000.0)
        UsdLux.DistantLight.Define(stage, "/World/G1Sun").CreateIntensityAttr(3000.0)
        report["lights_added"] = True

    report["physics_scene_changes"] = L.configure_physics_scene(stage, loco.physics_dt, cpu=True)
    if args.scenario == "hospital":
        try:
            from physics_setup import raise_gpu_capacities

            report["gpu_capacity_changes"] = raise_gpu_capacities(stage)
        except Exception as e:  # noqa: BLE001
            report["gpu_capacity_changes"] = f"physics_setup import failed: {e!r}"

    # ------------------------------------------------------------------ robot + cameras
    spawn_guess = (start_xy[0], start_xy[1], start_z + L.SPAWN_HEIGHT_M)
    report["author"] = loco.author(stage, spawn_guess, start_yaw)
    cam_nav_path = L.author_head_camera(stage, loco.prim_path, nav_spec, "g1_head_cam")
    cam_real_path = None if args.no_real_cam else L.author_head_camera(stage, loco.prim_path, real_spec,
                                                                       "g1_head_cam_real")
    chase_path = None
    if not args.no_chase:
        chase = UsdGeom.Camera.Define(stage, "/World/G1Chase")
        chase.CreateFocalLengthAttr(16.0)
        chase.CreateHorizontalApertureAttr(20.955)
        chase.CreateVerticalApertureAttr(20.955 * 9 / 16)
        chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        chase_path = "/World/G1Chase"

    def aim_chase(eye, target):
        # Through the Camera wrapper: it re-authors the prim's xform ops as translate/orient/
        # scale when it is created, so a raw xform op held from before that is invalid.
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                    Gf.Vec3d(0, 0, 1)).GetInverse()
        q = m.ExtractRotationQuat()
        im = q.GetImaginary()
        cams["chase"].set_world_pose(position=np.array(list(map(float, eye))),
                                     orientation=np.array([q.GetReal(), im[0], im[1], im[2]]),
                                     camera_axes="usd")

    # ------------------------------------------------------------------ play
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    # One plain update WITH playSimulations on: SimulationManager only builds the tensor
    # physics view if that setting is true when the PLAY event is handled.
    kit.update()
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    report["physics_view_ok"] = SimulationManager.get_physics_sim_view() is not None
    report["init"] = loco.initialize()
    report["contact_view"] = loco.record.get("contact_view")
    stepper = L.PhysicsStepper(loco.physics_dt)

    try:
        from pxr import PhysxSchema

        sc = L.physics_scene_prims(stage)[0]
        pxs = PhysxSchema.PhysxSceneAPI(sc)
        report["physics_scene_effective"] = {
            "path": str(sc.GetPath()),
            "timeStepsPerSecond": pxs.GetTimeStepsPerSecondAttr().Get(),
            "enableGPUDynamics": pxs.GetEnableGPUDynamicsAttr().Get(),
            "broadphaseType": pxs.GetBroadphaseTypeAttr().Get(),
            "solverType": pxs.GetSolverTypeAttr().Get(),
            "SimulationManager_physics_dt": SimulationManager.get_physics_dt(),
            "device": SimulationManager.get_physics_sim_device(),
        }
    except Exception as e:  # noqa: BLE001
        report["physics_scene_effective"] = repr(e)

    # ------------------------------------------------------------------ spawn on the floor
    floor = L.floor_below(start_xy[0], start_xy[1], start_z + L.SPAWN_PROBE_ABOVE_M, skip_prefix=loco.prim_path)
    floor_z, floor_prim = (floor if floor else (start_z, "none"))
    report["floor"] = {"z": floor_z, "prim": floor_prim}
    spawn = (start_xy[0], start_xy[1], floor_z + L.SPAWN_HEIGHT_M)
    loco.reset_to(spawn, start_yaw, floor_z=floor_z)
    report["spawn"] = {"pelvis": spawn, "yaw_deg": math.degrees(start_yaw)}

    # Lateral / forward clearance at chest height (hospital): which side has room for the turn.
    RAY_START_OUT_M = 0.35   # start outside the G1's torso box (a ray born inside a shape hits it at 0)

    def ray(origin, direction, dist=30.0):
        """Distance from `origin` (on the pelvis axis) to the first non-robot collider along
        `direction`, or None. Cast from RAY_START_OUT_M out, iterating past robot hits."""
        import carb
        from omni.physx import get_physx_scene_query_interface

        q = get_physx_scene_query_interface()
        dvec = np.array(list(map(float, direction)))
        o = np.array(list(map(float, origin))) + RAY_START_OUT_M * dvec
        for _ in range(20):
            hit = q.raycast_closest(carb.Float3(*o.tolist()), carb.Float3(*dvec.tolist()), dist)
            if not hit or not hit.get("hit"):
                return None, ""
            prim = str(hit.get("collision", ""))
            hp = np.array([float(hit["position"][k]) for k in range(3)])
            if not prim.startswith(loco.prim_path):
                return float(np.linalg.norm(hp - np.array(list(map(float, origin))))), prim
            o = (o + 0.03 * dvec) if float(hit.get("distance", 0.0)) < 1e-4 else (hp + 0.01 * dvec)
        return None, ""

    def clearances(x, y, yaw, h=1.0):
        out = {}
        for name, a in (("fwd", 0.0), ("left", math.pi / 2), ("right", -math.pi / 2), ("back", math.pi)):
            d, p = ray((x, y, floor_z + h), (math.cos(yaw + a), math.sin(yaw + a), 0.0))
            out[name] = (None if d is None else round(d, 3), p.split("/")[-1] if p else "")
        return out

    clear0 = clearances(start_xy[0], start_xy[1], start_yaw)
    report["start_clearance_m"] = clear0
    turn_sign = 1.0
    if args.scenario == "hospital":
        lft = clear0["left"][0] or 0.0
        rgt = clear0["right"][0] or 0.0
        turn_sign = 1.0 if lft >= rgt else -1.0
        report["turn_toward"] = "left" if turn_sign > 0 else "right"

    # ------------------------------------------------------------------ schedule
    if args.scenario == "flat":
        schedule = [("stand", 2.0, (0, 0, 0)), ("vx0.5", 6.0, (0.5, 0, 0)), ("wz+0.6", 4.0, (0, 0, 0.6)),
                    ("wz-0.6", 4.0, (0, 0, -0.6)), ("vy+0.3", 3.0, (0, 0.3, 0)), ("vx0.8", 5.0, (0.8, 0, 0)),
                    ("stop", 3.0, (0, 0, 0))]
        if not args.no_extension:
            schedule += [("EXT_vx1.0", 5.0, (1.0, 0, 0)), ("EXT_stop", 2.0, (0, 0, 0))]
    else:
        w = 0.15 * turn_sign
        schedule = [("stand", 2.0, (0, 0, 0)), ("straight0.6", 14.0, (0.6, 0, 0)),
                    ("turn_out", 3.0, (0.5, 0, w)), ("turn_back", 3.0, (0.5, 0, -w)),
                    ("straight0.6b", 6.0, (0.6, 0, 0)), ("stop", 3.0, (0, 0, 0))]
    report["schedule"] = [{"name": n, "seconds": s, "cmd": c} for n, s, c in schedule]
    bounds, acc = [], 0.0
    for n, s, c in schedule:
        bounds.append((acc, acc + s, n, c))
        acc += s
    total_s = acc

    def command_at(t):
        for a, b, n, c in bounds:
            if a <= t < b:
                return n, c
        return bounds[-1][2], bounds[-1][3]

    # ------------------------------------------------------------------ cameras (Isaac Sim wrappers)
    W_NAV, H_NAV = nav_spec["resolution_wh"]
    cams = {"nav": Camera(prim_path=cam_nav_path, resolution=(W_NAV, H_NAV))}
    if cam_real_path:
        # Same optics (FOV), half the pixels: it is a comparison view, not a policy input.
        cams["real"] = Camera(prim_path=cam_real_path, resolution=(W_NAV // 2, H_NAV // 2))
    if chase_path:
        cams["chase"] = Camera(prim_path=chase_path, resolution=(1280, 720))
    for c in cams.values():
        c.initialize()

    chase_yaw = start_yaw

    def place_chase(pos, yaw, snap=False):
        nonlocal_state["chase_yaw"] = yaw if snap else (
            nonlocal_state["chase_yaw"] + 0.15 * math.atan2(math.sin(yaw - nonlocal_state["chase_yaw"]),
                                                            math.cos(yaw - nonlocal_state["chase_yaw"])))
        cy = nonlocal_state["chase_yaw"]
        if args.scenario == "flat":
            back, side, up = 3.2, 1.4, 1.5
        else:
            back, side, up = 2.6, 0.0, 1.35
        eye = (pos[0] - back * math.cos(cy) - side * math.sin(cy),
               pos[1] - back * math.sin(cy) + side * math.cos(cy), floor_z + up)
        aim_chase(eye, (pos[0], pos[1], floor_z + 0.75))

    nonlocal_state = {"chase_yaw": start_yaw}
    if chase_path:
        place_chase(spawn, start_yaw, snap=True)

    def render():
        L.render_only(kapp, loco.physics_dt, stepper.t)

    # Warm the renderers up (no physics advances: render_only).
    tw = time.time()
    for _ in range(30):
        render()
    report["render_warmup_s"] = round(time.time() - tw, 2)

    # Is the rendered head camera where locomotion.camera_pose() says it is? (USD/Fabric pose
    # of the camera prim through the wrapper vs head_link's PhysX pose + the spec'd mount.)
    try:
        cpos, cq = cams["nav"].get_world_pose(camera_axes="usd")
        rq = L.quat_to_matrix_wxyz(np.asarray(cq, dtype=float))
        mine = loco.camera_pose(nav_spec)
        view_usd = -rq[:, 2]
        report["camera_mount_check"] = {
            "eye_usd": np.round(cpos, 4).tolist(), "eye_physx_plus_mount": np.round(mine["eye"], 4).tolist(),
            "eye_err_m": round(float(np.linalg.norm(np.asarray(cpos) - mine["eye"])), 5),
            "pitch_down_deg_usd": round(math.degrees(math.asin(max(-1, min(1, -view_usd[2])))), 3),
            "pitch_down_deg_physx_plus_mount": round(mine["pitch_down_deg"], 3),
            "height_above_floor_m": round(mine["height"], 4), "forward_of_pelvis_m": round(mine["forward_of_pelvis"], 4),
            "intrinsics": np.asarray(cams["nav"].get_intrinsics_matrix()).round(3).tolist(),
        }
    except Exception as e:  # noqa: BLE001
        report["camera_mount_check"] = repr(e)

    def grab(name):
        rgba = cams[name].get_rgba()
        if rgba is None or rgba.size == 0:
            return None
        return np.asarray(rgba)[:, :, :3].astype(np.uint8)

    # ---- frame-latency test: is a frame read after ONE render_only() of the CURRENT scene?
    lag = {}
    if chase_path:
        pA = (spawn[0], spawn[1], spawn[2])
        aim_chase((pA[0] - 3.0, pA[1] + 0.0, floor_z + 1.5), (pA[0], pA[1], floor_z + 0.7))
        for _ in range(4):
            render()
        imgA = grab("chase")
        aim_chase((pA[0] + 0.0, pA[1] + 3.0, floor_z + 1.5), (pA[0], pA[1], floor_z + 0.7))
        render()
        img1 = grab("chase")
        render()
        img2 = grab("chase")
        for _ in range(4):
            render()
        imgB = grab("chase")
        if all(i is not None for i in (imgA, img1, img2, imgB)):
            d = lambda a, b: float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())  # noqa: E731
            lag = {"after1_vs_new": d(img1, imgB), "after1_vs_old": d(img1, imgA),
                   "after2_vs_new": d(img2, imgB), "old_vs_new": d(imgA, imgB)}
            lag["frames_of_lag"] = 0 if lag["after1_vs_new"] < lag["after1_vs_old"] else (
                1 if lag["after2_vs_new"] < 0.5 * lag["old_vs_new"] else 2)
        report["frame_latency_test"] = lag
        place_chase(spawn, start_yaw, snap=True)
        for _ in range(4):
            render()
    renders_per_capture = 1 + int(lag.get("frames_of_lag", 0))
    report["renders_per_capture"] = renders_per_capture
    save_report()

    # ------------------------------------------------------------------ main loop
    gpu_after_setup = gpu.read()
    report["gpu_after_setup"] = gpu_after_setup
    steps_per_frame = int(round(1.0 / args.fps / loco.physics_dt))           # 20 at 10 fps
    total_steps = int(round(total_s / loco.physics_dt))
    full_every = int(round(args.full_res_every / loco.physics_dt))
    timing = {"physics_s": 0.0, "policy_s": 0.0, "render_s": 0.0, "grab_s": 0.0, "save_s": 0.0,
              "measure_s": 0.0}
    rows, bob, frames_meta = [], [], []
    fell_at, fall_why = None, None
    torso_contact_first = None
    t_loop = time.time()
    frame_idx = 0
    loco_policy_time = 0.0
    for k in range(total_steps):
        t = k * loco.physics_dt
        seg, cmd = command_at(t)
        loco.set_command(*cmd)
        a = time.perf_counter()
        ticked = loco.pre_physics_step()
        b = time.perf_counter()
        stepper()
        c = time.perf_counter()
        timing["policy_s"] += b - a
        timing["physics_s"] += c - b
        t_after = (k + 1) * loco.physics_dt
        if ticked:
            # 50 Hz: head-camera pose for the bob statistics (cheap tensor reads)
            m0 = time.perf_counter()
            lt = loco.link_poses()
            cp = loco.camera_pose(nav_spec, lt)
            st = loco.state()
            bob.append((round(t_after, 3), seg, cp["height"], cp["pitch_down_deg"], cp["roll_deg"],
                        cp["forward_of_pelvis"], cp["left_of_pelvis"], st["height"]))
            timing["measure_s"] += time.perf_counter() - m0
        if (k + 1) % steps_per_frame == 0:
            m0 = time.perf_counter()
            st = loco.state()
            cf = loco.contact_forces()
            fell, why = loco.fallen(st)
            x, y = float(st["pos"][0]), float(st["pos"][1])
            row = {"t": round(t_after, 3), "seg": seg, "cmd": list(map(float, loco.cmd)),
                   "pos": [round(float(v), 4) for v in st["pos"]], "yaw_deg": round(math.degrees(st["yaw"]), 3),
                   "v_b": [round(float(v), 4) for v in st["lin_b"]], "w_b": [round(float(v), 4) for v in st["ang_b"]],
                   "pelvis_h": round(st["height"], 4), "roll_deg": round(st["roll_deg"], 3),
                   "pitch_deg": round(st["pitch_deg"], 3), "tilt_deg": round(st["tilt_deg"], 3),
                   "contact_n": None if cf is None else {kk: round(v, 2) for kk, v in cf.items()},
                   "fallen": fell}
            if args.scenario == "hospital":
                d, p = ray((x, y, floor_z + 1.0), (math.cos(st["yaw"]), math.sin(st["yaw"]), 0.0))
                row["fwd_clear_m"] = None if d is None else round(d, 3)
                row["goal_dist_m"] = round(math.hypot(episode["goal"][0] - x, episode["goal"][1] - y), 3)
            rows.append(row)
            timing["measure_s"] += time.perf_counter() - m0
            if cf and cf.get(L.TORSO_LINK, 0.0) > L.TORSO_CONTACT_N and torso_contact_first is None:
                torso_contact_first = {"t": row["t"], "force_n": cf[L.TORSO_LINK]}
            if fell and fell_at is None:
                fell_at, fall_why = row["t"], why
                print(f"[g1walk] FELL at t={row['t']:.2f}: {why}", flush=True)

            # ---- render + capture
            if chase_path:
                place_chase(st["pos"], st["yaw"])
            r0 = time.perf_counter()
            for _ in range(renders_per_capture):
                render()
            r1 = time.perf_counter()
            timing["render_s"] += r1 - r0
            imgs = {n: grab(n) for n in cams}
            r2 = time.perf_counter()
            timing["grab_s"] += r2 - r1
            for n, im in imgs.items():
                if im is None:
                    continue
                pil = Image.fromarray(im)
                if pil.size != (960, 540):
                    pil = pil.resize((960, 540), Image.BILINEAR)
                pil.save(OUT / "frames" / f"{n}_{frame_idx:05d}.jpg", quality=88)
            if imgs.get("nav") is not None and ((k + 1) % full_every == 0 or frame_idx == 0):
                Image.fromarray(imgs["nav"]).save(OUT / "fullres" / f"nav_t{t_after:05.1f}.jpg", quality=92)
                if imgs.get("real") is not None:
                    Image.fromarray(imgs["real"]).save(OUT / "fullres" / f"real_t{t_after:05.1f}.jpg", quality=92)
            cpn = loco.camera_pose(nav_spec)
            frames_meta.append({"i": frame_idx, **row, "cam_h": round(cpn["height"], 4),
                                "cam_pitch": round(cpn["pitch_down_deg"], 3),
                                "cam_roll": round(cpn["roll_deg"], 3)})
            timing["save_s"] += time.perf_counter() - r2
            frame_idx += 1
            if frame_idx % 20 == 0:
                print(f"[g1walk] t={row['t']:5.1f} {seg:12s} cmd=({row['cmd'][0]:.2f},{row['cmd'][1]:.2f},"
                      f"{row['cmd'][2]:.2f}) v_b=({row['v_b'][0]:+.2f},{row['v_b'][1]:+.2f}) "
                      f"w={row['w_b'][2]:+.2f} h={row['pelvis_h']:.3f} tilt={row['tilt_deg']:.1f} "
                      f"cam_h={cpn['height']:.3f} pitch={cpn['pitch_down_deg']:.1f} "
                      f"wall={time.time() - t_loop:.1f}s", flush=True)
            if fell_at is not None and row["t"] > fell_at + 1.0:
                break
    wall_loop = time.time() - t_loop
    sim_done = stepper.t
    gpu.stop_flag = True

    # ------------------------------------------------------------------ analysis
    def seg_stats(name, skip=1.0):
        rs = [r for r in rows if r["seg"] == name]
        if not rs:
            return None
        t_start = min(r["t"] for r in rs) - 1.0 / args.fps
        steady = [r for r in rs if r["t"] >= t_start + skip] or rs
        cmd = np.array(steady[0]["cmd"])
        vb = np.array([r["v_b"] for r in steady])
        wz = np.array([r["w_b"][2] for r in steady])
        meas = np.array([vb[:, 0].mean(), vb[:, 1].mean(), wz.mean()])
        # displacement-based velocity over the steady window, in the mean heading frame
        p0, p1 = np.array(steady[0]["pos"][:2]), np.array(steady[-1]["pos"][:2])
        y0, y1 = math.radians(steady[0]["yaw_deg"]), math.radians(steady[-1]["yaw_deg"])
        dt = steady[-1]["t"] - steady[0]["t"]
        dyaw = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
        ym = y0 + dyaw / 2
        d = p1 - p0
        disp = None
        if dt > 0.3:
            disp = [round((math.cos(ym) * d[0] + math.sin(ym) * d[1]) / dt, 4),
                    round((-math.sin(ym) * d[0] + math.cos(ym) * d[1]) / dt, 4), round(dyaw / dt, 4)]
        err = meas - cmd
        rms = np.sqrt(((np.stack([vb[:, 0], vb[:, 1], wz], 1) - cmd) ** 2).mean(0))
        return {"cmd": cmd.round(3).tolist(), "n": len(steady), "window_s": round(dt, 2),
                "mean_measured_vx_vy_wz": meas.round(4).tolist(),
                "mean_error": err.round(4).tolist(), "rms_error": rms.round(4).tolist(),
                "displacement_vx_vy_wz": disp,
                "tracking_ratio": [round(float(meas[i] / cmd[i]), 3) if abs(cmd[i]) > 1e-6 else None for i in range(3)],
                "pelvis_h_min_max": [min(r["pelvis_h"] for r in rs), max(r["pelvis_h"] for r in rs)],
                "tilt_max_deg": max(r["tilt_deg"] for r in rs),
                "roll_absmax_deg": max(abs(r["roll_deg"]) for r in rs),
                "pitch_absmax_deg": max(abs(r["pitch_deg"]) for r in rs)}

    segments = {n: seg_stats(n) for n, _, _ in schedule}
    bob_arr = np.array([b[2:] for b in bob], dtype=float) if bob else np.zeros((0, 6))

    def bob_stats(mask):
        a = bob_arr[mask]
        if not len(a):
            return None
        return {"n": int(len(a)),
                "cam_height_m": {"mean": round(float(a[:, 0].mean()), 4), "std": round(float(a[:, 0].std()), 4),
                                 "min": round(float(a[:, 0].min()), 4), "max": round(float(a[:, 0].max()), 4)},
                "cam_pitch_down_deg": {"mean": round(float(a[:, 1].mean()), 3), "std": round(float(a[:, 1].std()), 3),
                                       "min": round(float(a[:, 1].min()), 3), "max": round(float(a[:, 1].max()), 3)},
                "cam_roll_deg": {"mean": round(float(a[:, 2].mean()), 3), "std": round(float(a[:, 2].std()), 3),
                                 "absmax": round(float(np.abs(a[:, 2]).max()), 3)},
                "cam_forward_of_pelvis_m": {"mean": round(float(a[:, 3].mean()), 4), "std": round(float(a[:, 3].std()), 4)},
                "cam_left_of_pelvis_m": {"mean": round(float(a[:, 4].mean()), 4), "std": round(float(a[:, 4].std()), 4)},
                "pelvis_height_m": {"mean": round(float(a[:, 5].mean()), 4), "min": round(float(a[:, 5].min()), 4),
                                    "max": round(float(a[:, 5].max()), 4)}}

    segs = np.array([b[1] for b in bob])
    tt = np.array([b[0] for b in bob])
    walking = np.array([s not in ("stand", "stop", "EXT_stop") for s in segs]) if len(segs) else np.zeros(0, bool)
    head = {"all_after_2s": bob_stats(tt >= 2.0) if len(tt) else None,
            "walking": bob_stats(walking & (tt >= 3.0)) if len(tt) else None,
            "standing": bob_stats(np.array([s == "stand" for s in segs]) & (tt >= 1.0)) if len(tt) else None}
    for n, _, _ in schedule:
        if len(segs):
            head[f"seg:{n}"] = bob_stats(segs == n)

    nf = max(1, frame_idx)
    report.update({
        "fell_at_s": fell_at, "fall_why": fall_why, "torso_contact_first": torso_contact_first,
        "seconds_simulated": round(sim_done, 3), "frames": frame_idx,
        "segments": segments, "head_camera": head,
        "cmd_clamped": loco.clamped_commands, "reverse_requests": loco.reverse_requests,
        "policy_ticks": loco.ticks, "physics_steps": stepper.steps,
        "cost": {
            "wall_loop_s": round(wall_loop, 2),
            "wall_per_sim_s": round(wall_loop / max(sim_done, 1e-9), 3),
            "breakdown_s": {k: round(v, 2) for k, v in timing.items()},
            "physics_ms_per_step": round(1000 * timing["physics_s"] / max(stepper.steps, 1), 3),
            "policy_ms_per_tick": round(1000 * timing["policy_s"] / max(loco.ticks, 1), 3),
            "render_ms_per_capture": round(1000 * timing["render_s"] / nf, 2),
            "grab_ms_per_capture": round(1000 * timing["grab_s"] / nf, 2),
            "save_ms_per_capture": round(1000 * timing["save_s"] / nf, 2),
            "cameras": {n: list(c.get_resolution()) for n, c in cams.items()},
            "renders_per_capture": renders_per_capture,
            "physics_only_wall_per_sim_s": round((timing["physics_s"] + timing["policy_s"]) / max(sim_done, 1e-9), 3),
        },
        "gpu_mem": gpu.summary(),
        "gpu_samples": gpu.samples[-200:],
        "launch_to_loop_end_s": round(time.time() - T_LAUNCH, 1),
    })
    if args.scenario == "hospital":
        report["final"] = {"pos": rows[-1]["pos"] if rows else None, "goal_dist_m": rows[-1].get("goal_dist_m") if rows else None,
                           "min_fwd_clear_m": min((r["fwd_clear_m"] for r in rows if r.get("fwd_clear_m") is not None), default=None)}
        report["end_clearance_m"] = clearances(rows[-1]["pos"][0], rows[-1]["pos"][1], math.radians(rows[-1]["yaw_deg"])) if rows else None
        root_dirty = stage.GetRootLayer().dirty
        report["stage_file"]["root_layer_dirty"] = bool(root_dirty)
    (OUT / "telemetry.json").write_text(json.dumps({"rows": rows, "frames": frames_meta, "bob_50hz": bob}, default=str))
    report["status"] = "ok"
    rc = 0
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    report["status"] = f"error: {e!r}"
    rc = 1
finally:
    try:
        if report.get("stage_file"):
            report["stage_file"]["md5_after"] = hashlib.md5(Path(report["stage_file"]["path"]).read_bytes()).hexdigest()
            report["stage_file"]["unchanged"] = report["stage_file"]["md5_after"] == report["stage_file"]["md5_before"]
    except Exception as e:  # noqa: BLE001
        report.setdefault("stage_file", {})["md5_after"] = repr(e)
    save_report()
    print(f"[g1walk] wrote {OUT / 'report.json'} status={report.get('status')}", flush=True)
    # The process exit code is NOT the verdict: kit.close() ends the process and nothing after
    # it runs (sys.exit included), so a failed run still exits 0. Read this line / report.json.
    print(f"[g1walk] STATUS={'OK' if rc == 0 else 'FAILED'}", flush=True)
    sys.stdout.flush()
    try:
        omni.timeline.get_timeline_interface().stop()
    except Exception:  # noqa: BLE001
        pass
    kit.close()
    sys.exit(rc)
