#!/usr/bin/env python3
"""Probe the Unitree G1 asset the walking policy needs, and render its OWN head camera.

What this measures (one headless Kit session, plain grid floor, nothing else in the world):

  1. The articulation as PhysX sees it: root path, every DOF in PhysX order, limits (rad),
     drive gains as authored (read back through the tensor API, i.e. SI per-radian units),
     max efforts, armatures, body names + masses, articulation-root settings.
  2. The default standing geometry, measured WITHOUT balance in the loop: gravity is
     switched off for every body, the joints are driven to Isaac Lab's G1_CFG default
     pose, and once they settle the lowest corner of either foot COLLIDER is the floor.
     standing pelvis z = pelvis z - that corner. Same for the camera height.
  3. A head camera mounted on head_link at the pose given on the command line (default:
     Unitree's d435_joint from g1_29dof_with_hand_rev_1_0.urdf, re-expressed on this
     asset's head), with D435i colour intrinsics. It renders the frame with the robot
     standing on the floor (still gravity-free, so the pose is exact), plus a level-pitch
     variant from the same point and a third-person view, so self-occlusion and pitch can
     be judged by LOOKING.
  4. Then gravity on, feet on the floor, PD holding the default pose (no policy): pelvis z
     and tilt over 2 s. A humanoid without a balance controller is expected to fall
     eventually; the number says how long the PD-only stance survives and what the
     contact-loaded standing height is while it does.

Optionally a second asset (--compare) is loaded off to the side (gravity off, never in the
head camera's field) to record its DOF order/gains/standing height as well.

Usage -- ALWAYS through the GPU0 slot wrapper, Isaac's python.sh, output to a file:

    ISAAC_SLOT_MIN_FREE_MIB=8000 ISAAC_SLOT_TAG=g1-asset nav/tools/isaac_slot.sh \\
        ~/robotics/isaacsim-6.0.1/python.sh nav/tools/probe_g1_asset.py \\
        --out-dir /tmp/.../scratchpad/g1/probe > /tmp/.../probe.log 2>&1; echo $?

Writes <out-dir>/probe.json and <out-dir>/*.png. Nothing in the repo is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

# nav/tools/profile.py shadows the stdlib `profile`, which warp imports while Kit starts.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]

CDN = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"
DEFAULT_ASSET = f"{CDN}/IsaacLab/Robots/Unitree/G1/g1_minimal.usd"
DEFAULT_COMPARE = f"{CDN}/Robots/Unitree/G1/g1.usd"
GRID_ENV = f"{CDN}/Environments/Grid/default_environment.usd"

# Isaac Lab G1_CFG (IsaacLab-2.2.1 source/isaaclab_assets/isaaclab_assets/robots/unitree.py
# lines 270-373, identical on GitHub main/develop) -- the actuator + init_state the
# Isaac-Velocity-*-G1-v0 policy was trained with.
G1_DEFAULT_JOINT_POS = {
    r".*_hip_pitch_joint": -0.20,
    r".*_knee_joint": 0.42,
    r".*_ankle_pitch_joint": -0.23,
    r".*_elbow_pitch_joint": 0.87,
    r"left_shoulder_roll_joint": 0.16,
    r"left_shoulder_pitch_joint": 0.35,
    r"right_shoulder_roll_joint": -0.16,
    r"right_shoulder_pitch_joint": 0.35,
    r"left_one_joint": 1.0,
    r"right_one_joint": -1.0,
    r"left_two_joint": 0.52,
    r"right_two_joint": -0.52,
}
# (regex, stiffness, damping, effort_limit_sim, armature) -- first match wins
G1_ACTUATORS = [
    (r".*_hip_yaw_joint", 150.0, 5.0, 300.0, 0.01),
    (r".*_hip_roll_joint", 150.0, 5.0, 300.0, 0.01),
    (r".*_hip_pitch_joint", 200.0, 5.0, 300.0, 0.01),
    (r".*_knee_joint", 200.0, 5.0, 300.0, 0.01),
    (r"torso_joint", 200.0, 5.0, 300.0, 0.01),
    (r".*_ankle_pitch_joint", 20.0, 2.0, 20.0, 0.01),
    (r".*_ankle_roll_joint", 20.0, 2.0, 20.0, 0.01),
    (r".*_shoulder_.*", 40.0, 10.0, 300.0, 0.01),
    (r".*_elbow_.*", 40.0, 10.0, 300.0, 0.01),
    (r".*_(zero|one|two|three|four|five|six)_joint", 40.0, 10.0, 300.0, 0.001),
]
G1_SPAWN_Z = 0.74  # G1_CFG.init_state.pos


def rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_to_quat(m):
    """Column-vector rotation matrix -> (w, x, y, z)."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def camera_quat_in_link(pitch_down_rad: float):
    """USD camera (looks -Z, up +Y) mounted in a link whose +X is forward, +Z up,
    then pitched DOWN by pitch_down_rad about the link's +Y (URDF/ROS convention: a
    positive rpy pitch rotates +X toward -Z, which is how d435_joint is written)."""
    m_level = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]  # columns: cam_x=-Y, cam_y=+Z, cam_z=-X
    return mat_to_quat(matmul(rot_y(pitch_down_rad), m_level))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--compare", default=DEFAULT_COMPARE, help="'' to skip")
    ap.add_argument("--cam-link", default="head_link", help="link (child of the robot prim) the camera rides on")
    # Unitree d435_joint (g1_29dof_with_hand_rev_1_0.urdf) is (0.0576235, 0.01753, 0.42987) in
    # the rev_1_0 torso_link = (0.05366, 0.01753, 0.47387) in its pelvis/head frame. The 37-DOF
    # asset's head mesh is a different revision: the best rigid fit of the rev_1_0 head onto it
    # (surface chamfer 14.9 -> 8.4 mm) is a shift of (-0.0055, 0, -0.0110), which puts the
    # camera at (0.0482, 0.0175, 0.4629) in g1_minimal's head_link frame (== pelvis frame).
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.0482, 0.0175, 0.4629],
                    help="camera origin in --cam-link's frame (m)")
    ap.add_argument("--near", type=float, default=0.05,
                    help="near clip (m). The real D435 sits behind a transparent visor ~12-14 mm "
                         "inside the head shell; in sim the shell is opaque, so it must be clipped")
    ap.add_argument("--cam-pitch-deg", type=float, default=47.6,
                    help="downward pitch (d435_joint rpy pitch 0.8307767 rad = 47.60 deg)")
    ap.add_argument("--hfov-deg", type=float, default=69.0)
    ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--stand-seconds", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"args": vars(args), "status": "started"}

    def save():
        (out_dir / "probe.json").write_text(json.dumps(report, indent=1, default=str))

    from isaacsim import SimulationApp

    kit = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
    try:
        _run(kit, args, out_dir, report, save)
        report["status"] = "completed"
    except Exception as e:  # noqa: BLE001
        report["status"] = "crashed"
        report["fatal"] = "".join(traceback.format_exception(e))
        print(report["fatal"], flush=True)
    save()
    print(f"[probe_g1] STATUS {report['status']} -> {out_dir / 'probe.json'}", flush=True)
    rc = 0 if report["status"] == "completed" else 1
    # Last statement on purpose: nothing after SimulationApp.close() executes.
    kit.close()
    return rc


def _run(kit, args, out_dir: Path, report: dict, save) -> None:
    import re

    import numpy as np
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    from isaacsim.core.prims import Articulation
    from isaacsim.sensors.camera import Camera

    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        kit.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateTimeStepsPerSecondAttr(200)  # Isaac Lab sim.dt = 0.005

    env = stage.DefinePrim("/World/Env", "Xform")
    env.GetReferences().AddReference(GRID_ENV)

    # distance markers straight ahead (visual only, no physics): judge pitch/FOV by eye
    markers = []
    for i, (x, y, col) in enumerate([(1.0, 0.35, (1, 0, 0)), (2.0, -0.35, (0, 0.8, 0)),
                                     (4.0, 0.5, (0, 0.3, 1)), (8.0, -0.6, (1, 0.8, 0))]):
        c = UsdGeom.Cube.Define(stage, f"/World/Markers/m{i}_{int(x)}m")
        c.CreateSizeAttr(1.0)
        c.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.5))
        xf.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 1.0))
        markers.append({"path": str(c.GetPath()), "x": x, "y": y, "height_m": 1.0, "color": col})
    report["markers"] = markers

    robots = {"main": ("/World/G1", args.asset, (0.0, 0.0, 1.5))}
    if args.compare:
        robots["compare"] = ("/World/G1_compare", args.compare, (-3.0, 3.0, 1.5))
    for key, (path, url, pos) in robots.items():
        p = stage.DefinePrim(path, "Xform")
        p.GetReferences().AddReference(url)
        xf = UsdGeom.Xformable(p)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

    print("[probe_g1] loading stage (CDN)...", flush=True)
    t0 = time.time()
    for _ in range(120):
        kit.update()
    stage.Load()
    for _ in range(30):
        kit.update()
    report["stage_load_s"] = round(time.time() - t0, 1)

    lights = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI) or "Light" in p.GetTypeName()]
    report["lights_in_env"] = lights
    if not lights:
        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000.0)
        dl = UsdLux.DistantLight.Define(stage, "/World/Sun")
        dl.CreateIntensityAttr(3000.0)
        report["lights_added"] = True

    # ------------------------------------------------------------- static (USD) facts
    xc = UsdGeom.XformCache()

    def static_facts(root: str) -> dict:
        d: dict = {"root_prim": root}
        roots = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        d["articulation_roots"] = roots
        if roots:
            rp = stage.GetPrimAtPath(roots[0])
            for an in ("physxArticulation:enabledSelfCollisions", "physxArticulation:solverPositionIterationCount",
                       "physxArticulation:solverVelocityIterationCount", "physxArticulation:sleepThreshold",
                       "physxArticulation:stabilizationThreshold"):
                a = rp.GetAttribute(an)
                d[an] = a.Get() if a and a.HasValue() else None
        joints = {}
        for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = {"type": p.GetTypeName(), "path": str(p.GetPath())}
            if p.HasAPI(UsdPhysics.DriveAPI, "angular"):
                dr = UsdPhysics.DriveAPI(p, "angular")
                j["usd_drive"] = {"type": dr.GetTypeAttr().Get(), "stiffness_per_deg": dr.GetStiffnessAttr().Get(),
                                  "damping_per_deg": dr.GetDampingAttr().Get(), "maxForce": dr.GetMaxForceAttr().Get(),
                                  "targetPosition_deg": dr.GetTargetPositionAttr().Get()}
            if p.IsA(UsdPhysics.RevoluteJoint):
                r = UsdPhysics.RevoluteJoint(p)
                j["usd_limits_deg"] = [r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()]
                j["axis"] = r.GetAxisAttr().Get()
            for an in ("physxJoint:armature", "physxJoint:maxJointVelocity", "physxJoint:jointFriction"):
                a = p.GetAttribute(an)
                if a and a.HasAuthoredValue():
                    j[an.split(":")[1]] = a.Get()
            joints[p.GetName()] = j
        d["usd_joints"] = joints
        return d

    def foot_corners(root: str) -> dict:
        """Lowest-relevant geometry of each foot, in the ankle_roll_link frame."""
        out = {}
        for side in ("left", "right"):
            link = stage.GetPrimAtPath(f"{root}/{side}_ankle_roll_link")
            Wl_inv = xc.GetLocalToWorldTransform(link).GetInverse()
            coll, vis = [], []
            for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                if not p.IsA(UsdGeom.Gprim):
                    continue
                rel = xc.GetLocalToWorldTransform(p) * Wl_inv
                is_coll = p.HasAPI(UsdPhysics.CollisionAPI)
                pts = []
                if p.IsA(UsdGeom.Mesh):
                    pts = [Gf.Vec3d(v) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()]
                elif p.IsA(UsdGeom.Cube):
                    s = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
                    pts = [Gf.Vec3d(a, b, c) for a in (-s, s) for b in (-s, s) for c in (-s, s)]
                elif p.IsA(UsdGeom.Sphere):
                    r = UsdGeom.Sphere(p).GetRadiusAttr().Get()
                    pts = [Gf.Vec3d(0, 0, -r)]
                pts = [rel.Transform(v) for v in pts]
                (coll if is_coll else vis).append((str(p.GetPath()), p.GetTypeName(), [list(v) for v in pts]))
            out[side] = {"collision": coll, "visual_count": len(vis),
                         "visual_min_z_local": min((v[2] for _, _, ps in vis for v in ps), default=None)}
        return out

    statics = {k: static_facts(v[0]) for k, v in robots.items()}
    feet = {k: foot_corners(v[0]) for k, v in robots.items()}
    report["static"] = statics
    report["foot_colliders_local"] = {k: {s: [(c[0], c[1], len(c[2])) for c in v[s]["collision"]] for s in v} for k, v in feet.items()}

    # ------------------------------------------------------------- head camera prim
    cam_parent = f"{robots['main'][0]}/{args.cam_link}"
    cam_path = f"{cam_parent}/g1_head_cam"
    assert stage.GetPrimAtPath(cam_parent).IsValid(), f"no {cam_parent}"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    w, h = args.res
    hap = 20.955
    focal = (hap / 2) / math.tan(math.radians(args.hfov_deg) / 2)
    vap = hap * h / w
    cam.CreateHorizontalApertureAttr(hap)
    cam.CreateVerticalApertureAttr(vap)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(args.near, 1000.0))
    cam.CreateProjectionAttr("perspective")
    cxf = UsdGeom.Xformable(cam)
    cxf.ClearXformOpOrder()
    t_op = cxf.AddTranslateOp()
    o_op = cxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    t_op.Set(Gf.Vec3d(*args.cam_pos))

    def set_pitch(deg):
        q = camera_quat_in_link(math.radians(deg))
        o_op.Set(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))

    set_pitch(args.cam_pitch_deg)
    vfov = 2 * math.degrees(math.atan(vap / 2 / focal))
    report["camera"] = {"prim": cam_path, "parent": cam_parent, "local_pos": args.cam_pos,
                        "pitch_down_deg": args.cam_pitch_deg, "focal_mm": focal, "h_aperture_mm": hap,
                        "v_aperture_mm": vap, "hfov_deg": args.hfov_deg, "vfov_deg": vfov,
                        "resolution": [w, h], "clipping": [args.near, 1000.0]}

    chase_path = "/World/ChaseCam"
    chase = UsdGeom.Camera.Define(stage, chase_path)
    chase.CreateFocalLengthAttr(18.0)
    chase.CreateHorizontalApertureAttr(20.955)
    chase.CreateVerticalApertureAttr(20.955 * 720 / 1280)
    chase.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    def aim(camprim, eye, target):
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        x = UsdGeom.Xformable(camprim)
        x.ClearXformOpOrder()
        x.AddTransformOp().Set(view.GetInverse())

    # camera marker (small red sphere), only visible in the chase render
    mk = UsdGeom.Sphere.Define(stage, f"{cam_parent}/g1_head_cam_marker")
    mk.CreateRadiusAttr(0.012)
    mk.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0)])
    UsdGeom.Xformable(mk).AddTranslateOp().Set(Gf.Vec3d(*args.cam_pos))
    UsdGeom.Imageable(mk).MakeInvisible()

    # ------------------------------------------------------------- play + articulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        kit.update()

    arts = {}
    for key, (path, _, _) in robots.items():
        root = statics[key]["articulation_roots"][0]
        a = Articulation(root)
        a.initialize()
        arts[key] = a

    def dyn_facts(a) -> dict:
        kps, kds = a.get_gains()
        lim = a.get_dof_limits()
        d = {"articulation_root": a.prim_paths[0] if hasattr(a, "prim_paths") else None,
             "num_dof": a.num_dof, "num_bodies": a.num_bodies,
             "dof_names": list(a.dof_names), "body_names": list(a.body_names),
             "dof_limits_rad": np.asarray(lim)[0].tolist(),
             "gains_SI_as_authored": {"kp": np.asarray(kps)[0].tolist(), "kd": np.asarray(kds)[0].tolist()},
             "max_efforts": np.asarray(a.get_max_efforts())[0].tolist(),
             "armatures": np.asarray(a.get_armatures())[0].tolist(),
             "max_joint_velocities": np.asarray(a.get_joint_max_velocities())[0].tolist(),
             "body_masses": np.asarray(a.get_body_masses())[0].tolist(),
             "joint_pos_at_play": np.asarray(a.get_joint_positions())[0].tolist()}
        d["total_mass"] = float(sum(d["body_masses"]))
        try:
            d["drive_types"] = np.asarray(a.get_drive_types())[0].tolist()
        except Exception as e:  # noqa: BLE001
            d["drive_types"] = repr(e)
        return d

    dyn = {k: dyn_facts(a) for k, a in arts.items()}
    report["articulation"] = dyn
    save()
    print("[probe_g1] main DOF order:", dyn["main"]["dof_names"], flush=True)

    # Isaac Lab actuator + default pose on the MAIN robot
    a = arts["main"]
    names = dyn["main"]["dof_names"]
    n = len(names)
    q0 = np.zeros(n)
    kp, kd, eff, arm = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    unmatched = []
    for i, nm in enumerate(names):
        for rx, val in G1_DEFAULT_JOINT_POS.items():
            if re.fullmatch(rx, nm):
                q0[i] = val
        for rx, s, dmp, e, ar in G1_ACTUATORS:
            if re.fullmatch(rx, nm):
                kp[i], kd[i], eff[i], arm[i] = s, dmp, e, ar
                break
        else:
            unmatched.append(nm)
    report["isaaclab_default_joint_pos_in_dof_order"] = q0.tolist()
    report["isaaclab_actuator_unmatched"] = unmatched
    a.set_gains(kps=kp[None], kds=kd[None])
    a.set_max_efforts(eff[None])
    a.set_armatures(arm[None])
    kps2, kds2 = a.get_gains()
    report["gains_SI_after_isaaclab"] = {"kp": np.asarray(kps2)[0].tolist(), "kd": np.asarray(kds2)[0].tolist()}

    nb = {k: x.num_bodies for k, x in arts.items()}
    for k, x in arts.items():
        x.set_body_disable_gravity(np.ones((1, nb[k]), dtype=bool))

    def body_poses(x):
        return np.asarray(x._physics_view.get_link_transforms()).reshape(nb_of(x), 7)  # x y z qx qy qz qw

    def nb_of(x):
        return x.num_bodies

    def quat_rot(q, v):  # q = (qx,qy,qz,qw)
        qx, qy, qz, qw = q
        r = Gf.Rotation(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
        return np.array(r.TransformDir(Gf.Vec3d(*map(float, v))))

    def hold(x, q, pos):
        x.set_world_poses(positions=np.array([pos]), orientations=np.array([[1.0, 0, 0, 0]]))
        x.set_joint_positions(q[None])
        x.set_joint_position_targets(q[None])
        x.set_joint_velocities(np.zeros((1, len(q))))
        x.set_velocities(np.zeros((1, 6)))

    def foot_floor(key, x, poses):
        names_b = list(x.body_names)
        lows = {}
        for side in ("left", "right"):
            bi = names_b.index(f"{side}_ankle_roll_link")
            p, q = poses[bi, :3], poses[bi, 3:]
            zs = [float((p + quat_rot(q, v))[2]) for _, _, pts in feet[key][side]["collision"] for v in pts]
            vz = feet[key][side]["visual_min_z_local"]
            lows[side] = {"collider_min_z": min(zs) if zs else None,
                          "ankle_roll_z": float(p[2]),
                          "visual_min_z_approx": float(p[2] + vz) if vz is not None else None}
        return lows

    # ---- phase A: gravity off, mid-air, joints to default pose
    hold(a, q0, (0.0, 0.0, 1.5))
    if "compare" in arts:
        nc = arts["compare"].num_dof
        hold(arts["compare"], np.zeros(nc), (-3.0, 3.0, 1.5))
    for _ in range(90):
        kit.update()
    qA = np.asarray(a.get_joint_positions())[0]
    poses = body_poses(a)
    bn = list(a.body_names)
    pel = poses[bn.index("pelvis")]
    lows = foot_floor("main", a, poses)
    floor_z = min(v["collider_min_z"] for v in lows.values())
    stand_pelvis = float(pel[2] - floor_z)
    hl = poses[bn.index(args.cam_link)]
    cam_w = hl[:3] + quat_rot(hl[3:], args.cam_pos)
    fwd_dir = quat_rot(hl[3:], [math.cos(math.radians(args.cam_pitch_deg)), 0, -math.sin(math.radians(args.cam_pitch_deg))])
    pel_x = quat_rot(pel[3:], [1, 0, 0])
    phaseA = {
        "joint_pos_err_max_rad": float(np.abs(qA - q0).max()),
        "pelvis_world": pel.tolist(),
        "foot_lows": lows,
        "standing_pelvis_height_m": stand_pelvis,
        "head_link_world": hl.tolist(),
        "head_link_minus_pelvis": (hl[:3] - pel[:3]).tolist(),
        "camera_world": cam_w.tolist(),
        "camera_height_above_floor_m": float(cam_w[2] - floor_z),
        "camera_forward_of_pelvis_m": float(np.dot(cam_w - pel[:3], pel_x)),
        "camera_lateral_of_pelvis_m": float(np.dot(cam_w - pel[:3], quat_rot(pel[3:], [0, 1, 0]))),
        "camera_view_dir_world": fwd_dir.tolist(),
        "camera_pitch_below_horizontal_deg": float(math.degrees(math.asin(-fwd_dir[2]))),
        "spawn_z_isaaclab": G1_SPAWN_Z,
        "isaaclab_spawn_gap_m": float(G1_SPAWN_Z - stand_pelvis),
        "body_world_z_minus_floor": {b: float(poses[i, 2] - floor_z) for i, b in enumerate(bn)},
    }
    # cross-check against the renderer's view of the camera prim
    try:
        m = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(cam_path))
        phaseA["camera_world_usd_xform"] = list(m.ExtractTranslation())
        vd = m.TransformDir(Gf.Vec3d(0, 0, -1))
        up = m.TransformDir(Gf.Vec3d(0, 1, 0))
        phaseA["camera_view_dir_usd_xform"] = list(vd.GetNormalized())
        phaseA["camera_image_up_usd_xform"] = list(up.GetNormalized())
    except Exception as e:  # noqa: BLE001
        phaseA["camera_world_usd_xform"] = repr(e)
    if "compare" in arts:
        c = arts["compare"]
        pc = body_poses(c)
        bc = list(c.body_names)
        lc = foot_floor("compare", c, pc)
        fz = min(v["collider_min_z"] for v in lc.values() if v["collider_min_z"] is not None) if any(
            v["collider_min_z"] is not None for v in lc.values()) else min(v["visual_min_z_approx"] for v in lc.values())
        pp = pc[bc.index("pelvis")]
        tl = pc[bc.index("torso_link")]
        d435_local = (0.0576235, 0.01753, 0.42987)  # g1_29dof_with_hand_rev_1_0.urdf d435_joint, torso_link frame
        cw = tl[:3] + quat_rot(tl[3:], d435_local)
        phaseA["compare"] = {"foot_lows": lc, "floor_z_used": fz, "standing_pelvis_height_zero_pose_m": float(pp[2] - fz),
                             "d435_world": cw.tolist(), "d435_height_above_floor_zero_pose_m": float(cw[2] - fz),
                             "d435_forward_of_pelvis_m": float(cw[0] - pp[0])}
    report["phaseA_gravity_off"] = phaseA
    save()
    print("[probe_g1] phase A:", json.dumps({k: phaseA[k] for k in ("standing_pelvis_height_m", "camera_height_above_floor_m",
          "camera_forward_of_pelvis_m", "camera_pitch_below_horizontal_deg", "joint_pos_err_max_rad")}), flush=True)

    # ---- render: robot standing on the floor in the exact default pose (gravity still off)
    hold(a, q0, (0.0, 0.0, stand_pelvis))
    for _ in range(30):
        kit.update()
    head = Camera(prim_path=cam_path, resolution=(w, h))
    head.initialize()
    chase_cam = Camera(prim_path=chase_path, resolution=(1280, 720))
    chase_cam.initialize()
    for _ in range(60):
        kit.update()
    # read the intrinsics back AFTER the Camera wrapper had its chance to touch them
    cp = stage.GetPrimAtPath(cam_path)
    report["camera"]["usd_readback"] = {an: cp.GetAttribute(an).Get() for an in (
        "focalLength", "horizontalAperture", "verticalAperture", "clippingRange", "projection")}
    try:
        report["camera"]["intrinsics_matrix_px"] = np.asarray(head.get_intrinsics_matrix()).tolist()
        hf = 2 * math.degrees(math.atan(w / 2 / report["camera"]["intrinsics_matrix_px"][0][0]))
        vf = 2 * math.degrees(math.atan(h / 2 / report["camera"]["intrinsics_matrix_px"][1][1]))
        report["camera"]["fov_from_intrinsics_deg"] = [hf, vf]
    except Exception as e:  # noqa: BLE001
        report["camera"]["intrinsics_matrix_px"] = repr(e)

    def grab(cobj, name):
        for _ in range(20):
            rgba = cobj.get_rgba()
            if rgba is not None and rgba.size:
                img = np.asarray(rgba)[:, :, :3].astype(np.uint8)
                if img.max() > 0:
                    Image.fromarray(img).save(out_dir / name)
                    return {"file": str(out_dir / name), "shape": list(img.shape), "mean": float(img.mean())}
            kit.update()
        return {"file": None}

    renders = {}
    renders["head_d435_mount"] = grab(head, "g1_head_cam_d435_pitch%.1f.png" % args.cam_pitch_deg)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    for _ in range(20):
        kit.update()
    renders["head_d435_mount_near0.01"] = grab(head, "g1_head_cam_d435_near0.01.png")
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(args.near, 1000.0))
    set_pitch(0.0)
    for _ in range(20):
        kit.update()
    renders["head_level"] = grab(head, "g1_head_cam_level.png")
    set_pitch(args.cam_pitch_deg)
    UsdGeom.Imageable(mk).MakeVisible()
    aim(chase.GetPrim(), (1.6, 1.3, 1.45), (0.0, 0.0, 0.85))
    for _ in range(30):
        kit.update()
    renders["chase_front"] = grab(chase_cam, "g1_chase_front.png")
    aim(chase.GetPrim(), (0.0, 2.2, 1.1), (0.0, 0.0, 0.85))
    for _ in range(30):
        kit.update()
    renders["chase_side"] = grab(chase_cam, "g1_chase_side.png")
    UsdGeom.Imageable(mk).MakeInvisible()
    report["renders"] = renders
    save()

    # ---- phase B: gravity on, PD holding the default pose, no policy
    hold(a, q0, (0.0, 0.0, stand_pelvis + 0.002))
    a.set_body_disable_gravity(np.zeros((1, nb["main"]), dtype=bool))
    trace = []
    t_start = timeline.get_current_time()
    while timeline.get_current_time() - t_start < args.stand_seconds:
        kit.update()
        p, q = a.get_world_poses()
        p, q = np.asarray(p)[0], np.asarray(q)[0]  # q = w x y z
        upz = Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))).TransformDir(Gf.Vec3d(0, 0, 1))[2]
        trace.append([round(timeline.get_current_time() - t_start, 4), float(p[2]), float(p[0]), float(p[1]),
                      float(math.degrees(math.acos(max(-1, min(1, upz)))))])
    tr = np.array(trace)
    poses = body_poses(a)
    lowsB = foot_floor("main", a, poses)
    report["phaseB_gravity_on_pd_only"] = {
        "trace_t_z_x_y_tiltdeg": trace[:: max(1, len(trace) // 80)],
        "n_samples": len(trace),
        "pelvis_z_first_0.25s_mean": float(tr[tr[:, 0] < 0.25, 1].mean()) if len(tr) else None,
        "pelvis_z_min": float(tr[:, 1].min()), "pelvis_z_final": float(tr[-1, 1]),
        "tilt_max_deg": float(tr[:, 4].max()), "tilt_final_deg": float(tr[-1, 4]),
        "fell": bool(tr[:, 1].min() < 0.5 * stand_pelvis),
        "foot_lows_final": lowsB,
        "joint_pos_final": np.asarray(a.get_joint_positions())[0].tolist(),
    }
    renders["head_after_phaseB"] = grab(head, "g1_head_cam_after_pd_stand.png")
    aim(chase.GetPrim(), (1.6, 1.3, 1.45), (0.0, 0.0, 0.85))
    for _ in range(10):
        kit.update()
    renders["chase_after_phaseB"] = grab(chase_cam, "g1_chase_after_pd_stand.png")
    save()
    timeline.stop()
    for _ in range(5):
        kit.update()


if __name__ == "__main__":
    sys.exit(main())
