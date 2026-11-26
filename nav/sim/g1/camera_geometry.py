"""Pinhole geometry of the G1's head camera for the arc menu.

`nav/arc_menu.project()` assumes a LEVEL lens: v = h/2 + fx * cam_h / depth. The G1's D435 mount
is pitched 12 deg down at rest and 15.3 deg while walking (the policy leans ~3 deg forward;
nav/results/g1 + scratchpad/g1/runner reports), so a menu drawn with the level formula lands every
arc 150-250 px too high. `project_ground()` is the replacement. Its pose arguments are the camera's
pose at the instant the frame was rendered, in ONE convention shared with
G1Locomotion.camera_pose(), which calls `camera_angles()` below to produce them:

    R_cam = Rz(yaw_deg) @ Ry(pitch_deg) @ Rx(roll_deg) @ M_LEVEL      (heading frame)

  heading frame  x forward along the PELVIS yaw, y LEFT, z up, origin on the floor under the
                 rotation centre (the pelvis) -- the frame the menu's arcs are drawn in;
  M_LEVEL        a level USD camera (looks -Z, image-up +Y) whose view is +x;
  yaw_deg        view yaw relative to the pelvis heading, + = looking LEFT (the waist joint turns
                 the head -4.5..+7.3 deg against the pelvis while walking);
  pitch_deg      + = looking DOWN (Ry(+t) turns +x toward -z);
  roll_deg       rotation about the VIEW axis, + = the image's right side goes DOWN.

Because pitch/roll/yaw are an exact decomposition of the rendered camera's rotation, the
projection is exact (the self-test checks it against a full 3-D pinhole built with
locomotion.camera_mount_matrix and random head poses: < 1e-6 px). Two things this replaced:
`camera_pose()` used to report roll as asin(right_z), which is the NEGATIVE of the roll this
function consumes (passing it straight in doubled the error: 35-39 px at 2 deg), and a roll of the
head about its own x axis also yaws a pitched view, which a pitch+roll-only model cannot represent
(~13 px residual). Both are covered by the decomposition.

Lens position: `cam_x` forward and `cam_y` left of the rotation centre, `cam_h` above the floor.
Square pixels (fy == fx), image origin top-left, v grows downward, the floor is z = 0.

At pitch = roll = yaw = 0 and cam_y = 0 it is exactly arc_menu.project():
    u = W/2 - fx * y / d        v = H/2 + fx * cam_h / d          (d = x - cam_x)

    python3 nav/sim/g1/camera_geometry.py     # self-test + the arc visibility table
"""

from __future__ import annotations

import math

# --- the camera (asset_spec.json nav_camera; optics are the D435i colour sensor's) -------------
W, H = 1920, 1080
HFOV_DEG = 69.0
VFOV_DEG = 42.273
FX = (W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)        # 1396.8 px, fy == fx
MOUNT_PITCH_DEG = 12.0                                         # asset_spec: re-pitched from the real 47.6
REAL_MOUNT_PITCH_DEG = 47.6

# --- measured while walking (flat_v3 + hospital_v2 + route runs; see the integration plan) ------
STAND_CAM_HEIGHT_M, STAND_PITCH_DEG = 1.093, 12.7
WALK_CAM_HEIGHT_M, WALK_PITCH_DEG, WALK_CAM_FORWARD_M = 1.089, 15.3, 0.074
WALK_CAM_LEFT_M = 0.02                                         # lens 0.006-0.031 m left of the pelvis

# A level USD camera looking down +x: columns are the camera's x (image right) = -y, its y
# (image up) = +z, its z (backwards) = -x. Same matrix as locomotion.camera_mount_matrix(0).
M_LEVEL = ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------- 3x3 helpers
def _mm(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _mt(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _mv(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def _rx(t):
    c, s = math.cos(t), math.sin(t)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _ry(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def camera_rotation(pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0):
    """The camera's rotation in the heading frame (columns = USD camera axes), module convention."""
    return _mm(_mm(_mm(_rz(math.radians(yaw_deg)), _ry(math.radians(pitch_deg))), _rx(math.radians(roll_deg))), M_LEVEL)


def camera_angles(r_cam_heading) -> tuple[float, float, float]:
    """Inverse of camera_rotation(): a USD camera rotation EXPRESSED IN THE HEADING FRAME (any 3x3
    sequence; columns = camera x/y/z) -> (pitch_deg, roll_deg, yaw_deg) in this module's convention.
    G1Locomotion.camera_pose() passes Rz(-pelvis_yaw) @ R_cam_world."""
    r = tuple(tuple(float(r_cam_heading[i][j]) for j in range(3)) for i in range(3))
    b = _mm(r, _mt(M_LEVEL))                                     # = Rz(yaw) Ry(pitch) Rx(roll)
    pitch = math.asin(max(-1.0, min(1.0, -b[2][0])))
    roll = math.atan2(b[2][1], b[2][2])
    yaw = math.atan2(b[1][0], b[0][0])
    return math.degrees(pitch), math.degrees(roll), math.degrees(yaw)


# ---------------------------------------------------------------------------- projection
def project_ground(x_forward: float, y_left: float, cam_h: float = WALK_CAM_HEIGHT_M,
                   cam_x: float = WALK_CAM_FORWARD_M, pitch_deg: float = WALK_PITCH_DEG,
                   roll_deg: float = 0.0, w: int = W, h: int = H, fov_deg: float = HFOV_DEG,
                   cam_y: float = 0.0, yaw_deg: float = 0.0) -> tuple[float, float] | None:
    """Floor point in the heading frame (origin under the rotation centre) -> pixel (u, v); None at
    or behind 5 cm of depth. Pose arguments in the module convention (module docstring)."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    r = camera_rotation(pitch_deg, roll_deg, yaw_deg)
    xc, yc, zc = _mv(_mt(r), (x_forward - cam_x, y_left - cam_y, -cam_h))
    depth = -zc
    if depth <= 0.05:
        return None
    return w / 2.0 + fx * xc / depth, h / 2.0 - fx * yc / depth


def unproject_floor(u: float, v: float, cam_h: float = WALK_CAM_HEIGHT_M, cam_x: float = WALK_CAM_FORWARD_M,
                    pitch_deg: float = WALK_PITCH_DEG, w: int = W, h: int = H, fov_deg: float = HFOV_DEG,
                    roll_deg: float = 0.0, cam_y: float = 0.0, yaw_deg: float = 0.0) -> tuple[float, float] | None:
    """Pixel -> floor point (x_forward, y_left) in the heading frame; None at or above the horizon."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    ray_cam = ((u - w / 2.0) / fx, -(v - h / 2.0) / fx, -1.0)
    dx, dy, dz = _mv(camera_rotation(pitch_deg, roll_deg, yaw_deg), ray_cam)
    if dz >= -1e-9:
        return None
    t = cam_h / -dz
    return cam_x + t * dx, cam_y + t * dy


def horizon_row(pitch_deg: float, h: int = H, fov_deg: float = HFOV_DEG, w: int = W) -> float:
    """Image row of the horizon at the image centre column (zero roll)."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return h / 2.0 - fx * math.tan(math.radians(pitch_deg))


def nearest_floor_ahead(cam_h: float, pitch_deg: float, cam_x: float = 0.0, h: int = H,
                        fov_deg: float = HFOV_DEG, w: int = W) -> float:
    """Forward distance (from the rotation centre) of the first floor row at the image bottom."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    ang = math.radians(pitch_deg) + math.atan((h / 2.0) / fx)
    return cam_h / math.tan(ang) + cam_x


def arc_xy(kappa: float, s: float) -> tuple[float, float]:
    """Point at arc length s on a constant-curvature path (left turn = kappa > 0)."""
    if abs(kappa) < 1e-9:
        return s, 0.0
    return math.sin(kappa * s) / kappa, (1.0 - math.cos(kappa * s)) / kappa


def visible_arc(kappa: float, length: float = 3.0, n: int = 3000, **cam) -> dict:
    """Which stretch of a menu arc lands inside the frame: visible length, first/last visible s, end pixel."""
    w, h = cam.get("w", W), cam.get("h", H)
    vis_s = []
    for i in range(n + 1):
        s = length * i / n
        p = project_ground(*arc_xy(kappa, s), **cam)
        if p is not None and 0 <= p[0] < w and 0 <= p[1] < h:
            vis_s.append(s)
    end = project_ground(*arc_xy(kappa, length), **cam)
    return {"kappa": kappa, "visible_m": round(len(vis_s) * length / n, 2),
            "first_s": None if not vis_s else round(vis_s[0], 2), "last_s": None if not vis_s else round(vis_s[-1], 2),
            "end_px": None if end is None else (round(end[0]), round(end[1]))}


# ---------------------------------------------------------------------------- live-pose API for the menu
# The arc menu MUST be drawn with the pose of the frame it is drawn on. g1/check (2026-10-05) measured the
# module's WALK_* constants against the live pose over 64 walking frames: 253 px worst error (head yaw
# -3.8..+6.7 deg against the pelvis, lens 0.008-0.034 m left of it); the live pose is exact to 0.0027 px.
# So the integration path is: G1Locomotion.camera_pose(spec)["menu_args"] read in the same update the
# frame is grabbed in, sent WITH the frame, and passed here. These functions have no defaults to fall back
# on: a missing key raises.
MENU_ARG_KEYS = ("cam_h", "cam_x", "cam_y", "pitch_deg", "roll_deg", "yaw_deg")


def _live(menu_args: dict) -> dict:
    missing = [k for k in MENU_ARG_KEYS if k not in menu_args or menu_args[k] is None]
    if missing:
        raise KeyError(f"menu_args is missing {missing}: the G1 menu must be drawn from the live camera pose "
                       f"(G1Locomotion.camera_pose()['menu_args']), never from module constants")
    return {k: float(menu_args[k]) for k in MENU_ARG_KEYS}


def project_ground_live(x_forward: float, y_left: float, menu_args: dict, **optics) -> tuple[float, float] | None:
    """project_ground() with the pose taken ONLY from `menu_args` (all six keys required)."""
    return project_ground(x_forward, y_left, **_live(menu_args), **optics)


def arc_badge(kappa: float, menu_args: dict, length: float = 3.0, min_visible_m: float = 0.5, n: int = 600,
              margin_px: int = 40, **optics) -> dict:
    """Where this frame can show a menu arc: its visible stretch and the pixel of the LAST visible point
    (the badge anchor -- the arc's end is out of frame for hard turns at this camera's pitch), or
    keep=False when less than `min_visible_m` of it is in frame (drop it from this frame's menu rather
    than offer the VLM a choice it cannot see). Measured for the G1 walking pose (15.3 deg, 1.089 m):
    kappa 0.60 shows 0.17 m of a 3 m arc, 0.35 shows 1.36 m; the first 1.55 m of every arc is below
    the frame."""
    cam = _live(menu_args)
    w, h = optics.get("w", W), optics.get("h", H)
    vis = []
    for i in range(n + 1):
        s_ = length * i / n
        p = project_ground(*arc_xy(kappa, s_), **cam, **optics)
        if p is not None and margin_px <= p[0] < w - margin_px and margin_px <= p[1] < h - margin_px:
            vis.append((s_, p))
    visible_m = len(vis) * length / n
    if not vis:
        return {"kappa": kappa, "keep": False, "visible_m": 0.0, "first_s": None, "last_s": None, "badge_px": None}
    return {"kappa": kappa, "keep": visible_m >= min_visible_m, "visible_m": round(visible_m, 3),
            "first_s": round(vis[0][0], 3), "last_s": round(vis[-1][0], 3),
            "badge_px": (round(vis[-1][1][0], 1), round(vis[-1][1][1], 1))}


# ---------------------------------------------------------------------------- self-test
def _pinhole_reference(head_rot, head_pos, pelvis_xy, pelvis_yaw, mount_local, mount_pitch_deg, pt_world,
                       w=W, h=H, fov_deg=HFOV_DEG):
    """Full 3-D pinhole, no shortcuts: a camera built exactly the way locomotion builds it (head link
    rotation @ camera_mount_matrix, eye = head pos + R_head @ mount offset), world point -> pixel."""
    import numpy as np

    import locomotion as L

    r_cam = np.asarray(head_rot) @ L.camera_mount_matrix(math.radians(mount_pitch_deg))
    eye = np.asarray(head_pos) + np.asarray(head_rot) @ np.asarray(mount_local)
    pc = r_cam.T @ (np.asarray(pt_world) - eye)
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return (w / 2.0 + fx * pc[0] / -pc[2], h / 2.0 - fx * pc[1] / -pc[2]), r_cam, eye


def _self_test() -> None:
    # (1) zero pose is arc_menu.project() (AlohaMini optics)
    for d, y in ((1.0, 0.0), (2.5, -0.7), (6.0, 1.2)):
        u, v = project_ground(d + 0.25, y, cam_h=0.354, cam_x=0.25, pitch_deg=0.0, w=1920, h=1080, fov_deg=90.1)
        fx = 960 / math.tan(math.radians(90.1) / 2)
        assert abs(u - (960 - fx * y / d)) < 1e-6 and abs(v - (540 + fx * 0.354 / d)) < 1e-6, (d, y, u, v)
    # (2) the closed form the asset spec quotes, v = H/2 + fx tan(atan2(h, d) - pitch), zero roll/yaw
    for d in (1.8, 3.0, 7.0):
        _, v = project_ground(d + 0.074, 0.0, cam_h=1.089, cam_x=0.074, pitch_deg=15.3)
        assert abs(v - (H / 2 + FX * math.tan(math.atan2(1.089, d) - math.radians(15.3)))) < 1e-6
    # (3) round trip, every argument nonzero
    cam = dict(cam_h=1.09, cam_x=0.07, cam_y=0.02, pitch_deg=15.3, roll_deg=-1.7, yaw_deg=4.2)
    for x, y in ((2.0, 0.0), (3.0, -0.8), (5.0, 1.5)):
        p = project_ground(x, y, **cam)
        q = unproject_floor(*p, **cam)
        assert abs(q[0] - x) < 1e-6 and abs(q[1] - y) < 1e-6, (x, y, q)
    # (4) the asset spec's own numbers at the nominal mount: horizon row 243, nearest floor ~1.81 m ahead of the lens
    assert abs(horizon_row(12.0) - 243.0) < 1.0, horizon_row(12.0)
    assert abs(nearest_floor_ahead(1.184, 12.0) - 1.81) < 0.06
    # (5) against a full 3-D pinhole built with locomotion.camera_mount_matrix and a perturbed head:
    #     head roll / pitch / yaw relative to the pelvis, lens off-centre, arbitrary world yaw
    import random

    import numpy as np

    try:
        import locomotion as L
    except ImportError:  # run from the repo root
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import locomotion as L

    def rz(t):
        return np.array(_rz(t))

    rng = random.Random(7)
    worst, worst_old = 0.0, 0.0
    mount = (0.0482, 0.0175, 0.4629)
    for _ in range(200):
        pyaw = rng.uniform(-math.pi, math.pi)
        dyaw, dpitch, droll = (math.radians(rng.uniform(-8, 8)), math.radians(rng.uniform(-4, 4)),
                               math.radians(rng.uniform(-4, 4)))
        r_head = rz(pyaw + dyaw) @ np.array(_ry(dpitch)) @ np.array(_rx(droll))
        pelvis = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), 0.63])
        head_pos = pelvis + rz(pyaw) @ np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.0])
        floor_z = 0.0
        r_cam = r_head @ L.camera_mount_matrix(math.radians(12.0))
        eye = head_pos + r_head @ np.array(mount)
        r_h = rz(-pyaw) @ r_cam
        pitch, roll, yaw = camera_angles(r_h)
        off = rz(-pyaw) @ (eye - pelvis)
        right_z_roll = math.degrees(math.asin(max(-1.0, min(1.0, r_cam[2, 0]))))   # the OLD camera_pose() roll
        for _k in range(5):
            xf, yl = rng.uniform(2.0, 8.0), rng.uniform(-2.0, 2.0)
            pw = np.array([pelvis[0], pelvis[1], floor_z]) + rz(pyaw) @ np.array([xf, yl, 0.0])
            ref, _, _ = _pinhole_reference(r_head, head_pos, pelvis[:2], pyaw, mount, 12.0, pw)
            if not (0 <= ref[0] < W and 0 <= ref[1] < H):
                continue
            got = project_ground(xf, yl, cam_h=float(eye[2] - floor_z), cam_x=float(off[0]), cam_y=float(off[1]),
                                 pitch_deg=pitch, roll_deg=roll, yaw_deg=yaw)
            worst = max(worst, abs(got[0] - ref[0]), abs(got[1] - ref[1]))
            old = project_ground(xf, yl, cam_h=float(eye[2] - floor_z), cam_x=float(off[0]), cam_y=float(off[1]),
                                 pitch_deg=pitch, roll_deg=right_z_roll, yaw_deg=yaw)   # ONLY the roll sign differs
            if old is not None:
                worst_old = max(worst_old, abs(old[0] - ref[0]), abs(old[1] - ref[1]))
    assert worst < 1e-6, worst
    assert worst_old > 5.0, worst_old            # the test can tell the old convention from the new one
    print(f"camera_geometry self-test: ok (3-D pinhole max err {worst:.2e} px; same call with the OLD "
          f"camera_pose() roll, asin(right_z): {worst_old:.1f} px)")


if __name__ == "__main__":
    _self_test()
    for label, cam in (("nominal 12.0 deg, 1.184 m", dict(cam_h=1.184, cam_x=0.0485, pitch_deg=12.0)),
                       ("walking mean 15.3 deg, 1.089 m", dict()),
                       ("walking 17.3 deg", dict(pitch_deg=17.3))):
        print(f"{label}: horizon row {horizon_row(cam.get('pitch_deg', WALK_PITCH_DEG)):.0f}, "
              f"first floor row of the frame {nearest_floor_ahead(cam.get('cam_h', WALK_CAM_HEIGHT_M), cam.get('pitch_deg', WALK_PITCH_DEG), cam.get('cam_x', WALK_CAM_FORWARD_M)):.2f} m ahead")
        for k in (0.0, 0.15, 0.35, 0.60):
            print("   ", visible_arc(k, **cam))
