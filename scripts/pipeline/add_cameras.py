"""Author the robot-mounted USD cameras onto the composed scene, matching the
OFFICIAL AlohaMini camera set from the LeRobot integration (see
third_party/lerobot_alohamini/src/lerobot/robots/alohamini/config_alohamini.py):

- "forward"      -- head camera on top of the lift column, facing the robot's
                    manipulation front (-Y: both grippers work toward -Y, verified by
                    probing the wrist link frames/bboxes at rest, NOT assumed from the
                    driving direction which is +X).
- "wrist_left"   -- on left_link5 (the gripper body after Wrist_Roll; link6 is the
                    MOVING jaw finger, wrong mount point), looking along the gripper.
- "wrist_right"  -- mirror of wrist_left on right_link5.

The cameras are children of the robot links, so they move with the wrist/lift like
the real ones. Intrinsics approximate a typical 640x480 USB webcam (~78 deg HFOV).

Runs as part of scripts/rebuild_all.sh (build_scene.py recreates scene.usda from
scratch, wiping these -- same reason configure_physics.py must re-run).

Usage:
    ~/isaacsim/python.sh scripts/pipeline/add_cameras.py
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/ root, for alohamini1_specs
from alohamini1_specs import (  # noqa: E402
    CAMERA_PRIM_PATHS,
    CHASE_CAMERA_PRIM_PATH,
    NAV_CAMERA_APERTURE,
    NAV_CAMERA_FOCAL_MM,
    NAV_CAMERA_HEIGHT_M,
    NAV_CAMERA_PRIM_PATH,
    NAV_HEADLIGHT_INTENSITY,
    NAV_HEADLIGHT_PRIM_PATH,
    NAV_HEADLIGHT_RADIUS_M,
)

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import UsdGeom, UsdLux  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(args.scene)
stage = usd_context.get_stage()
for _ in range(60):
    kit.update()
stage.Load()
for _ in range(10):
    kit.update()

# Local (parent-link-frame) mount poses. At the rest pose every link frame happens to
# be world-aligned (verified by probing ExtractRotationMatrix on each link), which is
# what made these offsets straightforward to derive from world-space measurements.
#
# Rotation cheat sheet for this specific rotateXYZ=(x, 0, 180) family (USD camera:
# -Z is the view direction, +Y is image-up; XformCommonAPI applies R = Rz*Ry*Rx):
#   view direction = (0, -sin x, -cos x),  image-up = (0, -cos x, sin x)
# The z=180 term is what keeps image-up pointing sensibly (world +Z-ish) -- without
# it the first render came out upside down AND, on the forward camera, the naive
# (-70, 0, 180) combination flipped the view to +Y entirely (it was staring at the
# rear table, upside down -- caught by rendering, not by eyeballing the math).
#
# - Wrist (x=157): view = (0, -0.39, 0.92), exactly the gripper axis at rest (link5
#   origin -> link6 finger center, measured). Camera sits behind/above the wrist;
#   fingers appear at the bottom of the frame, workspace beyond.
# - Forward (x=60): view = (0, -0.87, -0.5) = the manipulation front (-Y, both
#   grippers work toward -Y -- measured, NOT the +X driving direction), tilted 30 deg
#   down at the front table's work surface. MUST stay in front of the column's own
#   front face (world y<=-0.31): a nicer-sounding "above and behind the column top"
#   position is geometrically occluded -- any downward ray from back there dips below
#   the column's top front corner before clearing it (checked, not guessed). Note the
#   front table's two blocks span wider (+-0.25m at ~0.3m range) than the 78-deg HFOV
#   -- they sit at the frame edges by design; this is a context/navigation view, the
#   wrist cameras are the manipulation views.
CAMERA_SPECS = {
    "forward": {
        "path": CAMERA_PRIM_PATHS["forward"],
        # Raised well above the column top (world z~=1.21 vs table surface 0.994) --
        # at lower heights the camera is nearly level with the table and the near
        # surface fills the whole frame (verified by render).
        "translate": (0.0, -0.25, 1.15),
        "rotateXYZ": (55.0, 0.0, 180.0),
    },
    "wrist_left": {
        "path": CAMERA_PRIM_PATHS["wrist_left"],
        "translate": (0.0, 0.035, 0.01),
        "rotateXYZ": (157.0, 0.0, 180.0),
    },
    "wrist_right": {
        "path": CAMERA_PRIM_PATHS["wrist_right"],
        "translate": (0.0, 0.035, 0.01),
        "rotateXYZ": (157.0, 0.0, 180.0),
    },
    # --- Navigation camera: the only one that faces the DRIVING direction ---
    # Every camera above faces the manipulation front (-Y). The base drives +X. That
    # mismatch is fine for manipulation and fatal for navigation, so nav/ gets its own
    # camera rather than being handed a sideways view. Kept out of CAMERA_PRIM_PATHS
    # so the LeRobot observation contract is unchanged (see alohamini1_specs.py).
    #
    # THIS CAMERA IS A COPY OF NOVA CARTER'S FRONT HAWK, NOT A CHOICE. TIC-VLA is
    # trained through DynaNav's render of nova_carter_sensors.usd's
    # /chassis_link/front_hawk/left/camera_left; the intrinsics and the mount height in
    # alohamini1_specs.py were probed off that asset. Deviating from them moves the
    # input off the model's training distribution, which is a silent accuracy loss, not
    # a stylistic difference. The first version of this file used AlohaMini's own
    # webcam intrinsics (78 deg HFOV) at 1.15 m tilted 10 deg down; the policy could
    # not find the landmarks the prompts name, because a 90 deg view cropped to 78 deg
    # and pitched at the floor loses exactly the peripheral and distant context that
    # "the second aisle from the right" is expressed in.
    #
    # Rotation, using the same R = Rz*Ry*Rx convention as the cheat sheet above:
    #   after Rx(x):   view = (0, sin x, -cos x)      up = (0, cos x, sin x)
    #   after Rz(-90): view = (sin x, 0, -cos x)      up = (cos x, 0, sin x)
    # x=90 gives view = (1, 0, 0) and up = (0, 0, 1): straight down the driving
    # direction, dead level, image-up = world-up. Level is the point -- the Hawk sits
    # at pitch 0.0, measured, not assumed.
    "nav": {
        "path": NAV_CAMERA_PRIM_PATH,
        # x=+0.25 clears the base's own front face (half-extent ~0.21 in X, the same
        # number behind the 0.375 m turning swing radius in CLAUDE.md), so the chassis
        # does not occlude the lower frame. z from NAV_CAMERA_HEIGHT_M = Nova Carter's
        # own Hawk height. y=0: the Hawk's 0.075 offset is just the left eye of a
        # stereo pair, and the policy is fed one image.
        "translate": (0.25, 0.0, NAV_CAMERA_HEIGHT_M),
        "rotateXYZ": (90.0, 0.0, -90.0),
        "focal": NAV_CAMERA_FOCAL_MM,
        "aperture": NAV_CAMERA_APERTURE,
    },
    # --- Third-person chase camera: for WATCHING, never fed to the policy ---
    # Parented to base_link, so it inherits the base's position and yaw and stays
    # behind the robot as it turns, rather than being a fixed world camera the robot
    # drives away from. base_link rather than vertical_link on purpose: mounted on the
    # column it would ride up and down with the lift.
    #
    # Geometry: sitting 2.5 m behind and 1.8 m above the base, the robot's centre of
    # mass (~0.5 m up) sits atan(1.3 / 2.5) = 27.5 deg below the horizon. Using the
    # same (x, 0, -90) family as the nav camera, view = (sin x, 0, -cos x) is tilted
    # (90 - x) below horizontal, so x = 62.5 puts the robot squarely in frame with
    # room ahead of it to see where it is going.
    "chase": {
        "path": CHASE_CAMERA_PRIM_PATH,
        "translate": (-2.5, 0.0, 1.8),
        "rotateXYZ": (62.5, 0.0, -90.0),
    },
}

# --- Remove any camera of ours that is sitting somewhere it no longer belongs ---
# Defining a prim at the new path does NOT delete the old one. When camera_nav moved
# from the lift column to base_link, a plain re-run left BOTH prims in the stage: the
# code read the new one, Kit went on rendering the old one every frame, and a stale
# camera parented to the lift is exactly the kind of thing that looks fine in a diff
# and wrong in a frame. The invariant this enforces is "a camera we author exists at
# the path the spec names, and nowhere else" -- so moving a mount point is a one-line
# spec change from here on, not a spec change plus a manual USD cleanup someone has to
# remember. Only OUR camera names are touched; anything the environment asset brings
# with it is left alone.
_authored_paths = {spec["path"] for spec in CAMERA_SPECS.values()} | {NAV_HEADLIGHT_PRIM_PATH}
_authored_names = {p.rsplit("/", 1)[1] for p in _authored_paths}
for prim in list(stage.Traverse()):
    path = prim.GetPath().pathString
    if prim.GetName() in _authored_names and path not in _authored_paths:
        stage.RemovePrim(prim.GetPath())
        print(f"Removed stale camera prim: {path}")

for name, spec in CAMERA_SPECS.items():
    parent_path = spec["path"].rsplit("/", 1)[0]
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        raise RuntimeError(f"Camera parent link not found: {parent_path}")
    if parent.IsInstance():
        # Same pattern as fix_wheel_collision.py's wheels: children can't be added
        # under an instance -- un-instance this one copy first.
        parent.SetInstanceable(False)
        print(f"Un-instanced camera parent: {parent_path}")

    # Default intrinsics approximate a 640x480 USB webcam (~78 deg HFOV), which is what
    # the LeRobot cameras actually are. A spec can override them -- the nav camera does,
    # because it has to match the sensor TIC-VLA was trained through rather than
    # AlohaMini's own hardware.
    focal = spec.get("focal", 13.0)
    h_ap, v_ap = spec.get("aperture", (20.955, 15.716))

    cam = UsdGeom.Camera.Define(stage, spec["path"])
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(float(h_ap))
    cam.CreateVerticalApertureAttr(float(v_ap))
    cam.CreateClippingRangeAttr((0.01, 100.0))
    xform = UsdGeom.XformCommonAPI(cam.GetPrim())
    xform.SetTranslate(spec["translate"])
    xform.SetRotate(spec["rotateXYZ"])
    hfov = 2 * math.degrees(math.atan(h_ap / (2 * focal)))
    print(f"Authored camera '{name}' at {spec['path']}  (HFOV {hfov:.1f} deg)")

# --- Headlight, co-mounted with camera_nav ---
# DynaNav's own environments cannot be trusted to light themselves -- see the long
# comment in alohamini1_specs.py next to NAV_HEADLIGHT_INTENSITY for the measurements
# behind these numbers. Reuses the nav camera's own translate/rotateXYZ verbatim: a
# UsdLux light emits along local -Z exactly like a UsdGeom.Camera looks down local -Z,
# so whatever rotation points the camera down +X points this light the same way, no
# separate derivation needed.
nav_spec = CAMERA_SPECS["nav"]
headlight = UsdLux.DiskLight.Define(stage, NAV_HEADLIGHT_PRIM_PATH)
headlight.CreateIntensityAttr(NAV_HEADLIGHT_INTENSITY)
headlight.CreateRadiusAttr(NAV_HEADLIGHT_RADIUS_M)
headlight.CreateColorAttr((1.0, 0.98, 0.92))
headlight_xform = UsdGeom.XformCommonAPI(headlight.GetPrim())
headlight_xform.SetTranslate(nav_spec["translate"])
headlight_xform.SetRotate(nav_spec["rotateXYZ"])
print(f"Authored headlight at {NAV_HEADLIGHT_PRIM_PATH} (co-aimed with camera_nav)")

stage.Save()
print(f"\nSaved: {args.scene}")

kit.close()
