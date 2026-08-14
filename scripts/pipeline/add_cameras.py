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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/ root, for alohamini1_specs
from alohamini1_specs import (  # noqa: E402
    CAMERA_PRIM_PATHS,
    CHASE_CAMERA_PRIM_PATH,
    NAV_CAMERA_PRIM_PATH,
)

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

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
    # This one is NOT in the (x, 0, 180) family the cheat sheet above covers, so the
    # derivation, using the same R = Rz*Ry*Rx convention:
    #   after Rx(x):   view = (0, sin x, -cos x)      up = (0, cos x, sin x)
    #   after Rz(-90): view = (sin x, 0, -cos x)      up = (cos x, 0, sin x)
    # so x=80 gives view = (0.985, 0, -0.174): straight down +X, tilted 10 deg down,
    # with up = (0.174, 0, 0.985) -- image-up still world-up, i.e. not upside down.
    # 10 deg down rather than level so the floor immediately ahead is in frame (that
    # is where an obstacle the robot is about to hit actually appears) while keeping
    # the horizon visible for the distant landmarks the benchmark prompts name --
    # "the red emergency exit door", "the wall with the words 'Northside Branch
    # Library'". Tilt much further down and those leave the frame entirely.
    "nav": {
        "path": NAV_CAMERA_PRIM_PATH,
        # x=+0.25 clears the base's own front face (half-extent ~0.21 in X, the same
        # number behind the 0.375 m turning swing radius in CLAUDE.md), so the chassis
        # does not occlude the lower frame. z=1.15 matches the forward camera's height
        # on the column, which also puts it in the same ballpark as Nova Carter's
        # camera height -- the robot TIC-VLA was trained on.
        "translate": (0.25, 0.0, 1.15),
        "rotateXYZ": (80.0, 0.0, -90.0),
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

    cam = UsdGeom.Camera.Define(stage, spec["path"])
    cam.CreateFocalLengthAttr(13.0)          # ~78 deg HFOV at 20.955mm aperture
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(15.716)   # 4:3, matches 640x480
    cam.CreateClippingRangeAttr((0.01, 100.0))
    xform = UsdGeom.XformCommonAPI(cam.GetPrim())
    xform.SetTranslate(spec["translate"])
    xform.SetRotate(spec["rotateXYZ"])
    print(f"Authored camera '{name}' at {spec['path']}")

stage.Save()
print(f"\nSaved: {args.scene}")

kit.close()
