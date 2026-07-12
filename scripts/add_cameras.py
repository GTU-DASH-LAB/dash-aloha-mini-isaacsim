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
    ~/isaacsim/python.sh scripts/add_cameras.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alohamini1_specs import CAMERA_PRIM_PATHS  # noqa: E402

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
