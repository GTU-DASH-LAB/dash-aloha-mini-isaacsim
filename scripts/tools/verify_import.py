"""Phase 2 verification: open the imported Aloha.usda, sanity-check the articulation
tree, and capture a viewport screenshot for visual inspection.

Usage:
    ~/isaacsim/python.sh scripts/tools/verify_import.py
"""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/Aloha/Aloha.usda")
parser.add_argument("--screenshot", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/docs/import_verification.png")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Usd, UsdPhysics, UsdGeom  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(args.usd)
stage = usd_context.get_stage()

# open_stage / payload loading is asynchronous in Kit -- let it settle before traversing
for _ in range(60):
    kit.update()
stage.Load()  # force-load any unloaded payloads
for _ in range(10):
    kit.update()

print("=== Prim tree summary ===")
mesh_count = 0
joint_count = 0
art_roots = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh_count += 1
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        joint_count += 1
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_roots.append(str(prim.GetPath()))

print(f"Mesh prims: {mesh_count}")
print(f"Revolute/Prismatic joints: {joint_count}")
print(f"Articulation roots: {art_roots}")

# Step physics a few frames to make sure nothing throws / explodes immediately
for _ in range(30):
    kit.update()

# Compute the robot's world-space bounding box just for logging/sanity, then use the
# built-in frame_viewport_prims helper to actually aim the camera at it -- the default
# viewport camera does not auto-frame newly loaded geometry.
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
robot_prim = stage.GetPrimAtPath("/Aloha")
bbox = bbox_cache.ComputeWorldBound(robot_prim)
rng = bbox.ComputeAlignedRange()
print(f"Robot world bbox: min={rng.GetMin()} max={rng.GetMax()}")

from omni.kit.viewport.utility import get_active_viewport, frame_viewport_prims  # noqa: E402
import omni.kit.viewport.utility as vp_utility  # noqa: E402

viewport = get_active_viewport()
frame_viewport_prims(viewport, prims=["/Aloha"])
for _ in range(10):
    kit.update()
    time.sleep(0.1)

capture = vp_utility.capture_viewport_to_file(viewport, args.screenshot)
for _ in range(10):
    kit.update()

print(f"Screenshot saved to: {args.screenshot}")

kit.close()
