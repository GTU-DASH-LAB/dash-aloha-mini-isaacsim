"""Compose the working scene: a ready-made Isaac Sim environment (for lighting/ground)
plus the imported AlohaMini1 robot, referenced on top of it. Saves the result so later
scripts (physics config, control) can just open scene.usda directly.

Usage:
    ~/isaacsim/python.sh scripts/build_scene.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/Aloha/Aloha.usda")
parser.add_argument("--out", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--screenshot", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/docs/scene_verification.png")
parser.add_argument(
    "--environment-url",
    default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Simple_Room/simple_room.usd",
)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Sdf  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.new_stage()
stage = usd_context.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Reference the ready-made environment at the stage root
env_prim = stage.DefinePrim("/World/Environment", "Xform")
env_prim.GetReferences().AddReference(args.environment_url)

# Reference our robot on top of it
robot_prim = stage.DefinePrim("/World/Aloha", "Xform")
robot_prim.GetReferences().AddReference(args.robot_usd)
UsdGeom.XformCommonAPI(robot_prim).SetTranslate((0.0, 0.0, 0.0))

stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
if not stage.GetPrimAtPath("/World").IsValid():
    stage.DefinePrim("/World", "Xform")

stage.Export(args.out)
print(f"Wrote composed scene: {args.out}")

# Reopen the exported scene (fresh) to verify it stands on its own, then screenshot
usd_context.open_stage(args.out)
stage = usd_context.get_stage()
for _ in range(90):
    kit.update()
stage.Load()
for _ in range(20):
    kit.update()

from omni.kit.viewport.utility import get_active_viewport, frame_viewport_prims  # noqa: E402
import omni.kit.viewport.utility as vp_utility  # noqa: E402

viewport = get_active_viewport()
frame_viewport_prims(viewport, prims=["/World/Aloha"])
for _ in range(15):
    kit.update()

vp_utility.capture_viewport_to_file(viewport, args.screenshot)
for _ in range(10):
    kit.update()
print(f"Screenshot saved to: {args.screenshot}")

kit.close()
