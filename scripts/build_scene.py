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
    default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Office/office.usd",
)
parser.add_argument(
    "--pick-place-props", action=argparse.BooleanOptionalAction, default=True,
    help="Add a small physics-enabled table + graspable cubes near the robot for pick-and-place testing",
)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Sdf, UsdPhysics, Gf  # noqa: E402

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

if args.pick_place_props:
    # Small physics-enabled table + graspable cubes near the robot, positioned within
    # the arms' *actual* reach envelope. This was NOT guessed -- an earlier attempt
    # placed it at X=+0.3..0.35, Z=0.45 (assuming a ~0.3-0.4m horizontal reach like a
    # human arm) and empirical joint sweeps with proper drives showed the real
    # envelope is much smaller and closer to the base: X=-0.07..+0.03, Y=-0.09..-0.20
    # for the right arm (Z=0.87..1.05) -- mirrored in Y for the left arm. This robot's
    # arms are short and mounted high on the lift column, nothing like human proportions.
    table_top_z = 0.80
    table_prim = stage.DefinePrim("/World/PickPlaceTable", "Cube")
    table_geom = UsdGeom.Cube(table_prim)
    table_geom.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(table_prim).SetScale((0.15, 0.25, 0.02))
    UsdGeom.XformCommonAPI(table_prim).SetTranslate((-0.03, 0.0, table_top_z - 0.02))
    UsdPhysics.CollisionAPI.Apply(table_prim)  # static -- no RigidBodyAPI, doesn't move

    # One cube in each arm's confirmed reach zone (right: -Y, left: +Y, mirrored)
    cube_positions = [(-0.03, -0.15, table_top_z + 0.02), (-0.03, 0.15, table_top_z + 0.02)]
    cube_colors = [(0.8, 0.2, 0.2), (0.2, 0.4, 0.8)]
    for i, (pos, color) in enumerate(zip(cube_positions, cube_colors), start=1):
        cube_path = f"/World/PickCube{i}"
        cube_prim = stage.DefinePrim(cube_path, "Cube")
        cube_geom = UsdGeom.Cube(cube_prim)
        cube_geom.CreateSizeAttr(1.0)
        cube_geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.XformCommonAPI(cube_prim).SetScale((0.02, 0.02, 0.02))  # 4cm cube
        UsdGeom.XformCommonAPI(cube_prim).SetTranslate(pos)
        UsdPhysics.CollisionAPI.Apply(cube_prim)
        UsdPhysics.RigidBodyAPI.Apply(cube_prim)
        mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
        mass_api.CreateMassAttr(0.02)  # 20g -- light enough for the small gripper

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
