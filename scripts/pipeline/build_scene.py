"""Compose the working scene: a ready-made Isaac Sim environment (for lighting/ground)
plus the imported AlohaMini1 robot, referenced on top of it. Saves the result so later
scripts (physics config, control) can just open scene.usda directly.

Usage:
    ~/isaacsim/python.sh scripts/pipeline/build_scene.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/Aloha/Aloha.usda")
parser.add_argument("--out", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--screenshot", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/docs/scene_verification.png")
parser.add_argument(
    "--environment-url",
    # NOT Office/office.usd -- that asset is a full multi-story building (world bbox
    # roughly -528..535m in X, -294..382m in Y -- verified via BBoxCache, not a simple
    # reception room), which caused two real, verified bugs: (1) the robot spawned
    # overlapping some part of that building's geometry and physics exploded it across
    # the room within the first ~30 physics steps (confirmed via a step-by-step
    # translate/rotation trace -- robot ends up ~7.7m away, tipped 181 degrees), and
    # (2) Isaac Sim's "frame all" on stage-open zooms out to fit the whole building,
    # which is what looked like "starting from over the building very far away".
    # NOT Simple_Room/simple_room.usd either -- it's physically stable, but its big
    # center table (table_low_327, 3.2x1.6m) sits at the origin with its top at
    # Z~=0.01 and the environment's invisible ground collision plane is at Z=0 (i.e.
    # AT table height; the visible wood floor is 78cm lower at Z~=-0.77, all verified
    # via BBoxCache). The robot unavoidably spawns standing on that table, and driving
    # off its edge leaves it hovering 77cm above the visible floor on the invisible
    # plane -- looks broken even though physics is fine.
    # Simple_Warehouse/warehouse.usd was verified clean on all counts: sane ~24x38x9m
    # single-room bbox, real floor at Z=0, and the same step-by-step trace shows the
    # robot sitting still at translate=(0,0,0.0078), rotation=0 for the full 270-step
    # test -- see docs/scene_verification.png.
    default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Simple_Warehouse/warehouse.usd",
)
parser.add_argument(
    "--pick-place-props", action=argparse.BooleanOptionalAction, default=True,
    help="Add two NVIDIA packing tables (real legs + collision physics) flanking the "
    "robot, with four official colored blocks (red/green/blue/yellow, full rigid-body "
    "physics) on their work surfaces",
)
args = parser.parse_args()

import os  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Sdf, UsdPhysics  # noqa: E402

# Author the stage DIRECTLY into the output layer, not via new_stage()+Export().
# stage.Export() writes the fully FLATTENED composition: every referenced asset's
# geometry gets baked inline as local "Flattened_Prototype_N" prims (confirmed: no
# reference arcs survive in the exported file). With the warehouse environment that
# produced a 233MB scene.usda, which GitHub rejected on push (100MB hard limit).
# Exporting the root layer instead keeps environment/robot/props as lightweight
# reference arcs -- the file stays a few hundred KB and the referenced assets load
# from the CDN / local paths at open time (Kit caches CDN fetches locally).
out_layer = Sdf.Layer.FindOrOpen(args.out)
if out_layer is None:
    out_layer = Sdf.Layer.CreateNew(args.out)
else:
    out_layer.Clear()
stage = Usd.Stage.Open(out_layer)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# Reference the ready-made environment at the stage root
env_prim = stage.DefinePrim("/World/Environment", "Xform")
env_prim.GetReferences().AddReference(args.environment_url)

# Reference our robot on top of it. Relative path, not absolute: since the root layer
# now IS the committed scene.usda (see above), an absolute reference would break for
# anyone cloning the repo to a different path. Relative asset paths anchor to the
# layer that authors them, i.e. to assets/usd/.
robot_ref = "./" + os.path.relpath(args.robot_usd, os.path.dirname(os.path.abspath(args.out)))
robot_prim = stage.DefinePrim("/World/Aloha", "Xform")
robot_prim.GetReferences().AddReference(robot_ref)
UsdGeom.XformCommonAPI(robot_prim).SetTranslate((0.0, 0.0, 0.0))

# Physics scene -- must be authored EXPLICITLY here. The old new_stage()+Export()
# flow silently inherited Isaac Sim's new-stage template, which injects a physics
# scene prim; the raw Usd.Stage authoring above starts truly empty. Without a
# UsdPhysics.Scene prim, pressing Play (GUI or timeline.play()) simulates NOTHING --
# verified: healthy rigid-body blocks (rigidBodyEnabled=True, kinematic=False,
# collision+mass all composed) hung frozen mid-air through 300 played steps.
# isaacsim.core-driven scripts like verify_physics.py masked the bug because
# SimulationManager bootstraps its own physics context regardless.
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr(9.81)

ISAAC_ASSET_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac"

if args.pick_place_props:
    # Two official NVIDIA warehouse packing tables (real prop with legs + collision
    # physics baked in) and four official colored blocks (RigidBody+Collision+Mass all
    # baked in, 4.7cm, meter-scale -- verified by referencing each candidate asset and
    # traversing for applied physics APIs, not assumed).
    #
    # Asset choice notes (all BBoxCache-measured, not guessed):
    # - packing_table.usd: 2.474 x 0.782 x 1.083m, floor pivot (zmin=0). The RAW
    #   SM_HeavyDutyPackingTable_C02_01_physics.usd variant is authored in CENTIMETERS
    #   and composes 247m wide in this meters stage -- use the assembled asset only.
    # - SeattleLabTable/table.usd rejected: pivot is 1.04m below its own geometry.
    #
    # Placement: the tables must NOT overlap the robot anywhere. Beyond being the
    # requested behavior, a static prim interpenetrating the articulation at t=0 makes
    # PhysX's separation impulse launch the robot across the room (verified with an
    # earlier slab-table attempt that overlapped the resting arms: robot ended up
    # ~13.5m away, tipped 252 degrees). Robot bbox: X -0.21..0.21, Y -0.31..0.15.
    # Tables sit at Y=+-0.85 with their long (2.474m/X) side facing the robot:
    # nearest table edge at |Y|=0.46. That clears not just the robot's bbox (0.31)
    # but also its in-place ROTATION swing radius, sqrt(0.21^2+0.31^2)=0.375 -- with
    # the edge any closer, turning the base in place clips a table corner. One table
    # per arm side (arms face +-Y); reaching either means driving the base.
    for name, y_center, rot_z in (("PickTable", 0.85, 0.0), ("PlaceTable", -0.85, 180.0)):
        t = stage.DefinePrim(f"/World/{name}", "Xform")
        t.GetReferences().AddReference(f"{ISAAC_ASSET_BASE}/Props/PackingTable/packing_table.usd")
        UsdGeom.XformCommonAPI(t).SetTranslate((0.0, y_center, 0.0))
        # rotate the Place table so its work surface faces the robot too
        UsdGeom.XformCommonAPI(t).SetRotate((0.0, 0.0, rot_z))

    # Block spawn height: ~3cm above the work surface; they drop into place within
    # ~30 physics steps (0.5s) and then sit still. The table's bbox zmax (1.083) is
    # the shelf FRAME, not the work surface, so it can't be used for placement.
    # Measured on the composed scene: block centers settle at Z=1.0173 and stay there
    # through 300 played steps => work surface at ~0.994m (block half-height 0.0235).
    # NOTE when verifying: PhysX writes simulated transforms to the prim carrying
    # RigidBodyAPI, which in these assets is the Cube MESH CHILD
    # (/World/Block*/Cube), not this wrapper Xform -- reading the wrapper's transform
    # shows it frozen at spawn forever even while the block is visibly falling.
    block_z = 1.05
    blocks = [
        ("BlockRed", "red_block", -0.25, 0.55),     # PickTable side (+Y, left arm)
        ("BlockGreen", "green_block", 0.25, 0.55),
        ("BlockBlue", "blue_block", -0.25, -0.55),  # PlaceTable side (-Y, right arm)
        ("BlockYellow", "yellow_block", 0.25, -0.55),
    ]
    for name, asset, bx, by in blocks:
        b = stage.DefinePrim(f"/World/{name}", "Xform")
        b.GetReferences().AddReference(f"{ISAAC_ASSET_BASE}/Props/Blocks/{asset}.usd")
        UsdGeom.XformCommonAPI(b).SetTranslate((bx, by, block_z))

# Bake a default viewport camera pose close to the robot+tables. Isaac Sim's "frame
# all" on stage-open otherwise zooms out to fit the *entire* referenced environment's
# bounding box -- harmless for a small room, but this is what produced the "scene
# starts from over the building very far away" bug with the old Office environment
# (world bbox spanned roughly 1000m). Same corner-isometric direction Kit uses for a
# brand new stage's default camera (rotateXYZ=(54.73561, 0, 135) looking toward the
# origin), just pulled in from the stock (5,5,5) to (3.2,3.2,2.4) so it frames this
# robot (~1.1m tall) and both packing tables (spanning X +-1.24, Y +-1.13) without
# needing "frame all" to kick in.
persp_prim = stage.DefinePrim("/OmniverseKit_Persp", "Camera")
persp_xform = UsdGeom.XformCommonAPI(persp_prim)
persp_xform.SetTranslate((3.2, 3.2, 2.4))
persp_xform.SetRotate((54.73561, 0.0, 135.0))

stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
if not stage.GetPrimAtPath("/World").IsValid():
    stage.DefinePrim("/World", "Xform")

out_layer.Save()
# Release the authoring stage's hold on the layer so the omni context below opens a
# fresh copy from disk rather than the in-memory one.
stage = None
print(f"Wrote composed scene: {args.out}")

# Reopen the exported scene (fresh) to verify it stands on its own, then screenshot
usd_context = omni.usd.get_context()
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
