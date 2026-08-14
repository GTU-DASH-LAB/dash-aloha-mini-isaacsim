"""Compose a navigation stage: a DynaNav benchmark environment + AlohaMini.

Run build_nav_scene.sh, not this file directly -- this only authors and verifies the
layer. Anything placed after SimulationApp.close() does not execute (the process is
already on its way down), so the pipeline steps that follow have to be chained from
outside. That is exactly why scripts/rebuild_all.sh exists too.

This deliberately does NOT touch assets/usd/scene.usda. That scene is the verified
pick-and-place setup (two packing tables, four blocks) and regenerating it is a
documented footgun -- ../../CLAUDE.md records that build_scene.py recreates it from
scratch and silently wipes whatever configure_physics / fix_wheel_collision /
add_cameras layered on top. A navigation run needs a different environment and no
tables at all, so it gets its own stage file and leaves that one alone.

The layering is the same three-step dance the main pipeline uses, and for the same
reason: a freshly authored root layer has NO joint drives, NO wheel colliders and NO
cameras -- those are authored as overrides into the *scene* file, not into
Aloha.usda. Referencing the robot alone would produce a robot whose joints do
nothing. The .sh wrapper therefore chains the existing pipeline steps against the new
scene -- they all already take --scene, so nothing had to be forked.

Usage:
    nav/sim/build_nav_scene.sh warehouse
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))
sys.path.insert(0, str(REPO / "scripts"))

from episode import load_episode  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--episode", default="warehouse")
parser.add_argument("--robot-usd", default=str(REPO / "assets" / "usd" / "Aloha" / "Aloha.usda"))
parser.add_argument("--out", default=None, help="Defaults to assets/usd/nav_<episode>.usda")
args = parser.parse_args()

ep = load_episode(args.episode)
out_path = Path(args.out) if args.out else REPO / "assets" / "usd" / f"nav_{ep.name}.usda"

print(f"Building nav scene for episode '{ep.name}'")
print(f"  environment : {ep.scene}")
print(f"  start       : {ep.start} yaw={ep.start_yaw_deg} deg")
print(f"  goal        : {ep.goal}  ({ep.straight_line_distance_m:.2f} m away)")
print(f"  out         : {out_path}\n")

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# Author directly into the output layer rather than new_stage()+Export(). Export()
# flattens the whole composition and bakes every referenced mesh inline -- with the
# warehouse environment that produced a 233 MB file, over GitHub's 100 MB limit.
out_path.parent.mkdir(parents=True, exist_ok=True)
out_layer = Sdf.Layer.FindOrOpen(str(out_path))
if out_layer is None:
    out_layer = Sdf.Layer.CreateNew(str(out_path))
else:
    out_layer.Clear()

stage = Usd.Stage.Open(out_layer)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

env_prim = stage.DefinePrim("/World/Environment", "Xform")
env_prim.GetReferences().AddReference(ep.scene)

# The robot goes in at the EPISODE'S start pose, not the origin. This is not a
# convenience -- for the office episode it is the difference between working and
# exploding. That stage is 1063 m across and the origin is not inside the navigable
# area; CLAUDE.md records a spawn that overlapped building geometry and PhysX's
# separation impulse threw the whole articulation ~7.7 m and tipped it 181 degrees.
# Every episode's start coordinate comes from DynaNav, which drives from there
# successfully, so it is known-good free space.
robot_prim = stage.DefinePrim("/World/Aloha", "Xform")
robot_prim.GetReferences().AddReference(str(args.robot_usd))
UsdGeom.XformCommonAPI(robot_prim).SetTranslate(tuple(ep.start))
UsdGeom.XformCommonAPI(robot_prim).SetRotate((0.0, 0.0, ep.start_yaw_deg))

# Must be explicit. A raw-authored root layer inherits no physics scene, and without
# one Play simulates nothing at all -- rigid bodies just hang in the air (verified in
# this repo, and it passed verify_physics.py the whole time because isaacsim.core
# bootstraps its own physics context and masked it).
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr(9.81)

# Bake a viewpoint near the robot. Isaac Sim's "frame all" on stage-open zooms to fit
# the ENTIRE stage bbox, which on a 1 km environment is what "the scene starts from
# over the building very far" looked like. Placing the camera behind and above the
# start pose means the GUI opens looking at the robot regardless of stage scale.
yaw = math.radians(ep.start_yaw_deg)
cam = UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
UsdGeom.XformCommonAPI(cam.GetPrim()).SetTranslate((
    ep.start[0] - 4.0 * math.cos(yaw),
    ep.start[1] - 4.0 * math.sin(yaw),
    ep.start[2] + 3.0,
))
UsdGeom.XformCommonAPI(cam.GetPrim()).SetRotate((65.0, 0.0, ep.start_yaw_deg - 90.0))

out_layer.Save()
print(f"Authored layer: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

# Sanity: did the environment reference actually resolve? A silently-unresolved
# reference gives an empty stage that still "opens fine", which is exactly the
# failure mode that wasted a debugging round before (relative --scene path).
usd_context = omni.usd.get_context()
usd_context.open_stage(str(out_path))
check_stage = usd_context.get_stage()
for _ in range(120):  # CDN environments stream in; do not judge on frame 1
    kit.update()
check_stage.Load()
for _ in range(30):
    kit.update()

prim_count = sum(1 for _ in check_stage.Traverse())
robot_ok = check_stage.GetPrimAtPath("/World/Aloha/Geometry/base_link").IsValid()
print(f"Composed prims : {prim_count}")
print(f"Robot present  : {robot_ok}")
if prim_count < 100 or not robot_ok:
    print("\nERROR: composition looks empty -- a reference failed to resolve.", file=sys.stderr)
    kit.close()
    raise SystemExit(1)

print("\nLayer authored and verified. The joint drives / wheel colliders / nav camera")
print("are NOT applied yet -- run build_nav_scene.sh, which chains the pipeline steps.")

kit.close()
