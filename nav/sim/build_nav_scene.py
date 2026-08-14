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

A stage serves an ENVIRONMENT, not an episode. It used to be one stage per episode,
which meant eleven benchmark episodes cost eleven of these four-process build chains --
to produce stages that differed only in where one prim sat, six of them referencing the
same hospital.usd. The start pose was never load-bearing: the runner teleports to
`episode.start` before every run anyway. So `nav_hospital.usda` now serves all six
hospital episodes, and the whole ladder builds three stages instead of thirteen.

That also makes the UI useful. A running session can switch between every episode in
its environment -- same stage, new start pose and new goal -- instead of being pinned
to the single episode its stage was authored for.

Usage:
    nav/sim/build_nav_scene.sh hospital
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))
sys.path.insert(0, str(REPO / "scripts"))

from episode import env_name, episodes_by_env, representative_episode  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    default="warehouse",
    help="environment to build, e.g. hospital / office / warehouse",
)
parser.add_argument("--robot-usd", default=str(REPO / "assets" / "usd" / "Aloha" / "Aloha.usda"))
parser.add_argument("--out", default=None, help="Defaults to assets/usd/nav_<env>.usda")
args = parser.parse_args()

# Accept an episode name too, so an old invocation builds the right stage instead of
# failing with "unknown environment". Both spellings land on the same file.
env = args.env
if env not in episodes_by_env():
    from episode import load_episodes

    by_name = load_episodes()
    if env in by_name:
        env = env_name(by_name[env].scene)
        print(f"note: '{args.env}' is an episode; building its environment '{env}'\n")

ep = representative_episode(env)
out_path = Path(args.out) if args.out else REPO / "assets" / "usd" / f"nav_{env}.usda"
siblings = episodes_by_env()[env]

print(f"Building nav scene for environment '{env}'")
print(f"  usd         : {ep.scene}")
print(f"  episodes    : {len(siblings)} ({', '.join(e.name for e in siblings)})")
print(f"  authored at : {ep.start} yaw={ep.start_yaw_deg} deg (from '{ep.name}')")
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

# The robot goes in at a REAL start pose, not the origin. This is not a convenience --
# for the office environment it is the difference between working and exploding. That
# stage is 1063 m across and the origin is not inside the navigable area; CLAUDE.md
# records a spawn that overlapped building geometry and PhysX's separation impulse
# threw the whole articulation ~7.7 m and tipped it 181 degrees. Every episode's start
# coordinate comes from DynaNav, which drives from there successfully, so it is
# known-good free space -- which of them is used here does not matter, because the
# runner teleports to the episode's own start before each run.
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
print(f"Serves {len(siblings)} episode(s): {', '.join(e.name for e in siblings)}")

kit.close()
