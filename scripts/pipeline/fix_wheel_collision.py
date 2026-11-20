"""Fix the wheel-ground contact problem diagnosed via raycast: base_link's own shell
collision (Convex Hull and Convex Decomposition both did this) extends down far enough
to block ground contact before the wheels ever touch, so spinning the wheel joints
produces no base motion.

Fix:
1. Disable collision on base_link's own shell collision prim (it becomes a "ghost" for
   now -- fine for locomotion testing; other links still have working collision).
2. Add an explicit sphere collider (radius = WHEEL_RADIUS_M) to each wheel link. Sphere,
   not cylinder, specifically to avoid needing to compute each wheel's exact local spin
   axis after the URDF->USD axis conversion -- spheres are orientation-independent. This
   is a real simplification (a sphere can roll sideways too), acceptable for a
   holonomic 3-wheel base where some sideways compliance is expected anyway from the
   real omni-wheel rollers. Documented, not hidden.
3. Create and bind an explicit PhysX friction material to the wheels (the stage had
   ZERO physics materials anywhere before this -- confirmed via a full stage traverse).

Usage:
    ~/isaacsim/python.sh scripts/pipeline/fix_wheel_collision.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/ root, for alohamini1_specs
from alohamini1_specs import KIT_GPU0, WHEEL_RADIUS_M  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True, **KIT_GPU0})

import omni.usd  # noqa: E402
from pxr import UsdPhysics, UsdGeom, UsdShade, Sdf  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(args.scene)
stage = usd_context.get_stage()
for _ in range(60):
    kit.update()
stage.Load()
for _ in range(10):
    kit.update()


def fail_if_missing(missing: list[str], step: str) -> None:
    """End the step WITHOUT its sentinel (and without saving) if the robot was not all there.

    These used to be warnings the step printed and walked past, and Kit then exited 0 like
    it always does, so a build missing a joint drive or a wheel collider was published as
    complete. None of them fire on a real build (every nav and scene build log on disk),
    so any of them firing means the stage is not the robot this step was written for.
    os._exit(1), not kit.close() + SystemExit: close() ends the process with status 0
    before the raise is reached (measured 2026-10-05).
    """
    if not missing:
        return
    print(f"\nERROR: {step}: {len(missing)} expected robot prim(s) missing -- NOT saved:",
          file=sys.stderr)
    for what in missing:
        print(f"  - {what}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


# 1. Disable base_link's own shell collision (the culprit, confirmed via raycast).
# The collision prim is an instance proxy (shares a prototype with other links) --
# USD doesn't allow authoring overrides directly on instance proxies. Un-instance the
# specific instance root first (verified via find_instance_root.py: it's one level up,
# not the collision prim itself) so this one copy becomes independently editable
# without affecting other instances of the same prototype elsewhere in the scene.
missing: list[str] = []
instance_root_path = "/World/Aloha/Geometry/base_link/base_link"
instance_root_prim = stage.GetPrimAtPath(instance_root_path)
if instance_root_prim.IsValid() and instance_root_prim.IsInstance():
    instance_root_prim.SetInstanceable(False)
    print(f"Un-instanced: {instance_root_path}")
else:
    print(f"WARNING: expected instance root not found or not an instance: {instance_root_path}")
    if not instance_root_prim.IsValid():
        # Not an instance any more is a re-run on a fixed stage; ABSENT is a different robot.
        missing.append(f"base_link instance root {instance_root_path}")

shell_collision_path = "/World/Aloha/Geometry/base_link/base_link/base_link"
shell_prim = stage.GetPrimAtPath(shell_collision_path)
if shell_prim.IsValid() and shell_prim.HasAPI(UsdPhysics.CollisionAPI):
    UsdPhysics.CollisionAPI(shell_prim).CreateCollisionEnabledAttr(False)
    print(f"Disabled collision on: {shell_collision_path}")
else:
    print(f"WARNING: expected shell collision prim not found or missing CollisionAPI: {shell_collision_path}")
    # Left enabled, the shell's convex hull reaches below the tyres and the robot sits on
    # its belly (the 09-01 stub stage drove exactly that robot).
    missing.append(f"base_link shell collider {shell_collision_path}")

# 2. Create a friction material and bind it to the ground + wheels.
mat_path = "/World/Physics/WheelFrictionMaterial"
mat_prim = stage.DefinePrim(mat_path, "Material")
mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
mat_api.CreateStaticFrictionAttr(0.9)
mat_api.CreateDynamicFrictionAttr(0.8)
mat_api.CreateRestitutionAttr(0.0)
mat_shade = UsdShade.Material(mat_prim)
print(f"Created friction material: {mat_path}")

WHEEL_LINK_PATHS = {
    "wheel1": "/World/Aloha/Geometry/base_link/wheel1",
    "wheel2": "/World/Aloha/Geometry/base_link/wheel2",
    "wheel3": "/World/Aloha/Geometry/base_link/wheel3",
}

for wheel_name, link_path in WHEEL_LINK_PATHS.items():
    link_prim = stage.GetPrimAtPath(link_path)
    if not link_prim.IsValid():
        print(f"WARNING: wheel link not found: {link_path}")
        missing.append(f"wheel link {link_path}")
        continue

    # Each wheel link is itself an instance root (verified via find_instance_root2.py)
    # -- un-instance it so we can add a child collision shape.
    if link_prim.IsInstance():
        link_prim.SetInstanceable(False)
        print(f"Un-instanced: {link_path}")

    # Disable the original wheel mesh collision -- otherwise it stays active
    # alongside our new sphere (found via check_all_active_collisions.py: both were
    # enabled simultaneously, likely fighting each other / the mesh one has no
    # friction material bound).
    orig_mesh_path = f"{link_path}/{wheel_name}"
    orig_mesh_prim = stage.GetPrimAtPath(orig_mesh_path)
    if orig_mesh_prim.IsValid() and orig_mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(orig_mesh_prim).CreateCollisionEnabledAttr(False)
        print(f"Disabled original mesh collision on: {orig_mesh_path}")

    sphere_path = f"{link_path}/collision_sphere"
    sphere_geom = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere_geom.CreateRadiusAttr(WHEEL_RADIUS_M)
    sphere_prim = sphere_geom.GetPrim()

    UsdPhysics.CollisionAPI.Apply(sphere_prim)
    UsdShade.MaterialBindingAPI(sphere_prim).Bind(mat_shade, materialPurpose="physics")
    # Purely a physics proxy -- don't render it.
    UsdGeom.Imageable(sphere_prim).MakeInvisible()

    print(f"Added sphere collider (r={WHEEL_RADIUS_M}) to {wheel_name} at {sphere_path}")

# Also bind the friction material to the ground plane for a consistent friction pair.
ground_path = "/World/Environment/GroundPlane/CollisionPlane"
ground_prim = stage.GetPrimAtPath(ground_path)
if ground_prim.IsValid():
    UsdShade.MaterialBindingAPI(ground_prim).Bind(mat_shade, materialPurpose="physics")
    print(f"Bound friction material to ground: {ground_path}")
else:
    print(f"WARNING: ground plane not found at {ground_path}")

# The ground plane is the ENVIRONMENT's, not the robot's: outdoor_small has none, and that
# warning is expected there. Everything above is the robot, and must be there.
fail_if_missing(missing, "fix_wheel_collision")
stage.Save()
print(f"\nSaved: {args.scene}")

# The pipeline's proof that this step got to the end. Kit's python.sh exits 0 on an
# uncaught exception and after kit.close() whatever came first (measured 2026-10-05),
# so the exit status cannot say whether a step worked; build_nav_scene.sh requires
# this exact line instead. Printed last, just before close.
print("PIPELINE_STEP_OK fix_wheel_collision", flush=True)
kit.close()
