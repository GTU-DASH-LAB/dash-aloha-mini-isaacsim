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
    ~/isaacsim/python.sh scripts/fix_wheel_collision.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alohamini1_specs import WHEEL_RADIUS_M  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

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

# 1. Disable base_link's own shell collision (the culprit, confirmed via raycast).
# The collision prim is an instance proxy (shares a prototype with other links) --
# USD doesn't allow authoring overrides directly on instance proxies. Un-instance the
# specific instance root first (verified via find_instance_root.py: it's one level up,
# not the collision prim itself) so this one copy becomes independently editable
# without affecting other instances of the same prototype elsewhere in the scene.
instance_root_path = "/World/Aloha/Geometry/base_link/base_link"
instance_root_prim = stage.GetPrimAtPath(instance_root_path)
if instance_root_prim.IsValid() and instance_root_prim.IsInstance():
    instance_root_prim.SetInstanceable(False)
    print(f"Un-instanced: {instance_root_path}")
else:
    print(f"WARNING: expected instance root not found or not an instance: {instance_root_path}")

shell_collision_path = "/World/Aloha/Geometry/base_link/base_link/base_link"
shell_prim = stage.GetPrimAtPath(shell_collision_path)
if shell_prim.IsValid() and shell_prim.HasAPI(UsdPhysics.CollisionAPI):
    UsdPhysics.CollisionAPI(shell_prim).CreateCollisionEnabledAttr(False)
    print(f"Disabled collision on: {shell_collision_path}")
else:
    print(f"WARNING: expected shell collision prim not found or missing CollisionAPI: {shell_collision_path}")

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

stage.Save()
print(f"\nSaved: {args.scene}")

kit.close()
