"""Phase 3 verification: step physics, check for NaN/instability, then command a joint
position and confirm it actually reaches the target.

Usage:
    ~/isaacsim/python.sh scripts/pipeline/verify_physics.py
"""

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--screenshot", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/docs/physics_verification.png")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
import numpy as np  # noqa: E402

# STEP 1: open the composed scene and wait for all referenced assets to load.
usd_context = omni.usd.get_context()
usd_context.open_stage(args.scene)
stage = usd_context.get_stage()
for _ in range(60):
    kit.update()
stage.Load()
for _ in range(10):
    kit.update()

# STEP 2: start physics and wrap the robot in an Articulation handle.
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    kit.update()

art = Articulation(prim_paths_expr="/World/Aloha/Geometry/base_link")
art.initialize()

dof_names = art.dof_names
print(f"DOF count: {art.num_dof}")
print(f"DOF names: {dof_names}")

base_prim = stage.GetPrimAtPath("/World/Aloha/Geometry/base_link")
xform_cache = UsdGeom.XformCache()


def get_base_height():
    m = xform_cache.GetLocalToWorldTransform(base_prim)
    return m.ExtractTranslation()[2]


# STEP 3: stability check -- 120 steps under gravity; the base must neither sink nor
# float, and no joint may go NaN/Inf.
heights = []
for i in range(120):
    kit.update()
    if i % 20 == 0:
        heights.append(get_base_height())

print(f"Base height samples over 120 steps: {heights}")

positions = art.get_joint_positions()
velocities = art.get_joint_velocities()
nan_found = np.any(np.isnan(positions)) or np.any(np.isnan(velocities))
inf_found = np.any(np.isinf(positions)) or np.any(np.isinf(velocities))
print(f"NaN in joint state: {nan_found}, Inf in joint state: {inf_found}")
print(f"Joint positions after 120 steps: {dict(zip(dof_names, positions[0].tolist()))}")

# STEP 4: convergence check -- command one arm joint to a nonzero target and require
# it to actually get there (this is what catches a scene whose drives were wiped by a
# bare build_scene.py run -- see rebuild_all.sh).
if "left_joint1" in dof_names:
    idx = dof_names.index("left_joint1")
    target = 0.5  # radians, well within the +-1.92 rad limit
    target_positions = art.get_joint_positions()
    target_positions[0][idx] = target
    art.set_joint_position_targets(target_positions)

    for _ in range(180):
        kit.update()

    final_positions = art.get_joint_positions()
    achieved = final_positions[0][idx]
    error = abs(achieved - target)
    print(f"\nCommanded left_joint1 to {target:.3f} rad")
    print(f"Achieved after 180 steps: {achieved:.3f} rad (error={error:.4f} rad)")
    print(f"PASS (within 0.05 rad)" if error < 0.05 else f"FAIL (error too large)")
else:
    print("left_joint1 not found in dof_names!")

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
