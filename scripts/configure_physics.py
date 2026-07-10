"""Phase 3: apply joint drives on top of the composed scene.

- Arm joints (left/right_joint1..6): position drive, stiffness/damping/effort from
  alohamini1_specs.ARM_JOINT_GAINS (ported from NVIDIA's tuned SO-101 config).
- Lift joint (vertical_move): position drive, conservative stiffness/damping.
- Wheel joints (wheel1/2/3): velocity drive, target=0 for now -- Phase 4's control
  script sets nonzero targets via alohamini1_specs.body_to_wheel_speeds().

Edits are authored directly onto scene.usda (sparse overrides on top of the referenced
Aloha.usda/environment content -- standard USD workflow, doesn't require flattening).

Usage:
    ~/isaacsim/python.sh scripts/configure_physics.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alohamini1_specs import (  # noqa: E402
    ARM_JOINT_GAINS,
    ARM_SOLVER_POSITION_ITERATIONS,
    ARM_SOLVER_VELOCITY_ITERATIONS,
    LIFT_DAMPING,
    LIFT_STIFFNESS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import UsdPhysics, PhysxSchema  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(args.scene)
stage = usd_context.get_stage()
for _ in range(60):
    kit.update()
stage.Load()
for _ in range(10):
    kit.update()

PHYSICS_ROOT = "/World/Aloha/Physics"


def configure_position_drive(prim, stiffness, damping, effort, target=0.0, dof_type="angular"):
    drive = UsdPhysics.DriveAPI.Apply(prim, dof_type)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(float(effort))
    drive.CreateTargetPositionAttr(float(target))

    physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)
    physx_joint.CreateJointFrictionAttr(0.0)


def configure_velocity_drive(prim, effort, target=0.0, dof_type="angular"):
    # damping is a viscous gain (Nm per rad/s of velocity error). Wheel rotational
    # inertia is tiny (~5e-5 kg*m^2, from the URDF's own inertial values), so a huge
    # damping value causes bang-bang oscillation/chatter at 60Hz with only a few
    # velocity solver iterations -- verified empirically (wheel hit -19.7 rad/s
    # against a -2.6 rad/s target before this fix). A small damping is already enough
    # torque to reach these velocities given how little inertia there is to overcome.
    drive = UsdPhysics.DriveAPI.Apply(prim, dof_type)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(2.0)
    drive.CreateMaxForceAttr(float(effort))
    drive.CreateTargetVelocityAttr(float(target))


configured = []

for side in ("left", "right"):
    for idx, gains in ARM_JOINT_GAINS.items():
        joint_name = f"{side}_joint{idx}"
        prim = stage.GetPrimAtPath(f"{PHYSICS_ROOT}/{joint_name}")
        if not prim.IsValid():
            print(f"WARNING: joint prim not found: {joint_name}")
            continue
        configure_position_drive(
            prim, gains["stiffness"], gains["damping"], gains["effort"], dof_type="angular"
        )
        # Solver iteration counts are set at the articulation root, not per joint --
        # done once below.
        configured.append(joint_name)

lift_prim = stage.GetPrimAtPath(f"{PHYSICS_ROOT}/vertical_move")
if lift_prim.IsValid():
    configure_position_drive(lift_prim, LIFT_STIFFNESS, LIFT_DAMPING, 200.0, dof_type="linear")
    configured.append("vertical_move")
else:
    print("WARNING: vertical_move joint prim not found")

for wheel_name in ("wheel1", "wheel2", "wheel3"):
    prim = stage.GetPrimAtPath(f"{PHYSICS_ROOT}/{wheel_name}")
    if prim.IsValid():
        configure_velocity_drive(prim, effort=20.0, target=0.0, dof_type="angular")
        configured.append(wheel_name)
    else:
        print(f"WARNING: {wheel_name} joint prim not found")

# Articulation root solver iteration counts (matches NVIDIA's tuned SO-101 config).
# Root is base_link (confirmed via debug_stage.py: 'Articulation roots:
# [/Aloha/Geometry/base_link]' in the un-referenced asset -> /World/Aloha/Geometry/
# base_link once referenced under /World/Aloha in the composed scene).
art_root_prim = stage.GetPrimAtPath("/World/Aloha/Geometry/base_link")
if art_root_prim.IsValid():
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(art_root_prim)
    physx_art.CreateSolverPositionIterationCountAttr(ARM_SOLVER_POSITION_ITERATIONS)
    physx_art.CreateSolverVelocityIterationCountAttr(ARM_SOLVER_VELOCITY_ITERATIONS)
    # NVIDIA's single-arm SO-101 config disables this (makes sense there -- adjacent
    # links in one arm's own chain would otherwise constantly register false-positive
    # collisions right at their shared joint). But this robot's whole body -- both
    # arms, the lift, the base -- is ONE articulation, so disabling self-collision also
    # disabled collision between the left arm and right arm, or an arm and the lift/
    # base, letting them pass through each other with zero response. Enabled instead;
    # verified via verify_physics.py this doesn't introduce instability at the arms'
    # own adjacent joints (exact convergence, stable base height, unchanged from the
    # disabled case).
    physx_art.CreateEnabledSelfCollisionsAttr(True)

print(f"\nConfigured drives on {len(configured)} joints: {configured}")

stage.Save()
print(f"Saved: {args.scene}")

kit.close()
