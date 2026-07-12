"""One-off tool: read joint limits off NVIDIA's reference SO-ARM101-USD.usd so they can
be ported onto the AlohaMini1 URDF (whose exporter zeroed out all joint limits).

Usage:
    ~/isaacsim/python.sh scripts/tools/read_so101_joint_limits.py
"""

from isaacsim import SimulationApp

kit = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics  # noqa: E402

USD_PATH = "/home/gtu_dsa/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd"

stage = Usd.Stage.Open(USD_PATH)
if stage is None:
    raise RuntimeError(f"Could not open stage: {USD_PATH}")

print("=== Joints found ===")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        joint = UsdPhysics.Joint(prim)
        kind = "Revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "Prismatic"
        lower = upper = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            lower = rj.GetLowerLimitAttr().Get()
            upper = rj.GetUpperLimitAttr().Get()
        else:
            pj = UsdPhysics.PrismaticJoint(prim)
            lower = pj.GetLowerLimitAttr().Get()
            upper = pj.GetUpperLimitAttr().Get()
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        print(f"{prim.GetPath()} [{kind}] lower={lower} upper={upper} "
              f"body0={body0} body1={body1}")

kit.close()
