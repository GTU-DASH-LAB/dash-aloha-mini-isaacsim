"""Patch assets/upstream_alohamini1/urdf/Aloha.urdf in place.

The upstream URDF (SolidWorks-exported) has all revolute/prismatic joint limits zeroed
out (lower=upper=effort=velocity=0), which is a known exporter artifact -- see
CLAUDE.md. This script ports real values:

- Arm joints (`{side}_joint1..6`): limits read off NVIDIA's SO-ARM101-USD.usd via
  scripts/tools/read_so101_joint_limits.py (degrees -> radians), effort from
  Sim-to-Real-SO-101-Workshop's so101.py (effort_limit_sim=30 for all joints, their
  tuned simulation value, not the raw servo torque spec).
- `vertical_move` (lift): no authoritative spec found; placeholder range, flagged for
  empirical correction in Phase 2 once the mesh travel is visually inspected.

Joint mapping (same physical arm, both sides): joint1=Rotation, joint2=Pitch,
joint3=Elbow, joint4=Wrist_Pitch, joint5=Wrist_Roll, joint6=Jaw (gripper).

NOTE: sign polarity (which direction is "positive") was NOT verified against this
URDF's own axis vectors -- the AlohaMini URDF and NVIDIA's SO-101 USD are independently
authored assets. Magnitudes are trustworthy; polarity should be checked visually in
Phase 2 (does the rest pose land inside the range without immediately clamping?) and
flipped here if not.
"""

import math
import re
from pathlib import Path

URDF_PATH = Path(__file__).parent.parent / "assets/upstream_alohamini1/urdf/Aloha.urdf"

DEG = math.pi / 180.0

# (lower_deg, upper_deg, effort, velocity) per joint index 1..6, same for left/right
ARM_JOINT_LIMITS = {
    1: (-109.99987030029297, 109.99987030029297, 30, 3.0),   # Rotation
    2: (-100.00003814697266, 100.00003814697266, 30, 3.0),   # Pitch
    3: (-100.00003814697266, 90.00020599365234, 30, 3.0),    # Elbow
    4: (-94.99983215332031, 94.99983215332031, 30, 3.0),     # Wrist_Pitch
    5: (-160.00018310546875, 160.00018310546875, 30, 3.0),   # Wrist_Roll
    6: (-10.000003814697266, 100.00003814697266, 30, 3.0),   # Jaw (gripper)
}

# Real spec from liyiteng/lerobot_alohamini (src/lerobot/robots/alohamini/lift_axis.py):
# soft min 0mm, soft max 600mm, lead screw 84mm/rev (alohamini1). Effort/velocity are
# still engineering estimates (the real hardware spec is in motor ticks, not N/(m/s)).
LIFT_LIMIT = (0.0, 0.60, 200, 0.15)  # meters, meters; effort in N, velocity in m/s


def make_limit_block(indent: str, lower: float, upper: float, effort: float, velocity: float) -> str:
    return (
        f'{indent}<limit\n'
        f'{indent}  lower="{lower:.6f}"\n'
        f'{indent}  upper="{upper:.6f}"\n'
        f'{indent}  effort="{effort}"\n'
        f'{indent}  velocity="{velocity}" />'
    )


def patch():
    text = URDF_PATH.read_text()

    for side in ("left", "right"):
        for idx, (lo_deg, hi_deg, effort, vel) in ARM_JOINT_LIMITS.items():
            joint_name = f"{side}_joint{idx}"
            lo_rad, hi_rad = lo_deg * DEG, hi_deg * DEG

            # Match this joint's block up to and including its <limit .../> tag
            pattern = re.compile(
                r'(<joint\s*\n\s*name="' + re.escape(joint_name) + r'"\s*\n\s*type="revolute">.*?)'
                r'<limit\s*\n\s*lower="0"\s*\n\s*upper="0"\s*\n\s*effort="0"\s*\n\s*velocity="0"\s*/>',
                re.DOTALL,
            )
            match = pattern.search(text)
            if not match:
                raise RuntimeError(f"Could not find zeroed <limit> block for {joint_name}")

            new_limit = make_limit_block("    ", lo_rad, hi_rad, effort, vel)
            text = text[: match.start()] + match.group(1) + new_limit.strip() + text[match.end():]
            print(f"Patched {joint_name}: lower={lo_rad:.4f} upper={hi_rad:.4f} "
                  f"effort={effort} velocity={vel}")

    # Lift joint
    lo, hi, effort, vel = LIFT_LIMIT
    pattern = re.compile(
        r'(<joint\s*\n\s*name="vertical_move"\s*\n\s*type="prismatic">.*?)'
        r'<limit\s*\n\s*lower="0"\s*\n\s*upper="0"\s*\n\s*effort="0"\s*\n\s*velocity="0"\s*/>',
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find zeroed <limit> block for vertical_move")
    new_limit = make_limit_block("    ", lo, hi, effort, vel)
    text = text[: match.start()] + match.group(1) + new_limit.strip() + text[match.end():]
    print(f"Patched vertical_move: lower={lo} upper={hi} effort={effort} velocity={vel} "
          f"[PLACEHOLDER - verify in Phase 2]")

    URDF_PATH.write_text(text)
    print(f"\nWrote {URDF_PATH}")


if __name__ == "__main__":
    patch()
