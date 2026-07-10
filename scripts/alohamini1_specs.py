"""Single source of truth for AlohaMini1 physical constants used across scripts.

Sources:
- Arm joint gains: NVIDIA Sim-to-Real-SO-101-Workshop's so101.py (same physical arm)
- Lift travel range: liyiteng/lerobot_alohamini src/lerobot/robots/alohamini/lift_axis.py
- Wheel/base kinematics: liyiteng/lerobot_alohamini src/lerobot/robots/alohamini/lekiwi.py
  and model_specs.py (alohamini1 row)
"""

import math

# --- Arm joints (same for left_* and right_*) ---
# stiffness (Nm/rad), damping (Nm*s/rad), effort_limit (Nm) -- ported from NVIDIA's
# tuned Isaac Lab config for the same physical SO-101 arm.
ARM_JOINT_GAINS = {
    1: {"name": "Rotation", "stiffness": 55, "damping": 0.7, "effort": 30},
    2: {"name": "Pitch", "stiffness": 30, "damping": 0.8, "effort": 30},
    3: {"name": "Elbow", "stiffness": 25, "damping": 0.7, "effort": 30},
    4: {"name": "Wrist_Pitch", "stiffness": 12, "damping": 0.5, "effort": 30},
    5: {"name": "Wrist_Roll", "stiffness": 7, "damping": 0.5, "effort": 30},
    6: {"name": "Jaw", "stiffness": 4, "damping": 0.3, "effort": 30},
}
ARM_SOLVER_POSITION_ITERATIONS = 32
# NVIDIA's SO-101 config uses 1, but the wheel velocity drives needed more velocity
# iterations to converge without oscillating (verified empirically -- see
# configure_physics.py's configure_velocity_drive comment).
ARM_SOLVER_VELOCITY_ITERATIONS = 4

# Jaw (joint6) limits in radians, from the patched URDF (real SO-101 spec: -10 to 100
# degrees) -- used by control_terminal.py's gripper open/close shorthand.
# NOTE: which extreme is physically "open" vs "closed" has NOT been visually verified
# against this URDF's own axis convention -- same caveat as the other joint polarities
# (see plan.md/CLAUDE.md). Assumed lower=open/upper=closed for now; flip if wrong.
JAW_OPEN_RAD = -0.174533
JAW_CLOSED_RAD = 1.745329

# --- Lift (vertical_move, prismatic) ---
LIFT_MIN_M = 0.0
LIFT_MAX_M = 0.60
LIFT_LEAD_MM_PER_REV = 84.0  # lead screw pitch, alohamini1
LIFT_STIFFNESS = 5000.0  # N/m -- engineering estimate, not measured
LIFT_DAMPING = 200.0     # N*s/m -- engineering estimate

# --- Mobile base (LeKiwi 3-wheel omni base, alohamini1 dims) ---
WHEEL_RADIUS_M = 0.05
BASE_RADIUS_M = 0.125  # center to each wheel
# Wheel mounting angles (radians), from lekiwi.py: np.radians([240, 0, 120] - 90)
# = [150, -90, 30] degrees. IMPORTANT: this order does NOT naively match our URDF's
# wheel1/wheel2/wheel3 naming -- verified by computing each wheel's actual XY position
# angle from its URDF joint origin:
#   wheel1 origin xyz=(-0.1538, 0.091161)  -> atan2 ~= 149.4 deg  (matches the 150 slot)
#   wheel3 origin xyz=(-0.0020, -0.17875)  -> atan2 ~= -90.7 deg  (matches the -90 slot)
#   wheel2 origin xyz=(0.15563, 0.087993)  -> atan2 ~=  29.5 deg  (matches the 30 slot)
# So the correct pairing is ["wheel1", "wheel3", "wheel2"], not naive wheel1/2/3 order.
WHEEL_ANGLES_RAD = [math.radians(a - 90) for a in (240, 0, 120)]
WHEEL_NAMES = ["wheel1", "wheel3", "wheel2"]


def body_to_wheel_speeds(vx: float, vy: float, omega: float) -> list[float]:
    """Convert body-frame velocity (vx, vy in m/s, omega in rad/s) to per-wheel angular
    velocity (rad/s).

    This is a direct port of lekiwi.py's `_body_to_wheel_raw` kinematics (verified
    against the exact source, not a paraphrase): velocity_vector = [-vx, -vy, omega],
    wheel_linear = [cos(a), sin(a), base_radius] . velocity_vector per wheel angle a,
    wheel_angular = wheel_linear / wheel_radius. The real function takes omega in
    deg/s and converts internally; this one takes rad/s directly (same math).
    """
    velocity_vector = (-vx, -vy, omega)
    wheel_speeds = []
    for angle in WHEEL_ANGLES_RAD:
        row = (math.cos(angle), math.sin(angle), BASE_RADIUS_M)
        linear = sum(r * v for r, v in zip(row, velocity_vector))
        wheel_speeds.append(linear / WHEEL_RADIUS_M)
    return wheel_speeds
