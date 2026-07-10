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

# Joint limits in radians, same numbers as scripts/patch_urdf_joint_limits.py's
# ARM_JOINT_LIMITS (that script converts from the degree values read off
# SO-ARM101-USD.usd; kept as a separate radian copy here for convenience -- if you
# ever change one, change both). Used for REPL help text and clamping.
ARM_JOINT_LIMITS_RAD = {
    1: (-1.919862, 1.919862),   # Rotation
    2: (-1.745331, 1.745331),   # Pitch
    3: (-1.745331, 1.570800),   # Elbow
    4: (-1.658061, 1.658061),   # Wrist_Pitch
    5: (-2.792519, 2.792519),   # Wrist_Roll
    # Jaw (gripper): NOT the value read off NVIDIA's SO-ARM101-USD.usd (-0.174533,
    # 1.745329) -- that range doesn't match how THIS mesh's fingers actually close
    # (different CAD source than NVIDIA's reference asset). Empirically verified by
    # rendering close-up screenshots at several angles: -0.174533 leaves a visible gap
    # (not closed), 1.745329 is genuinely fully open, fingers first meet at -1.570796
    # (-90 deg), and -1.85 (~-106 deg) gives a bit of extra squeeze margin for gripping
    # thin objects while still looking visually clean (no bad mesh interpenetration).
    # Corrected in the URDF itself, not just here.
    6: (-1.85, 1.745329),   # Jaw (gripper) -- lower=closed, upper=open
}
# NVIDIA's SO-101 config uses position=32/velocity=1, tuned for a FIXED-base single
# arm. Our robot is floating-base (mobile), and the lift joint (vertical_move) sits
# directly between the floating root (base_link, heavy: wheels + everything) and
# vertical_link (carrying both arms) -- that specific joint needed far more solver
# iterations to converge at all. Empirically verified: at position=32/velocity=4 (the
# old values) and even position=64/velocity=8, the lift joint gets physically stuck
# near 0 regardless of commanded target or drive force (tried up to 5000N/50000
# stiffness -- still stuck), while position=128/velocity=16 converges to the exact
# target reliably. Ruled out self-collision and external furniture collision as causes
# first (same result with self-collision off, same result in an empty Grid
# environment) -- this is purely a solver-iteration insufficiency for this specific
# floating-root-adjacent joint. Arm joints (revolute, further down the chain) never
# needed this many iterations on their own, but the whole articulation shares one
# iteration count, so this covers everything.
ARM_SOLVER_POSITION_ITERATIONS = 128
ARM_SOLVER_VELOCITY_ITERATIONS = 16

# Jaw (joint6) limits in radians -- used by control_terminal.py's gripper open/close
# shorthand. Confirmed visually via close-up rendered screenshots at several joint
# angles (see ARM_JOINT_LIMITS_RAD[6] comment): lower=closed, upper=open.
JAW_OPEN_RAD = 1.745329
JAW_CLOSED_RAD = -1.85

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
