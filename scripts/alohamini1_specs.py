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

# Joint limits in radians, same numbers as scripts/pipeline/patch_urdf_joint_limits.py's
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

# --- Cameras (names/resolution/fps from the OFFICIAL LeRobot integration:
# third_party/lerobot_alohamini/src/lerobot/robots/alohamini/config_alohamini.py --
# all cameras there are OpenCV 640x480 @ 30fps; "forward" is active by default,
# "wrist_left"/"wrist_right" are scaffolded in the same config) ---
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 30
# Prim paths where scripts/pipeline/add_cameras.py authors the USD cameras (children of the
# robot links so they move with the wrist/column). LeRobot observation keys are
# "observation.images.<name>" for each <name> key here.
_BASE_LINK = "/World/Aloha/Geometry/base_link"
_ARM_CHAIN = f"{_BASE_LINK}/vertical_link"
CAMERA_PRIM_PATHS = {
    "forward": f"{_ARM_CHAIN}/camera_forward",
    "wrist_left": f"{_ARM_CHAIN}/left_link1/left_link2/left_link3/left_link4/left_link5/camera_wrist_left",
    "wrist_right": f"{_ARM_CHAIN}/right_link1/right_link2/right_link3/right_link4/right_link5/camera_wrist_right",
}

# --- Navigation camera (NOT part of the LeRobot camera set) ---
# Deliberately kept OUT of CAMERA_PRIM_PATHS: that dict defines the official LeRobot
# observation contract ("observation.images.<name>"), and adding a key there would
# silently change the shape of every recorded manipulation dataset.
#
# It exists because the three cameras above all face the MANIPULATION front, which is
# -Y (both grippers work toward -Y -- measured, see add_cameras.py), while the base's
# driving forward (vx) is +X. For manipulation that is correct. For navigation it is
# useless: the camera would show whatever is 90 degrees off the side of the direction
# of travel, and a vision-language navigation policy asked to "go straight ahead" would
# be reasoning about a view the robot is not driving into.
#
# Its INTRINSICS ARE NOT AlohaMini's -- they are copied verbatim off the sensor TIC-VLA
# was trained through. DynaNav feeds the policy the Nova Carter asset's own front Hawk
# stereo left eye (nova_carter_sensors.usd, /chassis_link/front_hawk/left/camera_left),
# rendered 1920x1080. Probed directly off that asset rather than guessed:
#
#     focal 2.8734 mm, aperture 5.760 x 3.600  ->  HFOV 90.1 deg, VFOV 64.1 deg
#     mounted x=0.100 y=0.075 z=0.346 on chassis_link, view (1,0,0), pitch 0.0 deg
#
# A camera is not a free choice here, it is part of the model's input distribution. The
# first version of this file used AlohaMini's own webcam-ish 78 deg HFOV at 1.15 m and
# tilted 10 deg down, and the policy could not see the landmarks the benchmark prompts
# name -- a 90 deg view cropped to 78 deg loses exactly the peripheral context that
# "the second aisle from the right" depends on.
# On base_link, not the lift column: Nova Carter's is on chassis_link, and a nav camera
# that rises and falls with the lift is a nav camera whose horizon moves for reasons the
# policy has no way to account for.
NAV_CAMERA_PRIM_PATH = f"{_BASE_LINK}/camera_nav"
NAV_CAMERA_RESOLUTION = (1920, 1080)     # DynaNav's render product size
NAV_CAMERA_FOCAL_MM = 2.8734347820281982
NAV_CAMERA_APERTURE = (5.760000228881836, 3.5999999046325684)
NAV_CAMERA_HEIGHT_M = 0.346              # Nova Carter's Hawk height on chassis_link

# A light co-mounted with camera_nav, facing the same +X driving direction it does.
# DynaNav's own environments cannot be trusted to light themselves: office.usd has 121
# small ceiling fixtures and NO ambient/dome light at all, and hospital.usd's one
# DomeLight is a dim overcast HDRI that (checked directly against the downloaded USD,
# not assumed) barely reaches an enclosed corridor anyway -- a dome lights exterior-
# facing surfaces, and a hallway with a ceiling has none. Measured against the darkest
# episode start in this benchmark (hospital_down_hallway2, a 30 m interior corridor):
# no light at all renders at mean pixel value ~14/255; this fixture at the settings
# below renders the same frame at ~88/255 with 0% of pixels clipped, and does not
# visibly wash out the brightest episode either (hospital_vending_machine, a
# window-lit room, +8/255 over its own unlit baseline of ~165). A physically-placed
# light was the only lever that actually worked: raising the renderer's filmIso
# (camera-sensitivity) setting looked identical in a quick check but the gain proved to
# be transient -- fully settled frames (150 sim steps post-teleport, not 40-60) came
# back indistinguishable from the filmIso=100 default, so something in the render
# pipeline re-normalizes against manual exposure changes but not against genuine scene
# radiance. Don't try filmIso again without that same settle time or you will
# rediscover the same false positive.
NAV_HEADLIGHT_PRIM_PATH = f"{_BASE_LINK}/camera_nav_headlight"
NAV_HEADLIGHT_INTENSITY = 45000.0
NAV_HEADLIGHT_RADIUS_M = 0.3

# --- Third-person chase camera (also NOT part of the LeRobot camera set) ---
# Parented to base_link rather than the lift column, so it does not ride up and down
# when the lift moves -- and so it inherits the base's yaw, giving a chase view that
# turns with the robot instead of a fixed world camera the robot drives out of.
CHASE_CAMERA_PRIM_PATH = f"{_BASE_LINK}/camera_chase"

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
