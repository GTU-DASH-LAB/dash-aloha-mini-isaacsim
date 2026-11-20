"""Unitree G1 that WALKS -- Isaac Lab's pretrained flat policy, in a plain Isaac Sim 6.0.1 script.

No Isaac Lab import anywhere. The pieces, in the order a script uses them:

  configure_physics_scene(stage)   BEFORE play. 200 Hz, TGS, CPU dynamics, in the SESSION layer.
  author_g1(stage, ...)            BEFORE play. References g1_minimal.usd and applies what Isaac
                                   Lab's spawn config applies (articulation/rigid-body props, foot
                                   material, contact reporting on feet + torso).
  author_head_camera(stage, ...)   BEFORE play. The G1's own D435 mount (asset_spec.json).
  G1Locomotion.initialize()        AFTER play. Binds DOF names, overwrites the USD drives with the
                                   trained PD gains/efforts/armature/velocity limits.
  G1Locomotion.reset_to(pos, yaw)  Training's reset pose, zero velocities, policy history cleared.
  loop: set_command(vx, vy, wz); pre_physics_step(); physics_step(); ...; render_only() when a
        frame is needed.

WHAT IS EXACTLY AS TRAINED (sources in scratchpad/g1/policy_notes.md, copied facts in
nav/results/g1/policy_spec.json):
  * physics dt 0.005 s, policy every 4 physics steps (50 Hz), PD targets HELD between ticks
    (IsaacLab-2.2.1 velocity_env_cfg.py:308-311). `pre_physics_step()` counts physics steps, so
    the policy rate is exact whatever cadence the caller steps in -- 3 or 4 steps per call at a
    60 Hz control loop still yields a tick every 20 ms of SIM time.
  * obs 123 = [v_b, w_b, g_b, cmd, q - q0, dq, last_action], v/w = root COM velocity rotated into
    the pelvis frame (walk_policy.base_frame_state). Joint ORDER is matched by NAME
    (G1WalkPolicy.bind), never assumed.
  * action -> target = q0 + 0.5 * a, no clip, all 37 joints (hands included).
  * implicit PD kp/kd/effort/armature/vel-limit per joint, written through the tensor API because
    the USD's own drives are a different robot (5730 N*m/rad, kd 0, armature 0.1).
  * articulation: self-collisions OFF, solver 8 position / 4 velocity iterations.
  * reset: `reset_joint_pos` (default clamped to 0.9 x soft limits, as training's reset event did),
    root 0.74 m above the floor, zero velocity, last action 0.

WHAT IS NOT, and why:
  * FOOT FRICTION. Training: robot shapes mu_s 0.8 / mu_d 0.6 against a 1.0/1.0 "multiply"
    ground, effective 0.8/0.6. Nav floors are what the environment authored -- the hospital's
    CollisionPlane is 0.9/0.8 "average", the Grid environment's has NO material (PhysX default
    0.5/0.5). Here the robot's own shapes get a material 0.8/0.6 with frictionCombineMode "max"
    (PhysX picks the higher-priority mode, and max outranks average/min/multiply), so every floor
    gives AT LEAST the trained friction: exactly 0.8/0.6 on an unmaterialed or slippery floor,
    the floor's own value on a grippier one (hospital: 0.9/0.8). Never less grip than the policy
    was trained on, no environment edit. Restitution combine "min" with 0 -> always 0, as trained.
  * CPU PhysX (numpy tensor backend). Training ran the GPU pipeline; Isaac Lab 2.2.1 on CPU and
    GPU agree with this deploy to 0.01 m/s (policy_notes.md), so this is not a fidelity loss.
  * Commands are clamped to the TRAINED box (vx [0, 1], vy +-0.5, wz +-1). A reverse request is
    clamped to vx = 0 and counted (`reverse_requests`): the policy was never trained to walk
    backwards. G1Base.apply() clamps FIRST and decides hold-or-walk on the CLAMPED command, so a
    pure reverse request (vx < 0, vy = wz = 0) becomes a zero command and HOLDS the pose like a
    STOP does (round 2 tested the raw request instead: the hold never engaged and a 10 s vx -0.3
    request drifted 0.296 m backwards on the policy's own zero-command drift, g1/check). A reverse
    request with a turn or a strafe in it keeps the turn/strafe and drops the reverse. So the
    AlohaMini runner's reverse-out-of-a-wedge recovery does NOT back a G1 out: it stands. A G1
    recovery has to turn in place instead. Only probes that pass `allow_untrained=True` ever walked
    backwards, and those fall at vx -0.4 / -0.6 within 1.1-1.5 s (vx -0.2 stands).
  * ZERO COMMAND IS NOT A STOP. The policy steps in place at (0, 0, 0) and drifts: body velocity
    (-0.032, +0.011, -0.006) measured here, -0.031 / -0.035 in Isaac Lab itself (the policy's own
    behaviour), ~0.64 m backwards over a 20 s STOP. G1Base(hold_on_stop=True) closes that with a
    small corrective command toward the pose latched when the zero command arrived (HOLD_* below).
    What it buys, measured as the MAXIMUM distance from the latched pose over the hold window (not
    the end point, which only says where inside the band the run happened to end): the robot stays
    within about HOLD_ENTER_M (5 cm) of the latch, cycling between drifting back on the policy's
    own ~0.03 m/s and a minimum forward correction (stats(): `hold_max_from_latch_m`; numbers in
    nav/results/g1/g1_regression_r4.json). It is a 5 cm band, not a 5 mm one. A robot that STOPS
    FROM A WALK overshoots the latch while braking (~0.2 m from 0.6 m/s) and, since it cannot be
    commanded backwards, comes back on that same ~0.03 m/s drift: for several seconds after a stop
    from speed it is still creeping backwards a few cm per second.

FALL DETECTION (`fallen()`), thresholds chosen against measured gait, not guessed:
  * pelvis height ABOVE THE LOWEST FOOT LINK < FALL_ABOVE_FEET_M (0.40 m). Measured walking value is
    reported in the run report (`above_feet`); it is relative to the feet, not to a floor height taken at
    spawn, so a ramp, a kerb or a staircase does not read as a fall (a floor-relative rule would call a
    robot that walked 0.3 m DOWN a slope "fallen"). A G1 that has sat or knelt has its pelvis at 0.3-0.4 m.
  * tilt of the pelvis z axis from vertical > FALL_TILT_DEG (40 deg). Measured tilt distribution
    (scratchpad/g1/runner/flat_all/report.json): pursuit-arc command streams <= 8.0 deg; in-range
    constant commands up to 18.5 deg (vx 0, vy -0.5, wz -1 -- strafe + spin leans the pelvis);
    randomised box-corner streams up to 19.6 deg without falling; the one fall recorded crossed
    36.6 deg and fell at 40.8 deg. No recorded run went past 20 deg and recovered.
  * WHICH RULE ENDS A G1 EPISODE: `fallen()` (40 deg / 0.40 m) -- the G1's own rule, reported as
    fault kind "fall" with the sub-rule that fired. The nav runner's AlohaMini rule (roll or pitch
    > 30 deg -> physics_fault) is NOT the G1's terminal rule; G1Base records when the pelvis first
    crossed it (`tilt30_first_t`) so the two can be compared. On the recorded data they agree on
    the outcome (nothing crossed 20 deg and recovered), so the choice moves a fall's TIME, not
    whether it happened.
  * pelvis height above the SPAWN floor (FALL_PELVIS_HEIGHT_M, 0.45 m) is still reported in `state()`
    and used when `floor_relative=True` (flat stages only).
  * FREE FALL is a separate fault (G1Base.physics_state(), kind "free_fall"): pelvis vertical speed
    below FREE_FALL_VZ_MPS (-1.5 m/s) continuously for FREE_FALL_HOLD_S (0.10 s) of sim time. Both
    fall rules above are about the robot's SHAPE, and a G1 dropping upright through the void keeps
    its shape: round 2 spawned over no floor and fallen() fired only after 4.75 s, at pelvis z
    -110.9 m (g1/check). Physics of the bound: free fall from rest reaches -1.5 m/s after 0.15 s
    (11 cm) and holding it 0.10 s more means >= 0.31 m of drop with nothing under the feet, more
    than the 0.135 m step-down the guard lets the robot take (guard.G1_STEP_DOWN_MAX_M). Measured
    margin on walked step-downs (0.135 / 0.20 m ledges at 0.5-1.0 m/s, guard off: vz_min and the
    longest run below the bound) is the "ledge" check in nav/results/g1/g1_regression_r4.json.
  * NO FLOOR under the start: G1Base.reset_to() still places the robot (so the pose is defined),
    but sets `no_floor`, records fault kind "no_floor" and RE-RECORDS it after every reset_stats(),
    so the runner's `if base.faults` rule ends the episode on its first step whatever order it
    resets in -- the same refusal run_navigation applies to AlohaMini (abort_reason "no_floor").
  * PHYSICS RATE: the policy is only in distribution at 200 Hz physics / 50 Hz policy. G1Base
    compares the step_dt omni.physx hands its PRE_STEP callback with the SPEC's physics_dt on every
    sub-step (`dt_mismatch_steps`, fault "physics_dt_mismatch"), and stats()["policy_hz"] is ticks
    per SIM second summed from those step_dt values, checked against the SPEC's policy_hz (50) --
    never against loco.decimation / loco.physics_dt, which are the values a glue bug would change
    (round 2 compared against loco.decimation, so a decimation-2 variant reported 100 Hz as OK).
  * torso contact > TORSO_CONTACT_N (1 N) is COUNTED, not a fall: it is Isaac Lab's own termination
    for this task (g1/rough_env_cfg.py:152 + velocity_env_cfg.py:271-273 illegal_contact on
    torso_link, threshold 1.0), i.e. the state the policy never saw past. For nav it means "walked
    into something with the chest". `post_physics_step()` reads it on EVERY physics sub-step (in
    callback mode from a PHYSICS_POST_STEP callback), so the counter works whichever way the
    caller steps; an earlier version counted it inside G1Base.apply() after the callback-mode
    return, where it could never increment.

PHYSICS HANDLE. Isaac Sim 6.0.1's Articulation._on_prim_deletion drops the articulation's physics
view on EVERY prim deletion anywhere in the stage (articulation.py:4974-4983 deletes _physics_view
without checking the path; XFormPrim's own handler does check). Camera.destroy() deletes render
product prims, so destroying ANY camera killed the G1 (AttributeError on the next tick, measured
by g1/check). `_ScopedArticulation` forwards only deletions of the robot's own prims or their
ancestors. If the view is lost anyway (timeline stop, robot deleted), the policy stops ticking and
G1Base reports fault kind "physics_handle_lost" -- an ERROR, not a fall. Still: create cameras
once per process and keep them.
"""

from __future__ import annotations

import json
import math
import sys
import time
import weakref
from pathlib import Path

import numpy as np

try:  # imported as nav/sim/g1 on sys.path, or as package `g1` from nav/sim
    from camera_geometry import camera_angles
    from walk_policy import G1WalkPolicy, base_frame_state
except ImportError:  # pragma: no cover
    from .camera_geometry import camera_angles
    from .walk_policy import G1WalkPolicy, base_frame_state

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# One copy of the headlight numbers: scripts/alohamini1_specs.py (measured there, see the long comment
# next to NAV_HEADLIGHT_INTENSITY).
sys.path.insert(0, str(REPO / "scripts"))
from alohamini1_specs import NAV_HEADLIGHT_INTENSITY, NAV_HEADLIGHT_RADIUS_M  # noqa: E402,F401  (AlohaMini's, for A/B tools)
RESULTS = REPO / "nav" / "results" / "g1"
# The deploy's inputs live NEXT TO THE CODE: nav/results/.gitignore ignores *.json and *.npz, so copies
# under nav/results/g1/ (where the prototype first wrote them; identical md5) would vanish from a checkout.
DATA = HERE / "data"
DEFAULT_POLICY_SPEC = DATA / "policy_spec.json"          # -> weights_npz "policy/policy_weights.npz" beside it
DEFAULT_ASSET_SPEC = DATA / "asset_spec.json"

# ---------------------------------------------------------------------------- thresholds
FALL_PELVIS_HEIGHT_M = 0.45   # above the SPAWN floor; flat stages only (see `fallen(floor_relative=True)`)
FALL_ABOVE_FEET_M = 0.40      # pelvis above the lowest foot link; terrain independent (module docstring)
FALL_TILT_DEG = 40.0
RUNNER_TILT_RULE_DEG = 30.0   # the nav runner's AlohaMini roll/pitch fault rule; RECORDED for the G1, not terminal
TORSO_CONTACT_N = 1.0
POLICY_HZ_TOLERANCE = 0.5     # runtime check: policy ticks per SIM second must be the SPEC's policy_hz +- this
STEP_DT_TOLERANCE_S = 1e-9    # a PhysX sub-step whose dt differs from the spec's physics_dt by more is a fault
FREE_FALL_VZ_MPS = -1.5       # pelvis vz below this ...
FREE_FALL_HOLD_S = 0.10       # ... continuously for this long (sim time) = free fall (module docstring)

# Pose hold for a zero command (G1Base(hold_on_stop=True)). P control toward the pose latched when
# the zero command arrived, inside the trained command box and never backwards. Gains/caps chosen
# small on purpose: the job is to cancel a 0.03 m/s drift, not to manoeuvre.
#
# Round-1 version: a 2 cm / 2 deg deadband and pure P. g1/check measured it settling at (-0.033 m,
# +0.022 m, 2.0 deg) -- just OUTSIDE its own deadband -- and sending ~(0.033, -0.022, +-0.037)
# forever: commands that small are below what the policy responds to (achieved vx ~ 0.91 cmd - 0.063,
# i.e. nothing forward below ~0.07 m/s), so they neither moved the robot nor went to zero.
# Now HYSTERESIS + a MINIMUM command: correction starts when the error leaves the ENTER band, is sent
# at no less than HOLD_MIN_* (large enough to move the robot), and stops (exact zero command) once the
# error is inside the EXIT band. Inside the bands the command is exactly 0.
HOLD_K_XY = 1.0               # (m/s) per m of position error, body frame
HOLD_K_YAW = 1.0              # (rad/s) per rad of heading error
HOLD_MAX_VX, HOLD_MAX_VY, HOLD_MAX_WZ = 0.3, 0.3, 0.5
HOLD_MIN_VX, HOLD_MIN_VY, HOLD_MIN_WZ = 0.15, 0.12, 0.15
HOLD_ENTER_M, HOLD_EXIT_M = 0.05, 0.02
HOLD_ENTER_RAD, HOLD_EXIT_RAD = math.radians(4.0), math.radians(1.5)

# Command slew limit (applied at every policy tick, in SIM time). g1/prototype stress: 1 of 9 randomised
# 60 s command streams fell -- a box-corner stream that flipped vy and wz sign within 0.3-0.5 s at vx 0
# -- while 0 of 3 pursuit-arc streams (smooth, unicycle) fell. Training resampled commands as steps only
# every 10 s, so second-scale sign reversals are outside what it saw. Rates: wz 2 rad/s^2 (a full
# -1 -> +1 reversal takes 1 s), vy 1 m/s^2 (-0.5 -> +0.5 in 1 s), vx 2 m/s^2 up; vx DOWN 4 m/s^2 so
# braking (the collision guard's stop) adds at most 0.25 s from full speed. None = off (parity probes).
DEFAULT_SLEW = {"vx_up": 2.0, "vx_down": 4.0, "vy": 1.0, "wz": 2.0}

# G1 head headlight (author_head_light). The SAME fixture type as AlohaMini's camera_nav headlight
# (DiskLight, colour (1, .98, .92)), but NOT its numbers, which belong to a lens 0.354 m above the floor:
#  * radius 0.10 m, not 0.30: a 0.30 m disk centred on the G1's lens reaches 0.16 m down the
#    torso and into the forearms (the hands sit up to 0.39 m in front of the pelvis), and its centre
#    sits inside the head shell (the lens is 13.7 mm behind the front-most head surface; mesh
#    vertices within 5 cm of the view axis end 13-18 mm ahead of the lens). Light emitted inside the
#    shell made the head render as a saturated white oval in every chase frame (g1/check).
#  * centred G1_HEADLIGHT_FORWARD_M ahead of the lens ALONG THE VIEW AXIS, clear of the shell, and
#    `visibleInPrimaryRay = False` so no camera sees the emitter itself.
#  * intensity calibrated for the G1 against AlohaMini's camera + headlight at the same poses
#    (nav/tools/g1_render_check.py, report in nav/results/g1/): see G1_HEADLIGHT_INTENSITY.
G1_HEADLIGHT_RADIUS_M = 0.10
G1_HEADLIGHT_FORWARD_M = 0.06
# Calibrated 2026-10-05 (nav/tools/g1_render_check.py, scratchpad/g1/repair1/render/report.json "light_pick"):
# mean luminance of a 1920x1080 frame, G1 head cam + this light vs AlohaMini camera_nav optics + AlohaMini's
# 45000 / r 0.30 light, same pelvis pose, each light alone, 150 updates to settle:
#   pose                       no light G1/Aloha   AlohaMini lit   G1 lit at x2.25 (this value)   ratio
#   hospital_down_hallway        20.0 / 27.5           82.4               91.5                  1.11
#   10 m down that corridor      14.1 / 14.0           85.2               92.4                  1.085
#   hospital_vending_machine    144.1 / 162.6         165.9              151.9                  0.916
#   hospital_exit_room          106.2 / 113.5         159.6              171.6                  1.076
# 405000 (= AlohaMini's light power at 1/9 the disk area) gave 0.68-0.93: the G1's lens is 1.09 m up, so
# the same power lands on floor three times as far away. x1.5 (607500) was also inside +-20% at all four
# poses (0.86-1.01); x2.25 has the smaller mean |log ratio| (0.087 vs 0.096). Window-lit rooms barely move.
G1_HEADLIGHT_INTENSITY = 911250.0
FOOT_CONTACT_N = 5.0          # a foot "in contact" (32 kg robot: stance foot carries ~150-300 N)
SPAWN_HEIGHT_M = 0.74         # G1_CFG.init_state.pos z (unitree.py:288); stands at 0.7216 -> 1.8 cm drop
SPAWN_PROBE_ABOVE_M = 0.30    # floor raycast starts this far above the episode z (base_drive.SPAWN_PROBE_ABOVE_M:
                              # the first 0.30 m above a DynaNav start cannot hold an umbrella or a canopy;
                              # a higher start of the ray could land the robot ON one)
WALK_PELVIS_HEIGHT_M = 0.63   # measured pelvis height above the floor while standing/walking (policy_notes.md,
                              # flat_v3/hospital_v2 reports: 0.605-0.660, mean 0.630)
SPAWN_WALKABLE_NORMAL_Z = 0.7

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_LINK = "torso_link"
HEAD_LINK = "head_link"


def load_specs(policy_spec: str | Path | None = None, asset_spec: str | Path | None = None):
    pspec = Path(policy_spec or DEFAULT_POLICY_SPEC)
    aspec = Path(asset_spec or DEFAULT_ASSET_SPEC)
    return pspec, json.loads(aspec.read_text())


# ---------------------------------------------------------------------------- small math
def yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def quat_to_matrix_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_roll_pitch_wxyz(q) -> tuple[float, float, float]:
    w, x, y, z = (float(c) for c in q)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return yaw, roll, pitch


def camera_mount_matrix(pitch_down_rad: float) -> np.ndarray:
    """Rotation of a USD camera (looks -Z, image-up +Y) inside a link whose +X is forward and
    +Z up, pitched DOWN by pitch_down_rad about the link's +Y (URDF convention: positive rpy
    pitch turns +X toward -Z). Same construction as nav/tools/probe_g1_asset.py, which the
    asset spec's renders were made with."""
    c, s = math.cos(pitch_down_rad), math.sin(pitch_down_rad)
    rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    m_level = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)  # cols: cam x=-Y, y=+Z, z=-X
    return rot_y @ m_level


def matrix_to_quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)


# ---------------------------------------------------------------------------- stage setup
def physics_scene_prims(stage) -> list:
    from pxr import UsdPhysics

    out = []
    for top in stage.GetPseudoRoot().GetChildren():
        for prim in (top, *top.GetChildren()):
            if prim.IsA(UsdPhysics.Scene):
                out.append(prim)
    return out


def configure_physics_scene(stage, physics_dt: float = 0.005, scene_path: str = "/World/PhysicsScene",
                            cpu: bool = True, broadphase: str = "MBP") -> list[str]:
    """Author the trained physics rate (and CPU dynamics) on every physics scene, in the SESSION
    layer so an opened nav_<env>.usda is never dirtied. Creates `scene_path` if the stage has
    none. Mirrors what isaacsim.core.api World(physics_dt=0.005, backend="numpy") writes -- the
    configuration the walk was validated under -- without needing a World. Returns change lines.
    """
    import carb
    from pxr import Gf, PhysxSchema, Usd, UsdPhysics

    changes: list[str] = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        scenes = physics_scene_prims(stage)
        if not scenes:
            sc = UsdPhysics.Scene.Define(stage, scene_path)
            sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
            sc.CreateGravityMagnitudeAttr(9.81)
            scenes = [sc.GetPrim()]
            changes.append(f"created {scene_path}")
        for prim in scenes:
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            want = {
                "TimeStepsPerSecond": int(round(1.0 / physics_dt)),
                "SolverType": "TGS",
                "EnableStabilization": False,
            }
            if cpu:
                want.update({"EnableGPUDynamics": False, "BroadphaseType": broadphase})
            else:
                # GPU dynamics + GPU broadphase, as the nav stages ship for AlohaMini and as Isaac Lab
                # trained. Readback stays ON (below), so the numpy tensor API keeps working exactly as
                # on CPU (SimulationManager reports device "cpu" while suppressReadback is False).
                want.update({"EnableGPUDynamics": True, "BroadphaseType": "GPU"})
            for name, value in want.items():
                attr = getattr(api, f"Get{name}Attr")()
                before = attr.Get() if attr and attr.IsValid() else None
                getattr(api, f"Create{name}Attr")().Set(value)
                changes.append(f"{prim.GetPath()}.physxScene:{name[0].lower() + name[1:]} {before} -> {value}")
    carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
    return changes


def author_g1(stage, prim_path: str, position, yaw: float, usd_url: str, policy_spec: dict,
              foot_friction_combine: str = "max") -> dict:
    """Reference the G1 and apply Isaac Lab's spawn-time properties. Call BEFORE play, with the
    session layer as edit target if the stage is a file that must stay clean."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    rec: dict = {"prim": prim_path, "usd": usd_url}
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_url)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, position)))
    q = yaw_quat_wxyz(yaw)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))

    bodies, roots, colliders = [], [], []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(p)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(p)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(str(p.GetPath()))
    if len(roots) != 1:
        raise RuntimeError(f"expected one articulation root under {prim_path}, found {[str(r.GetPath()) for r in roots]}")
    rec["articulation_root"] = str(roots[0].GetPath())
    rec["num_bodies"] = len(bodies)
    rec["colliders"] = colliders

    rbp = policy_spec["robot"]["rigid_body_props"]
    for b in bodies:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(b)
        rb.CreateDisableGravityAttr(False)
        rb.CreateRetainAccelerationsAttr(bool(rbp["retain_accelerations"]))
        rb.CreateLinearDampingAttr(float(rbp["linear_damping"]))
        rb.CreateAngularDampingAttr(float(rbp["angular_damping"]))
        rb.CreateMaxLinearVelocityAttr(float(rbp["max_linear_velocity"]))
        rb.CreateMaxAngularVelocityAttr(float(rbp["max_angular_velocity_deg_s"]))
        rb.CreateMaxDepenetrationVelocityAttr(float(rbp["max_depenetration_velocity"]))
    ap = policy_spec["robot"]["articulation_props"]
    pa = PhysxSchema.PhysxArticulationAPI.Apply(roots[0])
    pa.CreateEnabledSelfCollisionsAttr(bool(ap["enabled_self_collisions"]))
    pa.CreateSolverPositionIterationCountAttr(int(ap["solver_position_iteration_count"]))
    pa.CreateSolverVelocityIterationCountAttr(int(ap["solver_velocity_iteration_count"]))

    # Foot / body material (module docstring, "FOOT FRICTION"). Bound on every rigid body with
    # strongerThanDescendants, so it reaches the collider meshes underneath even if they are
    # instance proxies, and overrides anything the asset bound.
    rm = policy_spec["sim"]["robot_material"]
    mat_path = Sdf.Path(prim_path + "_PhysicsMaterial")
    mat = UsdShade.Material.Define(stage, mat_path)
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(float(rm["static_friction"]))
    mapi.CreateDynamicFrictionAttr(float(rm["dynamic_friction"]))
    mapi.CreateRestitutionAttr(float(rm["restitution"]))
    pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pm.CreateFrictionCombineModeAttr(foot_friction_combine)
    pm.CreateRestitutionCombineModeAttr("min")
    for b in bodies:
        UsdShade.MaterialBindingAPI.Apply(b).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    rec["robot_material"] = {"path": str(mat_path), "static": rm["static_friction"],
                             "dynamic": rm["dynamic_friction"], "restitution": rm["restitution"],
                             "friction_combine": foot_friction_combine, "restitution_combine": "min"}

    # Contact reporting on the feet and torso (threshold 0: report every contact).
    rep = []
    for name in (*FOOT_LINKS, TORSO_LINK):
        bp = stage.GetPrimAtPath(f"{prim_path}/{name}")
        if bp.IsValid():
            cr = PhysxSchema.PhysxContactReportAPI.Apply(bp)
            cr.CreateThresholdAttr(0.0)
            rep.append(str(bp.GetPath()))
    rec["contact_report_bodies"] = rep
    return rec


def author_head_camera(stage, robot_prim: str, cam_spec: dict, name: str = "g1_head_cam") -> str:
    """A UsdGeom.Camera on the G1's head_link at the spec'd mount, optics and pitch."""
    from pxr import Gf, UsdGeom

    parent = f"{robot_prim}/{cam_spec['parent_link_rel']}"
    if not stage.GetPrimAtPath(parent).IsValid():
        raise RuntimeError(f"no camera parent {parent}")
    path = f"{parent}/{name}"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(float(cam_spec["focal_length_mm"]))
    cam.CreateHorizontalApertureAttr(float(cam_spec["horizontal_aperture_mm"]))
    cam.CreateVerticalApertureAttr(float(cam_spec["vertical_aperture_mm"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(*map(float, cam_spec["clipping_range_m"])))
    cam.CreateProjectionAttr("perspective")
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, cam_spec["local_pos_m_in_head_link"])))
    q = matrix_to_quat_wxyz(camera_mount_matrix(math.radians(float(cam_spec["pitch_down_deg_rel_head"]))))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    return path


def author_head_light(stage, robot_prim: str, cam_spec: dict, name: str = "g1_head_cam_headlight",
                      intensity: float | None = None, radius_m: float | None = None,
                      forward_m: float | None = None) -> str:
    """The G1's counterpart of AlohaMini's camera_nav headlight (add_cameras.py:205-221): a DiskLight
    aimed exactly like the head camera, `forward_m` ahead of the lens along the view axis (outside
    the head shell), not visible to primary rays. Defaults: the G1_HEADLIGHT_* constants above.

    NOT part of the real G1 -- a rendering parity measure. DynaNav's environments light themselves
    poorly: a corridor with no light renders at ~14/255 mean pixel value, with AlohaMini's fixture
    ~88/255 (alohamini1_specs.py, NAV_HEADLIGHT_INTENSITY). A G1 frame without it is the dark one, and
    the policy was never shown a frame like that on any other robot in this repo. A UsdLux light
    emits along local -Z exactly as a UsdGeom.Camera looks down local -Z, so it takes the camera's
    own orientation verbatim."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    parent = f"{robot_prim}/{cam_spec['parent_link_rel']}"
    path = f"{parent}/{name}"
    light = UsdLux.DiskLight.Define(stage, path)
    light.CreateIntensityAttr(float(G1_HEADLIGHT_INTENSITY if intensity is None else intensity))
    light.CreateRadiusAttr(float(G1_HEADLIGHT_RADIUS_M if radius_m is None else radius_m))
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))
    light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    rot = camera_mount_matrix(math.radians(float(cam_spec["pitch_down_deg_rel_head"])))
    view_axis = rot @ np.array([0.0, 0.0, -1.0])
    fwd = G1_HEADLIGHT_FORWARD_M if forward_m is None else float(forward_m)
    pos = np.asarray(cam_spec["local_pos_m_in_head_link"], dtype=float) + fwd * view_axis
    xf = UsdGeom.Xformable(light)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    q = matrix_to_quat_wxyz(rot)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    return path


# ---------------------------------------------------------------------------- stepping
class PhysicsStepper:
    """One synchronous PhysX step of exactly `dt` (omni.physics.core simulate(); in 6.0.1 it
    blocks until results are written -- the same call isaacsim.core.api World.step(render=False)
    makes). Keeps its own sim clock. No rendering, no app update."""

    def __init__(self, dt: float):
        import omni.physics.core

        self.dt = float(dt)
        self.sim = omni.physics.core.get_physics_simulation_interface()
        self.t = 0.0
        self.steps = 0

    def __call__(self) -> None:
        self.sim.simulate(self.dt, self.t)
        self.t += self.dt
        self.steps += 1


def render_only(app, physics_dt: float = 0.005, sim_time: float = 0.0) -> None:
    """Update Kit (renderers, annotators, viewports) WITHOUT stepping physics -- what
    isaacsim.core.api SimulationContext.render() does: push PhysX results to Fabric, then one
    app.update() with /app/player/playSimulations off. A plain kit.update() while the timeline
    plays would ALSO advance PhysX by the app's own clock, on top of our explicit 5 ms steps."""
    import carb
    import omni.kit.app

    settings = carb.settings.get_settings()
    em = omni.kit.app.get_app().get_extension_manager()
    if em.is_extension_enabled("omni.physx.fabric"):
        try:
            from omni.physxfabric import get_physx_fabric_interface

            get_physx_fabric_interface().force_update(physics_dt, sim_time)
        except Exception:  # noqa: BLE001
            pass
    settings.set_bool("/app/player/playSimulations", False)
    app.update()
    settings.set_bool("/app/player/playSimulations", True)


def floor_below(x: float, y: float, z_from: float, skip_prefix: str = "/World/G1",
                max_dist: float = 10.0, max_hits: int = 60) -> tuple[float, str] | None:
    """First upward-facing collider below (x, y, z_from), skipping `skip_prefix` prims and steep
    faces. raycast_closest iterated past rejected hits (raycast_all can segfault Kit on the
    outdoor stage). Needs the physics scene parsed, i.e. after play.

    A ray that STARTS inside a shape (the G1's own torso box, when the probe height lands in
    it) reports that shape at distance 0 with the origin as the hit point; stepping past it by
    millimetres then never leaves the box. Zero-distance hits step 3 cm instead."""
    import carb
    from omni.physx import get_physx_scene_query_interface

    q = get_physx_scene_query_interface()
    z = z_from
    for _ in range(max_hits):
        hit = q.raycast_closest(carb.Float3(x, y, z), carb.Float3(0.0, 0.0, -1.0), max_dist)
        if not hit or not hit.get("hit"):
            return None
        prim = str(hit.get("collision", ""))
        hz = float(hit["position"][2])
        nz = float(hit["normal"][2])
        if not prim.startswith(skip_prefix) and nz >= SPAWN_WALKABLE_NORMAL_Z:
            return hz, prim
        z = (z - 0.03) if float(hit.get("distance", 0.0)) < 1e-4 else (hz - 0.005)
    return None


def g1_spawn_pose(start, robot_prim: str = "/World/G1") -> tuple[tuple[float, float, float], dict]:
    """(x, y, floor + SPAWN_HEIGHT_M) for the pelvis -- the G1 counterpart of base_drive's
    ground_spawn_pose(), which is AlohaMini geometry (three tyres, base_link rest height).
    The policy was trained from a 0.74 m spawn over a floor at 0, so the same drop (1.8 cm) is
    reproduced over whatever floor is under the start."""
    x, y, z = (float(v) for v in start)
    f = floor_below(x, y, z + SPAWN_PROBE_ABOVE_M, skip_prefix=robot_prim)
    if f is None:
        return (x, y, z + SPAWN_HEIGHT_M), {"episode_z": z, "ground_z": None, "ground_prim": "",
                                            "snapped_z": z + SPAWN_HEIGHT_M}
    gz, prim = f
    return (x, y, gz + SPAWN_HEIGHT_M), {"episode_z": round(z, 4), "ground_z": round(gz, 4),
                                         "ground_prim": prim, "snapped_z": round(gz + SPAWN_HEIGHT_M, 4)}


# ---------------------------------------------------------------------------- articulation
def _scoped_articulation(root_path: str, own_prefixes: list[str], name: str = "g1"):
    """isaacsim.core.prims.Articulation whose physics view survives deletions of UNRELATED prims
    (module docstring, PHYSICS HANDLE). `own_prefixes` is read at deletion time, so the caller can
    extend it after initialize() (with the link paths)."""
    from isaacsim.core.prims import Articulation

    class _ScopedArticulation(Articulation):
        def _on_prim_deletion(self, prim_path: str) -> None:
            own = getattr(self, "_g1_own_prefixes", None) or []
            p = str(prim_path).rstrip("/") or "/"
            relevant = p == "/" or not own or any(o == p or o.startswith(p + "/") for o in own)
            if relevant:
                self._g1_deletions_forwarded = getattr(self, "_g1_deletions_forwarded", 0) + 1
                return Articulation._on_prim_deletion(self, prim_path)
            self._g1_deletions_ignored = getattr(self, "_g1_deletions_ignored", 0) + 1
            return None

    art = _ScopedArticulation(root_path, name=name)
    art._g1_own_prefixes = own_prefixes
    return art


# ---------------------------------------------------------------------------- controller
class G1Locomotion:
    """The walking G1: policy + glue + measurements. Holds no simulation loop of its own."""

    def __init__(self, prim_path: str = "/World/G1", policy_spec: str | Path | None = None,
                 asset_spec: str | Path | None = None):
        pspec_path, self.asset_spec = load_specs(policy_spec, asset_spec)
        self.policy = G1WalkPolicy(pspec_path)
        self.spec = self.policy.spec
        self.prim_path = prim_path
        self.root_path = f"{prim_path}/{self.asset_spec['asset']['articulation_root_rel']}"
        self.usd_url = self.asset_spec["decision"]["spawn_usd"]
        self.physics_dt = float(self.spec["sim"]["physics_dt"])
        self.decimation = int(self.spec["sim"]["decimation"])
        rng = self.spec["commands"]["trained_ranges"]
        self.cmd_lo = np.array([rng["lin_vel_x"][0], rng["lin_vel_y"][0], rng["ang_vel_z"][0]])
        self.cmd_hi = np.array([rng["lin_vel_x"][1], rng["lin_vel_y"][1], rng["ang_vel_z"][1]])
        self.cmd = np.zeros(3)              # what the POLICY sees (after the slew limit)
        self.cmd_target = np.zeros(3)       # clamped command from set_command(); cmd follows it
        self.cmd_requested = np.zeros(3)
        self.slew: dict | None = None       # DEFAULT_SLEW-shaped dict, or None (cmd = target at once)
        self.slew_limited_ticks = 0
        self.clamped_commands = 0
        self.reverse_requests = 0
        self.pre_tick_hook = None           # called right before each policy tick (G1Base: hold recompute)
        self.pre_tick_hook_errors = 0
        self.floor_z = 0.0
        self.art = None
        self.n = 0
        self._count = 0
        self.ticks = 0
        self.sim_steps = 0              # physics steps seen by pre_physics_step (the sim clock)
        self.post_steps = 0             # ... and by post_physics_step; equal when both hooks are wired
        self.last_tick: dict | None = None
        self._contact = None
        self._contact_names: list[str] = []
        self._torso_cv = None
        self.handle_lost_at: float | None = None
        self.skipped_ticks = 0
        self.reset_contact_stats()
        self.record: dict = {}

    def sim_t(self) -> float:
        return self.sim_steps * self.physics_dt

    def reset_contact_stats(self) -> None:
        self.torso_contact_steps = 0    # physics sub-steps with net torso force > TORSO_CONTACT_N
        self.torso_contact_events = 0   # rising edges of that
        self.torso_contact_first_t: float | None = None
        self.torso_contact_max_n = 0.0
        self.torso_contact_read_errors = 0
        self._torso_in_contact = False

    # ------------------------------------------------------------------ authoring (pre-play)
    def author(self, stage, position, yaw: float, foot_friction_combine: str = "max") -> dict:
        self.record["author"] = author_g1(stage, self.prim_path, position, yaw, self.usd_url, self.spec,
                                          foot_friction_combine=foot_friction_combine)
        return self.record["author"]

    # ------------------------------------------------------------------ after play
    def initialize(self) -> dict:
        self._own_prefixes = [self.prim_path]
        self.art = _scoped_articulation(self.root_path, self._own_prefixes, name="g1")
        self.art.initialize()
        self.handle_lost_at = None
        names = list(self.art.dof_names)
        self.n = len(names)
        sim_idx = self.policy.bind(names)
        self.body_names = list(self.art.body_names)
        # The robot's own link prims (plus the robot prim above): deleting one of these, or an
        # ancestor, is the only deletion that may drop the physics view.
        self._own_prefixes.extend(f"{self.prim_path}/{b}" for b in self.body_names)
        self._head_idx = self.body_names.index(HEAD_LINK)
        self._foot_idx = [self.body_names.index(b) for b in FOOT_LINKS]

        act = self.spec["actuators_policy_order"]
        to_sim = self.policy.to_sim_order
        kp, kd = to_sim(np.array(act["stiffness"]), self.n), to_sim(np.array(act["damping"]), self.n)
        eff, arm = to_sim(np.array(act["effort_limit"]), self.n), to_sim(np.array(act["armature"]), self.n)
        self.art.set_gains(kps=kp.reshape(1, -1), kds=kd.reshape(1, -1))
        self.art.set_max_efforts(eff.reshape(1, -1))
        self.art.set_armatures(arm.reshape(1, -1))
        cur_v = np.asarray(self.art.get_joint_max_velocities()).reshape(-1)
        vlim = act.get("velocity_limit")
        if vlim is not None:
            v = np.array([cur_v[sim_idx[k]] if vl is None else vl for k, vl in enumerate(vlim)], dtype=float)
            self.art.set_max_joint_velocities(to_sim(v, self.n, fill=cur_v).reshape(1, -1))
        self.q_default_sim = to_sim(self.policy.default_q, self.n)
        self.q_reset_sim = to_sim(self.policy.reset_q, self.n)

        # Read everything back: a gain that did not land is the silent failure this guards.
        kps, kds = (np.asarray(g).reshape(-1) for g in self.art.get_gains())
        rb = {
            "dof_names_equal_policy_order": names == self.policy.joint_names,
            "kp_maxabs_err": float(np.abs(kps - kp).max()),
            "kd_maxabs_err": float(np.abs(kds - kd).max()),
            "effort_maxabs_err": float(np.abs(np.asarray(self.art.get_max_efforts()).reshape(-1) - eff).max()),
            "armature_maxabs_err": float(np.abs(np.asarray(self.art.get_armatures()).reshape(-1) - arm).max()),
            "num_dof": self.n, "num_bodies": len(self.body_names),
        }
        try:
            mats = np.asarray(self.art._physics_view.get_material_properties()).reshape(-1, 3)
            rb["shape_materials_static_dynamic_rest"] = mats.round(4).tolist()
        except Exception as e:  # noqa: BLE001
            rb["shape_materials_static_dynamic_rest"] = repr(e)
        self.record["initialize"] = rb

        # Contact forces on feet + torso, through the tensor API (PhysxContactReportAPI was
        # applied in author()). Optional: everything else works without it.
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            psv = SimulationManager.get_physics_sim_view()
            paths = [f"{self.prim_path}/{b}" for b in (*FOOT_LINKS, TORSO_LINK)]
            self._contact = [psv.create_rigid_contact_view(p) for p in paths]
            self._contact_names = [*FOOT_LINKS, TORSO_LINK]
            self._torso_cv = self._contact[-1]
            self.record["contact_view"] = "ok"
        except Exception as e:  # noqa: BLE001
            self._contact = None
            self._torso_cv = None
            self.record["contact_view"] = f"unavailable: {e!r}"
        return rb

    def handle_ok(self) -> bool:
        """The articulation's physics view exists (see PHYSICS HANDLE in the module docstring)."""
        return self.art is not None and getattr(self.art, "_physics_view", None) is not None

    # ------------------------------------------------------------------ reset / command
    def reset_to(self, position, yaw: float, floor_z: float | None = None) -> None:
        """Pelvis at `position` facing `yaw`, training's reset joint pose, everything at rest."""
        if floor_z is not None:
            self.floor_z = float(floor_z)
        q = yaw_quat_wxyz(yaw)
        self.art.set_world_poses(positions=np.array([list(map(float, position))]),
                                 orientations=np.array([list(q)]))
        self.art.set_joint_positions(self.q_reset_sim.reshape(1, -1))
        self.art.set_joint_velocities(np.zeros((1, self.n)))
        self.art.set_velocities(np.zeros((1, 6)))
        self.art.set_joint_position_targets(self.q_default_sim.reshape(1, -1))
        self.policy.reset()
        self._count = 0
        self.cmd[:] = 0.0
        self.cmd_target[:] = 0.0
        self.last_tick = None

    def clamp(self, vx: float, vy: float, wz: float, allow_untrained: bool = False,
              count: bool = True) -> np.ndarray:
        """The command clipped to the TRAINED box; with `count`, a reverse request (vx < 0) and any
        clipped command are counted (`reverse_requests`, `clamped_commands`). G1Base.apply() calls
        this first, to decide hold-or-walk on what the policy would actually be given."""
        req = np.array([vx, vy, wz], dtype=float)
        if count:
            self.cmd_requested = req
            if vx < 0.0:
                self.reverse_requests += 1
        c = req.copy() if allow_untrained else np.clip(req, self.cmd_lo, self.cmd_hi)
        if count and np.any(np.abs(c - req) > 1e-9):
            self.clamped_commands += 1
        return c

    def set_command(self, vx: float, vy: float, wz: float, allow_untrained: bool = False,
                    count: bool = True) -> tuple[float, float, float]:
        """Clamp to the TRAINED command box (counting what was clamped unless `count=False`, for
        commands G1Base generates itself) and make it the policy's target. `allow_untrained` skips
        the clamp -- only for probes of what happens outside it (e.g. reverse)."""
        c = self.clamp(vx, vy, wz, allow_untrained=allow_untrained, count=count)
        self.cmd_target = c
        if self.slew is None:
            self.cmd = c.copy()
        return tuple(float(v) for v in c)

    def _slew_step(self) -> None:
        """Move `cmd` toward `cmd_target` by at most rate * policy period per axis (DEFAULT_SLEW)."""
        if self.slew is None:
            self.cmd = self.cmd_target.copy()
            return
        dt = self.physics_dt * self.decimation
        d = self.cmd_target - self.cmd
        lo = np.array([-self.slew["vx_down"], -self.slew["vy"], -self.slew["wz"]]) * dt
        hi = np.array([self.slew["vx_up"], self.slew["vy"], self.slew["wz"]]) * dt
        step = np.clip(d, lo, hi)
        if np.any(np.abs(step - d) > 1e-12):
            self.slew_limited_ticks += 1
        self.cmd = self.cmd + step

    # ------------------------------------------------------------------ stepping hooks
    def pre_physics_step(self) -> bool:
        """Call before EVERY physics step. Runs the policy on every `decimation`-th call and
        holds the PD targets otherwise. Returns True on the steps that ticked. With the physics
        view gone it ticks nothing (there is nothing to command) and records when that started."""
        ticked = False
        self.sim_steps += 1
        if self._count % self.decimation == 0:
            if self.handle_ok():
                if self.pre_tick_hook is not None:
                    try:
                        self.pre_tick_hook()
                    except Exception:  # noqa: BLE001  (never raise from a physics callback)
                        self.pre_tick_hook_errors = getattr(self, "pre_tick_hook_errors", 0) + 1
                self._slew_step()
                self.tick()
                ticked = True
            else:
                self.skipped_ticks += 1
                if self.handle_lost_at is None:
                    self.handle_lost_at = round(self.sim_t(), 4)
        self._count += 1
        return ticked

    def post_physics_step(self) -> None:
        """Call after EVERY physics step: torso contact accounting (module docstring)."""
        self.post_steps += 1
        if self._torso_cv is None or not self.handle_ok():
            return
        try:
            f = np.asarray(self._torso_cv.get_net_contact_forces(self.physics_dt)).reshape(-1, 3)
            fn = float(np.linalg.norm(f.sum(axis=0)))
        except Exception:  # noqa: BLE001  (never raise from a physics callback)
            self.torso_contact_read_errors += 1
            return
        if fn > TORSO_CONTACT_N:
            self.torso_contact_steps += 1
            if self.torso_contact_first_t is None:
                self.torso_contact_first_t = round(self.sim_t(), 4)
            if not self._torso_in_contact:
                self.torso_contact_events += 1
            self._torso_in_contact = True
            self.torso_contact_max_n = max(self.torso_contact_max_n, fn)
        else:
            self._torso_in_contact = False

    def tick(self) -> None:
        pos, quat = self.art.get_world_poses()
        pos, quat = np.asarray(pos).reshape(-1), np.asarray(quat).reshape(-1)
        vel = np.asarray(self.art.get_velocities()).reshape(-1)
        q = np.asarray(self.art.get_joint_positions()).reshape(-1)
        dq = np.asarray(self.art.get_joint_velocities()).reshape(-1)
        lin_b, ang_b, grav_b = base_frame_state(quat, vel[:3], vel[3:])
        target, action, _obs = self.policy.step(lin_b, ang_b, grav_b, self.cmd, q, dq)
        self.art.set_joint_position_targets(self.policy.to_sim_order(target, self.n, fill=self.q_default_sim).reshape(1, -1))
        self.ticks += 1
        self.last_tick = {"pos": pos, "quat": quat, "lin_b": lin_b, "ang_b": ang_b, "grav_b": grav_b,
                          "action_absmax": float(np.abs(action).max())}

    def advance(self, physics_step, n_steps: int) -> None:
        for _ in range(n_steps):
            self.pre_physics_step()
            physics_step()
            self.post_physics_step()

    # ------------------------------------------------------------------ measurements
    def state(self) -> dict:
        pos, quat = self.art.get_world_poses()
        pos, quat = np.asarray(pos).reshape(-1), np.asarray(quat).reshape(-1)
        vel = np.asarray(self.art.get_velocities()).reshape(-1)
        lin_b, ang_b, grav_b = base_frame_state(quat, vel[:3], vel[3:])
        yaw, roll, pitch = yaw_roll_pitch_wxyz(quat)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(grav_b[2])))))
        lt = self.link_poses()
        above_feet = float(pos[2] - lt[self._foot_idx, 2].min())
        return {"pos": pos, "quat": quat, "yaw": yaw, "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch), "tilt_deg": tilt, "lin_w": vel[:3], "ang_w": vel[3:],
                "lin_b": lin_b, "ang_b": ang_b, "grav_b": grav_b, "height": float(pos[2] - self.floor_z),
                "above_feet": above_feet}

    def pose(self) -> tuple[float, float, float, float]:
        pos, quat = self.art.get_world_poses()
        pos, quat = np.asarray(pos).reshape(-1), np.asarray(quat).reshape(-1)
        return float(pos[0]), float(pos[1]), float(pos[2]), yaw_roll_pitch_wxyz(quat)[0]

    def fallen(self, state: dict | None = None, floor_relative: bool = False) -> tuple[bool, list[str]]:
        s = state or self.state()
        why = []
        if s["above_feet"] < FALL_ABOVE_FEET_M:
            why.append(f"pelvis {s['above_feet']:.3f} m above the feet < {FALL_ABOVE_FEET_M}")
        if floor_relative and s["height"] < FALL_PELVIS_HEIGHT_M:
            why.append(f"pelvis {s['height']:.3f} m above the spawn floor < {FALL_PELVIS_HEIGHT_M}")
        if s["tilt_deg"] > FALL_TILT_DEG:
            why.append(f"tilt {s['tilt_deg']:.1f} deg > {FALL_TILT_DEG}")
        return bool(why), why

    def contact_forces(self) -> dict | None:
        """Net contact force magnitude (N) on each foot and the torso over the LAST physics step."""
        if not self._contact:
            return None
        out = {}
        for name, cv in zip(self._contact_names, self._contact):
            f = np.asarray(cv.get_net_contact_forces(self.physics_dt)).reshape(-1, 3)
            out[name] = float(np.linalg.norm(f.sum(axis=0)))
        return out

    def link_pose(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        lt = np.asarray(self.art._physics_view.get_link_transforms()).reshape(-1, 7)  # x y z qx qy qz qw
        p = lt[idx, :3]
        qx, qy, qz, qw = lt[idx, 3:]
        return p, np.array([qw, qx, qy, qz])

    def link_poses(self) -> np.ndarray:
        return np.asarray(self.art._physics_view.get_link_transforms()).reshape(-1, 7)

    def camera_pose(self, cam_spec: dict, link_tf: np.ndarray | None = None) -> dict:
        """World pose of a head camera from head_link's PhysX pose + the spec'd mount, and the
        numbers the arc menu needs, in camera_geometry's convention so they go straight into
        project_ground(): `menu_args` = dict(cam_h, cam_x, cam_y, pitch_deg, roll_deg, yaw_deg)
        (height above the floor; lens forward/left of the pelvis in the heading frame; pitch down,
        roll about the view axis with + = image right side down, view yaw relative to the pelvis
        heading with + = left). The older keys stay for the tools that log them; `roll_deg` now
        carries project_ground's sign (it used to be the negative)."""
        lt = self.link_poses() if link_tf is None else link_tf
        p = lt[self._head_idx, :3]
        qx, qy, qz, qw = lt[self._head_idx, 3:]
        r_link = quat_to_matrix_wxyz((qw, qx, qy, qz))
        r_cam = r_link @ camera_mount_matrix(math.radians(float(cam_spec["pitch_down_deg_rel_head"])))
        eye = p + r_link @ np.asarray(cam_spec["local_pos_m_in_head_link"], dtype=float)
        view = -r_cam[:, 2]
        pelvis = self.art.get_world_poses()
        ppos = np.asarray(pelvis[0]).reshape(-1)
        pyaw = yaw_roll_pitch_wxyz(np.asarray(pelvis[1]).reshape(-1))[0]
        c, s = math.cos(-pyaw), math.sin(-pyaw)
        rz_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        fwd, left, _ = rz_inv @ (eye - ppos)
        pitch, roll, yaw_off = camera_angles(rz_inv @ r_cam)
        h = float(eye[2] - self.floor_z)
        return {"eye": eye, "height": h, "pitch_down_deg": pitch, "roll_deg": roll, "yaw_offset_deg": yaw_off,
                "view_yaw": math.atan2(view[1], view[0]), "pelvis_yaw": pyaw, "forward_of_pelvis": float(fwd),
                "left_of_pelvis": float(left),
                "menu_args": {"cam_h": h, "cam_x": float(fwd), "cam_y": float(left), "pitch_deg": pitch,
                              "roll_deg": roll, "yaw_deg": yaw_off}}


# ---------------------------------------------------------------------------- nav adapter
class G1Base:
    """KinematicBase's interface (nav/sim/base_drive.py) over a WALKING G1, for the nav runner.

    Two drive modes, same surface:

    CALLBACK (default, `physics_step=None`) -- the runner's loop is untouched. The scene runs at
    200 Hz (configure_physics_scene) and the runner keeps calling `kit.update()` once per 1/60 s
    control step; omni.physx then runs 3-4 physics sub-steps per update, and this class ticks the
    policy from a PHYSICS_PRE_STEP callback, every 4th sub-step (50 Hz of SIM time, whatever the
    update cadence), and does the per-sub-step bookkeeping (torso contact) from a PHYSICS_POST_STEP
    callback. `apply()` only stores the command. Nothing in the runner has to know about walking,
    which matters because the runner calls `kit.update()` in about a dozen other places (settling
    after a reset, holding when the camera returns nothing, ...): a policy driven from `apply()`
    would leave the robot unactuated in every one of them and it falls in 1.1 s.

    ONE live instance per process in callback mode: a second one would register a second pair of
    callbacks on the same G1 and tick the policy twice per physics step (100 Hz) -- which, measured
    by g1/check (decimation-2 variant), still WALKS, so nothing downstream would notice. __init__
    refuses while another instance is live (close() it first). Second line: the POLICY RATE CHECK.
    Every PRE_STEP callback records the step_dt omni.physx passes in; a sub-step whose dt is not the
    SPEC's physics_dt (0.005 s) counts in `dt_mismatch_steps` and records fault
    "physics_dt_mismatch", and stats()["policy_hz"] = policy ticks / (sum of those step_dt) is
    judged against the SPEC's policy_hz (50), never against loco.decimation or loco.physics_dt --
    those are exactly what a glue bug changes (round 2 compared with loco.decimation, so a policy
    ticking at 100 Hz reported policy_hz_ok True; g1/check). A stage left at PhysX's default 60 Hz
    now reports step_dt 0.0167 on its first sub-step instead of walking out of distribution.

    EXPLICIT (`physics_step=PhysicsStepper`) -- `apply(vx, vy, omega, dt)` advances physics itself,
    in whole steps of the stepper's dt with the remainder carried, and the caller renders with
    `render_only()`. What the walking tools (nav/tools/try_g1_walk.py) use. The stepper's dt is
    checked against the spec the same way.

    Nothing is teleported, so there is nothing to resync and no "jump" fault: the pose IS the
    sim's. Fault kinds in `first_fault`/`last_fault`/`fault_log` (the runner's `if base.faults`
    rule ends an episode on any of them):
      "fall"                 fallen() fired (tilt > 40 deg or pelvis < 0.40 m above the feet), with
                             the sub-rule; final, the policy cannot get up.
      "free_fall"            pelvis vz < FREE_FALL_VZ_MPS for FREE_FALL_HOLD_S (module docstring):
                             a G1 dropping upright keeps its shape, so fallen() alone saw a 4.75 s
                             fall into the void as upright.
      "no_floor"             reset_to() found no walkable floor under the start; re-recorded after
                             every reset_stats() so it survives the runner's reset order.
      "physics_dt_mismatch"  a sub-step at a dt other than the spec's (above).
      "physics_handle_lost"  an ERROR: the articulation's physics view is gone, nothing is commanding
                             the robot any more -- end the episode as an error, not a fall.
    physics_state()["terminal"] is True once any of these has happened in the episode.

    `hold_on_stop` (default True): a zero command holds the pose latched when it arrived, with a
    small corrective command (HOLD_* constants, hysteresis + minimum command), because the policy's
    own zero-command behaviour is stepping in place while drifting ~0.03 m/s backwards. The test is
    made on the CLAMPED command, so a reverse request (vx < 0 only) holds too. Any other command
    releases it. The hold command is RECOMPUTED AT EVERY POLICY TICK from the live pose (the loco's
    pre_tick_hook), not only inside apply(): a runner loop that calls kit.update() without apply()
    (settling, waiting on a frame) can no longer leave a stale correction running. What it holds
    to is a ~5 cm band (HOLD_ENTER_M), reported as the maximum distance from the latch
    (`hold_max_from_latch_m`, `hold_last_window_max_m`), not an end-point.

    `slew` (default DEFAULT_SLEW): per-axis rate limit on the command the policy sees, in sim time
    (module constant DEFAULT_SLEW for why). None disables it (parity probes).

    `position()` returns (x, y, FLOOR z), not the pelvis: collision_guard and lidar_2d add their
    mount height (0.30 m) to position[2], and a pelvis z of 0.63 m would put the guard's ray at
    0.93 m, over every table and chair leg it exists to catch. The floor estimate is the pelvis
    height minus its measured walking height (WALK_PELVIS_HEIGHT_M), so it follows a ramp.
    """

    RESYNC_TOLERANCE_M = float("inf")
    FAULT_LOG_MAX = 32
    _live = None                    # weakref to the callback-mode instance currently registered

    def __init__(self, loco: G1Locomotion, physics_step: PhysicsStepper | None = None, floor_relative: bool = False,
                 hold_on_stop: bool = True, slew: dict | None = DEFAULT_SLEW):
        self.loco = loco
        # The policy-rate reference comes from the SPEC (what training ran), never from loco's own
        # attributes: those are what a glue bug would change.
        sim = loco.spec["sim"]
        self.spec_dt = float(sim["physics_dt"])
        self.spec_decimation = int(sim["decimation"])
        self.spec_policy_hz = float(sim["policy_hz"])
        if abs(self.spec_policy_hz - 1.0 / (self.spec_dt * self.spec_decimation)) > 1e-6:
            raise ValueError(f"policy spec is inconsistent: policy_hz {self.spec_policy_hz} != "
                             f"1/(physics_dt {self.spec_dt} * decimation {self.spec_decimation})")
        self._holding = False
        self._hold_active = False                # hysteresis state: correcting (True) or quiet inside the band
        self.step = physics_step
        self.callback_mode = physics_step is None
        self.floor_relative = floor_relative
        self.hold_on_stop = hold_on_stop
        self._t_carry = 0.0
        self.physics_steps = 0          # sub-steps seen by the pre-step callback (callback mode)
        self.post_steps = 0             # ... and by the post-step callback
        self.sim_t_total = 0.0          # lifetime sim time, summed from the step_dt PhysX reports
        self._last_state: dict | None = None
        self._last_pose = (0.0, 0.0, 0.0, 0.0)
        self.spawn: dict = {}
        self.no_floor = False
        self._cb_ids: list = []
        self._t_pre: float | None = None
        self.reset_stats()
        if self.callback_mode:
            other = G1Base._live() if G1Base._live is not None else None
            if other is not None and other._cb_ids:
                raise RuntimeError("another G1Base is live in this process (callback mode); close() it first -- two "
                                   "would tick the policy twice per physics step")
            from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

            self._SimulationManager = SimulationManager
            self._cb_ids = [SimulationManager.register_callback(self._on_pre_step, event=SimulationEvent.PHYSICS_PRE_STEP),
                            SimulationManager.register_callback(self._on_post_step, event=SimulationEvent.PHYSICS_POST_STEP)]
            G1Base._live = weakref.ref(self)
        # only after the singleton check: a refused second instance must not take over the loco
        loco.slew = None if slew is None else dict(slew)
        loco.pre_tick_hook = self._pre_tick

    def close(self) -> None:
        for cb in self._cb_ids:
            self._SimulationManager.deregister_callback(cb)
        self._cb_ids = []
        if G1Base._live is not None and G1Base._live() is self:
            G1Base._live = None
        if getattr(self.loco.pre_tick_hook, "__self__", None) is self:
            self.loco.pre_tick_hook = None

    def _note_step_dt(self, dt: float, n: int = 1) -> None:
        """Bookkeeping for `n` physics sub-steps of `dt` seconds (the policy-rate check)."""
        self.sim_t_total += dt * n
        self.sim_dt_sum += dt * n
        self.step_dt_min = min(self.step_dt_min, dt)
        self.step_dt_max = max(self.step_dt_max, dt)
        if abs(dt - self.spec_dt) > STEP_DT_TOLERANCE_S:
            self.dt_mismatch_steps += n
            if self.dt_mismatch_first is None:
                self.dt_mismatch_first = {"t": round(self.sim_time(), 4), "step_dt": dt, "spec_physics_dt": self.spec_dt}
                self._fault("physics_dt_mismatch", step_dt=dt, spec_physics_dt=self.spec_dt)

    def _on_pre_step(self, step_dt, context=None) -> None:
        try:
            self._note_step_dt(float(step_dt))
        except Exception:  # noqa: BLE001  (never raise from a physics callback)
            self.step_dt_read_errors += 1
        self.physics_steps += 1
        self.loco.pre_physics_step()
        self._t_pre = time.perf_counter()

    def _on_post_step(self, step_dt, context=None) -> None:
        if self._t_pre is not None:          # wall time of the PhysX step itself (pre-step return -> post-step)
            self.physx_wall_s += time.perf_counter() - self._t_pre
            self.physx_timed_steps += 1
            self._t_pre = None
        self.post_steps += 1
        self.loco.post_physics_step()

    # KinematicBase-compatible surface ----------------------------------------------------
    def reset_to(self, position, yaw: float) -> None:
        """`position` is the EPISODE start (x, y, z); the pelvis height comes from a floor raycast
        (g1_spawn_pose), not from the caller -- AlohaMini's ground_spawn_pose() is wheel geometry.
        With no walkable floor under the start the robot is still placed (pelvis at episode z +
        SPAWN_HEIGHT_M), `no_floor` is set and fault "no_floor" recorded (and re-recorded by every
        reset_stats()): do not run such an episode."""
        pose, rec = g1_spawn_pose(position, self.loco.prim_path)
        self.spawn = rec
        self.no_floor = rec["ground_z"] is None
        floor = rec["ground_z"] if rec["ground_z"] is not None else float(position[2])
        self.loco.reset_to(pose, yaw, floor_z=floor)
        self._t_carry = 0.0
        self.fell = False
        self.free_fall = False
        self._ff_since = None
        self._hold = None
        self._holding = False
        self._hold_active = False
        if self.no_floor:
            self._record_no_floor()

    def _record_no_floor(self) -> None:
        self._fault("no_floor", episode_z=self.spawn.get("episode_z"), pelvis_spawn_z=self.spawn.get("snapped_z"),
                    why=["no walkable floor below the start (g1_spawn_pose ground_z None)"])

    def reset_stats(self) -> None:
        self.faults = 0
        self.first_fault: dict | None = None
        self.last_fault: dict | None = None
        self.fault_log: list[dict] = []
        self.resyncs = 0
        self.fell = False
        self.fall_time_s: float | None = None
        self.free_fall = False
        self.free_fall_t: float | None = None
        self._ff_since: float | None = None
        self.free_fall_longest_s = 0.0
        self.vz_min = float("inf")
        self.tilt30_first_t: float | None = None
        self.tilt_max_deg = 0.0
        self.z_min, self.z_max, self.z_step_max, self._z_prev = float("inf"), float("-inf"), 0.0, None
        self._hold = None
        self.hold_steps = 0
        self.hold_active_ticks = 0
        self.hold_windows = 0
        self.hold_max_from_latch_m = 0.0
        self._hold_win_max = 0.0
        self.reverse_requests_held = 0
        self.loco.reset_contact_stats()
        self.loco.reverse_requests = 0
        self.loco.clamped_commands = 0
        self._ticks0, self._steps0, self._post0 = self.loco.ticks, self.loco.sim_steps, self.loco.post_steps
        self._slew0 = self.loco.slew_limited_ticks
        self._handle_fault_logged = False
        self.physx_wall_s, self.physx_timed_steps = 0.0, 0
        self.sim_dt_sum = 0.0
        self.step_dt_min, self.step_dt_max = float("inf"), float("-inf")
        self.dt_mismatch_steps = 0
        self.dt_mismatch_first: dict | None = None
        self.step_dt_read_errors = 0
        if getattr(self, "no_floor", False):
            self._record_no_floor()

    @property
    def torso_contact_steps(self) -> int:
        return self.loco.torso_contact_steps

    def sync_from_sim(self) -> None:  # the sim is always the authority for a walking robot
        return None

    def _pose(self) -> tuple[float, float, float, float]:
        if self.loco.handle_ok():
            self._last_pose = self.loco.pose()
        return self._last_pose

    @property
    def yaw(self) -> float:
        return self._pose()[3]

    def position(self) -> tuple[float, float, float]:
        x, y, z, _ = self._pose()
        return (x, y, z - WALK_PELVIS_HEIGHT_M)

    def sim_time(self) -> float:
        return self.step.t if self.step is not None else self.sim_t_total

    def _fault(self, kind: str, **kw) -> None:
        self.faults += 1
        self.last_fault = {"t": round(self.sim_time(), 3), "kind": kind, **kw}
        self.first_fault = self.first_fault or self.last_fault
        if len(self.fault_log) < self.FAULT_LOG_MAX:
            self.fault_log.append(self.last_fault)

    def physics_state(self) -> dict:
        """The same keys KinematicBase.physics_state() returns (so the runner's tilt/speed rules
        apply unchanged) plus the G1's own: pelvis height above the feet, tilt, the fall and free-fall
        flags, `no_floor`, `physics_handle_lost` and `terminal`. Also where a fall / free fall is
        detected and counted, because the runner calls this once per step (the free-fall run length
        is measured between these calls, in sim time)."""
        if not self.loco.handle_ok():
            if not self._handle_fault_logged:
                self._handle_fault_logged = True
                self._fault("physics_handle_lost", why=[f"articulation physics view gone since sim t="
                                                        f"{self.loco.handle_lost_at}"])
            base = dict(self._last_state or {"speed_mps": 0.0, "vz_mps": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0,
                                              "z": 0.0, "pelvis_above_feet_m": 0.0, "tilt_deg": 0.0})
            base.update({"fallen": False, "free_fall": self.free_fall, "no_floor": self.no_floor,
                         "physics_handle_lost": True, "terminal": True})
            return base
        s = self.loco.state()
        h = s["above_feet"]
        t = self.sim_time()
        self.z_min, self.z_max = min(self.z_min, h), max(self.z_max, h)
        if self._z_prev is not None:
            self.z_step_max = max(self.z_step_max, abs(h - self._z_prev))
        self._z_prev = h
        self.tilt_max_deg = max(self.tilt_max_deg, s["tilt_deg"])
        if self.tilt30_first_t is None and max(abs(s["roll_deg"]), abs(s["pitch_deg"])) > RUNNER_TILT_RULE_DEG:
            self.tilt30_first_t = round(t, 3)
        vz = float(s["lin_w"][2])
        self.vz_min = min(self.vz_min, vz)
        if vz < FREE_FALL_VZ_MPS:
            if self._ff_since is None:
                self._ff_since = t
            run = t - self._ff_since
            self.free_fall_longest_s = max(self.free_fall_longest_s, run)
            if run >= FREE_FALL_HOLD_S - 1e-9 and not self.free_fall:
                self.free_fall = True
                self.free_fall_t = round(t, 3)
                self._fault("free_fall", vz_mps=round(vz, 3), run_s=round(run, 3),
                            pos=[round(float(v), 3) for v in s["pos"]],
                            why=[f"pelvis vz < {FREE_FALL_VZ_MPS} m/s for {run:.3f} s >= {FREE_FALL_HOLD_S}"])
        else:
            self._ff_since = None
        fell, why = self.loco.fallen(s, floor_relative=self.floor_relative)
        if fell and not self.fell:
            self.fell = True
            self.fall_time_s = round(t, 3)
            self._fault("fall", why=why, pos=[round(float(v), 3) for v in s["pos"]])
        out = {"speed_mps": float(np.linalg.norm(s["lin_w"])), "vz_mps": vz,
               "roll_deg": s["roll_deg"], "pitch_deg": s["pitch_deg"], "z": float(s["pos"][2]),
               "pelvis_above_feet_m": h, "tilt_deg": s["tilt_deg"], "fallen": bool(fell),
               "free_fall": self.free_fall, "no_floor": self.no_floor, "physics_handle_lost": False,
               "terminal": bool(self.fell or self.free_fall or self.no_floor or self.dt_mismatch_steps)}
        self._last_state = out
        return out

    def policy_sim_seconds(self) -> float:
        """Sim time since reset_stats(): summed from PhysX's own step_dt (callback mode) or the
        stepper's dt (explicit mode) -- not from loco.physics_dt."""
        return self.sim_dt_sum

    def policy_hz(self) -> float | None:
        """Policy ticks per SIM second since reset_stats(); None before 1 s of sim time."""
        sim_s = self.policy_sim_seconds()
        if sim_s < 1.0:
            return None
        return (self.loco.ticks - self._ticks0) / sim_s

    def stats(self) -> dict:
        """Everything an EpisodeResult should carry for a G1 run."""
        hz = self.policy_hz()
        want = self.spec_policy_hz
        dec_ok = (self.loco.decimation == self.spec_decimation
                  and abs(self.loco.physics_dt - self.spec_dt) <= STEP_DT_TOLERANCE_S)
        return {"policy_hz": None if hz is None else round(hz, 3), "policy_hz_expected": round(want, 3),
                "policy_hz_ok": (None if hz is None else
                                 bool(abs(hz - want) <= POLICY_HZ_TOLERANCE and self.dt_mismatch_steps == 0 and dec_ok)),
                "policy_hz_reference": "spec sim.policy_hz; sim seconds summed from PhysX step_dt",
                "spec_physics_dt": self.spec_dt, "spec_decimation": self.spec_decimation,
                "loco_physics_dt": self.loco.physics_dt, "loco_decimation": self.loco.decimation,
                "decimation_matches_spec": dec_ok,
                "step_dt_min_max": (None if self.step_dt_min == float("inf")
                                    else [round(self.step_dt_min, 9), round(self.step_dt_max, 9)]),
                "dt_mismatch_steps": self.dt_mismatch_steps, "dt_mismatch_first": self.dt_mismatch_first,
                "step_dt_read_errors": self.step_dt_read_errors,
                "sim_s_since_reset": round(self.sim_dt_sum, 4),
                "pre_post_steps": [self.loco.sim_steps - self._steps0, self.loco.post_steps - self._post0],
                "slew": self.loco.slew, "slew_limited_ticks": self.loco.slew_limited_ticks - self._slew0,
                "pre_tick_hook_errors": self.loco.pre_tick_hook_errors,
                "faults": self.faults, "fault_log": list(self.fault_log),
                "fell": self.fell, "fall_time_s": self.fall_time_s, "first_fault": self.first_fault,
                "free_fall": self.free_fall, "free_fall_t": self.free_fall_t,
                "free_fall_longest_run_s": round(self.free_fall_longest_s, 4),
                "vz_min_mps": None if self.vz_min == float("inf") else round(self.vz_min, 3),
                "no_floor": self.no_floor, "spawn": self.spawn,
                "tilt_max_deg": round(self.tilt_max_deg, 2), "tilt30_first_t": self.tilt30_first_t,
                "torso_contact_steps": self.loco.torso_contact_steps,
                "torso_contact_events": self.loco.torso_contact_events,
                "torso_contact_first_t": self.loco.torso_contact_first_t,
                "torso_contact_max_n": round(self.loco.torso_contact_max_n, 2),
                "torso_contact_read_errors": self.loco.torso_contact_read_errors,
                "reverse_requests": self.loco.reverse_requests, "clamped_commands": self.loco.clamped_commands,
                "reverse_requests_held": self.reverse_requests_held,
                "hold_on_stop": self.hold_on_stop, "hold_steps": self.hold_steps,
                "hold_active_ticks": self.hold_active_ticks, "hold_windows": self.hold_windows,
                "hold_max_from_latch_m": round(self.hold_max_from_latch_m, 4),
                "hold_last_window_max_m": round(self._hold_win_max, 4),
                "physics_handle_lost_at": self.loco.handle_lost_at, "skipped_ticks": self.loco.skipped_ticks,
                "physx_step_ms_mean": (round(1000 * self.physx_wall_s / self.physx_timed_steps, 3)
                                       if self.physx_timed_steps else None),
                "pelvis_above_feet_min": None if self.z_min == float("inf") else round(self.z_min, 3)}

    def _hold_command(self) -> tuple[float, float, float]:
        """Correction toward the latched pose: zero inside the EXIT band; once an error leaves the ENTER
        band, K*error but at least HOLD_MIN_* (and at most HOLD_MAX_*), until it is back inside EXIT.
        vx is never negative (untrained): a robot AHEAD of the latch comes back on the policy's own
        ~0.03 m/s backward drift at zero command. Also tracks the distance from the latch (the
        honest measure of what the hold buys: a band, not an end-point)."""
        x, y, _, yaw = self._pose()
        if self._hold is None:
            self._hold = (x, y, yaw)
            self.hold_windows += 1
            self._hold_win_max = 0.0
        hx, hy, hyaw = self._hold
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = hx - x, hy - y
        d = math.hypot(ex, ey)
        self._hold_win_max = max(self._hold_win_max, d)
        self.hold_max_from_latch_m = max(self.hold_max_from_latch_m, d)
        e_fwd, e_left = c * ex + s * ey, -s * ex + c * ey
        e_yaw = math.atan2(math.sin(hyaw - yaw), math.cos(hyaw - yaw))
        e_xy = math.hypot(max(e_fwd, 0.0), e_left)   # only what vx >= 0 / vy can correct
        if self._hold_active:
            if e_xy < HOLD_EXIT_M and abs(e_yaw) < HOLD_EXIT_RAD:
                self._hold_active = False
        elif e_xy > HOLD_ENTER_M or abs(e_yaw) > HOLD_ENTER_RAD:
            self._hold_active = True
        self.hold_steps += 1
        if not self._hold_active:
            return (0.0, 0.0, 0.0)
        self.hold_active_ticks += 1

        def mag(k_e, e, lo, hi, exit_band):
            if abs(e) < exit_band:
                return 0.0
            return math.copysign(min(max(abs(k_e), lo), hi), e)
        vx = mag(HOLD_K_XY * e_fwd, e_fwd, HOLD_MIN_VX, HOLD_MAX_VX, HOLD_EXIT_M) if e_fwd > 0 else 0.0
        vy = mag(HOLD_K_XY * e_left, e_left, HOLD_MIN_VY, HOLD_MAX_VY, HOLD_EXIT_M)
        wz = mag(HOLD_K_YAW * e_yaw, e_yaw, HOLD_MIN_WZ, HOLD_MAX_WZ, HOLD_EXIT_RAD)
        return (vx, vy, wz)

    def _pre_tick(self) -> None:
        """loco.pre_tick_hook: runs right before every policy tick (both drive modes)."""
        if self._holding:
            self.loco.set_command(*self._hold_command(), count=False)

    def apply(self, vx: float, vy: float, omega: float, dt: float) -> None:
        # Clamp FIRST (counting a reverse request / a clipped command once), then decide hold-or-walk
        # on what the policy would actually be given: a pure reverse request clamps to (0, 0, 0) and
        # must hold like a STOP (round 2 tested the raw request, so it never held; g1/check).
        c = self.loco.clamp(vx, vy, omega)
        zero = bool(np.all(np.abs(c) < 1e-9))
        if self.hold_on_stop and zero:
            if vx < 0.0:
                self.reverse_requests_held += 1
            if not self._holding:
                self._holding = True
                self._hold = None
                self._hold_active = False
            cmd = self._hold_command() if self.loco.handle_ok() else (0.0, 0.0, 0.0)
        else:
            self._holding = False
            self._hold = None
            self._hold_active = False
            cmd = tuple(float(v) for v in c)
        self.loco.set_command(*cmd, count=False)
        if self.callback_mode:
            return
        self._t_carry += float(dt)
        n = int(math.floor(self._t_carry / self.step.dt + 1e-6))
        self._t_carry -= n * self.step.dt
        self.loco.advance(self.step, n)
        if n:
            self._note_step_dt(float(self.step.dt), n)
        self.physics_state()

    def stop(self) -> None:
        self.apply(0.0, 0.0, 0.0, 0.0)
