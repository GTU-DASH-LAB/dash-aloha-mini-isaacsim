"""Phase 4: terminal control for the AlohaMini1 simulation.

Two modes:
  One-shot:  ~/isaacsim/python.sh scripts/control_terminal.py --arm left 1 0.5 --settle 2
  REPL:      ~/isaacsim/python.sh scripts/control_terminal.py --repl

Add --gui to open a visible Isaac Sim window (needs a display) so you can watch the
robot while typing commands in the same terminal:
  ~/isaacsim/python.sh scripts/control_terminal.py --repl --gui

REPL commands:
  arm <left|right> <1-6> <radians>   set one arm joint position target
  gripper <left|right> <open|close>  shorthand for joint6 to its limit
  lift <meters>                      set lift (vertical_move) position target, 0-0.6
  base <vx> <vy> <omega>             set sustained base velocity (m/s, m/s, rad/s)
  stop                               zero all wheel velocities
  status                             print current joint positions
  pose                               print base_link's world position
  screenshot <path>                  save a viewport screenshot framed on the robot
  wait <seconds>                     step physics for N seconds before next command
  quit / exit                        stop and close

PS4 controller mode (needs a controller connected + the `evdev` package + your user
in the `input` group -- run `sudo usermod -aG input $USER` and log back in if you get
a Permission Denied error reading /dev/input):
  ~/isaacsim/python.sh scripts/control_terminal.py --joystick --gui

  L1 held            -> control the RIGHT arm
  L2 held             -> control the LEFT arm
  L1 + L2 held together -> control BOTH arms at once, mirrored (opposite movement)
  R2 held             -> control the base (drive)
  (nothing held)      -> sticks/buttons do nothing

  Within arm mode:
    Left stick  X/Y -> joint1 (Rotation) / joint2 (Pitch)
    Right stick X/Y -> joint3 (Elbow) / joint4 (Wrist_Pitch)
    D-pad up/down    -> joint5 (Wrist_Roll)
    Cross / Circle   -> gripper open / close (joint6)
  Within base mode (R2 held):
    Left stick  X/Y -> vx / vy (m/s)
    Right stick X    -> omega (rad/s)

  NOTE: this mapping is implemented against the standard Linux evdev codes for a
  DualShock 4 (BTN_TL/BTN_TR for L1/R1, ABS_Z/ABS_RZ for analog L2/R2, ABS_X/Y and
  ABS_RX/RY for the sticks, ABS_HAT0X/Y for the d-pad). It has NOT been tested against
  a physical controller in this environment (none was connected). Run with
  --joystick-debug first to print raw events and confirm/adjust the codes in
  JOYSTICK_MAP below if your controller reports something different.
"""

import argparse
import queue
import sys
import threading
from pathlib import Path

try:
    # Isaac Sim's bundled Python is built without the stdlib `readline` module (no
    # libreadline linked in) -- gnureadline is a self-contained drop-in replacement.
    # pip install gnureadline
    import gnureadline as readline  # noqa: F401
except ImportError:
    try:
        import readline  # noqa: F401
    except ImportError:
        readline = None  # arrow-key history just won't work; everything else still does
        print("Note: no readline module available -- command history (Up arrow) is "
              "disabled. Run: ~/isaacsim/python.sh -m pip install gnureadline")

sys.path.insert(0, str(Path(__file__).parent))
from alohamini1_specs import (  # noqa: E402
    ARM_JOINT_GAINS,
    ARM_JOINT_LIMITS_RAD,
    JAW_CLOSED_RAD,
    JAW_OPEN_RAD,
    LIFT_MAX_M,
    LIFT_MIN_M,
    WHEEL_NAMES,
    body_to_wheel_speeds,
)

# Usage/description/limits for every REPL command. Shown in full by `help`, and
# shown automatically (instead of a bare "unknown command") when a recognized
# command is typed with the wrong number of arguments -- e.g. typing just "arm" or
# "base" alone shows you exactly how to use it rather than silently doing nothing.
_JOINT_LIMITS_TEXT = ", ".join(
    f"{ARM_JOINT_GAINS[i]['name']}(joint{i})=[{lo:.2f},{hi:.2f}]rad"
    for i, (lo, hi) in ARM_JOINT_LIMITS_RAD.items()
)
COMMAND_HELP = {
    "arm": {
        "usage": "arm <left|right> <1-6> <radians>",
        "description": "Set one arm joint's position target. Joint numbers: "
                        "1=Rotation, 2=Pitch, 3=Elbow, 4=Wrist_Pitch, 5=Wrist_Roll, "
                        "6=Jaw (gripper).",
        "limits": f"Per-joint radian limits: {_JOINT_LIMITS_TEXT}. Values outside "
                  f"this range will still be sent but the joint won't reach them "
                  f"(clamped by the physics drive, not by this script).",
    },
    "gripper": {
        "usage": "gripper <left|right> <open|close>",
        "description": "Shorthand for setting joint 6 (Jaw) to its open or closed limit.",
        "limits": f"open={JAW_OPEN_RAD:.4f} rad, close={JAW_CLOSED_RAD:.4f} rad. "
                  f"Which physical direction is 'open' vs 'closed' has not been "
                  f"visually verified (see CLAUDE.md) -- if it looks backwards, that's why.",
    },
    "lift": {
        "usage": "lift <meters>",
        "description": "Set the vertical lift column's position target.",
        "limits": f"Range [{LIFT_MIN_M:.2f}, {LIFT_MAX_M:.2f}] m. Values outside this "
                  f"range are clamped automatically by this script (not by physics).",
    },
    "base": {
        "usage": "base <vx> <vy> <omega>",
        "description": "Set a sustained base velocity: vx/vy in the robot's own "
                        "body frame (m/s, forward/strafe), omega is rotation (rad/s). "
                        "Stays active until you send another `base` command or `stop`.",
        "limits": "No hard limit enforced by this script, but real wheel-ground "
                  "traction physics doesn't work reliably (see CLAUDE.md) -- "
                  "locomotion is kinematically driven instead. IMPORTANT: driving "
                  "the base while also sending new `arm` commands measurably "
                  "degrades arm convergence -- `stop` the base first, then command "
                  "arms, for reliable results.",
    },
    "stop": {
        "usage": "stop",
        "description": "Zero the base velocity (equivalent to `base 0 0 0`). Does not "
                        "affect arm/lift/gripper targets.",
        "limits": "None.",
    },
    "status": {
        "usage": "status",
        "description": "Print every joint's current position.",
        "limits": "None.",
    },
    "pose": {
        "usage": "pose",
        "description": "Print base_link's current world position (x, y, z in meters).",
        "limits": "None.",
    },
    "screenshot": {
        "usage": "screenshot <path>",
        "description": "Save a viewport screenshot (PNG) framed on the robot to <path>.",
        "limits": "Only useful with --gui, or to inspect the file afterward -- "
                  "headless mode has no visible window but the file still gets written.",
    },
    "wait": {
        "usage": "wait <seconds>",
        "description": "Step physics for N seconds before reading the next command. "
                        "Use this after `base`/`arm`/`lift` commands to let motion settle.",
        "limits": "None, but very long waits will make the REPL unresponsive until done.",
    },
    "help": {
        "usage": "help [command]",
        "description": "List all commands, or show detailed usage/limits for one command.",
        "limits": "None.",
    },
    "quit": {
        "usage": "quit  (or: exit)",
        "description": "Stop physics and close the simulation.",
        "limits": "None.",
    },
}
COMMAND_HELP["exit"] = COMMAND_HELP["quit"]


def print_command_help(cmd: str | None = None):
    if cmd is None:
        print("Commands:")
        seen = set()
        for name, info in COMMAND_HELP.items():
            if info["usage"] in seen:
                continue
            seen.add(info["usage"])
            print(f"  {info['usage']}")
        print("Type `help <command>` for details on any one of them (e.g. `help arm`).")
        return
    info = COMMAND_HELP.get(cmd)
    if info is None:
        print(f"No such command: {cmd}. Type `help` to list all commands.")
        return
    print(f"Usage:       {info['usage']}")
    print(f"Description: {info['description']}")
    print(f"Limits:      {info['limits']}")

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--repl", action="store_true", help="Interactive command loop")
parser.add_argument("--arm", nargs=3, metavar=("SIDE", "JOINT", "RAD"), help="One-shot: set an arm joint")
parser.add_argument("--lift", type=float, metavar="METERS", help="One-shot: set lift height")
parser.add_argument("--base", nargs=3, type=float, metavar=("VX", "VY", "OMEGA"), help="One-shot: set base velocity")
parser.add_argument("--settle", type=float, default=2.0, help="Seconds to step physics after a one-shot command")
parser.add_argument("--gui", action="store_true", help="Open a visible window instead of running headless")
parser.add_argument("--joystick", action="store_true", help="PS4 controller control mode (see module docstring)")
parser.add_argument("--joystick-debug", action="store_true",
                     help="Just print raw controller events (no simulation) -- use this to verify/calibrate "
                          "button and axis codes against JOYSTICK_MAP before trusting --joystick")
args = parser.parse_args()

if args.joystick_debug:
    # Bail out before touching Isaac Sim at all -- this is a pure hardware-calibration
    # tool, no need to pay Kit's startup cost for it.
    import evdev

    devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
    if not devices:
        print("No input devices found at all under /dev/input. Is a controller connected? "
              "Also check you're in the 'input' group (groups | grep input) -- "
              "otherwise device nodes exist but you can't open them.")
        sys.exit(1)
    print("Devices found:")
    for d in devices:
        print(f"  {d.path}: {d.name}")
    candidates = [d for d in devices if any(
        s in d.name for s in ("Wireless Controller", "DualShock", "DualSense", "Sony", "PS4", "PLAYSTATION")
    )]
    if not candidates:
        print("\nNone of these look like a PS4/PlayStation controller by name. "
              "Pick the right one manually and pass --joystick-device /dev/input/eventN.")
        sys.exit(1)
    device = candidates[0]
    print(f"\nUsing: {device.path} ({device.name})")
    print("Press buttons / move sticks. Ctrl+C to stop.\n")
    for event in device.read_loop():
        if event.type != 0:  # skip EV_SYN sync events, just noise
            print(evdev.categorize(event))

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": not args.gui})

import math  # noqa: E402
import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from pxr import UsdGeom  # noqa: E402
import numpy as np  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(args.scene)
stage = usd_context.get_stage()
for _ in range(60):
    kit.update()
stage.Load()
for _ in range(10):
    kit.update()

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    kit.update()

art = Articulation(prim_paths_expr="/World/Aloha/Geometry/base_link")
art.initialize()
dof_names = art.dof_names

if args.gui:
    from omni.kit.viewport.utility import get_active_viewport, frame_viewport_prims
    frame_viewport_prims(get_active_viewport(), prims=["/World/Aloha"])
    for _ in range(15):
        kit.update()

# Maintain our own persistent target arrays instead of repeatedly reading back
# art.get_joint_positions()/get_joint_velocities() (the ACTUAL current position, not
# the previously-commanded target) and modifying one index at a time. That
# read-modify-write pattern is a real bug if you do it more than once before physics
# catches up: each call resets every OTHER joint's target back to wherever it
# currently is, silently clobbering earlier commands. Verified this was happening --
# commanding left_joint1 then lift then base left left_joint1 at ~0.05 rad instead of
# the commanded 0.4 rad, purely because set_lift()'s own read-modify-write reset it.
_position_targets = art.get_joint_positions().copy()
_velocity_targets = art.get_joint_velocities().copy()

base_prim = stage.GetPrimAtPath("/World/Aloha/Geometry/base_link")
xform_cache = UsdGeom.XformCache()

PHYSICS_DT = 1.0 / 60.0

# --- Kinematic base drive ---
# Real wheel-ground contact physics did not hold up under test (see CLAUDE.md /
# plan.md Phase 4 for the full investigation: fixed a genuine collision bug where
# base_link's own shell was blocking ground contact, added per-wheel sphere colliders
# + friction material, retuned velocity-drive damping that was causing oscillation --
# after all that, even a single wheel spinning in isolation produced ~200x less
# translation than expected). Per the plan's pre-approved fallback: base locomotion is
# kinematically driven (root pose directly set via set_world_poses(), which properly
# syncs with PhysX's internal state -- raw USD transform edits do NOT, verified
# directly), while wheel joints still spin at the visually-correct commanded rate.
#
# Tried setting root VELOCITY instead of teleporting POSITION, hoping it would be less
# invasive to the concurrently-running arm joint drives. It wasn't better: contact
# friction from the (still non-driving) wheel-ground contact damped out the injected
# velocity almost immediately, and refreshing every step to compensate caused Z-height
# drift (0.0078m -> 0.0611m over 3s -- some kind of energy injection from repeatedly
# forcing vz=0 against the contact solver). Reverted to position teleport.
#
# KNOWN LIMITATION: teleporting the root pose every step does measurably degrade
# concurrent ARM joint convergence (verified: commanding left_joint1 to 0.4 rad while
# also actively driving the base only reached ~0.05 rad). Recommended usage: drive the
# base, `stop` it, THEN command arm/lift moves -- not simultaneously. Sequential
# control was not degraded (arm/lift alone, or base alone, both converge correctly).
_base_pose = {"yaw": 0.0, "initialized": False}


def _init_base_pose():
    _, orientations = art.get_world_poses()
    w, x, y, z = orientations[0]
    _base_pose["yaw"] = 2.0 * math.atan2(z, w)
    _base_pose["initialized"] = True


_base_vel = {"vx": 0.0, "vy": 0.0, "omega": 0.0}


def _apply_kinematic_base_step(dt: float):
    vx, vy, omega = _base_vel["vx"], _base_vel["vy"], _base_vel["omega"]
    if vx == 0.0 and vy == 0.0 and omega == 0.0:
        return
    if not _base_pose["initialized"]:
        _init_base_pose()
    positions, _ = art.get_world_poses()
    t = positions[0]
    yaw = _base_pose["yaw"]
    dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
    dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
    new_yaw = yaw + omega * dt
    _base_pose["yaw"] = new_yaw
    new_positions = np.array([[t[0] + dx, t[1] + dy, t[2]]], dtype=np.float32)
    half = new_yaw / 2.0
    new_orientations = np.array([[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=np.float32)
    art.set_world_poses(positions=new_positions, orientations=new_orientations)


def step_seconds(seconds: float):
    for _ in range(int(seconds / PHYSICS_DT)):
        kit.update()
        _apply_kinematic_base_step(PHYSICS_DT)


def set_arm_joint(side: str, joint_idx: int, radians: float, quiet: bool = False):
    name = f"{side}_joint{joint_idx}"
    if name not in dof_names:
        if not quiet:
            print(f"ERROR: unknown joint {name}")
        return
    idx = dof_names.index(name)
    _position_targets[0][idx] = radians
    art.set_joint_position_targets(_position_targets)
    if not quiet:
        print(f"Set {name} target = {radians:.4f} rad")


def get_arm_joint_target(side: str, joint_idx: int) -> float:
    name = f"{side}_joint{joint_idx}"
    return float(_position_targets[0][dof_names.index(name)])


def set_gripper(side: str, action: str, quiet: bool = False):
    value = JAW_OPEN_RAD if action == "open" else JAW_CLOSED_RAD
    set_arm_joint(side, 6, value, quiet=quiet)


def set_lift(meters: float, quiet: bool = False):
    meters = max(LIFT_MIN_M, min(LIFT_MAX_M, meters))
    if "vertical_move" not in dof_names:
        if not quiet:
            print("ERROR: vertical_move not found")
        return
    idx = dof_names.index("vertical_move")
    _position_targets[0][idx] = meters
    art.set_joint_position_targets(_position_targets)
    if not quiet:
        print(f"Set lift target = {meters:.4f} m")


def get_lift_target() -> float:
    return float(_position_targets[0][dof_names.index("vertical_move")])


def set_base_velocity(vx: float, vy: float, omega: float, quiet: bool = False):
    # Wheel joints still spin at the visually-correct rate (real joint-level physics).
    wheel_speeds = body_to_wheel_speeds(vx, vy, omega)
    for wheel_name, speed in zip(WHEEL_NAMES, wheel_speeds):
        if wheel_name not in dof_names:
            if not quiet:
                print(f"ERROR: {wheel_name} not found")
            continue
        idx = dof_names.index(wheel_name)
        _velocity_targets[0][idx] = speed
    art.set_joint_velocity_targets(_velocity_targets)
    # Actual translation is kinematically driven via root pose teleport each step --
    # see _apply_kinematic_base_step / step_seconds. Only takes effect while stepping
    # through step_seconds() or the REPL main loop (both call it each frame).
    _base_vel["vx"], _base_vel["vy"], _base_vel["omega"] = vx, vy, omega
    if not quiet:
        print(f"Set base velocity vx={vx} vy={vy} omega={omega} (kinematic drive) -> "
              f"wheel spin {dict(zip(WHEEL_NAMES, wheel_speeds))}")


def print_status():
    positions = art.get_joint_positions()[0].tolist()
    for name, pos in zip(dof_names, positions):
        print(f"  {name}: {pos:.4f}")


def print_pose():
    xform_cache.Clear()
    m = xform_cache.GetLocalToWorldTransform(base_prim)
    t = m.ExtractTranslation()
    print(f"  base_link world position: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f}")


def take_screenshot(path: str):
    from omni.kit.viewport.utility import get_active_viewport, frame_viewport_prims
    import omni.kit.viewport.utility as vp_utility

    viewport = get_active_viewport()
    frame_viewport_prims(viewport, prims=["/World/Aloha"])
    for _ in range(15):
        kit.update()
    vp_utility.capture_viewport_to_file(viewport, path)
    for _ in range(10):
        kit.update()
    print(f"Screenshot saved to: {path}")


# --- PS4 controller mode ---
# Standard Linux evdev codes for a DualShock 4 (hid-sony/hid-playstation kernel
# driver). NOT verified against physical hardware in this environment -- run
# --joystick-debug to check/adjust these against your actual controller first.
JOYSTICK_MAP = {
    "BTN_TL": 310,       # L1
    "BTN_TR": 311,       # R1 (unused currently)
    "BTN_TL2": 312,      # L2 digital (some drivers also fire this alongside ABS_Z)
    "BTN_TR2": 313,      # R2 digital
    "BTN_SOUTH": 304,    # Cross -- gripper open
    "BTN_EAST": 305,     # Circle -- gripper close
    "ABS_X": 0,          # left stick X
    "ABS_Y": 1,          # left stick Y
    "ABS_Z": 2,          # L2 analog (0..255, 0=released)
    "ABS_RX": 3,         # right stick X
    "ABS_RY": 4,         # right stick Y
    "ABS_RZ": 5,         # R2 analog (0..255, 0=released)
    "ABS_HAT0X": 16,     # d-pad X
    "ABS_HAT0Y": 17,     # d-pad Y
}
ANALOG_TRIGGER_PRESS_THRESHOLD = 128  # out of 0..255 -- treat L2/R2 as "held" past this
STICK_DEADZONE = 0.15  # normalized -1..1, ignore noise near center
ARM_RATE_RAD_PER_SEC = 1.0  # max joint speed at full stick deflection
BASE_RATE_M_PER_SEC = 0.3
BASE_ROTATE_RATE_RAD_PER_SEC = 1.0


def find_joystick():
    import evdev

    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if any(s in dev.name for s in
               ("Wireless Controller", "DualShock", "DualSense", "Sony", "PS4", "PLAYSTATION")):
            return dev
    return None


def _normalize_axis(value: int, absinfo) -> float:
    lo, hi = absinfo.min, absinfo.max
    if hi == lo:
        return 0.0
    centered = (value - (hi + lo) / 2.0) / ((hi - lo) / 2.0)
    return max(-1.0, min(1.0, centered))


def _apply_deadzone(v: float) -> float:
    return 0.0 if abs(v) < STICK_DEADZONE else v


def run_joystick():
    import evdev

    device = find_joystick()
    if device is None:
        print("No PS4-like controller found (checked device names for 'Wireless "
              "Controller'/'DualShock'/'DualSense'/'Sony'/'PS4'/'PLAYSTATION'). "
              "Run --joystick-debug to see what's actually connected, or check "
              "you're in the 'input' group: groups | grep input")
        return
    print(f"Using controller: {device.path} ({device.name})")
    print("L1=right arm  L2=left arm  L1+L2=both (mirrored)  R2=base  -- see module docstring for axis mapping")

    absinfo = {code: device.absinfo(code) for code in
               (JOYSTICK_MAP["ABS_X"], JOYSTICK_MAP["ABS_Y"], JOYSTICK_MAP["ABS_RX"],
                JOYSTICK_MAP["ABS_RY"], JOYSTICK_MAP["ABS_Z"], JOYSTICK_MAP["ABS_RZ"])
               if code in dict(device.capabilities().get(3, []))}  # 3 = EV_ABS

    state = {
        "lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0,
        "l2": 0, "r2": 0, "l1": False, "hat_y": 0,
    }
    button_state = {"l1_held": False, "l2_held": False, "r2_held": False}
    last_gripper_press = {"south": False, "east": False}
    event_queue: queue.Queue = queue.Queue()

    def read_events():
        try:
            for event in device.read_loop():
                event_queue.put(event)
        except OSError as e:
            print(f"Controller disconnected or read error: {e}")

    reader = threading.Thread(target=read_events, daemon=True)
    reader.start()

    print("Reading controller input. Ctrl+C to stop.")
    last_mode = None
    try:
        while True:
            kit.update()
            _apply_kinematic_base_step(PHYSICS_DT)

            try:
                while True:
                    event = event_queue.get_nowait()
                    if event.type == 3:  # EV_ABS
                        if event.code == JOYSTICK_MAP["ABS_X"]:
                            state["lx"] = _apply_deadzone(_normalize_axis(event.value, absinfo[JOYSTICK_MAP["ABS_X"]]))
                        elif event.code == JOYSTICK_MAP["ABS_Y"]:
                            state["ly"] = _apply_deadzone(_normalize_axis(event.value, absinfo[JOYSTICK_MAP["ABS_Y"]]))
                        elif event.code == JOYSTICK_MAP["ABS_RX"]:
                            state["rx"] = _apply_deadzone(_normalize_axis(event.value, absinfo[JOYSTICK_MAP["ABS_RX"]]))
                        elif event.code == JOYSTICK_MAP["ABS_RY"]:
                            state["ry"] = _apply_deadzone(_normalize_axis(event.value, absinfo[JOYSTICK_MAP["ABS_RY"]]))
                        elif event.code == JOYSTICK_MAP["ABS_Z"]:
                            state["l2"] = event.value
                        elif event.code == JOYSTICK_MAP["ABS_RZ"]:
                            state["r2"] = event.value
                        elif event.code == JOYSTICK_MAP["ABS_HAT0Y"]:
                            state["hat_y"] = event.value
                    elif event.type == 1:  # EV_KEY
                        if event.code == JOYSTICK_MAP["BTN_TL"]:
                            button_state["l1_held"] = bool(event.value)
                        elif event.code == JOYSTICK_MAP["BTN_TL2"]:
                            button_state["l2_held"] = bool(event.value) or state["l2"] > ANALOG_TRIGGER_PRESS_THRESHOLD
                        elif event.code == JOYSTICK_MAP["BTN_TR2"]:
                            button_state["r2_held"] = bool(event.value) or state["r2"] > ANALOG_TRIGGER_PRESS_THRESHOLD
                        elif event.code == JOYSTICK_MAP["BTN_SOUTH"]:
                            last_gripper_press["south"] = bool(event.value)
                        elif event.code == JOYSTICK_MAP["BTN_EAST"]:
                            last_gripper_press["east"] = bool(event.value)
            except queue.Empty:
                pass

            # Analog triggers might not send a digital BTN event on every driver --
            # also treat crossing the threshold as "held" directly from the axis value.
            l2_held = button_state["l2_held"] or state["l2"] > ANALOG_TRIGGER_PRESS_THRESHOLD
            r2_held = button_state["r2_held"] or state["r2"] > ANALOG_TRIGGER_PRESS_THRESHOLD
            l1_held = button_state["l1_held"]

            if r2_held:
                mode = "base"
            elif l1_held and l2_held:
                mode = "both_sync"
            elif l1_held:
                mode = "right_arm"
            elif l2_held:
                mode = "left_arm"
            else:
                mode = "none"

            if mode != last_mode:
                print(f"Mode: {mode}")
                last_mode = mode

            dt = PHYSICS_DT
            if mode == "base":
                set_base_velocity(
                    state["ly"] * BASE_RATE_M_PER_SEC,
                    -state["lx"] * BASE_RATE_M_PER_SEC,
                    -state["rx"] * BASE_ROTATE_RATE_RAD_PER_SEC,
                    quiet=True,
                )
            else:
                set_base_velocity(0.0, 0.0, 0.0, quiet=True)

                sides = []
                mirror = {"left": 1.0, "right": 1.0}
                if mode == "right_arm":
                    sides = ["right"]
                elif mode == "left_arm":
                    sides = ["left"]
                elif mode == "both_sync":
                    sides = ["left", "right"]
                    mirror = {"left": -1.0, "right": 1.0}  # opposite movement, per spec

                for side in sides:
                    sign = mirror[side]
                    if state["lx"] != 0.0:
                        j1 = get_arm_joint_target(side, 1) + sign * state["lx"] * ARM_RATE_RAD_PER_SEC * dt
                        set_arm_joint(side, 1, j1, quiet=True)
                    if state["ly"] != 0.0:
                        j2 = get_arm_joint_target(side, 2) + sign * state["ly"] * ARM_RATE_RAD_PER_SEC * dt
                        set_arm_joint(side, 2, j2, quiet=True)
                    if state["rx"] != 0.0:
                        j3 = get_arm_joint_target(side, 3) + sign * state["rx"] * ARM_RATE_RAD_PER_SEC * dt
                        set_arm_joint(side, 3, j3, quiet=True)
                    if state["ry"] != 0.0:
                        j4 = get_arm_joint_target(side, 4) + sign * state["ry"] * ARM_RATE_RAD_PER_SEC * dt
                        set_arm_joint(side, 4, j4, quiet=True)
                    if state["hat_y"] != 0:
                        j5 = get_arm_joint_target(side, 5) + sign * state["hat_y"] * ARM_RATE_RAD_PER_SEC * dt
                        set_arm_joint(side, 5, j5, quiet=True)
                    if last_gripper_press["south"]:
                        set_gripper(side, "open", quiet=True)
                    elif last_gripper_press["east"]:
                        set_gripper(side, "close", quiet=True)
    except KeyboardInterrupt:
        print("\nStopped.")


def run_one_shot():
    did_something = False
    if args.arm:
        side, joint_idx, radians = args.arm
        set_arm_joint(side, int(joint_idx), float(radians))
        did_something = True
    if args.lift is not None:
        set_lift(args.lift)
        did_something = True
    if args.base is not None:
        set_base_velocity(*args.base)
        did_something = True
    if not did_something:
        print("No command given -- use --arm/--lift/--base, or --repl for interactive mode")
        return
    step_seconds(args.settle)
    print("\n--- Final joint state ---")
    print_status()


def run_repl():
    print("AlohaMini1 control REPL. Type `help` to list commands, `help <command>` for details.")
    cmd_queue: queue.Queue = queue.Queue()

    def read_stdin():
        # Using input() (not `for line in sys.stdin`) so that, on a real interactive
        # terminal, the `readline` module gives you arrow-key command history and
        # in-line editing for free -- press Up to recall the previous command, edit
        # it if you like, press Enter to run it. Falls back to plain line reads when
        # stdin isn't a TTY (e.g. piped input for scripted testing), same as before.
        while True:
            try:
                line = input()
            except EOFError:
                cmd_queue.put("quit")
                break
            cmd_queue.put(line.strip())
            if line.strip() in ("quit", "exit"):
                break

    reader = threading.Thread(target=read_stdin, daemon=True)
    reader.start()

    running = True
    while running:
        kit.update()
        _apply_kinematic_base_step(PHYSICS_DT)
        try:
            line = cmd_queue.get_nowait()
        except queue.Empty:
            continue
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd == "arm":
                if len(parts) == 4:
                    set_arm_joint(parts[1], int(parts[2]), float(parts[3]))
                else:
                    print_command_help("arm")
            elif cmd == "gripper":
                if len(parts) == 3:
                    set_gripper(parts[1], parts[2])
                else:
                    print_command_help("gripper")
            elif cmd == "lift":
                if len(parts) == 2:
                    set_lift(float(parts[1]))
                else:
                    print_command_help("lift")
            elif cmd == "base":
                if len(parts) == 4:
                    set_base_velocity(float(parts[1]), float(parts[2]), float(parts[3]))
                else:
                    print_command_help("base")
            elif cmd == "stop":
                set_base_velocity(0.0, 0.0, 0.0)
            elif cmd == "wait":
                if len(parts) == 2:
                    step_seconds(float(parts[1]))
                else:
                    print_command_help("wait")
            elif cmd == "status":
                print_status()
            elif cmd == "pose":
                print_pose()
            elif cmd == "screenshot":
                if len(parts) == 2:
                    take_screenshot(parts[1])
                else:
                    print_command_help("screenshot")
            elif cmd == "help":
                print_command_help(parts[1].lower() if len(parts) == 2 else None)
            elif cmd in ("quit", "exit"):
                running = False
            else:
                print(f"Unknown command: {line!r}. Type `help` to list commands.")
        except Exception as e:
            print(f"ERROR: {e}")


if args.joystick:
    run_joystick()
elif args.repl:
    run_repl()
else:
    run_one_shot()

kit.close()
