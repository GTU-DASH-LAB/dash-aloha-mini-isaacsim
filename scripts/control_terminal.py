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
"""

import argparse
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alohamini1_specs import (  # noqa: E402
    JAW_CLOSED_RAD,
    JAW_OPEN_RAD,
    LIFT_MAX_M,
    LIFT_MIN_M,
    WHEEL_NAMES,
    body_to_wheel_speeds,
)

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--repl", action="store_true", help="Interactive command loop")
parser.add_argument("--arm", nargs=3, metavar=("SIDE", "JOINT", "RAD"), help="One-shot: set an arm joint")
parser.add_argument("--lift", type=float, metavar="METERS", help="One-shot: set lift height")
parser.add_argument("--base", nargs=3, type=float, metavar=("VX", "VY", "OMEGA"), help="One-shot: set base velocity")
parser.add_argument("--settle", type=float, default=2.0, help="Seconds to step physics after a one-shot command")
parser.add_argument("--gui", action="store_true", help="Open a visible window instead of running headless")
args = parser.parse_args()

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


def set_arm_joint(side: str, joint_idx: int, radians: float):
    name = f"{side}_joint{joint_idx}"
    if name not in dof_names:
        print(f"ERROR: unknown joint {name}")
        return
    idx = dof_names.index(name)
    _position_targets[0][idx] = radians
    art.set_joint_position_targets(_position_targets)
    print(f"Set {name} target = {radians:.4f} rad")


def set_gripper(side: str, action: str):
    value = JAW_OPEN_RAD if action == "open" else JAW_CLOSED_RAD
    set_arm_joint(side, 6, value)


def set_lift(meters: float):
    meters = max(LIFT_MIN_M, min(LIFT_MAX_M, meters))
    if "vertical_move" not in dof_names:
        print("ERROR: vertical_move not found")
        return
    idx = dof_names.index("vertical_move")
    _position_targets[0][idx] = meters
    art.set_joint_position_targets(_position_targets)
    print(f"Set lift target = {meters:.4f} m")


def set_base_velocity(vx: float, vy: float, omega: float):
    # Wheel joints still spin at the visually-correct rate (real joint-level physics).
    wheel_speeds = body_to_wheel_speeds(vx, vy, omega)
    for wheel_name, speed in zip(WHEEL_NAMES, wheel_speeds):
        if wheel_name not in dof_names:
            print(f"ERROR: {wheel_name} not found")
            continue
        idx = dof_names.index(wheel_name)
        _velocity_targets[0][idx] = speed
    art.set_joint_velocity_targets(_velocity_targets)
    # Actual translation is kinematically driven via root pose teleport each step --
    # see _apply_kinematic_base_step / step_seconds. Only takes effect while stepping
    # through step_seconds() or the REPL main loop (both call it each frame).
    _base_vel["vx"], _base_vel["vy"], _base_vel["omega"] = vx, vy, omega
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
    print("AlohaMini1 control REPL. Commands: arm/gripper/lift/base/stop/status/quit")
    cmd_queue: queue.Queue = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
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
            if cmd == "arm" and len(parts) == 4:
                set_arm_joint(parts[1], int(parts[2]), float(parts[3]))
            elif cmd == "gripper" and len(parts) == 3:
                set_gripper(parts[1], parts[2])
            elif cmd == "lift" and len(parts) == 2:
                set_lift(float(parts[1]))
            elif cmd == "base" and len(parts) == 4:
                set_base_velocity(float(parts[1]), float(parts[2]), float(parts[3]))
            elif cmd == "stop":
                set_base_velocity(0.0, 0.0, 0.0)
            elif cmd == "wait" and len(parts) == 2:
                step_seconds(float(parts[1]))
            elif cmd == "status":
                print_status()
            elif cmd == "pose":
                print_pose()
            elif cmd == "screenshot" and len(parts) == 2:
                take_screenshot(parts[1])
            elif cmd in ("quit", "exit"):
                running = False
            else:
                print(f"Unknown command: {line}")
        except Exception as e:
            print(f"ERROR: {e}")


if args.repl:
    run_repl()
else:
    run_one_shot()

kit.close()
