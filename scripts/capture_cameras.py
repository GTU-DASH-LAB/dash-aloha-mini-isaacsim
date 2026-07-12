"""Grab frames from the robot-mounted cameras in the LeRobot observation format.

The camera set (names, 640x480 resolution, 30 fps) mirrors the OFFICIAL AlohaMini
LeRobot config (third_party/lerobot_alohamini/.../config_alohamini.py), so frames
recorded here drop straight into a LeRobot dataset:

    obs = get_camera_observation()
    # {"observation.images.forward":     (480, 640, 3) uint8,
    #  "observation.images.wrist_left":  (480, 640, 3) uint8,
    #  "observation.images.wrist_right": (480, 640, 3) uint8}

For episode recording, sample every 2nd physics step (physics runs at 60 Hz, the
official cameras are 30 fps).

Usage:
    # Save one frame per camera to docs/ and exit
    ~/isaacsim/python.sh scripts/capture_cameras.py --save-dir docs

    # Same, plus verify the wrist views actually change when the arm moves
    ~/isaacsim/python.sh scripts/capture_cameras.py --motion-test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alohamini1_specs import CAMERA_FPS, CAMERA_PRIM_PATHS, CAMERA_RESOLUTION  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--save-dir", default=None, help="Directory to write capture_<name>.png files into")
parser.add_argument("--motion-test", action="store_true",
                    help="Move the lift+arms and verify each wrist camera's image actually changes")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

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

cameras = {}
for name, prim_path in CAMERA_PRIM_PATHS.items():
    cam = Camera(prim_path=prim_path, resolution=CAMERA_RESOLUTION)
    cam.initialize()
    cameras[name] = cam

# Let the sensor pipelines warm up (first frames come back empty otherwise).
for _ in range(30):
    kit.update()


def get_camera_observation() -> dict:
    """One LeRobot-style observation dict: observation.images.<name> -> HxWx3 uint8."""
    obs = {}
    for name, cam in cameras.items():
        rgba = cam.get_rgba()
        if rgba is None or rgba.size == 0:
            obs[f"observation.images.{name}"] = None
            continue
        obs[f"observation.images.{name}"] = rgba[:, :, :3].astype(np.uint8)
    return obs


def save_observation(obs: dict, save_dir: str, suffix: str = ""):
    from PIL import Image

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in obs.items():
        if frame is None:
            continue
        name = key.rsplit(".", 1)[-1]
        path = out_dir / f"capture_{name}{suffix}.png"
        Image.fromarray(frame).save(path)
        print(f"Saved {path}")


obs = get_camera_observation()
w, h = CAMERA_RESOLUTION
print(f"Camera fps (official spec, sample every {60 // CAMERA_FPS} physics steps): {CAMERA_FPS}")
for key, frame in obs.items():
    ok = frame is not None and frame.shape == (h, w, 3) and frame.dtype == np.uint8
    detail = "None" if frame is None else f"shape={frame.shape} dtype={frame.dtype} mean={frame.mean():.1f}"
    print(f"{'OK  ' if ok else 'FAIL'} {key}: {detail}")

if args.save_dir:
    save_observation(obs, args.save_dir)

if args.motion_test:
    print("\n--- Motion test: lift up + wrist pitch, wrist views must change ---")
    art = Articulation(prim_paths_expr="/World/Aloha/Geometry/base_link")
    art.initialize()
    dof_names = art.dof_names
    targets = art.get_joint_positions().copy()
    targets[0][dof_names.index("vertical_move")] = 0.3
    for side in ("left", "right"):
        targets[0][dof_names.index(f"{side}_joint2")] = 1.0  # pitch arms forward/down
    art.set_joint_position_targets(targets)
    for _ in range(240):
        kit.update()

    obs_after = get_camera_observation()
    if args.save_dir:
        save_observation(obs_after, args.save_dir, suffix="_moved")
    all_ok = True
    for key in obs:
        before, after = obs[key], obs_after[key]
        if before is None or after is None:
            print(f"FAIL {key}: missing frame")
            all_ok = False
            continue
        diff = float(np.abs(after.astype(np.int16) - before.astype(np.int16)).mean())
        changed = diff > 2.0
        print(f"{'OK  ' if changed else 'FAIL'} {key}: mean abs pixel diff after motion = {diff:.2f}")
        all_ok = all_ok and changed
    print("MOTION TEST:", "PASS" if all_ok else "FAIL")

kit.close()
