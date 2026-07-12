"""Open the scene in the Isaac Sim GUI with one extra viewport per robot camera, so
all three camera views (forward / wrist_left / wrist_right) are visible live next to
the main perspective view. Physics starts playing automatically, so the wrist views
move when you command the arms (e.g. from a second terminal running
scripts/control/control_terminal.py --joystick-network, or the GUI joint sliders).

Usage:
    # GUI: main viewport + three camera viewports, live
    ~/isaacsim/python.sh scripts/cameras/view_cameras.py

    # Headless self-test: capture each camera viewport to PNG and exit (used to
    # verify this script works without needing someone at the GUI)
    ~/isaacsim/python.sh scripts/cameras/view_cameras.py --screenshot-test --out-dir docs

Manual alternative (no script): in any Isaac Sim viewport, click the camera icon
(top-left of the viewport) -> Cameras -> camera_forward / camera_wrist_left /
camera_wrist_right. Window > Viewport > Viewport 2 gives you a second viewport to
assign a different camera to.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/ root, for alohamini1_specs
from alohamini1_specs import CAMERA_PRIM_PATHS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/scene.usda")
parser.add_argument("--screenshot-test", action="store_true",
                    help="Run headless, capture each camera viewport to PNG, then exit")
parser.add_argument("--out-dir", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/docs",
                    help="Where --screenshot-test writes viewport_<name>.png")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": args.screenshot_test})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.kit.viewport.utility import create_viewport_window  # noqa: E402
import omni.kit.viewport.utility as vp_utility  # noqa: E402

# STEP 1: open the composed scene and wait for the referenced assets (environment,
# robot, tables) to finish loading.
usd_context = omni.usd.get_context()
usd_context.open_stage(args.scene)
stage = usd_context.get_stage()
for _ in range(60):
    kit.update()
stage.Load()
for _ in range(20):
    kit.update()

# STEP 2: one extra viewport window per robot camera, laid out side by side.
# 480x360 keeps the same 4:3 aspect as the real cameras (640x480).
windows = {}
for i, (name, cam_path) in enumerate(CAMERA_PRIM_PATHS.items()):
    window = create_viewport_window(
        name=f"Camera: {name}",
        camera_path=cam_path,
        width=480,
        height=360,
        position_x=60 + i * 500,
        position_y=60,
    )
    windows[name] = window
    print(f"Opened viewport '{name}' -> {cam_path}")
for _ in range(15):
    kit.update()

# STEP 3: start physics so the views are live (arms settle to rest pose, and any
# commands from another process/GUI sliders show up immediately).
timeline = omni.timeline.get_timeline_interface()
timeline.play()

if args.screenshot_test:
    # Let the render pipelines warm up, then grab each camera viewport once.
    for _ in range(60):
        kit.update()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, window in windows.items():
        path = str(out_dir / f"viewport_{name}.png")
        vp_utility.capture_viewport_to_file(window.viewport_api, path)
        for _ in range(10):
            kit.update()
        print(f"Captured {path}")
    kit.close()
else:
    # GUI mode: keep updating until the user closes the Isaac Sim window.
    print("All camera viewports open -- close the Isaac Sim window (or Ctrl+C) to exit.")
    try:
        while kit.is_running():
            kit.update()
    except KeyboardInterrupt:
        pass
    kit.close()
