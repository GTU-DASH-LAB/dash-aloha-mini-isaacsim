#!/usr/bin/env python3
"""Bring the policy up on a real robot.

    # 1. no robot at all -- webcam, printed commands. Do this first.
    python3 robot/run_robot.py --instruction "Go down the corridor." \
        --policy-host 192.168.1.50

    # 2. your own base, still held at zero by `enabled: false` in robot/robot.yaml
    python3 robot/run_robot.py --base mypkg.driver:MyBase --instruction "..."

    # 3. driving: set `enabled: true`, or POST /enable, with a hardware cutoff in reach.

The default `--base` is `PrintBase`, which moves nothing. That is not a placeholder to be
replaced before the first run -- it is the first run.

WHERE YOUR CODE PLUGS IN. `--camera`, `--base` and `--range` each take `module:attr`
pointing at anything callable that returns an object with the right methods. Your driver
imports nothing from this repository; see `robot/interfaces.py` for the three shapes and
`robot/README.md` for a worked example.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "nav" / "policy_server"))

import config as robot_config  # noqa: E402
import control_api  # noqa: E402
from client import PolicyClient, PolicyServerError  # noqa: E402
from runner import RobotRunner  # noqa: E402


def _load(spec: str, what: str):
    """Turn `package.module:factory` into the object it returns.

    Accepts `module:attr(arg=…)`? No -- deliberately. A spec that can carry arguments
    becomes a second configuration language living on the command line, and this repo
    already keeps its configuration in files that get committed. Write a zero-argument
    factory in your own module and put the arguments there, where they are reviewable.
    """
    if ":" not in spec:
        raise SystemExit(
            f"--{what} wants 'module:attr', e.g. 'mypkg.driver:MyBase'; got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(
            f"could not import {mod_name!r} for --{what}: {exc}\n"
            f"Your driver's directory must be on PYTHONPATH; it does not need to be "
            f"inside this repository.") from exc
    try:
        factory = getattr(mod, attr)
    except AttributeError as exc:
        raise SystemExit(f"{mod_name!r} has no attribute {attr!r}") from exc
    return factory()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the navigation policy on a real robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", required=True,
                    help="what to tell the model, in plain language")
    ap.add_argument("--camera", default="drivers.opencv_camera:OpenCVCamera",
                    help="module:attr returning a CameraSource")
    ap.add_argument("--base", default="drivers.sim_base:PrintBase",
                    help="module:attr returning a Base. The default moves nothing.")
    ap.add_argument("--range", dest="range_sensor",
                    default="drivers.null_range:NullRangeSensor",
                    help="module:attr returning a RangeSensor, or 'none'")
    ap.add_argument("--policy-host", default="127.0.0.1",
                    help="where launch_qwen.sh is running")
    ap.add_argument("--policy-port", type=int, default=8766)
    ap.add_argument("--limits", default=str(robot_config.DEFAULT_PATH),
                    help="the YAML the runner hot-reloads")
    ap.add_argument("--control-port", type=int, default=8770)
    ap.add_argument("--control-bind", default="127.0.0.1",
                    help="0.0.0.0 exposes the control API with no authentication")
    ap.add_argument("--no-control-api", action="store_true")
    ap.add_argument("--controller", default="braking",
                    help="braking (shipping) | pursuit | guided | holonomic")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--frame-dir", default="/tmp/qvla-robot-frames")
    ap.add_argument("--wait-ready-s", type=float, default=900.0,
                    help="how long to wait for the model to finish loading")
    args = ap.parse_args()

    policy = PolicyClient(args.policy_host, args.policy_port)
    print(f"[run] waiting for the policy server at "
          f"{args.policy_host}:{args.policy_port} ...", flush=True)
    try:
        # A cold start loads ~29 GiB of weights, so a robot that gave up after the
        # default socket timeout would look like a network problem and be a patience
        # problem. `wait_until_ready` polls /health until the model reports loaded.
        health = policy.wait_until_ready(timeout_s=args.wait_ready_s)
    except (PolicyServerError, OSError) as exc:
        print(f"[run] policy server unreachable: {exc}\n"
              f"      Start it with: nav/policy_server/launch_qwen.sh\n"
              f"      If it is on another machine, that machine must serve on an "
              f"address this robot can reach (NAV_POLICY_HOST=0.0.0.0).", flush=True)
        return 1
    print(f"[run] policy ready: {health.get('model')} "
          f"format={health.get('format')} think={health.get('think_level')}", flush=True)
    if not health.get("accepts_image_bytes"):
        # The path route reads files off the SERVER's disk. On a robot that shares no
        # filesystem with the GPU box, every frame would silently resolve to nothing.
        print("[run] this server predates the image-bytes route and cannot be fed by a "
              "robot. Restart it from this checkout.", flush=True)
        return 1

    limits = robot_config.LiveLimits(args.limits)
    lim = limits.current()
    print(f"[run] limits from {args.limits}: enabled={lim.enabled} "
          f"cruise={lim.cruise_mps} m/s max={lim.max_speed_mps} m/s "
          f"guard_stop={lim.guard_stop_m} m", flush=True)
    if not lim.enabled:
        print("[run] enabled=false -- the full loop runs and the base is held at ZERO. "
              "Watch the commands, then set enabled: true in the YAML (it is picked up "
              "without a restart).", flush=True)

    camera = _load(args.camera, "camera")
    base = _load(args.base, "base")
    range_sensor = (None if args.range_sensor.lower() == "none"
                    else _load(args.range_sensor, "range"))

    runner = RobotRunner(
        camera=camera, base=base, policy=policy, limits=limits,
        range_sensor=range_sensor, frame_dir=args.frame_dir,
        controller_name=args.controller)

    if not args.no_control_api:
        control_api.serve(runner, host=args.control_bind, port=args.control_port,
                          limits_path=args.limits)

    print(f"[run] instruction: {args.instruction!r}", flush=True)
    print(f"[run] frames -> {args.frame_dir}   Ctrl-C to stop.", flush=True)
    out = runner.run(args.instruction, timeout_s=args.timeout_s)
    print(f"[run] finished: {out['state']} ({out['message'] or 'no message'}) "
          f"after {out['elapsed_s']} s, {out['decisions']} decisions, "
          f"{out['guard_interventions']} guard stops, {out['recoveries']} recoveries",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
