"""WHEN does a GPU0-pinned Kit process open its ~250 MiB CUDA context on GPU1 (the policy server's card)?

g1/check (board 20:19) saw every Kit process on this machine holding 248-256 MiB on GPU1 although it is
launched with {"active_gpu": 0, "physics_gpu": 0, "multi_gpu": False}. CUDA_VISIBLE_DEVICES must not be set
for Kit (Kit warns it "can lead to undesired behavior or crashes"), so the question is which component
creates the context, and whether a Kit setting stops it.

Method: a thread polls `nvidia-smi --query-compute-apps` every 0.1 s and records, with wall time since this
script started, how much memory THIS pid holds on each GPU. The main thread marks phases:
  boot      SimulationApp(...) constructor (all startup extensions; Kit prints "[t s] [ext: X] startup"
            lines with its own clock, which starts within ~0.1 s of ours -- compare by time)
  stage     new stage + a cube + PhysicsScene, timeline.play(), 60 updates (CPU PhysX)
  warp      import warp; wp.init(); a 1-element array on cuda:0
  camera    isaacsim.sensors.camera.Camera 640x480 + 10 updates + get_rgba()
The report says which phase the GPU1 allocation first appeared in. Run variants with --extra to test
settings (each value is passed to Kit as a command-line setting).

    ISAAC_SLOT_MIN_FREE_MIB=4000 nav/tools/isaac_slot.sh ~/robotics/isaacsim-6.0.1/python.sh \
        nav/tools/probe_kit_gpu1_context.py --out <dir> [--extra --/renderer/multiGpu/enabled=false ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]   # nav/tools/profile.py shadows stdlib

T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--extra", nargs="*", default=[], help="extra Kit command-line settings, e.g. --/foo/bar=1")
ap.add_argument("--skip", default="", help="comma list of phases to skip: stage,warp,camera")
args, _unknown = ap.parse_known_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
PID = os.getpid()


def gpu_uuids() -> dict:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True)
    out = {}
    for line in r.stdout.strip().splitlines():
        i, u = [x.strip() for x in line.split(",")]
        out[u] = int(i)
    return out


UUID2IDX = gpu_uuids()
samples: list[tuple[float, dict]] = []
phase = {"name": "pre_boot"}
phase_marks: list[tuple[str, float]] = [("pre_boot", 0.0)]
stop = threading.Event()


def poll():
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        mine = {}
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[1] == str(PID):
                mine[UUID2IDX.get(parts[0], parts[0])] = int(parts[2])
        samples.append((round(time.time() - T0, 2), mine, phase["name"]))
        time.sleep(0.1)


th = threading.Thread(target=poll, daemon=True)
th.start()


def mark(name):
    phase["name"] = name
    phase_marks.append((name, round(time.time() - T0, 2)))
    print(f"[gpu1probe] phase {name} at {phase_marks[-1][1]} s", flush=True)


report: dict = {"pid": PID, "args": vars(args), "gpus": UUID2IDX}
mark("boot")
from isaacsim import SimulationApp  # noqa: E402

cfg = {"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False}
if args.extra:
    cfg["extra_args"] = list(args.extra)
kit = SimulationApp(cfg)
rc = 1
try:
    import carb

    s = carb.settings.get_settings()
    report["settings"] = {k: s.get(k) for k in ("/renderer/multiGpu/enabled", "/renderer/activeGpu", "/physics/cudaDevice",
                                               "/renderer/multiGpu/maxGpuCount", "/app/gpu/activeGpu",
                                               "/plugins/carb.cudainterop/enabled", "/rtx/multiGpu/enabled")}
    skip = set(x for x in args.skip.split(",") if x)
    if "stage" not in skip:
        mark("stage")
        import omni.timeline
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        for _ in range(5):
            kit.update()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        sc = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim()).CreateEnableGPUDynamicsAttr().Set(False)
        cube = UsdGeom.Cube.Define(stage, "/World/Cube")
        cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        omni.timeline.get_timeline_interface().play()
        for _ in range(60):
            kit.update()
    if "warp" not in skip:
        mark("warp")
        import warp as wp

        wp.init()
        a = wp.zeros(1, dtype=float, device="cuda:0")
        wp.synchronize()
        report["warp_devices"] = [str(d) for d in wp.get_devices()]
        for _ in range(10):
            kit.update()
    if "camera" not in skip:
        mark("camera")
        from isaacsim.sensors.camera import Camera

        cam = Camera(prim_path="/World/ProbeCam", resolution=(640, 480))
        cam.initialize()
        for _ in range(10):
            kit.update()
        rgba = cam.get_rgba()
        report["camera_frame_shape"] = None if rgba is None else list(rgba.shape)
    mark("idle_end")
    time.sleep(1.5)
    rc = 0
except Exception:  # noqa: BLE001
    import traceback

    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
finally:
    stop.set()
    th.join(timeout=2)
    first = {}
    for t, mine, ph in samples:
        for g, mib in mine.items():
            if g not in first and mib > 0:
                first[g] = {"t_s": t, "phase": ph, "mib": mib}
    report["phase_marks"] = phase_marks
    report["first_seen"] = first
    report["max_mib"] = {str(g): max((m.get(g, 0) for _, m, _ in samples), default=0) for g in UUID2IDX.values()}
    report["samples"] = samples
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[gpu1probe] first_seen {first} max {report['max_mib']}", flush=True)
    kit.close()
    os._exit(rc)
