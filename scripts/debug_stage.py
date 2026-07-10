"""Debug helper: dump the full prim tree (type, path) of the imported Aloha stage."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="/home/gtu_dsa/dash-aloha-mini-isaacsim/assets/usd/Aloha/Aloha.usda")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Usd  # noqa: E402

usd_context = omni.usd.get_context()
usd_context.open_stage(args.usd)
stage = usd_context.get_stage()

for _ in range(60):
    kit.update()
stage.Load()
for _ in range(10):
    kit.update()

print("=== Full prim tree ===")
count = 0
for prim in stage.Traverse():
    count += 1
    print(f"{prim.GetPath()}  [{prim.GetTypeName()}]  active={prim.IsActive()} "
          f"loaded={prim.IsLoaded()}")
print(f"\nTotal prims via Traverse(): {count}")

print("\n=== Root layer sublayers/payloads ===")
root_layer = stage.GetRootLayer()
print("subLayerPaths:", list(root_layer.subLayerPaths))

kit.close()
