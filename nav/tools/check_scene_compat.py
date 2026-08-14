"""Phase 8 spike: can Isaac Sim 6.0.1 open a DynaNav scene authored for Isaac Sim 5.0?

This is the single largest schedule risk in nav/plan.md -- every later phase assumes
the answer is yes. It also guards against the failure documented in ../CLAUDE.md, where
an environment turned out to be real-world scale (~1000 m across) and caused both a
camera-framing bug and a physics explosion at spawn. So we do not just check that the
stage opens; we measure how big it is and whether its references actually resolved.

Usage:
    /home/gtu-dsa/robotics/isaacsim-6.0.1/python.sh nav/tools/check_scene_compat.py
    ... --scene /path/to/other.usd
"""

import argparse
import json
import sys
import time

DYNANAV = "/home/gtu-dsa/robotics/TIC-VLA/DynaNav"

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=f"{DYNANAV}/assets/office.usd")
parser.add_argument("--load-seconds", type=float, default=90.0,
                    help="How long to pump the app while referenced assets stream in. "
                         "CDN-backed scenes need much longer than local ones.")
parser.add_argument("--json-out", default=None)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

kit = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, Gf  # noqa: E402

report = {"scene": args.scene, "opened": False}

# STEP 1: open the stage. open_stage returns (ok, error) in 6.x.
t0 = time.time()
usd_context = omni.usd.get_context()
result = usd_context.open_stage(args.scene)
ok = result[0] if isinstance(result, (tuple, list)) else bool(result)
report["open_stage_result"] = str(result)

stage = usd_context.get_stage()
if stage is None:
    report["error"] = "open_stage returned no stage"
    print(json.dumps(report, indent=2))
    kit.close()
    sys.exit(1)

# STEP 2: pump the app so referenced/payloaded assets stream in. A scene that pulls
# from Isaac's CDN resolves lazily -- checking the bounding box too early reports the
# box of an empty stage and looks like success.
deadline = time.time() + args.load_seconds
while time.time() < deadline:
    kit.update()
stage.Load()
for _ in range(30):
    kit.update()

report["opened"] = True
report["load_seconds"] = round(time.time() - t0, 1)

# STEP 3: inventory. Prim count separates "opened an empty stub" from "opened a scene".
prims = list(stage.Traverse())
report["prim_count"] = len(prims)
report["mesh_count"] = sum(1 for p in prims if p.IsA(UsdGeom.Mesh))
report["default_prim"] = str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None
report["up_axis"] = UsdGeom.GetStageUpAxis(stage)
report["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)

# STEP 4: did every reference actually resolve? Unresolved references are the specific
# way a 5.0-authored scene fails in 6.0.1 -- the stage still "opens", just empty.
report["invalid_prims"] = [str(p.GetPath()) for p in prims if not p.IsValid()][:20]
unloaded = [str(p.GetPath()) for p in prims if p.HasAuthoredPayloads() and not p.IsLoaded()]
report["unloaded_payloads"] = unloaded[:20]
report["unloaded_payload_count"] = len(unloaded)

# STEP 5: world-space bounding box -- the ~1000 m scale trap from ../CLAUDE.md.
try:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = cache.ComputeWorldBound(stage.GetPseudoRoot())
    rng = bound.ComputeAlignedRange()
    if rng.IsEmpty():
        report["bbox"] = None
        report["extent_m"] = None
    else:
        mn, mx = rng.GetMin(), rng.GetMax()
        report["bbox"] = {"min": [round(v, 2) for v in mn],
                          "max": [round(v, 2) for v in mx]}
        report["extent_m"] = [round(mx[i] - mn[i], 2) for i in range(3)]
except Exception as exc:  # noqa: BLE001
    report["bbox_error"] = repr(exc)

# STEP 6: verdict against the two things that actually matter downstream.
extent = report.get("extent_m")
verdict = []
if report["prim_count"] < 50:
    verdict.append("SUSPICIOUS: very few prims -- references may not have resolved")
if extent:
    largest = max(extent[0], extent[1])
    report["largest_horizontal_extent_m"] = largest
    if largest > 500:
        verdict.append(f"SCALE TRAP: {largest} m across -- see ../CLAUDE.md, this is the "
                       "class of environment that broke camera framing and physics")
    elif largest < 5:
        verdict.append(f"SUSPICIOUS: only {largest} m across -- likely an empty stub")
    else:
        verdict.append(f"scale OK: {largest} m across")
report["verdict"] = verdict or ["OK"]

print("\n===== DynaNav scene compatibility report =====")
print(json.dumps(report, indent=2))

if args.json_out:
    with open(args.json_out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.json_out}")

kit.close()
