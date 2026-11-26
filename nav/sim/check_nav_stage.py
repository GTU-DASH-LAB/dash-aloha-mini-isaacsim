#!/usr/bin/env python3
"""Does a nav stage file carry what all four build steps author? Exit 0 if so, 1 if not.

    python3 nav/sim/check_nav_stage.py assets/usd/nav_hospital.usda [more.usda ...]

WHY THIS EXISTS. build_nav_scene.sh used to trust each step's exit status, and Kit cannot
give it one: python.sh exits 0 after an uncaught Python exception and after kit.close(),
whatever happened first (measured 2026-10-05 through isaac_slot.sh, both rc=0). Only a
signal death (the 09-01 segfault, rc 139) ever reached `set -e`. So a step that raised
(add_cameras.py's "Camera parent link not found"), or that warned and skipped (a joint or
wheel prim it could not find), still let the build publish a stage missing that step's
work -- and nav/bench.sh caches a published stage by mtime. The steps now print a sentinel
the shell requires; this checks the RESULT, which is what actually matters and which no
exit status can fake. It reads the .usda as text (no pxr, so it runs in the system python
in milliseconds), and every expected name and count comes from alohamini1_specs.py.

What it looks for, per step (a stage built by the pipeline has all of it):
  1  build_nav_scene.py   the robot reference and a PhysicsScene (a raw-authored layer
                          has none -- CLAUDE.md, "Play simulates nothing without one")
  2  configure_physics.py a drive stiffness on every arm joint, the lift and each wheel,
                          and the articulation's solver iteration count
  3  fix_wheel_collision  one collision sphere per wheel and the friction material, the
                          base_link shell and wheel-mesh colliders disabled
  4  add_cameras.py       every camera (LeRobot set + camera_nav + camera_chase) and the
                          nav headlight
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from alohamini1_specs import (  # noqa: E402
    ARM_JOINT_GAINS,
    ARM_SOLVER_POSITION_ITERATIONS,
    CAMERA_PRIM_PATHS,
    CHASE_CAMERA_PRIM_PATH,
    NAV_CAMERA_PRIM_PATH,
    NAV_HEADLIGHT_PRIM_PATH,
    WHEEL_NAMES,
)


def _leaf(prim_path: str) -> str:
    return prim_path.rstrip("/").rsplit("/", 1)[-1]


def check(text: str) -> list[str]:
    """The problems with one stage's text; empty when it is complete."""
    problems: list[str] = []

    def need(step: str, ok: bool, what: str) -> None:
        if not ok:
            problems.append(f"step {step}: {what}")

    # 1/4 -- authored layer
    need("1/4", re.search(r'def \w* *"Aloha"', text) is not None
         and "Aloha.usda@" in text, 'no "Aloha" prim referencing Aloha.usda')
    need("1/4", re.search(r'def PhysicsScene "', text) is not None, "no PhysicsScene prim")

    # 2/4 -- joint drives
    drives = len(re.findall(r"drive:(?:angular|linear):physics:stiffness\s*=", text))
    want = 2 * len(ARM_JOINT_GAINS) + 1 + len(WHEEL_NAMES)
    need("2/4", drives >= want, f"{drives} drive stiffness attributes, want {want} "
                                "(both arms, the lift, every wheel)")
    need("2/4", re.search(r"physxArticulation:solverPositionIterationCount\s*=\s*"
                          rf"{ARM_SOLVER_POSITION_ITERATIONS}\b", text) is not None,
         f"articulation solverPositionIterationCount {ARM_SOLVER_POSITION_ITERATIONS} "
         "not authored")

    # 3/4 -- wheel colliders
    spheres = len(re.findall(r'def Sphere "collision_sphere"', text))
    need("3/4", spheres == len(WHEEL_NAMES),
         f"{spheres} wheel collision spheres, want {len(WHEEL_NAMES)}")
    need("3/4", re.search(r'def Material "WheelFrictionMaterial"', text) is not None,
         "no WheelFrictionMaterial")
    # ...and the colliders those spheres REPLACE switched off: base_link's shell (its hull
    # reaches below the tyres, so left on the robot rests on its belly -- the 09-01 stub
    # stage drove that robot) and each wheel's own mesh.
    need("3/4", re.search(r'over "base_link"\s*\{\s*bool physics:collisionEnabled = 0',
                          text) is not None,
         "base_link shell collider not disabled")
    disabled = len(re.findall(r"bool physics:collisionEnabled = 0", text))
    need("3/4", disabled >= 1 + len(WHEEL_NAMES),
         f"{disabled} colliders disabled, want {1 + len(WHEEL_NAMES)} (shell + every wheel mesh)")

    # 4/4 -- cameras
    for path in (*CAMERA_PRIM_PATHS.values(), NAV_CAMERA_PRIM_PATH, CHASE_CAMERA_PRIM_PATH):
        name = _leaf(path)
        need("4/4", re.search(rf'def Camera "{re.escape(name)}"', text) is not None,
             f'no camera "{name}"')
    light = _leaf(NAV_HEADLIGHT_PRIM_PATH)
    need("4/4", re.search(rf'def \w+ "{re.escape(light)}"', text) is not None,
         f'no "{light}"')
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    bad = 0
    for arg in argv[1:]:
        path = Path(arg)
        try:
            text = path.read_text()
        except OSError as exc:
            print(f"INCOMPLETE {path}: cannot read ({exc})")
            bad += 1
            continue
        problems = check(text)
        if problems:
            bad += 1
            print(f"INCOMPLETE {path} ({path.stat().st_size} bytes):")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
