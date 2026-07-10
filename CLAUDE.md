# CLAUDE.md — living project doc

> Update this file whenever you (an agent working this repo) learn something a future
> session would otherwise have to re-derive. This is not a changelog — prune stale
> entries, don't just append. See [`plan.md`](plan.md) for the phased task list; this
> file is for facts, decisions, and gotchas that don't fit a checkbox.

## What this is

Isaac Sim 6.0.1 simulation of **AlohaMini1** — a mobile robot (3-wheel omni base +
vertical lift) carrying two SO-101 arms — built from the real URDF/mesh assets in
[liyiteng/alohamini](https://github.com/liyiteng/alohamini) (Apache-2.0). Full physics,
controllable from a terminal script and from the Isaac Sim UI. No ROS2 by default (see
`plan.md` Phase 6 — ask before adding it).

## Ground rules

1. **Isaac Sim is heavy (GPU-bound).** Prefer headless (`--no-window`) verification
   scripts over launching the full GUI when just checking correctness. Close test
   instances after verifying rather than leaving them running.
2. **Step-by-step, verify after each step.** Don't chain multiple unverified phases in
   one shot — `plan.md` is phased specifically so each phase gets its own checkpoint.
3. **Report before silently downgrading fidelity.** E.g. if the wheel-physics default
   approach in Phase 3 turns out unstable, come back with what was tried and what the
   fallback looks like — don't just quietly simplify and move on.
4. **Don't overwrite the SolidWorks-derived mass/inertia values** in the URDF — they're
   real, computed from the actual CAD models. The only known-bad values are the
   joint `limit` blocks (all zeroed by the exporter — expected, needs patching).

## Key technical facts (established during planning, verified where noted)

- **Isaac Sim 6.0.1** lives at `~/isaacsim` (Kit's embedded Python is **3.12**). URDF
  importer extension: `isaacsim.asset.importer.urdf`.
- **Upstream URDF**: `assets/upstream_alohamini1/urdf/Aloha.urdf`, SolidWorks-exported
  (`sw_urdf_exporter`). 17 DOF: `wheel1/2/3` (continuous), `vertical_move` (prismatic,
  base_link→vertical_link), `left_joint1..6` (vertical_link→left_base(fixed)→arm chain),
  `right_joint1..6` (mirror). All revolute/prismatic joints have `lower=upper=effort=
  velocity=0` in the raw URDF — verified by direct grep, not an assumption. Must be
  patched before the arms/lift will move under drive control.
- **19 mesh files** vendored under `assets/upstream_alohamini1/meshes/` (~15MB total, no
  git-lfs needed). Package structure (`package.xml`, `CMakeLists.txt`, ROS2
  `ament_cmake` format) is vestigial from the original ROS2 visualization package this
  URDF shipped in — irrelevant to Isaac Sim import, kept only for provenance.
- **SO-101 actuator reference values** — from NVIDIA's own tuned config,
  `~/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/so101.py` (Isaac Lab,
  not standalone Isaac Sim, but the gains transfer since it's the same physical arm):

  | Joint (SO-101 name → our URDF joint) | stiffness | damping | effort_limit |
  |---|---|---|---|
  | Rotation → `{side}_joint1` | 55 | 0.7 | 30 |
  | Pitch → `{side}_joint2` | 30 | 0.8 | 30 |
  | Elbow → `{side}_joint3` | 25 | 0.7 | 30 |
  | Wrist_Pitch → `{side}_joint4` | 12 | 0.5 | 30 |
  | Wrist_Roll → `{side}_joint5` | 7 | 0.5 | 30 |
  | Jaw (gripper) → `{side}_joint6` | 4 | 0.3 | 30 |

  Solver settings used there: `solver_position_iteration_count=32`,
  `solver_velocity_iteration_count=1`, `enabled_self_collisions=False`. Good starting
  point for our arms too. Exact joint angle *limits* (radians) weren't in this script —
  they're baked into NVIDIA's compiled `SO-ARM101-USD.usd`; plan is to read them out
  programmatically (Phase 1) rather than guess from public specs.
- **No pre-built ALOHA/SO-101 asset ships with Isaac Sim 6.0.1** — confirmed by directly
  searching `~/isaacsim/exts` and the Nucleus asset cache. The closest bundled reference
  is a MuJoCo (not USD) test scene under
  `isaacsim.pip.newton/pip_prebundle/mujoco_warp/test_data/aloha_pot/` — not usable
  directly, XML format, different physics stack.
- **Wheel layout**: 3 wheels at roughly 120° apart (URDF origin `rpy` z-rotations
  ≈ -2.11, 2.09, and presumably ~0 for the third), consistent with a holonomic
  omni-wheel base. Not yet confirmed whether the STL mesh models actual roller
  geometry or a simplified wheel — check when the mesh is first visually inspected.

## File map

```
assets/upstream_alohamini1/   vendored URDF + meshes + config from liyiteng/alohamini
  urdf/Aloha.urdf             the robot description (needs joint-limit patching)
  meshes/*.STL                19 STL meshes (visual = collision source, CAD-exported)
  config/joint_names_Aloha.yaml   controller_joint_names list (17 entries)
  LICENSE                     upstream Apache-2.0 text
scripts/                      (to be populated — import script, control_terminal.py, etc.)
docs/                         (to be populated — usage docs once Phase 4/5 land)
plan.md                       phased task list, checkboxes
CLAUDE.md                     this file
```

## Current status

Phase 0 (scaffolding) done except the initial commit/push. Phase 1 not started.

## Next step

Patch `Aloha.urdf` joint limits (Phase 1), starting with reading real limits off
`SO-ARM101-USD.usd` via an Isaac Sim script.
