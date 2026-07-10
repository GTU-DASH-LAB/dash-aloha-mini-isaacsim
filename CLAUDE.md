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

## Gotchas hit so far

- **`package://` URDF mesh paths need `--ros-package` or meshes silently vanish.**
  `Aloha.urdf` references meshes as `package://Aloha/meshes/base_link.STL`. The URDF
  importer (`isaacsim.asset.importer.urdf`) does **not** error if it can't resolve that
  -- it just imports the robot with zero mesh geometry (correct link/joint hierarchy,
  correct physics, but literally invisible). No warning in the log either. Always pass
  `--ros-package "Aloha:/absolute/path/to/assets/upstream_alohamini1"` (the directory
  that contains `meshes/`) when importing this URDF. Caught by: import "succeeded" but
  a headless screenshot showed nothing at the origin, and a full stage prim dump
  showed zero `def Mesh` prims anywhere until this was added.
- **`stage.Traverse()` + `UsdGeom.Mesh` type-checking is unreliable on the imported
  asset** -- the importer's output uses point-instancing (`payloads/instances.usda`,
  `payloads/geometries.usd`) for the many symmetric/repeated parts, so a naive
  `prim.IsA(UsdGeom.Mesh)` walk from the root undercounts (reports 0 even when meshes
  are genuinely present and rendering). Don't trust that check for "did it import
  correctly" -- trust a bounding-box computation (`UsdGeom.BBoxCache` on `/Aloha`) and
  an actual rendered screenshot instead. `frame_viewport_prims(viewport,
  prims=["/Aloha"])` (from `omni.kit.viewport.utility`) is the reliable way to aim a
  headless camera at freshly imported geometry -- the default camera does not
  auto-frame it.
- **Screenshots of an empty stage are unlit (solid black except a dark silhouette).**
  Expected -- there's no light source yet. Don't mistake "dark but correctly shaped
  silhouette at the right bounding box" for a failure; it means geometry is fine and
  the next step (loading a proper ready-made environment) will supply lighting.

## Current status

**Phase 0, 1, 2 all done and verified.** Working pipeline: patched URDF ->
`isaacsim.asset.importer.urdf` -> `assets/usd/Aloha/Aloha.usda` -> composed with a
ready-made environment -> `assets/usd/scene.usda` (`scripts/build_scene.py`). Milestone
screenshot at `docs/scene_verification.png` confirms the whole robot renders correctly:
base grounded, lift column visible, both arms symmetric with gripper geometry, proper
lighting. Environment used: `Isaac/Environments/Grid/default_environment.usd` (plain
flat grid -- `Simple_Room` also works but has a furniture prop that the robot spawned
partly inside; switched to Grid to keep the base fully visible for now, revisit later
if a proper room look is wanted).

Isaac Sim's asset CDN root on this machine resolves to
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`
(confirmed reachable, network access works for streaming ready-made environments).

Not yet done: Phase 3 (per-joint drive stiffness/damping -- importer warned "Stiffness
and damping not available" for every joint, since the URDF has no `<dynamics>` tags --
expected, this is exactly what Phase 3 adds), wheel physics decision, terminal control
script, UI control verification.

## Next step

Phase 3: apply the SO-101 actuator gains table (above) as per-joint drives on
`left_joint1..6`/`right_joint1..6` in `scene.usda`, set a lift joint drive, and tackle
the wheel-drive decision (real per-wheel velocity drive + simplified collision proxy,
with a documented fallback if that's unstable -- see plan.md Phase 3).
