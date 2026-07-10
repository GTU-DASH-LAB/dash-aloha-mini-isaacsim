# CLAUDE.md — living project doc

> Update this file whenever you (an agent working this repo) learn something a future
> session would otherwise have to re-derive. This is not a changelog — prune stale
> entries, don't just append. See [`plan.md`](plan.md) for the phased task list; this
> file is for facts, decisions, and gotchas that don't fit a checkbox.

## What this is

Isaac Sim 6.0.1 simulation of **AlohaMini1** — a mobile robot (3-wheel omni base +
vertical lift) carrying two SO-101 arms — built from the real URDF/mesh assets in
[liyiteng/alohamini](https://github.com/liyiteng/alohamini) (Apache-2.0). Full physics,
controllable from `scripts/control_terminal.py` (terminal) and, in principle, from the
Isaac Sim UI (see Phase 5 caveat below — not independently verifiable from a headless
environment). No ROS2 by default (see `plan.md` Phase 6 — ask before adding it).

## Ground rules

1. **Isaac Sim is heavy (GPU-bound).** Prefer headless (`--no-window`) verification
   scripts over launching the full GUI when just checking correctness. Close test
   instances after verifying rather than leaving them running.
2. **Step-by-step, verify after each step.** Don't chain multiple unverified phases in
   one shot.
3. **Report before silently downgrading fidelity** — and if you do end up downgrading
   (e.g. the kinematic base-drive fallback below), document exactly what was tried and
   why, don't just leave a fallback in place unexplained.
4. **Don't overwrite the SolidWorks-derived mass/inertia values** in the URDF — they're
   real, computed from the actual CAD models.
5. **Verify numbers against raw source, not AI-paraphrased summaries** when porting
   constants from another repo — a paraphrase once missed a real sign-negation in the
   LeKiwi kinematics. Fetch and grep the actual file.
6. **When commanding multiple joint targets in the same session, don't
   read-modify-write from `get_joint_positions()`/`get_joint_velocities()` more than
   once per target array.** Those return the *actual current* state, not the
   previously-commanded target — a second call before the first target is reached will
   silently reset it. Use a persistent target array instead (see
   `control_terminal.py`'s `_position_targets`/`_velocity_targets`). This bit us for
   real: chaining an arm command then a lift command reset the arm's target back to
   ~0 with no error or warning.

## Key technical facts

- **Isaac Sim 6.0.1** lives at `~/isaacsim` (Kit's embedded Python is **3.12**). URDF
  importer extension: `isaacsim.asset.importer.urdf`.
- **Upstream URDF**: `assets/upstream_alohamini1/urdf/Aloha.urdf`, SolidWorks-exported.
  17 DOF: `wheel1/2/3` (continuous), `vertical_move` (prismatic, base_link→vertical_link),
  `left_joint1..6` / `right_joint1..6` (revolute, mirror SO-101's Rotation/Pitch/Elbow/
  Wrist_Pitch/Wrist_Roll/Jaw). Joint limits/effort/velocity were all zeroed by the
  exporter originally — now patched (`scripts/patch_urdf_joint_limits.py`).
- **19 mesh files** vendored under `assets/upstream_alohamini1/meshes/` (~15MB, no
  git-lfs needed).
- **All constants (arm gains, lift range, wheel kinematics, jaw limits) live in one
  place**: `scripts/alohamini1_specs.py`. Don't duplicate numbers elsewhere.
  Sources: arm gains from NVIDIA's `Sim-to-Real-SO-101-Workshop` (same physical arm,
  Isaac Lab config but gains transfer); lift range + wheel kinematics from
  `liyiteng/lerobot_alohamini` (real hardware control code for this exact robot).
- **No pre-built ALOHA/SO-101 asset ships with Isaac Sim 6.0.1** — confirmed by
  searching `~/isaacsim/exts` and the Nucleus asset cache.
- **Isaac Sim's asset CDN root** on this machine resolves to
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`.
- **Setting a prim's transform via raw USD (`UsdGeom.XformCommonAPI`) while physics is
  actively stepping has ~no effect.** PhysX keeps its own internal rigid body state and
  only pushes physics→USD, not the reverse, except through the tensor API. Verified
  directly: the same write persisted fine with the timeline paused, but did nothing
  once `timeline.play()` was active. Use `Articulation.set_world_poses()` (goes through
  `physics_view.set_root_transforms()`) to actually teleport a body during simulation.

## File map

```
assets/upstream_alohamini1/   vendored URDF + meshes + config from liyiteng/alohamini
assets/usd/Aloha/             imported USD (regenerate via urdf_import.py if URDF changes)
assets/usd/scene.usda         composed scene: environment + robot + drives + collision fixes
                               (regenerate: build_scene.py -> configure_physics.py -> fix_wheel_collision.py)
scripts/alohamini1_specs.py   single source of truth for all physical constants
scripts/patch_urdf_joint_limits.py   patches the URDF's joint limits (already applied)
scripts/build_scene.py        composes environment + robot -> scene.usda
scripts/configure_physics.py  applies joint drives (arms, lift, wheels) onto scene.usda
scripts/fix_wheel_collision.py   fixes the base-shell-blocks-wheels collision bug (see Gotchas)
scripts/verify_import.py      headless import sanity check + screenshot
scripts/verify_physics.py     headless physics stability check + joint command test
scripts/control_terminal.py   Phase 4 terminal control (one-shot + REPL)
docs/                         milestone screenshots + import logs
plan.md                       phased task list, checkboxes
CLAUDE.md                     this file
```

## Gotchas hit so far

- **`package://` URDF mesh paths need `--ros-package` or meshes silently vanish.**
  Always pass `--ros-package "Aloha:/absolute/path/to/assets/upstream_alohamini1"`.
- **`stage.Traverse()` + `UsdGeom.Mesh` type-checking undercounts** on this asset (point
  instancing). Verify via `UsdGeom.BBoxCache` + an actual screenshot, using
  `frame_viewport_prims(viewport, prims=[...])` to aim the headless camera (it doesn't
  auto-frame new geometry).
- **A wheel-name/angle mapping bug caught by checking raw URDF origins, not assuming.**
  The correct pairing is `["wheel1", "wheel3", "wheel2"]` against LeKiwi's `[150,-90,30]`
  degree angle order — see the derivation comment in `alohamini1_specs.py`.
- **Environment choice matters for visibility, not just lighting.** Using
  `Isaac/Environments/Grid/default_environment.usd` (plain floor) instead of
  `Simple_Room` (a furniture prop there obscured the robot's base).
- **Convex collision approximations can seal over intentional concavities.** Both
  Convex Hull *and* Convex Decomposition of `base_link`'s shell extended down far
  enough to block the wheels from ever touching ground — confirmed via raycast
  diagnostics (probing straight down at each wheel's XY position consistently hit
  `base_link`, not the wheel, at the wrong height). Fixed by disabling the shell's
  collision entirely and giving each wheel its own explicit sphere collider + friction
  material (`scripts/fix_wheel_collision.py`).
- **USD instance proxies can't be edited directly.** The offending collision prims
  were point-instanced (shared prototypes); authoring overrides on them raised
  `"authoring to an instance proxy is not allowed"`. Fix: walk up the prim ancestry
  checking `prim.IsInstance()` to find the actual instance root, then
  `prim.SetInstanceable(False)` on *that specific instance* (doesn't affect other
  instances of the same prototype elsewhere in the scene).
- **A drive's `damping` must be sized to the joint's actual inertia, not picked
  arbitrarily.** Used `damping=1e5` for wheel "velocity servos" (reasoning: "high
  damping = velocity control") — with the wheel's tiny rotational inertia
  (~5e-5 kg·m²) this caused violent bang-bang oscillation (a wheel hit -19.7 rad/s
  against a -2.6 rad/s target). Fixed: damping=2.0, plus bumped
  `solver_velocity_iteration_count` 1→4.
- **Real wheel traction stayed near-zero even after fixing collision + oscillation.**
  A single wheel spinning alone (no other wheels involved) produced ~200x less
  translation than expected. Not fully root-caused (diminishing returns past a certain
  point) — likely needs proper rolling-friction/anisotropic-friction tuning beyond a
  first pass. **Resolution**: kinematic base drive (see below), the plan's
  pre-approved fallback for exactly this situation.
- **Kinematic root-teleporting fights concurrent arm-joint convergence.**
  Continuously calling `set_world_poses()` on the articulation root every physics step
  while also trying to drive arm joints to new targets measurably degrades arm
  convergence (verified: `left_joint1` commanded to 0.4 rad reached only ~0.05 rad
  while the base was being actively teleported in the same loop). Tried setting root
  *velocity* instead (less invasive in theory) — didn't help translation (contact
  friction from the still-present, non-driving wheel contact damped it out between
  refreshes) and even refreshing every single step caused Z-height drift. Landed on:
  teleport-based kinematic drive for the base, used **sequentially** (drive base,
  `stop`, then command arms) rather than simultaneously. This is a known, documented
  limitation, not a silent one.

## Current status

**Phases 0-4 done and verified. Phase 5 partially verified (see caveat).**

Pipeline: patched URDF → `isaacsim.asset.importer.urdf` → composed with a ready-made
environment → joint drives applied → wheel collision fixed → terminal control script.

- **Visual**: robot renders correctly, grounded, both arms + lift column visible.
  Screenshots: `docs/scene_verification.png`, `docs/physics_verification.png`,
  `docs/control_demo.png` (shows a fully changed pose: lift extended, both arms moved,
  robot visibly repositioned from base driving).
- **Physics**: zero NaN/Inf across all verification runs, base height stable, arm/lift
  joints converge exactly to commanded targets (position drives, verified repeatedly).
- **Terminal control** (`scripts/control_terminal.py`): arm joints, gripper, lift all
  work correctly, including when chained together (after fixing the target-clobbering
  bug). Base locomotion works via kinematic drive, **used sequentially, not
  simultaneously with new arm commands** (documented limitation above).
- **UI control** (Phase 5): the articulation has proper `DriveAPI` position/velocity
  drives on every joint — exactly the mechanism the Property panel / Articulation
  Inspector's sliders use, and that mechanism is verified working via script. **Not
  independently verified by actually clicking a slider** — this environment has no
  display. See `plan.md` Phase 5 for the exact steps to check yourself in a
  non-headless Isaac Sim session.

## Next step

Nothing blocking. Optional future work: root-cause the wheel traction issue properly
(would remove the kinematic-drive limitation), or Phase 6 (ROS2) if requested.
