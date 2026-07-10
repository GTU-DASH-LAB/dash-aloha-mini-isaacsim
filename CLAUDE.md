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
- **Isaac Sim's bundled Python has no stdlib `readline` module** (built without
  libreadline). `import readline` raises `ModuleNotFoundError`. Fix:
  `~/isaacsim/python.sh -m pip install gnureadline`, then `import gnureadline as
  readline` (drop-in). Made this an optional import with a graceful fallback message
  rather than a hard dependency, since it only affects REPL command history/editing.
- **The user account on this machine is not in the `input` group**, so `evdev` can't
  open `/dev/input/eventN` device nodes (owned `root:input`, mode 0660) even once a
  controller is connected. Needs `sudo usermod -aG input $USER` + re-login. Documented
  prominently since it's a real blocker the first time someone tries `--joystick`.
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
- **AnyDesk (and remote desktop tools generally) don't forward USB/gamepad devices**,
  only keyboard/mouse/screen — confirmed via web search, not assumed. If the
  controller is on a different machine than the one running Isaac Sim, use
  `--joystick-network` (`control_terminal.py`) + `scripts/joystick_bridge_local.py`
  (runs on the controller's machine) instead of `--joystick`. Deliberately built on
  **TCP, not UDP**, specifically so a plain `ssh -L` tunnel works with no extra tools
  — `ssh -L` only forwards TCP by default, and getting UDP through would need
  sshuttle/a VPN. Verified end-to-end on the remote side with a real TCP client
  (connected, drove it through all four modes, clean disconnect handling) — the local
  half (pygame reading an actual controller on someone else's machine) is inherently
  untestable from here.
- **A real PS4 controller's button mapping on macOS differs from the generic Linux
  default** — confirmed via targeted single-button tests (not a guess): L1=button 9,
  R1=button 10 (not 4/5 as commonly assumed), L2/R2 are analog-only on axes 4/5 (no
  separate digital button event), and the d-pad reports as `hats=0` -- doesn't come
  through as a hat at all on this setup, so wrist-roll (joint5) control via d-pad
  doesn't work yet on this specific controller/OS combo (not root-caused further).
  `DEFAULT_MAPPING` in `joystick_bridge_local.py` reflects the verified values.
- **`enabled_self_collisions=False` (copied from NVIDIA's single-arm SO-101 config)
  let the left arm, right arm, lift, and base pass through each other.** Made sense
  for NVIDIA's case (a single arm's own adjacent links would otherwise constantly
  false-positive at their shared joint) but this robot's whole body is one
  articulation, so it silently disabled collision between logically-separate parts
  too. Flipped to `True` in `configure_physics.py`; verified via `verify_physics.py`
  this doesn't introduce instability at the arms' own joints (identical exact
  convergence and base-height stability to the disabled case).
- **Gripper control was snap-to-extreme, not proportional** — `set_gripper()` jumped
  straight to `JAW_OPEN_RAD`/`JAW_CLOSED_RAD` the instant a button was pressed, with
  no way to stop partway. Live user testing also suggested the open/closed labels
  might be backwards from physical reality (flagged as unverified when first
  written). Fixed by making joystick gripper control incremental/rate-based like
  every other joint (hold to move gradually, release to stop) -- this also sidesteps
  needing to know for certain which label is physically correct, since you can just
  watch it move and release when it looks right.
- **The Jaw joint's limits from NVIDIA's reference asset (-0.174533 to 1.745329 rad)
  do NOT correspond to a fully-closed gripper on THIS mesh.** Different CAD source
  than NVIDIA's SO-ARM101-USD.usd (same physical arm design, different jaw linkage
  geometry) -- so even though those numbers are real, verified values read directly
  off NVIDIA's asset, they don't transfer to this specific STL's finger geometry.
  User reported the gripper "wasn't closing all the way"; confirmed empirically by
  rendering close-up screenshots at several joint angles with the limit temporarily
  widened for testing: -0.174533 leaves a visible gap (not closed at all), 1.745329
  is genuinely fully open, and the fingers actually meet at **-1.570796 (-90 deg)** --
  well outside the original range. Corrected the real limit in `Aloha.urdf` itself
  (not just a script-side clamp) for both `left_joint6` and `right_joint6`, and
  updated `JAW_OPEN_RAD`/`JAW_CLOSED_RAD` + `ARM_JOINT_LIMITS_RAD[6]` in
  `alohamini1_specs.py` to match. Re-ran the full pipeline (reimport -> rebuild scene
  -> reconfigure physics -> refix wheel collision) and reverified: `docs/
  gripper_closed_final.png` shows the fingers actually touching, and
  `verify_physics.py` still passes (no NaN/Inf, exact convergence, stable base
  height) with the wider range.
- **Lesson for any other visually-unverified limit/direction assumption still in the
  codebase**: don't trust a reference asset's numeric range just because it's a real,
  correctly-read number from a legitimate source -- if the source is a *different*
  CAD/mesh than what's actually being simulated, the number can be confidently wrong
  for this specific geometry. Render a close-up screenshot at the extremes and
  actually look, the same way this was caught.
- **Widened the gripper close limit a bit further** (-1.570796 -> -1.85 rad, ~-106 deg)
  per user request for more squeeze margin (e.g. gripping thin objects) -- verified
  visually clean at -1.85 and even -2.0 (no bad mesh interpenetration), settled on
  -1.85 as a reasonable margin without excessive over-rotation.
- **Arm-to-arm self-collision verified actually working**, not just "doesn't destabilize
  rest pose" -- direct test: drove both arms' joint1 toward each other, and they got
  physically stuck partway (commanded -1.9 rad, actual only reached 0.66 rad) with
  heavily overlapping bounding boxes on the wrist links, confirming a real collision
  stopped the motion. (First attempt at this test used the wrong rotation sign and
  showed the arms moving apart instead of together, which looked like nothing was
  happening -- always double check the direction actually brings parts together
  before concluding collision "isn't working".)
- **Added lift control to the joystick scheme**: Triangle=up, Square=down, using the
  same increment-while-held pattern as the gripper, and independent of the L1/L2/R2
  arm/base mode selection (works no matter what's currently selected). On
  `joystick_bridge_local.py` (the macOS/pygame side actually in use), button indices
  2/3 for square/triangle are inferred from the standard SDL face-button ordering,
  not individually confirmed one-at-a-time the way L1/R1 were -- verify with --debug
  if it doesn't respond.

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
