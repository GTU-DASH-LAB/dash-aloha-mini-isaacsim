# CLAUDE.md — living project doc

> Update this file whenever you (an agent working this repo) learn something a future
> session would otherwise have to re-derive. This is not a changelog — prune stale
> entries, don't just append. See [`plan.md`](plan.md) for the phased task list; this
> file is for facts, decisions, and gotchas that don't fit a checkbox.

## What this is

Isaac Sim 6.0.1 simulation of **AlohaMini1** — a mobile robot (3-wheel omni base +
vertical lift) carrying two SO-101 arms — built from the real URDF/mesh assets in
[liyiteng/alohamini](https://github.com/liyiteng/alohamini) (Apache-2.0). Full physics,
controllable from `scripts/control/control_terminal.py` (terminal) and, in principle, from the
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
  exporter originally — now patched (`scripts/pipeline/patch_urdf_joint_limits.py`).
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

See `ARCHITECTURE.md` for the full annotated layout + data-flow diagram. Short form:

```
assets/upstream_alohamini1/   vendored URDF + meshes (liyiteng/alohamini, Apache-2.0)
assets/usd/Aloha/             imported robot USD (regenerate if URDF changes)
assets/usd/scene.usda         composed scene (~18KB of references + overrides --
                               regenerate ONLY via scripts/rebuild_all.sh)
scripts/alohamini1_specs.py   single source of truth for all constants + camera paths
scripts/rebuild_all.sh        the 4-step pipeline runner + verification
scripts/pipeline/             build_scene -> configure_physics -> fix_wheel_collision
                               -> add_cameras (+ verify_physics, patch_urdf_joint_limits)
scripts/control/              control_terminal.py (REPL/joystick/network) +
                               joystick_bridge_local.py (runs on YOUR laptop)
scripts/cameras/              capture_cameras.py (LeRobot frames) + view_cameras.py
                               (GUI live viewports)
scripts/tools/                one-off debug/inspection helpers
third_party/lerobot_alohamini git submodule -- official LeRobot integration (camera
                               specs + kinematics source of truth)
docs/                         verification screenshots ; ARCHITECTURE.md ; plan.md ;
                               CLAUDE.md (this file)
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
  material (`scripts/pipeline/fix_wheel_collision.py`).
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
  `--joystick-network` (`control_terminal.py`) + `scripts/control/joystick_bridge_local.py`
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
- **`isaacsim-mcp-server` (whats2000/isaacsim-mcp-server, set up in the separate
  `isaac-sim-ros2-mcp-setup` repo) does NOT work reliably against Isaac Sim 6.0.1**,
  confirmed directly, not just by trusting its own "5.1.0 only" documentation. Basic
  tools (`load_environment`, `load_usd`, `create_object`, `get_prim_info`,
  `step_simulation`) worked fine, but `capture_image` and even a trivial
  `execute_script` both failed, and the extension's log showed a real asyncio
  reentrancy bug in its `socket_server.py` (`_dispatch_command`/`execute_wrapper`
  trying to re-enter an already-executing task) -- a genuine event-loop
  incompatibility with how Kit 6.0.1 schedules tasks differently than 5.1.0, not a
  transient fluke. Useful for quick interactive exploration (e.g. discovering the
  `list_environments` catalog) but not reliable enough to depend on for this
  project -- stick with the direct `python.sh` scripts.
- **The `Office` environment was a real bug, not just an aesthetic choice, and got
  replaced by `Simple_Warehouse`.** User reported "the scene starts from over the
  building very far" and "the robot is not standing but 45 degree falling."
  Root-caused both to the same asset: `Office/office.usd`'s world bbox is roughly
  -528..535m in X, -294..382m in Y (verified via `BBoxCache` -- it's a full
  multi-story building, not a reception room). (1) The robot spawned overlapping some
  part of that building's geometry, and PhysX's separation impulse launched the whole
  articulation across the room within ~30 physics steps (traced step-by-step: ends up
  ~7.7m away, tipped 181 degrees). (2) Isaac Sim's "frame all" on stage-open zooms the
  camera out to fit the *entire* stage bbox, which for a 1000m-scale building is what
  looked like "over the building far away." Tested three replacements head-to-head
  with the same 270-step translate/rotation trace: `Grid`, `Simple_Room`, and
  `Simple_Warehouse` are all physically stable. `Simple_Room` was rejected anyway:
  its big center table (`table_low_327`, 3.2x1.6m) sits at the origin with its top at
  Z~=0.01 and the environment's invisible ground collision plane is AT table height
  (visible wood floor is 78cm lower, Z~=-0.77) -- the robot unavoidably spawns
  standing on furniture and driving off the edge leaves it hovering mid-air on the
  invisible plane. `Simple_Warehouse/warehouse.usd` has a real floor at Z=0 and fits
  the "factory" request. A close-up default camera pose is now baked into
  `scene.usda` (`/OmniverseKit_Persp`, translate=(3.2,3.2,2.4)) so the GUI's initial
  view never depends on "frame all" behavior, regardless of environment scale.
- **`stage.Export()` FLATTENS the whole composition -- `scene.usda` must be written
  via root-layer export instead.** Export() bakes every referenced asset's geometry
  inline (verified: no reference arcs survive, only local `Flattened_Prototype_N`
  prims); with the warehouse environment that produced a 233MB `scene.usda`, which
  GitHub *rejected on push* (100MB hard limit). `build_scene.py` now authors the
  stage directly into the output layer (`Sdf.Layer.CreateNew` + `Usd.Stage.Open`) and
  saves that layer -- `scene.usda` stays ~18KB of reference arcs + overrides.
  Consequences to keep in mind: (a) the robot must be referenced by RELATIVE path
  (`./Aloha/Aloha.usda`) or the committed file breaks off this machine; (b) opening
  the scene needs network the first time (CDN assets; Kit caches them); (c) diag/CI
  scripts should open `scene.usda` by ABSOLUTE path -- in one verified case a
  relative `--scene` arg made the relative robot reference silently fail to resolve
  (environment loaded, robot missing, empty bbox) while absolute-path opens of the
  same file were fine.
- **A raw-authored root layer has NO physics scene -- Play simulates nothing without
  one.** The old `new_stage()+Export()` flow silently inherited Isaac Sim's new-stage
  template, which injects a physics scene prim; `Usd.Stage.Open` on a fresh layer
  starts truly empty. Symptom: healthy rigid-body blocks (rigidBodyEnabled=True,
  kinematic=False, collision+mass all composed -- checked attribute by attribute)
  hung frozen mid-air through 300 played steps, in GUI-equivalent
  `timeline.play()` runs. `verify_physics.py` PASSed the whole time and masked the
  bug, because `isaacsim.core`'s SimulationManager bootstraps its own physics context
  -- do not take "verify_physics passes" as proof that plain GUI Play works.
  `build_scene.py` now authors `/World/PhysicsScene` explicitly.
- **Pick-and-place setup: two official NVIDIA packing tables + four official colored
  blocks, not hand-authored slabs.** Per request, tables are real props with legs
  (`Props/PackingTable/packing_table.usd`: 2.474x0.782x1.083m, floor pivot, collision
  physics baked in, comes with totes/crates) and must NOT overlap the robot anywhere
  (an earlier slab-table attempt that overlapped the resting arms reproduced the
  same PhysX separation-impulse explosion as the Office spawn: robot ended ~13.5m
  away, tipped 252 degrees). Asset gotchas found while choosing (all BBoxCache-
  measured): the raw `SM_HeavyDutyPackingTable_C02_01_physics.usd` variant is
  authored in CENTIMETERS and composes 247m wide in a meters stage -- only the
  assembled `packing_table.usd` is meter-scale; `SeattleLabTable/table.usd`'s pivot
  sits 1.04m below its own geometry. Blocks are `Props/Blocks/{red,green,blue,
  yellow}_block.usd` (4.7cm, RigidBody+Collision+Mass baked in). Placement: tables at
  Y=+-0.85, long side facing the robot -- nearest edge |Y|=0.46 clears both the robot
  bbox (0.31) and its in-place rotation swing radius sqrt(0.21^2+0.31^2)=0.375 (any
  closer and turning the base in place clips a table corner). Blocks spawn ~3cm above
  the work surface and settle at Z=1.0173 within ~30 steps (measured; the table's
  bbox zmax=1.083 is the shelf frame, NOT the work surface, which is at ~0.994m).
  **Measurement gotcha**: PhysX writes simulated transforms to the prim carrying
  RigidBodyAPI -- in these block assets that's the Cube MESH CHILD
  (`/World/Block*/Cube`), not the wrapper Xform. Tracking the wrapper shows the block
  "frozen at spawn" forever while it is actually falling -- this false reading burned
  a full debugging round before being caught.
- **Real bug, not a mystery: the lift joint (`vertical_move`) was physically stuck
  near 0 regardless of commanded target**, confirmed by the user reporting "the up
  down limits are wrong." Root-caused properly rather than guessed at: ruled out
  insufficient drive force (still stuck at 5000N/50000 stiffness, 25x/10x the
  original), ruled out self-collision (still stuck with it disabled), ruled out
  external furniture collision (identical result in an empty Grid environment,
  nothing to do with the Office swap). The actual cause: `ARM_SOLVER_POSITION_
  ITERATIONS`/`_VELOCITY_ITERATIONS` (32/4, from NVIDIA's fixed-base single-arm
  config) were insufficient for this floating-base robot's lift joint specifically --
  it sits directly between the floating root (base_link, heavy: wheels + everything)
  and vertical_link (carrying both arms), which apparently needs far more solver
  iterations to converge than a revolute joint further down the chain. 64 position
  iterations still failed; 128 works reliably (verified: target 0.6 -> actual
  0.5990-0.5994 across multiple rebuilds). Raised to position=128/velocity=16.
- **Process safeguard added**: `build_scene.py` recreates `scene.usda` from scratch
  every time (fresh references), which silently wipes out everything
  `configure_physics.py`/`fix_wheel_collision.py`/`add_cameras.py` layered on top --
  this exact mistake happened once (an environment swap left every joint drive at
  stiffness=0/damping=0, no error, just non-functional joints, only caught because a
  reach test showed target=-1.9 converging to actual=0.30). Added
  `scripts/rebuild_all.sh` to always run all four steps in the correct order plus a
  final `verify_physics.py` check -- use this instead of calling `build_scene.py`
  alone.
- **GUI Stop used to spam "Physics Simulation View is not created yet" warnings
  forever.** Timeline STOP destroys PhysX's simulation view; the control loops in
  `control_terminal.py` command every frame, so each frame logged a carb warning
  (~one per 5ms, user-reported). And after Play, the `Articulation`'s physics handle
  is stale -- `is_physics_handle_valid()` documents that `initialize()` must be
  called again, otherwise commands silently no-op. Fixed with a per-frame
  `sim_command_ready()` gate: one suspend notice on stop, full re-init + cached
  target-array refresh on resume (stale pre-stop targets must not fire), controller
  events/sockets still serviced while suspended. `sim stop|play` REPL command mirrors
  the GUI buttons and makes this path testable headless (verified: zero warnings in a
  scripted stop/command/resume session; joint converged to exactly 0.8 rad after
  resume).
- **Robot cameras follow the OFFICIAL LeRobot camera set, not invented names.**
  `third_party/lerobot_alohamini` (submodule -- the official LeRobot integration for
  this robot, same repo the wheel/lift constants came from) defines cameras
  `forward` / `wrist_left` / `wrist_right` (plus `backward`/`chest`, not implemented),
  all OpenCV 640x480 @ 30fps -- see `config_alohamini.py`. `scripts/pipeline/add_cameras.py`
  (step 4/4 of `rebuild_all.sh`) authors matching USD cameras on the robot links;
  `scripts/cameras/capture_cameras.py` returns frames as `observation.images.<name>` HxWx3
  uint8 dicts (LeRobot convention), sampling every 2nd physics step = 30fps.
  Findings baked into the mount poses (all measured, see comments in
  `add_cameras.py`):
  - Wrist cameras mount on `link5` (the gripper body after Wrist_Roll) -- `link6` is
    the MOVING jaw finger, the wrong mount point (also an instance; link5 is not).
  - The manipulation front is **-Y** -- BOTH grippers work toward -Y (arms are
    mirrored in X, not Y; measured from link frames/bboxes) -- while the driving
    "forward" (vx) is +X. The forward camera faces -Y.
  - Rotation gotcha: with `rotateXYZ=(x, 0, 180)`, view = (0, -sin x, -cos x) and
    image-up = (0, -cos x, sin x). The first naive attempt rendered upside down AND
    the forward camera actually faced +Y (staring at the rear table) -- caught by
    rendering every camera and LOOKING, not by trusting the math.
  - The forward camera must sit in FRONT of the column's front face (world
    y<=-0.31) and high above it (z~=1.21): behind-the-column placements are occluded
    by the column's own top corner (ray-checked), and low placements sit nearly
    level with the 0.994m table surface so the near tabletop fills the frame.
  - Verified end-to-end: all three frames (480,640,3) uint8, and a motion test
    (lift 0.3 + both arms pitch) changes every view (mean abs pixel diff 50-92) --
    `docs/cam_*.png` and `docs/capture_*_moved.png`.

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
- **Terminal control** (`scripts/control/control_terminal.py`): arm joints, gripper, lift all
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
