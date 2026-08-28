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
docs/ticvla-architecture/     TIC-VLA block diagrams, EN + TR, read off the source
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

## Language-driven navigation (`nav/`)

TIC-VLA + the DynaNav benchmark drive the base from a typed sentence. Full detail in
[`nav/README.md`](nav/README.md) and [`nav/plan.md`](nav/plan.md); the model itself —
every layer, dimension and both loss functions, read off the source and checked against
the checkpoint — is drawn in
[`docs/ticvla-architecture/`](docs/ticvla-architecture/README.md) (EN + TR). The facts
that affect the *rest* of this repo:

- **Two processes, and the split is forced.** TIC-VLA needs Python 3.11 (Isaac Sim
  5.0's NumPy 1.x ABI), AlohaMini needs 3.12; separately, a DynaNav scene alone was
  measured holding 25.4 GiB of GPU0's 31.35 GiB, so the 1.9 GB checkpoint could not
  share that card even if the Pythons matched. Sim → GPU0, policy → GPU1.
- **GPU pinning differs by process ON PURPOSE.** The policy server sets
  `CUDA_VISIBLE_DEVICES=1`, which is safe *only* because it never starts Kit. Isaac Sim
  is pinned through Kit's own `active_gpu`/`physics_gpu` config instead — Kit warns that
  the env var "can lead to undesired behavior or crashes".
- **There is now a fourth camera, `camera_nav`.** The three LeRobot cameras all face the
  manipulation front (−Y) while the base drives +X, so none of them looks where the
  robot is going. `camera_nav` is deliberately kept OUT of `CAMERA_PRIM_PATHS`: that
  dict is the LeRobot observation contract, and adding a key would silently change the
  shape of every recorded manipulation dataset. Its rotation is outside the
  `(x, 0, 180)` family the cheat sheet in `add_cameras.py` covers — `(90, 0, -90)`
  gives view `(1, 0, 0)` and image-up `(0, 0, 1)`, with the derivation written out there.
- **`camera_nav`'s intrinsics and mount are NOT AlohaMini's — they are a copy of Nova
  Carter's front Hawk.** A camera is part of a VLA's input distribution, not a styling
  choice: TIC-VLA is trained through DynaNav's render of
  `nova_carter_sensors.usd`'s `/chassis_link/front_hawk/left/camera_left`. Probed off
  that asset: focal 2.8734 mm, aperture 5.760×3.600 → **HFOV 90.1°**, mounted at
  **z=0.346 on `chassis_link`**, **pitch 0.0° (level)**, rendered **1920×1080**. The
  first version used AlohaMini's own webcam intrinsics (78° HFOV) at 1.15 m on the lift
  column tilted 10° down, and the policy could not find the landmarks the benchmark
  prompts name — 90° cropped to 78° and pitched into the floor loses exactly the
  peripheral and distant context that "the second aisle from the right" is expressed in.
  It sits on `base_link`, not `vertical_link`: a nav camera on the lift has a horizon
  that moves for reasons the policy cannot account for.
- **`add_cameras.py` deletes any camera of ours found off its spec'd path.** Defining a
  prim at a new path does not remove the old one. When `camera_nav` moved from the lift
  column to `base_link`, a plain re-run left BOTH prims in the stage — the code read the
  new one while Kit went on rendering the stale one every frame. The invariant is now
  enforced in the script, so changing a mount point is a one-line spec edit rather than
  a spec edit plus a USD cleanup someone has to remember.
- **And a fifth, `camera_chase`** — the third-person view the UI can open. Parented to
  `base_link` (not `vertical_link`, which would ride the lift) so it inherits position
  and yaw and stays behind the robot through turns. It is created **lazily, on the first
  UI request**: once an Isaac Sim `Camera` object exists, Kit renders its render product
  every frame whether or not anything reads it, so a headless benchmark should not pay
  for it. Also kept out of `CAMERA_PRIM_PATHS`, for the same observation-contract reason
  as `camera_nav`.
- **Nav scenes are separate stage files** (`assets/usd/nav_<episode>.usda`), built by
  `nav/sim/build_nav_scene.sh`. They never touch `scene.usda`. Note they still need all
  three pipeline steps applied — a freshly authored layer has no joint drives, no wheel
  colliders and no cameras, because those are overrides in the *scene* file rather than
  in `Aloha.usda`.
- **Anything after `SimulationApp.close()` does not execute.** The first version of the
  scene builder authored its layer, reported success, and applied none of the pipeline
  — no error anywhere. This is why both `rebuild_all.sh` and `build_nav_scene.sh` chain
  separate process invocations instead of looping in Python.
- **The kinematic base does not collide** (already documented above, but it bites
  harder here): a teleported body has no contact response and will pass straight
  through shelving. `nav/sim/collision_guard.py` raycasts instead. Verified live rather
  than assumed — `nav/tools/check_collision_guard.py` sweeps the ray fan through a
  circle and hits 9/16 bearings on real warehouse geometry at 6.8–12.5 m.
- **FastAPI + `from __future__ import annotations` = request models must be module
  level.** A pydantic model defined *inside* the route factory cannot be resolved by
  `get_type_hints()` against the function's module globals. FastAPI does not raise — it
  demotes the body parameter to a query string, and every POST answers
  `{"loc": ["query", "req"], "msg": "Field required"}`. Cost one 3-minute Isaac Sim
  restart to find, in `nav/ui/ui_bridge.py`.
- **Episodes come from `benchmark_full.yaml`, NEVER `benchmark_example.yaml`.** The
  latter is a four-episode smoke/demo config, and three of its four spawn the robot
  facing ~180° away from their own goal (office −167.7°, outdoor −180.0°, warehouse
  −166.7°). Running one looks precisely like a broken policy — the robot drives off the
  wrong way and never sees the landmark the prompt names, because the landmark is behind
  it. This is not a yaw-convention mismatch: DynaNav applies `start_yaw` as a plain
  `rotate_z_op.Set()` with forward=+X, identical to `base_drive.reset_to()`, and across
  all 85 real episodes not one is improved by a 180° flip. `nav/config/episodes.yaml`
  now annotates every episode with its own off-bearing; keep new ones inside ~30° or the
  run is not testing navigation.
- **Lesson, and it cost two runs: don't narrate a bad number into a good story.** Run 2's
  193° opening turn got written up in `nav/plan.md` as the task working — "it read the
  instruction and went looking for aisles." It was the robot turning around because the
  goal was behind it. The same file *also* carried an open question saying the smoke
  episodes driving away from the goal was not normal behaviour. A plausible explanation
  fitted to a suspicious measurement is worse than no explanation, because it closes the
  question.
- **`previous_waypoints_text` is not optional — it is TIC-VLA's temporal channel.**
  Calling `predict()` without it (the default `""`) makes every plan a fresh straight
  line, because the model has no way to know it has already been driving. Measured on
  one warehouse episode, that alone was the difference between closing **−0.40 m** (32 m
  of path travelled, ending at the perimeter wall) and **+8.73 m** (15.8 m of path,
  ending at the Aisle 05 entrance). DynaNav raises `ValueError` if the string is empty,
  and training always emitted one of two forms — with no history yet it still said
  "No waypoints available.", so the empty string is not the zero case. Built by
  `nav/sim/waypoint_history.py`, whose format is copied character for character from
  `ticvla/data/vlm_data.py:_format_previous_waypoints_text`; a reworded version reads
  fine to a human and sits off the model's distribution.
- **TIC-VLA is a latency-aware architecture, and we were running it in latency-free
  mode.** The paper's whole mechanism is the slow/fast split: a ~1.5 s VLM generation
  producing a KV cache, and a millisecond action expert driving on whatever cache is on
  hand. The server was calling blocking `predict()` and passing `time_delay=0.0`,
  `dx=dy=0.0`. Those zeros were *honest* while inference was synchronous — a frozen
  robot really has moved nowhere — but they meant the compensation pathway never
  carried a signal. Switching `/predict` to `predict_async` (the VLM backgrounds itself
  inside the server process, so the sim needs no threading at all) measured **1669 ms →
  34.8 ms mean per call, 48×**; `hospital_down_hallway2` went **322 s → 121 s wall** with
  path length unchanged at 31.72 → 31.68 m and success preserved. The arithmetic closes
  exactly: 123 calls × 1.6 s ≈ 197 s of the old run's 322 s was inference, and removing
  it lands on the observed number, so the saving is the inference cost and nothing else.
  Confirmed on three episodes. The cleanest signal is **wall seconds per policy call**,
  which normalises out episode length: `office_hallway_turn` 2.23/2.31/2.28/2.50 → 1.15,
  `hospital_exit_room` 2.18 → 1.03, `hospital_down_hallway2` 2.62 → 1.02. Every prior
  synchronous run of those episodes sits near 2.2–2.5 s/call and every async run near
  1.0–1.15. Note this is not the 35 ms figure: a "call" also buys 30 physics steps of
  rendering, physics and guard raycasts, which is the ~1.0 s floor. Inference went from
  roughly half that budget to 3% of it.
- **Measure staleness in SIM seconds, and check it against the 3 s horizon.** With async
  inference `time_delay` is the age of the cache in use — measured against the
  *second-to-last* generation start, since the cache was produced by the generation
  before the one now running. Observed here: **1.70 s mean, 2.00 s max**, safely under
  the action head's 3.0 s plan horizon. That margin is not guaranteed by anything in our
  code: DynaNav's 3 s staleness cutoff lives in its *behaviour* script, not in the model,
  so nothing in this stack bounds the delay. It stays small only because the sim runs at
  roughly 0.5× realtime, so a 2.05 s wall-clock generation costs just 1.0 s of sim time.
  Speed the sim up and the delay grows toward the horizon. `summarize_runs.py` prints it
  as `stale` for exactly this reason.
- **What tests the body-frame rotation is a nonzero `yaw_ref`, not a turn.** The `dx, dy`
  handed to the action head is the world displacement rotated by `-yaw_ref`. The obvious
  sanity check — "`|dy|` stayed at 0.009 m, so it works" — proves nothing: small `|dy|`
  only says the robot drove along its own heading, which is true whichever way you
  rotate. The reasoning that a *turning* episode is needed is also wrong. The real test
  is `atan2(dy, dx)` against `world_bearing(ref→cur) - yaw_ref`, and it discriminates
  whenever `yaw_ref != 0`. Verified across three episodes whose `yaw_ref` together covers
  roughly the whole circle (−117°..−89°, −92°..+9°, +81°..+180°): angle error **0.01–0.02°
  mean, 0.13° max**, where the flipped sign would give 100–168°. Residual length error is
  0.0008–0.0014 m and is not error at all — `robot_state` is bfloat16 on the server, so
  ~3 significant digits is the floor.
- **A benchmark that runs faster than realtime cannot measure a wall-clock-bound
  mechanism.** A standalone harness hammering `/predict` in a tight loop showed only 2
  generations in 25 calls and `time_delay` climbing to 12 s, which looks exactly like a
  throttling bug. It was not: the loop finished ~2.5 s of wall time while claiming 12 s
  of sim time, so only two ~1.5 s generations could physically fit. The real episode
  showed 1.5–2.0 s. Do not conclude anything about async behaviour from a harness whose
  clock does not advance the way the simulator's does.
- **A collision guard must block DIRECTIONS, not speed.** The first `collision_guard.py`
  scaled the whole velocity to zero whenever any ray — including one aimed 35° off the
  direction of travel — came back short. That is strictly less physical than the contact
  it stands in for: a real wall stops you along its normal and lets you slide. It
  deadlocked for real at the Aisle 05 entrance, where the robot hugs a rack endcap: an
  outer ray sits in the racking, translation goes to zero, and since the plan says
  "straight ahead" the controller's `omega` is ~0 too — neither driving nor turning, with
  the centre path clear. 2734 of 4201 steps frozen. The fix is structural, not tuning:
  project out the component of velocity pointing *into* each blocker (per blocker, not
  against an averaged bearing, or a corner leaves you driving into one of them) and keep
  the rest, capped at half speed. Same fan numbers, same 0.6 m: interventions went
  2734 → 0–80 and the motion is smooth. Do not "fix" a guard like this by widening
  thresholds until one episode passes; that is how a benchmark stops measuring anything.
- **TIC-VLA is a video model, and one still image is off-distribution.** Its own system
  prompt (emitted verbatim by `ticvla/data/vlm_data.py:_build_messages`) says it is given
  "a video consisting of visual observations, including historical and current frames",
  and DynaNav passes **four** frames at 3 s spacing, oldest first
  (`nova_carter_test_ticvla.py:_get_sampled_image_paths`). We were passing one.
  `nav/sim/frame_history.py` now samples `[-9s, -6s, -3s, now]` with DynaNav's edge cases
  — skip an offset the history cannot reach, always keep the current frame, dedupe with
  order preserved so a young history collapses to one frame rather than four copies of
  it. This is the visual twin of `previous_waypoints_text`; both were defaulted away.
- **Every run writes its trace to `nav/results/*.json`** (`EpisodeResult.save()`, gitignored).
  Runs used to be compared on final distance alone, which cannot tell a robot that drove
  to the wrong place from one that drove nowhere — and those have opposite fixes. The
  trace carries `(x, y, yaw)` per sample precisely because position alone cannot tell a
  robot that *chose* the wrong way from one that never turned. A 70 s episode is ~140
  points.
- **Use the `pursuit` controller, not `holonomic`** (now the default everywhere). TIC-VLA
  expresses "the target is off to your right" as a small *lateral offset*, because on the
  differential-drive Nova Carter it was trained on, lateral offset can only be satisfied
  by **rotating**. `HolonomicController` satisfies the same offset by **translating**: it
  is discharged as sideways drift, the heading error is driven back to ~0 before the next
  replan, and **the camera barely rotates**. The next frame therefore looks nearly
  identical, the policy asks for the same small offset again, and a loop meant to converge
  crawls. Four full runs on the benchmark instruction, controller the only variable;
  yaw measured over the approach window (t=0–21 s) because what accumulates after the
  robot passes the aisle mouth is wandering, not steering: holonomic turned right by
  **4.9° / 2.1°** (one of those runs turning the *wrong* way, to 105.9°), pursuit by
  **11.8° / 12.5°** — ~2.7× as far and consistently. On identical plans pursuit yields
  ~1.7× the yaw rate. Be clear about the size of the win, though: closest approach only
  improved from a mean of 6.52 m to 6.08 m. The deficit is 25°; the controller is worth
  about 9 of them. `controllers.py` had *predicted*
  this in its docstring — "full holonomy is dangerous for a vision-language policy" — and
  the `yaw_align=0.8` mitigation was assumed sufficient without being measured. It was not.
- **Instrument intent, not just outcome.** `EpisodeResult.plans` logs, per policy call,
  `(sim_time, plan_heading_deg, reach_m, bearing_to_goal_deg)`. The trace says where the
  robot *went*; it cannot distinguish a robot obeying a straight plan from one ignoring a
  turning plan, and those have opposite fixes. Both controllers steer at the same
  lookahead point, so plan heading is the controller-independent statement of policy
  intent. `bearing_to_goal` is scoring only — never fed to the policy.
- **Open: the policy does not ask to turn on the warehouse episode.** This is now measured
  rather than inferred. Across **129 policy calls: 2 asked for a right turn greater than
  5°**, mean asked **+1.1°** (very slightly *left*), while the bearing needed ran from
  −24.9° at the start to −80° by t=21 s and −170° by the end. Things ruled out, each by
  direct probe rather than argument:
  - *Not the controller* — pursuit recovers the turn but the deficit is 25°, not 1.7×.
  - *Not the lookahead rule.* The 30 waypoints are 10 Hz, i.e. exactly 3.0 s of motion, so
    a time-parameterized lookahead is defensible. It changes nothing: on a real start-frame
    plan, every waypoint from t=0.0 s to t=3.0 s lies within ±4° of straight (arc-length
    1.0 m → +1.4°; time-based 0.5/1.0/3.0 s → −2.3°/−0.6°/+3.8°). No sampling rule can
    extract a turn that is not in the plan.
  - *Not the conditioning.* Sweeping `previous_waypoints_text` (none / 3 s / 20 s straight
    / drifting left / drifting right) and `robot_state` vx (0.6 vs 1.5) moves the answer
    only between −6.5° and +3.4°. Implied plan speed stays ~0.55 m/s throughout.
  - *Not wiring* — directional language still moves it: "turn right" → −9.1°, "turn left"
    → +14.3°, "Stop. Do not move." → reach collapses to 0.21 m.

  There is no theta channel in the **action head**: `ticvla/models/ticvla.py` sets
  `action_dim=2, # Offset (dx, dy)`. Heading must be inferred from dx,dy by the
  controller — which is why the controller choice matters at all.
- **The `<answer>` triples are `(x, y, theta)` at 3 s / 6 s / 9 s — NOT `(x, y, z)`.**
  This corrects an earlier claim here. `ticvla/data/vlm_data.py:508-528` builds them as
  `for idx in [29, 59, 89]: theta = math.atan2(y, x + 1e-3)`, so the text head reaches
  **three times** the action head's 3.0 s horizon. `nav/sim/guidance.py` decodes it,
  including the `-100` sentinel (`vlm_data.py:519`) that the model legitimately emits
  when no 9 s future exists — treating that as a coordinate commands a hard left turn.
  Note theta is *redundant*: it is literally `atan2(y, x)` of the same pair, so nothing
  is gained by "sending theta". Only the horizon differs.
- **Steering on the 9 s guidance was tried and it does NOT work.** Recorded because the
  idea is attractive and cheap to re-derive. A single probe of the Aisle-05 start frame
  looked decisive — action head −6.5°, guidance −18.9°, needed −24.9° — but over a full
  141-call run the two channels have the same centre and the guidance has 3–5× the
  spread:

  | channel | mean | median | min | max | asked >5° right |
  |---|---|---|---|---|---|
  | action head (3 s) | +0.9° | — | — | — | 7 / 141 |
  | guidance (6 s) | +2.3° | +0.2° | −90.0° | +165.1° | 18 / 132 |

  Guidance was present in 94% of calls, so this is not a sentinel artifact — the signal
  is there and it is not informative. Steering on it made the episode *worse* (closest
  7.51 m vs pursuit's 5.77/6.39, over a 60.2 m path vs 44.1). Kept as the `guided`
  controller, not default, documented as a negative result.
- **Do obey the plan's SPEED, though — that half held up.** The action head is always
  exactly 30 waypoints at 10 Hz, so a plan's arc length over its fixed 3.0 s *is* a
  requested speed. Ignoring it is what let `warehouse_aisle6` close **92%** of a 34.3 m
  gap, reach **2.80 m** — 1.3 m short of the 1.5 m threshold — and then sail past to
  22.0 m. The policy was braking hard and we drove through it:

  | dist to goal | plan reach | implied speed | we drove |
  |---|---|---|---|
  | 4.14 m | 1.03 m | 0.34 m/s | 0.6 m/s |
  | 2.80 m *(closest)* | 0.65 m | **0.22 m/s** | 0.6 m/s |
  | 4.32 m | 0.35 m | 0.12 m/s | 0.6 m/s |

  DynaNav has the same blind spot and it costs them nothing, which is the trap: their
  episode *terminates* the instant the robot is within 1.5 m, so overshoot is unscored.
  **Parity with DynaNav's controller is not parity with DynaNav's scoring harness.**
  `braking` (= `pursuit` + `obey_plan_speed`) is now the default; `pursuit` keeps it off
  so it stays a clean parity baseline.
- **RESOLVED: measured speed exceeded the commanded limit because position was re-read
  from the sim every step.** `sync_from_sim()`'s docstring already said the integrated
  pose is the authority afterwards, because re-reading "would fight the teleport" — that
  was enforced for yaw and quietly not for x/y. The wheels carry no load but traction is
  not zero, so each step dragged the robot on top of the `vx*dt` the teleport applied,
  and reading that back as the next step's start made it compound: **0.663 m/s measured
  against a 0.600 m/s cap, 1.11×**. Over the 32.4 m `hospital_down_hallway` episode that
  is ~3.5 m of extra travel, against episodes that were plateauing 2.8–3.6 m short.
  `base_drive.py` now integrates x/y like yaw, resyncing only past 0.25 m — above
  per-step drag (~1 cm), below anything that matters — so a fall or an external reset
  still wins. A tighter tolerance lets the drag straight back in.
- **The speed cap is part of the policy's input distribution, not a safety knob.** It was
  0.6 m/s against DynaNav's Nova Carter 1.5. The policy reads its own past motion through
  `previous_waypoints_text` and plans at the speed it was trained at (mean **0.731 m/s
  asked** on office_nearest_elevator), so the cap clipped nearly every plan — and with it
  the *braking*, which is the entire point of the `braking` controller: `min(cap, plan)`
  engaged in **11 of 141 calls**. Now 1.5. Prompting cannot substitute; speed is not a
  channel the policy controls.
- **Nav stages are keyed on the ENVIRONMENT, not the episode.** The start pose used to be
  baked in at author time, so 13 episodes meant 13 four-process build chains for stages
  differing by one prim's position, six of them the same `hospital.usd`. The runner
  teleports to `episode.start` before every run, so the baked pose was never load-bearing
  — `episode.env_name()` is the key and the ladder builds 3 stages. The payoff is in the
  UI: every episode in the loaded environment is selectable and runnable in one session.
  Episodes elsewhere are shown but locked, because Isaac Sim cannot swap stages
  in-process at this version and a half-swapped stage fails as a *navigation* error, not
  a crash — a benchmark's worst failure mode, since it yields a plausible number.
- **The ladder once scored 13 episodes without driving a single metre, and the number it
  printed was 6/13.** Three ordinary decisions lined up into a silent lie, so all three are
  worth naming. (1) `run_navigation.py` read `info['num_action_chunks']` out of `/health`
  to build a *log message*; those are `server.py`'s keys, and `server_qwen.py` serves the
  same three ROUTES with a different payload, so pointing the runner at it raised KeyError
  inside `setup()`. (2) An uncaught exception exits Python with status 1, and `bench.sh`
  deliberately reads 1 as "the episode failed" rather than "the process died" — a good
  rule that here made a crash indistinguishable from a result. (3) `summarize_runs.py
  --latest` answers with the newest matching file and cannot know whether that file came
  from the run you just did; the newest were up to two weeks old. The tell was the policy
  server's own counter: `predictions` sat at 8 across a "completed" 13-episode ladder.
  Fixed on two independent axes — `ready_message()` never subscripts a payload it does not
  own, and `bench.sh` counts `nav/results/*_${EP}_${CONTROLLER}.json` before and after each
  run and calls it an ERROR when the count did not move. **Two servers matching on routes
  is not two servers matching on schema**, and after any benchmark run, read a counter that
  lives on the far side of the thing being measured.
- **6/13 on the benchmark ladder (Phase 18), from 0/13.** Two episodes beat DynaNav's own
  spl. The remaining failures are not one thing: some arrive and brake correctly ~2 m off
  the scored goal (`hospital_down_hallway` — plan reach collapses to 0.05 m at 2.11 m
  out), others cruise past at full plan speed (`hospital_vending_machine2`, whose variant
  1 succeeds at spl 0.98 — but see the next entry: "never sees the landmark" was the wrong
  reading of that one). And **`final` ≫
  `closest` in five of seven failures** — the robot arrives and then leaves. DynaNav never
  had to solve that: their harness terminates at 1.5 m, so their controller is never asked
  to stop. Parity with their controller is not parity with their scoring harness.
- **The vending-machine failure is NOT a prompting problem, and the attractive story
  about it is wrong.** Worth writing down because the wrong answer is very compelling.
  The two episodes share a scene, a start pose and a goal and differ only in the
  sentence — `hospital_vending_machine` "…at the vending machine at front." passes 3/4
  here, `hospital_vending_machine2` "…at the front of the vending machine at front left."
  passes 0/3. Every failing run shows the same signature: at ~4.5 m out, with the goal
  within a few degrees of dead ahead, the policy asks for another ~+9° **left** and keeps
  asking left while the goal slides off to the right, so the robot passes the machine
  1.5–2 m to its left. The miss is **lateral, not longitudinal** — it never stops short,
  it goes around. "front left" contradicting a machine that is by then dead ahead is an
  irresistible explanation. It is false. Held that exact frame and history fixed and
  varied only the sentence (`nav/tools/probe_stop_decision.py`):

  | instruction | asked heading |
  |---|---|
  | v2, "…at front left" (fails 0/3) | **+13.45°** |
  | v2 with left→**right** | **+12.44°** |
  | v1, "…at front" (passes 3/4) | **+20.56°** |
  | "Go straight ahead." (no object named) | **+15.61°** |
  | "Turn right." / "Turn left." | −14.11° / +21.51° |

  Swapping left for right moves the answer 1.0°, the *passing* sentence asks for **more**
  left than the failing one, and a sentence that never mentions the machine asks for the
  same left turn. The model is listening — 35.6° of authority between "turn left" and
  "turn right" — but the left bias is the **scene**, not the words: the machine stands
  against a wall, so a traversable forward plan bends around it. It first appears at
  ~5 m and grows as the machine fills more of the frame, which is what avoidance looks
  like. Also note "never sees the landmark" was wrong: `nav_000167.jpg` has the machine
  large and centred, 0.7° off the nose.
- **What actually separates success from failure there is whether the plan BRAKES, and
  it is inconsistent run to run.** Plan reach over the final approach, five runs:

  | run | reach on final approach | closest |
  |---|---|---|
  | v1 150443 | 0.551 → 0.574 | **1.50 ✓** |
  | v1 145416 | 0.715 → 0.625 → 0.479 → **0.424** | **1.50 ✓** |
  | v1 145727 | 0.93–1.08, flat | 2.12 ✗ |
  | v2 144637 | 1.04–1.14, flat | 1.85 ✗ |
  | v2 150620 | 0.99–1.08, flat | 2.13 ✗ |

  Perfect separation, and it **cuts across** the sentence split — v1's own failure is a
  no-brake run under the *passing* sentence. So the policy does decide when to stop (that
  authority was never missing) and the `braking` controller does honour it; what is
  unreliable is *recognising arrival*. The two are coupled: the runs that brake are the
  ones tracking y≈2.5–2.8 at x≈7.9, actually in front of the machine, while the failures
  are 0.9–1.7 m wider in y and never get in front of it. The wider arc comes from holding
  the left turn ~1 s longer after the spike, ≈0.5 m of lateral offset that then compounds.
- **`predict()` is deterministic but a RUN is not, and async is why.** Repeats on an
  identical frame + history + robot_state returned headings identical to **0.00° sd**
  across every arm above. Yet the same episode with the same sentence succeeds twice and
  fails once. The divergence cannot come from sampling: it comes from which KV cache
  happens to be ready at which step, which depends on wall-clock generation timing. Do
  not read a single episode outcome as a property of the prompt or the policy — n=1 on
  this stack is noise, and `stale` alternating 1.5/2.0 s is visible in every trace above.
- **"Slow down on approach" does not fix a lateral miss, and in this controller it
  tightens the turn.** `PursuitController` splits steering into `w_ff = 0.5 * v_cmd *
  kappa`, which is speed-invariant in *path shape* (dyaw/ds = kappa/2), and
  `w_fb = k_angular * yaw_err_filt`, which is not scaled by `v` at all. Halving speed
  therefore roughly doubles the yaw per metre travelled — the robot commits to whatever
  heading the plan asked for *sooner* in arc length, rather than getting "more time to
  react". Slowing is the right fix for overshooting *past* a target; it is not a fix for
  arriving 1.5 m to the side of one.

- **Swapping the VLM (Q-VLA): what a candidate has to pass, and it is not just shapes.**
  Design in [`nav/qvla_design.md`](nav/qvla_design.md), feasibility in
  [`nav/qwen_swap_plan.md`](nav/qwen_swap_plan.md), checker in
  `nav/tools/check_vlm_swap.py`. The gates that are easy to miss:
  - **TIC-VLA consumes a KV cache TENSOR, not text.** So a GGUF build — including the
    Ridge quantisation, which is the lightest option at 11.7 GB and does support vision
    through an `mmproj` — **cannot be used at all**: llama.cpp exposes no HF-style
    per-layer `past_key_values` to Python. Quantisation choice is constrained by what
    `transformers` can load, not by what runs fastest.
  - **`past_key_values[-1]` assumes every layer has a KV pair** (`ticvla.py:108`, with a
    `dim() != 4` assert on the next line). Qwen3.8-27B is a hybrid — 48 `linear_attention`
    + 16 `full_attention` — where most layers carry a recurrent state instead. It passes
    only because `full_attention_interval: 4` puts a full-attention layer at index 63 of
    64. Select the last full-attention layer from `layer_types` rather than relying on
    that. `check_vlm_swap.py` now reports this as its own section.
  - **A candidate worth swapping to is usually too new for the installed transformers.**
    4.57.6 cannot even read `qwen3_5` — but the shapes that decide the swap are plain
    JSON, so the checker falls back to fetching `config.json` off the hub.
  - **The weights have to fit GPU1 *while Isaac Sim holds GPU0*, and this line was wrong
    for weeks because of a unit error.** Measured off the safetensors headers without
    loading: FP8 is **28.75 GiB** (23.00 F8_E4M3 + 5.74 BF16) and fits a 31.36 GiB card
    with **2.61 GiB** spare. The note that used to stand here — "FP8 at 30.89 GB leaves
    0.5 GB on a 31.35 GiB card and cannot be the closed-loop build" — compared a **GB**
    figure against a **GiB** one. FP8 *is* the closed-loop build, single-card. The KV
    cache fits the headroom comfortably: only 16 of 64 layers are `full_attention`, so it
    is a few hundred MB.
  - **NVFP4 is smaller on disk and larger in memory, and is unusable on this stack.**
    21.81 GiB packed (6.97 GiB of nvfp4 over 14.97 B params, plus 14.84 GiB of F8_E4M3 and
    BF16 that were never 4-bit) — but compressed-tensors registers a forward pre-hook that
    decompresses on the FIRST forward, not at load. Hence a suspiciously fast "ready in
    6.5 s" followed by a death mid-generation. Decompressed to bf16 it is **42.73 GiB**,
    11 GiB over the card even with the card empty, and transformers 5.16.1 hardcodes
    `run_compressed=False` in `quantizer_compressed_tensors.py`, so there is no packed-
    inference path to fall back to. **On-disk size is not the inference footprint for
    4-bit.** The symptom to recognise is a `KeyError: 'weight_packed'` that is really a
    `torch.OutOfMemoryError` against a half-decompressed model — visible only after
    printing the traceback, since the one-line error named neither.
- **On a 27B, DECODE is the whole cost — prefill and resolution are not the lever, and
  the 200-token cap is not the token count.** Both halves of this were predicted wrong in
  `qvla_design.md` before being measured, so the numbers are worth keeping.
  `probe_qvla_latency.py` on Qwen3.8-27B-FP8, four real nav frames:

  | resolution | prompt tokens | prefill | decode | TOTAL |
  |---|---|---|---|---|
  | 448×448 (196 tok/frame, InternVL parity) | 796 | 0.47 s | 8.17 s | 8.64 s |
  | 1920×1080 native (2059 tok/frame) | 8236 | 4.57 s | 8.21 s | 12.78 s |

  Decode is **flat across a 10× change in prompt size** and is 95% of the call at the low
  resolution. Prefill scales exactly linearly (~0.55 ms/token), so cutting resolution is a
  real 9.7× on prefill and nearly nothing on the total — driving prefill to *zero* still
  leaves 8.2 s. And `ticvla.py:579`'s `max_new_tokens=200` is a cap, not a cost: sampled
  off the **live** policy server on real frames, generations are 131–135 tokens, mean
  **134**, tight because the output shape is fixed (`<think>` paragraph + one `<answer>`
  line of three triples). Budgeting 200 overstates a 27B call by 1.5×.
- **A multi-GPU split silently changes which FP8 kernels run, so a two-card timing is not
  a one-card timing.** transformers logs it plainly: spanning devices routes FP8 linear
  layers off DeepGEMM onto Triton/`grouped_mm`, because DeepGEMM's cached kernels are
  bound to a single CUDA context. **The provisional 16 tok/s was indeed the split talking.**
  It was quoted here on the belief that FP8 *had* to span both cards; that belief was the
  GB/GiB error above. Loaded onto one card (`CUDA_VISIBLE_DEVICES=1`, `QVLA_MAX_MEMORY=0:30`),
  DeepGEMM comes back and the same 8-call probe runs at **~2.7 s/call steady-state**
  (4.5 s mean including the first), 8 calls in 36 s wall. `qwen_load.py` drops zero entries
  from `max_memory` rather than capping them, because a device left in the map at 0 can
  still be spilled onto, and one spilled layer costs the whole DeepGEMM path.
- **`modules_to_not_convert` is PREFIX-matched, so an entry naming a module that does not
  exist can still disable one that does.** This produced a model that loaded, ran at full
  speed, raised nothing, and emitted pure gibberish. Qwen3.8-27B-FP8's skip list has 882
  entries including `layers.N.mlp.gate` and `...mlp.shared_expert_gate` for every layer —
  MoE routers, and this checkpoint is dense, so they look like harmless leftovers. They
  are not: `mlp.gate` is a **prefix of `mlp.gate_proj`**, a real linear in all 64 blocks.
  So `gate_proj` was skipped, stayed a plain `nn.Linear`, took its e4m3 weights raw, and
  its `weight_scale_inv` was dropped — the gate half of every SwiGLU off by a per-block
  scale. `"What is the capital of France?"` answered
  `'althocie不成这儿ardy礼拜 forth喜怒哀perationarked…'`; after dropping the 130 spurious
  entries, `'The capital of France is Paris.'` The **only** signal was a load report
  listing those `weight_scale_inv` keys as UNEXPECTED next to a note saying UNEXPECTED
  "can be ignored". **Read the load report, and treat an unexpected `*_scale*` key as an
  error.** Fixed in `nav/policy_server/qwen_load.py`, which everything in this repo loads
  Qwen through — at runtime, never in site-packages. Note it does not affect timing (same
  shapes, same FLOPs), so latency measured before the fix stands and quality did not.
- **Q-VLA-direct (VLM writes the waypoints, no action expert): the plan is well-formed,
  the SPEED is a single scalar per plan, and the STEERING is one-sided.** Measured
  against TIC-VLA on 8 moments of `office_hallway_turn`, identical frames/history/state,
  every call its own generation (`compare_qvla_ticvla.py`):

  | | asked heading @1 s | implied speed | vs bearing to goal |
  |---|---|---|---|
  | TIC-VLA (action expert) | −46.8° … +5.9°, tracks the turn | 0.62 m/s (0.18–0.86) | 20.9° mean |
  | Q-VLA-direct | **+0.00° in 8 of 8** | 0.56 m/s (0.01–0.70) | 29.1° mean |

  Format and parsing are not the problem: 36 generations, **0 parse failures, 0 errors**.
  Speed *varies with the image* — 7 distinct profiles across 8 frames — but this line used
  to say it was "genuinely read off the image", and that overstates what the numbers do.
  Every `pairs` plan is one scalar × `[1..6]`: perfectly linear, so there is **no
  within-plan braking at all**. The "braked to 0.01 m/s" event is `0.02` written six
  times — a stop, not a brake. The model picks a speed for the next 3 s and holds it.
  (`arc` is worse in a different way: non-linear profiles, but only 3 distinct ones across
  8 frames, byte-identical repeats.) What is flat outright is the lateral channel, and the
  failure is specific: told "Turn right." the model writes a textbook arc —
  `(0.32, -0.01), (0.64, -0.03), (0.96, -0.06), (1.28, -0.10), (1.60, -0.15), (1.92, -0.21)`,
  a proper parabola — and told "Turn left." it writes literal `0.00` six times. **It never
  emits a positive y.** Ruled out, each by direct probe:
  - *Not comprehension.* "y must be POSITIVE and grow along the list" → `0.00`. "Answer
    with the exact mirror image of a right turn, y of the opposite sign" → `0.00`.
  - *Not obstacle avoidance.* Dead in **5 of 5** different scenes across the episode
    (`probe_steering_authority.py` exists to make that distinction, because one frame
    cannot: a side dead *everywhere* is an encoding bug, a side dead in *some* scenes is
    a wall and is correct behaviour).
  - *Not the sign convention.* Flipping the prompt to y-positive-is-RIGHT (`QVLA_FLIP_Y=1`)
    does not move the dead side — it kills the live one. Both directions go to `0.00`.
    So the model holds its own FLU convention and will only steer when the prompt agrees
    with it; contradicted, it stops steering rather than following the wording.

  The conclusion for the architecture: **the action expert is doing real work that this
  27B cannot do in text as prompted**, and the gap is not prose quality. Any fix has to
  stop asking the model for a signed lateral offset inside a tuple — a direction word plus
  an unsigned magnitude, with the arc built our side. Note what this does *not* say: the
  slow/fast split is still worth having on speed alone, and Q-VLA's braking was as good as
  the action expert's.

  **Amended by the perception probe below.** Two claims here are wrong and one is right for
  the wrong reason. "Speed varies with the image" is not established: those 8 moments each
  carried their OWN history text, so image and history varied together; holding history
  fixed across 5 unrelated scenes gives ONE plan. "Not comprehension" is backwards — the
  model comprehends completely, and the probes that concluded otherwise could not see it
  because thinking was off. What survives is the headline, and now with a mechanism.
- **`QVLA_FORMAT=arc` collapses to a permanent stop in CLOSED loop, which eight open-loop
  frames could never have shown.** First closed-loop arc run, `hospital_vending_machine`:
  109 plans, **102 of them speed `0.000`**, last non-zero at t=9.0 s, 1.07 m of path in 54 s,
  timed out 12.5 m short. Not a stack failure — 45 generations, **0 parse failures, 0 gen
  errors**; the model generates cleanly and generates zeros: `STRAIGHT 0 | 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0`. Two consequences worth separating from the cause. First, an all-zero
  distance list makes the logged heading **meaningless**, not merely wrong: the waypoints
  collapse onto the origin and `atan2` returns whatever noise survives — hence `head=180.00`
  with `reach=0.550` in the trace. Don't read steering columns from a stopped plan.

  **Second: the prompt hypothesis written here first was WRONG, and it is left in place
  because of how convincing it was.** It said `arc` had a zero-attractor `pairs` lacked —
  both carry the same "if you have arrived, STOP" bullet, but `pairs` illustrates it with
  `(0.05, 0.00)` repeated (which still advances) while `arc` says "write STRAIGHT 0", so
  `STRAIGHT 0 | 0.0 ×6` is both the cheapest legal string and one the prompt blesses. It
  fit every observation available at the time. It was falsified by the very next run:
  `pairs` collapses identically, `(0.00, 0.00) ×6`, 136 of 141 plans at zero. The cause was
  never the prompt — see the horizon entry below. **A mechanism that explains the evidence
  is not thereby the mechanism**, and the tell was available: the same-frame probe that
  would have distinguished them costs one minute and was skipped in favour of a story.
- **The plan horizon has to outlast a generation, and copying the action expert's 3.0 s
  was the whole bug.** This is the real cause of both zero-collapses above, and the model
  was never involved. Feeding ONE frame seven different `time_delay` values returns the
  SAME cached plan, reported as:

  | `time_delay` | waypoints left | `plan_speed` |
  |---|---|---|
  | 0.0 | 30 | 1.353 |
  | 1.0 | 20 | 0.887 |
  | 2.0 | 10 | 0.420 |
  | **2.9** | **1** | **0.000** |

  The chain: generation takes 2.7–5 s against a 3.0 s horizon → the server slices the
  cached plan by staleness (`full[ceil(time_delay/DT):]`) → at 2.9 s one point survives →
  `plan_speed` returns 0.0 below two points → `braking` does
  `v_cap = min(v_cap, plan_speed(...))` → **full stop**. And staleness is *generation
  latency*, so standing still never clears it. 129 of 141 calls ran at staleness ≥ 3.0 s.
  **Why 3.0 s was the wrong number to inherit:** TIC-VLA's action expert re-runs on the
  current frame every control tick, so its waypoints are never stale — only the KV cache
  is, and 3.0 s is a *cache-freshness* budget. Remove the expert and the waypoints
  themselves become the thing that ages. The horizon is set by inference latency, not by
  parity with a head that no longer exists. Now `HORIZON_S = 10.0`, `N_WAYPOINTS = 100`,
  10 control points; `DT` stays 0.1 s so the control rate is the same 10 Hz. Verified:
  ~0.69 m/s held from staleness 0.0 through 9.0 s. **This is also the honest answer to
  "do we still need the action expert?" — yes, and not for steering quality. Its job is
  keeping a valid trajectory available at control rate, which a 27 B writing waypoints at
  ~0.3 Hz cannot do for a horizon that real time consumes at 1 s/s.**
- **Three separate constants were second copies of the horizon, and every one failed
  silently.** Worth naming as a class, because all three produced a plausible wrong number
  rather than an error, and two of them mimicked the bug being fixed. (1) `plan_speed`
  divided by a fixed `ACTION_HORIZON_S`, understating commanded speed in proportion to
  staleness — and via `obey_plan_speed` that is a brake that tightens the longer the model
  thinks. Now divides by the plan's own duration; identical for any untruncated plan, so
  recorded TIC-VLA numbers stand. (2) The parser's flat `6.0 m` plausibility cap. After the
  horizon change the model wrote a *correct* `(0.70, 0.00) … (7.00, 0.00)` and the parser
  discarded it — presenting as `has_plan: false`, all zeros, robot stopped, i.e. **exactly
  the symptom of the bug just fixed**. Only `parse_failures: 1` distinguished them. Now
  `MAX_REACH_M = 2.0 * HORIZON_S`, reproducing `6.0` exactly at 3 s. (3) The flat 110-token
  cap, sized for six pairs, which would have truncated ten mid-list. Plus the literal
  "3 seconds" in both prompts and the 3.0 s budget in `check_vlm_swap.py` /
  `probe_qvla_latency.py`, which would now fail a working configuration. **After changing a
  constant, grep for its value, not its name** — a copy has the number, not the identifier.
- **`/predict` re-serves a CACHED plan, so a probe loop over it measures one generation N
  times and prints N identical lines.** It only blocks on a generation when it has no plan;
  otherwise it returns the cached one instantly and regenerates in the background. A tight
  loop therefore produces a perfect-looking N-of-N result out of a single sample. This
  invalidated three conclusions stated as measurements: that history makes no difference
  (twice), that frame count makes no difference, and the "nine times out of nine" anchor
  reading behind commit `4ac6655` — that was 1–2 real generations re-served. **Every probe
  must POST `/reset` before every call**, which sets `plan=None` and forces a real
  generation. Redone that way the history answer inverted outright. Same class as the
  second-copy constants above: no error, no warning, a plausible number.
- **Q-VLA-direct with thinking OFF is not navigating — it copies a number out of the
  prompt and draws a straight line.** The single most important measurement on this branch,
  and it took a `/reset`-per-call probe to see. Five visually unrelated 4-frame windows from
  across an 881-frame capture, identical instruction and identical history, produced **one
  plan**: `(0.70, 0.00) … (7.00, 0.00)`, with `max|y| = 0.000` in every scene. Two of those
  scenes have a wall or a pillar a few metres ahead. Vary instead the *number in the prompt*,
  holding the scene fixed, and the plan tracks it: stated cap 0.8 → `0.70` m/s, 1.5 → `1.50`,
  2.5 → `0.05` — and `0.70`, `1.50`, `0.05` and `0.00` all appear verbatim in the prompt
  (`0.05` is the "for example (0.05, 0.00) repeated" bullet). This also explains the earlier
  0.7 m/s readings, which looked like agreement with reality and were agreement with the
  prompt's own "about 0.7 m/s" — removing that sentence moved the output to the new number.
  **A plan that matches a number you put in the prompt is not evidence the model agrees
  with you.**
- **The cause is that the model gets ZERO tokens to think, and turning thinking on shows it
  perceives and reasons correctly.** `QVLA_THINK` defaults to 0, and the no-think path
  prefills `<answer>(` — so the first token the model is free to choose is already inside a
  coordinate. With `QVLA_THINK=1` the same FP8 checkpoint describes the scenes accurately
  ("a large open room with a hexagonal tile floor, a black console table on the right, and
  windows with vertical blinds" — correct), does the trigonometry off the stated 90° FOV,
  and reaches the right decision. **Vision is not broken and the FP8 quantization is fine;
  neither was ever tested before, and both were quietly assumed.** Three regimes, each
  failing differently:

  | | parse OK | cost/plan | plan |
  |---|---|---|---|
  | think off | 5/5 | ~2.7 s | one straight line for all scenes, speed copied from prompt |
  | think on, unforced | **1/6** | 8–54 s | correct reasoning, never emits `</think>` or an answer |
  | think on, budget-forced | **5/5** | ~22 s | writes an answer, still the same straight line |

  Budget forcing is now in `_generate`: cap the thinking, then close the block ourselves
  with `</think>\n<answer>(` and decode only the answer. It fixes the structural failure
  completely (parse failures 5→0) and changes nothing about quality, which is the finding.
  Because the decisive artifact is this — the model's own reasoning, sitting in its context,
  saying **"To stop at it, I need to head toward it, i.e., move forward and left"** — and
  the very next tokens it writes are `y = 0.00`, ten times. **The lateral channel is dead at
  emission, not at perception and not at reasoning.** Greedy decoding into a heavily
  templated answer collapses onto a clean arithmetic sequence, and the reasoning does not
  reach the coordinates.

  So the answer to "do we still need the action expert?" is **yes**, now for a reason worth
  having: not that the 27B cannot see or cannot think about navigation — it demonstrably
  does both — but that it cannot convert a decision into coordinates, and cannot do it
  inside a control period even when it does. 22 s/plan against a 10 s horizon is 2.2× over
  budget, and raising the horizon to cover it means planning further ahead than the scene
  stays valid. Untested and the obvious next step: `do_sample=False` is the suspect for the
  mode collapse, and the code comment asserting "sampling buys nothing when the output is a
  list of coordinates" was never measured. TIC-VLA itself samples at 0.7.
- **The TIC-VLA dataset zips are `stored`, not deflated — 1.00× — and `_json` has no
  images.** Both measured off the completed `DynaNav_json.zip` by reading its central
  directory rather than extracting: 288,602 entries, 25.83 GB in and 25.83 GB out,
  151,183 `.txt` + 137,419 `.json`, zero images. Consequences: the frames are in the
  `_data` archives so all 538.29 GB is needed, and unzipping costs the archive size
  *again* — 538 GB of zips plus 538 GB extracted is ~1076 GB against ~936 GB free.
  **Extract each zip then delete it.** Reading the central directory is the cheap way to
  learn this; a ratio guessed from "it's JSON, it'll compress" would have been badly wrong
  in the safe direction and "it's JPEG, it won't" wrong in the dangerous one.

## Next step

Nothing blocking. Optional future work: root-cause the wheel traction issue properly
(would remove the kinematic-drive limitation, and probably explains the nav speed
discrepancy above), finish the remaining nav episodes × controllers
(`nav/plan.md` Phase 13), or Phase 6 (ROS2) if requested.
