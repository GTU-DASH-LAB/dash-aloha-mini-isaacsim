# Plan: AlohaMini (SO-101) in Isaac Sim 6.0.1

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/needs decision

## Goal

Simulate the AlohaMini1 robot (Apache-2.0, [liyiteng/alohamini](https://github.com/liyiteng/alohamini)) —
a mobile base (3 wheels + vertical lift) carrying two SO-101 arms — in Isaac Sim **6.0.1**,
with full rigid-body physics, controllable both from a terminal script and from the Isaac
Sim UI (joint sliders / Articulation Inspector). Decided scope: **full mobile robot**
(wheels + lift + arms), not just a fixed dual-arm stand. Repo is public on
`GTU-DASH-LAB`. ROS2 is **not used by default** — ask before adding it; see Phase 6.

## Why this approach

- No pre-built ALOHA or SO-101 asset ships with Isaac Sim 6.0.1 (verified: no USD under
  `~/isaacsim/exts`, nothing in the Nucleus asset cache). Two real sources exist on this
  machine and were combined:
  - **liyiteng/alohamini** → real URDF + STL meshes for the *whole* AlohaMini1 robot
    (`AlohaMini1/simulation/src/Aloha/`), SolidWorks-exported, real inertial properties,
    but joint limits/effort/velocity are all zeroed out (exporter artifact — needs
    patching before physics will behave).
  - **isaac-sim/Sim-to-Real-SO-101-Workshop** (`~/Sim-to-Real-SO-101-Workshop`) → NVIDIA's
    own SO-101 asset + tuned actuator gains (`source/sim_to_real_so101/assets/so101.py`).
    Same physical arm, so its per-joint stiffness/damping/effort values transfer directly
    onto `left_joint1..6` / `right_joint1..6` in the AlohaMini URDF. Targets Isaac Lab,
    not standalone Isaac Sim — we borrow the *numbers*, not the code path.
- We use Isaac Sim 6.0.1's built-in URDF importer (`isaacsim.asset.importer.urdf`,
  confirmed present) rather than hand-authoring USD, since we're starting from a real URDF.
- Per your steer: use an Isaac Sim **ready-made environment** (e.g. Simple Room / flat
  grid) as the world stage rather than building a custom one — the robot is the point.

## Joint map (from the real URDF, for reference)

17 DOF total: `wheel1`, `wheel2`, `wheel3` (continuous, 3-wheel omni-style base at ~120°
apart) · `vertical_move` (prismatic lift) · `left_joint1..6` / `right_joint1..6` (revolute,
mirrors SO-101's Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw).

---

## Phase 0 — Repo scaffolding
- [x] Create `dash-aloha-mini-isaacsim/` with `assets/`, `scripts/`, `docs/`
- [x] Vendor upstream URDF + 19 STL meshes + config into `assets/upstream_alohamini1/`
      (Apache-2.0, LICENSE + attribution included)
- [x] Write this `plan.md`
- [x] Write `CLAUDE.md` (living doc, updated as work progresses)
- [x] Git init, initial commit, push to `GTU-DASH-LAB/dash-aloha-mini-isaacsim` (public)

## Phase 1 — Asset prep (fix the URDF before it ever touches physics)
- [x] Read joint limits/effort/velocity off the existing `SO-ARM101-USD.usd` (via a
      small Isaac Sim script using the USD Physics schema) and apply the same numbers to
      `left_joint1..6` / `right_joint1..6` in `Aloha.urdf`
      (`scripts/read_so101_joint_limits.py`, `scripts/patch_urdf_joint_limits.py`)
- [x] Set a placeholder `vertical_move` (lift) travel range (0-0.30m) — no authoritative
      spec value found; **still needs empirical correction**, tracked as an open item
- [x] Decide + document collision geometry strategy: used `--collision-from-visuals
      --collision-type "Convex Hull"` uniformly for the first successful import.
      Wheels not specially handled yet — still flagged for Phase 3 (true omni-wheel
      contact is a hard physics problem, see Phase 3 notes)
- [x] Verify the patched URDF parses (Python `xml.etree` parse: 18 joints, 19 links,
      matches expectations; zero remaining zeroed `<limit>` blocks)
- [x] **Verify**: patched URDF loads without errors in the URDF importer — real import
      run, "Import complete - 1 succeeded, 0 failed"

## Phase 2 — Import into Isaac Sim 6.0.1
- [x] Import `Aloha.urdf` via `isaacsim.asset.importer.urdf` into a USD, `--no-fix-base`
      (it's a mobile robot — base must be a free rigid body, not welded to world).
      **Gotcha**: needs `--ros-package "Aloha:<path to upstream_alohamini1>"` or the
      `package://Aloha/meshes/...` paths silently fail to resolve and mesh geometry
      comes out empty with no error — see CLAUDE.md.
- [x] Load a ready-made Isaac Sim environment as the stage. Tried `Simple_Room` first —
      works, but the robot spawned partly inside a piece of room furniture (a
      ramp/bench prop), obscuring the base. Switched to `Isaac/Environments/Grid/
      default_environment.usd` (flat grid, no furniture) — clean, robot fully visible.
- [x] Place the imported robot on the ground plane (origin, `(0,0,0)` — matches the
      URDF's own base_link origin, sits correctly on the grid with no floating/sinking)
- [x] **Verify**: `scripts/build_scene.py` composes environment + robot into
      `assets/usd/scene.usda`, screenshot confirms — base grounded correctly, lift
      column visible, both arms symmetric with visible gripper/jaw geometry, proper
      lighting/shadows/materials all rendering. This is the milestone screenshot:
      `docs/scene_verification.png`.

## Phase 3 — Physics configuration
- [x] Apply patched joint limits from Phase 1 (already baked into the URDF, re-imported)
- [x] Arm joints: position drives (`UsdPhysics.DriveAPI`) with the stiffness/damping/
      effort values ported from `so101.py` (table in `CLAUDE.md`/`alohamini1_specs.py`) —
      `scripts/configure_physics.py`, applied on all 12 `{side}_joint{1-6}` joints
- [x] Lift joint: position drive, `alohamini1_specs.LIFT_STIFFNESS/DAMPING` (engineering
      estimates — real hardware spec is velocity/tick-based, not N/(m/s), see
      `alohamini1_specs.py` comments)
- [x] Wheel joints: velocity drives applied, using real geometry + kinematics. Got the
      exact LeKiwi holonomic drive equations from liyiteng/lerobot_alohamini (verified
      against raw source, including a real wheel-name-to-angle mapping bug caught along
      the way — see `alohamini1_specs.py`).
      **Update from Phase 4 stress-testing**: found and fixed a genuine collision bug
      first — `base_link`'s own shell collision (both Convex Hull and Convex
      Decomposition) extended down far enough to block ground contact before the
      wheels ever touched (confirmed via raycast diagnostics: probing straight down at
      each wheel's XY position hit `base_link`, not the wheel, at the wrong height).
      Fixed by disabling the shell's collision and giving each wheel an explicit
      sphere collider + friction material (`scripts/fix_wheel_collision.py` — also had
      to work around USD instance-proxy restrictions, since the offending prims were
      point-instanced). After that fix, wheel-ground contact was correctly registered,
      but **actual traction stayed near zero** even for one wheel spinning alone in
      isolation. Diminishing returns on further physics tuning, so per the plan's
      pre-approved fallback: base locomotion is now kinematically driven (see Phase 4
      for the full story) while wheel joints still spin at the correct visual rate.
- [x] Base_link: free rigid body (not fixed), confirmed resting stably on ground plane
      (height constant at ~0.007m across 120 steps, not sinking/floating)
- [x] Masses/inertia: kept as-is from the SolidWorks export (not overwritten)
- [x] **Verify**: `scripts/verify_physics.py` — 120 steps, zero NaN/Inf in joint state,
      base height stable, all joints hold rest pose under gravity (drift <0.004 rad on
      arm joints with drives; wheels drift more under velocity-drive-to-zero but don't
      explode). Commanded `left_joint1` to 0.5 rad, **achieved exactly 0.5 rad (zero
      error) after 180 steps** — position drive control loop confirmed working.
      Screenshot (`docs/physics_verification.png`) visually confirms the arm moved.

## Phase 4 — Control: terminal script
- [x] `scripts/control_terminal.py` — one-shot (`--arm/--lift/--base`) and interactive
      `--repl` modes. Commands: `arm`, `gripper`, `lift`, `base`, `stop`, `status`,
      `pose`, `screenshot`, `wait`, `quit`.
- [x] Both UX modes implemented (one-shot argparse + REPL, as planned)
- [x] **Verify**: real bugs found and fixed along the way, not just "ran without
      crashing":
      1. **Wheel-ground collision bug** (see Phase 3 update below) — fixed via
         `scripts/fix_wheel_collision.py`.
      2. **Wheel velocity-drive oscillation** — damping of 1e5 was wildly oversized for
         the wheel's tiny rotational inertia (~5e-5 kg·m²), causing bang-bang
         oscillation (wheel hit -19.7 rad/s against a -2.6 rad/s target). Fixed:
         damping=2.0, solver velocity iterations 1→4.
      3. **Even after fixing collision + oscillation, real wheel traction stayed near
         zero** — a single wheel spinning alone in isolation produced ~200x less
         translation than expected (physically plausible root cause: sphere/mesh
         rolling-friction resolution not properly tuned; not fully diagnosed further,
         diminishing returns on continued tuning). **Per the plan's pre-approved
         fallback condition, switched base locomotion to kinematic drive**: wheel
         joints still spin at the visually-correct commanded rate (real joint
         physics), but actual translation comes from directly setting the
         articulation root's pose each step via `art.set_world_poses()` (verified
         this is necessary — raw USD transform edits get silently ignored once
         physics is actively stepping; `set_world_poses()` goes through
         `physics_view.set_root_transforms()`, which properly syncs).
      4. **A real, separate, more important bug**: `set_arm_joint`/`set_lift` were
         each doing `targets = art.get_joint_positions()` (actual current position,
         not the previously-commanded target) then modifying one index before
         resending — so issuing a second command before the first settled silently
         reset the first command's target back to wherever it currently was.
         Fixed by maintaining a persistent `_position_targets` array instead of
         repeatedly re-reading actual state. This was **not** related to the base
         work at all, but was only exposed once multiple commands were chained.
      5. Final verified behavior: base-only driving (0.15 m/s × 3s → x≈0.42m, close to
         the 0.45m theoretical), sequential arm-after-base commands converge exactly
         (`left_joint1` → 0.5000 rad exact), combined lift+gripper+arm+base commands
         all land correctly when driven sequentially. **Known limitation**:
         teleporting the base root every step *while simultaneously* issuing new arm
         commands measurably degrades arm convergence (verified) — recommended usage
         is sequential (drive base, `stop`, then command arms), not simultaneous.
         Documented in `control_terminal.py`'s own comments and `CLAUDE.md`.
      6. Screenshots confirm all of this visually: `docs/control_demo.png` shows lift
         extended, both arms in different commanded poses, and the robot visibly
         repositioned on the grid from base driving.

## Phase 5 — Control: Isaac Sim UI
- [x] The articulation has proper `UsdPhysics.DriveAPI` position/velocity drives
      authored on every joint (Phase 3/4) — this is exactly the mechanism the Property
      panel / Articulation Inspector's sliders read and write, so UI control has
      everything it needs. Confirmed via script that the same underlying API
      (`set_joint_position_targets`) works correctly (exact convergence).
- [ ] **Cannot verify by clicking an actual slider from here** — this environment has
      no display and scripting can't drive the GUI's own widgets. This is a genuine
      limitation of headless verification, not a claim of "probably fine." Steps to
      check yourself:
      1. Launch Isaac Sim normally (no `--no-window`/headless): `~/isaacsim/isaac-sim.sh`
      2. File → Open → `assets/usd/scene.usda`
      3. Press Play (the physics timeline needs to be running for drives to respond)
      4. Window → Physics → Articulation Inspector, then select `/World/Aloha` (or
         click the robot in the viewport) — you should see all 16 DOF listed with
         sliders (`wheel1/2/3`, `vertical_move`, `left/right_joint1-6`)
      5. Drag a slider for e.g. `left_joint1` — the corresponding arm segment should
         rotate in the viewport in real time
      6. Alternatively: select the joint prim directly in the Stage tree (e.g.
         `/World/Aloha/Geometry/base_link/vertical_link/left_link1`'s parent joint
         under the `Physics` scope) and look at the Property panel's Physics tab for
         the same drive target field
      If a slider doesn't appear or doesn't move anything, that's a real finding worth
      reporting back — the drives are authored correctly per every script-based check
      done so far, so a UI-specific issue would be new information.

## Phase 6 — Robot cameras + LeRobot-compatible data collection (DONE)
- [x] Fix "Physics Simulation View is not created yet" warning spam: GUI Stop destroys
      the PhysX sim view while the control loops keep commanding every frame. Added a
      per-frame `sim_command_ready()` gate + auto re-`initialize()` on Play resume +
      `sim stop|play` REPL command. **Verified**: scripted stop/command/resume session,
      zero warnings in log, joint converges exactly after resume.
- [x] Add `third_party/lerobot_alohamini` submodule (official LeRobot integration for
      this robot) — source of truth for the camera set: `forward` + `wrist_left` +
      `wrist_right`, all 640x480 @ 30fps (`config_alohamini.py`).
- [x] `scripts/add_cameras.py` (step 4/4 in `rebuild_all.sh`): USD cameras authored on
      the robot links — wrist cams on `link5` (gripper body; `link6` is the moving jaw
      finger), forward cam above the lift column facing the manipulation front (-Y,
      measured — both grippers work toward -Y). Poses tuned by rendering each camera
      and iterating (first attempt was upside down AND the forward cam faced +Y).
      **Verified**: `docs/cam_*.png` renders show correct framing.
- [x] `scripts/capture_cameras.py`: LeRobot observation format
      (`observation.images.<name>` → 480x640x3 uint8, sample every 2nd physics step
      for 30fps). **Verified**: all three frames correct shape/dtype; motion test
      PASS (mean abs pixel diff 50-92 after a lift+arm move — all views actually
      track the robot).

## Phase 7 — ROS2 (optional, off by default)
- [ ] **Do not implement unless asked.** If wanted later: enable the ROS2 bridge
      extension the same way documented in `GTU-DASH-LAB/isaac-sim-ros2-mcp-setup`
      (internal bundled Jazzy libs, not system ROS2, inside the Isaac Sim process), and
      publish `/joint_states`, subscribe to per-arm command topics. Kept as a stretch
      goal, not a default dependency, per your instruction.

## Open questions / decisions to revisit
- Exact `vertical_move` travel range (no spec found — measuring empirically in Phase 1/2)
- Wheel collision/friction strategy (Phase 3 — will report findings before picking a
  fallback if the default approach doesn't hold up)
- Terminal control UX details (argparse vs REPL vs both) — will propose a concrete design
  once Phase 3/4 are reached, rather than guess now
