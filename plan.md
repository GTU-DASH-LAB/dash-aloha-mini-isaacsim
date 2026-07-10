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
- [ ] Load a ready-made Isaac Sim environment (Simple Room or flat grid) as the stage
- [ ] Place the imported robot on the ground plane
- [x] **Verify**: headless script loads the stage, computes world bbox on `/Aloha`
      (≈ 0.42m × 0.46m × 1.09m tall — plausible), frames the camera on it with
      `frame_viewport_prims`, and captures a screenshot — geometry confirmed present
      and correctly shaped (dark/unlit silhouette, expected with no light source yet;
      lighting comes next via the ready-made environment)

## Phase 3 — Physics configuration
- [ ] Apply patched joint limits from Phase 1
- [ ] Arm joints: `ImplicitActuator`-equivalent drives with the stiffness/damping/effort
      values ported from `so101.py` (see table in `CLAUDE.md` once transcribed)
- [ ] Lift joint: position drive, conservative stiffness/damping (tune empirically)
- [ ] Wheel joints: **decision point** — true wheel-ground friction contact for 3-wheel
      omni bases is genuinely hard to get stable in a general rigid-body solver. Default
      plan: real revolute joints + velocity drive per wheel (physically simulated
      rotation), with a *simplified* collision proxy (cylinder, not the exact roller
      mesh) and an isotropic-friction approximation. If that produces bad/unstable
      holonomic motion in testing, fall back to a documented simplified mode (direct
      base velocity command) as an alternative control path — will report back before
      silently doing this, since it's a fidelity trade-off you should sign off on
- [ ] Base_link: free rigid body (not fixed), resting on ground plane via wheel contacts
- [ ] Masses/inertia: already present from the SolidWorks export — keep them, don't
      overwrite, unless physics is visibly unstable
- [ ] **Verify**: run N physics steps headless, assert no NaN/Inf in any joint state or
      link pose, robot doesn't visibly explode/sink through the floor (check base height
      stays sane across steps)

## Phase 4 — Control: terminal script
- [ ] Standalone Python script (`scripts/control_terminal.py`) using Isaac Sim's
      Articulation API to command: arm joint positions (left/right independently),
      gripper open/close, lift height, base velocity (subject to Phase 3 outcome)
- [ ] Simple CLI interface — args or an interactive REPL loop (exact UX to be decided
      once Phase 3 lands; default plan is argparse for one-shot commands + a `--repl` mode)
- [ ] **Verify**: script commands a joint, read back actual joint state after settling,
      confirm it matches target within tolerance — for at least one arm joint, the lift,
      and (if implemented) base motion

## Phase 5 — Control: Isaac Sim UI
- [ ] Confirm the imported articulation shows up correctly in the Property panel /
      Articulation Inspector with per-joint sliders (this should come "for free" once
      Phase 3 drives are configured correctly — mostly a verification step, not new work)
- [ ] **Verify**: manually move a joint slider in the UI, confirm the robot responds
      (screenshot before/after)
- [ ] Document exact UI steps (which panel, how to select the articulation) in `CLAUDE.md`

## Phase 6 — ROS2 (optional, off by default)
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
