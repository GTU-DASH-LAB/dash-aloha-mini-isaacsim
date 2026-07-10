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
3. **Report before silently downgrading fidelity.** E.g. if the wheel contact physics
   turns out unstable once actually driven, come back with what was tried and what the
   fallback looks like — don't just quietly simplify and move on.
4. **Don't overwrite the SolidWorks-derived mass/inertia values** in the URDF — they're
   real, computed from the actual CAD models.
5. **Verify numbers against raw source, not AI-paraphrased summaries**, when porting
   constants from another repo — a paraphrase missed a real sign-negation in the LeKiwi
   kinematics once (see Gotchas). Fetch and grep the actual file.

## Key technical facts

- **Isaac Sim 6.0.1** lives at `~/isaacsim` (Kit's embedded Python is **3.12**). URDF
  importer extension: `isaacsim.asset.importer.urdf`.
- **Upstream URDF**: `assets/upstream_alohamini1/urdf/Aloha.urdf`, SolidWorks-exported.
  17 DOF: `wheel1/2/3` (continuous), `vertical_move` (prismatic, base_link→vertical_link),
  `left_joint1..6` / `right_joint1..6` (revolute, mirror SO-101's Rotation/Pitch/Elbow/
  Wrist_Pitch/Wrist_Roll/Jaw). Joint limits/effort/velocity were all zeroed by the
  exporter originally — now patched (see `scripts/patch_urdf_joint_limits.py`).
- **19 mesh files** vendored under `assets/upstream_alohamini1/meshes/` (~15MB, no
  git-lfs needed).
- **All constants (arm gains, lift range, wheel kinematics) live in one place**:
  `scripts/alohamini1_specs.py`. Don't duplicate numbers elsewhere — import from there.
  Sources: arm gains from NVIDIA's `Sim-to-Real-SO-101-Workshop` (same physical arm,
  Isaac Lab config but gains transfer); lift range + wheel kinematics from
  `liyiteng/lerobot_alohamini` (real hardware control code for this exact robot).
- **No pre-built ALOHA/SO-101 asset ships with Isaac Sim 6.0.1** — confirmed by
  searching `~/isaacsim/exts` and the Nucleus asset cache. Only a MuJoCo (not USD) test
  scene exists under `isaacsim.pip.newton/pip_prebundle/mujoco_warp/test_data/
  aloha_pot/` — different physics stack, not usable directly.
- **Isaac Sim's asset CDN root** on this machine resolves to
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`
  (network access confirmed working for streaming ready-made environments).

## File map

```
assets/upstream_alohamini1/   vendored URDF + meshes + config from liyiteng/alohamini
assets/usd/Aloha/             imported USD (regenerate via urdf_import.py if URDF changes)
assets/usd/scene.usda         composed scene: ready-made environment + robot + drives
                               (regenerate via build_scene.py + configure_physics.py)
scripts/alohamini1_specs.py   single source of truth for all physical constants
scripts/patch_urdf_joint_limits.py   patches the URDF's joint limits (already applied)
scripts/build_scene.py        composes environment + robot -> scene.usda
scripts/configure_physics.py  applies joint drives onto scene.usda
scripts/verify_import.py      headless import sanity check + screenshot
scripts/verify_physics.py     headless physics stability check + joint command test
docs/                         milestone screenshots + import logs
plan.md                       phased task list, checkboxes
CLAUDE.md                     this file
```

## Gotchas hit so far

- **`package://` URDF mesh paths need `--ros-package` or meshes silently vanish.**
  `Aloha.urdf` references meshes as `package://Aloha/meshes/base_link.STL`. The URDF
  importer does **not** error if it can't resolve that — it imports the robot with zero
  mesh geometry and no warning. Always pass `--ros-package
  "Aloha:/absolute/path/to/assets/upstream_alohamini1"`.
- **`stage.Traverse()` + `UsdGeom.Mesh` type-checking undercounts on this asset** — the
  importer's output uses point-instancing for symmetric parts, so a naive
  `prim.IsA(UsdGeom.Mesh)` walk reports 0 even when meshes are genuinely present and
  rendering. Verify via `UsdGeom.BBoxCache` + an actual screenshot instead.
  `frame_viewport_prims(viewport, prims=[...])` (from `omni.kit.viewport.utility`) is
  the reliable way to aim a headless camera at freshly imported/composed geometry — the
  default camera does not auto-frame it.
- **A wheel-name/angle mapping bug caught by checking raw URDF origins, not assuming.**
  `liyiteng/lerobot_alohamini`'s LeKiwi kinematics use wheel angles `[150, -90, 30]`
  degrees in that order. Naively pairing that with our URDF's `wheel1/wheel2/wheel3` in
  the same order is **wrong** — computed each wheel's actual position angle from its
  URDF joint origin xyz (`wheel1`≈149°, `wheel3`≈-91°, `wheel2`≈30°), giving the correct
  pairing `["wheel1", "wheel3", "wheel2"]`. This is baked into `alohamini1_specs.py`
  with the derivation in a comment — don't re-derive, but do sanity check if the URDF
  ever changes.
- **Environment choice matters for visibility, not just lighting.** `Simple_Room`
  renders fine but the robot spawned partly inside a furniture prop, obscuring the
  base. Using `Isaac/Environments/Grid/default_environment.usd` instead — plain floor,
  full robot visible. Revisit if a "real room" look is wanted later.

## Current status

**Phases 0-3 done and verified.** Pipeline: patched URDF → `isaacsim.asset.importer.urdf`
→ `assets/usd/Aloha/Aloha.usda` → composed with a ready-made environment →
`assets/usd/scene.usda` → joint drives applied. Verified end to end:
- Visual: robot renders correctly, grounded, both arms + lift column visible
  (`docs/scene_verification.png`)
- Physics: 120 steps, zero NaN/Inf, base height rock-stable, joints hold rest pose
  under gravity, commanded `left_joint1` to 0.5 rad and it landed exactly on target
  after settling (`docs/physics_verification.png`)

**Genuinely open, not a blocker**: wheel *kinematics* are solved (exact LeKiwi
equations ported and verified), but wheel *contact physics* under actual nonzero
velocity commands hasn't been stress-tested yet — current collision proxy is Convex
Hull on the real wheel mesh (not simplified). This is naturally Phase 4's first test
(driving the base). If unstable, fallback is a simplified collision proxy — will report
back rather than silently downgrading, per Ground Rule 3.

## Next step

Phase 4: terminal control script (`scripts/control_terminal.py`) — command arm joint
positions, gripper, lift height, and base velocity (via
`alohamini1_specs.body_to_wheel_speeds()`). The base-driving test is also where the
wheel contact physics question above gets answered empirically.
