# Architecture

How this repo is put together, and why. For the chronological task log see
[`plan.md`](plan.md); for hard-won gotchas and debugging history see
[`CLAUDE.md`](CLAUDE.md); for user-facing quick-start commands see
[`README.md`](README.md).

## The big picture

```
                    ┌─────────────────────────────────────────────┐
   real robot data  │  assets/upstream_alohamini1/ (URDF + STLs)  │  vendored from
                    └──────────────────┬──────────────────────────┘  liyiteng/alohamini
                                       │  isaacsim.asset.importer.urdf
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │        assets/usd/Aloha/Aloha.usda          │  imported robot
                    └──────────────────┬──────────────────────────┘
                                       │  scripts/rebuild_all.sh (4 steps + verify)
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │           assets/usd/scene.usda             │  the file you open /
                    │  warehouse + robot + tables + blocks +      │  control / record from
                    │  joint drives + wheel fix + cameras         │
                    └──────────────────┬──────────────────────────┘
                          ┌────────────┼────────────────┐
                          ▼            ▼                ▼
                   Isaac Sim GUI   scripts/control/   scripts/cameras/
                   (Play + joint   control_terminal   capture_cameras (LeRobot
                   sliders)        (REPL/joystick/    frames), view_cameras
                                   network bridge)    (live GUI viewports)
```

## Directory layout

```
assets/
  upstream_alohamini1/   vendored URDF + 19 STL meshes (Apache-2.0, liyiteng/alohamini)
  usd/Aloha/             robot USD produced by the URDF importer
  usd/scene.usda         composed scene -- ~18KB of *references* + physics overrides
                         (NOT flattened; see "scene.usda is small on purpose" below)
scripts/
  alohamini1_specs.py    single source of truth: all physical constants, camera
                         names/paths, wheel kinematics. Everything imports from here.
  rebuild_all.sh         THE way to rebuild scene.usda (runs the 4 pipeline steps in
                         order + a verification pass)
  pipeline/              scene-building steps, in run order:
    build_scene.py         1. environment + robot + tables/blocks + camera-ready layer
    configure_physics.py   2. joint drives (arms/lift position, wheels velocity)
    fix_wheel_collision.py 3. wheel sphere colliders + friction (see CLAUDE.md)
    add_cameras.py         4. robot-mounted cameras (forward + both wrists)
    verify_physics.py      final check: stability + a joint convergence test
    patch_urdf_joint_limits.py  one-time URDF fix (already applied; rerun only if
                                the vendored URDF is refreshed)
  control/
    control_terminal.py    terminal control: one-shot CLI, REPL, PS4 joystick
                           (local evdev), TCP network-bridge joystick
    joystick_bridge_local.py  runs on YOUR laptop (pygame) and streams the
                              controller to control_terminal --joystick-network
  cameras/
    capture_cameras.py     LeRobot-format frame grabbing (observation.images.<name>)
    view_cameras.py        Isaac Sim GUI with one live viewport per robot camera
  tools/                   one-off debug/inspection helpers (stage dumps, import
                           verification, reading limits off NVIDIA's reference asset)
third_party/
  lerobot_alohamini/       git submodule: the official LeRobot integration for this
                           robot -- source of truth for camera specs + kinematics
docs/                      verification screenshots (every claim in the docs has one)
```

## The scene pipeline (why `rebuild_all.sh` exists)

`build_scene.py` recreates `scene.usda` **from scratch** every run. Steps 2-4 layer
sparse USD overrides on top of it (drives, collision fixes, cameras). Running step 1
alone silently wipes steps 2-4 — no error, just joints with zero stiffness that
converge nowhere. This exact failure happened once and cost a debugging round, so:
**never run `pipeline/build_scene.py` by hand; always `./scripts/rebuild_all.sh`**,
which runs everything in order and finishes with `verify_physics.py` (stability +
joint-convergence check).

Key facts about the composed scene:

- **`scene.usda` is small on purpose** (~18KB). It holds *reference arcs* to the
  warehouse environment / table props / blocks (NVIDIA's asset CDN, cached locally by
  Kit after first open) and to the robot (relative path inside this repo), plus the
  physics/camera overrides. A flattened export was 233MB and GitHub rejected it.
- **It authors its own `/World/PhysicsScene`** — without that prim, pressing Play
  simulates nothing (Isaac's new-stage template normally injects one; a hand-authored
  root layer doesn't get it).
- **Environment choice is load-bearing, not cosmetic** — the Office asset is a
  ~1000m building that both broke camera framing and exploded the robot at spawn
  (geometry overlap); Simple_Room hides a giant table at the origin. The warehouse
  has a real floor at Z=0. Details in CLAUDE.md.

## Control (`scripts/control/`)

`control_terminal.py` — one process, four entry modes:

| Mode | Flag | Input |
|---|---|---|
| one-shot | `--arm/--lift/--base` | CLI args, settles, prints state |
| REPL | `--repl` | typed commands (`help` lists them; `sim stop/play` mirrors GUI) |
| joystick | `--joystick` | PS4 pad plugged into THIS machine (evdev) |
| network joystick | `--joystick-network` | JSON-over-TCP from `joystick_bridge_local.py` on your laptop |

Two implementation details worth knowing before editing it:

- **The base is kinematically driven** (root-pose teleport each frame via
  `Articulation.set_world_poses`), not by wheel-ground friction — real traction
  physics was tried extensively and documented in CLAUDE.md/plan.md Phase 4 before
  falling back. Wheel joints still spin at the visually correct rate.
- **Every PhysX-touching call goes through `sim_command_ready()`** — GUI Stop
  destroys the physics view (commands would warn-spam), and Play afterwards requires
  re-`initialize()`. The guard handles both transitions once, quietly.

## Cameras + data collection (`scripts/cameras/`)

Camera names, resolution and fps mirror the **official** AlohaMini LeRobot config
(`third_party/lerobot_alohamini/.../config_alohamini.py`): `forward`, `wrist_left`,
`wrist_right`, all 640x480 @ 30fps. Prim paths live in
`alohamini1_specs.CAMERA_PRIM_PATHS`; the cameras are authored by pipeline step 4 as
children of the robot links, so they move with the lift/wrists.

- `capture_cameras.py` → `observation.images.<name>` -> (480, 640, 3) uint8 dicts
  (exactly what a LeRobot dataset recorder consumes). Physics runs at 60Hz; sample
  every 2nd step for the official 30fps. `--motion-test` self-checks that every view
  actually changes when the robot moves.
- `view_cameras.py` → Isaac Sim GUI with one live viewport per camera (or
  `--screenshot-test` headless, which is how it's verified in CI-like runs).

## Verification philosophy

Every physics/geometry claim in this repo was verified by measurement, not eyeball:
rendered screenshots (committed to `docs/`), `BBoxCache` world-bounds queries,
actual-vs-target joint positions after settling, or raycast probes. When something
"looked right" but wasn't, the discrepancy is documented in CLAUDE.md's gotchas
(several were only caught this way — upside-down cameras, cubes falling through
tables, a "stuck" lift that was a solver-iteration issue). If you change physics or
geometry, add the same kind of check before trusting it.
