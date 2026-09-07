# dash-aloha-mini-isaacsim

**AlohaMini1** — a mobile base with a vertical lift and two SO-101 arms — simulated in
NVIDIA Isaac Sim 6.0.1 with full rigid-body physics, and driven from a typed sentence.
The policy is never given a goal coordinate. It gets its own camera and the words.

![A warehouse run: the robot's camera with seven coloured candidate paths drawn on the floor, next to a third-person view of the robot driving toward the aisle](docs/media/nav_warehouse_aisle05.gif)

> **“Proceed to the traffic cones at the entrance of Aisle 05 (the second aisle from the
> right).”** — the entire input, alongside the camera frame.
>
> **Left is everything the policy gets** — the robot's own camera, with the candidate
> paths drawn onto the floor and the one it picked lit up. **Right is a third-person view
> it never sees**, recorded so a viewer can check whether the robot then went where it
> said it would. Episode `warehouse`: arrived **1.50 m** from the cones over 15.4 m of
> path. Shown at 5× speed.

Three more, same two views stacked — the camera and its menu on top, the third-person
view below:

<table>
<tr>
<td width="33%"><img width="100%" src="docs/media/nav_hospital_wheelchairs.gif" alt="Hospital run: the robot drives past wheelchairs toward a water dispenser by a window"></td>
<td width="33%"><img width="100%" src="docs/media/nav_office_hallway_turn.gif" alt="Office run: the robot drives down a hallway and turns left at the end"></td>
<td width="33%"><img width="100%" src="docs/media/nav_hospital_vending_pivot.gif" alt="Hospital run with pivot options: two menu items are turn-in-place arrows instead of arcs"></td>
</tr>
<tr>
<td valign="top"><b><code>hospital_past_wheelchairs</code></b><br>“…stop at the water dispenser by the window.” 13.9 m of path, 31 s. <i>5×</i></td>
<td valign="top"><b><code>office_hallway_turn</code></b><br>“…then turn left at the end to approach the red emergency exit door.” A corner, not a straight line. <i>7×</i></td>
<td valign="top"><b><code>hospital_vending_machine</code></b><br>The pivot variant — menu items <b>1</b> and <b>6</b> are turn-in-place arrows, not arcs. <i>5×</i></td>
</tr>
</table>

The policy behind all four clips is a **frozen Qwen3.8-27B**: no fine-tuning, no
trajectory data, no action head. The whole thing is arcs drawn on the robot's own camera
and two calls asking which number to take. It scores **8 of 13** on the benchmark ladder,
against 6/13 for the fine-tuned VLA it replaced — the first three clips are episodes from
that scored run, the fourth is a variant arm that also offers turn-in-place. How it got
there, including the measurements that killed the more obvious approaches, is in
[`nav/README.md`](nav/README.md).

The same robot is also a manipulator — two SO-101 arms and a 0–0.6 m lift, driven from a
terminal REPL or a gamepad:

![Robot after a control sequence: lift extended, arms moved, base driven](docs/control_demo.png)

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — **start here to understand the repo**: layout,
  data flow, the scene pipeline, control paths, cameras
- [`plan.md`](plan.md) — phased implementation plan with checkboxes
- [`CLAUDE.md`](CLAUDE.md) — living project doc (facts, decisions, gotchas — kept current
  as work progresses)
- [`assets/upstream_alohamini1/`](assets/upstream_alohamini1/) — vendored URDF + meshes
  from [liyiteng/alohamini](https://github.com/liyiteng/alohamini) (Apache-2.0)
- [`scripts/control/control_terminal.py`](scripts/control/control_terminal.py) — terminal control (see
  Quick Start below)
- [`nav/README.md`](nav/README.md) — language-driven navigation: the arc menu, TIC-VLA,
  and the DynaNav benchmark, see below
- [`robot/README.md`](robot/README.md) — **putting that policy on a real robot**: the
  three interfaces you implement against your own camera and motor SDK, and how to bring
  it up without Isaac Sim, the submodules, or a GPU on the robot itself

Status: Phases 0-4 done and verified (import, physics, terminal control). Phase 5 (UI
control) has the underlying mechanism verified but not click-tested in an actual GUI
session — see `plan.md`/`CLAUDE.md` for exact steps to check yourself.

No ROS2 dependency by default. Isaac Sim 6.0.1 install expected at `~/isaacsim`.

## Language-driven navigation

The clips above. Two policies have driven this, both scored on benchmark
environments/instructions from **DynaNav**: **TIC-VLA**, a fine-tuned
vision-language-action model with a trained action head, and the **arc menu**, a frozen
Qwen3.8-27B that never saw a trajectory. Full detail — the architecture, the
version-pairing constraints that force it into two separate processes, every measured
result — lives in
[`nav/README.md`](nav/README.md), [`nav/plan.md`](nav/plan.md), and the "Language-driven
navigation" section of [`CLAUDE.md`](CLAUDE.md). Short version:

- One camera frame + a sentence go in (never a goal coordinate); a velocity command
  comes out, at 10Hz, mediated by a controller that turns the policy's short-horizon
  plan into a safe drive command.
- The DynaNav benchmark ships across **three environments** — hospital, office,
  warehouse — each with several scripted episodes (start pose, goal, instruction,
  success threshold). One Isaac Sim stage is built per *environment*, not per episode,
  since the runner teleports to each episode's own start pose before every run; the UI
  can then switch between every episode sharing that stage without a rebuild.
- Current result on the 13-episode ladder (easiest-first by our own difficulty
  ranking): **8/13 with the arc menu**, up from **6/13 with TIC-VLA**. Under TIC-VLA the
  mean SPL was 0.52 on the 11 episodes DynaNav also scores, two of them
  (`hospital_forward_staircase`, `hospital_exit_room`) beating DynaNav's own SPL
  outright. Full tables, and the failure modes behind the ones that still fail, are in
  `nav/plan.md`'s Phase 18 and the "Language-driven navigation" section of `CLAUDE.md`.
  Read `n=1` on this stack as noise: the same episode and sentence can succeed twice and
  fail once, because generation timing decides which plan is in hand at which step.

```bash
nav/sim/build_nav_scene.sh hospital     # once per environment (hospital/office/warehouse)
nav/run.sh --episode hospital_down_hallway
```

Then open `http://127.0.0.1:8080` — every episode in the loaded environment is
clickable and runnable from there; the others are listed with the relaunch command.

**Two full TIC-VLA runs at full resolution**, recorded from the live UI with the
third-person view on — download to play, since GitHub will not play a video that lives
in a repo:

- [`docs/nav_demo_hospital_forward_staircase.mp4`](docs/nav_demo_hospital_forward_staircase.mp4)
  — turn into a side hallway toward a staircase, `+66.1°`, one of the harder episodes by
  turn size. **Succeeds**, SPL 0.90, beating DynaNav's own 0.88.
- [`docs/nav_demo_hospital_down_hallway2.mp4`](docs/nav_demo_hospital_down_hallway2.mp4)
  — straight hallway approach to a bed beside a door, `32.5 m` start. **Succeeds**,
  SPL 1.00, matching DynaNav.

The clips at the top of this file are cut from arc-menu runs recorded the same way —
[`docs/media/README.md`](docs/media/README.md) says which run each one is and how to cut
another. Every run writes its menus and decisions as it goes, so any run can be replayed
into a video of what the model saw and chose:

```bash
python3 nav/tools/make_run_video.py --latest      # or --all
```

That tool is a debugging instrument first. A benchmark row says an episode closed 87% of
its gap with 506 collision-guard stops; it cannot say whether the model kept picking a
curve into a wall, lost the corridor at one specific corner, or was steering fine while
the controller scraped. Those have different fixes, and only the video separates them —
which is also why the third-person panel is recorded at all. A scrape, a pivot in place
and a reverse out of a wedge are all invisible from the camera doing the deciding,
because that camera moves with the robot.

## Cloning

```bash
git clone --recurse-submodules <this repo>
```

Two submodules, **both optional**, and it is worth knowing which you need before you
fetch either:

| submodule | needed for | size |
|---|---|---|
| `third_party/TIC-VLA` | reproducing the simulated navigation benchmark — the DynaNav episode definitions and the office/outdoor `.usd` scenes | several GB, mostly scene assets |
| `third_party/lerobot_alohamini` | the LeRobot data-collection path | small |

Neither is needed to run the navigation policy, and **neither is needed on a real
robot** — see [`robot/README.md`](robot/README.md). A plain `git clone` gives you a
working repo with both directories empty; fetch one later with:

```bash
git submodule update --init third_party/TIC-VLA
```

Everything else this code needs lives outside git — the model weights, the Isaac Sim
install, the virtualenvs. `nav/paths.py` resolves all of them, and running it prints
what it found and what is missing:

```bash
python3 nav/paths.py
```

Each root takes an environment variable (`TICVLA_ROOT`, `QVLA_MODEL_ROOT`,
`ISAACSIM_ROOT`) if yours live somewhere else. Nothing is hardcoded to one machine any
more; the workstation's own paths survive only as a last-resort fallback.

## Quick start

`assets/usd/scene.usda` is already committed and ready to use as-is — the default
environment is `Simple_Warehouse` (a "factory" setting) with two official NVIDIA
packing tables (real legs, totes/crates, collision physics) flanking the robot, and
four official colored blocks (red/green/blue/yellow, full rigid-body physics) resting
on their work surfaces. The tables sit just outside the base's in-place rotation
radius, so you can turn freely and drive up to either one for pick-and-place. The
default camera starts framed close to the robot regardless of environment size.

`scene.usda` is intentionally tiny (~18KB): it holds *references* to the environment
/props (Isaac's CDN) and the robot (relative path in this repo) plus the physics
overrides — opening it needs network the first time, then Kit caches the assets. See
`CLAUDE.md` before swapping in a different `--environment-url`: not all of Isaac
Sim's official environments are safe as-is (the `Office` environment used to be the
default here and had to be replaced — it's real-world building scale, ~1000m across,
which caused both a camera-framing bug and a physics explosion at spawn; `Simple_Room`
has furniture parked right at the origin).

If you change the URDF, or want to swap the environment, **always use
`scripts/rebuild_all.sh`**, not the individual scripts by hand:

```bash
./scripts/rebuild_all.sh                                    # rebuild with defaults
./scripts/rebuild_all.sh --environment-url <other CDN url>   # swap environment
./scripts/rebuild_all.sh --no-pick-place-props               # skip the table/cubes
```

Why not just `build_scene.py`: it recreates `scene.usda` from scratch every time,
which silently wipes out the joint drives and wheel-collision fixes that
`configure_physics.py`/`fix_wheel_collision.py` layer on top — `rebuild_all.sh` always
runs all three in the right order, plus a final `verify_physics.py` sanity check.

If you change the URDF itself, re-import first:
```bash
~/isaacsim/python.sh -m standalone_examples.api.isaacsim.asset.importer.urdf.urdf_import \
  --urdf assets/upstream_alohamini1/urdf/Aloha.urdf \
  --usd-path assets/usd \
  --ros-package "Aloha:$(pwd)/assets/upstream_alohamini1" \
  --collision-from-visuals --collision-type "Convex Decomposition" \
  --no-fix-base --merge-fixed-joints
./scripts/rebuild_all.sh
```

Control it from the terminal:

```bash
# One-shot
~/isaacsim/python.sh scripts/control/control_terminal.py --arm left 1 0.5 --settle 2

# Interactive
~/isaacsim/python.sh scripts/control/control_terminal.py --repl
> arm left 1 0.5
> gripper right close
> lift 0.3
> base 0.15 0 0
> wait 3
> stop
> screenshot out.png
> quit
```

Note: drive the base, then `stop` it, *before* issuing new arm commands — see
`CLAUDE.md`'s "kinematic root-teleporting fights concurrent arm-joint convergence" note
for why simultaneous base+arm commands don't converge as cleanly.

Type `help` in the REPL to list all commands, or `help <command>` (e.g. `help arm`)
for usage/limits on one of them — also shown automatically if you type a command name
with the wrong number of arguments (e.g. just `arm` or `base` alone). Command history
(Up arrow to recall the previous command) works if `gnureadline` is installed:

```bash
~/isaacsim/python.sh -m pip install gnureadline
```

## Robot cameras + data collection (LeRobot-compatible)

The robot carries three cameras matching the **official AlohaMini LeRobot config**
(names, 640x480 resolution, 30fps — from
[`third_party/lerobot_alohamini`](third_party/lerobot_alohamini), vendored as a
submodule; run `git submodule update --init` after cloning):

| Camera | Mount | View |
|---|---|---|
| `forward` | above the lift column | front workspace/table ([docs/cam_forward.png](docs/cam_forward.png)) |
| `wrist_left` | left gripper body (link5) | along the gripper, fingers at frame bottom |
| `wrist_right` | right gripper body (link5) | mirror of wrist_left |

**See all camera views live in Isaac Sim.** Either as a standalone viewer (main
viewport + one window per camera, physics running — command the robot from a second
terminal and watch the wrist views move):

```bash
~/isaacsim/python.sh scripts/cameras/view_cameras.py
```

...or built into the control REPL itself, no second process needed — `--cameras`
opens all three at startup, or type `cameras` at any point during the session:

```bash
~/isaacsim/python.sh scripts/control/control_terminal.py --repl --gui --cameras
> arm left 1 0.5      # watch the wrist_left window move as this runs
```

(Or manually in any Isaac Sim viewport: camera icon → Cameras → `camera_forward` /
`camera_wrist_left` / `camera_wrist_right`; Window → Viewport → Viewport 2 for extra
viewports.)

Grab frames in LeRobot's observation format (`observation.images.<name>` →
480×640×3 uint8):

```bash
# one frame per camera to docs/, plus a motion self-test
~/isaacsim/python.sh scripts/cameras/capture_cameras.py --save-dir docs --motion-test
```

For episode recording, import `get_camera_observation()` from that script (or copy
the ~15-line pattern) and sample every 2nd physics step — physics runs at 60Hz, the
official cameras are 30fps. The camera prim paths live in
`scripts/alohamini1_specs.py` (`CAMERA_PRIM_PATHS`) if you want to wire them into
your own pipeline.

Note: the cameras are part of the scene build — if you rebuild, always use
`./scripts/rebuild_all.sh` (it runs `add_cameras.py` as step 4/4).

## PS4 controller control

```bash
~/isaacsim/python.sh scripts/control/control_terminal.py --joystick --gui
```

Needs the `evdev` package (`~/isaacsim/python.sh -m pip install evdev`) and your user
in the `input` group — check with `groups | grep input`; if it's not listed:

```bash
sudo usermod -aG input $USER   # then log out and back in
```

Mapping: **L1**=control right arm, **L2**=control left arm, **L1+L2 together**=control
both arms mirrored (opposite movement), **R2**=control the base. Left stick and right
stick move different joints/axes depending on which mode is active — see the full
mapping table in `scripts/control/control_terminal.py`'s module docstring.

**This has not been tested against a physical controller** — none was connected in the
environment it was built in. It's implemented against the standard Linux `evdev` codes
for a DualShock 4, but exact button/axis codes can vary by driver. Run
`scripts/control/control_terminal.py --joystick-debug` first to print raw events from your
controller and confirm they match `JOYSTICK_MAP` at the top of the joystick section —
adjust the numbers there if your controller reports different codes.

### Controller plugged into a *different* machine (e.g. controlling this box over AnyDesk)

AnyDesk (and most remote-desktop tools) only forwards keyboard/mouse/screen, not
USB/gamepad devices. If your controller is plugged into your own local machine and
you're remoting into this one, use the network bridge instead of `--joystick`:

**On this machine** (the one running Isaac Sim):
```bash
~/isaacsim/python.sh scripts/control/control_terminal.py --joystick-network --port 9999 --gui
```

**On your local machine**:
```bash
python3 -m pip install pygame
python3 scripts/control/joystick_bridge_local.py --host <this-machine> --port 9999
```

If your local machine can reach this one directly (same LAN — this machine's address
is `10.1.18.165`), point `--host` straight at it. If not (likely, since you're going
through AnyDesk — probably a different network), tunnel over SSH instead. From your
local machine:
```bash
ssh -L 9999:localhost:9999 <your-username>@10.1.18.165
```
Leave that running in its own terminal/tab, then run `joystick_bridge_local.py --host
localhost --port 9999` — the tunnel forwards it through. This works with a plain SSH
tunnel (no extra VPN/tooling needed) because the bridge uses TCP, not UDP.

Verified end-to-end on the remote side (a real TCP client was connected and driven
through all four modes — `right_arm`, `both_sync`, `base`, back to `none` — with clean
disconnect handling). **The local half (pygame reading your actual controller) is not
verified** — I don't have access to your machine. Run `joystick_bridge_local.py --debug`
first to confirm the button/axis indices match `DEFAULT_MAPPING` in that script before
trusting it; override with `--button-l1`, `--axis-l2`, etc. if they don't.
