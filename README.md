# dash-aloha-mini-isaacsim

Simulating **AlohaMini1** (mobile base + vertical lift + two SO-101 arms) in NVIDIA
Isaac Sim 6.0.1, with full rigid-body physics, controllable from a terminal script and
from the Isaac Sim UI.

![Robot after a control sequence: lift extended, arms moved, base driven](docs/control_demo.png)

- [`plan.md`](plan.md) — phased implementation plan with checkboxes
- [`CLAUDE.md`](CLAUDE.md) — living project doc (facts, decisions, gotchas — kept current
  as work progresses)
- [`assets/upstream_alohamini1/`](assets/upstream_alohamini1/) — vendored URDF + meshes
  from [liyiteng/alohamini](https://github.com/liyiteng/alohamini) (Apache-2.0)
- [`scripts/control_terminal.py`](scripts/control_terminal.py) — terminal control (see
  Quick Start below)

Status: Phases 0-4 done and verified (import, physics, terminal control). Phase 5 (UI
control) has the underlying mechanism verified but not click-tested in an actual GUI
session — see `plan.md`/`CLAUDE.md` for exact steps to check yourself.

No ROS2 dependency by default. Isaac Sim 6.0.1 install expected at `~/isaacsim`.

## Quick start

Rebuild the scene from scratch (only needed if you change the URDF or want a clean
rebuild — `assets/usd/scene.usda` is already committed and ready to use as-is):

```bash
~/isaacsim/python.sh -m standalone_examples.api.isaacsim.asset.importer.urdf.urdf_import \
  --urdf assets/upstream_alohamini1/urdf/Aloha.urdf \
  --usd-path assets/usd \
  --ros-package "Aloha:$(pwd)/assets/upstream_alohamini1" \
  --collision-from-visuals --collision-type "Convex Decomposition" \
  --no-fix-base --merge-fixed-joints
~/isaacsim/python.sh scripts/build_scene.py
~/isaacsim/python.sh scripts/configure_physics.py
~/isaacsim/python.sh scripts/fix_wheel_collision.py
```

Control it from the terminal:

```bash
# One-shot
~/isaacsim/python.sh scripts/control_terminal.py --arm left 1 0.5 --settle 2

# Interactive
~/isaacsim/python.sh scripts/control_terminal.py --repl
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

## PS4 controller control

```bash
~/isaacsim/python.sh scripts/control_terminal.py --joystick --gui
```

Needs the `evdev` package (`~/isaacsim/python.sh -m pip install evdev`) and your user
in the `input` group — check with `groups | grep input`; if it's not listed:

```bash
sudo usermod -aG input $USER   # then log out and back in
```

Mapping: **L1**=control right arm, **L2**=control left arm, **L1+L2 together**=control
both arms mirrored (opposite movement), **R2**=control the base. Left stick and right
stick move different joints/axes depending on which mode is active — see the full
mapping table in `scripts/control_terminal.py`'s module docstring.

**This has not been tested against a physical controller** — none was connected in the
environment it was built in. It's implemented against the standard Linux `evdev` codes
for a DualShock 4, but exact button/axis codes can vary by driver. Run
`scripts/control_terminal.py --joystick-debug` first to print raw events from your
controller and confirm they match `JOYSTICK_MAP` at the top of the joystick section —
adjust the numbers there if your controller reports different codes.
