# Running this policy on your own robot

You tell the robot where to go in a sentence. A camera frame and that sentence go to a
vision-language model; a velocity command comes back. There is no map, no goal
coordinate, no waypoint list you have to author.

This directory is what makes that run on **your** hardware. You write three small classes
against your own camera and motor SDK; everything above them — the steering, the
collision guard, the stuck recovery, the latency compensation — is the same code that
produced the measured results in `nav/config/profiles/baseline.yaml`, not a
reimplementation of it.

**Read [Before you drive anything](#before-you-drive-anything) before the first `enabled:
true`.** It takes fifteen seconds and it catches the one mistake that has no error
message.

---

## What you need

| | |
|---|---|
| **A robot** | Differential drive is fine. It needs to accept a velocity command and report wheel odometry. No map, no localisation, no SLAM. |
| **One forward-facing camera** | A USB webcam works. ~448×448 is what the vision tower consumes, so resolution is not where your problems will come from. |
| **A GPU machine on the network** | ~30 GB of VRAM for the FP8 27B. It does **not** have to be the robot — this is the normal setup, and the robot only needs to reach it over HTTP. |
| *(recommended)* **Anything that measures distance** | Three ultrasonic sensors count. See [Do I need a range sensor](#do-i-need-a-range-sensor) — the honest answer is "the benchmark says no, your robot says yes". |

You do **not** need Isaac Sim, and you do not need the TIC-VLA submodule. Those are for
reproducing the benchmark. A plain `git clone` with no submodules initialised is enough
for everything on this page.

---

## The three things you write

They live in [`interfaces.py`](interfaces.py) as `typing.Protocol`s, which means **you
subclass nothing and import nothing from this repository**. A class with the right method
names *is* one of these. Write it against your own SDK, in your own package.

```python
class CameraSource:
    def grab(self) -> bytes | None: ...      # encoded JPEG/PNG bytes, or None
    def close(self) -> None: ...

class Base:
    def drive(self, vx, vy, omega) -> None:  # m/s forward, m/s left, rad/s CCW
    def stop(self) -> None: ...
    def odometry(self) -> tuple[float, float, float]:   # x_m, y_m, yaw_rad

class RangeSensor:                            # optional
    def scan(self) -> list[tuple[float, float]] | None: ...  # body-frame points
```

### A worked `Base`

The whole file. Nothing is elided.

```python
# ~/my_robot/driver.py
import math

class MyBase:
    """Wraps whatever your motor controller already speaks."""

    WHEEL_SEPARATION_M = 0.36

    def __init__(self):
        import serial
        self.port = serial.Serial("/dev/ttyUSB0", 115200, timeout=0)  # timeout=0!
        self.x = self.y = self.yaw = 0.0

    def drive(self, vx, vy, omega):
        # vy is ignored: this is a differential-drive base and the shipping
        # controller never asks for lateral motion. Accept it, don't raise.
        left  = vx - omega * self.WHEEL_SEPARATION_M / 2.0
        right = vx + omega * self.WHEEL_SEPARATION_M / 2.0
        self.port.write(f"V {left:.3f} {right:.3f}\n".encode())

    def stop(self):
        try:
            self.port.write(b"V 0 0\n")
        except Exception:
            pass          # stop() is on every exit path; it must not be able to throw

    def odometry(self):
        # However your encoders give it to you. Only DIFFERENCES are ever used,
        # so the origin is arbitrary and drift over a long run is harmless.
        return (self.x, self.y, self.yaw)
```

Then point the runner at it — your package just has to be importable:

```bash
PYTHONPATH=~/my_robot python3 robot/run_robot.py \
    --base driver:MyBase \
    --instruction "Go down the corridor and stop at the double doors."
```

Three properties of that file are load-bearing, and each one is a real failure:

- **`drive` must not block.** It is called at 20 Hz from the control loop. A driver that
  waits for the motor controller to acknowledge turns the collision guard's reaction time
  into whatever your serial link feels like today. Note `timeout=0`.
- **`stop` must not be able to throw.** It runs on shutdown, on Ctrl-C, on a policy
  error, and from the watchdog — every path where something has already gone wrong.
- **`odometry` needs the right *sign* on yaw, not the right *origin*.** Stale plans are
  rotated by the heading change since they were written, so a yaw that runs backwards
  un-rotates them twice.

### A `CameraSource`

Try [`drivers/opencv_camera.py`](drivers/opencv_camera.py) first — it covers USB, CSI and
RTSP through OpenCV and will probably work unmodified:

```bash
python3 robot/drivers/opencv_camera.py --source 0
# 640x480, 41.2 KiB JPEG -> /tmp/qvla-camera-check.jpg
```

If your camera has its own SDK (RealSense, ZED), write your own. One thing to copy from
that file regardless: **drain the camera on a background thread.** `VideoCapture.read()`
returns the *oldest* frame in the driver queue, not the newest, so reading once every 3
seconds hands the policy a view of where the robot used to be — smoothly, with
correct-looking imagery, and no error anywhere.

---

## Before you drive anything

The policy's entire output is a list of displacements in a **body-frame FLU** convention:

```
+x forward        +y LEFT        yaw counter-clockwise, radians
```

A robot whose `+y` is right drives the **mirror image of every plan it is given** —
confidently, smoothly, and into the wall on the other side. There is no exception, no
warning, and nothing in the logs looks wrong. This is the single most likely reason a
careful integration fails.

Fifteen seconds, with the robot on blocks or in an open space:

```bash
python3 - <<'EOF'
import sys, time; sys.path.insert(0, "/path/to/repo/robot")
from drivers.sim_base import PrintBase     # swap for YOUR base
base = PrintBase()
for name, cmd in [("forward", (0.15, 0, 0)), ("turn LEFT", (0, 0, 0.4))]:
    print(f">>> {name}"); t = time.time()
    while time.time() - t < 2.0:
        base.drive(*cmd); time.sleep(0.05)
    base.stop(); time.sleep(1.0)
EOF
```

**It must go forward, then turn left (counter-clockwise, seen from above).** If it turns
right, negate `omega` in your `drive()`. If "forward" is backwards, negate `vx`. Fix it in
the driver, not anywhere upstream.

---

## Bring the policy server up

On the GPU machine, from this checkout:

```bash
nav/policy_server/launch_qwen.sh
```

That reads [`nav/config/profiles/baseline.yaml`](../nav/config/profiles/baseline.yaml) —
the model, the arc set, the thinking level, the frame count, all pinned next to the scores
they earned. Don't set those by hand; a configuration that lives in two places drifts, and
this repository has produced two complete, plausible, wrong result tables that way.

Loading ~29 GB takes about 35 seconds. Check it landed:

```bash
python3 nav/tools/profile.py check
# OK    baseline @ 127.0.0.1:8766: 11 fields match, model loaded
```

That check is worth running every time. It reads `/health` back off the *running* server
and compares it field by field with the profile, which is how you learn that the process
you are talking to is not the one you think you launched.

**If the robot is a different machine**, serve on an address it can reach:

```bash
NAV_POLICY_HOST=0.0.0.0 nav/policy_server/launch_qwen.sh
```

There is no authentication on that port. Put it on a LAN you trust.

---

## Run it

### Stage 1 — no robot

Do this first, even if the robot is sitting right there. The question at this stage is
"does the camera work, does the policy answer, do the commands point where I expect", and
a robot underneath only adds its own failure modes to the answer.

```bash
python3 robot/run_robot.py \
    --policy-host 192.168.1.50 \
    --instruction "Go down the corridor and stop at the double doors."
```

The default `--base` prints what it *would* drive and moves nothing. Watch the commands
and the reasoning, and open the frames it saved in `/tmp/qvla-robot-frames/` — those are
exactly what the model saw.

### Stage 2 — your base, still held at zero

```bash
PYTHONPATH=~/my_robot python3 robot/run_robot.py \
    --base driver:MyBase --range driver:MyLidar \
    --policy-host 192.168.1.50 --instruction "..."
```

`enabled: false` in [`robot.yaml`](robot.yaml) is the default, and it means the entire
loop runs — camera, policy, guard, recovery, telemetry — with the base commanded to zero.
Push the robot around by hand and watch the odometry and the clearance move.

### Stage 3 — driving

**Have a hardware cutoff within reach.** `enabled` is a software flag; it cannot help you
when the software is what failed.

```bash
robot/tools/set_limit.py enabled true
```

No restart. The runner notices the file changed on its next control tick.

---

## Tuning it while it drives

Every limit lives in [`robot.yaml`](robot.yaml), and the runner re-reads it whenever the
file's mtime moves. That is deliberate: every number in there is one you will get wrong on
the first attempt in a real corridor, and if fixing it costs a restart — a fresh model
load, a robot walked back to its start — then in practice it does not get tuned, it gets
endured.

Three ways in, one file underneath:

```bash
# on the robot
robot/tools/set_limit.py cruise_mps 0.2
robot/tools/set_limit.py                       # show what is actually in force

# from your laptop
curl localhost:8770/status
curl -X POST localhost:8770/limits -d '{"cruise_mps": 0.2}'
curl -X POST localhost:8770/stop               # stops, and persists enabled:false

# or just edit robot.yaml in an editor while watching it drive
```

`POST /stop` writes `enabled: false` as well as ending the run, so a crash-restart cannot
bring the robot back up driving.

The ones worth touching, in the order you will reach for them:

| | |
|---|---|
| `cruise_mps` | **Start below what you think is right.** This is the one that matters. At 0.45 m/s a ~3 s generation covers 1.35 m *blind* — the robot commits to that distance before it can revise anything, and most overshoot failures live in exactly that gap. |
| `guard_stop_m` | Distance at which the base is stopped outright. Default 0.8 m, higher than the simulator's 0.6 m on purpose: a scan buffer holds one revolution, so a bearing can be ~100 ms old and the stop distance has to absorb the travel in that time. |
| `max_speed_mps` | The ceiling `cruise_mps` is clamped against. Separate so a hurried edit to the number you tune can't exceed the number you decided once, with a tape measure. |
| `chassis_radius_m` | Where your chassis ends. Returns inside it are dropped as self-hits — without this a robot that can see its own bumper guards against itself and never moves. |
| `decision_period_s` | Leave at 3.0. Shorter is not obviously better: the server serves a cached plan while it thinks, so asking more often mostly returns the same plan more often. |

`cruise_mps` is applied in two places, and the runner keeps them in sync for you: it caps
the controller locally *and* is pushed to the server, because the arc menu carries a
**direction** and not a speed. Changing only one leaves the server planning arcs at a pace
your robot never drives. The server also has a creep floor (0.20 m/s); ask for less and
the robot drives at your number while the arcs are shaped for the floor, which is the
conservative direction. It says so when it happens.

What is deliberately **not** in `robot.yaml`: the model, the thinking level, the arc set,
the frame count. Those are the measurement, they live in the profile, and a run with them
changed is a different experiment that should be labelled as one.

---

## What actually keeps the robot out of the wall

Two things, and it is worth knowing which is which.

**The policy** decides where to go, roughly every 3 seconds. It is the slow loop.

**The collision guard** runs at 20 Hz and is the only component that can stop the base
without waiting for the model to think. It reads your `RangeSensor`, casts a fan along the
direction of travel, scales the velocity down inside `guard_slow_m` and cancels it inside
`guard_stop_m`. It is the same `nav/sim/collision_guard.py` that produced every guard
number in `nav/results` — it turned out to have no Isaac coupling in scan mode, so it
ports to hardware unchanged rather than being rewritten.

Alongside it, **stuck recovery** notices the robot making no progress and reverses. It
distinguishes a *wedge* (drove into something) from a *balk* (stopped with clear floor
ahead) because those are evidence about different failures, and it tells the model which
one happened — something no single camera frame can show.

And a **watchdog** stops the base if a control tick stops arriving. That covers the case a
`finally: stop()` cannot: the loop still running but wedged — a camera driver blocked on a
USB reset, a socket read with no timeout — where the last velocity command stands and the
robot keeps going with nobody home.

### Do I need a range sensor

The measured baseline ran **without** one. On the 19-episode benchmark the lidar arm and
the no-lidar arm both scored 10/19 — five episodes won, five lost, which is noise on that
set and not a difference. The policy does not need it.

Add one anyway. What changes on hardware is not the success rate, it is the cost of being
wrong: in simulation a collision is a number in a results file. Without a range sensor the
guard has nothing to see with, and an obstacle that enters the frame between two decisions
gets hit at cruise speed. Three ultrasonic sensors satisfy the interface.

Pass `--range none` to run without one; you get [an explicit warning](drivers/null_range.py)
rather than silence.

---

## What to expect

On the 19-episode DynaNav benchmark, this configuration scores **10/19** — 9/13 indoor,
1/6 outdoor. Read those numbers for what they are: outdoor is close to unsolved, and the
noise floor on the indoor set is about ±2 episodes, so a change worth less than that is
not a change. The full table, the arms that were tried and rejected, and the caveats are
in [`nav/config/profiles/baseline.yaml`](../nav/config/profiles/baseline.yaml).

Your robot is not on that benchmark. Corridors, doorways and "go to the end and turn left"
are where this behaves like the results say. Cluttered rooms, glass, and anything needing
memory of a place it has already been are where it does not.

---

## When it doesn't work

| What you see | What it is |
|---|---|
| Robot drives the mirror image of what you asked | The FLU convention. [Run the check.](#before-you-drive-anything) |
| `policy server unreachable` | Server not up, or bound to `127.0.0.1` on a different machine. Relaunch with `NAV_POLICY_HOST=0.0.0.0`. |
| `this server predates the image-bytes route` | The server is running from an older checkout. Frames travel as bytes now because the robot and the GPU share no disk. |
| Robot drives on a view of where it used to be | Camera queue lag. Drain it on a thread; see `drivers/opencv_camera.py`. |
| Robot never moves, guard always blocked | `chassis_radius_m` too small — it is seeing its own bumper. Check `clearance_m` in `/status`. |
| It stops for no reason about once an hour | Range sensor returning stale or partial sweeps. Return `None` when a sweep isn't ready; the runner handles that. Never return `[]`, which is a positive claim that the world is clear. |
| Reasoning mentions a huge stall time | Your `odometry()` isn't moving — check the units are metres and the encoders are actually being read. |
| Model says "stopped" every decision | Look at the frames in `--frame-dir`. Usually the camera is dark, aimed at the ceiling, or auto-exposure never settled. |

`curl localhost:8770/status` gives you state, speed, clearance, guard interventions,
recoveries, and the model's own last reasoning — start there rather than in the logs.

---

## Files

| | |
|---|---|
| [`interfaces.py`](interfaces.py) | The three Protocols. Start here. |
| [`run_robot.py`](run_robot.py) | CLI. `--camera/--base/--range` take `module:attr`. |
| [`runner.py`](runner.py) | The control loop, and what it does differently from the simulator's, with the reasons. |
| [`config.py`](config.py) / [`robot.yaml`](robot.yaml) | Hot-reloaded limits. |
| [`sensing.py`](sensing.py) | Adapts your `RangeSensor` to what the guard already reads. |
| [`control_api.py`](control_api.py) | The HTTP surface behind `/status` and `/limits`. |
| [`drivers/`](drivers/) | Reference implementations. `opencv_camera.py` is the one likely to survive contact with your hardware. |
| [`tools/set_limit.py`](tools/set_limit.py) | Change one limit from the shell. |
