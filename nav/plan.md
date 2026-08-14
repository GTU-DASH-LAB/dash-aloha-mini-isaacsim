# Plan: language-driven navigation for AlohaMini, using TIC-VLA + DynaNav

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/needs decision

This is the plan for the `nav/` subsystem. For the base simulation work (import,
physics, terminal control, cameras) see the repo-root [`plan.md`](../plan.md).

## Goal

Type an English instruction into a web UI — e.g. *"Go straight ahead through the
hallway behind the black glass doors, then turn left at the end to approach the red
emergency exit door."* — and have **AlohaMini drive itself** through a
[DynaNav](https://github.com/ucla-mobility/TIC-VLA) benchmark scene in Isaac Sim
6.0.1, using the pretrained **TIC-VLA** policy as the brain.

The UI ships the four DynaNav smoke-episode prompts as one-click examples, so there is
always a known-good instruction to start from.

---

## Three findings that shape the whole design

Everything below follows from facts verified on this machine on 2026-08-14, not from
assumptions. Read this section before touching the code — each item killed a simpler
design that would otherwise look obviously correct.

### 1. TIC-VLA and AlohaMini **cannot share a process**

| | TIC-VLA / DynaNav | AlohaMini |
|---|---|---|
| Isaac Sim | **5.0.0** (NumPy 1.x ABI) | **6.0.1** |
| Python | **3.11** | **3.12** |
| venv | `~/envs/tic-vla` | Isaac Sim 6.0.1's own `python.sh` |

Isaac Sim's major version hard-pins the Python version (`memory/isaac-python-version-pairing.md`),
and `~/robotics/README.md` states the rule plainly: **never merge the two environments.**
So the policy cannot simply be imported into the sim script.

There is a second, independent reason, and it is measured rather than theoretical
(`memory/gtu-workstation-gpu-asymmetry.md`): the DynaNav **office scene alone leaves
Isaac Sim holding 25.42 GiB of GPU0's 31.35 GiB**, and loading the 1.9 GB TIC-VLA
checkpoint onto that same card dies with `CUDA out of memory`. *The scene is what is
large, not the model.* Even at matching Python versions, one process would not fit.

**Therefore: two processes, split across both GPUs.**

```
   ┌──────────────────────────────────────────────────────────┐
   │  Browser — prompt box + DynaNav example prompts          │
   └───────────────────────────┬──────────────────────────────┘
                               │ HTTP + MJPEG
   ┌───────────────────────────▼──────────────────────────────┐
   │  SIM PROCESS   ~/isaacsim/python.sh   py3.12   GPU0      │
   │    nav/ui/server.py       FastAPI, background thread     │
   │    nav/sim/run_navigation.py   Isaac Sim main loop       │
   │      scene = DynaNav env USD + AlohaMini                 │
   │      forward camera ──► PNG in a scratch dir             │
   │      waypoints ──► (vx,vy,ω) ──► body_to_wheel_speeds    │
   └───────────────────────────┬──────────────────────────────┘
                               │ HTTP  (frame paths + state → waypoints)
   ┌───────────────────────────▼──────────────────────────────┐
   │  POLICY SERVER   ~/envs/tic-vla/bin/python  py3.11  GPU1 │
   │    nav/policy_server/server.py                           │
   │      TICVLA(InternVL3-1B + TIC-VLA-model.ckpt)           │
   │      predict(images, instruction, state) → (T,2) waypts  │
   └──────────────────────────────────────────────────────────┘
```

Frames cross the boundary as **file paths, not bytes**: both processes are on one
machine, and `TICVLA.predict()` already takes `image_paths: list[str]`
(`DynaNav/ticvla.py:548`). Writing a PNG to a scratch dir and passing the path avoids
base64 entirely. HTTP is chosen over ZeroMQ because the payload is a small JSON blob,
localhost adds ~1 ms against ~0.5–2 s of inference, and `curl` can debug it.

### 2. AlohaMini's base is **kinematically teleported** — it will drive through walls

`scripts/control/control_terminal.py:434-486` is explicit about this. Real wheel-ground
contact was investigated and abandoned: even a single wheel spinning in isolation
produced ~200× less translation than expected, so the base is driven by
`art.set_world_poses()` every physics step while the wheel joints spin at the
visually-correct rate.

For terminal control that is a fine, documented trade-off. **For a navigation benchmark
it is disqualifying**: a teleported root has no collision response, so the robot ghosts
through walls and any collision-rate metric reads a meaningless 0%.

This is not hypothetical — DynaNav's own metrics are `total_collision_rate_percent`,
`human_collision_rate_percent`, and `spl`. Two of the three are nonsense on a
teleporting base.

**Therefore: Phase 11 adds a collision guard** — a PhysX scene query along the intended
motion vector *before* the teleport is applied. On a hit, the step is refused and the
episode records a real collision. This buys meaningful metrics without re-opening the
wheel-physics problem that already consumed a debugging round.

### 3. TIC-VLA has a **straight-line bias** — "intelligent" must be proven, not assumed

From this machine's own `DynaNav/benchmark_results/`:

| run | episode | path (m) | final error (m) | ok |
|---|---|---|---|---|
| 20260813_090804 | full set, 49 episodes | — | — | **27/49 = 55%** |
| 20260813_082927 | hospital *"go straight ahead…"* | 12.1 | **1.49** | ✅ |
| 20260813_082927 | office *"…then turn left at the end"* | 34.5 | **40.03** | ❌ |
| 20260813_120602 | outdoor | 68.4 | **83.76** | ❌ |
| 20260813_082927 | warehouse | 44.1 | 31.02 | ❌ |

The office episode **starts 13.74 m from the goal and ends 40 m away** after driving
34.5 m. The robot is not failing to move — it is moving confidently in the wrong
direction. It succeeds on `hospital`, whose instruction is literally *"Go straight
ahead and stop at the front of the vending machine"*, and fails the episodes that
require a turn.

So a demo that drives AlohaMini forward and stops near a goal **proves nothing** —
a policy that ignores the prompt entirely would produce the same video.

**Therefore: Phase 9 is a policy-sanity harness that runs before any closed-loop work**,
feeding one fixed frame with several contradictory instructions ("turn left" / "turn
right" / "stop") and asserting the waypoints actually differ. If they do not, the
integration is still worth building, but the honest claim becomes "AlohaMini executes
TIC-VLA's waypoints correctly", not "AlohaMini navigates intelligently". That
distinction is the difference between a thesis result and a demo.

---

## What the policy actually emits

Verified by reading `DynaNav/ticvla.py`, not inferred:

- `TICVLA.predict(image_paths, instruction, robot_state, …)` → `(response_text,
  waypoints, gen_start_step, kv_cache_available, gen_start_pose)` (`ticvla.py:548`)
- `waypoints` is `(1, T, 2)` — `action_dim=2` (`ticvla.py:80`), `num_action_chunks=30`
  by default (`ticvla.py:213`), overridable via `TICVLA_NUM_ACTION_CHUNKS`
- The pairs are **(dx, dy) displacements in the robot's body frame, FLU convention**
  (`behavior/nova_carter_test_ticvla.py:1369`)
- `robot_state` is 5 or 6 floats; the 6-form is unpacked as
  `[vx, vy, _, yaw_speed, dx, dy]` (`ticvla.py:626-637`)
- `response_text` is the VLM's free-text reasoning — worth surfacing in the UI, it is
  the most direct window into whether the model understood the prompt

## Controller: AlohaMini is holonomic, Nova Carter is not

DynaNav drives a **Nova Carter — differential drive**, so
`behavior/nova_carter_test_ticvla.py:1370-1416` collapses the (dx,dy) waypoints to
`(v, ω)` with pure pursuit and a yaw filter. It *has* to: a differential base cannot
strafe.

AlohaMini is a **LeKiwi 3-wheel omni base** — `body_to_wheel_speeds(vx, vy, omega)`
in `scripts/alohamini1_specs.py:105` is holonomic. It can track a (dx,dy) path
directly.

Both are implemented, selectable with `--controller`:

- **`pursuit`** (default) — a faithful port of the Nova Carter logic, `vy` forced to 0.
  Keeps motion inside TIC-VLA's training distribution, since the policy only ever saw
  non-holonomic robots. This is the scientifically defensible default.
- **`holonomic`** — tracks the lookahead point with `(vx, vy)` and controls yaw
  separately. Strictly better path-following on paper, but **out of distribution**: the
  policy's next observation comes from a viewpoint it would never have produced itself.
  Interesting to measure, not to assume.

Comparing the two on identical episodes is a real result and costs one flag.

---

## Phases

### Phase 8 — Spikes: prove the two risky assumptions before building on them
- [x] `nav/tools/check_scene_compat.py` — load DynaNav's `office.usd` (authored for
      Isaac Sim **5.0**) in Isaac Sim **6.0.1** headless
- [x] Check scale against `CLAUDE.md`'s ~1000 m warning
- [x] Stand up the policy server and confirm a `(T,2)` waypoint array comes back

**Result — the version gap is NOT a blocker.** `office.usd` opens cleanly in 6.0.1:

| | |
|---|---|
| opened | ✅ `True`, 188 s (CDN streaming) |
| prims / meshes | 4665 / 140 |
| invalid prims | **0** |
| unloaded payloads | **0** — every reference resolved |
| up axis / units | Z / 1.0 m per unit |

**But it walked straight into this repo's known scale trap.** The measured extent is
**1063.55 × 677.15 m**, and the stage references
`Assets/Isaac/4.5/Isaac/Environments/Office/` — *the same NVIDIA Office asset*
`../CLAUDE.md` says "had to be replaced — it's real-world building scale, ~1000 m
across, which caused both a camera-framing bug and a physics explosion at spawn."

This is survivable, because DynaNav itself drives Nova Carter through this scene
successfully. The extent is dominated by distant scenery; the actual episode runs from
`(-10.6, -12.1)` to `(-7.68, -25.53)`, about 13 m. So the two historical failures are
addressable rather than fundamental, and Phase 11 must handle both explicitly:
- **spawn** the robot at DynaNav's episode coordinates, never at the origin
- **author the camera** explicitly instead of relying on a frame-all default

**Policy server:** loads in **2.9 s** on physical GPU1 (`CUDA_VISIBLE_DEVICES=1`),
checkpoint accepted with `strict=True`, `num_action_chunks=30`, `/predict` round-trip
**1.0–1.5 s** per call.

### Phase 9 — Policy sanity (the gate; see finding 3)
- [x] `nav/tools/check_policy_sanity.py` — fixed frame, contradictory instructions,
      repeats to separate language response from sampling noise
- [x] Report per-instruction mean heading and path length
- [ ] Repeat with a frame from AlohaMini's own forward camera (different mounting
      height and FOV to Nova Carter — a real distribution shift). **Blocked on Phase 11**
- [x] Write the verdict into this file, whatever it is

**Verdict: PASS — TIC-VLA is genuinely language-conditioned.** Run on real
`front_frame_*.jpg` captures from this machine's own outdoor run:

| instruction | frame 000780 | frame 001167 |
|---|---|---|
| "Go straight ahead down the hallway." | −1.81°, len 1.52 | +0.16°, len 1.29 |
| "Turn **left** immediately…" | −0.95°, len 1.38 | **+14.35°**, len 1.19 |
| "Turn **right** immediately…" | **−23.80°**, len 1.24 | **−4.59°**, len 1.36 |
| "Stop. Do not move." | −164.98°, **len 0.38** | +169.24°, **len 0.43** |

Headings are FLU, so **+ is left and − is right — both frames are correctly signed**,
and "stop" collapses path length to ~0.4 against ~1.3 for the others. The instruction
changes the output far beyond sampling noise.

Two caveats that matter for later phases:

1. **Within-instruction spread is exactly 0.000°** across repeats. `predict()` sets
   `do_sample=True` but with `temperature=0.1, top_k=10` (`ticvla.py:585`), which is
   effectively greedy. Do not expect sampling to break the policy out of a stuck state
   — retrying an identical observation returns an identical plan.
2. **Turn authority is modest: ~±14° at the endpoint of a 30-chunk plan.** On frame
   000780 the left turn was only −0.95° while the right turn was −23.80°; the asymmetry
   did *not* reproduce on 001167, so it is frame-specific rather than a systematic
   left-turn failure — but it does show the turn signal is weak and scene-dependent.
   This is the most likely explanation for the office benchmark failure: not that the
   policy ignores "turn left at the end", but that ±14° per plan is too little authority
   to commit to a decisive turn. Worth testing directly, since it is a thesis-relevant
   result about the checkpoint rather than about AlohaMini.

### Phase 10 — Policy server (py3.11, GPU1)
- [ ] `nav/policy_server/server.py` — FastAPI; load `TICVLA` once at startup
- [ ] `POST /predict` → waypoints + reasoning text; `POST /reset` →
      `reset_episode_state()`; `GET /health` → device, dtype, chunk count
- [ ] `nav/policy_server/launch.sh` — sources nothing from Isaac; pins
      `CUDA_VISIBLE_DEVICES=1` (safe here: this process never starts Kit — see
      `memory/gtu-workstation-gpu-asymmetry.md`)
- [ ] `nav/policy_server/client.py` — thin client for the sim side, stdlib-only so it
      imports cleanly under Isaac Sim's Python
- [ ] Commit + push

### Phase 11 — Sim side: scene, camera, controller, collision guard (GPU0)
- [x] `nav/sim/episode.py` — episode config + success test + path metrics
- [x] `nav/config/episodes.yaml` — DynaNav's four episodes, prompts verbatim
- [x] `nav/sim/build_nav_scene.{py,sh}` — compose environment + robot (NOT in the plan
      originally; turned out to be required, see below)
- [x] `nav/sim/camera_source.py` — render the nav camera to a frame file
- [x] `nav/sim/controllers.py` — `PursuitController` + `HolonomicController`
- [x] `nav/sim/collision_guard.py` — PhysX ray fan, refuse the step on a hit
- [x] `nav/sim/base_drive.py` — velocity → pose (also not in the original plan)
- [x] `nav/sim/run_navigation.py` — the loop
- [x] Commit + push after each

**Four things the plan got wrong, all found by building it:**

1. **The `forward` camera is the wrong camera.** All three LeRobot cameras face the
   manipulation front (−Y); the base drives +X. The plan said "grab the `forward`
   camera each step", which would have fed the policy the view 90° off its direction
   of travel. Added a fourth camera, `camera_nav` (`rotateXYZ=(80,0,-90)` → view
   `(0.985, 0, -0.174)`), deliberately kept OUT of `CAMERA_PRIM_PATHS` so the LeRobot
   observation contract is unchanged.
2. **A nav scene has to be built, not just opened.** `scene.usda` is the pick-and-place
   setup and regenerating it is a documented footgun. A freshly authored layer has no
   joint drives, no wheel colliders and no cameras — those are overrides in the *scene*
   file, not in `Aloha.usda`.
3. **Nothing after `SimulationApp.close()` executes.** The first builder authored the
   layer, reported success, and applied none of the pipeline. Hence the `.sh` wrapper.
4. **Episode timeouts must count sim time, not wall clock.** Inference is synchronous
   and blocks ~1.0–1.5 s per call. Timing by wall clock spends most of a 70 s budget on
   the policy thinking and cuts the episode to roughly a third of its intended length —
   which would have looked like a navigation failure and been a stopwatch bug.

### Phase 12 — Web UI
- [x] `nav/ui/ui_bridge.py` — uvicorn on a worker thread of the sim process. It cannot
      be a second process (it needs live access to the running stage) and cannot own
      the main thread (Kit must). It only pushes jobs onto a queue and reads a status
      snapshot; the main loop executes them.
- [x] `nav/ui/static/index.html` — prompt box, **the four DynaNav prompts as clickable
      examples**, Run/Stop, live camera, controller selector. Zero external assets:
      this machine's onboard ethernet is dead, and a CDN font would make the UI look
      broken for reasons unrelated to the robot.
- [x] Surfaces distance-to-goal, sim + wall elapsed, policy calls, guard stops, and the
      VLM's reasoning text. Progress bar shows the *fraction of the initial gap closed*,
      not distance remaining.
- [ ] Top-down waypoint plot — deferred, not required for the deliverable
- [x] Commit + push

### Phase 13 — Run it, measure it, write it down
- [x] First full end-to-end run (warehouse, holonomic)
- [ ] Remaining episodes × both controllers
- [ ] Compare against DynaNav's own Nova Carter numbers
- [ ] Update `../CLAUDE.md` with every gotcha hit

**Run 1 — warehouse, holonomic, the benchmark's own prompt:**

```
[TIMEOUT] warehouse (holonomic)
  distance to goal : 26.32 m -> 14.89 m  (closed +11.43 m)
  path travelled   : 50.85 m in 70.0 s sim (324.3 s wall, 4201 steps, 141 policy calls)
  guard stops      : 0
```

Read this carefully, because "TIMEOUT" undersells it. The robot **spawns facing almost
exactly away from the goal** — DynaNav's start yaw is −141° and the goal bearing is
+52°, a 193° turn — and the first captured frame is a bare wall corner. It turned
around, drove out, and frame 140 shows it **inside an aisle between loaded pallet
racks**. That is the task working: it read the instruction and went looking for aisles.

It did not get there efficiently: 50.85 m of path to close 11.43 m of gap. Consistent
with Phase 9's finding that turn authority is only ~±14° per plan — enough to curve,
not enough to commit.

**Run 2 — same episode, started from the UI by clicking Run:**

```
[TIMEOUT] warehouse (holonomic)
  distance to goal : 26.32 m -> 12.38 m  (closed +13.94 m)
  path travelled   : 50.93 m in 70.0 s sim (329.4 s wall, 4201 steps, 141 policy calls)
  guard stops      : 0
```

Confirms the whole chain works from the browser: prompt in, robot drives, camera and
status stream back live. It also confirms the run is **near-deterministic** — same
prompt, same 4201 steps, same 141 calls, path length within 0.08 m of run 1 — exactly
what Phase 9's 0.000 within-instruction spread predicted.

The mid-run reasoning is the most convincing artifact the stack produces:

> *"I am in a warehouse with shelving and a wall on my right and a clear path. The
> critical obstacle is the right wall and cones, which are stationary... Next I
> continue forward, maintain clearance, **align to Aisle 5, and stop at its
> entrance**."*

That is the benchmark instruction being grounded in what the camera sees, not a canned
response.

**But the failure mode is now clear, and it is the one Phase 9 predicted.** Watching
distance-to-goal over run 2: 26.32 → 22.19 (6.5 s) → 21.51 (19 s) → **24.63 (27 s)** →
12.38 (70 s). It closes ground fast, then *loses* ground, and a frame grabbed at the
27 s mark shows the robot **nose-on to a blank wall**. It is not blocked — the guard
never fired, the wall was still metres away — it is mis-aimed and cannot turn hard
enough to recover, because turn authority is ~±14° per plan. It eventually curves
around and resumes, which is why the final number is still the best of the two runs.

This is the same shape as DynaNav's own published office failure (started 13.74 m from
the goal, finished 40 m away) and it now has a mechanism rather than a shrug:
**the policy points the right way but cannot commit to a turn.** Testable next steps,
in order of cheapness: raise `max_yaw_rate_radps`; shorten the pursuit lookahead so the
controller tracks the near part of the plan more aggressively; scale the emitted
heading rather than tracking it literally.

**Two real issues this surfaced, both recorded rather than papered over:**

- **Actual speed exceeds the commanded limit.** 50.85 m / 70.02 s = **0.726 m/s**
  against a `max_speed_mps` of 0.6. Most likely the kinematic teleport and the wheel
  spin *add*: `base_drive.apply()` teleports by `v·dt` **and** commands the wheel
  velocity targets, and while wheel traction is poor (`../CLAUDE.md`) it is not zero.
  Not yet confirmed — the honest statement is that there is ~20% unexplained
  translation. Test by zeroing the wheel targets and re-measuring.
- **Zero guard interventions needed checking, not celebrating.** 50 m through racking
  with no stop is equally what a broken raycast looks like.
  `nav/tools/check_collision_guard.py` sweeps the ray fan through a full circle:
  **9/16 bearings hit** real walls and pillars at 6.8–12.5 m. The guard is live; the
  robot genuinely kept its distance.

### Phase 14 — Watching the robot, and starting over (requested after run 2)
- [x] Third-person chase camera (`camera_chase`), parented to `base_link` in USD so it
      inherits the base's position **and yaw** — a fixed world camera would be a camera
      the robot drives away from. `base_link` rather than `vertical_link` on purpose:
      mounted on the column it would ride up and down with the lift.
- [x] Two live views side by side in the UI — nav (what the policy sees) and chase
      (what a human wants to see). A checkbox opens the second one.
- [x] **Created lazily, off by default.** Once an Isaac Sim `Camera` exists, Kit renders
      it every frame whether or not anyone reads it, so a headless benchmark run should
      not pay for a view nobody is watching. `try_create_chase()` returns `None` on a
      scene built before the camera existed, and the UI says to rebuild rather than
      throwing out of the render pipeline mid-episode.
- [x] Chase view refreshes every 15 physics steps (~4 Hz of sim time). Tying it to the
      replan cadence instead would give one frame per ~2.5 s of wall clock — a
      slideshow, not something you can watch the robot navigate in.
- [x] Reset button: `request_reset()` (UI thread, sets a flag) → `perform_reset()` (main
      thread, does the work). Touching the articulation from the UI thread races PhysX.
- [x] Reset aborts a run in progress, drains the job queue, zeroes the wheel velocity
      targets, syncs the integrated yaw to the teleported yaw, and clears the policy
      server's KV cache. Each of those is a bug if skipped: a stale queued job fires
      immediately after, the wheels keep spinning on the start line, the first drive step
      goes in the pre-reset heading, and the "fresh" run carries the failed attempt's
      context.
- [x] Between runs both panels go live (throttled to ~3 Hz). During a run the nav panel
      deliberately freezes on the frame the policy was actually given, so it lines up
      with the reasoning text beside it — but leaving it black while idle, next to a
      moving chase view, read as a broken UI rather than an idle one.
- [x] The chase toggle is restored from the runner on page load (`chase_enabled` in the
      status snapshot). The sim outlives the browser tab; without this, a reload left the
      checkbox off while the camera was still allocated and rendering every frame.
- [x] Verified: `docs/nav_ui_dual_view.png` (both views live, Reset present),
      `docs/nav_chase_midrun.jpg` (chase view mid-run, robot out on the warehouse floor
      with the pallet racking ahead). Reset checked end-to-end — aborted a run at 25.04 m,
      returned to idle, and the next run started from 26.32 m again, the same figure runs
      1 and 2 began at, with the chase view pixel-matching the pre-run frame.

**Gotcha that cost a restart:** the `ChaseRequest` pydantic model was first defined
*inside* `build_app()`. This file uses `from __future__ import annotations`, so FastAPI
resolves the annotation string with `get_type_hints()` against the function's **module**
globals — where a locally-defined class does not exist. It does not raise; FastAPI
silently demotes the parameter to a query string and every POST answers
`{"loc": ["query", "req"], "msg": "Field required"}`. Request models stay at module
level.

---

### Phase 15 — "it goes in the wrong direction and can't see Aisle 5" (user report, run 3)

Two complaints in one sentence, and they turned out to be two separate bugs. Both were
mine; neither was the policy.

**Bug 1 — the episodes were the wrong episodes.**

`nav/config/episodes.yaml` was built from `TIC-VLA/DynaNav/configs/benchmark_example.yaml`.
That file is a four-episode **smoke/demo** config, not the benchmark. Three of its four
episodes spawn the robot facing roughly 180° away from their own goal:

| episode | start_yaw | goal bearing | off by |
|---|---|---|---|
| hospital_smoke | 0.0 | +15.5 | +15.5 |
| office_smoke | 90.0 | −77.7 | **−167.7** |
| outdoor_smoke | −123.0 | +57.0 | **−180.0** |
| warehouse_smoke | −141.0 | +52.3 | **−166.7** |

- [x] Ruled out the obvious explanation first: a yaw-convention mismatch on our side.
      DynaNav applies `start_yaw` as a plain `rotate_z_op.Set(start_yaw)` on the robot
      prim (`benchmark.py:_spawn_robot`), with robot-forward = +X — byte for byte what
      `sim/base_drive.py:reset_to()` does. Checked against the real 85-episode
      `benchmark_full.yaml`: 54 episodes start within 30° of their goal bearing, 29
      within 30–90°, 2 within 90–150°, **none** beyond 150°, and **not one is improved
      by a 180° flip**. The convention is right. The smoke episodes are simply not aimed
      at anything.
- [x] Rewrote `episodes.yaml` from `benchmark_full.yaml`. Five real episodes, each
      annotated with its own off-bearing, all now within 30°:

      episode              yaw  bearing  off by   dist
      warehouse             90     65.1   -24.9   16.5 m   (episode_61, the Aisle 5 task)
      warehouse_aisle6      59     64.0     5.0   34.3 m   (episode_76)
      hospital             -90    -96.4    -6.4   32.5 m   (episode_6)
      office              -132   -134.3    -2.3   15.1 m   (episode_26)
      outdoor              160    161.9     1.9   18.4 m   (episode_51)

- [x] Header warning in the file so the next person does not "helpfully" copy the demo
      config back in. Running a smoke episode looks *exactly* like a broken policy — the
      robot drives off the wrong way and never sees the landmark the prompt names,
      because the landmark is behind it.

**What I got wrong before, for the record:** run 2's 193° opening turn is written up
earlier in this plan as the task working — "it read the instruction and went looking for
aisles." It was not. It was the robot turning around because the goal was behind it.
A plausible story fitted to a bad number is worse than no story, because it closes the
question.

**Bug 2 — the nav camera was AlohaMini's, not the one the policy was trained through.**

A camera is part of a VLA's input distribution, not a free design choice. TIC-VLA is fed
DynaNav's render of the Nova Carter asset's front Hawk stereo left eye. Probed straight
off `nova_carter_sensors.usd` (`/chassis_link/front_hawk/left/camera_left`) rather than
guessed:

| | ours (before) | Hawk (now) |
|---|---|---|
| HFOV | 77.8° | **90.1°** |
| height | 1.15 m (on the lift column) | **0.346 m** (on `base_link`) |
| pitch | 10° down | **0.0°, level** |
| render aspect | 640×480, 4:3 | **1920×1080, 16:9** |

- [x] All four now match. Every one of them crops or shifts exactly the peripheral and
      distant context that "the second aisle from the right" is *expressed in* — a
      90° view rendered at 78° and pitched into the floor cannot contain a phrase about
      the far end of a row of aisles.
- [x] Moved to `base_link`. On the lift column, the nav camera's horizon moves whenever
      the lift moves — for reasons the policy has no way to account for.
- [x] Mounted at x=+0.25 to clear the base's own front face (half-extent ~0.21 in X).
- [x] `add_cameras.py` now **deletes any camera of ours found off its spec'd path**
      before authoring. Defining `camera_nav` at its new path does not remove the old
      one; a plain re-run left both prims in the stage, with Kit rendering the stale
      lift-mounted one every frame. Moving a mount point is now a one-line spec change.
- [x] Verified by rendering and looking, per this repo's own rule. The start frame now
      has all six aisle placards legible and the three Aisle 05 traffic cones in shot:
      `docs/nav_start_frame_hawk.jpg`, `docs/nav_start_aisle05_zoom.jpg`.

**Bug 3 — found while verifying the first two: the policy was never told what it had
already done.**

With the camera and the start pose both correct, the run still failed, and the way it
failed was diagnostic: 32.04 m of path travelled to close **−0.40 m** of distance. Every
plan came back a fresh ~1.3 m straight line, so the robot sailed past the aisle it was
sent to and parked in the perimeter wall — while its own reasoning text said "facing a
wall". It could see correctly and still had no idea it had been driving straight for a
minute.

`predict()` was being called with `previous_waypoints_text=""` — the default — on all
141 calls of a run. That parameter is TIC-VLA's temporal channel, and it is not
optional decoration:

- DynaNav builds it every inference and raises
  `ValueError("ERROR: Empty previous waypoints text!")` if it comes out blank. The
  reference implementation treats the value we were sending as an impossible state.
- Training **always** emitted one of two strings, never nothing. With no history yet it
  still said "No waypoints available." — so the empty string is not the zero case, that
  line is.

- [x] `sim/waypoint_history.py`. Format copied character for character from
      `ticvla/data/vlm_data.py:_format_previous_waypoints_text`, because that is the
      function that produced the training text — a reworded sentence still reads fine to
      a human while sitting off distribution.
- [x] Sampling ported from `nova_carter_test_ticvla.py:644-665`: one sample per second of
      sim time, each the displacement over the previous second expressed in the body
      frame **as it was at the start of that second** (`R_prev.T @ delta_world`), not the
      current frame and not world. First sample is a (0,0,0) placeholder, filtered at
      format time.
- [x] Measured, same episode, same seed, only this changed:

      | | before | after |
      |---|---|---|
      | distance closed | **−0.40 m** | **+8.73 m** |
      | final distance | 16.91 m | 7.78 m |
      | path travelled | 32.04 m | 15.80 m |

      Half the path length for a result that goes from backwards to two-thirds of the
      way there. It stopped wandering, and its closing reasoning is "at the entrance of
      Aisle 05 … begun entering Aisle 05" rather than "facing a wall".

**Still TIMEOUT, and the remaining cause is now a different one.** The robot reaches the
Aisle 05 entrance and wedges on the rack endcap: 2734 of 4201 steps (65%) were guard
stops, against 723 before. `collision_guard.py` fans ±35° over 7 rays and stops at
0.6 m, so when the robot hugs an endcap corner the outer rays sit inside the racking and
translation is cancelled for most of the run while the centre path is clear.

**Bug 4 — the guard blocked speed when it should have blocked directions.**

Called guard *tuning* above. It was not; it was structural, and tuning it would have hidden
the defect. The old guard scaled the entire velocity to zero whenever any ray came back
short — including a ray aimed 35° off the direction of travel. That makes it strictly
less physical than the contact response it substitutes for: a real wall stops you along
its normal and lets you slide along it.

The deadlock follows: hugging an endcap, one outer ray sits in the racking → translation
0; the plan says "straight ahead" → heading error ~0 → the controller's
`omega = yaw_align * heading_error` ~0. Neither driving nor turning, beside an aisle it
could have driven into.

- [x] Project, don't scale. For each blocking ray bearing, remove the component of world
      velocity pointing *into* it and keep the remainder, capped at half speed:

      for angle in blockers:
          approach = wx*cos(angle) + wy*sin(angle)
          if approach > 0:                # only cancel motion TOWARD the hit
              wx -= approach*cos(angle); wy -= approach*sin(angle)

- [x] Per blocker, not against an averaged bearing — in a corner each hit has to remove
      its own component, and averaging leaves a velocity still driving into one of them.
      Successive projection converges to ~0 for a genuine dead end, which is correct there.
- [x] Same fan, same 0.6 m stop distance, no threshold widened. Interventions:
      **2734 → 32 → 0 → 80** across the runs after. The freezing is gone.

**Bug 5 — TIC-VLA is a video model and we were sending one still.**

Its system prompt says it verbatim: "a video consisting of visual observations, including
historical and current frames" (`ticvla/data/vlm_data.py:_build_messages`). DynaNav sends
**four** frames at 3 s spacing, oldest first
(`nova_carter_test_ticvla.py:_get_sampled_image_paths`).

- [x] `sim/frame_history.py` samples `[-9s, -6s, -3s, now]`, with DynaNav's edge cases:
      skip an offset the history cannot reach, always keep the current frame, dedupe
      preserving order (a young history collapses to one frame — sending the same JPEG
      four times is not a video, it is one frame plus a claim about motion that did not
      happen).
- [x] Also fixed `robot_state`'s `dx, dy`. DynaNav's is displacement since the VLM
      generation *started*, because it infers on a background thread while the robot keeps
      driving. Ours is synchronous — the robot is frozen for the whole call — so the honest
      value is 0.0, not the one-physics-step (~1 cm) displacement that used to go there.

- [x] `EpisodeResult.save()` writes every run to `nav/results/*.json` (gitignored), trace
      included, `(x, y, yaw)` per sample. Runs were being compared on final distance
      alone, which cannot distinguish a robot that drove to the wrong place from one that
      drove nowhere — opposite fixes. This is what produced the diagnosis below.

**Where it actually stands, measured (warehouse, holonomic, 70 s):**

| run | change | initial → final | closed | path | guard |
|---|---|---|---|---|---|
| 1 | camera + episodes fixed | 16.51 → 16.91 | −0.40 | 32.04 m | 723 |
| 2 | + waypoint history | 16.51 → 7.78 | **+8.73** | 15.80 m | 2734 |
| 3 | + directional guard | 16.51 → 28.80 | −12.30 | 46.98 m | 32 |
| 4 | + trace saving | 16.51 → 28.77 | −12.26 | 52.53 m | 0 |
| 5 | + 4-frame video | 16.51 → 28.86 | −12.35 | 48.34 m | 80 |

Run 2's good number was **the broken guard freezing the robot next to the aisle**, not
navigation. Removing the freeze made the number worse and the behaviour honest — which is
the right trade, but it has to be said out loud rather than being read as a regression.

**The trace names the remaining failure exactly.** Goal `(-3.26, 7.61)`; the robot starts
at `(-10.22, -7.36)` facing 90° (north) and needs to end up ~7 m **east**:

    t= 0.0s  (-10.22, -7.36)  yaw  90.0   d=16.51
    t=17.4s  (-10.14,  5.79)  yaw  93.9   d= 7.12   <- level with the goal, 7 m west of it
    t=20.9s  (-10.46,  7.92)  yaw 105.9   d= 7.20
    t=34.8s  (-11.95, 14.54)  yaw  90.7   d=11.11
    t=69.5s  (-25.50, 26.00)  yaw 240.9   d=28.86

It holds yaw ~88–94° for the first 20 s, drives dead straight past the aisle mouth at a
closest approach of **6.98 m** against a 1.5 m success threshold, and never turns east.

- [x] Ruled out an integration bug by probing the policy directly on the start frame.
      It responds correctly to directional language, so the plumbing is fine:

      | instruction | mean heading of returned waypoints |
      |---|---|
      | "Turn right and…" | **−9.1°** |
      | "Turn left and…" | **+14.3°** |
      | "Go straight ahead" | +2.1° |
      | "Stop. Do not move." | reach collapses to 0.21 m |
      | the benchmark instruction | **+1.2°** (straight) |

      From this viewpoint the model genuinely chooses straight. That is a model/viewpoint
      question, not a wiring one.

---

## Phase 16 — the controller was losing the turn (and what was left when it stopped)

Prompted by an observation from the bench: *"it goes between 4 and 5 to the end, maybe
because the controller is different and it needs theta rather than dx dy."* Half right,
and the half that was right was worth two runs.

**Bug 6 — `HolonomicController` converts turn intent into strafe, and breaks the loop.**

TIC-VLA expresses "the target is off to your right" as a small **lateral offset**, because
on the differential-drive Nova Carter it was trained on, that is the only thing a lateral
offset *can* mean: you satisfy it by rotating. `HolonomicController` has an omni base, so
it satisfies the identical offset by **translating**. The offset is discharged as sideways
drift, the heading error is driven back to ~0 before the next replan, and the camera never
rotates much — so the next frame looks nearly the same, the policy asks for the same
small offset again, and a loop meant to converge crawls.

`controllers.py` *predicted this in its own docstring* ("full holonomy is dangerous for a
vision-language policy") and then assumed `yaw_align=0.8` defused it. Nobody measured that
assumption for five runs.

- [x] Measured, same episode, same policy, controller the only variable:

      controller   rightward yaw by t=21 s   closest approach
      holonomic     4.9° / 2.1°              6.06 / 6.98 m
      pursuit      11.8° / 12.5°             5.77 / 6.39 m

      Four full runs, all on the benchmark Aisle-05 instruction. Yaw is measured over
      the approach window (t=0-21 s) only -- what accumulates after the robot passes
      the aisle mouth is wandering, not steering. Pursuit turns ~2.7x as far (mean
      12.2° vs 3.5°) and consistently; one holonomic run turned the WRONG way, to
      105.9°. The closest-approach gain is modest (mean 6.08 vs 6.52 m), which is the
      honest shape of the result: the deficit is 25° and the controller is worth ~9.

- [x] Static comparison on identical plans: pursuit gives ~1.7× the yaw rate (1.7 vs
      1.0 °/s at a 1.2° plan; 27.4 vs 16.0 °/s at 20°).
- [x] `pursuit` is now the default in `run.sh`, `run_navigation.py` and the UI dropdown.
      The omni base's lateral DOF is still used by `collision_guard.py` for sliding along
      obstacles, where there is no policy in the loop to confuse.

**Bug 7 — we were logging outcome without intent.**

Five runs were argued about on the basis of where the robot ended up. That cannot separate
a robot obeying a straight plan from one ignoring a turning plan, and those have opposite
fixes.

- [x] `EpisodeResult.plans` records `(sim_time, plan_heading_deg, reach_m,
      bearing_to_goal_deg)` per policy call. Both controllers steer at the same lookahead
      point, so plan heading is the controller-independent statement of intent.
      `bearing_to_goal` is scoring only, never fed to the policy.

**What it measured, first run — and this closes the question.** Warehouse, pursuit,
129 calls:

    t       asked    needed   deficit
    0.0     +0.1°    -24.9°    +25.1°
   10.5     -2.9°    -43.9°    +41.1°
   21.0     -0.7°    -80.3°    +79.6°
   35.0     +1.2°   -135.7°   +136.9°

    asked : min -8.9°  max +19.7°  mean +1.1°
    calls asking more than 5° right: 2 / 129

The policy is not being mis-executed. **It is not asking to turn.**

- [x] Ruled out the lookahead rule, which was the natural next suspect. The 30 waypoints
      are 10 Hz (`action_horizon_steps=30`), i.e. exactly 3.0 s of motion, so a
      time-parameterized lookahead is a defensible alternative to arc length. Probed
      against a real start-frame plan rather than assumed:

      | rule | heading |
      |---|---|
      | arc-length 1.0 m (ours) | +1.4° |
      | time-based 0.5 s | −2.3° |
      | time-based 1.0 s | −0.6° |
      | time-based 3.0 s | +3.8° |

      Every waypoint from t=0.0 s to t=3.0 s is within ±4° of straight. No sampling rule
      extracts a turn that is not in the plan. **Not implemented** — it would have been a
      plausible-looking change that fixed nothing.
- [x] Ruled out the conditioning. Sweeping `previous_waypoints_text` (none / 3 s / 20 s
      straight / drifting left / drifting right) and `robot_state` vx (0.6 vs 1.5) moves
      the answer only between −6.5° and +3.4°. Implied plan speed stays ~0.55 m/s.
- [x] Ruled out "send theta instead of dx,dy": there is no theta.
      `ticvla/models/ticvla.py` sets `action_dim=2,  # Offset (dx, dy)`, and the `(x,y,z)`
      triples in the `<answer>` text are position plus height. Heading has to be inferred
      from dx,dy by the controller — which is exactly why the controller choice mattered.

**Open, and stated as open:** from this viewpoint the policy will not commit to the ~25°
turn this episode needs. The next test is an episode requiring no large turn —
`warehouse_aisle6` (5.0° off bearing) or `hospital` (6.4° off, straight hallway) — to
separate "the pipeline cannot turn" from "this episode is hard". Not to be fixed by
widening the guard or nudging controller gains, because either would make the benchmark
stop measuring anything.

---

## Open questions

- **Does the DynaNav office scene load in Isaac Sim 6.0.1 at all?** Phase 8 answers
  this. It is the single largest schedule risk; everything downstream assumes it.
- ~~**Is the straight-line bias the policy, or this machine's integration of it?**~~
  **Answered in Phase 15, and the question contained the answer.** "The smoke episodes
  driving away from the goal is not normal for a policy at 55%" was correct — it was not
  the policy. Three of the four smoke episodes *spawn* facing away from their own goal,
  so a policy driving off in the wrong direction was the only thing it could have done.
  Fixed by using `benchmark_full.yaml`. Note the observation sat here as an open
  question for two runs while the behaviour it described was being written up elsewhere
  in this file as the task succeeding.
- **Should the arms and lift do anything during navigation?** Assumed parked in a safe
  tucked pose for now. Manipulation-after-navigation is a natural follow-on and is
  explicitly out of scope here.
- **Whose `num_action_chunks`?** Default 30. Nova Carter's pure-pursuit lookahead
  (`L_des = 1.0` m) was tuned against that horizon; if AlohaMini's speed limits differ
  materially, the lookahead needs retuning, not just the chunk count.
