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

---

## Open questions

- **Does the DynaNav office scene load in Isaac Sim 6.0.1 at all?** Phase 8 answers
  this. It is the single largest schedule risk; everything downstream assumes it.
- **Is the straight-line bias the policy, or this machine's integration of it?** The
  49-episode run scoring 55% argues the pipeline basically works, which makes a pure
  integration bug less likely — but the smoke episodes driving *away* from the goal is
  not normal for a policy at 55%. Phase 9 separates the two. Worth resolving regardless
  of AlohaMini, since it affects the thesis result directly.
- **Should the arms and lift do anything during navigation?** Assumed parked in a safe
  tucked pose for now. Manipulation-after-navigation is a natural follow-on and is
  explicitly out of scope here.
- **Whose `num_action_chunks`?** Default 30. Nova Carter's pure-pursuit lookahead
  (`L_des = 1.0` m) was tuned against that horizon; if AlohaMini's speed limits differ
  materially, the lookahead needs retuning, not just the chunk count.
