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
- [ ] `nav/tools/check_scene_compat.py` — load DynaNav's `office.usd` (authored for
      Isaac Sim **5.0**) in Isaac Sim **6.0.1** headless; report whether it opens, its
      world-space bounding box, and whether the CDN references resolve
- [ ] Check scale against `CLAUDE.md`'s warning: the *other* Office environment was
      real-world scale (~1000 m across) and caused both a camera-framing bug and a
      physics explosion at spawn. Confirm DynaNav's is sane before adopting it
- [ ] `nav/tools/check_policy_server.py` — stand up the policy server, POST one frame,
      confirm a `(T,2)` waypoint array comes back and log the latency
- [ ] Record both results in this file. **If the scene will not load in 6.0.1**, fall
      back to Isaac's `Simple_Warehouse` (already the repo default and known-good) and
      keep the DynaNav *prompts* — note it as a deviation rather than hiding it

### Phase 9 — Policy sanity (the gate; see finding 3)
- [ ] `nav/tools/check_policy_sanity.py` — one fixed frame, N contradictory
      instructions, compare the returned waypoint sets
- [ ] Report per-instruction mean heading and lateral spread; assert the sets are
      measurably different
- [ ] Repeat with a frame captured from AlohaMini's own forward camera (different
      mounting height and FOV to Nova Carter — a real distribution shift)
- [ ] Write the verdict into this file, whatever it is

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
- [ ] `nav/sim/episode.py` — episode config (start, start_yaw, goal, instruction,
      timeout) + success test + SPL/path-length metrics, mirroring DynaNav's definitions
      so numbers are comparable
- [ ] `nav/config/episodes.yaml` — the four DynaNav smoke episodes, re-homed to
      AlohaMini (start poses re-validated: AlohaMini's footprint and camera height
      differ from Nova Carter's, so a start pose that is collision-free for one is not
      automatically free for the other)
- [ ] `nav/sim/camera_source.py` — grab the `forward` camera each step, write PNG to a
      scratch dir, maintain the rolling history `predict()` expects
- [ ] `nav/sim/controllers.py` — `PursuitController` + `HolonomicController`, sharing a
      waypoint-tracking base class
- [ ] `nav/sim/collision_guard.py` — PhysX scene query ahead of the intended motion;
      refuse the kinematic step on a hit and record it (finding 2)
- [ ] `nav/sim/run_navigation.py` — the main loop tying it together
- [ ] Commit + push after each of the above

### Phase 12 — Web UI
- [ ] `nav/ui/server.py` — FastAPI in a background thread of the sim process (same
      pattern as `dash-so101`'s `ui/server.py`, which already solved the
      load-model-once problem)
- [ ] `nav/ui/static/index.html` — prompt box, **dropdown of the four DynaNav example
      prompts**, Start/Stop/Reset, live MJPEG of the forward camera
- [ ] Surface per-step: distance to goal, elapsed/timeout, current `(vx,vy,ω)`, the
      VLM's reasoning text, and a top-down plot of the waypoints
- [ ] Commit + push

### Phase 13 — Run it, measure it, write it down
- [ ] Run all four episodes with `--controller pursuit`; save results JSON in DynaNav's
      schema
- [ ] Repeat with `--controller holonomic`; compare
- [ ] Record success rate, SPL, path length, collisions — and state plainly how they
      compare to DynaNav's own Nova Carter numbers
- [ ] Update `../CLAUDE.md` with every gotcha hit
- [ ] Commit + push

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
