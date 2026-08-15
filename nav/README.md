# `nav/` — language-driven navigation for AlohaMini

Type a sentence, watch the robot drive. The policy is [TIC-VLA][ticvla]; the
environments and the prompts are [DynaNav][ticvla]'s published benchmark.

**The policy is never told where the goal is.** It gets one camera frame and one
sentence. `goal` in `config/episodes.yaml` exists only to score the run — the same
role it plays in DynaNav's benchmark. A version that fed the goal coordinate to a
path planner would look identical on video and would demonstrate nothing.

[ticvla]: https://github.com/  <!-- ../../TIC-VLA, vendored locally -->

## Quick start

```bash
nav/sim/build_nav_scene.sh hospital
```

```bash
nav/run.sh --episode hospital_down_hallway
```

Then open <http://127.0.0.1:8080>. Every DynaNav prompt is listed there, grouped by
environment. **Anything under the loaded environment can be selected and run right
there** — picking it teleports the robot to that episode's start and scores against
that episode's goal, so one `nav/run.sh` covers all seven hospital episodes. Episodes
from other environments are listed but greyed, with the relaunch command shown on
click: Isaac Sim cannot swap stages in-process at this version, and a half-swapped
stage fails as a *navigation* error rather than as a crash — a benchmark's worst
failure mode, because it produces a plausible number.

One stage per environment, three in total:

```bash
nav/sim/build_nav_scene.sh hospital    # 7 episodes
nav/sim/build_nav_scene.sh office      # 4 episodes
nav/sim/build_nav_scene.sh warehouse   # 2 episodes
```

Or let `nav/bench.sh` build what it needs as it walks the ladder.

**Two full runs recorded from the live UI, third-person view on** — both succeed:

<table>
<tr><td width="50%">

`hospital_forward_staircase` (+66.1° turn, SPL 0.90 — beats DynaNav's own 0.88)

<video src="../docs/nav_demo_hospital_forward_staircase.mp4" controls muted playsinline width="100%"></video>

</td><td width="50%">

`hospital_down_hallway2` (32.5 m straight approach, SPL 1.00)

<video src="../docs/nav_demo_hospital_down_hallway2.mp4" controls muted playsinline width="100%"></video>

</td></tr>
</table>

The UI also has:

- **third-person view** — a checkbox above the camera panel. Ticking it opens a second
  live view from a chase camera 2.5 m behind the robot, side by side with the nav
  camera, so you can watch the robot drive and see what it is reasoning over at the same
  time. It is off by default and created on first use: once an Isaac Sim `Camera`
  exists, Kit renders it every frame whether or not anyone is looking.
- **Reset** — teleports the robot back to the episode's start pose, aborting a run in
  progress. It also clears the policy server's KV cache, so the next run starts from a
  genuinely blank context rather than carrying the failed attempt's reasoning forward.

`run.sh` starts the policy server itself if it is not already up, and reuses it if it
is. Building the scene is a one-off per episode (it takes a few minutes — the
environments stream from NVIDIA's CDN).

## Why two processes

```
        Browser  :8080
           │  prompt in, camera + status out
           ▼
┌──────────────────────────────┐        ┌─────────────────────────────┐
│ SIM PROCESS      GPU0        │  HTTP  │ POLICY SERVER     GPU1      │
│ isaacsim-6.0.1   Python 3.12 │ :8765  │ ~/envs/tic-vla    Python 3.11│
│                              │───────▶│                             │
│ run_navigation.py + ui/      │◀───────│ TIC-VLA + InternVL3-1B      │
│ Isaac Sim, AlohaMini         │        │ frame path in, waypoints out│
└──────────────────────────────┘        └─────────────────────────────┘
```

This split is forced, not stylistic:

- **Python versions cannot be reconciled.** TIC-VLA targets Isaac Sim 5.0's NumPy 1.x
  ABI and needs 3.11; AlohaMini runs on Isaac Sim 6.0.1, which is 3.12. The root
  `robotics/CLAUDE.md` forbids merging those environments.
- **Neither fits beside the other on one card.** A DynaNav scene alone was measured
  holding 25.4 GiB of GPU0's 31.35 GiB. Loading the 1.9 GB checkpoint on the same
  card OOMs. *The scene is what is large, not the model.*

Frames cross the boundary as **file paths**, not bytes — both processes are on this
machine and `TICVLA.predict()` already takes `image_paths`.

GPU pinning differs by process **on purpose**: the policy server sets
`CUDA_VISIBLE_DEVICES=1`, which is safe only because it never starts Kit. Isaac Sim is
pinned through Kit's own `active_gpu`/`physics_gpu` config instead.

## Layout

| path | what it does |
|---|---|
| `config/episodes.yaml` | generated: 11 DynaNav-completed episodes + 2 manual, easiest first |
| `config/episodes_manual.yaml` | hand-written episodes, appended verbatim by the importer |
| `sim/import_benchmark.py` | rebuilds the ladder from DynaNav's results; ranks by `spl` |
| `policy_server/server.py` | FastAPI wrapper over `TICVLA.predict()` (py3.11, GPU1) |
| `policy_server/client.py` | stdlib-only client, so Isaac Sim's py3.12 can import it |
| `sim/build_nav_scene.sh` | compose environment + robot, then apply the pipeline |
| `sim/resolve_env.py` | which stage does this episode need? (no Kit boot) |
| `bench.sh` | the whole ladder: build per environment, run per episode |
| `sim/summarize_runs.py` | closest approach, `spl`, guard rate across saved runs |
| `sim/run_navigation.py` | the loop, and the UI's job queue |
| `sim/controllers.py` | `braking` (default), `pursuit`, `guided`, `holonomic` |
| `sim/collision_guard.py` | raycast fan — the base does not collide on its own |
| `ui/` | prompt box, live camera, benchmark examples |
| `tools/check_policy_sanity.py` | does the output depend on the prompt? |
| `tools/check_scene_compat.py` | will a 5.0-authored scene open in 6.0.1? |

## Things that will surprise you

**The `forward` camera is the wrong camera.** All three LeRobot cameras face the
manipulation front (−Y); the base drives +X. Navigation uses a fourth camera,
`camera_nav`, deliberately kept out of `CAMERA_PRIM_PATHS` so the LeRobot observation
contract is unchanged.

**`camera_nav` is a copy of Nova Carter's front Hawk, not AlohaMini's own camera.**
90.1° HFOV, 0.346 m up, dead level, rendered 1920×1080 — probed off
`nova_carter_sensors.usd`, because that is the sensor DynaNav renders and therefore the
one TIC-VLA was trained through. A camera is part of a VLA's input distribution, not a
styling choice. The first version used AlohaMini's own webcam intrinsics (78° HFOV) at
1.15 m on the lift column, tilted 10° down, and the policy could not find the landmarks
the prompts name: `docs/nav_start_frame_hawk.jpg` shows what it sees now — all six aisle
placards legible and the Aisle 05 traffic cones in frame from the start line.

**Use `benchmark_full.yaml`, never `benchmark_example.yaml`.** The example file is a
four-episode smoke config, and three of its four spawn the robot facing ~180° away from
their own goal. Running one is indistinguishable from a broken policy — the robot drives
off the wrong way and never sees the landmark, because the landmark is behind it. Every
episode in `config/episodes.yaml` is annotated with how far off its goal bearing it
starts; keep that inside ~30°.

**The base does not collide.** Locomotion is `set_world_poses()` teleportation —
`../CLAUDE.md` has the full story of why real wheel traction never held up. A
teleported body has no contact response and will pass through a wall, so
`collision_guard.py` raycasts ahead instead. That is genuinely weaker than physics: it
stops the robot, it does not push back.

**Episode timeouts count sim time, not wall clock.** Inference is synchronous and
blocks ~1.0–1.5 s per call while the robot stands still.

**The office episode is not the default.** It runs (`--episode office`) and Isaac Sim
6.0.1 opens it cleanly, but it is 1063 m across and is the same NVIDIA Office asset
that `../CLAUDE.md` records as having exploded this robot at spawn. `warehouse` is
equally a real benchmark episode and is already verified stable here.

**The policy is nearly deterministic.** `predict()` samples at temperature 0.1 with
top-k 10, which is effectively greedy — repeated calls on the same frame returned
headings identical to 0.000°. Retrying a stuck state will not shake it loose.

**Turn authority is modest.** About ±14° at the end of a 30-waypoint plan. The policy
does respond to "turn left" (verified, correctly signed), but not sharply.

## Checking it is real, not a moving robot

```bash
/home/gtu-dsa/envs/tic-vla/bin/python nav/tools/check_policy_sanity.py --frame <a.jpg> --repeats 3
```

Holds the image fixed and varies only the instruction. If the waypoints do not change,
nothing here is language-driven and the demo is just a robot driving forward. Results
from this machine are in [`plan.md`](plan.md) under Phase 9.

The same test is available through the UI: load a prompt marked *other scene* and run
it against the loaded environment. It should behave differently from the native one.
