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
nav/sim/build_nav_scene.sh warehouse
```

```bash
nav/run.sh
```

Then open <http://127.0.0.1:8080>. The four DynaNav prompts are listed in the UI;
click one to load it, or type your own.

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
| `config/episodes.yaml` | the four DynaNav episodes, prompts copied verbatim |
| `policy_server/server.py` | FastAPI wrapper over `TICVLA.predict()` (py3.11, GPU1) |
| `policy_server/client.py` | stdlib-only client, so Isaac Sim's py3.12 can import it |
| `sim/build_nav_scene.sh` | compose environment + robot, then apply the pipeline |
| `sim/run_navigation.py` | the loop, and the UI's job queue |
| `sim/controllers.py` | `pursuit` (DynaNav parity) and `holonomic` (omni base) |
| `sim/collision_guard.py` | raycast fan — the base does not collide on its own |
| `ui/` | prompt box, live camera, benchmark examples |
| `tools/check_policy_sanity.py` | does the output depend on the prompt? |
| `tools/check_scene_compat.py` | will a 5.0-authored scene open in 6.0.1? |

## Things that will surprise you

**The `forward` camera is the wrong camera.** All three LeRobot cameras face the
manipulation front (−Y); the base drives +X. Navigation uses a fourth camera,
`camera_nav`, deliberately kept out of `CAMERA_PRIM_PATHS` so the LeRobot observation
contract is unchanged.

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
