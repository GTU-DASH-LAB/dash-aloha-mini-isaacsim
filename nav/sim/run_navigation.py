"""The navigation loop: camera -> TIC-VLA -> controller -> kinematic base.

Runs inside Isaac Sim 6.0.1's Python 3.12. The policy lives in a separate process
(Python 3.11, GPU1) and is reached over HTTP -- see ../plan.md for why that split is
forced rather than chosen.

One thing to be clear about, because it is the entire point of the exercise: **the
policy is never told where the goal is.** `episode.goal` exists only to score the run,
exactly as it does in DynaNav's benchmark. TIC-VLA gets a camera frame and a sentence.
If the robot arrives, it arrived by reading "stop in front of Aisle 5, where traffic
cones and a yellow caution sign block the entrance" and finding that in the image. A
version that fed the goal coordinate to a planner would look identical on video and
would prove nothing.

Inference is ASYNCHRONOUS, the way the paper is. TIC-VLA is "Think-in-Control": a ~1 s
VLM generation producing a KV cache, and a millisecond action expert that drives on
whatever cache is currently on hand. The policy server calls `predict_async`, so the
slow half runs on a background thread over there and this loop only ever waits for the
fast half -- the robot no longer stands still while the model thinks.

The consequence is that every plan is built on thinking that is roughly one generation
old, and that staleness is declared rather than hidden: `time_delay` says how many
seconds old, and the dx,dy tail of `robot_state` says how far the robot travelled
meanwhile, rotated into the body frame the model was reasoning in. Those two fields are
the paper's latency compensation. Zeroing them does not remove the delay, it only stops
the model being able to correct for it.

This was synchronous until the async switch, and the zeros were correct then: a frozen
robot really has moved nowhere. If you ever put `predict` back, put the zeros back too.

**And that is exactly what `NAV_PLAN_PERIOD_S` does.** Asynchronous inference is the
right trade only while a generation is cheap against the plan it is spending. It stops
being right the moment thinking gets expensive, because "the robot keeps driving" and
"the robot drives blind" are the same sentence: the blind window IS the generation. The
13-episode ladder measured the cost and it is not subtle -- 7/13 with no thinking, 2/13
with a moderate budget, 2/13 with a large one, guard interventions rising from a few
hundred to over two thousand. That ladder did not measure whether more thinking helps.
It measured how far a robot travels with its eyes shut, which is proportional to how
long it thinks, and it therefore made the levels un-comparable by construction.

Set `NAV_PLAN_PERIOD_S` and the loop becomes classical sense -> plan -> act: drive the
current plan for exactly that many SIMULATED seconds, then hold still until a plan built
on the CURRENT frame arrives. The simulation genuinely freezes while it waits, because
the loop reaches `kit.update()` only after the call returns -- so no physics steps, no
rendering, no sim time, and the compensation zeros above become literally true again.
Thinking then costs wall-clock instead of costing collisions, which is the only regime in
which "does it decide better when it thinks longer?" is a question about deciding.

Usage:
    nav/run.sh --episode warehouse --controller braking
    NAV_PLAN_PERIOD_S=3.0 nav/run.sh --episode warehouse    # stop-and-think every 3 s
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "nav" / "sim", REPO / "nav" / "policy_server", REPO / "nav" / "ui",
          REPO / "scripts"):
    sys.path.insert(0, str(p))

from client import PolicyClient, PolicyServerError  # noqa: E402
from controllers import (  # noqa: E402
    DYNANAV_LOOKAHEAD_M,
    Command,
    _lookahead_point,
    make_controller,
    plan_speed,
)
from episode import (  # noqa: E402
    Episode,
    EpisodeResult,
    env_name,
    episodes_by_env,
    load_episode,
    load_episodes,
)
from frame_history import FrameHistory  # noqa: E402
from guidance import guidance_heading, parse_guidance  # noqa: E402
from stuck_recovery import StuckRecovery  # noqa: E402
from waypoint_history import WaypointHistory  # noqa: E402

PHYSICS_DT = 1.0 / 60.0

# Fixed planning period in SIMULATED seconds, or 0.0 for the paper's asynchronous mode.
#
# When set, this replaces the episode's own `replan_every_steps` and switches the loop to
# sense -> plan -> act: the robot drives one plan for exactly this long, then stops dead
# and waits for a plan generated from the frame it is standing on. See the module
# docstring for why -- in one line, an asynchronous loop drives blind for the length of a
# generation, so raising the thinking budget raises the distance travelled blind, and the
# thinking levels stop being comparable.
#
# The period is the ACTING window, so it must not exceed the plan horizon or the robot
# runs off the end of its own plan before the next decision: `HORIZON_S` is 10.0 at
# medium and 16.0 at high, against periods of 3.0 and 1.0 here, so there is a wide margin.
# `check_plan_period()` enforces it rather than trusting the arithmetic to stay true.
PLAN_PERIOD_S = float(os.environ.get("NAV_PLAN_PERIOD_S", "") or 0.0)


def plan_period_steps(episode_default: int) -> int:
    """Physics steps between decisions: the fixed period if set, else the episode's own.

    Kept as a function so the two places that need it (the chase-camera pairing and the
    replan test) cannot drift apart -- they must fire on the same steps or the recorded
    `chase_<step>.jpg` stops being the third-person twin of `menu_<step>.jpg`.
    """
    if PLAN_PERIOD_S > 0:
        return max(1, round(PLAN_PERIOD_S / PHYSICS_DT))
    return episode_default


def check_plan_period(policy, replan_steps: int, synchronous: bool) -> None:
    """Print the timing contract, and refuse a period the plan cannot cover.

    Printed and not merely asserted because the failure this guards against is silent:
    a period longer than the plan horizon leaves the robot driving off the end of its
    own trajectory for the remainder of every window, which looks like a policy that
    stops steering rather than like a misconfiguration. And a run that was MEANT to be
    synchronous but was launched without the variable produces a complete, plausible set
    of numbers measured under the other regime entirely -- the same class of mistake as
    scoring a ladder against a server still holding the previous thinking level.
    """
    period_s = replan_steps * PHYSICS_DT
    if not synchronous:
        print(f"[nav] planning: ASYNCHRONOUS, every {period_s:.2f} s sim "
              f"({replan_steps} steps); the robot drives the previous plan while the "
              f"model thinks", flush=True)
        return
    print(f"[nav] planning: SYNCHRONOUS, every {period_s:.2f} s sim ({replan_steps} "
          f"steps); the robot STOPS and waits for each decision", flush=True)
    # The horizon lives in the server, so ask it rather than assuming the pairing here.
    # An unreachable or older server is not fatal -- it just cannot answer, and a run
    # that skips the check is better than one that refuses to start over a status read.
    horizon = None
    try:
        horizon = float(policy.health()["horizon_s"])
    except Exception:
        # Includes the TIC-VLA server, which serves the same route with a different
        # schema and has no `horizon_s` at all. Never subscript a payload this process
        # does not own -- doing exactly that once turned a whole 13-episode ladder into
        # a KeyError inside setup() that scored as thirteen ordinary failures.
        pass
    if horizon is None:
        print("[nav] could not read horizon_s from the policy server; period unchecked",
              flush=True)
    elif period_s > horizon:
        raise SystemExit(
            f"[nav] plan period {period_s:.2f} s exceeds the policy's {horizon:.2f} s "
            f"horizon -- the robot would run off the end of every plan before the next "
            f"decision. Lower NAV_PLAN_PERIOD_S or raise the thinking level's horizon.")
    else:
        print(f"[nav] plan horizon {horizon:.1f} s covers the {period_s:.2f} s period",
              flush=True)


# Which of the three guidance horizons (3/6/9 s) gets logged in `plans`. Logged for
# every controller even though only `guided` steers on it, so the two channels can be
# compared within one run. Kept here rather than at the call site so the logged column
# always describes the same horizon the controller would have used.
GUIDANCE_HORIZON_S = 6.0

# How often a third-person frame is written to disk during a recorded run, in physics
# steps. 2 steps = 30 frames per simulated second.
#
# This used to fire only on decision steps, and a decision is taken every ~3 s, so the
# recorded third-person track was ~0.3 fps: a slideshow of a robot teleporting between
# poses, which is unreadable as motion. The video is the only way anyone but the runner
# sees what happened, so a track that cannot show HOW the robot moved cannot answer the
# question it is watched to answer.
#
# WHAT IT COSTS, measured rather than assumed, because "render every frame" sounds
# expensive and mostly is not: Kit renders this camera's render product every frame
# whether or not anything reads it (see camera_source.py), so the render is already paid
# for. What a capture actually costs is the readback plus the JPEG encode, measured at
# 1.70 ms against a 34.6 ms physics step. Every 2 steps is therefore ~2.5% of sim time,
# plus ~25 kB per frame on disk -- about 90 MB for a 100 s episode, deleted with the
# frames once the video is built.
#
# Overridable because that trade is different on a long episode than on a short one, and
# because the honest answer to "is this slowing the benchmark" is a number the operator
# can change, not a constant they have to trust.
CHASE_EVERY = max(1, int(os.environ.get("NAV_CHASE_EVERY", "2")))


def ready_message(info: dict) -> str:
    """Describe whichever policy server answered, without assuming which one it is.

    This line used to subscript the health payload directly for `num_action_chunks`
    and `device`. Those are `server.py`'s keys. `server_qwen.py` serves the same three
    ROUTES with a different payload, so pointing the runner at it raised KeyError
    inside `setup()` -- before a single step had been driven.

    That crash cost more than its size, because of how it was reported. An uncaught
    exception exits Python with status 1, and `bench.sh` reads 1 as "the episode
    failed", not "the process died". So the 13-episode ladder ran to completion in
    four minutes, scored every episode off the PREVIOUS run's result file, and printed
    a perfectly plausible 6/13 -- the exact "plausible number" failure the one-process-
    per-episode design exists to prevent. `bench.sh` now also requires a NEW result
    file per run, so the two guards are independent.

    Rule for anything reached over HTTP: match on routes AND payload, or use `.get`.
    """
    bits = []
    if info.get("num_action_chunks") is not None:
        bits.append(f"{info['num_action_chunks']} chunks")
    if info.get("mode"):        # the qvla-direct server names itself here
        bits.append(str(info["mode"]))
    if info.get("format"):
        bits.append(f"format={info['format']}")
    if info.get("device"):
        bits.append(f"on {info['device']}")
    return "policy ready" + (f" ({', '.join(bits)})" if bits else "")


class NavigationRunner:
    """Owns the Isaac Sim stage and executes one episode at a time.

    Kit must be driven from the main thread, so the UI server runs in a *worker*
    thread and hands jobs here rather than the other way round. `status` and
    `latest_jpeg` are the shared read-only-ish surface the UI polls; both are guarded
    by `_lock`.
    """

    def __init__(
        self,
        episode: Episode,
        controller_name: str = "braking",
        policy_host: str = "127.0.0.1",
        policy_port: int = 8765,
        scene_path: Path | None = None,
        scratch_dir: Path | str = "/tmp/alohamini-nav-frames",
        headless: bool = True,
        sim_gpu: int = 0,
    ):
        self.episode = episode
        self.controller_name = controller_name
        # Keyed on the ENVIRONMENT, not the episode -- six hospital episodes share one
        # stage and differ only by a start pose the runner sets itself. See
        # episode.env_name().
        self.scene_path = scene_path or (
            REPO / "assets" / "usd" / f"nav_{env_name(episode.scene)}.usda"
        )
        self.scratch_dir = Path(scratch_dir)
        self.headless = headless
        self.sim_gpu = sim_gpu
        self.policy = PolicyClient(policy_host, policy_port)

        self._lock = threading.Lock()
        self.status: dict = {
            "state": "starting",
            "episode": episode.name,
            "instruction": episode.instruction,
            "controller": controller_name,
            "distance_m": None,
            "initial_distance_m": None,
            "elapsed_s": 0.0,   # sim time
            "wall_s": 0.0,      # includes blocking inference
            "steps": 0,
            "policy_calls": 0,
            "guard_interventions": 0,
            "reasoning": "",
            "message": "",
            # False once a run uses an instruction other than the episode's own, at
            # which point the distance readouts no longer describe the task and the
            # UI stops presenting them as a score. See run_episode().
            "scored": True,
        }
        self.latest_jpeg: bytes | None = None
        # Third-person view. None until the UI asks for it -- an Isaac Sim Camera
        # allocates a render product that Kit then renders EVERY frame whether or not
        # anyone reads it, so a headless benchmark should not pay for a view nobody is
        # watching. `chase_wanted` is set from the UI thread; the main thread acts on it.
        self.chase_jpeg: bytes | None = None
        self.chase_wanted = False
        # Where this run's chase frames go, alongside the server's menus. Set per
        # episode from /reset's answer; None means the server is not recording, so
        # there is nothing to build a video out of and no reason to pay for the camera.
        self.record_dir: Path | None = None
        self.chase_available: bool | None = None  # None = not tried yet
        self._chase: object | None = None
        self._preview_t = 0.0  # throttles the between-runs camera refresh

        self.kit = None
        self.art = None
        self.camera = None
        self.guard = None
        self.recovery = None
        self.base = None
        self._abort = threading.Event()
        self._reset = threading.Event()
        # Set from the UI thread, applied by perform_reset() on the main thread.
        self._pending_episode: str | None = None
        # Kit must be driven from the main thread, so the UI thread cannot run an
        # episode itself -- it queues one and the main loop picks it up.
        self._jobs: queue.Queue[dict] = queue.Queue()
        self.last_result: EpisodeResult | None = None

    # ------------------------------------------------------------------ status
    def _set(self, **kwargs) -> None:
        with self._lock:
            self.status.update(kwargs)

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self.status)
        # Chase state is reported so a browser reload can restore the toggle. Without
        # it a reload leaves the checkbox off while the camera is still allocated and
        # rendering every frame -- paying the cost with nothing on screen.
        snap["chase_enabled"] = self.chase_wanted
        snap["chase_available"] = self.chase_available
        return snap

    def policy_label(self) -> str:
        """Identify the server that answered, for the run record.

        Read off `/health` rather than from our own flags: the port is what we chose, the
        model and format are what actually loaded, and it is the second pair that decides
        what the number means. A stale server left running in the wrong format is exactly
        the mistake this is here to make visible afterwards.
        """
        info = getattr(self, "_policy_info", None) or {}
        model = str(info.get("model", "?")).rsplit("/", 1)[-1]
        fmt = info.get("format")
        return f"{model}{f' [{fmt}]' if fmt else ''} @{self.policy.base.rsplit(':', 1)[-1]}"

    def request_stop(self) -> None:
        self._abort.set()

    def request_reset(self, episode: str | None = None) -> None:
        """Put the robot back on the start line, optionally of a different episode.

        Aborts a running episode too: resetting mid-run and letting the loop carry on
        against a teleported robot would produce a nonsense trace. The actual work
        happens on the main thread in `perform_reset()` -- this only raises the flag,
        because touching the articulation from the UI thread races PhysX. Same reason
        the episode switch is stashed rather than applied here.
        """
        if episode:
            self._pending_episode = episode
        self._abort.set()
        self._reset.set()

    # ------------------------------------------------------------------- chase
    def set_chase_enabled(self, enabled: bool) -> bool:
        """Turn the third-person view on/off. Returns whether it is now available."""
        self.chase_wanted = enabled
        if not enabled:
            return True
        # Creation itself must happen on the main thread; report what we know now.
        return self.chase_available is not False

    def _update_chase(self) -> None:
        """Main thread only. Creates the camera on first use, then grabs a frame."""
        if not self.chase_wanted:
            return
        if self._chase is None:
            if self.chase_available is False:
                return
            from camera_source import try_create_chase

            self._chase = try_create_chase()
            self.chase_available = self._chase is not None
            if self._chase is None:
                self._set(message=(
                    "this scene has no chase camera -- rebuild it with "
                    f"nav/sim/build_nav_scene.sh {self.environment}"
                ))
                return
        frame = self._chase.grab_jpeg()
        if frame:
            self.chase_jpeg = frame

    def _capture_chase(self, step: int) -> None:
        """Grab a third-person frame and, if this run is recording, write it to disk.

        Best-effort throughout: a missing third-person frame costs a panel in a video and
        must never cost a benchmark episode. That is why the whole body is wrapped rather
        than just the write -- `_update_chase` creates the camera on first use, and a scene
        without one has already been seen to raise from inside Kit rather than return None.
        """
        try:
            self._update_chase()
            if self.record_dir is not None and self.chase_jpeg:
                (self.record_dir / f"chase_{step:08d}.jpg").write_bytes(self.chase_jpeg)
        except Exception as exc:
            print(f"[nav] chase capture failed at step {step}: {exc}", flush=True)

    def _update_idle_views(self) -> None:
        """Main thread only. Keep both camera panels live between runs.

        While an episode runs, the nav panel deliberately shows the last frame the
        POLICY was given -- that is what makes the reasoning text next to it legible.
        Between runs no such frame exists, and a black panel beside a moving chase view
        reads as a broken UI rather than an idle one, so it shows a live preview.

        Throttled to ~10 Hz, matching the browser's chase timer. It used to be ~3 Hz
        against a 700 ms poll, which was the right pairing then; the chase view now has
        its own 100 ms timer, and leaving the producer at 3 Hz would just serve the same
        JPEG three times. Nothing is competing for the main thread while idle, and a
        `grab_jpeg()` costs 1.70 ms measured, so this is affordable in a way it would
        not be mid-episode. `grab_jpeg()` rather than `grab_to_file()` on purpose --
        the on-disk frame counter belongs to the policy's frames.
        """
        now = time.time()
        if now - self._preview_t < 0.1:
            return
        self._preview_t = now
        if self.camera is not None:
            frame = self.camera.grab_jpeg()
            if frame:
                self.latest_jpeg = frame
        self._update_chase()

    # ------------------------------------------------------------- episode set
    @property
    def environment(self) -> str:
        """The environment whose stage is loaded, e.g. "hospital"."""
        return env_name(self.episode.scene)

    def runnable_episodes(self) -> dict[str, Episode]:
        """Every episode this process can run right now.

        That is every episode sharing the loaded stage -- the whole point of building
        one stage per environment. Switching between them is a teleport plus a new
        goal; switching environment is not possible in-process, because Isaac Sim
        cannot swap stages cleanly at this version and a half-swapped stage fails as a
        navigation error rather than as a crash, i.e. it produces a plausible number.
        """
        return {
            ep.name: ep
            for ep in episodes_by_env().get(self.environment, [])
        }

    def check_episode(self, name: str) -> str | None:
        """None if `name` can be run here, otherwise why not, in a sentence."""
        available = self.runnable_episodes()
        if name in available:
            return None
        try:
            other = load_episode(name)
        except KeyError:
            return f"unknown episode {name!r}"
        return (
            f"'{name}' runs in the {env_name(other.scene)} environment; this process "
            f"has {self.environment} loaded. Relaunch with: "
            f"nav/run.sh --episode {name}"
        )

    # -------------------------------------------------------------- job queue
    def submit_job(
        self,
        instruction: str,
        controller: str | None = None,
        episode: str | None = None,
    ) -> bool:
        """Called from the UI thread. False if a run is already in flight.

        `episode` is validated by the caller through check_episode() -- it is applied
        on the main thread in run_episode(), because switching it moves the robot.
        """
        if self.snapshot()["state"] == "running":
            return False
        self._jobs.put(
            {"instruction": instruction, "controller": controller, "episode": episode}
        )
        return True

    def reset_pending(self) -> bool:
        return self._reset.is_set()

    def take_job(self, timeout: float = 0.05) -> dict | None:
        try:
            return self._jobs.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------- setup
    def setup(self) -> None:
        if not self.scene_path.is_file():
            raise FileNotFoundError(
                f"nav scene not built: {self.scene_path}\n"
                f"  build it with: nav/sim/build_nav_scene.sh {env_name(self.episode.scene)}"
            )

        from isaacsim import SimulationApp

        # Pin the simulator to GPU0 through Kit's OWN config, never through
        # CUDA_VISIBLE_DEVICES -- Kit warns that pinning it "can lead to undesired
        # behavior or crashes", and memory/gtu-workstation-gpu-asymmetry.md is
        # emphatic about it. The policy server does use the env var, which is safe
        # there only because that process never starts Kit.
        #
        # This is not tidiness: GPU1 is already holding InternVL3-1B plus the
        # checkpoint, and a DynaNav scene can take 25.4 GiB on its own.
        self.kit = SimulationApp({
            "headless": self.headless,
            "active_gpu": self.sim_gpu,
            "physics_gpu": self.sim_gpu,
            "multi_gpu": False,
        })

        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import Articulation

        from base_drive import KinematicBase
        from camera_source import NavCameraSource
        from collision_guard import CollisionGuard

        self._set(state="loading", message=f"opening {self.scene_path.name}")

        usd_context = omni.usd.get_context()
        # ABSOLUTE path. A relative --scene arg once made the relative robot reference
        # silently fail to resolve: environment loaded, robot missing, empty bbox, no
        # error anywhere (../../CLAUDE.md).
        usd_context.open_stage(str(self.scene_path.resolve()))
        stage = usd_context.get_stage()
        # These environments stream from the CDN. Judging the stage on frame 1 sees a
        # half-loaded world; the office scene needed 188 s to fully resolve.
        for _ in range(240):
            self.kit.update()
        stage.Load()
        for _ in range(30):
            self.kit.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(10):
            self.kit.update()

        self.art = Articulation("/World/Aloha/Geometry/base_link")
        self.art.initialize()
        dof_names = list(self.art.dof_names)

        self.base = KinematicBase(self.art, dof_names)
        self.base.sync_from_sim()

        self.camera = NavCameraSource(scratch_dir=self.scratch_dir)
        self.camera.warmup(self.kit)
        self.guard = CollisionGuard()
        self.recovery = StuckRecovery(self.guard)

        self._set(state="ready", message="waiting for policy server")
        info = self.policy.wait_until_ready()
        # Kept, not just printed: it is the only place the run learns which brain it is
        # about to drive, and `EpisodeResult.policy` records it alongside the result.
        self._policy_info = info
        self._set(state="idle", message=ready_message(info))

    # --------------------------------------------------------------- execution
    def run_episode(
        self,
        instruction: str | None = None,
        controller_name: str | None = None,
        episode: str | None = None,
    ) -> EpisodeResult:
        # Switching episode first, so `instruction=None` picks up the NEW episode's
        # prompt rather than the outgoing one's.
        if episode and episode != self.episode.name:
            reason = self.check_episode(episode)
            if reason:
                raise ValueError(reason)
            self.episode = self.runnable_episodes()[episode]
            self._set(episode=self.episode.name)

        instruction = instruction or self.episode.instruction
        if controller_name:
            self.controller_name = controller_name
        ep = self.episode
        self._abort.clear()

        controller = make_controller(
            self.controller_name, ep.max_speed_mps, ep.max_yaw_rate_radps
        )
        controller.reset()
        # Clears the KV cache between episodes, and names the episode so the arc-menu
        # server files this run's menus and decisions under it.
        reset_info = self.policy.reset(run=ep.name)

        # Where the server is filing this run's menus and decisions. The chase frames go
        # in the SAME directory, keyed on the same step number, so the video tool can
        # pair "what the robot saw and chose" with "what it looked like from outside"
        # without a second index that could drift out of step with the first.
        # None whenever the server is not recording, which is also exactly when there is
        # no video to build -- so no separate switch for this.
        rec_dir = reset_info.get("run_dir") if isinstance(reset_info, dict) else None
        self.record_dir = Path(rec_dir) if rec_dir else None
        if self.record_dir is not None:
            # Turning this on creates the camera, and once an Isaac Sim `Camera` exists
            # Kit renders its render product every frame whether or not anything reads
            # it -- so a run that is not being recorded must not pay for it.
            self.set_chase_enabled(True)

        # Put the robot on the start line. This is now REQUIRED, not a convenience:
        # nav stages are built per environment, so the stage opens at whatever pose it
        # was authored with rather than at this episode's. It also fixes something
        # that was quietly wrong before -- running twice in a row without pressing
        # Reset used to start the second run from wherever the first one stopped, and
        # score it against an `initial_distance` measured from there.
        self.base.reset_to(tuple(ep.start), ep.start_yaw_rad)
        for _ in range(30):  # let the teleport settle before anything measures it
            self.kit.update()

        self.base.sync_from_sim()
        self.guard.interventions = 0
        self.recovery.reset()

        start_pos = self.base.position()
        initial_distance = math.dist(start_pos[:2], ep.goal[:2])
        # `ep.goal` is the goal of the episode's OWN instruction. If the user typed
        # something else, every distance below still measures toward that original
        # goal, and reporting it unlabelled would be a plain lie -- "distance to goal
        # 14.2 m" while the robot correctly drives somewhere else entirely. There is
        # no goal coordinate for a free-text instruction and there cannot be one: the
        # whole point is that the policy is never given coordinates. So the numbers
        # stay (they are still the honest distance to a fixed point, useful for
        # watching motion) and the UI is told not to score them.
        scored = instruction.strip() == ep.instruction.strip()
        self._set(
            state="running",
            scored=scored,
            instruction=instruction,
            controller=self.controller_name,
            initial_distance_m=round(initial_distance, 3),
            distance_m=round(initial_distance, 3),
            policy_calls=0,
            steps=0,
            guard_interventions=0,
            reasoning="",
            message="",
        )

        command = Command(0.0, 0.0, 0.0)
        trace: list[tuple[float, float, float]] = [(*start_pos[:2], self.base.yaw)]
        # (t, asked_deg, reach_m, goal_rel_deg, guide_deg|None, plan_speed, staleness_s)
        plans: list[tuple[float, float, float, float, float | None, float, float]] = []
        path_length = 0.0
        policy_calls = 0
        # Forced re-decisions after a reverse, and the ones the server refused. Counted
        # apart because they mean different things: the first is the recovery working,
        # the second is a server that has no /replan route and a robot still being handed
        # the plan that wedged it.
        replans = 0
        replans_failed = 0
        prev_pos = start_pos
        step = 0
        t0 = time.time()
        success = False
        timed_out = False
        history = WaypointHistory(PHYSICS_DT)
        history.observe(0, start_pos, self.base.yaw)
        frames = FrameHistory()

        # Where the robot was each time the server started a new background VLM
        # generation: (step, x, y, yaw). The plan we get back is built on a KV cache
        # that is one generation old, so `[0]` -- the second-to-last start, the one that
        # actually produced the cache in use -- is the reference for how stale it is.
        # Capped at two entries for the same reason DynaNav caps it at two
        # (nova_carter_test_ticvla.py:952, `pop(0)` once the list exceeds 2): the older
        # ones describe caches that have already been superseded, and referencing them
        # would make the reported delay grow without bound.
        gen_starts: list[tuple[int, float, float, float]] = []

        # Physics steps between decisions, and whether the robot waits for each one. In
        # synchronous mode the wait is a real stop -- `kit.update()` is not reached while
        # the HTTP call blocks -- so `think_wall_s` accumulates time the robot spent
        # standing still, which is the price this mode pays and therefore the number that
        # has to be reported next to its result.
        replan_steps = plan_period_steps(ep.replan_every_steps)
        synchronous = PLAN_PERIOD_S > 0
        think_wall_s = 0.0
        check_plan_period(self.policy, replan_steps, synchronous)

        while True:
            # SIM time, not wall time. DynaNav's timeouts (70-100 s) budget how long the
            # robot has to do the task, not how long the hardware takes to decide. That
            # mattered enormously when every call froze the robot for ~1.0-1.5 s; with
            # `predict_async` the freeze is down to the action expert alone, so the two
            # clocks now run much closer together. Still sim time: the gap is smaller,
            # not gone, and a wall clock would silently re-scale every episode's budget
            # with GPU load -- a stopwatch bug that reads as a navigation failure.
            sim_time = step * PHYSICS_DT
            wall_time = time.time() - t0
            position = self.base.position()
            distance = math.dist(position[:2], ep.goal[:2])

            # Third-person capture, BEFORE the exit checks so the last frame written is
            # the robot at the goal rather than a metre short of it -- the arrival is the
            # one moment of an episode a viewer most wants to see.
            #
            # The `or` is not redundant with the modulus. A decision frame must ALWAYS have
            # a third-person twin, because that pairing is what the video is built on, and
            # `replan_every_steps` is not guaranteed to be a multiple of `CHASE_EVERY` --
            # tune either one and the twins would start going missing, silently, on exactly
            # the frames that carry the reasoning text.
            if step % CHASE_EVERY == 0 or step % replan_steps == 0:
                self._capture_chase(step)

            if distance <= ep.success_threshold_m:
                success = True
                break
            if sim_time > ep.timeout_s:
                timed_out = True
                break
            if self._abort.is_set():
                self._set(message="stopped by user")
                break

            # --- replan ---------------------------------------------------
            if step % replan_steps == 0:
                # The third-person twin of this decision was grabbed at the top of the
                # loop, a few lines and no simulation steps ago, so `chase_<step>.jpg` and
                # `menu_<step>.jpg` are still the same instant seen two ways.
                frame_path = self.camera.grab_to_file()
                if frame_path is None:
                    # An empty render means the sensor pipeline is not up. Driving on
                    # a stale plan here would look fine and mean nothing.
                    self._set(message="camera returned an empty frame; holding")
                    for _ in range(10):
                        self.kit.update()
                    continue

                frames.add(sim_time, frame_path)
                self.latest_jpeg = frame_path.read_bytes()

                # How stale is the thinking behind the plan we are about to get, and
                # how far has the robot travelled since that thinking began? Measured
                # against the generation that produced the cache now in use, and
                # expressed in THAT moment's body frame (FLU: +x forward, +y left) --
                # the frame the model was reasoning in. Rotating by -yaw_ref is the 2D
                # form of DynaNav's `R_start.T @ delta_world`
                # (nova_carter_test_ticvla.py:815-822); we have a yaw where they have a
                # quaternion, so the matrix collapses to a sin/cos pair.
                if synchronous:
                    # Nothing to compensate for. The call below blocks, and the loop does
                    # not reach `kit.update()` until it returns, so the simulation is
                    # frozen for the whole generation: no steps, no sim time, and a robot
                    # that has not moved a millimetre since the frame it is being asked
                    # about. These zeros are not a simplification, they are the
                    # measurement -- exactly as this file's header says of the
                    # synchronous era that preceded the async switch.
                    time_delay, dx, dy = 0.0, 0.0, 0.0
                elif gen_starts:
                    ref_step, ref_x, ref_y, ref_yaw = gen_starts[0]
                    time_delay = (step - ref_step) * PHYSICS_DT
                    dxw = position[0] - ref_x
                    dyw = position[1] - ref_y
                    cos_r, sin_r = math.cos(ref_yaw), math.sin(ref_yaw)
                    dx = cos_r * dxw + sin_r * dyw
                    dy = -sin_r * dxw + cos_r * dyw
                else:
                    # Nothing has been generated yet, so the first call is genuinely
                    # not stale -- it blocks until the first cache exists.
                    time_delay, dx, dy = 0.0, 0.0, 0.0

                try:
                    out = self.policy.predict(
                        # Four frames at 3 s spacing, oldest first -- TIC-VLA is a
                        # video model and its system prompt says so. One still gave
                        # it no evidence it had been moving at all.
                        image_paths=frames.sample(sim_time),
                        instruction=instruction,
                        # TICVLA unpacks the 6-form as [vx, vy, _, yaw_speed, dx, dy].
                        # dx/dy is the displacement since the VLM started thinking,
                        # which is real now that the server runs `predict_async` and the
                        # robot keeps driving through a generation. It used to be hard
                        # 0.0, and that was the honest value while inference was
                        # synchronous and the robot stood still for every call -- the
                        # zero described the world accurately. It would be a lie here.
                        robot_state=[
                            command.vx, command.vy, 0.0, command.omega, dx, dy,
                        ],
                        current_step=step,
                        # Paired with dx/dy above: together they are the whole of the
                        # paper's latency compensation. The action head is not asked to
                        # guess that its cache is old, it is told, in seconds.
                        time_delay=time_delay,
                        # What the robot has already done. Omitting this (the default
                        # "") is why every plan used to come back as a fresh straight
                        # line: with no record of having driven for the last minute,
                        # "go forward" is a perfectly good answer every single time.
                        # DynaNav treats an empty value here as a hard error.
                        previous_waypoints_text=history.prompt_text(sim_time),
                        robot_type=ep.robot_type,
                        # The one thing about the last few seconds the camera cannot
                        # show. After a reverse the view is the same obstacle from a
                        # metre further back, which looks like an ordinary approach and
                        # is not -- it is the approach that already failed. True for a
                        # few seconds after the manoeuvre, and it stops the moment the
                        # robot has driven clear of where it was released.
                        recovered=self.recovery.memory_live(sim_time, position),
                        # ...and WHICH manoeuvre, because the two need opposite
                        # sentences. After a wedge, something was in the way; after a
                        # balk, nothing was and the robot stopped anyway. Telling the
                        # model the wrong one of those is worse than telling it neither.
                        recovery_kind=self.recovery.memory_kind(sim_time, position) or "",
                        # The other half of "give it some history", and the half no
                        # prompt can supply: one frame of a stationary robot is the same
                        # picture as one frame of a moving one, so without a number from
                        # here the model cannot know that the last four decisions
                        # achieved nothing. Measured where the position is known.
                        stalled_s=self.recovery.stalled_s(sim_time, position),
                        # Do not hand back the cached plan: block until one exists for
                        # the frame grabbed a few lines above. The robot is already
                        # stopped -- this call is what stops it -- so the wait buys the
                        # guarantee that no metre of this episode is driven on thinking
                        # that predates the view it was driven through.
                        wait_fresh=synchronous,
                    )
                except PolicyServerError as exc:
                    self._set(state="error", message=str(exc))
                    raise

                import numpy as np

                # Non-None only when this call kicked off a NEW background generation;
                # the server skips starting one while another is still running, so most
                # calls report nothing and leave the reference where it was.
                gen_step = out.get("vlm_generation_start_step")
                if gen_step is not None:
                    gen_starts.append((int(gen_step), position[0], position[1], self.base.yaw))
                    del gen_starts[:-2]

                waypoints = np.asarray(out["waypoints"], dtype=float)
                # The text head reaches 9 s; the action head stops at 3. Only the
                # `guided` controller acts on this, but it is parsed and logged for
                # every controller so the two channels can be compared on the same
                # run instead of across runs.
                guidance = parse_guidance(out.get("reasoning"))
                command = controller(waypoints, guidance)
                policy_calls += 1
                # In synchronous mode this is time the robot spent stopped. It is the
                # entire price of the mode, and it does NOT appear in `elapsed_s`,
                # because sim time does not advance while the call blocks -- so without
                # this field a synchronous run and an asynchronous one look equally cheap
                # and the trade being made is invisible in the results file.
                think_wall_s += float(out.get("latency_s") or 0.0)

                # Record what was ASKED for, next to where the robot was, so the two
                # can be compared afterwards. Both controllers steer at the same
                # lookahead point, so this is the controller-independent statement of
                # the policy's intent -- and `goal_rel` is what it would have had to
                # say to be right. The policy is never told goal_rel; it is scoring
                # only, exactly like `episode.goal` itself.
                x_l, y_l, reach = _lookahead_point(waypoints, DYNANAV_LOOKAHEAD_M)
                goal_rel = math.degrees(
                    math.atan2(ep.goal[1] - position[1], ep.goal[0] - position[0])
                    - self.base.yaw
                )
                guide_deg = guidance_heading(guidance, GUIDANCE_HORIZON_S)
                plans.append((
                    round(sim_time, 2),
                    round(math.degrees(math.atan2(y_l, x_l)), 2),
                    round(reach, 3),
                    round((goal_rel + 180.0) % 360.0 - 180.0, 2),
                    round(math.degrees(guide_deg), 2) if guide_deg is not None else None,
                    round(plan_speed(waypoints), 3),
                    # How old the thinking behind this plan was, in seconds of sim
                    # time. Under synchronous inference this was 0.0 by construction
                    # and not worth recording; with `predict_async` it is the single
                    # number that says whether the async switch is operating in the
                    # regime the paper trained for. The action head plans 3.0 s
                    # ahead, so a delay approaching that means the robot is being
                    # steered by a plan that has nearly expired.
                    round(time_delay, 3),
                ))
                self._set(
                    policy_calls=policy_calls,
                    reasoning=(out.get("reasoning") or "")[:600],
                )

            # --- recover, then guard, then step ----------------------------
            # Recovery runs BEFORE the guard, and the guard then sees the reversing
            # command: `check` casts along the direction of travel, so a negative vx
            # aims the fan backwards and the manoeuvre is protected by the same code
            # that protects driving forward.
            drive, recovering = self.recovery.update(
                sim_time, position, self.base.yaw, command)
            if self.recovery.take_release():
                # The manoeuvre just ended. The plan the server is holding was generated
                # from INSIDE the wedge and, on every wedge worth recovering from, says
                # STOP -- so re-serving it drives the robot straight back into the desk
                # it just backed out of, and the reverse bought nothing. Dropping it
                # forces the next call to generate on the backed-off view. It also
                # invalidates any generation already in flight, which was started on the
                # old view too.
                #
                # `/replan` and not `/reset`: reset rebuilds the recording directory, and
                # with no run name that means None, which would silently stop recording
                # the decisions for the rest of the episode -- including the ones this
                # whole mechanism exists to produce.
                try:
                    self.policy.replan()
                    replans += 1
                except PolicyServerError as exc:
                    # Not fatal and not silent. A server without the route (server.py)
                    # holds no cached plan to clear, so there is nothing to fail at; the
                    # robot keeps driving either way and the count says what happened.
                    replans_failed += 1
                    print(f"[nav] replan after recovery failed: {exc}", flush=True)
            guard = self.guard.check(position, self.base.yaw, drive.vx, drive.vy)
            self.base.apply(
                # Already guarded, and directionally: what survives is the part of the
                # command that was not driving into anything.
                guard.vx,
                guard.vy,
                # Let the robot keep turning when the guard stops translation --
                # otherwise it wedges against a wall with no way to face away from it.
                # `drive.omega` rather than `command.omega`: while reversing it is 0,
                # and turning mid-reverse would take the robot out of the footprint it
                # just vacated, which is the only space this manoeuvre knows is free.
                drive.omega,
                PHYSICS_DT,
            )
            self.kit.update()

            new_pos = self.base.position()
            path_length += math.dist(new_pos[:2], prev_pos[:2])
            prev_pos = new_pos
            step += 1
            history.observe(step, new_pos, self.base.yaw)

            # Refresh the third-person view ~20x per simulated second. The old value
            # was every 15 steps, chosen when a replan cost ~1 s of blocking inference
            # and the argument was "at least don't tie it to the replan cadence". That
            # argument expired with `predict_async`, and 15 steps was leaving the view
            # at 1.9 fps of wall clock -- a slideshow.
            #
            # Raising it is nearly free, which is not obvious: Kit renders this camera
            # EVERY frame whether or not anyone reads it (camera_source.py's docstring
            # says so), so the throttle was rationing something already paid for. What
            # a refresh actually costs is the readback plus the JPEG encode. Measured
            # on the hospital stage: 0.27 ms readback, 1.70 ms for the whole
            # `grab_jpeg()`, against a 34.6 ms step in a real run. Every 3 steps is
            # therefore +1.6% sim time for 5x the frame rate.
            #
            # Do not push this to every step. 28 fps would cost ~5% and the browser
            # cannot consume it anyway -- the page fetches /chase.jpg on its own timer,
            # and these two numbers have to be raised together or neither moves.
            #
            # At the default `CHASE_EVERY` of 2 the capture at the top of the loop is
            # already faster than this, and running both would pay for two readbacks to
            # show one picture. This exists for the case where somebody raises
            # `NAV_CHASE_EVERY` to make a long episode cheaper to record -- the LIVE view
            # should not get slower because the RECORDING got sparser. Two rates, two
            # purposes, and the faster of them wins.
            if CHASE_EVERY > 3 and step % 3 == 0:
                self._update_chase()

            if step % 30 == 0:
                trace.append((*new_pos[:2], self.base.yaw))
                self._set(
                    steps=step,
                    distance_m=round(distance, 3),
                    elapsed_s=round(sim_time, 1),
                    wall_s=round(wall_time, 1),
                    guard_interventions=self.guard.interventions,
                    message=(
                        (f"backing off after stalling ({self.recovery.balks})"
                         if self.recovery.mode == "balk" else
                         f"backing out of a wedge ({self.recovery.engagements})")
                        if recovering else
                        # Checked before the guard line on purpose: right after a reverse
                        # the obstacle is usually still inside the slow band, so "blocked
                        # by desk" is true and is not the interesting half of what is
                        # happening. This is the half a watcher cannot infer from the
                        # picture.
                        "backed off -- asking where to go now"
                        if self.recovery.memory_live(sim_time, new_pos) else
                        f"blocked by {guard.hit_prim.split('/')[-1]} at {guard.distance_m:.2f} m"
                        if guard.blocked else ""
                    ),
                )

        self.base.stop()
        final_pos = self.base.position()
        final_distance = math.dist(final_pos[:2], ep.goal[:2])
        trace.append((*final_pos[:2], self.base.yaw))

        result = EpisodeResult(
            episode=ep.name,
            instruction=instruction,
            success=success,
            final_distance_m=final_distance,
            initial_distance_m=initial_distance,
            elapsed_s=step * PHYSICS_DT,
            wall_s=time.time() - t0,
            steps=step,
            policy_calls=policy_calls,
            path_length_m=path_length,
            guard_interventions=self.guard.interventions,
            recoveries=self.recovery.engagements,
            # A subset of `recoveries`, not a second total. Reported separately because the
            # two say different things about the run: a wedge means the robot drove into
            # something, a balk means the policy stopped without arriving. An episode with
            # four wedges and one with four balks have nothing in common, and a single
            # number would report them as the same run.
            balks=self.recovery.balks,
            recoveries_blocked_behind=self.recovery.blocked_behind,
            recovery_replans=replans,
            recovery_replans_failed=replans_failed,
            timed_out=timed_out,
            controller=self.controller_name,
            policy=self.policy_label(),
            plan_period_s=(replan_steps * PHYSICS_DT) if synchronous else 0.0,
            think_wall_s=round(think_wall_s, 1),
            base_z_step_max_m=round(self.base.z_step_max, 6),
            base_z_span_m=(round(self.base.z_max - self.base.z_min, 6)
                           if self.base.z_max > self.base.z_min else 0.0),
            trace=trace,
            plans=plans,
        )
        # Save unconditionally, including aborted runs. The run you most want the trace
        # for is the one that went wrong, and that is exactly the one someone stops early.
        try:
            saved = result.save()
            print(f"\nrun saved: {saved}")
        except OSError as exc:  # a full disk must not lose the episode itself
            print(f"\ncould not save run: {exc}")

        self._set(
            state="done",
            distance_m=round(final_distance, 3),
            steps=step,
            elapsed_s=round(result.elapsed_s, 1),
            message=result.summary().splitlines()[0],
        )
        return result

    def perform_reset(self) -> None:
        """Main thread only. Put the robot back on the episode's start line.

        Applies a pending episode switch first, so Reset doubles as "show me that
        episode's starting view" -- the natural way to browse an environment's
        episodes without committing to a run.
        """
        if self._pending_episode:
            name, self._pending_episode = self._pending_episode, None
            if not self.check_episode(name):
                self.episode = self.runnable_episodes()[name]
                self._set(episode=self.episode.name, instruction=self.episode.instruction)
        ep = self.episode
        self._reset.clear()
        self._abort.clear()
        # Drop anything queued. Hitting Reset means "start over", so a job submitted
        # before the reset must not fire immediately afterwards.
        while self.take_job(timeout=0.0) is not None:
            pass

        self.base.reset_to(tuple(ep.start), ep.start_yaw_rad)
        self.guard.interventions = 0
        # Clear the policy's KV cache as well. Without this the model carries the
        # previous attempt's context into a run that is supposed to start fresh, and
        # the "reset" run is not comparable to a first one.
        try:
            self.policy.reset()
        except PolicyServerError as exc:
            self._set(message=f"reset the robot, but the policy server did not: {exc}")

        # Let the teleport settle before anyone renders or measures.
        for _ in range(30):
            self.kit.update()
        self._preview_t = 0.0  # force a refresh; both panels must show the start pose
        self._update_idle_views()

        self._set(
            state="idle", distance_m=None, initial_distance_m=None, elapsed_s=0.0,
            wall_s=0.0, steps=0, policy_calls=0, guard_interventions=0, reasoning="",
            message=f"reset to start: {tuple(round(v, 2) for v in ep.start)} "
                    f"@ {ep.start_yaw_deg:g} deg",
        )

    def idle(self, seconds: float = 0.05) -> None:
        """Keep Kit alive between jobs so the viewport and camera stay live."""
        for _ in range(max(1, int(seconds / PHYSICS_DT))):
            self.kit.update()
        self._update_idle_views()

    def close(self) -> None:
        if self.kit is not None:
            self.kit.close()


def main() -> int:
    episodes = load_episodes()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="warehouse", choices=sorted(episodes))
    ap.add_argument(
        "--controller", default="braking",
        choices=["braking", "pursuit", "holonomic", "guided"]
    )
    ap.add_argument("--instruction", default=None,
                    help="Override the benchmark instruction (the UI's job, mostly).")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=8765)
    ap.add_argument("--gui", action="store_true", help="Run with a window instead of headless.")
    ap.add_argument("--ui-port", type=int, default=None,
                    help="Also serve the prompt UI on this port.")
    args = ap.parse_args()

    ep = load_episode(args.episode)
    runner = NavigationRunner(
        episode=ep,
        controller_name=args.controller,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        headless=not args.gui,
    )
    runner.setup()

    if args.ui_port:
        from ui_bridge import serve_ui  # local import: only needed in UI mode

        serve_ui(runner, port=args.ui_port)
        print(f"\n  UI: http://127.0.0.1:{args.ui_port}\n")
        # The UI thread submits jobs; this thread just keeps Kit turning.
        try:
            while True:
                # Reset is checked before jobs: if the user hit Reset while a run was
                # queued, they meant "start over", not "run the stale job first".
                if runner.reset_pending():
                    runner.perform_reset()
                    continue
                job = runner.take_job(timeout=0.05)
                if job is None:
                    runner.idle()
                    continue
                try:
                    result = runner.run_episode(
                        job["instruction"], job.get("controller"), job.get("episode")
                    )
                except ValueError as exc:
                    # An episode from another environment slipped past the API check.
                    # Report it and keep serving -- the UI is still usable.
                    runner._set(state="idle", message=str(exc))
                    print(f"\nrejected job: {exc}", flush=True)
                    continue
                runner.last_result = result
                print("\n" + result.summary(), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            runner.close()
        return 0

    result = runner.run_episode(args.instruction)
    print("\n" + result.summary())
    runner.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
