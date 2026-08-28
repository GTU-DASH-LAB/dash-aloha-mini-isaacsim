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

Usage:
    nav/run.sh --episode warehouse --controller braking
"""

from __future__ import annotations

import argparse
import math
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
from waypoint_history import WaypointHistory  # noqa: E402

PHYSICS_DT = 1.0 / 60.0

# Which of the three guidance horizons (3/6/9 s) gets logged in `plans`. Logged for
# every controller even though only `guided` steers on it, so the two channels can be
# compared within one run. Kept here rather than at the call site so the logged column
# always describes the same horizon the controller would have used.
GUIDANCE_HORIZON_S = 6.0


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
        self.chase_available: bool | None = None  # None = not tried yet
        self._chase: object | None = None
        self._preview_t = 0.0  # throttles the between-runs camera refresh

        self.kit = None
        self.art = None
        self.camera = None
        self.guard = None
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

        self._set(state="ready", message="waiting for policy server")
        info = self.policy.wait_until_ready()
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
        self.policy.reset()  # clear the KV cache between episodes

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
            if step % ep.replan_every_steps == 0:
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
                if gen_starts:
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

            # --- guard + step ---------------------------------------------
            guard = self.guard.check(position, self.base.yaw, command.vx, command.vy)
            self.base.apply(
                # Already guarded, and directionally: what survives is the part of the
                # command that was not driving into anything.
                guard.vx,
                guard.vy,
                # Let the robot keep turning when the guard stops translation --
                # otherwise it wedges against a wall with no way to face away from it.
                command.omega,
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
            if step % 3 == 0:
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
            timed_out=timed_out,
            controller=self.controller_name,
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
