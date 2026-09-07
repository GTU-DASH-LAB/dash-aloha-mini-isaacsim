#!/usr/bin/env python3
"""The control loop, on real hardware. Ported from nav/sim/run_navigation.py minus Isaac.

    runner = RobotRunner(camera=my_cam, base=my_base, policy=PolicyClient(host="gpu-box"))
    runner.run("Go down the corridor and stop at the double doors.")

WHAT IS THE SAME AS THE SIMULATOR, and is the same code rather than a reimplementation:
pure-pursuit steering (`nav/sim/controllers.py`), the collision guard
(`nav/sim/collision_guard.py`, whose Isaac import is lazy and unreachable in scan mode),
stuck recovery, frame history, waypoint history, guidance parsing, and the policy client.
Reimplementing any of those would have produced a robot that behaves like the benchmark
except where it does not, with no way to tell which half you are looking at.

WHAT IS NECESSARILY DIFFERENT, and each difference is a decision:

1. TIME IS WALL-CLOCK. The simulator advances time only when it steps, so a blocking
   policy call there costs nothing -- the robot is genuinely frozen mid-air. Here a
   blocking call is a robot still rolling with nobody watching, so the policy call runs
   on its own thread and the control loop never waits for it. This is the single largest
   structural change and it exists for safety, not throughput.

2. ODOMETRY COMES FROM YOU. The simulator integrates a kinematic base and knows the
   answer exactly. Only differences are used here -- displacement during a generation,
   displacement during a stall window -- so drift is harmless and an origin is arbitrary.

3. THERE IS NO GOAL AND NO SUCCESS. The benchmark scores against a coordinate the policy
   is never told. On your robot nothing knows where the elevator is, so the run ends when
   you stop it, when the timeout expires, or when the model says it has arrived.

4. THERE IS A WATCHDOG. In simulation a hung loop is a hung process. Here it is a moving
   vehicle, so a separate thread stops the base if a control tick has not completed
   recently -- the case a `finally: base.stop()` cannot cover, because the loop is still
   running, it is just no longer arriving anywhere.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "nav" / "sim"))
sys.path.insert(0, str(_REPO / "nav" / "policy_server"))

from client import PolicyClient, PolicyServerError  # noqa: E402
from collision_guard import CollisionGuard  # noqa: E402
from controllers import Command, make_controller, plan_speed  # noqa: E402
from frame_history import FrameHistory  # noqa: E402
from guidance import parse_guidance  # noqa: E402
from stuck_recovery import StuckRecovery  # noqa: E402
from waypoint_history import WaypointHistory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LiveLimits  # noqa: E402
from sensing import ScanBuffer  # noqa: E402

# How many consecutive policy errors before the run gives up. One is too few -- a single
# dropped connection on a wifi link should not end a drive -- and unbounded is wrong,
# because a robot driving on a plan whose author stopped answering ten minutes ago is
# exactly the failure this whole file is arranged to prevent.
MAX_CONSECUTIVE_POLICY_ERRORS = 3


@dataclass
class Telemetry:
    """What the run looks like from outside. Read by the control API; never drives."""

    state: str = "idle"              # idle | driving | holding | recovering | stopped
    message: str = ""
    instruction: str = ""
    elapsed_s: float = 0.0
    decisions: int = 0
    policy_errors: int = 0
    guard_interventions: int = 0
    recoveries: int = 0
    pivots: int = 0
    last_reasoning: str = ""
    vx: float = 0.0
    omega: float = 0.0
    plan_age_s: float = 0.0
    plan_speed_mps: float = 0.0
    clearance_m: float = float("inf")
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set(self, **kw) -> None:
        with self._lock:
            self.__dict__.update(kw)


class _Watchdog:
    """Stops the base if the control loop stops arriving.

    A `finally: base.stop()` covers the loop EXITING. It does not cover the loop still
    running while wedged -- a camera driver blocked on a USB reset, a socket read with no
    timeout -- where the last velocity command stands and the robot keeps going. That is
    the case this thread exists for, and it is the one that happens on hardware.
    """

    def __init__(self, base, timeout_s: float):
        self.base = base
        self.timeout_s = timeout_s
        self.last_tick = time.monotonic()
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def tick(self) -> None:
        self.last_tick = time.monotonic()

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.timeout_s / 4.0):
            if time.monotonic() - self.last_tick > self.timeout_s:
                if not self.tripped:
                    self.tripped = True
                    print(f"[watchdog] no control tick for {self.timeout_s}s -- stopping",
                          flush=True)
                try:
                    self.base.stop()
                except Exception as exc:  # noqa: BLE001 -- must not die while stopping
                    print(f"[watchdog] base.stop() raised: {exc}", flush=True)
            else:
                self.tripped = False


class RobotRunner:
    def __init__(
        self,
        camera,
        base,
        policy: PolicyClient,
        limits: LiveLimits,
        range_sensor=None,
        frame_dir: Path | str = "/tmp/qvla-robot-frames",
        controller_name: str = "braking",
        robot_type: str = "wheeled robot",
    ):
        self.camera = camera
        self.base = base
        self.policy = policy
        self.limits = limits
        self.range_sensor = range_sensor
        self.controller_name = controller_name
        self.robot_type = robot_type
        self.telemetry = Telemetry()

        self.frame_dir = Path(frame_dir)
        self.frame_dir.mkdir(parents=True, exist_ok=True)

        lim = limits.current()
        self.scan = ScanBuffer(chassis_radius_m=lim.chassis_radius_m)
        # `scan=` is what keeps this Isaac-free: with it the guard reads the buffer and
        # never reaches `_scene_query`, which is the only place it would import omni.
        self.guard = CollisionGuard(
            stop_distance_m=lim.guard_stop_m,
            slow_distance_m=lim.guard_slow_m,
            fan_half_angle_deg=lim.guard_fan_half_angle_deg,
            chassis_radius_m=lim.chassis_radius_m,
            scan=self.scan if range_sensor is not None else None,
        )
        self.recovery = StuckRecovery(self.guard)
        self._abort = threading.Event()

        # Set by the decision thread, consumed by the control loop. A plain slot rather
        # than a queue: a plan that arrived while a newer one was already requested is
        # not a backlog to work through, it is out of date, and the newest is the only
        # one worth driving.
        self._pending: dict | None = None
        self._pending_lock = threading.Lock()
        self._decision_thread: threading.Thread | None = None
        # The cruise speed the SERVER currently believes in. Tracked so a mid-drive edit
        # to robot.yaml reaches both halves; see `_push_speed`.
        self._server_cruise: float | None = None
        self._server_floor: float | None = None   # learned from /health, never hardcoded
        self._floor_warned: float | None = None

    # ----------------------------------------------------------------------------------
    def stop(self) -> None:
        """Ask the run to end. Safe from another thread -- the control API calls it."""
        self._abort.set()

    # ----------------------------------------------------------------------------------
    def _grab_frame(self, seq: int) -> Path | None:
        data = self.camera.grab()
        if not data:
            return None
        # Written to disk even though the bytes are what gets sent. Costs a few ms and
        # buys the thing you will want at 2am: the exact frames the policy saw, next to
        # the decisions it made on them. `FrameHistory` is also path-based, and reusing
        # it unchanged is worth more than avoiding the write.
        path = self.frame_dir / f"frame_{seq:06d}.jpg"
        path.write_bytes(data)
        return path

    def _start_decision(self, frames: list[str], instruction: str, step: int,
                        time_delay: float, dx: float, dy: float, command: Command,
                        history_text: str, position, yaw: float, now: float) -> None:
        """Fire the policy call on its own thread. The control loop keeps ticking.

        `now` is elapsed seconds since the run began -- the SAME clock `recovery.update`
        is fed. Passing `time.monotonic()` here instead type-checks, runs, and tells the
        model the robot has been stalled for about 1023089 seconds, which is the machine's
        uptime; it then plans for an emergency that is not happening.
        """

        def _work() -> None:
            try:
                self._push_speed()
                out = self.policy.predict(
                    # The bytes, not the paths: the server reads `image_paths` off its
                    # OWN disk, and the whole point of a robot is that it does not share
                    # one with the GPU. Same frames, same oldest-first order.
                    images=[Path(f).read_bytes() for f in frames],
                    instruction=instruction,
                    robot_state=[command.vx, command.vy, 0.0, command.omega, dx, dy],
                    current_step=step,
                    time_delay=time_delay,
                    previous_waypoints_text=history_text,
                    robot_type=self.robot_type,
                    recovered=self.recovery.memory_live(now, position),
                    recovery_kind=self.recovery.memory_kind(now, position) or "",
                    stalled_s=self.recovery.stalled_s(now, position),
                    scan_points=self.scan.scan_points(),
                )
                out["_error"] = None
            except (PolicyServerError, OSError) as exc:
                out = {"_error": str(exc)}
            with self._pending_lock:
                self._pending = out

        self._decision_thread = threading.Thread(target=_work, daemon=True)
        self._decision_thread.start()

    def _push_speed(self) -> None:
        """Keep the server's cruise speed equal to the one in robot.yaml.

        Called from the DECISION thread, immediately before `predict`, and not from the
        control loop: this is a blocking HTTP round trip, and putting one of those in the
        20 Hz loop makes the guard's reaction time a property of the network.

        Why it has to happen at all. `cruise_mps` is applied in two places -- it caps the
        controller locally, and it sets the pace of the arcs the server plans, because
        the menu carries a direction and not a speed. Pushing it only at startup means an
        operator who lowers the speed mid-drive slows the robot's own controller while
        the server keeps issuing plans at the old pace, and `BrakingPursuitController`
        takes `min(v_max, plan_speed)` against a number that no longer describes this
        robot. The measurable symptom is a robot that is slower but still overshoots.
        """
        want = self.limits.current().cruise_mps
        # The server refuses anything under its creep floor -- the bottom of its approach
        # ramp -- and answers 400. Clamped here against the floor the server ITSELF
        # reports, rather than against a constant copied into this file, because a copied
        # constant is how the two halves end up disagreeing in the first place.
        #
        # Only the SERVER's pace is clamped. The local cap stays at `cruise_mps`, and the
        # braking controller takes the minimum, so asking for 0.15 on a small robot still
        # drives at 0.15 -- the arcs are just shaped for 0.20, which is the conservative
        # direction. Without the clamp this posts a 400 every decision forever and the
        # robot drives at a speed the operator was told it would not.
        if self._server_floor is not None and want < self._server_floor:
            if self._floor_warned != want:
                self._floor_warned = want
                print(f"[runner] cruise_mps {want} is under the server's creep floor "
                      f"{self._server_floor}; it will plan arcs at {self._server_floor} "
                      f"while the robot drives at {want}.", flush=True)
            want = self._server_floor
        if self._server_cruise is not None and abs(want - self._server_cruise) < 1e-6:
            return
        try:
            self.policy.menu_speed(want)
            self._server_cruise = want
        except (PolicyServerError, OSError) as exc:
            # Not fatal, and deliberately not retried on a timer: the next decision comes
            # in ~3 s and will try again. What must not happen is silence, because the
            # robot is now driving at a speed the operator did not choose.
            print(f"[runner] server kept cruise {self._server_cruise}; "
                  f"could not set {want}: {exc}", flush=True)

    def _take_result(self) -> dict | None:
        with self._pending_lock:
            out, self._pending = self._pending, None
        return out

    # ----------------------------------------------------------------------------------
    def run(self, instruction: str, timeout_s: float = 600.0) -> dict:
        """Drive until told to stop, until `timeout_s`, or until the policy gives up."""
        lim = self.limits.current()
        dt = 1.0 / lim.control_hz

        # The history spacing must follow the decision period, not sit at its own default:
        # `sample()` walks back in `interval_s` steps from now, so a 3 s history against a
        # 1 s decision period hands the model the same frame four times and calls it a
        # memory. That exact mismatch is what the mem2 arms were testing.
        frames = FrameHistory(interval_s=lim.decision_period_s)
        history = WaypointHistory(physics_dt=dt)
        controller = make_controller(
            self.controller_name, lim.max_speed_mps, lim.max_yaw_rate_radps)
        self.recovery.reset()
        self.guard.interventions = 0

        # Tell the server the cruise speed before the first decision. Thereafter every
        # decision re-checks it, so an edit to robot.yaml reaches both halves.
        self._server_cruise = None
        try:
            self._server_floor = float(self.policy.health().get("menu_creep_mps") or 0.0)
        except (PolicyServerError, OSError, TypeError, ValueError):
            self._server_floor = None   # `server.py` has no menu and no floor
        self._push_speed()
        try:
            self.policy.reset()
        except (PolicyServerError, OSError):
            pass

        watchdog = _Watchdog(self.base, lim.watchdog_s)
        watchdog.start()

        command = Command(0.0, 0.0, 0.0)
        gen_starts: list[tuple[float, float, float, float]] = []
        pivot_remaining = 0.0
        pivot_deadline = 0.0
        prev_yaw = self.base.odometry()[2]
        decisions = 0
        policy_errors = 0
        recoveries = 0
        pivots = 0
        seq = 0
        step = 0

        t0 = time.monotonic()
        # Zero, not `t0`. Every other time in this loop is `now`, i.e. seconds SINCE t0,
        # and mixing the two scales here costs nothing at import and everything at run:
        # `now >= next_decision_t` compares ~0 against a monotonic clock reading in the
        # hundreds of thousands, is never true, and the robot drives a whole episode
        # without ever calling the policy -- while the loop, the guard and the recovery
        # all run normally and the logs look healthy.
        next_decision_t = 0.0
        self.telemetry.set(state="driving", instruction=instruction, message="")

        try:
            while not self._abort.is_set():
                tick_start = time.monotonic()
                now = tick_start - t0
                if now > timeout_s:
                    self.telemetry.set(state="stopped", message="timeout")
                    break

                lim = self.limits.current()
                dt = 1.0 / lim.control_hz
                # Applied every tick rather than at construction, because the operator is
                # editing these while watching the robot -- a limit that needs a restart
                # is a limit that does not get tuned. See robot/config.py.
                controller.v_max = lim.max_speed_mps
                controller.w_max = lim.max_yaw_rate_radps
                self.guard.stop_distance_m = lim.guard_stop_m
                self.guard.slow_distance_m = lim.guard_slow_m
                self.scan.chassis_radius_m = lim.chassis_radius_m

                position = self.base.odometry()
                pos_xy = (position[0], position[1], 0.0)
                yaw = position[2]
                history.observe(step, pos_xy, yaw)

                if self.range_sensor is not None:
                    self.scan.update(self.range_sensor.scan())

                # --- collect a finished decision ------------------------------------
                out = self._take_result()
                if out is not None:
                    if out.get("_error"):
                        policy_errors += 1
                        self.telemetry.set(policy_errors=policy_errors,
                                           message=f"policy error: {out['_error']}")
                        command = Command(0.0, 0.0, 0.0)
                        if policy_errors >= MAX_CONSECUTIVE_POLICY_ERRORS:
                            self.telemetry.set(
                                state="stopped",
                                message=f"policy unreachable after "
                                        f"{policy_errors} attempts")
                            break
                    else:
                        policy_errors = 0
                        decisions += 1
                        command, pivot_remaining, pivot_deadline = self._apply(
                            out, controller, lim, now, gen_starts, pos_xy, yaw)
                        if pivot_remaining:
                            pivots += 1
                        self.telemetry.set(
                            decisions=decisions,
                            last_reasoning=(out.get("reasoning") or "")[:600],
                            policy_errors=0)

                # --- ask for the next decision --------------------------------------
                busy = (self._decision_thread is not None
                        and self._decision_thread.is_alive())
                if now >= next_decision_t and not busy and not pivot_remaining:
                    frame_path = self._grab_frame(seq)
                    if frame_path is None:
                        # A camera that has stopped producing is precisely when a stale
                        # plan is most dangerous and least visible. Hold, do not coast.
                        command = Command(0.0, 0.0, 0.0)
                        self.telemetry.set(state="holding",
                                           message="camera returned no frame")
                    else:
                        seq += 1
                        frames.add(now, frame_path)
                        time_delay, dx, dy = self._staleness(gen_starts, now, pos_xy, yaw)
                        gen_starts.append((now, position[0], position[1], yaw))
                        del gen_starts[:-2]
                        self._start_decision(
                            frames.sample(now), instruction, step, time_delay, dx, dy,
                            command, history.prompt_text(now), pos_xy, yaw, now)
                        next_decision_t = now + lim.decision_period_s
                        self.telemetry.set(state="driving", message="")

                # --- close the loop on an in-place turn ------------------------------
                if pivot_remaining > 0.0:
                    # Accumulated from the yaw the base ACTUALLY reached, wrapped to
                    # (-pi, pi] before it is added. Yaw is read as an angle, not as a
                    # winding: one tick across the +/-pi seam is a few thousandths of a
                    # radian of real rotation and 6.28 of arithmetic, which would finish
                    # any turn instantly at the moment the robot happens to face west.
                    pivot_remaining -= abs((yaw - prev_yaw + math.pi)
                                           % (2 * math.pi) - math.pi)
                    if pivot_remaining <= 0.0 or now >= pivot_deadline:
                        # Look again NOW, from the direction just gained. Holding the
                        # rest of the planning period would spend it standing still on a
                        # view the robot has already turned away from.
                        pivot_remaining = 0.0
                        command = Command(0.0, 0.0, 0.0)
                        next_decision_t = now
                        self.telemetry.set(state="driving", message="")
                prev_yaw = yaw

                # --- recover, then guard, then drive --------------------------------
                # Recovery runs BEFORE the guard, and the guard then sees the reversing
                # command: `check` casts along the direction of travel, so a negative vx
                # aims the fan backwards and the manoeuvre is protected by the same code
                # that protects driving forward.
                drive, recovering = self.recovery.update(now, pos_xy, yaw, command)
                if recovering:
                    # `engagements` and not a local counter: `recovering` is true on every
                    # tick of a manoeuvre, so incrementing here would report one reverse
                    # as forty and make the number unreadable next to the sim's.
                    recoveries = self.recovery.engagements
                    self.telemetry.set(
                        state="recovering",
                        message=f"backing out ({self.recovery.mode})")
                if self.recovery.take_release():
                    # The plan the server holds was generated from INSIDE the wedge and
                    # almost certainly says STOP, so re-serving it drives straight back
                    # into what was just reversed out of. /replan and not /reset: reset
                    # rebuilds the recording directory and would silently stop recording.
                    try:
                        self.policy.replan()
                    except (PolicyServerError, OSError):
                        pass

                if not lim.enabled:
                    # The master switch. Everything above still runs -- frames, policy,
                    # guard, telemetry -- so you can watch what it WOULD do.
                    drive = Command(0.0, 0.0, 0.0)
                    self.telemetry.set(state="holding", message="disabled in robot.yaml")

                guarded = self.guard.check(pos_xy, yaw, drive.vx, drive.vy)
                vx, vy = guarded.vx, guarded.vy
                omega = drive.omega
                if guarded.blocked:
                    self.telemetry.set(
                        state="holding",
                        message=f"guard: obstacle at {guarded.distance_m:.2f} m")

                self.base.drive(vx, vy, omega)
                self.telemetry.set(
                    elapsed_s=round(now, 1), vx=round(vx, 3), omega=round(omega, 3),
                    guard_interventions=self.guard.interventions,
                    recoveries=recoveries, pivots=pivots,
                    clearance_m=round(self.scan.forward_clearance_m(), 2),
                    position=(round(position[0], 2), round(position[1], 2),
                              round(yaw, 3)))

                watchdog.tick()
                step += 1
                slack = dt - (time.monotonic() - tick_start)
                if slack > 0:
                    time.sleep(slack)
        except KeyboardInterrupt:
            self.telemetry.set(state="stopped", message="interrupted")
        finally:
            watchdog.close()
            # Ordered: stop the wheels first, then release the camera. A camera close
            # that throws must not be what stands between a running robot and a halt.
            try:
                self.base.stop()
            finally:
                try:
                    self.camera.close()
                except Exception as exc:  # noqa: BLE001
                    print(f"[runner] camera.close() raised: {exc}", flush=True)

        return self.telemetry.snapshot()

    # ----------------------------------------------------------------------------------
    def _staleness(self, gen_starts, now: float, position, yaw: float):
        """How old the thinking behind the plan in hand is, and how far we moved since.

        Expressed in the body frame of the moment that thinking STARTED, which is the
        frame the model was reasoning in. The simulator rotates by -yaw_ref with a
        sin/cos pair because it has a yaw where DynaNav has a quaternion; same here.

        `gen_starts[0]` and not `[-1]`: under async the server returns whatever is
        CACHED and starts a new generation, so the plan in hand is the one before the
        generation now running. Using the newest would report one period of staleness
        for two periods of driving -- and `time_delay` is not a log field, the server
        slices the plan by it, so a wrong value discards real plan.
        """
        if not gen_starts:
            return 0.0, 0.0, 0.0
        ref_t, ref_x, ref_y, ref_yaw = gen_starts[0]
        dxw = position[0] - ref_x
        dyw = position[1] - ref_y
        cos_r, sin_r = math.cos(ref_yaw), math.sin(ref_yaw)
        return (now - ref_t,
                cos_r * dxw + sin_r * dyw,
                -sin_r * dxw + cos_r * dyw)

    def _apply(self, out: dict, controller, lim, now: float, gen_starts, position,
               yaw: float):
        """Turn one policy response into a command, plus any pivot it asked for."""
        waypoints = np.asarray(out["waypoints"], dtype=float)
        guidance = parse_guidance(out.get("reasoning"))
        command = controller(waypoints, guidance)

        # The in-place turn arrives BESIDE the waypoints because it cannot be in them: a
        # turn is every waypoint at the origin, which is bit-for-bit a STOP and is what
        # `controller` just built. A server that does not know the field sends nothing
        # and the robot stops, which is the conservative half of the ambiguity.
        pivot_rad = float(out.get("pivot_rad") or 0.0)
        if pivot_rad:
            command = Command(
                0.0, 0.0, math.copysign(lim.max_yaw_rate_radps, pivot_rad))
            # 2.5x the ideal duration: a bound, not a schedule.
            deadline = now + 2.5 * abs(pivot_rad) / max(lim.max_yaw_rate_radps, 1e-3)
            return command, abs(pivot_rad), deadline

        self.telemetry.set(
            plan_age_s=round(float(out.get("latency_s") or 0.0), 2),
            plan_speed_mps=round(plan_speed(waypoints), 3))
        return command, 0.0, 0.0
