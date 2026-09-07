#!/usr/bin/env python3
"""The three things you implement to put this policy on your own robot.

    CameraSource   one forward-facing camera        REQUIRED
    Base           drive at a velocity, report odometry   REQUIRED
    RangeSensor    something that measures distance  OPTIONAL, and read the note

These are `typing.Protocol`s, so nothing needs to be subclassed and nothing needs to
import this file at runtime. A class with the right methods IS one of these. Write your
driver against your own hardware SDK, hand it to `robot/runner.py`, and the rest of the
stack -- pure-pursuit control, stuck recovery, the collision guard, frame history,
latency compensation -- is the same code that produced the measured results in
`nav/config/profiles/baseline.yaml`.

THE COORDINATE CONVENTION IS THE PART TO GET RIGHT. Everything below is body-frame FLU:

    +x  forward          +y  left          yaw  counter-clockwise, radians

This is DynaNav's convention and the policy was trained in it. It is not a formatting
preference: the policy's whole output is a list of displacements in this frame, so a
robot whose +y is right drives the mirror image of every plan it is given -- smoothly,
confidently, and into the wall on the other side. There is no error message for this.
The one-line check is in `robot/README.md` under "Before you drive anything", and it
takes about fifteen seconds to run.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CameraSource(Protocol):
    """One forward-facing camera. The only sensor the policy actually requires."""

    def grab(self) -> bytes | None:
        """Return the newest frame as ENCODED bytes -- a JPEG or PNG file's contents.

        Encoded, not a raw pixel buffer, for two reasons that both matter more than the
        convenience: the frames travel to the policy server over HTTP, and the server
        identifies re-sent frames by hashing exactly these bytes, which is what keeps a
        four-frame history from re-uploading three frames it already has.

        Return None if no frame is available yet. The runner treats that as "hold" and
        retries -- it does NOT drive on the previous plan, because a camera that has
        stopped producing is the case where a stale plan is most dangerous and least
        detectable.

        Roughly 448x448 is what the vision tower consumes; anything larger is downscaled
        server-side, so sending 4K only costs you upload time and JPEG encoding.
        """

    def close(self) -> None:
        """Release the device. Called once on shutdown, including after a crash."""


@runtime_checkable
class Base(Protocol):
    """The wheels, and whatever knows where they have taken you."""

    def drive(self, vx: float, vy: float, omega: float) -> None:
        """Command a body-frame velocity: m/s forward, m/s left, rad/s counter-clockwise.

        Called at the control rate (default 20 Hz), including with all-zero arguments --
        a zero here is an active command to stop, not an absence of one.

        `vy` is 0.0 for a differential-drive robot and the runner never asks it for
        anything else, because the shipping controller is pure pursuit. Accept and
        ignore it rather than raising, so a holonomic base can be dropped in later
        without changing this interface.

        Do NOT block. This is called from the control loop, and a driver that waits for
        a motor controller to acknowledge turns the guard's reaction time into whatever
        your serial link feels like today.
        """

    def stop(self) -> None:
        """Come to a halt now. Must be safe to call repeatedly and from any state.

        Called on shutdown, on Ctrl-C, on a policy-server error, and by the watchdog. It
        is the last thing that runs on every exit path, so it should not be able to
        throw -- if your transport can fail, catch it here and fail closed.
        """

    def odometry(self) -> tuple[float, float, float]:
        """Where you are now: (x_m, y_m, yaw_rad) in a fixed frame of YOUR choosing.

        Only differences are ever used -- how far the robot moved during a generation,
        and how far it moved during a stall window -- so the origin is arbitrary and
        drift over a long run is harmless. What is NOT harmless is the scale and the
        sign of yaw: `time_delay` compensation rotates a stale plan by the heading
        change since it was written, so a yaw that runs backwards un-rotates it twice.

        Wheel odometry is enough. This does not need a map, a filter, or a localiser.
        """


@runtime_checkable
class RangeSensor(Protocol):
    """Anything that measures distance to obstacles. Optional, and worth reading why.

    The measured baseline runs WITHOUT one: on the 19-episode benchmark the lidar arm
    and the no-lidar arm both scored 10/19, five episodes won and five lost, which is
    what noise looks like on that set rather than a difference. So the sensor is not
    what makes the policy work.

    It is still the thing to add first on real hardware, and for a reason the benchmark
    could not show. In simulation a collision is a number in a results file. On your
    robot it is your robot. The guard is the only component that can stop the base
    without waiting for the model to think, and without this it has nothing to see with.
    """

    def scan(self) -> list[tuple[float, float]] | None:
        """One sweep as body-frame (x_forward, y_left) points in metres.

        POINTS, not ranges, and the same format the policy server already accepts for
        `scan_points`. A range vector only means something next to the bearing
        convention and mount offset it was taken under, so shipping one would make two
        components agree about three things instead of one. Points have the frame
        already applied, and the same list feeds both consumers: the guard clips your
        velocity with it, and the server drops menu arcs you cannot physically drive.

        Drop returns that hit your own chassis before returning them -- nothing
        downstream can tell those from a wall 20 cm ahead, and a robot guarding against
        itself never moves.

        Three ultrasonic sensors are a valid implementation of this. So is a 2D lidar.
        Return None if the sweep is not ready; the runner treats that as "no scan this
        tick" and keeps the previous guard behaviour.
        """
