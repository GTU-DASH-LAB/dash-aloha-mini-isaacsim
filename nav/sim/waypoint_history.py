"""The robot's own recent motion, in the exact text form TIC-VLA was trained on.

TIC-VLA's prompt has two halves. The image says what the world looks like; this string
says what the robot has already *done* about it. Without it the model is asked "what
should I do next?" with no notion that it has been driving in a straight line for the
last minute — and it answers, reasonably, "go straight", forever. That is precisely the
failure this repo saw: 32 m of path travelled to close 0.4 m of distance, every plan a
fresh 1.3 m straight line, the robot sailing past the aisle it was sent to and parking
in the perimeter wall.

It is not an optional enrichment. DynaNav builds this string on every inference and
raises `ValueError("ERROR: Empty previous waypoints text!")` if it comes out blank —
the reference implementation treats an empty value as an impossible state, and we were
passing exactly that on all 141 calls of a run.

THE FORMAT IS COPIED, NOT DESIGNED. It has to match
`ticvla/data/vlm_data.py:_format_previous_waypoints_text` character for character,
because that is the function that produced the training text. Reword the sentence and
the string still looks perfectly sensible to a human while sitting off the distribution
the model learned. Both branches matter: with no history yet, training still emitted the
"No waypoints available." line, so an empty string is not the zero case — that line is.

Sampling convention, ported from DynaNav (`nova_carter_test_ticvla.py:644-665`):

  * one sample per 1.0 s of SIM time (it samples `frame_idx % 30 == 0` at 30 Hz; we run
    physics at 60 Hz, so the divisor differs and the meaning does not),
  * each sample is the displacement over the PREVIOUS second, expressed in the body
    frame as it was at the START of that second (`R_prev.T @ delta_world`) — not the
    current frame, and not world,
  * the first sample is a literal (0,0,0) placeholder and is filtered out when the text
    is built, so the first real second of motion is the first thing reported.
"""

from __future__ import annotations

import math


class WaypointHistory:
    """Accumulates 1 s body-frame displacements and renders TIC-VLA's history prompt."""

    def __init__(self, physics_dt: float, sample_period_s: float = 1.0):
        self.sample_period_s = sample_period_s
        # Sample every N physics steps rather than on a float time comparison: step
        # count is exact, and sim time is derived from it anyway.
        self.steps_per_sample = max(1, round(sample_period_s / physics_dt))
        self.reset()

    def reset(self) -> None:
        """Between episodes. A carried-over history describes a different run."""
        self._samples: list[tuple[float, list[float]]] = []
        self._prev_pos: tuple[float, float, float] | None = None
        self._prev_yaw: float | None = None
        self._last_sampled_step: int | None = None

    def observe(self, step: int, position: tuple[float, float, float], yaw: float) -> None:
        """Call every physics step. Records a sample only when one is due."""
        if step % self.steps_per_sample != 0 or step == self._last_sampled_step:
            return
        self._last_sampled_step = step
        t = step * self.sample_period_s / self.steps_per_sample

        if self._prev_pos is None:
            # DynaNav appends a literal zero on the first tick and filters it out at
            # format time. Kept rather than skipped so the sample indices keep lining
            # up with the wall-clock seconds they represent.
            self._samples.append((t, [0.0, 0.0, 0.0]))
        else:
            dx = position[0] - self._prev_pos[0]
            dy = position[1] - self._prev_pos[1]
            dz = position[2] - self._prev_pos[2]
            # R_prev.T @ delta_world, written out for the planar case: the robot only
            # ever yaws, so the full rotation matrix would be three lines of numpy to
            # express one 2D rotation by -yaw_prev.
            c, s = math.cos(self._prev_yaw), math.sin(self._prev_yaw)
            self._samples.append((t, [c * dx + s * dy, -s * dx + c * dy, dz]))

        self._prev_pos = tuple(position)
        self._prev_yaw = yaw

    def prompt_text(self, elapsed_s: float) -> str:
        """The string TIC-VLA expects. Never empty — see this module's docstring."""
        parts = [
            f"({x:.2f}, {y:.2f}, {z:.2f})"
            for _t, (x, y, z) in self._samples
            if abs(x) >= 1e-6 or abs(y) >= 1e-6 or abs(z) >= 1e-6
        ]
        if not parts:
            return f"From 0.0s to current timestamp time is {elapsed_s:.1f}s. No waypoints available."
        return (
            f"From 0.0s to current timestamp time is {elapsed_s:.1f}s. "
            f"(a list of waypoints 1s in between): {', '.join(parts)}\n"
            "Each waypoint (x, y, z) is the displacement over the previous 1.0s. "
            "x is forward, y is left, z is up."
        )
