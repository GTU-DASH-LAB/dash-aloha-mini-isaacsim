"""Velocity command -> robot pose, via the kinematic base drive.

This is the same mechanism as scripts/control/control_terminal.py's
`_apply_kinematic_base_step`, factored out so the navigation loop and the REPL do not
drift apart. The reasoning behind teleporting rather than driving the wheels lives in
../../CLAUDE.md; the short version is that real wheel traction never held up, and
set_world_poses() is the only way to actually move a body during simulation (raw USD
transform writes are ignored while physics is stepping -- PhysX only pushes
physics->USD, verified directly).

The wheel joints are still commanded at the kinematically correct rate. They carry no
load, but a robot gliding along with motionless wheels looks broken, and the wheel
speeds are the honest LeKiwi kinematics either way.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from alohamini1_specs import body_to_wheel_speeds  # noqa: E402

WHEEL_JOINTS = ("wheel1", "wheel2", "wheel3")


class KinematicBase:
    def __init__(self, articulation, dof_names: list[str]):
        self.art = articulation
        self.dof_names = dof_names
        self._wheel_indices = [
            dof_names.index(name) for name in WHEEL_JOINTS if name in dof_names
        ]
        self._velocity_targets = np.zeros((1, len(dof_names)), dtype=np.float32)
        self._yaw = 0.0
        self._xy: np.ndarray | None = None
        self._initialized = False
        # Vertical motion the DRIVE never asked for. `apply()` writes x and y but carries
        # z straight through from whatever the contact solver last produced, so any
        # bouncing of the wheel spheres against the floor is written back into the pose
        # every step -- and the chase camera hangs off `base_link`, so it lands in the
        # recorded video as a shake. Measured rather than argued about: `z_step_max` is
        # the largest single-step change, which is what the eye reads as jitter, and the
        # min/max span says whether the robot is also settling or climbing.
        self.z_min = float("inf")
        self.z_max = float("-inf")
        self.z_step_max = 0.0
        self._z_prev: float | None = None

    def _note_z(self, z: float) -> None:
        z = float(z)
        self.z_min = min(self.z_min, z)
        self.z_max = max(self.z_max, z)
        if self._z_prev is not None:
            self.z_step_max = max(self.z_step_max, abs(z - self._z_prev))
        self._z_prev = z

    # If the sim's idea of where the robot is diverges from ours by more than this,
    # something moved it that was not us -- a fall, a reset, a physics correction --
    # and the sim wins. Well above per-step wheel drag (~1 cm) and well below any
    # distance that matters to an episode, so it catches real events without letting
    # drag leak back in.
    RESYNC_TOLERANCE_M = 0.25

    def sync_from_sim(self) -> None:
        """Read the true pose out of the sim.

        Called once at episode start, and never again mid-episode: the integrated pose
        below IS the authority afterwards. Re-reading it every step would fight the
        teleport (the pose we just wrote is the pose we would read back).

        **That was true of yaw and stated here, but position was re-read every step
        anyway, and it cost 11%.** The wheels are commanded at the kinematically
        correct rate and carry no load, but traction is not zero, so each step the
        robot was dragged a little on top of the `vx * dt` the teleport had already
        applied -- and because the next step read that dragged position back as its
        starting point, the error compounded instead of cancelling. Measured over a
        full episode: 0.663 m/s mean against a 0.600 m/s cap, 1.11x.

        That is not a cosmetic discrepancy at this scale. Over the 32.4 m
        `hospital_down_hallway` episode it is ~3.5 m of extra travel, and the episode
        failed by plateauing 2.88 m from its goal. Integrating position the same way
        yaw already was makes the commanded speed exact.
        """
        positions, orientations = self.art.get_world_poses()
        w, _x, _y, z = orientations[0]
        self._yaw = 2.0 * math.atan2(z, w)
        self._xy = np.array([positions[0][0], positions[0][1]], dtype=np.float64)
        self._initialized = True

    @property
    def yaw(self) -> float:
        return self._yaw

    def position(self) -> tuple[float, float, float]:
        positions, _ = self.art.get_world_poses()
        # Sampled here rather than inside `apply()`: this is called once per loop
        # iteration whatever the command was, so a robot held still by the guard or
        # standing through a synchronous decision is still watched. `apply()` returns
        # early on a zero command and would miss exactly those.
        self._note_z(positions[0][2])
        return tuple(float(v) for v in positions[0])

    def apply(self, vx: float, vy: float, omega: float, dt: float) -> None:
        if not self._initialized:
            self.sync_from_sim()

        # Spin the wheels at the visually correct rate even though they carry no load.
        for idx, speed in zip(self._wheel_indices, body_to_wheel_speeds(vx, vy, omega)):
            self._velocity_targets[0][idx] = speed
        self.art.set_joint_velocity_targets(self._velocity_targets)

        if vx == 0.0 and vy == 0.0 and omega == 0.0:
            return

        positions, _ = self.art.get_world_poses()
        t = positions[0]
        yaw = self._yaw
        # Body-frame velocity rotated into world. Note vy is real here: on the
        # holonomic controller the robot genuinely translates sideways.
        dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        self._yaw = yaw + omega * dt

        # Integrate x/y ourselves rather than reading the sim's value back and adding
        # to it. See sync_from_sim(): reading back folds wheel drag into the next
        # step's starting point, so the error compounds and the robot travels 1.11x
        # its commanded speed. Resync only if something OTHER than us has clearly
        # moved the robot -- otherwise the drag we are trying to exclude comes
        # straight back in through the tolerance check.
        if np.hypot(t[0] - self._xy[0], t[1] - self._xy[1]) > self.RESYNC_TOLERANCE_M:
            self._xy[:] = (t[0], t[1])
        self._xy += (dx, dy)

        # Z is carried through untouched, never recomputed. Forcing a target height
        # against the contact solver caused measurable drift before (0.0078 -> 0.0611 m
        # over 3 s) -- let gravity and the wheel contacts settle it.
        new_positions = np.array([[self._xy[0], self._xy[1], t[2]]], dtype=np.float32)
        half = self._yaw / 2.0
        new_orientations = np.array(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=np.float32
        )
        self.art.set_world_poses(positions=new_positions, orientations=new_orientations)

    def stop(self) -> None:
        self.apply(0.0, 0.0, 0.0, 0.0)

    def reset_to(self, position: tuple[float, float, float], yaw: float) -> None:
        """Teleport back to a known pose — what the UI's Reset button does.

        Sets the integrated pose to match rather than re-reading it afterwards: the
        teleport and the internal state must agree, or the first drive step after a
        reset moves in the direction the robot was facing *before* it. The same now
        goes for x/y, which is integrated too -- leaving `_xy` stale here would make
        the first step teleport the robot straight back to where it was.

        Zeroes the wheel velocity targets too. Skipping that leaves the wheels spinning
        at whatever they were last commanded, which looks like the robot is still
        driving while it sits on the start line.
        """
        self._velocity_targets[:] = 0.0
        self.art.set_joint_velocity_targets(self._velocity_targets)

        half = yaw / 2.0
        self.art.set_world_poses(
            positions=np.array([list(position)], dtype=np.float32),
            orientations=np.array(
                [[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=np.float32
            ),
        )
        self._yaw = yaw
        self._xy = np.array([position[0], position[1]], dtype=np.float64)
        self._initialized = True
