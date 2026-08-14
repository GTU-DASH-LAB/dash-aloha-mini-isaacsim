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
        self._initialized = False

    def sync_from_sim(self) -> None:
        """Read the true pose out of the sim.

        Called once at episode start, and never again mid-episode: the integrated yaw
        below IS the authority afterwards. Re-reading it every step would fight the
        teleport (the pose we just wrote is the pose we would read back).
        """
        _, orientations = self.art.get_world_poses()
        w, _x, _y, z = orientations[0]
        self._yaw = 2.0 * math.atan2(z, w)
        self._initialized = True

    @property
    def yaw(self) -> float:
        return self._yaw

    def position(self) -> tuple[float, float, float]:
        positions, _ = self.art.get_world_poses()
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

        # Z is carried through untouched, never recomputed. Forcing a target height
        # against the contact solver caused measurable drift before (0.0078 -> 0.0611 m
        # over 3 s) -- let gravity and the wheel contacts settle it.
        new_positions = np.array([[t[0] + dx, t[1] + dy, t[2]]], dtype=np.float32)
        half = self._yaw / 2.0
        new_orientations = np.array(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=np.float32
        )
        self.art.set_world_poses(positions=new_positions, orientations=new_orientations)

    def stop(self) -> None:
        self.apply(0.0, 0.0, 0.0, 0.0)
