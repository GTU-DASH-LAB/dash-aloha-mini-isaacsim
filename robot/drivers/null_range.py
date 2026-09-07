#!/usr/bin/env python3
"""A `RangeSensor` that reports "I have measured nothing", which is not "nothing is there".

The distinction is the entire content of this file, and it is worth the file.

An empty sweep -- `[]` -- is a positive claim: the sensor looked and the world is clear.
The collision guard would let every velocity through and the policy server would mark
every arc drivable, both correctly, on a claim nobody actually made. `None` says the
sweep is not available, which is the truth when there is no sensor, and every consumer
already handles it: `ScanBuffer.update(None)` keeps the previous sweep, `scan_points()`
returns None until the first real one, and `CollisionGuard` built with `scan=None` is
simply a guard with nothing to see.

So passing this object and passing no sensor at all end in the same place. It exists to
be an explicit, greppable statement in a config -- and because the constructor is the
right spot for the warning below, where somebody reads it before the first drive.

RUNNING WITHOUT A RANGE SENSOR IS SUPPORTED AND IS HOW THE BASELINE WAS MEASURED. On the
19-episode benchmark the lidar arm and the no-lidar arm both scored 10/19 -- five
episodes won, five lost, which is noise on that set and not a difference. The policy does
not need one.

What changes on hardware is the cost of being wrong. In simulation a collision is a
number in a results file. The guard is the only thing in this stack that can stop the
base without waiting ~3 s for the model to think, and without a range sensor it has
nothing to work with -- so an obstacle that enters the frame between two decisions is
hit at cruise speed. Three ultrasonic sensors satisfy `RangeSensor` and are enough.
"""

from __future__ import annotations


class NullRangeSensor:
    """Always returns None. See the module docstring for why that is not `[]`."""

    def __init__(self, announce: bool = True):
        if announce:
            print("[range] no range sensor: the collision guard is inactive and the "
                  "robot will only stop when the policy says so, up to one decision "
                  "period (~3 s) later. See robot/drivers/null_range.py.", flush=True)

    def scan(self) -> list[tuple[float, float]] | None:
        return None
