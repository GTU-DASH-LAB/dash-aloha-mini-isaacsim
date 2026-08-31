"""Back out of a wedge, and only out of a wedge.

Reported from watching a run: the robot nosed up to a desk and then held there for the
rest of the episode. Nothing in the stack is wrong at that moment -- the menu offers
seven forward arcs and a STOP, and when a surface is half a metre off the nose every
forward arc drives into it, so STOP is the only answer that is not a collision. The
frame does not change, so the next decision is the same one, and so is the one after
that. The policy is not stuck in the sense of being confused; it is out of options.

Giving it a reverse option is what this is, and it belongs here rather than on the menu.
A drawn arc is something the model can see and judge; reversing is a manoeuvre into
space that is behind the camera and therefore not in the picture at all. Asking a
vision model to pick a path it cannot see is asking it to guess.

TWO THINGS MAKE THIS SAFE, and they are the whole design:

**Reversing straight back is provably clear.** The robot was just standing where it is
about to reverse into, one metre ago. So `omega` stays 0 for the whole manoeuvre: the
moment it turns while backing, it leaves its own footprint and is driving blind into
space nothing has checked. The rear is guarded anyway -- `CollisionGuard.check` casts
along the direction of TRAVEL, so handing it a negative vx aims the fan backwards with
no new code -- but that guard is the second line, not the first.

**It cannot undo an arrival.** `run_navigation.py` breaks out of the control loop the
instant `distance <= success_threshold_m`, so a robot that has arrived is not driving
any more and this never runs on it. That is the reason the trigger can afford to be as
blunt as "not moving, with something close in front": the expensive mistake, backing
away from a goal already reached, is structurally impossible rather than tuned against.

The trigger is deliberately not a new threshold. `wedged` means the nearest thing in
front is inside the guard's OWN `stop_distance_m` -- i.e. exactly the distance at which
the guard cancels forward motion, which is the same as saying every forward arc on the
menu is unusable. Inventing a separate number here would let the two disagree, and then
the robot would either back off while it could still drive forward, or sit inside a
wedge the recovery did not recognise.
"""

from __future__ import annotations

import math

BACK_SPEED_MPS = 0.25     # slow: this is a blind-ish manoeuvre, not a getaway
BACK_LIMIT_M = 0.8        # how far back it will ever go in one recovery
STALL_WINDOW_S = 3.0      # about one generation; shorter and a slow creep reads as stuck
STALL_PROGRESS_M = 0.10   # moved less than this in the window = not making progress
CLEAR_MARGIN_M = 0.4      # extra room past the guard's stop distance before releasing
COOLDOWN_S = 2.0          # let the policy see and act on the new view before re-arming
MAX_CONSECUTIVE = 4       # a wedge this deep is a result, not something to keep poking


class StuckRecovery:
    """Watches for a wedge and drives the robot backwards out of it.

    Owns no raycasting of its own: front clearance is measured with the run's existing
    `CollisionGuard`, so the recovery and the guard can never disagree about what
    "blocked" means.
    """

    def __init__(self, guard, back_speed_mps: float = BACK_SPEED_MPS,
                 back_limit_m: float = BACK_LIMIT_M):
        self.guard = guard
        self.back_speed_mps = back_speed_mps
        self.back_limit_m = back_limit_m
        self.engagements = 0          # how many times it fired, for the result record
        self.blocked_behind = 0       # fired, and could not reverse either
        self._history: list[tuple[float, float, float]] = []   # (t, x, y)
        self._active = False
        self._backed_m = 0.0
        self._consecutive = 0
        self._release_t = -1e9
        self._start = (0.0, 0.0)
        self._engage_t = 0.0
        # Twice the time the manoeuvre should need, so a rear obstacle shows up as
        # "reversing and not moving" instead of as a robot shoving at a wall for the
        # rest of the episode. The guard cancels the motion; nothing else would notice.
        self._give_up_s = 2.0 * back_limit_m / max(back_speed_mps, 1e-6) + 1.0

    def reset(self) -> None:
        self._history.clear()
        self._active = False
        self._backed_m = 0.0
        self._consecutive = 0
        self._release_t = -1e9

    def front_clearance_m(self, position, yaw: float) -> float:
        """Distance to the nearest thing ahead, or inf.

        `count=False` because this is a measurement, not a guard action: without it
        every probe would add to `guard.interventions` and the run would report
        hundreds of interventions it never had.
        """
        return self.guard.check(position, yaw, 1.0, 0.0, count=False).distance_m

    def update(self, sim_time: float, position, yaw: float, command):
        """Return the command to actually drive, and whether this is a recovery.

        Call once per control step with the controller's command; it is handed back
        unchanged whenever the robot is fine, which is almost always.
        """
        if self._active:
            return self._continue(sim_time, position, yaw, command)

        self._history.append((sim_time, position[0], position[1]))
        while self._history and sim_time - self._history[0][0] > STALL_WINDOW_S:
            self._history.pop(0)

        if sim_time - self._release_t < COOLDOWN_S:
            return command, False
        if self._consecutive >= MAX_CONSECUTIVE:
            return command, False
        # Need a full window before the displacement over it means anything -- at the
        # start of an episode the history is one sample and would read as zero progress.
        if len(self._history) < 2 or sim_time - self._history[0][0] < STALL_WINDOW_S:
            return command, False
        t0, x0, y0 = self._history[0]
        if math.dist((x0, y0), (position[0], position[1])) > STALL_PROGRESS_M:
            self._consecutive = 0
            return command, False
        if self.front_clearance_m(position, yaw) > self.guard.stop_distance_m:
            # Standing still with room ahead. That is the policy choosing to hold, and
            # overriding it here would drive through a deliberate stop.
            return command, False

        self._active = True
        self._backed_m = 0.0
        self._start = (position[0], position[1])
        self._engage_t = sim_time
        self.engagements += 1
        self._consecutive += 1
        return self._reverse(), True

    def _continue(self, sim_time: float, position, yaw: float, command):
        self._backed_m = math.dist(self._start, (position[0], position[1]))
        clear = self.front_clearance_m(position, yaw)
        stuck_behind = sim_time - self._engage_t > self._give_up_s
        if stuck_behind:
            # Wedged front AND back. Reversing harder will not help and the robot would
            # spend the episode pushing at whatever is behind it, so hand control back
            # and let the run record that this happened rather than hiding it in a
            # flat trace.
            self.blocked_behind += 1
        if (stuck_behind or self._backed_m >= self.back_limit_m
                or clear > self.guard.stop_distance_m + CLEAR_MARGIN_M):
            self._active = False
            self._release_t = sim_time
            self._history.clear()
            return command, False
        return self._reverse(), True

    def _reverse(self):
        from controllers import Command
        # omega is 0 on purpose. See the module docstring: straight back is the only
        # direction this manoeuvre knows to be free.
        return Command(vx=-self.back_speed_mps, vy=0.0, omega=0.0)
