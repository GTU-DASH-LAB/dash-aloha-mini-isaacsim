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

**Reversing is only half of it, and the first version shipped only that half.** Backing
away and handing control back silently is not a recovery, because nothing about the
decision has changed: the policy is looking at the same desk from a metre further away,
holding a cached plan that says STOP, and the frame that made it say STOP is the frame it
is still being shown. What that buys is a stall one metre further back -- and a WORSE
one, because at 1.4 m nothing is inside `stop_distance_m` any more, so the trigger above
will never fire again and the robot stands there for the rest of the episode.

So the manoeuvre ends in a re-decision, which is `release_pending` and `memory_live`
below. On release the runner throws the cached plan away (`PolicyClient.replan`), forcing
a generation on the backed-off view rather than re-serving the one taken from inside the
wedge; and for a few seconds afterwards it marks its `/predict` calls `recovered=True`,
which is how the model gets told the one thing the picture cannot show it -- that the
obstacle in front of it is the obstacle it just backed out of. See
`arc_menu.DESCRIBE_AFTER_RECOVERY`.

The memory ends on either of two conditions, and both are meaningful rather than tuned:
the robot has driven `RECOVERY_CLEARED_M` from where it was released, which is the
deadlock actually broken; or `RECOVERY_MEMORY_S` has passed, which is long enough for a
couple of generations to have been asked the question and is the cap on how long a fact
about the past is allowed to shape the present.
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
RECOVERY_MEMORY_S = 6.0   # ~3 generations: long enough to be asked, short enough to end
RECOVERY_CLEARED_M = 0.5  # driven this far from the release point = the deadlock is over


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
        # Set on the step the manoeuvre ends, cleared by whoever acts on it. A flag
        # rather than a third return value from `update`: the release is a one-off event
        # and every other step would have to carry a False for it, which is how a caller
        # ends up throwing the plan away on the wrong step.
        self.release_pending = False
        self._history: list[tuple[float, float, float]] = []   # (t, x, y)
        self._active = False
        self._backed_m = 0.0
        self._consecutive = 0
        self._release_t = -1e9
        self._release_pos = (0.0, 0.0)
        self._start = (0.0, 0.0)
        self._engage_t = 0.0
        # Twice the time the manoeuvre should need, so a rear obstacle shows up as
        # "reversing and not moving" instead of as a robot shoving at a wall for the
        # rest of the episode. The guard cancels the motion; nothing else would notice.
        self._give_up_s = 2.0 * back_limit_m / max(back_speed_mps, 1e-6) + 1.0
        # Twice the back limit, for the reason spelled out in `update`: a robot that has
        # only gone as far as one reverse could carry it has not escaped anything.
        self._escape_m = 2.0 * back_limit_m

    def reset(self) -> None:
        self._history.clear()
        self._active = False
        self._backed_m = 0.0
        self._consecutive = 0
        self._release_t = -1e9
        self.release_pending = False

    def take_release(self) -> bool:
        """True once per manoeuvre, on the step it ended. Clears itself.

        Read every step by the runner, which answers it by dropping the policy's cached
        plan. Consuming it here rather than letting the caller reset it keeps "acted on"
        and "happened" the same event -- a caller that forgets would otherwise replan on
        every step for the rest of the run.
        """
        fired, self.release_pending = self.release_pending, False
        return fired

    def memory_live(self, sim_time: float, position) -> bool:
        """Should the next decision be told the robot has just backed out of something?

        Two ways out, and the first is the one that matters: once the robot has actually
        driven `RECOVERY_CLEARED_M` away from where it was released, the wedge is behind
        it and describing it as "the thing directly in front of you" is no longer true.
        The timeout is the backstop for the case where it never moves -- a fact about the
        last few seconds should not be steering the policy a minute later.
        """
        if sim_time - self._release_t > RECOVERY_MEMORY_S:
            return False
        return math.dist(
            self._release_pos, (position[0], position[1])) < RECOVERY_CLEARED_M

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
        # Drop a sample only while the NEXT one is still old enough to span the window,
        # so `_history[0]` is the newest sample that is at least STALL_WINDOW_S old.
        #
        # The obvious version -- pop while `sim_time - _history[0][0] > STALL_WINDOW_S`
        # -- is wrong, and wrong in a way that leaves the feature looking like it works.
        # It pops until the span is <= 3.0 and the arming test below then requires the
        # span to be >= 3.0, so the two are complementary on a strict inequality and only
        # a span of EXACTLY 3.0 arms anything. At a 1/60 s step that is a float
        # coincidence: instrumented over 5 s of a robot that never moved with a wall
        # 0.4 m off its nose, the span sat at 2.9999999999999996 and the recovery never
        # fired once. It fired at all in longer runs only because clearing the history on
        # release reshuffles which samples exist -- i.e. the 3 s trigger was really a
        # random one.
        while (len(self._history) > 1
               and sim_time - self._history[1][0] >= STALL_WINDOW_S):
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
            # Moving, so leave it alone -- but only clear the consecutive count once the
            # robot is far enough from the last wedge to have actually left it, not
            # merely moving. The escape radius has to be bigger than the manoeuvre's own
            # amplitude or the manoeuvre counts as its own escape: reverse 0.8 m, drive
            # 0.8 m back into the same desk, and every step of that is "progress" by
            # displacement. A counter that resets on it can never reach MAX_CONSECUTIVE,
            # so the cap meant to stop the robot rocking in and out of one obstacle for a
            # whole episode would never once apply.
            if math.dist(self._start, (position[0], position[1])) > self._escape_m:
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
            self._release_pos = (position[0], position[1])
            # Raised on every release, including the give-up one. A robot wedged front
            # AND back needs the re-decision more than the others, not less: reversing
            # got it nothing, so the only thing left that can change is the answer.
            self.release_pending = True
            self._history.clear()
            # `command` is handed back for this one step and it is stale by definition --
            # it is the plan from inside the wedge. That is fine and it is not what the
            # robot ends up driving: the runner reads `take_release()` on this same step
            # and drops the cached plan, so the next call generates on the new view.
            return command, False
        return self._reverse(), True

    def _reverse(self):
        from controllers import Command
        # omega is 0 on purpose. See the module docstring: straight back is the only
        # direction this manoeuvre knows to be free.
        return Command(vx=-self.back_speed_mps, vy=0.0, omega=0.0)
