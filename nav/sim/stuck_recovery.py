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

**THE SECOND TRIGGER: standing still with nothing in front (a BALK).** The wedge above
requires an obstacle inside the guard's stop distance, and the line that enforced it used
to bail out with the comment "standing still with room ahead is the policy choosing to
hold, and overriding it would drive through a deliberate stop". That reasoning was wrong,
and the first full ladder is what showed it: `office_nearest_elevator` answered STOP on
**42 of its 65 decisions** from 5.5 m out, with its own description of those same frames
reading "there is a wall straight ahead, the most open walkable floor is to the RIGHT,
TARGET: not visible". Nothing was ever close enough to be a wedge, so the recovery never
fired once, and the episode ended 5.5 m short. `hospital_exit_room` failed the same way
with the guard involved: 2655 interventions, 12% of the gap closed.

The correction is the same structural argument that justifies taking STOP off the menu
after a reverse, and it is worth stating in full because it is what makes this safe:
**a robot still inside the control loop has not arrived.** `run_navigation.py` breaks the
instant `distance <= success_threshold_m`, so arrival is latched and terminal. Therefore a
stop that persists for `BALK_WINDOW_S` cannot be the correct stop -- there is no state of
the world in which standing still for five seconds, still being driven, is right. It is
not a deliberate stop being overridden; it is a stop that has already been proven wrong by
the fact that the loop is still running.

What a balk does about it is deliberately not "drive forward anyway". It reverses
`BALK_BACK_M` and then forces a re-decision, because the failure is one of INFORMATION,
not of will: the model said it could not see the target. Backing up half a metre widens
the field of view and puts a landmark that was cropped at the frame edge back inside it,
which is the one thing the robot can do to change the answer rather than to override it.
Overriding it -- picking a direction ourselves -- would be inventing a heading that
nothing in the stack has looked at.
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

# The BALK: standing still with nothing in front. Longer window than the wedge because
# there is no obstacle corroborating the stall -- the only evidence is time, so it has to
# be enough time that a slow crawl or one cautious generation cannot look like a stop.
BALK_WINDOW_S = 5.0
# And a shorter reverse, because the purpose is different. A wedge reverse is an ESCAPE
# and has to clear the obstacle; a balk reverse is a LOOK, and half a metre is already a
# visibly wider field of view -- enough for a landmark cropped out at the frame edge to
# come back into it. Backing further would only spend distance the robot then has to
# re-drive.
BALK_BACK_M = 0.5


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
        # Counted apart from `engagements` because they are evidence about different
        # failures: a wedge says the robot drove into something, a balk says the policy
        # stopped without arriving. A run reporting one is not the same run reporting the
        # other, and summing them would hide which of the two the episode actually hit.
        self.balks = 0
        self._mode = "wedge"          # which trigger is currently engaged
        self._release_mode = "wedge"  # ...and which one the live memory belongs to
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
        # The tallies are per EPISODE, and they were not being cleared. One instance is
        # built per process and `reset()` runs per episode, so in the UI -- where several
        # episodes share a process -- episode 2 reported episode 1's recoveries plus its
        # own. Headless runs never showed it because there the process is the episode.
        self.engagements = 0
        self.balks = 0
        self.blocked_behind = 0

    @property
    def mode(self) -> str:
        """"wedge" or "balk" -- which trigger is engaged RIGHT NOW.

        Distinct from `memory_kind`, which answers the same question about a manoeuvre that
        has already finished. The status line wants this one: while reversing, "backing out
        of a wedge" and "backing off after stalling" describe the same motion for opposite
        reasons, and a watcher who is told the wrong one will look for an obstacle that was
        never there.
        """
        return self._mode

    def take_release(self) -> bool:
        """True once per manoeuvre, on the step it ended. Clears itself.

        Read every step by the runner, which answers it by dropping the policy's cached
        plan. Consuming it here rather than letting the caller reset it keeps "acted on"
        and "happened" the same event -- a caller that forgets would otherwise replan on
        every step for the rest of the run.
        """
        fired, self.release_pending = self.release_pending, False
        return fired

    def stalled_s(self, sim_time: float, position) -> float:
        """How long the robot has been making no progress. 0.0 when it is moving.

        Reported to the policy so the model can be told what the picture cannot show it:
        one frame of a stationary robot is identical to one frame of a moving one. This is
        the honest form of "add some history" -- a number measured here, where position is
        known, rather than a story assembled on the server, where it is not.
        """
        if self._active:
            return 0.0
        # Walk BACKWARDS from the newest sample and stop at the first one that is far
        # away: the stall began there. Scanning forwards and taking the first near sample
        # instead would find a position the robot merely passed through earlier and report
        # a rock-out-and-back as one long stall -- overstating exactly the case where the
        # robot is moving the most.
        oldest = sim_time
        for t, x, y in reversed(self._history):
            if math.dist((x, y), (position[0], position[1])) > STALL_PROGRESS_M:
                break
            oldest = t
        return sim_time - oldest

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

    def memory_kind(self, sim_time: float, position) -> str | None:
        """"wedge", "balk" or None -- which manoeuvre the live memory came from.

        The two need different sentences and the difference is not cosmetic: after a wedge
        the thing in front is the thing that blocked you, after a balk there was never
        anything in front and the problem was that you could not see your target. Telling
        the model the wrong one of those is worse than telling it neither.
        """
        return self._release_mode if self.memory_live(sim_time, position) else None

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
        # Pruned to the LONGER of the two windows, because the balk test needs 5 s of
        # history to be able to say anything and the wedge test reads its own 3 s out of
        # the same list. One list, two spans, measured by `_moved_over` rather than by
        # index -- indexing element 0 only works while there is exactly one window.
        keep_s = max(STALL_WINDOW_S, BALK_WINDOW_S)
        while len(self._history) > 1 and sim_time - self._history[1][0] >= keep_s:
            self._history.pop(0)

        if sim_time - self._release_t < COOLDOWN_S:
            return command, False
        if self._consecutive >= MAX_CONSECUTIVE:
            return command, False
        # Need a full window before the displacement over it means anything -- at the
        # start of an episode the history is one sample and would read as zero progress.
        moved = self._moved_over(STALL_WINDOW_S, sim_time, position)
        if moved is None:
            return command, False
        if moved > STALL_PROGRESS_M:
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
            # Standing still with room ahead: not a wedge. This used to return here, on
            # the grounds that it is the policy choosing to hold and overriding it would
            # drive through a deliberate stop. See the module docstring for why that was
            # wrong -- arrival is latched and terminal, so a robot still being driven has
            # not arrived, and a stop this long has therefore already been proven wrong.
            # It still gets a LONGER window than a wedge, because time is the only
            # evidence here: no obstacle corroborates it.
            balked = self._moved_over(BALK_WINDOW_S, sim_time, position)
            if balked is None or balked > STALL_PROGRESS_M:
                return command, False
            self.balks += 1
            return self._engage("balk", sim_time, position)

        return self._engage("wedge", sim_time, position)

    def _moved_over(self, window_s: float, sim_time: float, position) -> float | None:
        """Distance covered over the last `window_s`, or None if history is too short.

        None and 0.0 are different answers and conflating them is what makes a stall
        detector fire on the first frame of an episode: an empty history has covered no
        distance, which is not the same as a robot that has covered no distance.
        """
        ref = None
        for t, x, y in self._history:
            if sim_time - t >= window_s:
                ref = (x, y)
            else:
                break
        if ref is None:
            return None
        return math.dist(ref, (position[0], position[1]))

    def _engage(self, mode: str, sim_time: float, position):
        self._active = True
        self._mode = mode
        self._backed_m = 0.0
        self._start = (position[0], position[1])
        self._engage_t = sim_time
        self.engagements += 1
        self._consecutive += 1
        return self._reverse(), True

    def _continue(self, sim_time: float, position, yaw: float, command):
        self._backed_m = math.dist(self._start, (position[0], position[1]))
        clear = self.front_clearance_m(position, yaw)
        # A balk ends on distance alone. There is no clearance condition to wait for --
        # the front was already clear when it engaged, which is what made it a balk -- and
        # the limit is shorter because the manoeuvre is a look, not an escape.
        limit = BALK_BACK_M if self._mode == "balk" else self.back_limit_m
        stuck_behind = sim_time - self._engage_t > self._give_up_s
        if stuck_behind:
            # Wedged front AND back. Reversing harder will not help and the robot would
            # spend the episode pushing at whatever is behind it, so hand control back
            # and let the run record that this happened rather than hiding it in a
            # flat trace.
            self.blocked_behind += 1
        if (stuck_behind or self._backed_m >= limit
                or (self._mode == "wedge"
                    and clear > self.guard.stop_distance_m + CLEAR_MARGIN_M)):
            self._active = False
            self._release_t = sim_time
            self._release_pos = (position[0], position[1])
            self._release_mode = self._mode
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
