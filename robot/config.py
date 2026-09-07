#!/usr/bin/env python3
"""Runtime limits, reloaded from disk while the robot is driving.

    limits = LiveLimits(Path("robot/robot.yaml"))
    ...
    lim = limits.current()          # cheap; re-reads only when the file's mtime moves

WHY THESE ARE HOT-RELOADED RATHER THAN COMMAND-LINE FLAGS. Every number in here is one
you will get wrong on the first attempt with a real robot in a real corridor, and the
loop that finds the right value is: watch it drive, decide it is too fast, change one
number. If that loop requires a restart it also requires a fresh model load, a new
episode, and a robot walked back to where it started -- so in practice the number does
not get tuned, it gets endured. Editing a file the running process notices closes the
loop to a couple of seconds. `robot/tools/set_limit.py` and the control API write the
same file, so all three routes are one mechanism.

WHAT IS DELIBERATELY NOT IN HERE: anything that changes what the POLICY is. The thinking
level, the arc set, the frame count and the model are pinned in
`nav/config/profiles/baseline.yaml` and belong to the measurement -- a run with those
changed is a different experiment and should be labelled as one. These are the limits
the VEHICLE is driven under, which is a separate question and yours to answer, because
the answer depends on a chassis nobody here has seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "robot.yaml"


@dataclass(frozen=True)
class Limits:
    """Everything the runner is allowed to change without restarting."""

    # --- what the operator reaches for first ------------------------------------------
    # The master switch. False means the loop keeps running -- camera, policy, guard,
    # logging all live -- and the base is commanded to zero. This is the setting to
    # drive with while you are still checking that the arrows point where you expect,
    # and it is what the control API's "stop" writes. It is NOT a substitute for a
    # hardware e-stop and this file says so where you will read it: a software flag
    # cannot help you when the software is the thing that failed.
    enabled: bool = False
    # Forward speed. Applied twice on purpose: it caps the controller here, and it is
    # pushed to the policy server's /menu_speed, because the arc menu has no speed
    # channel of its own -- the model picks a direction and the speed comes from
    # configuration. Capping only locally would leave the server planning arcs at a pace
    # the robot never drives, and the plan's own speed is what the braking controller
    # takes the minimum against.
    cruise_mps: float = 0.25
    # Hard ceiling. `cruise_mps` above it is clamped, not honoured. Separate from cruise
    # so that a hurried edit to the number you tune cannot exceed the number you decided
    # once, while sober, with a tape measure.
    max_speed_mps: float = 0.45
    max_yaw_rate_radps: float = 1.0

    # --- the guard: the only thing that stops the robot without asking the model -------
    # Below this distance the base is stopped outright. Default higher than the
    # simulator's 0.6 m, and that is not caution for its own sake: on a scan-driven
    # guard the buffer holds one revolution, so a bearing can be 100 ms old, and the
    # stop distance has to absorb the travel in that time. A guard tuned on the
    # simulator's instantaneous raycasts is optimistic by exactly that margin.
    guard_stop_m: float = 0.8
    guard_slow_m: float = 1.8
    guard_fan_half_angle_deg: float = 35.0
    # Where your own chassis ends. Returns closer than this are discarded as self-hits.
    chassis_radius_m: float = 0.35

    # --- rates ------------------------------------------------------------------------
    # How often the base is commanded. Independent of how often the policy is asked --
    # that is the whole shape of this system, a fast loop driving a plan that a slow loop
    # replaces. 20 Hz is comfortable; the guard and the recovery run at this rate.
    control_hz: float = 20.0
    # Seconds between policy calls. 3.0 matches the spacing the frame history was
    # designed around and the benchmark ran at. Shorter is not obviously better: the
    # server serves a cached plan while it thinks, so asking more often mostly returns
    # the same plan more often.
    decision_period_s: float = 3.0
    # Stop the base if a control tick has not completed in this long. Covers the case a
    # `stop()` in a `finally` cannot: the loop still running but wedged -- a camera
    # driver blocked on a USB reset, a policy call with no timeout -- where the last
    # velocity command stands indefinitely and the robot drives on with nobody home.
    watchdog_s: float = 1.0

    def validated(self) -> "Limits":
        """Clamp into a range that cannot hurt anyone, and say so rather than silently.

        A hand-edited YAML is the input here, and it is edited by someone watching a
        robot rather than reading a schema. A negative rate or a zero control frequency
        should not become a division by zero six layers down.
        """
        problems = []
        if self.cruise_mps > self.max_speed_mps:
            problems.append(
                f"cruise_mps {self.cruise_mps} exceeds max_speed_mps {self.max_speed_mps}"
                f"; clamped")
        cruise = min(max(self.cruise_mps, 0.0), max(self.max_speed_mps, 0.0))
        if self.guard_slow_m < self.guard_stop_m:
            problems.append(
                f"guard_slow_m {self.guard_slow_m} is inside guard_stop_m "
                f"{self.guard_stop_m}; raised to match")
        if self.control_hz <= 0.0:
            problems.append(f"control_hz {self.control_hz} must be positive; using 20")
        if problems:
            print("[limits] " + "; ".join(problems), flush=True)
        return Limits(
            enabled=bool(self.enabled),
            cruise_mps=cruise,
            max_speed_mps=max(self.max_speed_mps, 0.0),
            max_yaw_rate_radps=max(self.max_yaw_rate_radps, 0.0),
            guard_stop_m=max(self.guard_stop_m, 0.0),
            guard_slow_m=max(self.guard_slow_m, self.guard_stop_m),
            guard_fan_half_angle_deg=min(max(self.guard_fan_half_angle_deg, 1.0), 180.0),
            chassis_radius_m=max(self.chassis_radius_m, 0.0),
            control_hz=self.control_hz if self.control_hz > 0.0 else 20.0,
            decision_period_s=max(self.decision_period_s, 0.1),
            watchdog_s=max(self.watchdog_s, 0.05),
        )


def _from_mapping(raw: dict) -> Limits:
    known = {f.name for f in fields(Limits)}
    unknown = sorted(set(raw) - known)
    if unknown:
        # Loudly, because the failure it prevents is silent: a misspelled key in a YAML
        # that loads fine leaves the default in force, and the operator watches the
        # robot ignore a setting they are certain they changed.
        print(f"[limits] ignoring unknown key(s): {', '.join(unknown)}", flush=True)
    return Limits(**{k: v for k, v in raw.items() if k in known}).validated()


def load(path: Path | str = DEFAULT_PATH) -> Limits:
    path = Path(path)
    if not path.exists():
        return Limits().validated()
    raw = yaml.safe_load(path.read_text()) or {}
    return _from_mapping(raw.get("limits", raw))


def save(limits: Limits, path: Path | str = DEFAULT_PATH) -> None:
    """Write limits back, preserving the rest of the file.

    Used by the control API and `robot/tools/set_limit.py`. Reads first so that comments
    the operator added elsewhere in the document are not the price of changing a speed.
    """
    path = Path(path)
    raw = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    if "limits" not in raw and raw:
        raw = {"limits": raw}
    raw.setdefault("limits", {})
    raw["limits"].update({f.name: getattr(limits, f.name) for f in fields(Limits)})
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False))
    # Atomic, because the runner may be reading this file at any moment and a
    # half-written YAML would come back as a parse error mid-drive.
    tmp.replace(path)


class LiveLimits:
    """Holds the current limits and re-reads the file when it changes underneath."""

    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self._mtime: float | None = None
        self._limits = load(self.path)
        self._mtime = self._stamp()

    def _stamp(self) -> float | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def current(self) -> Limits:
        """The limits as of now. Safe to call every control tick -- it stats, not parses."""
        stamp = self._stamp()
        if stamp != self._mtime:
            self._mtime = stamp
            try:
                new = load(self.path)
            except (yaml.YAMLError, OSError) as exc:
                # Keep driving under the last good limits rather than dying on a typo.
                # The operator is editing this file next to a moving robot; a syntax
                # error there should cost them a message, not a crash with the base
                # still rolling.
                print(f"[limits] reload failed, keeping previous: {exc}", flush=True)
                return self._limits
            changed = [f.name for f in fields(Limits)
                       if getattr(new, f.name) != getattr(self._limits, f.name)]
            if changed:
                print("[limits] reloaded: "
                      + ", ".join(f"{n}={getattr(new, n)}" for n in changed), flush=True)
            self._limits = new
        return self._limits


def yaw_rate_for(limits: Limits) -> float:
    """The controller's yaw cap in rad/s, kept as a function so the unit is stated once."""
    return min(limits.max_yaw_rate_radps, 2.0 * math.pi)
