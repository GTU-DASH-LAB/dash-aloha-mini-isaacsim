"""Does the wedge recovery fire when it should, and stay out of the way when it should not?

The expensive mistake for this feature is not failing to recover -- that just leaves the
robot where it already was. It is firing on a robot that is deliberately holding still,
which turns a correct decision into a reversing manoeuvre. So the cases that matter most
here are the ones where nothing should happen.

Run against a stub guard rather than Isaac Sim, because the logic under test is a state
machine over time and clearance, and booting Kit to exercise it costs five minutes and
adds a stage whose geometry would then be the thing being measured.

Usage:
    python3 nav/tools/check_stuck_recovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from collision_guard import GuardResult  # noqa: E402
from controllers import Command  # noqa: E402
from stuck_recovery import BACK_LIMIT_M, STALL_WINDOW_S, StuckRecovery  # noqa: E402

DT = 1.0 / 60.0
DRIVING = Command(0.45, 0.0, 0.0)
HOLDING = Command(0.0, 0.0, 0.0)


class StubGuard:
    """Reports a fixed front clearance and counts how often it was asked to guard."""

    def __init__(self, clearance_m: float, stop_distance_m: float = 0.6):
        self.stop_distance_m = stop_distance_m
        self.clearance_m = clearance_m
        self.interventions = 0

    def check(self, position, yaw, vx, vy, count=True) -> GuardResult:
        if count:
            self.interventions += 1
        return GuardResult(False, self.clearance_m, vx, vy)


def simulate(clearance_m: float, command: Command, seconds: float,
             speed_mps: float = 0.0, open_up_after_s: float | None = None,
             rear_blocked: bool = False):
    """Drive a robot at a fixed speed for `seconds` and report what recovery did."""
    guard = StubGuard(clearance_m)
    rec = StuckRecovery(guard)
    x, t = 0.0, 0.0
    fired, reverse_steps, max_reverse_x, omega_seen = 0, 0, 0.0, set()
    was_active, start_x = False, 0.0
    for _ in range(int(seconds / DT)):
        if open_up_after_s is not None and t >= open_up_after_s:
            guard.clearance_m = 5.0
        drive, recovering = rec.update(t, (x, 0.0, 0.0), 0.0, command)
        if recovering and not was_active:
            fired += 1
            start_x = x
        if recovering:
            reverse_steps += 1
            omega_seen.add(round(drive.omega, 6))
            max_reverse_x = max(max_reverse_x, abs(x - start_x))
        was_active = recovering
        # `rear_blocked` is what a wall behind looks like from here: the guard cancels
        # the reverse velocity, so the robot is commanded backwards and does not move.
        moved = 0.0 if (recovering and rear_blocked) else (
            drive.vx if recovering else speed_mps)
        x += moved * DT
        t += DT
    return {"fired": fired, "reverse_steps": reverse_steps, "backed_m": max_reverse_x,
            "omega": omega_seen, "engagements": rec.engagements,
            "blocked_behind": rec.blocked_behind, "interventions": guard.interventions}


def main() -> int:
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str):
        print(f"  {'pass' if ok else 'FAIL'}  {name:38} {detail}")
        if not ok:
            fails.append(f"{name}: {detail}")

    print(f"stall window {STALL_WINDOW_S}s, back limit {BACK_LIMIT_M} m\n")

    # --- the cases where nothing must happen ---------------------------------
    r = simulate(0.4, DRIVING, 20.0, speed_mps=0.45)
    check("driving past a close wall", r["fired"] == 0,
          f"{r['fired']} engagements while still making progress")

    r = simulate(5.0, HOLDING, 20.0)
    check("holding still in open space", r["fired"] == 0,
          f"{r['fired']} engagements -- a deliberate stop must be left alone")

    r = simulate(0.4, HOLDING, STALL_WINDOW_S - 0.5)
    check("stopped, window not yet full", r["fired"] == 0,
          f"{r['fired']} engagements before {STALL_WINDOW_S}s of evidence")

    # --- the case it exists for ----------------------------------------------
    r = simulate(0.4, HOLDING, 20.0)
    check("wedged: fires", r["fired"] >= 1, f"{r['fired']} engagements")
    check("wedged: reverses", r["reverse_steps"] > 0,
          f"{r['reverse_steps']} steps of reverse")
    check("wedged: never turns", r["omega"] == {0.0},
          f"omega values seen: {sorted(r['omega'])} (must be 0 -- only the footprint "
          f"it just left is known clear)")
    check("wedged: respects the back limit",
          r["backed_m"] <= BACK_LIMIT_M + 0.05,
          f"backed {r['backed_m']:.2f} m against a {BACK_LIMIT_M} m limit")

    # --- releases as soon as the way opens ------------------------------------
    r = simulate(0.4, HOLDING, 20.0, open_up_after_s=STALL_WINDOW_S + 0.5)
    check("releases when the front clears", r["backed_m"] < BACK_LIMIT_M,
          f"backed only {r['backed_m']:.2f} m before handing control back")

    # --- wedged front AND back -------------------------------------------------
    r = simulate(0.4, HOLDING, 40.0, rear_blocked=True)
    check("gives up when it cannot reverse", r["blocked_behind"] >= 1,
          f"{r['blocked_behind']} give-ups recorded rather than shoving forever")

    # --- telemetry -------------------------------------------------------------
    r = simulate(5.0, HOLDING, 20.0)
    check("clearance probes are not interventions", r["interventions"] == 0,
          f"{r['interventions']} guard interventions counted from probing alone")

    print("\n" + "=" * 78)
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
    else:
        print("  PASS - recovery fires only on a wedge, reverses straight, and stops.")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
