#!/usr/bin/env python3
"""Does the controllers' heading filter still wrap, and did the P1 fix change anything else?

THE BUG (P1). `PursuitController` / `GuidedPursuitController` / `HolonomicController`
low-pass a heading error with

    state += alpha * wrap(err - state)        # the INCREMENT is wrapped, the STATE is not

(HEAD:nav/sim/controllers.py:236-240 pursuit, :323-327 holonomic, :433 guided). When a
STOP plan is reframed for the distance driven since it was generated, every waypoint sits
at the same point BEHIND the robot, the lookahead point is at heading +-pi, and the state
is dragged to just past pi. From there the *short* way round to a straight-ahead error of
0 is UPWARD, so the state settles near 2*pi and `omega = clip(k * state)` saturates for
the rest of the episode while the plan says "straight".

THE FIX (uncommitted). `is_stationary()` makes a plan with < 2 cm of path return
Command(0,0,0) without touching the filter, and `filter_angle()` wraps the STATE into
[-pi, pi] after every update. This tool checks, with no Isaac, no server and no GPU:

  (0) unit checks of the two helpers,
  (a) the OLD update (inline copy, cross-checked against `git show HEAD:...` while HEAD is
      still the old code) reproduces the saturating spin on a scripted sequence,
  (b) the REAL classes, imported from nav/sim/controllers.py, do not: STOP plans return
      Command(0,0,0), |state| <= pi at every step, straight plans give ~0 omega, and a
      plain constant-curvature turn gives the same omega as before the change,
  (c) every nav/results/*.json with a `plans` list, replayed through old and new filters:
      which recorded runs would have crossed the wrap, and when.

Exit status 0 only if every assertion passes. Usage:

    python3 nav/tools/check_controller_wrap.py
    python3 nav/tools/check_controller_wrap.py --full          # every row of the traces
    python3 nav/tools/check_controller_wrap.py --stops 20 --straight 30
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "sim"))

import controllers as C  # noqa: E402
from controllers import Command, make_controller  # noqa: E402
from guidance import guidance_heading  # noqa: E402

# nav/config/episodes.yaml:23-24 -- the module itself defines no speed or yaw-rate limit.
V_MAX = 1.5
W_MAX = 1.2
PI = math.pi
TWO_PI = 2.0 * math.pi
NAMES = ("pursuit", "braking", "guided", "holonomic")


# ---------------------------------------------------------------------------------------
# Assertion bookkeeping
# ---------------------------------------------------------------------------------------

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------------------
# The OLD update, inline.
#
# Copied from `git show HEAD:nav/sim/controllers.py` (pursuit :221-240, holonomic :314-327,
# guided :417-433): `len(waypoints) < 2` is the only stationary test, and the filter state
# is initialised unwrapped and only ever incremented by a wrapped step. Everything that did
# NOT change (`_lookahead_point`, `plan_speed`, `_EPS`, the gains) is imported from the real
# module so this copy differs from the new code in exactly the lines the fix touched.
# ---------------------------------------------------------------------------------------


def old_filter(state: float | None, new: float, alpha: float) -> float:
    if state is None:
        state = new
    wrapped = math.atan2(math.sin(new - state), math.cos(new - state))
    return state + alpha * wrapped


class OldPursuit:
    def __init__(self, v_max, w_max, lookahead_m=C.DYNANAV_LOOKAHEAD_M,
                 k_angular=C.DYNANAV_K_ANGULAR, alpha_filter=C.DYNANAV_ALPHA_FILTER,
                 obey_plan_speed=False, speed_scaled_feedback=False):
        self.v_max, self.w_max = v_max, w_max
        self.lookahead_m, self.k_angular, self.alpha_filter = lookahead_m, k_angular, alpha_filter
        self.obey_plan_speed = obey_plan_speed
        self.speed_scaled_feedback = speed_scaled_feedback
        self._yaw_err_filt: float | None = None

    def _steer(self, waypoints, x_l, y_l, dist, ref_heading) -> Command:
        self._yaw_err_filt = old_filter(self._yaw_err_filt, ref_heading, self.alpha_filter)
        kappa = 2.0 * y_l / (dist * dist)
        v_kappa = self.w_max / (abs(kappa) + C._EPS)
        v_cap = min(self.v_max, C.plan_speed(waypoints)) if self.obey_plan_speed else self.v_max
        v_cmd = float(np.clip(min(v_cap, v_kappa), 0.0, self.v_max))
        w_ff = 0.5 * v_cmd * kappa
        w_fb = self.k_angular * self._yaw_err_filt
        if self.speed_scaled_feedback:
            w_fb *= min(1.0, v_cmd / max(self.k_angular * dist, C._EPS))
        return Command(v_cmd, 0.0, float(np.clip(w_ff + w_fb, -self.w_max, self.w_max)))

    def __call__(self, waypoints, guidance=None) -> Command:
        if len(waypoints) < 2:
            return Command(0.0, 0.0, 0.0)
        x_l, y_l, dist = C._lookahead_point(waypoints, self.lookahead_m)
        if dist < C._EPS:
            return Command(0.0, 0.0, 0.0)
        return self._steer(waypoints, x_l, y_l, dist, math.atan2(y_l, x_l))


class OldBraking(OldPursuit):
    def __init__(self, *a, **kw):
        kw.setdefault("obey_plan_speed", True)
        kw.setdefault("speed_scaled_feedback", True)
        super().__init__(*a, **kw)


class OldGuided(OldPursuit):
    def __init__(self, *a, guidance_horizon_s=6.0, **kw):
        kw.setdefault("obey_plan_speed", True)
        super().__init__(*a, **kw)
        self.guidance_horizon_s = guidance_horizon_s

    def __call__(self, waypoints, guidance=None) -> Command:
        heading = guidance_heading(guidance, self.guidance_horizon_s)
        if heading is None:
            return OldPursuit.__call__(self, waypoints)
        if len(waypoints) < 2:
            return Command(0.0, 0.0, 0.0)
        x_l, y_l, dist = C._lookahead_point(waypoints, self.lookahead_m)
        if dist < C._EPS:
            return Command(0.0, 0.0, 0.0)
        return self._steer(waypoints, x_l, y_l, dist, heading)


class OldHolonomic:
    def __init__(self, v_max, w_max, lookahead_m=C.DYNANAV_LOOKAHEAD_M, yaw_align=0.8,
                 alpha_filter=C.DYNANAV_ALPHA_FILTER):
        self.v_max, self.w_max = v_max, w_max
        self.lookahead_m, self.yaw_align, self.alpha_filter = lookahead_m, yaw_align, alpha_filter
        self._heading_filt: float | None = None

    def __call__(self, waypoints, guidance=None) -> Command:
        if len(waypoints) < 2:
            return Command(0.0, 0.0, 0.0)
        x_l, y_l, dist = C._lookahead_point(waypoints, self.lookahead_m)
        if dist < C._EPS:
            return Command(0.0, 0.0, 0.0)
        self._heading_filt = old_filter(self._heading_filt, math.atan2(y_l, x_l), self.alpha_filter)
        vx, vy = self.v_max * (x_l / dist), self.v_max * (y_l / dist)
        omega = float(np.clip(self.yaw_align * self._heading_filt, -self.w_max, self.w_max))
        return Command(float(vx), float(vy), omega)


OLD = {"pursuit": OldPursuit, "braking": OldBraking, "guided": OldGuided,
       "holonomic": OldHolonomic}


def make_old(name: str):
    return OLD[name](V_MAX, W_MAX)


def line_of(needle: str) -> int:
    """1-based line in controllers.py that starts with `needle` (for citations in the report)."""
    for i, line in enumerate(Path(inspect.getsourcefile(C)).read_text().splitlines(), 1):
        if line.startswith(needle):
            return i
    return -1


def state_of(ctrl) -> float | None:
    for attr in ("_yaw_err_filt", "_heading_filt"):
        if hasattr(ctrl, attr):
            return getattr(ctrl, attr)
    return None


# ---------------------------------------------------------------------------------------
# Plans and the scripted sequence
# ---------------------------------------------------------------------------------------


def arc_plan(kappa: float, v: float = 0.7, n: int = 30, dt: float = 0.1) -> np.ndarray:
    """Constant-curvature arc from the origin along +x, n points dt apart at speed v.

    Positive kappa curves LEFT (+y), i.e. a positive yaw-rate command.
    """
    s = v * dt * np.arange(1, n + 1)
    if abs(kappa) < 1e-12:
        return np.stack([s, np.zeros_like(s)], axis=1)
    return np.stack([np.sin(kappa * s) / kappa, (1.0 - np.cos(kappa * s)) / kappa], axis=1)


def point_plan(x: float, y: float, n: int = 30) -> np.ndarray:
    """Every waypoint at the same point: a STOP, as reframed by the server."""
    return np.tile(np.array([x, y], dtype=float), (n, 1))


def guidance_at(heading: float, length_m: float = 3.0) -> np.ndarray:
    """A (3,3) of (x, y, theta) rows at 3/6/9 s that all point along `heading`."""
    x, y = length_m * math.cos(heading), length_m * math.sin(heading)
    return np.array([[x, y, heading]] * 3, dtype=float)


@dataclass
class Step:
    phase: str          # move | settle | stop | straight | behind | ahead
    label: str
    wp: np.ndarray
    guidance: np.ndarray | None = None


def build_sequence(sign: int, n_stops: int, n_straight: int) -> list[Step]:
    """Moving plans, then reframed STOPs behind the robot, then straight-ahead plans.

    `sign` mirrors the whole STOP phase (+1: the first STOP sits at +y). The moving phase
    ends with straight "settle" plans so the filter enters the STOPs near zero, which makes
    "|omega| < 0.05 on the straight plans afterwards" a statement about the STOPs and not
    about leftover turning. The STOP count is even, so the phase ends on the update that
    pushes an unwrapped state upward (see the report for the trace).
    """
    seq: list[Step] = []
    for kap in (0.35, 0.35, 0.20, 0.35, -0.35, -0.35, -0.20, -0.35):
        seq.append(Step("move", f"arc k={kap:+.2f}", arc_plan(kap)))
    for _ in range(6):
        seq.append(Step("settle", "straight", arc_plan(0.0)))
    for i in range(n_stops):
        y = sign * 0.05 * (1 if i % 2 == 0 else -1)
        seq.append(Step("stop", f"STOP@(-0.80,{y:+.2f})", point_plan(-0.8, y)))
    for _ in range(n_straight):
        seq.append(Step("straight", "straight", arc_plan(0.0)))
    return seq


def build_guided_stress(sign: int, n_behind: int, n_ahead: int) -> list[Step]:
    """Second `filter_angle` call site: the guided branch filters the GUIDANCE heading.

    Moving straight plans (never stationary) whose 6 s guidance alternates +-(pi - 0.05),
    i.e. nearly backward -- the guidance channel was logged at up to +165 deg
    (GuidedPursuitController docstring) -- then guidance straight ahead.
    """
    seq: list[Step] = []
    for _ in range(6):
        seq.append(Step("move", "guid 0", arc_plan(0.0), guidance_at(0.0)))
    for i in range(n_behind):
        h = sign * (PI - 0.05) * (1 if i % 2 == 0 else -1)
        seq.append(Step("behind", f"guid {math.degrees(h):+.1f}deg", arc_plan(0.0), guidance_at(h)))
    for _ in range(n_ahead):
        seq.append(Step("ahead", "guid 0", arc_plan(0.0), guidance_at(0.0)))
    return seq


@dataclass
class Row:
    phase: str
    label: str
    cmd: Command
    state: float | None


def run(ctrl, seq: list[Step]) -> list[Row]:
    rows = []
    for st in seq:
        cmd = ctrl(st.wp, st.guidance)
        rows.append(Row(st.phase, st.label, cmd, state_of(ctrl)))
    return rows


def max_abs_state(rows: list[Row]) -> float:
    """Largest |filter state| over the rows; NaN when the controller never set one.

    NaN fails every `<= pi` comparison, so a controller that swallows every plan (state
    stays None) is reported as a failed check instead of crashing the tool.
    """
    return max((abs(r.state) for r in rows if r.state is not None), default=float("nan"))


def fnum(x: float | None, fmt: str = "+.6f") -> str:
    return "None" if x is None else format(x, fmt)


def sfmt(x: float | None) -> str:
    return "   None " if x is None else f"{x:+8.4f}"


def print_trace(title: str, seq: list[Step], rows: list[Row], full: bool) -> None:
    print(f"\n  {title}")
    print(f"  {'#':>3} {'phase':<9} {'plan':<22} {'state[rad]':>10} {'state/2pi':>9} "
          f"{'vx':>7} {'omega':>8}")
    n_straight = sum(1 for r in rows if r.phase in ("straight", "ahead"))
    shown_straight = {0, 1, 2, 3, 4, 7, 11, 15, 19, n_straight - 1}
    k = 0
    first_after = next((i for i, r in enumerate(rows)
                        if r.phase in ("stop", "behind", "straight", "ahead")), len(rows))
    last_move = first_after - 1
    for i, r in enumerate(rows):
        keep = full
        if r.phase in ("stop", "behind"):
            keep = True
        elif r.phase in ("straight", "ahead"):
            keep = keep or k in shown_straight
            k += 1
        elif i == last_move:
            keep = True
        if not keep:
            continue
        st = r.state
        ratio = "" if st is None else f"{st / TWO_PI:+.3f}"
        flag = "  <-- |state| > pi" if st is not None and abs(st) > PI else ""
        print(f"  {i:>3} {r.phase:<9} {r.label:<22} {sfmt(st)} {ratio:>9} "
              f"{r.cmd.vx:7.3f} {r.cmd.omega:+8.4f}{flag}")


# ---------------------------------------------------------------------------------------
# (0) Unit checks of the two helpers
# ---------------------------------------------------------------------------------------


def unit_checks() -> None:
    banner("(0) helper unit checks  --  is_stationary / filter_angle")
    print(f"  STATIONARY_PATH_M = {C.STATIONARY_PATH_M}  (controllers.py:{line_of('STATIONARY_PATH_M = ')})")
    print(f"  is_stationary @ controllers.py:{inspect.getsourcelines(C.is_stationary)[1]}, "
          f"filter_angle @ controllers.py:{inspect.getsourcelines(C.filter_angle)[1]}")

    two = lambda d: np.array([[0.0, 0.0], [d, 0.0]])  # noqa: E731
    check(C.is_stationary(np.zeros((30, 2))), "all-zero STOP plan is stationary")
    check(C.is_stationary(point_plan(-0.8, 0.05)), "STOP reframed to (-0.8, +0.05) is stationary")
    check(C.is_stationary(point_plan(-0.8, -0.05)), "STOP reframed to (-0.8, -0.05) is stationary")
    check(C.is_stationary(np.zeros((1, 2))) and C.is_stationary(np.zeros((0, 2))),
          "fewer than 2 waypoints is stationary")
    check(C.is_stationary(two(0.019)), "total path 0.019 m < 0.02 m is stationary")
    check(not C.is_stationary(two(0.021)), "total path 0.021 m >= 0.02 m is NOT stationary")
    creep = np.stack([0.005 * np.arange(1, 101), np.zeros(100)], axis=1)  # 0.05 m/s over 10 s
    check(not C.is_stationary(creep),
          "0.05 m/s creep over a 10 s / 100-point horizon is NOT stationary",
          f"path = {np.sum(np.hypot(*np.diff(creep, axis=0).T)):.3f} m")
    check(not C.is_stationary(arc_plan(0.35)), "normal 0.7 m/s arc is NOT stationary")

    cmd = make_controller("braking", V_MAX, W_MAX)(creep)
    check(abs(cmd.vx - 0.0495) < 1e-3 and cmd.vx > 0,
          "braking still creeps at the plan's own 0.05 m/s (the threshold swallows no real plan)",
          f"vx = {cmd.vx:.4f}")

    rng = np.random.default_rng(0)
    prev = rng.uniform(-PI, PI, 200_000)
    new = rng.uniform(-PI, PI, 200_000)
    alpha = C.DYNANAV_ALPHA_FILTER
    out = np.array([C.filter_angle(float(p), float(n), alpha) for p, n in zip(prev, new)])
    check(bool(np.all(np.abs(out) <= PI)), "filter_angle output |.| <= pi on 200k random (prev,new)",
          f"max |out| = {np.max(np.abs(out)):.12f}")
    oldv = np.array([old_filter(float(p), float(n), alpha) for p, n in zip(prev, new)])
    in_range = np.abs(oldv) <= PI
    same = np.abs(out - oldv)[in_range]
    check(bool(np.all(same < 1e-12)),
          "where the old update stayed in range, filter_angle is bit-for-bit the same",
          f"{int(in_range.sum())}/{len(prev)} pairs in range, max |diff| = {same.max():.2e}")
    out_of_range = ~in_range
    diff = oldv[out_of_range] - out[out_of_range]
    k = np.round(diff / TWO_PI)
    check(bool(np.all(np.abs(diff - k * TWO_PI) < 1e-9)) and bool(np.all(np.abs(k) == 1)),
          "where the old update left (-pi, pi], filter_angle returns the same ANGLE minus exactly 2*pi",
          f"{int(out_of_range.sum())} pairs ({100.0 * out_of_range.mean():.2f}%) differ, all by exactly +-2pi")
    f0 = C.filter_angle(None, 3.5, alpha)
    check(abs(f0 - (3.5 - TWO_PI)) < 1e-12, "filter_angle(None, x) wraps the seed value",
          f"filter_angle(None, 3.5) = {f0:+.6f}")


# ---------------------------------------------------------------------------------------
# HEAD cross-check: is the inline old copy really the old code?
# ---------------------------------------------------------------------------------------


def load_head_module() -> tuple[types.ModuleType | None, str]:
    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", "HEAD:nav/sim/controllers.py"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        return None, f"git show failed ({exc})"
    mod = types.ModuleType("controllers_head")
    mod.__file__ = "HEAD:nav/sim/controllers.py"
    sys.modules["controllers_head"] = mod  # @dataclass resolves string annotations via sys.modules
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return None, f"could not exec HEAD controllers.py ({exc})"
    if hasattr(mod, "filter_angle"):
        return None, "HEAD already contains filter_angle -- the fix is committed, nothing old to compare"
    return mod, ""


def head_crosscheck(seqs: dict[str, list[Step]], stress: dict[str, list[Step]]) -> None:
    print("\n  inline-old vs `git show HEAD:nav/sim/controllers.py` (the code before the fix):")
    head, why = load_head_module()
    if head is None:
        print(f"  [SKIP] {why}")
        return
    worst = 0.0
    n_cmp = 0
    for name in NAMES:
        for tag, seq in list(seqs.items()) + list(stress.items()):
            if name != "guided" and tag.startswith("guided-stress"):
                continue
            h = head.make_controller(name, V_MAX, W_MAX)
            o = make_old(name)
            for st in seq:
                a, b = h(st.wp, st.guidance), o(st.wp, st.guidance)
                worst = max(worst, abs(a.vx - b.vx), abs(a.vy - b.vy), abs(a.omega - b.omega))
                sa, sb = state_of(h), state_of(o)
                if sa is not None and sb is not None:
                    worst = max(worst, abs(sa - sb))
                n_cmp += 1
    check(worst < 1e-12, "inline old copy == HEAD controllers.py (vx, vy, omega, state)",
          f"{n_cmp} commands compared across {len(NAMES)} controllers, max |diff| = {worst:.2e}")


# ---------------------------------------------------------------------------------------
# (a) The old bug
# ---------------------------------------------------------------------------------------


def section_a(seqs, stress, full: bool):
    banner("(a) OLD update reproduces the bug  --  state += alpha*wrap(err - state), state NOT wrapped")
    print(f"  gains: k_angular={C.DYNANAV_K_ANGULAR}, alpha={C.DYNANAV_ALPHA_FILTER}, "
          f"max_yaw_rate={W_MAX} rad/s, max_speed={V_MAX} m/s")
    print("  sequence: 8 arcs (|kappa| 0.20-0.35, 0.7 m/s) + 6 straight, THEN reframed STOPs at "
          "(-0.8, +-0.05) alternating, THEN straight-ahead plans.")
    old_rows: dict[str, dict[str, list[Row]]] = {}
    for tag, seq in seqs.items():
        old_rows[tag] = {}
        for name in NAMES:
            old_rows[tag][name] = run(make_old(name), seq)
        print_trace(f"OLD pursuit, scenario '{tag}'", seq, old_rows[tag]["pursuit"], full)

    print()
    for tag, seq in seqs.items():
        for name in ("pursuit", "braking", "holonomic", "guided"):
            rows = old_rows[tag][name]
            stops = [r for r in rows if r.phase == "stop"]
            straight = [r for r in rows if r.phase == "straight"]
            end_stop = stops[-1].state
            end = straight[-1]
            print(f"  OLD {name:<9} {tag:<8} state after STOPs {end_stop:+.4f}  "
                  f"state after straight plans {end.state:+.4f} ({end.state / TWO_PI:+.4f} x 2pi)  "
                  f"omega {end.cmd.omega:+.3f}   (STOP-phase vx min/max "
                  f"{min(r.cmd.vx for r in stops):+.2f}/{max(r.cmd.vx for r in stops):+.2f})")
    print()
    for tag in seqs:
        sign = +1 if tag == "plus" else -1
        for name in NAMES:
            rows = old_rows[tag][name]
            stops = [r for r in rows if r.phase == "stop"]
            straight = [r for r in rows if r.phase == "straight"]
            s_stop, s_end, w_end = stops[-1].state, straight[-1].state, straight[-1].cmd.omega
            check(abs(s_stop) > PI, f"OLD {name}/{tag}: state leaves (-pi, pi] during the STOPs",
                  f"state = {s_stop:+.4f}")
            check(abs(abs(s_end) - TWO_PI) < 0.05 and np.sign(s_end) == sign,
                  f"OLD {name}/{tag}: after straight plans the state sits at {'+' if sign > 0 else '-'}2pi",
                  f"state = {s_end:+.4f}, 2pi - |state| = {TWO_PI - abs(s_end):+.4f}")
            check(abs(w_end) >= W_MAX - 1e-12,
                  f"OLD {name}/{tag}: omega SATURATES at {W_MAX} while the plan says straight",
                  f"omega = {w_end:+.4f}, k*state = {C.DYNANAV_K_ANGULAR * s_end:+.3f}"
                  if name != "holonomic" else f"omega = {w_end:+.4f}, 0.8*state = {0.8 * s_end:+.3f}")

    print("\n  guided-branch stress (guidance alternating +-(pi-0.05), plans moving):")
    old_stress: dict[str, list[Row]] = {}
    for tag, seq in stress.items():
        rows = run(make_old("guided"), seq)
        old_stress[tag] = rows
        print_trace(f"OLD guided, scenario '{tag}'", seq, rows, full)
        sign = +1 if tag.endswith("plus") else -1
        behind = [r for r in rows if r.phase == "behind"]
        ahead = [r for r in rows if r.phase == "ahead"]
        check(abs(behind[-1].state) > PI, f"OLD guided/{tag}: state leaves (-pi, pi] on guidance",
              f"state = {behind[-1].state:+.4f}")
        check(abs(abs(ahead[-1].state) - TWO_PI) < 0.05 and np.sign(ahead[-1].state) == sign
              and abs(ahead[-1].cmd.omega) >= W_MAX - 1e-12,
              f"OLD guided/{tag}: settles at {'+' if sign > 0 else '-'}2pi and omega saturates",
              f"state = {ahead[-1].state:+.4f}, omega = {ahead[-1].cmd.omega:+.3f}")

    head_crosscheck(seqs, stress)
    return old_rows, old_stress


# ---------------------------------------------------------------------------------------
# (b) The real classes
# ---------------------------------------------------------------------------------------


def section_b(seqs, stress, old_rows, old_stress, full: bool) -> None:
    banner("(b) REAL classes from nav/sim/controllers.py on the same sequence")
    print(f"  make_controller(name, {V_MAX}, {W_MAX}); controllers.py: PursuitController "
          f":{inspect.getsourcelines(C.PursuitController)[1]}, "
          f"HolonomicController :{inspect.getsourcelines(C.HolonomicController)[1]}, "
          f"GuidedPursuitController :{inspect.getsourcelines(C.GuidedPursuitController)[1]}")

    zero = Command(0.0, 0.0, 0.0)
    for tag, seq in seqs.items():
        # `seq` minus the STOPs: because STOPs must leave the filter untouched, the real
        # controllers' straight-phase output has to be IDENTICAL with and without them.
        seq_nostop = [s for s in seq if s.phase != "stop"]
        print(f"\n  --- scenario '{tag}' ---")
        for name in NAMES:
            rows = run(make_controller(name, V_MAX, W_MAX), seq)
            ctl_rows = run(make_controller(name, V_MAX, W_MAX), seq_nostop)
            if name == "pursuit":
                print_trace(f"NEW {name}, scenario '{tag}'", seq, rows, full)
            stops = [r for r in rows if r.phase == "stop"]
            straight = [r for r in rows if r.phase == "straight"]
            ctl_straight = [r for r in ctl_rows if r.phase == "straight"]
            first_stop = next(i for i, r in enumerate(rows) if r.phase == "stop")
            state_before = rows[first_stop - 1].state
            state_after = stops[-1].state
            max_state = max_abs_state(rows)
            max_straight_w = max(abs(r.cmd.omega) for r in straight)
            max_diff = max(abs(a.cmd.omega - b.cmd.omega) for a, b in zip(straight, ctl_straight))
            old_end = old_rows[tag][name][-1]
            print(f"  NEW {name:<9} {tag:<5} STOP cmds all (0,0,0): {all(r.cmd == zero for r in stops)}  "
                  f"state before/after STOPs {fnum(state_before)}/{fnum(state_after)}  max|state| "
                  f"{max_state:.4f}  max|omega| on straight {max_straight_w:.4f}  "
                  f"final omega {straight[-1].cmd.omega:+.5f}  (OLD final omega {old_end.cmd.omega:+.3f})")
            check(all(r.cmd == zero for r in stops),
                  f"NEW {name}/{tag}: every reframed STOP returns Command(0,0,0)",
                  f"{len(stops)} STOP plans")
            check(state_after == state_before,
                  f"NEW {name}/{tag}: the STOPs leave the filter state untouched",
                  f"{fnum(state_before, '+.9f')} -> {fnum(state_after, '+.9f')}")
            check(max_state <= PI,
                  f"NEW {name}/{tag}: |filter state| <= pi at every one of {len(rows)} steps",
                  f"max |state| = {max_state:.6f}")
            check(max_straight_w < 0.05,
                  f"NEW {name}/{tag}: |omega| < 0.05 on every straight plan after the STOPs",
                  f"max |omega| = {max_straight_w:.5f} over {len(straight)} plans "
                  f"(OLD pursuit-style sat at {W_MAX})")
            check(max_diff == 0.0,
                  f"NEW {name}/{tag}: straight-phase omega is IDENTICAL with the STOPs removed",
                  f"max |diff| = {max_diff:.1e}")

    # --- plain turning plan: the fix must change nothing where nothing was wrong -----
    print("\n  --- constant-curvature arc, kappa = +-0.35 1/m, 30 points x 0.1 s at 0.7 m/s, 10 repeated calls ---")
    sample = arc_plan(0.35)
    x_l, y_l, dist = C._lookahead_point(sample, C.DYNANAV_LOOKAHEAD_M)
    print(f"  lookahead point ({x_l:.3f}, {y_l:.3f}), dist {dist:.3f}, plan_speed {C.plan_speed(sample):.4f} m/s, "
          f"yaw_err {math.atan2(y_l, x_l):+.4f} rad")
    for name in NAMES:
        for kappa in (+0.35, -0.35):
            wp = arc_plan(kappa)
            variants: list[tuple[str, np.ndarray | None]] = [("", None)]
            if name == "guided":
                gd = np.array([[*wp_pt, math.atan2(wp_pt[1], wp_pt[0])]
                               for wp_pt in arc_plan(kappa, n=90)[[29, 59, 89]]])
                variants.append(("+guidance", gd))
            for vtag, gd in variants:
                new, old = make_controller(name, V_MAX, W_MAX), make_old(name)
                max_abs, max_ratio_dev, sign_ok = 0.0, 0.0, True
                for _ in range(10):
                    cn, co = new(wp, gd), old(wp, gd)
                    max_abs = max(max_abs, abs(cn.omega - co.omega), abs(cn.vx - co.vx),
                                  abs(cn.vy - co.vy))
                    sign_ok &= (np.sign(cn.omega) == np.sign(co.omega) == np.sign(kappa))
                    if abs(co.omega) > 1e-9:
                        max_ratio_dev = max(max_ratio_dev, abs(cn.omega / co.omega - 1.0))
                check(sign_ok and max_abs < 1e-9 and max_ratio_dev < 1e-9,
                      f"NEW {name}{vtag}/kappa {kappa:+.2f}: omega sign and magnitude unchanged vs old",
                      f"new omega {cn.omega:+.6f}, old {co.omega:+.6f}, max |diff| over 10 calls {max_abs:.1e}")

    # --- guided branch: second filter_angle call site -------------------------------
    print("\n  --- guided branch (filters the GUIDANCE heading): guidance alternating +-(pi-0.05) ---")
    for tag, seq in stress.items():
        sign = +1 if tag.endswith("plus") else -1
        rows = run(make_controller("guided", V_MAX, W_MAX), seq)
        ahead = [r for r in rows if r.phase == "ahead"]
        max_state = max_abs_state(rows)
        tail = ahead[-5:]
        print_trace(f"NEW guided, scenario '{tag}'", seq, rows, full)
        check(max_state <= PI, f"NEW guided/{tag}: |state| <= pi at every step",
              f"max |state| = {max_state:.6f}")
        check(max(abs(r.cmd.omega) for r in tail) < 0.05,
              f"NEW guided/{tag}: |omega| < 0.05 on the last 5 straight-ahead guidance steps",
              f"final omega = {ahead[-1].cmd.omega:+.6f} (OLD {old_stress[tag][-1].cmd.omega:+.3f}, "
              f"state {old_stress[tag][-1].state:+.3f})")


# ---------------------------------------------------------------------------------------
# (c) Replay of recorded runs
# ---------------------------------------------------------------------------------------


@dataclass
class Replay:
    path: Path
    episode: str
    lidar: str
    controller: str
    policy: str
    n_plans: int
    n_used: int
    n_stop_proxy: int
    n_behind: int
    old_cross: int
    old_outside: int
    old_first_t: float | None
    old_max: float
    old_final: float | None
    new_max: float
    new_stop_max: float
    new_cross: int
    blown: bool                 # path > 2 x speed cap x duration: a physics blow-up, not driving
    post_fix: bool              # carries `config`, which EpisodeResult only gained with the fix
    yaw_after: float | None     # |recorded yaw swept| in the 20 s after the replayed 1st crossing, rad
    yaw_any: float | None       # largest |recorded yaw swept| in ANY 20 s window, rad


def _yaw_sweeps(trace, elapsed_s: float, t_cross: float | None, window_s: float = 20.0):
    """Recorded yaw swept in `window_s`: after `t_cross`, and the worst window anywhere."""
    if len(trace) < 10 or elapsed_s <= 0:
        return None, None
    dt = elapsed_s / (len(trace) - 1)
    k = max(1, int(window_s / dt))
    u = [float(trace[0][2])]
    for t in trace[1:]:
        u.append(u[-1] + math.atan2(math.sin(t[2] - u[-1]), math.cos(t[2] - u[-1])))

    def sweep(i0: int) -> float:
        return abs(u[min(len(u) - 1, i0 + k)] - u[i0])

    worst = max(sweep(i) for i in range(len(u) - 1))
    after = None if t_cross is None else sweep(min(len(u) - 2, int(round(t_cross / dt))))
    return after, worst


def replay_file(path: Path, alpha: float) -> Replay | None:
    d = json.loads(path.read_text())
    plans = d.get("plans") or []
    if not plans:
        return None
    old = new = new_stop = None
    old_cross = old_out = n_used = n_stop = n_behind = new_cross = 0
    first_t = None
    old_max = new_max = new_stop_max = 0.0
    was_out = False
    for p in plans:
        if len(p) < 3 or not isinstance(p[1], (int, float)):
            continue
        t, heading, reach = float(p[0]), math.radians(float(p[1])), float(p[2])
        if reach < C._EPS:      # the controllers return Command(0,0,0) before the filter
            continue
        n_used += 1
        if abs(heading) > math.radians(150.0):
            n_behind += 1
        old = old_filter(old, heading, alpha)
        outside = abs(old) > PI
        if outside:
            old_out += 1
            if not was_out:
                old_cross += 1
                if first_t is None:
                    first_t = t
        was_out = outside
        old_max = max(old_max, abs(old))

        new = C.filter_angle(new, heading, alpha)
        new_max = max(new_max, abs(new))
        new_cross += abs(new) > PI

        # `plan_speed` is arc / (N * 0.1 s); under 0.002 m/s is < 2 cm of path for N <= 100,
        # which is every STOP (all waypoints one point => exactly 0.0). Needs the 6th field.
        spd = float(p[5]) if len(p) > 5 and p[5] is not None else None
        if spd is not None and spd < 0.002:
            n_stop += 1
            continue
        new_stop = C.filter_angle(new_stop, heading, alpha)
        new_stop_max = max(new_stop_max, abs(new_stop))
    yaw_after, yaw_any = _yaw_sweeps(d.get("trace") or [], float(d.get("elapsed_s") or 0.0), first_t)
    blown = float(d.get("path_length_m") or 0.0) > 2.0 * V_MAX * float(d.get("elapsed_s") or 0.0)
    return Replay(
        path=path, episode=str(d.get("episode", "?")), lidar=str(d.get("lidar") or "-"),
        controller=str(d.get("controller", "?")), policy=str(d.get("policy", "-")),
        n_plans=len(plans), n_used=n_used, n_stop_proxy=n_stop, n_behind=n_behind,
        old_cross=old_cross, old_outside=old_out, old_first_t=first_t, old_max=old_max,
        old_final=old, new_max=new_max, new_stop_max=new_stop_max, new_cross=int(new_cross),
        blown=blown, post_fix="config" in d, yaw_after=yaw_after, yaw_any=yaw_any,
    )


def section_c(results_dir: Path, alpha: float) -> None:
    banner(f"(c) replay of recorded runs  --  alpha = {alpha}, plan heading (2nd field) through old vs new filter")
    files = sorted(glob.glob(str(results_dir / "*.json")))
    reps: list[Replay] = []
    unreadable = 0
    for f in files:
        try:
            r = replay_file(Path(f), alpha)
        except Exception:  # noqa: BLE001
            unreadable += 1
            continue
        if r is not None:
            reps.append(r)
    n_calls = sum(r.n_used for r in reps)
    print(f"  {len(files)} result files in {results_dir.relative_to(REPO)}, {len(reps)} with a non-empty "
          f"'plans' list ({n_calls} filter updates replayed), {unreadable} unreadable")
    print("  old filter: skip a plan when its lookahead reach < 1e-3 m (controller returns before the filter),")
    print("  NEW(filter) = filter_angle on the same inputs; NEW(+stop) additionally drops plans whose logged")
    print("  plan_speed < 0.002 m/s (every STOP; is_stationary() is the real test, this is its proxy).")

    crossers = [r for r in reps if r.old_cross > 0]
    crossers.sort(key=lambda r: (r.episode, r.path.name))
    print(f"\n  OLD filter leaves (-pi, pi] in {len(crossers)} of {len(reps)} replayed runs:")
    if crossers:
        print(f"  {'run (timestamp_episode_controller)':<52} {'lidar':<8} {'ctrl':<8} {'calls':>5} "
              f"{'stop':>4} {'behind':>6} {'cross':>5} {'out':>4} {'1st cross t':>11} "
              f"{'max|old|':>8} {'final old':>9} {'yaw/20s':>8}")
        for r in crossers:
            flag = ("B" if r.blown else "") + ("P" if r.post_fix else "") + ("*" if r.controller == "guided" else "")
            ya = "-" if r.yaw_after is None else f"{math.degrees(r.yaw_after):.0f}"
            print(f"  {r.path.stem:<52} {r.lidar:<8} {r.controller:<8} {r.n_used:>5} {r.n_stop_proxy:>4} "
                  f"{r.n_behind:>6} {r.old_cross:>5} {r.old_outside:>4} {r.old_first_t:>10.1f}s "
                  f"{r.old_max:>8.3f} {r.old_final:>+9.3f} {ya:>8} {flag}")
        print("  columns: calls = filter updates replayed, stop = plans with plan_speed < 0.002, behind = plans")
        print("  > 150 deg off the nose, cross = times the state left the range, out = updates spent outside it,")
        print("  yaw/20s = degrees of yaw the RECORDED trace swept in the 20 s after the replayed first crossing.")
        print("  flags: B blown-up run (path > 2 x 1.5 m/s x duration), P recorded after the fix (has 'config'),")
        print("         * guided steers on guidance, which is not the heading replayed here.")

    print("\n  per (episode, lidar arm): runs replayed / runs where the OLD filter crossed")
    groups: dict[tuple[str, str], list[Replay]] = {}
    for r in reps:
        groups.setdefault((r.episode, r.lidar), []).append(r)
    print(f"  {'episode':<28} {'lidar':<8} {'runs':>4} {'crossed':>7} {'crossings':>9}")
    for (ep, ld), rs in sorted(groups.items()):
        c = [r for r in rs if r.old_cross > 0]
        if c:
            print(f"  {ep:<28} {ld:<8} {len(rs):>4} {len(c):>7} {sum(r.old_cross for r in c):>9}")
    by_ep: dict[str, list[Replay]] = {}
    for r in reps:
        by_ep.setdefault(r.episode, []).append(r)
    quiet = sorted(ep for ep, rs in by_ep.items() if all(r.old_cross == 0 for r in rs))
    print(f"  never crossed in ANY run or arm ({len(quiet)} episodes): {', '.join(quiet) if quiet else '-'}")
    print("  lidar '-' = the run predates the `lidar` field (episode.py:247: '' must not be read as fan).")

    n_blown = sum(1 for r in crossers if r.blown)
    print(f"\n  physics blow-ups among the crossing runs (flag B): {n_blown} of {len(crossers)}; their headings "
          f"after the blow-up are not driving.")
    if crossers:
        print(f"  in the crossing runs: {sum(r.n_behind for r in crossers)} plans point > 150 deg behind the robot, "
              f"{sum(r.n_stop_proxy for r in crossers)} are STOP-like (plan_speed < 0.002)")

    # Is the replay a model of what the robot DID? Compare against the recorded yaw. A
    # saturated omega is 1.2 rad/s = 24 rad = 1375 deg in 20 s; a full turn in 20 s is far
    # above anything driving a corridor produces. Pre-fix runs only: a run recorded after
    # the fix would not spin whatever the old filter says, and would decay this check.
    turn = 2.0 * PI
    pre = [r for r in reps if not r.post_fix and not r.blown]
    pre_cross = [r for r in pre if r.old_cross > 0 and r.yaw_after is not None]
    pre_quiet = [r for r in pre if r.old_cross == 0 and r.yaw_any is not None]
    hit = sum(1 for r in pre_cross if r.yaw_after >= turn)
    base = sum(1 for r in pre_quiet if r.yaw_any >= turn)
    if pre_cross and pre_quiet:
        p_hit, p_base = hit / len(pre_cross), base / len(pre_quiet)
        print("\n  does the replay predict what the robot did? (recorded yaw, pre-fix non-blown runs)")
        print(f"    replayed crossing  -> >= 1 full turn within 20 s of it : {hit}/{len(pre_cross)} runs ({100 * p_hit:.0f}%)")
        print(f"    no replayed crossing -> >= 1 full turn in ANY 20 s window: {base}/{len(pre_quiet)} runs ({100 * p_base:.1f}%)")
        ratio = p_hit / p_base if p_base > 0 else float("inf")
        print(f"    enrichment {ratio:.1f}x. NOT 100%: crossing is the trigger the replay can see, and the 08-31 runs")
        print("    may predate some controller changes; the non-crossing spins have another cause or older code.")
        check(p_hit >= 0.5 and ratio >= 3.0,
              "replayed OLD-filter crossings predict recorded spins (>= 50% of runs and >= 3x the baseline rate)",
              f"{hit}/{len(pre_cross)} = {100 * p_hit:.0f}% vs {100 * p_base:.1f}% baseline = {ratio:.1f}x")
    print("  caveat: the replay is open-loop. Headings after a crossing were produced by a robot that was spinning,")
    print("  so it shows WHEN the old filter leaves the range, not what a robot running the new code would have seen.")

    new_max = max((r.new_max for r in reps), default=0.0)
    new_stop_max = max((r.new_stop_max for r in reps), default=0.0)
    new_cross = sum(r.new_cross for r in reps)
    check(len(reps) > 0, "found recorded runs with a non-empty 'plans' list", f"{len(reps)} runs")
    check(new_cross == 0 and new_max <= PI,
          "NEW filter_angle never leaves (-pi, pi] on any recorded run",
          f"max |state| over {n_calls} updates = {new_max:.6f}, runs crossing = 0/{len(reps)}")
    check(new_stop_max <= PI, "NEW filter_angle + stationary skip never leaves (-pi, pi]",
          f"max |state| = {new_stop_max:.6f}")
    if not crossers:
        print("  NOTE: no recorded run crossed with the OLD filter; part (a) is then the only reproduction.")


# ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stops", type=int, default=16, help="reframed STOP plans (even; default 16)")
    ap.add_argument("--straight", type=int, default=25, help="straight plans after the STOPs")
    ap.add_argument("--alpha", type=float, default=C.DYNANAV_ALPHA_FILTER)
    ap.add_argument("--results", type=Path, default=REPO / "nav" / "results")
    ap.add_argument("--full", action="store_true", help="print every row of every trace")
    args = ap.parse_args()
    if args.stops % 2:
        ap.error("--stops must be even (the STOP phase must end on the upward-pushing update)")

    print(f"check_controller_wrap  --  {REPO}")
    print(f"controllers.py: {inspect.getsourcefile(C)}")

    seqs = {"plus": build_sequence(+1, args.stops, args.straight),
            "minus": build_sequence(-1, args.stops, args.straight)}
    stress = {"guided-stress-plus": build_guided_stress(+1, args.stops, args.straight),
              "guided-stress-minus": build_guided_stress(-1, args.stops, args.straight)}

    unit_checks()
    old_rows, old_stress = section_a(seqs, stress, args.full)
    section_b(seqs, stress, old_rows, old_stress, args.full)
    section_c(args.results, args.alpha)

    banner("SUMMARY")
    failed = [name for ok, name in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} assertions passed")
    for name in failed:
        print(f"  FAILED: {name}")
    print("  RESULT:", "PASS" if not failed else "FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
