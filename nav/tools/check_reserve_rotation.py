#!/usr/bin/env python3
"""Does re-serving a cached arc hand the robot back the right rest of it? No Isaac, no model.

    nav/tools/check_reserve_rotation.py                  # everything; exit 0 only if all pass
    nav/tools/check_reserve_rotation.py --grids operating
    nav/tools/check_reserve_rotation.py --trace 0.35 1.5 0.45   # every decision of one cell
    nav/tools/check_reserve_rotation.py --logs           # + pick-up periods from nav/results

Exit codes: 0 pass, 1 a criterion failed, 2 the code this check models has drifted.

WHY THIS EXISTS. `server_qwen.py` is asynchronous: while a ~2 s generation decodes, every
control call is answered with the PREVIOUS plan, re-expressed in the robot's current body
frame by `reframe(plan, dx, dy, dyaw, age)` and sliced by the plan's age. Two defects lived
in how the runner described that plan to the server, and neither raises or moves a counter:

  P1  the runner sent dx, dy and time_delay but NOT dyaw (`PredictRequest.yaw_delta_rad`
      defaulted to 0.0 and nothing set it), so a cached arc was translated into the new
      frame and never rotated. A robot already half-way round the arc was asked to take
      the same turn again from its new heading, on every re-serve: about twice the yaw the
      menu asked for. Fixed by client.py / run_navigation.py sending `yaw_delta_rad`.
  P2  the runner picked the reference BEFORE each call as `gen_starts[0]`. On the call at
      which a generation has just finished the server serves THAT plan (and spawns the
      next), but the runner still anchored on the previous generation's start. dx, dy,
      dyaw and time_delay were one pick-up period too old on about one call per generation
      (logged time_delay cycled 2.0 / 2.5 / 3.0 where 1.5 / 2.0 / 2.5 was true). Fixed by
      the runner also sending `pose_world` and `sim_time`: the server keeps the pose of the
      call that SPAWNED each generation and reframes against that plan's own start.

THREE CLIENTS ARE COMPARED, because the point is to tell each fix from its absence:
  old   pre-P1: no yaw_delta, anchor gen_starts[0].
  p1    what shipped before the P2 fix: yaw_delta sent, anchor still gen_starts[0].
  new   what is in the tree: the same values are still computed and sent, plus pose_world
        and sim_time, and the server anchors on the served plan's own generation start.
  fresh the control: the same controller driven on a plan generated at the current pose
        every call, no staleness. It bounds what pursuit itself does to a clean arc, so
        1.00x there means any other ratio is the re-serve.

WHAT IS REAL AND WHAT IS A MODEL. The arithmetic that decides the answer is real code: the
controller is `controllers.make_controller("braking", ...)`, the arc is
`arc_menu.plan_from_kappa`, and `reframe()` is the server's own text, executed (see
REFRAME_VERBATIM). Importing `server_qwen` would load torch and a 30 GB model and
`run_navigation` needs Kit, so the PROTOCOL between them -- what the server does when a call
arrives, what the runner does before and after one -- is re-written here in ~60 lines. Every
line of that rewrite is pinned to the source it mimics by ANCHORS: the check refuses to run
(exit 2, not 1) the moment any of them changes, because a replay of code that has moved on
is the worst kind of green. The guard's teeth are tested on every run (`selftest_guard`):
mutated copies of reframe(), of the server's anchor arithmetic and of the runner / client
call are fed to the same matcher and must all be rejected. The protocol model is also
checked against the world: its steady-state `time_delay` cycle at a 1.5 s pick-up period is
2.0 / 2.5 / 3.0, and so is the one in nav/results/20260905-092943_office_hallway_turn_braking.json.

THE LOOP BEING REPLAYED (async, the default: NAV_PLAN_PERIOD_S unset):
  * the runner calls /predict every `replan_every_steps` = 30 physics steps = 0.5 s;
  * on a call at which the server is idle it STARTS a generation on that call's frame and
    still answers with the cached plan; the generation lands G seconds later (sim time) and
    the next one starts at the first call after that, so the effective period P is G rounded
    UP to the 0.5 s call grid, and is printed;
  * every generation is "the same arc again": `plan_from_kappa(kappa, v, 100, 0.1)` in the
    body frame of the pose its frame was grabbed at;
  * the controller turns the live plan into a command HELD until the next call; the base
    integrates a unicycle at PHYSICS_DT.
Consequence worth knowing before reading any number: a plan is served at ages P..2P, never
0..P -- it was spawned at the previous pick-up and is replaced at the next one.

THE METRIC. Executed yaw over a FIXED DISTANCE against what the chosen arc asks for over
it, `kappa * s`. Distance, not time: an over-rotating robot also drives a different path,
and a time window would let it cover a different length. The window starts after the first
`--s-lo` metres (start-up) and spans `--dist` metres; the least-squares slope of yaw against
distance is printed beside the net figure, as a check that the net figure is not an accident
of where the window's edges fall.

SECTIONS (the verdict names the ones that count toward the exit code):
  1. identity   a robot placed exactly ON the cached arc must be handed back exactly the
                rest of it, against an independent SE(2) oracle, on EVERY call including
                the one that picks up a freshly finished plan. No controller, no loop,
                nothing chaotic. `new` must be exact; `p1` must be exact except on pick-up
                calls and wrong on every one of them; `old` must be wrong on every call.
  2. operating  v = baseline.yaml's QVLA_MENU_SPEED (0.45), G in {1.0, 1.5, 2.1}: where the
                logged async runs actually live (`--logs`: pick-up period p50 1.0 s, 82% <=
                1.5 s, 9% >= 2.5 s over 4286 intervals when this was written). GATED: `new`
                within 1.0 +- tol of kappa * s in every cell, AND `old` outside it in every
                cell, otherwise the criterion cannot tell the fix from its absence. Beside
                it, per cell: on how many served calls the client's anchor differed from the
                served plan's own start, the largest |time_delay| error that implied, and
                the age the server actually sliced by -- the P2 bug measured directly --
                and the server's own `anchor_fixes` counter, which must agree.
     2b. the same cells over a 60 m window. INFORMATION ONLY -- see WINDOW below.
  3. long G     INFORMATION ONLY, never gated: G in {2.1, 3.0, 4.2} at the server's default
                0.7 m/s (3a) and at the profile's 0.45 m/s (3b). A plan is served at ages
                P..2P, and the 3 m arc (`plan_from_kappa`, arc_len 3.0) is spent at 4.29 s at
                0.7 m/s and 6.67 s at 0.45, so with P = 2.5 / 3.0 / 4.5 s the plan driven has
                partly or wholly run out onto its straight tail: no client can hold the band,
                and the lateral error to the plan (`peak lateral error`) grows to metres.
                The last column re-runs `new` with a 12 m arc, which the server does not
                build, as the control: it returns to 0.99-1.08x at 0.7 m/s, so the cause is
                the arc, not the frame the plan is expressed in.
  4. model      the protocol model against the logged time_delay cycle (see above).

WINDOW. The first default was 30 m. At P = 2.5 s (G = 2.1) a same-arc-forever loop is not
stable: the first lateral excursion over 0.1 m comes at 44-72 s into the run for `new` (51 s
to never for `p1`), the error alternates in sign and grows ~1.5x per generation, and over 60 m
`new` under-rotates to 0.61-0.94x against `p1`'s 0.84-1.00x. It is not the arc running out, not
rounding (section 1 is exact), and not the frame the plan is expressed in: it is pursuit on a
plan 2.5-5 s old, and the exact anchor reaches it sooner than the stale one. (Measured while
writing this, not reproduced by the check: a 10 m arc gives the same 60 m ratios, and using the
server's anchor for only the geometry or only the age is as bad as using it for both, 0.62-0.64x
at kappa 0.15.) The gate is the steady-state turn rate, so it uses 20 m after the first 6 m of
start-up: that ends before the growth moves the ratio out of the band, although it has already
begun (peak lateral error 0.76-0.93 m at kappa 0.15 and 0.35). 2b prints the long window and a
`note:` flags every gated cell whose peak lateral error is over 0.25 m. `--dist 30` restores the
strict reading and then FAILS at G = 2.1 for kappa 0.15 and 0.35 (0.71x, 0.81x): the check being
honest about a loop that is marginal at P = 2.5 s, not about the re-expressed plan.

MEASURED (tree as of 2026-10-04; deterministic, re-run to reproduce):
  1. NEW exact to 2.4e-15 m on all 540 calls of 9 cells, 0 waypoints dropped, including the
     11-29 pick-up calls per cell. P1 client the same on every other call and off by 0.46-1.49 m
     on every pick-up call, with P / 0.1 s = 10 / 15 / 25 waypoints dropped. OLD off by 0.14-2.8 m
     on every call.
  2. old 1.92-2.03x, P1 client 0.94-1.00x, NEW 0.98-1.01x, fresh 1.00-1.01x, all nine cells.
     The client's anchor is not the served plan's start on 57/116, 38/117 and 23/117 served
     calls at P = 1.0 / 1.5 / 2.5 s; its time_delay is off by exactly P on those, the age the
     server slices by never. The server's `anchor_fixes` equals that count in every cell.
  4. the model's client time_delay cycles 3.0 / 2.0 / 2.5 at P = 1.5 s against a true age of
     1.5 / 2.0 / 2.5 s; the logged run shows 2.0 / 2.5 / 3.0 repeating, the same cycle.
KNOWN, NOT MODELLED: pivots and recoveries (a call off the 0.5 s grid; QVLA_MENU_PIVOTS is 0
in baseline.yaml), /replan, /reset, sync and bounded modes, the lidar menu filter. In those
the anchors still pin the lines but the replay does not exercise them.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import math
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "nav/policy_server/server_qwen.py"
CLIENT = REPO / "nav/policy_server/client.py"
RUNNER = REPO / "nav/sim/run_navigation.py"
BASE_DRIVE = REPO / "nav/sim/base_drive.py"
EPISODES = REPO / "nav/config/episodes.yaml"
PROFILE = REPO / "nav/config/profiles/baseline.yaml"
LOGGED_RUN = REPO / "nav/results/20260905-092943_office_hallway_turn_braking.json"

sys.path.insert(0, str(REPO / "nav/sim"))
sys.path.insert(0, str(REPO / "nav"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from arc_menu import CURVATURE_SETS, DEFAULT_LENGTH_M, plan_from_kappa  # noqa: E402
from controllers import Command, make_controller  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_DRIFT = 0, 1, 2
EXACT_M = 1e-12      # float rounding through a few sin/cos is ~1e-14; a float32 slip is 1e-7
WRONG_M = 1e-3       # "visibly wrong": a millimetre, far above rounding, far below a real error
CONTROL_ARC_M = 12.0   # longer than the plan can be consumed in the long-G grids (6.3 m)
LATERAL_NOTE_M = 0.25  # peak cross-track error above which a passing cell is annotated
CLIENTS = ("old", "p1", "new")


# --------------------------------------------------------------------------------------
# The server's own reframe(), verbatim, and the guard that keeps it verbatim.
# --------------------------------------------------------------------------------------
# Copied from nav/policy_server/server_qwen.py, `def reframe`. It is EXECUTED below, so the
# thing under test is this text and not a paraphrase of it; `verify_reframe()` compares it
# against the live file on every run and raises Drift on any difference at all, including
# the docstring, because "only the comment changed" is how a copy gets trusted after the
# code did change.
REFRAME_VERBATIM = r'''def reframe(plan: np.ndarray, dx: float, dy: float, dyaw: float,
            age_s: float) -> np.ndarray:
    """Express a plan written `age_s` ago in the CURRENT body frame, and drop what is spent.

    Two corrections, and both are needed:
      - spatial: the plan's origin was the robot's pose at generation. Subtract the
        translation since, then rotate by -dyaw. This is the same 2D form of
        `R_start.T @ delta_world` the runner already uses for dx, dy.
      - temporal: waypoint i was for t = (i+1)*0.1 s after generation. Points whose time
        has passed are behind the robot and are dropped. The plan therefore shortens as
        it ages, which is honest -- a 3 s plan really does run out 3 s later.
    """
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    rel = plan - np.array([dx, dy])
    out = np.stack([c * rel[:, 0] - s * rel[:, 1],
                    s * rel[:, 0] + c * rel[:, 1]], axis=-1)
    # round() first: an age built as a difference of two float times lands a hair above
    # an exact multiple of DT on ~40% of off-grid calls, and ceil would cut one point late.
    keep = int(np.ceil(round(max(0.0, age_s) / DT, 9)))
    return out[keep:]'''


class Drift(Exception):
    """The code this check models has changed. Raised, never swallowed: exit 2."""


def _code_tokens(text: str) -> list[tuple[str, int]]:
    """(token text, 1-based line) with comments, blank lines and layout removed.

    Python's own tokenizer, so a comment inserted between two lines of a statement -- which
    is how most of the edits this check is pinned against have been made -- does not hide
    the statement, while a changed name, operator or literal still does. A snippet that
    ends mid-statement is fine: the tokens produced before the error are kept.
    """
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENDMARKER}
    out: list[tuple[str, int]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in skip:
                out.append((tok.string, tok.start[0]))
    except (tokenize.TokenError, IndentationError):
        pass
    return out


class Source:
    """A source file that can be searched ignoring comments and layout, by line number."""

    def __init__(self, path: Path, text: str | None = None):
        self.path = path
        self.text = path.read_text() if text is None else text
        self._tokens = _code_tokens(self.text)

    def rel(self) -> str:
        return str(self.path.relative_to(REPO))

    def find(self, snippet: str) -> int | None:
        """Line of the first token of `snippet`, matching whole tokens in order."""
        want = [t for t, _ in _code_tokens(snippet)]
        words = [t for t, _ in self._tokens]
        for i in range(len(words) - len(want) + 1):
            if words[i:i + len(want)] == want:
                return self._tokens[i][1]
        return None

    def function(self, name: str) -> tuple[str, int]:
        for node in ast.parse(self.text).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                seg = ast.get_source_segment(self.text, node)
                assert seg is not None
                return seg, node.lineno
        raise Drift(f"{self.rel()}: no top-level function {name!r}")

    def constant(self, name: str):
        """Evaluate a top-level `NAME = <literal arithmetic>` (e.g. `1.0 / 60.0`)."""
        for node in ast.parse(self.text).body:
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                try:
                    return eval(compile(ast.Expression(node.value), "<const>", "eval"),
                                {"__builtins__": {}}, {})
                except Exception as exc:
                    raise Drift(f"{self.rel()}: {name} is no longer a plain literal "
                                f"expression ({exc})") from exc
        raise Drift(f"{self.rel()}: no top-level constant {name!r}")


def _rstrip_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines())


def verify_reframe(copy: str, src: Source) -> int:
    live, lineno = src.function("reframe")
    if _rstrip_lines(live) != _rstrip_lines(copy):
        diff = "\n".join(difflib.unified_diff(
            _rstrip_lines(copy).splitlines(), _rstrip_lines(live).splitlines(),
            "check_reserve_rotation.REFRAME_VERBATIM",
            f"{src.rel()}:{lineno} reframe()", lineterm=""))
        raise Drift("reframe() in the server no longer matches the copy this check "
                    "executes. Re-read the new function, decide whether the replay still "
                    "models it, then paste it into REFRAME_VERBATIM:\n" + diff)
    return lineno


# (file, snippet, what the replay relies on it for). Matched by whole tokens, so layout and
# line wrapping may change; names, operators and literals may not.
ANCHORS: list[tuple[Path, str, str]] = [
    # --- the runner: cadence, controller, the reference, and what goes on the wire
    (RUNNER, "PHYSICS_DT = 1.0 / 60.0", "physics step"),
    (RUNNER, 'PLAN_PERIOD_S = float(os.environ.get("NAV_PLAN_PERIOD_S", "") or 0.0)',
     "no fixed planning period unless asked"),
    (RUNNER, 'PLAN_MODE = (os.environ.get("NAV_PLAN_MODE", "").strip().lower() or '
             '("sync" if PLAN_PERIOD_S > 0 else "async"))',
     "with no period set the mode is async"),
    (RUNNER, "if PLAN_PERIOD_S > 0: return max(1, round(PLAN_PERIOD_S / PHYSICS_DT)) "
             "return episode_default", "decision period falls back to the episode's"),
    (RUNNER, "replan_steps = plan_period_steps(ep.replan_every_steps)", "decision period"),
    (RUNNER, "next_decision_step = step + replan_steps", "next call is one period on"),
    (RUNNER, "controller = make_controller( self.controller_name, ep.max_speed_mps, "
             "ep.max_yaw_rate_radps ) controller.reset()", "controller and its limits"),
    (RUNNER, '"--controller", default="braking",', "controller default"),
    (RUNNER, "sim_time = step * PHYSICS_DT", "sim_time is the step count times the step"),
    (RUNNER, "ref_step, ref_x, ref_y, ref_yaw = gen_starts[-1 if bounded else 0]",
     "which start the CLIENT's pose delta is taken from (async: [0], one period stale "
     "on a pick-up call)"),
    (RUNNER, "time_delay = (step - ref_step) * PHYSICS_DT", "the client's staleness"),
    (RUNNER, "dxw = position[0] - ref_x dyw = position[1] - ref_y", "world displacement"),
    (RUNNER, "dx = cos_r * dxw + sin_r * dyw dy = -sin_r * dxw + cos_r * dyw",
     "displacement into the reference body frame"),
    (RUNNER, "yaw_delta = math.atan2(math.sin(self.base.yaw - ref_yaw), "
             "math.cos(self.base.yaw - ref_yaw))", "the client's yaw delta"),
    (RUNNER, "command.vx, command.vy, 0.0, command.omega, dx, dy,", "robot_state layout"),
    (RUNNER, "time_delay=time_delay,", "time_delay is sent"),
    (RUNNER, "yaw_delta_rad=yaw_delta,", "yaw delta is sent"),
    (RUNNER, "pose_world=[float(position[0]), float(position[1]), float(self.base.yaw)],",
     "THIS call's pose is sent: the server's anchor needs it"),
    (RUNNER, "sim_time=float(sim_time),", "THIS call's clock is sent"),
    (RUNNER, "gen_starts.append((int(gen_step), position[0], position[1], self.base.yaw)) "
             "del gen_starts[:-2]", "a start is recorded AFTER predict(), last two kept"),
    (RUNNER, 'round(out["plan_age_s"] if out.get("plan_age_s") is not None else time_delay, 3),',
     "the logged staleness is the server's, when it reports one"),
    (RUNNER, "command = controller(waypoints, guidance)", "one command per call, held"),
    (RUNNER, "guard.vx, guard.vy, drive.omega, PHYSICS_DT,", "base is driven every step"),
    (CLIENT, "yaw_delta_rad: float = 0.0, pose_world: list[float] | None = None, "
             "sim_time: float | None = None,", "predict() takes all three"),
    (CLIENT, '"yaw_delta_rad": yaw_delta_rad,', "the client puts the yaw delta on the wire"),
    (CLIENT, '"pose_world": pose_world,', "the client puts the pose on the wire"),
    (CLIENT, '"sim_time": sim_time,', "the client puts the clock on the wire"),
    (CLIENT, '"time_delay": time_delay,', "time_delay on the wire"),
    # --- the base: a unicycle
    (BASE_DRIVE, "dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt", "x uses OLD yaw"),
    (BASE_DRIVE, "dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt", "y uses OLD yaw"),
    # apply() routes the step through _resolve_motion() (the contact seam), which hands
    # back (dx, dy, omega * dt) untouched when no motion_filter is attached -- the base
    # this replay models -- and, with the runner's contact layer attached, returns the
    # very same objects on every step nothing blocks (kinematic_contact.motion_filter),
    # so this model is the free-space base either way. A blocked step is a different base.
    (BASE_DRIVE, "dx, dy, dyaw = self._resolve_motion(dx, dy, omega * dt, dt, float(t[2]))",
     "the commanded yaw step goes through the seam"),
    (BASE_DRIVE, "if self.motion_filter is None: return dx, dy, dyaw",
     "with no filter the seam passes the command through"),
    (BASE_DRIVE, "self._yaw = yaw + dyaw", "yaw is integrated, not wrapped"),
    # --- the server: what a call does
    (SERVER, "yaw_delta_rad: float = 0.0", "the request field exists"),
    (SERVER, "pose_world: list[float] | None = None sim_time: float | None = None",
     "the request carries the caller's pose and clock"),
    (SERVER, 'busy = _state["gen_thread"] is not None', "busy == a generation is running"),
    (SERVER, "if req.pose_world is not None and len(req.pose_world) >= 3 and "
             "req.sim_time is not None: gen_pose = (float(req.pose_world[0]), "
             "float(req.pose_world[1]), float(req.pose_world[2]), float(req.sim_time))",
     "a call's pose is (x, y, yaw, sim_time), or None if either field is missing"),
    (SERVER, "req.stalled_s, req.scan_points, gen_pose), daemon=True)",
     "the spawning call's pose goes to the generation it spawns"),
    (SERVER, "elif not busy: started_step = req.current_step th = _spawn() "
             "if not have_plan: th.join(timeout=GEN_TIMEOUT_S)",
     "an idle server spawns on the call, and only the very first call waits"),
    (SERVER, "vlm_generation_start_step=started_step,", "the spawn is reported back"),
    (SERVER, '_state["plan"] = plan _state["plan_gen_step"] = step '
             '_state["plan_gen_pose"] = gen_pose',
     "a finished generation replaces the cached plan AND its start pose"),
    (SERVER, 'finally: with _lock: _state["gen_thread"] = None',
     "and the server is idle again from that moment"),
    (SERVER, 'plan = _state["plan"] plan_gen_pose = _state["plan_gen_pose"]',
     "plan and its start pose are read together"),
    (SERVER, "dx, dy = (req.robot_state + [0.0] * 6)[4:6] "
             "dyaw, age = float(req.yaw_delta_rad), max(0.0, float(req.time_delay))",
     "fallback: the CLIENT's dx, dy, yaw delta and age"),
    (SERVER, "if gen_pose is not None and plan_gen_pose is not None: "
             "gx, gy, gyaw, gt = plan_gen_pose x, y, yaw, t = gen_pose wx, wy = x - gx, y - gy",
     "server anchor: needs both poses, displacement is this call minus the plan's start"),
    (SERVER, "c, s_ = math.cos(gyaw), math.sin(gyaw) "
             "dx, dy = c * wx + s_ * wy, -s_ * wx + c * wy",
     "server anchor: displacement into the plan's own start frame"),
    (SERVER, "dyaw = math.atan2(math.sin(yaw - gyaw), math.cos(yaw - gyaw))",
     "server anchor: wrapped yaw delta since the plan's start"),
    (SERVER, "own_age = max(0.0, t - gt) if abs(own_age - age) > 1e-3: with _lock: "
             '_state["anchor_fixes"] += 1 age = own_age',
     "server anchor: age is sim-time since the plan's start; a disagreement is counted"),
    (SERVER, "full = reframe(plan, float(dx), float(dy), dyaw, 0.0) "
             "live = full[int(np.ceil(round(age / DT, 9))):] if len(live) == 0: live = full[-1:]",
     "reframe with the chosen yaw delta, slice by the chosen age, hold the last point"),
    (SERVER, "plan = plan_from_kappa(kappa, speed, N_WAYPOINTS, DT)",
     "an arc choice becomes a plan"),
    (SERVER, 'MENU_SPEED_MPS = float(os.environ.get("QVLA_MENU_SPEED", "0.7"))',
     "cruise speed default (0.7 m/s)"),
    (SERVER, 'THINK_LEVEL = os.environ.get("QVLA_THINK_LEVEL", "medium").strip()',
     "default level, hence the 10 s horizon"),
    (SERVER, 'HORIZON_S = float(os.environ.get("QVLA_HORIZON_S", LEVEL["horizon_s"]))',
     "horizon comes from the level"),
    (SERVER, "N_WAYPOINTS = int(round(HORIZON_S * _CONTROL_HZ))", "plan length"),
    (SERVER, "DT = HORIZON_S / N_WAYPOINTS", "plan spacing"),
]


def verify_anchors(texts: dict[Path, str] | None = None) -> list[tuple[str, int, str]]:
    """Every ANCHOR must match. `texts` swaps in other text for a path (the guard self-test)."""
    texts = texts or {}
    sources: dict[Path, Source] = {}
    found, missing = [], []
    for path, snippet, why in ANCHORS:
        src = sources.setdefault(path, Source(path, texts.get(path)))
        line = src.find(snippet)
        if line is None:
            missing.append(f"  {src.rel()}: {snippet!r}\n      (replay relies on it for: {why})")
        else:
            found.append((src.rel(), line, why))
    if missing:
        raise Drift("source lines the replay is pinned to are gone or changed:\n"
                    + "\n".join(missing))
    return found


# (label, file, regex, replacement): each mutation is a plausible one-line edit that would
# make the replay model something the code no longer does. Applied to COPIES of the text.
SOURCE_MUTATIONS: list[tuple[str, Path, str, str]] = [
    ("server: anchor rotation sign flipped", SERVER,
     r"-s_ \* wx \+ c \* wy", "s_ * wx + c * wy"),
    ("server: anchor yaw delta dropped", SERVER,
     r"dyaw = math\.atan2\(math\.sin\(yaw - gyaw\), math\.cos\(yaw - gyaw\)\)", "dyaw = 0.0"),
    ("server: anchor age discarded", SERVER, r"\n\s+age = own_age\n", "\n"),
    ("server: reframe() called with the sign flipped", SERVER,
     r"full = reframe\(plan, float\(dx\), float\(dy\), dyaw, 0\.0\)",
     "full = reframe(plan, float(dx), float(dy), -dyaw, 0.0)"),
    ("server: anchor start pose not kept with the plan", SERVER,
     r'\n\s+_state\["plan_gen_pose"\] = gen_pose\n', "\n"),
    ("runner: pose_world dropped from the call", RUNNER,
     r"pose_world=\[float\(position\[0\]\).*?float\(self\.base\.yaw\)\],", ""),
    ("runner: sim_time dropped from the call", RUNNER, r"sim_time=float\(sim_time\),", ""),
    ("runner: yaw_delta dropped from the call", RUNNER, r"yaw_delta_rad=yaw_delta,", ""),
    ("client: pose_world not put on the wire", CLIENT, r'"pose_world": pose_world,', ""),
]


# --------------------------------------------------------------------------------------
# Constants, read from source rather than typed here a second time.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    physics_dt: float
    replan_steps: int
    v_max: float
    w_max: float
    horizon_s: float
    n_waypoints: int
    dt: float
    cruise_task: float          # server default, 0.7 m/s
    cruise_profile: float | None  # baseline.yaml's QVLA_MENU_SPEED, if present
    arc_len: float
    controller: str


def load_config() -> Config:
    runner, server = Source(RUNNER), Source(SERVER)
    physics_dt = float(runner.constant("PHYSICS_DT"))
    ep = yaml.safe_load(EPISODES.read_text())
    overrides = [k for k, e in ep["episodes"].items()
                 if {"replan_every_steps", "max_speed_mps", "max_yaw_rate_radps"} & set(e)]
    if overrides:
        raise Drift(f"{EPISODES.name}: episodes {overrides[:3]} override the defaults the "
                    f"replay uses; model them or restrict the replay to the defaults")
    d = ep["defaults"]
    levels = server.constant("_LEVELS")
    horizon = float(levels["medium"]["horizon_s"])
    hz = float(server.constant("_CONTROL_HZ"))
    n = int(round(horizon * hz))
    prof = yaml.safe_load(PROFILE.read_text()) if PROFILE.exists() else {}
    prof_speed = (prof.get("env") or {}).get("QVLA_MENU_SPEED")
    m = re.search(r'MENU_SPEED_MPS = float\(os\.environ\.get\("QVLA_MENU_SPEED", "([0-9.]+)"\)\)',
                  server.text)
    assert m, "MENU_SPEED_MPS default not found (anchor should have caught this)"
    return Config(
        physics_dt=physics_dt, replan_steps=int(d["replan_every_steps"]),
        v_max=float(d["max_speed_mps"]), w_max=float(d["max_yaw_rate_radps"]),
        horizon_s=horizon, n_waypoints=n, dt=horizon / n,
        cruise_task=float(m.group(1)),
        cruise_profile=float(prof_speed) if prof_speed is not None else None,
        arc_len=float(DEFAULT_LENGTH_M), controller="braking")


# --------------------------------------------------------------------------------------
# The replay.
# --------------------------------------------------------------------------------------
def _exec_reframe(cfg: Config):
    ns = {"math": math, "np": np, "DT": cfg.dt}
    exec(compile(REFRAME_VERBATIM, "<server_qwen.reframe>", "exec"), ns)
    return ns["reframe"]


def runner_pose_delta(ref: tuple[float, float, float], pose: tuple[float, float, float]):
    """dx, dy, yaw_delta exactly as run_navigation.py computes them from a reference start.

    Pinned by the RUNNER anchors on `dxw`/`dyw`, `dx`/`dy` and `yaw_delta`.
    """
    ref_x, ref_y, ref_yaw = ref
    x, y, yaw = pose
    dxw, dyw = x - ref_x, y - ref_y
    cos_r, sin_r = math.cos(ref_yaw), math.sin(ref_yaw)
    dx = cos_r * dxw + sin_r * dyw
    dy = -sin_r * dxw + cos_r * dyw
    yaw_delta = math.atan2(math.sin(yaw - ref_yaw), math.cos(yaw - ref_yaw))
    return dx, dy, yaw_delta


class Served(NamedTuple):
    live: np.ndarray
    started: int | None        # step of the generation this call spawned, if it did
    served_from: int | None    # start step of the plan this call is answered with
    age_used: float            # the age the slice was cut at
    dyaw_used: float


class ArcServer:
    """What server_qwen.py does with a call, for a model that always picks one arc.

    Generations are scheduled in whole physics steps. A generation started at step s lands
    at s + latency, at which point it replaces the cached plan and the server is idle; the
    NEXT call (not that instant) is what starts the following one -- exactly the ordering
    the anchors pin: spawn happens inside predict(), commit inside the worker thread. A
    generation carries the pose of the call that spawned it, as `gen_pose` does there.
    """

    def __init__(self, cfg: Config, reframe, kappa: float, speed: float, latency_steps: int,
                 arc_len_m: float | None = None):
        self.cfg, self.reframe = cfg, reframe
        self.kappa, self.speed, self.latency = kappa, speed, latency_steps
        self.arc_len_m = cfg.arc_len if arc_len_m is None else arc_len_m   # control only
        self.plan: np.ndarray | None = None
        self.plan_start: int | None = None      # the server's own `plan_gen_step`
        self.plan_pose = None                   # ... and `plan_gen_pose`
        self.running = None                     # (start, lands, plan, pose)
        self.anchor_fixes = 0                   # the server's own counter, same condition

    def _commit(self, step: int) -> None:
        if self.running is not None and step >= self.running[1]:
            self.plan_start, self.plan, self.plan_pose = (
                self.running[0], self.running[2], self.running[3])
            self.running = None

    def predict(self, step: int, robot_state: list[float], time_delay: float,
                yaw_delta_rad: float, pose_world=None, sim_time=None) -> Served:
        self._commit(step)
        started = None
        busy, have_plan = self.running is not None, self.plan is not None
        gen_pose = None
        if pose_world is not None and len(pose_world) >= 3 and sim_time is not None:
            gen_pose = (float(pose_world[0]), float(pose_world[1]), float(pose_world[2]),
                        float(sim_time))
        if not busy:                                           # `elif not busy:` in predict()
            started = step
            plan = plan_from_kappa(self.kappa, self.speed, self.cfg.n_waypoints, self.cfg.dt,
                                   self.arc_len_m)
            if not have_plan:                                  # the very first call waits
                self.plan_start, self.plan, self.plan_pose = step, plan, gen_pose
            else:
                self.running = (step, step + self.latency, plan, gen_pose)
        dx, dy = (robot_state + [0.0] * 6)[4:6]
        dyaw, age = float(yaw_delta_rad), max(0.0, float(time_delay))
        if gen_pose is not None and self.plan_pose is not None:
            gx, gy, gyaw, gt = self.plan_pose
            x, y, yaw, t = gen_pose
            wx, wy = x - gx, y - gy
            c, s_ = math.cos(gyaw), math.sin(gyaw)
            dx, dy = c * wx + s_ * wy, -s_ * wx + c * wy
            dyaw = math.atan2(math.sin(yaw - gyaw), math.cos(yaw - gyaw))
            own_age = max(0.0, t - gt)
            if abs(own_age - age) > 1e-3:
                self.anchor_fixes += 1
            age = own_age
        full = self.reframe(self.plan, float(dx), float(dy), dyaw, 0.0)
        live = full[int(np.ceil(round(age / self.cfg.dt, 9))):]
        if len(live) == 0:
            live = full[-1:]
        return Served(live, started, self.plan_start, age, dyaw)


def client_call(server: ArcServer, client: str, step: int, dt: float,
                pose: tuple[float, float, float], command: Command, gen_starts: list):
    """One /predict call as run_navigation.py makes it, for each generation of the client.

    All three compute the same thing from `gen_starts[0]` (async); they differ in what they
    send. -> (client_ref, time_delay_sent, Served).
    """
    ref = None
    time_delay = dx = dy = yaw_delta = 0.0
    if gen_starts:
        ref = gen_starts[0][0]                                             # async: [0]
        dx, dy, yaw_delta = runner_pose_delta(gen_starts[0][1:], pose)
        time_delay = (step - ref) * dt
        if client == "old":
            yaw_delta = 0.0                                                # no yaw_delta yet
    extra = {}
    if client == "new":
        extra = dict(pose_world=[pose[0], pose[1], pose[2]], sim_time=step * dt)
    served = server.predict(step, [command.vx, command.vy, 0.0, command.omega, dx, dy],
                            time_delay, yaw_delta, **extra)
    if served.started is not None:                  # recorded AFTER predict(), last two kept
        gen_starts.append((int(served.started), *pose))
        del gen_starts[:-2]
    return ref, time_delay, served


class Decision(NamedTuple):
    step: int
    client_ref: int | None     # start step the CLIENT measured from (None before the first)
    served_from: int | None    # start step of the plan the server actually served
    td_sent: float             # time_delay the client computed and sent
    age_used: float            # age the server sliced by
    dyaw_used: float           # yaw delta the server rotated by
    n_live: int
    y_first: float


@dataclass
class Trace:
    s: np.ndarray
    yaw: np.ndarray
    decisions: list[Decision]
    stalled: bool
    anchor_fixes: int


def simulate(cfg: Config, reframe, kappa: float, speed: float, latency_s: float, *,
             client: str, dist: float, t_max_s: float = 600.0,
             arc_len_m: float | None = None) -> Trace:
    """One closed loop. `client` is one of old / p1 / new (see the module docstring) or
    "fresh": a plan generated at the current pose on every call, no staleness at all.
    `arc_len_m` is a CONTROL, not something the server does: a longer arc than 3 m, to ask
    whether a miss is the arc running out."""
    dt = cfg.physics_dt
    server = ArcServer(cfg, reframe, kappa, speed, max(1, round(latency_s / dt)), arc_len_m)
    ctrl = make_controller(cfg.controller, cfg.v_max, cfg.w_max)
    ctrl.reset()
    x = y = yaw = s = 0.0
    gen_starts: list[tuple[int, float, float, float]] = []
    command = Command(0.0, 0.0, 0.0)
    next_decision, step = 0, 0
    S, YAW, decisions = [0.0], [0.0], []
    fresh_plan = plan_from_kappa(kappa, speed, cfg.n_waypoints, cfg.dt)
    while s < dist and step * dt < t_max_s:
        if step == next_decision:
            if client == "fresh":
                waypoints = fresh_plan
            else:
                ref, td, sv = client_call(server, client, step, dt, (x, y, yaw), command,
                                          gen_starts)
                decisions.append(Decision(step, ref, sv.served_from, td, sv.age_used,
                                          sv.dyaw_used, len(sv.live), float(sv.live[0][1])))
                waypoints = sv.live
            command = ctrl(np.asarray(waypoints, dtype=float), None)
            next_decision = step + cfg.replan_steps
        # base.apply(): unicycle, x and y from the OLD yaw (base_drive.py)
        if command.vx != 0.0 or command.vy != 0.0 or command.omega != 0.0:
            dxw = (command.vx * math.cos(yaw) - command.vy * math.sin(yaw)) * dt
            dyw = (command.vx * math.sin(yaw) + command.vy * math.cos(yaw)) * dt
            yaw += command.omega * dt
            x += dxw
            y += dyw
            s += math.hypot(dxw, dyw)
        step += 1
        S.append(s)
        YAW.append(yaw)
    return Trace(np.asarray(S), np.asarray(YAW), decisions, stalled=s < dist,
                 anchor_fixes=server.anchor_fixes)


@dataclass
class Window:
    net_deg: float
    asked_deg: float
    ratio: float
    slope_ratio: float
    peak_lat_m: float


def measure(tr: Trace, kappa: float, s_lo: float, dist: float, dt: float) -> Window:
    s_hi = s_lo + dist
    asked = math.degrees(kappa * dist)
    if tr.s[-1] < s_hi:
        return Window(float("nan"), asked, float("nan"), float("nan"), float("nan"))
    net = float(np.interp(s_hi, tr.s, tr.yaw) - np.interp(s_lo, tr.s, tr.yaw))
    mask = (tr.s >= s_lo) & (tr.s <= s_hi)
    slope = float(np.polyfit(tr.s[mask], tr.yaw[mask], 1)[0])
    lat = [abs(d.y_first) for d in tr.decisions
           if s_lo <= tr.s[min(d.step, len(tr.s) - 1)] <= s_hi]
    return Window(math.degrees(net), asked, net / (kappa * dist), slope / kappa,
                  max(lat) if lat else float("nan"))


@dataclass
class Anchor:
    """The P2 bug, measured: how the client's anchor compared with the plan actually served."""
    n: int             # served calls that had a client reference at all
    n_mismatch: int    # client reference != start of the plan served
    td_err: float      # max |time_delay the client SENT - the served plan's true age|
    age_err: float     # max |age the server SLICED BY - the served plan's true age|
    fixes: int         # the server's own anchor_fixes counter


def anchor_stats(tr: Trace, dt: float) -> Anchor:
    served = [d for d in tr.decisions if d.client_ref is not None]
    true = [(d.step - d.served_from) * dt for d in served]
    return Anchor(len(served), sum(1 for d in served if d.client_ref != d.served_from),
                  max((abs(d.td_sent - t) for d, t in zip(served, true)), default=0.0),
                  max((abs(d.age_used - t) for d, t in zip(served, true)), default=0.0),
                  tr.anchor_fixes)


def effective_period_s(cfg: Config, latency_s: float) -> float:
    steps = max(1, round(latency_s / cfg.physics_dt))
    return math.ceil(steps / cfg.replan_steps) * cfg.replan_steps * cfg.physics_dt


def arc_left(cfg: Config, speed: float, period_s: float) -> float:
    """Share of the served window [P, 2P) of plan age that still lies on the 3 m arc.

    A plan is served at ages P..2P (spawned at the previous pickup, replaced at the next),
    never 0..P. Past arc_len / speed the plan is a straight line along its final heading,
    so below 1.0 the plan being driven no longer contains the whole turn it was chosen for.
    """
    return min(1.0, max(0.0, (cfg.arc_len / speed - period_s) / period_s))


# --------------------------------------------------------------------------------------
# 1. Identity: a robot exactly on the cached arc, against an independent oracle.
# --------------------------------------------------------------------------------------
def _se2(p) -> np.ndarray:
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([[c, -s, p[0]], [s, c, p[1]], [0.0, 0.0, 1.0]])


def _on_circle(g0, kappa: float, speed: float, t: float) -> tuple[float, float, float]:
    """World pose of a robot that has driven `t` s at constant curvature from pose g0."""
    h = kappa * speed * t
    local = (math.sin(h) / kappa, (1.0 - math.cos(h)) / kappa, h) if abs(kappa) > 1e-9 \
        else (speed * t, 0.0, 0.0)
    w = _se2(g0) @ np.array([local[0], local[1], 1.0])
    return float(w[0]), float(w[1]), g0[2] + h


class CallId(NamedTuple):
    pickup: bool         # client reference != served plan's start
    age_s: float
    err_m: float         # worst waypoint error over the waypoints both sides still hold
    dropped: int         # waypoints the true remainder has and the served one lacks
    on_arc: float        # robot vs the served plan's pose at that age (m / rad, the larger)


def identity_run(cfg: Config, reframe, kappa: float, speed: float, latency_s: float,
                 client: str, g0=(3.0, -2.0, 1.0), t_end_s: float = 30.0) -> list[CallId]:
    """Drive a robot kinematically along ONE circle (every generation is the same arc, so
    the robot is on every plan's arc at its own age), call on the runner's cadence, and
    compare each served plan with the true remainder computed from poses by SE(2) matrices
    -- a different path to the answer from reframe()'s `rel`/`rotate`. g0 is neither the
    origin nor axis-aligned, so a dropped rotation or a swapped sin/cos cannot hide."""
    dt = cfg.physics_dt
    server = ArcServer(cfg, reframe, kappa, speed, max(1, round(latency_s / dt)))
    plan = plan_from_kappa(kappa, speed, cfg.n_waypoints, cfg.dt)
    gen_starts: list = []
    start_pose: dict[int, tuple[float, float, float]] = {}
    out: list[CallId] = []
    for step in range(0, int(round(t_end_s / dt)), cfg.replan_steps):
        pose = _on_circle(g0, kappa, speed, step * dt)
        ref, _, sv = client_call(server, client, step, dt, pose, Command(0.0, 0.0, 0.0),
                                 gen_starts)
        if sv.started is not None:
            start_pose[sv.started] = pose
        gp = start_pose[sv.served_from]
        age_steps = step - sv.served_from
        keep_true = math.ceil(round(age_steps * dt / cfg.dt, 9))
        keep_served = len(plan) - len(sv.live)
        want = (np.c_[plan, np.ones(len(plan))] @ (np.linalg.inv(_se2(pose)) @ _se2(gp)).T)[:, :2]
        lo = max(keep_true, keep_served)
        err = float(np.max(np.hypot(*(sv.live[lo - keep_served:] - want[lo:]).T)))
        k = int(round(age_steps * dt / cfg.dt))                           # waypoint index k-1
        here = plan[k - 1] if k > 0 else (0.0, 0.0)                       # age 0: the origin
        pw = _se2(gp) @ np.array([here[0], here[1], 1.0])
        ph = gp[2] + kappa * min(speed * age_steps * dt, cfg.arc_len)
        on_arc = max(math.hypot(pw[0] - pose[0], pw[1] - pose[1]),
                     abs(math.atan2(math.sin(ph - pose[2]), math.cos(ph - pose[2]))))
        out.append(CallId(ref is not None and ref != sv.served_from, age_steps * dt, err,
                          keep_served - keep_true, on_arc))
    return out


def _wrong(c: CallId) -> bool:
    return c.err_m > WRONG_M or c.dropped != 0


def identity_section(cfg, reframe, kappas, speed, latencies) -> tuple[list[str], bool]:
    lines = ["  kappa    G     P | calls pickups  on-arc | NEW: all calls  pickup calls  dropped"
             " | P1 client anchor: other calls  pickup calls  dropped | OLD: worst   best"]
    ok = True
    for kappa in kappas:
        for g in latencies:
            runs = {c: identity_run(cfg, reframe, kappa, speed, g, c) for c in CLIENTS}
            new, p1, old = runs["new"], runs["p1"], runs["old"]
            pick = [c for c in p1 if c.pickup]
            other = [c for c in p1 if not c.pickup]
            on_arc = max(c.on_arc for c in new)
            e_new = max(c.err_m for c in new)
            e_new_pick = max((c.err_m for c in new if c.pickup), default=0.0)
            d_new = max(abs(c.dropped) for c in new)
            e_oth = max((c.err_m for c in other), default=0.0)
            d_oth = max((abs(c.dropped) for c in other), default=0)
            e_pick = max((c.err_m for c in pick), default=0.0)
            d_pick = max((c.dropped for c in pick), default=0)
            moved = [c for c in old if c.age_s > 0]
            e_old_hi = max(c.err_m for c in moved)
            e_old_lo = min(c.err_m for c in moved)
            row_ok = (on_arc < 1e-9 and e_new <= EXACT_M and d_new == 0
                      and e_oth <= EXACT_M and d_oth == 0
                      and len(pick) > 0 and all(_wrong(c) for c in pick)
                      and all(_wrong(c) for c in moved))
            ok = ok and row_ok
            lines.append(
                f"  {kappa:5.2f} {g:4.1f}s {effective_period_s(cfg, g):4.1f}s | "
                f"{len(new):5d} {len(pick):7d} {on_arc:7.0e} | "
                f"{e_new:14.1e} {e_new_pick:13.1e} {d_new:8d} | "
                f"{e_oth:26.1e} {e_pick:12.3f}m {d_pick:7d} | "
                f"{e_old_hi:8.3f}m {e_old_lo:6.3f}m{'' if row_ok else '  FAIL'}")
    return lines, ok


# --------------------------------------------------------------------------------------
# 3. The protocol model against the log.
# --------------------------------------------------------------------------------------
def model_cycle(cfg: Config, reframe, speed: float) -> tuple[list[float], list[float]]:
    """Steady-state time_delay the client SENDS and the served plan's TRUE age, at P = 1.5 s."""
    tr = simulate(cfg, reframe, 0.35, speed, 1.5, client="p1", dist=12.0)
    served = [d for d in tr.decisions if d.client_ref is not None][-6:]
    return ([round(d.td_sent, 3) for d in served],
            [round((d.step - d.served_from) * cfg.physics_dt, 3) for d in served])


def logged_cycle() -> list[float] | None:
    """The logged time_delay of decisions 14-25 (steady state) of the run the model is held to."""
    import json

    if not LOGGED_RUN.exists():
        return None
    plans = json.loads(LOGGED_RUN.read_text()).get("plans") or []
    return [p[6] for p in plans[14:26]] if len(plans) >= 26 else None


def ceil_noise(cfg: Config) -> tuple[float, float]:
    """Share of calls whose server age lands one waypoint late, on / off the 30-step grid.

    The server cuts `ceil((t - gt) / DT)` where t and gt are `step * PHYSICS_DT`: a
    difference of two rounded products, where the old client sent `(step - ref) * dt`, one
    rounded product. Counted only at the ages where it can matter, exact multiples of DT.
    """
    dt, per = cfg.physics_dt, int(round(cfg.dt / cfg.physics_dt))

    def share(starts, ages) -> float:
        bad = tot = 0
        for ref in starts:
            for n in ages:
                tot += 1
                bad += math.ceil(round(max(0.0, (ref + n) * dt - ref * dt) / cfg.dt, 9)) != n // per
        return bad / tot

    g = cfg.replan_steps
    return (share(range(0, 36000, g), range(g, 10 * g, g)),
            share(range(0, 6000), range(per, 10 * g, per)))


# --------------------------------------------------------------------------------------
def selftest_guard() -> str:
    """The drift guard must actually fire. A guard that cannot fail is decoration."""
    src = Source(SERVER)
    n = 0
    for label, mutate in (("rotation sign flipped", lambda t: t.replace("-dyaw", "dyaw")),
                          ("age dropped", lambda t: t.replace("out[keep:]", "out")),
                          ("docstring edited", lambda t: t.replace("honest", "dishonest"))):
        mutated = mutate(REFRAME_VERBATIM)
        assert mutated != REFRAME_VERBATIM, f"selftest mutation {label!r} changed nothing"
        try:
            verify_reframe(mutated, src)
        except Drift:
            n += 1
            continue
        raise SystemExit(f"selftest: a copy with {label} was NOT rejected -- the guard is dead")
    for label, path, pattern, repl in SOURCE_MUTATIONS:
        text = path.read_text()
        mutated, hits = re.subn(pattern, repl, text, count=1, flags=re.S)
        assert hits == 1 and mutated != text, f"selftest mutation {label!r} changed nothing"
        try:
            verify_anchors({path: mutated})
        except Drift:
            n += 1
            continue
        raise SystemExit(f"selftest: a copy with {label} was NOT rejected -- the guard is dead")
    return f"{n}/{n} mutated copies of the server, runner and client rejected"


def logged_pickup_periods() -> list[float]:
    """Seconds between plan pick-ups in the async arc-menu runs on disk (`--logs`).

    A pick-up shows as the logged `time_delay` (plans[i][6]) dropping by more than half a
    call. Needs nav/results, which is gitignored: absent data is "no sample", not an error.
    Runs logged before the P2 fix carry the client's time_delay, later ones the server's;
    either way the DROP is the pick-up.
    """
    import json

    out: list[float] = []
    for f in sorted((REPO / "nav/results").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("plan_mode") != "async" or "[menu]" not in str(d.get("policy", "")):
            continue
        plans = d.get("plans") or []
        td, tt = [p[6] for p in plans], [p[0] for p in plans]
        drops = [i for i in range(1, len(td)) if td[i] < td[i - 1] - 0.4]
        out += [tt[b] - tt[a] for a, b in zip(drops, drops[1:])]
    return out


# --------------------------------------------------------------------------------------
@dataclass
class Cell:
    speed: float
    kappa: float
    g: float
    period: float
    arc_left: float
    w: dict[str, Window]       # fresh / old / p1 / new
    anchors: dict[str, Anchor]  # p1 / new
    mirror_ok: bool


def run_cell(cfg, reframe, kappa, v, g, s_lo, dist, control_arc_m=None) -> Cell:
    kw = dict(dist=s_lo + dist)
    runs = {c: simulate(cfg, reframe, kappa, v, g, client=c, **kw)
            for c in ("fresh",) + CLIENTS}
    if control_arc_m is not None:
        runs["new_long_arc"] = simulate(cfg, reframe, kappa, v, g, client="new",
                                        arc_len_m=control_arc_m, **kw)
    mir = simulate(cfg, reframe, -kappa, v, g, client="new", **kw)
    w = {n: measure(tr, kappa, s_lo, dist, cfg.physics_dt) for n, tr in runs.items()}
    wm = measure(mir, -kappa, s_lo, dist, cfg.physics_dt)
    sym = abs(wm.net_deg + w["new"].net_deg) <= 1e-6 * max(1.0, abs(w["new"].net_deg))
    return Cell(v, kappa, g, effective_period_s(cfg, g), arc_left(cfg, v, effective_period_s(cfg, g)),
                w, {c: anchor_stats(runs[c], cfg.physics_dt) for c in ("p1", "new")}, sym)


def cell_ok(c: Cell, tol: float) -> bool:
    r = c.w["new"].ratio
    return not math.isnan(r) and abs(r - 1.0) <= tol and c.mirror_ok


def cell_discriminates(c: Cell, tol: float) -> bool:
    r = c.w["old"].ratio
    return not math.isnan(r) and abs(r - 1.0) > tol


def print_grid(title: str, cells: list[Cell], tol: float, gated: bool,
               anchors: bool = True) -> None:
    print(f"\n== {title} ==")
    ctl = "new_long_arc" in cells[0].w
    print(f"{'v':>4} {'kappa':>5} {'G':>4} {'P':>5} | {'asked':>6} | {'fresh':>6} | "
          f"{'OLD (pre-P1)':>20} | {'P1 client, own anchor':>22} | "
          f"{'NEW, server anchor':>22} | {'peak lateral error':>18} {'arc left':>8}"
          + (f" | {'NEW, 12 m arc (control)':>23}" if ctl else ""))
    for c in cells:
        w = c.w
        flag = ""
        if gated:
            flag = ("" if cell_ok(c, tol) else "  FAIL") + (
                "" if cell_discriminates(c, tol) else "  NOT DISCRIMINATING")
        print(f"{c.speed:4.2f} {c.kappa:5.2f} {c.g:4.1f} {c.period:4.1f}s | "
              f"{w['new'].asked_deg:5.0f}d | {w['fresh'].ratio:5.2f}x | "
              + " | ".join(f"{w[k].ratio:6.2f}x (slope {w[k].slope_ratio:4.2f})"
                            for k in ("old", "p1", "new"))
              + f" | p1 {w['p1'].peak_lat_m:5.2f}m new {w['new'].peak_lat_m:5.2f}m "
              f"{100 * c.arc_left:7.0f}%"
              + (f" | {w['new_long_arc'].ratio:6.2f}x (slope {w['new_long_arc'].slope_ratio:4.2f})"
                 if ctl else "") + flag)
    if not anchors:
        return
    print("  anchor, per cell (served calls whose client reference is NOT the start of the "
          "plan served; times in s):")
    print(f"  {'kappa':>5} {'G':>4} {'P':>5} | {'P1 client: ref!=served':>23} {'td sent err':>12} "
          f"{'age used err':>13} | {'NEW: ref!=served':>17} {'td sent err':>12} "
          f"{'age used err':>13} {'server anchor_fixes':>20}")
    for c in cells:
        a, b = c.anchors["p1"], c.anchors["new"]
        print(f"  {c.kappa:5.2f} {c.g:4.1f} {c.period:4.1f}s | "
              f"{a.n_mismatch:>13d}/{a.n:<9d} {a.td_err:12.2f} {a.age_err:13.2f} | "
              f"{b.n_mismatch:>9d}/{b.n:<7d} {b.td_err:12.2f} {b.age_err:13.2f} "
              f"{b.fixes:20d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.15, 0.35, 0.60])
    ap.add_argument("--operating-latencies", type=float, nargs="+", default=[1.0, 1.5, 2.1],
                    help="generation latency G in sim seconds, the range logged async runs "
                         "actually have (see --logs)")
    ap.add_argument("--long-latencies", type=float, nargs="+", default=[2.1, 3.0, 4.2],
                    help="G of the information-only long-latency grids")
    ap.add_argument("--long-speed", type=float, default=None,
                    help="default: the server's own default (0.7 m/s)")
    ap.add_argument("--operating-speed", type=float, default=None,
                    help="default: baseline.yaml QVLA_MENU_SPEED (0.45 m/s)")
    ap.add_argument("--dist", type=float, default=20.0,
                    help="fixed distance window, m. 30 was the first default; at P = 2.5 s "
                         "(G = 2.1) the idealised loop starts to oscillate after ~40 s, so a "
                         "30 m window measures that instead of the turn rate. See 2b.")
    ap.add_argument("--long-dist", type=float, default=60.0,
                    help="window of the information-only repeat of the operating grid, m")
    ap.add_argument("--s-lo", type=float, default=6.0, help="metres skipped as start-up")
    ap.add_argument("--tol", type=float, default=0.10, help="|new ratio - 1| must be <= this")
    ap.add_argument("--grids", choices=("operating", "info", "both"), default="both")
    ap.add_argument("--logs", action="store_true",
                    help="also print pick-up periods measured from nav/results (information)")
    ap.add_argument("--trace", type=float, nargs=3, metavar=("KAPPA", "G", "V"),
                    help="print every decision of one cell")
    ap.add_argument("--client", choices=CLIENTS, default="new",
                    help="which client --trace replays (default: new)")
    args = ap.parse_args()

    try:
        print("== source anchors (the replay is pinned to these lines) ==")
        reframe_line = verify_reframe(REFRAME_VERBATIM, Source(SERVER))
        print(f"  {SERVER.relative_to(REPO)}:{reframe_line}  reframe() == REFRAME_VERBATIM "
              f"(byte-for-byte, docstring included)")
        for rel, line, why in verify_anchors():
            print(f"  {rel}:{line}  {why}")
        print(f"  guard self-test: {selftest_guard()}")
        cfg = load_config()
    except Drift as exc:
        print(f"\nDRIFT -- the code under test is no longer what this check models:\n{exc}",
              file=sys.stderr)
        return EXIT_DRIFT

    offered = {abs(k) for k in CURVATURE_SETS["coarse"]}
    for k in args.kappas:
        if round(abs(k), 2) not in offered:
            print(f"warning: kappa {k} is not on the coarse menu {sorted(offered)}")
    v_long = args.long_speed or cfg.cruise_task
    v_oper = args.operating_speed or cfg.cruise_profile or cfg.cruise_task
    reframe = _exec_reframe(cfg)

    print("\n== model constants (all read from source) ==")
    print(f"  PHYSICS_DT {cfg.physics_dt:.6f} s | a call every {cfg.replan_steps} steps = "
          f"{cfg.replan_steps * cfg.physics_dt:.2f} s | async (client reference = gen_starts[0])")
    print(f"  controller {cfg.controller}, v_max {cfg.v_max} m/s, w_max {cfg.w_max} rad/s")
    print(f"  plan: {cfg.n_waypoints} points x {cfg.dt:.2f} s = {cfg.horizon_s:.0f} s; "
          f"arc {cfg.arc_len:.1f} m then straight on the final heading")
    print(f"  cruise: server default {cfg.cruise_task} m/s; baseline.yaml QVLA_MENU_SPEED = "
          f"{cfg.cruise_profile}")
    print(f"  window: {args.dist:.0f} m after the first {args.s_lo:.0f} m; "
          f"criterion: new within 1.0 +- {args.tol}, old outside it")

    if args.trace:
        kappa, g, v = args.trace
        tr = simulate(cfg, reframe, kappa, v, g, client=args.client, dist=args.s_lo + args.dist)
        print(f"\n== decisions: kappa {kappa}, G {g} s, v {v} ({args.client} client) ==")
        print("  step   t(s)  client_ref  served_from  td_sent  age_used  dyaw_used(deg)"
              "  live_pts  y_first(m)")
        for d in tr.decisions:
            flag = ("  <-- client anchor != served plan"
                    if d.client_ref is not None and d.client_ref != d.served_from else "")
            print(f"  {d.step:5d} {d.step * cfg.physics_dt:6.1f}  {str(d.client_ref):>10}  "
                  f"{str(d.served_from):>11}  {d.td_sent:7.2f}  {d.age_used:8.2f}  "
                  f"{math.degrees(d.dyaw_used):14.1f}  {d.n_live:8d}  {d.y_first:10.2f}{flag}")
        return EXIT_OK

    verdicts: dict[str, bool | None] = {}

    # ---- 1. identity: does re-serving reproduce the exact rest of the arc? ----------------
    print(f"\n== 1. re-serve identity (robot exactly on the arc, v {v_oper} m/s, "
          f"G {args.operating_latencies}; no controller) ==")
    lines, id_ok = identity_section(cfg, reframe, args.kappas, v_oper, args.operating_latencies)
    print("\n".join(lines))
    verdicts["identity"] = id_ok
    print(f"  -> {'PASS' if id_ok else 'FAIL'}: NEW exact to {EXACT_M:.0e} m with no waypoint "
          f"dropped, on pick-up calls too; P1 client exact except on pick-up calls, wrong on "
          f"every one of them (err > {WRONG_M * 1000:.0f} mm or points dropped); OLD wrong "
          f"on every call")

    # ---- 2/4. closed loops --------------------------------------------------------------
    # (key, title, speed, latencies, window m, counts toward the exit code)
    grids = []
    if args.grids in ("operating", "both"):
        grids.append(("operating", f"2. closed loop, the operating regime of logged runs "
                      f"(v {v_oper} m/s, G {args.operating_latencies}, {args.dist:.0f} m "
                      f"window) -- GATED", v_oper, args.operating_latencies, args.dist, True))
    if args.grids in ("info", "both"):
        grids.append(("2b", f"2b. INFORMATION ONLY, not gated: the same cells over a "
                      f"{args.long_dist:.0f} m window. At P = 2.5 s the closed loop oscillates "
                      f"laterally (peak lateral error) and under-rotates once the window reaches "
                      f"it. Section 1 shows the re-expressed plan is exact, so this is pursuit "
                      f"on a plan 2.5-5 s old, not the frame it is expressed in; the exact "
                      f"anchor is the worse of the two clients here",
                      v_oper, args.operating_latencies, args.long_dist, False))
        note = ("The last column repeats NEW with a 12 m arc, which the server does not build: "
                "where it is back inside the band the miss was the 3 m arc running out inside "
                "the served window of ages P..2P, not the anchor")
        grids.append(("3a", f"3a. INFORMATION ONLY, not gated: long latencies at the server's "
                      f"default speed (v {v_long} m/s, G {args.long_latencies}). The 3 m arc is "
                      f"spent at {cfg.arc_len / v_long:.2f} s. {note}",
                      v_long, args.long_latencies, args.dist, False))
        if v_oper != v_long:
            grids.append(("3b", f"3b. INFORMATION ONLY, not gated: the same long latencies at "
                          f"the baseline profile's speed (v {v_oper} m/s, G "
                          f"{args.long_latencies}). The 3 m arc is spent at "
                          f"{cfg.arc_len / v_oper:.2f} s",
                          v_oper, args.long_latencies, args.dist, False))
    all_cells: dict[str, list[Cell]] = {}
    for name, title, v, gs, dist, gated in grids:
        cells = [run_cell(cfg, reframe, k, v, g, args.s_lo, dist,
                          CONTROL_ARC_M if name.startswith("3") else None)
                 for k in args.kappas for g in gs]
        all_cells[name] = cells
        print_grid(title, cells, args.tol, gated, anchors=name != "2b")
        ratios = {k: [c.w[k].ratio for c in cells if not math.isnan(c.w[k].ratio)]
                  for k in ("old", "p1", "new")}
        print("  " + " | ".join(
            f"{lbl}: {min(r):.2f}x .. {max(r):.2f}x (mean {sum(r) / len(r):.2f}x)"
            for lbl, r in (("old", ratios["old"]), ("p1 client", ratios["p1"]),
                           ("new", ratios["new"])) if r))
        if not gated:
            print(f"  [not in the exit code] {sum(cell_ok(c, args.tol) for c in cells)} of "
                  f"{len(cells)} cells have new within 1.0 +- {args.tol}")
            continue
        bad = [c for c in cells if not cell_ok(c, args.tol)]
        blind = [c for c in cells if not cell_discriminates(c, args.tol)]
        verdicts[name] = not bad and not blind
        print(f"  -> {'PASS' if verdicts[name] else 'FAIL'}: new within 1.0 +- {args.tol} in "
              f"{len(cells) - len(bad)} of {len(cells)} cells; old outside it in "
              f"{len(cells) - len(blind)} of {len(cells)}")
        if blind:
            print("  NOT DISCRIMINATING: the old client also lands inside the tolerance in "
                  "some cell, so this criterion cannot tell the fix from its absence")
        wob = [c for c in cells if c.w["new"].peak_lat_m > LATERAL_NOTE_M]
        if wob:
            print(f"  note: in {len(wob)} gated cell(s) (P = "
                  f"{', '.join(sorted({f'{c.period:.1f}' for c in wob}))} s) the new client's "
                  f"peak lateral error is {max(c.w['new'].peak_lat_m for c in wob):.2f} m: the "
                  f"loop is oscillating there. A ratio inside the band means the window ends "
                  f"before that growth moves it out; 2b shows a longer one")

    # ---- 3. the protocol model against the log -------------------------------------------
    sent, true = model_cycle(cfg, reframe, v_oper)
    cyc_ok = (sorted(set(sent[-3:])) == [2.0, 2.5, 3.0] and sent[:3] == sent[3:]
              and sorted(set(true[-3:])) == [1.5, 2.0, 2.5] and true[:3] == true[3:])
    verdicts["model cycle"] = cyc_ok
    print("\n== 4. the protocol model against the world (P = 1.5 s, steady state) ==")
    print(f"  client sends time_delay  {sent}")
    print(f"  true age of the plan     {true}")
    logged = logged_cycle()
    print(f"  logged, {LOGGED_RUN.name}, decisions 14-25: "
          f"{logged if logged is not None else 'not on this machine (information only)'}")
    print(f"  -> {'PASS' if cyc_ok else 'FAIL'}: the client's cycle is 2.0 / 2.5 / 3.0 and the "
          f"truth 1.5 / 2.0 / 2.5, one period apart on the pick-up call only")
    on_grid, off_grid = ceil_noise(cfg)
    print(f"  information: the server cuts ceil(round((t - gt) / DT, 9)) from two rounded sim times; at "
          f"ages that are exact multiples of DT that is one waypoint late on "
          f"{100 * on_grid:.1f}% of calls on the 0.5 s call grid and "
          f"{100 * off_grid:.0f}% off it (calls shifted by an in-place turn)")

    if args.logs:
        per = logged_pickup_periods()
        print("\n== pick-up period of plans in logged async arc-menu runs (nav/results) ==")
        if per:
            a = np.asarray(per)
            q = {p: float(np.percentile(a, p)) for p in (25, 50, 75, 90, 95, 99)}
            print(f"  n = {len(a)} intervals | " + "  ".join(f"p{p} {v:.1f}s" for p, v in q.items())
                  + f" | <= 1.5 s: {100 * (a <= 1.5).mean():.0f}%  >= 2.5 s: "
                  f"{100 * (a >= 2.5).mean():.1f}%  >= 4.5 s: {100 * (a >= 4.5).mean():.1f}%")
        else:
            print("  no async [menu] result files found")

    gated_cells = all_cells.get("operating", [])
    mirror = [c.mirror_ok for c in gated_cells]
    if mirror:
        print(f"\n  mirror symmetry (yaw of -kappa == -yaw of kappa, new client): "
              f"{sum(mirror)}/{len(mirror)} operating cells exact")
        verdicts["mirror"] = all(mirror)
    fixes_ok = all(c.anchors["new"].fixes == c.anchors["new"].n_mismatch for c in gated_cells)
    if gated_cells:
        verdicts["server counter"] = fixes_ok
        print(f"  server anchor_fixes == client-vs-served mismatches in every operating cell: "
              f"{fixes_ok}")

    print("\n== verdict ==")
    for k, v in verdicts.items():
        print(f"  {k:26s} {'PASS' if v else 'FAIL'}")
    info = [n for n in all_cells if n != "operating"]
    if info:
        print(f"  {'information, not gated':26s} {', '.join(info)}")
    return EXIT_OK if all(verdicts.values()) else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
