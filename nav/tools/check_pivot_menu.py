#!/usr/bin/env python3
"""Prove the in-place-turn menu is well formed, without loading a model.

    nav/tools/check_pivot_menu.py                       # all checks, writes a sample menu
    nav/tools/check_pivot_menu.py --frame some.jpg      # render onto a chosen frame
    nav/tools/check_pivot_menu.py --out /tmp/pivot      # where the sample menus go

WHY THIS EXISTS. The pivot adds an action the model can choose, and every part of that
addition fails QUIETLY:

  * a label collision (an arc and a pivot both numbered 5) makes one of them
    unselectable and the other ambiguous -- and produces a perfectly normal-looking run
  * a `_STOP_LABEL` that did not move when the menu grew turns "stop" into "turn right"
  * a glyph drawn below the horizon lands on the floor among the arcs and stops reading
    as "in place"
  * a prompt that names the wrong two numbers asks the model to pick paths that are not
    the turns, and the model will comply
  * a sign flip sends the robot the other way, which looks like a bad decision rather
    than a bad wire

None of those raise. All of them yield a complete result file. The closed-loop run costs
an hour of GPU time and answers "did success go up?", which is not a question that can
distinguish a working pivot from a broken one -- so the wiring is checked here, on the
CPU, in a second, before the campaign is built on it.

The checks are deliberately structural rather than perceptual. Whether the model UNDER-
STANDS the glyph is what the ladder measures; whether the glyph is where the prompt says
it is, and carries the number the parser will map back to a rotation, is decidable here.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))
sys.path.insert(0, str(REPO / "nav/policy_server"))

# Pivots are OFF by default in the server, which is the right default for a benchmark and
# the wrong one for a test of pivots. Set before the import, because both flags are read
# at module scope and `_STOP_LABEL` is derived from them there.
os.environ.setdefault("QVLA_FORMAT", "menu")
os.environ["QVLA_MENU_PIVOTS"] = "1"

import numpy as np  # noqa: E402
from arc_menu import (  # noqa: E402
    PIVOT_DEG, PIVOT_RAD, allocate_labels, make_arcs, parse_choice_speed, pivot_word,
    render_menu, select_system, stop_label,
)

FRAMES = Path("/tmp/alohamini-nav-frames")


def _fail(msgs: list[str], cond: bool, msg: str) -> None:
    if not cond:
        msgs.append(msg)


def check_allocation(fails: list[str], n_arcs: int) -> tuple[list[int], tuple[int, int]]:
    """Labels are unique, cover 1..N, and STOP sits one past all of them."""
    stop = stop_label(n_arcs, True)
    _fail(fails, stop == n_arcs + 3,
          f"stop_label is {stop}, expected {n_arcs + 3} ({n_arcs} arcs + 2 pivots + 1)")
    _fail(fails, stop_label(n_arcs, False) == n_arcs + 1,
          "the no-pivot menu's STOP label moved -- earlier runs are no longer described "
          "by this code")

    # Every call, not one: the shuffle is per call and a collision that happens one time
    # in fifty is still a collision in a thirteen-episode ladder.
    rng = np.random.default_rng(0)
    seen_pivots = set()
    labels = pivots = None
    for _ in range(200):
        labels, pivots = allocate_labels(rng, n_arcs, True)
        _fail(fails, pivots is not None, "allocate_labels returned no pivot labels")
        if pivots is None:
            break
        allocated = list(labels) + list(pivots)
        _fail(fails, sorted(allocated) == list(range(1, n_arcs + 3)),
              f"labels are not a permutation of 1..{n_arcs + 2}: {sorted(allocated)}")
        _fail(fails, stop not in allocated,
              f"STOP's label {stop} was also handed to a path or a pivot")
        seen_pivots.add(pivots)
        if fails:
            break
    # The shuffle has to actually shuffle. A frozen pair would pass every check above and
    # reintroduce exactly the confound the permutation exists to remove.
    _fail(fails, len(seen_pivots) > 4,
          f"the pivot labels barely move: only {len(seen_pivots)} distinct pairs in 200 "
          f"calls -- the shuffle is not reaching them")
    return list(labels or []), tuple(pivots or (0, 0))


def check_signs(fails: list[str]) -> None:
    """Left glyph turns left, and the words match the sign."""
    _fail(fails, PIVOT_RAD > 0, f"PIVOT_RAD is {PIVOT_RAD}, expected a positive magnitude")
    _fail(fails, abs(math.degrees(PIVOT_RAD) - PIVOT_DEG) < 1e-9,
          "PIVOT_RAD and PIVOT_DEG disagree")
    _fail(fails, "left" in pivot_word(+PIVOT_RAD),
          f"pivot_word(+) says {pivot_word(+PIVOT_RAD)!r}, expected a left turn")
    _fail(fails, "right" in pivot_word(-PIVOT_RAD),
          f"pivot_word(-) says {pivot_word(-PIVOT_RAD)!r}, expected a right turn")


def check_mapping(fails: list[str], stop: int, pivots: tuple[int, int]) -> None:
    """The label -> radians map the server builds, checked at its two ends.

    Rebuilt here rather than imported because it is one line inside `menu_plan`, behind
    two generations. One line is worth restating; an hour of GPU time to reach it is not.
    """
    by_pivot = {int(pivots[0]): PIVOT_RAD, int(pivots[1]): -PIVOT_RAD}
    _fail(fails, by_pivot[pivots[0]] > 0,
          "the LEFT glyph does not map to a positive (leftward) rotation")
    _fail(fails, by_pivot[pivots[1]] < 0,
          "the RIGHT glyph does not map to a negative (rightward) rotation")
    _fail(fails, stop not in by_pivot, "STOP is reachable as a pivot")


def check_prompt(fails: list[str], stop_label: int, pivots: tuple[int, int]) -> str:
    """The selector prompt names both turns, and only when they are on the menu."""
    left, right = pivots
    with_p = select_system(stop_label, stop_allowed=True, speed_choice=False,
                           pivot_labels=pivots)
    without = select_system(stop_label, stop_allowed=True, speed_choice=False)

    for label, side in ((left, "left"), (right, "right")):
        _fail(fails, str(label) in with_p,
              f"the prompt never names the {side} turn's label ({label})")
    _fail(fails, "in place" in with_p, "the prompt never says the turns are in place")
    _fail(fails, f"{PIVOT_DEG:.0f} degrees" in with_p,
          f"the prompt never states the turn angle ({PIVOT_DEG:.0f} degrees)")
    _fail(fails, str(stop_label) in with_p, "the prompt never names the STOP label")
    # The anti-spinning brake. A turn is always collision-free and therefore always the
    # safe answer; without a cost attached it is how an episode is spent pirouetting.
    _fail(fails, "spinning" in with_p,
          "the prompt has no sentence warning against repeated turning")
    # And the no-pivot prompt must be unchanged in the ways that matter: it is what every
    # earlier measurement was taken under.
    _fail(fails, "in place" not in without,
          "the NO-pivot prompt mentions turning in place -- the two variants have leaked")
    return with_p


def check_parsing(fails: list[str], pivots: tuple[int, int]) -> None:
    """A reply naming a pivot survives the parser that was written for paths."""
    left, right = pivots
    for reply, want in ((f"{left}", left), (f"{right}", right),
                        (f" {left}\n", left), (f"Answer: {right}", right)):
        got, level = parse_choice_speed(reply)
        _fail(fails, got == want,
              f"parse_choice_speed({reply!r}) gave {got}, expected {want}")
        _fail(fails, level is None,
              f"parse_choice_speed({reply!r}) invented speed level {level}")
    # With the speed question open the model answers two numbers, and the pivot has to
    # survive being the first of them -- a turn in place has no speed, and a parser that
    # read the pair the other way round would turn "turn left, slowly" into a path.
    got, level = parse_choice_speed(f"{left} 3")
    _fail(fails, got == left and level == 3,
          f"parse_choice_speed('{left} 3') gave ({got}, {level}), expected ({left}, 3)")


def check_render(fails: list[str], frame: Path, out_dir: Path,
                 labels: list[int], pivots: tuple[int, int]) -> list[Path]:
    """Draw a real frame both ways and check the glyphs land above the horizon.

    Above the horizon is the point of the placement and it is a geometric guarantee, not
    a hope: `project` puts every floor point below h/2, so the top band is the one region
    of the frame no arc can ever reach. A glyph drawn among the arcs would read as another
    place to drive to.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    arcs = make_arcs()
    plain = out_dir / "menu_no_pivots.jpg"
    withp = out_dir / "menu_with_pivots.jpg"
    render_menu(str(frame), str(plain), arcs, labels)
    render_menu(str(frame), str(withp), arcs, labels, pivot_labels=pivots)

    a, b = Image.open(plain).convert("RGB"), Image.open(withp).convert("RGB")
    _fail(fails, a.size == b.size, f"the two renders differ in size: {a.size} vs {b.size}")
    if a.size != b.size:
        return [plain, withp]
    w, h = a.size

    # Where the two images differ IS where the glyphs were drawn, so no assumption about
    # the drawing code is needed to find them -- only that the rest of the frame is
    # untouched, which is itself worth checking.
    diff = np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)).sum(axis=2)
    ys, xs = np.nonzero(diff > 24)
    _fail(fails, len(ys) > 0, "the pivot glyphs changed nothing in the image")
    if len(ys) == 0:
        return [plain, withp]
    _fail(fails, ys.max() < h / 2,
          f"a pivot glyph reaches y={ys.max()} of {h} -- it is drawn on the floor, "
          f"among the arcs, instead of above the horizon")
    # One cluster per side, and neither anywhere near the middle where the arcs fan out.
    left_px = (xs < w / 3).sum()
    right_px = (xs > 2 * w / 3).sum()
    middle = ((xs >= w / 3) & (xs <= 2 * w / 3)).sum()
    _fail(fails, left_px > 200 and right_px > 200,
          f"the glyphs are not on both sides: {left_px} px left, {right_px} px right")
    _fail(fails, middle == 0,
          f"{middle} px of glyph land in the middle third, where the arcs are")

    # Legibility at the resolution the MODEL sees, which is not the resolution this was
    # drawn at: the processor downsamples to QVLA_MAX_PIXELS. A number that survives the
    # render and dies in the resize is a number nobody can answer.
    max_px = int(os.environ.get("QVLA_MAX_PIXELS", 200704))
    scale = min(1.0, math.sqrt(max_px / float(w * h)))
    small = b.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    small.save(out_dir / "menu_with_pivots_asmodel.jpg", quality=92)
    glyph_px_small = math.hypot(xs.max() - xs.min(), 0) * scale
    _fail(fails, glyph_px_small > 24,
          f"at the model's resolution the glyph pair spans {glyph_px_small:.0f} px -- "
          f"too small to read a digit out of")
    return [plain, withp, out_dir / "menu_with_pivots_asmodel.jpg"]


def check_server_wiring(fails: list[str], n_arcs: int) -> None:
    """The flag reaches the server, and the server derives the same numbers.

    SKIPPED where torch is missing. Everything above is pure geometry and text and runs
    anywhere; this one needs the policy server's own interpreter, and refusing to check
    the parts that do not would make this tool unrunnable from a plain shell -- which is
    where it is most useful. `run_pivot_study.sh` invokes it under the server's venv, so
    the campaign path always gets the full check.
    """
    try:
        server = importlib.import_module("server_qwen")
    except ModuleNotFoundError as exc:
        print(f"   (skipping the server wiring check -- {exc.name} is not importable "
              f"here; run under the policy server's venv for it)")
        return
    _fail(fails, server.MENU_PIVOTS,
          "MENU_PIVOTS is False in the server -- QVLA_MENU_PIVOTS did not reach it")
    _fail(fails, len(server._MENU_ARCS) == n_arcs,
          f"the server has {len(server._MENU_ARCS)} arcs, arc_menu builds {n_arcs}")
    _fail(fails, server._STOP_LABEL == stop_label(n_arcs, True),
          f"the server's STOP label is {server._STOP_LABEL}, not "
          f"{stop_label(n_arcs, True)}")
    # The two prompt variants must be distinct objects AND distinct text: a cache keyed
    # on the wrong tuple would hand the pivot prompt to a no-pivot menu, silently.
    with_p = server._menu_system(True, False, (8, 9))
    without = server._menu_system(True, False, None)
    _fail(fails, with_p != without,
          "the cached prompt builder returns the same text with and without pivots")
    _fail(fails, "pivot_rad" in server.PredictResponse.model_fields,
          "PredictResponse has no pivot_rad field -- the turn cannot reach the runner")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, default=None,
                    help="camera frame to draw on; defaults to the newest scratch frame")
    ap.add_argument("--out", type=Path, default=Path("/tmp/pivot-check"))
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    n_arcs = len(make_arcs())
    stop = stop_label(n_arcs, True)

    fails: list[str] = []
    labels, pivots = check_allocation(fails, n_arcs)
    check_signs(fails)
    check_mapping(fails, stop, pivots)
    prompt = check_prompt(fails, stop, pivots)
    check_parsing(fails, pivots)
    check_server_wiring(fails, n_arcs)

    print(f"-- arcs {n_arcs}  pivots {pivots}  STOP {stop}")
    print(f"-- one shuffle: arcs {labels} + pivots {list(pivots)}")

    frame = args.frame
    if frame is None:
        cand = sorted(FRAMES.glob("nav_*.jpg"), key=lambda p: p.stat().st_mtime)
        frame = cand[-1] if cand else None
    if frame is None or not frame.is_file():
        print(f"   (no camera frame at {FRAMES} -- skipping the render check)")
    else:
        print(f"-- rendering onto {frame}")
        for out in check_render(fails, frame, args.out, labels, pivots):
            print(f"   wrote {out}")

    if args.show_prompt:
        print("\n" + "=" * 72)
        print(prompt)
        print("=" * 72)

    if fails:
        print()
        for f in fails:
            print(f"!! {f}")
        return 1
    print("\n-- PIVOT MENU OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
