"""The model sees the wall and picks the arc into it. Three ways to close that gap.

Established by the two probes before this one, on the same twelve frames:

    it can separate a blocked way from an open one   100%  (70/70 forced choice)
    it names the obstacle and its distance            6/6  ("a large red double door")
    it avoids the straight arc when told "drive safely"  11%
    its arc choice changes when the image is mirrored     0/18

So the percept exists and never reaches the answer. The suspect is the shape of the
question: one image, seven arcs, `max_new_tokens=8`, and a demand for a bare digit gives
the model nowhere to put the reasoning it has already demonstrated it can do. Under that
pressure the centre arc is the cheapest token, and the centre arc is what comes out.

Three repairs, each keeping the frozen model and the drawn menu, ordered by what they cost
in the control loop:

  DIGIT    the current query, unchanged. The baseline, re-measured here rather than quoted
           from the previous run, so all three are read off the same frames and the same
           label permutations.
  THINK    one call, but room to speak: name what is in the way, then give the number. Adds
           decode tokens, no extra round trip.
  CHAIN    two calls. Ask what is in the way on the bare frame -- the exact question that
           scored 6/6 -- then hand that answer to the arc query as context. Doubles the
           round trips, and is the only variant where the percept is produced under a
           question that is already known to work.

CHAIN is the one to beat on accuracy and the one to be suspicious of on latency: 0.31 s was
the entire argument for this architecture over TIC-VLA's expert, so a repair that costs
0.9 s has to be worth the third of a second it gives back.

Every control from `probe_arc_obstacles.py` is kept, because a repair is easy to fake:
open-corridor frames catch a variant that just learned to swerve, and the mirror pairs
catch one that learned a side. Latency is reported per variant, not averaged away.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arc_menu import make_arcs, render_menu  # noqa: E402
from probe_arc_obstacles import BLOCKED, OPEN, flip  # noqa: E402
from probe_arc_selection import SYSTEM  # noqa: E402
from probe_free_space import DESCRIBE, SYSTEM as PLAIN_SYSTEM  # noqa: E402

STRAIGHT_K = 0.15
USER = "Navigation instruction: {instr}\n\nWhich numbered path do you take?"

# Same prompt as DIGIT up to the final sentence. Replacing only the last line keeps the
# grounding identical, so a difference in the result is about the room to think and not
# about some other wording that drifted in alongside it.
THINK_SYSTEM = SYSTEM.replace(
    "Answer with the number of the chosen path and nothing else.",
    "First name in one short sentence anything that is in the robot's way -- a wall, a "
    "door, a doorframe, an object -- and which paths run into it. Then, on a new line, "
    "write the number of the chosen path and nothing else.")


def post(base: str, images: list[str], system: str, user: str,
         max_new: int) -> tuple[str, float]:
    body = {"image_paths": images, "system": system, "user": user,
            "max_new_tokens": max_new}
    req = urllib.request.Request(base + "/raw", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["text"], out["latency_s"]


def _last_label(text: str, k: int) -> int | None:
    """Last standalone 1..k in the reply.

    THINK's reasoning carries its own numbers ("about 1 metre away"), so the first integer
    is the wrong one to take and the last is the right one: the prompt puts the choice at
    the end. Bounded to the menu so a stray distance past the end cannot be read as a pick.
    """
    hits = re.findall(rf"\b([1-{k}])\b", text)
    return int(hits[-1]) if hits else None


def v_digit(base, menu, raw, instr, k):
    text, lat = post(base, [menu], SYSTEM, USER.format(instr=instr), 8)
    m = re.search(r"\d+", text)
    return (int(m.group()) if m else None), lat, text.strip()[:60]


def v_think(base, menu, raw, instr, k):
    text, lat = post(base, [menu], THINK_SYSTEM, USER.format(instr=instr), 110)
    return _last_label(text, k), lat, text.strip().replace("\n", " ")[:60]


def v_chain(base, menu, raw, instr, k):
    seen, lat1 = post(base, [raw], PLAIN_SYSTEM, DESCRIBE, 60)
    seen = seen.strip().replace("\n", " ")
    user = (f"What the robot can see ahead of it: {seen}\n\n"
            + USER.format(instr=instr))
    text, lat2 = post(base, [menu], SYSTEM, user, 8)
    m = re.search(r"\d+", text)
    return (int(m.group()) if m else None), lat1 + lat2, seen[:60]


# CHAIN's own numbers say what is missing from it. It lifted wall-avoidance from 22% to
# 72% while leaving the direction of the swerve near chance (60% vs 43%), and the reason is
# visible in the text it passes along: "there is a wall directly in the robot's way,
# approximately 0.5 to 1 meter away" reports that the way is blocked and never says which
# way is open. So the second call learns to turn and has to guess where.
#
# The fix is to ask the first call for the thing the second call is missing, and it is
# worth trying precisely because of what is already measured: a free-space question scores
# 100%, and a direction word scores 100%. If the description names the open side, the two
# capabilities that each work alone are finally wired to each other -- and if it still
# fails, the failure is in the drawn menu rather than in either half, which sends this to
# geometry with no ambiguity left.
DESCRIBE_SIDED = (
    "Answer in two short sentences. First: is there a wall, a door, a doorframe or a large "
    "object straight ahead of the robot within about 3 metres? Name it, or say the way "
    "ahead is open. Second: say which direction has the most open, walkable floor the "
    "robot could drive along -- far left, left, straight ahead, right, or far right.")


def v_sided(base, menu, raw, instr, k):
    seen, lat1 = post(base, [raw], PLAIN_SYSTEM, DESCRIBE_SIDED, 80)
    seen = seen.strip().replace("\n", " ")
    user = (f"What the robot can see ahead of it: {seen}\n\n" + USER.format(instr=instr))
    text, lat2 = post(base, [menu], SYSTEM, user, 8)
    m = re.search(r"\d+", text)
    return (int(m.group()) if m else None), lat1 + lat2, seen[:60]


VARIANTS = {"DIGIT": v_digit, "THINK": v_think, "CHAIN": v_chain, "SIDED": v_sided}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--instruction", default="Drive safely.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out-dir", default="/tmp/arc-repair")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arcs = make_arcs()
    k = len(arcs)

    # Menus are rendered once and shared by all three variants, so they differ only in the
    # question asked. Same pixels, same shuffles, same order.
    rng = random.Random(args.seed)
    scenes = []
    for fid, side in [(f, s) for f, s in BLOCKED] + [(f, 0) for f in OPEN]:
        kind = "blocked" if any(fid == b for b, _ in BLOCKED) else "open"
        for mir in (False, True):
            raw = f"{args.frame_dir}/nav_{fid:06d}.jpg"
            if mir:
                raw = flip(raw, str(out / f"flip_{fid:06d}.jpg"))
            for p in range(args.perms):
                labels = list(range(1, k + 1))
                rng.shuffle(labels)
                menu = str(out / f"m_{fid:06d}{'_m' if mir else ''}_p{p}.jpg")
                render_menu(raw, menu, arcs, labels)
                scenes.append({
                    "frame": fid, "kind": kind, "mirrored": mir, "perm": p,
                    "raw": raw, "menu": menu,
                    "by_label": {lab: arcs[i].kappa for i, lab in enumerate(labels)},
                    "open_side": -side if mir else side,
                })

    print(f"{len(scenes)} scenes x {len(VARIANTS)} variants, "
          f"instruction {args.instruction!r}\n")
    results: dict[str, list[dict]] = {}
    for name, fn in VARIANTS.items():
        rows = []
        for sc in scenes:
            choice, lat, note = fn(base, sc["menu"], sc["raw"], args.instruction, k)
            rows.append({**{q: sc[q] for q in
                            ("frame", "kind", "mirrored", "perm", "open_side")},
                         "choice": choice, "note": note, "latency_s": lat,
                         "kappa": sc["by_label"].get(choice)})
        results[name] = rows
        blk = [r for r in rows if r["kind"] == "blocked" and r["kappa"] is not None]
        opn = [r for r in rows if r["kind"] == "open" and r["kappa"] is not None]
        sided = [r for r in blk if r["open_side"] != 0]
        pairs = [(a, b) for fid, _s in BLOCKED for p in range(args.perms)
                 for a in [next((r["kappa"] for r in blk if r["frame"] == fid
                                 and not r["mirrored"] and r["perm"] == p), None)]
                 for b in [next((r["kappa"] for r in blk if r["frame"] == fid
                                 and r["mirrored"] and r["perm"] == p), None)]
                 if a is not None and b is not None]
        results[name + "_stats"] = {  # type: ignore[assignment]
            "avoid": sum(abs(r["kappa"]) > STRAIGHT_K for r in blk) / max(len(blk), 1),
            "keep": sum(abs(r["kappa"]) <= STRAIGHT_K for r in opn) / max(len(opn), 1),
            "side": sum(r["kappa"] * r["open_side"] > 0
                        for r in sided) / max(len(sided), 1),
            "mirror": sum(a * b < 0 for a, b in pairs) / max(len(pairs), 1),
            "lat": statistics.median(r["latency_s"] for r in rows),
            "unparsed": sum(r["kappa"] is None for r in rows),
        }
        s = results[name + "_stats"]
        print(f"  {name:6} avoid {s['avoid'] * 100:3.0f}%  keep {s['keep'] * 100:3.0f}%  "
              f"open-side {s['side'] * 100:3.0f}%  mirror {s['mirror'] * 100:3.0f}%  "
              f"{s['lat']:.2f}s  unparsed {s['unparsed']}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'variant':8}{'avoid wall':>12}{'keep straight':>15}{'open side':>11}"
          f"{'mirror':>9}{'latency':>10}")
    print(f"{'':8}{'(blocked)':>12}{'(open)':>15}{'(ch. 43%)':>11}{'(ch. 0%)':>9}{'':>10}")
    for name in VARIANTS:
        s = results[name + "_stats"]
        print(f"{name:8}{s['avoid'] * 100:11.0f}%{s['keep'] * 100:14.0f}%"
              f"{s['side'] * 100:10.0f}%{s['mirror'] * 100:8.0f}%{s['lat']:9.2f}s")

    best = max(VARIANTS, key=lambda n: (results[n + "_stats"]["avoid"]
                                        + results[n + "_stats"]["keep"]
                                        + results[n + "_stats"]["mirror"]))
    bs = results[best + "_stats"]
    if bs["avoid"] > 0.7 and bs["keep"] > 0.7 and bs["mirror"] > 0.7:
        verdict = (f"{best} REPAIRS IT at {bs['lat']:.2f}s per decision -- the percept "
                   f"reaches the choice once the question makes room for it. Wire this "
                   f"variant to the controller.")
    elif bs["avoid"] > 0.7 and bs["keep"] < 0.5:
        verdict = (f"{best} only learned to swerve: it avoids {bs['avoid'] * 100:.0f}% of "
                   f"blocked straights but holds straight on only {bs['keep'] * 100:.0f}% "
                   f"of open corridors. That is a bias, not obstacle avoidance. This is "
                   f"exactly what the open controls exist to catch.")
    else:
        verdict = (f"NONE OF THE THREE repairs it -- best is {best} at "
                   f"{bs['avoid'] * 100:.0f}% avoid / {bs['mirror'] * 100:.0f}% mirror. "
                   f"The model can answer about free space when asked directly and cannot "
                   f"apply it to a drawn menu however the question is framed. Filter the "
                   f"menu geometrically before it is drawn, and leave the model the job it "
                   f"does at 100%: choosing a direction from the instruction.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
