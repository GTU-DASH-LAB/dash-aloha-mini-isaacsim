"""WMNav, OpenFMNav and L3MVN run against our frames, using their own code and prompts.

`probe_value_map.py` handled VLFM, whose contribution is a model (BLIP-2 ITM) and so ports
as a model. The other three contribute PROMPTS and a scoring loop around a frozen LLM, and
a prompt ports perfectly: it is text, it costs nothing to move, and our policy server
already serves the model it would be talking to. So there is no reason to argue about
whether they would work here. Run them.

Every prompt below is copied out of the repository that published it, not paraphrased:

    WMNav        github.com/B0B8K1ng/WMNavigation  src/WMNav_agent.py:_construct_prompt
    OpenFMNav    github.com/yxKryptonite/OpenFMNav vl_prompt/prompt/{discover,scoring}.py
    L3MVN        github.com/ybgdgh/L3MVN           main_llm_zeroshot.py:construct_dist

WHAT EACH ONE ACTUALLY CONTRIBUTES, READ OFF ITS OWN SOURCE.

  WMNav/action      Numbered red arrows are drawn on the observation and the VLM picks one
                    by number. That is our arc menu with a different prompt, which makes it
                    the single most directly comparable thing in any of the four papers.
                    Two differences from ours are load-bearing and both are testable here:
                    it makes the model TALK BEFORE ANSWERING ("First, tell me what you
                    see... Lastly, explain which action achieves that best"), and it keeps
                    an action 0 meaning "turn around". `probe_arc_repair.py` already
                    measured the first of those as the THINK repair; this measures WMNav's
                    own wording of it.
  WMNav/predicting  A 0-10 score per direction off a panorama -- the curiosity value map.
                    Its criterion (1) is explicitly geometric: "If there is no visible way
                    to move to other areas ... assign a score of 0", where a way out means
                    "a turn in the corner, an open door, a hallway". That is a free-space
                    question asked in words, which is exactly the channel §6.2 found
                    missing, so it is worth a lot more than its role in the paper suggests.
  OpenFMNav        Two stages. `discover` asks a VLM to name what is in view; `scoring`
                    hands those descriptions to an LLM and asks which area is likeliest to
                    contain the goal. Both stages are run here, in that order, on the same
                    sector crops, so what is measured is OpenFMNav's pipeline and not our
                    guess at it.
  L3MVN            `construct_dist` builds the string "A room containing <obj>, <obj>, and
                    <label>." and scores it with GPT2-large, once per candidate category.
                    There is no image in that function and no geometry -- it is a textual
                    co-occurrence prior over object classes. It is run anyway, on the object
                    lists OpenFMNav's discover stage produced, because the honest way to
                    report "this cannot answer our question" is to run it and show what it
                    answers instead.

THE SUBSTITUTION, STATED ONCE AND APPLIED EVERYWHERE.

All three are object-goal navigators: their prompts say NAVIGATE TO THE NEAREST {goal}.
Our labelled frames carry no object category -- the question they ask is "which way is
drivable", under the neutral instruction "Drive safely." So WMNav/action is run in two
variants and both are reported:

    verbatim   their prompt untouched, with goal = "open floor". Tests WMNav as published.
    neutral    their prompt with only the goal sentence swapped for our neutral
               instruction, everything else -- the arrow description, the talk-then-answer
               scaffold, the {'action': N} format, the closed-doors note -- left alone.
               Tests WMNav's PROMPTING contribution separately from its task framing.

Reporting one without the other would be a choice about which result to show, so both are
in the table. The same substitution logic applies to the other two and is noted per system.

SCORED WITH `probe_arc_obstacles.py`'s RULES, ON ITS FRAMES, AGAINST ITS 27 B NUMBERS.
Same twelve labelled frames, same label permutations, same mirror control, same columns:
avoided the straight arc on blocked frames, stayed straight on open ones, turned toward the
labelled open side, and -- the one that needs no labels -- did the choice negate when the
image was flipped. The baseline row is our own arc-menu prompt on the same server.

Usage:
    /home/gtu-dsa/envs/qvla/bin/python nav/tools/probe_objectnav_prompts.py --port 8766
"""

from __future__ import annotations

import sys
from pathlib import Path

# See the long note at the top of `probe_value_map.py`: `nav/tools/profile.py` shadows the
# stdlib `profile`, cProfile imports it, and torch imports cProfile. L3MVN's stage loads
# GPT2-large, so this file needs the same eviction.
# Matching only `sys.path[0]` is not enough: an importer that put a RELATIVE "nav/tools"
# on the path leaves a string that is the same directory and not the same text, and the
# shadowing comes back. Compare resolved paths and drop every entry that names this one.
_TOOLS = str(Path(__file__).resolve().parent)


def _same_dir(entry: str) -> bool:
    try:
        return str(Path(entry or ".").resolve()) == _TOOLS
    except OSError:
        return False


_evicted = [p for p in sys.path if _same_dir(p)]
sys.path[:] = [p for p in sys.path if not _same_dir(p)]
import cProfile  # noqa: E402, F401  -- imported for its side effect on sys.modules

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

sys.path.insert(0, _TOOLS)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arc_menu import make_arcs, render_menu  # noqa: E402
from probe_arc_obstacles import BLOCKED, FRAME_DIR, OPEN, flip  # noqa: E402
from probe_arc_selection import SYSTEM as QVLA_SYSTEM  # noqa: E402
from probe_value_map import badge_columns, crops  # noqa: E402

STRAIGHT_K = 0.15
NEUTRAL = "Drive safely."
GOAL = "open floor"        # the noun substituted into the verbatim object-goal prompts

# ======================================================================================
# WMNav -- src/WMNav_agent.py, _construct_prompt(). Copied, not paraphrased. The f-string
# branches are resolved for our case: no subtask (the first run of an episode), and the
# turn-around cooldown satisfied so the "action 0" sentence is present, because our menu
# has no pivot and dropping the sentence would quietly change what is being tested.
# ======================================================================================
# NOT an f-string: the braces are for `.format()`, and the four-brace run is deliberate.
# WMNav builds this piece as a PLAIN string inside an otherwise-f-string concatenation, so
# the doubled braces never collapse and the model is literally shown `{{'action': ...}}`.
# `.format()` halves a doubled brace, so four here reproduces their two exactly. Their
# typo "acheives" is kept for the same reason: this is their prompt, not a tidied one.
WMNAV_ACTION = (
    "TASK: NAVIGATE TO THE NEAREST {goal}, and get as close to it as possible. Use your "
    "prior knowledge about where items are typically located within a home. "
    "There are {n} red arrows superimposed onto your observation, which represent "
    "potential actions. "
    "These are labeled with a number in a white circle, which represent the location you "
    "would move to if you took that action. NOTE: choose action 0 if you want to TURN "
    "AROUND or DONT SEE ANY GOOD ACTIONS. "
    "First, tell me what you see in your sensor observation, and if you have any leads on "
    "finding the {goal}. Second, tell me which general direction you should go in. "
    "Lastly, explain which action acheives that best, and return it as "
    "{{{{'action': <action_key>}}}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO "
    "NOT NEED TO GO UP OR DOWN STAIRS"
)

# Their criteria (1) and (2) also carry three sentences disambiguating chair/sofa/bed for
# the HM3D category set. They are dropped here and nowhere else, because the goal noun is
# not one of those categories and keeping them would be quoting a prompt at a task it
# cannot refer to. Everything that bears on the geometric judgement is verbatim.
WMNAV_PREDICTING = (
    "The agent has been tasked with navigating to a {goal}. The agent has sent you the "
    "panoramic image describing your surrounding environment, each image contains a label "
    "indicating the relative rotation angle({angles}) with red fonts. "
    "Your job is to assign a score to each direction (ranging from 0 to 10), judging "
    "whether this direction is worth exploring. The following criteria should be used: "
    "To help you describe the layout of your surrounding,  please follow my step-by-step "
    "instructions: "
    "(1) If there is no visible way to move to other areas and it is clear that the target "
    "is not in sight, assign a score of 0. "
    "(2) If the {goal} is found, assign a score of 10.  "
    "(3) If there is a way to move to another area, assign a score based on your estimate "
    "of the likelihood of finding a {goal}, using your common sense. Moving to another "
    "area means there is a turn in the corner, an open door, a hallway, etc. Note you "
    "CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not "
    "considered. "
    "For each direction, provide an explanation for your assigned score. Format your "
    "answer in the json {{{pairs}}}."
)

# ======================================================================================
# OpenFMNav -- vl_prompt/prompt/discover.py and scoring.py. The scoring prompt ships as a
# five-message few-shot chat; our /raw route takes one system and one user string, so the
# two worked examples are inlined into the system prompt in the order the repo sends them.
# That preserves the conditioning; it does not preserve the role boundaries, and that is
# the one deviation in this file's use of OpenFMNav.
# ======================================================================================
# vl_prompt/prompt/discover.py, verbatim. Read rule (1) before reading the result: the
# perception stage is INSTRUCTED to discard "things that are part of the house, like
# ceiling, wall, floor, window", and rule (4) to discard doors as too common. Those four
# nouns are the entire vocabulary our question is asked in. This is not a limitation this
# probe imposes on OpenFMNav -- it is OpenFMNav's own design, and it is correct for
# OpenFMNav, whose question is which ROOM a sofa is likely to be in. It is quoted rather
# than fixed, because fixing it would mean reporting on a prompt nobody published.
OPENFMNAV_DISCOVER_SYSTEM = """You are an intelligent assistant called DiscoverVLM that \
can understand natural language and scene images. Given a list of objects and an image, \
your goal is to discover new objects in the image that are not in the list.

You should consider the following rules when discovering new objects:

(1) You should first consider, what's in the image? Note that you should only include \
objects in the house, and avoid things that are part of the house, like ceiling, wall, \
floor, window etc and avoid room names, like bedroom, kitchen, etc.

(2) Considering the given object list, you should only output things that are not in the \
list or are not similar to things in the list because your duty is to discover new things. \
For example, if the given object list contains "couch" or "tv", you should not output \
"sofa" or "television" because they are similar.

(3) Confirm that the objects you output are in the image. For example, if the image is a \
bedroom, you should not output "bathtub" because it is impossible to find a bathtub in a \
bedroom. And also confirm the objects you output don't violate rule (1).

(4) Avoid objects are common everywhere. For example, objects like light switch and door \
are present in every room, so you should not output them.

Your output should be in the form of "Answer: <list of objects>" such as:

Answer: ["chair", "bed", "bottle"]
"""
# p_manager.get_discover_prompt: question = f"Current object list: {objects}\n{USER}".
# The list is empty because this is the first observation of an episode.
OPENFMNAV_DISCOVER_USER = ("Current object list: []\n"
                           "What objects can you see in the image?")

# vl_prompt/prompt/scoring.py, verbatim -- SYSTEM_PROMPT then both USER/ASSISTANT pairs,
# including the "Thought:" bodies. An earlier draft here kept only the "Answer: [...]"
# lines, which would have been a different prompt: the examples are what make the model
# reason per description before scoring, and dropping them tests a condensation nobody
# published. Their five chat messages are flattened into one system string because /raw
# takes one system and one user; that is this file's only deviation from OpenFMNav, and
# it changes the role boundaries, not the conditioning text.
OPENFMNAV_SCORING_SYSTEM = """You are an intelligent embodied agent called ReasonLLM that \
follows an instruction to navigate in an indoor environment. You are firstly given an \
object goal class for you to find, which is called the goal.

Then, at each step, your task is to take several descriptions of what an area contains to \
output scores for these areas to contain the goal. Each score is a floating point number \
between 0 and 1.

Your output should be a list of scores.

At each step, you should consider:

(1) For each description, according to what the area contains, is it possible that the \
goal is also in this area? To better do reasoning, you can imagine what kind of room the \
area is in, for example, a bedroom, a living room, a bathroom, etc. Based on the common \
sense, you can judge the possibility that the goal to be in this area.

(2) If the goal class is already in the description, the score should be 1 without any \
hesitation.

(3) If one area contains nothing, it is still possible that the goal is in that area. Give \
a score of 0.4 to 0.6. Score the area higher in that case when other areas are not likely \
to contain the goal.

(4) If there are no current frontiers, skip the thought and output 'No frontiers'.

The following two exchanges are worked examples of the task.

USER:
Goal: toilet

- Description 0: The area contains a towel, a bathtub and a sink.

- Description 1: The area contains a bed and a plant.

- Description 2: The area contains a sofa, a TV and a table.

ASSISTANT:
Thought: Let's analyze each description.

- Description 0: this area contains a bathtub and a sink, so it is possibly a bathroom, \
and the goal is toilet, so it is possible that the goal is in this area, I will give a \
score of 0.9

- Description 1: this area contains a bed and a plant, so it is possibly a bedroom, and \
the goal is toilet, so it is not likely that the goal is in this area. I will give it a 0.2

- Description 2: this area contains a sofa, a TV and a table, so it is possibly a living \
room, and the goal is toilet, so it is also not likely that the goal is in this area. I \
will give it a 0.3

Answer: [0.9, 0.2, 0.3]

USER:
Goal: bed

- Description 0: The area contains a towel, a bathtub and a sink.

- Description 1: The area contains a bed and a plant.

- Description 2: The area contains a sofa, a TV and a table.

ASSISTANT:
Thought: Let's analyze each description.

- Description 0: this area contains a bathtub and a sink, so it is possibly a bathroom, \
and the goal is bed, so it is not possible that the goal is in this area. My score is 0.1

- Description 1: this area contains a bed, which is the goal, so the score is 1

- Description 2: this area contains a sofa, a TV and a table, so it is possibly a living \
room. The goal is a bed, so it can be near this area. I will give it a 0.5

Answer: [0.1, 1, 0.5]"""


def ask(base: str, images: list[str], system: str, user: str,
        max_new: int = 8) -> tuple[str, float]:
    body = {"image_paths": images, "system": system, "user": user,
            "max_new_tokens": max_new}
    req = urllib.request.Request(base + "/raw", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out["text"], out["latency_s"]


def last_int(text: str, strict: bool) -> int | None:
    """Read the chosen action. `strict` means: only the {'action': N} form counts.

    The smoke test is the reason this takes a flag. WMNav's prompt makes the model talk
    first and answer last, so a reply cut off by the token cap contains paragraphs of
    correct reasoning and no answer at all -- and a loose "last integer in the string"
    rule happily returns the "2" of a numbered list item. That does not degrade the
    result, it FABRICATES one: an arm scoring near chance off enumeration digits is
    indistinguishable from an arm that answered and was wrong.

    So for the talking arms, no {'action': N} means no answer, counted as unparsed. The
    bare-digit arm (ours, capped at 8 tokens) has no prose to be confused by and keeps
    the loose rule it was measured with.
    """
    m = re.findall(r"['\"]?action['\"]?\s*[:=]\s*['\"]?(\d+)", text, re.I)
    if m:
        return int(m[-1])
    return None if strict else (int(n[-1]) if (n := re.findall(r"\d+", text)) else None)


def parse_answer_list(text: str) -> list[str]:
    """DiscoverVLM answers `Answer: ["chair", "bed"]`. An empty list is a real answer.

    Rule (1) of its prompt forbids reporting walls, floors, ceilings and windows, so on a
    frame whose distinguishing feature IS a wall the correct output under that prompt is
    a short list or none at all. Returning `[]` rather than falling back to raw text keeps
    that visible instead of smuggling the wall back in as prose.
    """
    tail = text.split("Answer:")[-1]
    m = re.search(r"\[(.*?)\]", tail, re.S)
    body = m.group(1) if m else ""
    return [o.strip().strip('"\'') for o in body.split(",") if o.strip().strip('"\' ')]


def phrase(objs: list[str]) -> str:
    """Their description sentence. Rule (3) of the scoring prompt covers the empty case."""
    if not objs:
        return "The area contains nothing."
    if len(objs) == 1:
        return f"The area contains a {objs[0]}."
    return f"The area contains a {', a '.join(objs[:-1])} and a {objs[-1]}."


def argmax_kappa(scores: list[float], kappas: list[float]) -> float:
    """Turn a per-sector score list into one curvature, averaging over TIES.

    A plain `max(range(k), key=scores.__getitem__)` returns the FIRST maximum, and both
    value-map prompts saturate: WMNav's scale is 0-10 and a frame with an open right half
    scores [10, 10, 10, 8, 5, 0, 0], three sectors tied at the ceiling. Taking the first
    of those is not a reading of the value map, it is a reading of the iteration order --
    and since sector 0 is one end of the menu, it is a constant bias toward that end,
    which is exactly the positional prior the mirror control exists to catch. Averaging
    the tied sectors' curvatures gives the centroid of the region the prompt liked, which
    is what "the value map points that way" means.
    """
    top = max(scores)
    tied = [kappas[j] for j, s in enumerate(scores) if s == top]
    return sum(tied) / len(tied)


def scores_from_json(text: str, keys: list[str]) -> list[float] | None:
    """Pull WMNav's per-direction scores out of its JSON answer, keyed by angle label."""
    got: list[float] = []
    for k in keys:
        m = re.search(rf"['\"]{k}['\"]\s*:\s*\{{[^}}]*?['\"]Score['\"]\s*:\s*([0-9.]+)", text)
        if not m:
            m = re.search(rf"['\"]{k}['\"]\s*:\s*([0-9.]+)", text)
        if not m:
            return None
        got.append(float(m.group(1)))
    return got


def score_rows(rows: list[dict]) -> dict[str, float | int]:
    """The four columns from `probe_arc_obstacles.py`, computed identically."""
    blk = [r for r in rows if r["kind"] == "blocked" and r["kappa"] is not None]
    opn = [r for r in rows if r["kind"] == "open" and r["kappa"] is not None]
    sided = [r for r in blk if r["open_side"] != 0]
    pairs = []
    for fid in {r["frame"] for r in blk}:
        for p in {r["perm"] for r in blk}:
            a = [r for r in blk if r["frame"] == fid and r["perm"] == p and not r["mirrored"]]
            b = [r for r in blk if r["frame"] == fid and r["perm"] == p and r["mirrored"]]
            if a and b:
                pairs.append((a[0]["kappa"], b[0]["kappa"]))
    flipped = sum(a * b < 0 for a, b in pairs)
    return {
        "n_blocked": len(blk), "n_open": len(opn),
        "avoid": sum(abs(r["kappa"]) > STRAIGHT_K for r in blk) / max(len(blk), 1),
        "keep": sum(abs(r["kappa"]) <= STRAIGHT_K for r in opn) / max(len(opn), 1),
        "toward": sum(r["kappa"] * r["open_side"] > 0 for r in sided) / max(len(sided), 1),
        "mirror": flipped / max(len(pairs), 1), "n_mirror": len(pairs),
        "turn_around": sum(1 for r in rows if r.get("turn_around")),
        "unparsed": sum(1 for r in rows
                        if r["kappa"] is None and not r.get("turn_around")),
        "latency": statistics.fmean([r["latency_s"] for r in rows]) if rows else 0.0,
    }


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--frame-dir", default=str(FRAME_DIR))
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sector-w", type=int, default=640)
    ap.add_argument("--out-dir", default="/tmp/objectnav-prompts")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--skip", default="", help="Comma-separated stage names to skip.")
    # The mirror control needs no hand labels, so it is the one column that can be made
    # well-powered cheaply -- §8.2's `itm/clear-wall` scored 6/6 on six labelled frames and
    # 70% over 56, which is what choosing a winner off n=6 buys. Same remedy here.
    ap.add_argument("--value-sweep", type=int, default=0,
                    help="Extra unlabelled frames to run the WMNav value prompt on, "
                         "orig vs mirrored, for the negation rate alone.")
    ap.add_argument("--sweep-dir", default="/tmp/alohamini-nav-frames")
    ap.add_argument("--sweep-range", default="601:880",
                    help="The intact capture block; see nav/config/probe_frames.json.")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arcs = make_arcs()
    k = len(arcs)
    kappas = [a.kappa for a in arcs]
    cols = badge_columns(arcs)
    frames = [(f, s, "blocked") for f, s in BLOCKED] + [(f, 0, "open") for f in OPEN]
    results: dict[str, dict] = {}
    all_rows: dict[str, list[dict]] = {}

    def raw(fid: int, mirrored: bool) -> str:
        p = f"{args.frame_dir}/nav_{fid:06d}.jpg"
        return flip(p, str(out / f"flip_{fid:06d}.jpg")) if mirrored else p

    # ---------------------------------------------------------------- menu-picking arms
    # QVLA's own prompt is re-run here rather than quoted from probe_arc_obstacles, so
    # every row in the final table came off one server in one sitting.
    wm_verbatim = WMNAV_ACTION.format(goal=GOAL.upper(), n=k)
    wm_neutral = (
        wm_verbatim
        .replace(f"TASK: NAVIGATE TO THE NEAREST {GOAL.upper()}, and get as close to it "
                 f"as possible. Use your prior knowledge about where items are typically "
                 f"located within a home. ",
                 f"TASK: {NEUTRAL} Stay on open, walkable floor and do not run into a "
                 f"wall, a door frame or an object. ")
        .replace(f"if you have any leads on finding the {GOAL.upper()}",
                 "which parts of the floor ahead are open"))
    assert wm_neutral != wm_verbatim, "the neutral rewrite matched nothing"
    # 700 tokens, not 320: at 320 the smoke test cut the model off mid-sentence on the
    # SECOND of the three things WMNav's prompt asks for, so the answer it exists to
    # produce was never reached. A cap that truncates the prompt's own structure measures
    # the cap. Observed replies land near 400 tokens.
    menu_arms = {
        "qvla (ours)": (QVLA_SYSTEM, f"Navigation instruction: {NEUTRAL}\n\n"
                                     f"Which numbered path do you take?", 8, False),
        "wmnav verbatim": ("", wm_verbatim, 700, True),
        "wmnav neutral": ("", wm_neutral, 700, True),
    }
    for name, (system, user, max_new, strict) in menu_arms.items():
        if name in skip:
            continue
        banner(f"MENU ARM -- {name}")
        rng = random.Random(args.seed)
        rows: list[dict] = []
        t0 = time.time()
        for fid, side, kind in frames:
            for mirrored in (False, True):
                src = raw(fid, mirrored)
                for p in range(args.perms):
                    labels = list(range(1, k + 1))
                    rng.shuffle(labels)
                    menu = str(out / f"m_{fid:06d}{'_m' if mirrored else ''}_p{p}.jpg")
                    render_menu(src, menu, arcs, labels)
                    by_label = {lab: arcs[i] for i, lab in enumerate(labels)}
                    reply, lat = ask(base, [menu], system, user, max_new)
                    choice = last_int(reply, strict)
                    arc = by_label.get(choice)
                    rows.append({
                        "frame": fid, "kind": kind, "mirrored": mirrored, "perm": p,
                        "open_side": (-side if mirrored else side),
                        # WMNav keeps an action 0 meaning "turn around", and our menu has
                        # no such arc. A 0 is therefore neither a parse failure nor a
                        # curvature -- it is a refusal of the whole menu, counted in its
                        # own column so it inflates nothing and is hidden by nothing.
                        "turn_around": choice == 0,
                        "choice": choice, "kappa": None if arc is None else arc.kappa,
                        "reply": reply.strip()[-120:], "latency_s": lat,
                    })
            r = [x for x in rows if x["frame"] == fid and x["kappa"] is not None]
            print(f"  f{fid:<4d} {kind:<7s} mean kappa "
                  f"orig {statistics.fmean([x['kappa'] for x in r if not x['mirrored']] or [0]):+.2f}"
                  f" / mirror {statistics.fmean([x['kappa'] for x in r if x['mirrored']] or [0]):+.2f}",
                  flush=True)
        results[name] = score_rows(rows) | {"wall_s": time.time() - t0}
        all_rows[name] = rows

    # ------------------------------------------------------- WMNav's curiosity value map
    # Seven sector crops handed over as one multi-image call, labelled with the bearing
    # each one is centred on, which is the panorama-with-angle-labels WMNav's prompt
    # describes. The angles are ours (the arcs' bearings), not their 30/90/.../330, because
    # ours is a 90 degree camera and inventing views it cannot see would be fabricating the
    # input rather than porting the prompt.
    if "wmnav value" not in skip:
        banner("VALUE MAP -- wmnav predicting (0-10 per direction)")
        from PIL import Image
        keys = [f"a{i}" for i in range(k)]
        pairs = ", ".join("'%s': {'Score': <0-10>, 'Explanation': <why>}" % x for x in keys)
        rows = []
        t0 = time.time()
        for fid, side, kind in frames:
            for mirrored in (False, True):
                im = Image.open(raw(fid, mirrored)).convert("RGB")
                paths = []
                for i, win in enumerate(crops(im, cols, args.sector_w)):
                    p = out / f"sec_{fid:06d}{'_m' if mirrored else ''}_{i}.jpg"
                    win.save(p, quality=92)
                    paths.append(str(p))
                user = WMNAV_PREDICTING.format(
                    goal=GOAL, angles=", ".join(keys), pairs=pairs)
                user += ("\n\nThe images are given in order and their labels are, "
                         "left to right across the robot's view: " + ", ".join(keys) + ".")
                reply, lat = ask(base, paths, "", user, 900)
                sc = scores_from_json(reply, keys)
                kap = argmax_kappa(sc, kappas) if sc else None
                rows.append({"frame": fid, "kind": kind, "mirrored": mirrored, "perm": 0,
                             "open_side": (-side if mirrored else side),
                             "scores": sc, "kappa": kap,
                             "reply": reply.strip()[:200], "latency_s": lat})
            print(f"  f{fid:<4d} {kind:<7s} "
                  f"{[r['scores'] for r in rows if r['frame'] == fid]}", flush=True)
        results["wmnav value"] = score_rows(rows) | {"wall_s": time.time() - t0}
        all_rows["wmnav value"] = rows

        # The unlabelled mirror sweep. No hand labels are consulted, so the only thing it
        # can report is whether flipping the image negates the chosen curvature -- and a
        # model reading a fixed sector index scores 0 here, not chance, because the badge
        # columns are symmetric about the optical axis.
        if args.value_sweep:
            lo, hi = (int(x) for x in args.sweep_range.split(":"))
            pool = sorted(Path(args.sweep_dir).glob("nav_*.jpg"))
            pool = [p for p in pool if lo <= int(p.stem.split("_")[1]) <= hi]
            step = max(len(pool) // args.value_sweep, 1)
            pool = pool[::step][:args.value_sweep]
            print(f"\n  mirror sweep: {len(pool)} unlabelled frames from {args.sweep_dir} "
                  f"[{lo}:{hi}]", flush=True)
            neg = same = usable = 0
            sweep_rows = []
            for n_done, src in enumerate(pool, 1):
                ks = []
                for mirrored in (False, True):
                    path = (flip(str(src), str(out / f"sw_{src.stem}_m.jpg"))
                            if mirrored else str(src))
                    im = Image.open(path).convert("RGB")
                    paths = []
                    for i, win in enumerate(crops(im, cols, args.sector_w)):
                        p = out / f"sw_{src.stem}{'_m' if mirrored else ''}_{i}.jpg"
                        win.save(p, quality=92)
                        paths.append(str(p))
                    user = WMNAV_PREDICTING.format(goal=GOAL, angles=", ".join(keys),
                                                   pairs=pairs)
                    user += ("\n\nThe images are given in order and their labels are, "
                             "left to right across the robot's view: " + ", ".join(keys)
                             + ".")
                    reply, _ = ask(base, paths, "", user, 900)
                    sc = scores_from_json(reply, keys)
                    ks.append(argmax_kappa(sc, kappas) if sc else None)
                sweep_rows.append({"frame": src.stem, "kappa": ks})
                if None not in ks:
                    usable += 1
                    neg += ks[0] * ks[1] < 0
                    same += ks[0] == ks[1]
                if n_done % 5 == 0 or n_done == len(pool):
                    print(f"    {n_done}/{len(pool)}  negated {neg}/{usable}  "
                          f"identical {same}/{usable}", flush=True)
            results["wmnav value"]["sweep_neg"] = neg
            results["wmnav value"]["sweep_same"] = same
            results["wmnav value"]["sweep_n"] = usable
            all_rows["wmnav value sweep"] = sweep_rows

    # ------------------------------------------------------- OpenFMNav, both its stages
    discovered: dict[tuple[int, bool], list[list[str]]] = {}
    if "openfmnav" not in skip:
        banner("OPENFMNAV -- discover (VLM names objects) then scoring (LLM ranks areas)")
        from PIL import Image
        rows = []
        t0 = time.time()
        for fid, side, kind in frames:
            for mirrored in (False, True):
                im = Image.open(raw(fid, mirrored)).convert("RGB")
                objs: list[list[str]] = []
                raws: list[str] = []
                for i, win in enumerate(crops(im, cols, args.sector_w)):
                    p = out / f"sec_{fid:06d}{'_m' if mirrored else ''}_{i}.jpg"
                    win.save(p, quality=92)
                    # 900 tokens, and it took two failed caps to find. DiscoverVLM's
                    # prompt has four numbered rules and the model walks all of them in
                    # prose before emitting `Answer:`; replies run ~1850 characters. At 96
                    # AND at 400 every sector came back `[]`, which parses as "this prompt
                    # reports nothing here" -- the exact conclusion this stage exists to
                    # test, reached from a truncation. A cap below the prompt's own output
                    # length does not weaken a result, it manufactures one.
                    txt, _ = ask(base, [str(p)], OPENFMNAV_DISCOVER_SYSTEM,
                                 OPENFMNAV_DISCOVER_USER, 900)
                    objs.append(parse_answer_list(txt))
                    raws.append(txt.strip())
                discovered[(fid, mirrored)] = objs
                # Their USER turn's exact shape, including the blank line between entries
                # and the "contains nothing" case rule (3) is written for.
                descs = [phrase(o) for o in objs]
                user = (f"Goal: {GOAL}\n\n" + "\n\n".join(
                    f"- Description {i}: {d}" for i, d in enumerate(descs)) + "\n")
                reply, lat = ask(base, [], OPENFMNAV_SCORING_SYSTEM, user, 700)
                nums = re.findall(r"[0-9]*\.?[0-9]+", reply.split("Answer:")[-1])
                sc = [float(x) for x in nums[:k]] if len(nums) >= k else None
                kap = argmax_kappa(sc, kappas) if sc else None
                # How often DiscoverVLM's own reasoning names a wall, floor, ceiling or
                # door and then discards it under rule (1) or (4). This is the number the
                # section turns on, so it is counted rather than quoted from one example:
                # the stage is not blind to the obstacle, it is instructed to drop it.
                dropped = sum(bool(re.search(r"(wall|floor|ceiling|door)[^.]{0,80}"
                                             r"(exclud|not includ|structural|avoid|common)",
                                             t, re.I)) for t in raws)
                rows.append({"frame": fid, "kind": kind, "mirrored": mirrored, "perm": 0,
                             "open_side": (-side if mirrored else side),
                             "objects": objs, "descriptions": descs, "scores": sc,
                             "kappa": kap, "empty_sectors": sum(not o for o in objs),
                             "sectors_dropping_structure": dropped, "discover_raw": raws,
                             "reply": reply.strip()[:300], "latency_s": lat})
            fr = [r for r in rows if r["frame"] == fid]
            print(f"  f{fid:<4d} {kind:<7s} scores={[r['scores'] for r in fr]} "
                  f"structure-dropped {sum(r['sectors_dropping_structure'] for r in fr)}"
                  f"/{7 * len(fr)} sectors", flush=True)
        results["openfmnav"] = score_rows(rows) | {"wall_s": time.time() - t0}
        all_rows["openfmnav"] = rows

    # ----------------------------------------------------------------- L3MVN, as written
    l3mvn: dict = {}
    if "l3mvn" not in skip and discovered:
        banner("L3MVN -- construct_dist(), GPT2-large, verbatim from main_llm_zeroshot.py")
        l3mvn = run_l3mvn(discovered)

    # ------------------------------------------------------------------------ the table
    banner("ALL ARMS, probe_arc_obstacles.py's columns, same twelve frames")
    print(f"  {'arm':<16}{'avoid':>9}{'keep':>9}{'toward':>9}{'mirror':>10}"
          f"{'turn-arnd':>11}{'unparsed':>10}{'latency':>10}")
    print(f"  {'':<16}{'blocked':>9}{'open':>9}{'open side':>9}{'flip':>10}"
          f"{'(no arc)':>11}{'':>10}{'s/call':>10}")
    print(f"  {'chance':<16}{'57%':>9}{'43%':>9}{'43%':>9}{'0% prior':>10}"
          f"{'':>11}{'':>10}{'':>10}")
    for name, r in results.items():
        print(f"  {name:<16}{r['avoid'] * 100:>8.0f}%{r['keep'] * 100:>8.0f}%"
              f"{r['toward'] * 100:>8.0f}%{r['mirror'] * 100:>9.0f}%"
              f"{r['turn_around']:>11}{r['unparsed']:>10}{r['latency']:>10.2f}")
    print(f"\n  n = {results[next(iter(results))]['n_blocked']} blocked decisions, "
          f"{results[next(iter(results))]['n_open']} open, "
          f"{results[next(iter(results))]['n_mirror']} mirror pairs per menu arm; "
          f"the sector arms are one decision per frame per flip.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"results": results, "rows": all_rows, "l3mvn": l3mvn,
             "goal_substituted": GOAL, "neutral": NEUTRAL, "perms": args.perms,
             "sector_w": args.sector_w, "frames": [f for f, _, _ in frames]}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


def run_l3mvn(discovered: dict[tuple[int, bool], list[list[str]]]) -> dict:
    """L3MVN's frontier score, unmodified, on the objects OpenFMNav's stage found.

    This is `construct_dist` from `main_llm_zeroshot.py` with nothing changed but the
    variable names: build "A room containing <obj>, <obj>, and <label>.", score it with
    GPT2-large, once per candidate label, and take the distribution over labels. The
    scoring function there is `configure_lm("GPT2-large")`, which returns the summed
    log-probability of the sentence, so that is what is computed.

    It is run to show what it returns, not to score it against the open-side labels,
    because it CANNOT be scored against them: the output is a distribution over object
    CATEGORIES for one area, with no term that depends on which direction the area is in.
    Two sectors containing the same objects get the same number whether one is a wall and
    the other is a corridor. That is not a criticism of L3MVN -- it is what a co-occurrence
    prior is -- it is the reason it has nothing to say about our task.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "openai-community/gpt2-large"
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name).eval()

    def scoring_fxn(text: str):
        """main_llm_zeroshot.py:425. Returns -loss, the MEAN NLL, not the summed."""
        tokens_tensor = tok.encode(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output = lm(tokens_tensor, labels=tokens_tensor)
            loss = output[0]
            return -loss

    # constants.py:19, in their order.
    category_to_id = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]

    def construct_dist(objs: list[str]):
        """main_llm_zeroshot.py:471, unchanged including the trailing ", and"."""
        query_str = "A room containing "
        for ob in objs:
            query_str += ob + ", "
        query_str += "and"
        TEMP = []
        for label in category_to_id:
            TEMP_STR = query_str + " "
            TEMP_STR += label + "."
            TEMP.append(scoring_fxn(TEMP_STR))
        return torch.tensor(TEMP)

    print(f"  loaded {name}, {sum(p.numel() for p in lm.parameters()) / 1e6:.0f} M params")

    # Their line 754 is `frontier_score_list[e].append(new_dist[category_to_id.index(cname)])`
    # -- the frontier's score IS the goal category's softmax mass. Run that lookup with our
    # goal before anything else, because it is the shortest true statement about whether
    # L3MVN can express our question, and it is their code raising, not ours.
    try:
        category_to_id.index(GOAL)
        reachable = True
    except ValueError as e:
        reachable = False
        print(f"\n  category_to_id.index({GOAL!r}) -> ValueError: {e}")
        print(f"  Their scoring is INDEXED by goal category. Ours is {GOAL!r}, and the list"
              f"\n  is {category_to_id}. There is no entry to read, so the"
              f"\n  frontier score L3MVN would compute for our task does not exist. Below is"
              f"\n  the distribution it computes anyway, to show what it ranks instead.\n")
    dump: dict = {"labels": category_to_id, "goal_in_list": reachable, "per_sector": {}}
    for (fid, mirrored), sectors in sorted(discovered.items()):
        if mirrored:
            continue
        print(f"  frame {fid}")
        for i, objs in enumerate(sectors):
            # Their line 743: a frontier with no detected objects never reaches the LLM at
            # all -- it falls through to a geometric score. Recorded, not scored.
            if not objs:
                dump["per_sector"][f"{fid}/{i}"] = {"objects": [], "skipped": "len(objs)==0"}
                print(f"    sector {i}  objects=[]  ->  never reaches the LM "
                      f"(their `if len(objs_list)>0`)")
                continue
            dist = F.softmax(construct_dist(objs), dim=0)
            best = category_to_id[int(dist.argmax())]
            probs = [round(float(x), 3) for x in dist]
            dump["per_sector"][f"{fid}/{i}"] = {"objects": objs, "softmax": probs,
                                                "argmax": best}
            print(f"    sector {i}  objects={objs}\n"
                  f"        softmax over {category_to_id} = {probs}  -> {best!r}")
    print("\n  Read what that distribution ranges over: six object CATEGORIES, for ONE"
          "\n  area. There is no bearing in it, no image in the function that produced it,"
          "\n  and no term that could differ between a sector that is a wall and a sector"
          "\n  that is open floor except via the object names. Two sectors that contain the"
          "\n  same objects get the same number whichever one you can drive through."
          "\n\n  That is not a defect. It is what a co-occurrence prior is, and in L3MVN it"
          "\n  is correct, because the thing that knows WHERE anything is is the semantic"
          "\n  map -- `local_map[e][se_cn+4, fmb[0]:fmb[1], fmb[2]:fmb[3]]`, a depth"
          "\n  projection. The LM ranks frontiers the map has already found. Remove the"
          "\n  depth and you have removed the half of L3MVN that answers our question.")
    return dump


if __name__ == "__main__":
    raise SystemExit(main())
