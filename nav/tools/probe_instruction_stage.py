"""A multi-stage instruction gets its LATER direction word applied at the FIRST decision.

`hospital_exit_room` is the only episode of the five re-run at `medium` that still fails,
and its whole failure is two decisions long. Instruction:

    "Exit the room and turn right to enter the hallway. Continue straight ahead and to
     the end of the hallway."

At step 0 and step 30 the describe call says, correctly, *"The way ahead is open, with a
doorway leading into a hallway visible in the distance. The most open, walkable floor is
straight ahead. TARGET: 10 m"* -- and the select call answers a right-hand arc, kappa
-0.35 and then -0.60. Those are the ONLY two decisions in the whole 160-decision episode
where describe and select disagree; from step 120 to the end they agree every time. By
then the robot is facing the room's cabinets, `TARGET: not visible` for the remaining 150
decisions, and it spends the episode bouncing along furniture. 257 guard interventions.

The suspected mechanism is that the selector has no notion of WHICH STAGE of the
instruction it is on, and this stack has already measured that it obeys hard: an
instruction moves the arc-menu plan **103 degrees**, against ~9 for TIC-VLA's action
expert. "Turn right" belongs to stage two -- after the doorway -- and gets spent on stage
one. `office_hallway_turn2` has the identical two-stage shape and passes, which fits: its
stage-one word is "straight ahead", so the same bug cannot show.

That story explains every observation, which is exactly why it must not be believed yet.
The same class of story was believed once on this branch -- the `arc` zero-collapse "stop
attractor" -- and the next run falsified it. So: hold the frame, hold the description the
model itself produced on that frame, and vary only the question.

  full      what actually ran. Must reproduce the right-hand arc, or the probe is not
            reaching the thing being explained.
  stage1    same prompt, direction word deleted from the instruction. If this goes
            straight, the word is what pulls it -- nothing else changed.
  staged    the candidate repair: tell the selector the instruction is a SEQUENCE, that
            it should carry out the earliest part not yet done, and that when a direction
            word disagrees with the description of the floor, the floor wins.
  described a control in the opposite direction: the description is rewritten to say the
            open floor is to the right. `staged` must still turn right here, or it has
            not learned to read the description -- it has learned to go straight.

And the control that decides whether the repair is safe to ship: run every arm on the
LATER frames of the same episode, where the description says right and right is correct.
A repair that turns those straight would trade one failure for a worse one, and would do
it while looking like a fix on the frame it was written for.

Runs against the live server's `/raw`, which declines while a plan generation is in
flight, so it cannot race a ladder for the GPU. Loading a second copy is not an option --
the FP8 weights hold 29 of GPU1's 32 GiB.

    nav/tools/probe_instruction_stage.py --run /tmp/qvla-menus/<dir> [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from arc_menu import DEFAULT_CURVATURES, direction_word, select_system  # noqa: E402

STOP_LABEL = len(DEFAULT_CURVATURES) + 1
BASE = "http://127.0.0.1:8766"

# Repair candidates. The first run of this probe tested only `LONG`, and it fixed the
# failing frame and then broke the ANSWER FORMAT on a third of the control frames -- the
# model replied "Based on the" instead of a digit, which in the live system is a decision
# with no plan behind it. So format compliance is now a measured column, not an
# assumption, and the wording is varied against it.
#
# Three placements, because where a rule sits changes whether it reads as a rule or as an
# invitation to discuss. LONG and SHORT go in the system prompt; USER goes in the message
# body next to the question, which is where `SELECT_AFTER_BALK` already lives and which
# leaves the system prompt's closing "answer with the number and nothing else" as the last
# thing read before answering.
RULES = {
    "long": ("system",
             "The instruction may describe several moves in order. Carry out the earliest "
             "part that has not been done yet, not the whole sentence at once -- a turn "
             "that comes after a doorway is for after the doorway. If a direction word in "
             "the instruction disagrees with where the description says the open floor "
             "actually is, follow the description: you cannot turn into a wall to obey a "
             "word."),
    "short": ("system",
              "The instruction may list several moves in order; do only the earliest one "
              "not yet done. Where it names a direction the description does not call "
              "open, follow the description."),
    "user": ("user",
             "The instruction may list several moves in order. Do only the earliest one "
             "that has not been done yet, and where it names a direction the description "
             "above does not call open, follow the description."),
}


def post(images: list[str], system: str, user: str, max_new: int = 8) -> str:
    body = {"image_paths": images, "system": system, "user": user,
            "max_new_tokens": max_new}
    req = urllib.request.Request(BASE + "/raw", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["text"]


def kappa_of(reply: str, labels: list[int]) -> float | None:
    """Reply -> curvature, using the label permutation THIS menu image was drawn with.

    The labels are shuffled per call in the live system and baked into the saved jpeg, so
    the mapping cannot be recomputed -- it has to come out of the record. Returns None for
    STOP or an unreadable answer, which are reported separately rather than folded into
    a direction.
    """
    m = re.search(r"\d+", reply)
    if not m:
        return None
    choice = int(m.group())
    if choice not in labels:
        return None
    return DEFAULT_CURVATURES[labels.index(choice)]


def strip_direction(instruction: str) -> str:
    """The same sentence with its first turn phrase removed, and nothing else touched."""
    out = re.sub(r"\s*and turn (?:left|right)\b", "", instruction, count=1,
                 flags=re.IGNORECASE)
    return re.sub(r",?\s*then turn (?:left|right)\b[^.]*", "", out, count=1,
                  flags=re.IGNORECASE)


_ANCHOR = ("Your job is to choose the one path that best carries out the navigation "
           "instruction.")


def _flip(seen: str) -> str:
    """Rewrite the description so the open floor is on the RIGHT, changing nothing else.

    The control that keeps a repair honest: a rule that made the model go straight
    regardless would look identical to a working one on the failing frame. Returns the
    original unchanged if no phrase matched, and the caller reports that rather than
    silently scoring a control that never applied -- the first run of this probe had one
    of those and it read as a mysterious result instead of a broken arm.
    """
    for a, b in (("floor is straight ahead", "floor is to the right"),
                 ("is straight ahead", "is to the right"),
                 ("straight ahead.", "to the right.")):
        if a in seen:
            return seen.replace(a, b, 1)
    return seen


def arms(rec: dict, seen: str) -> dict[str, tuple[str, str, bool]]:
    """(system, user, is_flip_control) per arm. `seen` is the description used."""
    instr = rec["instruction"]
    plain = select_system(STOP_LABEL)

    def user(text: str, instruction: str, extra: str = "") -> str:
        return (f"What the robot can see ahead of it: {text}\n\n"
                + (f"{extra}\n\n" if extra else "")
                + f"Navigation instruction: {instruction}\n\n"
                "Which numbered path do you take?")

    out: dict[str, tuple[str, str, bool]] = {
        "full":   (plain, user(seen, instr), False),
        "stage1": (plain, user(seen, strip_direction(instr)), False),
    }
    flipped = _flip(seen)
    for name, (where, rule) in RULES.items():
        system = plain.replace(_ANCHOR, _ANCHOR + " " + rule) if where == "system" else plain
        if where == "system":
            assert system != plain, "rule not inserted -- select_system wording moved"
        extra = rule if where == "user" else ""
        out[name] = (system, user(seen, instr, extra), False)
        # Same repair, description flipped to say RIGHT. Must follow the description.
        out[name + "+flip"] = (system, user(flipped, instr, extra), flipped != seen)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a /tmp/qvla-menus run directory")
    ap.add_argument("--steps", default="0,30",
                    help="decision steps to treat as the failing frames")
    ap.add_argument("--control-steps", default="300,360,510",
                    help="later steps where the description says right and right is right")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    recs = {r["step"]: r for r in
            (json.loads(l) for l in
             (run / "decisions.jsonl").read_text().splitlines() if l.strip())}

    groups = [("FAILING FRAMES (description says straight, episode turned right)",
               [int(s) for s in args.steps.split(",") if s]),
              ("CONTROL FRAMES (description says right -- right is correct here)",
               [int(s) for s in args.control_steps.split(",") if s])]

    tally: dict[str, dict[str, list[float | None]]] = {}
    for title, steps in groups:
        print(f"\n=== {title} ===")
        for step in steps:
            rec = recs.get(step)
            if rec is None:
                print(f"  step {step}: not in this run"); continue
            menu = str(run / rec["menu"])
            seen = str(rec.get("free_space", "")).strip()
            print(f"\n  step {step}  (episode chose k={rec.get('kappa')})")
            print(f"    it said: {seen[:150]}")
            for name, (system, user, flipped_ok) in arms(rec, seen).items():
                if name.endswith("+flip") and not flipped_ok:
                    print(f"    {name:<12} SKIPPED -- no phrase to flip in this description")
                    continue
                ks, replies, bad = [], [], 0
                for _ in range(args.repeats):
                    reply = post([menu], system, user)
                    # A reply that is not a bare number is a decision with NO PLAN in the
                    # live loop, so it is counted separately from a legitimate STOP.
                    if not re.fullmatch(r"\s*\d+\s*\.?\s*", reply):
                        bad += 1
                    ks.append(kappa_of(reply, rec["labels"]))
                    replies.append(reply.strip()[:14])
                words = [("none" if k is None else direction_word(k)) for k in ks]
                tally.setdefault(title, {}).setdefault(name, []).extend(
                    [(k, r) for k, r in zip(ks, replies)])
                flag = f"  [{bad}/{args.repeats} NOT A NUMBER]" if bad else ""
                print(f"    {name:<12} {', '.join(words):<38} raw={replies}{flag}")

    print("\n=== summary (negative k = right, positive = left) ===")
    for title, per_arm in tally.items():
        print(f"  {title}")
        for name, pairs in per_arm.items():
            good = [k for k, _ in pairs if k is not None]
            mean = sum(good) / len(good) if good else float("nan")
            bad = sum(1 for _, r in pairs if not re.fullmatch(r"\s*\d+\s*\.?\s*", r))
            print(f"    {name:<12} mean k {mean:+.3f}   "
                  f"({sum(k < -0.05 for k in good)} right, "
                  f"{sum(abs(k) <= 0.05 for k in good)} straight, "
                  f"{sum(k > 0.05 for k in good)} left)"
                  f"   unparseable {bad}/{len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
