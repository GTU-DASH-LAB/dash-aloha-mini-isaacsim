"""Can the model tell, from the frame in front of it, that a stage of the instruction is done?

    nav/tools/probe_stage_state.py

THE DECISION THIS GATES. `probe_goal_direction.py` established that the policy cannot be
repaired by asking a better question: the goal is invisible on 65% of all decisions and on
92% of the ones needing a turn past 90 degrees, so "which way must I go" has no answer in
the pixels and the model rightly says `straight ahead`. What is missing is STATE -- which
clause of a multi-part instruction is in progress.

The proposed architecture turns the sentence into a program counter: split the instruction
into ordered clauses, ask the describe call one binary question per decision (is the
current clause's landmark behind us yet), latch the stage when it goes true, and show the
select call only the clause it is on. Every part of that is cheap EXCEPT the binary
question, which the model has to be able to answer. This probe asks whether it can, before
any of the machinery is built.

WHY THE LABELS ARE FREE, and this is the whole trick. Two moments in a run have a stage
state that needs no annotation:

  * THE FIRST DECISION. The robot is at its start pose and has not moved. No clause is
    done. Certain NO.
  * THE LAST DECISION OF A RUN THAT SUCCEEDED. Success is recomputed from the trace against
    the goal threshold, so a run scored `Y` really did arrive, and every precondition on
    the way to arriving was therefore met. Certain YES.

No doorway coordinates, no hand-labelled frames, no judgement calls. The cost is that only
the two ends of a run are usable, which is fine -- a channel that cannot separate "have not
started" from "have arrived" will not separate anything finer, and that is the question
being asked.

WHAT THE CONTROLS RULE OUT. A model that answers NO to everything scores 100% on the first
frames and 0% on the last; one that answers YES to everything does the reverse. Both are
reported, so a headline accuracy cannot hide either. Per-episode rows are printed for the
same reason the hard study prints seeds individually: one episode whose landmark happens to
stay in frame would otherwise carry a mean that means nothing.

The answer is forced with a prefill, for the reason `SELECT_PREFILL` documents -- 392 of
6673 campaign decisions answered an "answer with X and nothing else" instruction with prose
instead, and a probe that scores unparsed prose as a wrong answer is measuring the answer
contract rather than the capability.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))
sys.path.insert(0, str(REPO / "nav/sim"))
sys.path.insert(0, str(REPO / "nav/tools"))
from arc_menu import CAM_FOV_DEG, CAM_HEIGHT_M  # noqa: E402
from probe_goal_direction import _menu_dir, ask  # noqa: E402
from summarize_runs import load_config, score  # noqa: E402

STUDY = REPO / "nav/results/hard_study"

# The first clause of each instruction, as a thing that is either behind the robot or not.
# Written from the episode's own sentence rather than invented: these are the words the
# policy is already given, so a model that cannot answer about them cannot be helped by a
# stage tracker built on the same sentence.
STAGE_Q = {
    "hospital_exit_room":
        "has the robot already driven out of the room and into the hallway",
    "hospital_forward_staircase":
        "has the robot already reached the staircase",
    "warehouse":
        "has the robot already reached the traffic cones at the entrance of Aisle 05",
    "warehouse_aisle6":
        "has the robot already driven into Aisle 06",
    "hospital_down_hallway2":
        "has the robot already reached the red bed to the right of the doors",
    "hospital_past_wheelchairs":
        "has the robot already driven past the wheelchairs",
}

SYSTEM = (f"You are looking through the forward camera of a small wheeled robot. The "
          f"camera is {CAM_HEIGHT_M:.2f} m above the floor, level, with a "
          f"{CAM_FOV_DEG:.0f} degree horizontal field of view. Answer only about what is "
          f"visible in this image.")

USER = ("The robot has been told: \"{instruction}\"\n\n"
        "Looking at this image, {question}? Answer with one word, YES or NO.")

PREFILL = "The answer is "


def _yes(text: str) -> bool | None:
    m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    return m.group(1).lower() == "yes" if m else None


def main() -> int:
    cfg = load_config()
    rows: list[tuple[str, str, bool, Path]] = []   # episode, instruction, truth, image
    for arm in ("ref_s0", "ref_s1", "ref_s2", "refrep_s0"):
        for f in sorted((STUDY / arm).glob("*.json")):
            if f.stem == "arm":
                continue
            r = score(f, cfg)
            if r is None:
                continue
            stamp, rest = f.stem.split("_", 1)
            ep = rest.rsplit("_", 1)[0]
            d = _menu_dir(ep, stamp)
            if d is None or ep not in STAGE_Q:
                continue
            imgs = sorted(d.glob("menu_*.jpg"))
            if len(imgs) < 4:
                continue
            instr = cfg[ep]["instruction"]
            rows.append((ep, instr, False, imgs[0]))          # start pose: certain NO
            if r["success"]:
                rows.append((ep, instr, True, imgs[-1]))      # arrived: certain YES

    if not rows:
        print("no usable runs on disk")
        return 1

    print(__doc__.split("\n\n")[0])
    print(f"\n{len(rows)} frames with a label that needs no annotation: "
          f"{sum(not t for *_, t, _ in [(0, 0, r[2], r[3]) for r in rows])} at the start "
          f"pose (NO), {sum(r[2] for r in rows)} at the end of a successful run (YES)\n")
    print(f"  {'episode':<28}{'truth':>7}{'said':>7}   {'':<4}")
    print("  " + "-" * 50)

    per: dict[str, list[int]] = {}
    said_yes = right = unparsed = 0
    for ep, instr, truth, img in rows:
        got = _yes(ask([str(img)], SYSTEM,
                       USER.format(instruction=instr, question=STAGE_Q[ep]), 12))
        ok = got is not None and got == truth
        right += ok
        said_yes += got is True
        unparsed += got is None
        per.setdefault(ep, [0, 0])
        per[ep][0] += ok
        per[ep][1] += 1
        print(f"  {ep:<28}{'YES' if truth else 'NO':>7}"
              f"{('YES' if got else 'NO') if got is not None else '??':>7}   "
              f"{'ok' if ok else 'X'}")

    n = len(rows)
    n_yes = sum(r[2] for r in rows)
    print(f"\n\nPER EPISODE\n")
    for ep, (ok, tot) in sorted(per.items()):
        print(f"  {ep:<28}{ok}/{tot}")

    print(f"\n\nVERDICT\n")
    print(f"  correct                        {right}/{n}  ({right / n:.0%})")
    print(f"  answered YES                   {said_yes}/{n}   "
          f"(all-NO would be {n - n_yes}/{n} correct, all-YES {n_yes}/{n})")
    if unparsed:
        print(f"  unparsed                       {unparsed}/{n}")
    print("\n  The number to beat is the better of the two constant answers above, not 50%.")
    print("  Beating it means the stage channel exists and the program counter can be built.")
    print("  Failing to means these episodes are out of reach for a policy that decides")
    print("  from one frame, and the next move is a different architecture, not a prompt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
