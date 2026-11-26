#!/usr/bin/env python3
"""Replay a run's select calls with and without the odometry sentence (QVLA_ODOMETRY).

    nav/tools/probe_turn_odometry.py <run menus dir> <EpisodeResult json> [--port 8766]

Each decision's select call is rebuilt from what the run recorded -- the menu image it was
shown, the description, the history line, the instruction, the label shuffle -- and sent to
the server's /raw twice: exactly as it ran, and with `arc_menu.odometry_note` added after
the history line, computed from the run's own trace at that decision. The first column is a
reconstruction check (it must reproduce the recorded choice); the second is the effect.

Open loop: in the real loop a different choice changes every later frame, so this says
whether the sentence flips the decision IN the state the old prompt drove the robot into,
not what the trajectory would have been. The spin guard's arc removal cannot be replayed
here -- the recorded menu image already has every arc drawn -- and needs no replay: it is a
filter, the dropped arcs simply cannot be chosen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))
sys.path.insert(0, str(REPO / "nav" / "sim"))

from arc_menu import (  # noqa: E402
    SELECT_PREFILL, SELECT_PREFILL_SPEED, SPEED_CHOICE_FROM_M, direction_word,
    SPIN_MIN_KAPPA, odometry_note, parse_choice_speed, select_system, stop_label,
)
from turn_odometer import TurnOdometer  # noqa: E402

N_ARCS = 7
STOP = stop_label(N_ARCS, False)
KAPPAS = [-0.6, -0.35, -0.15, 0.0, 0.15, 0.35, 0.6]   # coarse menu, label order = index


def raw(port, image, system, user, prefill, max_new=8):
    body = json.dumps({"image_paths": [image], "system": system, "user": user,
                       "prefill": prefill, "max_new_tokens": max_new}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/raw", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["text"]


def word(choice, labels):
    if choice is None:
        return "?"
    if choice == STOP:
        return "STOP"
    if choice in labels:
        return direction_word(KAPPAS[labels.index(choice)])
    return f"off-menu {choice}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("result")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--notes", action="store_true", help="print the sentence added")
    a = ap.parse_args()
    run = Path(a.run_dir)
    recs = [json.loads(l) for l in (run / "decisions.jsonl").read_text().splitlines() if l.strip()]
    while len(recs) > 1 and recs[0]["step"] > recs[1]["step"]:
        recs.pop(0)                      # a previous tour step's generation, see make_slide_video
    tr = json.loads(Path(a.result).read_text())["trace"]
    flips = same = guarded = 0
    seen = None            # the server's `target_seen` memory, replayed from the record
    for r in recs[::a.every]:
        k = min(len(tr) - 1, int(r["step"]) // 30)     # trace is sampled every 30 steps
        odo = TurnOdometer()
        odo.reset(0.0, *tr[0][:2], tr[0][2])
        for i in range(1, k + 1):
            odo.update(i * 0.5, *tr[i][:2], tr[i][2])
        pl = odo.payload()
        note, spin = odometry_note(pl, r.get("target_m"), seen)
        if r.get("target_m") is not None:
            seen = {"turned_deg": pl["turned_deg"], "since_s": pl["since_s"],
                    "target_m": r["target_m"]}
        ask_speed = r.get("target_m") is not None and r["target_m"] <= SPEED_CHOICE_FROM_M
        system = select_system(STOP, stop_allowed=not r.get("kind"), speed_choice=ask_speed)
        hist = r.get("history") or ""
        tail = (f"Navigation instruction: {r['instruction']}\n\n"
                + ("Which numbered path do you take, and how fast?" if ask_speed
                   else "Which numbered path do you take?"))
        head = f"What the robot can see ahead of it: {r['free_space']}\n\n" + (f"{hist}\n\n" if hist else "")
        prefill = SELECT_PREFILL_SPEED if ask_speed else SELECT_PREFILL
        img = str((run / r["menu"]).resolve())
        base = parse_choice_speed(raw(a.port, img, system, head + tail, prefill))[0]
        new = parse_choice_speed(raw(a.port, img, system, head + (f"{note}\n\n" if note else "") + tail,
                                     prefill))[0]
        rec_w, base_w, new_w = word(r["choice"], r["labels"]), word(base, r["labels"]), word(new, r["labels"])
        flips += base_w != new_w
        same += base_w == rec_w
        # The guard takes these arcs off the real menu; the recorded image still has them.
        off = (spin and new in r["labels"]
               and KAPPAS[r["labels"].index(new)] * spin >= SPIN_MIN_KAPPA)
        guarded += bool(off)
        print(f"t={r['step']/60:5.1f}s turned {odo.payload()['turned_deg']:7.1f}  "
              f"recorded {rec_w:11s} replay {base_w:11s} +odometry {new_w:11s}"
              f"{'  [spin guard' + (': that arc is off the menu]' if off else ']') if spin else ''}"
              f"{'  | ' + note[:160] if a.notes else ''}", flush=True)
    print(f"\n{len(recs[::a.every])} decisions: replay reproduced the recording on {same}; "
          f"the odometry sentence changed the choice on {flips}; {guarded} of its choices "
          f"are arcs the spin guard removes")


if __name__ == "__main__":
    main()
