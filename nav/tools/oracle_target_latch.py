"""Gate G-c: put the model's MEASURED bearings through the lidar and see what P2 would latch.

    nav/tools/oracle_target_latch.py                      # strip ruler, default rule
    nav/tools/oracle_target_latch.py --variant grid --standoff 0.8 --agree 1.0

P2 latches a world point for the instruction's target: the describe call names a ruler
column, the column is a bearing from the CAMERA, the lidar ranges along it, and two
sightings that agree become the latch; the robot then drives to a standoff short of the
latched return and stops there on geometry. This replays exactly that, offline, on the
G-b frames (`probe_target_bearing.py`), using:

  * the model's own answers on those frames (visible? which column?) -- not an assumed
    error distribution, the real one, including its false sightings;
  * the replayed lidar sweep at the same pose (`replay_lidar_trace.py`, 500 beams).

NO NEW ASSUMPTION ABOUT THE SENSOR. The replayed runs predate the column model, so the
sweep comes from a sensor at the rotation centre; the camera is 0.25 m ahead of it. The
camera ray is therefore marched outward from the camera and stops at the first sample
that lies beyond the lidar's return on the beam pointing at it -- the parallax-correct
intersection of a camera bearing with a lidar scan, not "the beam with the same angle".

A LATCH needs two sightings of one run whose ranged points agree within --agree metres;
the latch is their mean. The stop point is --standoff metres short of it, back toward the
robot's pose at the second sighting. Success = stop point within the episode's own
success threshold less --margin (0.3 m, so a latch that only grazes the threshold does
not count).

PASS (from the P2 proposal, written before this ran):
  >= 6 of the 18 failed runs would have latched within threshold - 0.3 m, AND
  <= 1 of the 20 successful runs would have latched onto a point > 2 m from the goal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nav.sim.episode import load_episode  # noqa: E402

CAM_FWD_M = 0.25


def range_along_camera_ray(z, k: int, cam_bearing_deg: float) -> tuple[float, float] | None:
    """World (x, y) where a camera bearing meets the replayed scan at pose k, or None."""
    x, y, _, yaw = z["pose_sim"][k]
    B = z["bearings_body"]
    R = z["ranges"][k].astype(float)
    selfh = z["self_hit"][k]
    R = np.where(selfh, np.nan, R)
    b = math.radians(cam_bearing_deg)
    for depth in np.arange(0.10, 12.0, 0.02):
        px, py = CAM_FWD_M + depth * math.cos(b), depth * math.sin(b)   # body frame
        beta = math.atan2(py, px) % (2 * math.pi)
        i = int(round(beta / (2 * math.pi / len(B)))) % len(B)
        r = R[i]
        if np.isfinite(r) and math.hypot(px, py) >= r:
            c, s = math.cos(yaw), math.sin(yaw)
            return x + c * px - s * py, y + s * px + c * py
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=REPO / "nav/results/replay/gb_probe/rows.json")
    ap.add_argument("--replay", type=Path, default=REPO / "nav/results/replay/gb")
    ap.add_argument("--variant", default="strip")
    ap.add_argument("--agree", type=float, default=1.0)
    ap.add_argument("--standoff", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--truth", action="store_true",
                    help="oracle upper bound: use the TRUE bearing and perfect visibility")
    args = ap.parse_args()

    text = args.rows.read_text()
    allrows = ([json.loads(l) for l in text.splitlines() if l.strip()]
               if args.rows.suffix == ".jsonl" else json.loads(text))
    rows = [r for r in allrows if r["variant"] == args.variant]
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["result"]].append(r)

    out = []
    for run, R in sorted(by_run.items()):
        z = np.load(args.replay / run / "replay.npz")
        s = json.loads((args.replay / run / "summary.json").read_text())
        ep = load_episode(s["episode"])
        gx, gy = ep.goal[0], ep.goal[1]
        thr = float(ep.success_threshold_m)
        sightings = []
        for r in sorted(R, key=lambda r: r["trace_index"]):
            if args.truth:
                if r["kind"] != "in":
                    continue
                bearing = r["bearing"]
            else:
                if not r["visible"] or r["pred_bearing"] is None:
                    continue
                bearing = r["pred_bearing"]
            p = range_along_camera_ray(z, r["k"], bearing)
            if p is None:
                continue
            sightings.append((r["trace_index"], p, r["k"], r["kind"]))
        latch = None
        for a in range(len(sightings)):
            for b in range(a + 1, len(sightings)):
                pa, pb = sightings[a][1], sightings[b][1]
                if math.dist(pa, pb) <= args.agree:
                    latch = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, sightings[b][2])
                    break
            if latch:
                break
        row = dict(run=run, episode=s["episode"], success=s["run"]["success"], thr=thr,
                   sightings=len(sightings),
                   ctrl_sightings=sum(1 for q in sightings if q[3] == "ctrl"))
        if latch:
            lx, ly, k2 = latch
            rx, ry = z["pose_sim"][k2][:2]
            d = math.hypot(lx - rx, ly - ry)
            f = max(0.0, (d - args.standoff) / d) if d > 1e-6 else 0.0
            sx, sy = rx + f * (lx - rx), ry + f * (ly - ry)
            row.update(latched=True, latch_err=round(math.hypot(lx - gx, ly - gy), 2),
                       stop_err=round(math.hypot(sx - gx, sy - gy), 2))
            row["win"] = row["stop_err"] <= thr - args.margin
        else:
            row.update(latched=False, latch_err=None, stop_err=None, win=False)
        out.append(row)

    fails = [r for r in out if not r["success"]]
    oks = [r for r in out if r["success"]]
    rescued = [r for r in fails if r["win"]]
    false_latch = [r for r in oks if r["latched"] and r["latch_err"] > 2.0]
    print(f"variant={args.variant} agree={args.agree} m standoff={args.standoff} m "
          f"margin={args.margin} m{'  [TRUTH BEARINGS -- upper bound]' if args.truth else ''}")
    print(f"{'run':52}{'ok':>4}{'sight':>6}{'latch':>7}{'latch_err':>10}{'stop_err':>9}"
          f"{'thr':>5}{'win':>5}")
    for r in out:
        print(f"{r['run'][:51]:52}{'Y' if r['success'] else '-':>4}{r['sightings']:>6}"
              f"{'Y' if r['latched'] else '-':>7}"
              f"{'' if r['latch_err'] is None else r['latch_err']:>10}"
              f"{'' if r['stop_err'] is None else r['stop_err']:>9}{r['thr']:>5.1f}"
              f"{'Y' if r['win'] else '':>5}")
    ok = len(rescued) >= 6 and len(false_latch) <= 1
    print(f"\nfailed runs rescued: {len(rescued)}/{len(fails)} (need >= 6)   "
          f"successes latched > 2 m off: {len(false_latch)}/{len(oks)} (need <= 1)")
    print(f"latched at all: failures {sum(r['latched'] for r in fails)}/{len(fails)}, "
          f"successes {sum(r['latched'] for r in oks)}/{len(oks)}")
    print(f"G-c {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
