"""Gate G-b: can the 27B say WHERE in the frame the instruction's target is, to a few degrees?

    nav/tools/probe_target_bearing.py --replay nav/results/replay/gb
    nav/tools/probe_target_bearing.py --replay nav/results/replay/gb --variants strip --limit 20

WHY. P2 (lidar-ranged target latch) turns one describe-call answer into a world point: the
model names a ruler column, the column becomes a bearing, and the lidar ranges along that
bearing. Everything downstream inherits the bearing error, and a false "visible" becomes a
confident latch onto the wrong thing -- the main way P2 can do harm. So before any of it is
built, this asks the deployed model the question on frames whose geometry is known.

THE FRAMES are re-renders of the Sept lidar A/B runs (19 fan + 19 c1@0.30, 10/19 each) at
poses chosen by `scratchpad p2/select_poses.py`: in view (|bearing| <= 40 deg) in six
distance bins, plus up to two CONTROL poses per run with the goal >= 60 deg off the nose,
i.e. outside the 90 deg frame. Rendered by `replay_lidar_trace.py --indices --frames`, so
the robot, the stage and camera_nav are the runner's own.

THE QUESTION is the deployed describe call -- FREE_SPACE_SYSTEM + DESCRIBE_SIDED, no
thinking, the medium budget -- with the TARGET line extended by a COLUMN line, and a ten-
column ruler drawn on the frame. Two ruler variants, same pixels otherwise:
  strip   numbered bands along the bottom edge only
  grid    the same bands plus faint full-height column guides

TRUTH, and its known bias. Episodes give a goal FLOOR POINT, not the object the sentence
names; the replay smoke run measured the vending machine 0.8-0.9 m beyond its goal point.
So the true bearing here is the goal point's, seen from the CAMERA (0.25 m ahead of
base_link, which is where the image is taken), and part of any error is this proxy. The
distance-scaled slack column reports how much error a 1 m object offset alone could make.

PASS (from the P2 proposal, written before this ran):
  in-view frames <= 8 m:  visible-rate >= 70%, median |bearing error| <= 8 deg,
                          gross errors (> 20 deg) <= 15%
  control frames:         'visible' <= 15%
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav"))
sys.path.insert(0, str(REPO))
from arc_menu import DESCRIBE_SIDED, FREE_SPACE_SYSTEM  # noqa: E402
from nav.sim.episode import load_episode  # noqa: E402

W, H = 1920, 1080
FX = (W / 2.0) / math.tan(math.radians(90.1308) / 2.0)   # 957.8 px, add_cameras.py:201
CAM_FWD_M = 0.25                                          # camera_nav on base_link
N_COLS = 10
IN_VIEW_DEG, CTRL_DEG = 40.0, 60.0
MAX_NEW = 150                                             # medium describe 110 + COLUMN line

RULER_NOTE = (" A numbered ruler along the bottom edge of the image splits it into ten "
              "equal-width columns, numbered 0 at the far left to 9 at the far right.")
TARGET_COLUMN = (
    " Third: the robot has been told \"{instruction}\". Finish your answer with exactly "
    "these two lines:\n"
    "TARGET: <number> m   (roughly how far away the place or object that instruction "
    "names is, or exactly 'TARGET: not visible' if you cannot see it)\n"
    "COLUMN: <0-9>   (the ruler column that the centre of that place or object is in, or "
    "exactly 'COLUMN: none' if you cannot see it)")


def column_bearing_deg(col: int) -> float:
    """Ruler column -> bearing of its centre from the camera, + = left (FLU)."""
    u = (col + 0.5) * W / N_COLS
    return math.degrees(math.atan((W / 2.0 - u) / FX))


def draw_ruler(src: Path, dst: Path, grid: bool) -> None:
    im = Image.open(src).convert("RGB")
    assert im.size == (W, H), im.size
    d = ImageDraw.Draw(im, "RGBA")
    band = 64
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
    except OSError:
        font = ImageFont.load_default()
    cw = W / N_COLS
    for k in range(N_COLS):
        x0, x1 = k * cw, (k + 1) * cw
        shade = (0, 0, 0, 170) if k % 2 == 0 else (40, 40, 40, 170)
        d.rectangle([x0, H - band, x1, H], fill=shade)
        d.text(((x0 + x1) / 2, H - band / 2), str(k), fill=(255, 255, 255, 255),
               font=font, anchor="mm")
        if k:
            d.line([x0, H - band - 24, x0, H], fill=(255, 255, 255, 230), width=3)
            if grid:
                d.line([x0, 0, x0, H - band - 24], fill=(255, 255, 255, 70), width=2)
    im.save(dst, quality=92)


def parse(text: str) -> tuple[bool | None, float | None, int | None]:
    """(visible, distance_m, column). visible None = no TARGET line at all."""
    t = re.search(r"TARGET:\s*(not visible|[-+]?\d+(?:\.\d+)?)", text, re.I)
    c = re.search(r"COLUMN:\s*(none|\d)", text, re.I)
    if t is None:
        visible, dist = None, None
    elif t.group(1).lower() == "not visible":
        visible, dist = False, None
    else:
        visible, dist = True, float(t.group(1))
    col = int(c.group(1)) if c and c.group(1).isdigit() else None
    if col is not None and visible is None:
        visible = True
    return visible, dist, col


def ask(port: int, image: Path, system: str, user: str) -> tuple[str, float]:
    body = json.dumps({"image_paths": [str(image)], "system": system, "user": user,
                       "max_new_tokens": MAX_NEW, "think": False}).encode()
    for attempt in range(30):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/raw", data=body,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            return out["text"], float(out["latency_s"])
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise
            time.sleep(2.0)                 # a plan generation is in flight; wait it out
    raise RuntimeError("server stayed busy")


def truth(summary: dict, pose: dict) -> tuple[float, float]:
    gx, gy = summary["goal"]["xy"]
    x, y, _, yaw = pose["sim_pose"]
    cx, cy = x + CAM_FWD_M * math.cos(yaw), y + CAM_FWD_M * math.sin(yaw)
    b = math.degrees(math.atan2(gy - cy, gx - cx) - yaw)
    return (b + 180.0) % 360.0 - 180.0, math.hypot(gx - cx, gy - cy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True,
                    help="parent dir of replay_lidar_trace.py outputs (one per result)")
    ap.add_argument("--variants", default="strip,grid")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--limit", type=int, default=None, help="first N frames only (smoke)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where rulered frames and answers go (default <replay>/../gb_probe)")
    args = ap.parse_args()
    # Absolute, because the SERVER opens these paths from its own working directory.
    args.replay = args.replay.resolve()
    out = (args.out or args.replay.parent / "gb_probe").resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variants.split(",") if v]

    items = []
    for sdir in sorted(p for p in args.replay.iterdir() if (p / "summary.json").is_file()):
        s = json.loads((sdir / "summary.json").read_text())
        ep = load_episode(s["episode"])
        for pose in s["poses"]:
            frame = sdir / "frames" / f"nav_{pose['k']:03d}.jpg"
            if not frame.is_file():
                continue
            b, d = truth(s, pose)
            kind = "in" if abs(b) <= IN_VIEW_DEG else "ctrl" if abs(b) >= CTRL_DEG else "edge"
            if kind == "edge":
                continue
            ray = pose.get("goal_exact_ray") or {}
            rng = ray.get("range_m")
            # Line of sight at the lidar plane: the reference ray along the true bearing
            # reaches the goal's neighbourhood (or runs out of C1 range beyond 12 m).
            los = (rng is None and pose["goal_dist_m"] > 12.0) or (
                rng is not None and rng >= pose["goal_dist_m"] - 1.5)
            items.append(dict(result=sdir.name, episode=s["episode"],
                              success=s["run"]["success"], lidar=s["run"]["lidar"],
                              instruction=ep.instruction, k=pose["k"],
                              trace_index=pose["trace_index"], kind=kind,
                              bearing=round(b, 2), dist=round(d, 2), los=bool(los),
                              frame=str(frame)))
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} frames: {sum(i['kind'] == 'in' for i in items)} in view, "
          f"{sum(i['kind'] == 'ctrl' for i in items)} controls; variants {variants}")

    rows = []
    log = (out / "answers.jsonl").open("a")
    for n, it in enumerate(items):
        for v in variants:
            img = out / "frames" / f"{it['result']}_{it['k']:03d}_{v}.jpg"
            if not img.is_file():
                draw_ruler(Path(it["frame"]), img, grid=(v == "grid"))
            text, lat = ask(args.port, img, FREE_SPACE_SYSTEM + RULER_NOTE,
                            DESCRIBE_SIDED + TARGET_COLUMN.format(instruction=it["instruction"]))
            vis, dist_est, col = parse(text)
            pred = column_bearing_deg(col) if col is not None else None
            row = dict(it, variant=v, text=text, latency_s=lat, visible=vis,
                       dist_est=dist_est, column=col, pred_bearing=pred,
                       err=None if pred is None else round(pred - it["bearing"], 2))
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if n % 20 == 0:
            print(f"  {n + 1}/{len(items)}", flush=True)
    log.close()
    return report(rows, variants, out)


def report(rows: list[dict], variants: list[str], out: Path) -> int:
    lines, verdict = [], {}
    p = lines.append
    for v in variants:
        R = [r for r in rows if r["variant"] == v]
        p(f"\n=== variant: {v}  ({len(R)} answers, median latency "
          f"{statistics.median(r['latency_s'] for r in R):.2f} s)")
        noparse = sum(r["visible"] is None for r in R)
        p(f"  no TARGET line at all: {noparse}")
        p(f"  {'set':<26}{'n':>4}{'visible':>9}{'col given':>10}{'med|err|':>10}"
          f"{'p90|err|':>10}{'>20deg':>8}{'slack1m':>9}")
        bins = [("in view <=8 m", lambda r: r["kind"] == "in" and r["dist"] <= 8),
                ("  of which LOS", lambda r: r["kind"] == "in" and r["dist"] <= 8 and r["los"]),
                ("in view 8-12 m", lambda r: r["kind"] == "in" and 8 < r["dist"] <= 12),
                ("in view 12-35 m", lambda r: r["kind"] == "in" and r["dist"] > 12),
                ("in view, failed runs", lambda r: r["kind"] == "in" and not r["success"]),
                ("CONTROL (out of view)", lambda r: r["kind"] == "ctrl")]
        stats = {}
        for name, f in bins:
            S = [r for r in R if f(r)]
            if not S:
                continue
            vis = sum(bool(r["visible"]) for r in S) / len(S)
            errs = [abs(r["err"]) for r in S if r["err"] is not None and r["visible"]]
            med = statistics.median(errs) if errs else float("nan")
            p90 = sorted(errs)[int(0.9 * (len(errs) - 1))] if errs else float("nan")
            gross = sum(e > 20 for e in errs) / len(errs) if errs else float("nan")
            slack = statistics.median(math.degrees(math.atan(1.0 / max(r["dist"], 0.5)))
                                      for r in S)
            stats[name] = dict(n=len(S), vis=vis, med=med, gross=gross)
            p(f"  {name:<26}{len(S):>4}{100 * vis:>8.0f}%{len(errs):>10}{med:>9.1f}d"
              f"{p90:>9.1f}d{100 * gross:>7.0f}%{slack:>8.1f}d")
        a, c = stats.get("in view <=8 m"), stats.get("CONTROL (out of view)")
        ok = bool(a and c and a["vis"] >= 0.70 and a["med"] <= 8.0 and a["gross"] <= 0.15
                  and c["vis"] <= 0.15)
        verdict[v] = ok
        p(f"  G-b {'PASS' if ok else 'FAIL'}: need visible>=70%, median<=8, gross<=15% "
          f"on in-view<=8 m; visible<=15% on controls")
    text = "\n".join(lines)
    print(text)
    (out / "report.txt").write_text(text + "\n")
    json.dump(rows, (out / "rows.json").open("w"))
    return 0 if any(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
