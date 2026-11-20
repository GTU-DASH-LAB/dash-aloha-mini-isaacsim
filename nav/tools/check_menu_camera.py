#!/usr/bin/env python3
"""Per-request cameras for the arc menu: prove the old path is untouched and the new one is right.

    ~/envs/qvla/bin/python nav/tools/check_menu_camera.py --out <dir> [--frames DIR] [--orig PATH]

Run it with the POLICY SERVER'S interpreter (~/envs/qvla/bin/python): byte-identity of a JPEG is
a property of the Pillow/libjpeg build as much as of the drawing code, and the recorded menus
were encoded by that interpreter.

WHAT IS CHECKED, and why each one is the check rather than a proxy:

  A  recorded   No-camera menus re-rendered with the NEW code from the raw camera frames they
                were drawn on, compared against the recorded menu_*.jpg byte for byte and pixel
                for pixel. Raw frames live in /tmp/alohamini-nav-frames (overwritten every
                episode), so --frames should point at a snapshot; a frame is matched to its menu
                by mtime (the server renders ~10-50 ms after the runner writes the frame) and a
                wrong match cannot pass, because a wrong frame differs everywhere.
  B  old==new   The ORIGINAL arc_menu (--orig, the copy the running server loaded) against the
                new one, under BOTH QVLA_MENU_GEOM modes, on recorded decisions -- their own
                labels, and also with pivots and with a lidar-style drawn subset -- so every
                branch of render_menu's no-camera path is exercised. Legacy-geometry records
                have no raw frames left; for those the recorded menu itself is the canvas.
  C  prompts    select_system for every flag combination and the free-space system prompts:
                old == new without a camera, and with a camera whose `prompt` flag is off.
  D  geometry   MenuCamera.projector against nav/sim/g1/camera_geometry.project_ground (an
                independent implementation g1/check validated against rendered poses), over
                random poses; hfov vs fx; rescaled frames; zero attitude == project().
  E  server     server_qwen imported offline (CPU, no model; the two generations stubbed),
                driven through FastAPI's TestClient: a /predict with a G1 camera, one without,
                and malformed cameras -> 422; decisions.jsonl rows and /health checked.
  F  G1 frames  Real G1 head-cam frames: red floor markers at known world points (flat stage)
                detected and compared with where the camera puts them, live pose vs the
                walking constants; hospital frames drawn with the nav mount, the real 47.6 deg
                mount (pose derived from the same head link) and, for contrast, the level
                no-camera menu. Images are written for LOOKING at.
  G  pivots     Clearance between the 3 m arcs and the pivot glyphs for the G1 mounts.

report.json in --out; exit status 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# nav/tools/profile.py shadows the stdlib module (CLAUDE.md); keep this directory off the path.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != TOOLS]
REPO = TOOLS.parent.parent
NAV = REPO / "nav"
SCRATCH = Path("/tmp/claude-1000/-home-gtu-dsa-robotics-dash-aloha-mini-isaacsim/"
               "b4469602-c8d7-462c-adbc-d4a0ef742011/scratchpad")
DEF_ORIG = SCRATCH / "g1nav" / "server_orig" / "arc_menu.py"
DEF_FRAMES = SCRATCH / "g1nav" / "menucam" / "frames_snapshot"
G1_FLAT = SCRATCH / "g1" / "repair4" / "camera_flat"
G1_HOSP = SCRATCH / "g1" / "repair4" / "camera_hospital"
G1_ARCS = [-0.35, -0.15, 0.0, 0.15, 0.35]

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

RESULTS: dict = {}
FAILS: list[str] = []


def check(name: str, ok: bool, **info) -> bool:
    RESULTS.setdefault("checks", []).append({"name": name, "ok": bool(ok), **info})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {json.dumps(info, default=str)[:400]}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def load_module(path: Path, alias: str, geom: str, arcs: str = "coarse"):
    """Import an arc_menu file under its own name with QVLA_MENU_GEOM fixed at import."""
    os.environ["QVLA_MENU_GEOM"] = geom
    os.environ["QVLA_MENU_ARCS"] = arcs
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod                  # dataclasses resolve annotations through it
    spec.loader.exec_module(mod)
    assert mod.MENU_GEOM == geom
    return mod


def pixels(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).astype(np.int16)


def runs_dir() -> Path:
    return NAV / "results" / "menus"


def read_rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in open(run / "decisions.jsonl")]


def match_frames(run: Path, frames: list[tuple[float, Path]]) -> list[tuple[dict, Path]]:
    """(decision row, raw frame) pairs: the newest frame written within 1 s before the menu."""
    out = []
    for r in read_rows(run):
        menu = run / r["menu"]
        if not menu.is_file():
            continue
        t = menu.stat().st_mtime
        cand = [(m, f) for m, f in frames if m <= t and t - m < 1.0]
        if cand:
            out.append((r, max(cand)[1]))
    return out


# ------------------------------------------------------------------------------- A + B
def section_identity(args, new_m, old_m, new_l, old_l, out: Path) -> None:
    frames = sorted((f.stat().st_mtime, f) for f in Path(args.frames).glob("nav_*.jpg"))
    runs = sorted(p for p in runs_dir().iterdir() if (p / "decisions.jsonl").is_file())
    pairs = []
    for run in runs:
        for r, f in match_frames(run, frames):
            pairs.append((run, r, f))
    tmp = out / "identity"
    tmp.mkdir(parents=True, exist_ok=True)
    arcs = {"mounted": new_m.make_arcs(), "legacy": new_l.make_arcs()}

    # A: new code, no camera, raw frame -> must BE the recorded file.
    n_bytes = n_pix = 0
    worst = 0
    per_run: dict = {}
    for run, r, f in pairs:
        geom = r.get("menu_geom", "legacy")
        mod = new_m if geom == "mounted" else new_l
        drawn = r.get("drawn")
        labels = r["labels"]
        a = arcs[geom]
        if drawn is not None:
            keep = [i for i, lab in enumerate(labels) if lab in drawn]
            a, labels = [a[i] for i in keep], [labels[i] for i in keep]
        piv = tuple(r["pivot_labels"]) if r.get("pivot_labels") else None
        o = tmp / f"A_{run.name}_{r['menu']}"
        mod.render_menu(str(f), str(o), a, labels, pivot_labels=piv)
        rec = run / r["menu"]
        same_bytes = o.read_bytes() == rec.read_bytes()
        d = int(np.abs(pixels(o) - pixels(rec)).max())
        n_bytes += same_bytes
        n_pix += d == 0
        worst = max(worst, d)
        per_run.setdefault(run.name, [0, 0])
        per_run[run.name][0] += 1
        per_run[run.name][1] += d == 0
        o.unlink()
    check("A recorded menus reproduced from raw frames by the NEW code (no camera)",
          len(pairs) >= 20 and n_bytes == len(pairs) and worst == 0,
          decisions=len(pairs), byte_identical=n_bytes, pixel_identical=n_pix,
          max_abs_pixel_diff=worst, per_run=per_run,
          pillow=Image.__version__ if hasattr(Image, "__version__") else None)

    # B: old == new, both geometries, every no-camera branch.
    rng = random.Random(11)
    sample = rng.sample(pairs, min(args.n_sample, len(pairs))) if pairs else []
    legacy_rows = []
    for run in runs[: args.n_legacy_runs]:
        rows = read_rows(run)
        if rows and rows[0].get("menu_geom", "legacy") == "legacy":
            legacy_rows += [(run, r, run / r["menu"]) for r in rows[:: max(1, len(rows) // 3)][:3]
                            if (run / r["menu"]).is_file()]
    cases = [(run, r, f, "raw") for run, r, f in sample] + \
            [(run, r, f, "recorded-menu-as-canvas") for run, r, f in legacy_rows]
    n = worst = n_bytes_diff = 0
    variants_seen = set()
    for run, r, canvas, kind in cases:
        labels = r["labels"]
        n_arcs = len(labels)
        for geom, nm, om in (("legacy", new_l, old_l), ("mounted", new_m, old_m)):
            full = nm.make_arcs()
            assert [x.kappa for x in full] == [x.kappa for x in om.make_arcs()]
            variants = [("recorded", full, labels, None)]
            variants.append(("pivots", full, labels, (n_arcs + 1, n_arcs + 2)))
            keep = [0, 2, 3, 5]
            variants.append(("drawn-subset", [full[i] for i in keep], [labels[i] for i in keep],
                             None))
            for vname, a, lab, piv in variants:
                o1, o2 = tmp / "old.jpg", tmp / "new.jpg"
                om.render_menu(str(canvas), str(o1), a, lab, pivot_labels=piv)
                nm.render_menu(str(canvas), str(o2), a, lab, pivot_labels=piv)
                same = o1.read_bytes() == o2.read_bytes()
                d = int(np.abs(pixels(o1) - pixels(o2)).max())
                worst = max(worst, d)
                n_bytes_diff += not same
                n += 1
                variants_seen.add((geom, vname, kind))
    check("B original arc_menu == new arc_menu, no camera, both geometries, all branches",
          n > 0 and worst == 0 and n_bytes_diff == 0, renders_compared=n,
          byte_mismatches=n_bytes_diff, decisions=len(cases),
          legacy_records=len(legacy_rows), raw_frame_records=len(sample),
          max_abs_pixel_diff=worst, variants=sorted(variants_seen))


# ------------------------------------------------------------------------------- C
def section_prompts(new_m, old_m, new_l, old_l) -> None:
    bad = []
    n = 0
    cam_noprompt = new_m.MenuCamera.from_request(
        {"hfov_deg": 69.0, "cam_h": 1.09, "pitch_deg": 15.0, "arcs": G1_ARCS})
    for nm, om in ((new_m, old_m), (new_l, old_l)):
        for stop, sa, sc, piv, gd in itertools.product(
                (8, 10), (True, False), (True, False), (None, (8, 9), (3, 5)), (True, False)):
            a = om.select_system(stop, stop_allowed=sa, speed_choice=sc, pivot_labels=piv,
                                 goal_dir=gd)
            b = nm.select_system(stop, stop_allowed=sa, speed_choice=sc, pivot_labels=piv,
                                 goal_dir=gd)
            c = nm.select_system(stop, stop_allowed=sa, speed_choice=sc, pivot_labels=piv,
                                 goal_dir=gd, camera=cam_noprompt)
            n += 1
            if not (a == b == c):
                bad.append((nm.MENU_GEOM, stop, sa, sc, piv, gd))
        for two in (False, True):
            const = om.FREE_SPACE_SYSTEM_2 if two else om.FREE_SPACE_SYSTEM
            if not (const == nm.free_space_system(None, two)
                    == nm.free_space_system(cam_noprompt, two)):
                bad.append((nm.MENU_GEOM, "free_space", two))
        for name in ("DESCRIBE_SIDED", "DESCRIBE_TARGET", "STAGE_RULE", "SELECT_PREFILL"):
            if getattr(om, name) != getattr(nm, name):
                bad.append((nm.MENU_GEOM, name))
    cam_p = new_m.MenuCamera.from_request({"hfov_deg": 69.0, "cam_h": 1.093, "pitch_deg": 12.6,
                                           "prompt": True, "robot": "humanoid robot"})
    sel = new_m.select_system(8, camera=cam_p)
    fs = new_m.free_space_system(cam_p)
    check("C prompts unchanged without a camera (and with prompt=false); camera sentence only on request",
          not bad and "humanoid robot" in sel and "tilted 13 degrees down" in sel
          and "69 degree" in fs and "1.09 m" in fs,
          combinations=n, mismatches=bad[:5], prompt_true_first_line=sel.split("\n")[0])


# ------------------------------------------------------------------------------- D
def section_geometry(new_m, new_l) -> None:
    sys.path.insert(0, str(NAV / "sim" / "g1"))
    import camera_geometry as CG

    rng = random.Random(3)
    worst = 0.0
    n = 0
    for _ in range(3000):
        pose = dict(cam_h=rng.uniform(0.25, 1.4), cam_x=rng.uniform(-0.1, 0.4),
                    cam_y=rng.uniform(-0.05, 0.05), pitch_deg=rng.uniform(-5, 55),
                    roll_deg=rng.uniform(-8, 8), yaw_deg=rng.uniform(-10, 10))
        fov = rng.choice((69.0, 90.1, 58.0))
        w, h = rng.choice(((1920, 1080), (1280, 720), (640, 480)))
        cam = new_m.MenuCamera.from_request({"hfov_deg": fov, **pose})
        proj = cam.projector(w, h)
        x, y = rng.uniform(0.3, 9.0), rng.uniform(-4.0, 4.0)
        a = proj(x, y)
        b = CG.project_ground(x, y, w=w, h=h, fov_deg=fov, **pose)
        if (a is None) != (b is None):
            worst = max(worst, 1e9)
            continue
        if a is None:
            continue
        n += 1
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    check("D projection == camera_geometry.project_ground over random poses",
          worst < 1e-6 and n > 2000, compared=n, max_abs_px=worst)

    # fx form vs hfov form, and rescaled frames
    cam_f = new_m.MenuCamera.from_request({"width": 1920, "height": 1080, "fx": 1396.80859375,
                                           "fy": 1396.80859375, "cx": 960.0, "cy": 540.0,
                                           "cam_h": 1.09, "cam_x": 0.07, "pitch_deg": 15.3,
                                           "roll_deg": -1.7, "yaw_deg": 4.2})
    cam_v = new_m.MenuCamera.from_request({"hfov_deg": 69.0, "cam_h": 1.09, "cam_x": 0.07,
                                           "pitch_deg": 15.3, "roll_deg": -1.7, "yaw_deg": 4.2})
    e1 = e2 = 0.0
    pf, pv = cam_f.projector(1920, 1080), cam_v.projector(1920, 1080)
    ph = cam_f.projector(960, 540)
    for x, y in ((2.0, 0.0), (3.0, -0.8), (5.0, 1.5), (2.5, 0.4)):
        a, b, c = pf(x, y), pv(x, y), ph(x, y)
        e1 = max(e1, abs(a[0] - b[0]), abs(a[1] - b[1]))
        e2 = max(e2, abs(a[0] / 2 - c[0]), abs(a[1] / 2 - c[1]))
    check("D fx form == hfov form (69 deg D435), and a half-size frame scales the intrinsics",
          e1 < 1e-3 and e2 < 1e-9, fx_vs_hfov_px=e1, half_size_px=e2)

    # zero attitude == project(), both geometries
    worst = 0.0
    for mod in (new_m, new_l):
        cam = mod.MenuCamera.from_request({"hfov_deg": mod.CAM_FOV_DEG, "cam_h": mod.CAM_HEIGHT_M,
                                           "cam_x": mod.CAM_FORWARD_M})
        proj = cam.projector(1920, 1080)
        for arc in mod.make_arcs():
            for x, y in arc.points:
                a, b = proj(float(x), float(y)), mod.project(float(x), float(y), 1920, 1080)
                if (a is None) != (b is None):
                    worst = 1e9
                elif a is not None:
                    worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    check("D a level request camera with the server's own optics == project()",
          worst < 1e-9, max_abs_px=worst)

    # visible lengths vs camera_geometry.arc_badge at the walking pose (margin 0)
    cam = new_m.MenuCamera.from_request({"hfov_deg": 69.0, "cam_h": CG.WALK_CAM_HEIGHT_M,
                                         "cam_x": CG.WALK_CAM_FORWARD_M,
                                         "pitch_deg": CG.WALK_PITCH_DEG})
    arcs = new_m.make_arcs()
    vis = cam.visible_m(arcs, 1920, 1080)
    ref = [CG.visible_arc(a.kappa)["visible_m"] for a in arcs]
    check("D visible metres per arc match camera_geometry.visible_arc (G1 walking pose)",
          max(abs(a - b) for a, b in zip(vis, ref)) < 0.03,
          kappas=[a.kappa for a in arcs], visible_m=vis, camera_geometry=ref)


# ------------------------------------------------------------------------------- E
SERVER_CHILD = r'''
import json, os, sys, time, threading
from pathlib import Path
sys.path.insert(0, os.environ["MC_NAV"] + "/policy_server")
import server_qwen as S
from fastapi.testclient import TestClient
from PIL import Image

out = Path(os.environ["MC_OUT"])
res = {}
S._state["model"] = object()                 # no weights: the two generations are stubbed
seen = {"systems": [], "menus": []}
real_render = S.render_menu
def spy_render(img, menu, arcs, labels, **kw):
    seen["menus"].append({"labels": list(labels), "kappas": [a.kappa for a in arcs],
                          "camera": kw.get("camera") is not None, "colours": kw.get("colours")})
    return real_render(img, menu, arcs, labels, **kw)
S.render_menu = spy_render
def fake_tta(image_paths, system, user, think_budget, max_new, prefill=""):
    seen["systems"].append(system)
    if prefill:                                    # the select call: answer the straight arc
        m = seen["menus"][-1]
        lab = m["labels"][min(range(len(m["kappas"])), key=lambda i: abs(m["kappas"][i]))]
        return prefill + str(lab), "", 0.0
    return "The way ahead is open. The most open, walkable floor is straight ahead. TARGET: not visible", "", 0.0
S.think_then_answer = fake_tta

c = TestClient(S.app)
frame = os.environ["MC_FRAME"]
args = json.loads(os.environ["MC_CAMERA"])
r = c.post("/reset", json={"run": "check_menu_camera"})
res["reset"] = r.json()
run_dir = Path(r.json()["run_dir"])
base = {"image_paths": [frame], "instruction": "Walk down the hallway.", "wait_fresh": True}
r1 = c.post("/predict", json={**base, "current_step": 0, "camera": args})
res["with_camera"] = {"status": r1.status_code, "n_wp": len(r1.json().get("waypoints", [])),
                      "first_wp": r1.json().get("waypoints", [[None]])[0] if r1.status_code == 200 else r1.text}
r2 = c.post("/predict", json={**base, "current_step": 30})
res["without_camera"] = {"status": r2.status_code, "n_wp": len(r2.json().get("waypoints", []))}
bad = {}
for name, cam in (("unknown_key", {**args, "pitch": 15.0}),
                  ("no_cam_h", {k: v for k, v in args.items() if k != "cam_h"}),
                  ("fx_and_hfov", {**args, "hfov_deg": 69.0}),
                  ("arc_not_on_menu", {**args, "arcs": [0.25]}),
                  ("nan_pitch", {**args, "pitch_deg": float("nan")}),
                  ("pitch_90", {**args, "pitch_deg": 95.0})):
    try:
        rr = c.post("/predict", json={**base, "current_step": 60, "camera": cam})
        bad[name] = [rr.status_code, rr.text[:160]]
    except Exception as exc:                       # NaN may not even serialise
        bad[name] = ["client-side", repr(exc)[:160]]
res["bad"] = bad
h = c.get("/health").json()
res["health_menu_camera"] = h.get("menu_camera")
res["rows"] = [json.loads(l) for l in open(run_dir / "decisions.jsonl")]
res["menus_drawn"] = seen["menus"]
res["systems"] = seen["systems"]
res["run_dir"] = str(run_dir)
(out / "server_child.json").write_text(json.dumps(res, indent=1, default=str))
print("SERVER_CHILD_DONE")
'''


def section_server(args, out: Path, new_m, old_m, frame: Path, cam_args: dict) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", QVLA_FORMAT="menu", QVLA_MENU_GEOM="mounted",
               QVLA_MENU_ARCS="coarse", QVLA_MENU_PIVOTS="0", QVLA_MENU_FRAMES="1",
               QVLA_THINK_LEVEL="medium", QVLA_MENU_SPEED="0.45",
               QVLA_MENU_DIR=str(out / "server_menus"), QVLA_UPLOAD_DIR=str(out / "server_uploads"),
               MC_NAV=str(NAV), MC_OUT=str(out), MC_FRAME=str(frame), MC_CAMERA=json.dumps(cam_args))
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", SERVER_CHILD], env=env, capture_output=True,
                       text=True, timeout=600)
    (out / "server_child.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if "SERVER_CHILD_DONE" not in p.stdout:
        check("E server offline harness ran", False, rc=p.returncode, stderr_tail=p.stderr[-800:])
        return
    res = json.loads((out / "server_child.json").read_text())
    rows = res["rows"]
    ok_rows = len(rows) == 2
    rc, rn = (rows + [{}, {}])[:2]
    full = [a.kappa for a in new_m.make_arcs()]
    lab_by_k = {k: lab for k, lab in zip(full, rc.get("labels", []))}
    want_drawn = [lab_by_k[k] for k in full if k in G1_ARCS]
    cam = rc.get("camera") or {}
    check("E /predict with a G1 camera: 200, plan, row records the request camera + subset + visibility",
          ok_rows and res["with_camera"]["status"] == 200 and res["with_camera"]["n_wp"] > 0
          and cam.get("source") == "request" and cam.get("arcs") == G1_ARCS
          and rc.get("drawn") == want_drawn and len(rc.get("visible_m") or []) == 7
          and abs(cam.get("pitch_deg", 0) - cam_args["pitch_deg"]) < 1e-4
          and cam.get("drawn_wh") == [1920, 1080] and rc.get("kappa") == 0.0,
          row_camera=cam, drawn=rc.get("drawn"), want_drawn=want_drawn,
          visible_m=rc.get("visible_m"), kappa=rc.get("kappa"))
    check("E /predict without a camera: row records the server's level camera, drawn null",
          ok_rows and res["without_camera"]["status"] == 200
          and (rn.get("camera") or {}).get("source") == "server"
          and (rn.get("camera") or {}).get("geom") == "mounted" and rn.get("drawn") is None
          and rn.get("visible_m") is None,
          row_camera=rn.get("camera"))
    bad = res["bad"]
    check("E malformed cameras are refused with 422 (never a silently level menu)",
          all(v[0] in (422, "client-side") for v in bad.values()) and bad["unknown_key"][0] == 422,
          bad=bad)
    hc = res["health_menu_camera"] or {}
    check("E /health reports menu_camera.supported, the keys, the default camera and counts",
          hc.get("supported") is True and hc.get("requests") == 1 and hc.get("menus") == 1
          and (hc.get("default") or {}).get("geom") == "mounted"
          and (hc.get("last") or {}).get("source") == "request",
          menu_camera={k: hc.get(k) for k in ("supported", "requests", "menus", "default")})
    md = res["menus_drawn"]
    check("E camera menu keeps each arc's full-menu colour; no-camera menu passes no colours",
          len(md) == 2 and md[0]["camera"] and md[0]["colours"] == [1, 2, 3, 4, 5]
          and not md[1]["camera"] and md[1]["colours"] is None, menus=md)
    # The no-camera decision's prompts must be the original module's words.
    sys_old = old_m.select_system(8, stop_allowed=True, speed_choice=False, pivot_labels=None,
                                  goal_dir=False)
    systems = res["systems"]
    check("E the model calls see the ORIGINAL system prompts with and without a camera (prompt off)",
          len(systems) == 4 and systems[0] == systems[2] == old_m.FREE_SPACE_SYSTEM
          and systems[1] == systems[3] == sys_old)
    # And the no-camera menu the server wrote IS the original render on that frame.
    run_dir = Path(res["run_dir"])
    o = out / "server_nocam_old.jpg"
    old_m.render_menu(str(frame), str(o), old_m.make_arcs(), rn["labels"], pivot_labels=None)
    same = o.read_bytes() == (run_dir / rn["menu"]).read_bytes()
    check("E the server's no-camera menu is byte-identical to the original renderer's",
          same, menu=str(run_dir / rn["menu"]), wall_s=round(time.time() - t0, 1))
    RESULTS["server_camera_menu"] = str(run_dir / rc["menu"])


# ------------------------------------------------------------------------------- F
def derived_real_pose(menu_args: dict, nav_mount_deg: float = 12.0, real_mount_deg: float = 47.6
                      ) -> dict:
    """The real D435 mount's pose from the nav mount's: same head link, same lens position,
    mount pitched (47.6 - 12) deg further about the head link's +Y.
    R_nav = B(p, r, y) M_LEVEL = R_link Ry(12) M_LEVEL  =>  B_real = B(p, r, y) Ry(35.6)."""
    p, r, y = (math.radians(menu_args[k]) for k in ("pitch_deg", "roll_deg", "yaw_deg"))

    def rz(t):
        return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])

    def rx(t):
        return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])

    b = rz(y) @ ry(p) @ rx(r) @ ry(math.radians(real_mount_deg - nav_mount_deg))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -b[2, 0]))))
    roll = math.degrees(math.atan2(b[2, 1], b[2, 2]))
    yaw = math.degrees(math.atan2(b[1, 0], b[0, 0]))
    return {**menu_args, "pitch_deg": pitch, "roll_deg": roll, "yaw_deg": yaw}


def detect_markers(img: np.ndarray, preds: list[tuple[int, tuple[float, float]]],
                   win_u: int = 60, win_v: int = 34):
    r, g, b = (img[..., k].astype(np.int32) for k in range(3))
    red = (r > 90) & (r > 2 * g) & (r > 2 * b)
    H, W = red.shape
    out = {}
    for k, (u, v) in preds:
        u0, v0 = int(round(u)), int(round(v))
        if not (win_u <= u0 < W - win_u and win_v <= v0 < H - win_v):
            continue
        sub = red[v0 - win_v:v0 + win_v, u0 - win_u:u0 + win_u]
        ys, xs = np.nonzero(sub)
        # one whole marker inside the window, or no measurement
        if len(xs) < 25 or xs.min() == 0 or ys.min() == 0 or xs.max() == 2 * win_u - 1 \
                or ys.max() == 2 * win_v - 1:
            continue
        out[k] = (u0 - win_u + xs.mean(), v0 - win_v + ys.mean())
    return out


def section_g1_frames(new_m, out: Path, spec: dict) -> None:
    sys.path.insert(0, str(NAV / "sim" / "g1"))
    import camera_geometry as CG

    nav_spec = spec["nav_camera"]
    sys.path.insert(0, str(NAV / "policy_server"))
    import client as CL

    # ---------------- flat stage: markers at known world points
    rep = json.loads((NAV / "results" / "g1" / "g1_camera_flat_r4.json").read_text())
    rows = rep["video_rows"]
    markers = [(float(x), float(y)) for x in np.arange(1.2, 9.01, 0.6) for y in np.arange(-3.0, 3.01, 0.6)]
    MARKER_TOP_Z = 0.01
    stats = {"live": [], "const": [], "live_yaw_fit": [], "const_yaw_fit": [], "fit_yaw_deg": []}
    per_frame = []
    look = None
    for row in rows:
        if row["seg"] not in ("stand", "straight") or row["i"] % 2:
            continue
        fpath = G1_FLAT / "frames" / "nav" / f"{row['i']:05d}.jpg"
        if not fpath.is_file():
            continue
        img = np.asarray(Image.open(fpath).convert("RGB"))
        px, py = row["pos"][0], row["pos"][1]

        def heading_pts(yaw_rad):
            c, s = math.cos(yaw_rad), math.sin(yaw_rad)
            return [(c * (mx - px) + s * (my - py), -s * (mx - px) + c * (my - py)) for mx, my in markers]

        live_args = dict(row["menu_args"])
        live_args["cam_h"] -= MARKER_TOP_Z
        const_args = {"cam_h": CG.WALK_CAM_HEIGHT_M - MARKER_TOP_Z, "cam_x": CG.WALK_CAM_FORWARD_M,
                      "cam_y": 0.0, "pitch_deg": CG.WALK_PITCH_DEG, "roll_deg": 0.0, "yaw_deg": 0.0}

        def project_all(cam_args, yaw_rad):
            cam = new_m.MenuCamera.from_request(CL.camera_from_spec(nav_spec, cam_args))
            proj = cam.projector(img.shape[1], img.shape[0])
            pts = heading_pts(yaw_rad)
            return [(k, proj(x, y)) for k, (x, y) in enumerate(pts)
                    if x - cam_args["cam_x"] < 4.0 and proj(x, y) is not None]

        preds = project_all(live_args, 0.0)
        dets = detect_markers(img, preds)
        if len(dets) < 4:
            continue

        def resid(cam_args, yaw_rad):
            pr = dict(project_all(cam_args, yaw_rad))
            e = [math.hypot(dets[k][0] - pr[k][0], dets[k][1] - pr[k][1]) for k in dets if k in pr]
            return float(np.median(e)) if e else 1e9, e

        def fit(cam_args):
            best = (1e9, 0.0)
            for yd in np.arange(-3.0, 3.001, 0.05):
                m, _ = resid(cam_args, math.radians(yd))
                best = min(best, (m, yd))
            return best

        m_live, e_live = resid(live_args, 0.0)
        m_const, _ = resid(const_args, 0.0)
        f_live, y_live = fit(live_args)
        f_const, _ = fit(const_args)
        stats["live"].append(m_live)
        stats["const"].append(m_const)
        stats["live_yaw_fit"].append(f_live)
        stats["const_yaw_fit"].append(f_const)
        stats["fit_yaw_deg"].append(y_live)
        per_frame.append({"i": row["i"], "seg": row["seg"], "markers": len(dets),
                          "median_px_live": round(m_live, 2), "median_px_const": round(m_const, 2),
                          "median_px_live_yawfit": round(f_live, 2), "fit_yaw_deg": round(y_live, 2),
                          "median_px_const_yawfit": round(f_const, 2),
                          "pitch_deg": row["menu_args"]["pitch_deg"], "roll_deg": row["menu_args"]["roll_deg"]})
        if look is None and row["seg"] == "straight" and len(dets) >= 8:
            look = (row, fpath, preds, dets, y_live)
    summ = {k: round(float(np.median(v)), 2) for k, v in stats.items() if v}
    RESULTS["g1_flat_markers"] = {"summary_median": summ, "frames": per_frame}
    # 10 px, not ~0: these rows read the pose one kit.update() AFTER the grab, and g1's round-2
    # pixel latency test found the rendered frame is the pose of the update BEFORE (lag1 median
    # 3.1 px vs lag0 14 px). The head jitters 1-2 deg of roll within 0.1 s even standing, so a
    # one-update lag alone is worth several px. Fitting yaw+roll+pitch offsets per frame brings
    # these frames to 1.4-5 px with roll offsets of -1.0..+1.5 deg and no consistent sign
    # (scratchpad/g1nav/menucam/diag_markers.py) -- jitter, not a convention error, which would
    # be one sign and grow with |roll|.
    check("F G1 flat frames: red floor markers land where the live camera puts them",
          bool(per_frame) and summ["live_yaw_fit"] < 10.0
          and summ["const_yaw_fit"] > 3 * summ["live_yaw_fit"],
          frames=len(per_frame), median_px=summ,
          note="pelvis yaw is not in the rows (stand/straight from yaw 0); 'yawfit' fits it, 1 DOF; "
               "the rows read the pose one update after the grab (g1 r2: the frame is lag1)")
    if look is not None:
        row, fpath, preds, dets, y_live = look
        args = CL.camera_from_spec(nav_spec, row["menu_args"], arcs=G1_ARCS, name="g1_head_nav")
        cam = new_m.MenuCamera.from_request(args)
        o = out / f"F_flat_markers_{row['i']:05d}.jpg"
        labels = [3, 6, 1, 7, 2, 5, 4]
        full = new_m.make_arcs()
        keep, _ = new_m.camera_keep(full, cam, 1920, 1080)
        new_m.render_menu(str(fpath), str(o), [full[i] for i in keep], [labels[i] for i in keep],
                          camera=cam, colours=keep)
        im = Image.open(o).convert("RGB")
        d = ImageDraw.Draw(im)
        # predicted marker centres (cyan crosses) from the LIVE pose at the fitted pelvis yaw
        live_args = dict(row["menu_args"])
        live_args["cam_h"] -= MARKER_TOP_Z
        cam_m = new_m.MenuCamera.from_request(CL.camera_from_spec(nav_spec, live_args))
        proj = cam_m.projector(1920, 1080)
        c, s = math.cos(math.radians(y_live)), math.sin(math.radians(y_live))
        px, py = row["pos"][0], row["pos"][1]
        for mx, my in markers:
            x, y = c * (mx - px) + s * (my - py), -s * (mx - px) + c * (my - py)
            p = proj(x, y)
            if p is None or x > 4.0:
                continue
            d.line([(p[0] - 22, p[1]), (p[0] + 22, p[1])], fill=(0, 255, 255), width=4)
            d.line([(p[0], p[1] - 22), (p[0], p[1] + 22)], fill=(0, 255, 255), width=4)
        for k, (u, v) in dets.items():
            d.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 255, 0), width=3)
        im.save(o, quality=92)
        RESULTS["look_flat"] = str(o)

    # ---------------- hospital: nav mount, real mount, and the level menu, same instant
    hrep = json.loads((NAV / "results" / "g1" / "g1_camera_hospital_r4.json").read_text())
    picks = []
    for row in hrep["video_rows"]:
        if row["i"] in (40, 120, 200, 280):
            picks.append(row)
    real_spec = spec["head_camera_real"]
    tiles = []
    vis_tab = []
    for row in picks:
        nav_f = G1_HOSP / "frames" / "nav" / f"{row['i']:05d}.jpg"
        real_f = G1_HOSP / "frames" / "real" / f"{row['i']:05d}.jpg"
        if not nav_f.is_file():
            continue
        labels = [3, 6, 1, 7, 2, 5, 4]
        full = new_m.make_arcs()
        # nav mount, G1 subset
        cam = new_m.MenuCamera.from_request(
            CL.camera_from_spec(nav_spec, row["menu_args"], arcs=G1_ARCS, name="g1_head_nav"))
        keep, vis = new_m.camera_keep(full, cam, 1920, 1080)
        o1 = out / f"F_hosp_{row['i']:05d}_nav.jpg"
        new_m.render_menu(str(nav_f), str(o1), [full[i] for i in keep], [labels[i] for i in keep],
                          camera=cam, colours=keep)
        # the same frame drawn as if it were camera_nav (no camera in the request)
        o0 = out / f"F_hosp_{row['i']:05d}_levelmenu.jpg"
        new_m.render_menu(str(nav_f), str(o0), full, labels)
        row_vis = {"i": row["i"], "nav_pitch": row["menu_args"]["pitch_deg"],
                   "nav_visible_m": dict(zip([a.kappa for a in full], vis))}
        o2 = None
        if real_f.is_file():
            rargs = derived_real_pose(row["menu_args"])
            camr = new_m.MenuCamera.from_request(
                CL.camera_from_spec(real_spec, rargs, name="g1_head_real"))
            keep_r, vis_r = new_m.camera_keep(full, camr, 1920, 1080)
            o2 = out / f"F_hosp_{row['i']:05d}_real.jpg"
            new_m.render_menu(str(real_f), str(o2), full, labels, camera=camr)
            row_vis.update(real_pitch=round(rargs["pitch_deg"], 2),
                           real_visible_m=dict(zip([a.kappa for a in full], vis_r)))
        vis_tab.append(row_vis)
        tiles.append((o0, o1, o2))
    if tiles:
        tw, th = 640, 360
        sheet = Image.new("RGB", (tw * 3, th * len(tiles)), (0, 0, 0))
        for r_, (o0, o1, o2) in enumerate(tiles):
            for c_, o in enumerate((o0, o1, o2)):
                if o is not None:
                    sheet.paste(Image.open(o).convert("RGB").resize((tw, th)), (c_ * tw, r_ * th))
        d = ImageDraw.Draw(sheet)
        for c_, t in enumerate(("no camera (level camera_nav model)", "G1 nav mount, live pose",
                                "G1 real 47.6 deg mount (derived pose)")):
            d.rectangle([c_ * tw, 0, c_ * tw + 420, 26], fill=(0, 0, 0))
            d.text((c_ * tw + 6, 6), t, fill=(255, 255, 255))
        sp = out / "F_hosp_contact_sheet.jpg"
        sheet.save(sp, quality=90)
        RESULTS["look_hospital_sheet"] = str(sp)
    RESULTS["g1_hospital_visibility"] = vis_tab


# ------------------------------------------------------------------------------- G
def section_pivots(new_m) -> None:
    sys.path.insert(0, str(NAV / "sim" / "g1"))
    r = 96 * 1.05
    glyph_bottom = 0.28 * 1080 + 1.3 * r
    full = new_m.make_arcs()
    worst_nav = 1e9
    for pitch in np.arange(10.0, 18.01, 1.0):
        for h in (1.05, 1.09, 1.2):
            for roll in (-4.0, 0.0, 4.0):
                cam = new_m.MenuCamera.from_request({"hfov_deg": 69.0, "cam_h": h, "cam_x": 0.07,
                                                     "pitch_deg": float(pitch), "roll_deg": roll,
                                                     "arcs": G1_ARCS})
                proj = cam.projector(1920, 1080)
                for a in full:
                    if a.kappa not in G1_ARCS:
                        continue
                    vs = [p[1] for p in (proj(float(x), float(y)) for x, y in a.points)
                          if p is not None and 0 <= p[0] < 1920 and 0 <= p[1] < 1080]
                    if vs:
                        worst_nav = min(worst_nav, min(vs) - glyph_bottom)
    cam = new_m.MenuCamera.from_request({"hfov_deg": 69.0, "cam_h": 1.18, "cam_x": 0.05,
                                         "pitch_deg": 47.6})
    proj = cam.projector(1920, 1080)
    top_real = min(p[1] for a in full for p in (proj(float(x), float(y)) for x, y in a.points)
                   if p is not None and 0 <= p[0] < 1920 and 0 <= p[1] < 1080)
    RESULTS["pivot_clearance"] = {"glyph_bottom_row": glyph_bottom,
                                  "nav_mount_min_gap_px": round(worst_nav, 1),
                                  "real_mount_topmost_arc_row": round(top_real, 1)}
    check("G pivot glyphs clear the G1 nav-mount arcs (pitch 10-18, roll +-4, h 1.05-1.2)",
          worst_nav > 0, min_gap_px=round(worst_nav, 1),
          real_mount_topmost_arc_row=round(top_real, 1),
          note="the real 47.6 deg mount puts arcs under the glyphs: keep pivots off there")


# ------------------------------------------------------------------------------- cost
def section_cost(new_m, frame: Path, out: Path) -> None:
    cam = new_m.MenuCamera.from_request({"width": 1920, "height": 1080, "fx": 1396.80859375,
                                         "cam_h": 1.09, "cam_x": 0.07, "pitch_deg": 15.3,
                                         "roll_deg": -1.2, "yaw_deg": 2.0, "arcs": G1_ARCS})
    full = new_m.make_arcs()
    labels = [3, 6, 1, 7, 2, 5, 4]
    o = out / "cost.jpg"
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        new_m.render_menu(str(frame), str(o), full, labels)
        t.append(time.perf_counter() - t0)
    tc, tk = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        keep, _ = new_m.camera_keep(full, cam, 1920, 1080)
        t1 = time.perf_counter()
        new_m.render_menu(str(frame), str(o), [full[i] for i in keep], [labels[i] for i in keep],
                          camera=cam, colours=keep)
        tc.append(time.perf_counter() - t0)
        tk.append(t1 - t0)
    RESULTS["cost_ms"] = {"render_no_camera": round(1000 * float(np.median(t)), 1),
                          "camera_keep": round(1000 * float(np.median(tk)), 1),
                          "camera_keep_plus_render": round(1000 * float(np.median(tc)), 1)}
    print(f"[info] cost {RESULTS['cost_ms']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=str(DEF_FRAMES),
                    help="snapshot of /tmp/alohamini-nav-frames with mtimes preserved (cp -p)")
    ap.add_argument("--orig", default=str(DEF_ORIG), help="the arc_menu.py the running server loaded")
    ap.add_argument("--n-sample", type=int, default=24)
    ap.add_argument("--n-legacy-runs", type=int, default=10,
                    help="oldest runs to sample legacy-geometry records from (3 each)")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: A,B,C,D,E,F,G")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip().upper() for s in args.skip.split(",") if s.strip())

    new_path = NAV / "arc_menu.py"
    new_l = load_module(new_path, "arc_menu_new_legacy", "legacy")
    old_l = load_module(Path(args.orig), "arc_menu_old_legacy", "legacy")
    new_m = load_module(new_path, "arc_menu_new_mounted", "mounted")
    old_m = load_module(Path(args.orig), "arc_menu_old_mounted", "mounted")
    RESULTS["inputs"] = {"orig": args.orig, "frames": args.frames, "python": sys.executable,
                         "pillow": getattr(Image, "__version__", None)}
    spec = json.loads((NAV / "results" / "g1" / "asset_spec.json").read_text())

    if not ({"A", "B"} <= skip):
        section_identity(args, new_m, old_m, new_l, old_l, out)
    if "C" not in skip:
        section_prompts(new_m, old_m, new_l, old_l)
    if "D" not in skip:
        section_geometry(new_m, new_l)
    hosp_frame = G1_HOSP / "frames" / "nav" / "00120.jpg"
    if "E" not in skip and hosp_frame.is_file():
        sys.path.insert(0, str(NAV / "policy_server"))
        import client as CL
        hrow = next(r for r in json.loads((NAV / "results" / "g1" / "g1_camera_hospital_r4.json")
                                          .read_text())["video_rows"] if r["i"] == 120)
        cam_args = CL.camera_from_spec(spec["nav_camera"], hrow["menu_args"], arcs=G1_ARCS,
                                       name="g1_head_nav")
        section_server(args, out, new_m, old_m, hosp_frame, cam_args)
    if "F" not in skip:
        section_g1_frames(new_m, out, spec)
    if "G" not in skip:
        section_pivots(new_m)
    if hosp_frame.is_file():
        section_cost(new_m, hosp_frame, out)
    RESULTS["failed"] = FAILS
    (out / "report.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\n{len(RESULTS.get('checks', [])) - len(FAILS)}/{len(RESULTS.get('checks', []))} checks "
          f"passed; report {out / 'report.json'}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
