"""Build every figure in the QVLA paper from the repository's own data.

Run from the repository root:

    python3 midi_conference/qvla_paper/scripts/make_figures.py [overview|menu|timeline|results|qual]

Every figure is drawn at its printed size -- at most the 122 mm LNCS text width -- and is
included in LaTeX at scale 1, so a label drawn at 6.5 pt is printed at 6.5 pt (Springer's
floor is 6 pt). Fonts are embedded as TrueType, never Type 3. Series are separated by
marker shape and hatching as well as colour, so the figures survive greyscale printing.

Data sources, all inside this repository:
  - arc geometry, plan builder and speed law: nav/arc_menu.py (imported, not re-typed)
  - closed-loop results: nav/results/*.json, scored by nav/sim/summarize_runs.score()
  - qualitative frames: nav/results/videos/*.mp4 (decoded with ffmpeg)
  - language-authority probes: the probe logs summarised in CLAUDE.md / nav/vlm_survey.md
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "figures"
sys.path.insert(0, str(REPO / "nav"))
sys.path.insert(0, str(REPO / "nav" / "sim"))
import arc_menu as AM  # noqa: E402
import summarize_runs as SR  # noqa: E402

TEXTWIDTH_PT = 122.0 / 25.4 * 72.0          # 345.8 pt, the LNCS text block
PT = 1.0 / 72.0                              # inches per point
FS = 6.5                                     # base label size, pt
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "Liberation Sans", "font.size": FS,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "axes.titlesize": 7.0, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
    "axes.titlepad": 3.0, "lines.linewidth": 1.0,
})
PAL = [tuple(c / 255 for c in rgb) for rgb in AM._PALETTE]
DARK = "#1f1f1f"
CRUISE = 0.45        # QVLA_MENU_SPEED in nav/config/profiles/baseline.yaml

# Closed-loop ladders used in the paper, identified by their result-file time window.
# A ladder is a window, not "the latest run": result files do not record every server
# knob, so two ladders with identical labels can be different experiments (CLAUDE.md).
QVLA_WINDOW = ("20260904-232005", "20260905-022414")     # shipping profile, 19 episodes
TICVLA_WINDOW = ("20260814-223021", "20260814-235645")   # TIC-VLA, braking controller
INDOOR = ["office_nearest_elevator", "hospital_down_hallway", "hospital_down_hallway2",
          "office_passing_hallway", "hospital_vending_machine", "hospital_vending_machine2",
          "office_hallway_turn", "office_hallway_turn2", "hospital_past_wheelchairs",
          "hospital_forward_staircase", "hospital_exit_room", "warehouse", "warehouse_aisle6"]


def new_fig(w_pt: float, h_pt: float):
    return plt.figure(figsize=(w_pt * PT, h_pt * PT))


def save(fig, name: str, **kw) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def ladder(window, policy_sub: str = "", controller: str = "braking") -> dict:
    cfg = SR.load_config()
    rows = {}
    for p in sorted(glob.glob(str(REPO / "nav" / "results" / "*.json"))):
        ts = os.path.basename(p)[:15]
        if not (window[0] <= ts <= window[1]):
            continue
        d = json.load(open(p))
        if policy_sub and policy_sub not in d.get("policy", ""):
            continue
        if controller and d.get("controller") != controller:
            continue
        r = SR.score(Path(p), cfg)
        if r is not None:
            rows[d["episode"]] = r
    return rows


# --------------------------------------------------------------------------------------
# Fig. 1 -- system overview.  Axes are in points, so a box's width can be checked
# against its text: Liberation Sans at 6.5 pt averages ~3.3 pt per character.
# --------------------------------------------------------------------------------------
def fig_overview() -> None:
    W, H = TEXTWIDTH_PT, 142.0
    fig = new_fig(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ls="-", title_size=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                    fc=fc, ec=DARK, lw=0.6, ls=ls))
        ax.text(x + w / 2, y + h - 4.5, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=DARK)
        for i, s in enumerate(lines):
            ax.text(x + w / 2, y + h - 14.0 - 7.6 * i, s, ha="center", va="top",
                    fontsize=FS, color=DARK)

    def arrow(p, q, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                     color=DARK, ls=ls, shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={rad}"))

    def note(x, y, s, ha="center", va="bottom"):
        ax.text(x, y, s, ha=ha, va=va, fontsize=FS, color=DARK, style="italic")

    c_sim, c_menu, c_vlm, c_ctl = "#e3ebf6", "#e6f2df", "#fbe9d6", "#ede6f5"
    top, h = 86.0, 52.0            # top row: perception and decision (one per ~2.1 s)
    bot = 4.0                      # bottom row: execution (0.5 s replan, 60 Hz tracking)

    box(2, top, 66, h, "Isaac Sim 6.0.1",
        ["AlohaMini robot", "DynaNav scenes", "physics 60 Hz", "(GPU 0)"], c_sim)
    box(86, top, 74, h, "Arc menu",
        ["7 arcs, 3 m long", "$\\kappa\\in[-0.6,\\,0.6]$ m$^{-1}$",
         "labels 1-7 shuffled", "8 = STOP (not drawn)"], c_menu)
    ax.add_patch(FancyBboxPatch((176, top), 168, h, boxstyle="round,pad=0,rounding_size=4",
                                fc=c_vlm, ec=DARK, lw=0.6))
    ax.text(260, top + h - 4.5, "Qwen3.8-27B-FP8, frozen (GPU 1)", ha="center", va="top",
            fontsize=7.0, fontweight="bold")
    for x0, bw, title, lines in (
            (179, 80, "(1) describe", ["raw frame + instruction", "blocked? open side?",
                                   "TARGET: $d$ m / not visible"]),
            (265, 76, "(2) select", ["menu + description", "+ recent-choice note",
                                 "“The number is …”"])):
        ax.add_patch(FancyBboxPatch((x0, top + 4), bw, 34, boxstyle="round,pad=0,rounding_size=3",
                                    fc="white", ec=DARK, lw=0.5))
        ax.text(x0 + bw / 2, top + 35.0, title, ha="center", va="top", fontsize=FS,
                fontweight="bold")
        for i, s in enumerate(lines):
            ax.text(x0 + bw / 2, top + 27.0 - 7.4 * i, s, ha="center", va="top", fontsize=FS)
    arrow((259, top + 21), (265, top + 21))

    box(262, bot, 82, h, "Plan builder",
        ["arc $\\kappa$, then straight", "10 s = 100 waypoints", "speed from $d$"], c_ctl)
    box(176, bot, 70, h, "/predict (async)",
        ["newest plan, sliced", "by its staleness;", "replan every 0.5 s"], c_ctl)
    box(90, bot, 70, h, "Braking pursuit",
        ["1 m lookahead", "$v\\leq$ plan speed", "$|\\omega|\\leq1.2$ rad/s"], c_ctl)
    box(2, bot, 72, h, "Collision guard",
        ["7 rays over ±35°", "stop at 0.6 m,", "slide along walls"], c_ctl)

    y_mid = top + 26
    arrow((68, y_mid), (86, y_mid))
    note(77, y_mid + 2, "RGB")
    note(77, y_mid - 2, "448$^2$", va="top")
    arrow((160, y_mid), (176, y_mid))
    y_low = bot + 26
    arrow((302, top), (302, bot + h))
    note(305, (top + bot + h) / 2, "label, $d$", ha="left", va="center")
    arrow((262, y_low), (246, y_low))
    arrow((176, y_low), (160, y_low))
    arrow((90, y_low), (74, y_low))
    arrow((38, bot + h), (38, top))
    note(41, (top + bot + h) / 2, "$(v,\\omega)$", ha="left", va="center")

    # recovery loop (dashed): the only path by which the robot ever reverses
    rx, ry, rw, rh = 84, 60, 172, 22
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh, boxstyle="round,pad=0,rounding_size=3",
                                fc="white", ec=DARK, lw=0.6, ls=(0, (3, 2))))
    ax.text(rx + rw / 2, ry + rh / 2 + 3.9, "Recovery (wedge 3 s / balk 5 s): reverse 0.5-0.8 m,",
            ha="center", va="center", fontsize=FS)
    ax.text(rx + rw / 2, ry + rh / 2 - 3.9, "/replan, STOP removed, a note added to both calls",
            ha="center", va="center", fontsize=FS)
    arrow((rx + rw - 30, ry + rh), (rx + rw - 30, top), ls=(0, (3, 2)))
    arrow((rx, ry + rh / 2), (60, bot + h), ls=(0, (3, 2)), rad=0.25)
    save(fig, "fig1_overview")


# --------------------------------------------------------------------------------------
# Fig. 2 -- (a) the menu's arcs and the plan one of them becomes, (b) the speed law
# --------------------------------------------------------------------------------------
LINESTYLES = ["-", "--", "-.", "-", "-.", "--", "-"]


def fig_menu_geometry() -> None:
    W, H = TEXTWIDTH_PT, 112.0
    fig = new_fig(W, H)
    arcs = AM.make_arcs(AM.CURVATURE_SETS["coarse"], AM.DEFAULT_LENGTH_M, 80)

    # (a) top-down in the robot frame; x forward is drawn up, y left is drawn left
    ax = fig.add_axes([26 / W, 21 / H, 150 / W, 80 / H])
    chosen = 0.35
    plan = AM.plan_from_kappa(chosen, CRUISE, 100, 0.1)          # what label +0.35 drives
    ax.plot(-plan[:, 1], plan[:, 0], color="0.55", lw=2.6, solid_capstyle="round", zorder=1)
    ax.plot(-plan[9::10, 1], plan[9::10, 0], ls="none", marker="o", ms=1.8, color=DARK,
            zorder=4)
    ax.text(-plan[-1, 1], plan[-1, 0] + 0.14, "10 s plan\nfor $\\kappa$=+0.35",
            ha="center", va="bottom", fontsize=FS - 0.5, linespacing=1.0)
    for i, a in enumerate(arcs):
        ax.plot(-a.points[:, 1], a.points[:, 0], color=PAL[i], ls=LINESTYLES[i], lw=1.1,
                zorder=3)
        ex, ey = a.end
        th = a.kappa * AM.DEFAULT_LENGTH_M                      # heading at the arc's end
        lx, ly = ex + 0.36 * math.cos(th), ey + 0.36 * math.sin(th)
        lab = "0" if a.kappa == 0 else f"{a.kappa:+.2f}".replace("-", "−")
        if a.kappa != chosen:
            ax.text(-ly, lx, lab, ha="center", va="center", fontsize=FS - 0.5)
    ax.add_patch(Circle((0, 0), AM.CHASSIS_RADIUS_M, fc="white", ec=DARK, lw=0.6, zorder=5))
    ax.add_patch(Polygon([(-0.14, -0.1), (0.14, -0.1), (0, 0.22)], closed=True, fc=DARK,
                         zorder=6))
    half = math.radians(AM.CAM_FOV_DEG / 2)
    for s in (-1, 1):
        ax.plot([0, s * 5 * math.tan(half)], [0, 5], color="0.6", lw=0.5, ls=(0, (2, 2)),
                zorder=0)
    ax.text(4.2, 1.15, "camera\n90° FOV", ha="right", va="top", fontsize=FS - 0.5,
            color="0.35", linespacing=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-0.5, 4.1)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["3", "2", "1", "0", "−1", "−2", "−3"])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("$y$, left positive (m)")
    ax.set_ylabel("$x$, forward (m)")
    ax.set_title("(a) the seven arcs, top-down; labels are $\\kappa$ in m$^{-1}$", loc="left",
                 fontsize=FS)

    # (b) speed of the plan built from the answer
    ax = fig.add_axes([222 / W, 21 / H, 118 / W, 80 / H])
    d = np.linspace(0, 8, 801)
    ramp = np.array([AM.speed_for(x, CRUISE) for x in d])
    ax.axvspan(0, AM.SPEED_CHOICE_FROM_M, fc="0.93", ec="0.7", hatch="////", lw=0.0, zorder=0)
    ax.plot(d, ramp, color=DARK, lw=1.1, zorder=3, label="ramp on $d$")
    lv = np.arange(0, AM.SPEED_LEVELS + 1)
    v_lv = [AM.speed_from_level(int(k), CRUISE) for k in lv]
    ax.plot(np.full_like(lv, 1.5, dtype=float), v_lv, ls="none", marker="s", ms=2.2,
            mfc="white", mec=DARK, mew=0.6, zorder=4, label="model's level 0-10")
    ax.text(0.15, 0.135, "$d\\leq3$ m: model\nnames a speed", ha="left", va="top",
            fontsize=FS - 0.5, linespacing=1.0,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.text(7.9, CRUISE + 0.012, "cruise 0.45", ha="right", va="bottom", fontsize=FS - 0.5)
    ax.text(7.9, AM.CREEP_MPS - 0.012, "creep 0.20", ha="right", va="top", fontsize=FS - 0.5)
    ax.axhline(AM.CREEP_MPS, color="0.6", lw=0.4, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.56)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("TARGET estimate $d$ (m)")
    ax.set_ylabel("plan speed (m/s)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.44), frameon=False, handlelength=1.4,
              borderaxespad=0.2, handletextpad=0.4, fontsize=FS - 0.5)
    ax.set_title("(b) speed from the TARGET line", loc="left", fontsize=FS)
    save(fig, "fig2_menu")


# --------------------------------------------------------------------------------------
# Fig. 4 -- qualitative decisions, decoded from the recorded run videos
# --------------------------------------------------------------------------------------
# (video, seconds into the video). The video frame is 2560x1020: the model's own
# 1280x720 menu view on the left under a 54 px header, the chase camera on the right, and
# the model's text below. The header and text are cropped (they would print below 6 pt)
# and quoted in the LaTeX caption instead. Times were found by indexing one frame per
# decision; the header strip changes whenever a new decision lands.
QUAL = [   # one row of three, so the camera-ready fits 8 pages
    ("20260831-231900_office_hallway_turn__medium.mp4", 34.0, "a"),
    ("20260831-231900_office_hallway_turn__medium.mp4", 43.0, "b"),
    ("20260831-225617_office_nearest_elevator__medium.mp4", 38.5, "c"),
]


def grab(video: str, t: float, tmp: Path) -> np.ndarray:
    out = tmp / f"{Path(video).stem}_{t:.1f}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}",
                    "-i", str(REPO / "nav/results/videos" / video),
                    "-frames:v", "1", str(out)], check=True)
    im = plt.imread(out)
    return im[54:686, 0:1280, :3]           # menu view only; header and watermark removed


def fig_qualitative() -> None:
    with tempfile.TemporaryDirectory() as td:
        ims = [grab(v, t, Path(td)) for v, t, _ in QUAL]
    gap = 3.0
    n = len(ims)
    Wf = TEXTWIDTH_PT                     # one full-width row; each frame ~40 mm wide
    w = (Wf - (n - 1) * gap) / n
    h = w * ims[0].shape[0] / ims[0].shape[1]
    Hf = h
    fig = new_fig(Wf, Hf)
    for k, (im, (_, _, tag)) in enumerate(zip(ims, QUAL)):
        ax = fig.add_axes([k * (w + gap) / Wf, 0.0, w / Wf, 1.0])
        ax.imshow(im, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        ax.text(0.012, 0.975, f"({tag})", transform=ax.transAxes, ha="left", va="top",
                fontsize=7.0, fontweight="bold", color=DARK,
                bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    save(fig, "fig4_qualitative", dpi=300)


# --------------------------------------------------------------------------------------
# Timeline -- one run, five moments: third-person view, the model's menu with the chosen
# arc outlined, and the describe call's own words. The run's decisions.jsonl lived in
# /tmp/qvla-menus and is gone; the quotes are read off the text panel that
# make_run_video.py burnt into the same video from that log, and abridged with "...".
# Times are sim seconds (= decision step / 60); success latched at 56.85 s, 1.496 m.
# --------------------------------------------------------------------------------------
TIMELINE_VIDEO = "20260831-225617_office_nearest_elevator__medium.mp4"
TIMELINE = [   # (video t, label, quoted description, choice, choice colour)
    (2.0, "1.5 s", "“The way ahead is open … TARGET: not visible”",
     "→ straight, 0.45 m/s", DARK),
    (25.0, "25 s", "“a wall and two elevator doors straight ahead … TARGET: 3 m”",
     "→ straight, 0.20 m/s", DARK),
    (38.5, "38.5 s", "“a wall and a dark doorframe straight ahead … far left … "
     "TARGET: not visible”", "→ STOP (= blocked)", "#b3261e"),
    (43.0, "43 s", "+ note “has not moved for about 5 seconds”; “… far left”",
     "→ hard left, κ = +0.60", "#b35900"),
    (50.5, "50 s", "“a dark elevator door straight ahead … TARGET: 2 m”",
     "→ straight, 0.20 m/s\ngoal reached at 56.9 s", "#1e6b2f"),
]


def grab_full(video: str, t: float, tmp: Path) -> np.ndarray:
    out = tmp / f"{Path(video).stem}_{t:.1f}_full.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}",
                    "-i", str(REPO / "nav/results/videos" / video),
                    "-frames:v", "1", str(out)], check=True)
    return plt.imread(out)[54:686, :, :3]   # both views; header and text panel removed


def wrap_to(fig, text: str, width_pt: float, **kw) -> str:
    """Greedy word wrap measured with the real font, so no line exceeds width_pt."""
    r = fig.canvas.get_renderer()
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        t = fig.text(0, 0, trial, **kw)
        wide = t.get_window_extent(renderer=r).width * 72.0 / fig.dpi > width_pt
        t.remove()
        if wide and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return "\n".join(lines + [cur])


def fig_timeline() -> None:
    with tempfile.TemporaryDirectory() as td:
        ims = [grab_full(TIMELINE_VIDEO, t, Path(td)) for t, *_ in TIMELINE]
    n, gap, fs = len(ims), 3.0, 6.0
    W = TEXTWIDTH_PT
    w = (W - (n - 1) * gap) / n
    crop_chase = (60, 612)                   # rows of the 632-row view; ceiling/edge trimmed
    hc = w * (crop_chase[1] - crop_chase[0]) / 1280
    hm = w * 632 / 1280
    lead, label_h, text_lines = 1.2 * fs, 8.0, 5
    H = label_h + hc + 1.5 + hm + 2.0 + text_lines * lead
    fig = new_fig(W, H)
    for k, (im, (_, tl, quote, choice, col)) in enumerate(zip(ims, TIMELINE)):
        x = k * (w + gap)
        y_top = H - label_h
        fig.text((x + 1.0) / W, (H - 1.0) / H, tl, ha="left", va="top", fontsize=6.5,
                 fontweight="bold", color=DARK)
        for y0, h, view in ((y_top - hc, hc, im[crop_chase[0]:crop_chase[1], 1280:2560]),
                            (y_top - hc - 1.5 - hm, hm, im[:, 0:1280])):
            ax = fig.add_axes([x / W, y0 / H, w / W, h / H])
            ax.imshow(view, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
        y_text = y_top - hc - 1.5 - hm - 2.0
        body = wrap_to(fig, quote, w - 1.0, fontsize=fs, style="italic")
        fig.text(x / W, y_text / H, body, ha="left", va="top", fontsize=fs, style="italic",
                 color="0.25", linespacing=1.12)
        nl = body.count("\n") + 1
        r = fig.canvas.get_renderer()
        for i, line in enumerate(choice.split("\n")):
            assert nl + i + 1 <= text_lines, (tl, nl)
            t = fig.text(x / W, (y_text - (nl + i) * lead) / H, line, ha="left", va="top",
                         fontsize=fs, fontweight="bold", color=col)
            assert t.get_window_extent(renderer=r).width * 72.0 / fig.dpi <= w + 0.5, line
    save(fig, "fig_timeline", dpi=300)


# --------------------------------------------------------------------------------------
# Fig. 3 -- (a) instruction authority, (b) per-episode closest approach
# --------------------------------------------------------------------------------------
# (a) Heading separation between "turn left" and "turn right" on the same observation.
# TIC-VLA rows: planned heading, same frame / history / robot state, instruction the only
# change (probe_stop_decision.py and the sweeps recorded in CLAUDE.md); the dagger row
# overwrites the VLM's reasoning text itself (probe_cache_bandwidth.py), which is more
# control than any prompt has. QVLA row: probe_arc_selection.py, mean curvature separation
# 1.190 1/m over 144 trials, as the endpoint bearing of a 3 m arc (chord angle kL/2).
AUTH = [
    ("TIC-VLA, corridor", 0.04),
    ("TIC-VLA, text overwritten†", 17.2),
    ("TIC-VLA, warehouse", 23.4),
    ("TIC-VLA, vending", 35.6),
    ("QVLA (ours)", math.degrees(1.190 * AM.DEFAULT_LENGTH_M / 2.0)),
]
SHORT = {"office_nearest_elevator": "elevator", "hospital_down_hallway": "hallway",
         "hospital_down_hallway2": "hallway2", "office_passing_hallway": "passing",
         "hospital_vending_machine": "vending", "hospital_vending_machine2": "vending2",
         "office_hallway_turn": "turn", "office_hallway_turn2": "turn2",
         "hospital_past_wheelchairs": "wheelchairs", "hospital_forward_staircase": "staircase",
         "hospital_exit_room": "exit room", "warehouse": "warehouse",
         "warehouse_aisle6": "aisle 6"}


CAMPAIGN = "cr"   # nav/tools/run_camera_ready.sh, tags [medium_]cr_<arm>_r<round>
ARMS = [("ticvla", "TIC-VLA", "^", "white", "#2f5f9e", 14),
        ("qvla", "QVLA", "o", "#d9731f", DARK, 11),
        ("straight", "no model", "s", "0.55", "0.25", 9)]


def campaign() -> dict:
    """arm -> round -> episode -> scored row, by run tag (see camera_ready_table.py)."""
    sys.path.insert(0, str(REPO / "nav" / "tools"))
    import camera_ready_table as CRT
    return CRT.collect(CAMPAIGN)


def fig_results() -> None:
    data = campaign()
    W, H = TEXTWIDTH_PT, 122.0
    fig = new_fig(W, H)

    ax = fig.add_axes([96 / W, 38 / H, 72 / W, 60 / H])
    ys = np.arange(len(AUTH))[::-1]
    for y, (lab, val) in zip(ys, AUTH):
        ours = lab.startswith("QVLA")
        ax.barh(y, val, height=0.6, color="#f0a35e" if ours else "#a9c1e0",
                hatch="//////" if ours else "", edgecolor=DARK, lw=0.5)
        ax.text(val + 2.5, y, f"{val:.2g}°" if val < 10 else f"{val:.0f}°",
                va="center", ha="left", fontsize=FS)
    ax.axvline(90, color="0.35", lw=0.5, ls=(0, (2, 2)))
    ax.text(88, len(AUTH) - 0.62, "a 90° turn", ha="right", va="bottom", fontsize=FS - 0.5)
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_ylim(-0.6, len(AUTH) - 0.02)
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in AUTH])
    ax.set_xlabel("left vs. right heading (deg)")
    ax.set_title("(a) authority of the instruction", loc="right", fontsize=FS)

    # (b) successes per episode over the campaign's rounds, one marker per policy,
    # offset sideways so ties stay visible.
    ax = fig.add_axes([206 / W, 38 / H, 136 / W, 60 / H])
    x = np.arange(len(INDOOR))
    n_rounds = max((len(r) for r in data.values()), default=0)
    for k, (arm, name, mk, fc, ec, size) in enumerate(ARMS):
        rounds = data.get(arm, {})
        if not rounds:
            continue
        succ = np.array([sum(bool(rounds[r].get(e, {}).get("success")) for r in rounds)
                         for e in INDOOR])
        mean = np.mean([sum(bool(row["success"]) for row in rounds[r].values())
                        for r in rounds])
        ax.scatter(x + (k - 1) * 0.24, succ, marker=mk, s=size, facecolor=fc, edgecolor=ec,
                   lw=0.6, zorder=3 + k, label=f"{name} {mean:.1f}")
    ax.set_ylim(-0.4, n_rounds + 0.4)
    ax.set_yticks(range(n_rounds + 1))
    for yy in range(n_rounds + 1):
        ax.axhline(yy, color="0.9", lw=0.4, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in INDOOR], rotation=60, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1.0)
    ax.set_xlim(-0.6, len(INDOOR) - 0.4)
    ax.set_ylabel(f"successes in {n_rounds} rounds")
    ax.legend(loc="lower center", frameon=False, handletextpad=0.05, borderaxespad=0.0,
              bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handlelength=1.0)
    ax.set_title("(b) successes per indoor episode", loc="left", fontsize=FS, pad=11)
    save(fig, "fig3_results")


if __name__ == "__main__":
    which = sys.argv[1:] or ["overview", "menu", "timeline", "results"]
    if "overview" in which:
        fig_overview()
    if "menu" in which:
        fig_menu_geometry()
    if "qual" in which:
        fig_qualitative()
    if "timeline" in which:
        fig_timeline()
    if "results" in which:
        fig_results()
