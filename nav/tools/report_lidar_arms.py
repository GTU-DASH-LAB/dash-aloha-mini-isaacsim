#!/usr/bin/env python3
"""Render the lidar A/B ladder as an HTML email, and optionally send it.

    python3 nav/tools/report_lidar_arms.py --out /tmp/report.html
    python3 nav/tools/report_lidar_arms.py --send

Scoring lives in `compare_lidar_arms.pair_arms()` and is imported, never re-derived --
the 25-minute ladder clustering and the refusal to backfill `lidar: ""` are exactly the
rules CLAUDE.md records as failing silently once they exist in two copies. This file is
presentation only. If a number here disagrees with the terminal tool, this file is wrong.

TWO THINGS THE LAYOUT IS DOING ON PURPOSE, both of which a prettier table would lose:

  * The noise floor is at the TOP, above the totals, not in a footnote. Three clean
    prior ladders scored 2/13, 8/13, 8/13 on the same code path, so a reader who sees
    "+2" before seeing that spread has already drawn the wrong conclusion, and no
    caveat further down undoes it.
  * `was` sits to the LEFT of both arms. An episode that has passed before carries no
    information when it flips again; an episode at 0/3 is the only place a single
    ladder can say something a repeat could not. Putting the prior first makes the
    reader weight the row before reading its outcome.

EMAIL HTML IS NOT WEB HTML. Every style is inline: Gmail and Outlook strip or partially
honour `<style>` blocks, and a stylesheet that silently does not apply produces an
unreadable wall of text rather than an error. Layout is tables for the same reason --
flex and grid are unreliable across clients. No external fonts, no images, no scripts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import urllib.request
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nav" / "tools"))

from compare_lidar_arms import pair_arms  # noqa: E402

# One palette, named once. Slightly blue-shifted neutrals rather than pure grey so the
# table rules sit under the ink instead of competing with it.
INK = "#171a1f"
MUTED = "#5d6875"
FAINT = "#8b96a3"
RULE = "#e3e7ec"
BAND = "#f4f6f9"
PAGE = "#ffffff"
WIN = "#136c3c"
LOSS = "#a52218"
WARN_BG = "#fdf6e3"
WARN_INK = "#7a5c12"
WARN_RULE = "#e6d5a8"

SANS = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif")
MONO = ("ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace")


def health(url: str) -> dict | None:
    """The policy server's own counters, read from the far side of the thing measured.

    CLAUDE.md: a 13-episode ladder once printed a plausible 6/13 against a server that
    had served 8 predictions. `predictions` is the number that catches that, and it has
    to come from the server rather than from our own log. None on any failure -- a
    missing counter block is honest, a fabricated one is the failure being guarded.
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def cell(text: str, *, mono: bool = False, bold: bool = False, colour: str = INK,
         align: str = "left", pad: str = "7px 7px", extra: str = "") -> str:
    font = MONO if mono else SANS
    weight = "600" if bold else "400"
    # Numeric cells never wrap. At an email's ~700 px a `1.50 m` breaking after the
    # digits puts the unit on its own line and the column stops reading as a column.
    nowrap = "white-space:nowrap;" if mono else ""
    return (f'<td style="padding:{pad};font-family:{font};font-size:13px;'
            f'font-weight:{weight};color:{colour};text-align:{align};'
            f'border-bottom:1px solid {RULE};{nowrap}{extra}">{text}</td>')


def head_cell(text: str, align: str = "left") -> str:
    return (f'<th style="padding:6px 7px;font-family:{SANS};font-size:10px;'
            f'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;'
            f'color:{FAINT};text-align:{align};border-bottom:2px solid {RULE};'
            f'white-space:nowrap;">{escape(text)}</th>')


def signed(v: float, places: int = 0) -> str:
    """`+3`, `-2`, and a bare `0` — never `-0`, which reads as a rounding bug."""
    if abs(v) < 0.5 / (10 ** places):
        return "0" if places == 0 else f"{0:.{places}f}"
    return f"{v:+.{places}f}"


def section(title: str, sub: str = "") -> str:
    out = (f'<h2 style="margin:32px 0 4px;font-family:{SANS};font-size:15px;'
           f'font-weight:700;color:{INK};letter-spacing:-0.01em;">{escape(title)}</h2>')
    if sub:
        out += (f'<p style="margin:0 0 12px;font-family:{SANS};font-size:12.5px;'
                f'line-height:1.55;color:{MUTED};">{sub}</p>')
    return out


def para(text: str, colour: str = MUTED, size: str = "12.5px") -> str:
    return (f'<p style="margin:0 0 10px;font-family:{SANS};font-size:{size};'
            f'line-height:1.6;color:{colour};">{text}</p>')


def build(c, counters: dict | None, when: str) -> str:
    t = c.totals()
    delta = t["b_pass"] - t["a_pass"]
    dcolour = WIN if delta > 0 else LOSS if delta < 0 else MUTED

    P: list[str] = []
    P.append(
        f'<div style="background:{PAGE};padding:28px 30px 36px;max-width:760px;'
        f'margin:0 auto;">'
    )

    # --- header ---------------------------------------------------------------------
    P.append(
        f'<p style="margin:0 0 2px;font-family:{SANS};font-size:10px;font-weight:600;'
        f'letter-spacing:0.1em;text-transform:uppercase;color:{FAINT};">'
        f'dash-aloha-mini-isaacsim &nbsp;·&nbsp; nav benchmark</p>'
        f'<h1 style="margin:0 0 6px;font-family:{SANS};font-size:23px;font-weight:700;'
        f'letter-spacing:-0.02em;color:{INK};">2D lidar A/B — {t["n"]} episodes, '
        f'paired</h1>'
        f'<p style="margin:0 0 22px;font-family:{SANS};font-size:12.5px;color:{MUTED};">'
        f'{escape(when)} &nbsp;·&nbsp; controller <code style="font-family:{MONO};">'
        f'{escape(c.controller)}</code> &nbsp;·&nbsp; both arms re-run fresh, same '
        f'server, same policy</p>'
    )

    # --- headline -------------------------------------------------------------------
    P.append(
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
        f'style="border-collapse:collapse;background:{BAND};border:1px solid {RULE};'
        f'margin:0 0 18px;"><tr>'
    )
    for label, val, sub in (
        (f"A · {c.a_label}", f'{t["a_pass"]}/{t["n"]}', "7-ray fan, reference"),
        (f"B · {c.b_label}", f'{t["b_pass"]}/{t["n"]}', "RPLIDAR C1, 500 pts/rev"),
        ("difference", f'{delta:+d}', "read the band below first"),
    ):
        colour = dcolour if label == "difference" else INK
        P.append(
            f'<td style="padding:16px 18px;border-right:1px solid {RULE};width:33%;'
            f'vertical-align:top;">'
            f'<div style="font-family:{SANS};font-size:10px;font-weight:600;'
            f'letter-spacing:0.08em;text-transform:uppercase;color:{FAINT};'
            f'margin-bottom:5px;">{escape(label)}</div>'
            f'<div style="font-family:{MONO};font-size:27px;font-weight:700;'
            f'color:{colour};line-height:1.1;">{val}</div>'
            f'<div style="font-family:{SANS};font-size:11px;color:{FAINT};'
            f'margin-top:4px;">{escape(sub)}</div></td>'
        )
    P.append("</tr></table>")

    # --- the caveat, above the table and not below it -------------------------------
    if c.history:
        P.append(
            f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
            f'style="border-collapse:collapse;margin:0 0 8px;"><tr>'
            f'<td style="background:{WARN_BG};border:1px solid {WARN_RULE};'
            f'border-left:3px solid {WARN_INK};padding:13px 16px;font-family:{SANS};'
            f'font-size:12.5px;line-height:1.6;color:{WARN_INK};">'
            f'<b>Read this before the totals.</b> Three clean single-pass ladders '
            f'recorded before the sensor existed scored '
            f'<b>{", ".join(c.prior_scores())}</b> on the same stack. That spread is the '
            f'noise floor: <code style="font-family:{MONO};">predict()</code> is '
            f'deterministic but a <i>run</i> is not — which KV cache is ready at which '
            f'step depends on wall-clock generation timing, and the same episode with '
            f'the same sentence has succeeded twice and failed once. '
            f'<b>A one- or two-episode move in either direction is not a result.</b>'
            f'</td></tr></table>'
        )

    # --- the paired table ------------------------------------------------------------
    P.append(section(
        "Episode by episode",
        'Paired, never pooled. "9/19 against 8/19" is equally consistent with nothing '
        'changing and with nine episodes flipping each way; only the join names which '
        'ones moved. Read <b>flip</b> first and <b>closed</b> second — arrival is a hard '
        'radius, so a run can improve a long way without flipping, or flip on 8&nbsp;cm.'
    ))
    # Nine columns is more than a phone has. Clients that honour overflow scroll it;
    # the ones that do not shrink the table, which is the same outcome they would have
    # reached anyway -- what neither does is push the page body sideways.
    P.append('<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">')
    P.append(
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
        f'style="border-collapse:collapse;min-width:640px;">'
        f'<tr>{head_cell("episode")}{head_cell("goal ≤", "center")}'
        f'{head_cell("was", "center")}'
        f'{head_cell("A " + c.a_label, "center")}{head_cell("B " + c.b_label, "center")}'
        f'{head_cell("flip", "center")}{head_cell("closed A", "right")}'
        f'{head_cell("closed B", "right")}{head_cell("Δ", "right")}</tr>'
    )
    for i, ep in enumerate(c.order):
        a, b = c.arm_a[ep], c.arm_b[ep]
        fl = c.flip(ep)
        zebra = f"background:{BAND};" if i % 2 else ""
        w = c.was(ep)
        # An episode at 0/N is the only row a single ladder can inform. Marked in the
        # `was` column itself so it is visible while reading, not only in the list below.
        was_txt = f"{w[0]}/{w[1]}" if w else "—"
        was_col = WARN_INK if w and w[0] == 0 else FAINT
        ca, cb = a["closed_frac"] * 100, b["closed_frac"] * 100
        d = cb - ca

        def verdict(row: dict) -> str:
            ok = row["success"]
            mark = "✓" if ok else "✗"
            col = WIN if ok else LOSS
            return (f'<span style="color:{col};font-weight:700;">{mark}</span>'
                    f'<span style="color:{MUTED};"> {row["closest_m"]:.2f} m</span>')

        flip_html = {
            "WON": f'<span style="color:{WIN};font-weight:700;">WON</span>',
            "LOST": f'<span style="color:{LOSS};font-weight:700;">LOST</span>',
        }.get(fl, f'<span style="color:{FAINT};">=</span>')

        # The arrival radius is not 1.5 m everywhere -- the outdoor set overrides it both
        # up (4.0 m) and down (1.0 m). Shown per row because otherwise a `3.99 m` marked
        # pass sitting under a `1.89 m` marked fail reads as a scoring error.
        thr = a["threshold_m"]
        P.append(
            f"<tr>"
            + cell(escape(ep), extra=zebra + "white-space:nowrap;")
            + cell(f"{thr:.1f} m", mono=True,
                   colour=INK if abs(thr - 1.5) > 1e-6 else FAINT,
                   align="center", extra=zebra)
            + cell(was_txt, mono=True, colour=was_col, align="center", extra=zebra)
            + cell(verdict(a), mono=True, align="center", extra=zebra)
            + cell(verdict(b), mono=True, align="center", extra=zebra)
            + cell(flip_html, align="center", extra=zebra)
            + cell(f"{ca:.0f}%", mono=True, colour=MUTED, align="right", extra=zebra)
            + cell(f"{cb:.0f}%", mono=True, colour=MUTED, align="right", extra=zebra)
            + cell(signed(d), mono=True, align="right",
                   colour=WIN if d > 1 else LOSS if d < -1 else FAINT, extra=zebra)
            + "</tr>"
        )
    P.append(
        f"<tr>"
        + cell("TOTAL", bold=True, extra=f"border-top:2px solid {RULE};")
        + cell("", extra=f"border-top:2px solid {RULE};")
        + cell("", extra=f"border-top:2px solid {RULE};")
        + cell(f'{t["a_pass"]} / {t["n"]}', mono=True, bold=True, align="center",
               extra=f"border-top:2px solid {RULE};")
        + cell(f'{t["b_pass"]} / {t["n"]}', mono=True, bold=True, align="center",
               extra=f"border-top:2px solid {RULE};")
        + cell(f'<span style="color:{dcolour};font-weight:700;">{signed(delta)}</span>',
               align="center", extra=f"border-top:2px solid {RULE};")
        + cell(f'{t["a_closed"]:.0f}%', mono=True, bold=True, align="right",
               extra=f"border-top:2px solid {RULE};")
        + cell(f'{t["b_closed"]:.0f}%', mono=True, bold=True, align="right",
               extra=f"border-top:2px solid {RULE};")
        + cell(signed(t["b_closed"] - t["a_closed"]), mono=True, bold=True,
               align="right", extra=f"border-top:2px solid {RULE};")
        + "</tr></table></div>"
    )
    P.append(
        f'<p style="margin:8px 0 0;font-family:{SANS};font-size:11.5px;color:{FAINT};'
        f'line-height:1.55;"><b>goal ≤</b> = that episode\'s own arrival radius, which '
        f'is DynaNav\'s and is not 1.5&nbsp;m everywhere — the outdoor set overrides it '
        f'up to 4.0&nbsp;m and down to 1.0&nbsp;m, so a 3.99&nbsp;m pass and a '
        f'1.89&nbsp;m fail can both be correct. '
        f'<b>was</b> = passes over the '
        f'{len(c.history)} clean single-pass ladders from before the sensor existed. '
        f'<b>closed</b> = fraction of the initial goal distance the robot actually '
        f'closed; it moves when the pass/fail threshold does not. Distances shown are '
        f'<b>closest approach</b>, not final — a robot that reaches 2.8&nbsp;m and then '
        f'drives away ends at 22&nbsp;m, and those are opposite failures.</p>'
    )

    won, lost = c.won(), c.lost()
    P.append(
        f'<p style="margin:14px 0 0;font-family:{SANS};font-size:12.5px;'
        f'line-height:1.7;color:{INK};">'
        f'<span style="color:{WIN};font-weight:700;">won ({len(won)})</span> '
        f'<code style="font-family:{MONO};font-size:12px;color:{MUTED};">'
        f'{escape(", ".join(won)) or "—"}</code><br>'
        f'<span style="color:{LOSS};font-weight:700;">lost ({len(lost)})</span> '
        f'<code style="font-family:{MONO};font-size:12px;color:{MUTED};">'
        f'{escape(", ".join(lost)) or "—"}</code></p>'
    )

    # --- the rows that can actually carry information --------------------------------
    if c.never:
        P.append(section(
            "The only rows one ladder can settle",
            "Everywhere else the prior already contains both outcomes, so a flip there "
            "sits inside the noise named above. These episodes have <b>never passed on "
            "any prior ladder</b>, so a pass here is the one result a repeat could not "
            "have produced by chance."
        ))
        P.append(
            f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
            f'style="border-collapse:collapse;">'
            f'<tr>{head_cell("episode")}{head_cell("prior", "center")}'
            f'{head_cell("A " + c.a_label, "center")}'
            f'{head_cell("B " + c.b_label, "center")}{head_cell("now")}</tr>'
        )
        for ep in c.never:
            aa, bb = c.arm_a[ep]["success"], c.arm_b[ep]["success"]
            w = c.was(ep)
            verdict, vcol = (
                ("passes on BOTH arms", WIN) if aa and bb else
                (f"passes on {c.b_label} ONLY — the lidar arm", WIN) if bb else
                (f"passes on {c.a_label} only — the reference arm", MUTED) if aa else
                ("still fails on both", LOSS)
            )
            P.append(
                "<tr>"
                + cell(escape(ep), bold=True, extra="white-space:nowrap;")
                + cell(f"0/{w[1]}" if w else "—", mono=True, colour=WARN_INK,
                       align="center")
                + cell(f'{"✓" if aa else "✗"} {c.arm_a[ep]["closest_m"]:.2f} m',
                       mono=True, colour=WIN if aa else LOSS, align="center")
                + cell(f'{"✓" if bb else "✗"} {c.arm_b[ep]["closest_m"]:.2f} m',
                       mono=True, colour=WIN if bb else LOSS, align="center")
                + cell(escape(verdict), colour=vcol, bold=True)
                + "</tr>"
            )
        P.append("</table>")

    # --- aggregates that speak about the sensor rather than the policy ---------------
    P.append(section(
        "Aggregates",
        "<b>Guard interventions is the one number here that does not pass through the "
        "VLM</b> — the guard is the half of this change that is pure geometry. More "
        "returns should mean <i>more</i> interventions; a large drop would mean the "
        "robot stopped reaching the obstacles rather than that it stopped hitting them. "
        "Read path length next to <b>closed</b> and never alone: a short path is either "
        "a tidy route or a run that parked, and <code style=\"font-family:" + MONO
        + ";\">warehouse_aisle6</code> has produced both."
    ))
    P.append(
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
        f'style="border-collapse:collapse;">'
        f'<tr>{head_cell("")}{head_cell("A " + c.a_label, "right")}'
        f'{head_cell("B " + c.b_label, "right")}{head_cell("Δ", "right")}</tr>'
    )
    for label, av, bv, fmt, better in (
        ("guard interventions", t["a_guard"], t["b_guard"], "{:.0f}", None),
        ("path driven (m)", t["a_path"], t["b_path"], "{:.0f}", None),
        ("mean SPL", t["a_spl"], t["b_spl"], "{:.2f}", "up"),
        ("mean closed (%)", t["a_closed"], t["b_closed"], "{:.0f}", "up"),
    ):
        d = bv - av
        dcol = FAINT if better is None else (WIN if d > 0 else LOSS if d < 0 else FAINT)
        P.append(
            "<tr>"
            + cell(escape(label), colour=MUTED)
            + cell(fmt.format(av), mono=True, align="right")
            + cell(fmt.format(bv), mono=True, align="right")
            + cell(signed(d, 2 if fmt.endswith("2f") else 0),
                   mono=True, colour=dcol, align="right")
            + "</tr>"
        )
    P.append("</table>")

    # --- what was actually wired, so the reader can judge the claim ------------------
    P.append(section(
        "What the sensor is wired into",
        "One <code style=\"font-family:" + MONO + ";\">SweepingLidar2D</code>, so both "
        "consumers see the same returns taken at the same instants. Arm A leaves it off "
        "entirely, which is what keeps every earlier ladder number comparable."
    ))
    P.append(
        f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
        f'style="border-collapse:collapse;border:1px solid {RULE};"><tr>'
        f'<td style="padding:14px 16px;width:50%;vertical-align:top;'
        f'border-right:1px solid {RULE};font-family:{SANS};font-size:12.5px;'
        f'line-height:1.6;color:{MUTED};">'
        f'<b style="color:{INK};">Low loop — collision guard</b><br>'
        f'The fan is 7 rays over ±35°, 11.7° apart: a <b>12 cm gap at the 0.6 m stop '
        f'distance</b>, 30 cm at 1.5 m. A chair leg fits through it. The C1\'s 0.72° '
        f'closes those to 0.8 cm and 1.9 cm. Cost is staleness — the beam revisits a '
        f'bearing every 100 ms — which the raised 0.75 m stop distance absorbs.'
        f'</td>'
        f'<td style="padding:14px 16px;width:50%;vertical-align:top;'
        f'font-family:{SANS};font-size:12.5px;line-height:1.6;color:{MUTED};">'
        f'<b style="color:{INK};">High loop — VLM arc menu</b><br>'
        f'Per-arc clearance drops arcs that end in a wall before the model is asked to '
        f'choose. Deliberately <i>not</i> the "openings outside the frame" channel, '
        f'which the pre-build gate measured at <b>−11 pp</b> and which is not shipped. '
        f'Gated on ≥50 points so an empty buffer records <i>null</i> rather than a '
        f'plausible all-clear row.'
        f'</td></tr></table>'
    )

    # --- unpaired, never silently dropped -------------------------------------------
    if c.only_a or c.only_b:
        rows = []
        if c.only_a:
            rows.append(f"only in <b>{escape(c.a_label)}</b>: "
                        f"{escape(', '.join(c.only_a))}")
        if c.only_b:
            rows.append(f"only in <b>{escape(c.b_label)}</b>: "
                        f"{escape(', '.join(c.only_b))}")
        P.append(
            f'<table role="presentation" cellpadding="0" cellspacing="0" width="100%" '
            f'style="border-collapse:collapse;margin:24px 0 0;"><tr>'
            f'<td style="background:{WARN_BG};border:1px solid {WARN_RULE};'
            f'border-left:3px solid {LOSS};padding:13px 16px;font-family:{SANS};'
            f'font-size:12.5px;line-height:1.6;color:{WARN_INK};">'
            f'<b>Not paired — excluded from every number above.</b> A missing run is '
            f'usually a crash rather than a result.<br>' + "<br>".join(rows)
            + '</td></tr></table>'
        )

    # --- the far-side counter --------------------------------------------------------
    if counters:
        bits = " &nbsp;·&nbsp; ".join(
            f'<b style="color:{INK};">{counters[k]}</b> {k}'
            for k in ("predictions", "generations", "generation_errors",
                      "parse_failures")
            if k in counters
        )
        P.append(section(
            "Policy server counters",
            "Read from the far side of the thing being measured. This ladder's "
            "ancestor once printed a plausible 6/13 against a server that had served "
            "8 predictions; this line is the check that catches it."
        ))
        P.append(para(bits + " &nbsp;(cumulative over both arms — the server was "
                      "relaunched before arm A and not restarted between them)"))

    P.append(
        f'<p style="margin:34px 0 0;padding-top:14px;border-top:1px solid {RULE};'
        f'font-family:{MONO};font-size:11px;color:{FAINT};line-height:1.6;">'
        f'nav/tools/report_lidar_arms.py &nbsp;·&nbsp; scoring imported from '
        f'compare_lidar_arms.pair_arms() &nbsp;·&nbsp; reproduce with<br>'
        f'python3 nav/tools/compare_lidar_arms.py --history</p>'
    )
    P.append("</div>")
    return "".join(P)


def plain(c, when: str) -> str:
    """The text/plain fallback. Says what happened on its own, per notify_run's rule."""
    t = c.totals()
    lines = [
        f"2D lidar A/B ladder, {t['n']} paired episodes, controller {c.controller}",
        when,
        "",
        f"  A {c.a_label:<10} {t['a_pass']}/{t['n']}   mean closed {t['a_closed']:.0f}%",
        f"  B {c.b_label:<10} {t['b_pass']}/{t['n']}   mean closed {t['b_closed']:.0f}%",
        f"  difference       {t['b_pass'] - t['a_pass']:+d}",
        "",
        f"  won  ({len(c.won())}): {', '.join(c.won()) or '-'}",
        f"  lost ({len(c.lost())}): {', '.join(c.lost()) or '-'}",
    ]
    if c.history:
        lines += [
            "",
            f"  NOISE FLOOR: three prior clean ladders scored "
            f"{', '.join(c.prior_scores())} on the same stack.",
            "  A one- or two-episode move is not a result.",
        ]
    if c.never:
        lines += ["", "  Never passed before (the only rows one ladder can settle):"]
        for ep in c.never:
            aa, bb = c.arm_a[ep]["success"], c.arm_b[ep]["success"]
            lines.append(f"    {ep:<26} A {'pass' if aa else 'fail'}   "
                         f"B {'pass' if bb else 'fail'}")
    lines += ["", "Full table in the HTML part, or:",
              "  python3 nav/tools/compare_lidar_arms.py --history"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="fan")
    ap.add_argument("--b", default="c1@0.30")
    ap.add_argument("--controller", default="braking")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/lidar_ab_report.html"))
    ap.add_argument("--health", default="http://127.0.0.1:8766/health")
    ap.add_argument("--send", action="store_true", help="mail it via notify_run.py")
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()

    c = pair_arms(args.a, args.b, args.controller, history=True)
    when = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    body = build(c, health(args.health), when)
    args.out.write_text(body)
    print(f"wrote {args.out} ({len(body) / 1024:.1f} KB)")

    t = c.totals()
    subject = args.subject or (
        f"Lidar A/B: {c.a_label} {t['a_pass']}/{t['n']} vs {c.b_label} "
        f"{t['b_pass']}/{t['n']} ({t['b_pass'] - t['a_pass']:+d})"
    )
    if args.send:
        # Through the CLI rather than by importing send(): the key-handling and the
        # User-Agent that Cloudflare requires live there, and there must not be a
        # second copy of either.
        subprocess.run(
            [sys.executable, str(REPO / "nav/tools/notify_run.py"),
             "--subject", subject, "--body", plain(c, when),
             "--html", str(args.out)],
            check=True,
        )
    else:
        print(f"subject would be: {subject}\n(pass --send to mail it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
