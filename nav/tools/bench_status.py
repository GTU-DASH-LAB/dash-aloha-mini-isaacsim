"""Print what the ladder is doing right now, in one screen of text.

Written for the hourly mail during an unattended run, so it answers the three questions
you actually have at 3 a.m. in that order: is it still moving, what has it scored so far,
and is the thing it depends on still alive.

THE LAST ONE IS NOT PADDING. The expensive failure on this stack is not a crash -- it is
the policy server dying at episode 3 while `bench.sh` cheerfully runs the remaining ten
against a dead port in seconds each, producing a full set of plausible zeros. So the
server's own counters are printed next to the episode count: `generations` climbing is
the only proof from the far side of the measurement that anything is being decided.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROGRESS = REPO / "nav/results/ladder_progress.log"
MENUS = Path(os.environ.get("QVLA_MENU_DIR", "/tmp/qvla-menus"))
PORT = os.environ.get("NAV_POLICY_PORT", "8766")


def _health() -> dict | None:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def _ladder() -> list[str]:
    import yaml
    cfg = yaml.safe_load((REPO / "nav/config/episodes.yaml").read_text())
    return list(cfg["episodes"])


def status() -> str:
    out: list[str] = []
    done: list[tuple[str, str, str]] = []
    if PROGRESS.is_file():
        for line in PROGRESS.read_text().splitlines():
            t, e, c, v = (line.split("\t") + ["", "", ""])[:4]
            done.append((t, e, v))

    try:
        names = _ladder()
    except Exception:
        names = []
    ok = sum(v.startswith("SUCCESS") for _, _, v in done)
    out.append(f"{len(done)} of {len(names) or '?'} episodes scored, {ok} succeeded")

    # The run directory the server is writing into right now. `decisions.jsonl` growing
    # between two hourly mails is the difference between "slow episode" and "wedged".
    live = sorted(MENUS.glob("*/"), key=lambda p: p.stat().st_mtime)
    if live:
        d = live[-1]
        age = time.time() - d.stat().st_mtime
        n = sum(1 for _ in (d / "decisions.jsonl").open()) if (
            d / "decisions.jsonl").is_file() else 0
        out += ["", f"running: {d.name}",
                f"  {n} decisions so far, last written {age / 60:.1f} min ago"]
        if (d / "decisions.jsonl").is_file() and n:
            last = json.loads((d / "decisions.jsonl").read_text().splitlines()[-1])
            out += [f"  step {last.get('step')}  ->  "
                    + ("STOP" if last.get("stop") else f"path {last.get('choice')}"
                       f"  kappa {last.get('kappa')}")
                    + ("   [after reversing out]" if last.get("recovered") else ""),
                    f"  it says: {str(last.get('free_space', ''))[:300]}"]

    h = _health()
    out += ["", "policy server:"]
    if h and "__error__" in h:
        out.append(f"  UNREACHABLE on :{PORT} -- {h['__error__']}")
        out.append("  (every episode from here on will score without deciding anything)")
    elif h:
        out.append("  " + "  ".join(
            f"{k}={v}" for k, v in h.items()
            if k in ("format", "predictions", "generations", "parse_failures",
                     "gen_errors", "stale_discards", "menu_speed_mps", "has_plan")))

    if done:
        out += ["", "scored so far:"]
        out += [f"  {t[11:19]}  {e:<34} {v}" for t, e, v in done]
    if names:
        left = [n for n in names if n not in {e for _, e, _ in done}]
        out += ["", f"still to run ({len(left)}): " + ", ".join(left)]
    return "\n".join(out)


if __name__ == "__main__":
    print(status())
    sys.exit(0)
