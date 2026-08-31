"""Email a benchmark update, optionally with the run's video attached.

A 13-episode ladder is roughly two hours of nobody watching, and the failure that costs
the most is the silent one: the policy server dies at episode 3 and the remaining ten
"run" in seconds against a dead port. So this exists to put the numbers somewhere they
will be seen while the ladder is still running, rather than in a log read afterwards.

Reuses `~/.config/resend/api_key` (chmod 600) exactly as `~/robotics/scripts/
notify-progress.sh` does -- same key, same sender, same recipient. It is a separate file
rather than a flag on that script because that one is the machine-setup notifier shared
with the rest of the robotics tree, and this one carries attachments and benchmark
vocabulary that have no business there.

ATTACHMENTS ARE CAPPED AND THE CAP IS NOT COSMETIC. Resend rejects a request whose total
body exceeds 40 MB, and the rejection arrives as a 4xx after the whole payload has been
uploaded -- so an oversized attachment costs the upload AND loses the message. Anything
above `MAX_ATTACH_MB` is named in the body by path instead of being sent.

Usage:
    python3 nav/tools/notify_run.py --subject "..." --body "..." [--attach video.mp4]
    ... | python3 nav/tools/notify_run.py --subject "..."          # body on stdin
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

KEYFILE = Path.home() / ".config/resend/api_key"
TO = os.environ.get("RESEND_TO", "fouadiadhami@gmail.com")
FROM = os.environ.get("RESEND_FROM", "onboarding@resend.dev")
API = "https://api.resend.com/emails"
# Base64 inflates by 4/3, so a 12 MB file is ~16 MB on the wire and two of them would not
# fit. One video per message, well inside the limit.
MAX_ATTACH_MB = 12.0


def send(subject: str, body: str, attach: list[Path] | None = None) -> bool:
    """True if Resend accepted it. Never raises -- a failed email must not fail a run."""
    if not KEYFILE.is_file() or not KEYFILE.read_text().strip():
        print(f"[notify] no API key at {KEYFILE}", file=sys.stderr)
        return False

    payload: dict = {"from": FROM, "to": [TO], "subject": subject, "text": body}
    skipped: list[str] = []
    files = []
    for p in attach or []:
        if not p.is_file():
            skipped.append(f"{p} (missing)")
            continue
        mb = p.stat().st_size / 1e6
        if mb > MAX_ATTACH_MB:
            skipped.append(f"{p} ({mb:.1f} MB, over the {MAX_ATTACH_MB:.0f} MB cap)")
            continue
        files.append({"filename": p.name,
                      "content": base64.b64encode(p.read_bytes()).decode()})
    if skipped:
        # Said in the body, not swallowed. "The video was too big" and "there was no
        # video" look identical from an inbox and mean different things about the run.
        payload["text"] += "\n\nnot attached:\n  " + "\n  ".join(skipped)
    if files:
        payload["attachments"] = files

    # The User-Agent is load-bearing, which is not obvious. Resend sits behind Cloudflare
    # and a default `Python-urllib/3.x` is refused with HTTP 403 "error code: 1010" --
    # a ban on the client string, not on the key, and the message says nothing about
    # either. `scripts/notify-progress.sh` never hit it because curl sends its own.
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEYFILE.read_text().strip()}",
                 "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"[notify] sent: {subject}"
                  + (f" (+{len(files)} attachment)" if files else ""))
            return resp.status < 300
    except urllib.error.HTTPError as exc:
        print(f"[notify] HTTP {exc.code}: {exc.read()[:300]!r}", file=sys.stderr)
    except Exception as exc:
        print(f"[notify] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", default=None, help="omit to read the body from stdin")
    ap.add_argument("--attach", action="append", default=[], type=Path)
    args = ap.parse_args()
    body = args.body if args.body is not None else sys.stdin.read()
    # 0 even on failure: this is called from a benchmark hook, and `set -e` turning a
    # bounced email into an aborted 13-episode ladder is a worse outcome than a missing
    # message. The stderr line above is the record.
    send(args.subject, body, args.attach)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
