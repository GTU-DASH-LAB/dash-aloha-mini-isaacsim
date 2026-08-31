#!/usr/bin/env bash
# Mail the ladder's status once an hour until the ladder is gone, then mail once more.
#
#   setsid nohup nav/tools/hourly_report.sh > /tmp/hourly.log 2>&1 &
#
# Separate from the per-episode hook on purpose. The hook fires on an EVENT and an
# episode can take twenty minutes or can wedge and take an hour, so a hook alone gives no
# way to tell "still working" from "died quietly" -- which is the whole question when
# nobody is at the machine. This fires on the CLOCK, so silence from it means the
# reporter itself is gone rather than the run being slow.
#
# It also sends the last word. A ladder that finishes at 04:10 should not wait until
# somebody wakes up and asks.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
INTERVAL="${1:-3600}"
export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"

# `ps -eo args | grep -c '[b]ench.sh'` and not `pgrep -f bench.sh`: pgrep -f matches the
# pattern against THIS script's own command line too, so a wait loop written that way
# never ends. The bracket is what keeps grep from finding its own argument.
running() { [ "$(ps -eo args | grep -c '[b]ench\.sh')" -gt 0 ]; }

n=0
while :; do
  # Sleep FIRST. This starts right after the ladder does, and a status mail sent before
  # the first episode has produced anything is a mail that says nothing.
  sleep "$INTERVAL"
  n=$((n + 1))
  BODY="$(python3 nav/tools/bench_status.py 2>&1)"
  if running; then
    python3 nav/tools/notify_run.py \
      --subject "[qvla ladder] hourly update #$n -- $(date +%H:%M)" --body "$BODY"
  else
    python3 nav/tools/notify_run.py \
      --subject "[qvla ladder] FINISHED -- $(date +%H:%M)" \
      --body "The benchmark process is no longer running.

$BODY

Videos: $REPO/nav/results/videos
Logs:   $REPO/nav/results/logs"
    exit 0
  fi
done
