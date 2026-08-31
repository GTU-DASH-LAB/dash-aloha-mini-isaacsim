#!/usr/bin/env bash
# Mail the ladder's status once an hour until the ladder is gone, then mail once more.
#
#   setsid nohup nav/tools/hourly_report.sh [interval_s] [bench_pid] > /tmp/hourly.log 2>&1 &
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
BENCH_PID="${2:-}"
export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"

# Given the ladder's pid, ask about that process and nothing else. The name-matching
# fallback below is correct but not exact, and the inexactness runs one way: any shell
# whose command line merely MENTIONS bench.sh counts, including the interactive one that
# launched this. That only ever delays the finished-mail by an interval, so it is a fine
# fallback and a poor default.
#
# The fallback is `ps -eo args | grep -c '[b]ench.sh'` and not `pgrep -f bench.sh`,
# because pgrep -f matches the pattern against the invoking shell's own command line, so
# a wait loop written that way never ends. The bracket keeps grep from finding its own
# argument for the same reason.
running() {
  if [ -n "$BENCH_PID" ]; then
    kill -0 "$BENCH_PID" 2>/dev/null
  else
    [ "$(ps -eo args | grep -c '^bash .*[b]ench\.sh')" -gt 0 ]
  fi
}

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
