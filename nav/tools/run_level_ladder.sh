#!/usr/bin/env bash
# Run one benchmark ladder at one thinking level, server restart included.
#
#   nav/tools/run_level_ladder.sh high                       # every episode
#   nav/tools/run_level_ladder.sh very_high --only ep1,ep2   # a subset
#   nav/tools/run_level_ladder.sh medium --tag rest          # tag `medium_rest`
#
# The level lives in the SERVER, not in the runner, because the horizon it drags with
# it changes how a plan is sliced -- so every level costs a full model reload. Three
# levels is therefore three restarts, and doing them by hand is three chances to run a
# ladder against a server still holding the previous level. That failure is silent: it
# produces thirteen complete, plausible numbers measuring the wrong configuration.
#
# So this script does the whole cycle and, before it starts driving, reads `think_level`
# back off /health and REFUSES to run if it is not the level asked for. The check is
# against the server's own report, not against what was passed to the launcher.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

LEVEL="${1:?usage: run_level_ladder.sh <medium|high|very_high> [--only eps] [--tag suffix]}"
shift
ONLY=""; SUFFIX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --tag)  SUFFIX="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"
export QVLA_RUN_TAG="${LEVEL}${SUFFIX:+_$SUFFIX}"
LOGDIR="$REPO/nav/results/logs"
mkdir -p "$LOGDIR"

# Match on the interpreter path AND the script, via awk rather than `pgrep -f`. `pgrep -f`
# matches the pattern against the invoking shell's own command line, so a pattern that
# appears in this file can find this file -- that has cost three separate sessions here.
# awk's own command line is `awk {...}`, which cannot match either field.
qwen_pids() {
  ps -eo pid,args | awk '$2 ~ /qvla\/bin\/python$/ && $3 == "server_qwen.py" { print $1 }'
}

for pid in $(qwen_pids); do
  echo "-- stopping policy server $pid"
  kill "$pid" 2>/dev/null
done
for _ in $(seq 1 60); do
  [ -z "$(qwen_pids)" ] && break
  command sleep 1
done
if [ -n "$(qwen_pids)" ]; then
  echo "!! a policy server is still alive; refusing to load a second 28 GiB model" >&2
  exit 1
fi

echo "-- starting policy server at think level: $LEVEL"
nohup nav/policy_server/launch_qwen.sh "$LEVEL" \
    > "$LOGDIR/server_qwen_${LEVEL}.log" 2>&1 &
SERVER_PID=$!

HEALTH=""
for _ in $(seq 1 90); do
  HEALTH="$(curl -sf --max-time 5 "http://127.0.0.1:${NAV_POLICY_PORT}/health" || true)"
  [ -n "$HEALTH" ] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!! policy server died on startup:" >&2
    tail -25 "$LOGDIR/server_qwen_${LEVEL}.log" >&2
    exit 1
  fi
  command sleep 5
done
if [ -z "$HEALTH" ]; then
  echo "!! policy server never answered on :${NAV_POLICY_PORT}" >&2
  exit 1
fi

GOT="$(printf '%s' "$HEALTH" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("think_level",""))')"
if [ "$GOT" != "$LEVEL" ]; then
  echo "!! server reports think_level=$GOT, asked for $LEVEL -- refusing to run" >&2
  exit 1
fi
echo "-- server up and reporting think_level=$GOT"
echo

nav/bench.sh ${ONLY:+--only "$ONLY"} --on-episode nav/tools/on_episode.sh
RC=$?
echo "-- ladder ${QVLA_RUN_TAG} finished (rc=$RC)"
exit $RC
