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

# Every field the ENVIRONMENT claims to have set, checked against what the server says it
# actually loaded. It began as the think_level check alone and grew for the same reason
# that one exists: each of these lives in the server, none of them is visible in a result
# file, and every one of them fails by producing a complete and plausible ladder that
# measured something else. The arc SET is checked by name rather than by counting arcs, so
# the sizes stay in `CURVATURE_SETS` and nowhere else.
CHECK="$(printf '%s' "$HEALTH" | EXPECT_LEVEL="$LEVEL" python3 -c '
import json, os, sys
h = json.load(sys.stdin)
want = {"think_level": os.environ["EXPECT_LEVEL"]}
# The menu keys are only on /health when the server is serving a menu at all; under the
# waypoint formats their absence is correct and must not read as a mismatch.
if "arc_set" in h:
    want["arc_set"]     = os.environ.get("QVLA_MENU_ARCS", "coarse").strip().lower()
    want["menu_frames"] = max(1, int(os.environ.get("QVLA_MENU_FRAMES", "1")))
    want["menu_seed"]   = int(os.environ.get("QVLA_MENU_SEED", "0"))
    want["menu_pivots"] = os.environ.get("QVLA_MENU_PIVOTS", "0").lower() in (
        "1", "true", "yes")
    if want["menu_pivots"]:
        want["pivot_deg"] = float(os.environ.get("QVLA_PIVOT_DEG", "15"))
def agrees(got, exp):
    # `pivot_deg` is a float on one side and null on a server without the pivots, so this
    # cannot be a bare subtraction: a missing key has to read as a MISMATCH and not as a
    # TypeError, which would fail the ladder with a traceback instead of a diagnosis.
    if isinstance(exp, float):
        return isinstance(got, (int, float)) and abs(float(got) - exp) < 1e-6
    return got == exp

bad = [f"{k}: server says {h.get(k)!r}, environment asked for {v!r}"
       for k, v in want.items() if not agrees(h.get(k), v)]
print("\n".join(bad) if bad else "OK " + "  ".join(f"{k}={h.get(k)}" for k in want))
' 2>&1)"
case "$CHECK" in
  OK\ *) echo "-- server up, configuration verified: ${CHECK#OK }" ;;
  *) echo "!! the running server is not the one asked for -- refusing to drive:" >&2
     printf '   %s\n' "$CHECK" >&2
     exit 1 ;;
esac
echo

nav/bench.sh ${ONLY:+--only "$ONLY"} --on-episode nav/tools/on_episode.sh
RC=$?
echo "-- ladder ${QVLA_RUN_TAG} finished (rc=$RC)"
exit $RC
