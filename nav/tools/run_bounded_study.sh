#!/usr/bin/env bash
# The bounded-async arm of the planning study: is a CAPPED blind window enough?
#
#   setsid nohup nav/tools/run_bounded_study.sh > /tmp/bounded_study.log 2>&1 &
#   nav/tools/run_bounded_study.sh --only hospital_exit_room     # smoke test one episode
#   nav/tools/run_bounded_study.sh --skip-preflight              # server already proven
#
# THE QUESTION, and it is not the same one `run_sync_study.sh` asks. That campaign
# removed the blind window entirely -- the robot stops dead at every decision -- which
# makes "does more thinking help?" answerable but costs a full stop per decision. This
# one asks whether the stop was necessary at all, or whether merely BOUNDING the window
# at one planning period recovers the same safety at a fraction of the price.
#
# The mechanism is a pipeline: at call k the server hands back the plan generated from
# call (k-1)'s images and immediately starts generating from call k's. The robot always
# drives on thinking exactly one period old, and stalls only for whatever part of a
# generation did not fit inside the period.
#
# So the cost is legible BEFORE the run: compare a generation against a period. The sim
# runs at roughly 0.4x realtime, so a 3 s period is ~7.5 s of wall clock, against ~2.7 s
# for a medium generation -- the bound should never engage and this should be free. At
# `high` a generation is several times longer and the margin is genuinely unknown, which
# is why both levels are run rather than the cheap one being assumed to generalise.
# `bounded_stalls` on the server is the measurement, and 0 is the interesting answer.
#
# TWO LADDERS, medium and high, at the same 3 s period the synchronous arm used. Same 13
# indoor episodes, same controller, archived alongside it under nav/results/sync_study/
# so `sync_study_table.py` puts all three regimes in one matrix. The 1 s period is
# deliberately NOT run here: at 1 s a medium generation cannot fit, so bounded degrades
# toward synchronous and the two columns would measure the same thing twice.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

ONLY_OVERRIDE=""; PREFLIGHT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY_OVERRIDE="$2"; shift 2 ;;
    --skip-preflight) PREFLIGHT=0; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

STUDY="$REPO/nav/results/sync_study"
mkdir -p "$STUDY"
export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"

PERIOD="3.0"
export NAV_PLAN_PERIOD_S="$PERIOD"
export NAV_PLAN_MODE="bounded"

# The same 13 indoor episodes the synchronous arm and the async baseline both ran. The
# six outdoor episodes stay out: they postdate the baseline, and nav_outdoor_small.usda
# is a 1.1 KB stub the scene pipeline never ran on -- no joint drives, no wheel
# colliders, no camera_nav -- which is the leading candidate for the 36 km and 51 km path
# lengths two of them recorded. Derived rather than hardcoded so a reordering cannot
# silently drop one, and asserted at 13 so an addition cannot silently join.
EPISODES="$(python3 - <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("nav/config/episodes.yaml"))["episodes"]
names = [n for n in cfg if not n.startswith("outdoor_")]
if len(names) != 13:
    sys.exit(f"expected 13 indoor episodes, found {len(names)}: {names}")
print(",".join(names))
PY
)" || { echo "!! could not build the episode list" >&2; exit 1; }
[ -n "$ONLY_OVERRIDE" ] && EPISODES="$ONLY_OVERRIDE"
# awk on the field count, not `tr | wc -l`: a comma-separated list with no trailing
# newline has one fewer newline than it has items, so `wc -l` reports 12 for 13 episodes
# -- a header that undercounts the campaign it is announcing.
N_EP=$(printf '%s' "$EPISODES" | awk -F, '{print NF}')

export QVLA_LADDER_ONLY="$EPISODES"

echo "================================================================"
echo "  bounded-async study -- 2 ladders x $N_EP episodes, period ${PERIOD}s"
echo "  results: $STUDY"
echo "================================================================"
printf '  %s\n' $(printf '%s' "$EPISODES" | tr ',' ' ')
echo

# ---------------------------------------------------------------- preflight
# One short episode, then read the regime back off the artifacts. Every way the
# sim-side variable and the server-side field can fail to meet produces a complete and
# plausible result file rather than an error -- see check_bounded_mode.py -- so a
# campaign that skips this can spend four hours measuring a regime it is not in. The
# cost is one model load and one 13 m episode against a ~4 h campaign.
if [ "$PREFLIGHT" = "1" ]; then
  PRE_EP="office_hallway_turn2"
  echo "-- PREFLIGHT: $PRE_EP at medium, to prove the mode before committing the campaign"
  QVLA_LADDER_ONLY="$PRE_EP" \
    nav/tools/run_level_ladder.sh medium --only "$PRE_EP" --tag "b${PERIOD}pre" \
      > "$STUDY/preflight.log" 2>&1
  PRE_RC=$?
  # Not piped into `tee`, on purpose: the exit status is the whole point of this call,
  # and a pipeline reports the pager's status instead. Written, then shown.
  python3 nav/tools/check_bounded_mode.py --episode "$PRE_EP" --period "$PERIOD" \
      > "$STUDY/preflight_check.txt" 2>&1
  CHECK_RC=$?
  cat "$STUDY/preflight_check.txt"
  if [ "$PRE_RC" != "0" ] || [ "$CHECK_RC" != "0" ]; then
    echo "!! preflight failed (ladder rc=$PRE_RC, check rc=$CHECK_RC) -- refusing to"
    echo "!! run two ladders that would be labelled with a regime they did not use."
    python3 nav/tools/notify_run.py \
      --subject "[bounded-study] PREFLIGHT FAILED -- campaign not started" \
      --body "$(cat "$STUDY/preflight_check.txt")

ladder rc=$PRE_RC, check rc=$CHECK_RC
log: $STUDY/preflight.log" || true
    exit 1
  fi
  echo "-- preflight passed"
  echo
fi

# ---------------------------------------------------------------- the ladders
for LEVEL in medium high; do
  NAME="bounded${PERIOD}_${LEVEL}"
  DEST="$STUDY/$NAME"

  # A re-run must not be scored against the run it replaces. The archive keeps every
  # JSON it is given and on_episode.sh appends to the progress file, so a second campaign
  # into the same paths reads as one campaign of 26 episodes, 13 of them stale. Moved
  # aside rather than deleted -- a partial condition is still the only copy of whatever
  # it measured, and this script is not the right place to decide it is worthless.
  if [ -d "$DEST" ] && [ -n "$(ls -1 "$DEST"/*.json 2>/dev/null)" ]; then
    BAK="$DEST.superseded-$(date +%Y%m%d-%H%M%S)"
    echo "-- $NAME already holds results; moving them to $(basename "$BAK")"
    mv "$DEST" "$BAK"
  fi
  mkdir -p "$DEST"
  rm -f "nav/results/ladder_progress_${LEVEL}_b${PERIOD}.log"

  # Results are written into nav/results/ and COPIED here, selected by mtime against
  # this marker. Deliberately not redirected at the source: bench.sh counts files there
  # before and after each episode to catch a run that produced no measurement, and
  # pointing that glob elsewhere would disarm the check that exists because a ladder once
  # scored 13 episodes off files up to two weeks old.
  MARK="$DEST/.started"
  : > "$MARK"

  echo "================================================================"
  echo "  CONDITION $NAME  --  bounded async, period ${PERIOD}s, think level $LEVEL"
  echo "  started $(date -Is)"
  echo "================================================================"

  nav/tools/run_level_ladder.sh "$LEVEL" --only "$EPISODES" --tag "b${PERIOD}" \
      > "$DEST/ladder.log" 2>&1 &
  LADDER_PID=$!

  # One reporter per condition, so the tag in its subject and the progress file it reads
  # are always the condition actually running. It exits by itself when the ladder pid
  # goes away, sending a last mail as it does.
  QVLA_RUN_TAG="${LEVEL}_b${PERIOD}" \
    setsid nohup nav/tools/hourly_report.sh 3600 "$LADDER_PID" \
      > "$DEST/hourly.log" 2>&1 &

  wait "$LADDER_PID"
  RC=$?
  echo "-- condition $NAME finished (rc=$RC) at $(date -Is)"

  # `-newer` against the marker rather than a name pattern: the filenames carry only a
  # timestamp and the episode, nothing that says which condition produced them.
  find nav/results -maxdepth 1 -name '*.json' -newer "$MARK" \
      -exec cp -p {} "$DEST/" \; 2>/dev/null
  cp -p "nav/results/ladder_progress_${LEVEL}_b${PERIOD}.log" "$DEST/" 2>/dev/null
  cp -p "nav/results/logs/server_qwen_${LEVEL}.log" "$DEST/server.log" 2>/dev/null
  find nav/results/videos -maxdepth 1 -name "*__${LEVEL}_b${PERIOD}.mp4" -newer "$MARK" \
      -exec cp -p {} "$DEST/" \; 2>/dev/null
  N_JSON=$(ls -1 "$DEST"/*.json 2>/dev/null | wc -l)
  echo "-- archived $N_JSON result files into $DEST"

  # The bound's own verdict, straight off the server that just ran the ladder: how often
  # a generation failed to fit inside a period, and what that cost. This is the number
  # the whole regime is judged on and it lives nowhere else, so it is captured here
  # before the next condition restarts the server and resets the counters.
  curl -sf --max-time 10 "http://127.0.0.1:${NAV_POLICY_PORT}/health" \
      > "$DEST/health.json" 2>/dev/null
  BOUND="$(python3 - "$DEST/health.json" <<'PY'
import json, sys
try:
    h = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"(no /health captured: {exc})")
    sys.exit(0)
calls, stalls = h.get("bounded_calls", 0), h.get("bounded_stalls", 0)
pct = f"{100.0 * stalls / calls:.0f}%" if calls else "n/a"
print(f"bounded_calls={calls} stalls={stalls} ({pct}) "
      f"stall_s={h.get('bounded_stall_s')} timeouts={h.get('bounded_timeouts')}")
PY
)"
  echo "-- $BOUND"

  python3 nav/tools/sync_study_table.py > "$STUDY/comparison.txt" 2>&1
  python3 nav/tools/notify_run.py \
    --subject "[bounded-study] $NAME done ($N_JSON runs, rc=$RC)" \
    --body "Condition $NAME finished at $(date -Is).
bounded async, period ${PERIOD}s, think level $LEVEL, $N_EP episodes, rc=$RC

How often the bound engaged (0 stalls = the period always hid the generation):
  $BOUND

$(cat "$STUDY/comparison.txt")

archived to: $DEST" || echo "!! condition mail failed (continuing)"
done

echo "================================================================"
echo "  bounded study complete at $(date -Is)"
echo "================================================================"
python3 nav/tools/sync_study_table.py --verbose | tee "$STUDY/comparison.txt"
python3 nav/tools/notify_run.py \
  --subject "[bounded-study] BOTH BOUNDED LADDERS DONE" \
  --body "$(cat "$STUDY/comparison.txt")

Full results: $STUDY" || true
