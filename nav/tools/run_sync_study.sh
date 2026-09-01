#!/usr/bin/env bash
# The synchronous-planning study: does more thinking help once the robot STOPS for it?
#
#   setsid nohup nav/tools/run_sync_study.sh > /tmp/sync_study.log 2>&1 &
#   nav/tools/run_sync_study.sh --only hospital_exit_room     # smoke test one episode
#
# THE QUESTION. The async ladder scored 7/13 at medium, 2/13 at high and 2/13 at
# very_high -- thinking harder made the robot measurably worse. The cause was not the
# thinking: `/predict` served a cached plan while regenerating in the background, so the
# robot drove BLIND for the length of every generation and that blind window grew with
# the budget. The trade was never "better decisions vs. worse decisions", it was "better
# decisions vs. longer stretches of nobody steering", and past medium the second term won.
#
# NAV_PLAN_PERIOD_S removes the second term entirely: the robot stands still for each
# decision. So this campaign asks the original question -- does a bigger thinking budget
# buy better navigation? -- for the first time without the confound.
#
# FOUR LADDERS, and the two axes are deliberate. The PERIOD (3 s, 1 s) trades decision
# frequency against how much of the wall clock is spent standing still; the LEVEL
# (medium, high) trades decision quality against the same. They are not independent --
# halving the period doubles the number of generations, so 1 s at `high` is the most
# expensive corner by a wide margin -- which is exactly why all four corners are needed
# to say anything about either axis.
#
# The order below is the order asked for, not the cheapest one. Grouping by level would
# save two model reloads (~5 min each), which is nothing against a ~7 h campaign, and
# running the conditions in the stated order means a partial campaign is still readable
# in the order somebody expects to read it.
#
# WHY THIS IS NOT A LOOP INSIDE bench.sh. Each condition needs its own policy server,
# because the thinking level lives in the server and drags a plan horizon with it.
# `run_level_ladder.sh` already does that cycle including the check that refuses to run
# when /health reports a level other than the one asked for -- the failure it prevents is
# silent, producing thirteen complete and plausible numbers for the wrong configuration.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

ONLY_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY_OVERRIDE="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

STUDY="$REPO/nav/results/sync_study"
mkdir -p "$STUDY"
export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"

# THE LADDER IS THE 13 INDOOR EPISODES, and the async baseline this is compared against
# is the same 13. The six outdoor episodes are excluded on purpose: they were added after
# that baseline was measured, so including them would make the comparison the whole
# campaign exists for un-runnable, and `assets/usd/nav_outdoor_small.usda` is a 1.1 KB
# stub that never had the pipeline applied to it -- no joint drives, no wheel colliders,
# no camera_nav -- which is a strong candidate for the 36 km and 51 km path lengths two
# of them recorded. Derived rather than hardcoded so a reordering of the config cannot
# silently drop one, and asserted at 13 so an ADDITION cannot silently join.
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
# -- a header and a mail body that undercount the campaign they are announcing.
N_EP=$(printf '%s' "$EPISODES" | awk -F, '{print NF}')

# Exported so bench_status.py's hourly mail counts against the ladder actually running
# rather than against every episode in the config.
export QVLA_LADDER_ONLY="$EPISODES"

echo "================================================================"
echo "  synchronous-planning study -- 4 ladders x $N_EP episodes"
echo "  results: $STUDY"
echo "================================================================"
printf '  %s\n' $(printf '%s' "$EPISODES" | tr ',' ' ')
echo

CONDITIONS=("3.0 medium" "3.0 high" "1.0 medium" "1.0 high")

for COND in "${CONDITIONS[@]}"; do
  read -r PERIOD LEVEL <<<"$COND"
  NAME="p${PERIOD}_${LEVEL}"
  DEST="$STUDY/$NAME"

  # A re-run must not be scored against the run it is replacing. Both of these are
  # append-or-accumulate by nature -- the archive directory keeps every JSON it is given
  # and `on_episode.sh` appends to the progress file -- so a second campaign into the
  # same paths would read as one campaign of 26 episodes, 13 of them stale. Moved aside
  # rather than deleted: a partial condition is still the only copy of whatever it
  # measured, and this script is not the right place to decide it is worthless.
  if [ -d "$DEST" ] && [ -n "$(ls -1 "$DEST"/*.json 2>/dev/null)" ]; then
    BAK="$DEST.superseded-$(date +%Y%m%d-%H%M%S)"
    echo "-- $NAME already holds results; moving them to $(basename "$BAK")"
    mv "$DEST" "$BAK"
  fi
  mkdir -p "$DEST"
  rm -f "nav/results/ladder_progress_${LEVEL}_p${PERIOD}.log"

  # A timestamp file is how this condition's results are told apart from the previous
  # condition's afterwards. The runner writes every run into nav/results/ (deliberately
  # not redirected: bench.sh counts files there before and after each episode to catch a
  # run that produced no measurement, and pointing that glob somewhere else would disarm
  # the check that exists because a ladder once scored 13 episodes off files up to two
  # weeks old). So the results are COPIED here, selected by mtime against this marker.
  MARK="$DEST/.started"
  : > "$MARK"

  export NAV_PLAN_PERIOD_S="$PERIOD"
  echo "================================================================"
  echo "  CONDITION $NAME  --  period ${PERIOD}s, think level $LEVEL"
  echo "  started $(date -Is)"
  echo "================================================================"

  nav/tools/run_level_ladder.sh "$LEVEL" --only "$EPISODES" --tag "p${PERIOD}" \
      > "$DEST/ladder.log" 2>&1 &
  LADDER_PID=$!

  # One reporter per condition, so the tag in its subject line and the progress file it
  # reads are always the condition that is actually running. It exits by itself when the
  # ladder pid goes away, sending a last mail as it does.
  QVLA_RUN_TAG="${LEVEL}_p${PERIOD}" \
    setsid nohup nav/tools/hourly_report.sh 3600 "$LADDER_PID" \
      > "$DEST/hourly.log" 2>&1 &

  wait "$LADDER_PID"
  RC=$?
  echo "-- condition $NAME finished (rc=$RC) at $(date -Is)"

  # Archive: the result JSONs this condition produced, its progress file, and its logs.
  # `-newer` against the marker rather than a name pattern, because the filenames carry
  # only a timestamp and the episode -- nothing that says which condition they came from.
  find nav/results -maxdepth 1 -name '*.json' -newer "$MARK" \
      -exec cp -p {} "$DEST/" \; 2>/dev/null
  cp -p "nav/results/ladder_progress_${LEVEL}_p${PERIOD}.log" "$DEST/" 2>/dev/null
  cp -p "nav/results/logs/server_qwen_${LEVEL}.log" "$DEST/server.log" 2>/dev/null
  find nav/results/videos -maxdepth 1 -name "*__${LEVEL}_p${PERIOD}.mp4" -newer "$MARK" \
      -exec cp -p {} "$DEST/" \; 2>/dev/null
  N_JSON=$(ls -1 "$DEST"/*.json 2>/dev/null | wc -l)
  echo "-- archived $N_JSON result files into $DEST"

  # The running scoreboard, mailed at every condition boundary. Sent even when the
  # condition errored -- especially then, because a campaign that quietly lost its third
  # ladder at 2 a.m. should say so at 2 a.m. and not at the end.
  python3 nav/tools/sync_study_table.py > "$STUDY/comparison.txt" 2>&1
  python3 nav/tools/notify_run.py \
    --subject "[sync-study] $NAME done ($N_JSON runs, rc=$RC)" \
    --body "Condition $NAME finished at $(date -Is).
period ${PERIOD}s, think level $LEVEL, $N_EP episodes, rc=$RC

$(cat "$STUDY/comparison.txt")

archived to: $DEST" || echo "!! condition mail failed (continuing)"
done

echo "================================================================"
echo "  study complete at $(date -Is)"
echo "================================================================"
python3 nav/tools/sync_study_table.py --verbose | tee "$STUDY/comparison.txt"
python3 nav/tools/notify_run.py \
  --subject "[sync-study] ALL FOUR CONDITIONS DONE" \
  --body "$(cat "$STUDY/comparison.txt")

Full results: $STUDY" || true
