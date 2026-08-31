#!/usr/bin/env bash
# Run the DynaNav benchmark ladder end to end, easiest episode first.
#
#   nav/bench.sh                          # every episode in episodes.yaml, braking
#   nav/bench.sh --controller pursuit     # the DynaNav-parity baseline
#   nav/bench.sh --only office_elevator,hospital_red   # just these
#   nav/bench.sh --from hospital_vending  # resume partway down the ladder
#
# WHY THIS IS A SHELL LOOP AND NOT A PYTHON ONE. Isaac Sim cannot swap stages cleanly
# inside one process at this version, and a half-swapped stage fails as a navigation
# error rather than as a crash -- the worst possible failure mode for a benchmark,
# because it produces a plausible number. So: one run process per episode,
# sequentially. That is the price of a number you can trust.
#
# BUILDS ARE PER ENVIRONMENT, RUNS ARE PER EPISODE. A stage used to be authored per
# episode, which made 13 episodes cost 13 four-process Isaac Sim build chains for
# stages that differed only in where one prim sat -- six of them the same hospital.
# The runner teleports to `episode.start` before every run, so the baked pose was
# never load-bearing. 13 builds -> 3.
#
# Scenes are cached. A rebuild only happens when the .usda is missing or older than
# the episode config, so a re-run of the ladder skips straight to driving.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

CONTROLLER="braking"
ONLY=""
FROM=""
KEEP_GOING=1
# Optional command run after every episode, as: $ON_EPISODE <episode> <controller>
# <verdict>. It is what turns an unattended two-hour ladder into something you can
# follow -- build the video, send the mail, whatever. Deliberately a hook and not
# built in: none of that belongs in the thing whose job is to produce a trustworthy
# number, and a hook that fails must never be able to change one.
ON_EPISODE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --controller) CONTROLLER="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --from)       FROM="$2"; shift 2 ;;
    --stop-on-fail) KEEP_GOING=0; shift ;;
    --on-episode) ON_EPISODE="$2"; shift 2 ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

CONFIG="$REPO/nav/config/episodes.yaml"
LOGDIR="$REPO/nav/results/logs"
mkdir -p "$LOGDIR"

# Episode order comes from the config file, which import_benchmark.py writes
# easiest-first. Reading it here rather than re-ranking keeps one ordering in one
# place: if the ladder looks wrong, it is wrong in the generated YAML, not in two
# subtly different sorts.
mapfile -t EPISODES < <(python3 - "$CONFIG" "$ONLY" "$FROM" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))["episodes"]
names = list(cfg)
only = [s for s in sys.argv[2].split(",") if s]
if only:
    missing = [s for s in only if s not in names]
    if missing:
        sys.exit(f"no such episode(s): {', '.join(missing)}")
    names = [n for n in names if n in only]
start = sys.argv[3]
if start:
    if start not in names:
        sys.exit(f"no such episode: {start}")
    names = names[names.index(start):]
print("\n".join(names))
PY
) || { echo "could not read episode list" >&2; exit 1; }

echo "ladder: ${#EPISODES[@]} episodes, controller=$CONTROLLER"
printf '  %s\n' "${EPISODES[@]}"
echo

PASS=0; FAIL=0; ERR=0
for EP in "${EPISODES[@]}"; do
  echo "================================================================"
  echo "  $EP  ($CONTROLLER)"
  echo "================================================================"
  ENV_NAME="$(python3 nav/sim/resolve_env.py "$EP")" || { ERR=$((ERR+1)); continue; }
  SCENE="$REPO/assets/usd/nav_${ENV_NAME}.usda"

  if [ ! -f "$SCENE" ] || [ "$CONFIG" -nt "$SCENE" ]; then
    echo "-- building $SCENE (serves every $ENV_NAME episode)"
    if ! nav/sim/build_nav_scene.sh "$ENV_NAME" > "$LOGDIR/build_${ENV_NAME}.log" 2>&1; then
      echo "!! BUILD FAILED -- see $LOGDIR/build_${ENV_NAME}.log"
      tail -15 "$LOGDIR/build_${ENV_NAME}.log"
      ERR=$((ERR+1)); [ $KEEP_GOING -eq 1 ] && continue || exit 1
    fi
    echo "-- built"
  else
    echo "-- scene cached ($ENV_NAME)"
  fi

  echo "-- running (this blocks until the episode ends or times out)"
  # Count the results this episode already has. `summarize_runs.py --latest` returns
  # the NEWEST matching run, and it has no idea whether that run is the one we just
  # asked for. When `run_navigation.py` died in setup(), the ladder scored all 13
  # episodes off files up to two weeks old and printed a plausible 6/13. The trailing
  # `_${CONTROLLER}.json` anchors the glob, so `hospital_down_hallway` does not match
  # `hospital_down_hallway2`.
  BEFORE=$(ls -1 nav/results/*_"${EP}_${CONTROLLER}".json 2>/dev/null | wc -l)

  nav/run.sh --episode "$EP" --controller "$CONTROLLER" --no-ui \
      > "$LOGDIR/run_${EP}_${CONTROLLER}.log" 2>&1
  RC=$?

  AFTER=$(ls -1 nav/results/*_"${EP}_${CONTROLLER}".json 2>/dev/null | wc -l)
  if [ "$AFTER" -eq "$BEFORE" ]; then
    # No new file. Whatever rc says, this run produced no measurement -- do not let
    # summarize_runs.py answer with somebody else's.
    echo "!! NO RESULT WRITTEN (rc=$RC) -- run produced no new file in nav/results/;"
    echo "   refusing to score it off an older run. See $LOGDIR/run_${EP}_${CONTROLLER}.log"
    grep -m1 -B2 -A6 "Traceback (most recent call last)" \
        "$LOGDIR/run_${EP}_${CONTROLLER}.log" | sed 's/^/   /'
    ERR=$((ERR+1))
    [ $KEEP_GOING -eq 0 ] && exit 1
    echo
    continue
  fi

  # Headless run_navigation.py returns 0 on a completed episode and 1 on a failed
  # one -- a failed EPISODE, not a failed process. Anything else is a real crash.
  # Conflating the two would have logged every unsuccessful episode as an
  # infrastructure error and hidden the actual benchmark result behind a stack
  # trace, which is precisely the failure this harness exists to avoid.
  if [ $RC -le 1 ]; then
    VERDICT=$(python3 nav/sim/summarize_runs.py --latest "$EP" --controller "$CONTROLLER" --oneline)
    echo "   $VERDICT"
    case "$VERDICT" in
      SUCCESS*) PASS=$((PASS+1)) ;;
      NO-RUN*)  echo "   (no result file -- treating as an error)"; ERR=$((ERR+1)) ;;
      *)        FAIL=$((FAIL+1)) ;;
    esac
  else
    echo "!! RUN CRASHED (rc=$RC) -- see $LOGDIR/run_${EP}_${CONTROLLER}.log"
    tail -15 "$LOGDIR/run_${EP}_${CONTROLLER}.log"
    ERR=$((ERR+1))
    [ $KEEP_GOING -eq 0 ] && exit 1
    VERDICT="CRASHED (rc=$RC)"
  fi

  # After the verdict is counted, so a broken hook cannot move the score. Errors are
  # printed and swallowed for the same reason: an email that bounces at episode 4 must
  # not end a ladder that still has nine episodes to run.
  if [ -n "$ON_EPISODE" ]; then
    "$ON_EPISODE" "$EP" "$CONTROLLER" "${VERDICT:-unknown}" \
      || echo "!! on-episode hook failed for $EP (continuing)"
  fi
  echo
done

echo "================================================================"
echo "  ladder done: $PASS succeeded, $FAIL failed, $ERR errored"
echo "================================================================"
python3 nav/sim/summarize_runs.py --controller "$CONTROLLER"
