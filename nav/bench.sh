#!/usr/bin/env bash
# Run the DynaNav benchmark ladder end to end, easiest episode first.
#
#   nav/bench.sh                          # every episode in episodes.yaml, braking
#   nav/bench.sh --controller pursuit     # the DynaNav-parity baseline
#   nav/bench.sh --only office_elevator,hospital_red   # just these
#   nav/bench.sh --from hospital_vending  # resume partway down the ladder
#
# WHY THIS IS A SHELL LOOP AND NOT A PYTHON ONE. Each episode needs its own pair of
# processes, and neither can be reused:
#
#   1. The start pose is baked into the USD stage at author time, so a new episode
#      means a new `nav_<episode>.usda` -- built by a chain of four Isaac Sim
#      invocations, because SimulationApp.close() takes the interpreter down with it
#      and nothing after it runs.
#   2. The runner then opens that stage. Isaac Sim cannot swap stages cleanly inside
#      one process at this version, and a half-swapped stage fails as a navigation
#      error rather than as a crash -- the worst possible failure mode for a
#      benchmark, because it produces a plausible number.
#
# So: one build process chain plus one run process per episode, sequentially. A full
# 11-episode ladder is a couple of hours. That is the price of a number you can trust.
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
while [ $# -gt 0 ]; do
  case "$1" in
    --controller) CONTROLLER="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --from)       FROM="$2"; shift 2 ;;
    --stop-on-fail) KEEP_GOING=0; shift ;;
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
  SCENE="$REPO/assets/usd/nav_${EP}.usda"

  if [ ! -f "$SCENE" ] || [ "$CONFIG" -nt "$SCENE" ]; then
    echo "-- building $SCENE"
    if ! nav/sim/build_nav_scene.sh "$EP" > "$LOGDIR/build_${EP}.log" 2>&1; then
      echo "!! BUILD FAILED -- see $LOGDIR/build_${EP}.log"
      tail -15 "$LOGDIR/build_${EP}.log"
      ERR=$((ERR+1)); [ $KEEP_GOING -eq 1 ] && continue || exit 1
    fi
    echo "-- built"
  else
    echo "-- scene cached"
  fi

  echo "-- running (this blocks until the episode ends or times out)"
  nav/run.sh --episode "$EP" --controller "$CONTROLLER" --no-ui \
      > "$LOGDIR/run_${EP}_${CONTROLLER}.log" 2>&1
  RC=$?

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
  fi
  echo
done

echo "================================================================"
echo "  ladder done: $PASS succeeded, $FAIL failed, $ERR errored"
echo "================================================================"
python3 nav/sim/summarize_runs.py --controller "$CONTROLLER"
