#!/usr/bin/env bash
# The six episodes the campaign could not do, run three times each under five menus.
#
#   setsid nohup nav/tools/run_hard_study.sh > /tmp/hard_study.log 2>&1 &
#   nav/tools/run_hard_study.sh --arms ref,all --seeds 0     # a short version
#
# WHY THESE SIX. Across the six arms already measured -- baseline, prefill, prefill+high,
# pivot, pivot+high, bounded -- 46 of 78 runs succeeded, and the successes were not spread
# evenly. Two episodes succeeded every time and these six did not reach half:
#
#     hospital_forward_staircase    0 / 6
#     warehouse_aisle6              1 / 6
#     hospital_exit_room            2 / 6
#     hospital_down_hallway2        3 / 7
#     warehouse                     3 / 6
#     hospital_past_wheelchairs     3 / 6
#
# Everything above 50% flips with the configuration and tells us nothing; these are where
# the remaining failure actually lives. Hardcoded rather than derived from the results on
# disk, and that is the point: the set is a property of the campaign that has ALREADY run,
# and a list recomputed at launch would quietly change composition as this study adds its
# own results, which would make the study's own arms incomparable to each other.
#
# WHY THREE SEEDS, when nothing here samples. Decoding is greedy, so the model is a
# function of its input; the one thing that is drawn is the label permutation, from
# QVLA_MENU_SEED. That permutation is the whole defence against "answer the middle number"
# scoring, so a policy that only works under one arrangement of the digits is not a policy
# that works. Three seeds is also the missing ingredient in every number this campaign has
# produced so far: single runs cannot resolve a 9-against-7, and six arms of single runs
# produced ten episodes out of thirteen that flipped at least once.
#
# It buys one free check as well. Arm `ref` at seed 0 is bit-for-bit the configuration that
# already ran as p3.0_medium. If those six episodes do not reproduce, then something
# outside the seed is moving between runs, and every comparison in this file -- and in the
# campaign before it -- is measuring that instead.
#
# THE ARMS are one factor at a time, plus the combination:
#
#     ref       coarse 7 arcs,  no turns,       one frame     the current best
#     all       fine 11 arcs,   15 deg turns,   two frames    what was asked for
#     pivot15   coarse 7 arcs,  15 deg turns,   one frame     the turn alone
#     memory    coarse 7 arcs,  no turns,       two frames    the memory alone
#     fine11    fine 11 arcs,   no turns,       one frame     the finer menu alone
#
# `ref` runs first because nothing else is readable without it, and `all` second because it
# is the question -- so an interrupted study still answers "did it help" even if it cannot
# yet say which part helped.
#
# Level is `medium` throughout and that is a result, not a shortcut: `high` cost 5x the
# thinking time for identical decision counts and 3.5x the collisions, because 89-93% of
# its reasoning blocks were cut off mid-sentence by the 180-token budget. Spending this
# study's hours on more repeats of medium buys more than spending them on high.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

# name : arc_set : pivots : frames [: seeds]
#
# `refrep` is `ref` a second time at seed 0 and nothing else, and it runs SECOND rather
# than last. It is the determinism check, and a determinism check that arrives after the
# results it licenses has arrived too late: if the same configuration run twice does not
# give the same six verdicts, then no arm below it has been shown to differ from any
# other, and the remaining eleven hours should be spent on that instead.
ARMS_ALL="ref:coarse:0:1 refrep:coarse:0:1:0 all:fine:1:2 pivot15:coarse:1:1 \
memory:coarse:0:2 fine11:fine:0:1"
SEEDS="0 1 2"
SEEDS_FROM_CLI=0
ARM_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arms)  ARM_FILTER="$2"; shift 2 ;;
    --seeds) SEEDS="$(printf '%s' "$2" | tr ',' ' ')"; SEEDS_FROM_CLI=1; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

ARMS=""
for spec in $ARMS_ALL; do
  name="${spec%%:*}"
  case ",${ARM_FILTER}," in
    ,,) ARMS="$ARMS $spec" ;;
    *",$name,"*) ARMS="$ARMS $spec" ;;
  esac
done
[ -n "$ARMS" ] || { echo "!! --arms matched nothing in: $ARMS_ALL" >&2; exit 2; }

# Validated against the config rather than trusted. A misspelled episode does not fail --
# `bench.sh --only` simply runs the ones it recognises -- so the arm would come back with
# five results, be archived, tabulated and mailed, and read as a complete condition.
EPISODES="hospital_forward_staircase,warehouse_aisle6,hospital_exit_room,\
hospital_down_hallway2,warehouse,hospital_past_wheelchairs"
python3 - "$EPISODES" <<'PY' || exit 1
import sys, yaml
want = sys.argv[1].split(",")
have = yaml.safe_load(open("nav/config/episodes.yaml"))["episodes"]
missing = [e for e in want if e not in have]
if missing:
    sys.exit(f"!! not in nav/config/episodes.yaml: {missing}")
print(f"-- {len(want)} episodes validated against the config")
PY

STUDY="$REPO/nav/results/hard_study"
mkdir -p "$STUDY"
LEVEL="medium"
PERIOD="3.0"

export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"
export NAV_PLAN_MODE="sync"
export NAV_PLAN_PERIOD_S="$PERIOD"
export QVLA_LADDER_ONLY="$EPISODES"

N_ARM=$(printf '%s' "$ARMS" | wc -w)
N_SEED=$(printf '%s' "$SEEDS" | wc -w)
N_EP=$(printf '%s' "$EPISODES" | awk -F, '{print NF}')
# Counted by walking the arms rather than multiplying, because an arm may pin its own
# seeds. A header that says 18 ladders when 16 will run is the kind of small lie that
# makes somebody think the campaign died early.
N_LADDER=0
for spec in $ARMS; do
  IFS=: read -r _ _ _ _ S_ <<<"$spec"
  if [ -n "${S_:-}" ] && [ "$SEEDS_FROM_CLI" = 0 ]; then
    N_LADDER=$((N_LADDER + $(printf '%s' "$S_" | tr ',' ' ' | wc -w)))
  else
    N_LADDER=$((N_LADDER + N_SEED))
  fi
done
echo "================================================================"
echo "  hard-episode study -- $N_ARM arms, $N_EP episodes each"
echo "  = $N_LADDER ladders, $((N_LADDER * N_EP)) runs, ~9 min each"
echo "  results: $STUDY"
echo "================================================================"
printf '  %s\n' $(printf '%s' "$EPISODES" | tr ',' ' ')
echo

python3 nav/tools/notify_run.py \
  --subject "[hard] study restarted -- $((N_LADDER * N_EP)) runs on the 6 unsolved episodes" \
  --body "Focusing on the episodes that stayed broken, as asked.

RESTARTED after one ladder, because that ladder's own reproducibility check failed and
found a real defect. The label generator was built once per server process and drawn from
across a whole ladder, so an episode's permutation depended on how many decisions the
episodes BEFORE it had consumed. hospital_exit_room at seed 0 drew [5,6,1,2,7,4,3] in the
13-episode ladder and [6,7,1,2,5,4,3] in the 6-episode one -- a different menu, a
different answer, from the same seed.

That is worse than an irreproducible re-run. The decision count differs between ARMS too,
so an arm whose first episode finished in 30 decisions handed its second episode a
different permutation than an arm that took 45: part of every arm-to-arm difference in
this campaign was a different draw rather than a different policy. The generator is now
reseeded per episode from (seed, episode name), so a permutation depends on nothing else.

The six, and their record across the six arms measured so far:

  hospital_forward_staircase    0 / 6
  warehouse_aisle6              1 / 6
  hospital_exit_room            2 / 6
  hospital_down_hallway2        3 / 7
  warehouse                     3 / 6
  hospital_past_wheelchairs     3 / 6

Everything above 50% flips with the configuration, so it measures noise. These do not.

$N_ARM arms, each run at three label seeds, all at think level medium:

  ref       coarse 7 arcs,  no turns,       one frame     the current best
  refrep    ref again at seed 0                           the determinism check
  all       fine 11 arcs,   15 deg turns,   two frames    what was asked for
  pivot15   coarse 7 arcs,  15 deg turns,   one frame     the turn alone
  memory    coarse 7 arcs,  no turns,       two frames    the memory alone
  fine11    fine 11 arcs,   no turns,       one frame     the finer menu alone

Three seeds because decoding is greedy -- the only thing drawn is the label permutation,
and a policy that only works under one arrangement of the digits is not a policy that
works.

refrep runs SECOND, not last. It is ref again with identical settings at the same seed,
and it asks the only question that licenses everything below it: is this stack a function
of its inputs? If the same configuration run twice does not give the same six verdicts,
then no arm has been shown to differ from any other and the remaining hours belong to
that instead. It is now answerable at all only because of the reseeding fix above.

Mail follows at every arm boundary. log: /tmp/hard_study.log" || echo "!! start mail failed"

for spec in $ARMS; do
  IFS=: read -r ARM ARCSET PIV FRAMES ARM_SEEDS <<<"$spec"
  export QVLA_MENU_ARCS="$ARCSET"
  export QVLA_MENU_PIVOTS="$PIV"
  export QVLA_MENU_FRAMES="$FRAMES"
  export QVLA_PIVOT_DEG="15"
  # Set unconditionally, including for the arms that do not use it. An env var left over
  # from the previous arm is exactly the failure `run_level_ladder.sh` now refuses to
  # drive through, and the cheapest place to not have it is here.

  # An arm may pin its own seeds. Only `refrep` does: repeating a determinism check at
  # three seeds would answer the same question three times at three times the cost.
  # `--seeds` on the command line still wins, so a deliberate re-run is not overridden.
  ARM_SEED_LIST="$SEEDS"
  [ -n "${ARM_SEEDS:-}" ] && [ "$SEEDS_FROM_CLI" = 0 ] && \
    ARM_SEED_LIST="$(printf '%s' "$ARM_SEEDS" | tr ',' ' ')"

  for SEED in $ARM_SEED_LIST; do
    export QVLA_MENU_SEED="$SEED"
    NAME="${ARM}_s${SEED}"
    DEST="$STUDY/$NAME"
    TAG="${ARM}_s${SEED}"

    if [ -d "$DEST" ] && [ -n "$(ls -1 "$DEST"/*.json 2>/dev/null)" ]; then
      BAK="$DEST.superseded-$(date +%Y%m%d-%H%M%S)"
      echo "-- $NAME already holds results; moving them to $(basename "$BAK")"
      mv "$DEST" "$BAK"
    fi
    mkdir -p "$DEST"
    rm -f "nav/results/ladder_progress_${LEVEL}_${TAG}.log"
    MARK="$DEST/.started"
    : > "$MARK"

    echo "================================================================"
    echo "  ARM $NAME  --  $ARCSET arcs, pivots=$PIV, frames=$FRAMES, seed $SEED"
    echo "  started $(date -Is)"
    echo "================================================================"

    nav/tools/run_level_ladder.sh "$LEVEL" --only "$EPISODES" --tag "$TAG" \
        > "$DEST/ladder.log" 2>&1 &
    LADDER_PID=$!
    QVLA_RUN_TAG="${LEVEL}_${TAG}" \
      setsid nohup nav/tools/hourly_report.sh 3600 "$LADDER_PID" \
        > "$DEST/hourly.log" 2>&1 &
    wait "$LADDER_PID"
    RC=$?
    echo "-- $NAME finished (rc=$RC) at $(date -Is)"

    find nav/results -maxdepth 1 -name '*.json' -newer "$MARK" \
        -exec cp -p {} "$DEST/" \; 2>/dev/null
    cp -p "nav/results/ladder_progress_${LEVEL}_${TAG}.log" "$DEST/" 2>/dev/null
    cp -p "nav/results/logs/server_qwen_${LEVEL}.log" "$DEST/server.log" 2>/dev/null
    find nav/results/videos -maxdepth 1 -name "*__${LEVEL}_${TAG}.mp4" -newer "$MARK" \
        -exec cp -p {} "$DEST/" \; 2>/dev/null
    N_JSON=$(ls -1 "$DEST"/*.json 2>/dev/null | wc -l)
    echo "-- archived $N_JSON result files into $DEST"
    # The configuration is written next to the results, not only into a log that the next
    # arm overwrites. Six months from now the directory name is the only thing left saying
    # what `all_s2` was, and a directory name is not a record.
    printf '{"arm":"%s","arc_set":"%s","pivots":%s,"frames":%s,"seed":%s,"level":"%s",\n "period_s":%s,"episodes":"%s","rc":%s,"finished":"%s"}\n' \
      "$ARM" "$ARCSET" "$([ "$PIV" = 1 ] && echo true || echo false)" "$FRAMES" \
      "$SEED" "$LEVEL" "$PERIOD" "$EPISODES" "$RC" "$(date -Is)" > "$DEST/arm.json"
  done

  # One mail per ARM, after its seeds -- not per ladder. A per-ladder mail would send
  # fifteen, and the only comparison worth reading is across seeds anyway: one seed's
  # success on one episode is the unit of noise this study exists to average over.
  python3 nav/tools/hard_study_table.py > "$STUDY/comparison.txt" 2>&1
  python3 nav/tools/notify_run.py \
    --subject "[hard] arm $ARM done ($N_SEED seeds x $N_EP episodes)" \
    --body "Arm $ARM finished at $(date -Is).
$ARCSET arcs, pivots=$PIV, memory frames=$FRAMES, level $LEVEL, seeds: $SEEDS

$(cat "$STUDY/comparison.txt")

archived to: $STUDY/${ARM}_s*" || echo "!! arm mail failed (continuing)"
done

echo "================================================================"
echo "  hard-episode study complete at $(date -Is)"
echo "================================================================"
python3 nav/tools/hard_study_table.py --verbose | tee "$STUDY/comparison.txt"
python3 nav/tools/notify_run.py \
  --subject "[hard] ALL ARMS DONE -- do 15 degrees and a memory frame fix the six?" \
  --body "$(cat "$STUDY/comparison.txt")

Full results: $STUDY" || true
