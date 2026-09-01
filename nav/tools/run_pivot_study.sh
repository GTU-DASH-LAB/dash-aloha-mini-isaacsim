#!/usr/bin/env bash
# The in-place-turn arm of the planning study: is the MENU the thing that is missing?
#
#   setsid nohup nav/tools/run_pivot_study.sh > /tmp/pivot_study.log 2>&1 &
#   nav/tools/run_pivot_study.sh --only office_hallway_turn2    # smoke test one episode
#   nav/tools/run_pivot_study.sh --skip-preflight               # server already proven
#
# THE QUESTION. The synchronous arm answered the one it was built for -- removing the
# blind window took medium from 7/13 to 9/13 -- and then showed, cleanly, that the
# remaining failures are not a thinking problem at all:
#
#                        frozen*   guard    recoveries   STOP answers
#     the failures       16-35%    0-299      1-4        up to 45%
#     the successes         0%     0-61        0           <=1
#     (* share of trace samples within 0.5 m of the sample 20 back)
#
# The failures are not "drove to the wrong place". They are "ended up facing a direction
# it could not navigate out of, and answered STOP until the clock ran out". All three
# successes ALSO drove straight ~90% of the time, so the straight bias is not what
# separates them -- the ability to reorient is.
#
# And the menu cannot reorient. Every arc is 3 m of FORWARD travel; the tightest sweeps
# 103 degrees but still ends 1.63 m down the corridor. With a wall a metre ahead every
# drawn path runs into it and STOP is the only honest answer on the menu. The selector
# prompt has said "do not answer STOP merely because the way ahead is tight" since the
# first ladder, and it does not hold, because it is asking for an action that is not there.
#
# So this arm adds one: two white rotation glyphs above the horizon, 30 degrees in place,
# direction chosen by the model. 30 and not 90 because the robot re-decides the moment a
# turn finishes -- three small turns with a look between them beat one large blind one.
#
# THE CONTROL is p3.0_medium, run under exactly this period and this level and differing
# only in the menu. Same 13 indoor episodes, same controller, same synchronous regime,
# archived alongside it so `sync_study_table.py` puts them in one matrix.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

ONLY_OVERRIDE=""; PREFLIGHT=1; LEVELS="medium high"
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY_OVERRIDE="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
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
export NAV_PLAN_MODE="sync"
# The one variable that makes this arm different from p3.0_*. Exported here and read at
# module scope by the policy server, which `run_level_ladder.sh` launches as a child, so
# it reaches the model through the environment and nowhere else -- which is exactly why
# the preflight reads it back off /health rather than trusting this line.
export QVLA_MENU_PIVOTS=1

# The same 13 indoor episodes every other arm ran. Derived rather than hardcoded so a
# reordering cannot silently drop one, and asserted at 13 so an addition cannot join.
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
# newline has one fewer newline than it has items, so `wc -l` reports 12 for 13.
N_EP=$(printf '%s' "$EPISODES" | awk -F, '{print NF}')

export QVLA_LADDER_ONLY="$EPISODES"

echo "================================================================"
echo "  in-place-turn study -- $(printf '%s' "$LEVELS" | wc -w) ladders x $N_EP episodes"
echo "  synchronous, period ${PERIOD}s, control = p${PERIOD}_medium (9/13)"
echo "  results: $STUDY"
echo "================================================================"
printf '  %s\n' $(printf '%s' "$EPISODES" | tr ',' ' ')
echo

# ---------------------------------------------------------------- preflight
# Two checks, and they answer different questions. The first is pure CPU and proves the
# menu is WELL FORMED -- labels unique, STOP moved, glyphs above the horizon, prompt
# naming the right two numbers. The second needs one real episode and proves the turn
# CROSSES the wire: the server's count of turns handed out has to equal the runner's count
# of turns performed.
#
# Neither is optional theatre. Every failure mode here yields thirteen complete episodes,
# and the two that matter most -- the server never seeing the flag, the runner ignoring the
# field -- are indistinguishable from "the model was offered the turn and did not want it",
# which is a real possible result. Cost: one model load and one 13 m episode against a
# ~3 h campaign.
if [ "$PREFLIGHT" = "1" ]; then
  echo "-- PREFLIGHT 1/2: the menu itself (CPU only)"
  /home/gtu-dsa/envs/qvla/bin/python nav/tools/check_pivot_menu.py \
      --out "$STUDY/pivot_menu_sample" > "$STUDY/preflight_menu.txt" 2>&1
  MENU_RC=$?
  cat "$STUDY/preflight_menu.txt"
  if [ "$MENU_RC" != "0" ]; then
    echo "!! the menu is malformed -- refusing to spend GPU time on it."
    python3 nav/tools/notify_run.py \
      --subject "[pivot-study] PREFLIGHT FAILED -- malformed menu" \
      --body "$(cat "$STUDY/preflight_menu.txt")" || true
    exit 1
  fi

  PRE_EP="office_hallway_turn2"
  echo
  echo "-- PREFLIGHT 2/2: $PRE_EP at medium, to prove the turn reaches the robot"
  QVLA_LADDER_ONLY="$PRE_EP" \
    nav/tools/run_level_ladder.sh medium --only "$PRE_EP" --tag "pvpre" \
      > "$STUDY/preflight_pivot.log" 2>&1
  PRE_RC=$?
  # Written, then shown. Not piped into `tee`: the exit status is the whole point of the
  # call and a pipeline would report the pager's instead.
  python3 nav/tools/check_pivot_run.py --episode "$PRE_EP" \
      > "$STUDY/preflight_pivot_check.txt" 2>&1
  CHECK_RC=$?
  cat "$STUDY/preflight_pivot_check.txt"
  if [ "$PRE_RC" != "0" ] || [ "$CHECK_RC" != "0" ]; then
    echo "!! preflight failed (ladder rc=$PRE_RC, check rc=$CHECK_RC) -- refusing to run"
    echo "!! ladders that would be labelled with an action the robot cannot take."
    python3 nav/tools/notify_run.py \
      --subject "[pivot-study] PREFLIGHT FAILED -- campaign not started" \
      --body "$(cat "$STUDY/preflight_pivot_check.txt")

ladder rc=$PRE_RC, check rc=$CHECK_RC
log: $STUDY/preflight_pivot.log" || true
    exit 1
  fi
  echo "-- preflight passed"
  echo
fi

# ---------------------------------------------------------------- the ladders
for LEVEL in $LEVELS; do
  NAME="pivot${PERIOD}_${LEVEL}"
  DEST="$STUDY/$NAME"

  # A re-run must not be scored against the run it replaces: the archive keeps every JSON
  # it is given and on_episode.sh APPENDS to the progress file, so a second campaign into
  # the same paths reads as one campaign of 26 episodes, 13 of them stale. Moved aside
  # rather than deleted -- a partial condition is still the only copy of what it measured.
  if [ -d "$DEST" ] && [ -n "$(ls -1 "$DEST"/*.json 2>/dev/null)" ]; then
    BAK="$DEST.superseded-$(date +%Y%m%d-%H%M%S)"
    echo "-- $NAME already holds results; moving them to $(basename "$BAK")"
    mv "$DEST" "$BAK"
  fi
  mkdir -p "$DEST"
  rm -f "nav/results/ladder_progress_${LEVEL}_pv.log"

  # Results are written into nav/results/ and COPIED here, selected by mtime against this
  # marker. Deliberately not redirected at the source: bench.sh counts files there before
  # and after each episode to catch a run that produced no measurement, and pointing that
  # glob elsewhere would disarm a check that exists because a ladder once scored 13
  # episodes off files up to two weeks old.
  MARK="$DEST/.started"
  : > "$MARK"

  echo "================================================================"
  echo "  CONDITION $NAME  --  turns on the menu, period ${PERIOD}s, level $LEVEL"
  echo "  started $(date -Is)"
  echo "================================================================"

  nav/tools/run_level_ladder.sh "$LEVEL" --only "$EPISODES" --tag "pv" \
      > "$DEST/ladder.log" 2>&1 &
  LADDER_PID=$!

  # One reporter per condition, so the tag in its subject and the progress file it reads
  # are always the condition actually running. It exits by itself when the ladder pid goes
  # away, sending a last mail as it does.
  QVLA_RUN_TAG="${LEVEL}_pv" \
    setsid nohup nav/tools/hourly_report.sh 3600 "$LADDER_PID" \
      > "$DEST/hourly.log" 2>&1 &

  wait "$LADDER_PID"
  RC=$?
  echo "-- condition $NAME finished (rc=$RC) at $(date -Is)"

  # `-newer` against the marker rather than a name pattern: the filenames carry only a
  # timestamp and the episode, nothing that says which condition produced them.
  find nav/results -maxdepth 1 -name '*.json' -newer "$MARK" \
      -exec cp -p {} "$DEST/" \; 2>/dev/null
  cp -p "nav/results/ladder_progress_${LEVEL}_pv.log" "$DEST/" 2>/dev/null
  cp -p "nav/results/logs/server_qwen_${LEVEL}.log" "$DEST/server.log" 2>/dev/null
  find nav/results/videos -maxdepth 1 -name "*__${LEVEL}_pv.mp4" -newer "$MARK" \
      -exec cp -p {} "$DEST/" \; 2>/dev/null
  N_JSON=$(ls -1 "$DEST"/*.json 2>/dev/null | wc -l)
  echo "-- archived $N_JSON result files into $DEST"

  # Whether the action was USED, straight off the server that just ran the ladder, before
  # the next condition restarts it and resets the counters. This is the number that says
  # whether a null result means "turning did not help" or "the model never turned", and
  # those have completely different next steps.
  curl -sf --max-time 10 "http://127.0.0.1:${NAV_POLICY_PORT}/health" \
      > "$DEST/health.json" 2>/dev/null
  TURNS="$(python3 - "$DEST" <<'PY'
import json, pathlib, sys
dest = pathlib.Path(sys.argv[1])
try:
    h = json.load(open(dest / "health.json"))
except Exception as exc:
    print(f"(no /health captured: {exc})")
    h = {}
served = h.get("pivots")
done = took = 0
for f in dest.glob("*.json"):
    if f.name == "health.json":
        continue
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    if "pivots" not in r:
        continue
    done += 1
    took += int(r["pivots"])
print(f"turns served={served} executed={took} over {done} episodes "
      f"({took / done:.1f} per episode)" if done else
      f"turns served={served}, no episode reported a count")
PY
)"
  echo "-- $TURNS"

  python3 nav/tools/sync_study_table.py > "$STUDY/comparison.txt" 2>&1
  python3 nav/tools/notify_run.py \
    --subject "[pivot-study] $NAME done ($N_JSON runs, rc=$RC)" \
    --body "Condition $NAME finished at $(date -Is).
in-place turns ON the menu, synchronous, period ${PERIOD}s, level $LEVEL, $N_EP episodes.

Was the new action used at all (0 would mean the prompt, not the action, is the problem):
  $TURNS

$(cat "$STUDY/comparison.txt")

archived to: $DEST" || echo "!! condition mail failed (continuing)"
done

echo "================================================================"
echo "  pivot study complete at $(date -Is)"
echo "================================================================"
python3 nav/tools/sync_study_table.py --verbose | tee "$STUDY/comparison.txt"
python3 nav/tools/notify_run.py \
  --subject "[pivot-study] ALL PIVOT LADDERS DONE" \
  --body "$(cat "$STUDY/comparison.txt")

Full results: $STUDY" || true
