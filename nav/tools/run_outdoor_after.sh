#!/usr/bin/env bash
# Wait for a running ladder to finish, then bring up the outdoor scene and run it.
#
#   setsid nohup nav/tools/run_outdoor_after.sh <pid_to_wait_for> \
#       > nav/results/logs/outdoor_chain.log 2>&1 &
#
# Three things have to happen in this order and none of them can happen while the
# three-level ladder is still driving:
#
#   1. `episodes.yaml` is REGENERATED, which is what pulls the six outdoor episodes out of
#      `episodes_manual.yaml` and into the ladder. It cannot be done sooner: `bench.sh`
#      rebuilds a stage whenever the config is newer than it, so regenerating mid-run
#      would fire an Isaac Sim build chain before every remaining episode, on the GPU the
#      run is already using.
#   2. `nav_outdoor_small.usda` is BUILT. First time this scene has ever been staged here.
#   3. The six episodes run, with the same per-episode video + mail hook as everything
#      else.
#
# The regeneration is verified rather than trusted: the generator ranks against DynaNav's
# result file, and if that file ever changes the ladder silently becomes a different set
# of episodes. So the previous config is kept and every episode name in it must still be
# present afterwards, or this stops without building anything.
#
# The gate before running the other two thinking levels is "did the scene RUN", not "did
# it succeed". A scene that drives and fails is a real measurement and worth having at all
# three levels; a scene that never built is not worth repeating twice more.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
export NAV_POLICY_PORT="${NAV_POLICY_PORT:-8766}"

WAIT_PID="${1:-}"
OUTDOOR="outdoor_umbrellas,outdoor_pillars,outdoor_library,outdoor_upsway,outdoor_upsway_far,outdoor_ramp_fountain"
LOGDIR="$REPO/nav/results/logs"; mkdir -p "$LOGDIR"

mail() { python3 nav/tools/notify_run.py --subject "$1" --body "$2" || true; }

if [ -n "$WAIT_PID" ]; then
  echo "-- waiting for pid $WAIT_PID (the thinking-level ladders) to finish"
  # `command sleep`, matching run_level_ladder.sh: a plain `sleep` is intercepted in this
  # environment. And `kill -0` on the pid, never `pgrep -f` -- a pattern naming this
  # script would match this script's own command line and wait forever.
  while kill -0 "$WAIT_PID" 2>/dev/null; do command sleep 60; done
  echo "-- pid $WAIT_PID is gone at $(date -Is); starting the outdoor scene"
fi

BACKUP="/tmp/episodes_before_outdoor.yaml"
cp nav/config/episodes.yaml "$BACKUP"
if ! python3 nav/sim/import_benchmark.py --write > "$LOGDIR/import_outdoor.log" 2>&1; then
  mail "[bench] outdoor: episode regeneration FAILED" "$(tail -30 "$LOGDIR/import_outdoor.log")"
  exit 1
fi

# Every episode that was in the ladder must still be in it, and all six new ones must have
# arrived. Either half failing means the generated config is not the one this was planned
# against, and building a stage off it would produce numbers for a different benchmark.
CHECK=$(python3 - "$BACKUP" "$OUTDOOR" <<'PY'
import sys, yaml
old = set(yaml.safe_load(open(sys.argv[1]))["episodes"])
new = set(yaml.safe_load(open("nav/config/episodes.yaml"))["episodes"])
lost = sorted(old - new)
missing = [e for e in sys.argv[2].split(",") if e not in new]
if lost:    print("LOST:", ", ".join(lost))
if missing: print("MISSING:", ", ".join(missing))
if not lost and not missing:
    print(f"OK {len(old)} -> {len(new)} episodes, all six outdoor present")
PY
)
echo "$CHECK"
case "$CHECK" in
  OK*) : ;;
  *) cp "$BACKUP" nav/config/episodes.yaml
     mail "[bench] outdoor: config check failed, reverted" "$CHECK"
     exit 1 ;;
esac

mail "[bench] outdoor scene starting" \
     "The three thinking-level ladders have finished. Regenerating the episode list added
the six outdoor episodes and lost none of the existing ones:

  $CHECK

Now building assets/usd/nav_outdoor_small.usda -- the first time this repo has ever
staged DynaNav's fourth scene -- and running:

  $OUTDOOR

Two caveats that belong with every number these produce. DynaNav spawns 100-200 animated
pedestrians in these episodes and our stage is static, so our street is empty: easier, in
exactly the dimension the benchmark is named after. And this is the first scene with
terrain -- one instruction is literally 'go down the ramp'. The base is a kinematic
teleport and the collision guard casts a horizontal ray fan; neither has been asked to
handle a slope before, so a failure here may be the base, not the policy."

echo "################  OUTDOOR @ medium  ################"
nav/tools/run_level_ladder.sh medium --only "$OUTDOOR" --tag outdoor \
    2>&1 | tee "$LOGDIR/ladder_outdoor_medium.log"
RC=${PIPESTATUS[0]}

SCORED=$(wc -l < nav/results/ladder_progress_medium_outdoor.log 2>/dev/null || echo 0)
echo "-- outdoor at medium: rc=$RC, $SCORED episodes scored"

if [ "$SCORED" -lt 1 ]; then
  mail "[bench] outdoor scene did NOT run" \
"The outdoor ladder scored no episodes (rc=$RC). Most likely the stage build failed --
this is the first time nav_outdoor_small.usda has ever been built. Not repeating it at
the other two thinking levels, because nothing would be measured.

Build log tail:
$(tail -25 "$LOGDIR/build_outdoor_small.log" 2>/dev/null || echo '(no build log)')

Ladder log tail:
$(tail -25 "$LOGDIR/ladder_outdoor_medium.log")"
  exit 1
fi

for LEVEL in high very_high; do
  echo "################  OUTDOOR @ $LEVEL  ################"
  nav/tools/run_level_ladder.sh "$LEVEL" --only "$OUTDOOR" --tag outdoor \
      2>&1 | tee "$LOGDIR/ladder_outdoor_${LEVEL}.log"
  echo "-- outdoor at $LEVEL finished with rc=${PIPESTATUS[0]}"
done

BODY=$(python3 - <<'PY'
import pathlib
out = []
for lvl in ("medium", "high", "very_high"):
    p = pathlib.Path(f"nav/results/ladder_progress_{lvl}_outdoor.log")
    if not p.is_file():
        out += [f"{lvl}: did not run", ""]; continue
    rows = [l.split("\t") for l in p.read_text().splitlines() if l.strip()]
    ok = sum(r[3].startswith("SUCCESS") for r in rows if len(r) > 3)
    out.append(f"{lvl}: {ok}/{len(rows)} succeeded")
    out += [f"   {r[1]:<24} {r[3][:110]}" for r in rows if len(r) > 3]
    out.append("")
out.append("Reminder: no DynaNav reference number exists for any outdoor episode, our "
           "street has no pedestrians in it, and this is the only scene with terrain.")
print("\n".join(out))
PY
) || BODY="(scoreboard failed)"
mail "[bench] outdoor scene: all thinking levels done" "$BODY"
