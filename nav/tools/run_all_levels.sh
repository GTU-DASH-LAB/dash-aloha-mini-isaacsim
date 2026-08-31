#!/usr/bin/env bash
# Run the whole benchmark ladder once per thinking level, unattended, start to finish.
#
#   setsid nohup nav/tools/run_all_levels.sh > nav/results/logs/all_levels.log 2>&1 &
#   nav/tools/run_all_levels.sh high very_high        # a subset of levels, in order
#
# One level at a time, in the order given, each a full server restart plus all thirteen
# episodes. `run_level_ladder.sh` does one level and refuses to drive unless /health
# reports the level it was asked for; this only sequences them and refuses to let one bad
# level end the night.
#
# THAT LAST PART IS THE WHOLE POINT. A level can fail for reasons that say nothing about
# the other two -- a model reload that OOMs because something else grabbed GPU1, a scene
# rebuild, a wedged episode. Stopping there would throw away the levels not yet run, and
# the person who asked for this is asleep. So a failed level is recorded and the next one
# starts. The summary at the end names which levels ran and which did not, because "no
# mail arrived for very_high" and "very_high scored zero" must not look the same in the
# morning.
#
# Videos and per-episode mail come from the `--on-episode` hook inside each ladder, so
# results arrive through the night rather than in one batch at the end.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

LEVELS=("$@")
[ ${#LEVELS[@]} -eq 0 ] && LEVELS=(medium high very_high)

LOGDIR="$REPO/nav/results/logs"
mkdir -p "$LOGDIR"
STARTED="$(date -Is)"
declare -a OUTCOME

for LEVEL in "${LEVELS[@]}"; do
  echo
  echo "################################################################"
  echo "#  LEVEL: $LEVEL      ($(date -Is))"
  echo "################################################################"
  nav/tools/run_level_ladder.sh "$LEVEL" 2>&1 | tee "$LOGDIR/ladder_${LEVEL}.log"
  # The ladder's own status, not tee's. Piping into tee makes $? the pager's, which is the
  # exact mistake that once reported a failed Isaac Sim install as a success.
  RC=${PIPESTATUS[0]}
  OUTCOME+=("$LEVEL rc=$RC")
  echo "-- level $LEVEL finished with rc=$RC at $(date -Is)"
done

echo
echo "################################################################"
echo "#  ALL LEVELS DONE   started $STARTED   finished $(date -Is)"
printf '#    %s\n' "${OUTCOME[@]}"
echo "################################################################"

# One closing mail with every level's scoreboard side by side. The per-episode hook has
# already sent thirteen mails per level; this is the only place the three are comparable,
# which is the question the whole exercise was set up to answer.
BODY=$(python3 - "${LEVELS[@]}" <<'PY'
import pathlib, sys
out = []
for lvl in sys.argv[1:]:
    p = pathlib.Path(f"nav/results/ladder_progress_{lvl}.log")
    if not p.is_file():
        out += [f"{lvl}: no progress file -- this level did not run", ""]
        continue
    rows = [l.split("\t") for l in p.read_text().splitlines() if l.strip()]
    ok = sum(r[3].startswith("SUCCESS") for r in rows if len(r) > 3)
    out.append(f"{lvl}: {ok}/{len(rows)} succeeded")
    for r in rows:
        if len(r) > 3:
            out.append(f"   {r[1]:<30} {r[3][:110]}")
    out.append("")
print("\n".join(out))
PY
) || BODY="(could not build the scoreboard)"
python3 nav/tools/notify_run.py \
  --subject "[bench] all thinking levels finished: ${LEVELS[*]}" --body "$BODY" || true
