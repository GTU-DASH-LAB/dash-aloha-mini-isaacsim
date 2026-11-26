#!/usr/bin/env bash
# Run the two arc-menu probes against each candidate VLM, and against the live baseline.
#
# The probes are the benchmark and they are not modified. Each candidate gets a
# `server_vlm.py` on GPU0 at port 8767; the 27B baseline stays where it is on GPU1 at
# 8766 and is scored through the same probes in the same session, so the reference row is
# measured here rather than quoted from a document written weeks ago.
#
# Ordering is smallest-first to match the download order -- a candidate that has not
# arrived yet is SKIPPED with a line saying so, never silently omitted, so a five-row
# table cannot turn into a four-row table without the reason being visible.
#
# Process hygiene, because this repo has paid for it twice: the server PID comes from `$!`
# and is killed by number. No `pkill -f` -- its pattern matches the invoking shell's own
# command line, which has already killed this shell three times in one session.
#
# Usage:
#   nav/tools/bench_vlm_ladder.sh                 # baseline + all five
#   nav/tools/bench_vlm_ladder.sh qwen3vl-4b      # just these

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/gtu-dsa/envs/qvla/bin/python
OUT="${VLM_BENCH_OUT:-$REPO/nav/results/vlm_survey}"
PORT="${VLM_BENCH_PORT:-8767}"
BASE_PORT="${VLM_BENCH_BASE_PORT:-8766}"
LOGS="$OUT/logs"
ALL=(smolvlm2-2.2b qwen3vl-4b gemma3-4b internvl3.5-4b qwen3vl-8b)
CANDS=("${@:-}")
[ -z "${CANDS[0]:-}" ] && CANDS=("${ALL[@]}")

mkdir -p "$OUT" "$LOGS"
cd "$REPO"

# One probe pair against whatever is listening on $1, tagged $2.
run_probes() {
  local port="$1" tag="$2"
  echo "    probe_arc_selection ..."
  "$PY" nav/tools/probe_arc_selection.py --port "$port" --scenes 8 --perms 3 \
      --json-out "$OUT/${tag}__selection.json" > "$LOGS/${tag}__selection.log" 2>&1
  local a=$?
  echo "    probe_arc_repair ..."
  "$PY" nav/tools/probe_arc_repair.py --port "$port" --perms 3 \
      --json-out "$OUT/${tag}__repair.json" > "$LOGS/${tag}__repair.log" 2>&1
  local b=$?
  # Report both, not their conjunction: one probe failing and one succeeding is a
  # different situation from both failing, and `&&` would erase the distinction.
  echo "    selection=$a repair=$b"
  [ $a -eq 0 ] && [ $b -eq 0 ]
}

# Poll /health until the model is loaded. Bounded, because a server that died at import
# time never answers and an unbounded loop would hang the whole ladder on it.
wait_healthy() {
  local port="$1" pid="$2" i
  for i in $(seq 1 240); do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/health" | grep -q '"ok":true'; then
      return 0
    fi
    # A dead server cannot become healthy; stop waiting the moment it exits rather than
    # burning the full four minutes to reach the same conclusion.
    kill -0 "$pid" 2>/dev/null || return 1
    command sleep 2
  done
  return 1
}

# The baseline row can be scored before a single candidate has finished downloading, and
# then not re-scored when the candidates run. Kept as an opt-out rather than a default so
# that a plain invocation always produces a table with something to compare against.
echo "=== baseline: Qwen3.8-27B-FP8 on :$BASE_PORT (already running, GPU1)"
if [ -n "${VLM_BENCH_SKIP_BASELINE:-}" ]; then
  echo "    skipped (VLM_BENCH_SKIP_BASELINE set) -- reusing $OUT/baseline-qwen27b__*.json"
elif curl -sf --max-time 5 "http://127.0.0.1:$BASE_PORT/health" > "$OUT/baseline__health.json"; then
  run_probes "$BASE_PORT" "baseline-qwen27b" || echo "    BASELINE PROBES FAILED"
else
  echo "    no baseline on :$BASE_PORT -- skipping the reference row"
fi

for name in "${CANDS[@]}"; do
  dir="/home/gtu-dsa/robotics/models/$name"
  echo
  echo "=== $name"
  if [ ! -f "$dir/config.json" ]; then
    echo "    SKIP -- not downloaded yet ($dir)"
    continue
  fi

  CUDA_VISIBLE_DEVICES=0 "$PY" nav/policy_server/server_vlm.py \
      --candidate "$name" --port "$PORT" > "$LOGS/${name}__server.log" 2>&1 &
  pid=$!
  if wait_healthy "$PORT" "$pid"; then
    curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" > "$OUT/${name}__health.json"
    echo "    loaded: $(cat "$OUT/${name}__health.json")"
    run_probes "$PORT" "$name" || echo "    PROBES FAILED for $name"
  else
    echo "    LOAD FAILED -- last 15 lines of $LOGS/${name}__server.log:"
    tail -15 "$LOGS/${name}__server.log" | sed 's/^/      /'
  fi

  kill "$pid" 2>/dev/null
  # Give the card back before the next load, or candidate N+1 OOMs against N's weights
  # and the failure gets attributed to the wrong model.
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; command sleep 1; done
  kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  command sleep 3
done

echo
echo "=== done. results in $OUT"
