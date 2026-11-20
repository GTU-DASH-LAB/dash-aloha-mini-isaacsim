#!/usr/bin/env bash
# Re-run the closed-loop comparison of the MIDI 2026 QVLA paper on ONE stack.
#
#   nav/tools/run_camera_ready.sh                       # 3 rounds x (qvla ticvla straight)
#   ROUNDS=1 ARMS=straight TAG_PREFIX=smoke nav/tools/run_camera_ready.sh \
#       --only office_nearest_elevator                  # a smoke test
#
# WHY. Both reviewers' main objection was that TIC-VLA had been measured once, on an older
# stack than QVLA. Here every arm runs on the same commit, the same 13 indoor episodes and
# the same controller, guard, contact layer and recovery reverse, three times each.
# The arms:
#   qvla      the paper's policy: baseline profile, frozen Qwen3.8-27B-FP8 (GPU1)
#   ticvla    TIC-VLA, official checkpoint, asynchronous as DynaNav runs it (GPU1)
#   straight  no model: the straightest drawn arc at cruise speed, every decision
#             (QVLA_SELECTOR=straight in server_qwen.py)
#
# The rounds are INTERLEAVED (qvla, ticvla, straight, then again), not blocked. Anything
# that drifts over a 12-hour campaign, such as GPU clocks or another user's job on the
# machine, then lands on every arm alike, not on whichever arm happened to run last.
#
# Each ladder gets its own server, started and checked here, and its own tag,
# <prefix>_<arm>_r<round>. Every result file records the tag under config.env
# (QVLA_RUN_TAG). Results go to nav/results/ like every other run. One line per episode
# goes to nav/results/camera_ready_progress.log.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

ROUNDS="${ROUNDS:-3}"
ARMS="${ARMS:-qvla ticvla straight}"
TAG_PREFIX="${TAG_PREFIX:-cr}"
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

LOGDIR="$REPO/nav/results/logs"
mkdir -p "$LOGDIR"
PROGRESS="$REPO/nav/results/camera_ready_progress.log"
HOOK="$REPO/nav/tools/on_episode_quiet.sh"
say() { echo "$(date -Iseconds) $*" | tee -a "$PROGRESS"; }

# The baseline profile, simulator side included (NAV_LIDAR=0 and friends), so all three
# arms drive under identical sim settings. Not hand-typed; see profile.py for why.
eval "$(python3 nav/tools/profile.py env --sim)" || { echo "profile env failed" >&2; exit 1; }

# Matched on interpreter AND script via awk, never `pgrep -f`, which matches its own shell.
qwen_pids() { ps -eo pid,args | awk '$2 ~ /qvla\/bin\/python$/ && $3 == "server_qwen.py" {print $1}'; }
tic_pids()  { ps -eo pid,args | awk '$2 ~ /tic-vla\/bin\/python$/ && $3 == "server.py" {print $1}'; }
stop_servers() {
  local pids
  pids="$(qwen_pids) $(tic_pids)"
  [ -z "${pids// /}" ] && return 0
  kill $pids 2>/dev/null
  for _ in $(seq 1 90); do
    [ -z "$(qwen_pids)$(tic_pids)" ] && return 0
    command sleep 1
  done
  say "!! policy servers still alive: $(qwen_pids) $(tic_pids)"
  return 1
}

# The two qwen-server arms go through run_level_ladder.sh, which starts the server, reads
# think_level / arc_set / menu settings back off /health and refuses on any mismatch.
run_qwen_arm() {   # $1 = selector (vlm|straight), $2 = tag
  stop_servers || return 1
  QVLA_SELECTOR="$1" LADDER_ON_EPISODE="$HOOK" \
      nav/tools/run_level_ladder.sh medium --tag "$2" ${ONLY:+--only "$ONLY"}
}

run_ticvla_arm() {   # $1 = tag
  stop_servers || return 1
  local models
  models="$(python3 nav/paths.py --models)"
  # launch.sh runs under `set -u` and names these two, so they must be set even though
  # server.py resolves the same defaults on its own.
  export TICVLA_BASE_MODEL_PATH="${TICVLA_BASE_MODEL_PATH:-$models/InternVL3-1B}"
  export TICVLA_CHECKPOINT_PATH="${TICVLA_CHECKPOINT_PATH:-$models/TIC-VLA-model.ckpt}"
  NAV_POLICY_PORT=8765 nohup nav/policy_server/launch.sh \
      > "$LOGDIR/server_ticvla_$1.log" 2>&1 &
  local pid=$! up=""
  for _ in $(seq 1 90); do
    up="$(curl -sf --max-time 5 http://127.0.0.1:8765/health || true)"
    [ -n "$up" ] && break
    kill -0 "$pid" 2>/dev/null || { say "!! TIC-VLA server died on startup"; \
                                     tail -20 "$LOGDIR/server_ticvla_$1.log"; return 1; }
    command sleep 2
  done
  [ -n "$up" ] || { say "!! TIC-VLA server never answered"; return 1; }
  say "-- TIC-VLA server up: $up"
  QVLA_RUN_TAG="${TAG_PREFIX}_$1" NAV_POLICY_PORT=8765 \
      nav/bench.sh ${ONLY:+--only "$ONLY"} --on-episode "$HOOK"
}

say "== camera-ready campaign: rounds=$ROUNDS arms='$ARMS' prefix=$TAG_PREFIX" \
    "commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)" \
    "${ONLY:+only=$ONLY}"
for R in $(seq 1 "$ROUNDS"); do
  for ARM in $ARMS; do
    TAG="${ARM}_r${R}"
    say "== round $R arm $ARM (tag ${TAG_PREFIX}_${TAG})"
    case "$ARM" in
      qvla)     run_qwen_arm vlm "${TAG_PREFIX}_${TAG}" ;;
      straight) run_qwen_arm straight "${TAG_PREFIX}_${TAG}" ;;
      ticvla)   run_ticvla_arm "$TAG" ;;
      *) say "!! unknown arm $ARM"; continue ;;
    esac
    say "== done round $R arm $ARM (rc=$?)"
  done
done
stop_servers
say "== campaign finished"
