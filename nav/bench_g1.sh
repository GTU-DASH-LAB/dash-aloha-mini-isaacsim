#!/usr/bin/env bash
# Run DynaNav episodes with the walking Unitree G1 (nav/sim/run_navigation_g1.py), one process per
# episode -- the G1 counterpart of nav/bench.sh, and for the same reason: Isaac Sim cannot swap
# stages cleanly in one process, and a half-swapped stage fails as a navigation error, i.e. as a
# plausible number.
#
#   nav/bench_g1.sh --only hospital_vending_machine                    # real policy (:8766), p12 camera
#   nav/bench_g1.sh --only hospital_vending_machine --camera real      # the real 47.6 deg mount
#   nav/bench_g1.sh --only hospital_vending_machine,hospital_down_hallway \
#       --policy stub --results-dir /tmp/g1stub                        # no model: scripted arcs
#
# flags:  --only A,B   --from A   --policy qwen|stub   --camera p12|real   --results-dir DIR
#         --stub-script NAME|FILE   --timeout S   --tag TAG   --video   --stop-on-fail
#
# Every run goes through nav/tools/isaac_slot.sh (GPU0 slot + GPU0/GPU1 memory gates; Kit is pinned
# to GPU0 by the runner itself). A --policy qwen run ALSO holds the policy lock
# ($NAV_POLICY_LOCK, default /tmp/claude-1000/isaac-slots/policy8766.lock) for its whole duration:
# other workflows drive the same 27B server, and two episodes interleaving on it would each be
# measured against the other's KV cache and recording directory. flock WAITS; it never breaks it.
#
# THE VERDICT IS THE NEW RESULT FILE, not the exit status: Isaac's python.sh exits 0 whatever
# happened. Each episode must add exactly one <stamp>_<episode>_g1pursuit.json to the results
# directory, or it is an ERROR -- the guard bench.sh learned when a dead setup() scored a whole
# ladder off two-week-old files. The controller name "g1pursuit" also keeps a G1 row from ever
# matching bench.sh's "*_<episode>_braking.json" glob while an AlohaMini ladder runs.
# Stub runs must name a --results-dir (never nav/results: they are not policy measurements).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

CONTROLLER="g1pursuit"
POLICY="qwen"; CAMERA="p12"; ONLY=""; FROM=""; RESULTS=""; STUB_SCRIPT="exercise"; TIMEOUT=""
TAG="g1nav"; VIDEO=0; KEEP_GOING=1
while [ $# -gt 0 ]; do
  case "$1" in
    --only)        ONLY="$2"; shift 2 ;;
    --from)        FROM="$2"; shift 2 ;;
    --policy)      POLICY="$2"; shift 2 ;;
    --camera)      CAMERA="$2"; shift 2 ;;
    --results-dir) RESULTS="$2"; shift 2 ;;
    --stub-script) STUB_SCRIPT="$2"; shift 2 ;;
    --timeout)     TIMEOUT="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    --video)       VIDEO=1; shift ;;
    --stop-on-fail) KEEP_GOING=0; shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
case "$POLICY" in qwen|stub) ;; *) echo "--policy must be qwen or stub" >&2; exit 2 ;; esac
case "$CAMERA" in p12|real) ;; *) echo "--camera must be p12 or real" >&2; exit 2 ;; esac
if [ -z "$RESULTS" ]; then
  if [ "$POLICY" = "stub" ]; then
    echo "--policy stub needs --results-dir (a stub run must not land in nav/results)" >&2; exit 2
  fi
  RESULTS="$REPO/nav/results"
fi
mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"
LOGDIR="$RESULTS/logs"; mkdir -p "$LOGDIR"

CONFIG="${NAV_EPISODES_CONFIG:-episodes.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$REPO/nav/config/$CONFIG" ;; esac
[ -f "$CONFIG" ] || { echo "no episode table at $CONFIG" >&2; exit 2; }
export NAV_EPISODES_CONFIG="$CONFIG"

mapfile -t EPISODES < <(python3 - "$CONFIG" "$ONLY" "$FROM" <<'PY'
import sys, yaml
names = list(yaml.safe_load(open(sys.argv[1]))["episodes"])
only = [s for s in sys.argv[2].split(",") if s]
if only:
    missing = [s for s in only if s not in names]
    if missing:
        sys.exit(f"no such episode(s): {', '.join(missing)}")
    names = [n for n in names if n in only]
if sys.argv[3]:
    if sys.argv[3] not in names:
        sys.exit(f"no such episode: {sys.argv[3]}")
    names = names[names.index(sys.argv[3]):]
print("\n".join(names))
PY
) || { echo "could not read episode list" >&2; exit 1; }

if [ -z "${ISAACSIM_PYTHON:-}" ]; then
  ISAACSIM_PYTHON="$(python3 "$REPO/nav/paths.py" --isaacsim-python)" || exit 1
fi
POLICY_PORT="${NAV_POLICY_PORT:-8766}"
POLICY_LOCK="${NAV_POLICY_LOCK:-/tmp/claude-1000/isaac-slots/policy8766.lock}"

echo "G1 ladder: ${#EPISODES[@]} episode(s), policy=$POLICY camera=$CAMERA controller=$CONTROLLER"
printf '  %s\n' "${EPISODES[@]}"
echo "  results -> $RESULTS"
if [ "$POLICY" = "qwen" ]; then
  H="$(curl -sf --max-time 10 "http://127.0.0.1:${POLICY_PORT}/health" || true)"
  if [ -z "$H" ]; then
    echo "!! nothing answers on :${POLICY_PORT} -- start the arc-menu server first" >&2; exit 1
  fi
  echo "  policy on :${POLICY_PORT} -- $(printf '%s' "$H" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print("  ".join(f"{k}={h[k]}" for k in ("model", "format", "think_level", "horizon_s", "menu_speed_mps",
                                        "menu_geom", "run_tag") if k in h))')"
  echo "  (the server must understand the /predict \"camera\" field, or it draws AlohaMini geometry)"
  echo "  policy lock: $POLICY_LOCK"
fi
echo

PASS=0; FAIL=0; ERR=0
for EP in "${EPISODES[@]}"; do
  echo "================================================================"
  echo "  $EP  (g1, $POLICY, camera $CAMERA)"
  echo "================================================================"
  ENV_NAME="$(python3 nav/sim/resolve_env.py "$EP")" || { ERR=$((ERR+1)); continue; }
  SCENE="$REPO/assets/usd/nav_${ENV_NAME}.usda"
  if [ ! -f "$SCENE" ]; then
    echo "!! no stage $SCENE -- build it with nav/sim/build_nav_scene.sh $ENV_NAME (not done here)"
    ERR=$((ERR+1)); [ $KEEP_GOING -eq 1 ] && continue || exit 1
  fi
  if ! python3 nav/sim/check_nav_stage.py "$SCENE" > "$LOGDIR/stagecheck_${ENV_NAME}.log" 2>&1; then
    echo "!! CACHED STAGE INCOMPLETE -- refusing to run on it:"; cat "$LOGDIR/stagecheck_${ENV_NAME}.log"
    ERR=$((ERR+1)); [ $KEEP_GOING -eq 1 ] && continue || exit 1
  fi

  LOG="$LOGDIR/run_${EP}_${CONTROLLER}_${CAMERA}_$(date +%Y%m%d-%H%M%S).log"
  BEFORE=$(ls -1 "$RESULTS"/*_"${EP}_${CONTROLLER}".json 2>/dev/null | wc -l)
  CMD=(env ISAAC_SLOTS="${ISAAC_SLOTS:-8}" ISAAC_SLOT_MIN_FREE_MIB="${ISAAC_SLOT_MIN_FREE_MIB:-8000}"
       ISAAC_SLOT_MEM_TIMEOUT_S="${ISAAC_SLOT_MEM_TIMEOUT_S:-5400}" ISAAC_SLOT_TAG="${TAG}-${EP}"
       nav/tools/isaac_slot.sh "$ISAACSIM_PYTHON" nav/sim/run_navigation_g1.py
       --episode "$EP" --camera "$CAMERA" --policy "$POLICY" --policy-port "$POLICY_PORT"
       --results-dir "$RESULTS" --scratch-dir "$RESULTS/frames/${EP}_${CAMERA}")
  [ "$POLICY" = "stub" ] && CMD+=(--stub-script "$STUB_SCRIPT" --record-root "$RESULTS/runs")
  [ -n "$TIMEOUT" ] && CMD+=(--timeout "$TIMEOUT")
  echo "-- running ($(date +%H:%M:%S)); log $LOG"
  T0=$(date +%s)
  if [ "$POLICY" = "qwen" ]; then
    mkdir -p "$(dirname "$POLICY_LOCK")"
    flock "$POLICY_LOCK" "${CMD[@]}" > "$LOG" 2>&1
  else
    "${CMD[@]}" > "$LOG" 2>&1
  fi
  RC=$?
  AFTER=$(ls -1 "$RESULTS"/*_"${EP}_${CONTROLLER}".json 2>/dev/null | wc -l)
  STATUS_LINE="$(grep -a '^\[g1nav\] STATUS=' "$LOG" | tail -n 1)"
  echo "   $(( $(date +%s) - T0 )) s wall, rc=$RC; ${STATUS_LINE:-no STATUS line}"
  if [ "$AFTER" -ne $((BEFORE + 1)) ]; then
    echo "!! NO NEW RESULT ($BEFORE -> $AFTER files) -- not scoring this episode; see $LOG"
    grep -a -m1 -A8 "Traceback (most recent call last)" "$LOG" | sed 's/^/   /'
    ERR=$((ERR+1)); [ $KEEP_GOING -eq 1 ] && { echo; continue; } || exit 1
  fi
  NEWEST="$(ls -1t "$RESULTS"/*_"${EP}_${CONTROLLER}".json | head -n 1)"
  VERDICT="$(python3 - "$NEWEST" <<'PY'
import json, math, sys
d = json.load(open(sys.argv[1]))
g = d.get("g1") or {}
goal = None
tr = d.get("trace") or []
v = "SUCCESS" if d.get("success") else (d.get("abort_reason") or ("TIMEOUT" if d.get("timed_out") else "FAILED")).upper()
sp = g.get("speed_tracking") or {}
print(f"{v}  {d['initial_distance_m']:.2f} -> {d['final_distance_m']:.2f} m, path {d['path_length_m']:.1f} m in "
      f"{d['elapsed_s']:.1f} s sim, calls {d['policy_calls']}, guard {d['guard_interventions']}, "
      f"turns {len(g.get('recovery_turns') or [])}, pivots {(g.get('controller') or {}).get('pivots')}, "
      f"falls {int(bool((g.get('base') or {}).get('fell')))}, walk {sp.get('achieved_mps')}/{sp.get('commanded_mps')} m/s, "
      f"wall/sim {(g.get('cost') or {}).get('wall_per_sim_s')}  [{sys.argv[1].rsplit('/', 1)[-1]}]")
PY
)"
  echo "   $VERDICT"
  case "$VERDICT" in SUCCESS*) PASS=$((PASS+1)) ;; *) FAIL=$((FAIL+1)) ;; esac
  if [ $VIDEO -eq 1 ]; then
    python3 nav/tools/g1_nav_video.py --result "$NEWEST" > "$LOGDIR/video_${EP}.log" 2>&1 \
      && echo "   video: $(tail -n 1 "$LOGDIR/video_${EP}.log")" || echo "   !! video build failed (see $LOGDIR/video_${EP}.log)"
  fi
  echo
done
echo "================================================================"
echo "  G1 ladder done: $PASS succeeded, $FAIL failed, $ERR errored"
echo "================================================================"
[ $ERR -eq 0 ]
