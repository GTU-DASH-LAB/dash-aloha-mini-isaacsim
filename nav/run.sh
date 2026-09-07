#!/usr/bin/env bash
# Run a navigation episode. Starts the policy server if it is not already up.
#
#   nav/run.sh                                   # warehouse, braking, UI on :8080
#   nav/run.sh --episode hospital_vending --gui  # with a window
#   nav/run.sh --controller pursuit --no-ui      # DynaNav-parity controller, CLI only
#
# TWO PROCESSES, and the split is forced rather than stylistic (see nav/plan.md):
#   policy  ~/envs/tic-vla/bin/python   py3.11  GPU1   TIC-VLA + InternVL3-1B
#   sim     isaacsim-6.0.1/python.sh    py3.12  GPU0   Isaac Sim + AlohaMini
# TIC-VLA needs 3.11 (Isaac Sim 5.0's NumPy 1.x ABI) and AlohaMini needs 3.12; on top
# of that the DynaNav office scene alone holds 25.4 GiB of GPU0's 31.35 GiB, so the
# checkpoint could not share that card even if the Pythons matched.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Resolved through nav/paths.py so a clone elsewhere only has to set ISAACSIM_ROOT
# (or ISAACSIM_PYTHON) instead of editing this script. Two steps, not
# `${X:-$(cmd)}`: that form hides a failing resolve from `set -e` and would run
# the pipeline below with an empty interpreter path.
if [ -z "${ISAACSIM_PYTHON:-}" ]; then
  ISAACSIM_PYTHON="$(python3 "$REPO/nav/paths.py" --isaacsim-python)"
fi
PYTHON_SH="$ISAACSIM_PYTHON"
POLICY_PORT="${NAV_POLICY_PORT:-8765}"
UI_PORT="${NAV_UI_PORT:-8080}"

EPISODE="warehouse"; CONTROLLER="braking"; GUI=""; USE_UI=1; INSTRUCTION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --episode)     EPISODE="$2"; shift 2 ;;
    --controller)  CONTROLLER="$2"; shift 2 ;;
    --instruction) INSTRUCTION="$2"; shift 2 ;;
    --gui)         GUI="--gui"; shift ;;
    --no-ui)       USE_UI=0; shift ;;
    --ui-port)     UI_PORT="$2"; shift 2 ;;
    -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Stages are per environment. Every episode in the same environment shares one, and
# once the session is up the UI can switch between all of them without a rebuild.
ENV_NAME="$(python3 "$REPO/nav/sim/resolve_env.py" "$EPISODE")" || exit 1
SCENE="$REPO/assets/usd/nav_${ENV_NAME}.usda"
if [ ! -f "$SCENE" ]; then
  echo "Nav scene missing for the '$ENV_NAME' environment (episode '$EPISODE')." >&2
  echo "Build it first:  nav/sim/build_nav_scene.sh $ENV_NAME" >&2
  exit 1
fi

# Reuse a policy server that is already up. Loading InternVL3-1B plus the 1.9 GB
# checkpoint takes ~3 s and ~2 GiB, so there is no reason to pay it twice -- and two
# servers on one port would just fail to bind.
if curl -sf "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null 2>&1; then
  echo "Policy server already up on :${POLICY_PORT}"
else
  echo "Starting policy server on :${POLICY_PORT} (GPU1)…"
  "$REPO/nav/policy_server/launch.sh" > "${TMPDIR:-/tmp}/nav-policy-server.log" 2>&1 &
  POLICY_PID=$!
  # Kill only the server WE started; leave a pre-existing one alone.
  trap 'kill '"$POLICY_PID"' 2>/dev/null || true' EXIT
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$POLICY_PID" 2>/dev/null; then
      echo "ERROR: policy server died on startup. Log:" >&2
      tail -30 "${TMPDIR:-/tmp}/nav-policy-server.log" >&2
      exit 1
    fi
    sleep 2
  done
fi

ARGS=(--episode "$EPISODE" --controller "$CONTROLLER" --policy-port "$POLICY_PORT")
[ -n "$GUI" ] && ARGS+=("$GUI")
[ -n "$INSTRUCTION" ] && ARGS+=(--instruction "$INSTRUCTION")
[ "$USE_UI" = "1" ] && ARGS+=(--ui-port "$UI_PORT")

cd "$REPO"
echo "Episode    : $EPISODE"
echo "Environment: $ENV_NAME  (the UI can switch to any episode in it)"
echo "Controller : $CONTROLLER"
[ "$USE_UI" = "1" ] && echo "UI         : http://127.0.0.1:${UI_PORT}"
echo

exec "$PYTHON_SH" nav/sim/run_navigation.py "${ARGS[@]}"
