#!/usr/bin/env bash
# Build a navigation scene end to end:  author layer -> drives -> colliders -> cameras
#
# Each step is its own process because SimulationApp.close() takes the interpreter
# down with it -- code placed after it never runs (found the hard way; the authoring
# script silently "succeeded" while applying none of the pipeline). Same reason
# scripts/rebuild_all.sh chains separate invocations.
#
# The three pipeline steps are the SAME ones the pick-and-place scene uses, just
# pointed at a different --scene. A freshly authored layer has no joint drives, no
# wheel colliders and no cameras: those live as overrides in the scene file, not in
# Aloha.usda. Skip them and you get a robot whose joints do nothing.
#
# One stage per ENVIRONMENT, not per episode. The runner teleports to the episode's
# start pose before every run, so six hospital episodes share one nav_hospital.usda
# instead of costing six of these chains. An episode name is accepted too and resolves
# to its environment, so an old command line still builds the right file.
#
# Usage:  nav/sim/build_nav_scene.sh [environment]   (default: warehouse)
set -euo pipefail

ENV_NAME="${1:-warehouse}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_SH="${ISAACSIM_PYTHON:-/home/gtu-dsa/robotics/isaacsim-6.0.1/python.sh}"

# Resolve the name here as well as inside the Python: the three pipeline steps below
# each run in their own process and need the .usda path.
ENV_NAME="$(python3 "$REPO/nav/sim/resolve_env.py" "$ENV_NAME")" || exit 1
SCENE="$REPO/assets/usd/nav_${ENV_NAME}.usda"

if [ ! -x "$PYTHON_SH" ]; then
  echo "ERROR: Isaac Sim 6.0.1 python.sh not found at $PYTHON_SH" >&2
  echo "       (Isaac Sim 6.x pairs with Python 3.12 -- do NOT point this at 5.0.0)" >&2
  exit 1
fi

cd "$REPO"

echo "############ 1/4  author nav_${ENV_NAME}.usda ############"
"$PYTHON_SH" nav/sim/build_nav_scene.py --env "$ENV_NAME"

for step in configure_physics fix_wheel_collision add_cameras; do
  case "$step" in
    configure_physics)   n="2/4" ;;
    fix_wheel_collision) n="3/4" ;;
    add_cameras)         n="4/4" ;;
  esac
  echo
  echo "############ $n  $step ############"
  "$PYTHON_SH" "scripts/pipeline/${step}.py" --scene "$SCENE"
done

echo
echo "Nav scene ready: $SCENE"
echo "Serves every $ENV_NAME episode; the runner teleports to whichever you ask for."
echo "Run it with:     nav/run.sh --episode <name>"
