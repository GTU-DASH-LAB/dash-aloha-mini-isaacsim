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
# ATOMIC. All four steps write to a temporary file next to the real one, and it is moved
# into place only after the last step exits 0. The first outdoor build (2026-09-01)
# segfaulted in step 1 AFTER saving the layer; `set -e` then skipped steps 2-4, which is
# right, but the 1.1 KB stub was already sitting at nav_outdoor_small.usda, and nav/bench.sh
# treats any stage file newer than the config as cached. Every outdoor run until the
# 09-10 rebuild therefore drove a different robot -- hull collider on, mesh wheels, no
# joint drives, no cameras -- with nothing anywhere saying so. A build that fails now
# leaves the previous stage (or none) exactly as it was.
#
# "FAILS" CANNOT MEAN "EXITS NON-ZERO". Kit's python.sh exits 0 after an uncaught Python
# exception and after kit.close(), whatever came first (measured 2026-10-05: both rc=0);
# only a signal death like the 09-01 segfault ever reached `set -e`. So a step that raised,
# or warned and skipped a prim it could not find, still got published. Two checks now
# stand between a step and the publish, and neither trusts an exit status:
#   * each step must print "PIPELINE_STEP_OK <script>" -- its last line before
#     kit.close() -- or the build stops there;
#   * before the rename, nav/sim/check_nav_stage.py reads the finished file and requires
#     every step's artefacts (robot + PhysicsScene, 16 joint drives, 3 wheel spheres +
#     friction material, every camera + the headlight). nav/bench.sh runs the same check
#     on a cached stage before trusting it.
#
# Usage:  nav/sim/build_nav_scene.sh [environment]   (default: warehouse)
#         NAV_SCENE_OUT=/abs/path.usda nav/sim/build_nav_scene.sh hospital   # build elsewhere
set -euo pipefail

ENV_NAME="${1:-warehouse}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Resolved through nav/paths.py so a clone elsewhere only has to set ISAACSIM_ROOT
# (or ISAACSIM_PYTHON) instead of editing this script. Two steps, not
# `${X:-$(cmd)}`: that form hides a failing resolve from `set -e` and would run
# the pipeline below with an empty interpreter path.
if [ -z "${ISAACSIM_PYTHON:-}" ]; then
  ISAACSIM_PYTHON="$(python3 "$REPO/nav/paths.py" --isaacsim-python)"
fi
PYTHON_SH="$ISAACSIM_PYTHON"

# Resolve the name here as well as inside the Python: the three pipeline steps below
# each run in their own process and need the .usda path.
ENV_NAME="$(python3 "$REPO/nav/sim/resolve_env.py" "$ENV_NAME")" || exit 1
SCENE="${NAV_SCENE_OUT:-$REPO/assets/usd/nav_${ENV_NAME}.usda}"
# Absolute, because the steps run after `cd "$REPO"` and must open the stage by absolute
# path anyway (a relative --scene once dropped the robot reference without an error).
SCENE="$(cd "$(dirname "$SCENE")" && pwd)/$(basename "$SCENE")"
# Same directory as the final file, for two reasons: `mv` within one filesystem is an
# atomic rename, and any relative asset path a step authors resolves identically from
# the temporary name and the final one. Hidden and named so it is obviously not a stage.
BUILD="$(dirname "$SCENE")/.$(basename "$SCENE" .usda).build-$$.usda"
STEPLOG="$(dirname "$SCENE")/.$(basename "$SCENE" .usda).build-$$.log"
rm -f "$BUILD" "$STEPLOG"
trap 'rm -f "$BUILD" "$STEPLOG"' EXIT

if [ ! -x "$PYTHON_SH" ]; then
  echo "ERROR: Isaac Sim 6.0.1 python.sh not found at $PYTHON_SH" >&2
  echo "       (Isaac Sim 6.x pairs with Python 3.12 -- do NOT point this at 5.0.0)" >&2
  exit 1
fi

cd "$REPO"

# Run one step, show its output as it comes, and fail unless it both exited 0 and printed
# its sentinel. Output goes through tee to a log so the sentinel can be checked; the
# step's own status is read from PIPESTATUS, never from tee's.
run_step() {
  local name="$1"; shift
  set +e
  "$PYTHON_SH" "$@" 2>&1 | tee "$STEPLOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: $name exited $rc -- stage NOT published" >&2
    exit 1
  fi
  if ! grep -qx "PIPELINE_STEP_OK $name" "$STEPLOG"; then
    echo "ERROR: $name exited 0 but never printed 'PIPELINE_STEP_OK $name' -- it did not" >&2
    echo "       reach its end (Kit exits 0 after an uncaught exception). Stage NOT published." >&2
    exit 1
  fi
}

echo "############ 1/4  author nav_${ENV_NAME}.usda ############"
run_step build_nav_scene nav/sim/build_nav_scene.py --env "$ENV_NAME" --out "$BUILD"

for step in configure_physics fix_wheel_collision add_cameras; do
  case "$step" in
    configure_physics)   n="2/4" ;;
    fix_wheel_collision) n="3/4" ;;
    add_cameras)         n="4/4" ;;
  esac
  echo
  echo "############ $n  $step ############"
  run_step "$step" "scripts/pipeline/${step}.py" --scene "$BUILD"
done

echo
echo "############ check  every step's artefacts are in the file ############"
if ! python3 nav/sim/check_nav_stage.py "$BUILD"; then
  echo "ERROR: the built stage is missing a step's work -- stage NOT published" >&2
  exit 1
fi

# Only reached when all four steps finished and the result checked out. The rename is
# the publish.
mv -f "$BUILD" "$SCENE"
rm -f "$STEPLOG"
trap - EXIT

echo
echo "Nav scene ready: $SCENE"
echo "Serves every $ENV_NAME episode; the runner teleports to whichever you ask for."
echo "Run it with:     nav/run.sh --episode <name>"
