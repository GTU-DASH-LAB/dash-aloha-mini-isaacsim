#!/usr/bin/env bash
# Launch the frozen-Qwen arc-menu policy server (its own venv, GPU1).
#
#   nav/policy_server/launch_qwen.sh                 # medium
#   nav/policy_server/launch_qwen.sh very_high       # a thinking level
#
# `launch.sh` next door is for TIC-VLA and cannot serve this one: it validates a
# DynaNav checkout and a 1.9 GB checkpoint that this policy has no use for, and it
# ends in `exec "$VENV_PY" server.py`. Two servers, two entry points.
#
# WHY THIS EXISTS AT ALL RATHER THAN AN ENV-VAR INCANTATION IN SOMEBODY'S SHELL.
# The same thirteen episodes get run once per thinking level, so this command is
# typed three times with one word different, and the one word decides what is being
# measured. A ladder tagged `very_high` against a server left at `medium` produces a
# complete, plausible, wrong set of numbers -- the failure mode this repo has already
# been burned by twice. Here the level is argument one, it is echoed at startup, and
# `/health` reports it back so `bench_status.py` can check the label against reality.
#
# This process must NEVER import Isaac Sim. That is what makes CUDA_VISIBLE_DEVICES
# safe to use here -- Kit warns that pinning it "can lead to undesired behavior or
# crashes", so the simulator pins its GPU through Kit's own config instead.
set -euo pipefail
cd "$(dirname "$0")"

# Argument one, first, so the profile below cannot overwrite it -- `--soft` emits
# set-if-unset assignments, so anything already exported at this point wins.
export QVLA_THINK_LEVEL="${1:-${QVLA_THINK_LEVEL:-medium}}"

# EVERY DEFAULT BELOW USED TO BE TYPED HERE. They now live in one file,
# `nav/config/profiles/baseline.yaml`, alongside the scores that configuration actually
# earned and the four arms that were measured against it and lost. Two copies of a
# configuration is how a 19-episode campaign came to run with the memory frame off while
# testing the memory frame: the value was in the launcher, the experiment set it
# elsewhere, and nothing compared them. `nav/tools/profile.py check` closes that loop by
# reading /health back after startup.
#
# Override any of it the ordinary way -- `QVLA_MENU_FRAMES=2 launch_qwen.sh` still works,
# and now shows up as a deliberate deviation from a named profile rather than as an
# unremarkable env var. Pick a different profile with QVLA_PROFILE=<name>.
QVLA_PROFILE=${QVLA_PROFILE:-baseline}
# Two steps, not `eval "$(...)"`, and that is the whole point of the line. `set -e` does
# not see through a command substitution inside `eval`: a failed profile load yields an
# empty string, `eval ""` succeeds, and the script sails on to start a server with none
# of its configuration and no error -- which is the same silent-wrong-config failure this
# file was just rewritten to prevent. Assigning first makes the substitution's own exit
# status the assignment's, so a typo'd profile name stops here.
# Path is relative to the script's own directory, which the `cd` above made the cwd.
_PROFILE_ENV="$(python3 ../tools/profile.py env --soft --profile "$QVLA_PROFILE")"
eval "$_PROFILE_ENV"

VENV_PY=${VENV_PY:-/home/gtu-dsa/envs/qvla/bin/python}

# The card choice and the 30 GiB memory cap are pinned in the profile, not defaulted
# here, because both were wrong once: spanning the two cards routes FP8 off DeepGEMM
# onto Triton at ~4x the cost, and a device left in the map at 0 can still be spilled
# onto, which loses the DeepGEMM path for the whole model over a single layer.
export CUDA_VISIBLE_DEVICES=${NAV_POLICY_GPU:-1}

echo "Q-VLA arc-menu policy server"
echo "  profile    : $QVLA_PROFILE   (nav/tools/profile.py check -- after startup)"
echo "  python     : $VENV_PY"
echo "  model      : $QVLA_MODEL"
echo "  format     : $QVLA_FORMAT   speed cap ${QVLA_MENU_SPEED} m/s"
echo "  thinking   : $QVLA_THINK_LEVEL"
# Echoed because these two are the settings that have already cost a campaign: a whole
# 19-episode ladder ran at frames=1 while believing it was testing frames=2, and nothing
# on screen or in the results said otherwise.
echo "  menu       : frames ${QVLA_MENU_FRAMES}  pivots ${QVLA_MENU_PIVOTS}  arcs ${QVLA_MENU_ARCS}"
echo "  GPU        : physical ${CUDA_VISIBLE_DEVICES} (seen as cuda:0), ${QVLA_MAX_MEMORY}"
echo "  port       : $NAV_POLICY_PORT"
echo

exec "$VENV_PY" server_qwen.py
