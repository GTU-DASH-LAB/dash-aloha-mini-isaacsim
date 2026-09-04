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

export QVLA_THINK_LEVEL="${1:-${QVLA_THINK_LEVEL:-medium}}"

VENV_PY=${VENV_PY:-/home/gtu-dsa/envs/qvla/bin/python}
export QVLA_MODEL=${QVLA_MODEL:-Qwen/Qwen3.8-27B-FP8}
export QVLA_FORMAT=${QVLA_FORMAT:-menu}
export QVLA_MENU_SPEED=${QVLA_MENU_SPEED:-0.45}
export NAV_POLICY_PORT=${NAV_POLICY_PORT:-8766}

# Both halves of this line are load-bearing and were each wrong once. GPU1 alone,
# because spanning both cards routes FP8 off DeepGEMM onto Triton and costs ~4x; and
# `0:30` because after the pin below the card is device 0, with 30 GiB of its 31.36
# left for the 28.75 GiB of weights. A device left in the map at 0 can still be
# spilled onto, and one spilled layer loses the DeepGEMM path for the whole model.
export CUDA_VISIBLE_DEVICES=${NAV_POLICY_GPU:-1}
export QVLA_MAX_MEMORY=${QVLA_MAX_MEMORY:-0:30}

echo "Q-VLA arc-menu policy server"
echo "  python     : $VENV_PY"
echo "  model      : $QVLA_MODEL"
echo "  format     : $QVLA_FORMAT   speed cap ${QVLA_MENU_SPEED} m/s"
echo "  thinking   : $QVLA_THINK_LEVEL"
echo "  GPU        : physical ${CUDA_VISIBLE_DEVICES} (seen as cuda:0), ${QVLA_MAX_MEMORY}"
echo "  port       : $NAV_POLICY_PORT"
echo

exec "$VENV_PY" server_qwen.py
