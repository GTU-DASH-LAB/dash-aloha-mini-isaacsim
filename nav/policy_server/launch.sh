#!/usr/bin/env bash
# Launch the TIC-VLA policy server (Python 3.11 venv, GPU1).
#
# This process must NEVER import Isaac Sim. That is what makes CUDA_VISIBLE_DEVICES
# safe to use here -- Kit warns at startup that pinning it "can lead to undesired
# behavior or crashes", so the *simulator* pins its GPU a different way
# (--/physics/cudaDevice=N). See memory/gtu-workstation-gpu-asymmetry.md.
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=${VENV_PY:-/home/gtu-dsa/envs/tic-vla/bin/python}
export TICVLA_DYNANAV_ROOT=${TICVLA_DYNANAV_ROOT:-/home/gtu-dsa/robotics/TIC-VLA/DynaNav}
export TICVLA_BASE_MODEL_PATH=${TICVLA_BASE_MODEL_PATH:-/home/gtu-dsa/robotics/models/InternVL3-1B}
export TICVLA_CHECKPOINT_PATH=${TICVLA_CHECKPOINT_PATH:-/home/gtu-dsa/robotics/models/TIC-VLA-model.ckpt}
export NAV_POLICY_PORT=${NAV_POLICY_PORT:-8765}

# GPU1 is on a PCIe x4 link, which throttles host<->device transfer but NOT compute.
# Per step this carries one camera image over and a few dozen floats back, so the
# narrow link is irrelevant here -- and it keeps all 31 GiB of GPU0 for the scene.
export CUDA_VISIBLE_DEVICES=${NAV_POLICY_GPU:-1}
export NAV_POLICY_DEVICE=cuda:0   # physical GPU1, after the pin above

for p in "$TICVLA_DYNANAV_ROOT" "$TICVLA_BASE_MODEL_PATH" "$TICVLA_CHECKPOINT_PATH"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing required path: $p" >&2
    exit 1
  fi
done

echo "TIC-VLA policy server"
echo "  python     : $VENV_PY"
echo "  DynaNav    : $TICVLA_DYNANAV_ROOT"
echo "  checkpoint : $TICVLA_CHECKPOINT_PATH"
echo "  GPU        : physical ${CUDA_VISIBLE_DEVICES} (seen as cuda:0)"
echo "  port       : $NAV_POLICY_PORT"
echo

exec "$VENV_PY" server.py
