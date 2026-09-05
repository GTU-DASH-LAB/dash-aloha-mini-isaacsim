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
# Resolved rather than hardcoded: nav/paths.py prefers the third_party/TIC-VLA submodule
# and falls back to this machine's historical checkout, so this launcher works both here
# and in a fresh clone. `server.py` resolves the same way on its own -- exporting it here
# only makes the value visible to DynaNav's code, which reads this variable by name.
#
# Assigned in two steps for the reason launch_qwen.sh spells out: `X=${X:-$(cmd)}` hides
# a failing cmd from `set -e`, and an empty root here would be exported into DynaNav's
# own code, which reads this variable by name and would look for its assets under /.
if [ -z "${TICVLA_DYNANAV_ROOT:-}" ]; then
  TICVLA_DYNANAV_ROOT="$(python3 ../paths.py --dynanav)"
fi
export TICVLA_DYNANAV_ROOT
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
