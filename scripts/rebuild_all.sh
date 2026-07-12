#!/usr/bin/env bash
# Rebuild the full pipeline in the correct order. build_scene.py recreates scene.usda
# from scratch (fresh references to the environment + robot), which wipes out anything
# layered on top by configure_physics.py / fix_wheel_collision.py -- so those two MUST
# run again after any build_scene.py call, every time. Run this instead of calling the
# individual scripts by hand to avoid silently ending up with an unconfigured scene
# (this exact mistake happened once: an environment swap left every joint at
# stiffness=0/damping=0, with no visible error -- see CLAUDE.md).
#
# Usage: scripts/rebuild_all.sh [extra args passed through to build_scene.py]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== 1/4 build_scene.py ==="
~/isaacsim/python.sh scripts/build_scene.py "$@"

echo "=== 2/4 configure_physics.py ==="
~/isaacsim/python.sh scripts/configure_physics.py

echo "=== 3/4 fix_wheel_collision.py ==="
~/isaacsim/python.sh scripts/fix_wheel_collision.py

echo "=== 4/4 add_cameras.py ==="
~/isaacsim/python.sh scripts/add_cameras.py

echo "=== Done. Verifying... ==="
~/isaacsim/python.sh scripts/verify_physics.py
