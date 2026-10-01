#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="artifacts/interactive-gui-$(date +%Y%m%d-%H%M%S)"
export MUSA_IMAGE="${MUSA_IMAGE:-dl-musa-gui:latest}"
export MTHREADS_VISIBLE_DEVICES="${MTHREADS_VISIBLE_DEVICES:-0}"
export MUJOCO_VNC=1

exec "$repo_root/docker/run-musa.sh" /workspace/dl/docker/start-mujoco-vnc.sh \
  python scripts/run_live.py --keep-alive --gui \
    --sim-seconds "${MUJOCO_SIM_SECONDS:-10}" \
    --out "$run_id" \
    --video "$run_id/baseline.mp4"
