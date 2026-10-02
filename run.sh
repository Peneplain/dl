#!/usr/bin/env bash
# Run supported workflows from the S4000 host using the existing vendor environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$command" in
  help|-h|--help)
    cat <<'EOF'
Usage: ./run.sh COMMAND [arguments]

Run these commands from the server host in ~/dl:
  shell       Enter the offline RGB rendering container
  check       Verify local assets, sources and MUSA operators
  tests       Run all tests, including actual RGB/MP4 rendering
  smoke       Check synthetic reference conversion (--out required)
  live        Text/reference -> SONIC -> simulation (run_live.py arguments)
  render      Replay recorded states (render_expert_rollout.py arguments)
  ardy        Generate a reference only (run_ardy.py arguments)
  service     Keep ARDY loaded for JSONL requests (ardy_service.py arguments)
  sonic       Probe frozen SONIC ONNX graphs (--out required)
  fetch       Download/verify pinned assets (fetch_baseline.py arguments)
  convert     Convert a motion CSV (prepare_reference.py arguments)
  deploy      Export deploy motion CSVs (prepare_deploy_motion.py arguments)
  package     Create a baseline source archive (--out required)
  gui         Start the interactive MuJoCo viewer and VNC
  build       Build the offline RGB image
  build-gui   Build the GUI image
  build-base  Build the control-only image

The default image is dl-musa-render:latest; override MUSA_IMAGE if needed.
Every run/render output must be fresh. See docs/commands.md for examples.
EOF
    exit 0
    ;;
  build) exec docker build --target render -f Dockerfile.musa -t dl-musa-render:latest "$@" . ;;
  build-gui) exec docker build --target gui -f Dockerfile.musa -t dl-musa-gui:latest "$@" . ;;
  build-base) exec docker build -f Dockerfile.musa -t dl-musa:latest "$@" . ;;
  gui) exec "$repo_root/docker/run-mujoco-gui.sh" "$@" ;;
  shell) invocation=(bash "$@") ;;
  check)
    invocation=(bash -c 'python scripts/check_baseline.py --device musa && python scripts/check_backend.py --device musa' "$@")
    ;;
  tests) invocation=(env RUN_MUJOCO_RENDER_TESTS=1 python -m unittest discover -s tests -v "$@") ;;
  smoke) script=smoke.py ;;
  live) script=run_live.py ;;
  render) script=render_expert_rollout.py ;;
  ardy)
    invocation=(python scripts/run_ardy.py --device musa --text-device musa --text-dtype bfloat16 "$@")
    ;;
  service) script=ardy_service.py ;;
  sonic) script=check_sonic_onnx.py ;;
  fetch) script=fetch_baseline.py ;;
  convert) script=prepare_reference.py ;;
  deploy) script=prepare_deploy_motion.py ;;
  package) script=package_baseline.py ;;
  *) echo "Unknown command: $command. Run ./run.sh help." >&2; exit 2 ;;
esac

export MUSA_IMAGE="${MUSA_IMAGE:-dl-musa-render:latest}"
if [[ -n "${script:-}" ]]; then invocation=(python "scripts/$script" "$@"); fi
exec "$repo_root/docker/run-musa.sh" "${invocation[@]}"
