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
  batch       Collect attempts; --ardy or --kimodo, --batch N, optional --grasp
  manual      Enter prompt JSON; --ardy or --kimodo, optional --gui / --grasp
              Grasp starts at the table; add --walk to approach from farther back
  render      Render selected attempts or --all after collection
  ardy        Generate a reference only (run_ardy.py arguments)
  service     Keep ARDY loaded for JSONL requests (ardy_service.py arguments)
  sonic       Probe frozen SONIC ONNX graphs (--out required)
  fetch       Download/verify pinned assets (fetch_baseline.py arguments)
  convert     Convert a motion CSV (prepare_reference.py arguments)
  deploy      Export deploy motion CSVs (prepare_deploy_motion.py arguments)
  package     Create a baseline source archive (--out required)
  build       Build the offline RGB image
  build-gui   Build the GUI image
  build-base  Build the control-only image

Choose a frozen generator with --ardy (default) or --kimodo.
Example: ./run.sh batch --ardy --grasp --batch 20 --seed 10000
Resume:  ./run.sh batch --resume output/batch-TIME
The default image is dl-musa-render:latest; override MUSA_IMAGE if needed.
Sessions use output/batch-TIME or output/manual-TIME. See docs/commands.md.
EOF
    exit 0
    ;;
  build) exec docker build --target render -f Dockerfile.musa -t dl-musa-render:latest "$@" . ;;
  build-gui) exec docker build --target gui -f Dockerfile.musa -t dl-musa-gui:latest "$@" . ;;
  build-base) exec docker build -f Dockerfile.musa -t dl-musa:latest "$@" . ;;
  shell) invocation=(bash "$@") ;;
  check)
    invocation=(bash -c 'python scripts/check_baseline.py --device musa && python scripts/check_backend.py --device musa' "$@")
    ;;
  tests) invocation=(env RUN_MUJOCO_RENDER_TESTS=1 python -m unittest discover -s tests -v "$@") ;;
  smoke) script=smoke.py ;;
  batch|manual)
    invocation=(python scripts/run.py "$command" "$@")
    if [[ "$command" == manual && " $* " == *" --gui "* ]]; then
      export MUSA_IMAGE="${MUSA_IMAGE:-dl-musa-gui:latest}"
      export MUJOCO_VNC=1
      invocation=(/workspace/dl/docker/start-mujoco-vnc.sh "${invocation[@]}")
    fi
    ;;
  render) script=render.py ;;
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
