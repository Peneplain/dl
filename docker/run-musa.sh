#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${MUSA_IMAGE:-dl-musa:latest}"
visible_devices="${MTHREADS_VISIBLE_DEVICES:-all}"

if ! docker info --format '{{json .Runtimes}}' | grep -q 'mthreads'; then
  echo "Docker runtime 'mthreads' is not registered. Run the host setup from README.md." >&2
  exit 1
fi

tty_flags=(--interactive)
if [[ -t 0 && -t 1 ]]; then
  tty_flags+=(--tty)
fi

# Preserve an SSH X11 display when one is available.  SSH forwarding uses a
# TCP listener on host localhost (for example, localhost:10.0), so the
# container needs host networking for that case.  Local Unix-socket displays
# only need the socket mount.  Headless runs leave these arrays empty.
display_flags=()
network_flags=()
venv_flags=()

# The baseline installer targets this repository-local environment. Put it
# first on PATH when it exists so `run-musa.sh bash` and direct commands use
# the same pinned torch/torch_musa/ONNX stack without requiring an extra
# `source` step inside the container.
venv_dir="/workspace/dl/.venv-baseline-musa"
if [[ -f "$repo_root/.venv-baseline-musa/pyvenv.cfg" ]]; then
  venv_flags+=(
    -e "VIRTUAL_ENV=${venv_dir}"
    -e "PATH=${venv_dir}/bin:/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn_bench/bin:/usr/local/musa/mccl_test:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  )
fi

if [[ -n "${DISPLAY:-}" ]]; then
  display_flags+=(-e "DISPLAY=${DISPLAY}")
  xauthority_file="${XAUTHORITY:-${HOME:-}/.Xauthority}"
  if [[ -f "$xauthority_file" ]]; then
    display_flags+=(-e XAUTHORITY=/tmp/.Xauthority
                    -v "$xauthority_file:/tmp/.Xauthority:ro")
  fi
  if [[ "$DISPLAY" == localhost:* || "$DISPLAY" == 127.0.0.1:* ]]; then
    network_flags+=(--network=host)
  elif [[ -d /tmp/.X11-unix ]]; then
    display_flags+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
  fi
fi

exec docker run --rm "${tty_flags[@]}" \
  "${network_flags[@]}" \
  --runtime=mthreads \
  --shm-size="${MUSA_SHM_SIZE:-16g}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${display_flags[@]}" \
  "${venv_flags[@]}" \
  -e MTHREADS_VISIBLE_DEVICES="${visible_devices}" \
  -v "${repo_root}:/workspace/dl" \
  -w /workspace/dl \
  "${image}" "$@"
