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

exec docker run --rm "${tty_flags[@]}" \
  --runtime=mthreads \
  --shm-size="${MUSA_SHM_SIZE:-16g}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MTHREADS_VISIBLE_DEVICES="${visible_devices}" \
  -v "${repo_root}:/workspace/dl" \
  -w /workspace/dl \
  "${image}" "$@"
