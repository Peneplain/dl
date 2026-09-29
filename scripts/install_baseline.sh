#!/usr/bin/env bash
# Run INSIDE the existing MUSA container. No CUDA/TRT extras or torch replacement.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run_dir="artifacts/baseline-install-$(date +%Y%m%d-%H%M%S)"
mkdir -p artifacts
mkdir "$run_dir"
python - "$run_dir/vendor-constraints.txt" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
import torch
import torch_musa
if not torch.musa.is_available():
    raise SystemExit("MUSA unavailable: run inside the supported S4000 container")
pins = []
for name in ("torch", "torch_musa", "torchvision", "torchaudio", "triton"):
    try:
        pins.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
Path(sys.argv[1]).write_text("\n".join(pins) + "\n")
print("Preserving installed vendor packages:", ", ".join(pins))
PY
# Dependency conflicts fail resolution instead of replacing a pinned vendor package.
python -m pip install -c "$run_dir/vendor-constraints.txt" -r requirements-baseline.txt \
  2>&1 | tee "$run_dir/install.log"
python - "$run_dir/vendor-constraints.txt" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    name, expected = line.split("==", 1)
    if metadata.version(name) != expected:
        raise SystemExit(f"Vendor package changed: {name}")
PY
python -m pip freeze > "$run_dir/pip-freeze.txt"
python scripts/check_backend.py --device musa \
  2>&1 | tee "$run_dir/backend.log"
echo "Dependencies installed. Next: python scripts/fetch_baseline.py --only all"
