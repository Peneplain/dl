"""Check baseline dependencies, pinned sources and complete local weights without inference."""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import LOCK, ROOT, checked_checkout, verify_assets
from baseline.text_encoder import check_transformers_version
from scripts.run_ardy import device_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--device", default="musa")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    failures = []

    def check(label, fn):
        try:
            result = fn()
            print(f"[OK]   {label}: {result if result is not None else 'passed'}", flush=True)
        except Exception as error:
            failures.append(label)
            print(f"[FAIL] {label}: {type(error).__name__}: {error}", flush=True)

    # MUSA must be registered before ARDY or Transformers are imported.
    check("device", lambda: device_for(args.device))
    for name in ("torch", "mujoco", "onnxruntime", "transformers", "peft", "hydra", "einops", "vector_quantize_pytorch"):
        check(name, lambda name=name: getattr(importlib.import_module(name), "__version__", "imported"))
    check("text encoder compatibility", check_transformers_version)
    for key in ("ardy", "sonic"):
        check(f"{key} source", lambda key=key: checked_checkout(ROOT / "third_party" / key, lock[key]["commit"]))
    for key in ("sonic", "ardy", "llama_base", "text_base", "text_adapter"):
        check(f"{key} weights", lambda key=key: verify_assets(args.assets, key))
    print("Preflight only; actual model inference and MuJoCo execution are separate checks.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
