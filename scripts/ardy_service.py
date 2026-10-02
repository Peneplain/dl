"""Persistent text-to-ARDY reference service.

The service loads the frozen text encoder and ARDY model once, then accepts
JSONL requests on stdin. Each request produces the same offline reference
files as ``run_ardy.py`` without reloading the 8B text model.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.ardy import ArdyService
from baseline.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="musa")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--out-root", type=Path, default=ROOT / "output/ardy-service")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)
    service = ArdyService(args)
    print("READY", file=sys.stderr, flush=True)
    request_index = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        try:
            request = json.loads(line) if line.startswith("{") else {"prompt": line}
            prompt = request["prompt"]
            request_index += 1
            name = request.get("name", f"request-{request_index:04d}")
            name = Path(str(name)).name
            output = args.out_root / name
            report = service.generate(
                prompt, float(request.get("duration", 2.0)), int(request.get("seed", 0)), output)
            print(json.dumps({"status": "passed", "output": str(output),
                              "motion_seconds": report["motion_seconds"]}), flush=True)
        except Exception as error:
            print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
                  flush=True)


if __name__ == "__main__":
    main()
