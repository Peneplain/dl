"""Execute frozen SONIC ONNX graphs on CPU using synthetic inputs; no physics.

This is an operator/latency gate, not an observation adapter or control loop.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import ROOT, sha256, verify_assets, write_json


def probe(model_path, repeats, threads):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    feed, inputs = {}, []
    dtypes = {"tensor(float)": np.float32, "tensor(double)": np.float64,
              "tensor(float16)": np.float16, "tensor(int64)": np.int64,
              "tensor(int32)": np.int32, "tensor(bool)": np.bool_}
    for node in session.get_inputs():
        shape = list(node.shape)
        # Only batch may be dynamic. Do not invent an observation dimension.
        if shape and not isinstance(shape[0], int):
            shape[0] = 1
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"Unresolved ONNX input shape {node.name}: {node.shape}")
        if node.type not in dtypes:
            raise ValueError(f"Unsupported input dtype {node.type}")
        feed[node.name] = np.zeros(shape, dtype=dtypes[node.type])
        inputs.append({"name": node.name, "shape": shape, "dtype": node.type})
    # Three unmeasured warm-ups; each graph is probed independently.
    for _ in range(3):
        session.run(None, feed)
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = session.run(None, feed)
        latencies.append((time.perf_counter() - start) * 1000)
        if not all(np.isfinite(value).all() for value in outputs):
            raise FloatingPointError("Nonfinite SONIC ONNX output")
    return {"sha256": sha256(model_path), "providers": session.get_providers(),
            "inputs": inputs,
            "outputs": [{"name": node.name, "shape": list(value.shape)}
                        for node, value in zip(session.get_outputs(), outputs)],
            "latency_ms_p50": float(np.percentile(latencies, 50)),
            "latency_ms_p95": float(np.percentile(latencies, 95)), "repeats": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("--repeats and --threads must be positive")
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    args.out.mkdir(parents=True)
    report = {"status": "running", "stage": "sonic_onnx_cpu_operators",
              "synthetic_inputs": True, "graphs_connected": False,
              "physics_executed": False, "task_success": None,
              "platform": platform.platform(), "threads": args.threads, "warmups": 3}
    try:
        import onnxruntime as ort
        report["onnxruntime"] = ort.__version__
        report["asset_manifest_sha256"] = verify_assets(args.assets, "sonic")
        report["observation_config_sha256"] = sha256(args.assets / "sonic/observation_config.yaml")
        report["models"] = {}
        for name in ("encoder", "decoder"):
            report["models"][name] = probe(args.assets / f"sonic/model_{name}.onnx", args.repeats, args.threads)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
