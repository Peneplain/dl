"""Run one-process text -> ARDY -> SONIC -> MuJoCo baseline execution.

The SONIC loop stays at 50 Hz while ARDY generates a requested reference in a
worker thread. A completed reference is installed through the shared timestamped
buffer, with a short transition from the measured pose. The default run is
headless; GUI forwarding is handled by ``docker/run-musa.sh`` when available.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.ardy import ArdyService
from baseline.common import ROOT, write_json
from baseline.simulation import SimulationStop, SonicSimulation


def parse_request(line, index):
    line = line.strip()
    if not line:
        return None
    if line.lower() in {"quit", "exit"}:
        return {"quit": True}
    request = json.loads(line) if line.startswith("{") else {"prompt": line}
    prompt = str(request.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("request needs a nonempty prompt")
    return {"prompt": prompt, "duration": float(request.get("duration", 2.0)),
            "seed": int(request.get("seed", 0)), "name": f"request-{index:04d}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="musa")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--sonic-repo", type=Path, default=ROOT / "third_party/sonic")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        help="Install an existing reference.npz before starting input")
    parser.add_argument("--prompt", help="Queue one initial text command")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-seconds", type=float, default=30.0)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fast", action="store_true",
                        help="Run faster than wall-clock instead of pacing 50 Hz")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    if args.sim_seconds <= 0 or args.threads < 1 or args.duration < 0.08:
        parser.error("sim-seconds, threads and duration must be positive; duration >= 0.08")
    if args.history_frames < 4 or args.history_frames % 4:
        parser.error("history-frames must be a multiple of four and at least four")
    args.out.mkdir(parents=True)
    (args.out / "ardy").mkdir()

    simulation = SonicSimulation(args.assets, args.sonic_repo, args.out, threads=2)
    requests = queue.Queue()
    stop_input = threading.Event()
    request_index = 0
    if args.prompt:
        request_index += 1
        requests.put({"prompt": args.prompt, "duration": args.duration,
                      "seed": args.seed, "name": f"request-{request_index:04d}"})

    def read_input():
        nonlocal request_index
        for line in sys.stdin:
            if stop_input.is_set():
                break
            try:
                request_index += 1
                request = parse_request(line, request_index)
                if request is not None:
                    requests.put(request)
                    if request.get("quit"):
                        break
            except Exception as error:
                print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
                      file=sys.stderr, flush=True)

    input_thread = threading.Thread(target=read_input, name="stdin-reader", daemon=True)
    input_thread.start()
    service = None
    service_lock = threading.Lock()

    def generate(request, history):
        nonlocal service
        with service_lock:
            if service is None:
                print("Loading ARDY service in the background...", file=sys.stderr, flush=True)
                service = ArdyService(args)
        output = args.out / "ardy" / request["name"]
        return service.generate(request["prompt"], request["duration"], request["seed"],
                                output, history_qpos=history)

    future = None
    active_request = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ardy-generator")
    final = {"status": "running", "physics_executed": False, "sonic_executed": False,
             "task_success": None, "commands": []}
    deadline = float(args.sim_seconds)
    if args.reference:
        simulation.install(simulation.load_reference(args.reference))
        final["reference"] = str(args.reference)
        deadline = max(deadline, float(simulation.end_time) + 2.0)

    try:
        while simulation.data.time < deadline or future is not None or not requests.empty():
            if future is None:
                try:
                    request = requests.get_nowait()
                except queue.Empty:
                    request = None
                if request is not None:
                    if request.get("quit"):
                        break
                    history = simulation.history_qpos(args.history_frames)
                    active_request = request
                    final["commands"].append({k: request[k] for k in ("prompt", "duration", "seed")})
                    simulation.event("ardy_request", prompt=request["prompt"],
                                     duration=request["duration"], seed=request["seed"])
                    future = executor.submit(generate, request, history)

            if future is not None and future.done():
                try:
                    report = future.result()
                    reference = simulation.load_reference(
                        args.out / "ardy" / active_request["name"] / "reference.npz")
                    simulation.install(reference)
                    final.setdefault("ardy_reports", []).append(report)
                    deadline = max(deadline, float(simulation.end_time) + 2.0)
                    print(json.dumps({"status": "installed", "prompt": active_request["prompt"],
                                      "motion_seconds": report["motion_seconds"]}), flush=True)
                except Exception as error:
                    simulation.event("ardy_request_failed", error=f"{type(error).__name__}: {error}")
                    print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
                          file=sys.stderr, flush=True)
                finally:
                    future = None
                    active_request = None

            started = time.perf_counter()
            try:
                simulation.tick()
            except SimulationStop as error:
                simulation.event("simulation_stop", reason=str(error))
                final["stop_reason"] = str(error)
                break
            if not args.fast:
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - started)))
    finally:
        stop_input.set()
        executor.shutdown(wait=False, cancel_futures=False)
        final.update(simulation.summary())
        final["status"] = "passed" if final.get("stop_reason") is None else "stopped"
        write_json(args.out / "report.json", final)
        simulation.close()


if __name__ == "__main__":
    main()
