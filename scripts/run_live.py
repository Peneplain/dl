"""Run one-process text -> ARDY -> SONIC -> MuJoCo baseline execution.

The SONIC loop stays at 50 Hz while ARDY generates a requested reference in a
worker thread. A completed reference is installed through the shared timestamped
buffer, with a short transition from the current checked nominal pose. The default run is
headless; the optional GUI uses the container's configured X display.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.ardy import ArdyService
from baseline.common import ROOT, write_json
from baseline.simulation import SimulationStop, SonicSimulation
from baseline.sonic_policy import SonicPolicy


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
    duration = float(request.get("duration", 2.0))
    if not math.isfinite(duration) or duration < .08:
        raise ValueError("request duration must be finite and at least 0.08 seconds")
    return {"prompt": prompt, "duration": duration,
            "seed": int(request.get("seed", 0)), "name": f"request-{index:04d}"}


def finish_run(simulation, final, run_dir):
    """Seal execution logs before rendering, and retain failures in both stages."""
    video_path = simulation.video_path
    if final.get("stop_reason") is not None:
        final["status"] = "stopped"
    elif final.get("ardy_failed") or final.get("runtime_error"):
        final["status"] = "failed"
    else:
        final["status"] = "passed"
    final["execution_status"] = final["status"]
    try:
        final.update(simulation.summary())
    except Exception as error:
        final.update(status="failed", recording_error=f"{type(error).__name__}: {error}")
    finally:
        try:
            simulation.end_run()
        except Exception as error:
            final.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
    # Rendering wall time is reported separately from the control loop.
    write_json(run_dir / "report.json", final)
    if video_path is not None and not final.get("recording_error") and not final.get("cleanup_error"):
        try:
            from baseline.rendering import render_rollout
            report = render_rollout(run_dir, video=video_path)
            final["vision"] = report
            final["video"] = report["video"]
        except Exception as error:
            final["status"] = "failed"
            final["render_error"] = f"{type(error).__name__}: {error}"
            print(f"Offline rendering failed; saved rollout can be retried: {error}",
                  file=sys.stderr, flush=True)
        finally:
            write_json(run_dir / "report.json", final)
    return final


def run_interactive(args):
    """Run independent prompt directories while reusing the frozen models."""
    args.out.mkdir(parents=True, exist_ok=False)
    session = {"status": "running", "runs": []}
    print("Loading ARDY service before accepting prompts...", file=sys.stderr, flush=True)
    service = ArdyService(args)
    print("Loading SONIC policy before accepting prompts...", file=sys.stderr, flush=True)
    sonic_load_started = time.perf_counter()
    policy = SonicPolicy(args.assets, args.sonic_repo, threads=2)
    video_name = Path(args.video).name if args.video else None
    simulation = SonicSimulation(args.assets, args.sonic_repo, None,
                                 threads=2, policy=policy, gui=args.gui)
    print(f"ARDY and SONIC ready in {time.perf_counter() - sonic_load_started:.1f}s; "
          f"{'MuJoCo viewer is open. ' if args.gui else ''}Enter a prompt (or 'quit' to stop).",
          file=sys.stderr, flush=True)

    requests = queue.Queue()
    stop_input = threading.Event()
    input_closed = threading.Event()
    request_index = 0
    if args.prompt:
        request_index += 1
        requests.put({"prompt": args.prompt, "duration": args.duration,
                      "seed": args.seed, "name": f"request-{request_index:04d}"})

    def read_input():
        nonlocal request_index
        try:
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
        finally:
            input_closed.set()

    input_thread = threading.Thread(target=read_input, name="stdin-reader", daemon=True)
    input_thread.start()

    try:
        while not input_closed.is_set() or not requests.empty():
            try:
                request = requests.get(timeout=.1)
            except queue.Empty:
                simulation.sync_viewer()
                continue
            if request.get("quit"):
                break

            run_number = int(request["name"].split("-")[-1])
            run_dir = args.out / f"run-{run_number:04d}"
            run_dir.mkdir(parents=False, exist_ok=False)
            (run_dir / "ardy").mkdir()
            video_path = run_dir / video_name if video_name else None
            simulation.start_run(run_dir, video_path)
            final = {
                "status": "running", "physics_executed": False,
                "sonic_executed": False, "task_success": None,
                "commands": [{k: request[k] for k in ("prompt", "duration", "seed")}],
                "ardy_failed": False, "fast": args.fast, "keep_alive": True,
                "run_id": run_dir.name, "models_preloaded": True,
            }
            try:
                history = simulation.history_qpos(args.history_frames)
                simulation.event("ardy_request", prompt=request["prompt"],
                                 duration=request["duration"], seed=request["seed"])
                output = run_dir / "ardy" / request["name"]
                try:
                    with ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="ardy-request") as executor:
                        future = executor.submit(
                            service.generate, request["prompt"], request["duration"],
                            request["seed"], output, history_qpos=history)
                        while not future.done():
                            simulation.sync_viewer()
                            time.sleep(.02)
                        report = future.result()
                    reference = simulation.load_reference(output / "reference.npz")
                    simulation.install(reference)
                    final["ardy_reports"] = [report]
                    deadline = max(float(args.sim_seconds), float(simulation.end_time) + 2.0)
                    print(json.dumps({"status": "installed", "run": run_dir.name,
                                      "prompt": request["prompt"],
                                      "motion_seconds": report["motion_seconds"]}), flush=True)
                except Exception as error:
                    final["ardy_failed"] = True
                    final["ardy_errors"] = [{"type": type(error).__name__,
                                             "message": str(error),
                                             "request": request["name"]}]
                    simulation.event("ardy_request_failed",
                                     error=f"{type(error).__name__}: {error}")
                    print(json.dumps({"status": "failed", "run": run_dir.name,
                                      "error": f"{type(error).__name__}: {error}"}),
                          file=sys.stderr, flush=True)
                    deadline = float(args.sim_seconds)

                while simulation.data.time < deadline - 1e-8:
                    started = time.perf_counter()
                    try:
                        simulation.tick()
                    except SimulationStop as error:
                        simulation.event("simulation_stop", reason=str(error))
                        final["stop_reason"] = str(error)
                        break
                    if not args.fast:
                        time.sleep(max(0.0, .02 - (time.perf_counter() - started)))
            except BaseException as error:
                final["runtime_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                finish_run(simulation, final, run_dir)
                session["runs"].append({"run": run_dir.name, "status": final["status"]})
                simulation.reset()
        session["status"] = ("failed" if any(r["status"] != "passed" for r in session["runs"])
                             else "passed")
    except BaseException as error:
        session.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_input.set()
        simulation.close()
        write_json(args.out / "session.json", session)
    if session["status"] == "failed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="musa")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--sonic-repo", type=Path, default=ROOT / "third_party/sonic")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path,
                        help="After execution, render saved states to RGB NPZ and this MP4")
    parser.add_argument("--gui", action="store_true",
                        help="Keep a passive MuJoCo viewer open during the interactive session")
    parser.add_argument("--reference", type=Path,
                        help="Install an existing reference.npz before starting input")
    parser.add_argument("--prompt", help="Queue one initial text command")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-seconds", type=float, default=30.0)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run without wall-clock pacing; pause simulated time while ARDY generates",
    )
    parser.add_argument(
        "--keep-alive", "--interactive", dest="keep_alive", action="store_true",
        help="Keep this run open for more JSONL prompts until quit or EOF",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    if args.video and args.video.exists():
        parser.error("Choose a fresh --video path")
    if (not math.isfinite(args.sim_seconds) or not math.isfinite(args.duration)
            or args.sim_seconds <= 0 or args.threads < 1 or args.duration < 0.08):
        parser.error("sim-seconds, threads and duration must be positive; duration >= 0.08")
    if args.history_frames < 4 or args.history_frames % 4:
        parser.error("history-frames must be a multiple of four and at least four")
    if args.keep_alive:
        if args.reference:
            parser.error("--reference is for a single run; interactive runs use text prompts")
        run_interactive(args)
        return
    # Keep the output directory a write-once run boundary, including against
    # two processes racing to use the same timestamped path.
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "ardy").mkdir()

    try:
        simulation = SonicSimulation(args.assets, args.sonic_repo, args.out, threads=2,
                                     video_path=args.video, gui=args.gui)
    except BaseException as error:
        write_json(args.out / "report.json", {
            "status": "failed", "execution_status": "failed", "stage": "simulation_init",
            "physics_executed": False, "sonic_executed": False, "task_success": None,
            "runtime_error": f"{type(error).__name__}: {error}"})
        raise
    requests = queue.Queue()
    stop_input = threading.Event()
    input_closed = threading.Event()
    request_index = 0
    if args.prompt:
        request_index += 1
        requests.put({"prompt": args.prompt, "duration": args.duration,
                      "seed": args.seed, "name": f"request-{request_index:04d}"})

    def read_input():
        nonlocal request_index
        try:
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
        finally:
            input_closed.set()

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
             "task_success": None, "commands": [], "ardy_failed": False,
             "fast": args.fast, "keep_alive": args.keep_alive}
    deadline = float(args.sim_seconds)
    try:
        if args.reference:
            simulation.install(simulation.load_reference(args.reference))
            final["reference"] = str(args.reference)
            deadline = max(deadline, float(simulation.end_time) + 2.0)
        while (simulation.data.time < deadline - 1e-8 or future is not None or not requests.empty()
               or (args.keep_alive and not input_closed.is_set())):
            if future is None:
                # Once the current simulated segment is complete, interactive
                # mode waits here instead of advancing an idle simulation.
                if args.keep_alive and simulation.data.time >= deadline and not input_closed.is_set():
                    try:
                        request = requests.get(timeout=.1)
                    except queue.Empty:
                        request = None
                else:
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
                    final["ardy_failed"] = True
                    final.setdefault("ardy_errors", []).append(
                        {"type": type(error).__name__, "message": str(error),
                         "request": active_request["name"] if active_request else None}
                    )
                    simulation.event("ardy_request_failed", error=f"{type(error).__name__}: {error}")
                    print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
                          file=sys.stderr, flush=True)
                finally:
                    future = None
                    active_request = None

            # Offline batches and persistent interactive runs must not turn
            # ARDY wall time into recorded choreography time. Plain real-time
            # mode intentionally keeps the nominal SONIC hold running.
            if (args.fast or args.keep_alive) and future is not None:
                time.sleep(.005)
                continue

            started = time.perf_counter()
            try:
                simulation.tick()
            except SimulationStop as error:
                simulation.event("simulation_stop", reason=str(error))
                final["stop_reason"] = str(error)
                break
            if not args.fast:
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - started)))
    except BaseException as error:
        final["runtime_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_input.set()
        executor.shutdown(wait=False, cancel_futures=False)
        try:
            finish_run(simulation, final, args.out)
        finally:
            simulation.close()
    if final["status"] in {"failed", "stopped"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
