"""Two execution modes: batch collection and one-command-at-a-time manual control."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from baseline.common import ROOT, sha256, write_json
from baseline.console import ManualConsole
from baseline.grasp import DEFAULT_START_BACK, DEFAULT_TABLE_STANDOFF, PHASES
from baseline.session import (TERMINAL, latest_result, make_plan, next_attempt, result_line,
                              session_directory, summarize)


def parse_request(payload, index, args):
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    allowed = {"prompt", "duration", "seed", "cube_xy", "phase_prompts", "reference"}
    if set(payload) - allowed:
        raise ValueError("Unknown JSON keys: " + ", ".join(sorted(set(payload) - allowed)))
    prompt = payload.get("prompt", args.prompt)
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("prompt must be a nonempty string")
    duration = payload.get("duration", args.duration)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not np.isfinite(duration) or not .08 <= duration <= 25:
        raise ValueError("duration must be finite and between .08 and 25 seconds")
    seed = payload.get("seed", args.seed + index - 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    phase_prompts = payload.get("phase_prompts", args.phase_prompts or {})
    if not isinstance(phase_prompts, dict) or set(phase_prompts) - {phase for phase, _ in PHASES}:
        raise ValueError("phase_prompts must map approach/settle/prepare/reach/lower/close/lift/hold to strings")
    if any(not isinstance(v, str) or not v.strip() for v in phase_prompts.values()):
        raise ValueError("phase_prompts values must be nonempty strings")
    xy = payload.get("cube_xy", args.cube_xy)
    if not args.grasp and (xy is not None or phase_prompts):
        raise ValueError("cube_xy and phase_prompts require --grasp")
    if args.grasp:
        if xy is None:
            rng = np.random.default_rng(seed)
            xmin, xmax, ymin, ymax = args.xy_range
            xy = [float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax))]
        try:
            xy = np.asarray(xy, dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or not (.34 <= xy[0] <= .55 and -.38 <= xy[1] <= -.08):
                raise ValueError
        except (ValueError, TypeError) as error:
            raise ValueError("cube_xy must fit X=[.34,.55], Y=[-.38,-.08]") from error
        xy = xy.tolist()
    reference = payload.get("reference")
    if reference is not None:
        if args.grasp or prompt:
            raise ValueError("reference requires an empty task and no prompt")
        if not isinstance(reference, str) or not Path(reference).is_file():
            raise ValueError("reference must name an existing reference.npz")
        reference = str(Path(reference).resolve())
    return {"index": index, "task": "grasp" if args.grasp else None, "prompt": prompt,
            "duration": float(duration), "seed": seed, "cube_xy": xy,
            "phase_prompts": phase_prompts, "prompt_profile": args.prompt_profile if args.grasp else None,
            "reference": reference, "reference_sha256": sha256(reference) if reference else None}


def parser_for_run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["batch", "manual"])
    parser.add_argument("--grasp", action="store_true", help="Enable the tabletop block grasp task; default: empty scene")
    parser.add_argument("--batch", type=int, default=1, help="Number of batch attempts (default: 1)")
    parser.add_argument("--resume", type=Path, help="Resume a batch folder using its saved configuration")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prompt", help="Motion text; absent by default")
    parser.add_argument("--duration", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube-xy", nargs=2, type=float)
    parser.add_argument("--start-back", type=float, default=DEFAULT_START_BACK,
                        help="Initial extra distance behind the approach target, in metres (grasp only)")
    parser.add_argument("--table-standoff", type=float, default=DEFAULT_TABLE_STANDOFF,
                        help="Target root distance from the front table edge, in metres (grasp only)")
    parser.add_argument("--xy-range", nargs=4, type=float, default=[.36, .46, -.30, -.16])
    parser.add_argument("--phase-prompts", type=Path, help="JSON file overriding individual grasp phase prompts")
    parser.add_argument("--prompt-profile", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--gui", action="store_true", help="Manual mode only; a persistent MuJoCo viewer")
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="musa")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--sonic-repo", type=Path, default=ROOT / "third_party/sonic")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def parse_args(argv=None):
    parser = parser_for_run()
    args = parser.parse_args(argv)
    if args.resume:
        if args.mode != "batch":
            parser.error("--resume is for batch mode")
        try:
            saved = json.loads((args.resume / "plan.json").read_text())
            config = saved["config"]
            if saved["schema_version"] != 2 or config["mode"] != "batch":
                raise ValueError("Unsupported saved plan")
            # Resume is entirely defined by the saved plan; no silent overrides.
            supplied = list(sys.argv[1:] if argv is None else argv)
            if any(s.startswith("--") and s.split("=")[0] != "--resume" for s in supplied):
                parser.error("Use only: batch --resume PATH; settings come from plan.json")
            for key, value in config.items():
                setattr(args, key, Path(value) if key in {"ardy_repo", "sonic_repo", "assets"} else value)
        except (OSError, KeyError, ValueError) as error:
            parser.error(str(error))
    elif args.phase_prompts:
        try:
            args.phase_prompts = json.loads(args.phase_prompts.read_text())
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.batch < 1 or args.seed < 0 or args.seed + args.batch - 1 >= 2**32:
        parser.error("batch must be positive; seeds must fit uint32")
    if args.mode == "manual" and (args.batch != 1 or args.plan_only):
        parser.error("--batch and --plan-only are for batch mode")
    if args.mode == "batch" and args.gui:
        parser.error("--gui is for manual mode")
    if args.history_frames < 4 or args.history_frames % 4 or args.threads < 1:
        parser.error("history-frames must be a positive multiple of four; threads positive")
    if not (.05 <= args.start_back <= .6 and .20 <= args.table_standoff <= .55):
        parser.error("start-back must be .05-.6 m; table-standoff must be .20-.55 m")
    xmin, xmax, ymin, ymax = args.xy_range
    if not (.34 <= xmin <= xmax <= .55 and -.38 <= ymin <= ymax <= -.08):
        parser.error("xy-range must fit X=[.34,.55], Y=[-.38,-.08]")
    if not args.grasp and args.prompt_profile != "focused":
        parser.error("prompt-profile requires --grasp")
    if not args.grasp and (args.start_back != DEFAULT_START_BACK or args.table_standoff != DEFAULT_TABLE_STANDOFF):
        parser.error("start-back and table-standoff require --grasp")
    try:
        parse_request({}, 1, args)
    except ValueError as error:
        parser.error(str(error))
    return args


def config_for(args):
    return {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
            if k not in {"resume", "output_root", "plan_only"}}


def main(argv=None):
    args = parse_args(argv)
    if args.resume:
        output = args.resume.resolve()
        saved = json.loads((output / "plan.json").read_text())
        plan = make_plan(config_for(args), saved["requests"])
    else:
        output = args.output_root.resolve()
        requests = [parse_request({}, i + 1, args) for i in range(args.batch)] if args.mode == "batch" else []
        plan = make_plan(config_for(args), requests)
    with session_directory(output, plan, resume=bool(args.resume),
                           mode=None if args.resume else args.mode) as output:
        print(f"Output: {output}\nTask: {'tabletop grasp' if args.grasp else 'none (robot and ground only)'}", flush=True)
        requests = plan["requests"].copy()
        if args.plan_only:
            _, text = summarize(output, plan, "planned")
            print(text)
            return 0
        from baseline.execution import ExecutionRuntime
        runtime = ExecutionRuntime(args, output)
        status = "running"
        try:
            if args.mode == "batch":
                print("[LOAD] Preparing all models before batch execution.", flush=True)
                runtime.preload()
                print("[BATCH] Models ready. Starting planned attempts.", flush=True)
                for request in requests:
                    old = latest_result(output, request["index"], verify=True)
                    if old and old["status"] in TERMINAL:
                        print("[SKIP] " + result_line(old), flush=True)
                        continue
                    attempt = next_attempt(output, request)
                    print(f"\n[{request['index']}/{len(requests)}] {attempt.name} seed={request['seed']}", flush=True)
                    result = runtime.run(request, attempt, plan["plan_sha256"])
                    print(result_line({**result, "attempt": attempt.name}), flush=True)
                    summarize(output, plan, status)
                    if result["status"] == "failed" and (result["stage"] == "initializing" or result.get("model_load_failed")):
                        status = "blocked"
                        break
                else:
                    status = "complete"
            else:
                console = ManualConsole(sys.stdin)
                print("[LOAD] Preparing models before accepting JSON; input disabled.", flush=True)
                try:
                    with console.busy() as discard:
                        runtime.discard_input = discard
                        runtime.preload()
                finally:
                    runtime.discard_input = lambda: None
                print('Enter JSON, for example {"prompt":"A person raises the right hand slowly.","duration":2,"seed":0}', flush=True)
                print('In grasp mode, {} starts the default phase sequence. Enter quit to exit; input received while BUSY is discarded.', flush=True)
                while True:
                    print("READY > ", end="", flush=True)
                    line = console.read_line(runtime.idle)
                    if line is None or line.strip().lower() in {"quit", "exit"}:
                        break
                    if not line.strip():
                        continue
                    try:
                        request = parse_request(json.loads(line), len(requests) + 1, args)
                    except (ValueError, TypeError) as error:
                        print(f"Invalid input: {error}", flush=True)
                        continue
                    requests.append(request)
                    write_json(output / "requests.json", requests)
                    attempt = next_attempt(output, request)
                    print(f"BUSY [{attempt.name}] Input disabled. Wait until execution finishes before entering another command.", flush=True)
                    with console.busy() as discard:
                        runtime.discard_input = discard
                        result = runtime.run(request, attempt, plan["plan_sha256"])
                    runtime.discard_input = lambda: None
                    print(result_line({**result, "attempt": attempt.name}), flush=True)
                    if console.discarded:
                        print("Input received during execution was discarded. Enter it again after READY.", flush=True)
                    summarize(output, plan, status, requests=requests)
                status = "complete"
        except Exception:
            status = "failed"
            raise
        except BaseException:
            status = "interrupted"
            raise
        finally:
            try:
                runtime.close()
            finally:
                summary, text = summarize(output, plan, status, requests=requests)
                print("\n" + text + f"\nReport: {output / 'summary.md'}", flush=True)
        return 1 if summary["failures"] or summary["pending"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
