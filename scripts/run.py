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
from baseline.session import (TERMINAL, digest, latest_result, make_plan, next_attempt,
                              result_line, session_directory, summarize)


# Exploratory K0 wrist-target calibration; shared grounding and gates stay fixed.
KIMODO_WRIST_OFFSET = (.125, .035, .08)


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


def parser_for_run(*, allow_abbrev=True):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=allow_abbrev)
    parser.add_argument("mode", choices=["batch", "manual"])
    parser.add_argument("--grasp", action="store_true", help="Enable the tabletop block grasp task; default: empty scene")
    generator = parser.add_mutually_exclusive_group()
    generator.add_argument("--ardy", dest="kimodo", action="store_const", const=False,
                           help="Use the pinned frozen ARDY generator (default)")
    generator.add_argument("--kimodo", dest="kimodo", action="store_const", const=True,
                           help="Use the pinned Kimodo-G1-RP-v1 generator")
    parser.set_defaults(kimodo=False)
    parser.add_argument("--batch", type=int, default=1, help="Number of batch attempts (default: 1)")
    parser.add_argument("--resume", type=Path, help="Resume a batch folder using its saved configuration")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prompt", help="Motion text; absent by default")
    parser.add_argument("--duration", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube-xy", nargs=2, type=float)
    parser.add_argument("--start-back", type=float, default=DEFAULT_START_BACK,
                        help="Extra distance behind the approach target, in metres (--grasp --walk only)")
    parser.add_argument("--walk", action="store_true",
                        help="Start behind the table and execute the walking approach (grasp only)")
    parser.add_argument("--direct-start", action="store_true",
                        help="Compatibility alias for the default no-walk grasp start (grasp only)")
    parser.add_argument("--table-standoff", type=float, default=DEFAULT_TABLE_STANDOFF,
                        help="Target root distance from the front table edge, in metres (grasp only)")
    parser.add_argument("--finger-kp", type=float, default=6., help="Finger torque stiffness (Nm/rad; default 6)")
    parser.add_argument("--finger-kd", type=float, default=.4, help="Finger torque damping (Nm s/rad; default .4)")
    parser.add_argument("--hand-approach", choices=["preshaped", "open"], default="open",
                        help="Finger posture during reach/lower; close only after alignment (grasp only)")
    parser.add_argument("--hold-seconds", type=float, default=5.,
                        help="Final hold duration, within the 30-second trial budget (grasp only)")
    parser.add_argument("--acquisition-z-min", type=float, default=None,
                        help="Optional lower bound for the dynamic wrist-frame gate (metres; grasp only)")
    parser.add_argument("--wrist-offset", type=float, nargs=3, default=None,
                        help="Optional wrist-frame offset override; default uses the measured hand-center site")
    parser.add_argument("--alignment-replans", type=int, default=2,
                        help="Short measured-state lower-phase replans after a missed gate (grasp only)")
    parser.add_argument("--alignment-replan-seconds", type=float, default=1.6,
                        help="Duration of each lower-phase alignment replan (grasp only)")
    parser.add_argument("--acquisition-transition", type=float, default=.3,
                        help="Measured-pose acquisition hold transition in seconds (grasp only)")
    parser.add_argument("--contact-profile", choices=["legacy", "elliptic"], default="elliptic",
                        help="Shared grasp contact solver profile (grasp only)")
    parser.add_argument("--lift-seconds", type=float, default=4.8,
                        help="Generated lift duration, within the common timeout (grasp only)")
    parser.add_argument("--xy-range", nargs=4, type=float, default=[.36, .46, -.30, -.16])
    parser.add_argument("--phase-prompts", type=Path, help="JSON file overriding individual grasp phase prompts")
    parser.add_argument("--prompt-profile", choices=["focused", "legacy"], default="focused")
    parser.add_argument("--gui", action="store_true", help="Manual mode only; a persistent MuJoCo viewer")
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="musa")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--risk-checkpoint", "--risk", dest="risk_checkpoint", type=Path,
                        help="Optional frozen Predictive Risk checkpoint; requires --residual-checkpoint and --risk-gate")
    parser.add_argument("--residual-checkpoint", "--residual", dest="residual_checkpoint", type=Path,
                        help="Optional frozen Residual checkpoint used with --risk-checkpoint")
    parser.add_argument("--risk-gate", "--gate", dest="risk_gate", type=Path,
                        help="Validation-calibrated Risk gate used with the learned controller")
    parser.add_argument("--risk-interface", choices=["P"], default=None,
                        help="Learned interface; the current physical controller is P")
    parser.add_argument("--risk-device", default=None,
                        help="Torch device for Risk/Residual inference (default: --device)")
    parser.add_argument("--risk-update-hz", type=float, default=None,
                        help="Risk/Residual update frequency, constrained to 10-20 Hz (default: 10)")
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--sonic-repo", type=Path, default=ROOT / "third_party/sonic")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--kimodo-repo", type=Path, default=ROOT / "third_party/kimodo",
                        help="Pinned Kimodo source checkout (used only with --kimodo)")
    parser.add_argument("--kimodo-assets", type=Path, default=ROOT / "checkpoints/kimodo",
                        help="Pinned Kimodo checkpoint root (used only with --kimodo)")
    parser.add_argument("--kimodo-diffusion-steps", type=int, default=100,
                        help="Kimodo denoising steps (default: 100)")
    parser.add_argument("--kimodo-constraint-guidance", type=float, default=2.0,
                        help="Kimodo pose-constraint guidance scale (default: 2)")
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument("--kimodo-project-constraints", dest="kimodo_no_projection",
                            action="store_false",
                            help="Experimental nominal wrist projection (off by default; not grasp-validated)")
    projection.add_argument("--kimodo-no-projection", dest="kimodo_no_projection",
                            action="store_true", help="Keep raw Kimodo rotations (default)")
    parser.set_defaults(kimodo_no_projection=True)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def parse_args(argv=None, *, allow_abbrev=True):
    parser = parser_for_run(allow_abbrev=allow_abbrev)
    args = parser.parse_args(argv)
    resumed = bool(args.resume)
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
                path_keys = {"ardy_repo", "sonic_repo", "assets", "kimodo_repo", "kimodo_assets",
                             "risk_checkpoint", "residual_checkpoint", "risk_gate"}
                setattr(args, key, Path(value) if key in path_keys and value is not None else value)
            # Older Kimodo plans used raw exports. Preserve resumed experiments.
            if config.get("kimodo", False) and "kimodo_no_projection" not in config:
                args.kimodo_no_projection = True
            # Plans written before --walk used direct_start=False for the walking
            # path. Preserve that immutable behavior when they are resumed.
            if "walk" not in config:
                args.walk = bool(args.grasp) and not bool(config.get("direct_start", False))
        except (OSError, KeyError, ValueError) as error:
            parser.error(str(error))
    elif args.phase_prompts:
        try:
            args.phase_prompts = json.loads(args.phase_prompts.read_text())
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.walk and args.direct_start:
        parser.error("--walk and --direct-start are mutually exclusive")
    if not resumed and args.grasp:
        # A grasp attempt starts at the grounded table approach target unless
        # the caller explicitly opts into the walking approach.
        args.direct_start = not args.walk
        if args.kimodo and args.wrist_offset is None:
            # Store the selected K0 calibration in the immutable plan. Explicit
            # overrides and resumed plans retain their original values.
            args.wrist_offset = list(KIMODO_WRIST_OFFSET)
    if args.batch < 1 or args.seed < 0 or args.seed + args.batch - 1 >= 2**32:
        parser.error("batch must be positive; seeds must fit uint32")
    if args.mode == "manual" and (args.batch != 1 or args.plan_only):
        parser.error("--batch and --plan-only are for batch mode")
    if args.mode == "batch" and args.gui:
        parser.error("--gui is for manual mode")
    learned_paths = (args.risk_checkpoint, args.residual_checkpoint, args.risk_gate)
    if any(path is not None for path in learned_paths) and not all(path is not None for path in learned_paths):
        parser.error("--risk-checkpoint, --residual-checkpoint and --risk-gate must be supplied together")
    if all(path is not None for path in learned_paths):
        if args.kimodo:
            parser.error("--kimodo cannot be combined with Risk/Residual artifacts; run B0 or P separately")
        if not args.grasp:
            parser.error("learned Risk/Residual inference requires --grasp so measured task sensors are available")
        for label, learned_path in zip(("risk-checkpoint", "residual-checkpoint", "risk-gate"), learned_paths):
            if not learned_path.is_file():
                parser.error(f"{label} does not exist: {learned_path}")
        if args.risk_interface is None:
            args.risk_interface = "P"
        if args.risk_device is None:
            args.risk_device = args.device
        if args.risk_update_hz is None:
            args.risk_update_hz = 10.
        if not np.isfinite(args.risk_update_hz) or not 10 <= args.risk_update_hz <= 20:
            parser.error("risk-update-hz must be finite and between 10 and 20")
    elif any(value is not None for value in (args.risk_interface, args.risk_device, args.risk_update_hz)):
        parser.error("--risk-interface, --risk-device and --risk-update-hz require all three learned controller artifacts")
    if args.history_frames < 4 or args.history_frames % 4 or args.threads < 1:
        parser.error("history-frames must be a positive multiple of four; threads positive")
    if args.kimodo_diffusion_steps < 1:
        parser.error("kimodo-diffusion-steps must be positive")
    if (not np.isfinite(args.kimodo_constraint_guidance)
            or args.kimodo_constraint_guidance <= 0):
        parser.error("kimodo-constraint-guidance must be finite and positive")
    if not (.05 <= args.start_back <= .6 and .20 <= args.table_standoff <= .55):
        parser.error("start-back must be .05-.6 m; table-standoff must be .20-.55 m")
    if (not np.isfinite([args.finger_kp, args.finger_kd, args.hold_seconds]).all()
            or not (0 < args.finger_kp <= 20 and 0 <= args.finger_kd <= 2 and 3 <= args.hold_seconds <= 20)):
        parser.error("finger-kp must be (0,20], finger-kd [0,2], hold-seconds [3,20]")
    if not args.grasp and (args.hand_approach != "open" or args.hold_seconds != 5.):
        parser.error("hand-approach and hold-seconds require --grasp")
    if args.acquisition_z_min is not None and (not np.isfinite(args.acquisition_z_min)
                                                or not -.20 <= args.acquisition_z_min <= 0):
        parser.error("acquisition-z-min must be finite and between -.20 and 0 m")
    if args.wrist_offset is not None and (not np.isfinite(args.wrist_offset).all()
            or not (.05 <= args.wrist_offset[0] <= .30 and -.15 <= args.wrist_offset[1] <= .15
                    and -.15 <= args.wrist_offset[2] <= .15)):
        parser.error("wrist-offset must fit X=[.05,.30], Y=[-.15,.15], Z=[-.15,.15] m")
    if args.alignment_replans < 0 or args.alignment_replans > 4:
        parser.error("alignment-replans must be between 0 and 4")
    if not np.isfinite(args.alignment_replan_seconds) or not .4 <= args.alignment_replan_seconds <= 3.0:
        parser.error("alignment-replan-seconds must be between .4 and 3 seconds")
    if not np.isfinite(args.acquisition_transition) or not .1 <= args.acquisition_transition <= .8:
        parser.error("acquisition-transition must be between .1 and .8 seconds")
    if not np.isfinite(args.lift_seconds) or not 3.2 <= args.lift_seconds <= 6.4:
        parser.error("lift-seconds must be between 3.2 and 6.4 seconds")
    if not args.grasp and (args.acquisition_z_min is not None or args.wrist_offset is not None
                          or args.alignment_replans != 2 or args.alignment_replan_seconds != 1.6
                          or args.acquisition_transition != .3 or args.contact_profile != "elliptic"
                          or args.lift_seconds != 4.8):
        parser.error("wrist, acquisition, contact and lift options require --grasp")
    xmin, xmax, ymin, ymax = args.xy_range
    if not (.34 <= xmin <= xmax <= .55 and -.38 <= ymin <= ymax <= -.08):
        parser.error("xy-range must fit X=[.34,.55], Y=[-.38,-.08]")
    if not args.grasp and args.prompt_profile != "focused":
        parser.error("prompt-profile requires --grasp")
    if not args.grasp and args.walk:
        parser.error("walk requires --grasp")
    if not args.grasp and args.direct_start:
        parser.error("direct-start requires --grasp")
    if not args.grasp and (args.start_back != DEFAULT_START_BACK or args.table_standoff != DEFAULT_TABLE_STANDOFF):
        parser.error("start-back and table-standoff require --grasp")
    try:
        parse_request({}, 1, args)
    except ValueError as error:
        parser.error(str(error))
    return args


def config_for(args):
    derived = {"risk_checkpoint_sha256", "residual_checkpoint_sha256", "risk_gate_sha256",
               "risk_learning_source_sha256"}
    excluded = {"resume", "output_root", "plan_only"} | derived
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in excluded}
    paths = (("risk_checkpoint", "risk_checkpoint_sha256"),
             ("residual_checkpoint", "residual_checkpoint_sha256"),
             ("risk_gate", "risk_gate_sha256"))
    if all(config.get(key) for key, _ in paths):
        for key, hash_key in paths:
            config[hash_key] = sha256(config[key])
        config["risk_learning_source_sha256"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "risk_residual").rglob("*.py"))
        }
    return config


def build_risk_residual_provider(args):
    """Load a provenance-checked controller for the shared batch executor."""
    if args.risk_checkpoint is None:
        return None, None
    import torch

    from baseline.runtime import device_for
    from risk_residual.checkpoints import load_controller
    from risk_residual.runtime import ReferenceCorrectionProvider, SimulationHistoryBuilder

    device = device_for(args.risk_device)
    controller = load_controller(args.risk_checkpoint, args.residual_checkpoint, args.risk_gate,
                                 device=device, method=args.risk_interface)
    provider = ReferenceCorrectionProvider(
        controller, SimulationHistoryBuilder(controller.config), update_hz=args.risk_update_hz)

    def metadata(artifact):
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        return {
            "path": str(Path(artifact).resolve()),
            "sha256": sha256(artifact),
            "schema": payload.get("schema"),
            "kind": payload.get("kind"),
            "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
            "baseline_lock_sha256": payload.get("baseline_lock_sha256"),
            "synthetic_inputs": payload.get("synthetic_inputs"),
            "window_hashes": payload.get("window_hashes"),
            "model_config": payload.get("model_config"),
            "model_config_sha256": (digest(payload["model_config"])
                                    if payload.get("model_config") is not None else None),
            "interface": payload.get("interface"),
            "risk_sha256": payload.get("risk_sha256"),
        }

    gate = json.loads(args.risk_gate.read_text())
    controller_metadata = {
        "schema": "dl-risk-residual-runtime-v1",
        "interface": args.risk_interface,
        "device": str(device),
        "update_hz": float(args.risk_update_hz),
        "risk": metadata(args.risk_checkpoint),
        "residual": metadata(args.residual_checkpoint),
        "gate": {
            "path": str(args.risk_gate.resolve()),
            "sha256": sha256(args.risk_gate),
            "threshold": gate.get("threshold"),
            "split": gate.get("split"),
            "risk_sha256": gate.get("risk_sha256"),
            "dataset_manifest_sha256": gate.get("dataset_manifest_sha256"),
            "validation_windows_sha256": gate.get("validation_windows_sha256"),
            "probability_transform": gate.get("probability_transform"),
            "temperature": gate.get("temperature"),
        },
        "learning_source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "risk_residual").rglob("*.py"))
        },
        "dataset_manifest_sha256": gate.get("dataset_manifest_sha256"),
    }
    risk_config_hash = controller_metadata["risk"].get("model_config_sha256")
    residual_config_hash = controller_metadata["residual"].get("model_config_sha256")
    if risk_config_hash != residual_config_hash:
        raise ValueError("Risk and residual model configurations differ")
    controller_metadata["model_config_sha256"] = risk_config_hash
    return provider, controller_metadata


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
        provider, controller_metadata = build_risk_residual_provider(args)
        args.controller_metadata = controller_metadata
        args.execution_method = "risk_residual_batch" if provider is not None else None
        if controller_metadata is not None:
            write_json(output / "controller.json", controller_metadata)
        runtime = ExecutionRuntime(args, output, correction_provider=provider)
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
