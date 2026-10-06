"""Run B0/P pilot episodes through one shared frozen baseline executor.

Pass the same request file and baseline options to both methods. This entry point
collects pilot evidence; it does not invent a held-out split or claim the planned
20-pair, three-seed evaluation budget. P requires real, provenance-checked Risk,
Residual and validation gate artifacts. No teacher participates in a trial.
"""

import argparse
import json
from pathlib import Path

from baseline.common import ROOT, sha256, write_json
from baseline.session import digest, make_plan, next_attempt, wilson
from scripts.run import config_for, parse_args as baseline_args, parse_request


def prepare_plan(args):
    options = list(args.baseline_options)
    if options and options[0] == "--":
        options.pop(0)
    forbidden = {"--resume", "--batch", "--plan-only", "--output-root", "--gui"}
    if any(option.split("=", 1)[0] in forbidden for option in options):
        raise ValueError("Requests and fresh output are controlled by this experiment entry point")
    baseline = baseline_args(["batch", "--grasp", *options])
    payloads = json.loads(args.requests.read_text())
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("--requests must contain a nonempty JSON list of baseline request objects")
    # Explicit seeds prevent independently invoked methods from changing their
    # initial scene and ARDY random stream through implicit defaults.
    if any(not isinstance(row, dict) or "seed" not in row for row in payloads):
        raise ValueError("Every paired request must declare its seed")
    requests = [parse_request(row, index + 1, baseline) for index, row in enumerate(payloads)]
    seeds = [row["seed"] for row in requests]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Pilot episode seeds must be unique")
    shared = make_plan(config_for(baseline), requests)
    plan = {"schema_version": 1, "method": args.method, "evaluation_scope": "pilot",
            "teacher_used": False, "baseline_plan": shared,
            "paired_request_sha256": digest(requests),
            "experiment_source_sha256": sha256(Path(__file__)),
            "update_hz": args.update_hz if args.method == "P" else None,
            "learning_device": args.learning_device if args.method == "P" else None,
            "controller_artifacts": {}}
    if args.method == "P":
        if any(path is None for path in (args.risk, args.residual, args.gate)):
            raise ValueError("P requires --risk, --residual and --gate")
        plan["controller_artifacts"] = {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                                         for key, path in (("risk", args.risk),
                                                           ("residual", args.residual), ("gate", args.gate))}
        plan["learning_source_sha256"] = {str(path.relative_to(ROOT)): sha256(path)
                                          for path in sorted((ROOT / "risk_residual").rglob("*.py"))}
    elif any(path is not None for path in (args.risk, args.residual, args.gate)):
        raise ValueError("B0 does not accept learned controller artifacts")
    plan["plan_sha256"] = digest(plan)
    return baseline, plan


def correction_provider(args):
    if args.method == "B0":
        return None
    # B0 reaches its executor with no learning imports or checkpoint loads.
    from baseline.runtime import device_for
    from risk_residual.checkpoints import load_controller
    from risk_residual.runtime import ReferenceCorrectionProvider, SimulationHistoryBuilder

    gate = json.loads(args.gate.read_text())
    if (gate.get("probability_transform", "raw_sigmoid") != "raw_sigmoid"
            or gate.get("temperature", 1.0) != 1.0):
        raise ValueError("This pilot entry accepts raw-sigmoid gates only; temperature diagnostics are not gates")
    device = device_for(args.learning_device)
    controller = load_controller(args.risk, args.residual, args.gate, device=device, method="P")
    return ReferenceCorrectionProvider(controller, SimulationHistoryBuilder(controller.config),
                                       update_hz=args.update_hz)


def inference_summary(output):
    import numpy as np

    records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    decisions = [row for row in records if row["event"] == "risk_residual"]
    summary = {"decisions": len(decisions),
               "activations": sum(bool(row["active"]) for row in decisions),
               "history_gaps": sum(row["event"] == "risk_history_gap" for row in records)}
    summary["activation_rate"] = summary["activations"] / len(decisions) if decisions else None
    for key in ("risk_ms", "residual_ms", "controller_ms", "desired_offset_norm"):
        values = [row[key] for row in decisions if key in row]
        summary[key + "_p50"] = float(np.percentile(values, 50)) if values else None
        summary[key + "_p95"] = float(np.percentile(values, 95)) if values else None
    return summary


def run(args):
    from baseline.execution import ExecutionRuntime

    baseline, plan = prepare_plan(args)
    provider = correction_provider(args)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    runtime = ExecutionRuntime(baseline, args.out, correction_provider=provider)
    reports, status, error_message = [], "running", None
    try:
        runtime.preload()
        for request in plan["baseline_plan"]["requests"]:
            output = next_attempt(args.out, request)
            report = runtime.run(request, output, plan["baseline_plan"]["plan_sha256"])
            report.update(method=args.method, evaluation_scope="pilot",
                          experiment_plan_sha256=plan["plan_sha256"],
                          paired_episode_id=digest(request),
                          paired_request_sha256=plan["paired_request_sha256"],
                          controller_artifacts=plan["controller_artifacts"])
            # Count actual physics trials before processing optional diagnostics.
            # A malformed event stream must not erase an executed failure.
            reports.append({**report, "attempt": output.name})
            try:
                report["learning_inference"] = inference_summary(output) if (output / "events.jsonl").exists() else None
            except Exception as diagnostic_error:
                report["learning_inference"] = None
                report["diagnostic_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            write_json(output / "report.json", report)
            reports[-1] = {**report, "attempt": output.name}
            if report["status"] == "failed":
                status = "failed"
                break
        else:
            status = "complete"
    except (KeyboardInterrupt, SystemExit) as error:
        status = "interrupted"
        error_message = f"{type(error).__name__}: {error}"
        raise
    except Exception as error:
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            runtime.close()
        finally:
            successes = sum(row.get("task_success") is True for row in reports)
            summary = {"schema_version": 1, "status": status, "method": args.method,
                       "evaluation_scope": "pilot", "teacher_used": False,
                       "planned": len(plan["baseline_plan"]["requests"]), "completed": len(reports),
                       "successes": successes, "success_rate": successes / len(reports) if reports else None,
                       "success_wilson_95": wilson(successes, len(reports)),
                       "plan_sha256": plan["plan_sha256"], "attempts": reports}
            if error_message is not None:
                summary["error"] = error_message
            write_json(args.out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("B0", "P"), required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--risk", type=Path)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--learning-device", default="musa")
    parser.add_argument("--update-hz", type=float, default=10)
    parser.add_argument("baseline_options", nargs=argparse.REMAINDER,
                        help="Shared baseline flags after --; batch/grasp are supplied automatically")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    import math
    if not math.isfinite(args.update_hz) or not 10 <= args.update_hz <= 20:
        parser.error("update-hz must be finite and between 10 and 20")
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
