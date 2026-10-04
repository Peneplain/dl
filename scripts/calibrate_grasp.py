"""Paired hand-controller calibration using saved or freshly generated references.

Every condition reruns from reset through frozen SONIC. By default, available
nominal references are reused and missing phases are freshly generated and logged.
With --fresh-motion, ARDY generates every moving phase from executed history.
These diagnostic/calibration rollouts do not replace held-out evaluation.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.common import sha256, write_json
from baseline.execution import ExecutionRuntime
from baseline.session import make_plan
from scripts.run import config_for, parse_args


VARIANTS = {
    "legacy": (4., .2, "preshaped"),
    "damped": (4., .4, "preshaped"),
    "strong": (6., .4, "preshaped"),
    "open": (6., .4, "open"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--standoff", type=float)
    parser.add_argument("--hold-seconds", type=float, default=10.)
    parser.add_argument("--acquisition-z-min", type=float, default=None)
    parser.add_argument("--wrist-offset", type=float, nargs=3, default=None)
    parser.add_argument("--alignment-replans", type=int, default=2)
    parser.add_argument("--alignment-replan-seconds", type=float, default=1.6)
    parser.add_argument("--acquisition-transition", type=float, default=.3)
    parser.add_argument("--contact-profile", choices=["legacy", "elliptic"], default="elliptic")
    parser.add_argument("--lift-seconds", type=float, default=3.2)
    parser.add_argument("--fresh-motion", action="store_true",
                        help="Generate every moving phase from the current executed history")
    options = parser.parse_args(argv)
    options.out.mkdir(parents=True, exist_ok=False)
    sources = {json.loads((path / "request.json").read_text())["seed"]: path
               for path in options.source.glob("attempt-*") if (path / "request.json").is_file()}
    runtime = None
    reports = []
    try:
        for variant in options.variants:
            kp, kd, posture = VARIANTS[variant]
            for seed in options.seeds:
                source = sources[seed]
                settings = json.loads((source / "settings.json").read_text())
                request = json.loads((source / "request.json").read_text())
                standoff = options.standoff if options.standoff is not None else settings["table_standoff_m"]
                command = ["batch", "--grasp", "--seed", str(seed), "--table-standoff", str(standoff),
                                   "--finger-kp", str(kp), "--finger-kd", str(kd),
                                   "--hand-approach", posture, "--hold-seconds", str(options.hold_seconds),
                                   "--contact-profile", options.contact_profile,
                                   "--lift-seconds", str(options.lift_seconds),
                                   "--alignment-replans", str(options.alignment_replans),
                                   "--alignment-replan-seconds", str(options.alignment_replan_seconds),
                                   "--acquisition-transition", str(options.acquisition_transition)]
                if options.acquisition_z_min is not None:
                    command += ["--acquisition-z-min", str(options.acquisition_z_min)]
                if options.wrist_offset is not None:
                    command += ["--wrist-offset", *map(str, options.wrist_offset)]
                args = parse_args(command)
                output = options.out / f"{variant}-seed-{seed:03d}"
                output.mkdir()
                if runtime is None:
                    runtime = ExecutionRuntime(args, options.out)
                    original_generate = runtime.generate
                runtime.args = args
                if runtime.simulation is not None:
                    runtime.simulation.finger_kp, runtime.simulation.finger_kd = kp, kd
                def generate(prompt, duration, phase_seed, destination, **kwargs):
                    nominal = source / "ardy" / destination.name / "reference.npz"
                    if nominal.is_file() and not options.fresh_motion:
                        destination.mkdir(parents=True)
                        shutil.copyfile(nominal, destination / "reference.npz")
                        provenance = {"status": "passed", "diagnostic_saved_reference": True,
                                      "source_reference": str(nominal), "source_reference_sha256": sha256(nominal),
                                      "generation_reexecuted": False, "phase_seed": phase_seed}
                        write_json(destination / "report.json", provenance)
                        return provenance
                    return original_generate(prompt, duration, phase_seed, destination, **kwargs)
                runtime.generate = generate
                plan = make_plan(config_for(args), [request])
                write_json(output / "plan.json", plan)
                write_json(output / "request.json", request)
                result = runtime.run(request, output, plan["plan_sha256"])
                reports.append({"variant": variant, "seed": seed, "output": str(output),
                                "task_success": result["task_success"], "failure_reason": result.get("failure_reason"),
                                "max_clearance_m": result.get("max_block_clearance_m"),
                                "hold_seconds": result.get("max_continuous_hold_seconds"),
                                "simulation_seconds": result.get("simulation_seconds")})
                print(json.dumps(reports[-1]), flush=True)
                write_json(options.out / "diagnostics.json", {
                    "scope": ("fresh ARDY closed-loop calibration" if options.fresh_motion else
                              "saved-reference hand-controller calibration; not held-out evaluation"),
                    "command": sys.argv, "script_sha256": sha256(__file__), "trials": reports})
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    main()
