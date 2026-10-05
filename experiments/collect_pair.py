"""Collect a verified clean-teacher / arm-perturbed-nominal simulation pair.

Both branches reuse the same saved ARDY references and start from the same
episode reset. The activation fingerprint rejects any pre-intervention drift.
This is a controlled perturbation source, not held-out task evaluation.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

from baseline.common import sha256, write_json
from baseline.execution import ExecutionRuntime
from baseline.session import make_plan
from risk_residual.teacher import ScheduledArmOffset
from scripts.run import config_for, parse_args


def source_args(source):
    request = json.loads((source / "request.json").read_text())
    settings = json.loads((source / "settings.json").read_text())
    if request.get("task") != "grasp":
        raise ValueError("A paired teacher requires a grasp source attempt")
    command = ["batch", "--grasp", "--seed", str(request["seed"]),
               "--cube-xy", *map(str, request["cube_xy"]),
               "--table-standoff", str(settings["table_standoff_m"]),
               "--finger-kp", str(settings["finger_kp_nm_per_rad"]),
               "--finger-kd", str(settings["finger_kd_nm_s_per_rad"]),
               "--hand-approach", settings["hand_approach"],
               "--hold-seconds", str(settings["final_hold_seconds"]),
               "--contact-profile", settings["contact_profile"],
               "--lift-seconds", str(settings["lift_seconds"]),
               "--alignment-replans", str(settings.get("alignment_replans", 2)),
               "--alignment-replan-seconds", str(settings.get("alignment_replan_seconds", 1.6)),
               "--acquisition-transition", str(settings.get("acquisition_transition_seconds", .3)),
               "--prompt-profile", request["prompt_profile"]]
    if not settings["direct_start"]:
        command += ["--walk", "--start-back", str(settings["start_back_m"])]
    if settings.get("nominal_wrist_offset_m") is not None:
        command += ["--wrist-offset", *map(str, settings["nominal_wrist_offset_m"])]
    if settings.get("acquisition_z_min_m") is not None:
        command += ["--acquisition-z-min", str(settings["acquisition_z_min_m"])]
    return parse_args(command), request


def collect(source, output, *, phase, delay, joint, amplitude):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh paired collection output directory")
    args, request = source_args(source)
    # Demand saved references throughout; a fresh ARDY sample would change the
    # counterfactual action and invalidate the clean/perturbed comparison.
    output.mkdir(parents=True)
    provenance = {"source_attempt": str(source), "source_report_sha256": sha256(source / "report.json"),
                  "phase": phase, "delay": delay, "joint": joint, "amplitude": amplitude,
                  "branches": {}, "pair_state_verified": False, "teacher_verified": False}
    runtime = None
    try:
        for name, perturb in (("teacher", False), ("nominal", True)):
            branch = output / name
            branch.mkdir()
            provider = ScheduledArmOffset(phase=phase, delay=delay, joint=joint,
                                          amplitude=amplitude, perturb=perturb)
            if runtime is None:
                runtime = ExecutionRuntime(args, output, correction_provider=provider)
            else:
                runtime.correction_provider = provider

            def saved_generate(prompt, duration, seed, destination, **kwargs):
                saved = source / "ardy" / destination.name / "reference.npz"
                if not saved.is_file():
                    raise FileNotFoundError(f"Missing exact saved ARDY reference: {saved}")
                destination.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(saved, destination / "reference.npz")
                result = {"status": "passed", "generation_reexecuted": False,
                          "source_reference": str(saved), "source_reference_sha256": sha256(saved),
                          "phase_seed": seed}
                write_json(destination / "report.json", result)
                return result

            runtime.generate = saved_generate
            plan = make_plan(config_for(args), [request])
            write_json(branch / "plan.json", plan)
            write_json(branch / "request.json", request)
            report = runtime.run(request, branch, plan["plan_sha256"])
            provenance["branches"][name] = {
                "report": str(branch / "report.json"), "report_sha256": sha256(branch / "report.json"),
                "scene_sha256": sha256(branch / "scene.xml"),
                "model_sha256": json.loads((branch / "rollout/metadata.json").read_text())["model_sha256"],
                "task_success": bool(report.get("task_success")), "status": report["status"],
                "physics_executed": bool(report.get("physics_executed")),
                "sonic_executed": bool(report.get("sonic_executed")),
                "activation": provider.activation,
            }
            write_json(output / "pair.json", provenance)
        clean = provenance["branches"]["teacher"]
        perturbed = provenance["branches"]["nominal"]
        left, right = clean["activation"], perturbed["activation"]
        provenance["pair_state_verified"] = bool(left and right and
            left["snapshot_sha256"] == right["snapshot_sha256"] and
            left["frame_index"] == right["frame_index"] and
            abs(left["time"] - right["time"]) < 1e-8 and
            clean["scene_sha256"] == perturbed["scene_sha256"] and
            clean["model_sha256"] == perturbed["model_sha256"])
        provenance["teacher_verified"] = bool(provenance["pair_state_verified"]
                                               and clean["task_success"] and clean["status"] == "passed")
        provenance["nominal_task_success"] = perturbed["task_success"]
        provenance["recovery_verified"] = bool(provenance["teacher_verified"] and
            perturbed["status"] in {"passed", "stopped"} and
            perturbed["physics_executed"] and perturbed["sonic_executed"] and
            not perturbed["task_success"])
        provenance["status"] = "paired" if provenance["pair_state_verified"] else "unpaired"
        write_json(output / "pair.json", provenance)
        return provenance
    finally:
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=("reach", "lower", "lift"), default="lower")
    parser.add_argument("--delay", type=float, default=.2)
    parser.add_argument("--joint", default="right_wrist_pitch_joint")
    parser.add_argument("--amplitude", type=float, default=.1)
    args = parser.parse_args()
    result = collect(args.source, args.out, phase=args.phase, delay=args.delay,
                     joint=args.joint, amplitude=args.amplitude)
    print(json.dumps({"status": result["status"], "pair_state_verified": result["pair_state_verified"],
                      "teacher_verified": result["teacher_verified"],
                      "recovery_verified": result["recovery_verified"],
                      "nominal_task_success": result["nominal_task_success"]}), flush=True)
    return 0 if result["pair_state_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
