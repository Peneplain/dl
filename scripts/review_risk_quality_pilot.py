"""Certify one exact saved-reference limitation without changing old evidence.

The failed initial run stays failed. This certificate authorizes processing only
its twelve untouched jobs, preserving and excluding the known invalid attempt.
Other failures, corrupted streams and changed snapshots cannot use this route.
"""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.teacher_risk_pipeline import now, read_json
from scripts.risk_quality_continue import host_path, sha, verified_pair, verify_hashes
from scripts.risk_quality_pipeline import process_exists


VALID_JOB_IDS = ("82d5cbfd8efd62db4a7c", "d2c17d17bf361d989a94")
EXCLUDED_JOB_ID = "f0d97ec9f1fd970b463a"
INITIAL_IDS = (VALID_JOB_IDS[0], EXCLUDED_JOB_ID, VALID_JOB_IDS[1])
MISSING_PREFIX = "FileNotFoundError: Missing exact saved ARDY reference: "
MISSING_RELATIVE = Path("ardy/lower_replan_01/reference.npz")
SCHEMA = "dl-risk-quality-pilot-review-v1"
EXPECTED_PLAN_SHA256 = "d9bf8ee9296f8b5836802cd33f5db39459b0d146ea09d9a62a2120b029ddc9c3"
ACTIVATION_KEYS = {"time", "frame_index", "phase", "snapshot_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def branch_evidence(repo, branch, *, perturb, job):
    report_path = host_path(repo, branch["report"])
    report = read_json(report_path)
    outputs = report.get("outputs_sha256", {})
    required = {"events.jsonl", "task.csv", "rollout/metadata.json", "rollout/states.npz",
                "rollout/scene.mjb", "scene.xml"}
    if (not required.issubset(outputs) or sha(report_path) != branch["report_sha256"] or
            report.get("status") != branch["status"] or
            report.get("task_success") is not branch["task_success"] or
            report.get("physics_executed") is not True or report.get("sonic_executed") is not True or
            branch.get("physics_executed") is not True or branch.get("sonic_executed") is not True):
        raise ValueError("Actual branch outcome or required physical evidence differs")
    hashes = {str(report_path): sha(report_path)}
    for relative, expected in outputs.items():
        artifact = (report_path.parent / relative).resolve()
        if not artifact.is_relative_to(report_path.parent.resolve()) or sha(artifact) != expected:
            raise ValueError("Branch output evidence changed: " + relative)
        hashes[str(artifact)] = expected
    if (outputs["scene.xml"] != branch["scene_sha256"] or
            outputs["rollout/scene.mjb"] != branch["model_sha256"] or
            read_json(report_path.parent / "rollout/metadata.json")["model_sha256"] != branch["model_sha256"]):
        raise ValueError("Actual compiled physical model differs from pair provenance")
    events = [json.loads(line) for line in (report_path.parent / "events.jsonl").read_text().splitlines()]
    activations = [event for event in events if event.get("event") == "teacher_pair_activation"]
    activation = branch.get("activation")
    if (len(activations) != 1 or not isinstance(activation, dict) or set(activation) != ACTIVATION_KEYS or
            not isinstance(activation["time"], (int, float)) or isinstance(activation["time"], bool) or
            not math.isfinite(activation["time"]) or activation["time"] < 0 or
            not isinstance(activation["frame_index"], int) or isinstance(activation["frame_index"], bool) or
            activation["frame_index"] < 0 or activation["phase"] != job["perturbation"]["phase"] or
            not isinstance(activation["snapshot_sha256"], str) or
            re.fullmatch(r"[0-9a-f]{64}", activation["snapshot_sha256"]) is None or
            any(activations[0].get(key) != value for key, value in activation.items()) or
            activations[0].get("sim_time") != activation.get("time") or
            activations[0].get("perturb") is not perturb or
            activations[0].get("joint") != job["perturbation"]["joint"] or
            activations[0].get("amplitude") != (job["perturbation"]["amplitude"] if perturb else 0.)):
        raise ValueError("Pair activation differs from its hashed actual event stream")
    return report, hashes


def snapshot_review(repo, first):
    """Re-verify the exact initial artifacts; no server job or data write occurs."""
    state, plan, report, index = [read_json(first / relative) for relative in
                                 ("state.json", "pairs/plan.json", "pairs/report.json", "index/sources.json")]
    if (state.get("schema") != "dl-risk-quality-pipeline-v1" or state.get("run_id") != "risk-quality-261006" or
            state.get("status") != "failed" or state.get("stage") != "teacher-pilot" or
            first.resolve() != (repo / "output/risk-quality-261006").resolve()):
        raise ValueError("Certificate route is only for the exact failed initial teacher pilot")
    if any(process_exists(state.get(key)) for key in ("pid", "child_pid")):
        raise ValueError("Recorded first-stage parent or child is still alive; review cannot proceed")
    verify_hashes(repo, state["source_sha256"])
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (digest(unsigned) != plan.get("plan_sha256") or plan.get("plan_sha256") != EXPECTED_PLAN_SHA256 or
            plan.get("sources_sha256") != sha(first / "index/sources.json")):
        raise ValueError("Immutable pair/source plan differs")
    verify_hashes(repo, plan["baseline_sources_sha256"])
    parents = index["parents"]
    owners = {(row["prompt_group"], str(row["scene_seed"])): row["split"] for row in parents}
    if (len(parents) != 15 or len(owners) != 15 or
            sum(split == "train" for split in owners.values()) != 10 or
            sum(split == "val" for split in owners.values()) != 5):
        raise ValueError("Indexed pilot differs from its fixed fifteen-parent budget")
    jobs = {job["id"]: job for job in plan["jobs"]}
    if (len(jobs) != 15 or tuple(job["id"] for job in plan["jobs"][:3]) != INITIAL_IDS or
            set(report.get("jobs", {})) != set(INITIAL_IDS) or report.get("planned_pairs") != 15 or
            report.get("pending_pairs") != 12 or report.get("failed_pairs") != 1 or
            report.get("plan_sha256") != plan["plan_sha256"]):
        raise ValueError("The review applies only to the exact initial three jobs and twelve pending jobs")
    parents_by_id = {parent["parent_id"]: parent for parent in parents}
    if (len(parents_by_id) != 15 or any(job["parent"] != parents_by_id.get(job["parent"]["parent_id"])
            or host_path(repo, job["source"]).resolve() != host_path(repo, job["parent"]["nominal"]).resolve()
            for job in jobs.values())):
        raise ValueError("Planned job does not belong to its exact indexed original parent")
    hashes = {str(first / relative): sha(first / relative) for relative in
              ("state.json", "pairs/plan.json", "pairs/report.json", "pairs/sources.json", "index/sources.json")}
    immutable = {str(first / relative): hashes[str(first / relative)] for relative in
                 ("state.json", "pairs/plan.json", "index/sources.json")}
    recoveries, clean_successes = 0, 0
    reviewed_parents, recovery_parents = set(), {"train": set(), "val": set()}
    branches_by_job = {}
    for key in INITIAL_IDS:
        job, result = jobs[key], report["jobs"][key]
        verify_hashes(repo, job["source_files_sha256"])
        hashes.update(job["source_files_sha256"])
        immutable.update(job["source_files_sha256"])
        pair_path = host_path(repo, result["pair"])
        pair = read_json(pair_path)
        hashes[str(pair_path)] = immutable[str(pair_path)] = sha(pair_path)
        if (host_path(repo, pair["source_attempt"]).resolve() != host_path(repo, job["source"]).resolve() or
                pair.get("source_report_sha256") != sha(host_path(repo, job["source"]) / "report.json") or
                pair.get("pair_state_verified") is not True or pair.get("teacher_verified") is not True or
                result.get("teacher_verified") is not True or
                any(pair[name] != job["perturbation"][name] for name in ("phase", "delay", "joint", "amplitude"))):
            raise ValueError("Pair source/teacher/perturbation provenance changed")
        actual = {}
        for name in ("teacher", "nominal"):
            actual[name], evidence = branch_evidence(repo, pair["branches"][name], perturb=name == "nominal", job=job)
            hashes.update(evidence)
            immutable.update(evidence)
        nominal, teacher = pair["branches"]["nominal"], pair["branches"]["teacher"]
        if (nominal["activation"] != teacher["activation"] or
                nominal["scene_sha256"] != teacher["scene_sha256"] or
                nominal["model_sha256"] != teacher["model_sha256"]):
            raise ValueError("Pair physical/controller activation did not match")
        if teacher["status"] != "passed" or teacher["task_success"] is not True:
            raise ValueError("Every reviewed clean teacher must actually succeed")
        clean_successes += 1
        if key in VALID_JOB_IDS:
            recovered = verified_pair(repo, job, result)
            recoveries += recovered
            if recovered:
                recovery_parents[job["parent"]["split"]].add(job["parent"]["parent_id"])
        else:
            missing = host_path(repo, job["source"]).resolve() / MISSING_RELATIVE
            error = actual["nominal"].get("runtime_error", "")
            error_path = error.removeprefix(MISSING_PREFIX)
            if (result.get("status") != "invalid_execution" or pair.get("recovery_verified") is not False or
                    result.get("recovery_verified") is not False or nominal["status"] != "failed" or
                    nominal["task_success"] is not False or actual["nominal"].get("failure_reason") != "lower_error" or
                    actual["nominal"].get("model_load_failed") is not False or
                    not error.startswith(MISSING_PREFIX) or
                    host_path(repo, error_path).resolve() != missing or missing.exists()):
                raise ValueError("Invalid attempt is not the exact known missing lower-replan-reference limitation")
        reviewed_parents.add(job["parent"]["parent_id"])
        branches_by_job[key] = {name: {"status": value["status"], "task_success": value["task_success"]}
                               for name, value in pair["branches"].items()}
    if recoveries < 1 or report.get("recovery_verified") != recoveries:
        raise ValueError("Reviewed valid pairs must include a physically verified recovery")
    current_sources = read_json(first / "pairs/sources.json")
    if (current_sources.get("pair_plan_sha256") != plan["plan_sha256"] or
            current_sources.get("original_sources_sha256") != plan["sources_sha256"] or
            current_sources.get("baseline_lock_sha256") != plan["baseline_lock_sha256"]):
        raise ValueError("Initial exported sources differ from the immutable pair plan")
    excluded_path = host_path(repo, report["jobs"][EXCLUDED_JOB_ID]["pair"]).resolve()
    if any(host_path(repo, row["pair"]).resolve() == excluded_path for row in current_sources["parents"] if row.get("pair")):
        raise ValueError("Known invalid pair must not be exported as a training source")
    valid_exports = {host_path(repo, report["jobs"][key]["pair"]).resolve() for key in VALID_JOB_IDS}
    if {host_path(repo, row["pair"]).resolve() for row in current_sources["parents"] if row.get("pair")} != valid_exports:
        raise ValueError("Initial exported sources must contain only the two reviewed valid pairs")
    return {"schema": SCHEMA, "first": str(first.resolve()), "run_id": state["run_id"],
            "admitted_first_status": "failed", "admitted_first_stage": "teacher-pilot",
            "approval": "Process only the unchanged twelve pending candidates; preserve and exclude one known invalid attempt.",
            "scope": "Exploratory continuation with nonrandom replay censoring caused by feedback replanning; invalid execution is excluded from physical outcome and training denominators.",
            "source_plan_sha256": plan["sources_sha256"], "pair_plan_sha256": plan["plan_sha256"],
            "reviewer_source_sha256": sha(Path(__file__)), "launch_snapshot_sha256": hashes,
            "immutable_evidence_sha256": immutable,
            "original_state": state, "initial_report": report, "initial_sources": current_sources,
            "excluded_job_ids": [EXCLUDED_JOB_ID], "valid_job_ids": list(VALID_JOB_IDS),
            "remaining_job_ids": [key for key in jobs if key not in INITIAL_IDS],
            "branches": branches_by_job,
            "counts": {"initial_pairs": 3, "valid_pairs": 2, "excluded_invalid_pairs": 1,
                       "verified_recoveries": recoveries, "verified_clean_successes": clean_successes,
                       "reviewed_independent_parents": len(reviewed_parents),
                       "new_recovery_parents_by_split": {split: len(values) for split, values in recovery_parents.items()},
                       "independent_new_parents": 15, "planned_pairs": 15, "remaining_pairs": 12}}


def create_certificate(repo, first, output):
    if output.exists():
        raise FileExistsError("Use a fresh certificate directory")
    review = snapshot_review(repo, first)
    output.mkdir(parents=True, exist_ok=False)
    review["reviewed_at"] = now()
    review["certificate_sha256"] = digest(review)
    # Copy original JSON bytes for independent audit, without touching the run.
    for relative in ("state.json", "pairs/report.json", "pairs/plan.json", "pairs/sources.json", "index/sources.json"):
        target = output / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((first / relative).read_bytes())
        if sha(target) != review["launch_snapshot_sha256"][str(first / relative)]:
            raise ValueError("Original evidence changed while copying review snapshot")
    (output / "certificate.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return output / "certificate.json"


def validate_certificate(repo, first, path):
    certificate = read_json(path)
    if certificate.get("certificate_sha256") != digest({key: value for key, value in certificate.items()
                                                       if key != "certificate_sha256"}):
        raise ValueError("Review certificate digest changed")
    current = snapshot_review(repo, first)
    if any(certificate.get(key) != value for key, value in current.items()):
        raise ValueError("Pilot artifacts or reviewer code changed after explicit review")
    for relative in ("state.json", "pairs/report.json", "pairs/plan.json", "pairs/sources.json", "index/sources.json"):
        copy = path.parent / "snapshot" / relative
        if sha(copy) != certificate["launch_snapshot_sha256"][str(first / relative)]:
            raise ValueError("Frozen certificate snapshot changed")
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=Path("output/risk-quality-261006"))
    parser.add_argument("--out", type=Path, required=True, help="Fresh review output directory")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    first, output = host_path(repo, args.first), host_path(repo, args.out)
    if not output.parent.resolve().is_relative_to((repo / "output").resolve()):
        parser.error("Review artifacts must be inside repository output")
    with (repo / "output/risk-quality-pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(create_certificate(repo, first, output))


if __name__ == "__main__":
    main()
