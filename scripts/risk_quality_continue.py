"""Continue one reviewed, bounded teacher pilot through exploratory B0/P checks.

Host standard library only. No new scene budget, test selection, downloads,
DDP, automatic retries of rejected pairs, or edits to old experiment evidence.
Start once with nohup after terminal pilot-review, or with an explicit review
certificate for the one documented missing-reference execution limitation.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.teacher_risk_pipeline import Pipeline, atomic_json, check_audit, check_pairs, now, read_json


THRESHOLDS = [.14, .14, .1, .12]

# Runs inside the existing container; host orchestration stays stdlib-only.
DIAGNOSE_CODE = '''import json,sys,numpy as np
from pathlib import Path
from experiments.diagnose_labels import diagnose,branch_layout,physical_hold,reasons,summarize,KEYS
manifest,source,destination=map(Path,sys.argv[1:])
report=diagnose(manifest,source_path=source)
meta=json.loads(manifest.read_text())
parents={p["parent_id"]:p for p in json.loads(source.read_text())["parents"]}
report["control_cohorts"]={}
for split in ("train","val"):
    with np.load(manifest.parent/meta["windows"][split],allow_pickle=False) as archive:
        arrays={key:archive[key] for key in KEYS}
    branch,success,attempts,origins=branch_layout(arrays,parents)
    stable=physical_hold(arrays,branch,success,attempts)
    cohort=np.asarray([parents[str(pid)]["control_cohort"] for pid in arrays["parent_id"]])
    report["control_cohorts"][split]={}
    for name in ("old-dense-velocity","current-sparse-velocity"):
        mask=cohort==name
        report["control_cohorts"][split][name]={
          "verified_corrections":int((mask&arrays["correction_sample"]).sum()),
          "physical_stable_hold":summarize(arrays,stable&mask,
             np.asarray(meta["provenance"]["label_thresholds"]["tracking"]),reasons(arrays),origins=origins)}
destination.write_text(json.dumps(report,indent=2,allow_nan=False)+"\\n")
'''


class InsufficientEvidence(ValueError):
    """A measured pilot gate needs a separately bounded next plan, not retries."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host_path(repo, value):
    path = Path(value)
    if path.is_relative_to("/workspace/dl"):
        path = repo / path.relative_to("/workspace/dl")
    if not path.is_absolute():
        path = repo / path
    return path


def verify_hashes(repo, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing pinned source hashes")
    for path, expected in hashes.items():
        if sha(host_path(repo, path)) != expected:
            raise ValueError("Pinned source/evidence changed: " + path)


def verified_pair(repo, job, result):
    if result.get("status") != "paired":
        raise ValueError("Pilot pair is not a valid completed physical pair")
    verify_hashes(repo, job["source_files_sha256"])
    pair = read_json(host_path(repo, result["pair"]))
    if (not pair.get("pair_state_verified") or
            host_path(repo, pair["source_attempt"]) != host_path(repo, job["source"]) or
            pair["source_report_sha256"] != sha(host_path(repo, job["source"]) / "report.json") or
            any(pair[key] != job["perturbation"][key] for key in ("phase", "delay", "joint", "amplitude"))):
        raise ValueError("Pair provenance differs from its immutable job")
    branches = pair["branches"]
    for branch in branches.values():
        report_path = host_path(repo, branch["report"])
        report = read_json(report_path)
        if (sha(report_path) != branch["report_sha256"] or
                sha(report_path.parent / "scene.xml") != branch["scene_sha256"] or
                sha(report_path.parent / "rollout/scene.mjb") != branch["model_sha256"] or
                report.get("status") not in {"passed", "stopped"} or
                report.get("status") != branch["status"] or
                bool(report.get("task_success")) != branch["task_success"] or
                not report.get("physics_executed") or not report.get("sonic_executed")):
            raise ValueError("Pair branch outcome or physical evidence changed")
    nominal, teacher = branches["nominal"], branches["teacher"]
    a, b = nominal["activation"], teacher["activation"]
    if (a is None or b is None or a["snapshot_sha256"] != b["snapshot_sha256"] or
            a["frame_index"] != b["frame_index"] or abs(a["time"] - b["time"]) > 1e-8 or
            nominal["scene_sha256"] != teacher["scene_sha256"] or
            nominal["model_sha256"] != teacher["model_sha256"]):
        raise ValueError("Pair did not start from the same physical/controller state")
    recovery = teacher["status"] == "passed" and teacher["task_success"] and not nominal["task_success"]
    if bool(pair.get("recovery_verified")) != recovery or bool(result.get("recovery_verified")) != recovery:
        raise ValueError("Recovery flag disagrees with verified branch outcomes")
    return recovery


def pilot_gate(repo, first):
    state = read_json(first / "state.json")
    if (state.get("schema") != "dl-risk-quality-pipeline-v1" or
            state.get("run_id") != "risk-quality-261006" or
            state.get("status") != "pilot-review" or state.get("stage") != "pilot-review"):
        raise ValueError("The exact first quality run has not reached terminal pilot-review")
    verify_hashes(repo, state["source_sha256"])
    original = read_json(first / "index/sources.json")["parents"]
    unique = {(row["prompt_group"], str(row["scene_seed"])) for row in original}
    if (len(unique) > 15 or not unique or any(row["split"] not in {"train", "val"} for row in original)):
        raise ValueError("New teacher parents exceed the bounded train/val pilot")
    plan = read_json(first / "pairs/plan.json")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest():
        raise ValueError("Pair plan digest changed")
    if plan.get("sources_sha256") != sha(first / "index/sources.json"):
        raise ValueError("Original indexed pilot parents changed since pair planning")
    verify_hashes(repo, plan["baseline_sources_sha256"])
    report = read_json(first / "pairs/report.json")
    jobs = {job["id"]: job for job in plan["jobs"]}
    if (not jobs or len(jobs) > len(unique) * 3 or len(jobs) > 45 or
            report.get("plan_sha256") != plan.get("plan_sha256") or
            len(report.get("jobs", {})) != min(3, len(jobs)) or
            set(report["jobs"]) - set(jobs) or report.get("failed_pairs")):
        raise ValueError("Initial bounded pair pilot or saved plan is inconsistent")
    recovery = sum(verified_pair(repo, jobs[key], result) for key, result in report["jobs"].items())
    clean_success = sum(read_json(host_path(repo, result["pair"]))["branches"]["teacher"]["task_success"]
                        for result in report["jobs"].values())
    if clean_success < 1:
        raise InsufficientEvidence("No physically verified clean teacher success in initial pairs")
    return state, plan, {"initial_pairs": len(report["jobs"]), "verified_recoveries": recovery,
                         "verified_clean_successes": clean_success,
                         "independent_new_parents": len(unique), "planned_pairs": len(jobs)}


def pilot_admission(repo, first, certificate_path=None):
    """The original three-valid-pair gate is unchanged without a certificate."""
    if certificate_path is None:
        state, plan, counts = pilot_gate(repo, first)
        return state, plan, counts, None
    # Lazy import avoids a dependency cycle with the stdlib review command.
    from scripts.review_risk_quality_pilot import validate_certificate
    certificate = validate_certificate(repo, first, certificate_path)
    return (certificate["original_state"], read_json(first / "pairs/plan.json"),
            certificate["counts"], certificate)


def preserved_review(repo, first, certificate, report):
    """Mutable collector summaries may advance; old evidence cannot be retried."""
    verify_hashes(repo, certificate["immutable_evidence_sha256"])
    old_results = certificate["initial_report"]["jobs"]
    if any(report.get("jobs", {}).get(key) != original for key, original in old_results.items()):
        raise ValueError("A previously reviewed pair was altered or retried")
    expected = set(old_results) | set(certificate["remaining_job_ids"])
    if set(report.get("jobs", {})) != expected or report.get("plan_sha256") != certificate["pair_plan_sha256"]:
        raise ValueError("Collector expanded beyond the fixed original remaining candidates")
    sources = read_json(first / "pairs/sources.json")
    valid_paths = {host_path(repo, row["pair"]).resolve() for row in report["jobs"].values()
                   if row["status"] == "paired"}
    if any(host_path(repo, row["pair"]).resolve() not in valid_paths
           for row in sources["parents"] if row.get("pair")):
        raise ValueError("Invalid/rejected execution was exported as a training source")
    outcomes = {}
    for result in report["jobs"].values():
        outcomes[result["status"]] = outcomes.get(result["status"], 0) + 1
    return {"attempted_candidates": len(expected), "outcomes": outcomes,
            "known_invalid_preserved": certificate["excluded_job_ids"],
            "physical_pair_denominator": outcomes.get("paired", 0),
            "scope": "Exploratory complete-case pairs; feedback-replan censoring is nonrandom. Invalid executions are not physical failures or training pairs."}


def calibration_gate(repo, artifact_path, diagnosis_path, old_manifest):
    artifact, manifest = read_json(artifact_path), read_json(old_manifest)
    if (artifact.get("schema") != "dl-label-calibration-v1" or
            artifact.get("split") != "train" or artifact.get("calibration_split") != "train" or
            artifact.get("exploratory") is not True or artifact.get("frozen") is not True or
            artifact.get("test_used") is not False or artifact.get("labels_written") is not False or
            artifact.get("automatic_application") is not False or
            artifact.get("candidate_tracking_thresholds_rad") != THRESHOLDS or
            artifact.get("input_manifest_sha256") != sha(old_manifest) or
            artifact.get("input_source_plan_sha256") != manifest["provenance"]["source_plan_sha256"] or
            artifact.get("input_diagnosis_sha256") != sha(diagnosis_path) or
            artifact.get("calibrator_revision_sha256") != sha(repo / "experiments/calibrate_labels.py")):
        raise ValueError("Label thresholds lack frozen train-only diagnosis/provenance")
    return artifact


def merge_sources(old, current):
    cohorts = {
        "old-dense-velocity": "Immutable original teacher controls differentiated full dense nominal references.",
        "current-sparse-velocity": "Current controls preserve B0 sparse nominal velocity plus correction-only derivative."}
    parents, ids, owners = [], set(), {"scene_seed": {}, "prompt_group": {}}
    for cohort, source in zip(cohorts, (old, current)):
        for original in source["parents"]:
            row = {**original, "control_cohort": cohort}
            if row["parent_id"] in ids or row["split"] not in {"train", "val", "test"}:
                raise ValueError("Duplicate/invalid parent in merged sources")
            if cohort == "current-sparse-velocity" and row["split"] == "test":
                raise ValueError("The current pilot must not add test parents")
            ids.add(row["parent_id"])
            for field, values in owners.items():
                value = str(row[field])
                if value in values and values[value] != row["split"]:
                    raise ValueError("Parent split leakage: " + field)
                values[value] = row["split"]
            parents.append(row)
    if old["baseline_lock_sha256"] != current["baseline_lock_sha256"]:
        raise ValueError("Frozen baseline asset locks differ")
    return {"parents": parents, "baseline_lock_sha256": old["baseline_lock_sha256"],
            "control_cohorts": cohorts,
            "scope": "Mixed immutable control cohorts for exploratory supervised training; not current-controller evaluation."}


def select_seed(summary):
    jobs = summary.get("jobs", [])
    if (summary.get("status") != "passed" or {job.get("seed") for job in jobs} != {0, 1} or
            len(jobs) != 2 or any(job.get("status") != "passed" or
                not isinstance(job.get("best_val_loss"), (int, float)) or
                not math.isfinite(job["best_val_loss"]) for job in jobs)):
        raise ValueError("Both independent Risk seeds must finish with finite validation loss")
    return min(jobs, key=lambda row: (row["best_val_loss"], row["seed"]))["seed"]


def diagnostic_gate(diagnosis):
    if diagnosis.get("schema") != "dl-label-diagnosis-v1" or set(diagnosis.get("splits", {})) != {"train", "val"}:
        raise ValueError("Validation diagnostic must contain train/val only")
    if diagnosis.get("tracking_thresholds_rad") != THRESHOLDS:
        raise ValueError("Validation diagnostic changed the frozen label candidate")
    observed = diagnosis["splits"]["val"]["physical_stable_hold"]
    windows, positives = observed["windows"], observed["reason_windows"]["tracking"]
    if not windows or positives / windows > .10:
        raise InsufficientEvidence("Frozen label candidate violates the predeclared validation stable-hold tracking-positive <=10% guard; do not retune from validation")
    cohorts = diagnosis["control_cohorts"]
    if (cohorts["train"]["old-dense-velocity"]["verified_corrections"] < 28 or
            cohorts["val"]["old-dense-velocity"]["verified_corrections"] < 7):
        raise InsufficientEvidence("Old verified direct correction supervision was lost during conversion")
    return {"validation_stable_hold_windows": windows, "tracking_positive": positives,
            "tracking_positive_fraction": positives / windows, "guard": .10,
            "control_cohorts": cohorts}


def claim_first(first, output):
    claim = {"schema": "dl-risk-quality-continuation-claim-v1", "first": str(first.resolve()),
             "output": str(output.resolve()), "pid": os.getpid(), "at": now()}
    with (first / "continuation-claim.json").open("x") as stream:
        json.dump(claim, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return claim


class Continuation(Pipeline):
    def __init__(self, repo, output, first, calibration, diagnosis, pilot_review=None):
        self.first, self.old = first, repo / "output/teacher-risk-261006"
        self.review, self.pair_plan, review_counts, self.certificate = pilot_admission(repo, first, pilot_review)
        calibration_gate(repo, calibration, diagnosis, self.old / "dataset/manifest.json")
        self.calibration = calibration
        super().__init__(repo, output, first / "index/sources.json",
                         repo / "configs/learning/risk-quality-pilot.json", 2)
        self.inputs.update({str(host_path(repo, path).resolve()): value
                            for path, value in self.review["source_sha256"].items()})
        if self.certificate is not None:
            self.inputs.update({str(host_path(repo, path).resolve()): value
                                for path, value in self.certificate["immutable_evidence_sha256"].items()})
            from scripts import review_risk_quality_pilot
            for path in (pilot_review, Path(review_risk_quality_pilot.__file__),
                         *(pilot_review.parent / "snapshot" / relative for relative in
                           ("state.json", "pairs/report.json", "pairs/plan.json", "pairs/sources.json", "index/sources.json"))):
                self.inputs[str(path.resolve())] = sha(path)
        for path in (Path(__file__).absolute(), first / "state.json", first / "pairs/plan.json",
                     calibration, diagnosis, self.old / "pairs/sources.json", self.old / "dataset/manifest.json",
                     repo / "configs/learning/p.json", repo / "output/data-collection-261005/prompts/val-a.json"):
            self.inputs[str(path)] = sha(path)
        self.state.update(schema="dl-risk-quality-continuation-v1", source_sha256=self.inputs,
                          review_counts=review_counts,
                          scope="Bounded expansion, train-only exploratory labels, two Risk seeds, one Residual, three validation pilots; no final test",
                          budgets={"maximum_new_parents": 15, "maximum_pairs": 45, "risk_seeds": 2,
                                   "risk_max_epochs": 30, "residual_seeds": 1, "residual_max_epochs": 30,
                                   "paired_validation_trials": 3},
                          selection_rule="Lowest best total val loss; ties choose lower seed; raw validation max-F1 gate")
        worktree_status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                                                 cwd=repo, text=True).splitlines()
        self.state.update(source_worktree_dirty=bool(worktree_status), source_worktree_status=worktree_status,
                          code_version="git_commit is the base revision; source_sha256 identifies the actual pinned working-tree code.")
        if self.certificate is not None:
            self.state["explicit_pilot_review"] = {"path": str(pilot_review), "sha256": sha(pilot_review),
                "certificate_sha256": self.certificate["certificate_sha256"],
                "original_status": "failed", "excluded_job_ids": self.certificate["excluded_job_ids"],
                "remaining_job_ids": self.certificate["remaining_job_ids"], "scope": self.certificate["scope"]}
        self.save()

    def checked(self, stage, command):
        if self.execute(stage, command):
            raise RuntimeError(stage + " failed; inspect preserved stage log")

    def gpu(self, stage):
        return self.gpus(stage)[0]

    def run(self):
        self.event("pilot-approved", self.state["review_counts"])
        pairs = self.first / "pairs"
        expansion_limit = (len(self.certificate["remaining_job_ids"]) if self.certificate is not None
                           else len(self.pair_plan["jobs"]))
        code = self.execute("teacher-expand", self.docker(self.gpu("teacher-expand"), "python", "-m",
            "experiments.collect_pairs", "--sources", self.container_path(self.sources), "--out",
            self.container_path(pairs), "--resume", "--limit", str(expansion_limit)))
        report = read_json(pairs / "report.json")
        check_pairs(report)
        if self.certificate is not None:
            self.event("reviewed-evidence-preserved", preserved_review(self.repo, self.first, self.certificate, report))
        if code not in (0, 1):
            raise RuntimeError("Unexpected teacher collector exit status")
        recovery_parents = {"train": set(), "val": set()}
        jobs = {job["id"]: job for job in self.pair_plan["jobs"]}
        for key, result in report["jobs"].items():
            if result["status"] == "paired" and verified_pair(self.repo, jobs[key], result):
                parent = jobs[key]["parent"]
                recovery_parents[parent["split"]].add(parent["parent_id"])
        counts = {key: len(value) for key, value in recovery_parents.items()}
        self.event("teacher-plan-processed", {**{key: report[key] for key in
            ("planned_pairs", "pending_pairs", "recovery_verified", "failed_pairs")},
            "new_recovery_parents_by_split": counts})
        if any(counts[split] < 1 for split in ("train", "val")):
            raise InsufficientEvidence("New independent recovery coverage is absent in train or val; preserve completed pairs and review a separately bounded collection plan. Counts: " + str(counts))
        self.event("teacher-expansion-verified", {"new_recovery_parents_by_split": counts})
        merged = merge_sources(read_json(self.old / "pairs/sources.json"), read_json(pairs / "sources.json"))
        merged["source_plans_sha256"] = {str(path): sha(path) for path in
                                        (self.old / "pairs/sources.json", pairs / "sources.json")}
        merged["label_calibration_sha256"] = sha(self.calibration)
        sources = self.output / "merged-sources.json"
        atomic_json(sources, merged)
        self.inputs[str(sources)] = sha(sources)
        dataset = self.output / "dataset"
        self.checked("convert", self.docker(self.gpu("convert"), "python", "-m", "experiments.build_dataset",
            "--sources", self.container_path(sources), "--out", self.container_path(dataset),
            "--tracking-thresholds", *map(str, THRESHOLDS), "--skip-ineligible"))
        audit_path = self.output / "dataset-audit.json"
        self.checked("audit", self.docker(self.gpu("audit"), "python", "-m", "experiments.audit_dataset",
            "--data", self.container_path(dataset / "manifest.json"), "--out", self.container_path(audit_path)))
        audit = read_json(audit_path)
        check_audit(audit)
        self.event("dataset-audit-passed", {"splits": audit["splits"], "risk": audit["risk"],
            "residual": audit["residual"], "new_recovery_parents_by_split": counts,
            "manifest": str(dataset / "manifest.json"), "label_thresholds_rad": THRESHOLDS})
        for path in (dataset / "manifest.json", *(dataset / f"{split}.npz" for split in ("train", "val", "test"))):
            self.inputs[str(path)] = sha(path)
        diagnostic = self.output / "frozen-label-validation.json"
        self.checked("label-validation", self.docker(self.gpu("label-validation"), "python", "-c", DIAGNOSE_CODE,
            self.container_path(dataset / "manifest.json"), self.container_path(sources), self.container_path(diagnostic)))
        guard = diagnostic_gate(read_json(diagnostic))
        self.event("frozen-label-validation-passed", guard)
        while True:
            available = self.gpus("risk")
            if len(available) >= 2:
                break
            self.state.update(status="waiting-for-two-gpus", stage="risk")
            self.save()
            time.sleep(30)
        risk = self.output / "risk"
        self.state["risk_gpus"] = available[:2]
        self.checked("risk", ["bash", str(self.repo / "scripts/train_risk_multiseed.sh"),
            ",".join(map(str, available[:2])), self.repository_path(dataset / "manifest.json"),
            self.repository_path(self.config), self.repository_path(risk)])
        summary = read_json(risk / "summary.json")
        chosen = select_seed(summary)
        for seed in (0, 1):
            self.checked(f"risk-val-{seed}", self.docker(self.gpu(f"risk-val-{seed}"), "python", "-m",
                "experiments.evaluate_risk", "--risk", self.container_path(risk / f"seed-{seed}/best.pt"),
                "--data", self.container_path(dataset / "manifest.json"), "--out",
                self.container_path(self.output / f"risk-validation/seed-{seed}"), "--device", "musa"))
        checkpoint, gate = risk / f"seed-{chosen}/best.pt", self.output / f"risk-validation/seed-{chosen}/gate.json"
        self.event("risk-training-complete", {"jobs": summary["jobs"], "chosen_seed": chosen,
            "selection_rule": self.state["selection_rule"], "checkpoint": str(checkpoint), "gate": str(gate)})
        residual = self.output / "residual"
        self.checked("residual", self.docker(self.gpu("residual"), "python", "-m", "experiments.train",
            "--stage", "residual", "--interface", "P", "--seed", "0", "--device", "musa", "--data",
            self.container_path(dataset / "manifest.json"), "--risk", self.container_path(checkpoint),
            "--config", self.container_path(self.repo / "configs/learning/p.json"), "--out", self.container_path(residual)))
        if read_json(residual / "report.json").get("status") != "passed":
            raise ValueError("Residual training did not pass")
        self.event("residual-training-complete", read_json(residual / "report.json"))
        requests = self.output / "paired-requests.json"
        prompts = read_json(self.repo / "output/data-collection-261005/prompts/val-a.json")
        atomic_json(requests, [{"seed": seed, "phase_prompts": prompts} for seed in range(24000, 24003)])
        outcomes = {}
        for method in ("B0", "P"):
            arguments = ["python", "-m", "experiments.run_policy", "--method", method, "--requests",
                         self.container_path(requests), "--out", self.container_path(self.output / f"policy/{method}")]
            if method == "P":
                arguments += ["--risk", self.container_path(checkpoint), "--residual",
                              self.container_path(residual / "best.pt"), "--gate", self.container_path(gate)]
            arguments += ["--", "--device", "musa", "--text-device", "musa", "--table-standoff", ".20", "--hold-seconds", "10"]
            self.checked("policy-" + method, self.docker(self.gpu("policy-" + method), *arguments))
            outcomes[method] = read_json(self.output / f"policy/{method}/summary.json")
            if outcomes[method].get("completed") != 3 or outcomes[method].get("status") != "complete":
                raise ValueError("Matched exploratory policy trials are incomplete")
        a, b = outcomes["B0"]["attempts"], outcomes["P"]["attempts"]
        if [row["paired_episode_id"] for row in a] != [row["paired_episode_id"] for row in b]:
            raise ValueError("B0/P exploratory episodes are not matched")
        self.event("exploratory-policy-complete", {method: {key: result[key] for key in
            ("completed", "successes", "success_rate", "success_wilson_95")} for method, result in outcomes.items()})
        self.state.update(status="complete", stage="complete", finished_at=now(),
                          conclusion="Exploratory validation pilot only; three pairs cannot establish grasp improvement.")
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=Path("output/risk-quality-261006"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--pilot-review", type=Path,
                        help="Explicit immutable certificate for the exact known failed teacher pilot; no generic failure override")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    first, calibration, diagnosis, output = [host_path(repo, path) for path in
                                            (args.first, args.calibration, args.diagnosis, args.out)]
    pilot_review = host_path(repo, args.pilot_review) if args.pilot_review is not None else None
    if output.exists() or not output.parent.resolve().is_relative_to((repo / "output").resolve()):
        parser.error("Use a fresh output directory inside repository output")
    # The shared first-stage lock prevents mutation of a still-running pilot.
    with (repo / "output/risk-quality-pipeline.lock").open("a") as first_lock, \
            (repo / "output/risk-quality-continue.lock").open("a") as continuation_lock:
        for lock in (first_lock, continuation_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        pipeline = None
        try:
            pipeline = Continuation(repo, output, first, calibration, diagnosis, pilot_review)
            pipeline.check_sources()
            pipeline.state["continuation_claim"] = claim_first(first, output)
            pipeline.save()
            pipeline.run()
        except BaseException as error:
            status = ("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else
                      "insufficient-evidence" if isinstance(error, InsufficientEvidence) else "failed")
            if pipeline is not None:
                failed_stage = pipeline.state["stage"]
                pipeline.state.update(status=status, error=f"{type(error).__name__}: {error}", finished_at=now())
                pipeline.event("failed", {"stage": failed_stage, "error": pipeline.state["error"]})
            else:
                atomic_json(output / "state.json", {"schema": "dl-risk-quality-continuation-v1", "run_id": output.name,
                    "status": status, "stage": "preflight", "error": f"{type(error).__name__}: {error}", "updated_at": now()})
            raise


if __name__ == "__main__":
    main()
