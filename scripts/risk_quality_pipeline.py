"""Bounded first quality stage: old Risk evaluation, 15 new parents, 3 pairs.

Stop at a measured pilot review gate before expanding teacher collection or
rebuilding data. Uses existing local assets/container; never downloads models.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.teacher_risk_pipeline import Pipeline, atomic_json, now, read_json


GROUPS = (("train-a", "train", 12000), ("train-b", "train", 13000),
          ("val-a", "val", 22000))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def host_path(repo, value):
    path = Path(value)
    if path.is_relative_to("/workspace/dl"):
        path = repo / path.relative_to("/workspace/dl")
    return path if path.is_absolute() else repo / path


def process_exists(pid):
    if pid is None:
        return False
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise ValueError("Invalid recorded pipeline PID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_parent_batch(summary):
    rows = summary.get("attempts", [])
    if (summary.get("status") != "complete" or summary.get("pending") != 0
            or summary.get("completed") != 5 or summary.get("planned") != 5
            or len(rows) != 5
            or any(row.get("status") not in {"passed", "stopped"}
                   or row.get("physics_executed") is not True
                   or row.get("sonic_executed") is not True
                   or not isinstance(row.get("task_success"), bool) for row in rows)):
        raise ValueError("Parent pilot is incomplete or contains execution errors")


def checked_batch(repo, root, seed):
    """Validate saved source/config/actual episode evidence before reusing a batch."""
    batches = list(root.glob("batch-*"))
    if len(batches) != 1 or not batches[0].is_dir():
        raise ValueError("Expected exactly one complete parent batch; do not rerun partial evidence")
    batch = batches[0]
    plan, summary = read_json(batch / "plan.json"), read_json(batch / "summary.json")
    check_parent_batch(summary)
    if (plan.get("method") != "B0" or plan.get("plan_sha256") != digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"})
            or summary.get("plan_sha256") != plan["plan_sha256"]
            or plan.get("lock_sha256") != sha(repo / "configs/baseline.lock.json")):
        raise ValueError("Parent baseline plan/asset lock changed")
    for path, expected in plan["source_sha256"].items():
        if sha(host_path(repo, path)) != expected:
            raise ValueError("Parent baseline source changed: " + path)
    group = next((name for name, _, start in GROUPS if start == seed), None)
    if group is None:
        raise ValueError("Unknown fixed parent pilot seed group")
    prompts = read_json(repo / f"output/data-collection-261005/prompts/{group}.json")
    expected_config = {"mode": "batch", "grasp": True, "batch": 5, "prompt": None, "duration": 2.,
        "seed": seed, "cube_xy": None, "start_back": .45, "walk": False, "direct_start": True,
        "table_standoff": .20, "finger_kp": 6., "finger_kd": .4, "hand_approach": "open",
        "hold_seconds": 10., "acquisition_z_min": None, "wrist_offset": None, "alignment_replans": 2,
        "alignment_replan_seconds": 1.6, "acquisition_transition": .3, "contact_profile": "elliptic",
        "lift_seconds": 4.8, "xy_range": [.36, .46, -.30, -.16], "phase_prompts": prompts,
        "prompt_profile": "focused", "gui": False, "device": "musa", "text_device": "musa",
        "text_dtype": "bfloat16", "history_frames": 16, "threads": 4}
    config = dict(plan["config"])
    for key, path in (("ardy_repo", "third_party/ardy"), ("sonic_repo", "third_party/sonic"),
                      ("assets", "checkpoints/baseline")):
        expected_config[key] = str((repo / path).resolve())
        config[key] = str(host_path(repo, config[key]).resolve())
    if config != expected_config:
        raise ValueError("Parent batch configuration differs from the fixed quality pilot")
    requests = plan.get("requests", [])
    if (len(requests) != 5 or [row.get("seed") for row in requests] != list(range(seed, seed + 5))
            or [row.get("index") for row in requests] != list(range(1, 6))
            or any(row.get("task") != "grasp" or row.get("phase_prompts") != prompts
                   or row.get("prompt") is not None or row.get("duration") != 2.
                   or row.get("prompt_profile") != "focused" or row.get("reference") is not None
                   or row.get("reference_sha256") is not None for row in requests)):
        raise ValueError("Parent batch differs from its fixed five-scene budget")
    observed_successes = 0
    for request, row in zip(requests, summary["attempts"]):
        name = row.get("attempt", "")
        if not re.fullmatch(r"attempt-\d{5}", name) or name != f"attempt-{request['index']:05d}":
            raise ValueError("Parent batch contains unexpected/retried attempts")
        attempt = batch / name
        report = read_json(attempt / "report.json")
        if (report != {key: value for key, value in row.items() if key != "attempt"}
                or report.get("request") != request or report.get("plan_sha256") != plan["plan_sha256"]
                or read_json(attempt / "request.json") != request):
            raise ValueError("Actual parent report differs from its completed summary/plan")
        outputs = report.get("outputs_sha256", {})
        required = {"nominal_context.csv", "task.csv", "rollout/metadata.json", "rollout/states.npz",
                    "rollout/scene.mjb"}
        if not required.issubset(outputs):
            raise ValueError("Missing hashed parent training evidence")
        for path, expected in outputs.items():
            artifact = (attempt / path).resolve()
            if not artifact.is_relative_to(attempt.resolve()) or sha(artifact) != expected:
                raise ValueError("Parent episode evidence changed: " + path)
        observed_successes += report["task_success"]
    if (summary.get("successes") != observed_successes or summary.get("failures") != 5 - observed_successes
            or summary.get("success_rate") != observed_successes / 5):
        raise ValueError("Parent success/failure counts disagree with actual outcomes")
    return batch, summary


def accept_parent_exit(code, summary):
    # scripts/run.py returns 1 for physical task failures, even when all trials
    # completed correctly. Execution errors/incomplete evidence still block.
    if code not in (0, 1):
        raise RuntimeError(f"Parent batch exited with unexpected status {code}")
    check_parent_batch(summary)


def check_pair_pilot(report):
    jobs = list(report.get("jobs", {}).values())
    expected = min(3, report.get("planned_pairs", 0))
    if not expected or len(jobs) != expected:
        raise ValueError("Teacher pilot did not execute its bounded candidate budget")
    if any(row.get("status") != "paired" for row in jobs):
        raise ValueError("Teacher pilot contains an invalid or failed execution; review before expansion")


class QualityPipeline(Pipeline):
    def __init__(self, repo, output, preferred_gpu=2):
        self.resumed = False
        old = repo / "output/teacher-risk-261006"
        super().__init__(repo, output, old / "pairs/sources.json",
                         repo / "configs/learning/risk-quality-pilot.json", preferred_gpu)
        self.old = old
        paths = [Path(__file__).absolute(), old / "dataset/manifest.json", old / "dataset/val.npz"]
        paths += [old / f"risk/seed-{seed}/best.pt" for seed in (0, 1)]
        paths += [repo / f"output/data-collection-261005/prompts/{name}.json" for name, _, _ in GROUPS]
        for path in paths:
            self.inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.state.update(schema="dl-risk-quality-pipeline-v1",
                          source_sha256=self.inputs,
                          scope="Validation diagnosis and bounded new teacher pilot; no test evaluation",
                          budgets={"new_train_parents": 10, "new_val_parents": 5,
                                   "initial_pairs": 3, "new_test_parents": 0})
        self.save()

    @classmethod
    def resume(cls, repo, output, preferred_gpu=2):
        """Recover completed stages without relaxing their baseline/data locks."""
        previous_bytes = (output / "state.json").read_bytes()
        previous = json.loads(previous_bytes)
        if (previous.get("schema") != "dl-risk-quality-pipeline-v1"
                or previous.get("run_id") != output.name
                or previous.get("status") not in {"failed", "interrupted"}
                or host_path(repo, previous.get("output", "")).resolve() != output.resolve()):
            raise ValueError("Resume requires this exact failed/interrupted terminal quality run")
        if any(process_exists(previous.get(key)) for key in ("pid", "child_pid")):
            raise ValueError("Recorded parent/child is still active; refuse duplicate work")
        if (output / "continuation-claim.json").exists():
            raise ValueError("A continuation already claimed this first-stage run")
        inputs = dict(previous["source_sha256"])
        runner = repo / "scripts/risk_quality_pipeline.py"
        runner_key = str(runner)
        if runner_key not in inputs:
            raise ValueError("Previous run did not pin its runner source")
        upgrade = None
        for path, expected in inputs.items():
            actual = sha(host_path(repo, path))
            if actual == expected:
                continue
            if path != runner_key:
                raise ValueError("Pinned source/evidence changed; resume cannot relax it: " + path)
            revision = previous.get("runner_source_revision", {}).get("git_commit", previous["git_commit"])
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Previous runner has no immutable Git revision")
            old_blob = subprocess.check_output(["git", "show", revision + ":scripts/risk_quality_pipeline.py"], cwd=repo)
            if hashlib.sha256(old_blob).hexdigest() != expected:
                raise ValueError("Old runner pin does not match its recorded Git blob")
            current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            new_blob = subprocess.check_output(["git", "show", current_commit + ":scripts/risk_quality_pipeline.py"], cwd=repo)
            if hashlib.sha256(new_blob).hexdigest() != actual:
                raise ValueError("Commit the reviewed runner repair before resuming")
            upgrade = {"path": runner_key, "old_sha256": expected, "new_sha256": actual,
                       "old_git_commit": revision, "new_git_commit": current_commit}
        pipeline = cls.__new__(cls)
        pipeline.repo, pipeline.output, pipeline.preferred_gpu = repo, output, preferred_gpu
        pipeline.old = repo / "output/teacher-risk-261006"
        pipeline.sources = pipeline.old / "pairs/sources.json"
        pipeline.config = repo / "configs/learning/risk-quality-pilot.json"
        pipeline.inputs, pipeline.state, pipeline.resumed = inputs, previous.copy(), True
        # Validate all available completed evidence BEFORE changing state or pins.
        pipeline.preflight_existing()
        backup_root = output / "resume-history"
        index = 1
        while (backup_root / f"resume-{index:03d}").exists():
            index += 1
        backup = backup_root / f"resume-{index:03d}"
        artifacts = {str(path.relative_to(output)): sha(path) for path in output.rglob("*")
                     if path.is_file() and not path.is_relative_to(backup_root)}
        backup.mkdir(parents=True, exist_ok=False)
        (backup / "previous-state.json").write_bytes(previous_bytes)
        atomic_json(backup / "artifacts.json", {"schema": "dl-risk-quality-resume-evidence-v1",
                    "at": now(), "previous_state_sha256": hashlib.sha256(previous_bytes).hexdigest(),
                    "artifact_sha256": artifacts, "runner_upgrade": upgrade})
        if upgrade:
            pipeline.inputs[runner_key] = upgrade["new_sha256"]
            pipeline.state["runner_source_revision"] = {"git_commit": upgrade["new_git_commit"],
                                                        "sha256": upgrade["new_sha256"]}
        pipeline.state.pop("error", None)
        pipeline.state.pop("finished_at", None)
        pipeline.state.update(pid=os.getpid(), child_pid=None, status="running", stage="resume-preflight",
                              source_sha256=pipeline.inputs, resume_backup=str(backup))
        pipeline.event("resumed", {"backup": str(backup), "runner_upgrade": upgrade,
                       "behavior": "Reuse verified complete stages; refuse partial artifacts; no new trials for completed parents"})
        return pipeline

    def checked_evaluation(self, seed):
        destination = self.output / "old-risk" / f"seed-{seed}"
        report = read_json(destination / "report.json")
        gate = read_json(destination / "gate.json")
        risk_hash, manifest_hash = sha(self.old / f"risk/seed-{seed}/best.pt"), sha(self.old / "dataset/manifest.json")
        if (report.get("status") != "passed" or report.get("schema") != "dl-risk-evaluation-v1"
                or report.get("split") != "val" or report.get("seed") != seed
                or report.get("risk_sha256") != risk_hash
                or report.get("dataset_manifest_sha256") != manifest_hash
                or report.get("windows_sha256") != sha(self.old / "dataset/val.npz")
                or report.get("predictions_sha256") != sha(destination / "predictions.npz")
                or report.get("gate") != gate or gate.get("split") != "val"
                or gate.get("risk_sha256") != risk_hash
                or gate.get("dataset_manifest_sha256") != manifest_hash
                or gate.get("probability_transform", "raw_sigmoid") != "raw_sigmoid"
                or gate.get("temperature", 1.) != 1.):
            raise ValueError("Existing old Risk validation is incomplete or changed")
        for event in self.state.get("events", []):
            if event["stage"] == "old-validation-complete":
                recorded = next((row for row in event["details"]["evaluations"] if row["seed"] == seed), None)
                if recorded is None or recorded["gate"] != gate or recorded["checkpoint_epoch"] != report["checkpoint_epoch"]:
                    raise ValueError("Old validation differs from its original completion event")
        return report

    def preflight_existing(self):
        for seed in (0, 1):
            destination = self.output / "old-risk" / f"seed-{seed}"
            if destination.exists():
                self.checked_evaluation(seed)
        for name, _, seed in GROUPS:
            root = self.output / "parents" / name
            if root.exists() and any(root.iterdir()):
                checked_batch(self.repo, root, seed)
        # This repair resumes the complete parent-exit failure. Later partial
        # index/pair runs need their own evidence review, never automatic retries.
        if (self.output / "index").exists() or (self.output / "pairs").exists():
            raise ValueError("Resume does not retry existing index/pair stages; review those artifacts separately")

    def event_once(self, stage, details):
        if not any(event["stage"] == stage for event in self.state["events"]):
            self.event(stage, details)

    def stage_name(self, stage, suffix):
        if not self.resumed or not (self.output / (stage + suffix)).exists():
            return stage
        index = 1
        while (self.output / (f"{stage}-resume-{index:03d}" + suffix)).exists():
            index += 1
        return f"{stage}-resume-{index:03d}"

    def execute(self, stage, command):
        # Preserve old logs if a previously missing stage has an orphaned log.
        return super().execute(self.stage_name(stage, ".log"), command)

    def gpu(self, stage):
        available = self.gpus(self.stage_name(stage, "-gpu-snapshot.txt"))
        return self.preferred_gpu if self.preferred_gpu in available else available[0]

    def checked_execute(self, stage, command):
        code = self.execute(stage, command)
        if code:
            raise RuntimeError(f"{stage} exited with status {code}; inspect its log")

    def run(self):
        evaluations = []
        for seed in (0, 1):
            stage = f"old-risk-seed-{seed}"
            destination = self.output / "old-risk" / f"seed-{seed}"
            if not destination.exists():
                self.checked_execute(stage, self.docker(self.gpu(stage), "python", "-m",
                    "experiments.evaluate_risk", "--risk",
                    self.container_path(self.old / f"risk/seed-{seed}/best.pt"), "--data",
                    self.container_path(self.old / "dataset/manifest.json"), "--out",
                    self.container_path(destination), "--device", "musa", "--temperature-report"))
            elif not self.resumed:
                raise ValueError("Old validation destination unexpectedly exists in a fresh run")
            report = self.checked_evaluation(seed)
            evaluations.append({"seed": seed, "report": str(destination / "report.json"),
                                "checkpoint_epoch": report["checkpoint_epoch"], "gate": report["gate"]})
        self.event_once("old-validation-complete", {"evaluations": evaluations})
        assignments = []
        for name, split, seed in GROUPS:
            stage = "parent-" + name
            root = self.output / "parents" / name
            reused = root.exists() and any(root.iterdir())
            code = None
            if reused:
                if not self.resumed:
                    raise ValueError("Parent destination unexpectedly exists in a fresh run")
                batch, summary = checked_batch(self.repo, root, seed)
            else:
                code = self.execute(stage, self.docker(self.gpu(stage), "python", "scripts/run.py",
                    "batch", "--grasp", "--batch", "5", "--seed", str(seed), "--phase-prompts",
                    self.container_path(self.repo / f"output/data-collection-261005/prompts/{name}.json"),
                    "--table-standoff", ".20", "--hold-seconds", "10", "--output-root",
                    self.container_path(root)))
                batch, summary = checked_batch(self.repo, root, seed)
                accept_parent_exit(code, summary)
            assignments.append({"run": self.container_path(batch), "split": split,
                                "prompt_group": "quality-" + name})
            self.event_once(stage + "-complete", {**{key: summary[key] for key in
                       ("planned", "completed", "successes", "failures", "success_rate")},
                       "execution_reused": reused, "collector_exit_code": code})
        declarations = self.output / "batches.json"
        atomic_json(declarations, {"batches": assignments})
        index = self.output / "index"
        self.checked_execute("index", self.docker(self.gpu("index"), "python", "-m",
            "experiments.plan_data", "index", "--batches", self.container_path(declarations),
            "--out", self.container_path(index)))
        indexed = read_json(index / "report.json")
        if indexed.get("pending") or indexed.get("unusable") or not indexed.get("teacher_candidates"):
            raise ValueError("New parent evidence is not ready for a teacher pilot")
        self.event("parents-indexed", {key: indexed[key] for key in
                   ("planned_parents", "candidate_parents", "teacher_candidates", "pending", "unusable")})
        pairs = self.output / "pairs"
        code = self.execute("teacher-pilot", self.docker(self.gpu("teacher-pilot"), "python", "-m",
            "experiments.collect_pairs", "--sources", self.container_path(index / "sources.json"),
            "--out", self.container_path(pairs), "--limit", "3"))
        if code:
            raise RuntimeError("Teacher pilot execution failed; preserve artifacts and inspect report")
        report = read_json(pairs / "report.json")
        check_pair_pilot(report)
        self.event("teacher-pilot-complete", {key: report[key] for key in
                   ("planned_pairs", "pending_pairs", "recovery_verified", "failed_pairs")})
        self.state.update(status="pilot-review", stage="pilot-review", finished_at=now(),
                          next_action="Review verified pairs before resuming remaining candidates, conversion and training")
        self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse verified complete validation/parent stages from a dead failed/interrupted run")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    output = args.out if args.out.is_absolute() else repo / args.out
    if not output.parent.resolve().is_relative_to((repo / "output").resolve()):
        parser.error("Quality artifacts must be inside repository output")
    with (repo / "output/risk-quality-pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            pipeline = QualityPipeline.resume(repo, output, args.gpu)
        else:
            output.mkdir(parents=True, exist_ok=False)
            pipeline = QualityPipeline(repo, output, args.gpu)
        try:
            pipeline.run()
        except BaseException as error:
            pipeline.state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                                  error=f"{type(error).__name__}: {error}", finished_at=now())
            pipeline.save()
            raise


if __name__ == "__main__":
    main()
