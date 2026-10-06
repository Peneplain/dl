"""Bounded first quality stage: old Risk evaluation, 15 new parents, 3 pairs.

Stop at a measured pilot review gate before expanding teacher collection or
rebuilding data. Uses existing local assets/container; never downloads models.
"""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.teacher_risk_pipeline import Pipeline, atomic_json, now, read_json


GROUPS = (("train-a", "train", 12000), ("train-b", "train", 13000),
          ("val-a", "val", 22000))


def check_parent_batch(summary):
    if (summary.get("status") != "complete" or summary.get("pending") != 0
            or summary.get("completed") != 5 or summary.get("planned") != 5
            or any(row.get("status") not in {"passed", "stopped"}
                   for row in summary.get("attempts", []))):
        raise ValueError("Parent pilot is incomplete or contains execution errors")


def check_pair_pilot(report):
    jobs = list(report.get("jobs", {}).values())
    expected = min(3, report.get("planned_pairs", 0))
    if not expected or len(jobs) != expected:
        raise ValueError("Teacher pilot did not execute its bounded candidate budget")
    if any(row.get("status") != "paired" for row in jobs):
        raise ValueError("Teacher pilot contains an invalid or failed execution; review before expansion")


class QualityPipeline(Pipeline):
    def __init__(self, repo, output, preferred_gpu=2):
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

    def gpu(self, stage):
        available = self.gpus(stage)
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
            self.checked_execute(stage, self.docker(self.gpu(stage), "python", "-m",
                "experiments.evaluate_risk", "--risk",
                self.container_path(self.old / f"risk/seed-{seed}/best.pt"), "--data",
                self.container_path(self.old / "dataset/manifest.json"), "--out",
                self.container_path(destination), "--device", "musa", "--temperature-report"))
            report = read_json(destination / "report.json")
            if report.get("status") != "passed" or report.get("split") != "val":
                raise ValueError("Old Risk validation did not complete")
            evaluations.append({"seed": seed, "report": str(destination / "report.json"),
                                "checkpoint_epoch": report["checkpoint_epoch"], "gate": report["gate"]})
        self.event("old-validation-complete", {"evaluations": evaluations})
        assignments = []
        for name, split, seed in GROUPS:
            stage = "parent-" + name
            root = self.output / "parents" / name
            self.checked_execute(stage, self.docker(self.gpu(stage), "python", "scripts/run.py",
                "batch", "--grasp", "--batch", "5", "--seed", str(seed), "--phase-prompts",
                self.container_path(self.repo / f"output/data-collection-261005/prompts/{name}.json"),
                "--table-standoff", ".20", "--hold-seconds", "10", "--output-root",
                self.container_path(root)))
            batches = list(root.glob("batch-*/summary.json"))
            if len(batches) != 1:
                raise ValueError("Expected one fresh parent batch")
            summary = read_json(batches[0])
            check_parent_batch(summary)
            assignments.append({"run": self.container_path(batches[0].parent), "split": split,
                                "prompt_group": "quality-" + name})
            self.event(stage + "-complete", {key: summary[key] for key in
                       ("planned", "completed", "successes", "failures", "success_rate")})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.out if args.out.is_absolute() else repo / args.out
    if not output.parent.resolve().is_relative_to((repo / "output").resolve()):
        parser.error("Quality artifacts must be inside repository output")
    with (repo / "output/risk-quality-pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
