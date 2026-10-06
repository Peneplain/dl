"""Durable server workflow: collect pairs, convert, audit, train Risk seeds.

Uses only the Python standard library on the host. Models run inside the
existing MUSA container. Each free physical GPU trains one independent seed.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def idle_gpus(snapshot):
    """Require zero used memory/utilization and no listed process, not spare VRAM."""
    rows = re.findall(r"^\s*(\d+)\s+[^|\n]+\|[^|\n]+\|\s*(\d+)%\s+(\d+)MiB\(\d+MiB\)",
                      snapshot, re.MULTILINE)
    if not rows:
        raise ValueError("Unrecognized mthreads-gmi output; cannot choose GPUs safely")
    processes = snapshot.split("Processes:", 1)[-1] if "Processes:" in snapshot else ""
    busy = {int(x) for x in re.findall(r"^\s*(\d+)\s+\d+\s+.+\s+\d+MiB\s*$",
                                      processes, re.MULTILINE)}
    return [int(index) for index, utilization, memory in rows
            if int(utilization) == 0 and int(memory) == 0 and int(index) not in busy]


def check_pairs(report):
    # Individual rejected pairs are diagnostic outcomes. Incomplete collection
    # must not advance, even if a collector process exited without an exception.
    if (report.get("status") != "complete" or report.get("pending_pairs") != 0
            or not report.get("planned_pairs")
            or len(report.get("jobs", {})) != report["planned_pairs"]):
        raise ValueError("Teacher collection is incomplete or has no candidates")


def check_audit(report):
    for stage in ("risk", "residual"):
        if not report.get(stage, {}).get("ready"):
            raise ValueError(f"Final dataset is not ready for {stage}: "
                             + "; ".join(report.get(stage, {}).get("issues", [])))


class Pipeline:
    def __init__(self, repo, output, sources, config, preferred_gpu=4):
        self.repo, self.output = repo, output
        self.sources, self.config, self.preferred_gpu = sources, config, preferred_gpu
        self.state = {"schema": "dl-teacher-risk-pipeline-v1", "run_id": output.name,
                      "pid": os.getpid(), "status": "running", "stage": "preflight",
                      "started_at": now(), "events": [], "output": str(output),
                      "scope": "Controlled teacher pairs and independent Risk seeds; no DDP or P evaluation"}
        self.inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for folder in ("baseline", "risk_residual", "experiments")
                       for path in (repo / folder).rglob("*.py")}
        for path in (sources, config, repo / "configs/data/perturbations.json",
                     repo / "scripts/train_risk_multiseed.sh", Path(__file__).absolute()):
            self.inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.state["source_sha256"] = self.inputs
        self.state["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        self.save()

    def save(self):
        self.state["updated_at"] = now()
        atomic_json(self.output / "state.json", self.state)

    def event(self, stage, details):
        event_id = f"DL-PIPELINE-{self.output.name}-{len(self.state['events']) + 1}"
        self.state["events"].append({"id": event_id, "stage": stage, "at": now(),
                                     "details": details})
        self.save()

    def check_sources(self):
        for path, expected in self.inputs.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Pipeline source changed during execution: " + path)

    def gpus(self, stage):
        while True:
            snapshot = subprocess.check_output(["mthreads-gmi"], text=True, timeout=20)
            (self.output / f"{stage}-gpu-snapshot.txt").write_text(snapshot)
            available = idle_gpus(snapshot)
            if available:
                self.state.update(status="running", available_gpus=available)
                self.save()
                return available
            self.state.update(status="waiting-for-gpu", stage=stage)
            self.save()
            time.sleep(30)

    def docker(self, gpu, *arguments):
        return ["env", f"MTHREADS_VISIBLE_DEVICES={gpu}", "MUSA_IMAGE=dl-musa-render:latest",
                str(self.repo / "docker/run-musa.sh"), "env", "MUSA_VISIBLE_DEVICES=0", *arguments]

    def container_path(self, path):
        try:
            return str(Path("/workspace/dl") / path.relative_to(self.repo))
        except ValueError:
            return str(path)

    def repository_path(self, path):
        # The shell launcher reads files on the host, then passes paths to its
        # container. Repository-relative paths work in both mount namespaces.
        try:
            return str(path.relative_to(self.repo))
        except ValueError:
            return str(path)

    def execute(self, stage, command):
        self.check_sources()
        log = self.output / f"{stage}.log"
        self.state.update(status="running", stage=stage, command=command, log=str(log))
        print(f"[{now()}] {stage}: {shlex.join(command)}", flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=self.repo, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            self.state["child_pid"] = process.pid
            self.save()
            while True:
                try:
                    return process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if stage == "teacher":
                        try:
                            report = read_json(self.output / "pairs/report.json")
                            self.state["teacher_progress"] = {key: report[key] for key in
                                ("planned_pairs", "pending_pairs", "recovery_verified", "failed_pairs")}
                        except (OSError, ValueError, KeyError):
                            pass
                    self.save()

    def run(self):
        available = self.gpus("teacher")
        gpu = self.preferred_gpu if self.preferred_gpu in available else available[0]
        self.state["teacher_gpu"] = gpu
        pairs, dataset = self.output / "pairs", self.output / "dataset"
        code = self.execute("teacher", self.docker(gpu, "python", "-m", "experiments.collect_pairs",
                            "--sources", self.container_path(self.sources),
                            "--out", self.container_path(pairs)))
        report = read_json(pairs / "report.json")
        check_pairs(report)
        if code not in (0, 1):
            raise RuntimeError(f"Teacher collector exited with unexpected status {code}")
        summary = {key: report[key] for key in
                   ("planned_pairs", "pending_pairs", "recovery_verified", "failed_pairs")}
        summary["collector_exit_code"] = code
        self.event("teacher-complete", summary)
        # The collector may return 1 for rejected individual pairs. A complete
        # report permits conversion; actual Risk AND Residual audits decide readiness.
        code = self.execute("convert", self.docker(gpu, "python", "-m", "experiments.build_dataset",
                            "--sources", self.container_path(pairs / "sources.json"),
                            "--out", self.container_path(dataset), "--tracking-thresholds",
                            ".06", ".06", ".1", ".1", "--skip-ineligible"))
        if code:
            raise RuntimeError(f"Dataset conversion failed with exit status {code}")
        self.event("conversion-complete", read_json(dataset / "report.json"))
        audit_path = self.output / "dataset-audit.json"
        code = self.execute("audit", self.docker(gpu, "python", "-m", "experiments.audit_dataset",
                            "--data", self.container_path(dataset / "manifest.json"),
                            "--out", self.container_path(audit_path)))
        if code:
            raise RuntimeError(f"Dataset audit failed with exit status {code}")
        audit = read_json(audit_path)
        check_audit(audit)
        self.event("audit-passed", audit)
        available = self.gpus("risk")
        self.state["risk_gpus"] = available
        risk_output = self.output / "risk"
        code = self.execute("risk", ["bash", str(self.repo / "scripts/train_risk_multiseed.sh"),
                            ",".join(map(str, available)), self.repository_path(dataset / "manifest.json"),
                            self.repository_path(self.config), self.repository_path(risk_output)])
        summary = read_json(risk_output / "summary.json")
        if (code or summary.get("status") != "passed" or len(summary.get("jobs", [])) != len(available)
                or any(job.get("status") != "passed" for job in summary["jobs"])):
            raise RuntimeError("One or more Risk training seeds failed; inspect per-seed logs")
        self.event("risk-complete", summary)
        self.state.update(status="complete", stage="complete", finished_at=now())
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("output/data-collection-261005/index/sources.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/learning/p.json"))
    parser.add_argument("--teacher-gpu", type=int, default=4)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = [path if path.is_absolute() else repo / path for path in (args.out, args.sources, args.config)]
    output, sources, config = paths
    (repo / "output").mkdir(exist_ok=True)
    with (repo / "output/teacher-risk-pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        pipeline = Pipeline(repo, output, sources, config, args.teacher_gpu)
        try:
            pipeline.run()
        except Exception as error:
            pipeline.state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
            pipeline.event("failed", {"error": pipeline.state["error"], "stage": pipeline.state["stage"]})
            raise


if __name__ == "__main__":
    main()
