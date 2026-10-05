"""Collect resumable controlled teacher pairs from indexed successful B0 episodes.

This command executes physics only when called explicitly. Planning and indexing
remain separate. Every branch inherits its original parent's declared split.
"""

import argparse
import fcntl
import json
from pathlib import Path
import re

from baseline.common import LOCK, ROOT, sha256, write_json
from baseline.session import digest, source_hashes
from experiments.build_dataset import pair_details
from experiments.collect_pair import collect
from risk_residual.data import audit_parents
from risk_residual.teacher import ScheduledArmOffset


def source_snapshot(source):
    source = Path(source)
    report = json.loads((source / "report.json").read_text())
    recorded = report.get("outputs_sha256", {})
    paths = [source / name for name in ("report.json", "request.json", "settings.json")]
    paths.extend(sorted((source / "ardy").glob("*/reference.npz")))
    plan = source / "plan.json" if (source / "plan.json").is_file() else source.parent / "plan.json"
    if plan.is_file():
        paths.append(plan)
    if not any(path.name == "reference.npz" for path in paths):
        raise ValueError("Teacher candidate has no saved ARDY references")
    for path in paths:
        if path.is_relative_to(source) and path.name != "report.json":
            relative = str(path.relative_to(source))
            if recorded and (relative not in recorded or sha256(path) != recorded[relative]):
                raise ValueError("Teacher source evidence changed: " + relative)
    return {str(path.resolve()): sha256(path) for path in paths}


def pair_manifest(sources_path, perturbations_path):
    sources = json.loads(Path(sources_path).read_text())
    audit_parents(sources["parents"])
    if sources.get("baseline_lock_sha256", sha256(LOCK)) != sha256(LOCK):
        raise ValueError("Source index uses a different frozen baseline")
    variants = json.loads(Path(perturbations_path).read_text())["perturbations"]
    names = set()
    for variant in variants:
        name = variant["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) or name in names:
            raise ValueError("Perturbation IDs must be unique path-safe names")
        names.add(name)
        ScheduledArmOffset(**{key: variant[key] for key in ("phase", "delay", "joint", "amplitude")},
                           perturb=True)
    if not variants:
        raise ValueError("At least one bounded perturbation is required")
    jobs = []
    for parent in sources["parents"]:
        if parent.get("pair"):
            raise ValueError("Pair only original indexed B0 episodes, not existing paired branches")
        path = Path(parent["nominal"]).resolve()
        report_path = path / "report.json"
        if parent.get("source_report_sha256") and sha256(report_path) != parent["source_report_sha256"]:
            raise ValueError("Source report changed after indexing")
        report = json.loads(report_path.read_text())
        if report.get("status") != "passed" or not report.get("task_success"):
            continue
        if not report.get("physics_executed") or not report.get("sonic_executed"):
            raise ValueError("Successful teacher candidate did not execute physics and SONIC")
        snapshot = source_snapshot(path)
        for variant in variants:
            job = {"parent": parent, "source": str(path), "perturbation": variant,
                   "source_files_sha256": snapshot}
            job["id"] = digest(job)[:20]
            jobs.append(job)
    manifest = {"schema": "dl-pair-batch-v1", "sources_sha256": sha256(sources_path),
                "perturbations_sha256": sha256(perturbations_path), "jobs": jobs,
                "baseline_lock_sha256": sha256(LOCK), "baseline_sources_sha256": source_hashes(),
                "collector_sha256": sha256(Path(__file__).with_name("collect_pair.py")),
                "batch_collector_sha256": sha256(__file__),
                "teacher_sha256": sha256(ROOT / "risk_residual/teacher.py")}
    manifest["plan_sha256"] = digest(manifest)
    return manifest


def export_sources(original, manifest, results, output):
    branches = {}
    for job in manifest["jobs"]:
        result = results.get(job["id"])
        if not result or result["status"] != "paired":
            continue
        pair = checked_pair(job, result["pair"])
        parent = job["parent"]
        nominal = Path(pair["branches"]["nominal"]["report"]).parent
        branch = {**parent, "parent_id": parent["parent_id"] + "--" + job["perturbation"]["id"],
                  "nominal": str(nominal), "pair": result["pair"],
                  "source_report_sha256": sha256(nominal / "report.json")}
        branches.setdefault(parent["parent_id"], []).append(branch)
    parents = []
    for parent in original["parents"]:
        # The converter includes each pair's clean branch. Do not add the same
        # original clean episode again when paired branches are available.
        parents.extend(branches.get(parent["parent_id"], [parent]))
    audit_parents(parents)
    write_json(output / "sources.json", {"parents": parents,
               "baseline_lock_sha256": sha256(LOCK), "pair_plan_sha256": manifest["plan_sha256"],
               "original_sources_sha256": manifest["sources_sha256"]})


def checked_pair(job, path):
    pair = pair_details(path)
    if (Path(pair["source_attempt"]).resolve() != Path(job["source"]).resolve() or
            pair["source_report_sha256"] != job["source_files_sha256"][str(Path(job["source"]) / "report.json")] or
            any(pair[key] != job["perturbation"][key] for key in ("phase", "delay", "joint", "amplitude"))):
        raise ValueError("Paired evidence belongs to a different source or perturbation")
    return pair


def collect_batch(sources_path, perturbations_path, output, *, resume=False, limit=None):
    output = Path(output).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest = pair_manifest(sources_path, perturbations_path)
    if output.exists():
        if not resume:
            raise FileExistsError("Use a fresh pair directory or explicit --resume")
        saved = json.loads((output / "plan.json").read_text())
        if saved != manifest:
            raise ValueError("Resume source, references, collector or perturbations changed")
    else:
        if resume:
            raise FileNotFoundError("Resume pair directory does not exist")
        output.mkdir(parents=True)
        write_json(output / "plan.json", manifest)
    original = json.loads(Path(sources_path).read_text())
    with (output / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_path = output / "report.json"
        saved_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        if saved_report and saved_report.get("plan_sha256") != manifest["plan_sha256"]:
            raise ValueError("Pair report belongs to a different collection plan")
        results = saved_report.get("jobs", {})
        if set(results) - {job["id"] for job in manifest["jobs"]}:
            raise ValueError("Pair report contains undeclared jobs")
        attempted = 0
        def save():
            export_sources(original, manifest, results, output)
            pending = sum(job["id"] not in results or results[job["id"]]["status"] == "interrupted"
                          for job in manifest["jobs"])
            status = "no-teacher-candidates" if not manifest["jobs"] else ("pending" if pending else "complete")
            report = {"status": status, "planned_pairs": len(manifest["jobs"]),
                      "pending_pairs": pending, "recovery_verified": sum(row.get("recovery_verified", False)
                                                                          for row in results.values()),
                      "failed_pairs": sum(row["status"] in {"failed", "unpaired", "invalid_execution"}
                                          for row in results.values()),
                      "jobs": results, "plan_sha256": manifest["plan_sha256"]}
            write_json(report_path, report)
            return report
        save()
        for job in manifest["jobs"]:
            previous = results.get(job["id"])
            if previous and previous["status"] != "interrupted":
                if previous["status"] == "paired":
                    checked_pair(job, previous["pair"])
                continue
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            root = output / "pairs" / job["id"]
            root.mkdir(parents=True, exist_ok=True)
            attempt = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:03d}"
            print(f"[PAIR] {job['parent']['parent_id']} {job['perturbation']['id']} -> {attempt}", flush=True)
            try:
                pair = collect(job["source"], attempt,
                               **{key: job["perturbation"][key] for key in ("phase", "delay", "joint", "amplitude")})
                status = "unpaired"
                if pair["pair_state_verified"]:
                    checked_pair(job, attempt / "pair.json")
                    status = ("paired" if all(branch["status"] in {"passed", "stopped"} and
                              branch["physics_executed"] and branch["sonic_executed"]
                              for branch in pair["branches"].values()) else "invalid_execution")
                results[job["id"]] = {"status": status, "pair": str(attempt / "pair.json"),
                                      "teacher_verified": pair["teacher_verified"],
                                      "recovery_verified": pair["recovery_verified"]}
            except KeyboardInterrupt:
                results[job["id"]] = {"status": "interrupted", "attempt": str(attempt)}
                save()
                raise
            except Exception as error:
                results[job["id"]] = {"status": "failed", "attempt": str(attempt),
                                      "error": f"{type(error).__name__}: {error}"}
            save()
        return save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--perturbations", type=Path, default=ROOT / "configs/data/perturbations.json")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum newly attempted pairs in this invocation")
    args = parser.parse_args()
    result = collect_batch(args.sources, args.perturbations, args.out, resume=args.resume, limit=args.limit)
    print(json.dumps({key: value for key, value in result.items() if key != "jobs"}), flush=True)
    if result["failed_pairs"] or result["status"] == "no-teacher-candidates":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
