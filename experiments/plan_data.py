"""Plan disjoint B0 batches, then index completed episodes without running physics."""

import argparse
import csv
import json
from pathlib import Path
import re
import shlex

from baseline.common import LOCK, ROOT, sha256, write_json
from baseline.grasp import PHASES
from baseline.session import digest, latest_result, make_plan, session_directory
from risk_residual.data import audit_parents
from risk_residual.provenance import own_prompt
from scripts.run import config_for, parse_args, parse_request

COLLECTION_SCHEMA = "dl-collection-v1"


def checked_plan(path):
    plan = json.loads(Path(path).read_text())
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != digest(unsigned):
        raise ValueError(f"Collection/session plan hash mismatch: {path}")
    return plan


def plan_collection(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh collection plan directory")
    spec = json.loads(config_path.read_text())
    baseline_args = spec.get("baseline_args", [])
    reserved = {"--seed", "--batch", "--phase-prompts", "--prompt", "--resume",
                "--output-root", "--plan-only", "--gui"}
    if (not isinstance(baseline_args, list) or
            any(not isinstance(value, str) or value.split("=")[0] in reserved
                for value in baseline_args)):
        raise ValueError("baseline_args cannot override collection seeds, prompts or output")
    groups, parents, owners, planned, names = [], [], {}, [], set()
    phases = {phase for phase, _ in PHASES}
    for group in spec["groups"]:
        name, split = group["id"], group["split"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) or name in names:
            raise ValueError("Prompt group IDs must be unique lowercase path-safe names")
        names.add(name)
        count, seed = group["count"], group["seed_start"]
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed + count > 2**32):
            raise ValueError("Each group needs a positive count and a uint32 seed range")
        prompts = group["phase_prompts"]
        if not isinstance(prompts, dict) or set(prompts) != phases:
            raise ValueError("Each prompt group must explicitly define all eight phases")
        args = parse_args(["batch", "--grasp", *baseline_args,
                           "--seed", str(seed), "--batch", str(count)])
        args.phase_prompts = prompts
        requests = [parse_request({}, index + 1, args) for index in range(count)]
        own_prompt(owners, requests[0], split)
        batch = make_plan(config_for(args), requests)
        relative = f"batches/{name}"
        groups.append({"id": name, "split": split, "batch": relative,
                       "plan_sha256": batch["plan_sha256"], "count": count})
        planned.append((name, batch, prompts))
        parents.extend({"parent_id": f"{name}-{request['seed']}", "split": split,
                        "prompt_group": name, "scene_seed": request["seed"],
                        "batch": relative, "index": request["index"]}
                       for request in requests)
    audit_parents(parents)
    if {parent["split"] for parent in parents} != {"train", "val", "test"}:
        raise ValueError("Collection plan must contain train, val and test groups")
    collection = {"schema": COLLECTION_SCHEMA, "config_sha256": sha256(config_path),
                  "baseline_lock_sha256": sha256(LOCK), "groups": groups,
                  "parents": parents, "status": "planned", "physics_executed": False}
    collection["plan_sha256"] = digest(collection)
    output.mkdir(parents=True)
    commands = []
    for name, batch, prompts in planned:
        with session_directory(output / "batches" / name, batch):
            pass
        write_json(output / "prompts" / f"{name}.json", prompts)
        commands.append("./run.sh batch --resume " + shlex.quote(str(output / "batches" / name)))
    write_json(output / "collection-plan.json", collection)
    (output / "commands.txt").write_text("\n".join(commands) + "\n")
    return collection


def collection_batches(path):
    path = Path(path).resolve()
    collection = checked_plan(path)
    if collection.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("Unsupported collection schema")
    assignments = []
    for group in collection["groups"]:
        run = (path.parent / group["batch"]).resolve()
        if not run.is_relative_to(path.parent):
            raise ValueError("Batch path escapes its collection directory")
        batch = checked_plan(run / "plan.json")
        if batch["plan_sha256"] != group["plan_sha256"]:
            raise ValueError("A batch changed after its split was fixed")
        assignments.append({"run": str(run), "split": group["split"], "prompt_group": group["id"]})
    return assignments, sha256(path)


def index_batches(assignments, output, *, collection_sha256=None):
    """Reject leakage and changed evidence; report short/unsupported episodes explicitly."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh index directory")
    declarations, sessions, owners, identifiers = [], [], {}, set()
    for assignment in assignments:
        run = Path(assignment["run"]).resolve()
        split, group = assignment["split"], assignment["prompt_group"]
        if not isinstance(group, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", group):
            raise ValueError("Invalid prompt group name")
        plan = checked_plan(run / "plan.json")
        if plan.get("method") != "B0" or plan["config"]["mode"] != "batch":
            raise ValueError("Index only B0 batch sessions")
        if plan["lock_sha256"] != sha256(LOCK):
            raise ValueError("Session uses a different frozen baseline lock")
        for request in plan["requests"]:
            if request.get("task") != "grasp":
                raise ValueError("Training batches must use --grasp")
            own_prompt(owners, request, split)
            parent = {"parent_id": f"{group}-{request['seed']}-{request['index']}",
                      "split": split, "prompt_group": group, "scene_seed": request["seed"]}
            if parent["parent_id"] in identifiers:
                raise ValueError("Duplicate parent; use distinct group names for separate sessions")
            identifiers.add(parent["parent_id"])
            declarations.append(parent)
            sessions.append((run, plan, request, parent))
    # Audit ALL declarations, including pending attempts, before inspecting outcomes.
    audit_parents(declarations)
    parents, inventory = [], []
    for run, plan, request, parent in sessions:
        row = {**parent, "run": str(run), "index": request["index"]}
        result = latest_result(run, request["index"], verify=True)
        if result is None or result["status"] not in {"passed", "stopped", "failed"}:
            inventory.append({**row, "eligibility": "pending"})
            continue
        if result.get("request") != request or result.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("Attempt request or plan differs from its declared collection")
        attempt = run / result["attempt"]
        row.update(attempt=str(attempt), task_success=bool(result.get("task_success")),
                   failure_reason=result.get("failure_reason"))
        required = ("nominal_context.csv", "task.csv", "rollout/metadata.json", "rollout/states.npz",
                    "rollout/scene.mjb")
        missing = [name for name in required if not (attempt / name).is_file()]
        reason = None
        if result["status"] == "failed" or not result.get("physics_executed") or not result.get("sonic_executed"):
            reason = "invalid_execution"
        elif missing:
            reason = "missing_training_streams: " + ", ".join(missing)
        elif json.loads((attempt / "rollout/metadata.json").read_text()).get("status") != "complete":
            reason = "incomplete_rollout"
        else:
            with (attempt / "nominal_context.csv").open(newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            if count < 16:
                reason = "insufficient_history"
        if reason:
            inventory.append({**row, "eligibility": "unusable", "reason": reason})
            continue
        parents.append({**parent, "nominal": str(attempt),
                        "source_report_sha256": sha256(attempt / "report.json")})
        inventory.append({**row, "eligibility": "candidate", "teacher_candidate": bool(result.get("task_success"))})
    output.mkdir(parents=True)
    sources = {"parents": parents, "collection_sha256": collection_sha256,
               "baseline_lock_sha256": sha256(LOCK)}
    write_json(output / "sources.json", sources)
    report = {"status": "indexed", "physics_executed": False, "dataset_written": False,
              "planned_parents": len(declarations), "candidate_parents": len(parents),
              "teacher_candidates": sum(row.get("teacher_candidate", False) for row in inventory),
              "pending": sum(row["eligibility"] == "pending" for row in inventory),
              "unusable": sum(row["eligibility"] == "unusable" for row in inventory), "episodes": inventory}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Write fixed splits and runnable batch plans; no model load")
    plan.add_argument("--config", type=Path, default=ROOT / "configs/data/collection.json")
    plan.add_argument("--out", type=Path, required=True)
    index = commands.add_parser("index", help="Index completed batches; no simulation")
    group = index.add_mutually_exclusive_group(required=True)
    group.add_argument("--collection", type=Path)
    group.add_argument("--batches", type=Path, help="JSON with batches: run, split, prompt_group")
    index.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan_collection(args.config, args.out)
        result = {key: result[key] for key in ("status", "physics_executed", "plan_sha256")}
    else:
        assignments, fingerprint = (collection_batches(args.collection) if args.collection else
                                    (json.loads(args.batches.read_text())["batches"], sha256(args.batches)))
        result = index_batches(assignments, args.out, collection_sha256=fingerprint)
        result.pop("episodes")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
