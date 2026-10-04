"""Session plans, flat attempt directories and readable evaluation reports."""

from contextlib import contextmanager
from datetime import datetime
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from baseline.common import LOCK, ROOT, sha256, write_json

TERMINAL = {"passed", "failed", "stopped"}
REASONS = {
    "timeout": "Timeout: lift-and-hold criteria not met",
    "prohibited_robot_table_contact": "Prohibited robot-table contact",
    "prohibited_robot_block_contact": "Non-right-hand robot-block contact",
    "prohibited_robot_floor_contact": "Non-foot robot-floor contact",
    "block_floor_contact": "Block-floor contact",
    "fall": "Robot fall",
    "joint_velocity_limit": "Joint velocity limit exceeded",
    "interrupted": "Interrupted by user",
    "approach_target_missed": "Walking did not reach the grounded approach target",
    "approach_not_settled": "Robot did not stop and settle before reaching",
    "prepare_not_settled": "Robot did not remain stable after upper-body preparation",
    "prepare_target_drift": "Robot drifted outside the approach target during preparation",
    "grasp_alignment_missed": "Hand did not reach the block acquisition region",
    "grasp_not_acquired": "Closing did not establish stable opposing finger contacts",
    "grasp_lost_after_success": "Block was lost after reaching the hold threshold",
    "approach_too_close": "Walking crossed the table standoff limit",
    "direct_start_not_settled": "Direct-start pose did not remain stable before manipulation",
}


def timestamp(*, microseconds=False):
    pattern = "%y%m%d-%H%M%S" + ("-%f" if microseconds else "")
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime(pattern)


def source_hashes():
    paths = [ROOT / "scripts/run.py", *sorted((ROOT / "baseline").rglob("*.py"))]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_plan(config, requests):
    plan = {"schema_version": 2, "method": "B0", "timezone": "Asia/Shanghai",
            "config": config, "requests": requests, "source_sha256": source_hashes(),
            "lock_sha256": sha256(LOCK)}
    plan["plan_sha256"] = digest(plan)
    return plan


@contextmanager
def session_directory(output, plan, resume=False, mode=None):
    output = Path(output)
    if mode is not None:
        if resume or mode not in {"batch", "manual"}:
            raise ValueError("Automatic session naming needs batch/manual mode without resume")
        while True:
            candidate = output / f"{mode}-{timestamp()}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                output = candidate
                break
            except FileExistsError:
                # Concurrent launches retain second precision without sharing
                # a directory or overwriting a previous session.
                time.sleep(.05)
    elif resume:
        if not output.is_dir():
            raise FileNotFoundError("Resume directory does not exist")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this session") from error
        if resume:
            saved = json.loads((output / "plan.json").read_text())
            if saved != plan:
                raise ValueError("Resume source/configuration/model lock differs from the saved plan")
        else:
            write_json(output / "plan.json", plan)
        yield output


def attempt_dirs(output, index):
    name = f"attempt-{index:05d}"
    return sorted(p for p in Path(output).glob(name + "*")
                  if p.is_dir() and (p.name == name or p.name.startswith(name + "-retry-")))


def latest_result(output, index, verify=False):
    paths = attempt_dirs(output, index)
    if not paths or not (paths[-1] / "report.json").is_file():
        return None
    row = json.loads((paths[-1] / "report.json").read_text())
    if verify and row.get("status") in TERMINAL:
        for name, expected in row.get("outputs_sha256", {}).items():
            path = paths[-1] / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Attempt evidence changed: {path}")
    return {**row, "attempt": paths[-1].name}


def next_attempt(output, request):
    paths = attempt_dirs(output, request["index"])
    suffix = f"-retry-{len(paths) + 1:02d}" if paths else ""
    path = Path(output) / f"attempt-{request['index']:05d}{suffix}"
    path.mkdir(exist_ok=False)
    write_json(path / "request.json", request)
    return path


def wilson(successes, count):
    if count == 0:
        return None
    z, p = 1.959963984540054, successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def result_label(row):
    if row.get("status") not in TERMINAL:
        return "PENDING"
    if row.get("task_success") is True:
        return "SUCCESS"
    if row.get("task_success") is False or row.get("status") != "passed":
        return "FAILED"
    return "COMPLETED"  # No task evaluator: execution completion is not task success.


def result_line(row):
    label = result_label(row)
    request = row.get("request", {})
    reason = readable_reason(row) or "—"
    sim = row.get("simulation_seconds", 0)
    return (f"[{label:9}] {row['attempt']}  seed={request.get('seed', '?')}  "
            f"sim={sim:.2f}s  phase={row.get('stage', 'pending')}  {reason}")


def readable_reason(row):
    code = row.get("failure_reason") or ""
    reason = REASONS.get(code, code)
    if row.get("runtime_error"):
        reason += ": " + row["runtime_error"]
    contact = row.get("first_prohibited_contact") or {}
    if contact.get("bodies"):
        reason += " (" + " / ".join(contact["bodies"]) + ")"
    return reason


def summarize(output, plan, status, requests=None):
    output = Path(output)
    requests = plan["requests"] if requests is None else requests
    rows = []
    for request in requests:
        row = latest_result(output, request["index"])
        rows.append(row or {"attempt": f"attempt-{request['index']:05d}", "request": request,
                            "status": "pending", "task_success": None})
    completed = [r for r in rows if r["status"] in TERMINAL]
    evaluated = [r for r in completed if r.get("task_success") is not None]
    successes = sum(r.get("task_success") is True for r in evaluated)
    failures = [r for r in rows if result_label(r) == "FAILED"]
    summary = {"schema_version": 2, "status": status, "method": "B0",
               "planned": len(requests), "completed": len(completed),
               "pending": len(requests) - len(completed), "evaluated": len(evaluated),
               "successes": successes, "failures": len(failures),
               "execution_only": sum(result_label(r) == "COMPLETED" for r in rows),
               "success_rate": successes / len(evaluated) if evaluated else None,
               "success_wilson_95": wilson(successes, len(evaluated)),
               "plan_sha256": plan["plan_sha256"], "expert_dataset": False,
               "failure_reasons": {}, "attempts": rows}
    for row in failures:
        reason = row.get("failure_reason", "runtime_error")
        summary["failure_reasons"][reason] = summary["failure_reasons"].get(reason, 0) + 1
    write_json(output / "summary.json", summary)
    total = (f"{output.name}: {status}\n"
             f"Completed {len(completed)}/{len(requests)} | Success {successes} | Failure {len(failures)} | "
             f"Execution only {summary['execution_only']} | Pending {summary['pending']}")
    if evaluated:
        lo, hi = summary["success_wilson_95"]
        total += f"\nGrasp success rate {successes}/{len(evaluated)} ({successes / len(evaluated):.1%}), Wilson 95% [{lo:.1%}, {hi:.1%}]"
    text = total + "\n\n" + "\n".join(result_line(r) for r in rows) + "\n"
    text += "\nSUCCESS=task succeeded; FAILED=failed; COMPLETED=execution finished without task evaluation; PENDING=unfinished.\n"
    (output / "summary.txt").write_text(text)
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    markdown = ["# " + output.name, "", total.split("\n", 1)[1], "",
                "| Attempt | Result | Seed | Phase | Sim s | Reason | Prompt |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["attempt", "result", "seed", "phase", "simulation_seconds", "reason", "prompt"])
    for row in rows:
        request = row.get("request", {})
        columns = [row["attempt"], result_label(row), request.get("seed"), row.get("stage", "pending"),
                   f"{row.get('simulation_seconds', 0):.2f}", row.get("failure_reason") or "",
                   request.get("prompt") or ("(task phase prompts)" if request.get("task") else "(standing / saved reference)")]
        writer.writerow(columns)
        columns[0] = f"[{row['attempt']}]({row['attempt']}/report.json)"
        columns[5] = readable_reason(row)
        markdown.append("| " + " | ".join(escape(c) for c in columns) + " |")
    (output / "summary.md").write_text("\n".join(markdown) + "\n")
    (output / "results.csv").write_text(csv_buffer.getvalue())
    for label, filename in (("SUCCESS", "successes.txt"), ("FAILED", "failures.txt")):
        (output / filename).write_text("".join(r["attempt"] + "\n" for r in rows if result_label(r) == label))
    return summary, text
