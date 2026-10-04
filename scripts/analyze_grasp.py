"""Audit recorded grasp trials without rerunning physics or changing measurements."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from baseline.common import sha256, write_json


def audit_nominal_context(trial):
    """Validate the causal 50 Hz P input stream without reading task labels."""
    path = trial / "nominal_context.csv"
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        offsets = sorted(name for name in fields if name.startswith("nominal_time_offset:"))
        required = {"sim_time", "frame_index", "phase", "root_x", "state_q:" + "left_hip_pitch_joint"}
        missing = sorted(required - set(fields))
        if missing or len(offsets) < 2:
            return {"status": "invalid", "path": str(path), "missing": missing,
                    "nominal_offset_columns": len(offsets)}
        position_columns = [name for name in fields if name.startswith("nominal_pos:")]
        velocity_columns = [name for name in fields if name.startswith("nominal_vel:")]
        quaternion_columns = [name for name in fields if name.startswith("nominal_quat:")]
        expected = len(offsets)
        if (len(position_columns) != expected * 29 or len(velocity_columns) != expected * 29
                or len(quaternion_columns) != expected * 4):
            return {"status": "invalid", "path": str(path), "reason": "lookahead_width_mismatch",
                    "rows": 0, "nominal_offsets": len(offsets)}
        rows = 0
        previous_time = None
        previous_frame = None
        first_offsets = None
        first_sim_time = None
        last_time = None
        for row in reader:
            rows += 1
            sim_time = float(row["sim_time"])
            frame = int(row["frame_index"])
            if not math.isfinite(sim_time) or (previous_time is not None and sim_time <= previous_time):
                return {"status": "invalid", "path": str(path), "reason": "nonmonotonic_sim_time",
                        "rows": rows}
            if previous_frame is not None and frame != previous_frame + 1:
                return {"status": "invalid", "path": str(path), "reason": "noncontiguous_frame_index",
                        "rows": rows}
            offsets_now = tuple(float(row[name]) for name in offsets)
            if not all(math.isfinite(value) for value in offsets_now):
                return {"status": "invalid", "path": str(path), "reason": "nonfinite_offsets",
                        "rows": rows}
            if first_offsets is None:
                first_offsets = offsets_now
                first_sim_time = sim_time
            if any(b <= a for a, b in zip(offsets_now, offsets_now[1:])):
                return {"status": "invalid", "path": str(path), "reason": "nonmonotonic_offsets",
                        "rows": rows}
            quat = np.array([float(row[name]) for name in quaternion_columns], dtype=float)
            if len(quat) % 4 or len(quat) == 0 or not np.isfinite(quat).all():
                return {"status": "invalid", "path": str(path), "reason": "invalid_nominal_quaternions",
                        "rows": rows}
            if not np.allclose(np.linalg.norm(quat.reshape(-1, 4), axis=1), 1., atol=2e-3):
                return {"status": "invalid", "path": str(path), "reason": "nonunit_nominal_quaternions",
                        "rows": rows}
            previous_time, previous_frame, last_time = sim_time, frame, sim_time
        if rows == 0:
            return {"status": "invalid", "path": str(path), "reason": "empty"}
        return {"status": "passed", "path": str(path), "rows": rows,
                "first_sim_time": first_sim_time,
                "last_sim_time": last_time, "nominal_offsets": list(first_offsets),
                "nominal_position_columns": len(position_columns),
                "nominal_velocity_columns": len(velocity_columns),
                "nominal_quaternion_columns": len(quaternion_columns)}


def analyze_trial(trial):
    report = json.loads((trial / "report.json").read_text())
    with (trial / "task.csv").open() as handle:
        records = list(csv.DictReader(handle))
    rows = {key: np.array([row[key] for row in records], dtype=float)
            for key in ("sim_time", "thumb_force_n", "finger_force_n", "lowest_clearance_m",
                        "held_seconds", "block_x", "block_y", "block_z", "wrist_x", "wrist_y", "wrist_z")}
    phase = np.array([row["phase"] for row in records])
    contact = (rows["thumb_force_n"] > .01) & (rows["finger_force_n"] > .01)
    held = contact & (rows["lowest_clearance_m"] >= .05)
    success_time = report.get("success_sim_time")
    after = rows["sim_time"] >= success_time if success_time is not None else np.zeros(len(phase), bool)
    # New reports keep late loss sticky.  Fall back to the final held sample
    # for historical reports that predate the explicit field.
    late_loss = bool(report.get("lost_after_success", False)
                     or report.get("post_success_loss_samples", 0) > 0)
    dropped = bool(after.any() and (late_loss or not held[-1]))
    result = {"attempt": trial.name, "source": str(trial),
              "seed": report["request"]["seed"], "cube_xy": report["request"]["cube_xy"],
              "recorded_task_success": report["task_success"],
              "threshold_reached": bool(report.get("success_threshold_reached", success_time is not None)),
              "recorded_failure_reason": report.get("failure_reason"),
              "dropped_after_success_threshold": dropped,
              "late_loss_after_success": late_loss,
              "retained_at_end": bool(report.get("retained_at_end", held[-1])) and not late_loss,
              "final_clearance_m": float(rows["lowest_clearance_m"][-1]),
              "success_sim_time": success_time, "phases": {},
              "alignment": report.get("grasp_alignment"),
              "nominal_context": audit_nominal_context(trial),
              "report_sha256": sha256(trial / "report.json"),
              "task_sha256": sha256(trial / "task.csv")}
    if after.any():
        losses = np.flatnonzero(after & ~held)
        result["first_post_success_loss_sim_time"] = float(rows["sim_time"][losses[0]]) if len(losses) else None
    for name in np.unique(phase):
        mask = phase == name
        t = rows["sim_time"][mask]
        wrist = np.stack([rows[f"wrist_{axis}"][mask] for axis in "xyz"], axis=1)
        block = np.stack([rows[f"block_{axis}"][mask] for axis in "xyz"], axis=1)
        speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1) / np.diff(t)
        # Displacement over the phase distinguishes body drift from grip-relative slip.
        relative = block - wrist
        result["phases"][name] = {
            "first_sim_time": float(t[0]), "last_sim_time": float(t[-1]),
            "opposing_contact_fraction": float(contact[mask].mean()),
            "thumb_force_mean_n": float(rows["thumb_force_n"][mask].mean()),
            "finger_force_mean_n": float(rows["finger_force_n"][mask].mean()),
            "final_clearance_m": float(rows["lowest_clearance_m"][mask][-1]),
            "wrist_speed_p95_mps": float(np.percentile(speed, 95)) if len(speed) else None,
            "wrist_displacement_m": float(np.linalg.norm(wrist[-1] - wrist[0])),
            "block_relative_displacement_m": float(np.linalg.norm(relative[-1] - relative[0])),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    # Accept both normal session attempts and calibration labels such as
    # ``open-seed-012`` without changing the recorded files.
    trials = sorted(path for path in args.run.iterdir()
                    if path.is_dir() and (path / "task.csv").is_file()
                    and (path / "report.json").is_file())
    results = [analyze_trial(trial) for trial in trials]
    counts = Counter(row["recorded_failure_reason"] or "threshold_success" for row in results)
    summary = {"scope": "recorded physical trials; read-only audit", "command": sys.argv,
               "analysis_sha256": sha256(__file__), "numpy_version": np.__version__,
               "attempts": len(results), "recorded_threshold_successes": sum(row["threshold_reached"] for row in results),
               "successes_with_later_drop": sum(row["dropped_after_success_threshold"] for row in results),
               "retained_successes": sum(row["recorded_task_success"] and row["retained_at_end"] for row in results),
               "reasons": dict(counts), "trials": results}
    write_json(args.out / "audit.json", summary)
    lines = ["# Recorded grasp audit", "", f"Source: {args.run}", "",
             "| Seed | Recorded result | Final clearance (m) | Retained at end | Post-success drop |",
             "| --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| {row['seed']} | {row['recorded_failure_reason'] or 'threshold success'} | "
                     f"{row['final_clearance_m']:.4f} | {row['retained_at_end']} | {row['dropped_after_success_threshold']} |")
    (args.out / "audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
