"""A bounded physical pilot must pass execution checks before expansion."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.risk_quality_pipeline import (GROUPS, QualityPipeline, accept_parent_exit, checked_batch,
                                          check_pair_pilot, check_parent_batch, digest, host_path, sha)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_batch(repo, root, seed):
    batch = root / "batch-261006-151900"
    batch.mkdir(parents=True)
    requests = [{"index": i + 1, "seed": seed + i, "task": "grasp", "cube_xy": [.4, -.2],
                 "prompt": None, "duration": 2., "prompt_profile": "focused",
                 "reference": None, "reference_sha256": None,
                 "phase_prompts": {"reach": "Reach toward the block."}}
                for i in range(5)]
    plan = {"schema_version": 2, "method": "B0", "requests": requests,
            "config": {"mode": "batch", "grasp": True, "batch": 5, "seed": seed, "prompt": None,
                "duration": 2., "cube_xy": None, "start_back": .45, "walk": False, "direct_start": True,
                "table_standoff": .2, "finger_kp": 6., "finger_kd": .4, "hand_approach": "open",
                "hold_seconds": 10., "acquisition_z_min": None, "wrist_offset": None,
                "alignment_replans": 2, "alignment_replan_seconds": 1.6, "acquisition_transition": .3,
                "contact_profile": "elliptic", "lift_seconds": 4.8, "xy_range": [.36, .46, -.30, -.16],
                "phase_prompts": {"reach": "Reach toward the block."}, "prompt_profile": "focused",
                "gui": False, "device": "musa", "text_device": "musa", "text_dtype": "bfloat16",
                "history_frames": 16, "threads": 4, "ardy_repo": "/workspace/dl/third_party/ardy",
                "sonic_repo": "/workspace/dl/third_party/sonic", "assets": "/workspace/dl/checkpoints/baseline"},
            "lock_sha256": sha(repo / "configs/baseline.lock.json"),
            "source_sha256": {name: sha(repo / name) for name in ("baseline/common.py", "scripts/run.py")}}
    plan["plan_sha256"] = digest(plan)
    write(batch / "plan.json", plan)
    rows = []
    for i, request in enumerate(requests):
        attempt = batch / f"attempt-{i + 1:05d}"
        write(attempt / "request.json", request)
        files = ("nominal_context.csv", "task.csv", "rollout/metadata.json", "rollout/states.npz",
                 "rollout/scene.mjb", "ardy/lower/reference.npz")
        for name in files:
            path = attempt / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name + " immutable physical fixture evidence").encode())
        report = {"status": "passed" if i < 2 else "stopped", "stage": "hold",
                  "physics_executed": True, "sonic_executed": True, "task_success": i < 2,
                  "request": request, "plan_sha256": plan["plan_sha256"],
                  "failure_reason": None if i < 2 else "timeout",
                  "outputs_sha256": {name: sha(attempt / name) for name in (*files, "request.json")}}
        write(attempt / "report.json", report)
        rows.append({**report, "attempt": attempt.name})
    summary = {"status": "complete", "planned": 5, "completed": 5, "pending": 0,
               "successes": 2, "failures": 3, "success_rate": .4,
               "plan_sha256": plan["plan_sha256"], "attempts": rows}
    write(batch / "summary.json", summary)
    return batch, summary


class ResumeFixture:
    old_commit = "9c382dc085531dfb6d9dc661ed1b2621b33fec97"
    new_commit = "a" * 40
    old_blob = b"the original approved quality runner\n"
    new_blob = b"the reviewed parent-exit/resume repair\n"

    def __init__(self, root):
        self.repo = root / "repo"
        self.output = self.repo / "output/risk-quality-261006"
        self.old = self.repo / "output/teacher-risk-261006"
        self.output.mkdir(parents=True)
        for name in ("baseline/common.py", "scripts/run.py", "configs/baseline.lock.json",
                     "configs/learning/risk-quality-pilot.json"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixed unchanged source or configuration\n")
        runner = self.repo / "scripts/risk_quality_pipeline.py"
        runner.write_bytes(self.new_blob)
        for name, _, _ in GROUPS:
            write(self.repo / f"output/data-collection-261005/prompts/{name}.json",
                  {"reach": "Reach toward the block."})
        write(self.old / "dataset/manifest.json", {"windows": {"val": "val.npz"}})
        (self.old / "dataset/val.npz").write_bytes(b"unchanged validation windows")
        write(self.old / "pairs/sources.json", {"parents": []})
        inputs = {str(self.repo / name): sha(self.repo / name) for name in
                  ("baseline/common.py", "scripts/run.py", "configs/baseline.lock.json",
                   "configs/learning/risk-quality-pilot.json")}
        inputs[str(runner)] = __import__("hashlib").sha256(self.old_blob).hexdigest()
        for path in (self.old / "dataset/manifest.json", self.old / "dataset/val.npz", self.old / "pairs/sources.json"):
            inputs[str(path)] = sha(path)
        evaluations = []
        for seed in (0, 1):
            checkpoint = self.old / f"risk/seed-{seed}/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"unchanged seed {seed} checkpoint".encode())
            inputs[str(checkpoint)] = sha(checkpoint)
            destination = self.output / f"old-risk/seed-{seed}"
            destination.mkdir(parents=True)
            (destination / "predictions.npz").write_bytes(f"fixed seed {seed} predictions".encode())
            gate = {"split": "val", "risk_sha256": sha(checkpoint),
                    "dataset_manifest_sha256": sha(self.old / "dataset/manifest.json"),
                    "threshold": .18 if seed == 0 else .31,
                    "probability_transform": "raw_sigmoid", "temperature": 1.}
            report = {"schema": "dl-risk-evaluation-v1", "status": "passed", "split": "val", "seed": seed,
                      "checkpoint_epoch": 3, "risk_sha256": sha(checkpoint),
                      "dataset_manifest_sha256": sha(self.old / "dataset/manifest.json"),
                      "windows_sha256": sha(self.old / "dataset/val.npz"), "gate": gate,
                      "predictions_sha256": sha(destination / "predictions.npz")}
            write(destination / "gate.json", gate)
            write(destination / "report.json", report)
            evaluations.append({"seed": seed, "report": str(destination / "report.json"),
                                "checkpoint_epoch": 3, "gate": gate})
        self.batch, _ = make_batch(self.repo, self.output / "parents/train-a", 12000)
        self.state = {"schema": "dl-risk-quality-pipeline-v1", "run_id": self.output.name,
                      "status": "failed", "stage": "parent-train-a", "pid": 1010421, "child_pid": 1014707,
                      "output": str(self.output), "git_commit": self.old_commit, "source_sha256": inputs,
                      "started_at": "2026-10-06T07:17:39+00:00", "finished_at": "2026-10-06T07:23:30+00:00",
                      "error": "parent-train-a exited with status 1", "events": [
                          {"id": "DL-PIPELINE-risk-quality-261006-1", "stage": "old-validation-complete",
                           "at": "2026-10-06T07:18:45+00:00", "details": {"evaluations": evaluations}}]}
        write(self.output / "state.json", self.state)
        (self.output / "parent-train-a.log").write_text("five physical trials completed; 2 successes, 3 task failures\n")

    def git(self, command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return self.new_commit + "\n"
        if command == ["git", "show", self.old_commit + ":scripts/risk_quality_pipeline.py"]:
            return self.old_blob
        if command == ["git", "show", self.new_commit + ":scripts/risk_quality_pipeline.py"]:
            return self.new_blob
        raise AssertionError("Unexpected subprocess in stdlib fixture: " + repr(command))

    def resume(self):
        with patch("scripts.risk_quality_pipeline.process_exists", return_value=False), \
                patch("scripts.risk_quality_pipeline.subprocess.check_output", side_effect=self.git):
            return QualityPipeline.resume(self.repo, self.output)


class QualityGateTests(unittest.TestCase):
    def test_task_failures_count_but_execution_errors_stop(self):
        summary = {"status": "complete", "planned": 5, "completed": 5, "pending": 0,
                   "attempts": [{"status": "stopped", "physics_executed": True,
                                 "sonic_executed": True, "task_success": False} for _ in range(5)]}
        check_parent_batch(summary)
        accept_parent_exit(1, summary)
        accept_parent_exit(0, summary)
        with self.assertRaises(RuntimeError):
            accept_parent_exit(2, summary)
        for changed in ({"pending": 1}, {"completed": 4}, {"status": "blocked"},
                        {"attempts": [{"status": "failed"}]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_parent_batch({**summary, **changed})

    def test_physics_sonic_and_actual_five_rows_are_required_even_for_exit_zero(self):
        base = {"status": "complete", "planned": 5, "completed": 5, "pending": 0,
                "attempts": [{"status": "stopped", "physics_executed": True,
                              "sonic_executed": True, "task_success": False} for _ in range(5)]}
        for key in ("physics_executed", "sonic_executed"):
            broken = json.loads(json.dumps(base))
            broken["attempts"][0][key] = False
            with self.assertRaises(ValueError):
                accept_parent_exit(0, broken)
        with self.assertRaises(ValueError):
            check_parent_batch({**base, "attempts": base["attempts"][:4]})

    def test_three_valid_pairs_allow_review_not_unvalidated_expansion(self):
        report = {"planned_pairs": 12, "pending_pairs": 9,
                  "jobs": {str(i): {"status": "paired", "recovery_verified": False} for i in range(3)}}
        check_pair_pilot(report)
        for changed in ({"jobs": {}}, {"planned_pairs": 0},
                        {"jobs": {str(i): {"status": "failed"} for i in range(3)}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_pair_pilot({**report, **changed})

    def test_resumed_orchestration_reuses_validation_and_complete_train_a_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            original_state = (fixture.output / "state.json").read_bytes()
            original_files = {str(path.relative_to(fixture.output)): sha(path) for path in fixture.output.rglob("*")
                              if path.is_file() and path.name != "state.json"}
            pipeline = fixture.resume()
            calls = []
            def execute(stage, command):
                calls.append(stage)
                pipeline.check_sources()
                if stage in {"parent-train-b", "parent-val-a"}:
                    name = stage.removeprefix("parent-")
                    seed = next(seed for group, _, seed in GROUPS if group == name)
                    make_batch(fixture.repo, fixture.output / "parents" / name, seed)
                    return 1  # Correct task failures must not abort valid physics.
                if stage == "index":
                    write(fixture.output / "index/report.json", {"planned_parents": 15, "candidate_parents": 15,
                          "teacher_candidates": 6, "pending": 0, "unusable": 0})
                    write(fixture.output / "index/sources.json", {"parents": []})
                    return 0
                if stage == "teacher-pilot":
                    write(fixture.output / "pairs/report.json", {"planned_pairs": 18, "pending_pairs": 15,
                          "recovery_verified": 1, "failed_pairs": 0,
                          "jobs": {str(i): {"status": "paired"} for i in range(3)}})
                    return 0
                raise AssertionError("A completed stage was rerun: " + stage)
            with patch.object(pipeline, "gpu", return_value=2), patch.object(pipeline, "execute", side_effect=execute):
                pipeline.run()
            self.assertEqual(calls, ["parent-train-b", "parent-val-a", "index", "teacher-pilot"])
            for name, expected in original_files.items():
                self.assertEqual(sha(fixture.output / name), expected, name)
            backup = Path(pipeline.state["resume_backup"])
            self.assertEqual((backup / "previous-state.json").read_bytes(), original_state)
            inventory = json.loads((backup / "artifacts.json").read_text())
            self.assertEqual(inventory["runner_upgrade"]["old_git_commit"], fixture.old_commit)
            self.assertEqual(inventory["runner_upgrade"]["new_git_commit"], fixture.new_commit)
            self.assertEqual(pipeline.state["events"][0], fixture.state["events"][0])
            self.assertEqual(sum(event["stage"] == "old-validation-complete" for event in pipeline.state["events"]), 1)
            self.assertEqual(pipeline.state["stage"], "pilot-review")
            self.assertTrue(next(event for event in pipeline.state["events"]
                                 if event["stage"] == "parent-train-a-complete")["details"]["execution_reused"])

    def test_resume_rejects_active_recorded_parent_or_child_without_mutating_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            before = (fixture.output / "state.json").read_bytes()
            for active in (1010421, 1014707):
                with patch("scripts.risk_quality_pipeline.process_exists", side_effect=lambda pid: pid == active), \
                        self.assertRaisesRegex(ValueError, "active"):
                    QualityPipeline.resume(fixture.repo, fixture.output)
            self.assertEqual((fixture.output / "state.json").read_bytes(), before)
            self.assertFalse((fixture.output / "resume-history").exists())

    def test_resume_only_allows_git_verified_runner_revision_and_keeps_all_other_source_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            (fixture.repo / "baseline/common.py").write_text("changed baseline must be rejected")
            with self.assertRaisesRegex(ValueError, "cannot relax"):
                fixture.resume()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            fixture.old_blob = b"wrong historical Git source"
            with self.assertRaisesRegex(ValueError, "Git blob"):
                fixture.resume()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            fixture.new_blob = b"current working copy is not committed"
            with self.assertRaisesRegex(ValueError, "Commit"):
                fixture.resume()

    def test_complete_summary_cannot_hide_mutated_report_artifact_or_baseline_source(self):
        for target in ("report", "artifact", "source"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                fixture = ResumeFixture(Path(temporary))
                if target == "report":
                    path = fixture.batch / "attempt-00001/report.json"
                    report = json.loads(path.read_text())
                    report["task_success"] = False
                    write(path, report)
                elif target == "artifact":
                    (fixture.batch / "attempt-00001/rollout/states.npz").write_bytes(b"changed")
                else:
                    (fixture.repo / "scripts/run.py").write_text("changed frozen baseline")
                with self.assertRaises(ValueError):
                    checked_batch(fixture.repo, fixture.output / "parents/train-a", 12000)

    def test_host_container_config_paths_match_but_changed_pilot_settings_do_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            # Current container paths are accepted by normalizing into the
            # canonical host repository, rather than changing the frozen plan.
            checked_batch(fixture.repo, fixture.output / "parents/train-a", 12000)
            path = fixture.batch / "plan.json"
            plan = json.loads(path.read_text())
            plan["config"]["hold_seconds"] = 5.
            plan["plan_sha256"] = digest({key: value for key, value in plan.items() if key != "plan_sha256"})
            write(path, plan)
            summary_path = fixture.batch / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["plan_sha256"] = plan["plan_sha256"]
            write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "configuration"):
                checked_batch(fixture.repo, fixture.output / "parents/train-a", 12000)

    def test_resume_refuses_incomplete_or_mutated_existing_validation_and_later_stages(self):
        for target in ("prediction", "incomplete-validation", "partial-parent", "index"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                fixture = ResumeFixture(Path(temporary))
                before = (fixture.output / "state.json").read_bytes()
                if target == "prediction":
                    (fixture.output / "old-risk/seed-0/predictions.npz").write_bytes(b"changed predictions")
                elif target == "incomplete-validation":
                    (fixture.output / "old-risk/seed-0/report.json").unlink()
                elif target == "partial-parent":
                    summary_path = fixture.batch / "summary.json"
                    summary = json.loads(summary_path.read_text())
                    summary.update(status="interrupted", pending=1, completed=4)
                    write(summary_path, summary)
                else:
                    (fixture.output / "index").mkdir()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    fixture.resume()
                self.assertEqual((fixture.output / "state.json").read_bytes(), before)
                self.assertFalse((fixture.output / "resume-history").exists())


if __name__ == "__main__":
    unittest.main()
