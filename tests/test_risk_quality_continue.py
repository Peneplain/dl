"""Bounded continuation gates and immutable source/cohort provenance."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.risk_quality_continue import (InsufficientEvidence, THRESHOLDS, calibration_gate,
    claim_first, diagnostic_gate, host_path, merge_sources, pilot_gate, select_seed, sha, verify_hashes)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ContinuationTests(unittest.TestCase):
    def test_host_container_mapping_and_changed_source_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "baseline/fixture.py"
            source.parent.mkdir()
            source.write_text("frozen fixture")
            self.assertEqual(host_path(repo, "/workspace/dl/baseline/fixture.py"), source)
            hashes = {"/workspace/dl/baseline/fixture.py": sha(source)}
            verify_hashes(repo, hashes)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_hashes(repo, hashes)

    def pilot(self, repo, *, recovery=False):
        first = repo / "output/risk-quality-261006"
        marker = repo / "baseline/fixture.py"
        marker.parent.mkdir()
        marker.write_text("immutable source")
        pinned = {str(marker): sha(marker)}
        write(first / "state.json", {"schema": "dl-risk-quality-pipeline-v1", "run_id": first.name,
              "status": "pilot-review", "stage": "pilot-review", "source_sha256": pinned})
        parent = {"parent_id": "quality-12000", "prompt_group": "quality-train-a",
                  "scene_seed": 12000, "split": "train"}
        write(first / "index/sources.json", {"parents": [parent]})
        source = first / "source"
        write(source / "report.json", {"status": "passed", "task_success": True})
        jobs, results = [], {}
        for index in range(3):
            variant = {"phase": "lower", "joint": "right_shoulder_pitch_joint", "delay": .2,
                       "amplitude": .1, "id": f"fixture-{index}"}
            job = {"id": str(index), "parent": parent, "source": str(source), "perturbation": variant,
                   "source_files_sha256": {str(source / "report.json"): sha(source / "report.json")}}
            pair_path = first / f"pairs/proofs/{index}/pair.json"
            branches = {}
            for name in ("teacher", "nominal"):
                branch = pair_path.parent / name
                success = name == "teacher" or not recovery
                write(branch / "report.json", {"status": "passed", "task_success": success,
                      "physics_executed": True, "sonic_executed": True})
                (branch / "scene.xml").write_text("same physical scene")
                (branch / "rollout").mkdir()
                (branch / "rollout/scene.mjb").write_text("same compiled model")
                branches[name] = {"report": str(branch / "report.json"), "report_sha256": sha(branch / "report.json"),
                     "scene_sha256": sha(branch / "scene.xml"), "model_sha256": sha(branch / "rollout/scene.mjb"),
                     "status": "passed", "task_success": success,
                     "activation": {"snapshot_sha256": "same pre-intervention controller/physics snapshot",
                                    "frame_index": 15, "time": .3}}
            write(pair_path, {"source_attempt": str(source), "source_report_sha256": sha(source / "report.json"),
                              "pair_state_verified": True, "recovery_verified": recovery,
                              "branches": branches, **{key: variant[key] for key in ("phase", "joint", "delay", "amplitude")}})
            jobs.append(job)
            results[str(index)] = {"status": "paired", "pair": str(pair_path), "recovery_verified": recovery}
        plan = {"jobs": jobs, "baseline_sources_sha256": pinned,
                "sources_sha256": sha(first / "index/sources.json")}
        plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write(first / "pairs/plan.json", plan)
        write(first / "pairs/report.json", {"jobs": results, "plan_sha256": plan["plan_sha256"], "failed_pairs": 0})
        return first

    def test_zero_initial_recoveries_can_expand_valid_clean_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            first = self.pilot(repo)
            _, _, counts = pilot_gate(repo, first)
            self.assertEqual(counts["verified_recoveries"], 0)
            self.assertEqual(counts["verified_clean_successes"], 3)
            state = json.loads((first / "state.json").read_text())
            state["status"] = "running"
            write(first / "state.json", state)
            with self.assertRaisesRegex(ValueError, "terminal"):
                pilot_gate(repo, first)

    def test_pair_branch_evidence_change_stops_before_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            first = self.pilot(repo, recovery=True)
            self.assertEqual(pilot_gate(repo, first)[2]["verified_recoveries"], 3)
            (first / "pairs/proofs/0/teacher/report.json").write_text("changed outcome")
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                pilot_gate(repo, first)

    def test_one_first_run_can_be_consumed_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory)
            claim_first(first, first / "next")
            with self.assertRaises(FileExistsError):
                claim_first(first, first / "another-next")
            self.assertEqual(json.loads((first / "continuation-claim.json").read_text())["output"],
                             str((first / "next").resolve()))

    def test_merging_preserves_splits_and_marks_control_cohorts(self):
        old = {"baseline_lock_sha256": "frozen", "parents": [
            {"parent_id": "old", "split": "test", "scene_seed": 30000, "prompt_group": "test-a"}]}
        new = {"baseline_lock_sha256": "frozen", "parents": [
            {"parent_id": "new", "split": "val", "scene_seed": 22000, "prompt_group": "quality-val-a"}]}
        merged = merge_sources(old, new)
        self.assertEqual([row["split"] for row in merged["parents"]], ["test", "val"])
        self.assertEqual([row["control_cohort"] for row in merged["parents"]],
                         ["old-dense-velocity", "current-sparse-velocity"])
        new["parents"][0]["scene_seed"] = 30000
        with self.assertRaisesRegex(ValueError, "leakage"):
            merge_sources(old, new)
        new["parents"][0].update(scene_seed=22000, split="test")
        with self.assertRaisesRegex(ValueError, "add test"):
            merge_sources(old, new)

    def test_seed_choice_is_predeclared_finite_loss_rule(self):
        summary = {"status": "passed", "jobs": [{"seed": seed, "status": "passed", "best_val_loss": .1}
                                                for seed in (1, 0)]}
        self.assertEqual(select_seed(summary), 0)
        summary["jobs"][0]["best_val_loss"] = float("nan")
        with self.assertRaises(ValueError):
            select_seed(summary)

    def test_frozen_train_only_calibration_checks_all_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            helper, diagnosis, manifest, artifact = [repo / name for name in
                ("experiments/calibrate_labels.py", "diagnosis.json", "manifest.json", "calibration.json")]
            helper.parent.mkdir()
            helper.write_text("calibration revision fixture")
            write(diagnosis, {"schema": "dl-label-diagnosis-v1"})
            write(manifest, {"provenance": {"source_plan_sha256": "source-plan"}})
            value = {"schema": "dl-label-calibration-v1", "split": "train", "calibration_split": "train",
                     "exploratory": True, "frozen": True, "test_used": False, "labels_written": False,
                     "automatic_application": False,
                     "candidate_tracking_thresholds_rad": THRESHOLDS, "input_manifest_sha256": sha(manifest),
                     "input_source_plan_sha256": "source-plan", "input_diagnosis_sha256": sha(diagnosis),
                     "calibrator_revision_sha256": sha(helper)}
            write(artifact, value)
            calibration_gate(repo, artifact, diagnosis, manifest)
            write(artifact, {**value, "calibration_split": "val"})
            with self.assertRaisesRegex(ValueError, "train-only"):
                calibration_gate(repo, artifact, diagnosis, manifest)

    def test_validation_guard_stops_without_retuning_or_old_correction_loss(self):
        report = {"schema": "dl-label-diagnosis-v1", "tracking_thresholds_rad": THRESHOLDS,
                  "splits": {"train": {}, "val": {"physical_stable_hold": {
                      "windows": 100, "reason_windows": {"tracking": 10}}}},
                  "control_cohorts": {"train": {"old-dense-velocity": {"verified_corrections": 28}},
                                      "val": {"old-dense-velocity": {"verified_corrections": 7}}}}
        self.assertEqual(diagnostic_gate(report)["tracking_positive_fraction"], .1)
        report["splits"]["val"]["physical_stable_hold"]["reason_windows"]["tracking"] = 11
        with self.assertRaises(InsufficientEvidence):
            diagnostic_gate(report)
        report["splits"]["val"]["physical_stable_hold"]["reason_windows"]["tracking"] = 0
        report["control_cohorts"]["val"]["old-dense-velocity"]["verified_corrections"] = 6
        with self.assertRaisesRegex(InsufficientEvidence, "lost"):
            diagnostic_gate(report)


if __name__ == "__main__":
    unittest.main()
