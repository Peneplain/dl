"""Split isolation, causal eligibility, resumable pair provenance and training readiness."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baseline.common import LOCK, ROOT, sha256, write_json
from baseline.grasp import FOCUSED_PROMPTS
from baseline.session import make_plan
from experiments.build_dataset import InsufficientHistory, build, inspect, read_context
from experiments.collect_pair import source_args
from experiments.collect_pairs import collect_batch
from experiments.plan_data import collection_batches, index_batches, plan_collection
from experiments.smoke import make_fixture
from risk_residual.audit import readiness, require_ready, supervision_summary
from risk_residual.config import ModelConfig
from risk_residual.data import WindowDataset
from risk_residual.provenance import own_prompt, prompt_identity
from scripts.run import config_for, parse_args, parse_request


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def batch(self, name, *, seed=1, count=1):
        run = self.root / name
        run.mkdir()
        args = parse_args(["batch", "--grasp", "--seed", str(seed), "--batch", str(count)])
        requests = [parse_request({}, index + 1, args) for index in range(count)]
        plan = make_plan(config_for(args), requests)
        write_json(run / "plan.json", plan)
        return run, plan

    def episode(self, run, plan, index, *, success=False, frames=16):
        request = plan["requests"][index - 1]
        attempt = run / f"attempt-{index:05d}"
        (attempt / "rollout").mkdir(parents=True)
        write_json(attempt / "request.json", request)
        write_json(attempt / "rollout/metadata.json", {"status": "complete"})
        (attempt / "rollout/states.npz").write_bytes(b"recorded states")
        (attempt / "rollout/scene.mjb").write_bytes(b"compiled scene")
        (attempt / "task.csv").write_text("sim_time\n0\n")
        (attempt / "nominal_context.csv").write_text("sim_time\n" + "0\n" * frames)
        outputs = {str(path.relative_to(attempt)): sha256(path) for path in attempt.rglob("*") if path.is_file()}
        write_json(attempt / "report.json", {"status": "passed" if success else "stopped",
                   "request": request, "plan_sha256": plan["plan_sha256"], "task_success": success,
                   "physics_executed": True, "sonic_executed": True, "outputs_sha256": outputs})
        return attempt

    def test_collection_plan_is_disjoint_and_does_not_load_models(self):
        output = self.root / "collection"
        with patch("baseline.execution.ExecutionRuntime", side_effect=AssertionError("No physics")):
            plan = plan_collection(ROOT / "configs/data/collection.json", output)
        self.assertEqual(len(plan["parents"]), 200)
        self.assertFalse(plan["physics_executed"])
        owners = {}
        for parent in plan["parents"]:
            owners.setdefault(parent["scene_seed"], set()).add(parent["split"])
        self.assertTrue(all(len(value) == 1 for value in owners.values()))
        batches, _ = collection_batches(output / "collection-plan.json")
        self.assertEqual(len(batches), 6)
        report = index_batches(batches, self.root / "index")
        self.assertEqual(report["pending"], 200)
        self.assertEqual(report["candidate_parents"], 0)

    def test_plan_rejects_seed_overlap_before_writing(self):
        spec = json.loads((ROOT / "configs/data/collection.json").read_text())
        spec["groups"][2]["seed_start"] = spec["groups"][0]["seed_start"]
        path = self.root / "config.json"
        write_json(path, spec)
        output = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "scene_seed leakage"):
            plan_collection(path, output)
        self.assertFalse(output.exists())

    def test_effective_default_and_explicit_prompts_cannot_cross_splits(self):
        implicit = {"prompt_profile": "focused", "phase_prompts": {}}
        explicit = {"prompt_profile": "focused", "phase_prompts": {
            key: "  " + value.upper() + "  " for key, value in FOCUSED_PROMPTS.items()}}
        self.assertEqual(prompt_identity(implicit), prompt_identity(explicit))
        owners = {}
        own_prompt(owners, implicit, "train")
        with self.assertRaisesRegex(ValueError, "Effective prompt"):
            own_prompt(owners, explicit, "test")

    def test_index_retains_failures_and_reports_short_and_pending_attempts(self):
        run, plan = self.batch("run", count=4)
        self.episode(run, plan, 1, success=True)
        self.episode(run, plan, 2, success=False)
        self.episode(run, plan, 3, frames=8)
        report = index_batches([{"run": str(run), "split": "train", "prompt_group": "a"}],
                               self.root / "index")
        self.assertEqual(report["candidate_parents"], 2)
        self.assertEqual(report["teacher_candidates"], 1)
        self.assertEqual(report["unusable"], 1)
        self.assertEqual(report["pending"], 1)
        sources = json.loads((self.root / "index/sources.json").read_text())
        self.assertEqual([parent["scene_seed"] for parent in sources["parents"]], [1, 2])

    def test_index_rejects_changed_evidence_and_effective_prompt_leakage(self):
        run, plan = self.batch("run")
        attempt = self.episode(run, plan, 1)
        (attempt / "rollout/states.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            index_batches([{"run": str(run), "split": "train", "prompt_group": "a"}], self.root / "bad")
        other, _ = self.batch("other", seed=100)
        with self.assertRaisesRegex(ValueError, "Effective prompt"):
            index_batches([{"run": str(run), "split": "train", "prompt_group": "a"},
                           {"run": str(other), "split": "val", "prompt_group": "b"}], self.root / "bad")

    def pair_sources(self):
        source = self.root / "source"
        (source / "ardy/lower").mkdir(parents=True)
        (source / "ardy/lower/reference.npz").write_bytes(b"saved motion")
        write_json(source / "report.json", {"status": "passed", "task_success": True,
                   "physics_executed": True, "sonic_executed": True})
        write_json(source / "request.json", {})
        write_json(source / "settings.json", {})
        sources = self.root / "sources.json"
        write_json(sources, {"baseline_lock_sha256": sha256(LOCK), "parents": [
            {"parent_id": "parent-a", "split": "val", "prompt_group": "val-a", "scene_seed": 12,
             "nominal": str(source), "source_report_sha256": sha256(source / "report.json")}]})
        variants = self.root / "perturbations.json"
        write_json(variants, {"perturbations": [{"id": "offset", "phase": "lower", "delay": .2,
                   "joint": "right_shoulder_pitch_joint", "amplitude": .15}]})
        return source, sources, variants

    def fake_collect(self, source, output, **parameters):
        output.mkdir(parents=True)
        branches = {}
        for name, success in (("teacher", True), ("nominal", False)):
            branch = output / name
            (branch / "rollout").mkdir(parents=True)
            (branch / "scene.xml").write_text("<scene/>")
            write_json(branch / "rollout/metadata.json", {"model_sha256": "same-model"})
            report = branch / "report.json"
            write_json(report, {"status": "passed" if success else "stopped", "task_success": success,
                       "physics_executed": True, "sonic_executed": True})
            branches[name] = {"report": str(report), "report_sha256": sha256(report),
                              "scene_sha256": sha256(branch / "scene.xml"), "model_sha256": "same-model",
                              "status": "passed" if success else "stopped", "task_success": success,
                              "physics_executed": True, "sonic_executed": True,
                              "activation": {"snapshot_sha256": "same-state", "time": 2., "frame_index": 100}}
        pair = {"source_attempt": str(source), "source_report_sha256": sha256(Path(source) / "report.json"),
                **parameters, "pair_state_verified": True, "teacher_verified": True,
                "recovery_verified": True, "branches": branches}
        write_json(output / "pair.json", pair)
        return pair

    def test_pairs_resume_without_duplicate_execution_and_keep_parent_split(self):
        source, sources, variants = self.pair_sources()
        output = self.root / "pairs"
        with patch("experiments.collect_pairs.collect", side_effect=self.fake_collect) as collect:
            result = collect_batch(sources, variants, output)
            self.assertEqual(result["recovery_verified"], 1)
            collect_batch(sources, variants, output, resume=True)
            self.assertEqual(collect.call_count, 1)
        parent = json.loads((output / "sources.json").read_text())["parents"][0]
        self.assertEqual(parent["split"], "val")
        self.assertEqual(parent["scene_seed"], 12)
        self.assertEqual(parent["source_report_sha256"], sha256(Path(parent["nominal"]) / "report.json"))
        (source / "ardy/lower/reference.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Resume source"):
            collect_batch(sources, variants, output, resume=True)

    def test_failed_pair_is_recorded_without_invented_correction_or_silent_retry(self):
        _, sources, variants = self.pair_sources()
        output = self.root / "pairs"
        with patch("experiments.collect_pairs.collect", side_effect=RuntimeError("missing saved reference")) as collect:
            result = collect_batch(sources, variants, output)
            collect_batch(sources, variants, output, resume=True)
            self.assertEqual(collect.call_count, 1)
        self.assertEqual(result["failed_pairs"], 1)
        parent = json.loads((output / "sources.json").read_text())["parents"][0]
        self.assertNotIn("pair", parent)

    def test_no_successful_source_reports_blocker_without_running_physics(self):
        source, sources, variants = self.pair_sources()
        write_json(source / "report.json", {"status": "stopped", "task_success": False})
        plan = json.loads(sources.read_text())
        plan["parents"][0]["source_report_sha256"] = sha256(source / "report.json")
        write_json(sources, plan)
        with patch("experiments.collect_pairs.collect", side_effect=AssertionError("No physics")):
            result = collect_batch(sources, variants, self.root / "pairs")
        self.assertEqual(result["status"], "no-teacher-candidates")
        self.assertEqual(result["planned_pairs"], 0)

    def test_interrupted_pair_keeps_old_attempt_and_can_resume(self):
        _, sources, variants = self.pair_sources()
        output = self.root / "pairs"
        def interrupt(source, path, **parameters):
            path.mkdir(parents=True)
            raise KeyboardInterrupt
        with patch("experiments.collect_pairs.collect", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                collect_batch(sources, variants, output)
        with patch("experiments.collect_pairs.collect", side_effect=self.fake_collect):
            result = collect_batch(sources, variants, output, resume=True)
        self.assertEqual(result["recovery_verified"], 1)
        self.assertEqual(len(list((output / "pairs").glob("*/attempt-*"))), 2)

    def test_inspection_can_skip_short_history_but_cannot_skip_corruption(self):
        path = self.root / "sources.json"
        write_json(path, {"parents": [{"parent_id": "a", "split": "train", "prompt_group": "a", "scene_seed": 1}]})
        with patch("experiments.build_dataset.extract_parent", side_effect=InsufficientHistory("too short")):
            result = inspect(path, thresholds=np.ones(4), skip_ineligible=True)
        self.assertEqual(result["parents"][0]["windows"], 0)
        with patch("experiments.build_dataset.extract_parent", side_effect=ValueError("corrupted state")):
            with self.assertRaisesRegex(ValueError, "corrupted"):
                inspect(path, thresholds=np.ones(4), skip_ineligible=True)
        context = self.root / "context.csv"
        context.write_text("sim_time\n0\n")
        with self.assertRaises(InsufficientHistory):
            read_context(context)

    def test_readiness_masks_censored_negatives_and_requires_residual_validation_categories(self):
        manifest = make_fixture(self.root / "fixture", ModelConfig(width=32, heads=4, layers=1))
        arrays = WindowDataset(manifest, "train", ModelConfig(width=32, heads=4, layers=1)).arrays
        summary = supervision_summary(arrays)
        self.assertTrue(readiness({"train": summary, "val": summary}, "residual")["ready"])
        changed = {key: value.copy() for key, value in arrays.items()}
        changed["intervention_valid"][:] = False
        changed["correction_sample"][:] = False
        bad = supervision_summary(changed)
        self.assertEqual(bad["risk_negative"], 0)
        self.assertEqual(bad["risk_censored"], len(arrays["parent_id"]))
        with self.assertRaisesRegex(ValueError, "val: no correction_samples"):
            require_ready({"train": summary, "val": bad}, "residual")

    def test_source_replay_preserves_history_and_backend_settings(self):
        source = self.root / "source"
        source.mkdir()
        write_json(source / "request.json", {"task": "grasp", "seed": 1, "cube_xy": [.4, -.2], "prompt_profile": "focused"})
        write_json(source / "settings.json", {"table_standoff_m": .2, "finger_kp_nm_per_rad": 6,
                   "finger_kd_nm_s_per_rad": .4, "hand_approach": "open", "final_hold_seconds": 10,
                   "contact_profile": "elliptic", "lift_seconds": 4.8, "direct_start": True})
        write_json(self.root / "plan.json", {"config": {"history_frames": 24, "threads": 2, "device": "cpu"}})
        args, _ = source_args(source)
        self.assertEqual(args.history_frames, 24)
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.device, "cpu")

    def test_conversion_round_trip_and_report_expose_missing_correction_supervision(self):
        config = ModelConfig(width=32, heads=4, layers=1)
        fixture = make_fixture(self.root / "fixture", config)
        metadata = json.loads(fixture.read_text())
        rows = {}
        for split in ("train", "val", "test"):
            arrays = WindowDataset(fixture, split, config).arrays
            for index, pid in enumerate(arrays["parent_id"]):
                rows[str(pid)] = {key: value[index] for key, value in arrays.items()}
        extra = {"parent_id": "short", "split": "train", "prompt_group": "short", "scene_seed": "short"}
        sources = self.root / "sources.json"
        write_json(sources, {"parents": [*metadata["parents"], extra]})
        def extract(parent, thresholds):
            if parent["parent_id"] == "short":
                raise InsufficientHistory("short episode")
            return [rows[parent["parent_id"]]], "model", {
                "parent_id": parent["parent_id"], "prompt_identity": parent["split"],
                "source_files_sha256": {"report.json": "hash"}, "decision_time_bounds": [1., 1.]}
        with patch("experiments.build_dataset.extract_parent", side_effect=extract):
            report = build(sources, self.root / "dataset", thresholds=np.ones(4),
                           skip_ineligible=True, require="residual")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["skipped_parents"]), 1)
        self.assertEqual(report["splits"]["val"]["correction_samples"], 4)
        for row in rows.values():
            row["correction_sample"] = False
            row["teacher_verified"] = False
            row["teacher_snapshot"] = ""
            row["offset_target"] = np.zeros_like(row["offset_target"])
        with patch("experiments.build_dataset.extract_parent", side_effect=extract):
            report = build(sources, self.root / "risk-only", thresholds=np.ones(4),
                           skip_ineligible=True, require="residual")
        self.assertEqual(report["status"], "not-ready")
        self.assertTrue(report["readiness"]["risk"]["ready"])
        self.assertFalse(report["readiness"]["residual"]["ready"])
        self.assertTrue((self.root / "risk-only/manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
