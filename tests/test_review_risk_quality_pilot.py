"""A single explicit missing-reference admission cannot bypass evidence gates."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import review_risk_quality_pilot as reviewer
from scripts.risk_quality_continue import (Continuation, InsufficientEvidence, claim_first, host_path,
    pilot_admission, pilot_gate, preserved_review, sha)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def fixture(repo):
    """Real branch/report conventions; only tensor bytes replace simulator IO."""
    first = repo / "output/risk-quality-261006"
    marker = repo / "baseline/frozen.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("frozen controller fixture")
    pinned = {"/workspace/dl/baseline/frozen.py": sha(marker)}
    write(first / "state.json", {"schema": "dl-risk-quality-pipeline-v1", "run_id": first.name,
        "status": "failed", "stage": "teacher-pilot", "pid": None, "child_pid": None,
        "source_sha256": pinned})
    parents = []
    for group, split, start in (("train-a", "train", 12000), ("train-b", "train", 13000), ("val-a", "val", 22000)):
        for offset in range(5):
            source = first / "parents" / group / f"attempt-{offset + 1:05d}"
            write(source / "report.json", {"status": "passed", "task_success": True})
            parents.append({"parent_id": f"quality-{group}-{start + offset}-{offset + 1}",
                "prompt_group": "quality-" + group, "scene_seed": start + offset, "split": split,
                "nominal": "/workspace/dl/" + str(source.relative_to(repo))})
    write(first / "index/sources.json", {"parents": parents})
    jobs, results = [], {}
    selected = [parents[2], parents[3], parents[10], parents[11], parents[12]]
    for index in range(15):
        parent = selected[index // 3]
        # Real saved plan ordering is independently specified by the fixture.
        initial = ("82d5cbfd8efd62db4a7c", "f0d97ec9f1fd970b463a", "d2c17d17bf361d989a94")
        key = initial[index] if index < 3 else f"pending-{index:02d}"
        variant = {"id": f"variant-{index % 3}", "phase": "lift" if index % 3 == 2 else "lower", "delay": .2,
                   "joint": "right_shoulder_pitch_joint", "amplitude": (.15, -.15, .1)[index % 3]}
        source = host_path(repo, parent["nominal"])
        job = {"id": key, "parent": parent, "source": parent["nominal"], "perturbation": variant,
               "source_files_sha256": {str(Path(parent["nominal"]) / "report.json"): sha(source / "report.json")}}
        jobs.append(job)
        if index >= 3:
            continue
        pair_path = first / "pairs/pairs" / key / "attempt-001/pair.json"
        branches = {}
        for name in ("teacher", "nominal"):
            branch = pair_path.parent / name
            success = name == "teacher" or key == reviewer.VALID_JOB_IDS[1]
            status = "failed" if key == reviewer.EXCLUDED_JOB_ID and name == "nominal" else "passed"
            activation = {"time": 1.2, "frame_index": 60, "phase": variant["phase"], "snapshot_sha256": "a" * 64}
            event = {"event": "teacher_pair_activation", "sim_time": 1.2, "wall_time": 99., **activation,
                "perturb": name == "nominal", "joint": variant["joint"],
                "amplitude": variant["amplitude"] if name == "nominal" else 0.}
            artifacts = {"scene.xml": "same physical scene", "rollout/scene.mjb": "same compiled model",
                "rollout/states.npz": "recorded simulator states", "task.csv": "task signals",
                "events.jsonl": json.dumps(event) + "\n", "nominal_context.csv": "nominal inputs",
                "effective_context.csv": "executed inputs", "ardy/lower/reference.npz": "saved reference"}
            for relative, value in artifacts.items():
                path = branch / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            model_sha = sha(branch / "rollout/scene.mjb")
            write(branch / "rollout/metadata.json", {"model_sha256": model_sha})
            outputs = {relative: sha(branch / relative) for relative in (*artifacts, "rollout/metadata.json")}
            report = {"status": status, "task_success": success, "physics_executed": True,
                "sonic_executed": True, "model_load_failed": False, "outputs_sha256": outputs}
            if status == "failed":
                report.update(failure_reason="lower_error", runtime_error=reviewer.MISSING_PREFIX +
                    str(source / reviewer.MISSING_RELATIVE))
            write(branch / "report.json", report)
            branches[name] = {"status": status, "task_success": success, "physics_executed": True,
                "sonic_executed": True, "report": "/workspace/dl/" + str((branch / "report.json").relative_to(repo)),
                "report_sha256": sha(branch / "report.json"), "scene_sha256": outputs["scene.xml"],
                "model_sha256": model_sha, "activation": activation}
        recovery = key == reviewer.VALID_JOB_IDS[0]
        write(pair_path, {"source_attempt": job["source"], "source_report_sha256": sha(source / "report.json"),
            "pair_state_verified": True, "teacher_verified": True, "recovery_verified": recovery,
            "branches": branches, **{name: variant[name] for name in ("phase", "delay", "joint", "amplitude")}})
        results[key] = {"status": "invalid_execution" if key == reviewer.EXCLUDED_JOB_ID else "paired", "pair": str(pair_path),
                        "teacher_verified": True, "recovery_verified": recovery}
    plan = {"schema": "dl-pair-batch-v1", "jobs": jobs, "baseline_sources_sha256": pinned,
            "sources_sha256": sha(first / "index/sources.json"), "baseline_lock_sha256": "frozen assets"}
    plan["plan_sha256"] = reviewer.digest(plan)
    write(first / "pairs/plan.json", plan)
    write(first / "pairs/report.json", {"status": "pending", "jobs": results,
        "plan_sha256": plan["plan_sha256"], "planned_pairs": 15, "pending_pairs": 12,
        "failed_pairs": 1, "recovery_verified": 1})
    exported = [dict(selected[0], parent_id=selected[0]["parent_id"] + "--" + jobs[index]["perturbation"]["id"],
                     nominal=str(Path(results[jobs[index]["id"]]["pair"]).parent / "nominal"),
                     pair=results[jobs[index]["id"]]["pair"]) for index in (0, 2)]
    write(first / "pairs/sources.json", {"parents": exported + [row for row in parents if row != selected[0]],
        "baseline_lock_sha256": "frozen assets", "pair_plan_sha256": plan["plan_sha256"],
        "original_sources_sha256": plan["sources_sha256"]})
    return first, plan


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.first, self.plan = fixture(self.repo)
        pin = patch.object(reviewer, "EXPECTED_PLAN_SHA256", self.plan["plan_sha256"])
        pin.start()
        self.addCleanup(pin.stop)

    def certificate(self):
        return reviewer.create_certificate(self.repo, self.first, self.repo / "output/explicit-review")

    def mutate_branch_report(self, job_id, branch, change):
        """Re-sign pair metadata to test actual semantic guards, not stale SHA only."""
        result = json.loads((self.first / "pairs/report.json").read_text())["jobs"][job_id]
        pair_path = Path(result["pair"])
        pair = json.loads(pair_path.read_text())
        report_path = host_path(self.repo, pair["branches"][branch]["report"])
        report = json.loads(report_path.read_text())
        change(report, report_path.parent)
        write(report_path, report)
        pair["branches"][branch]["report_sha256"] = sha(report_path)
        write(pair_path, pair)

    def test_explicit_review_preserves_failure_and_records_independent_coverage(self):
        old = {str(path): path.read_bytes() for path in self.first.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "terminal pilot-review"):
            pilot_admission(self.repo, self.first)
        certificate = self.certificate()
        state, _, counts, value = pilot_admission(self.repo, self.first, certificate)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(counts["valid_pairs"], 2)
        self.assertEqual(counts["verified_clean_successes"], 3)
        self.assertEqual(counts["reviewed_independent_parents"], 1)
        self.assertEqual(counts["new_recovery_parents_by_split"], {"train": 1, "val": 0})
        self.assertEqual(len(value["remaining_job_ids"]), 12)
        self.assertEqual(value["excluded_job_ids"], [reviewer.EXCLUDED_JOB_ID])
        self.assertEqual({str(path): path.read_bytes() for path in self.first.rglob("*") if path.is_file()}, old)
        with self.assertRaises(FileExistsError):
            self.certificate()

    def test_active_parent_child_or_permission_denied_cannot_be_reviewed(self):
        for key in ("pid", "child_pid"):
            with self.subTest(key=key):
                state = json.loads((self.first / "state.json").read_text())
                state[key] = 456
                write(self.first / "state.json", state)
                with patch("scripts.risk_quality_pipeline.os.kill", side_effect=PermissionError), \
                        self.assertRaisesRegex(ValueError, "still alive"):
                    self.certificate()
                state[key] = None
                write(self.first / "state.json", state)

    def test_only_exact_known_missing_reference_error_is_admitted(self):
        cases = (("runtime_error", "RuntimeError: unrelated failure"), ("failure_reason", "hold_error"),
                 ("model_load_failed", True), ("runtime_error", reviewer.MISSING_PREFIX + "/different/reference.npz"))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                first_original = {path: path.read_bytes() for path in self.first.rglob("*") if path.is_file()}
                self.mutate_branch_report(reviewer.EXCLUDED_JOB_ID, "nominal", lambda report, _: report.update({field: value}))
                with self.assertRaisesRegex(ValueError, "exact known"):
                    self.certificate()
                for path, data in first_original.items():
                    path.write_bytes(data)
        missing = host_path(self.repo, self.plan["jobs"][1]["source"]) / reviewer.MISSING_RELATIVE
        missing.parent.mkdir(parents=True)
        missing.write_text("invented reference")
        with self.assertRaisesRegex(ValueError, "exact known"):
            self.certificate()

    def test_rollout_corruption_and_even_resigned_activation_corruption_are_rejected(self):
        pair_path = Path(json.loads((self.first / "pairs/report.json").read_text())["jobs"][reviewer.VALID_JOB_IDS[0]]["pair"])
        states = pair_path.parent / "nominal/rollout/states.npz"
        original = states.read_bytes()
        states.write_text("modified recorded states")
        with self.assertRaisesRegex(ValueError, "output evidence changed"):
            self.certificate()
        states.write_bytes(original)
        def changed_event(report, branch):
            event = json.loads((branch / "events.jsonl").read_text())
            event["frame_index"] += 1
            (branch / "events.jsonl").write_text(json.dumps(event) + "\n")
            report["outputs_sha256"]["events.jsonl"] = sha(branch / "events.jsonl")
        self.mutate_branch_report(reviewer.VALID_JOB_IDS[0], "nominal", changed_event)
        with self.assertRaisesRegex(ValueError, "hashed actual event"):
            self.certificate()

    def test_certificate_or_current_summary_changes_require_new_explicit_review(self):
        certificate = self.certificate()
        original = certificate.read_bytes()
        value = json.loads(certificate.read_text())
        value["counts"]["valid_pairs"] = 3
        write(certificate, value)
        with self.assertRaisesRegex(ValueError, "digest changed"):
            reviewer.validate_certificate(self.repo, self.first, certificate)
        certificate.write_bytes(original)
        report = json.loads((self.first / "pairs/report.json").read_text())
        report["status"] = "changed"
        write(self.first / "pairs/report.json", report)
        with self.assertRaisesRegex(ValueError, "after explicit review"):
            reviewer.validate_certificate(self.repo, self.first, certificate)

    def test_invalid_pair_cannot_drop_or_weaken_same_state_fingerprint(self):
        pair_path = Path(json.loads((self.first / "pairs/report.json").read_text())["jobs"][reviewer.EXCLUDED_JOB_ID]["pair"])
        original = pair_path.read_bytes()
        cases = (("snapshot_sha256", None), ("snapshot_sha256", "not-a-hash"), ("frame_index", True),
                 ("phase", "hold"), ("time", -1.))
        for key, value in cases:
            with self.subTest(key=key, value=value):
                pair = json.loads(original)
                for branch in pair["branches"].values():
                    if value is None:
                        del branch["activation"][key]
                    else:
                        branch["activation"][key] = value
                write(pair_path, pair)
                with self.assertRaisesRegex(ValueError, "hashed actual event"):
                    self.certificate()
                pair_path.write_bytes(original)
    def completed_report(self):
        report = json.loads((self.first / "pairs/report.json").read_text())
        report.update(status="complete", pending_pairs=0)
        for job in self.plan["jobs"][3:]:
            report["jobs"][job["id"]] = {"status": "failed", "error": "fixture preserved rejection"}
        return report

    def test_completed_expansion_preserves_old_results_and_never_exports_invalid(self):
        certificate = json.loads(self.certificate().read_text())
        report = self.completed_report()
        counts = preserved_review(self.repo, self.first, certificate, report)
        self.assertEqual(counts["attempted_candidates"], 15)
        self.assertEqual(counts["physical_pair_denominator"], 2)
        changed = copy.deepcopy(report)
        changed["jobs"][reviewer.EXCLUDED_JOB_ID]["pair"] = "retried/attempt-002/pair.json"
        with self.assertRaisesRegex(ValueError, "altered or retried"):
            preserved_review(self.repo, self.first, certificate, changed)
        sources = json.loads((self.first / "pairs/sources.json").read_text())
        sources["parents"].append({"pair": report["jobs"][reviewer.EXCLUDED_JOB_ID]["pair"]})
        write(self.first / "pairs/sources.json", sources)
        with self.assertRaisesRegex(ValueError, "training source"):
            preserved_review(self.repo, self.first, certificate, report)

    def test_expansion_limit_and_missing_validation_recovery_stop_downstream(self):
        certificate = json.loads(self.certificate().read_text())
        write(self.first / "pairs/report.json", self.completed_report())
        pipeline = object.__new__(Continuation)
        pipeline.repo, pipeline.first = self.repo, self.first
        pipeline.sources = self.first / "index/sources.json"
        pipeline.pair_plan, pipeline.certificate = self.plan, certificate
        pipeline.state = {"review_counts": certificate["counts"]}
        executed, events = [], []
        pipeline.execute = lambda stage, command: executed.append((stage, command)) or 1
        pipeline.gpu = lambda stage: 0
        pipeline.docker = lambda gpu, *arguments: list(arguments)
        pipeline.container_path = lambda path: str(path)
        pipeline.event = lambda stage, details: events.append((stage, details))
        with self.assertRaisesRegex(InsufficientEvidence, "absent in train or val"):
            pipeline.run()
        self.assertEqual(len(executed), 1)
        self.assertIn("--resume", executed[0][1])
        self.assertEqual(executed[0][1][-2:], ["--limit", "12"])
        self.assertEqual(dict(events)["teacher-plan-processed"]["new_recovery_parents_by_split"], {"train": 1, "val": 0})

    def test_certificate_constructor_pins_host_paths_but_not_mutable_collector_summaries(self):
        certificate = self.certificate()
        paths = ("calibration.json", "diagnosis.json", "configs/learning/p.json",
                 "output/teacher-risk-261006/pairs/sources.json", "output/teacher-risk-261006/dataset/manifest.json",
                 "output/data-collection-261005/prompts/val-a.json")
        for relative in paths:
            write(self.repo / relative, {})
        def initialize(pipeline, repo, output, sources, config, preferred):
            pipeline.repo, pipeline.output, pipeline.sources = repo, output, sources
            pipeline.inputs, pipeline.state = {}, {}
        with patch("scripts.risk_quality_continue.calibration_gate"), \
                patch("scripts.risk_quality_continue.Pipeline.__init__", initialize), \
                patch("scripts.risk_quality_continue.subprocess.check_output", return_value=" M scripts/risk_quality_continue.py\n"), \
                patch.object(Continuation, "save"):
            pipeline = Continuation(self.repo, self.repo / "output/next", self.first,
                                    self.repo / "calibration.json", self.repo / "diagnosis.json", certificate)
        pipeline.check_sources()
        # Runtime module pins may themselves live under /workspace/dl when
        # these tests run in the container. Only fixture-origin keys require
        # translation into this fixture's independent host repository.
        value = json.loads(certificate.read_text())
        fixture_pins = {**value["original_state"]["source_sha256"],
                        **value["immutable_evidence_sha256"]}
        for origin, expected in fixture_pins.items():
            mapped = host_path(self.repo, origin).resolve()
            self.assertEqual(pipeline.inputs[str(mapped)], expected)
            self.assertEqual(sha(mapped), expected)
            if origin.startswith("/workspace/dl/"):
                self.assertTrue(mapped.is_relative_to(self.repo.resolve()))
                self.assertNotIn(origin, pipeline.inputs)
        self.assertNotIn(str(self.first / "pairs/report.json"), pipeline.inputs)
        self.assertNotIn(str(self.first / "pairs/sources.json"), pipeline.inputs)
        self.assertIn(str(certificate.resolve()), pipeline.inputs)
        self.assertEqual(host_path(self.repo, "/data/group3/dl-output/fixture"), Path("/data/group3/dl-output/fixture"))
        claim_first(self.first, self.repo / "output/next")
        with self.assertRaises(FileExistsError):
            claim_first(self.first, self.repo / "output/second-next")

    def test_changed_production_plan_pin_is_rejected(self):
        with patch.object(reviewer, "EXPECTED_PLAN_SHA256", "different known plan"):
            with self.assertRaisesRegex(ValueError, "Immutable pair/source plan"):
                self.certificate()

    def test_copy_of_same_named_first_cannot_get_another_continuation_claim(self):
        copied = self.repo / "output/copied/risk-quality-261006"
        shutil.copytree(self.first, copied)
        with self.assertRaisesRegex(ValueError, "exact failed initial"):
            reviewer.snapshot_review(self.repo, copied)


if __name__ == "__main__":
    unittest.main()
