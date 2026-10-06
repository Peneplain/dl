"""Workflow gates must reject incomplete data and preserve busy GPUs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.teacher_risk_pipeline import Pipeline, check_audit, check_pairs, idle_gpus


class PipelineGateTests(unittest.TestCase):
    def test_gpu_allocation_excludes_existing_processes_and_busy_devices(self):
        snapshot = """
0    MTT S4000 |00000000:08:00.0 |0%    0MiB(49152MiB)
1    MTT S4000 |00000000:09:00.0 |0%    16550MiB(49152MiB)
2    MTT S4000 |00000000:0e:00.0 |5%    0MiB(49152MiB)
3    MTT S4000 |00000000:11:00.0 |0%    0MiB(49152MiB)
Processes:
0    1234    python    0MiB
1    2345    python    16550MiB
"""
        self.assertEqual(idle_gpus(snapshot), [3])
        with self.assertRaises(ValueError):
            idle_gpus("Driver error")

    def test_completed_collection_can_include_rejected_pairs_but_not_pending_jobs(self):
        report = {"status": "complete", "pending_pairs": 0, "planned_pairs": 2,
                  "failed_pairs": 1, "jobs": {"a": {"status": "paired"}, "b": {"status": "failed"}}}
        check_pairs(report)
        for changed in ({"pending_pairs": 1}, {"status": "pending"}, {"jobs": {"a": {}}},
                        {"planned_pairs": 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_pairs({**report, **changed})

    def test_risk_readiness_alone_cannot_start_final_training(self):
        ready = {"risk": {"ready": True}, "residual": {"ready": True}}
        check_audit(ready)
        with self.assertRaisesRegex(ValueError, "no correction_samples"):
            check_audit({**ready, "residual": {"ready": False,
                         "issues": ["val: no correction_samples"]}})
        with self.assertRaisesRegex(ValueError, "risk"):
            check_audit({**ready, "risk": {"ready": False, "issues": ["train: no risk_negative"]}})

    def workflow(self, directory, *, residual_ready):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.repo = Path(directory)
        pipeline.output = pipeline.repo / "output/pipeline"
        pipeline.output.mkdir(parents=True)
        pipeline.sources = pipeline.repo / "sources.json"
        pipeline.config = pipeline.repo / "configs/learning/p.json"
        pipeline.preferred_gpu = 4
        pipeline.state = {"events": []}
        stages, commands = [], []

        def execute(stage, command):
            stages.append(stage)
            commands.append(command)
            if stage == "teacher":
                path = pipeline.output / "pairs/report.json"
                report = {"status": "complete", "planned_pairs": 2, "pending_pairs": 0,
                          "recovery_verified": 1, "failed_pairs": 1, "jobs": {"a": {}, "b": {}}}
            elif stage == "convert":
                path, report = pipeline.output / "dataset/report.json", {"status": "passed"}
            elif stage == "audit":
                path = pipeline.output / "dataset-audit.json"
                report = {"risk": {"ready": True}, "residual": {"ready": residual_ready,
                          "issues": [] if residual_ready else ["val: no correction_samples"]}}
            else:
                path = pipeline.output / "risk/summary.json"
                report = {"status": "passed", "jobs": [{"status": "passed"} for _ in range(7)]}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report))
            return 1 if stage == "teacher" else 0

        return pipeline, stages, commands, execute

    def test_complete_workflow_uses_all_free_gpus_and_paths_visible_in_both_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline, stages, commands, execute = self.workflow(directory, residual_ready=True)
            with patch.object(pipeline, "gpus", side_effect=[[4], [0, 2, 3, 4, 5, 6, 7]]), \
                    patch.object(pipeline, "execute", side_effect=execute):
                pipeline.run()
            self.assertEqual(stages, ["teacher", "convert", "audit", "risk"])
            self.assertEqual(pipeline.state["status"], "complete")
            self.assertEqual(commands[-1][2:], ["0,2,3,4,5,6,7",
                "output/pipeline/dataset/manifest.json", "configs/learning/p.json", "output/pipeline/risk"])
            self.assertIn("/workspace/dl/sources.json", commands[0])

    def test_failed_final_audit_stops_before_any_risk_training(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline, stages, _, execute = self.workflow(directory, residual_ready=False)
            with patch.object(pipeline, "gpus", return_value=[4]), \
                    patch.object(pipeline, "execute", side_effect=execute), \
                    self.assertRaisesRegex(ValueError, "no correction_samples"):
                pipeline.run()
            self.assertEqual(stages, ["teacher", "convert", "audit"])


if __name__ == "__main__":
    unittest.main()
