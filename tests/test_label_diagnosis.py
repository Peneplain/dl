"""Diagnostic invariants: causal masks, source identity and measured stable holds."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.diagnose_labels import (branch_layout, diagnose, digest, physical_hold,
                                        reasons, sensitivity)
from risk_residual.config import PHASES


def arrays(count):
    return {"future_valid": np.ones((count, 8, 4), bool),
            "track_target": np.zeros((count, 8, 4), np.float32),
            "contact_mask": np.zeros((count, 8, 4), bool),
            "contact_target": np.zeros((count, 8, 4), np.float32),
            "balance_target": np.zeros((count, 8, 4), np.float32),
            "intervention_valid": np.ones(count, bool),
            "intervention_target": np.zeros(count, np.float32),
            "correction_sample": np.zeros(count, bool),
            "context": np.zeros((count, 8, len(PHASES) + 14), np.float32),
            "nominal_times": np.broadcast_to(np.arange(8) * .04 + .3, (count, 8)).copy()}


class LabelDiagnosisTests(unittest.TestCase):
    def test_reasons_keep_recovery_and_censored_positives_separate(self):
        a = arrays(4)
        a["track_target"][0, 0, 1] = 1
        a["contact_mask"][1, 0, 1] = True
        a["contact_target"][1, 0, 1] = 1
        a["future_valid"][2, 1:] = False
        a["intervention_valid"][2] = False
        a["correction_sample"][3] = True
        a["future_valid"][3, 1:] = False
        a["intervention_target"][:] = [1, 1, 0, 1]
        result = reasons(a)
        self.assertEqual(result["tracking"].tolist(), [True, False, False, False])
        self.assertEqual(result["contact"].tolist(), [False, True, False, False])
        self.assertEqual(result["verified_recovery"].tolist(), [False, False, False, True])
        a["intervention_valid"][2] = True
        with self.assertRaisesRegex(ValueError, "disagree"):
            reasons(a)

    def test_sensitivity_never_removes_physical_or_recovery_labels(self):
        a = arrays(5)
        a["track_target"][0, 0, 1] = 2
        a["contact_mask"][1, 0, 1] = True
        a["contact_target"][1, 0, 1] = 1
        a["balance_target"][2, 0, 3] = 1
        a["correction_sample"][3] = True
        a["future_valid"][4, 1:] = False
        a["intervention_valid"][4] = False
        result = sensitivity(a, np.array([.06, .06, .1, .1]),
                             [[.2, .2, .2, .2]], np.array([True, False, False, False, False]))[0]
        self.assertEqual(result["positive"], 3)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["censored"], 1)
        self.assertEqual(result["verified_recovery_kept_positive"], 1)
        self.assertEqual(result["verified_recovery_tracking_trigger"], 0)
        self.assertEqual(result["physical_stable_hold_positive"], 0)
        np.testing.assert_equal(a["track_target"][0, 0, 1], 2)

    def test_test_split_is_refused_before_reading_a_manifest(self):
        with self.assertRaisesRegex(ValueError, "test stays held out"):
            diagnose("missing-file.json", splits=("test",))

    def test_branch_boundary_requires_one_reset_and_unchanged_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nominal, teacher = root / "nominal", root / "teacher"
            nominal.mkdir()
            teacher.mkdir()
            for path, success in ((nominal, False), (teacher, True)):
                (path / "report.json").write_text(json.dumps({"task_success": success}))
            pair = root / "pair.json"
            pair.write_text(json.dumps({"source_attempt": str(root / "original"), "branches": {
                "teacher": {"report": str(teacher / "report.json"),
                            "report_sha256": digest(teacher / "report.json")}}}))
            parent = {"nominal": str(nominal), "pair": str(pair),
                      "source_report_sha256": digest(nominal / "report.json")}
            a = {"parent_id": np.array(["p"] * 4), "decision_time": np.array([.3, .4, .3, .4])}
            cohort, outcome, _, origins = branch_layout(a, {"p": parent})
            self.assertEqual(cohort.tolist(), ["paired_nominal"] * 2 + ["teacher_clean"] * 2)
            self.assertEqual(outcome.tolist(), [False, False, True, True])
            self.assertEqual(len(set(origins)), 1)
            a["decision_time"] = np.array([.3, .4, .5, .6])
            with self.assertRaisesRegex(ValueError, "boundary"):
                branch_layout(a, {"p": parent})

    def test_physical_hold_requires_clearance_contacts_complete_future_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.csv"
            times = np.arange(113) * .005
            task.write_text("sim_time,lowest_clearance_m,thumb_force_n,finger_force_n\n" +
                            "".join(f"{t:.8f},.06,.02,.03\n" for t in times))
            report = root / "report.json"
            report.write_text(json.dumps({"outputs_sha256": {"task.csv": digest(task)}}))
            a = arrays(4)
            a["context"][:, :, PHASES.index("hold")] = 1
            a["nominal_times"][:] = np.arange(8) * .04 + .1
            a["future_valid"][1, -1] = False
            success = np.array([True, True, False, True])
            cohort = np.array(["teacher_clean", "teacher_clean", "teacher_clean", "paired_nominal"])
            attempts = np.array([str(root)] * 4, dtype=object)
            self.assertEqual(physical_hold(a, cohort, success, attempts).tolist(), [True, False, False, False])
            # A slip between the 40 ms label samples must reject the interval.
            task.write_text(task.read_text().replace("0.11500000,.06", "0.11500000,.04"))
            report.write_text(json.dumps({"outputs_sha256": {"task.csv": digest(task)}}))
            self.assertFalse(physical_hold(a, cohort, success, attempts).any())


if __name__ == "__main__":
    unittest.main()
