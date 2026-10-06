"""Train-only candidate derivation, floors, held-out boundaries and provenance."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.calibrate_labels import calibrate, derive
from risk_residual.config import BODY_GROUPS


def diagnosis():
    p95 = (.13788280141353604, .13251932995319363, .05060421420633789, .11984458810091017)
    return {"schema": "dl-label-diagnosis-v1", "manifest_sha256": "a" * 64,
            "source_plan_sha256": "b" * 64, "tracking_thresholds_rad": [.06, .06, .1, .1],
            "splits": {
                "train": {"physical_stable_hold": {
                    "original_episodes": 35, "windows": 6712,
                    "per_episode_max_horizon_p95_rad": {
                        body: {"p50": value / 2, "p90": value * .9, "p95": value, "p99": value * 2}
                        for body, value in zip(BODY_GROUPS, p95)}}},
                "val": {"physical_stable_hold": {"original_episodes": 12, "windows": 2234,
                                                   "risk_positive": 2234}}}}


class LabelCalibrationTests(unittest.TestCase):
    def test_exact_train_rule_matches_recorded_candidate_without_using_val_quantiles(self):
        report = diagnosis()
        before = copy.deepcopy(report)
        result = derive(report)
        self.assertEqual(result["candidate_tracking_thresholds_rad"], [.14, .14, .1, .12])
        self.assertFalse(result["validation_check"]["used_for_selection"])
        self.assertEqual(report, before)
        report["splits"]["val"]["physical_stable_hold"]["per_episode_max_horizon_p95_rad"] = {
            body: {"p95": 100.} for body in BODY_GROUPS}
        report["splits"]["val"]["physical_stable_hold"]["risk_positive"] = 0
        self.assertEqual(derive(report)["candidate_tracking_thresholds_rad"], [.14, .14, .1, .12])

    def test_floor_never_lowers_existing_threshold_and_rounds_up(self):
        report = diagnosis()
        report["tracking_thresholds_rad"] = [.145, .2, .1, .15]
        self.assertEqual(derive(report)["candidate_tracking_thresholds_rad"], [.15, .2, .1, .15])

    def test_insufficient_independent_episodes_and_empty_evidence_are_rejected(self):
        for key, value in (("original_episodes", 9), ("windows", 0)):
            report = diagnosis()
            report["splits"]["train"]["physical_stable_hold"][key] = value
            with self.assertRaisesRegex(ValueError, "at least 10 independent"):
                derive(report)

    def test_test_diagnosis_and_missing_provenance_are_rejected(self):
        report = diagnosis()
        report["splits"]["test"] = {}
        with self.assertRaisesRegex(ValueError, "test stays held out"):
            derive(report)
        report = diagnosis()
        report["source_plan_sha256"] = None
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            derive(report)

    def test_nonfinite_and_unordered_train_statistics_are_rejected(self):
        for value in (float("nan"), -1, float("inf")):
            report = diagnosis()
            report["splits"]["train"]["physical_stable_hold"]["per_episode_max_horizon_p95_rad"][BODY_GROUPS[0]]["p95"] = value
            with self.assertRaises(ValueError):
                derive(report)
        report = diagnosis()
        report["splits"]["train"]["physical_stable_hold"]["per_episode_max_horizon_p95_rad"][BODY_GROUPS[0]]["p50"] = 1
        with self.assertRaisesRegex(ValueError, "Unordered"):
            derive(report)

    def test_artifact_pins_exact_input_and_cannot_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "diagnosis.json", root / "calibration.json"
            source.write_text(json.dumps(diagnosis()))
            before = source.read_bytes()
            expected = hashlib.sha256(before).hexdigest()
            result = calibrate(source, out, expected_diagnosis_sha256=expected)
            self.assertEqual(result["input_diagnosis_sha256"], expected)
            self.assertEqual(result, json.loads(out.read_text()))
            self.assertEqual(source.read_bytes(), before)
            preserved = out.read_bytes()
            with self.assertRaises(FileExistsError):
                calibrate(source, out)
            self.assertEqual(out.read_bytes(), preserved)
            with self.assertRaisesRegex(ValueError, "differs"):
                calibrate(source, root / "mismatch.json", expected_diagnosis_sha256="0" * 64)
            self.assertFalse((root / "mismatch.json").exists())


if __name__ == "__main__":
    unittest.main()
