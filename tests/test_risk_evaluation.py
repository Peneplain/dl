"""Masking, score ties, degenerate classes, parent clustering and gate provenance."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.common import sha256
from experiments.calibrate import select_threshold
from experiments.evaluate_risk import (binary_metrics, episode_cluster_ids, evaluate_arrays,
                                      fit_temperature, future_onset_metrics, infer_checkpoint, parent_summary,
                                      precision_recall)
from experiments.smoke import make_fixture
from risk_residual.config import ModelConfig, SCHEMA
from risk_residual.data import Normalizer, WindowDataset
from risk_residual.models import RiskModel
from baseline.common import LOCK


class RiskEvaluationTests(unittest.TestCase):
    def test_skewed_always_high_is_not_balanced_accuracy(self):
        labels = np.r_[np.ones(93), np.zeros(7)]
        result = binary_metrics(np.ones(100), labels)
        self.assertEqual(result["accuracy"], .93)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["false_positive_rate"], 1.)
        self.assertAlmostEqual(result["risk_average_precision"], .93)
        self.assertAlmostEqual(result["low_risk_average_precision"], .07)

    def test_tied_scores_group_before_integrating_pr(self):
        result = precision_recall([.9, .9, .1, .1], [1, 0, 1, 0])
        self.assertEqual(result["thresholds"], [.9, .1])
        self.assertEqual(result["average_precision"], .5)
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(precision_recall([.9, .8, .2, .1], [1, 1, 0, 0])["average_precision"], 1.)

    def test_unavailable_labels_and_missing_classes_are_explicit(self):
        result = binary_metrics([.8, np.nan], [1, np.nan], mask=np.array([True, False]))
        self.assertEqual(result["available"], 1)
        self.assertIsNone(result["specificity"])
        self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(result["low_risk_average_precision"])
        empty = binary_metrics([], [])
        self.assertEqual(empty["available"], 0)
        self.assertIsNone(empty["accuracy"])
        with self.assertRaises(ValueError):
            binary_metrics([1.1], [1])
        with self.assertRaises(ValueError):
            binary_metrics([.8], [1], mask=np.array([1]))

    def test_cluster_bootstrap_resamples_whole_parents(self):
        # With 1 all-negative and 1 all-positive cluster, about half the draws
        # contain only one class. Undefined BA must not become artificial 0/1.
        result = parent_summary(np.array([.1] * 20 + [.9] * 20), np.array([0] * 20 + [1] * 20),
                                np.ones(40, bool), np.array(["a"] * 20 + ["b"] * 20),
                                threshold=.5, bootstrap=100, seed=7)
        self.assertEqual(result["parents"], 2)
        self.assertEqual(result["bootstrap"]["risk_recall"]["ci95"], [1., 1.])
        self.assertLess(result["bootstrap"]["balanced_accuracy"]["defined_draws"], 100)
        self.assertEqual(result["macro"]["specificity"]["contributing_parents"], 1)

    def test_temperature_requires_validation_and_both_classes(self):
        result = fit_temperature([8., -8., 8., -8.], [1, 0, 0, 1])
        self.assertGreater(result["temperature"], 1.)
        self.assertLessEqual(result["fitted_validation_nll"], result["raw_validation_nll"])
        self.assertFalse(result["runtime_applied"])
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_temperature([1, -1], [1, 0], split="test")
        with self.assertRaises(ValueError):
            fit_temperature([1, 1], [1, 1])

    def test_existing_f1_tie_rule_is_preserved(self):
        result = select_threshold([.1, .9], [0, 1])
        self.assertEqual(result["threshold"], .9)
        self.assertEqual(result["validation_f1"], 1.)

    def test_current_reactive_does_not_establish_future_prediction_or_mix_direct_recovery(self):
        # 0 already current risk; 1 future onset; 2 complete stable; 3 recovery
        # without any proxy; 4 censored-negative future; 5 partial positive future.
        n = 6
        a = {"intervention_logit": np.zeros(n), "probability": np.array([.9, .9, .1, .8, .2, .7]),
             "future_valid": np.ones((n, 8, 4), bool), "contact_mask": np.zeros((n, 8, 4), bool),
             "track_target": np.zeros((n, 8, 4)), "contact_target": np.zeros((n, 8, 4)),
             "balance_target": np.zeros((n, 8, 4)), "correction_sample": np.array([0, 0, 0, 1, 0, 0], bool),
             "intervention_target": np.array([1, 1, 0, 1, 0, 1]),
             "intervention_valid": np.array([1, 1, 1, 1, 0, 1], bool),
             "phase_index": np.zeros(n, int)}
        a["track_target"][0, 0, 0] = 1.1
        a["track_target"][1, 2, 0] = 1.1
        a["track_target"][5, 1, 0] = 1.1
        a["future_valid"][[4, 5], 4:] = False
        # Masked NaNs must not produce an onset or consume a negative label.
        a["track_target"][[4, 5], 4:] = np.nan
        result = future_onset_metrics(a, threshold=.5)
        counts = result["counts"]
        self.assertEqual(counts["already_current_proxy_positive"], 1)
        self.assertEqual(counts["future_only_proxy_positive"], 2)
        self.assertEqual(counts["direct_recovery_only_positive"], 1)
        self.assertEqual(counts["future_onset_positive"], 2)
        self.assertEqual(counts["future_onset_negative"], 1)
        self.assertEqual(counts["future_onset_censored"], 1)
        onset = result["current_proxy_negative_future_onset"]
        self.assertEqual(onset["raw_gate"]["risk_recall"], 1.)
        self.assertEqual(onset["raw_gate"]["false_positive_rate"], 0.)
        self.assertEqual(onset["current_proxy_reactive"]["risk_recall"], 0.)
        self.assertEqual(result["direct_recovery_only"]["raw_gate"]["risk_recall"], 1.)
        self.assertEqual(result["by_phase"]["hold"]["current_proxy_negative_future_onset"]["raw_gate"]["available"], 0)

    def test_strict_checkpoint_predictions_and_window_hash(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ModelConfig(width=16, layers=1, heads=4, dropout=0)
            manifest = make_fixture(root / "data", config, count=4)
            data = WindowDataset(manifest, "train", config)
            checkpoint = root / "risk.pt"
            torch.save({"schema": SCHEMA, "kind": "risk", "baseline_lock_sha256": sha256(LOCK),
                        "model_config": config.to_dict(), "model": RiskModel(config).state_dict(),
                        "normalizer": Normalizer.fit(data).state_dict(), "epoch": 3, "seed": 0,
                        "synthetic_inputs": True, "dataset_manifest_sha256": sha256(manifest),
                        "window_hashes": {"val": sha256(root / "data/val.npz")}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                infer_checkpoint(checkpoint, manifest)
            a, provenance = infer_checkpoint(checkpoint, manifest, allow_synthetic=True, batch_size=2)
            self.assertEqual(provenance["checkpoint_epoch"], 3)
            self.assertEqual(len(a["intervention_logit"]), 4)
            report = evaluate_arrays(a, threshold=.5, bootstrap=20)
            self.assertEqual(report["by_phase"]["stand"]["available"], 0)
            self.assertIsNone(report["auxiliary_by_body"]["torso"]["contact"]["accuracy"])
            with (root / "data/val.npz").open("ab") as handle:
                handle.write(b"changed hash")
            with self.assertRaisesRegex(ValueError, "changed"):
                infer_checkpoint(checkpoint, manifest, allow_synthetic=True)


if __name__ == "__main__":
    unittest.main()
