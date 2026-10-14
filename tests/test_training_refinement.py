"""Residual draw budgets and truthful training provenance in the real CPU loop."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baseline.common import sha256, write_json
from experiments.smoke import make_fixture
from experiments.train import train
from risk_residual.checkpoints import load_checkpoint
from risk_residual.config import ModelConfig


class TrainingRefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ModelConfig(width=16, heads=4, layers=1, dropout=0)
        self.manifest = make_fixture(self.root / "data", self.config, count=6)
        # Keep one available Risk window outside both residual supervision sets.
        # The default residual budget must follow selected windows, not all six.
        for split in ("train", "val"):
            path = self.manifest.parent / f"{split}.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["stable_sample"][-1] = False
            arrays["residual_valid"][-1] = False
            np.savez_compressed(path, **arrays)
        self.settings = {"model": self.config.to_dict(), "risk_options": {},
            "training": {"epochs": 3, "batch_size": 4, "learning_rate": .001,
                         "weight_decay": .01, "gradient_clip": 1., "correction_fraction": .5,
                         "alpha": 1., "beta": .1, "auxiliary_weight": 1.}}

    def run_stage(self, name, stage, budget=None, risk=None):
        settings = {**self.settings, "training": {**self.settings["training"], **(budget or {})}}
        config = self.root / f"{name}-config.json"
        write_json(config, settings)
        out = self.root / name
        out.mkdir()
        args = SimpleNamespace(config=config, data=self.manifest, stage=stage, risk=risk,
                               device="cpu", threads=1, seed=7, allow_synthetic=True, interface="P", out=out)
        with redirect_stdout(io.StringIO()):
            report = train(args)
        metrics = json.loads((out / "metrics.json").read_text())
        return report, metrics, out

    def frozen_risk(self):
        report, _, out = self.run_stage("risk", "risk", {"epochs": 1})
        self.assertEqual(report["training_controls"]["risk_sampling"]["samples_per_epoch"], 6)
        return out / "best.pt"

    def test_default_selected_budget_matches_explicit_legacy_budget_and_loads_checkpoint(self):
        risk = self.frozen_risk()
        risk_sha = sha256(risk)
        default, metrics, out = self.run_stage("default", "residual", risk=risk)
        explicit, _, explicit_out = self.run_stage("explicit-legacy", "residual",
                                                  {"residual_samples_per_epoch": 5}, risk)
        for report in (default, explicit):
            sampling = report["training_controls"]["residual_sampling"]
            self.assertEqual(report["training_controls"]["risk_sampling"]["mode"], "not_applied")
            self.assertEqual((sampling["unique_correction_samples"], sampling["unique_stable_samples"],
                              sampling["unique_selected_samples"], sampling["samples_per_epoch"]), (3, 2, 5, 5))
            self.assertTrue(sampling["replacement"])
            self.assertEqual(report["updates"], 6)
        self.assertEqual(default["training_controls"]["residual_sampling"]["budget_source"], "selected_count_default")
        self.assertEqual([row["updates_this_epoch"] for row in metrics], [2, 2, 2])
        self.assertEqual([row["cumulative_updates"] for row in metrics], [2, 4, 6])
        _, _, checkpoint = load_checkpoint(out / "last.pt", kind="residual")
        _, _, explicit_checkpoint = load_checkpoint(explicit_out / "last.pt", kind="residual")
        for key, tensor in checkpoint["model"].items():
            self.assertTrue(torch.equal(tensor, explicit_checkpoint["model"][key]), key)
        self.assertEqual(checkpoint["updates"], 6)
        self.assertEqual(checkpoint["risk_sha256"], risk_sha)
        self.assertEqual(sha256(risk), risk_sha)

    def test_optional_budget_controls_optimizer_steps_and_reports_expected_exposure(self):
        risk = self.frozen_risk()
        report, metrics, out = self.run_stage("bounded", "residual",
                                            {"residual_samples_per_epoch": 11}, risk)
        sampling = report["training_controls"]["residual_sampling"]
        self.assertEqual(sampling["samples_per_epoch"], 11)
        self.assertEqual(sampling["sampling_mass"], {"correction": .5, "stable": .5})
        self.assertEqual(sampling["expected_draws_per_epoch"], {"correction": 5.5, "stable": 5.5})
        self.assertAlmostEqual(sampling["expected_draws_per_unique_sample_per_epoch"]["correction"], 5.5 / 3)
        self.assertAlmostEqual(sampling["expected_draws_per_unique_sample_per_epoch"]["stable"], 5.5 / 2)
        self.assertEqual([row["updates_this_epoch"] for row in metrics], [3, 3, 3])
        self.assertEqual([row["cumulative_updates"] for row in metrics], [3, 6, 9])
        self.assertEqual([row["learning_rate"] for row in metrics], [.001] * 3)
        self.assertEqual(report["updates"], 9)
        _, _, checkpoint = load_checkpoint(out / "last.pt", kind="residual")
        self.assertEqual(checkpoint["training_controls"]["residual_sampling"], sampling)
        self.assertEqual(checkpoint["updates"], 9)
        self.assertEqual(checkpoint["window_hashes"]["train"], sha256(self.manifest.parent / "train.npz"))

    def test_bounded_budget_composes_with_existing_patience_and_exact_best_selection(self):
        risk = self.frozen_risk()
        report, metrics, out = self.run_stage("stopped", "residual",
            {"epochs": 5, "residual_samples_per_epoch": 11,
             "early_stopping": {"patience": 1, "min_delta": 100.}}, risk)
        self.assertEqual((report["actual_epochs"], report["budget_epochs"], report["updates"]), (2, 5, 6))
        self.assertTrue(report["stopped_early"])
        best = min(metrics, key=lambda row: row["val_loss"])
        self.assertEqual(report["best_epoch"], best["epoch"])
        _, _, checkpoint = load_checkpoint(out / "best.pt", kind="residual")
        self.assertEqual(checkpoint["epoch"], best["epoch"])
        self.assertEqual(checkpoint["updates"], best["cumulative_updates"])

    def test_invalid_or_risk_stage_budget_is_rejected_before_device_or_data_use(self):
        config = self.root / "invalid-config.json"
        cases = [("residual", value) for value in (0, -1, True, 1., "8", None)] + [("risk", 4)]
        for stage, samples in cases:
            with self.subTest(stage=stage, samples=samples):
                write_json(config, {**self.settings, "training": {**self.settings["training"],
                                                                   "residual_samples_per_epoch": samples}})
                args = SimpleNamespace(config=config, stage=stage)
                with patch("experiments.train.device_for") as device, \
                        self.assertRaisesRegex(ValueError, "positive integer for residual training only"):
                    train(args)
                device.assert_not_called()


class FutureOnsetLossTests(unittest.TestCase):
    def test_future_onset_masks_current_positive_recovery_and_censored_negative(self):
        import torch
        from risk_residual.losses import future_onset_targets
        n, h, b = 4, 8, 4
        batch = {
            'future_valid': torch.ones(n, h, b, dtype=torch.bool),
            'contact_mask': torch.zeros(n, h, b, dtype=torch.bool),
            'track_target': torch.zeros(n, h, b),
            'balance_target': torch.zeros(n, h, b),
            'contact_target': torch.zeros(n, h, b),
            'correction_sample': torch.zeros(n, dtype=torch.bool),
            'intervention_target': torch.zeros(n),
        }
        batch['track_target'][0, 0, 0] = 1.0  # current violation: excluded
        batch['track_target'][1, 2, 1] = 1.0  # future onset: included positive
        batch['correction_sample'][2] = True  # verified recovery: excluded
        batch['future_valid'][3, 4:] = False  # censored future negative: excluded
        target, mask = future_onset_targets(batch)
        self.assertEqual(mask.tolist(), [False, True, False, False])
        self.assertEqual(target[1].item(), 1.0)

    def test_future_onset_loss_has_finite_nonzero_gradient_and_zero_weight_is_legacy(self):
        import torch
        from risk_residual.losses import risk_loss
        from types import SimpleNamespace
        n, h, b = 2, 8, 4
        batch = {
            'future_valid': torch.ones(n, h, b, dtype=torch.bool),
            'contact_mask': torch.zeros(n, h, b, dtype=torch.bool),
            'track_target': torch.zeros(n, h, b),
            'balance_target': torch.zeros(n, h, b),
            'contact_target': torch.zeros(n, h, b),
            'intervention_valid': torch.ones(n, dtype=torch.bool),
            'intervention_target': torch.zeros(n),
            'correction_sample': torch.zeros(n, dtype=torch.bool),
        }
        batch['track_target'][1, 2, 1] = 1.0
        logit = torch.zeros(n, requires_grad=True)
        output = SimpleNamespace(
            tracking=torch.zeros(n, h, b, requires_grad=True),
            contact_logits=torch.zeros(n, h, b, requires_grad=True),
            balance_logits=torch.zeros(n, h, b, requires_grad=True),
            intervention_logit=logit,
        )
        base, _ = risk_loss(output, batch, future_onset_weight=0.0)
        refined, parts = risk_loss(output, batch, future_onset_weight=1.0)
        self.assertNotEqual(refined.item(), base.item())
        self.assertGreater(parts['future_onset'].item(), 0.0)
        refined.backward()
        self.assertTrue(torch.isfinite(logit.grad).all())
        self.assertNotEqual(logit.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
