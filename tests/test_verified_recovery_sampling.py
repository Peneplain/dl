"""Verified recovery exposure preserves censored background and training contracts."""

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
from experiments.train import train, verified_recovery_weights
from risk_residual.checkpoints import load_checkpoint
from risk_residual.config import ModelConfig
from risk_residual.data import Normalizer, WindowDataset
from torch.utils.data import DataLoader


def weight_fixture():
    ids = np.array(["origin-a-first", "origin-a-variant", "origin-b", "negative", "proxy-positive", "censored"])
    arrays = {"correction_sample": np.array([True, True, True, False, False, False]),
              "teacher_verified": np.array([True, True, True, False, False, False]),
              "intervention_valid": np.array([True, True, True, True, True, False]),
              "intervention_target": np.array([1., 1., 1., 0., 1., np.nan]), "parent_id": ids}
    parents = {str(pid): {"split": "train", "prompt_group": "train-prompts", "scene_seed": str(i)}
               for i, pid in enumerate(ids)}
    parents["origin-a-variant"]["scene_seed"] = parents["origin-a-first"]["scene_seed"]
    return arrays, parents


class VerifiedRecoveryWeightTests(unittest.TestCase):
    def test_origin_mass_balances_variants_and_retains_all_background_including_censoring(self):
        arrays, parents = weight_fixture()
        weights, report = verified_recovery_weights(arrays, parents, .05)
        np.testing.assert_allclose(weights, [.0125, .0125, .025, .95 / 3, .95 / 3, .95 / 3], rtol=0, atol=1e-15)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertEqual((report["unique_recovery_windows"], report["unique_recovery_origins"],
                          report["unique_background_windows"], report["samples_per_epoch"]), (3, 2, 3, 6))
        self.assertEqual(report["sampling_mass"], {"recovery": .05, "background": .95})
        self.assertAlmostEqual(report["expected_draws_per_epoch"]["recovery"], .3)
        self.assertEqual(report["censored_background_windows"], 1)
        self.assertEqual(report["excluded_censored"], 0)
        self.assertEqual([row["windows"] for row in report["recovery_origins"]], [2, 1])
        self.assertAlmostEqual(report["recovery_origins"][0]["expected_draws_per_window_per_epoch"], .075)
        self.assertTrue(np.isnan(arrays["intervention_target"][-1]))  # Masked target is retained, not relabeled.

    def test_each_correction_requires_complete_recovery_proof_instead_of_silent_filtering(self):
        for key, value in (("teacher_verified", False), ("intervention_valid", False),
                           ("intervention_target", 0.), ("intervention_target", np.nan)):
            with self.subTest(key=key, value=value):
                arrays, parents = weight_fixture()
                arrays[key][0] = value
                with self.assertRaisesRegex(ValueError, "Every correction window"):
                    verified_recovery_weights(arrays, parents, .05)

    def test_invalid_masks_unavailable_origins_and_nontraining_parents_are_rejected(self):
        arrays, parents = weight_fixture()
        arrays["teacher_verified"] = arrays["teacher_verified"].astype(float)
        with self.assertRaisesRegex(ValueError, "sampling arrays"):
            verified_recovery_weights(arrays, parents, .05)
        arrays, parents = weight_fixture()
        parents["censored"]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "TRAIN parents"):
            verified_recovery_weights(arrays, parents, .05)
        arrays, parents = weight_fixture()
        parents["origin-b"].pop("scene_seed")
        with self.assertRaisesRegex(ValueError, "origin provenance"):
            verified_recovery_weights(arrays, parents, .05)
        parents["origin-b"]["scene_seed"] = None
        with self.assertRaisesRegex(ValueError, "origin provenance"):
            verified_recovery_weights(arrays, parents, .05)
        arrays, parents = weight_fixture()
        arrays["correction_sample"][:] = False
        with self.assertRaisesRegex(ValueError, "recovery and background"):
            verified_recovery_weights(arrays, parents, .05)


class VerifiedRecoveryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ModelConfig(width=16, heads=4, layers=1, dropout=0)
        self.manifest = make_fixture(self.root / "data", self.config, count=8)
        meta = json.loads(self.manifest.read_text())
        for parent in meta["parents"]:
            if parent["parent_id"] in {"train-0", "train-2"}:
                parent.update(prompt_group="same-original-prompt", scene_seed="10000")
        write_json(self.manifest, meta)
        path = self.manifest.parent / "train.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        # A censored background window retains six available future time steps
        # for the auxiliary heads while its binary intervention target is absent.
        arrays["future_valid"][-1, -2:] = False
        for key in ("track_target", "contact_target", "balance_target"):
            arrays[key][-1, -2:] = np.nan
        arrays["intervention_valid"][-1] = False
        arrays["intervention_target"][-1] = np.nan
        arrays["stable_sample"][-1] = False
        arrays["residual_valid"][-1] = False
        np.savez_compressed(path, **arrays)
        self.settings = {"model": self.config.to_dict(), "risk_options": {},
            "training": {"epochs": 2, "batch_size": 3, "learning_rate": .001,
                         "weight_decay": .01, "gradient_clip": 1., "correction_fraction": .5,
                         "alpha": 1., "beta": .1, "auxiliary_weight": 1.}}

    def run_stage(self, name, budget):
        config_path = self.root / f"{name}-config.json"
        write_json(config_path, {**self.settings, "training": {**self.settings["training"], **budget}})
        out = self.root / name
        out.mkdir()
        args = SimpleNamespace(config=config_path, data=self.manifest, stage="risk", risk=None,
                               device="cpu", threads=1, seed=7, allow_synthetic=True, interface="P", out=out)
        with redirect_stdout(io.StringIO()):
            report = train(args)
        return report, json.loads((out / "metrics.json").read_text()), out

    def test_actual_risk_loop_uses_declared_draw_budget_natural_normalizer_and_validation(self):
        before = {split: sha256(self.manifest.parent / f"{split}.npz") for split in ("train", "val")}
        loaders = []
        def record_loader(dataset, **kwargs):
            loaders.append((dataset, kwargs))
            return DataLoader(dataset, **kwargs)
        with patch("experiments.train.DataLoader", side_effect=record_loader):
            report, metrics, out = self.run_stage("recovery", {"risk_sampling": "verified_recovery",
                                                              "risk_recovery_fraction": .05})
        sampling = report["training_controls"]["risk_sampling"]
        self.assertEqual((sampling["samples_per_epoch"], sampling["unique_recovery_windows"],
                          sampling["unique_recovery_origins"], sampling["censored_background_windows"]), (8, 4, 3, 1))
        self.assertEqual([row["updates_this_epoch"] for row in metrics], [3, 3])
        self.assertEqual([row["cumulative_updates"] for row in metrics], [3, 6])
        self.assertEqual((report["actual_epochs"], report["budget_epochs"], report["updates"]), (2, 2, 6))
        self.assertTrue(all(np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"]) for row in metrics))
        training, validation = loaders
        self.assertEqual(training[1]["sampler"].num_samples, 8)
        self.assertTrue(training[1]["sampler"].replacement)
        expected_weights, _ = verified_recovery_weights(training[0].arrays, training[0].parents, .05)
        np.testing.assert_array_equal(training[1]["sampler"].weights.numpy(), expected_weights)
        self.assertGreater(training[1]["sampler"].weights[-1], 0.)
        self.assertEqual((len(validation[0]), validation[1].get("sampler"), validation[1]["shuffle"]), (8, None, False))
        _, _, checkpoint = load_checkpoint(out / "best.pt", kind="risk")
        best = min(metrics, key=lambda row: row["val_loss"])
        self.assertEqual((report["best_epoch"], checkpoint["epoch"], checkpoint["updates"]),
                         (best["epoch"], best["epoch"], best["cumulative_updates"]))
        self.assertEqual(checkpoint["training_controls"]["risk_sampling"], sampling)
        natural = Normalizer.fit(WindowDataset(self.manifest, "train", self.config)).state_dict()
        for key, tensor in natural.items():
            self.assertTrue(torch.equal(tensor, checkpoint["normalizer"][key]), key)
        self.assertEqual(before, {split: sha256(self.manifest.parent / f"{split}.npz") for split in before})

    def test_default_natural_training_matches_explicit_natural_model_tensors_exactly(self):
        default, metrics, out = self.run_stage("default", {})
        explicit, explicit_metrics, explicit_out = self.run_stage("natural", {"risk_sampling": "natural"})
        self.assertEqual(metrics, explicit_metrics)
        self.assertEqual(default["training_controls"], explicit["training_controls"])
        _, _, default_checkpoint = load_checkpoint(out / "last.pt", kind="risk")
        _, _, explicit_checkpoint = load_checkpoint(explicit_out / "last.pt", kind="risk")
        for key, tensor in default_checkpoint["model"].items():
            self.assertTrue(torch.equal(tensor, explicit_checkpoint["model"][key]), key)

    def test_invalid_fraction_mode_or_residual_stage_is_rejected_before_device_or_data_use(self):
        invalid = [("risk", {"risk_sampling": "verified_recovery", "risk_recovery_fraction": value})
                   for value in (True, False, 0, 1, -.1, 1.1, float("nan"), float("inf"), "0.05", None)]
        invalid += [("risk", {"risk_sampling": "verified_recovery"}),
                    ("risk", {"risk_sampling": "natural", "risk_recovery_fraction": .05}),
                    ("risk", {"risk_sampling": "balanced_intervention", "risk_recovery_fraction": .05}),
                    ("risk", {"risk_sampling": True}),
                    ("risk", {"risk_sampling": []}),
                    ("residual", {"risk_sampling": "verified_recovery", "risk_recovery_fraction": .05}),
                    ("residual", {"risk_recovery_fraction": .05})]
        for stage, budget in invalid:
            with self.subTest(stage=stage, budget=budget):
                config = self.root / "invalid.json"
                # Deliberately include nonstandard JSON NaN/Infinity tokens to
                # verify they are rejected even though json.loads accepts them.
                config.write_text(json.dumps({**self.settings, "training": {**self.settings["training"], **budget}}))
                with patch("experiments.train.device_for") as device, self.assertRaises(ValueError):
                    train(SimpleNamespace(config=config, stage=stage))
                device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
