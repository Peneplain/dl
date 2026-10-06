"""Optional patience/sampling preserve natural validation and default budgets."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.smoke import make_fixture
from experiments.train import EarlyStopping, balanced_intervention_weights, train
from risk_residual.config import ModelConfig
from risk_residual.checkpoints import load_checkpoint
from baseline.common import write_json


class TrainingControlTests(unittest.TestCase):
    def test_patience_is_disabled_by_default_and_delta_only_affects_patience(self):
        disabled = EarlyStopping()
        self.assertFalse(disabled.update({"val_loss": 1}))
        self.assertFalse(disabled.update({"val_loss": 2}))
        stop = EarlyStopping({"patience": 2, "min_delta": .01})
        self.assertFalse(stop.update({"val_loss": 1.}))
        self.assertFalse(stop.update({"val_loss": .995}))
        self.assertTrue(stop.update({"val_loss": .994}))
        self.assertEqual(stop.best, 1.)
        stop = EarlyStopping({"patience": 1, "monitor": "val_intervention"})
        self.assertFalse(stop.update({"val_loss": 0., "val_components": {"intervention": 1.}}))
        self.assertTrue(stop.update({"val_loss": -.1, "val_components": {"intervention": 1.1}}))
        with self.assertRaises(ValueError):
            EarlyStopping({"patience": 0})

    def test_balancing_uses_both_train_classes_and_excludes_censoring(self):
        arrays = {"intervention_target": np.array([1., 1., 1., 0., np.nan]),
                  "intervention_valid": np.array([True, True, True, True, False])}
        weights, report = balanced_intervention_weights(arrays)
        self.assertAlmostEqual(weights[:3].sum(), .5)
        self.assertEqual(weights[3], .5)
        self.assertEqual(weights[4], 0.)
        self.assertEqual(report["samples_per_epoch"], 5)
        self.assertEqual(report["excluded_censored"], 1)
        with self.assertRaises(ValueError):
            balanced_intervention_weights({"intervention_valid": np.ones(3, bool),
                                            "intervention_target": np.ones(3)})

    def test_real_training_loop_saves_actual_and_planned_budget_with_loadable_checkpoints(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ModelConfig(width=16, heads=4, layers=1, dropout=0)
            manifest = make_fixture(root / "data", config, count=4)
            settings = {"model": config.to_dict(), "risk_options": {},
                        "training": {"epochs": 5, "batch_size": 4, "learning_rate": .001,
                                     "weight_decay": .01, "gradient_clip": 1., "correction_fraction": .5,
                                     "alpha": 1., "beta": .1, "auxiliary_weight": 1.,
                                     "early_stopping": {"patience": 1, "min_delta": 100.}}}
            path = root / "config.json"
            write_json(path, settings)
            output = root / "risk"
            output.mkdir()
            args = SimpleNamespace(config=path, data=manifest, stage="risk", risk=None, device="cpu",
                                   threads=1, seed=7, allow_synthetic=True, interface="P", out=output)
            report = train(args)
            self.assertEqual(report["actual_epochs"], 2)
            self.assertEqual(report["budget_epochs"], 5)
            self.assertTrue(report["stopped_early"])
            self.assertEqual(report["updates"], 2)
            _, _, checkpoint = load_checkpoint(output / "last.pt", kind="risk")
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(len(json.loads((output / "metrics.json").read_text())), 2)
            self.assertEqual(checkpoint["training_controls"]["checkpoint_selection"],
                             "exact minimum total validation loss")
            settings["training"].pop("early_stopping")
            settings["training"]["epochs"] = 3
            settings["training"]["risk_sampling"] = "balanced_intervention"
            write_json(path, settings)
            args.out = root / "balanced"
            args.out.mkdir()
            report = train(args)
            self.assertEqual(report["epochs"], 3)
            self.assertFalse(report["stopped_early"])
            self.assertEqual(report["training_controls"]["risk_sampling"]["validation_distribution"],
                             "unchanged natural validation windows")


if __name__ == "__main__":
    unittest.main()
