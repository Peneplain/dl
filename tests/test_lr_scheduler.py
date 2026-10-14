"""Validation-loss plateau scheduling stays optional and auditable."""

import unittest

import numpy as np
import torch

from experiments.train import EarlyStopping, validation_scheduler_options


class LearningRateSchedulerTests(unittest.TestCase):
    def test_missing_scheduler_preserves_fixed_learning_rate(self):
        early_stop = EarlyStopping()
        self.assertIsNone(validation_scheduler_options(None, 1e-4, early_stop))
        self.assertIsNone(validation_scheduler_options({}, 1e-4, early_stop))

    def test_default_plateau_options_are_explicit_and_match_checkpoint_metric(self):
        early_stop = EarlyStopping({"patience": 8, "monitor": "val_loss"})
        options = validation_scheduler_options({"type": "reduce_on_plateau"}, 1e-4, early_stop)
        self.assertEqual(options, {
            "mode": "min", "factor": .5, "patience": 2,
            "threshold": 0., "threshold_mode": "abs", "min_lr": 1e-6,
        })

    def test_plateau_scheduler_reduces_only_after_validation_plateau(self):
        early_stop = EarlyStopping({"patience": 8, "monitor": "val_loss"})
        options = validation_scheduler_options(
            {"factor": .5, "patience": 0, "min_lr": 1e-6}, 1e-3, early_stop)
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)
        scheduler.step(0.9)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)

    def test_invalid_scheduler_or_mismatched_stopping_monitor_is_rejected(self):
        early_stop = EarlyStopping({"patience": 8, "monitor": "val_loss"})
        invalid = [
            {"factor": 0}, {"factor": 1}, {"factor": True},
            {"patience": -1}, {"min_lr": 1e-4},
            {"monitor": "val_intervention"}, {"unexpected": 1},
        ]
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    validation_scheduler_options(settings, 1e-4, early_stop)

        with self.assertRaises(ValueError):
            validation_scheduler_options(
                {"monitor": "val_loss"}, 1e-4,
                EarlyStopping({"patience": 8, "monitor": "val_intervention"}))

    def test_scheduler_inputs_require_finite_numeric_values(self):
        early_stop = EarlyStopping()
        for settings in ({"factor": np.nan}, {"min_lr": np.inf}, {"patience": 1.5}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    validation_scheduler_options(settings, 1e-4, early_stop)


if __name__ == "__main__":
    unittest.main()
