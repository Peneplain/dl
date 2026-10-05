"""Meaningful P invariants, loss gradients, leakage rejection and frozen training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from baseline.adapters.joints import ARM_INDICES
from baseline.common import write_json
from experiments.calibrate import select_threshold
from experiments.smoke import make_fixture
from experiments.statistics import paired_bootstrap, wilson
from risk_residual.config import ModelConfig
from risk_residual.data import Normalizer, WindowDataset, audit_parents, nominal_risk_labels
from risk_residual.losses import residual_loss, risk_loss
from risk_residual.models import ResidualModel, RiskModel, freeze_risk, risk_features
from risk_residual.runtime import InferenceWindow, PredictiveController
from scripts.package_baseline import source_files


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = ModelConfig(width=32, heads=4, layers=1, dropout=0)
        self.manifest = make_fixture(Path(self.temp.name) / 'data', self.config)
        self.data = WindowDataset(self.manifest, 'train', self.config)
        self.batch = next(iter(torch.utils.data.DataLoader(self.data, batch_size=8)))
        self.normalizer = Normalizer.fit(self.data)
        self.inputs = self.normalizer(self.batch)

    def test_default_proposal_network_forward_backward_and_finite_gradients(self):
        # Exercise the actual 256x4/8 architecture separately from the tiny fixture trainer.
        risk = RiskModel()
        output = risk(**{key: value[:1] for key, value in self.inputs.items()})
        self.assertEqual(tuple(output.tokens.shape), (1, 8, 4, 32))
        batch = {key: value[:1] for key, value in self.batch.items()}
        loss, _ = risk_loss(output, batch)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in risk.parameters() if p.grad is not None))
        residual = ResidualModel()
        prediction = residual(**{key: value[:1] for key, value in self.inputs.items()},
                              features=output.tokens.detach())
        residual_loss(prediction, batch)[0].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in residual.parameters()))

    def test_matching_feature_slots_and_arm_only_bounds(self):
        risk, residual = RiskModel(self.config), ResidualModel(self.config)
        output = risk(**self.inputs)
        for interface in ('B1', 'B2', 'I1', 'I2', 'I3', 'I4', 'P', 'pooled'):
            self.assertEqual(risk_features(output, interface).shape, output.tokens.shape)
        with torch.no_grad():
            residual.offset.bias.fill_(100)
        correction = residual(**self.inputs, features=output.tokens)
        nonarms = [j for j in range(29) if j not in ARM_INDICES]
        self.assertEqual(correction[..., nonarms].abs().sum().item(), 0)
        self.assertLessEqual(correction.abs().max().item(), .150001)

    def test_masked_nan_targets_do_not_poison_gradients(self):
        risk = RiskModel(self.config)
        self.batch['future_valid'][:, -2:] = False
        self.batch['contact_mask'].zero_()
        for key in ('track_target', 'balance_target'):
            self.batch[key][:, -2:] = torch.nan
        self.batch['contact_target'].fill_(torch.nan)
        output = risk(**self.inputs)
        loss, parts = risk_loss(output, self.batch)
        self.assertEqual(parts['contact'].item(), 0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in risk.parameters() if p.grad is not None))

    def test_frozen_risk_and_disjoint_identity_losses(self):
        risk, residual = freeze_risk(RiskModel(self.config)), ResidualModel(self.config)
        with torch.no_grad():
            output = risk(**self.inputs)
        prediction = residual(**self.inputs, features=output.tokens)
        loss, parts = residual_loss(prediction, self.batch)
        self.assertGreater(parts['correction'].item(), 0)
        self.assertEqual(parts['identity'].item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in risk.parameters()))
        self.batch['stable_sample'].fill_(True)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            residual_loss(prediction, self.batch)

    def test_low_risk_skips_residual_inference(self):
        risk, residual = RiskModel(self.config), ResidualModel(self.config)
        with torch.no_grad():
            risk.intervention.weight.zero_()
            risk.intervention.bias.fill_(-100)
        controller = PredictiveController(risk, residual, self.normalizer, threshold=.5)
        a = self.data.arrays
        window = InferenceWindow(float(a['decision_time'][0]), a['history_times'][0], a['history'][0],
                                 a['nominal_times'][0], a['nominal'][0], a['context'][0])
        with patch.object(residual, 'forward', side_effect=AssertionError('must be skipped')):
            desired, report = controller.predict(window)
        self.assertFalse(report['active'])
        np.testing.assert_array_equal(desired, 0)
        window.history_times = window.history_times + .02
        with self.assertRaisesRegex(ValueError, 'clocks'):
            controller.predict(window)

    def test_manifest_rejects_branch_and_scene_leakage(self):
        parents = [{'parent_id': 'a', 'split': 'train', 'prompt_group': 'x', 'scene_seed': 1},
                   {'parent_id': 'b', 'split': 'test', 'prompt_group': 'y', 'scene_seed': 1}]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            audit_parents(parents)
        meta = json.loads(self.manifest.read_text())
        meta['parents'][0]['split'] = 'test'
        write_json(self.manifest, meta)
        with self.assertRaisesRegex(ValueError, 'parent'):
            WindowDataset(self.manifest, 'train', self.config)

    def test_teacher_and_nominal_counterfactual_provenance_required(self):
        path = self.manifest.parent / 'train.npz'
        a = self.data.arrays.copy()
        a['teacher_snapshot'] = np.array(['wrong'] * len(self.data))
        np.savez(path, **a)
        with self.assertRaisesRegex(ValueError, 'SAME snapshot'):
            WindowDataset(self.manifest, 'train', self.config)
        a['teacher_snapshot'] = a['nominal_snapshot']
        a['risk_source'] = np.array(['corrected'] * len(self.data))
        np.savez(path, **a)
        with self.assertRaisesRegex(ValueError, 'nominal branch'):
            WindowDataset(self.manifest, 'train', self.config)

    def test_censored_negative_and_open_hand_labels(self):
        valid = np.ones((8, 4), bool)
        valid[-1] = False
        labels = nominal_risk_labels(np.zeros((8, 4)), np.ones((8, 4)), np.zeros((8, 4)),
                                     valid, np.zeros((8, 4), bool), np.ones(4))
        self.assertFalse(labels['intervention_valid'])
        self.assertEqual(labels['intervention_target'], 0)
        error = np.zeros((8, 4))
        error[1, 0] = 2
        labels = nominal_risk_labels(error, np.zeros((8, 4)), np.zeros((8, 4)),
                                     valid, np.zeros((8, 4), bool), np.ones(4))
        self.assertTrue(labels['intervention_valid'])
        self.assertEqual(labels['intervention_target'], 1)

    def test_verified_recovery_is_positive_without_auxiliary_threshold_violation(self):
        path = self.manifest.parent / 'train.npz'
        arrays = {key: value.copy() for key, value in self.data.arrays.items()}
        index = int(np.flatnonzero(arrays['correction_sample'])[0])
        for key in ('track_target', 'contact_target', 'balance_target'):
            arrays[key][index] = 0
        np.savez(path, **arrays)
        WindowDataset(self.manifest, 'train', self.config)
        arrays['intervention_target'][index] = 0
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, 'verified recovery'):
            WindowDataset(self.manifest, 'train', self.config)

    def test_baseline_archive_excludes_new_learning_modules_and_tests(self):
        root = Path(__file__).resolve().parents[1]
        files = {str(path.relative_to(root)) for path in source_files(root)}
        self.assertNotIn('tests/test_risk_residual.py', files)
        self.assertFalse(any(p.startswith(('risk_residual/', 'experiments/', 'configs/learning/')) for p in files))

    def test_statistics_pair_episode_ids_and_threshold_validation(self):
        low, high = wilson(0, 20)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0)
        result = paired_bootstrap({'a': 0, 'b': 0}, {'a': 1, 'b': 1}, draws=50)
        self.assertEqual(result['ci95'], [1, 1])
        with self.assertRaises(ValueError):
            paired_bootstrap({'a': 1}, {'b': 1})
        self.assertGreater(select_threshold([.1, .9], [0, 1])['validation_f1'], .99)
        with self.assertRaises(ValueError):
            select_threshold([.1, .2], [0, 0])


if __name__ == '__main__':
    unittest.main()
