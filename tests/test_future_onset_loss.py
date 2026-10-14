import unittest
from types import SimpleNamespace

import torch

from risk_residual.losses import future_onset_targets, risk_loss


class FutureOnsetLossTests(unittest.TestCase):
    def make_batch(self):
        n, h, b = 5, 8, 4
        valid = torch.ones(n, h, b, dtype=torch.bool)
        contact_mask = torch.zeros_like(valid)
        track = torch.zeros(n, h, b)
        contact = torch.zeros_like(track)
        balance = torch.zeros_like(track)
        correction = torch.zeros(n, dtype=torch.bool)

        # Current proxy positive: excluded from future-onset supervision.
        track[0, 0, 0] = 1.0
        # Future onset with complete future: valid positive.
        track[1, 1, 0] = 1.0
        # Future onset with a censored future: still valid because a positive is observed.
        track[2, 1, 0] = 1.0
        valid[2, 3:] = False
        # Future negative with incomplete future: censored, so invalid.
        valid[3, 3:] = False
        # Verified recovery is excluded even when its nominal future is positive.
        correction[4] = True
        track[4, 1, 0] = 1.0
        return {
            "future_valid": valid,
            "contact_mask": contact_mask,
            "track_target": track,
            "contact_target": contact,
            "balance_target": balance,
            "correction_sample": correction,
            "intervention_target": torch.tensor([1., 1., 1., 0., 1.]),
            "intervention_valid": torch.ones(n, dtype=torch.bool),
        }

    def output(self, n=5):
        shape = (n, 8, 4)
        return SimpleNamespace(
            tracking=torch.zeros(shape, requires_grad=True),
            contact_logits=torch.zeros(shape, requires_grad=True),
            balance_logits=torch.zeros(shape, requires_grad=True),
            intervention_logit=torch.zeros(n, requires_grad=True),
        )

    def test_targets_exclude_current_recovery_and_censored_negative(self):
        target, valid = future_onset_targets(self.make_batch())
        self.assertEqual(valid.tolist(), [False, True, True, False, False])
        self.assertEqual(target.tolist(), [False, True, True, False, True])

    def test_default_weight_preserves_existing_total_and_new_term_has_gradient(self):
        batch = self.make_batch()
        output = self.output()
        total, parts = risk_loss(output, batch, auxiliary_weight=1.0)
        expected = parts["intervention"] + sum(parts[key] for key in ("tracking", "contact", "balance"))
        self.assertTrue(torch.allclose(total, expected))
        self.assertGreater(parts["future_onset"].item(), 0.)
        weighted, _ = risk_loss(output, batch, auxiliary_weight=1.0, future_onset_weight=1.0)
        self.assertGreater((weighted - total).item(), 0.)
        weighted.backward()
        self.assertTrue(torch.isfinite(output.intervention_logit.grad).all())
        self.assertGreater(output.intervention_logit.grad.abs().sum().item(), 0.)

    def test_masked_nan_targets_do_not_poison_future_onset_gradients(self):
        batch = self.make_batch()
        batch["track_target"][3, 3:] = torch.nan
        batch["balance_target"][3, 3:] = torch.nan
        batch["contact_target"].fill_(torch.nan)
        output = self.output()
        loss, _ = risk_loss(output, batch, future_onset_weight=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(output.intervention_logit.grad).all())


if __name__ == "__main__":
    unittest.main()
