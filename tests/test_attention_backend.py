"""Regression checks for the full-model MUSA dropout/SDPA descriptor failure."""

import unittest
from unittest.mock import patch

import torch

from risk_residual.models import PortableEncoderLayer


class PortableAttentionTests(unittest.TestCase):
    def test_unfused_attention_preserves_outputs_gradients_and_checkpoint_keys(self):
        torch.manual_seed(17)
        standard = torch.nn.TransformerEncoderLayer(32, 4, 128, 0., batch_first=True,
                                                    norm_first=True, activation="gelu")
        portable = PortableEncoderLayer(32, 4, 128, 0., batch_first=True,
                                        norm_first=True, activation="gelu")
        portable.load_state_dict(standard.state_dict(), strict=True)
        self.assertEqual(set(portable.state_dict()), set(standard.state_dict()))
        a = torch.randn(2, 12, 32, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[0, -2:] = True
        expected = standard(a, src_key_padding_mask=mask)
        actual = portable.forward_unfused(b, src_key_padding_mask=mask)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(b.grad, a.grad, atol=2e-6, rtol=2e-5)
        for name, parameter in standard.named_parameters():
            torch.testing.assert_close(dict(portable.named_parameters())[name].grad,
                                       parameter.grad, atol=2e-6, rtol=2e-5)

    def test_full_width_dropout_backward_never_calls_sdpa(self):
        layer = PortableEncoderLayer(256, 8, 1024, .1, batch_first=True,
                                     norm_first=True, activation="gelu")
        x = torch.randn(2, 48, 256, requires_grad=True)
        with patch("torch.nn.functional.scaled_dot_product_attention",
                   side_effect=RuntimeError("Vendor descriptor rejected")):
            result = layer.forward_unfused(x)
            result.square().mean().backward()
        self.assertEqual(layer.self_attn.dropout, .1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
