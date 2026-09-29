"""Tiny offline checkpoints exercise the real loading stack; no 8B weights needed."""

import importlib.util
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.common import LOCK, ROOT, checked_checkout


class TextEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("torch", "transformers", "peft", "hydra", "vector_quantize_pytorch"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"Install requirements-baseline.txt for text integration tests ({name})")
        source = ROOT / "third_party/ardy"
        if not source.is_dir():
            raise unittest.SkipTest("Pinned ARDY source is needed for text integration tests")
        from baseline.text_encoder import check_transformers_version
        check_transformers_version()
        checked_checkout(source, json.loads(LOCK.read_text())["ardy"]["commit"])
        sys.path.insert(0, str(source))

    def tiny_config(self):
        from transformers import LlamaConfig
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, pad_token_id=0)
        config._attn_implementation = "eager"
        return config

    def test_bidirectional_attention_and_padding(self):
        import torch
        from baseline.llama import LocalLlamaBiModel
        torch.manual_seed(7)
        model = LocalLlamaBiModel(self.tiny_config()).eval()
        with torch.no_grad():
            a = model(torch.tensor([[1, 2, 3]])).last_hidden_state
            b = model(torch.tensor([[1, 2, 4]])).last_hidden_state
            self.assertGreater((a[:, 0] - b[:, 0]).abs().max().item(), 1e-5)
            mask = torch.tensor([[0, 1, 1]])
            a = model(torch.tensor([[3, 1, 2]]), attention_mask=mask).last_hidden_state
            b = model(torch.tensor([[4, 1, 2]]), attention_mask=mask).last_hidden_state
            torch.testing.assert_close(a[:, 1:], b[:, 1:])

    def test_loads_both_adapters_offline_and_preserves_prompt_pooling(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
        from baseline.llama import LocalLlamaBiModel
        from baseline.text_encoder import LocalTextEncoder

        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.tiny_config()
            backbone = LlamaForCausalLM(config)
            backbone.save_pretrained(root / "llama_base")
            base_q = backbone.model.layers[0].self_attn.q_proj.weight.detach().clone()
            for name, value in (("text_base", 0.03), ("text_adapter", 0.07)):
                adapted = get_peft_model(LocalLlamaBiModel(config), LoraConfig(
                    r=2, lora_alpha=2, target_modules=["q_proj"], lora_dropout=0.0))
                for param_name, parameter in adapted.named_parameters():
                    if "lora_" in param_name:
                        torch.nn.init.constant_(parameter, value)
                adapted.save_pretrained(root / name)
            config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            config.save_pretrained(root / "text_base")
            # Match the published MNTP config; v5 omits this field on save.
            config_path = root / "text_base/config.json"
            saved = json.loads(config_path.read_text())
            saved["_name_or_path"] = config._name_or_path
            config_path.write_text(json.dumps(saved))
            tokenizer = Tokenizer(WordLevel({"[EOS]": 0, "[UNK]": 1, "A": 2,
                                              "person": 3, "stands": 4, "still": 5}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="[EOS]",
                                    unk_token="[UNK]").save_pretrained(root / "text_base")
            with patch.object(socket.socket, "connect", side_effect=AssertionError("Network access during offline load")):
                encoder = LocalTextEncoder(root)
                q_layer = encoder.model.model.base_model.model.layers[0].self_attn.q_proj
                torch.testing.assert_close(q_layer.base_layer.weight, base_q + 2 * 0.03**2)
                torch.testing.assert_close(q_layer.get_delta_weight("default"),
                                            torch.full_like(base_q, 2 * 0.07**2))
                self.assertFalse(any(p.requires_grad for p in encoder.model.parameters()))
                prepared = encoder.model.prepare_for_tokenization("!@#$%^&*()A person stands still.")
                self.assertIn("<|start_header_id|>user<|end_header_id|>", prepared)
                features, lengths = encoder(["A person stands still."])
                expected = encoder.model.encode(["A person stands still."], device="cpu",
                                                batch_size=1, show_progress_bar=False)
                torch.testing.assert_close(features[:, 0], expected)
                self.assertEqual(features.shape, (1, 1, 16))
                self.assertEqual(lengths, [1])
                self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
