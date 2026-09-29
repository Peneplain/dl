"""Local-only loading of the frozen Llama + MNTP + supervised LLM2Vec stack."""

import json
from importlib.metadata import version
from pathlib import Path


class LocalTextEncoder:
    """Use ARDY's tokenizer/pooling with explicit local paths for all three models.

    ``text_base`` is the historical directory name for the MNTP adapter, not
    the 8B backbone. Loading it through Transformers' adapter auto-discovery
    follows a remote base_model_name_or_path and breaks offline installations.
    The caller verifies the source checkout and asset manifests before loading.
    """

    def __init__(self, assets, *, dtype="float32", device="cpu"):
        check_transformers_version()
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer
        from ardy.model.llm2vec.llm2vec import LLM2Vec
        from baseline.llama import LocalLlamaBiModel

        assets = Path(assets).resolve()
        mntp = assets / "text_base"
        supervised = assets / "text_adapter"
        config = AutoConfig.from_pretrained(str(mntp), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(mntp), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model, loading = LocalLlamaBiModel.from_pretrained(
            str(assets / "llama_base"), config=config, local_files_only=True,
            dtype=getattr(torch, dtype), attn_implementation="eager", output_loading_info=True)
        if (loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs")
                or set(loading.get("unexpected_keys", [])) - {"lm_head.weight"}):
            raise ValueError(f"Incomplete or incompatible Llama backbone: {loading}")
        # Preserve the upstream prompt template, which dispatches by this name.
        model.config._name_or_path = json.loads((mntp / "config.json").read_text())["_name_or_path"]
        model = PeftModel.from_pretrained(model, str(mntp), local_files_only=True,
                                          is_trainable=False)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, str(supervised), local_files_only=True,
                                          is_trainable=False)
        options_path = supervised / "llm2vec_config.json"
        options = json.loads(options_path.read_text()) if options_path.exists() else {}
        self.model = LLM2Vec(model, tokenizer, **options).to(device)
        self.model.eval().requires_grad_(False)
        self.device = str(device)

    def __call__(self, texts):
        import torch

        # The upstream encode() dispatcher chooses CUDA multiprocessing whenever
        # >1 CUDA device exists, even if the caller explicitly requested CPU.
        # Its pinned single-batch implementation retains the same prompt/pooling.
        with torch.no_grad():
            batches = [self.model._encode([self.model._convert_to_str("", text)],
                                          device=self.device) for text in texts]
        return torch.cat(batches).float()[:, None], [1] * len(texts)


def check_transformers_version():
    installed = version("transformers")
    if installed != "5.8.1":
        raise RuntimeError(f"Expected transformers==5.8.1, got {installed}; reinstall requirements-baseline.txt")
