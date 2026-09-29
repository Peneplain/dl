"""Compatibility for ARDY's bidirectional Llama under Transformers 5.8.1."""

import torch
from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel


class LocalLlamaBiModel(LlamaBiModel):
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                past_key_values=None, use_cache=False, **kwargs):
        # Transformers 5 builds a causal mask directly in LlamaModel.forward;
        # ARDY's legacy _update_causal_mask override is no longer called.
        # A supplied 4D additive mask bypasses that construction. Only padding
        # keys are masked, so every real token sees the entire prompt.
        if past_key_values is not None or use_cache:
            raise ValueError("Bidirectional text encoding does not support a KV cache")
        if self.config._attn_implementation != "eager":
            raise ValueError("The baseline text encoder requires eager attention")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, length = inputs_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((batch, length), device=inputs_embeds.device)
        if attention_mask.shape != (batch, length):
            raise ValueError("Expected a 2D text padding mask")
        mask = torch.zeros((batch, 1, 1, length), dtype=inputs_embeds.dtype,
                           device=inputs_embeds.device)
        mask = mask.masked_fill(attention_mask[:, None, None, :] == 0,
                                torch.finfo(inputs_embeds.dtype).min)
        return super().forward(inputs_embeds=inputs_embeds,
                               attention_mask=mask.expand(batch, 1, length, length),
                               use_cache=False, **kwargs)
