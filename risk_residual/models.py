"""Proposal Risk/Residual Transformers with matched body/time conditioning slots."""

from dataclasses import dataclass

import torch
from torch import nn

from baseline.adapters.joints import ARM_INDICES
from risk_residual.config import CONTEXT_DIM, NOMINAL_DIM, STATE_DIM, ModelConfig


@dataclass
class RiskOutput:
    tokens: torch.Tensor               # [N,8,4,32]
    tracking: torch.Tensor             # nonnegative, threshold-normalized error
    contact_logits: torch.Tensor       # [N,8,4], supervised only for closed hands
    balance_logits: torch.Tensor       # [N,8,4]
    intervention_logit: torch.Tensor   # [N]

    @property
    def probability(self):
        return self.intervention_logit.sigmoid()

    def auxiliary(self):
        return torch.stack((self.tracking, self.contact_logits.sigmoid(),
                            self.balance_logits.sigmoid()), dim=-1)


class TemporalBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_projection = nn.Linear(STATE_DIM, config.width)
        self.nominal_projection = nn.Linear(NOMINAL_DIM + CONTEXT_DIM, config.width)
        self.history_position = nn.Parameter(torch.randn(config.history_steps, config.width) * .02)
        self.time_position = nn.Parameter(torch.randn(config.horizon, config.width) * .02)
        self.body_position = nn.Parameter(torch.randn(config.body_groups, config.width) * .02)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, 4 * config.width,
                                            config.dropout, activation="gelu", batch_first=True,
                                            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers,
                                              norm=nn.LayerNorm(config.width),
                                              enable_nested_tensor=False)

    def forward(self, history, nominal, context, conditioning=None):
        c = self.config
        n = history.shape[0]
        for tensor, shape in ((history, (n, c.history_steps, STATE_DIM)),
                              (nominal, (n, c.horizon, NOMINAL_DIM)),
                              (context, (n, c.horizon, CONTEXT_DIM))):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Input shape {tuple(tensor.shape)} != {shape}")
        past = self.state_projection(history) + self.history_position
        future = self.nominal_projection(torch.cat((nominal, context), dim=-1))
        future = (future[:, :, None] + self.time_position[None, :, None]
                  + self.body_position[None, None])
        if conditioning is not None:
            future = future + conditioning
        # Bidirectional attention sees past execution and FUTURE NOMINAL references only.
        encoded = self.encoder(torch.cat((past, future.flatten(1, 2)), dim=1))
        return encoded[:, c.history_steps:].reshape(n, c.horizon, c.body_groups, c.width)


class RiskModel(nn.Module):
    def __init__(self, config=None, *, no_future=False, no_history=False):
        super().__init__()
        self.config = config or ModelConfig()
        self.no_future, self.no_history = no_future, no_history
        self.backbone = TemporalBackbone(self.config)
        self.tokens = nn.Linear(self.config.width, self.config.token_dim)
        self.track = nn.Sequential(nn.Linear(self.config.token_dim, 1), nn.Softplus())
        self.contact = nn.Linear(self.config.token_dim, 1)
        self.balance = nn.Linear(self.config.token_dim, 1)
        self.intervention = nn.Linear(self.config.token_dim, 1)

    def forward(self, history, nominal, context):
        if self.no_history:
            history = history[:, -1:].expand_as(history)
        if self.no_future:
            nominal = torch.zeros_like(nominal)
        z = self.tokens(self.backbone(history, nominal, context))
        return RiskOutput(z, self.track(z).squeeze(-1), self.contact(z).squeeze(-1),
                          self.balance(z).squeeze(-1),
                          self.intervention(z.mean((1, 2))).squeeze(-1))


def risk_features(output, interface):
    """Parameter-free adapters preserve identical 8x4x32 slots and decoder capacity."""
    z = output.tokens
    if interface in {"B1", "B2", "I1"}:
        return torch.zeros_like(z)
    if interface == "I2":
        return output.probability[:, None, None, None].expand_as(z)
    if interface in {"I3", "I4"}:
        auxiliary = output.auxiliary()
        if interface == "I3":
            auxiliary = auxiliary.mean(1, keepdim=True).expand_as(auxiliary)
        # Repeat the three threshold-normalized auxiliary channels, with no
        # trainable adapter that would change comparison capacity.
        return auxiliary.repeat(1, 1, 1, (z.shape[-1] + 2) // 3)[..., :z.shape[-1]]
    if interface == "P":
        return z
    if interface == "pooled":
        return z.mean((1, 2), keepdim=True).expand_as(z)
    raise ValueError(f"Unknown risk interface: {interface}")


class ResidualModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = TemporalBackbone(self.config)
        self.risk_projection = nn.Linear(self.config.token_dim, self.config.width, bias=False)
        self.offset = nn.Linear(self.config.width * self.config.body_groups, 14)
        # Start at the identity reference while allowing the head to learn.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer("arm_indices", torch.tensor(ARM_INDICES))

    def forward(self, history, nominal, context, features):
        c = self.config
        if tuple(features.shape) != (history.shape[0], c.horizon, c.body_groups, c.token_dim):
            raise ValueError("Risk feature slots must be [N,8,4,32]")
        hidden = self.backbone(history, nominal, context, self.risk_projection(features))
        raw = self.offset(hidden.flatten(2))
        bounded = raw.clamp(-c.max_offset, c.max_offset)
        # Out-of-place scatter has gradients for the 14 arm outputs only.
        full = bounded.new_zeros((*bounded.shape[:-1], 29))
        full = full.index_copy(-1, self.arm_indices, bounded)
        return full


def freeze_risk(model):
    model.requires_grad_(False)
    model.eval()
    return model
