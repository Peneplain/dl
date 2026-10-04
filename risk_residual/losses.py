"""Masked supervision; corrected futures are never nominal risk targets."""

import torch
from torch.nn import functional as F

from baseline.adapters.joints import ARM_INDICES


def masked_mean(value, mask):
    mask = torch.broadcast_to(mask.bool(), value.shape)
    # where (not multiplication) also masks NaNs in unavailable targets.
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def risk_loss(output, batch, *, auxiliary_weight=1.0):
    valid = batch["future_valid"].bool()
    contact_valid = valid & batch["contact_mask"].bool()
    # Sanitize masked targets BEFORE differentiable operations: 0 * NaN in a
    # backward pass is still NaN even if a downstream reduction is masked.
    track = torch.where(valid, batch["track_target"], torch.zeros_like(output.tracking))
    contact = torch.where(contact_valid, batch["contact_target"],
                          torch.zeros_like(output.contact_logits))
    balance = torch.where(valid, batch["balance_target"], torch.zeros_like(output.balance_logits))
    intervention_valid = batch["intervention_valid"].bool()
    intervention = torch.where(intervention_valid, batch["intervention_target"],
                               torch.zeros_like(output.intervention_logit))
    parts = {
        "tracking": masked_mean((output.tracking - track).square(), valid),
        "contact": masked_mean(F.binary_cross_entropy_with_logits(
            output.contact_logits, contact, reduction="none"), contact_valid),
        "balance": masked_mean(F.binary_cross_entropy_with_logits(
            output.balance_logits, balance, reduction="none"), valid),
        "intervention": masked_mean(F.binary_cross_entropy_with_logits(
            output.intervention_logit, intervention, reduction="none"), intervention_valid),
    }
    total = parts["intervention"] + auxiliary_weight * sum(parts[k] for k in
                                                           ("tracking", "contact", "balance"))
    return total, parts


def residual_loss(prediction, batch, *, alpha=1.0, beta=.1):
    correction = batch["correction_sample"].bool()
    stable = batch["stable_sample"].bool()
    if (correction & stable).any():
        raise ValueError("Correction and stable identity sets must be disjoint")
    available = batch["residual_valid"].bool()
    prediction = prediction[..., list(ARM_INDICES)]
    target = batch["offset_target"][..., list(ARM_INDICES)]
    correction_mask = correction[:, None] & available
    identity_mask = stable[:, None] & available
    target = torch.where(correction_mask[..., None], target, torch.zeros_like(target))
    # Frobenius norm over arms, averaged over available times then samples.
    def per_sample(value, valid, selected):
        counts = valid.sum(-1)
        mean = torch.where(valid, value, torch.zeros_like(value)).sum(-1) / counts.clamp_min(1)
        return masked_mean(mean, selected & (counts > 0))
    correction_loss = per_sample((prediction - target).square().sum(-1),
                                 correction_mask, correction)
    identity = per_sample(prediction.square().sum(-1), identity_mask, stable)
    pair_valid = available[:, 1:] & available[:, :-1] & (correction | stable)[:, None]
    changes = (prediction[:, 1:] - prediction[:, :-1]).square().sum(-1)
    smooth = masked_mean(torch.where(pair_valid, changes, torch.zeros_like(changes)).sum(-1),
                         (correction | stable) & pair_valid.any(-1))
    return correction_loss + alpha * identity + beta * smooth, {
        "correction": correction_loss, "identity": identity, "smoothness": smooth}
