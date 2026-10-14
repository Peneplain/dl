"""Masked supervision; corrected futures are never nominal risk targets."""

import torch
from torch.nn import functional as F

from baseline.adapters.joints import ARM_INDICES


def masked_mean(value, mask):
    mask = torch.broadcast_to(mask.bool(), value.shape)
    # where (not multiplication) also masks NaNs in unavailable targets.
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def future_onset_targets(batch):
    """Return the canonical future-only proxy target and its valid mask.

    A window is eligible when the complete current step is observed and has no
    current nominal violation, it is not a verified recovery window, and the
    remaining horizon either contains a violation or is completely observed.
    This mirrors ``experiments.evaluate_risk.future_onset_metrics``: partial
    future negatives remain censored, while partial future positives are valid.
    """
    valid = batch["future_valid"].bool()
    contact_mask = batch["contact_mask"].bool()
    track = torch.where(valid, batch["track_target"], torch.zeros_like(batch["track_target"]))
    balance = torch.where(valid, batch["balance_target"], torch.zeros_like(batch["balance_target"]))
    contact_valid = valid & contact_mask
    contact = torch.where(contact_valid, batch["contact_target"],
                          torch.zeros_like(batch["contact_target"]))
    violations = valid & ((track >= 1) | (balance > .5) |
                          (contact_mask & (contact > .5)))
    current = violations[:, 0].any(-1)
    current_complete = valid[:, 0].all(-1)
    future = violations[:, 1:].any((1, 2))
    complete_future = valid[:, 1:].all((1, 2))
    recovery = batch["correction_sample"].bool()
    onset_eligible = current_complete & ~current & ~recovery
    onset_valid = onset_eligible & (future | complete_future)
    return future.to(dtype=batch["intervention_target"].dtype), onset_valid


def risk_loss(output, batch, *, auxiliary_weight=1.0, future_onset_weight=0.0,
              future_onset_positive_weight=1.0):
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
    onset_target, onset_valid = future_onset_targets(batch)
    onset_target = torch.where(onset_valid, onset_target,
                               torch.zeros_like(onset_target))
    independent_future_logits = getattr(output, "future_onset_logit", None)
    future_logits = (independent_future_logits
                     if independent_future_logits is not None
                     else output.intervention_logit)
    if future_onset_positive_weight <= 0 or not torch.isfinite(torch.as_tensor(future_onset_positive_weight)):
        raise ValueError("future_onset_positive_weight must be finite and positive")
    pos_weight = torch.as_tensor(float(future_onset_positive_weight), device=future_logits.device,
                                 dtype=future_logits.dtype)
    parts = {
        "tracking": masked_mean((output.tracking - track).square(), valid),
        "contact": masked_mean(F.binary_cross_entropy_with_logits(
            output.contact_logits, contact, reduction="none"), contact_valid),
        "balance": masked_mean(F.binary_cross_entropy_with_logits(
            output.balance_logits, balance, reduction="none"), valid),
        "intervention": masked_mean(F.binary_cross_entropy_with_logits(
            output.intervention_logit, intervention, reduction="none"), intervention_valid),
        "future_onset": masked_mean(F.binary_cross_entropy_with_logits(
            future_logits, onset_target, reduction="none", pos_weight=pos_weight), onset_valid),
    }
    total = (parts["intervention"] +
             auxiliary_weight * sum(parts[k] for k in ("tracking", "contact", "balance")) +
             future_onset_weight * parts["future_onset"])
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
