"""Report usable supervision and reject training sets missing required categories."""

import numpy as np

from risk_residual.config import PHASES


def supervision_summary(arrays):
    valid = np.asarray(arrays["intervention_valid"], dtype=bool)
    positive = valid & (np.asarray(arrays["intervention_target"]) > .5)
    negative = valid & ~positive
    correction = np.asarray(arrays["correction_sample"], dtype=bool)
    stable = np.asarray(arrays["stable_sample"], dtype=bool)
    ids = np.asarray(arrays["parent_id"])
    context = np.asarray(arrays["context"])
    phases = context[:, 0, :len(PHASES)].argmax(-1)
    return {"windows": len(ids), "parents": len(np.unique(ids)),
            "risk_positive": int(positive.sum()), "risk_negative": int(negative.sum()),
            "risk_censored": int((~valid).sum()),
            "correction_samples": int(correction.sum()), "stable_samples": int(stable.sum()),
            "correction_parents": len(np.unique(ids[correction])),
            "stable_parents": len(np.unique(ids[stable])),
            "future_valid_fraction": float(np.asarray(arrays["future_valid"]).mean()),
            "phase_windows": {phase: int((phases == index).sum()) for index, phase in enumerate(PHASES)}}


def readiness(summaries, stage):
    issues = []
    for split in ("train", "val"):
        summary = summaries.get(split)
        if summary is None or not summary["windows"]:
            issues.append(f"{split}: no usable windows")
            continue
        required = ("risk_positive", "risk_negative") if stage == "risk" else ("correction_samples", "stable_samples")
        for key in required:
            if not summary[key]:
                issues.append(f"{split}: no {key}")
    return {"ready": not issues, "issues": issues,
            "scope": "supervision availability only; does not establish sample size or model performance"}


def require_ready(summaries, stage):
    result = readiness(summaries, stage)
    if not result["ready"]:
        raise ValueError(f"Dataset is not ready for {stage} training: " + "; ".join(result["issues"]))
    return result
