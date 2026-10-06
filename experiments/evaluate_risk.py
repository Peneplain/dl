"""Audit a real Risk checkpoint without using test data to fit or select a gate.

Exports every prediction and its masks for reproduction. Window scores describe
overlapping windows; uncertainty resamples scene/prompt episode clusters, not
individual windows. A validation-fitted temperature is an optional diagnostic
only: the frozen gate emitted here always uses the original raw probability.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common import LOCK, sha256, write_json
from baseline.runtime import device_for
from experiments.calibrate import select_threshold
from risk_residual.checkpoints import load_checkpoint
from risk_residual.config import BODY_GROUPS, PHASES
from risk_residual.data import WindowDataset


def binary_arrays(probability, labels, mask=None):
    probability, labels = np.asarray(probability, dtype=float), np.asarray(labels)
    if probability.shape != labels.shape:
        raise ValueError("Probability and label shapes differ")
    mask = np.ones(labels.shape, bool) if mask is None else np.asarray(mask)
    if mask.shape != labels.shape or mask.dtype != np.bool_:
        raise ValueError("The label mask must have the same shape and bool dtype")
    p, y = probability[mask].ravel(), labels[mask].ravel()
    if (not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Available probabilities/labels must be finite probabilities and binary labels")
    return p, y.astype(bool)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def precision_recall(probability, labels):
    """Exact score-tie groups and step-integrated AP, not trapezoidal PR area."""
    p, y = binary_arrays(probability, labels)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not len(y):
        return {"average_precision": None, "positive_support": 0, "negative_support": 0,
                "thresholds": [], "precision": [], "recall": [], "roc_auc": None}
    order = np.argsort(-p, kind="stable")
    scores, target = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(target)[ends]
    fp = ends + 1 - tp
    precision = tp / (tp + fp)
    recall = tp / positives if positives else np.zeros(len(tp))
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else None
    auc = (float(np.trapz(np.r_[0., recall], np.r_[0., fp / negatives]))
           if positives and negatives else None)
    return {"average_precision": ap, "positive_support": positives, "negative_support": negatives,
            "thresholds": scores[ends].tolist(), "precision": precision.tolist(),
            "recall": recall.tolist() if positives else [None] * len(tp), "roc_auc": auc}


def from_counts(tp, fp, tn, fn):
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "risk_recall": recall, "false_positive_rate": ratio(fp, fp + tn),
            "specificity": specificity, "precision": ratio(tp, tp + fp),
            "negative_predictive_value": ratio(tn, tn + fn),
            "balanced_accuracy": ((recall + specificity) / 2
                                  if recall is not None and specificity is not None else None),
            "accuracy": ratio(tp + tn, tp + fp + tn + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "activation_rate": ratio(tp + fp, tp + fp + tn + fn)}


def binary_metrics(probability, labels, *, threshold=.5, mask=None, curves=False):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability")
    p, y = binary_arrays(probability, labels, mask)
    prediction = p >= threshold
    result = from_counts(np.sum(prediction & y), np.sum(prediction & ~y),
                         np.sum(~prediction & ~y), np.sum(~prediction & y))
    result.update(available=len(y), positive=int(y.sum()), negative=int((~y).sum()),
                  threshold=float(threshold), brier=float(np.mean((p - y) ** 2)) if len(y) else None)
    clipped = p.clip(1e-12, 1 - 1e-12)
    result["binary_cross_entropy"] = (float(np.mean(-np.where(y, np.log(clipped), np.log1p(-clipped))))
                                      if len(y) else None)
    for name, score, target in (("risk", p, y), ("low_risk", 1 - p, ~y)):
        pr = precision_recall(score, target)
        result[name + "_average_precision"] = pr["average_precision"]
        if curves:
            result[name + "_precision_recall"] = pr
    result["roc_auc"] = precision_recall(p, y)["roc_auc"]
    return result


def episode_cluster_ids(dataset):
    """Variants and clean replays sharing scene/prompt remain in one cluster.

    This declared grouping is conservative when multiple originals share a
    scene seed. It does not claim that authored paraphrase groups are random.
    """
    return np.asarray([json.dumps([dataset.parents[str(pid)]["prompt_group"],
                                   dataset.parents[str(pid)]["scene_seed"]], separators=(",", ":"))
                       for pid in dataset.arrays["parent_id"]])


def parent_summary(probability, labels, valid, parent_id, *, threshold, bootstrap=1000, seed=0):
    """Parent macro point estimates plus cluster bootstrap of window metrics.

    Threshold is fixed throughout bootstrap: these are conditional diagnostic
    intervals, not uncertainty in threshold selection or trained parameters.
    """
    p, y, valid, parent_id = map(np.asarray, (probability, labels, valid, parent_id))
    binary_arrays(p, y, valid)
    if p.ndim != 1 or parent_id.shape != p.shape or bootstrap < 0:
        raise ValueError("Invalid parent identifiers or bootstrap count")
    ids = np.unique(parent_id[valid])
    names = ("risk_recall", "false_positive_rate", "specificity", "precision", "balanced_accuracy",
             "risk_average_precision", "low_risk_average_precision", "brier")
    rows, counts = [], []
    for pid in ids:
        row = binary_metrics(p, y, threshold=threshold, mask=valid & (parent_id == pid))
        rows.append({"parent_id": str(pid), **{k: v for k, v in row.items()
                                              if not k.endswith("precision_recall")}})
        counts.append([row[k] for k in ("tp", "fp", "tn", "fn")])
    macro = {}
    for name in names:
        values = [row[name] for row in rows if row[name] is not None]
        macro[name] = {"mean": float(np.mean(values)) if values else None,
                       "contributing_parents": len(values)}
    ci = {}
    if len(ids) >= 2 and bootstrap:
        counts = np.asarray(counts, dtype=np.int64)
        draws = np.random.default_rng(seed).integers(0, len(ids), (bootstrap, len(ids)))
        reduced = [from_counts(*row) for row in counts[draws].sum(1)]
        for name in names[:5]:
            values = [row[name] for row in reduced if row[name] is not None]
            ci[name] = {"ci95": np.quantile(values, [.025, .975]).tolist() if values else None,
                        "defined_draws": len(values), "draws": bootstrap}
    return {"parents": len(ids), "macro": macro, "per_parent": rows, "bootstrap": ci,
            "bootstrap_seed": seed, "bootstrap_threshold_fixed": True,
            "note": "Overlapping windows are correlated; resample complete parent/episode clusters."}


def sigmoid(logit):
    logit = np.asarray(logit, dtype=float)
    return np.exp(-np.logaddexp(0., -logit))


def fit_temperature(logits, labels, *, split="val"):
    """Bounded scalar temperature; reports same-validation fit optimistically."""
    if split != "val":
        raise ValueError("Temperature fitting is restricted to validation data")
    logits, labels = np.asarray(logits, float), np.asarray(labels)
    if (logits.ndim != 1 or labels.shape != logits.shape or not np.isfinite(logits).all()
            or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2):
        raise ValueError("Temperature fitting needs finite logits and both validation classes")
    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., scaled) - labels * scaled))
    # Optimize log(T) within explicit bounds without adding a scipy dependency.
    lower, upper = np.log(.05), np.log(20.)
    golden = (np.sqrt(5.) - 1.) / 2
    a, b = lower, upper
    for _ in range(100):
        x, z = b - golden * (b - a), a + golden * (b - a)
        if objective(x) < objective(z):
            b = z
        else:
            a = x
    candidates = [lower, upper, (a + b) / 2, 0.]
    optimum = min(candidates, key=objective)
    return {"temperature": float(np.exp(optimum)), "raw_validation_nll": objective(0.),
            "fitted_validation_nll": objective(optimum), "bounds": [.05, 20.],
            "fit_split": "val", "runtime_applied": False,
            "note": "Fit and diagnostic scores reuse validation; not an independent calibration assessment."}


def infer_checkpoint(risk_path, manifest_path, *, device="cpu", split="val", batch_size=64,
                     allow_synthetic=False):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model, normalizer, meta = load_checkpoint(risk_path, kind="risk", device=device)
    data = WindowDataset(manifest_path, split, model.config)
    if (meta["dataset_manifest_sha256"] != sha256(manifest_path)
            or data.manifest["provenance"]["baseline_lock_sha256"] != sha256(LOCK)):
        raise ValueError("Evaluation data differs from risk training provenance")
    if bool(meta.get("synthetic_inputs")) != bool(data.manifest.get("synthetic_inputs")):
        raise ValueError("Checkpoint and dataset synthetic provenance differ")
    if meta.get("synthetic_inputs") and not allow_synthetic:
        raise ValueError("Synthetic evaluation requires explicit --allow-synthetic")
    data_root = Path(manifest_path).resolve().parent
    validation_path = (data_root / data.manifest["windows"]["val"]).resolve()
    if not validation_path.is_relative_to(data_root):
        raise ValueError("Validation windows must remain inside the dataset directory")
    validation_hash = sha256(validation_path)
    if meta["window_hashes"]["val"] != validation_hash:
        raise ValueError("Validation windows changed since risk training")
    collected = {key: [] for key in ("intervention_logit", "probability", "tracking", "contact_logits", "balance_logits")}
    with torch.inference_mode():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
            inputs = {key: value.to(device) for key, value in batch.items()}
            output = model(**normalizer(inputs))
            for key in collected:
                value = getattr(output, key).cpu().numpy()
                if not np.isfinite(value).all():
                    raise FloatingPointError("Nonfinite model prediction: " + key)
                collected[key].append(value)
    predictions = {key: np.concatenate(value) for key, value in collected.items()}
    for key in ("parent_id", "decision_time", "intervention_valid", "intervention_target", "future_valid",
                "contact_mask", "track_target", "contact_target", "balance_target", "correction_sample"):
        predictions[key] = data.arrays[key]
    phase = data.arrays["context"][:, 0, :len(PHASES)]
    if not np.allclose(phase.sum(-1), 1) or not np.isin(phase, [0, 1]).all():
        raise ValueError("Evaluation requires a one-hot decision phase")
    predictions["phase_index"] = phase.argmax(-1)
    predictions["episode_cluster_id"] = episode_cluster_ids(data)
    provenance = {"risk_sha256": sha256(risk_path), "dataset_manifest_sha256": sha256(manifest_path),
                  "windows_sha256": sha256(data.path), "split": split, "checkpoint_epoch": meta["epoch"],
                  "validation_windows_sha256": validation_hash,
                  "seed": meta["seed"], "device": str(device), "model_config": model.config.to_dict(),
                  "synthetic_inputs": bool(meta.get("synthetic_inputs")), "physics_executed": False,
                  "label_thresholds": data.manifest["provenance"]["label_thresholds"],
                  "task_success": None}
    return predictions, provenance


def auxiliary_metrics(a):
    result = {}
    for i, body in enumerate(BODY_GROUPS):
        valid = a["future_valid"][:, :, i]
        error = a["tracking"][:, :, i][valid] - a["track_target"][:, :, i][valid]
        result[body] = {"tracking": {"available": int(valid.sum()),
                                    "mse": float(np.mean(error ** 2)) if len(error) else None,
                                    "mae": float(np.mean(np.abs(error))) if len(error) else None,
                                    "units": "tracking threshold normalized"},
                        "contact": binary_metrics(sigmoid(a["contact_logits"][:, :, i]),
                                                  a["contact_target"][:, :, i],
                                                  mask=valid & a["contact_mask"][:, :, i]),
                        "balance": binary_metrics(sigmoid(a["balance_logits"][:, :, i]),
                                                  a["balance_target"][:, :, i], mask=valid)}
    return result


def future_onset_metrics(a, *, threshold):
    """Separate future proxy onset from already observable proxy violations.

    h=0 is the decision state, not an independently predicted future. A partial
    future may establish a violation, but a proxy-negative future requires all
    remaining steps. Verified recovery labels are reported separately and are
    excluded from the pure future-onset subset. No threshold is fitted here.
    """
    valid = np.asarray(a["future_valid"], dtype=bool)
    violations = valid & ((a["track_target"] >= 1) | (a["balance_target"] > .5)
                          | (a["contact_mask"] & (a["contact_target"] > .5)))
    current = violations[:, 0].any(-1)
    current_complete = valid[:, 0].all(-1)
    future = violations[:, 1:].any((1, 2))
    complete_future = valid[:, 1:].all((1, 2))
    proxy_any = current | future
    recovery = np.asarray(a["correction_sample"], dtype=bool)
    intervention_valid = np.asarray(a["intervention_valid"], dtype=bool)
    labels = a["intervention_target"]
    p = a.get("probability", sigmoid(a["intervention_logit"]))
    # A reactive positive is established even if another body is unavailable;
    # an observed reactive negative requires all current bodies to be available.
    reactive_available = intervention_valid & (current | current_complete)
    onset_eligible = current_complete & ~current & ~recovery
    onset_valid = onset_eligible & (future | complete_future)
    direct_only = recovery & ~proxy_any

    def compare(mask, target):
        return {"raw_gate": binary_metrics(p, target, threshold=threshold, mask=mask),
                "current_proxy_reactive": binary_metrics(current.astype(float), target,
                                                          threshold=.5, mask=mask)}

    counts = {"intervention_positive": int((intervention_valid & (labels == 1)).sum()),
              "already_current_proxy_positive": int((intervention_valid & current).sum()),
              "future_only_proxy_positive": int((intervention_valid & ~current & future).sum()),
              "verified_recovery_windows": int(recovery.sum()),
              "direct_recovery_only_positive": int(direct_only.sum()),
              "current_proxy_negative_complete": int((current_complete & ~current).sum()),
              "future_onset_available": int(onset_valid.sum()),
              "future_onset_positive": int((onset_valid & future).sum()),
              "future_onset_negative": int((onset_valid & ~future).sum()),
              "future_onset_censored": int((onset_eligible & ~onset_valid).sum()),
              "recovery_excluded_from_future_onset": int((current_complete & ~current & recovery).sum())}
    for key in ("parent_id", "episode_cluster_id"):
        if key in a:
            counts["future_onset_" + key + "_count"] = len(np.unique(a[key][onset_valid]))
    result = {"counts": counts,
              "all_observed_intervention": compare(reactive_available, labels),
              "current_proxy_negative_future_onset": compare(onset_valid, future),
              "verified_recovery": compare(intervention_valid & recovery, labels),
              "direct_recovery_only": compare(intervention_valid & direct_only, labels),
              "by_phase": {},
              "definition": "Current proxy uses h=0 nominal tracking/contact/balance violations; future onset uses h>0.",
              "selection": "Use the same frozen raw gate threshold; no subset threshold fitting.",
              "note": "Proxy onset is not proof of future physical task failure; verified recovery is separate supervision."}
    for index, phase in enumerate(PHASES):
        phase_mask = a["phase_index"] == index
        result["by_phase"][phase] = {
            "all_observed_intervention": compare(reactive_available & phase_mask, labels),
            "current_proxy_negative_future_onset": compare(onset_valid & phase_mask, future),
            "verified_recovery": compare(intervention_valid & recovery & phase_mask, labels),
            "direct_recovery_only": compare(intervention_valid & direct_only & phase_mask, labels)}
    return result


def evaluate_arrays(a, *, threshold, bootstrap=1000, bootstrap_seed=0):
    p = a.get("probability", sigmoid(a["intervention_logit"]))
    labels, valid = a["intervention_target"], a["intervention_valid"]
    result = {"windows": len(p), "censored": int((~valid).sum()),
              "intervention": binary_metrics(p, labels, threshold=threshold, mask=valid, curves=True),
              "always_high_risk": binary_metrics(np.ones_like(p), labels, threshold=.5, mask=valid),
              "always_low_risk": binary_metrics(np.zeros_like(p), labels, threshold=.5, mask=valid),
              "by_phase": {}, "auxiliary_by_body": auxiliary_metrics(a)}
    result["predictive_onset"] = future_onset_metrics(a, threshold=threshold)
    for i, phase in enumerate(PHASES):
        mask = valid & (a["phase_index"] == i)
        result["by_phase"][phase] = binary_metrics(p, labels, threshold=threshold, mask=mask)
    result["parent_macro"] = parent_summary(p, labels, valid, a["parent_id"], threshold=threshold,
                                             bootstrap=0)
    result["episode_cluster"] = parent_summary(p, labels, valid, a["episode_cluster_id"],
                                                threshold=threshold, bootstrap=bootstrap, seed=bootstrap_seed)
    result["episode_cluster"]["grouping"] = "same prompt_group and scene_seed; all perturbation/clean branches together"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--calibration", type=Path, help="Frozen validation gate; required for final test evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--temperature-report", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    if (args.out.exists() or args.threads < 1 or args.bootstrap < 0
            or args.split == "test" and (args.calibration is None or args.temperature_report)):
        parser.error("Use a fresh output, valid counts, and a pre-frozen raw validation gate for test; no test fitting")
    torch.set_num_threads(args.threads)
    a, provenance = infer_checkpoint(args.risk, args.data, device=device_for(args.device), split=args.split,
                                     batch_size=args.batch_size, allow_synthetic=args.allow_synthetic)
    valid = a["intervention_valid"]
    if args.calibration:
        gate = json.loads(args.calibration.read_text())
        if (gate.get("split") != "val" or gate.get("risk_sha256") != provenance["risk_sha256"]
                or gate.get("dataset_manifest_sha256") != provenance["dataset_manifest_sha256"]
                or gate.get("validation_windows_sha256", provenance["validation_windows_sha256"])
                    != provenance["validation_windows_sha256"]
                or gate.get("probability_transform", "raw_sigmoid") != "raw_sigmoid"
                or gate.get("temperature", 1.) != 1.):
            raise ValueError("The frozen gate must belong to this checkpoint/data and use raw sigmoid")
    else:
        gate = {**select_threshold(a["probability"][valid], a["intervention_target"][valid]),
                "split": "val", "risk_sha256": provenance["risk_sha256"],
                "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
                "validation_windows_sha256": provenance["validation_windows_sha256"],
                "synthetic_inputs": provenance["synthetic_inputs"], "probability_transform": "raw_sigmoid",
                "temperature": 1., "validation_windows": int(valid.sum())}
    result = {"schema": "dl-risk-evaluation-v1", "status": "passed", **provenance,
              "gate": gate, **evaluate_arrays(a, threshold=gate["threshold"], bootstrap=args.bootstrap,
                                               bootstrap_seed=args.bootstrap_seed),
              "selection_note": "Validation F1 selection is diagnostic; test may only use a pre-frozen gate."}
    if args.temperature_report:
        calibration = fit_temperature(a["intervention_logit"][valid], a["intervention_target"][valid])
        scaled = sigmoid(a["intervention_logit"] / calibration["temperature"])
        alternative_gate = select_threshold(scaled[valid], a["intervention_target"][valid])
        result["temperature_diagnostic"] = {**calibration, "candidate_gate": alternative_gate,
                                            "metrics": binary_metrics(scaled, a["intervention_target"], mask=valid,
                                                                      threshold=alternative_gate["threshold"])}
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "predictions.npz", **a)
    result["predictions_sha256"] = sha256(args.out / "predictions.npz")
    write_json(args.out / "gate.json", gate)
    write_json(args.out / "report.json", result)
    print(json.dumps({key: result[key] for key in ("status", "split", "checkpoint_epoch", "seed", "gate")}, indent=2))


if __name__ == "__main__":
    main()
