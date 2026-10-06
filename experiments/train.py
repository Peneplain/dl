"""Train risk first; freeze it and train residuals using PREDICTED features.

python -m experiments.train --stage risk --data datasets/task/manifest.json ...
Synthetic fixtures require --allow-synthetic and cannot become task checkpoints.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from baseline.common import LOCK, ROOT, sha256, write_json
from baseline.runtime import device_for, synchronize
from risk_residual.checkpoints import load_checkpoint
from risk_residual.audit import require_ready, supervision_summary
from risk_residual.config import SCHEMA, ModelConfig
from risk_residual.data import Normalizer, WindowDataset
from risk_residual.losses import residual_loss, risk_loss
from risk_residual.models import ResidualModel, RiskModel, freeze_risk, risk_features


class EarlyStopping:
    """Optional validation-only stopping; best checkpoint still uses exact val_loss.

    This controls the actual number of epochs, not model architecture or loss.
    A positive min_delta affects patience only, never the best.pt selection rule.
    """

    def __init__(self, settings=None):
        settings = {} if settings is None else settings
        if not isinstance(settings, dict) or set(settings) - {"patience", "min_delta", "monitor"}:
            raise ValueError("Invalid early_stopping fields")
        self.enabled = bool(settings)
        self.patience = settings.get("patience", 0)
        self.min_delta = settings.get("min_delta", 0.)
        self.monitor = settings.get("monitor", "val_loss")
        if self.enabled and (not isinstance(self.patience, int) or isinstance(self.patience, bool)
                             or self.patience < 1 or not np.isfinite(self.min_delta) or self.min_delta < 0
                             or self.monitor not in {"val_loss", "val_intervention"}):
            raise ValueError("Early stopping needs positive patience, finite nonnegative delta, and validation monitor")
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, metrics):
        if not self.enabled:
            return False
        value = (metrics["val_loss"] if self.monitor == "val_loss"
                 else metrics["val_components"]["intervention"])
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite early stopping monitor")
        if value < self.best - self.min_delta:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def balanced_intervention_weights(arrays):
    """Equal expected Risk class mass, fitted on TRAIN labels only.

    Censored negatives receive zero sampling mass, so their auxiliary labels are
    omitted by this explicitly optional ablation. Epoch sample count remains the
    full original training count to preserve the planned optimizer budget.
    """
    valid, labels = np.asarray(arrays["intervention_valid"]), np.asarray(arrays["intervention_target"])
    if valid.dtype != np.bool_ or labels.shape != valid.shape or valid.ndim != 1:
        raise ValueError("Invalid intervention sampling arrays")
    if not np.isin(labels[valid], [0, 1]).all():
        raise ValueError("Sampling requires binary available labels")
    positive, negative = valid & (labels == 1), valid & (labels == 0)
    if not positive.any() or not negative.any():
        raise ValueError("Balanced Risk sampling requires both TRAIN classes")
    weights = np.zeros(len(valid), np.float64)
    weights[positive] = .5 / positive.sum()
    weights[negative] = .5 / negative.sum()
    return weights, {"mode": "balanced_intervention", "fit_split": "train", "replacement": True,
                     "positive_windows": int(positive.sum()), "negative_windows": int(negative.sum()),
                     "excluded_censored": int((~valid).sum()), "samples_per_epoch": len(valid),
                     "validation_distribution": "unchanged natural validation windows",
                     "note": "Window balancing is not an increase in independent episodes."}


def reduction_counts(stage, batch):
    """Aggregate validation by supervised elements/sets, independent of batch partition."""
    if stage == "risk":
        valid = batch["future_valid"]
        return {"tracking": valid.sum(), "balance": valid.sum(),
                "contact": (valid & batch["contact_mask"]).sum(),
                "intervention": batch["intervention_valid"].sum()}
    available = batch["residual_valid"]
    correction, stable = batch["correction_sample"], batch["stable_sample"]
    return {"correction": (correction & available.any(-1)).sum(),
            "identity": (stable & available.any(-1)).sum(),
            "smoothness": ((correction | stable) &
                           (available[:, 1:] & available[:, :-1]).any(-1)).sum()}


def train(args):
    settings = json.loads(args.config.read_text())
    config = ModelConfig(**settings["model"])
    budget = settings["training"]
    early_stop = EarlyStopping(budget.get("early_stopping"))
    sampling_mode = budget.get("risk_sampling", "natural")
    if sampling_mode not in {"natural", "balanced_intervention"}:
        raise ValueError("risk_sampling must be natural or balanced_intervention")
    if args.stage != "risk" and (sampling_mode != "natural" or early_stop.monitor == "val_intervention"):
        raise ValueError("Risk-specific sampling/monitor cannot be used for residual training")
    if (budget["epochs"] < 1 or budget["batch_size"] < 1
            or not 0 < budget["correction_fraction"] < 1
            or any(not np.isfinite(budget[key]) or budget[key] < 0 for key in
                   ("learning_rate", "weight_decay", "alpha", "beta", "auxiliary_weight"))
            or budget["learning_rate"] == 0 or not np.isfinite(budget["gradient_clip"])
            or budget["gradient_clip"] <= 0):
        raise ValueError("Invalid fixed training budget")
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "musa":
        torch.musa.manual_seed_all(args.seed)
    train_data = WindowDataset(args.data, "train", config)
    val_data = WindowDataset(args.data, "val", config)
    require_ready({"train": supervision_summary(train_data.arrays),
                   "val": supervision_summary(val_data.arrays)}, args.stage)
    if train_data.manifest.get("synthetic_inputs", False) and not args.allow_synthetic:
        raise ValueError("Synthetic fixtures require explicit --allow-synthetic")
    if train_data.manifest["provenance"]["baseline_lock_sha256"] != sha256(LOCK):
        raise ValueError("Dataset was collected with a different frozen baseline lock")
    risk = None
    risk_options = settings.get("risk_options", {})
    if args.stage == "risk":
        if args.risk is not None:
            raise ValueError("--risk applies only to residual training")
        model = RiskModel(config, **risk_options).to(device)
        normalizer = Normalizer.fit(train_data).to(device)
    else:
        if args.risk is None:
            raise ValueError("Residual training requires a frozen --risk checkpoint")
        risk, normalizer, risk_meta = load_checkpoint(args.risk, kind="risk", device=device)
        if (risk.config != config or risk_meta["dataset_manifest_sha256"] != sha256(args.data)
                or risk_meta["risk_options"] != risk_options
                or risk_meta["window_hashes"] != {"train": sha256(train_data.path), "val": sha256(val_data.path)}):
            raise ValueError("Frozen risk configuration/data differs from residual training")
        freeze_risk(risk)
        model = ResidualModel(config).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    training = train_data
    validation = val_data
    sampling_report = {"mode": "natural", "fit_split": "train", "samples_per_epoch": len(train_data)}
    if args.stage == "risk" and sampling_mode == "balanced_intervention":
        weights, sampling_report = balanced_intervention_weights(train_data.arrays)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(train_data),
                                         replacement=True, generator=generator)
    if args.stage == "residual":
        a = train_data.arrays
        selected = np.flatnonzero(a["correction_sample"] | a["stable_sample"])
        correction_count = int(a["correction_sample"].sum())
        stable_count = int(a["stable_sample"].sum())
        if not correction_count or not stable_count:
            raise ValueError("Residual training needs disjoint correction and stable identity sets")
        ratio = budget["correction_fraction"]
        weights = np.where(a["correction_sample"][selected], ratio / correction_count,
                            (1 - ratio) / stable_count)
        training = Subset(train_data, selected.tolist())
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(selected),
                                         replacement=True, generator=generator)
        selected_val = np.flatnonzero(val_data.arrays["correction_sample"] |
                                      val_data.arrays["stable_sample"])
        if not len(selected_val):
            raise ValueError("No residual validation samples")
        validation = Subset(val_data, selected_val.tolist())
    loaders = {
        "train": DataLoader(training, batch_size=budget["batch_size"], sampler=sampler,
                             shuffle=sampler is None, generator=generator, num_workers=0),
        "val": DataLoader(validation, batch_size=budget["batch_size"], shuffle=False, num_workers=0),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget["learning_rate"],
                                  weight_decay=budget["weight_decay"])
    provenance = {
        "kind": args.stage, "schema": SCHEMA, "model_config": config.to_dict(),
        "settings": settings, "seed": args.seed, "interface": args.interface,
        "risk_options": risk_options, "baseline_lock_sha256": sha256(LOCK),
        "dataset_manifest_sha256": sha256(args.data),
        "window_hashes": {"train": sha256(train_data.path), "val": sha256(val_data.path)},
        "risk_sha256": sha256(args.risk) if args.risk else None,
        "synthetic_inputs": bool(train_data.manifest.get("synthetic_inputs", False)),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()), "adapter_parameters": 0,
        "inference_inputs": ["history", "nominal", "context"],
        "loss_normalization": "risk: available elements; residual: arm squared norm/time/sample; smoothness: time-sum/sample",
        "training_controls": {"risk_sampling": sampling_report,
                              "early_stopping": {"enabled": early_stop.enabled, "patience": early_stop.patience,
                                                 "min_delta": early_stop.min_delta, "monitor": early_stop.monitor},
                              "budget_epochs": budget["epochs"],
                              "checkpoint_selection": "exact minimum total validation loss"},
    }
    history = []
    best = float("inf")
    best_epoch = 0
    updates = 0
    start = time.perf_counter()
    for epoch in range(budget["epochs"]):
        metrics = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            model.train(split == "train")
            numerators, denominators = {}, {}
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                inputs = normalizer(batch)
                with torch.set_grad_enabled(split == "train"):
                    if args.stage == "risk":
                        loss, parts = risk_loss(model(**inputs), batch,
                                            auxiliary_weight=budget["auxiliary_weight"])
                    else:
                        with torch.no_grad():
                            features = risk_features(risk(**inputs), args.interface)
                        prediction = model(**inputs, features=features)
                        loss, parts = residual_loss(prediction, batch, alpha=budget["alpha"], beta=budget["beta"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training/validation loss")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), budget["gradient_clip"],
                                                       error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                for key, count in reduction_counts(args.stage, batch).items():
                    count = int(count)
                    numerators[key] = numerators.get(key, 0.) + float(parts[key].detach()) * count
                    denominators[key] = denominators.get(key, 0) + count
            reduced = {key: numerators[key] / max(1, denominators[key]) for key in numerators}
            metrics[split + "_components"] = reduced
            if args.stage == "risk":
                metrics[split + "_loss"] = reduced["intervention"] + budget["auxiliary_weight"] * sum(
                    reduced[key] for key in ("tracking", "contact", "balance"))
            else:
                metrics[split + "_loss"] = (reduced["correction"] + budget["alpha"] * reduced["identity"]
                                            + budget["beta"] * reduced["smoothness"])
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        checkpoint = {**provenance, "epoch": epoch + 1, "updates": updates,
                      "model": model.state_dict(), "normalizer": normalizer.state_dict()}
        torch.save(checkpoint, args.out / "last.pt")
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            best_epoch = epoch + 1
            torch.save(checkpoint, args.out / "best.pt")
        write_json(args.out / "metrics.json", history)
        if early_stop.update(metrics):
            break
    synchronize(device)
    return {**provenance, "status": "passed", "epochs": len(history), "actual_epochs": len(history),
            "budget_epochs": budget["epochs"], "stopped_early": len(history) < budget["epochs"],
            "stop_reason": "validation_patience" if len(history) < budget["epochs"] else "fixed_budget_completed",
            "best_epoch": best_epoch, "updates": updates,
            "wall_seconds": time.perf_counter() - start, "best_val_loss": best,
            "best_sha256": sha256(args.out / "best.pt"),
            "physics_executed": False, "task_success": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("risk", "residual"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learning/p.json")
    parser.add_argument("--risk", type=Path)
    parser.add_argument("--interface", choices=("B1", "I2", "I3", "I4", "P", "pooled"), default="P")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        parser.error("Use a fresh --out directory and positive --threads")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "physics_executed": False, "task_success": None}
    try:
        report = train(args)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
