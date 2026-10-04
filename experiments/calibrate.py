"""Freeze a gate threshold using validation labels only (maximum F1 pilot rule)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.common import sha256
from baseline.runtime import device_for
from risk_residual.checkpoints import load_checkpoint
from risk_residual.data import WindowDataset


def select_threshold(probability, labels):
    probability, labels = np.asarray(probability), np.asarray(labels)
    if (probability.ndim != 1 or labels.shape != probability.shape or len(labels) == 0
            or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()
            or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2):
        raise ValueError("Calibration requires finite probabilities and both positive/negative validation labels")
    best = None
    for threshold in np.linspace(0, 1, 101):
        predicted = probability >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        candidate = (f1, -int(predicted.sum()), threshold)
        if best is None or candidate > best:
            best = candidate
    return {"threshold": float(best[2]), "validation_f1": float(best[0]),
                "rule": "max F1 on grid 0:.01:1; tie: fewer activations, then higher threshold"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="musa")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output file")
    device = device_for(args.device)
    model, normalizer, meta = load_checkpoint(args.risk, kind="risk", device=device)
    if meta["dataset_manifest_sha256"] != sha256(args.data):
        raise ValueError("Calibration manifest differs from risk training")
    data = WindowDataset(args.data, "val", model.config)
    if meta["window_hashes"]["val"] != sha256(data.path):
        raise ValueError("Validation windows changed since risk training")
    probabilities, labels = [], []
    with torch.inference_mode():
        for batch in DataLoader(data, batch_size=64):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**normalizer(batch))
            valid = batch["intervention_valid"]
            probabilities.extend(output.probability[valid].cpu().tolist())
            labels.extend(batch["intervention_target"][valid].cpu().tolist())
    result = dict(select_threshold(probabilities, labels), split="val", risk_sha256=sha256(args.risk),
                  dataset_manifest_sha256=sha256(args.data), validation_windows=len(labels),
                  synthetic_inputs=bool(meta["synthetic_inputs"]))
    # Exclusive creation preserves the validation-freeze boundary.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
