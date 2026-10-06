"""Strict checkpoint interfaces and provenance; never load arbitrary pickled objects."""

from pathlib import Path

import torch

from baseline.common import LOCK, sha256
from risk_residual.config import SCHEMA, ModelConfig
from risk_residual.data import Normalizer
from risk_residual.models import ResidualModel, RiskModel


def load_checkpoint(path, *, kind, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("kind") != kind
            or checkpoint.get("baseline_lock_sha256") != sha256(LOCK)):
        raise ValueError("Checkpoint schema, kind or frozen baseline lock mismatch")
    config = ModelConfig(**checkpoint["model_config"])
    options = checkpoint.get("risk_options", {})
    model = RiskModel(config, **options) if kind == "risk" else ResidualModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    normalizer = Normalizer().to(device)
    normalizer.load_state_dict(checkpoint["normalizer"], strict=True)
    return model, normalizer, checkpoint


def validate_gate(calibration, risk_path, risk_meta):
    """Reject gates whose probabilities do not match runtime's raw sigmoid."""
    if (calibration.get("split") != "val"
            or calibration.get("risk_sha256") != sha256(risk_path)
            or calibration.get("dataset_manifest_sha256")
            != risk_meta.get("dataset_manifest_sha256")):
        raise ValueError("Gate must be calibrated on validation data for this risk checkpoint")
    if (calibration.get("probability_transform", "raw_sigmoid") != "raw_sigmoid"
            or calibration.get("temperature", 1.0) != 1.0):
        raise ValueError("Runtime requires a raw_sigmoid gate with temperature 1")
    validation_hash = calibration.get("validation_windows_sha256")
    if (validation_hash is not None
            and validation_hash != risk_meta.get("window_hashes", {}).get("val")):
        raise ValueError("Gate validation windows differ from the risk checkpoint")
    threshold = calibration.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1):
        raise ValueError("Gate threshold must be a finite probability")


def load_controller(risk_path, residual_path, calibration_path, *, device="cpu", no_gate=False,
                    method=None):
    import json

    from risk_residual.runtime import PredictiveController

    risk, normalizer, risk_meta = load_checkpoint(risk_path, kind="risk", device=device)
    residual, residual_norm, meta = load_checkpoint(residual_path, kind="residual", device=device)
    if (meta.get("risk_sha256") != sha256(risk_path)
            or meta.get("dataset_manifest_sha256") != risk_meta.get("dataset_manifest_sha256")):
        raise ValueError("Residual was not trained with this frozen risk/data manifest")
    if any(not torch.equal(value, residual_norm.state_dict()[key])
           for key, value in normalizer.state_dict().items()):
        raise ValueError("Risk and residual normalization differ")
    calibration = json.loads(Path(calibration_path).read_text())
    validate_gate(calibration, risk_path, risk_meta)
    if meta.get("synthetic_inputs") or calibration.get("synthetic_inputs"):
        raise ValueError("Synthetic checkpoints cannot be loaded as physical task controllers")
    method = method or meta["interface"]
    allowed = {"B1", "B2", "I1"} if meta["interface"] == "B1" else {meta["interface"]}
    if method not in allowed:
        raise ValueError("B1/B2/I1 must reuse B1; other interfaces require their own residual checkpoint")
    return PredictiveController(risk, residual, normalizer, threshold=calibration["threshold"],
                                interface=method, no_gate=no_gate)
