"""Audited, pre-split windows. No pickle and no inference inputs from labels."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from baseline.adapters.joints import ARM_INDICES
from risk_residual.config import CONTEXT_DIM, NOMINAL_DIM, PHASES, SCHEMA, STATE_DIM

INPUT_KEYS = ("history", "nominal", "context")
TARGET_KEYS = ("future_valid", "contact_mask", "track_target", "contact_target",
               "balance_target", "intervention_valid", "intervention_target", "offset_target",
               "residual_valid", "correction_sample", "stable_sample")


def audit_parents(parents):
    """All nominal/counterfactual/teacher branches inherit a parent's split."""
    by_id, owners = {}, {"prompt_group": {}, "scene_seed": {}}
    for parent in parents:
        pid, split = parent["parent_id"], parent["split"]
        if not isinstance(pid, str) or not pid or pid in by_id or split not in {"train", "val", "test"}:
            raise ValueError("Invalid/duplicate parent ID or split")
        by_id[pid] = parent
        for key, values in owners.items():
            value = str(parent[key])
            if value in values and values[value] != split:
                raise ValueError(f"{key} leakage across parent splits: {value}")
            values[value] = split
    if not by_id:
        raise ValueError("Empty parent split manifest")
    return by_id


def nominal_risk_labels(tracking_error, contact_lost, instability, valid, contact_required,
                        tracking_thresholds):
    """Derive labels from a NOMINAL branch's executed future, never corrected outcomes.

    Input tracking errors and thresholds use the same per-body metric/units.
    Contact-required is phase/hand dependent and false for torso/lower body.
    A partial future establishes positives, but cannot establish a negative.
    """
    error = np.asarray(tracking_error, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    required = np.asarray(contact_required, dtype=bool)
    thresholds = np.asarray(tracking_thresholds, dtype=float)
    if (error.shape != (8, 4) or valid.shape != error.shape or required.shape != error.shape
            or thresholds.shape != (4,) or not np.isfinite(thresholds).all()
            or (thresholds <= 0).any() or required[:, 2:].any()):
        raise ValueError("Invalid body/time labels, contact mask or tracking thresholds")
    contact, balance = np.asarray(contact_lost), np.asarray(instability)
    for value, mask in ((error, valid), (contact, valid & required), (balance, valid)):
        if value.shape != error.shape or not np.isfinite(value[mask]).all():
            raise ValueError("Nonfinite or mismatched available labels")
    normalized = np.where(valid, error / thresholds, 0)
    violation = valid & ((normalized >= 1) | (balance > .5) | (required & (contact > .5)))
    return {"track_target": normalized.astype(np.float32),
                "contact_target": np.where(required & valid, contact, 0).astype(np.float32),
                "balance_target": np.where(valid, balance, 0).astype(np.float32),
                "future_valid": valid, "contact_mask": required,
                "intervention_target": np.float32(violation.any()),
                "intervention_valid": np.bool_(violation.any() or valid.all())}


class WindowDataset(Dataset):
    """Manifest + one NPZ per split; audit before handing any windows to training.

    Provenance fields are attestations from the collector, not proof that an
    external teacher actually succeeded. Physical verification remains required.
    """

    def __init__(self, manifest_path, split, config):
        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Dataset schema differs from the versioned inference interface")
        self.parents = audit_parents(self.manifest["parents"])
        provenance = self.manifest.get("provenance", {})
        required = ("baseline_lock_sha256", "scene_hashes", "label_thresholds", "collector_revision")
        if any(not provenance.get(key) for key in required):
            raise ValueError("Missing collection/label provenance")
        root = manifest_path.resolve().parent
        path = (root / self.manifest["windows"][split]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Windows must be inside the dataset directory")
        self.path = path
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        a = self.arrays
        n, h, k = len(a["history"]), config.horizon, config.history_steps
        expected = {"history": (n, k, STATE_DIM), "nominal": (n, h, NOMINAL_DIM),
                    "context": (n, h, CONTEXT_DIM), "offset_target": (n, h, 29),
                    "residual_valid": (n, h)}
        expected.update({key: (n, h, 4) for key in
                         ("future_valid", "contact_mask", "track_target", "contact_target", "balance_target")})
        expected.update({key: (n,) for key in ("intervention_valid", "intervention_target",
                         "correction_sample", "stable_sample", "parent_id", "risk_source",
                         "teacher_verified", "nominal_snapshot", "teacher_snapshot", "decision_time")})
        expected.update(history_times=(n, k), nominal_times=(n, h))
        if n == 0:
            raise ValueError("Empty window split")
        for key, shape in expected.items():
            if key not in a or a[key].shape != shape:
                raise ValueError(f"Invalid/missing {key}; expected {shape}")
        for key in INPUT_KEYS + ("decision_time", "history_times", "nominal_times"):
            if not np.isfinite(a[key]).all():
                raise ValueError(f"Nonfinite input: {key}")
        for key in ("future_valid", "contact_mask", "intervention_valid", "residual_valid",
                    "correction_sample", "stable_sample", "teacher_verified"):
            if a[key].dtype != np.bool_:
                raise ValueError(f"Mask must have bool dtype: {key}")
        if (not np.allclose(np.diff(a["history_times"]), .02, atol=1e-6, rtol=0)
                or not np.allclose(np.diff(a["nominal_times"]), .04, atol=1e-6, rtol=0)
                or not np.allclose(a["history_times"][:, -1], a["decision_time"], atol=1e-6, rtol=0)
                or not np.allclose(a["nominal_times"][:, 0], a["decision_time"], atol=1e-6, rtol=0)):
            raise ValueError("History/future timestamps violate the inference clocks")
        for pid in a["parent_id"]:
            if str(pid) not in self.parents or self.parents[str(pid)]["split"] != split:
                raise ValueError("Window parent does not belong to this split")
        if np.any(a["risk_source"] != "nominal") or np.any(a["nominal_snapshot"] == ""):
            raise ValueError("Risk labels require nominal branch snapshot provenance")
        correction, stable = a["correction_sample"], a["stable_sample"]
        if (correction & stable).any():
            raise ValueError("Correction/identity sets overlap")
        if np.any(correction & (~a["teacher_verified"] |
                               (a["nominal_snapshot"] != a["teacher_snapshot"]))):
            raise ValueError("Correction requires a verified teacher from the SAME snapshot")
        valid = a["future_valid"]
        contact_valid = valid & a["contact_mask"]
        # Only arms/hands can receive phase-dependent grasp-contact labels.
        if a["contact_mask"][:, :, 2:].any():
            raise ValueError("Torso/lower-body contacts are not grasp-loss labels")
        closed_phase = a["context"][:, :, PHASES.index("close"):PHASES.index("hold") + 1].sum(-1) > .5
        if np.any(a["contact_mask"] & ~closed_phase[..., None]):
            raise ValueError("Open-hand phases cannot supervise grasp loss")
        for key, mask in (("track_target", valid), ("contact_target", contact_valid),
                          ("balance_target", valid), ("intervention_target", a["intervention_valid"])):
            values = a[key][mask]
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid available target: {key}")
            if key != "track_target" and (values > 1).any():
                raise ValueError(f"Probability target outside [0,1]: {key}")
        violations = valid & ((a["track_target"] >= 1) | (a["balance_target"] > .5)
                              | (a["contact_mask"] & (a["contact_target"] > .5)))
        # Verified same-state recovery is an independent intervention-needed
        # label, even when short-horizon tracking/contact proxies stay below
        # their thresholds. Auxiliary risk heads remain nominal-only.
        positive = violations.any((1, 2)) | correction
        expected_valid = positive | valid.all((1, 2))
        if (not np.array_equal(expected_valid, a["intervention_valid"])
                or np.any(a["intervention_target"][expected_valid] != positive[expected_valid])):
            raise ValueError("Intervention labels must reflect nominal violations or verified recovery; mask censored negatives")
        if np.any(stable & (~valid.all((1, 2)) | positive)):
            raise ValueError("Stable identity samples require a complete stable nominal future")
        selected = correction | stable
        if np.any(selected & ~a["residual_valid"].any(-1)):
            raise ValueError("Selected residual samples need available targets")
        mask = selected[:, None] & a["residual_valid"]
        offsets = a["offset_target"][mask]
        nonarms = [i for i in range(29) if i not in ARM_INDICES]
        if (not np.isfinite(offsets).all() or (np.abs(offsets) > config.max_offset + 1e-6).any()
                or (offsets[:, nonarms] != 0).any()
                or (a["offset_target"][stable[:, None] & a["residual_valid"]] != 0).any()):
            raise ValueError("Correction target violates bounds, arm mask or identity set")

    def __len__(self):
        return len(self.arrays["history"])

    def __getitem__(self, index):
        return {key: torch.as_tensor(np.array(self.arrays[key][index], copy=True))
                .to(dtype=torch.bool if self.arrays[key].dtype == np.bool_ else torch.float32)
                for key in INPUT_KEYS + TARGET_KEYS}


class Normalizer(torch.nn.Module):
    """Fit continuous inputs on TRAIN windows only; context remains in documented units."""
    def __init__(self):
        super().__init__()
        for key, width in (("history", STATE_DIM), ("nominal", NOMINAL_DIM)):
            self.register_buffer(key + "_mean", torch.zeros(width))
            self.register_buffer(key + "_std", torch.ones(width))

    @classmethod
    def fit(cls, training_dataset):
        if any(training_dataset.parents[str(pid)]["split"] != "train"
               for pid in training_dataset.arrays["parent_id"]):
            raise ValueError("Normalization can only be fitted on training parents")
        result = cls()
        for key in ("history", "nominal"):
            values = training_dataset.arrays[key].astype(np.float64)
            getattr(result, key + "_mean").copy_(torch.from_numpy(values.mean((0, 1))).float())
            getattr(result, key + "_std").copy_(torch.from_numpy(values.std((0, 1)).clip(1e-3)).float())
        # Binary contacts and categorical phases keep their exact encoding,
        # including phases that have zero variance in a small pilot split.
        for region in (slice(79, 81), slice(-len(PHASES), None)):
            result.history_mean[region] = 0
            result.history_std[region] = 1
        return result

    def forward(self, batch):
        return {key: ((batch[key] - getattr(self, key + "_mean")) / getattr(self, key + "_std")
                      if key != "context" else batch[key]) for key in INPUT_KEYS}
