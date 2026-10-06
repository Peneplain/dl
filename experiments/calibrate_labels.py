"""Derive an exploratory tracking-threshold candidate from TRAIN evidence only.

For each body, take the diagnosis report's across-episode p95 of the per-episode
p95 of maximum-horizon tracking error. Round upward to .01 rad, with the old
threshold as a floor. This writes a fresh provenance artifact, never labels.
Validation counts provide context only; no validation statistic selects values.
"""

import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import re

from risk_residual.config import BODY_GROUPS


MIN_ORIGINAL_EPISODES = 10
RAD_STEP = Decimal("0.01")


def positive_number(value, name, *, zero_allowed=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero_allowed and value == 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero_allowed else 'positive'} number")
    return Decimal(str(value))


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def pinned_hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a recorded SHA-256 hash")
    return value


def derive(report):
    """Use train only; require physically stable evidence from independent origins."""
    if report.get("schema") != "dl-label-diagnosis-v1":
        raise ValueError("Input must be a versioned label diagnosis report")
    splits = report.get("splits", {})
    if "train" not in splits or any(split not in ("train", "val") for split in splits):
        raise ValueError("Input diagnosis must use train/val only; test stays held out")
    manifest_hash = pinned_hash(report.get("manifest_sha256"), "manifest_sha256")
    source_hash = pinned_hash(report.get("source_plan_sha256"), "source_plan_sha256")
    old = report.get("tracking_thresholds_rad", ())
    if not isinstance(old, list) or len(old) != len(BODY_GROUPS):
        raise ValueError("Diagnosis must record four original tracking thresholds")
    floors = [positive_number(value, "Original threshold") for value in old]
    stable = splits["train"].get("physical_stable_hold", {})
    episodes = count(stable.get("original_episodes"), "Train stable original episodes")
    windows = count(stable.get("windows"), "Train physically stable hold windows")
    if episodes < MIN_ORIGINAL_EPISODES or windows == 0 or episodes > windows:
        raise ValueError("Calibration needs stable hold windows from at least 10 independent original episodes")
    per_episode = stable.get("per_episode_max_horizon_p95_rad", {})
    candidates, derivation = [], {}
    for body, floor in zip(BODY_GROUPS, floors):
        summary = per_episode.get(body)
        if not isinstance(summary, dict):
            raise ValueError(f"Missing per-episode tracking distribution for {body}")
        quantiles = [positive_number(summary.get(key), f"{body} {key}", zero_allowed=True)
                     for key in ("p50", "p90", "p95", "p99")]
        if any(left > right for left, right in zip(quantiles, quantiles[1:])):
            raise ValueError(f"Unordered episode quantiles for {body}")
        raw_p95 = quantiles[2]
        rounded = raw_p95.quantize(RAD_STEP, rounding=ROUND_CEILING)
        candidate = max(floor, rounded).quantize(RAD_STEP, rounding=ROUND_CEILING)
        candidates.append(float(candidate))
        derivation[body] = {"train_across_episode_p95_rad": float(raw_p95),
                            "rounded_train_p95_rad": float(rounded),
                            "previous_tracking_threshold_rad": float(floor),
                            "candidate_tracking_threshold_rad": float(candidate)}
    validation_check = None
    if "val" in splits:
        observed = splits["val"].get("physical_stable_hold", {})
        validation_check = {
            "used_for_selection": False,
            "physically_stable_hold_windows": count(observed.get("windows"), "Val stable hold windows"),
            "original_episodes": count(observed.get("original_episodes"), "Val stable original episodes"),
            "positive_at_previous_thresholds": count(observed.get("risk_positive"), "Val prior proxy positives"),
            "scope": "Counts under the existing manifest only; candidate labels are not evaluated here."}
        if validation_check["positive_at_previous_thresholds"] > validation_check["physically_stable_hold_windows"]:
            raise ValueError("Validation proxy positives exceed available stable hold windows")
    return {"schema": "dl-label-calibration-v1", "exploratory": True, "frozen": True,
            "split": "train", "calibration_split": "train", "test_used": False,
            "labels_written": False, "automatic_application": False,
            "input_manifest_sha256": manifest_hash, "input_source_plan_sha256": source_hash,
            "body_groups": list(BODY_GROUPS), "previous_tracking_thresholds_rad": list(map(float, floors)),
            "candidate_tracking_thresholds_rad": candidates,
            "train_evidence": {"physically_stable_hold_windows": windows,
                               "independent_original_episodes": episodes},
            "rule": "For each body: train across-episode p95 of per-episode p95 of window maximum "
                    "horizon tracking error; ceil to .01 rad; retain original threshold as a floor; "
                    "ceil the resulting maximum to .01 rad. Minimum 10 independent original episodes.",
            "derivation": derivation, "validation_check": validation_check,
            "scope": "Exploratory proxy calibration from physically stable successful holds; "
                     "this is not corrected ground truth and does not establish grasp improvement. "
                     "Other task phases are not established safe by this hold calibration.",
            "rebuild_requirements": [
                "Create a new dataset directory/manifest; preserve the source data and old dataset.",
                "Keep verified same-state recovery decisions positive independently of tracking thresholds.",
                "Preserve nominal contact and balance labels, phase-dependent contact masks and missing-future censoring.",
                "Audit the frozen candidate on validation evidence before training; do not retune using test."]}


def calibrate(diagnosis_path, output_path, *, expected_diagnosis_sha256=None):
    diagnosis_path, output_path = Path(diagnosis_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("Choose a fresh calibration output file")
    raw = diagnosis_path.read_bytes()
    diagnosis_hash = hashlib.sha256(raw).hexdigest()
    if expected_diagnosis_sha256 is not None:
        pinned_hash(expected_diagnosis_sha256, "expected diagnosis hash")
        if diagnosis_hash != expected_diagnosis_sha256:
            raise ValueError("Diagnosis SHA-256 differs from the requested input")
    result = derive(json.loads(raw))
    result["input_diagnosis"] = str(diagnosis_path)
    result["input_diagnosis_sha256"] = diagnosis_hash
    result["calibrator_revision_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        handle.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="Fresh exploratory JSON artifact; no dataset is changed")
    parser.add_argument("--expected-diagnosis-sha256")
    args = parser.parse_args()
    result = calibrate(args.diagnosis, args.out,
                       expected_diagnosis_sha256=args.expected_diagnosis_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
