"""Read-only label diagnosis on train/val; never select thresholds or relabel data.

The optional source plan recovers nominal/clean cohorts from the converter's
documented concatenation order. Physical stable-hold candidates additionally
require recorded clearance/opposing contacts throughout the prediction window.
Successful episode outcomes alone do not make reach/lower/lift windows stable.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from risk_residual.config import BODY_GROUPS, PHASES


KEYS = ("parent_id", "decision_time", "nominal_times", "history", "nominal", "context",
        "future_valid", "track_target", "contact_mask", "contact_target", "balance_target",
        "intervention_valid", "intervention_target", "correction_sample")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(handle)


def _digest(handle):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def reasons(arrays):
    valid = arrays["future_valid"].astype(bool)
    track = valid & (arrays["track_target"] >= 1)
    contact = valid & arrays["contact_mask"].astype(bool) & (arrays["contact_target"] > .5)
    balance = valid & (arrays["balance_target"] > .5)
    correction = arrays["correction_sample"].astype(bool)
    positive = (track | contact | balance).any((1, 2)) | correction
    expected_valid = positive | valid.all((1, 2))
    if (not np.array_equal(expected_valid, arrays["intervention_valid"])
            or np.any(arrays["intervention_target"][expected_valid] != positive[expected_valid])):
        raise ValueError("Stored intervention labels disagree with nominal proxies/recovery masks")
    return {"tracking": track.any((1, 2)), "contact": contact.any((1, 2)),
            "balance": balance.any((1, 2)), "verified_recovery": correction,
            "body_tracking": track.any(1)}


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(zip(("p50", "p90", "p95", "p99"),
                    map(float, np.quantile(values, (.5, .9, .95, .99)))))


def branch_layout(arrays, parents):
    """Fail on ambiguous ordering instead of inventing branch provenance."""
    ids, times = arrays["parent_id"], arrays["decision_time"]
    cohort = np.full(len(ids), "unknown", dtype="<U24")
    outcome = np.zeros(len(ids), dtype=bool)
    attempt = np.full(len(ids), "", dtype=object)
    origin = np.full(len(ids), "", dtype=object)
    for pid in dict.fromkeys(map(str, ids)):
        indices = np.flatnonzero(ids == pid)
        if not np.all(np.diff(indices) == 1):
            raise ValueError("Source-aware diagnosis requires contiguous converter parent blocks")
        parent = parents[pid]
        path = Path(parent["nominal"])
        recorded = path / "report.json"
        if parent.get("source_report_sha256") != digest(recorded):
            raise ValueError("Nominal report changed since source indexing")
        report = json.loads(recorded.read_text())
        if not parent.get("pair"):
            if np.any(np.diff(times[indices]) <= 0):
                raise ValueError("Original branch decisions must increase")
            cohort[indices] = "original"
            outcome[indices] = bool(report.get("task_success"))
            attempt[indices] = str(path)
            origin[indices] = str(path.resolve())
            continue
        pair = json.loads(Path(parent["pair"]).read_text())
        reset = np.flatnonzero(np.diff(times[indices]) < 0)
        if len(reset) != 1:
            raise ValueError("Cannot establish nominal/clean boundary from converter ordering")
        cut = int(reset[0] + 1)
        nominal, clean = indices[:cut], indices[cut:]
        if np.any(np.diff(times[nominal]) <= 0) or np.any(np.diff(times[clean]) <= 0):
            raise ValueError("Branch decisions must increase")
        teacher_report = Path(pair["branches"]["teacher"]["report"])
        if digest(teacher_report) != pair["branches"]["teacher"]["report_sha256"]:
            raise ValueError("Clean branch report changed since collection")
        teacher = json.loads(teacher_report.read_text())
        cohort[nominal], cohort[clean] = "paired_nominal", "teacher_clean"
        outcome[nominal] = bool(report.get("task_success"))
        outcome[clean] = bool(teacher.get("task_success"))
        attempt[nominal], attempt[clean] = str(path), str(teacher_report.parent)
        origin[indices] = str(Path(pair["source_attempt"]).resolve())
    return cohort, outcome, attempt, origin


def physical_hold(arrays, cohort, success, attempts):
    """Use only successful clean/original holds with measured physical evidence.

    Candidate selection is independent of tracking thresholds. It requires all
    future samples, 5 cm block clearance, opposing forces > .01 N and no recorded
    balance violation. These observed windows diagnose the label proxy; they are
    not guaranteed counterfactual negatives for every possible correction.
    """
    phase = arrays["context"][:, 0, :len(PHASES)].argmax(-1)
    candidates = (success & np.isin(cohort, ("original", "teacher_clean"))
                  & (phase == PHASES.index("hold"))
                  & arrays["future_valid"].all((1, 2))
                  & ~(arrays["balance_target"] > .5).any((1, 2)))
    accepted = np.zeros(len(candidates), dtype=bool)
    for attempt in dict.fromkeys(attempts[candidates]):
        path = Path(attempt) / "task.csv"
        report = json.loads((Path(attempt) / "report.json").read_text())
        if report.get("outputs_sha256", {}).get("task.csv") != digest(path):
            raise ValueError("Physical task evidence changed since collection")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        times = np.asarray([float(row["sim_time"]) for row in rows])
        measured = np.asarray([[float(row[key]) for key in
                                ("lowest_clearance_m", "thumb_force_n", "finger_force_n")]
                               for row in rows])
        if not np.isfinite(measured).all():
            raise ValueError("Physical clearance/contact evidence must be finite")
        safe = (measured[:, 0] >= .05) & (measured[:, 1:] > .01).all(1)
        if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Physical evidence times must increase")
        indices = np.flatnonzero(candidates & (attempts == attempt))
        future = arrays["nominal_times"][indices][:, [0, -1]]
        positions = np.searchsorted(times, future, side="right") - 1
        available = positions >= 0
        positions = np.clip(positions, 0, len(times) - 1)
        available &= (future - times[positions] <= .005001)
        # Inspect every recorded 5 ms physics sample between the endpoints,
        # not merely the eight 40 ms labels. A brief intervening slip is unsafe.
        unsafe = ~safe
        unsafe[1:] |= np.diff(times) > .005001
        failures = np.concatenate(([0], np.cumsum(unsafe)))
        continuous = (failures[positions[:, 1] + 1] - failures[positions[:, 0]]) == 0
        accepted[indices] = available.all(1) & continuous
    return accepted


def summarize(arrays, selected, thresholds, triggers, *, origins=None):
    selected = np.asarray(selected, dtype=bool)
    valid = arrays["future_valid"][selected].astype(bool)
    raw = arrays["track_target"][selected] * thresholds
    label_valid = arrays["intervention_valid"][selected]
    positive = arrays["intervention_target"][selected] > .5
    result = {"windows": int(selected.sum()),
              "parent_variants": len(set(map(str, arrays["parent_id"][selected]))),
              "risk_positive": int((label_valid & positive).sum()),
              "risk_negative": int((label_valid & ~positive).sum()),
              "risk_censored": int((~label_valid).sum()),
              "reason_windows": {key: int(value[selected].sum()) for key, value in triggers.items()
                                 if key != "body_tracking"},
              "body_tracking_violation_windows": dict(zip(BODY_GROUPS,
                  map(int, triggers["body_tracking"][selected].sum(0)))),
              "tracking_rad": {}, "max_horizon_tracking_rad": {}}
    if origins is not None:
        result["original_episodes"] = len(set(origins[selected]))
        result["per_episode_max_horizon_p95_rad"] = {}
    for body, name in enumerate(BODY_GROUPS):
        result["tracking_rad"][name] = quantiles(raw[:, :, body][valid[:, :, body]])
        complete = valid[:, :, body].all(1)
        result["max_horizon_tracking_rad"][name] = quantiles(raw[complete, :, body].max(1))
        if origins is not None:
            maximum = raw[complete, :, body].max(1)
            selected_origins = origins[selected][complete]
            episode_p95 = [float(np.quantile(maximum[selected_origins == origin], .95))
                           for origin in dict.fromkeys(selected_origins)]
            result["per_episode_max_horizon_p95_rad"][name] = quantiles(episode_p95)
    if selected.any():
        delta = arrays["nominal"][selected, 0, :29] - arrays["history"][selected, -1, :29]
        result["current_joint_error_rad"] = {
            name: {"signed_median": float(np.median(delta[:, joint])),
                   "mae": float(np.mean(np.abs(delta[:, joint])))}
            for joint, name in enumerate(ISAACLAB_JOINT_NAMES)}
        result["nominal_horizon_position_spread_max_rad"] = float(np.max(np.abs(
            arrays["nominal"][selected, :, :29] - arrays["nominal"][selected, :1, :29])))
    return result


def sensitivity(arrays, thresholds, candidates, stable):
    """Diagnostic grid only: never choose a winner or save replacement labels."""
    raw = arrays["track_target"] * thresholds
    valid = arrays["future_valid"].astype(bool)
    forced = arrays["correction_sample"].astype(bool)
    physical = (valid & ((arrays["balance_target"] > .5)
                 | (arrays["contact_mask"] & (arrays["contact_target"] > .5)))).any((1, 2))
    rows = []
    for threshold in candidates:
        threshold = np.asarray(threshold, dtype=float)
        if threshold.shape != (4,) or not np.isfinite(threshold).all() or np.any(threshold <= 0):
            raise ValueError("Each diagnostic candidate needs four positive thresholds")
        tracking = (valid & (raw >= threshold)).any((1, 2))
        positive = tracking | physical | forced
        rows.append({"tracking_thresholds_rad": list(map(float, threshold)),
                     "positive": int(positive.sum()),
                     "negative": int((~positive & valid.all((1, 2))).sum()),
                     "censored": int((~positive & ~valid.all((1, 2))).sum()),
                     "physical_stable_hold_windows": int(stable.sum()),
                     "physical_stable_hold_positive": int((stable & positive).sum()),
                     "verified_recovery_windows": int(forced.sum()),
                     "verified_recovery_tracking_trigger": int((forced & tracking).sum()),
                     "verified_recovery_physical_trigger": int((forced & physical).sum()),
                     "verified_recovery_kept_positive": int((forced & positive).sum())})
    return rows


def diagnose(manifest_path, *, splits=("train", "val"), source_path=None, candidates=()):
    if any(split not in ("train", "val") for split in splits):
        raise ValueError("Threshold diagnosis is restricted to train/val; test stays held out")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    thresholds = np.asarray(manifest["provenance"]["label_thresholds"]["tracking"], dtype=float)
    if thresholds.shape != (4,) or not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("Manifest needs four positive tracking thresholds")
    parents = None
    if source_path is not None:
        source_path = Path(source_path).resolve()
        if digest(source_path) != manifest["provenance"]["source_plan_sha256"]:
            raise ValueError("Source plan is not the manifest's collection provenance")
        parents = {parent["parent_id"]: parent for parent in json.loads(source_path.read_text())["parents"]}
        pinned = {entry["parent_id"]: entry["source_files_sha256"]
                  for entry in manifest["provenance"].get("source_files", ())}
        for pid, parent in parents.items():
            if parent.get("pair") and (not pinned.get(pid, {}).get("pair.json")
                                      or digest(parent["pair"]) != pinned[pid]["pair.json"]):
                raise ValueError("Pair cohort evidence changed since dataset conversion")
    result = {"schema": "dl-label-diagnosis-v1", "manifest_sha256": digest(manifest_path),
              "source_plan_sha256": digest(source_path) if source_path else None,
              "tracking_thresholds_rad": list(map(float, thresholds)),
              "scope": "Read-only train/val proxy diagnosis; no relabeling or automatic calibration. "
                       "Window counts are correlated; cohorts use converter branch order. "
                       "Stable holds require measured clearance/contact plus successful outcomes.",
              "splits": {}}
    for split in splits:
        archive_path = (manifest_path.parent / manifest["windows"][split]).resolve()
        if not archive_path.is_relative_to(manifest_path.parent):
            raise ValueError("Split archive must be inside the dataset")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in KEYS}
        triggers = reasons(arrays)
        phase = arrays["context"][:, 0, :len(PHASES)].argmax(-1)
        cohort, success, attempts, origins = branch_layout(arrays, parents) if parents is not None else (
            np.full(len(phase), "unknown"), None, None, None)
        stable = physical_hold(arrays, cohort, success, attempts) if parents is not None else np.zeros(len(phase), bool)
        overall = summarize(arrays, np.ones(len(phase), bool), thresholds, triggers, origins=origins)
        overall["phases"] = {name: summarize(arrays, phase == index, thresholds, triggers, origins=origins)
                             for index, name in enumerate(PHASES) if (phase == index).any()}
        overall["cohorts"] = {}
        for category in sorted(set(cohort)):
            for outcome in (False, True) if success is not None else (None,):
                selected = (cohort == category) & ((success == outcome) if outcome is not None else True)
                if not selected.any():
                    continue
                key = category + ("_success" if outcome else "_failure") if outcome is not None else category
                summary = summarize(arrays, selected, thresholds, triggers, origins=origins)
                summary["phases"] = {name: summarize(arrays, selected & (phase == index), thresholds,
                    triggers, origins=origins) for index, name in enumerate(PHASES)
                    if (selected & (phase == index)).any()}
                overall["cohorts"][key] = summary
        overall["physical_stable_hold"] = summarize(arrays, stable, thresholds, triggers, origins=origins)
        overall["threshold_sensitivity"] = sensitivity(arrays, thresholds, candidates, stable)
        result["splits"][split] = overall
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--candidate-thresholds", type=float, nargs=4, action="append", default=[],
                        metavar=("LEFT", "RIGHT", "TORSO", "LOWER"),
                        help="Read-only diagnostic sensitivity; does not select or change labels")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.manifest, splits=args.splits, source_path=args.sources,
                              candidates=args.candidate_thresholds), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
