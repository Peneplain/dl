"""Synthetic forward/backward + two-stage training; never physical evidence."""

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baseline.adapters.joints import ARM_INDICES
from baseline.common import LOCK, sha256, write_json
from experiments.train import train
from risk_residual.config import CONTEXT_DIM, NOMINAL_DIM, PHASES, SCHEMA, STATE_DIM, ModelConfig


def make_fixture(root, config, *, count=8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(7)
    parents = []
    for split in ("train", "val", "test"):
        n, h, k = count, config.horizon, config.history_steps
        ids = np.array([f"{split}-{i}" for i in range(n)])
        parents.extend({"parent_id": str(pid), "split": split, "prompt_group": str(pid), "scene_seed": str(pid)}
                       for pid in ids)
        history = rng.normal(0, .1, (n, k, STATE_DIM)).astype(np.float32)
        nominal = rng.normal(0, .1, (n, h, NOMINAL_DIM)).astype(np.float32)
        # Valid wxyz and one-hot phase fields even for interface fixtures.
        history[:, :, 58:62] = [1, 0, 0, 0]
        history[:, :, -len(PHASES):] = np.eye(len(PHASES), dtype=np.float32)[PHASES.index("lift")]
        nominal[:, :, 58:62] = [1, 0, 0, 0]
        context = np.zeros((n, h, CONTEXT_DIM), np.float32)
        context[:, :, PHASES.index("lift")] = 1
        correction = np.arange(n) % 2 == 0
        target = np.zeros((n, h, 29), np.float32)
        for j in ARM_INDICES:
            target[correction, :, j] = .02
        valid = np.ones((n, h, 4), bool)
        contact_mask = np.zeros_like(valid)
        contact_mask[:, :, :2] = True
        balance = np.zeros((n, h, 4), np.float32)
        track = np.where(correction[:, None, None], 1.2, .1) * np.ones((n, h, 4))
        arrays = {
            "history": history, "nominal": nominal, "context": context,
            "decision_time": np.ones(n), "history_times": np.tile(1 - np.arange(k - 1, -1, -1) * .02, (n, 1)),
            "nominal_times": np.tile(1 + np.arange(h) * .04, (n, 1)),
            "future_valid": valid, "contact_mask": contact_mask, "track_target": track.astype(np.float32),
            "contact_target": np.zeros_like(balance), "balance_target": balance,
            "intervention_valid": np.ones(n, bool), "intervention_target": correction.astype(np.float32),
            "offset_target": target, "residual_valid": np.ones((n, h), bool),
            "correction_sample": correction, "stable_sample": ~correction, "parent_id": ids,
            "risk_source": np.array(["nominal"] * n), "teacher_verified": correction,
            "nominal_snapshot": ids, "teacher_snapshot": np.where(correction, ids, ""),
        }
        np.savez_compressed(root / f"{split}.npz", **arrays)
    write_json(root / "manifest.json", {
        "schema": SCHEMA, "synthetic_inputs": True, "parents": parents,
        "windows": {split: f"{split}.npz" for split in ("train", "val", "test")},
        "provenance": {"baseline_lock_sha256": sha256(LOCK), "scene_hashes": ["synthetic-no-scene"],
                        "label_thresholds": {"tracking": [1, 1, 1, 1]}, "collector_revision": "synthetic-v1"}})
    return root / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "synthetic_inputs": True, "physics_executed": False, "task_success": None}
    try:
        config = ModelConfig(width=32, heads=4, layers=1, dropout=0.)
        manifest = make_fixture(args.out / "data", config)
        settings = {"model": asdict(config), "risk_options": {"no_future": False, "no_history": False},
                        "training": {"epochs": 2, "batch_size": 4, "learning_rate": .001, "weight_decay": .01,
                                      "gradient_clip": 1., "correction_fraction": .5, "alpha": 1., "beta": .1,
                                      "auxiliary_weight": 1.}}
        config_path = args.out / "config.json"
        write_json(config_path, settings)
        common = {"config": config_path, "data": manifest, "seed": 7, "device": args.device, "threads": 1,
                      "allow_synthetic": True, "interface": "P"}
        results = {}
        for stage in ("risk", "residual"):
            out = args.out / stage
            out.mkdir()
            options = SimpleNamespace(**common, stage=stage, out=out,
                                      risk=None if stage == "risk" else args.out / "risk/best.pt")
            results[stage] = train(options)
            write_json(out / "report.json", results[stage])
        report.update(status="passed", stages=results, fixture_model=config.to_dict(),
                      note="Reduced-width synthetic pipeline check; not grasp data or full-model/physics evidence")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
