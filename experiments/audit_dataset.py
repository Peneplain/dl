"""Audit converted NPZ supervision without training or simulation."""

import argparse
import json
from pathlib import Path

from baseline.common import write_json
from risk_residual.audit import readiness, supervision_summary
from risk_residual.config import ModelConfig
from risk_residual.data import WindowDataset


def audit(manifest):
    summaries = {split: supervision_summary(WindowDataset(manifest, split, ModelConfig()).arrays)
                 for split in ("train", "val", "test")}
    return {"splits": summaries, "risk": readiness(summaries, "risk"),
            "residual": readiness(summaries, "residual"), "physics_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require", choices=("risk", "residual"))
    args = parser.parse_args()
    result = audit(args.data)
    if args.out:
        if args.out.exists():
            parser.error("Use a fresh audit report path")
        write_json(args.out, result)
    print(json.dumps(result), flush=True)
    if args.require and not result[args.require]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
