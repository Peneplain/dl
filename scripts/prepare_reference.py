"""Adapt an ARDY G1 MuJoCo-qpos CSV to a named, resampled SONIC v1 reference."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from baseline.adapters.joints import ISAACLAB_JOINT_NAMES  # noqa: E402
from baseline.adapters.reference import ReferenceSequence  # noqa: E402
from baseline.adapters.sonic import JointStreamEncoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpos-csv", required=True)
    parser.add_argument("--joint-names", required=True,
                        help="JSON list of the 29 joint names in the CSV's source order")
    parser.add_argument("--source-fps", type=float, default=25)
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output .npz; optional sibling .packet is offline")
    parser.add_argument("--packet", action="store_true", help="Write bytes only; opens no network socket")
    args = parser.parse_args()
    if args.source_fps <= 0:
        parser.error("--source-fps must be positive")
    qpos = np.loadtxt(args.qpos_csv, delimiter=",", skiprows=args.skiprows, ndmin=2)
    if qpos.shape[1] != 36:
        parser.error("Expected 36 qpos columns: root xyz, root wxyz, 29 body joints")
    names = json.loads(Path(args.joint_names).read_text())
    reference = ReferenceSequence.from_named_joints(np.arange(len(qpos)) / args.source_fps,
                                                   qpos[:, 7:], qpos[:, 3:7], names)
    times = np.arange(int(np.floor(reference.times[-1] * 50 + 1e-7)) + 1) / 50
    reference = reference.sample(times)
    out = Path(args.out)
    if out.suffix != ".npz":
        parser.error("--out must end in .npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, times=reference.times, joint_pos=reference.joint_pos,
                        joint_vel=reference.velocities(), body_quat=reference.body_quat,
                        joint_names=np.array(ISAACLAB_JOINT_NAMES), fps=np.int64(50))
    if args.packet:
        out.with_suffix(".packet").write_bytes(JointStreamEncoder().encode(reference))
    print(f"Adapted {len(reference.times)} frames at 50 Hz to {out}; no simulation executed.")


if __name__ == "__main__":
    main()
