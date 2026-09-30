"""Convert an ARDY reference into SONIC deploy CSV motion data."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _write_csv(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    np.savetxt(path, values.reshape(values.shape[0], -1), delimiter=",", header="values", comments="")


def _root_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(quat_xyzw)
    result = np.zeros((len(rotations), 3), dtype=np.float64)
    if len(rotations) > 1:
        delta = rotations[1:] * rotations[:-1].inv()
        result[1:] = delta.as_rotvec() / dt
        result[0] = result[1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="ARDY reference.npz")
    parser.add_argument("--motion-csv", type=Path, required=True, help="ARDY motion.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="deploy motion root directory")
    parser.add_argument("--name", default="ardy_kick")
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat"], dtype=np.float64)
        times = np.asarray(data["times"], dtype=np.float64)

    motion = np.loadtxt(args.motion_csv, delimiter=",", ndmin=2)
    if motion.shape[1] != 36:
        raise ValueError(f"Expected 36 motion.csv columns, got {motion.shape[1]}")
    if len(motion) != len(times):
        source_times = np.linspace(times[0], times[-1], len(motion))
        root_pos = np.column_stack(
            [np.interp(times, source_times, motion[:, axis]) for axis in range(3)]
        )
    else:
        root_pos = motion[:, :3]

    if joint_pos.shape != joint_vel.shape or joint_pos.shape[1] != 29:
        raise ValueError("reference.npz must contain joint_pos and joint_vel with shape [N, 29]")
    if body_quat.shape != (len(times), 4):
        raise ValueError("reference.npz body_quat must have shape [N, 4] in wxyz order")

    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    dt = float(np.median(np.diff(times)))
    root_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = _root_angular_velocity(body_quat, dt)

    _write_csv(out / "joint_pos.csv", joint_pos)
    _write_csv(out / "joint_vel.csv", joint_vel)
    _write_csv(out / "body_pos.csv", root_pos[:, None, :])
    _write_csv(out / "body_quat.csv", body_quat[:, None, :])
    _write_csv(out / "body_lin_vel.csv", root_vel[:, None, :])
    _write_csv(out / "body_ang_vel.csv", root_ang_vel[:, None, :])

    metadata = (
        f"Metadata for: {args.name}\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[ 0 ]\n\n"
        f"Total timesteps: {len(times)}\n\n"
        f"Data arrays summary:\n"
        f"  joint_pos: {joint_pos.shape} (float64)\n"
        f"  joint_vel: {joint_vel.shape} (float64)\n"
        f"  body_pos_w: {(len(times), 1, 3)} (float64)\n"
        f"  body_quat_w: {(len(times), 1, 4)} (float64)\n"
    )
    (out / "metadata.txt").write_text(metadata)
    (out / "info.txt").write_text(
        json.dumps(
            {
                "source_reference": str(args.reference),
                "source_motion": str(args.motion_csv),
                "fps": 1.0 / dt,
                "frames": len(times),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(times)} frames at {1.0 / dt:.3f} Hz: {out}")


if __name__ == "__main__":
    main()
