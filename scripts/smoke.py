"""Check B0 reference conversion and packet fields using a synthetic CPU fixture."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.sonic import HEADER_SIZE
from baseline.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error("Choose a fresh output directory")
    out.mkdir(parents=True)
    report = {"stage": "baseline_reference_fixture", "status": "running",
              "synthetic_inputs": True, "ardy_sonic_checked": False,
              "physics_executed": False, "task_success": None}
    try:
        times = np.arange(9) / 25
        canonical = np.arange(29)[None, :] * 0.01 + times[:, None] * 0.05
        qpos = np.zeros((len(times), 36))
        qpos[:, 2:4] = 1  # root z=1 and identity wxyz
        qpos[:, 7:] = canonical[:, ::-1]
        np.savetxt(out / "source.csv", qpos, delimiter=",")
        write_json(out / "source_names.json", list(reversed(ISAACLAB_JOINT_NAMES)))
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_reference.py"),
                        "--qpos-csv", str(out / "source.csv"),
                        "--joint-names", str(out / "source_names.json"),
                        "--source-fps", "25", "--out", str(out / "reference.npz"),
                        "--packet"], check=True, cwd=ROOT)
        with np.load(out / "reference.npz", allow_pickle=False) as reference:
            expected_times = np.arange(17) / 50
            np.testing.assert_allclose(reference["times"], expected_times)
            np.testing.assert_allclose(reference["joint_pos"],
                                       np.arange(29)[None, :] * 0.01 + expected_times[:, None] * 0.05,
                                       atol=1e-7)
            np.testing.assert_allclose(reference["joint_vel"], 0.05, atol=2e-6)
            np.testing.assert_array_equal(reference["joint_names"], ISAACLAB_JOINT_NAMES)
            packet = (out / "reference.packet").read_bytes()
            if not packet.startswith(b"pose"):
                raise ValueError("Missing SONIC topic")
            header = json.loads(packet[4:4 + HEADER_SIZE].rstrip(b"\0"))
            if header["v"] != 1 or header["count"] != 17 or header["endian"] != "le":
                raise ValueError("Wrong packet version, count or byte order")
            position = 4 + HEADER_SIZE
            expected = {"joint_pos": reference["joint_pos"], "joint_vel": reference["joint_vel"],
                        "body_quat": reference["body_quat"], "frame_index": np.arange(17),
                        "catch_up": np.array([0])}
            types = {"f32": "<f4", "i64": "<i8", "u8": "u1"}
            if [field["name"] for field in header["fields"]] != list(expected):
                raise ValueError("Unexpected wire fields")
            for field in header["fields"]:
                dtype = np.dtype(types[field["dtype"]])
                count = int(np.prod(field["shape"]))
                value = np.frombuffer(packet, dtype=dtype, count=count, offset=position)
                np.testing.assert_array_equal(value.reshape(field["shape"]), expected[field["name"]])
                position += count * dtype.itemsize
            if position != len(packet):
                raise ValueError("Trailing packet bytes")
        report.update(status="passed", source_frames=9, reference_frames=17)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
