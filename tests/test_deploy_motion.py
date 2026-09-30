import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DeployMotionTests(unittest.TestCase):
    def test_ardy_reference_is_exported_at_deploy_layout(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            reference = work / "reference.npz"
            motion = work / "motion.csv"
            out = work / "deploy"
            times = np.arange(3, dtype=np.float64) * 0.02
            joint_pos = np.arange(87, dtype=np.float32).reshape(3, 29)
            joint_vel = np.ones((3, 29), dtype=np.float32)
            quat = np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32)
            np.savez(reference, times=times, joint_pos=joint_pos,
                     joint_vel=joint_vel, body_quat=quat)
            np.savetxt(motion, np.zeros((3, 36)), delimiter=",")

            result = subprocess.run(
                [sys.executable, "-B", str(root / "scripts/prepare_deploy_motion.py"),
                 "--reference", str(reference), "--motion-csv", str(motion),
                 "--out-dir", str(out), "--name", "kick"],
                cwd=root, check=True, capture_output=True, text=True,
            )
            self.assertIn("Prepared 3 frames at 50.000 Hz", result.stdout)
            directory_path = out / "kick"
            for filename in (
                "joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv",
                "body_lin_vel.csv", "body_ang_vel.csv", "metadata.txt", "info.txt",
            ):
                self.assertTrue((directory_path / filename).is_file())
            self.assertEqual(np.loadtxt(directory_path / "joint_pos.csv", delimiter=",", skiprows=1).shape,
                             (3, 29))
            self.assertIn("Body part indexes:\n[ 0 ]", (directory_path / "metadata.txt").read_text())


if __name__ == "__main__":
    unittest.main()
