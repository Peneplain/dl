"""Protect bring-up evidence and the source-only upload boundary."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baseline.common import LOCK, checked_checkout, sha256, verify_assets
from scripts.package_baseline import source_files


class BaselineBringupTests(unittest.TestCase):
    def test_asset_corruption_and_wrong_release_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sonic").mkdir()
            weight = root / "sonic/model_encoder.onnx"
            weight.write_bytes(b"fixture, not model weights")
            manifest = {"source": json.loads(LOCK.read_text())["sonic"],
                        "files": {weight.name: sha256(weight)}}
            path = root / "sonic.manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(verify_assets(root, "sonic"), sha256(path))
            weight.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "changed or incomplete"):
                verify_assets(root, "sonic")
            manifest["source"]["revision"] = "wrong-release"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source lock"):
                verify_assets(root, "sonic")

    def test_archive_excludes_weights_credentials_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scripts/run_ardy.py", "scripts/__pycache__/foo.py", "scripts/secret.pt",
                         ".env", ".env.secret", "checkpoints/model.onnx", "artifacts/run.json",
                         "third_party/source.py", "README.md", "configs/baseline.lock.json",
                         "risk_residual/models/model.py", "scripts/train_risk.py",
                         "configs/default.yaml", "docs/experiments.md"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            actual = {str(path.relative_to(root)) for path in source_files(root)}
            self.assertEqual(actual, {"README.md", "scripts/run_ardy.py", "configs/baseline.lock.json"})

    def test_archive_rejects_source_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").symlink_to(LOCK)
            with self.assertRaisesRegex(ValueError, "symlink"):
                source_files(root)

    def test_dirty_upstream_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            path = root / "model.py"
            path.write_text("original")
            subprocess.run(["git", "-C", directory, "add", "model.py"], check=True)
            subprocess.run(["git", "-C", directory, "-c", "user.name=Test", "-c",
                            "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                            "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(checked_checkout(root, commit), commit)
            path.write_text("local modification")
            with self.assertRaisesRegex(RuntimeError, "dirty=True"):
                checked_checkout(root, commit)

    def test_missing_assets_leave_failure_report_and_nonzero_exit(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            result = subprocess.run([sys.executable, str(root / "scripts/check_sonic_onnx.py"),
                                     "--assets", directory, "--out", str(out)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["physics_executed"])
            self.assertIsNone(report["task_success"])


if __name__ == "__main__":
    unittest.main()
