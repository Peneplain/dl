"""Protect bring-up evidence and the source-only upload boundary."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from baseline.common import (LOCK, asset_files, checked_checkout, sha256,
                             validate_asset_layout, verify_assets, verify_download_metadata, write_json)
from scripts.fetch_baseline import checkout, main as fetch_main
from scripts.package_baseline import source_files


class BaselineBringupTests(unittest.TestCase):
    def sonic_fixture(self, root):
        directory = root / "sonic"
        directory.mkdir()
        source = json.loads(LOCK.read_text())["sonic"]
        for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"):
            (directory / name).write_bytes(b"fixture, not model weights")
        write_json(directory / "config.json", {
            "schema_version": 1, "model_id": source["repo_id"],
            "variants": {"default": {"deployment_files": [
                "model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]}}})
        manifest = {"source": source, "files": {name: sha256(path)
                    for name, path in asset_files(directory).items()}}
        write_json(root / "sonic.manifest.json", manifest)
        return manifest

    def test_asset_corruption_and_wrong_release_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.sonic_fixture(root)
            weight = root / "sonic/model_encoder.onnx"
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

    def test_manifest_cannot_omit_loaded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.sonic_fixture(root)
            manifest["files"].pop("model_decoder.onnx")
            write_json(root / "sonic.manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "differ from"):
                verify_assets(root, "sonic")

    def test_llama_shards_must_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = json.loads(LOCK.read_text())["llama_base"]
            write_json(root / "config.json", {"model_type": "llama"})
            write_json(root / "model.safetensors.index.json", {
                "weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
            (root / "model-1.safetensors").write_bytes(b"first shard")
            with self.assertRaisesRegex(FileNotFoundError, "model-2"):
                validate_asset_layout(root, "llama_base", source)
            (root / "model-2.safetensors").write_bytes(b"second shard")
            validate_asset_layout(root, "llama_base", source)

    def test_completed_download_needs_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sonic_fixture(root)
            with patch.object(sys, "argv", ["fetch_baseline.py", "--only", "sonic", "--assets", directory]), \
                    patch.dict(sys.modules, {"huggingface_hub": None}):
                fetch_main()

    def test_failed_download_preserves_manifest(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sonic_fixture(root)
            path = root / "sonic.manifest.json"
            original = path.read_bytes()
            (root / "sonic/model_decoder.onnx").write_bytes(b"incomplete")
            def unavailable(**kwargs):
                raise ConnectionError("offline")
            with patch.object(sys, "argv", ["fetch_baseline.py", "--only", "sonic", "--assets", directory]), \
                    patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=unavailable)}):
                with self.assertRaises(ConnectionError):
                    fetch_main()
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "changed or incomplete"):
                verify_assets(root, "sonic")

    def test_offline_registration_requires_matching_download_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.sonic_fixture(root)
            (root / "sonic.manifest.json").unlink()
            argv = ["fetch_baseline.py", "--only", "sonic", "--assets", directory, "--offline"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"huggingface_hub": None}):
                with self.assertRaisesRegex(ValueError, "No Hugging Face download metadata"):
                    fetch_main()
                for name, digest in manifest["files"].items():
                    metadata = root / "sonic/.cache/huggingface/download" / (name + ".metadata")
                    metadata.parent.mkdir(parents=True, exist_ok=True)
                    metadata.write_text(manifest["source"]["revision"] + "\n" + digest + "\n0\n")
                fetch_main()
                verify_assets(root, "sonic")
                metadata.write_text("wrong-revision\n" + digest + "\n0\n")
                with self.assertRaisesRegex(ValueError, "revision does not match"):
                    verify_download_metadata(root / "sonic", manifest["source"], manifest["files"])
                metadata.write_text(manifest["source"]["revision"] + "\n" + "0" * 64 + "\n0\n")
                with self.assertRaisesRegex(ValueError, "ETag"):
                    verify_download_metadata(root / "sonic", manifest["source"], manifest["files"])

    def test_hf_cached_snapshot_supports_offline_verification(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            source = {"repo_id": "test/model", "revision": "pinned-revision"}
            root = Path(directory) / "models--test--model"
            snapshot = root / "snapshots/pinned-revision"
            snapshot.mkdir(parents=True)
            (root / "blobs").mkdir()
            data = b"fixture"
            git_hash = hashlib.sha1(b"blob 7\0" + data).hexdigest()
            blob = root / "blobs" / git_hash
            blob.write_bytes(data)
            (snapshot / "config.json").symlink_to(blob)
            verify_download_metadata(snapshot, source, {"config.json": sha256(blob)})

    def test_online_retry_cannot_relabel_corrupt_cached_files(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.sonic_fixture(root)
            manifest_path = root / "sonic.manifest.json"
            original = manifest_path.read_bytes()
            for name, digest in manifest["files"].items():
                metadata = root / "sonic/.cache/huggingface/download" / (name + ".metadata")
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(manifest["source"]["revision"] + "\n" + digest + "\n0\n")
            (root / "sonic/model_decoder.onnx").write_bytes(b"corrupt cached weight")
            with patch.object(sys, "argv", ["fetch_baseline.py", "--only", "sonic", "--assets", directory]), \
                    patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: None)}):
                with self.assertRaisesRegex(ValueError, "ETag"):
                    fetch_main()
            self.assertEqual(manifest_path.read_bytes(), original)

    def test_archive_excludes_weights_credentials_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scripts/run_ardy.py", "scripts/ardy_service.py",
                         "scripts/prepare_deploy_motion.py", "scripts/__pycache__/foo.py", "scripts/secret.pt",
                         ".env", ".env.secret", "checkpoints/model.onnx", "output/run.json",
                         "third_party/source.py", "README.md", "configs/baseline.lock.json",
                         "risk_residual/models/model.py", "scripts/train_risk.py",
                         "configs/default.yaml", "docs/experiments.md", "baseline/adapters/joints 2.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            actual = {str(path.relative_to(root)) for path in source_files(root)}
            self.assertEqual(actual, {"README.md", "scripts/run_ardy.py", "scripts/ardy_service.py",
                                      "scripts/prepare_deploy_motion.py", "configs/baseline.lock.json"})

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
            # A non-repository child must not resolve to this parent checkout.
            child = root / "child"
            child.mkdir()
            (child / "partial.txt").write_text("interrupted source setup")
            with self.assertRaisesRegex(RuntimeError, "separate Git checkout"):
                checked_checkout(child, commit)
            with self.assertRaisesRegex(RuntimeError, "separate Git checkout"):
                checkout(child, {"url": "https://example.invalid/repo.git", "commit": commit})
            path.write_text("local modification")
            with self.assertRaisesRegex(RuntimeError, "dirty=True"):
                checked_checkout(root, commit)

    def test_hydrated_lfs_files_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            path = root / "mesh.STL"
            data = b"binary STL contents"
            digest = hashlib.sha256(data).hexdigest()
            path.write_text("version https://git-lfs.github.com/spec/v1\n"
                            f"oid sha256:{digest}\nsize {len(data)}\n")
            subprocess.run(["git", "-C", directory, "add", "mesh.STL"], check=True)
            subprocess.run(["git", "-C", directory, "-c", "user.name=Test", "-c",
                            "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                            "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"], text=True).strip()
            path.write_bytes(data)
            self.assertEqual(checked_checkout(root, commit), commit)

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
