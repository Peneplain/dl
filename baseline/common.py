"""Source and weight provenance for the frozen baseline."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/baseline.lock.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_checkout(path, commit):
    path = Path(path).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if actual != commit or dirty:
        raise RuntimeError(f"Expected clean upstream {commit} at {path}; got {actual}, dirty={bool(dirty)}")
    return actual


def verify_assets(asset_root, key):
    """Require completed downloads and hash the exact files before inference."""
    asset_root = Path(asset_root)
    lock = json.loads(LOCK.read_text())
    manifest_path = asset_root / f"{key}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"] != lock[key] or not manifest.get("files"):
        raise ValueError(f"Asset manifest does not match the source lock: {manifest_path}")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid asset path: {name}")
        if sha256(asset_root / key / relative) != expected:
            raise ValueError(f"Asset changed or incomplete: {key}/{name}")
    return sha256(manifest_path)
