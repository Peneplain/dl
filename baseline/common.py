"""Source and weight provenance for the frozen baseline."""

import hashlib
import json
import re
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
    checked_repo_root(path)
    git = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
    actual = subprocess.check_output(
        git + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        git + ["status", "--porcelain", "--untracked-files=no"],
        text=True)
    dirty = []
    for line in status.splitlines():
        if not line:
            continue
        # A hydrated Git LFS file has an LFS pointer in the committed index and
        # binary contents in the worktree. It is required for local MuJoCo
        # execution, so it is clean with respect to source provenance.
        if line.startswith(" M ") and _is_hydrated_lfs_file(path, line[3:]):
            continue
        dirty.append(line)
    if actual != commit or dirty:
        raise RuntimeError(f"Expected clean upstream {commit} at {path}; got {actual}, dirty={bool(dirty)}")
    return actual


def _is_hydrated_lfs_file(repo: Path, relative_path: str) -> bool:
    """Return whether a worktree change replaces a committed LFS pointer."""
    pointer = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo),
         "show", f"HEAD:{relative_path}"],
        capture_output=True,
    )
    if pointer.returncode or not pointer.stdout.startswith(
        b"version https://git-lfs.github.com/spec/v1\n"
    ):
        return False
    pointer_text = pointer.stdout.decode("ascii", errors="ignore")
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer_text, re.MULTILINE)
    size_match = re.search(r"^size (\d+)$", pointer_text, re.MULTILINE)
    if not oid_match or not size_match:
        return False
    worktree_path = repo / relative_path
    try:
        if not worktree_path.is_file() or worktree_path.stat().st_size != int(size_match.group(1)):
            return False
        return sha256(worktree_path) == oid_match.group(1)
    except OSError:
        return False


def checked_repo_root(path):
    """Do not let Git silently discover the project's parent repository."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing upstream source: {path}; run fetch_baseline.py --only sources or import its Git bundle")
    result = subprocess.run(["git", "-c", f"safe.directory={path}", "-C", str(path),
                             "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True)
    if result.returncode or Path(result.stdout.strip()).resolve() != path:
        raise RuntimeError(f"Expected a separate Git checkout at {path}")


def asset_files(directory):
    return {str(p.relative_to(directory)): p for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(directory).parts}


def verify_download_metadata(download_dir, source, files):
    """Validate HF local-dir or snapshot provenance before offline registration.

    A directory name alone is not evidence of a pinned download. Require the
    downloader's revision/ETag records and match each file against its ETag.
    ``files`` maps paths relative to download_dir to their computed SHA256.
    """
    download_dir = Path(download_dir).resolve()
    cache_name = "models--" + source["repo_id"].replace("/", "--")
    snapshot = (download_dir.name == source["revision"] and
                download_dir.parent.name == "snapshots" and
                download_dir.parent.parent.name == cache_name)
    for name, digest in files.items():
        path = download_dir / name
        if snapshot and path.is_symlink() and path.resolve().parent == download_dir.parent.parent / "blobs":
            etag = path.resolve().name
        else:
            metadata = download_dir / ".cache/huggingface/download" / (name + ".metadata")
            if not metadata.is_file():
                raise ValueError(f"No Hugging Face download metadata for {path}; resume the pinned download online")
            lines = metadata.read_text().splitlines()
            if len(lines) < 2 or lines[0] != source["revision"]:
                raise ValueError(f"Downloaded revision does not match the lock: {path}")
            etag = lines[1]
        if len(etag) == 64:
            actual = digest
        elif len(etag) == 40:
            blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    blob.update(chunk)
            actual = blob.hexdigest()
        else:
            raise ValueError(f"Invalid Hugging Face ETag for {path}")
        if actual != etag:
            raise ValueError(f"Downloaded file does not match its ETag: {path}")


def validate_asset_layout(directory, key, source):
    """Check completeness before accepting a manifest, including every Llama shard."""
    directory = Path(directory)
    model_dir = directory / source["repo_id"].split("/")[-1] if key == "ardy" else directory
    required = {
        "ardy": ["config.yaml", "denoiser.safetensors", "tokenizer.safetensors",
                 "stats/motion/mean.npy", "stats/motion/std.npy",
                 "stats/pre_quantization/mean.npy", "stats/pre_quantization/std.npy",
                 "stats/post_quantization/mean.npy", "stats/post_quantization/std.npy"],
        "sonic": ["config.json", "model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"],
        "llama_base": ["config.json", "model.safetensors.index.json"],
        "text_base": ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "adapter_config.json", "adapter_model.safetensors"],
        "text_adapter": ["adapter_config.json", "adapter_model.safetensors"],
    }[key].copy()
    if key == "llama_base":
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        shards = set(index.get("weight_map", {}).values())
        if not shards or any(not isinstance(s, str) or Path(s).name != s or
                             not s.endswith(".safetensors") for s in shards):
            raise ValueError("Invalid Llama shard index")
        required.extend(sorted(shards))
        if (model_dir / "adapter_config.json").exists():
            raise ValueError("llama_base must contain the full Llama model, not a PEFT adapter")
    for name in required:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {key} asset: {path}")
    if key == "sonic":
        published = json.loads((model_dir / "config.json").read_text())
        expected = ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]
        if (published.get("schema_version") != 1 or published.get("model_id") != source["repo_id"]
                or published.get("variants", {}).get("default", {}).get("deployment_files") != expected):
            raise ValueError("SONIC default encoder/decoder/config release mismatch")


def verify_assets(asset_root, key):
    """Require completed downloads and hash the exact files before inference."""
    asset_root = Path(asset_root)
    lock = json.loads(LOCK.read_text())
    manifest_path = asset_root / f"{key}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"] != lock[key] or not manifest.get("files"):
        raise ValueError(f"Asset manifest does not match the source lock: {manifest_path}")
    validate_asset_layout(asset_root / key, key, lock[key])
    if set(asset_files(asset_root / key)) != set(manifest["files"]):
        raise ValueError(f"Asset files differ from the completed manifest: {manifest_path}")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid asset path: {name}")
        if sha256(asset_root / key / relative) != expected:
            raise ValueError(f"Asset changed or incomplete: {key}/{name}")
    return sha256(manifest_path)
