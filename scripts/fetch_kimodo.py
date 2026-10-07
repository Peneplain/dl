"""Fetch or verify the separately pinned Kimodo source and G1 checkpoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import (ROOT, asset_files, checked_checkout, checked_repo_root,
                             sha256, write_json)
from baseline.kimodo import _verify_checkpoint

LOCK_PATH = ROOT / "configs/kimodo.lock.json"


def _checkout(destination: Path, source: dict) -> str:
    destination = Path(destination)
    if not destination.exists() or (destination.is_dir() and not any(destination.iterdir())):
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(["git", "-c", f"safe.directory={destination}", "-C", str(destination),
                        "remote", "add", "origin", source["url"]], check=True)
    checked_repo_root(destination)
    git = ["git", "-c", f"safe.directory={destination}", "-C", str(destination)]
    head = subprocess.run(git + ["rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
        subprocess.run(git + ["fetch", "--depth", "1", "origin", source["commit"]], check=True)
        subprocess.run(git + ["checkout", "--detach", "FETCH_HEAD"], check=True)
    return checked_checkout(destination, source["commit"])


def _verify_model(asset_root: Path, lock: dict) -> Path:
    model_dir, manifest_digest = _verify_checkpoint(asset_root, lock["model"])
    if manifest_digest is None:
        raise ValueError(
            f"Missing {asset_root / 'kimodo.manifest.json'}. Fetch online with "
            "python scripts/fetch_kimodo.py --only checkpoint, or transfer the "
            "complete checkpoint directory including Hugging Face metadata."
        )
    return model_dir


def _fetch_model(asset_root: Path, lock: dict) -> None:
    from huggingface_hub import snapshot_download

    source = lock["checkpoint"]
    model_dir = asset_root / source["repo_id"].split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["config.yaml", "model.safetensors", "stats/**", "LICENSE", "README.md"]
    print(f"Downloading {source['repo_id']} @ {source['revision']}", flush=True)
    snapshot_download(repo_id=source["repo_id"], revision=source["revision"],
                      local_dir=model_dir, allow_patterns=patterns)
    # Validate expected model files before creating a manifest; do not bless a
    # partial or unpinned local copy merely because the downloader returned.
    _verify_checkpoint_layout(model_dir)
    files = {name: sha256(path) for name, path in asset_files(model_dir).items()}
    files_for_metadata = {str(path.relative_to(model_dir)): digest
                          for name, path in asset_files(model_dir).items()
                          if (digest := files[name])}
    from baseline.common import verify_download_metadata
    verify_download_metadata(model_dir, source, files_for_metadata)
    write_json(asset_root / "kimodo.manifest.json", {"source": source, "files": files})
    _verify_model(asset_root, lock)
    print(f"Verified {len(files)} Kimodo files; manifest: {asset_root / 'kimodo.manifest.json'}",
          flush=True)


def _verify_checkpoint_layout(model_dir: Path) -> None:
    required = (
        "config.yaml", "model.safetensors",
        "stats/motion/body/mean.npy", "stats/motion/body/std.npy",
        "stats/motion/global_root/mean.npy", "stats/motion/global_root/std.npy",
        "stats/motion/local_root/mean.npy", "stats/motion/local_root/std.npy",
    )
    for name in required:
        path = model_dir / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty Kimodo model asset: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("source", "checkpoint", "all"), default="all")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/kimodo")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/kimodo")
    parser.add_argument("--offline", action="store_true",
                        help="Verify a complete existing source checkout and checkpoint manifest only")
    args = parser.parse_args(argv)
    lock = json.loads(LOCK_PATH.read_text())
    try:
        if args.only in ("source", "all"):
            if args.offline:
                checked_checkout(args.repo, lock["source"]["commit"])
            else:
                _checkout(args.repo, lock["source"])
            print(f"Verified Kimodo source at {lock['source']['commit']}: {args.repo}", flush=True)
        if args.only in ("checkpoint", "all"):
            args.assets.mkdir(parents=True, exist_ok=True)
            if args.offline:
                _verify_model(args.assets, lock)
                print(f"Verified Kimodo checkpoint and manifest: {args.assets}", flush=True)
            else:
                try:
                    _verify_model(args.assets, lock)
                except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
                    _fetch_model(args.assets, lock)
                else:
                    print(f"Verified existing Kimodo checkpoint; no download needed: {args.assets}",
                          flush=True)
    except Exception as error:
        print(f"Kimodo setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        print("If this machine cannot reach GitHub/Hugging Face, run on a connected machine:",
              file=sys.stderr)
        print("  python scripts/fetch_kimodo.py --only all", file=sys.stderr)
        print("Then transfer the pinned source and checkpoint to the same project path:",
              file=sys.stderr)
        print("  rsync -a --progress third_party/kimodo/ \\", file=sys.stderr)
        print("    <user>@<host>:/path/to/dl/third_party/kimodo/", file=sys.stderr)
        print("  rsync -a --progress checkpoints/kimodo/ \\", file=sys.stderr)
        print("    <user>@<host>:/path/to/dl/checkpoints/kimodo/", file=sys.stderr)
        print("Then verify here with: python scripts/fetch_kimodo.py --offline", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
