"""Fetch pinned B0 source and weights. Downloads never enter the project Git index."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import (LOCK, ROOT, asset_files, checked_checkout, checked_repo_root,
                             sha256, validate_asset_layout, verify_assets,
                             verify_download_metadata, write_json)


def checkout(destination, source):
    if not destination.exists() or (destination.is_dir() and not any(destination.iterdir())):
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(["git", "-c", f"safe.directory={destination}", "-C", str(destination),
                        "remote", "add", "origin", source["url"]], check=True)
    checked_repo_root(destination)
    git = ["git", "-c", f"safe.directory={destination}", "-C", str(destination)]
    # Resume an interrupted initial fetch; never reset an existing checkout.
    head = subprocess.run(git + ["rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
        subprocess.run(git + ["fetch", "--depth", "1", "origin", source["commit"]], check=True)
        subprocess.run(git + ["checkout", "--detach", "FETCH_HEAD"], check=True)
    checked_checkout(destination, source["commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["sources", "ardy", "sonic", "llama", "text", "all"], default="all")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--offline", action="store_true",
                        help="Verify existing files; register unmanifested HF downloads using their revision/ETag metadata")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    if args.only in ("sources", "all"):
        for key in ("ardy", "sonic"):
            if args.offline:
                checked_checkout(ROOT / "third_party" / key, lock[key]["commit"])
            else:
                checkout(ROOT / "third_party" / key, lock[key])
    keys = {"sources": [], "ardy": ["ardy"], "sonic": ["sonic"],
            "llama": ["llama_base"],
            "text": ["llama_base", "text_base", "text_adapter"],
            "all": ["sonic", "ardy", "llama_base", "text_base", "text_adapter"]}[args.only]
    if not keys:
        return
    patterns = {
        "ardy": ["config.yaml", "*.safetensors", "stats/**", "LICENSE", "README.md"],
        "sonic": ["config.json", "model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml", "README.md", "LICENSE*"],
        "llama_base": ["config.json", "model.safetensors.index.json", "model-*.safetensors", "LICENSE*", "README.md"],
        "text_base": ["*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json", "README.md", "LICENSE*"],
        "text_adapter": ["*.json", "*.safetensors", "README.md", "LICENSE*"],
    }
    for key in keys:
        source = lock[key]
        destination = args.assets.resolve() / key
        download_dir = destination / source["repo_id"].split("/")[-1] if key == "ardy" else destination
        manifest_path = args.assets / f"{key}.manifest.json"
        if manifest_path.exists():
            try:
                verify_assets(args.assets, key)
            except (ValueError, KeyError, OSError) as error:
                print(f"Rechecking {key}: {error}", flush=True)
            else:
                print(f"Verified existing {key}; no download needed", flush=True)
                continue
        if not args.offline:
            from huggingface_hub import snapshot_download
            print(f"Downloading {source['repo_id']} @ {source['revision']}", flush=True)
            snapshot_download(repo_id=source["repo_id"], revision=source["revision"],
                              local_dir=download_dir, allow_patterns=patterns[key])
        validate_asset_layout(destination, key, source)
        files = {name: sha256(path) for name, path in asset_files(destination).items()}
        download_files = {str(path.relative_to(download_dir)): files[name]
                          for name, path in asset_files(destination).items()}
        # snapshot_download can reuse a corrupt local file if its metadata still
        # looks current. Never bless it with a new hash just because HF returned.
        verify_download_metadata(download_dir, source, download_files)
        # Replace atomically only after success. Existing manifests remain useful
        # after a network failure; verification still rejects changed/partial files.
        write_json(manifest_path, {"source": source, "files": files})
        print(f"Verified {len(files)} files; manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
