"""Fetch pinned B0 source and weights. Downloads never enter the project Git index."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import LOCK, ROOT, checked_checkout, sha256, write_json


def checkout(destination, source):
    if not destination.exists():
        destination.mkdir(parents=True)
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", source["url"]], check=True)
    # Resume an interrupted initial fetch; never reset an existing checkout.
    head = subprocess.run(["git", "-C", str(destination), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", source["commit"]], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
    checked_checkout(destination, source["commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["sources", "ardy", "sonic", "text", "all"], default="all")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    if args.only in ("sources", "all"):
        for key in ("ardy", "sonic"):
            checkout(ROOT / "third_party" / key, lock[key])
    keys = {"sources": [], "ardy": ["ardy"], "sonic": ["sonic"],
            "text": ["text_base", "text_adapter"],
            "all": ["sonic", "ardy", "text_base", "text_adapter"]}[args.only]
    if not keys:
        return
    from huggingface_hub import snapshot_download
    patterns = {
        "ardy": ["config.yaml", "*.safetensors", "stats/**", "LICENSE", "README.md"],
        "sonic": ["config.json", "model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml", "README.md", "LICENSE*"],
        "text_base": ["*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json", "README.md", "LICENSE*"],
        "text_adapter": ["*.json", "*.safetensors", "README.md", "LICENSE*"],
    }
    required = {"ardy": ["config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/motion/mean.npy"],
                "sonic": ["config.json", "model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"],
                "text_base": ["config.json", "tokenizer.json"],
                "text_adapter": ["adapter_config.json", "adapter_model.safetensors"]}
    for key in keys:
        source = lock[key]
        destination = args.assets.resolve() / key
        download_dir = destination / source["repo_id"].split("/")[-1] if key == "ardy" else destination
        # An interrupted/changed download must not retain a completed manifest.
        manifest_path = args.assets / f"{key}.manifest.json"
        manifest_path.unlink(missing_ok=True)
        print(f"Downloading {source['repo_id']} @ {source['revision']}", flush=True)
        snapshot_download(repo_id=source["repo_id"], revision=source["revision"],
                          local_dir=download_dir, allow_patterns=patterns[key])
        for name in required[key]:
            if not (download_dir / name).is_file():
                raise FileNotFoundError(download_dir / name)
        if key == "sonic":
            published = json.loads((destination / "config.json").read_text())
            expected = ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]
            if (published.get("schema_version") != 1 or published.get("model_id") != source["repo_id"]
                    or published["variants"]["default"]["deployment_files"] != expected):
                raise ValueError("SONIC default encoder/decoder/config release mismatch")
        files = {str(p.relative_to(destination)): sha256(p)
                 for p in sorted(destination.rglob("*"))
                 if p.is_file() and ".cache" not in p.relative_to(destination).parts}
        if key == "text_base" and not any(name.endswith(".safetensors") for name in files):
            raise FileNotFoundError("LLM2Vec base weights are missing")
        write_json(manifest_path, {"source": source, "files": files})
        print(f"Verified {len(files)} files; manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
