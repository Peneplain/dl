"""Create a source-only upload archive, including uncommitted implementation files."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"AGENTS.md", "README.md", "Dockerfile.musa", ".gitignore", ".dockerignore",
             "pyproject.toml", "requirements-musa.txt", "requirements-cpu.txt",
             "requirements-baseline.txt", "configs/baseline.lock.json",
             "docs/baseline.md", "docs/integration.md", "docs/verification.md",
             "docs/proposal.tex", "scripts/check_backend.py", "scripts/check_baseline.py", "scripts/check_sonic_onnx.py",
             "scripts/fetch_baseline.py", "scripts/install_baseline.sh",
             "scripts/package_baseline.py", "scripts/prepare_reference.py",
             "scripts/prepare_deploy_motion.py",
             "scripts/run_ardy.py", "scripts/ardy_service.py", "scripts/smoke.py"}
SOURCE_DIRS = {"baseline", "docker", "tests"}
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".tex"}


def source_files(root):
    files = [root / name for name in sorted(TOP_FILES) if (root / name).is_file()]
    for directory in sorted(SOURCE_DIRS):
        files.extend(path for path in sorted((root / directory).rglob("*"))
                     if path.is_file() and path.suffix in SUFFIXES
                     and (path.suffix != ".py" or path.stem.isidentifier())
                     and not any(part.startswith(".") or part == "__pycache__"
                                 for part in path.relative_to(root).parts))
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing symlink/outside source: {path}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh .tar.gz path under artifacts/")
    args = parser.parse_args()
    if not args.out.name.endswith(".tar.gz"):
        parser.error("--out must end in .tar.gz")
    if args.out.exists():
        parser.error("Choose a fresh --out file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tarfile.open(args.out, "x:gz") as archive:
        for path in source_files(ROOT):
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo("dl-b0/" + name)
            info.size = len(data)
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("dl-b0/SOURCE_MANIFEST.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_name(args.out.name + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"{args.out}: {len(manifest)} source files, SHA256 {digest}")
    print("No weights, datasets, upstream checkouts, environments, or run artifacts included.")


if __name__ == "__main__":
    main()
