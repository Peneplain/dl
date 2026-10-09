"""Create a source-only upload archive, including uncommitted implementation files."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"AGENTS.md", "README.md", "run.sh", "Dockerfile.musa", ".gitignore", ".dockerignore",
             "pyproject.toml", "requirements-musa.txt", "requirements-cpu.txt",
             "requirements-baseline.txt", "requirements-sonic-sim.txt", "configs/baseline.lock.json",
             "docs/baseline.md", "docs/commands.md", "docs/grasp.md", "docs/learning.md",
             "docs/project_context.md", "docs/track4_requirements.md",
             "configs/grasp-prompts.json", "docs/verification.md",
             "docs/proposal.tex", "scripts/check_backend.py", "scripts/check_baseline.py", "scripts/check_sonic_onnx.py",
             "scripts/fetch_baseline.py", "scripts/install_baseline.sh",
             "scripts/package_baseline.py", "scripts/prepare_reference.py",
             "scripts/prepare_deploy_motion.py",
             "scripts/render.py",
             "scripts/run_ardy.py", "scripts/ardy_service.py", "scripts/run.py", "scripts/smoke.py"}
BASELINE_TESTS = {
    "tests/test_baseline_bringup.py", "tests/test_deploy_motion.py",
    "tests/test_grasp.py", "tests/test_reference_timing.py", "tests/test_rollout_rendering.py",
    "tests/test_sessions.py", "tests/test_simulation.py", "tests/test_text_encoder.py",
}
TOP_FILES.update(BASELINE_TESTS)
KIMODO_FILES = {
    "configs/kimodo.lock.json", "scripts/fetch_kimodo.py", "tests/test_kimodo.py",
    "tests/test_kimodo_projection.py",
}
SOURCE_DIRS = {"baseline", "docker"}
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".tex"}


def source_files(root, scope="b0"):
    if scope not in {"b0", "kimodo"}:
        raise ValueError(f"Unknown packaging scope: {scope}")
    top_files = set(TOP_FILES)
    if scope == "kimodo":
        top_files.update(KIMODO_FILES)
    files = [root / name for name in sorted(top_files) if (root / name).is_file()]
    for directory in sorted(SOURCE_DIRS):
        files.extend(path for path in sorted((root / directory).rglob("*"))
                     if path.is_file() and path.suffix in SUFFIXES
                     and (scope == "kimodo" or not path.relative_to(root).as_posix().startswith("baseline/kimodo"))
                     and (path.suffix != ".py" or path.stem.isidentifier())
                     and not any(part.startswith(".") or part == "__pycache__"
                                 for part in path.relative_to(root).parts))
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing symlink/outside source: {path}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh .tar.gz path under output/")
    parser.add_argument("--scope", choices=("b0", "kimodo"), default="b0",
                        help="Explicit source scope; default is the declared B0 archive")
    args = parser.parse_args()
    if not args.out.name.endswith(".tar.gz"):
        parser.error("--out must end in .tar.gz")
    if args.out.exists():
        parser.error("Choose a fresh --out file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tarfile.open(args.out, "x:gz") as archive:
        for path in source_files(ROOT, args.scope):
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo("dl-" + args.scope + "/" + name)
            info.size = len(data)
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("dl-" + args.scope + "/SOURCE_MANIFEST.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_name(args.out.name + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"{args.out}: {len(manifest)} {args.scope} source files, SHA256 {digest}")
    print("No weights, datasets, upstream checkouts, environments, or run artifacts included.")


if __name__ == "__main__":
    main()
