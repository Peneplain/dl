"""Render selected attempts or a whole batch/manual session, after collection."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.common import write_json
from baseline.session import timestamp


def select_attempts(run, attempts=None, all_attempts=False, successes=False):
    run = Path(run)
    if (run / "rollout/states.npz").is_file():
        if attempts or all_attempts or successes:
            raise ValueError("An attempt path needs no --attempts/--all/--successes selector")
        return [run]
    if not run.is_dir():
        raise FileNotFoundError(run)
    candidates = sorted(p for p in run.glob("attempt-*") if p.is_dir())
    if not any((attempts, all_attempts, successes)):
        raise ValueError("For a session choose --attempts 1 3, --all, or --successes")
    if attempts:
        selected = []
        for item in attempts:
            name = f"attempt-{int(item):05d}" if item.isdigit() else item
            if not name.startswith("attempt-") or Path(name).name != name:
                raise ValueError(f"Invalid attempt selector: {item}")
            path = run / name
            if not path.is_dir():
                raise FileNotFoundError(path)
            if path not in selected:
                selected.append(path)
        return selected
    if successes:
        return [p for p in candidates if (p / "report.json").is_file()
                and json.loads((p / "report.json").read_text()).get("task_success") is True]
    return candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Session or single attempt folder")
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument("--attempts", nargs="+", help="Attempt numbers or exact folder names")
    selectors.add_argument("--all", action="store_true", dest="all_attempts")
    selectors.add_argument("--successes", action="store_true")
    parser.add_argument("--camera", action="append", help="Default: third_person; repeat for multiple cameras")
    parser.add_argument("--width", type=int, default=640, help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Image height (default: 480)")
    parser.add_argument("--fps", type=float, default=25, help="RGB and MP4 simulation-time sampling rate")
    parser.add_argument("--no-video", action="store_true", help="Only write vision/images.npz")
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-15)
    parser.add_argument("--distance", type=float, default=3)
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1 or (not args.no_video and (args.width % 2 or args.height % 2)):
        parser.error("Image dimensions must be positive, and even for video")
    if not 0 < args.fps <= 50:
        parser.error("fps must be in (0, 50]")
    try:
        selected = select_attempts(args.run, args.attempts, args.all_attempts, args.successes)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not selected:
        print("No matching attempts to render.")
        return 0
    from baseline.rendering import render_rollout
    receipt = {"schema_version": 1, "command": sys.argv, "attempts": []}
    receipt_path = args.run / f"render-{timestamp(microseconds=True)}.json"
    print(f"Render settings: {args.width}x{args.height}, {args.fps:g} FPS; "
          f"cameras: {', '.join(args.camera or ['third_person'])}", flush=True)
    for count, attempt in enumerate(selected, 1):
        started = time.perf_counter()
        out = attempt / "vision"
        if out.exists():
            out = attempt / ("vision-" + timestamp(microseconds=True))
        print(f"[{count}/{len(selected)}] {attempt.name} -> {out}", flush=True)
        def progress(stage, done, total):
            elapsed = time.perf_counter() - started
            remaining = (elapsed / done * (total - done)) if done and stage == "render" else None
            eta = f", ETA {remaining:.0f}s" if remaining is not None else ""
            print(f"  {stage}: {done}/{total} ({done / total:.0%}), {elapsed:.0f}s{eta}", flush=True)
        row = {"attempt": str(attempt), "output": str(out)}
        try:
            report = render_rollout(attempt, out=out, cameras=args.camera, width=args.width,
                                    height=args.height, rgb_fps=args.fps, video_fps=args.fps,
                                    video=None if args.no_video else out / "video.mp4", progress=progress,
                                    azimuth=args.azimuth, elevation=args.elevation, distance=args.distance)
            row.update(status="passed", frames=report["frames"], wall_seconds=report["render_wall_seconds"])
            print(f"  DONE: {out / ('images.npz' if args.no_video else 'video.mp4')}", flush=True)
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"  FAILED: {row['error']}", flush=True)
            if not isinstance(error, Exception):
                receipt["attempts"].append(row)
                write_json(receipt_path, receipt)
                raise
        receipt["attempts"].append(row)
        write_json(receipt_path, receipt)
    failed = sum(row["status"] != "passed" for row in receipt["attempts"])
    print(f"Rendered {len(selected) - failed}/{len(selected)}; failed {failed}. Report: {receipt_path}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
