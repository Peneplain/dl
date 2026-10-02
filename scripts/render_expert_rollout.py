"""Render saved simulation states to vision/images.npz and optional MP4.

The filename does not imply teacher/expert supervision. This replays recorded
executed states only and never loads a policy or advances physics.
Set MUJOCO_GL=osmesa (software), egl, or glfw before starting Python.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory containing rollout/")
    parser.add_argument("--out", type=Path, help="Fresh output directory; default: RUN/vision")
    parser.add_argument("--camera", action="append",
                        help="third_person (default) or a named MJCF camera; repeat for multiple views")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--lookat", type=float, nargs=3, default=[0, 0, .75])
    parser.add_argument("--distance", type=float, default=3)
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-15)
    parser.add_argument("--video", type=Path, help="Optional MP4 using the first camera")
    parser.add_argument("--video-fps", type=float, default=25,
                        help="MP4 simulation-time rate; images.npz still contains every saved frame")
    args = parser.parse_args()
    from baseline.rendering import render_rollout

    report = render_rollout(args.run, out=args.out, cameras=args.camera,
                            width=args.width, height=args.height, lookat=args.lookat,
                            distance=args.distance, azimuth=args.azimuth, elevation=args.elevation,
                            video=args.video, video_fps=args.video_fps)
    print(f"Rendered {report['frames']} states: {report['images_path']}")


if __name__ == "__main__":
    main()
