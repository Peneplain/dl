"""Offline MuJoCo RGB rendering; no ARDY, SONIC or learning checkpoints needed."""

from pathlib import Path
import os
import subprocess
import sys
import time
import uuid

import mujoco
import numpy as np

from baseline.common import sha256, write_json
from baseline.rollout import SavedRollout


class RGBVideoWriter:
    def __init__(self, path, width, height, fps):
        import imageio_ffmpeg

        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("Video path must end in .mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite video: {self.path}")
        self.log = self.path.with_suffix(self.path.suffix + ".ffmpeg.log").open("x")
        self.temporary = self.path.with_name(f".{self.path.stem}.{uuid.uuid4().hex}.partial.mp4")
        try:
            self.process = subprocess.Popen(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                 "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p", str(self.temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise
        self.frames = 0
        self.closed = False

    def write(self, rgb):
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("ffmpeg did not finish; see " + str(self.log.name))
            if code:
                raise RuntimeError(f"ffmpeg exited with status {code}; see {self.log.name}")
            # Publish only a completed MP4, without overwriting a competing file.
            os.link(self.temporary, self.path)
        finally:
            self.temporary.unlink(missing_ok=True)
            self.log.close()


def video_frame_indices(times, fps):
    """Nearest saved states on a constant simulation-time grid, never wall time."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be positive and finite")
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()):
        raise ValueError("Video source timestamps must be finite and strictly increasing")
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * fps + 1e-8)) + 1) / fps
    right = np.minimum(np.searchsorted(times, grid), len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(grid - times[left] < times[right] - grid, left, right)


def render_rollout(run, *, out=None, cameras=None, width=640, height=480,
                   lookat=(0, 0, .75), distance=3.0, azimuth=135, elevation=-15,
                   video=None, video_fps=25, progress=None, rgb_fps=None):
    """Render all saved frames. Default camera is fixed in world coordinates.

    Named MJCF cameras are supported, including head/wrist cameras when the
    recorded scene defines them. RGB arrays use [frame, camera, height, width, 3].
    """
    run = Path(run)
    out = Path(out) if out is not None else run / "vision"
    cameras = list(cameras or ["third_person"])
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("Choose distinct camera names")
    if width < 1 or height < 1 or (video is not None and (width % 2 or height % 2)):
        raise ValueError("Image dimensions must be positive; MP4 dimensions must be even")
    if (np.shape(lookat) != (3,) or not np.isfinite(lookat).all()
            or not np.isfinite([distance, azimuth, elevation]).all() or distance <= 0):
        raise ValueError("Camera parameters must be finite, with positive distance")
    if video is not None and (not np.isfinite(video_fps) or video_fps <= 0):
        raise ValueError("Video FPS must be positive and finite")
    if video is not None and Path(video).exists():
        raise FileExistsError(f"Refusing to overwrite video: {video}")
    if video is not None and Path(video).suffix.lower() != ".mp4":
        raise ValueError("Video path must end in .mp4")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "renderer": "mujoco.Renderer",
              "source_run": str(run), "mujoco_version": mujoco.__version__,
              "numpy_version": np.__version__, "command": sys.argv,
              "render_source_sha256": sha256(__file__),
              "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
              "camera_names": cameras, "width": width, "height": height,
              "third_person": {"lookat": list(lookat), "distance": distance,
                               "azimuth": azimuth, "elevation": elevation},
              "physics_reexecuted": False, "teacher_used": False,
              "images_status": "pending", "video_status": "pending" if video else "disabled"}
    writer = None
    rgb = None
    scratch = out / ".rgb.npy"
    try:
        saved = SavedRollout(run)
        report.update(model_sha256=saved.metadata["model_sha256"],
                      states_sha256=saved.metadata["states_sha256"])
        camera_objects = []
        for name in cameras:
            if name == "third_person":
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = lookat
                camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            else:
                camera = mujoco.mj_name2id(saved.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if camera < 0:
                    raise ValueError(f"Camera {name!r} is not defined in the recorded scene")
            camera_objects.append(camera)
        # Size the offscreen framebuffer on this replay model only.
        saved.model.vis.global_.offwidth = max(width, saved.model.vis.global_.offwidth)
        saved.model.vis.global_.offheight = max(height, saved.model.vis.global_.offheight)
        n = len(saved.sim_time)
        indices = (np.arange(n) if rgb_fps is None else video_frame_indices(saved.sim_time, rgb_fps))
        indices = np.unique(indices)
        times = saved.sim_time[indices]
        frames = saved.frame_index[indices]
        n = len(indices)
        report["rgb_fps"] = rgb_fps
        report["source_frames"] = len(saved.sim_time)
        last_progress = time.perf_counter()
        if progress:
            progress("render", 0, n)
        # A full 30 s RGB episode can exceed a gigabyte. Stream to disk rather
        # than retaining all rendered images in process memory.
        rgb = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.uint8,
                                       shape=(n, len(cameras), height, width, 3))
        with mujoco.Renderer(saved.model, height=height, width=width) as renderer:
            for frame, state_index in enumerate(indices):
                data = saved.restore(state_index)
                for slot, camera in enumerate(camera_objects):
                    renderer.update_scene(data, camera=camera)
                    rgb[frame, slot] = renderer.render()
                if progress and (frame == n - 1 or time.perf_counter() - last_progress >= 5):
                    progress("render", frame + 1, n)
                    report.update(rendered_frames=frame + 1, total_frames=n)
                    write_json(out / "report.json", report)
                    last_progress = time.perf_counter()
        rgb.flush()
        if progress:
            progress("compress RGB", n, n)
        images = out / "images.npz"
        temporary_images = out / ".images.npz.tmp"
        with temporary_images.open("xb") as handle:
            np.savez_compressed(handle, rgb=rgb, camera_names=np.array(cameras),
                                sim_time=times, frame_index=frames,
                                state_index=indices,
                                model_sha256=np.array(saved.metadata["model_sha256"]),
                                states_sha256=np.array(saved.metadata["states_sha256"]))
        temporary_images.rename(images)
        report.update(images_status="passed", frames=n, images_path=str(images),
                      images_sha256=sha256(images), rgb_shape=list(rgb.shape),
                      first_sim_time=float(times[0]), last_sim_time=float(times[-1]))
        # Seal RGB first. Encoder failures must not discard successfully rendered images.
        video_indices = np.array([], dtype=np.int64)
        if video is not None:
            report["video_status"] = "encoding"
            video_indices = video_frame_indices(times, video_fps)
            if progress:
                progress("encode MP4", 0, len(video_indices))
            writer = RGBVideoWriter(video, width, height, video_fps)
            for index in video_indices:
                writer.write(rgb[index, 0])
            writer.close()
            report["video_status"] = "passed"
        report["video"] = ({"path": str(video), "renderer": "mujoco.Renderer",
                            "camera": cameras[0], "fps": video_fps, "frames": writer.frames,
                            "width": width, "height": height, "status": "passed",
                            "codec": "libx264", "crf": 18, "pixel_format": "yuv420p",
                            "sha256": sha256(video),
                            "state_index": indices[video_indices].tolist(),
                            "presentation_time": (np.arange(len(video_indices)) / video_fps).tolist(),
                            "frame_index": frames[video_indices].tolist(),
                            "source_sim_time": times[video_indices].tolist()}
                           if writer is not None else None)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["images_status"] != "passed":
            report["images_status"] = "failed"
        elif video is not None:
            report["video_status"] = "failed"
        raise
    finally:
        try:
            if writer is not None and not writer.closed:
                try:
                    writer.close()
                except Exception as error:
                    report["encoder_cleanup_error"] = f"{type(error).__name__}: {error}"
        finally:
            if rgb is not None:
                del rgb
            scratch.unlink(missing_ok=True)
            (out / ".images.npz.tmp").unlink(missing_ok=True)
            report["render_wall_seconds"] = time.perf_counter() - started
            write_json(out / "report.json", report)
    return report
