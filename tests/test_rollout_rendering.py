"""Visual replay fixtures; these do not establish G1 tracking or grasp success."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from baseline.rollout import RolloutRecorder, SavedRollout
from baseline.rendering import render_rollout, video_frame_indices


SCENE = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1" rgba=".5 .5 .5 1"/>
    <body name="robot" pos="0 0 .5">
      <freejoint name="root"/>
      <geom type="sphere" size=".1" rgba=".2 .3 .9 1"/>
      <camera name="head" pos="1 0 .5" xyaxes="0 1 0 -.4 0 1"/>
      <body name="arm" pos="0 0 .1">
        <joint name="arm_joint" axis="0 1 0"/>
        <geom type="capsule" size=".03" fromto="0 0 0 .2 0 0"/>
        <body name="finger" pos=".2 0 0">
          <joint name="finger_joint" axis="0 0 1"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .07 0 0"/>
          <camera name="wrist" pos=".2 0 .2" xyaxes="0 1 0 -1 0 1"/>
        </body>
      </body>
    </body>
    <body name="block" pos=".5 0 .15">
      <freejoint name="block_root"/>
      <geom type="box" size=".07 .07 .07" rgba=".9 .1 .1 1"/>
    </body>
    <body name="marker" mocap="true" pos="-.5 0 .3">
      <geom type="sphere" size=".04" rgba=".1 .9 .1 1"/>
    </body>
  </worldbody>
  <actuator><motor joint="arm_joint"/><motor joint="finger_joint"/></actuator>
</mujoco>
"""


def record_fixture(run):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    data.qvel[model.joint("block_root").dofadr[0]] = -1
    data.ctrl[:] = [.1, .02]
    recorder = RolloutRecorder(model, run)
    expected = []
    for frame in range(6):
        if frame:
            for _ in range(4):
                mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        recorder.record(data, frame - 1, frame * .01)
        expected.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(),
                         data.geom_xpos.copy(), data.mocap_pos.copy()))
    recorder.close()
    return expected


class RolloutTests(unittest.TestCase):
    def test_round_trip_includes_free_root_fingers_dynamic_block_and_mocap(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            expected = record_fixture(run)
            saved = SavedRollout(run)
            np.testing.assert_array_equal(saved.frame_index, np.arange(-1, 5))
            self.assertIn("finger_joint", saved.metadata["joint_names"])
            # Random access, not accumulated re-simulation.
            with patch("mujoco.mj_step", side_effect=AssertionError("No physics in replay")):
                for frame in [5, 0, 2, 1]:
                    data = saved.restore(frame)
                    for actual, target in zip(
                            (data.qpos, data.qvel, data.ctrl, data.geom_xpos, data.mocap_pos),
                            expected[frame]):
                        np.testing.assert_allclose(actual, target, atol=1e-12)
            self.assertFalse(np.array_equal(expected[0][0], expected[-1][0]))

    def test_changed_model_or_states_are_rejected(self):
        for filename in ["scene.mjb", "states.npz"]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                record_fixture(run)
                with (run / "rollout" / filename).open("ab") as handle:
                    handle.write(b"corrupted")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    SavedRollout(run)

    def test_reset_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = mujoco.MjModel.from_xml_string(SCENE)
            data = mujoco.MjData(model)
            recorder = RolloutRecorder(model, directory)
            recorder.record(data, -1, 0)
            with self.assertRaisesRegex(ValueError, "increase"):
                recorder.record(data, 0, 1)
            recorder.close()
            recorder.close()
            with self.assertRaises(FileExistsError):
                RolloutRecorder(model, directory)

    def test_video_sampling_uses_simulation_timestamps_at_arbitrary_rates(self):
        times = np.arange(51) / 50
        indices = video_frame_indices(times, 30)
        self.assertEqual(len(indices), 31)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 50)
        np.testing.assert_allclose(times[indices], np.arange(31) / 30, atol=.01)

    def test_video_sampling_rejects_invalid_source_times(self):
        for times in [[], [0, 0], [0, float("nan")], [0, float("inf")]]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                video_frame_indices(times, 25)

    def test_unknown_camera_keeps_failure_report_and_saved_states(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record_fixture(run)
            with self.assertRaisesRegex(ValueError, "not defined"):
                render_rollout(run, cameras=["missing"])
            report = json.loads((run / "vision/report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse((run / "vision/.rgb.npy").exists())
            SavedRollout(run)


@unittest.skipUnless(os.environ.get("RUN_MUJOCO_RENDER_TESTS") == "1",
                     "Set RUN_MUJOCO_RENDER_TESTS=1 with a working MuJoCo GL backend")
class RGBRenderingTests(unittest.TestCase):
    def test_actual_rgb_multicamera_and_mp4_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record_fixture(run)
            with patch("mujoco.mj_step", side_effect=AssertionError("No physics in renderer")):
                report = render_rollout(run, cameras=["third_person", "head", "wrist"],
                                        width=128, height=96, lookat=(.2, 0, .4), distance=2,
                                        video=run / "test.mp4", video_fps=30)
            with np.load(run / "vision/images.npz", allow_pickle=False) as images:
                self.assertEqual(images["rgb"].shape, (6, 3, 96, 128, 3))
                self.assertEqual(images["rgb"].dtype, np.uint8)
                np.testing.assert_array_equal(images["frame_index"], np.arange(-1, 5))
                self.assertGreater(np.std(images["rgb"][0]), 5)
                self.assertTrue(np.any(images["rgb"][0] != images["rgb"][-1]))
                self.assertTrue(np.any(images["rgb"][0, 0] != images["rgb"][0, 1]))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["video"]["frames"], 7)
            self.assertGreater((run / "test.mp4").stat().st_size, 0)
            self.assertFalse((run / "vision/.rgb.npy").exists())
            # Replay the same executed states with another fixed camera.
            render_rollout(run, out=run / "vision-alt", width=64, height=48, azimuth=0)
            with self.assertRaises(FileExistsError):
                render_rollout(run)

    def test_encoder_failure_preserves_complete_rgb_and_failure_status(self):
        class FailedWriter:
            def __init__(self, *args, **kwargs):
                self.frames = 0
                self.closed = False

            def write(self, frame):
                self.frames += 1

            def close(self):
                self.closed = True
                raise RuntimeError("synthetic encoder failure")

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            record_fixture(run)
            with patch("baseline.rendering.RGBVideoWriter", FailedWriter):
                with self.assertRaisesRegex(RuntimeError, "synthetic encoder"):
                    render_rollout(run, width=64, height=48, video=run / "broken.mp4")
            with np.load(run / "vision/images.npz", allow_pickle=False) as images:
                self.assertEqual(images["rgb"].shape, (6, 1, 48, 64, 3))
            report = json.loads((run / "vision/report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["images_status"], "passed")
            self.assertEqual(report["video_status"], "failed")
            self.assertFalse((run / "broken.mp4").exists())


if __name__ == "__main__":
    unittest.main()
