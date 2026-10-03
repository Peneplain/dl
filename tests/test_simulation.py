"""Reference timing and terminal hold behavior without loading model weights."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from baseline.adapters.reference import ReferenceBuffer, ReferenceSequence
from baseline.simulation import SonicSimulation


class SimulationReferenceTests(unittest.TestCase):
    def test_reference_limits_apply_to_all_29_body_joints(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        lower = np.arange(29, dtype=np.float32) / 100
        upper = lower + 1
        simulation.ranges = np.stack([lower, upper], axis=1)
        reference = ReferenceSequence([0., .02],
                                      np.stack([lower - .04, upper + .04]),
                                      [[1, 0, 0, 0], [1, 0, 0, 0]])
        checked = simulation.validate_reference(reference)
        np.testing.assert_allclose(checked.joint_pos[0], lower)
        np.testing.assert_allclose(checked.joint_pos[1], upper)

    def test_acquisition_hold_uses_actual_pose_without_mutating_physics(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        pose = np.r_[[.1, 0., .75, 1., 0., 0., 0.], np.arange(29) / 100.]
        original = pose.copy()
        simulation.body_pose = lambda: pose
        simulation.current_ref = np.ones(29)
        simulation.install = Mock()
        simulation.event = Mock()
        simulation.hold_measured_pose("grasp_alignment_reached")
        reference = simulation.install.call_args.args[0]
        np.testing.assert_allclose(reference.joint_pos, np.repeat(pose[None, 7:], 2, axis=0), atol=1e-7)
        np.testing.assert_array_equal(reference.velocities(), np.zeros((2, 29)))
        np.testing.assert_array_equal(pose, original)
        self.assertEqual(simulation.install.call_args.kwargs["transition"], .1)

    def test_executor_holds_actual_endpoint_and_logs_coverage_once(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.policy = SimpleNamespace(future_count=10, future_step=5)
        simulation.data = SimpleNamespace(time=.19)
        simulation.buffer = ReferenceBuffer()
        simulation.buffer.sequence = ReferenceSequence(
            [0, .1, .2], np.tile([0, .1, .2], (29, 1)).T,
            np.tile([1, 0, 0, 0], (3, 1)))
        simulation.end_time = .2
        simulation.current_ref = np.zeros(29)
        simulation.current_quat = np.array([1, 0, 0, 0])
        simulation.holding = False
        simulation.terminal_hold_logged = False
        events = []
        simulation.event = lambda name, **fields: events.append(name)
        simulation.lookahead()
        simulation.lookahead()
        self.assertEqual(events.count("reference_terminal_hold"), 1)
        simulation.data.time = .21
        positions, velocities, _ = simulation.lookahead()
        np.testing.assert_allclose(positions, .2)
        np.testing.assert_array_equal(velocities, np.zeros((10, 29)))
        simulation.lookahead()
        self.assertEqual(simulation.buffer.underruns, 1)
        self.assertEqual(events.count("buffer_underrun"), 1)

    def test_declared_reference_fps_must_match_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
            for fps, times in [(50.5, [0, .02]), (50, [0, .04])]:
                np.savez(path, fps=fps, times=times, joint_names=ISAACLAB_JOINT_NAMES,
                         joint_pos=np.zeros((2, 29)), body_quat=[[1, 0, 0, 0]] * 2)
                with self.subTest(fps=fps, times=times), self.assertRaises(ValueError):
                    SonicSimulation.load_reference(path)


if __name__ == "__main__":
    unittest.main()
