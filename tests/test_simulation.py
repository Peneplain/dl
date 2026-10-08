"""Reference timing and terminal hold behavior without loading model weights."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from baseline.adapters.reference import ReferenceBuffer, ReferenceSequence
from baseline.adapters.joints import ARM_INDICES
from baseline.reference_checks import ReferenceChecks
from baseline.simulation import SonicSimulation


class SimulationReferenceTests(unittest.TestCase):
    def test_optional_correction_preserves_nominal_and_limits_current_offset(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.policy = SimpleNamespace(future_count=10, future_step=5)
        simulation.data = SimpleNamespace(time=0.)
        simulation.buffer = ReferenceBuffer()
        times = np.arange(51) * .02
        simulation.buffer.push(ReferenceSequence(
            times, np.full((len(times), 29), .5),
            np.tile([1, 0, 0, 0], (len(times), 1))))
        simulation.end_time = float(times[-1])
        simulation.current_ref = np.full(29, .5)
        simulation.current_quat = np.array([1, 0, 0, 0])
        simulation.holding = False
        simulation.terminal_hold_logged = False
        simulation.ranges = np.tile([-2., 2.], (29, 1))
        simulation.reference_checks = ReferenceChecks(simulation.ranges)
        simulation.learned_correction_used = False
        simulation.event = Mock()
        provider = Mock()
        provider.request.side_effect = lambda sim, ref: np.broadcast_to(
            np.array([.1 if i in ARM_INDICES else 0 for i in range(29)]),
            ref.joint_pos.shape).copy()
        simulation.correction_provider = provider
        positions, velocities, _ = simulation.lookahead()
        np.testing.assert_allclose(simulation._nominal_lookahead[0], .5)
        self.assertAlmostEqual(positions[0, ARM_INDICES[0]], .51, places=5)
        self.assertTrue(simulation.learned_correction_used)
        nonarms = [i for i in range(29) if i not in ARM_INDICES]
        np.testing.assert_allclose(positions[:, nonarms], .5)
        self.assertTrue(np.isfinite(velocities).all())
        simulation.data.time = .02
        provider.request.side_effect = lambda sim, ref: np.zeros_like(ref.joint_pos)
        positions, _, _ = simulation.lookahead()
        self.assertAlmostEqual(positions[0, ARM_INDICES[0]], .5, places=5)
        self.assertTrue(simulation.learned_correction_used)

    def test_zero_correction_does_not_mark_learning_used(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.policy = SimpleNamespace(future_count=10, future_step=5)
        simulation.data = SimpleNamespace(time=0.)
        simulation.buffer = ReferenceBuffer()
        times = np.arange(51) * .02
        simulation.buffer.push(ReferenceSequence(
            times, np.full((len(times), 29), .5),
            np.tile([1, 0, 0, 0], (len(times), 1))))
        simulation.end_time = float(times[-1])
        simulation.current_ref = np.full(29, .5)
        simulation.current_quat = np.array([1, 0, 0, 0])
        simulation.holding = False
        simulation.terminal_hold_logged = False
        simulation.ranges = np.tile([-2., 2.], (29, 1))
        simulation.reference_checks = ReferenceChecks(simulation.ranges)
        simulation.learned_correction_used = False
        simulation.event = Mock()
        provider = Mock()
        provider.request.side_effect = lambda sim, ref: np.zeros_like(ref.joint_pos)
        simulation.correction_provider = provider
        simulation.lookahead()
        self.assertFalse(simulation.learned_correction_used)

    def test_fully_clipped_correction_does_not_mark_learning_used(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.policy = SimpleNamespace(future_count=10, future_step=5)
        simulation.data = SimpleNamespace(time=0.)
        simulation.buffer = ReferenceBuffer()
        times = np.arange(51) * .02
        positions = np.full((len(times), 29), .5)
        arm_index = next(iter(ARM_INDICES))
        positions[:, list(ARM_INDICES)] = 2.0
        simulation.buffer.push(ReferenceSequence(
            times, positions, np.tile([1, 0, 0, 0], (len(times), 1))))
        simulation.end_time = float(times[-1])
        simulation.current_ref = positions[0].copy()
        simulation.current_quat = np.array([1, 0, 0, 0])
        simulation.holding = False
        simulation.terminal_hold_logged = False
        simulation.ranges = np.tile([-2., 2.], (29, 1))
        simulation.reference_checks = ReferenceChecks(simulation.ranges)
        simulation.learned_correction_used = False
        simulation.event = Mock()
        provider = Mock()
        provider.request.side_effect = lambda sim, ref: np.broadcast_to(
            np.array([.1 if i in ARM_INDICES else 0 for i in range(29)]),
            ref.joint_pos.shape).copy()
        simulation.correction_provider = provider
        positions_out, _, _ = simulation.lookahead()
        self.assertAlmostEqual(positions_out[0, arm_index], 2.0, places=5)
        self.assertFalse(simulation.learned_correction_used)

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
        self.assertEqual(simulation.install.call_args.kwargs["transition"], .3)

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
