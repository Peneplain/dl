import unittest

import numpy as np

from baseline.adapters.reference import BufferUnderrun, ReferenceBuffer, ReferenceSequence
from baseline.adapters.sonic import JointStreamEncoder


class ReferenceTimingTests(unittest.TestCase):
    def test_terminal_hold_recomputes_velocity_and_preserves_quaternion(self):
        sequence = ReferenceSequence([0, .1, .2], np.tile([0, .1, .2], (29, 1)).T,
                                     np.tile([1, 0, 0, 0], (3, 1)))
        held = sequence.sample_with_terminal_hold([.1, .2, .3, .4])
        np.testing.assert_allclose(held.joint_pos[:, 0], [.1, .2, .2, .2])
        np.testing.assert_allclose(held.velocities()[-2:], np.zeros((2, 29)), atol=1e-7)
        after = sequence.sample_with_terminal_hold([.3, .4])
        np.testing.assert_array_equal(after.velocities(), np.zeros((2, 29)))
        np.testing.assert_array_equal(after.body_quat, [[1, 0, 0, 0]] * 2)
        with self.assertRaises(BufferUnderrun):
            sequence.sample_with_terminal_hold([-.1, 0])
        with self.assertRaises(BufferUnderrun):
            sequence.sample([.1, .3])

    def test_buffer_rejects_nonfinite_times_and_noninteger_counts(self):
        buffer = ReferenceBuffer()
        for now, count, dt in [(np.nan, 10, .02), (0, 2.5, .02), (0, 2, np.inf)]:
            with self.subTest(now=now, count=count, dt=dt), self.assertRaises(ValueError):
                buffer.sample(now, count, dt)

    def test_packet_gap_is_rejected_even_at_large_timestamps(self):
        def sequence(start):
            return ReferenceSequence(start + np.arange(3) * 0.02, np.zeros((3, 29)),
                                     np.tile([1, 0, 0, 0], (3, 1)))
        encoder = JointStreamEncoder()
        encoder.encode(sequence(10000))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            encoder.encode(sequence(10000.08))
        encoder.encode(sequence(10000.06))


if __name__ == "__main__":
    unittest.main()
