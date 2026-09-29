import unittest

import numpy as np

from baseline.adapters.reference import ReferenceSequence
from baseline.adapters.sonic import JointStreamEncoder


class ReferenceTimingTests(unittest.TestCase):
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
