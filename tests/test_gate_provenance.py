"""Runtime gates must use the probabilities and validation data they declare."""

import tempfile
import unittest
from pathlib import Path

from baseline.common import sha256
from risk_residual.checkpoints import validate_gate


class GateProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.risk = Path(temporary.name) / "risk.pt"
        self.risk.write_bytes(b"checkpoint provenance fixture")
        self.meta = {"dataset_manifest_sha256": "manifest", "window_hashes": {"val": "windows"}}
        self.gate = {"split": "val", "risk_sha256": sha256(self.risk),
                     "dataset_manifest_sha256": "manifest", "threshold": .5,
                     "probability_transform": "raw_sigmoid", "temperature": 1.,
                     "validation_windows_sha256": "windows"}

    def test_matching_and_legacy_raw_gates(self):
        validate_gate(self.gate, self.risk, self.meta)
        legacy = {key: value for key, value in self.gate.items() if key not in
                  {"probability_transform", "temperature", "validation_windows_sha256"}}
        validate_gate(legacy, self.risk, self.meta)

    def test_temperature_or_window_mismatch_cannot_change_runtime_gate(self):
        for key, value in (("temperature", 2.), ("probability_transform", "temperature_scaled"),
                           ("validation_windows_sha256", "different"), ("threshold", float("nan")),
                           ("threshold", True), ("split", "test")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_gate({**self.gate, key: value}, self.risk, self.meta)


if __name__ == "__main__":
    unittest.main()
