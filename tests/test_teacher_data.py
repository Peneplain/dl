"""Regression checks for paired teacher collection and audited conversion."""

import json
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.common import sha256
from experiments.build_dataset import inspect, pair_details, planned, read_context
from risk_residual.teacher import ScheduledArmOffset


class TeacherDataTests(unittest.TestCase):
    def test_single_parent_inspection_does_not_write_a_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sources.json"
            path.write_text(json.dumps({"parents": [{"parent_id": "one", "split": "train",
                                                     "prompt_group": "p", "scene_seed": 7}]}))
            fake_window = {"intervention_target": np.float32(1),
                           "correction_sample": True, "stable_sample": False}
            provenance = {"decision_time_bounds": [2., 2.], "prompt_identity": "prompt"}
            with patch("experiments.build_dataset.extract_parent",
                       return_value=([fake_window], "model-hash", provenance)), \
                 patch("experiments.build_dataset.window_arrays", return_value={}), \
                 patch("experiments.build_dataset.WindowDataset"):
                result = inspect(path, thresholds=np.ones(4))
            self.assertEqual(result["parents"][0]["correction_samples"], 1)
            self.assertFalse(result["dataset_written"])
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_context_preserves_finger_header_order_and_rejects_nonfinite_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nominal_context.csv"
            names = list(ISAACLAB_JOINT_NAMES)
            fingers = [f"joint_{i:02d}" for i in range(13, -1, -1)]
            fields = ["sim_time", "frame_index", "phase", "root_z",
                      "root_qw", "root_qx", "root_qy", "root_qz",
                      "root_wx", "root_wy", "root_wz"]
            fields += [f"{prefix}:{name}" for prefix in ("state_q", "state_dq")
                       for name in names]
            fields += [f"finger_ref:{name}" for name in fingers]
            fields += [f"nominal_time_offset:{slot:02d}" for slot in range(10)]
            fields += [f"nominal_pos:{slot:02d}:{name}" for slot in range(10) for name in names]
            fields += [f"nominal_vel:{slot:02d}:{name}" for slot in range(10) for name in names]
            fields += [f"nominal_quat:{slot:02d}:{axis}" for slot in range(10)
                       for axis in "wxyz"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fields)
                writer.writeheader()
                for frame in range(16):
                    row = dict.fromkeys(fields, 0.)
                    row.update(sim_time=frame * .02, frame_index=frame, phase="lower",
                               root_z=.8, root_qw=1.)
                    row.update({f"finger_ref:{name}": i for i, name in enumerate(fingers)})
                    row.update({f"nominal_time_offset:{slot:02d}": slot * .1
                                for slot in range(10)})
                    row.update({f"nominal_quat:{slot:02d}:w": 1. for slot in range(10)})
                    writer.writerow(row)
            rows = read_context(path)
            np.testing.assert_array_equal(rows[0]["fingers"], np.arange(14))
            # A value-level corruption must be rejected before it can reach a training window.
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                records = list(reader)
                headers = reader.fieldnames
            records[0]["nominal_vel:00:" + names[0]] = "nan"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, headers)
                writer.writeheader()
                writer.writerows(records)
            with self.assertRaisesRegex(ValueError, "Invalid phase"):
                read_context(path)

    def test_arm_perturbation_activates_only_after_phase_delay(self):
        provider = ScheduledArmOffset(phase="lower", delay=.04,
                                      joint="right_wrist_pitch_joint", amplitude=.1, perturb=True)
        reference = ReferenceSequence(np.arange(46) * .02, np.zeros((46, 29)),
                                      np.tile([1., 0., 0., 0.], (46, 1)))
        simulation = SimpleNamespace(task_phase="reach", data=SimpleNamespace(time=1.),
                                     frame_index=50, event=lambda *args, **kwargs: None)
        with patch("risk_residual.teacher.activation_fingerprint", return_value="same"):
            self.assertFalse(np.count_nonzero(provider.request(simulation, reference)))
            simulation.task_phase = "lower"
            self.assertFalse(np.count_nonzero(provider.request(simulation, reference)))
            simulation.data.time = 1.02
            self.assertFalse(np.count_nonzero(provider.request(simulation, reference)))
            simulation.data.time = 1.04
            simulation.frame_index += 2
            result = provider.request(simulation, reference)
        index = ISAACLAB_JOINT_NAMES.index("right_wrist_pitch_joint")
        np.testing.assert_allclose(result[:, index], .1)
        self.assertFalse(np.count_nonzero(result[:, [i for i in range(29) if i != index]]))
        self.assertEqual(provider.activation["frame_index"], 52)
        with self.assertRaises(ValueError):
            ScheduledArmOffset(phase="lower", delay=0, joint="left_knee_joint",
                               amplitude=.1, perturb=True)

    def test_planned_reference_interpolates_risk_clock(self):
        times = np.arange(10) * .1
        row = {"time": 2., "slot_times": times,
               "pos": np.repeat(times[:, None], 29, axis=1),
               "quat": np.tile([1., 0., 0., 0.], (10, 1))}
        result = planned(row, 2. + np.arange(8) * .04)
        self.assertEqual(result.shape, (8, 62))
        np.testing.assert_allclose(result[:, 0], np.arange(8) * .04, atol=1e-7)
        np.testing.assert_allclose(result[:, 29], 1., atol=1e-6)

    def test_pair_rejects_different_activation_and_changed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branches = {}
            for name in ("teacher", "nominal"):
                branch = root / name
                (branch / "rollout").mkdir(parents=True)
                (branch / "scene.xml").write_text("<scene/>")
                (branch / "rollout/metadata.json").write_text('{"model_sha256":"model"}')
                report = branch / "report.json"
                report.write_text('{"status":"passed","task_success":true,"physics_executed":true,"sonic_executed":true}')
                branches[name] = {"report": str(report), "report_sha256": sha256(report),
                                  "scene_sha256": sha256(branch / "scene.xml"),
                                  "model_sha256": "model",
                                  "task_success": True, "status": "passed",
                                  "physics_executed": True, "sonic_executed": True,
                                  "activation": {"snapshot_sha256": "same", "frame_index": 42,
                                                 "time": 2.5}}
            pair = {"pair_state_verified": True, "teacher_verified": True,
                    "recovery_verified": False,
                    "branches": branches}
            path = root / "pair.json"
            path.write_text(json.dumps(pair))
            self.assertEqual(pair_details(path), pair)
            pair["branches"]["nominal"]["activation"]["snapshot_sha256"] = "different"
            path.write_text(json.dumps(pair))
            with self.assertRaisesRegex(ValueError, "nominal decision state"):
                pair_details(path)
            pair["branches"]["nominal"]["activation"]["snapshot_sha256"] = "same"
            path.write_text(json.dumps(pair))
            (root / "teacher/report.json").write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError, "changed since collection"):
                pair_details(path)

    def test_recovery_requires_failed_nominal_with_valid_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branches = {}
            for name, success in (("teacher", True), ("nominal", False)):
                branch = root / name
                (branch / "rollout").mkdir(parents=True)
                (branch / "scene.xml").write_text("<scene/>")
                (branch / "rollout/metadata.json").write_text('{"model_sha256":"model"}')
                report = branch / "report.json"
                report.write_text(json.dumps({"status": "passed", "task_success": success,
                                              "physics_executed": True, "sonic_executed": True}))
                branches[name] = {"report": str(report), "report_sha256": sha256(report),
                                  "scene_sha256": sha256(branch / "scene.xml"),
                                  "model_sha256": "model", "task_success": success,
                                  "status": "passed", "physics_executed": True,
                                  "sonic_executed": True,
                                  "activation": {"snapshot_sha256": "same", "frame_index": 42,
                                                 "time": 2.5}}
            path = root / "pair.json"
            pair = {"pair_state_verified": True, "teacher_verified": True,
                    "recovery_verified": True, "branches": branches}
            path.write_text(json.dumps(pair))
            self.assertTrue(pair_details(path)["recovery_verified"])
            pair["branches"]["nominal"]["physics_executed"] = False
            path.write_text(json.dumps(pair))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                pair_details(path)


if __name__ == "__main__":
    unittest.main()
