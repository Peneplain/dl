"""Offline Kimodo boundary and selector tests; no model inference is claimed here."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.kimodo import (
    KimodoService, _JointRotationConstraint, _kimodo_joint_names,
    _resample_reference, _verify_checkpoint,
)
from baseline.session import make_plan
from scripts.package_baseline import source_files
from scripts.run import config_for, parse_args


class KimodoConversionTests(unittest.TestCase):
    def test_resample_preserves_named_reference_and_exact_50hz_grid(self):
        times = np.arange(0., 1.001, 1 / 30.)
        positions = np.stack([times + j for j in range(29)], axis=-1)
        quats = np.tile(np.array([[1., 0., 0., 0.]], dtype=np.float32), (len(times), 1))
        reference = _resample_reference(ReferenceSequence(times, positions, quats), 50.)
        self.assertEqual(reference.joint_pos.shape[1], 29)
        self.assertTrue(np.allclose(np.diff(reference.times), .02))
        self.assertTrue(np.allclose(reference.body_quat[:, 0], 1.))

    def test_joint_order_is_read_from_converter_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1.xml"
            world = ET.Element("worldbody")
            for name in ISAACLAB_JOINT_NAMES:
                ET.SubElement(world, "joint", name=name, type="hinge")
            root = ET.Element("mujoco")
            root.append(world)
            ET.ElementTree(root).write(path)
            fake = SimpleNamespace(xml_path=path)
            self.assertEqual(_kimodo_joint_names(path, fake), tuple(ISAACLAB_JOINT_NAMES))

    def test_joint_rotation_constraint_indices_use_the_bound_torch_module(self):
        import torch

        skeleton = SimpleNamespace(bone_index={"waist_yaw_skel": 2, "waist_roll_skel": 4})
        frame_indices = torch.tensor([0, 3], dtype=torch.long)
        rotations = torch.eye(3).repeat(2, 2, 1, 1)
        constraint = _JointRotationConstraint(
            skeleton, frame_indices, rotations, ("waist_yaw_skel", "waist_roll_skel"), torch
        )
        self.assertEqual(
            constraint._indices().tolist(),
            [[0, 2], [0, 4], [3, 2], [3, 4]],
        )
        cropped = constraint.crop_move(1, 4)
        self.assertEqual(cropped._indices().tolist(), [[2, 2], [2, 4]])

    def test_measured_history_is_permuted_by_name_before_native_conversion(self):
        service = object.__new__(KimodoService)
        service.fps = 30.0
        service.joint_names = tuple(reversed(ISAACLAB_JOINT_NAMES))
        captured = {}

        def convert(qpos, source_fps):
            captured["qpos"] = qpos.copy()
            captured["fps"] = source_fps
            return {"fixture": qpos}

        service.cpu_converter = SimpleNamespace(qpos_to_motion_dict=convert)
        shared = np.arange(36, dtype=np.float32)[None]
        original = shared.copy()
        service._cpu_motion_from_qpos(shared)
        native = captured["qpos"]
        self.assertEqual(native.shape, (8, 36))
        np.testing.assert_array_equal(native[:, :7], np.repeat(shared[:, :7], 8, axis=0))
        np.testing.assert_array_equal(native[0, 7:], shared[0, 7:][::-1])
        np.testing.assert_array_equal(native[-1], native[0])
        np.testing.assert_array_equal(shared, original)
        self.assertEqual(captured["fps"], 30.0)

    def test_shared_constraints_keep_the_declared_standing_rotation_anchor(self):
        import sys
        import torch

        class FakeConstraint:
            def __init__(self, skeleton, indices, positions, rotations, root, *, joint_names):
                self.frame_indices = indices
                self.positions = positions
                self.rotations = rotations
                self.joint_names = joint_names

            def to(self, **kwargs):
                return self

        service = object.__new__(KimodoService)
        service.torch = torch
        service.device = torch.device("cpu")
        service.fps = 30.0
        skeleton = SimpleNamespace(
            bone_index={"right_wrist_yaw_skel": 0, "right_hand_roll_skel": 1},
            right_hand_joint_names=("right_wrist_yaw_skel", "right_hand_roll_skel"),
        )
        service.model = SimpleNamespace(skeleton=skeleton)
        service._kimodo_matrix = lambda: np.eye(3, dtype=np.float32)
        anchor = np.zeros(36, dtype=np.float32)
        anchor[:3] = [1.0, 2.0, 0.75]
        anchor[3] = 1.0
        latest = anchor.copy()
        latest[:3] += [.1, -.1, .05]
        captured = []

        def standing_motion(qpos):
            captured.append(qpos.copy())
            return {"posed_joints": np.array([[[1., 2., .9], [1.1, 2., .9]]]),
                    "global_rot_mats": np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()}

        service._standing_motion = standing_motion
        # The declared anchor and desired wrist share a non-identity rotation.
        # Their rotation delta must therefore remain identity.
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        shared = {"schema_version": 1, "coordinate_frame": "mujoco_world", "units": "SI",
                  "frame_indices": [0, 4], "standing_qpos": anchor.tolist(),
                  "standing_wrist_rotation": rotation.tolist(),
                  "wrist_positions": [[1., 2., .9], [1.3, 2., .8]],
                  "wrist_rotations": [rotation.tolist()] * 2}
        native = SimpleNamespace(EndEffectorConstraintSet=FakeConstraint,
                                 Root2DConstraintSet=FakeConstraint)
        with patch.dict(sys.modules, {"kimodo.constraints": native}):
            constraints, _ = service._shared_constraints(shared, 8, latest)
        np.testing.assert_array_equal(captured[0], anchor)
        np.testing.assert_allclose(constraints[0].positions[-1, 0], [0.3, 2., 0.05], atol=1e-6)
        np.testing.assert_allclose(constraints[0].positions[-1, 1], [0.4, 2., 0.05], atol=1e-6)
        np.testing.assert_allclose(constraints[0].rotations, np.broadcast_to(np.eye(3), (2, 2, 3, 3)))
        self.assertEqual(constraints[0].frame_indices.tolist(), [0, 5])

    def test_checkpoint_requires_pinned_layout_but_allows_missing_optional_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "Kimodo-G1-RP-v1"
            required = (
                "config.yaml", "model.safetensors",
                "stats/motion/body/mean.npy", "stats/motion/body/std.npy",
                "stats/motion/global_root/mean.npy", "stats/motion/global_root/std.npy",
                "stats/motion/local_root/mean.npy", "stats/motion/local_root/std.npy",
            )
            for name in required:
                path = model / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            model_dir, manifest = _verify_checkpoint(root, "Kimodo-G1-RP-v1")
            self.assertEqual(model_dir, model.resolve())
            self.assertIsNone(manifest)


class KimodoSelectionTests(unittest.TestCase):
    def test_default_is_ardy_and_flag_is_explicit(self):
        self.assertFalse(parse_args(["batch"]).kimodo)
        args = parse_args(["batch", "--kimodo"])
        self.assertTrue(args.kimodo)
        self.assertEqual(args.kimodo_diffusion_steps, 100)
        self.assertTrue(args.kimodo_no_projection)
        self.assertFalse(parse_args(["batch", "--kimodo", "--kimodo-project-constraints"]).kimodo_no_projection)
        self.assertTrue(parse_args(["batch", "--kimodo", "--kimodo-no-projection"]).kimodo_no_projection)

    def test_kimodo_grasp_calibration_does_not_change_ardy_or_explicit_overrides(self):
        self.assertIsNone(parse_args(["batch", "--grasp"]).wrist_offset)
        self.assertIsNone(parse_args(["batch", "--kimodo"]).wrist_offset)
        self.assertEqual(parse_args(["batch", "--kimodo", "--grasp"]).wrist_offset,
                         [.125, .035, .08])
        args = parse_args(["manual", "--kimodo", "--grasp", "--wrist-offset", ".125", ".035", ".04"])
        self.assertEqual(args.wrist_offset, [.125, .035, .04])

    def test_resume_preserves_raw_projection_and_uncalibrated_grasp_settings(self):
        config = config_for(parse_args(["batch", "--kimodo", "--grasp"]))
        config["wrist_offset"] = None
        for projection in (True, False, None):
            saved = dict(config)
            if projection is None:
                saved.pop("kimodo_no_projection")
            else:
                saved["kimodo_no_projection"] = projection
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "plan.json").write_text(json.dumps(
                    {"schema_version": 2, "config": saved}))
                args = parse_args(["batch", "--resume", directory])
                self.assertIsNone(args.wrist_offset)
                self.assertEqual(args.kimodo_no_projection,
                                 True if projection is None else projection)

    def test_projection_flags_are_mutually_exclusive(self):
        from contextlib import redirect_stderr
        import io
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["batch", "--kimodo", "--kimodo-project-constraints", "--kimodo-no-projection"])

    def test_constraint_guidance_is_explicit_finite_and_positive(self):
        from contextlib import redirect_stderr
        import io

        args = parse_args(["batch", "--kimodo", "--kimodo-constraint-guidance", "1"])
        self.assertEqual(args.kimodo_constraint_guidance, 1.0)
        for value in ("nan", "inf", "0", "-1"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["batch", "--kimodo", "--kimodo-constraint-guidance", value])

    def test_plan_and_archive_scope_are_separate(self):
        root = Path(__file__).resolve().parents[1]
        b0 = {p.relative_to(root).as_posix() for p in source_files(root, "b0")}
        kimodo = {p.relative_to(root).as_posix() for p in source_files(root, "kimodo")}
        self.assertNotIn("baseline/kimodo.py", b0)
        self.assertIn("baseline/kimodo.py", kimodo)
        self.assertNotIn("baseline/kimodo_projection.py", b0)
        self.assertIn("baseline/kimodo_projection.py", kimodo)
        self.assertNotIn("tests/test_kimodo_projection.py", b0)
        self.assertIn("tests/test_kimodo_projection.py", kimodo)
        self.assertNotIn("configs/kimodo.lock.json", b0)
        self.assertIn("configs/kimodo.lock.json", kimodo)

        b0_args = parse_args(["batch"])
        kimodo_args = parse_args(["batch", "--kimodo"])
        b0_plan = make_plan(config_for(b0_args), [])
        kimodo_plan = make_plan(config_for(kimodo_args), [])
        self.assertEqual(b0_plan["method"], "B0")
        self.assertEqual(kimodo_plan["method"], "KIMODO")
        self.assertIn("kimodo_lock_sha256", kimodo_plan)
        self.assertNotIn("kimodo_lock_sha256", b0_plan)

    def test_selected_service_is_lazy_and_default_path_does_not_import_kimodo(self):
        import sys
        from baseline.execution import ExecutionRuntime

        args = parse_args(["batch", "--kimodo"])
        with tempfile.TemporaryDirectory() as directory:
            fake = SimpleNamespace(KimodoService=lambda _args: "kimodo-service")
            with patch.dict(sys.modules, {"baseline.kimodo": fake}):
                runtime = ExecutionRuntime(args, directory)
                self.assertEqual(runtime.method, "KIMODO")
                self.assertEqual(runtime.generator_name, "Kimodo")
                self.assertEqual(runtime.ensure_service(), "kimodo-service")
                runtime.close()

        args = parse_args(["batch"])
        with tempfile.TemporaryDirectory() as directory:
            fake = SimpleNamespace(ArdyService=lambda _args: "ardy-service")
            # Importing Kimodo is forbidden while actually loading default B0.
            with patch.dict(sys.modules, {"baseline.ardy": fake, "baseline.kimodo": None}):
                runtime = ExecutionRuntime(args, directory)
                self.assertEqual(runtime.method, "B0")
                self.assertEqual(runtime.generator_name, "ARDY")
                self.assertEqual(runtime.ensure_service(), "ardy-service")
                runtime.close()

    def test_hand_center_metadata_is_kimodo_only_and_does_not_mutate_shared_goals(self):
        from baseline.execution import ExecutionRuntime

        constraints = {"schema_version": 1, "wrist_positions": [[0., 0., .8]]}
        for selected in (False, True):
            argv = ["batch", "--grasp"] + (["--kimodo"] if selected else [])
            with tempfile.TemporaryDirectory() as directory:
                runtime = ExecutionRuntime(parse_args(argv), directory)
                captured = {}
                def generate(*args, **kwargs):
                    captured.update(kwargs)
                    return "generated"
                runtime.service = SimpleNamespace(generate=generate)
                runtime.run_worker = lambda work, stage: work()
                with patch("baseline.execution.grasp_center_local", return_value=np.array([.125, .035, 0.])) as geometry:
                    self.assertEqual(runtime.generate("prompt", pose_constraints=constraints), "generated")
                if selected:
                    geometry.assert_called_once_with(runtime.simulation)
                    self.assertEqual(captured["pose_constraints"]["wrist_effector_offset_m"], [.125, .035, 0.])
                    self.assertIsNot(captured["pose_constraints"], constraints)
                else:
                    geometry.assert_not_called()
                    self.assertIs(captured["pose_constraints"], constraints)
                self.assertNotIn("wrist_effector_offset_m", constraints)
                runtime.close()


if __name__ == "__main__":
    unittest.main()
