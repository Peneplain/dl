"""Synthetic FK projection tests, not model compatibility or grasp evidence."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.kimodo_projection import KimodoConstraintProjector, RIGHT_ARM_JOINTS


def fixture_xml(path):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", limited="true", range="-3 3")
    ET.SubElement(default, "geom", type="sphere", size=".02", mass=".1")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="pelvis")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom")
    arm = body
    axes = ("0 1 0", "1 0 0", "0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 0 1")
    positions = ("0 -.2 .2", "0 0 0", "0 0 0", "0 0 -.2", "0 0 -.2", "0 0 0", "0 0 0")
    for name, axis, pos in zip(RIGHT_ARM_JOINTS, axes, positions):
        arm = ET.SubElement(arm, "body", name=name.replace("_joint", "_link"), pos=pos)
        ET.SubElement(arm, "joint", name=name, axis=axis)
        ET.SubElement(arm, "geom")
    for name in ISAACLAB_JOINT_NAMES:
        if name in RIGHT_ARM_JOINTS:
            continue
        other = ET.SubElement(body, "body", name=name.replace("_joint", "_link"))
        ET.SubElement(other, "joint", name=name, axis="0 1 0")
        ET.SubElement(other, "geom")
    ET.ElementTree(root).write(path)


def goals(position, rotation, anchor):
    return {"schema_version": 1, "coordinate_frame": "mujoco_world", "units": "SI",
            "frame_indices": [0, 4], "wrist_positions": [list(position), list(position)],
            "wrist_rotations": [rotation.tolist(), rotation.tolist()],
            "standing_qpos": anchor.tolist()}


class KimodoProjectionTests(unittest.TestCase):
    def test_timestamped_goals_use_25hz_to_30hz_and_slerp(self):
        anchor = np.r_[1., 2., .8, 1., 0., 0., 0., np.zeros(29)]
        c = goals([0., 0., 0.], np.eye(3), anchor)
        c["wrist_positions"][1] = [1., 0., 0.]
        c["wrist_rotations"][1] = Rotation.from_euler("z", 90, degrees=True).as_matrix().tolist()
        positions, rotations, result_anchor = KimodoConstraintProjector._goals(c, 6, 30.)
        self.assertAlmostEqual(positions[2, 0], (2. / 30.) / (4. / 25.))
        self.assertAlmostEqual(positions[-1, 0], 1.)
        self.assertAlmostEqual(Rotation.from_matrix(rotations[2]).as_euler("zyx")[0],
                               np.pi / 2 * positions[2, 0])
        np.testing.assert_array_equal(result_anchor, anchor)

    def test_projection_improves_pose_and_preserves_all_non_arm_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "g1_fixture.xml"
            fixture_xml(xml)
            # Deliberately scramble stream order: the boundary must use names.
            names = tuple(reversed(ISAACLAB_JOINT_NAMES))
            projector = KimodoConstraintProjector(xml, names)
            row = np.r_[0., 0., .8, 1., 0., 0., 0., np.full(29, .01)]
            anchor = row.copy()
            anchor[:2] = [1., 2.]
            world = row.copy()
            world[:2] += anchor[:2]
            arm = row[projector.arm_columns].copy()
            arm[0] -= .08
            arm[3] += .1
            target, rotation = projector._pose(world, arm)
            nominal = np.repeat(row[None], 6, axis=0)
            fitted, report = projector.project(nominal, goals(target, rotation, anchor), 30.)
            untouched = np.ones(36, dtype=bool)
            untouched[projector.arm_columns] = False
            np.testing.assert_array_equal(fitted[:, untouched], nominal[:, untouched])
            self.assertGreater(report["position_error_before_m_mean"], 1e-3)
            self.assertLess(report["position_error_after_m_max"], 1e-3)
            # Orientation is a soft preference after physical-hinge projection.
            self.assertLess(report["rotation_error_after_rad_max"], .01)
            self.assertLessEqual(report["observed_max_offset_rad"], projector.MAX_OFFSET_RAD)
            self.assertFalse(report["physics_stepped"])
            self.assertFalse(report["execution_feedback_used"])
            self.assertFalse(report["teacher_used"])

    def test_effector_offset_is_rotated_and_fitted_without_non_arm_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "g1_fixture.xml"
            fixture_xml(xml)
            projector = KimodoConstraintProjector(xml, ISAACLAB_JOINT_NAMES)
            row = np.r_[0., 0., .8, 1., 0., 0., 0., np.full(29, .01)]
            arm = row[projector.arm_columns].copy()
            arm[0] -= .1
            arm[3] += .12
            target, rotation = projector._pose(row, arm)
            constraints = goals(target, rotation, row)
            constraints["wrist_effector_offset_m"] = [.125, .035, 0.]
            fitted, report = projector.project(row[None], constraints, 30.)
            position, result_rotation = projector._pose(fitted[0])
            offset = np.asarray(constraints["wrist_effector_offset_m"])
            error = np.linalg.norm(position + result_rotation @ offset - target - rotation @ offset)
            self.assertLess(error, 1e-3)
            self.assertAlmostEqual(error, report["effector_error_after_m_terminal"])
            self.assertEqual(report["effector_offset_wrist_m"], offset.tolist())
            self.assertGreater(report["effector_error_before_m_mean"], error)
            untouched = np.ones(36, dtype=bool)
            untouched[projector.arm_columns] = False
            np.testing.assert_array_equal(fitted[0, untouched], row[untouched])
            for invalid in ([0., 0.], [1., 0., 0.], [float("nan"), 0., 0.]):
                constraints["wrist_effector_offset_m"] = invalid
                with self.assertRaises(ValueError):
                    projector.project(row[None], constraints, 30.)

    def test_unreachable_goals_remain_bounded_and_report_error(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "g1_fixture.xml"
            fixture_xml(xml)
            projector = KimodoConstraintProjector(xml, ISAACLAB_JOINT_NAMES)
            row = np.r_[0., 0., .8, 1., 0., 0., 0., np.zeros(29)]
            fitted, report = projector.project(row[None], goals([5., 5., 5.], np.eye(3), row), 30.)
            self.assertLessEqual(np.max(np.abs(fitted[:, 7:] - row[None, 7:])),
                                 projector.MAX_OFFSET_RAD + 1e-8)
            self.assertGreater(report["position_error_after_m_max"], 1.)

    def test_root_path_is_not_projected_and_invalid_goals_are_rejected(self):
        projector = KimodoConstraintProjector.__new__(KimodoConstraintProjector)
        q = np.r_[0., 0., .8, 1., 0., 0., 0., np.zeros(29)][None]
        out, report = projector.project(q, {"kind": "root_path"}, 30.)
        np.testing.assert_array_equal(out, q)
        self.assertFalse(report["applied"])
        c = goals([0., 0., 0.], np.eye(3), q[0])
        c["wrist_rotations"][0][0][0] = 2.
        with self.assertRaises(ValueError):
            projector._goals(c, 6, 30.)
        for fps in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                projector.project(q, None, fps)


if __name__ == "__main__":
    unittest.main()
