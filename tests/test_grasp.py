"""Task/batch interface checks using synthetic states, not physical grasp evidence."""

import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from baseline.common import ROOT
from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.grasp import (GraspEvaluator, build_scene, phase_constraints, wrist_pose,
                            ground_scene, approach_constraints, settle_constraints, approach_state,
                            PREPARE_PITCH_RAD, hand_alignment, initial_body_reference,
                            DIRECT_START_PHASES)
from baseline.simulation import SonicSimulation


FIXTURE = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="1 1 .1"/>
  <geom name="task_table" type="box" pos=".4 -.2 .67" size=".24 .24 .03"/>
  <body name="pelvis" pos="0 0 .8"><freejoint name="root"/>
    <geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
    <body name="right_wrist_yaw_link" pos=".4 -.2 .15">
      <site name="task_grasp_center" pos=".125 .035 0" size=".004"/>
      <body name="right_hand_thumb_1_link" pos="0 .03 0">
        <geom type="box" size=".02 .012 .02" mass=".1"/>
      </body>
      <body name="right_hand_index_1_link" pos="0 -.03 0">
        <geom type="box" size=".02 .012 .02" mass=".1"/>
      </body>
    </body>
  </body>
  <body name="task_block" pos=".4 -.2 .95"><freejoint name="task_block_free"/>
    <geom name="task_block_geom" type="box" size=".03 .03 .03" mass=".08"/>
  </body>
</worldbody></mujoco>"""


class GraspMetricTests(unittest.TestCase):
    def fixture(self, directory):
        model = mujoco.MjModel.from_xml_string(FIXTURE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        simulation = SimpleNamespace(model=model, data=data, root_q=0, frame_index=0,
                                     finger_names=(), finger_q=np.array([], dtype=int),
                                     finger_target=np.zeros(0))
        return simulation, GraspEvaluator(simulation, directory)

    def test_contact_hold_is_continuous_and_open_hand_is_not_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                def force(model, data, contact_index, output):
                    output[0] = 1.0
                with patch("baseline.grasp.mujoco.mj_contactForce", side_effect=force):
                    evaluator.observe(sim)
                    self.assertIsNone(evaluator.failure)
                    self.assertIsNone(evaluator.success_time)
                    evaluator.phase = "lift"
                    sim.data.time = .005
                    evaluator.observe(sim)
                    sim.data.time = 1.0
                    evaluator.observe(sim)
                    self.assertIsNone(evaluator.success_time)
                    with patch("baseline.grasp.mujoco.mj_contactForce"):
                        sim.data.time = 1.005
                        evaluator.observe(sim)
                    self.assertIsNone(evaluator.hold_start)
                    sim.data.time = 1.01
                    evaluator.observe(sim)
                    sim.data.time = 3.01
                    evaluator.observe(sim)
                    self.assertTrue(evaluator.summary()["task_success"])
                    # A later fall invalidates the earlier success.
                    sim.data.qpos[2] = .2
                    sim.data.time = 3.015
                    evaluator.observe(sim)
                    self.assertFalse(evaluator.summary()["task_success"])
                    self.assertEqual(evaluator.failure, "fall")
            finally:
                evaluator.close()

    def test_rotated_lowest_point(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                sim.data.qpos[evaluator.block_q + 2] = .785
                sim.data.qpos[evaluator.block_q + 3:evaluator.block_q + 7] = [
                    np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0]
                evaluator.observe(sim)
                self.assertAlmostEqual(evaluator.max_clearance, .085 - np.sqrt(2) * .03)
            finally:
                evaluator.close()

    def test_later_contact_loss_is_not_reported_as_retained_success(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            evaluator.phase = "hold"
            try:
                def force(model, data, index, output):
                    output[0] = 1.
                with patch("baseline.grasp.mujoco.mj_contactForce", side_effect=force):
                    evaluator.observe(sim)
                    sim.data.time = 2.
                    evaluator.observe(sim)
                self.assertTrue(evaluator.summary()["task_success"])
                sim.data.time = 2.005
                with patch("baseline.grasp.mujoco.mj_contactForce"):
                    evaluator.observe(sim)
                result = evaluator.summary()
                self.assertTrue(result["success_threshold_reached"])
                self.assertFalse(result["retained_at_end"])
                self.assertFalse(result["task_success"])
                self.assertEqual(result["failure_reason"], "grasp_lost_after_success")
                self.assertEqual(result["post_success_loss_samples"], 1)
            finally:
                evaluator.close()

    def test_hand_table_contacts_are_allowed_but_other_robot_contacts_stop(self):
        import xml.etree.ElementTree as ET

        cases = [(name, True) for name in (
            "right_hand_index_1_link", "left_hand_thumb_1_link",
            "right_hand_palm_link", "left_hand_palm_link",
            "right_wrist_yaw_link", "left_wrist_yaw_link")]
        cases += [(name, False) for name in (
            "right_wrist_pitch_link", "right_elbow_link", "torso_link")]
        for table_name in ("task_table", "task_leg_fixture"):
            for body_name, allowed in cases:
                with self.subTest(table=table_name, body=body_name), \
                        tempfile.TemporaryDirectory() as directory:
                    xml = ET.fromstring(FIXTURE)
                    xml.find(".//geom[@name='task_table']").set("name", table_name)
                    wrist = xml.find(".//body[@name='right_wrist_yaw_link']")
                    for child in list(wrist):
                        wrist.remove(child)
                    if body_name == "right_wrist_yaw_link":
                        contact_body = wrist
                        contact_body.set("pos", ".4 -.2 -.095")
                    else:
                        contact_body = ET.SubElement(xml.find(".//body[@name='pelvis']"),
                                                     "body", name=body_name, pos=".4 -.2 -.095")
                    ET.SubElement(contact_body, "geom", type="box", size=".02 .012 .02", mass=".1")
                    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    sim = SimpleNamespace(model=model, data=data, root_q=0, frame_index=0,
                                          finger_names=(), finger_q=np.array([], dtype=int),
                                          finger_target=np.zeros(0))
                    evaluator = GraspEvaluator(sim, directory)
                    try:
                        self.assertGreater(data.ncon, 0)
                        evaluator.observe(sim)
                        if allowed:
                            self.assertIsNone(evaluator.failure)
                            self.assertIsNone(evaluator.first_failure)
                            self.assertEqual(evaluator.summary()["hand_table_contact_steps"], 1)
                        else:
                            self.assertEqual(evaluator.failure, "prohibited_robot_table_contact")
                            self.assertIn(body_name, evaluator.first_failure["bodies"])
                        # Table contact cannot substitute for opposing block contacts.
                        self.assertEqual(evaluator.contact_steps, 0)
                        self.assertFalse(evaluator.summary()["task_success"])
                    finally:
                        evaluator.close()

    def test_named_finger_bounds_and_rate(self):
        sim = SonicSimulation.__new__(SonicSimulation)
        sim.finger_names = ("right_hand_index_0_joint", "left_hand_index_0_joint")
        sim.finger_ranges = np.array([[0, 1.5], [0, 1.5]])
        sim.finger_target = np.zeros(2)
        sim.command_fingers({"right_hand_index_0_joint": 1.0})
        np.testing.assert_allclose(sim.finger_target, [.05, 0])
        with self.assertRaises(ValueError):
            sim.command_fingers({"right_hand_index_0_joint": 2.0})
        with self.assertRaises(ValueError):
            sim.command_fingers({"wrong_joint": 1.0})

    def test_reach_raises_before_crossing_table_and_rotations_are_proper(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                start, rotation = wrist_pose(sim)
                constraints = phase_constraints(sim, "reach", 2.4, np.zeros(36), rotation)
                frames = np.asarray(constraints["frame_indices"])
                positions = np.asarray(constraints["wrist_positions"])
                early = frames <= .45 * 59
                np.testing.assert_allclose(positions[early, :2], np.tile(start[:2], (early.sum(), 1)))
                self.assertGreaterEqual(positions[-1, 2], start[2])
                rotations = np.asarray(constraints["wrist_rotations"])
                np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-8)
                np.testing.assert_allclose(rotations[0], rotation, atol=1e-8)
                self.assertEqual(constraints["frame_indices"][-1], 59)
            finally:
                evaluator.close()

    def test_task_scene_keeps_two_free_joints_and_cameras(self):
        repo = ROOT / "third_party/sonic"
        if not repo.is_dir():
            self.skipTest("Pinned SONIC source/meshes not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.xml"
            settings = build_scene(repo, path, [.4, -.22])
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(int(model.joint("floating_base_joint").type[0]), mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(int(model.joint("task_block_free").type[0]), mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(model.nu, 43)
            self.assertEqual(model.ncam, 2)
            root_q = int(model.joint("floating_base_joint").qposadr[0])
            self.assertAlmostEqual(model.qpos0[root_q], -.34)
            self.assertEqual(model.opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
            self.assertEqual(model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
            self.assertEqual(model.opt.impratio, 10.)
            self.assertEqual(model.opt.noslip_iterations, 0)
            self.assertEqual(settings["grounding_source"], "mujoco_state")
            self.assertFalse(settings["camera_input"])

    def test_direct_start_places_root_at_target_and_skips_locomotion_phases(self):
        repo = ROOT / "third_party/sonic"
        if not repo.is_dir():
            self.skipTest("Pinned SONIC source/meshes not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.xml"
            settings = build_scene(repo, path, [.4, -.22], start_back=0., direct_start=True)
            model = mujoco.MjModel.from_xml_path(str(path))
            root_q = int(model.joint("floating_base_joint").qposadr[0])
            np.testing.assert_allclose(model.qpos0[root_q:root_q + 2], [.11, -.08], atol=1e-8)
            self.assertTrue(settings["direct_start"])
            self.assertTrue(settings["hand_table_contact_allowed"])
            self.assertIn("non-hand robot-table", settings["prohibited_contacts"])
            np.testing.assert_allclose(settings["robot_start_xy_m"], [.11, -.08], atol=1e-8)
            self.assertEqual([name for name, _ in DIRECT_START_PHASES],
                             ["reach", "lower", "close", "lift", "hold"])
            self.assertEqual(settings["phases"], [list(phase) for phase in DIRECT_START_PHASES])

    def test_acquisition_region_uses_measured_wrist_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                wrist, rotation = wrist_pose(sim)
                for local, expected in (([.13, .045, -.005], True),
                                        ([.13, .045, -.043], False),
                                        ([.13, .045, -.07], False),
                                        ([.20, .045, -.005], False)):
                    sim.data.qpos[evaluator.block_q:evaluator.block_q + 3] = wrist + rotation @ local
                    result = hand_alignment(sim)
                    self.assertEqual(result["ready"], expected)
                    np.testing.assert_allclose(result["cube_in_wrist_m"], local)
            finally:
                evaluator.close()

    def test_task_initial_pose_does_not_change_frozen_policy_defaults(self):
        default = np.arange(29) / 100.
        before = default.copy()
        parked = initial_body_reference(default)
        np.testing.assert_array_equal(default, before)
        changed = {ISAACLAB_JOINT_NAMES[i] for i in np.flatnonzero(parked != default)}
        self.assertEqual(changed, {f"{side}_{joint}_joint" for side in ("left", "right")
                                  for joint in ("shoulder_pitch", "elbow")})

    def test_preparation_keeps_hand_back_and_conditions_only_small_waist_pitch(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                start, rotation = wrist_pose(sim)
                standing = np.r_[sim.data.qpos[:7], np.zeros(29)]
                original = standing.copy()
                goals = phase_constraints(sim, "prepare", 1.6, standing, rotation)
                count = len(goals["frame_indices"])
                np.testing.assert_allclose(goals["wrist_positions"], np.repeat(start[None], count, axis=0))
                np.testing.assert_allclose(goals["wrist_rotations"], np.repeat(rotation[None], count, axis=0))
                self.assertEqual(goals["torso_pitch_rad"][0], 0)
                self.assertAlmostEqual(goals["torso_pitch_rad"][-1], PREPARE_PITCH_RAD)
                self.assertTrue((np.diff(goals["torso_pitch_rad"]) >= 0).all())
                np.testing.assert_array_equal(standing, original)
            finally:
                evaluator.close()

    def test_grounding_uses_current_state_and_is_translation_equivariant(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                initial = ground_scene(sim)
                self.assertAlmostEqual(initial["approach_target_xy"][0], -.04)
                delta = np.array([1.1, -.7, 0.])
                sim.data.qpos[:3] += delta
                sim.data.qpos[evaluator.block_q:evaluator.block_q + 3] += delta
                sim.model.geom_pos[sim.model.geom("task_table").id] += delta
                mujoco.mj_forward(sim.model, sim.data)
                shifted = ground_scene(sim)
                np.testing.assert_allclose(shifted["approach_target_xy"], np.array(initial["approach_target_xy"]) + delta[:2])
                np.testing.assert_allclose(shifted["cube_in_base"], initial["cube_in_base"])
                self.assertAlmostEqual(shifted["position_error_m"], initial["position_error_m"])
                # Object displacement changes the grounded target without changing text.
                sim.data.qpos[evaluator.block_q + 1] += .1
                mujoco.mj_forward(sim.model, sim.data)
                self.assertAlmostEqual(ground_scene(sim)["approach_target_xy"][1], shifted["approach_target_xy"][1] + .1)
            finally:
                evaluator.close()

    def test_approach_path_reaches_target_and_does_not_pin_feet(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, evaluator = self.fixture(directory)
            try:
                path = approach_constraints(sim, 3.2)
                self.assertEqual(path["kind"], "root_path")
                self.assertNotIn("standing_qpos", path)
                np.testing.assert_allclose(path["root_positions_xy"][0], sim.data.qpos[:2])
                np.testing.assert_allclose(path["root_positions_xy"][-1], ground_scene(sim)["approach_target_xy"])
                self.assertEqual(path["frame_indices"][-1], 79)
                sim.body_pose = lambda: np.r_[sim.data.qpos[:7], np.zeros(29)]
                stop = settle_constraints(sim, 1.6)
                np.testing.assert_allclose(stop["root_positions_xy"], [sim.data.qpos[:2]] * 2)
            finally:
                evaluator.close()

    def test_arrival_checks_position_heading_speed_and_both_feet(self):
        import xml.etree.ElementTree as ET
        xml = ET.fromstring(FIXTURE)
        pelvis = xml.find(".//body[@name='pelvis']")
        for side, y in (("left", .1), ("right", -.1)):
            foot = ET.SubElement(pelvis, "body", name=f"{side}_ankle_roll_link", pos=f"0 {y} -.79")
            ET.SubElement(foot, "geom", type="box", size=".06 .03 .02", mass=".1")
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        data = mujoco.MjData(model)
        sim = SimpleNamespace(model=model, data=data, root_q=0, root_v=0)
        mujoco.mj_forward(model, data)
        data.qpos[:2] = ground_scene(sim)["approach_target_xy"]
        mujoco.mj_forward(model, data)
        self.assertTrue(approach_state(sim)["ready"])
        data.qvel[0] = .2
        self.assertFalse(approach_state(sim)["ready"])
        data.qvel[0] = 0
        data.qpos[0] -= .2
        mujoco.mj_forward(model, data)
        self.assertFalse(approach_state(sim)["ready"])
        data.qpos[0] += .2
        data.qpos[3:7] = [np.cos(.2), 0, 0, np.sin(.2)]
        mujoco.mj_forward(model, data)
        self.assertFalse(approach_state(sim)["ready"])
        data.qpos[3:7] = [1, 0, 0, 0]
        model.body_pos[model.body("right_ankle_roll_link").id, 2] += .2
        mujoco.mj_forward(model, data)
        self.assertEqual(approach_state(sim)["feet_on_floor"], ["left"])
        self.assertFalse(approach_state(sim)["ready"])


class PoseConstraintTests(unittest.TestCase):
    def test_root_waypoint_axes_headings_and_history_offsets(self):
        repo = ROOT / "third_party/ardy"
        if not repo.is_dir():
            self.skipTest("Pinned ARDY skeleton source not installed")
        sys.path.insert(0, str(repo))
        import torch
        from ardy.skeleton import G1Skeleton34
        from baseline.ardy import ArdyService
        captured = []
        def conditions(constraints, **kwargs):
            captured.extend(constraints)
            return torch.zeros(kwargs["length"], 1), torch.zeros(kwargs["length"], 1)
        service = ArdyService.__new__(ArdyService)
        service.torch, service.device = torch, torch.device("cpu")
        service.converter = SimpleNamespace(mujoco_to_ardy_matrix=torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]))
        service.model = SimpleNamespace(skeleton=G1Skeleton34(), motion_rep=SimpleNamespace(create_conditions_from_constraints=conditions))
        goals = {"schema_version": 1, "kind": "root_path", "coordinate_frame": "mujoco_world", "units": "SI",
                 "frame_indices": [0, 4], "root_positions_xy": [[-.4, -.2], [-.1, -.15]],
                 "root_heading_rad": [0., np.pi / 2]}
        service.pose_conditions(goals, frames=8, history_count=16)
        constraint = captured[0]
        self.assertEqual(constraint.frame_indices.tolist(), [16, 20])
        np.testing.assert_allclose(constraint.root_2d, [[-.2, -.4], [-.15, -.1]], atol=1e-6)
        np.testing.assert_allclose(constraint.global_root_heading, [0, np.pi / 2], atol=1e-6)
        with self.assertRaises(ValueError):
            service.pose_conditions({**goals, "root_positions_xy": [[float("nan"), 0], [0, 0]]}, 8, 16)

    def test_hand_constraint_preserves_geometry_and_offsets_future_frames(self):
        repo = ROOT / "third_party/ardy"
        if not repo.is_dir():
            self.skipTest("Pinned ARDY skeleton source not installed")
        sys.path.insert(0, str(repo))
        import torch
        from ardy.skeleton import G1Skeleton34
        from baseline.ardy import ArdyService

        skeleton = G1Skeleton34()
        template = {"posed_joints": skeleton.neutral_joints.float()[None, None].repeat(1, 4, 1, 1),
                    "global_rot_mats": torch.eye(3).repeat(1, 4, skeleton.nbjoints, 1, 1)}
        captured = []
        def conditions(constraints, **kwargs):
            captured.extend(constraints)
            return torch.zeros(kwargs["length"], 1), torch.zeros(kwargs["length"], 1)
        service = ArdyService.__new__(ArdyService)
        service.torch, service.device = torch, torch.device("cpu")
        coordinate = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        service.converter = SimpleNamespace(mujoco_to_ardy_matrix=coordinate)
        service.model = SimpleNamespace(skeleton=skeleton, motion_rep=SimpleNamespace(
            inverse=lambda *a, **kw: template, create_conditions_from_constraints=conditions))
        service.history_features = lambda poses: torch.zeros(1, 4, 1)
        standing = np.zeros(36)
        standing[3] = 1
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        requested = np.array([[.2, -.2, .9], [.35, -.25, 1.0]])
        goals = {"schema_version": 1, "coordinate_frame": "mujoco_world", "units": "SI",
                 "frame_indices": [0, 4], "wrist_positions": requested.tolist(),
                 "standing_qpos": standing.tolist(), "standing_wrist_rotation": np.eye(3).tolist(),
                 "wrist_rotations": np.stack([np.eye(3), rotation]).tolist()}
        service.pose_conditions(goals, frames=8, history_count=16)
        constraint = captured[0]
        self.assertEqual(constraint.frame_indices.tolist(), [16, 20])
        wrist = skeleton.bone_index["right_wrist_yaw_skel"]
        endpoint = skeleton.bone_index["right_hand_roll_skel"]
        np.testing.assert_allclose(constraint.global_joints_positions[:, wrist].numpy() @ coordinate.numpy(),
                                   requested, atol=1e-6)
        original_offset = (template["posed_joints"][0, -1, endpoint] -
                           template["posed_joints"][0, -1, wrist]).numpy() @ coordinate.numpy()
        final_offset = (constraint.global_joints_positions[-1, endpoint] -
                        constraint.global_joints_positions[-1, wrist]).numpy() @ coordinate.numpy()
        np.testing.assert_allclose(final_offset, rotation @ original_offset, atol=1e-6)
        self.assertIn(endpoint, constraint.pos_indices.tolist())
        # The additional torso condition must retain hand/foot goals and use
        # future frame indices. It must not replace all body coordinates.
        captured.clear()
        histories = []
        def record_history(poses):
            histories.append(poses.copy())
            return torch.zeros(1, len(poses), 1)
        service.history_features = record_history
        service.pose_conditions({**goals, "torso_pitch_rad": [0., .14]}, frames=8, history_count=16)
        self.assertEqual(len(captured), 2)
        self.assertIn(skeleton.bone_index["left_ankle_roll_skel"], captured[0].pos_indices.tolist())
        torso = skeleton.bone_index["waist_pitch_skel"]
        self.assertEqual(captured[1].indices.tolist(), [[16, torso], [20, torso]])
        pitch_index = 7 + ISAACLAB_JOINT_NAMES.index("waist_pitch_joint")
        np.testing.assert_allclose(histories[-1][:, pitch_index], [0., .14, .14, .14])
        other = [i for i in range(36) if i != pitch_index]
        np.testing.assert_array_equal(histories[-1][:, other], np.repeat(standing[None, other], 4, axis=0))
        for pitch in ([0., float("nan")], [.7, .7], [0.]):
            with self.assertRaises(ValueError):
                service.pose_conditions({**goals, "torso_pitch_rad": pitch}, frames=8, history_count=16)
        with self.assertRaises(ValueError):
            service.pose_conditions({**goals, "frame_indices": [0, 8]}, frames=8, history_count=16)
        with self.assertRaises(ValueError):
            service.pose_conditions({**goals, "coordinate_frame": "camera"}, frames=8, history_count=16)


if __name__ == "__main__":
    unittest.main()
