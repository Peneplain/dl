"""Shared simulator grounding, phased grasp goals and physical task assessment.

Goals condition ARDY; this module never substitutes IK/PD body control for SONIC.
Pilot scene/hand settings need calibration before freezing a comparison suite.
"""

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES


PHASES = (("approach", 2.0), ("settle", .8), ("prepare", 1.2), ("reach", 3.2), ("lower", 3.2),
          ("close", 2.0), ("lift", 4.8), ("hold", 5.0))
DIRECT_START_PHASES = PHASES[3:]
HOLD_PHASES = frozenset({"settle", "close", "hold"})
DEFAULT_START_BACK = .45
DEFAULT_TABLE_STANDOFF = .20
APPROACH_POSITION_TOLERANCE = .06
APPROACH_SPEED_LIMIT = .12
APPROACH_HEADING_TOLERANCE = np.deg2rad(12)
SETTLE_HOLD_SECONDS = .4
PREPARE_PITCH_RAD = float(np.deg2rad(8))
DEFAULT_PROMPT = "Stand still, reach with your right hand, grasp the block on the table, and lift it."
FOCUSED_PROMPTS = {
    "approach": "A person walks forward with two normal alternating steps and then stops upright. Both hands remain beside the hips.",
    "settle": "A person stops walking and stands still upright with both feet planted and both hands relaxed beside the body.",
    "prepare": "A person stands with both feet firmly planted, gently inclines the upper body forward to face down toward a block on the table, and pauses. Both hands remain beside the body.",
    "reach": "A person stands upright and slowly bends the right elbow, raising the open right hand upward beside the body, then extending the hand forward above the table. Both feet stay planted and the left arm stays at the side.",
    "lower": "A person stands upright and slowly lowers the open right hand beside a small block on a table, positioning the fingers around the block. Both feet stay planted and the left arm stays at the side.",
    "close": "A person stands upright with the right hand around a small block on a table, keeping the right wrist and elbow still while closing the fingers. Both feet stay planted and the left arm stays at the side.",
    "lift": "A person stands upright and slowly raises the right hand straight upward while holding a small block. The right elbow bends gently, both feet stay planted, and the left arm stays at the side.",
    "hold": "A person stands upright, holding a small block steadily in the raised right hand with the right elbow bent. The right hand stays still, both feet stay planted, and the left arm stays at the side.",
}
RIGHT_CLOSED = {
    "right_hand_thumb_0_joint": 0.6,
    "right_hand_thumb_1_joint": -0.55,
    "right_hand_thumb_2_joint": -1.0,
    "right_hand_middle_0_joint": 1.0,
    "right_hand_middle_1_joint": 1.1,
    "right_hand_index_0_joint": 1.0,
    "right_hand_index_1_joint": 1.1,
}
RIGHT_PRESHAPED = {name: (1.3 if name.endswith("_1_joint") and "thumb" not in name else 0.)
                  for name in RIGHT_CLOSED}


def is_hand_body(name):
    """Include fingers and palms, which the pinned G1 mounts on wrist-yaw bodies."""
    return (name.startswith(("left_hand_", "right_hand_"))
            or name in {"left_wrist_yaw_link", "right_wrist_yaw_link"})


def initial_body_reference(default):
    """Park the arms behind the table before the first physics step."""
    reference = np.array(default, dtype=float, copy=True)
    for side in ("left", "right"):
        reference[ISAACLAB_JOINT_NAMES.index(f"{side}_shoulder_pitch_joint")] = .4
        reference[ISAACLAB_JOINT_NAMES.index(f"{side}_elbow_joint")] = .8
    return reference


def grasp_center_local(simulation):
    """Return the configured hand-center site in the measured wrist frame."""
    site = simulation.model.site("task_grasp_center").id
    wrist, rotation = wrist_pose(simulation)
    return rotation.T @ (simulation.data.site_xpos[site] - wrist)


def hand_alignment(simulation, z_min=None):
    """Check the block against the measured hand center and block dimensions."""
    block = int(simulation.model.joint("task_block_free").qposadr[0])
    wrist, rotation = wrist_pose(simulation)
    local = rotation.T @ (simulation.data.qpos[block:block + 3] - wrist)
    center = grasp_center_local(simulation)
    geom = simulation.model.geom("task_block_geom").id
    half = np.asarray(simulation.model.geom_size[geom], dtype=float)
    # Lateral margins allow opposing pads to surround the block. Vertically,
    # require the block near the grasp center before stopping descent: the old
    # 1.5x margin closed on the top edge and lost contact during lift.
    margin = half * np.array([1.5, 1.5, 1. / 3.])
    lower = center - margin
    upper = center + margin
    if z_min is not None:
        lower[2] = max(lower[2], float(z_min))
    ready = bool(np.all(local > lower) and np.all(local < upper))
    return {"source": "mujoco_state", "cube_in_wrist_m": local.tolist(),
            "grasp_center_wrist_m": center.tolist(), "gate_lower_wrist_m": lower.tolist(),
            "gate_upper_wrist_m": upper.tolist(), "ready": ready}


def build_scene(repo, path, xy, start_back=DEFAULT_START_BACK, table_standoff=DEFAULT_TABLE_STANDOFF,
                direct_start=False, contact_profile="elliptic"):
    """Extend pinned SONIC assets, without modifying their source checkout."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Cube XY must contain two finite world coordinates")
    # Keep the whole block on the tabletop and away from the initial robot.
    if not (.34 <= xy[0] <= .55 and -.38 <= xy[1] <= -.08):
        raise ValueError("Cube XY outside the right-hand pilot workspace")
    if (not np.isfinite([start_back, table_standoff]).all()
            or not (0 <= start_back <= .6 and .20 <= table_standoff <= .55)):
        raise ValueError("start_back must be 0-.6 m; table_standoff must be .20-.55 m")
    directory = Path(repo).resolve() / "gear_sonic/data/robot_model/model_data/g1"
    robot = directory / "g1_29dof_with_hand.xml"
    root = ET.parse(robot).getroot()
    root.set("model", "b0_tabletop_state_sequencer_pilot_v3")
    root.find("compiler").set("meshdir", str(directory / "meshes"))
    scene = ET.parse(directory / "scene_43dof.xml").getroot()
    for child in scene:
        if child.tag == "include":
            continue
        destination = root.find(child.tag)
        if destination is None:
            root.append(child)
        else:
            destination.attrib.update(child.attrib)
            destination.extend(child)
    if contact_profile not in {"legacy", "elliptic"}:
        raise ValueError("Unknown contact solver profile")
    contact_solver = {"cone": "pyramidal", "solver": "Newton", "impratio": 1.,
                      "tolerance": 1e-8, "noslip_iterations": 0}
    if contact_profile == "elliptic":
        contact_solver.update(cone="elliptic", impratio=10., tolerance=1e-10)
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.attrib.update({key: str(value) for key, value in contact_solver.items()})
    world = root.find("worldbody")
    # This is an initial condition, applied before physics, not a motion command.
    pelvis = root.find(".//body[@name='pelvis']")
    initial_position = np.fromstring(pelvis.get("pos"), sep=" ")
    initial_position[0] = .52 - .21 - table_standoff - start_back
    if direct_start:
        initial_position[1] = xy[1] + .14
    pelvis.set("pos", " ".join(str(float(v)) for v in initial_position))
    ET.SubElement(world, "geom", name="task_table", type="box",
                  pos="0.52 -0.20 0.67", size="0.21 0.24 0.03", rgba="0.45 0.3 0.18 1",
                  friction="1.0 0.005 0.0001")
    for x in (.35, .69):
        for y in (-.40, 0.0):
            ET.SubElement(world, "geom", name=f"task_leg_{x}_{y}", type="box",
                          pos=f"{x} {y} 0.32", size="0.025 0.025 0.32",
                          rgba="0.35 0.25 0.15 1")
    block = ET.SubElement(world, "body", name="task_block", pos=f"{xy[0]} {xy[1]} 0.732")
    ET.SubElement(block, "freejoint", name="task_block_free")
    ET.SubElement(block, "geom", name="task_block_geom", type="box", size="0.03 0.03 0.03",
                  mass="0.08", rgba="0.9 0.25 0.12 1", friction="1.0 0.005 0.0001",
                  condim="4")
    wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    ET.SubElement(wrist, "site", name="task_grasp_center", pos="0.125 0.035 0",
                  size="0.004", rgba="0 1 0 0.5")
    # Optical axes: local -Z looks forward/down; +Y is image up.
    head = root.find(".//body[@name='torso_link']")
    if head is None:
        raise ValueError("Pinned G1 torso body missing")
    camera = head.find("camera[@name='head_camera']")
    if camera is None:
        camera = ET.SubElement(head, "camera")
    camera.attrib.clear()
    camera.attrib.update(name="head_camera", pos="0.06 0 0.35",
                         xyaxes="0 -1 0 0.5 0 0.8660254", fovy="75")
    ET.SubElement(wrist, "camera", name="wrist_camera", pos="0.04 -0.05 0.06",
                  xyaxes="0 -1 0 0 0 1", fovy="90")
    path = Path(path)
    with path.open("x") as handle:
        handle.write(ET.tostring(root, encoding="unicode"))
    return {"schema_version": 2, "table_top_m": .70, "block_side_m": .06,
            "table_position_m": [.52, -.20, .67], "table_half_extents_m": [.21, .24, .03],
            "block_mass_kg": .08, "cube_xy_m": xy.tolist(), "root_free": True,
            "robot_start_xy_m": initial_position[:2].tolist(),
            "start_back_m": start_back, "table_standoff_m": table_standoff,
            "direct_start": bool(direct_start),
            "contact_profile": contact_profile, "contact_solver": contact_solver,
            "grounding_source": "mujoco_state", "camera_input": False,
            "approach_position_tolerance_m": APPROACH_POSITION_TOLERANCE,
            "approach_speed_limit_mps": APPROACH_SPEED_LIMIT,
            "approach_heading_tolerance_rad": float(APPROACH_HEADING_TOLERANCE),
            "settle_hold_seconds": SETTLE_HOLD_SECONDS,
            "prepare_waist_pitch_rad": PREPARE_PITCH_RAD, "articulated_neck": False,
            "block_dynamic": True, "object_attachment": False, "settings_status": "pilot",
            "success_clearance_m": .05, "success_hold_seconds": 2.0, "timeout_seconds": 30.0,
            "finger_closed_rad": RIGHT_CLOSED, "finger_rate_rad_per_second": 2.5,
            "finger_preshaped_rad": RIGHT_PRESHAPED,
            "initial_arm_angles_rad": {"shoulder_pitch": .4, "elbow": .8},
            "phases": [list(p) for p in (DIRECT_START_PHASES if direct_start else PHASES)],
            "hold_phases": sorted(HOLD_PHASES),
            "reference_transition_seconds": {"approach": .2, "generated_arm": .4,
                                             "measured_acquisition_hold": .3},
            "acquisition_region_wrist_m": None,
            "acquisition_gate_source": "measured_hand_center_and_block_geometry",
            "hand_table_contact_allowed": True,
            "prohibited_contacts": ["non-hand robot-table", "non-right-hand robot-block",
                                    "non-foot robot-floor", "block-floor"],
            "cameras": ["head_camera", "wrist_camera"]}


def wrist_pose(simulation):
    body = simulation.model.body("right_wrist_yaw_link").id
    return simulation.data.xpos[body].copy(), simulation.data.xmat[body].reshape(3, 3).copy()


def ground_scene(simulation, table_standoff=DEFAULT_TABLE_STANDOFF):
    """Read current privileged state; no images, learned perception, or future states."""
    model, data = simulation.model, simulation.data
    table = model.geom("task_table").id
    table_position = data.geom_xpos[table].copy()
    table_rotation = data.geom_xmat[table].reshape(3, 3)
    normal, lateral = table_rotation[:, 0], table_rotation[:, 1]
    if abs(normal[2]) > 1e-5 or abs(lateral[2]) > 1e-5:
        raise ValueError("The tabletop approach requires a horizontal table")
    front = table_position - normal * model.geom_size[table, 0]
    block = int(model.joint("task_block_free").qposadr[0])
    cube = data.qpos[block:block + 3].copy()
    root = data.qpos[simulation.root_q:simulation.root_q + 3].copy()
    quat = data.qpos[simulation.root_q + 3:simulation.root_q + 7]
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    # Put the block on the robot's right side, aligned with the right shoulder.
    target = front - table_standoff * normal + (np.dot(cube - table_position, lateral) + .14) * lateral
    target_heading = float(np.arctan2(normal[1], normal[0]))
    actual_heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    heading_error = float(np.arctan2(np.sin(target_heading - actual_heading),
                                   np.cos(target_heading - actual_heading)))
    return {"schema_version": 1, "source": "mujoco_state", "camera_input": False,
            "coordinate_frame": "mujoco_world", "units": "SI", "sim_time": float(data.time),
            "table_position": table_position.tolist(), "table_front": front.tolist(),
            "table_top_m": float(table_position[2] + model.geom_size[table, 2]),
            "cube_position": cube.tolist(), "cube_in_base": (rotation.T @ (cube - root)).tolist(),
            "root_position": root.tolist(), "approach_target_xy": target[:2].tolist(),
            "approach_heading_rad": target_heading, "heading_error_rad": heading_error,
            "position_error_m": float(np.linalg.norm(target[:2] - root[:2])),
            "root_to_table_front_m": float(np.dot(front - root, normal))}


def approach_state(simulation, table_standoff=DEFAULT_TABLE_STANDOFF):
    state = ground_scene(simulation, table_standoff)
    speed = float(np.linalg.norm(simulation.data.qvel[simulation.root_v:simulation.root_v + 2]))
    feet = set()
    for contact in simulation.data.contact[:simulation.data.ncon]:
        geoms = (int(contact.geom1), int(contact.geom2))
        for floor, foot in (geoms, geoms[::-1]):
            name = simulation.model.body(int(simulation.model.geom_bodyid[foot])).name or ""
            if simulation.model.geom(floor).name == "floor" and "ankle_roll" in name and contact.dist <= .001:
                feet.add("left" if name.startswith("left") else "right")
    state.update(root_speed_mps=speed, feet_on_floor=sorted(feet), ready=(
        state["position_error_m"] <= APPROACH_POSITION_TOLERANCE and
        abs(state["heading_error_rad"]) <= APPROACH_HEADING_TOLERANCE and
        speed <= APPROACH_SPEED_LIMIT and feet == {"left", "right"}))
    return state


def approach_constraints(simulation, duration, table_standoff=DEFAULT_TABLE_STANDOFF):
    """A world-space root path; feet remain free to produce a walking gait."""
    state = ground_scene(simulation, table_standoff)
    frames = int(duration * 25)
    indices = np.unique(np.r_[np.arange(0, frames, 4), frames - 1]).astype(int)
    start = np.asarray(state["root_position"][:2])
    target = np.asarray(state["approach_target_xy"])
    u = np.clip(indices / (.75 * (frames - 1)), 0, 1)
    u = u * u * (3 - 2 * u)
    positions = start + u[:, None] * (target - start)
    return {"schema_version": 1, "kind": "root_path", "coordinate_frame": "mujoco_world", "units": "SI",
            "frame_indices": indices.tolist(), "root_positions_xy": positions.tolist(),
            "root_heading_rad": [state["approach_heading_rad"]] * len(indices), "grounding": state}


def settle_constraints(simulation, duration):
    """Stop at the measured position without pinning a potentially lifted foot."""
    pose = simulation.body_pose()
    frames = int(duration * 25)
    return {"schema_version": 1, "kind": "root_path", "coordinate_frame": "mujoco_world", "units": "SI",
            "frame_indices": [0, frames - 1], "root_positions_xy": [pose[:2].tolist()] * 2,
            "root_heading_rad": [float(Rotation.from_quat(pose[[4, 5, 6, 3]]).as_euler("xyz")[2])] * 2}


def phase_constraints(simulation, phase, duration, standing_qpos, standing_wrist_rotation,
                      wrist_offset=None):
    """Ground one phase in current GT state; interpolate sparse world wrist goals."""
    block = simulation.model.joint("task_block_free").qposadr[0]
    center = simulation.data.qpos[block:block + 3].copy()
    # Use the measured hand-center site in the wrist frame.  An explicit offset
    # remains available for controlled ablations and is never tied to cube XY.
    local_offset = grasp_center_local(simulation) if wrist_offset is None else np.asarray(wrist_offset)
    start, start_rotation = wrist_pose(simulation)
    goal = center - start_rotation @ local_offset
    frames = int(duration * 25)
    indices = np.unique(np.r_[np.arange(0, frames, 4), frames - 1]).astype(int)
    if phase == "lift":
        goal = start + np.array([0., 0., .14])
    u = np.clip(indices / max(1, .75 * (frames - 1)), 0, 1)
    u = u * u * (3 - 2 * u)
    positions = start + u[:, None] * (goal - start)
    if phase == "reach":
        # First raise the hanging hand without crossing the table edge. Only
        # then move over the tabletop; one straight segment clips the fingers.
        table = simulation.model.geom("task_table").id
        top = simulation.data.geom_xpos[table, 2] + simulation.model.geom_size[table, 2]
        high = max(start[2], float(top) + .24)
        t = indices / max(1, frames - 1)
        up = np.clip(t / .45, 0, 1)
        up = up * up * (3 - 2 * up)
        across = np.clip((t - .45) / .35, 0, 1)
        across = across * across * (3 - 2 * across)
        positions[:, :2] = start[:2] + across[:, None] * (goal[:2] - start[:2])
        positions[:, 2] = start[2] + up * (high - start[2])
    end_rotation = start_rotation if phase == "lift" else np.eye(3)
    rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([start_rotation, end_rotation])))(u).as_matrix()
    yaw = float(standing_qpos[7 + ISAACLAB_JOINT_NAMES.index("waist_yaw_joint")])
    roll = float(standing_qpos[7 + ISAACLAB_JOINT_NAMES.index("waist_roll_joint")])
    pitch = float(standing_qpos[7 + ISAACLAB_JOINT_NAMES.index("waist_pitch_joint")])
    torso_yaw = np.full(len(indices), yaw)
    torso_roll = np.full(len(indices), roll)
    torso_pitch = np.full(len(indices), pitch)
    if phase == "prepare":
        # The pinned G1 has a rigid head. A small waist inclination is a nominal
        # pose condition, not a camera observation or an injected body command.
        positions = np.repeat(start[None], len(indices), axis=0)
        rotations = np.repeat(start_rotation[None], len(indices), axis=0)
        torso_pitch = pitch + u * (PREPARE_PITCH_RAD - pitch)
    return {"schema_version": 1, "coordinate_frame": "mujoco_world", "units": "SI",
            "frame_indices": indices.tolist(), "wrist_positions": positions.tolist(),
            "wrist_rotations": rotations.tolist(),
            "torso_yaw_rad": torso_yaw.tolist(), "torso_roll_rad": torso_roll.tolist(),
            "torso_pitch_rad": torso_pitch.tolist(),
            "standing_qpos": np.asarray(standing_qpos).tolist(),
            "standing_wrist_rotation": np.asarray(standing_wrist_rotation).tolist()}


def phase_prompt(instruction, phase, profile="focused"):
    if profile == "focused":
        return (instruction.strip() + " " if instruction else "") + FOCUSED_PROMPTS[phase]
    if profile != "legacy":
        raise ValueError("Unknown grasp prompt profile")
    if phase in {"approach", "settle", "prepare"}:
        return (instruction.strip() + " " if instruction else "") + FOCUSED_PROMPTS[phase]
    actions = {"reach": "Reach the right hand above the block while standing still.",
               "lower": "Lower and align the right hand beside the block while standing still.",
               "close": "Keep the right wrist still while grasping the block.",
               "lift": "Raise the right hand slowly to lift the block while standing still.",
               "hold": "Hold the raised right hand still while standing upright."}
    return (instruction or DEFAULT_PROMPT) + " Current step: " + actions[phase]


class GraspEvaluator:
    """Assess every physics step, including rotated block extent and finger contacts."""

    def __init__(self, simulation, output):
        self.simulation = simulation
        self.model = simulation.model
        self.block_q = int(self.model.joint("task_block_free").qposadr[0])
        self.block_geom = self.model.geom("task_block_geom").id
        self.phase = "stand"
        self.failure = None
        self.hold_start = None
        self.max_hold = 0.0
        self.max_clearance = -1.0
        self.success_time = None
        self.retained_at_end = False
        self.post_success_loss_samples = 0
        self.lost_after_success = False
        self.contact_steps = 0
        self.hand_table_contact_steps = 0
        self.samples = 0
        self.first_failure = None
        self.opposing_contact_start = None
        self.opposing_contact_seconds = 0.
        self.file = (Path(output) / "task.csv").open("x", buffering=1)
        self.writer = csv.writer(self.file)
        self.finger_names = simulation.finger_names
        self.writer.writerow(["sim_time", "control_frame", "phase", "block_x", "block_y", "block_z",
                              "lowest_clearance_m", "thumb_force_n", "finger_force_n", "held_seconds",
                              "wrist_x", "wrist_y", "wrist_z"] +
                             [f"finger_q:{n}" for n in self.finger_names] +
                             [f"finger_ref:{n}" for n in self.finger_names])

    def observe(self, simulation):
        data, model = simulation.data, self.model
        pos = data.qpos[self.block_q:self.block_q + 3]
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, data.qpos[self.block_q + 3:self.block_q + 7])
        clearance = float(pos[2] - np.abs(mat.reshape(3, 3)[2]) @ model.geom_size[self.block_geom] - .70)
        thumb, finger = 0.0, 0.0
        hand_table_contact = False
        for contact_index, contact in enumerate(data.contact[:data.ncon]):
            if contact.dist > 0:
                continue
            geoms = (int(contact.geom1), int(contact.geom2))
            names = [model.geom(g).name or "" for g in geoms]
            bodies = [model.body(int(model.geom_bodyid[g])).name or "" for g in geoms]
            previous_failure = self.failure
            if any(n.startswith("task_table") or n.startswith("task_leg") for n in names):
                for geom, body in zip(geoms, bodies):
                    if model.geom_bodyid[geom] != 0 and geom != self.block_geom:
                        if is_hand_body(body):
                            hand_table_contact = True
                        else:
                            self.failure = self.failure or "prohibited_robot_table_contact"
            if "floor" in names:
                other = geoms[1 - names.index("floor")]
                body = model.body(int(model.geom_bodyid[other])).name or ""
                if other == self.block_geom:
                    self.failure = self.failure or "block_floor_contact"
                elif model.geom_bodyid[other] != 0 and "ankle_roll" not in body:
                    self.failure = self.failure or "prohibited_robot_floor_contact"
            if self.block_geom in geoms:
                other = geoms[1 - geoms.index(self.block_geom)]
                body = bodies[1 - geoms.index(self.block_geom)]
                allowed = body.startswith("right_hand_") or body == "right_wrist_yaw_link"
                if model.geom_bodyid[other] != 0 and not allowed:
                    self.failure = self.failure or "prohibited_robot_block_contact"
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                if body.startswith("right_hand_thumb_"):
                    thumb += max(0.0, float(force[0]))
                if body.startswith(("right_hand_index_", "right_hand_middle_")):
                    finger += max(0.0, float(force[0]))
            if previous_failure is None and self.failure is not None:
                self.first_failure = {"sim_time": float(data.time), "geoms": names,
                                      "bodies": bodies, "penetration_m": float(contact.dist)}
        root = data.qpos[simulation.root_q:simulation.root_q + 7]
        upright = 1 - 2 * (root[4] ** 2 + root[5] ** 2)
        if root[2] < .35 or upright < .4:
            self.failure = self.failure or "fall"
        grasp_contact = thumb > .01 and finger > .01
        if grasp_contact:
            if self.opposing_contact_start is None:
                self.opposing_contact_start = float(data.time)
            self.opposing_contact_seconds = float(data.time) - self.opposing_contact_start
        else:
            self.opposing_contact_start = None
            self.opposing_contact_seconds = 0.
        self.contact_steps += int(grasp_contact)
        self.hand_table_contact_steps += int(hand_table_contact)
        held = 0.0
        if (self.phase in {"lift", "hold"} and clearance >= .05 and grasp_contact
                and self.failure is None and data.time <= 30.0 + 1e-8):
            if self.hold_start is None:
                self.hold_start = float(data.time)
            held = float(data.time) - self.hold_start
            self.max_hold = max(self.max_hold, held)
            if held >= 2.0 - 1e-8 and self.success_time is None:
                self.success_time = float(data.time)
        else:
            self.hold_start = None
        self.retained_at_end = bool(self.hold_start is not None)
        if self.success_time is not None and not self.retained_at_end:
            self.post_success_loss_samples += 1
            # A threshold event is evidence, not a reversible success state.
            # Keep a late loss sticky even if the fingers happen to re-contact
            # the block before the episode ends.
            self.lost_after_success = True
        self.max_clearance = max(self.max_clearance, clearance)
        self.samples += 1
        # mj_step leaves derived poses at the beginning of its integration step.
        mujoco.mj_kinematics(model, data)
        wrist, _ = wrist_pose(simulation)
        self.writer.writerow([data.time, simulation.frame_index, self.phase, *pos,
                              clearance, thumb, finger, held, *wrist,
                              *data.qpos[simulation.finger_q], *simulation.finger_target])

    def summary(self):
        retained = self.retained_at_end and not self.lost_after_success
        success = self.success_time is not None and self.failure is None and retained
        reason = self.failure or ("grasp_lost_after_success" if self.lost_after_success
                                  else None if success else "timeout")
        return {"task_success": success, "success_sim_time": self.success_time,
                "failure_reason": reason, "success_threshold_reached": self.success_time is not None,
                "retained_at_end": retained,
                "lost_after_success": self.lost_after_success,
                "post_success_loss_samples": self.post_success_loss_samples,
                "max_block_clearance_m": self.max_clearance, "max_continuous_hold_seconds": self.max_hold,
                "opposing_finger_contact_steps": self.contact_steps, "physics_samples": self.samples,
                "hand_table_contact_steps": self.hand_table_contact_steps,
                "assessment_hz": 200, "user_corrections": 0, "teacher_used": False,
                "first_prohibited_contact": self.first_failure,
                "learned_correction_used": bool(
                    getattr(self.simulation, "learned_correction_used", False)
                ), "expert_valid": False}

    def close(self):
        self.file.close()
