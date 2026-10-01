"""Default frozen SONIC G1-mode ONNX adapter, matching pinned C++ deployment.

Reference angles are absolute; executed angles are relative to default pose.
History is oldest first, frame major within each feature group. Quaternions
are wxyz and orientation features flatten the first two matrix columns by row.
"""

import ast
from collections import deque
from dataclasses import dataclass
import re

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.common import LOCK, checked_checkout, verify_assets
import json


@dataclass
class PolicyParameters:
    names: tuple
    default: np.ndarray
    scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


def read_parameters(repo):
    """Read the pinned upstream constants; evaluate arithmetic only, never code."""
    path = repo / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    original = path.read_text()
    source = re.sub(r"//[^\n]*", "", original)
    constants = {}

    def arithmetic(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -arithmetic(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = arithmetic(node.left), arithmetic(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError("Unsupported upstream parameter expression")

    def value(expr):
        return arithmetic(ast.parse(expr.strip(), mode="eval").body)

    for name, expr in re.findall(r"const (?:double|float) (\w+)\s*=\s*([^;]+);", source):
        constants[name] = value(expr)

    def array(name):
        match = re.search(r"\b" + name + r"\s*=\s*\{(.*?)\};", source, re.S)
        if match is None: raise ValueError(f"Missing upstream parameter {name}")
        return np.array([value(s) for s in match[1].split(",") if s.strip()])

    block = re.search(r"\bdefault_angles\s*=\s*\{(.*?)\};", original, re.S)[1]
    names = tuple(re.findall(r"//\s*(\w+_joint)", block))
    if len(names) != 29 or set(names) != set(ISAACLAB_JOINT_NAMES):
        raise ValueError("Upstream default-angle names do not match G1")
    order = [names.index(name) for name in ISAACLAB_JOINT_NAMES]
    if not np.array_equal(order, array("mujoco_to_isaaclab")):
        raise ValueError("Named joint mapping differs from upstream mapping")
    return PolicyParameters(ISAACLAB_JOINT_NAMES, array("default_angles")[order],
                            array("g1_action_scale")[order], array("kps")[order],
                            array("kds")[order])


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def heading(wxyz):
    forward = rotation(wxyz).apply([1, 0, 0])
    return Rotation.from_euler("z", np.arctan2(forward[..., 1], forward[..., 0]))


def dimensions(name):
    fixed = {"token_state": 64, "encoder_mode_4": 4,
             "motion_root_z_position": 1, "motion_anchor_orientation": 6,
             "vr_3point_local_target": 9, "vr_3point_local_orn_target": 12}
    if name in fixed: return fixed[name]
    match = re.fullmatch(r"(.+)_(\d+)frame_step(\d+)", name)
    if not match: raise ValueError(f"Unsupported SONIC observation: {name}")
    width = {"motion_joint_positions": 29, "motion_joint_velocities": 29,
             "motion_joint_positions_lowerbody": 12, "motion_joint_velocities_lowerbody": 12,
             "motion_joint_positions_wrists": 6, "motion_root_z_position": 1,
             "motion_anchor_orientation": 6, "smpl_joints": 72,
             "smpl_anchor_orientation": 6, "his_base_angular_velocity": 3,
             "his_body_joint_positions": 29, "his_body_joint_velocities": 29,
             "his_last_actions": 29, "his_gravity_dir": 3}[match[1]]
    return width * int(match[2])


class SonicPolicy:
    def __init__(self, assets, repo, threads=2):
        import onnxruntime as ort

        self.source_commit = checked_checkout(repo, json.loads(LOCK.read_text())["sonic"]["commit"])
        self.asset_hash = verify_assets(assets, "sonic")
        self.parameters = read_parameters(repo)
        config = yaml.safe_load((assets / "sonic/observation_config.yaml").read_text())
        self.encoder_names = [o["name"] for o in config["encoder"]["encoder_observations"] if o["enabled"]]
        self.decoder_names = [o["name"] for o in config["observations"] if o["enabled"]]
        mode = next(m for m in config["encoder"]["encoder_modes"] if m["name"] == "g1")
        self.required = set(mode["required_observations"])
        if mode["mode_id"] != 0 or self.required != {
            "encoder_mode_4", "motion_joint_positions_10frame_step5",
            "motion_joint_velocities_10frame_step5", "motion_anchor_orientation_10frame_step5"}:
            raise ValueError("Only the pinned default G1 encoder mode is supported")
        self.future_count, self.future_step = 10, 5
        self.control_dt = .02
        self.lookahead_seconds = (self.future_count - 1) * self.future_step * self.control_dt
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(assets / "sonic/model_encoder.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(assets / "sonic/model_decoder.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        for graph, names, output in ((self.encoder, self.encoder_names, 64),
                                     (self.decoder, self.decoder_names, 29)):
            width = sum(dimensions(n) for n in names)
            if graph.get_inputs()[0].shape[-1] != width or graph.get_outputs()[0].shape[-1] != output:
                raise ValueError("SONIC graph/config dimensions differ")
        self.reset()

    def reset(self):
        self.history = deque(maxlen=10)
        self.last_action = np.zeros(29)
        self.alignment = Rotation.identity()

    def align(self, base_quat, reference_quat):
        self.alignment = heading(base_quat) * heading(reference_quat).inv()

    def observe(self, q, dq, quat, gyro):
        gravity = rotation(quat).inv().apply([0, 0, -1])
        self.history.append({"his_body_joint_positions": q - self.parameters.default,
                             "his_body_joint_velocities": np.array(dq),
                             "his_last_actions": self.last_action.copy(),
                             "his_base_angular_velocity": np.array(gyro),
                             "his_gravity_dir": gravity})

    def encoder_observation(self, pos, vel, quat, base_quat):
        if np.shape(pos) != (10, 29) or np.shape(vel) != (10, 29) or np.shape(quat) != (10, 4):
            raise ValueError("Expected ten G1 lookahead samples")
        relative = rotation(base_quat).inv() * self.alignment * rotation(quat)
        # Upstream GatherEncoderMode writes the scalar mode ID then zero padding,
        # not a one-hot vector. G1 mode is therefore four zeros.
        values = {"encoder_mode_4": [0, 0, 0, 0],
                  "motion_joint_positions_10frame_step5": pos,
                  "motion_joint_velocities_10frame_step5": vel,
                  "motion_anchor_orientation_10frame_step5": relative.as_matrix()[..., :2]}
        return np.concatenate([np.asarray(values[n]).reshape(-1) if n in self.required
                               else np.zeros(dimensions(n)) for n in self.encoder_names]).astype(np.float32)[None]

    def decoder_observation(self, tokens):
        groups = []
        for name in self.decoder_names:
            if name == "token_state":
                groups.append(np.asarray(tokens).reshape(-1))
                continue
            key = name.removesuffix("_10frame_step1")
            width = dimensions(name) // 10
            missing = np.zeros((10 - len(self.history), width))
            if key == "his_gravity_dir": missing[:, 2] = 1  # upstream rotates with zero-entry quaternion
            groups.append(np.concatenate([missing.reshape(-1)] + [h[key] for h in self.history]))
        return np.concatenate(groups).astype(np.float32)[None]

    def act(self, pos, vel, quat, base_quat):
        encoder_obs = self.encoder_observation(pos, vel, quat, base_quat)
        tokens = self.encoder.run(None, {self.encoder.get_inputs()[0].name: encoder_obs})[0]
        decoder_obs = self.decoder_observation(tokens)
        action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: decoder_obs})[0][0]
        if not np.isfinite(action).all(): raise FloatingPointError("Nonfinite SONIC action")
        self.last_action = action.copy()
        return self.parameters.default + self.parameters.scale * action, action
