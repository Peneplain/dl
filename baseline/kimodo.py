"""Local Kimodo text-to-reference service for the shared SONIC executor.

This module intentionally owns only the Kimodo generator boundary. The ARDY
service, SONIC policy, reference checks, timestamped buffer, and MuJoCo
execution remain shared or untouched. Kimodo does not provide ARDY-style
history conditioning; an executed history is retained as an artifact for
matched logging, while ``SonicSimulation.install`` performs the shared
measured-pose transition.
"""

from __future__ import annotations

import gc
import json
import os
import random
from functools import partial
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.adapters.sonic import JointStreamEncoder
from baseline.common import ROOT, checked_checkout, sha256, write_json
from baseline.runtime import device_for, synchronize

KIMODO_LOCK = ROOT / "configs/kimodo.lock.json"


class _JointRotationConstraint:
    """Small Kimodo-native constraint for selected global joint rotations.

    Kimodo's public EndEffectorConstraintSet intentionally exposes only hands
    and feet. ARDY's shared schema also carries measured torso joint targets;
    this adapter contributes only those rotation channels without pinning all
    body positions.
    """

    def __init__(self, skeleton, frame_indices, global_joints_rots, joint_names, torch):
        self.torch = torch
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.joint_names = tuple(joint_names)
        self.joint_indices = torch.as_tensor(
            [skeleton.bone_index[name] for name in self.joint_names],
            dtype=torch.long, device=frame_indices.device)
        self.global_joints_rots = global_joints_rots

    def _indices(self):
        n = len(self.frame_indices)
        j = len(self.joint_indices)
        return self.torch.stack(
            [self.frame_indices[:, None].expand(n, j),
             self.joint_indices[None, :].expand(n, j)], dim=-1).reshape(-1, 2)

    def update_constraints(self, data_dict, index_dict):
        data_dict["global_joints_rots"].append(self.global_joints_rots.reshape(-1, 3, 3))
        index_dict["global_joints_rots"].append(self._indices())

    def crop_move(self, start, end):
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return _JointRotationConstraint(
            self.skeleton, self.frame_indices[mask] - start,
            self.global_joints_rots[mask], self.joint_names, self.torch)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.frame_indices = self.frame_indices.to(device=device)
            self.joint_indices = self.joint_indices.to(device=device)
            self.global_joints_rots = self.global_joints_rots.to(device=device)
            if hasattr(self.skeleton, "to"):
                self.skeleton = self.skeleton.to(device)
        if dtype is not None:
            self.global_joints_rots = self.global_joints_rots.to(dtype=dtype)
        return self


def _lock() -> dict[str, Any]:
    return json.loads(KIMODO_LOCK.read_text())


def _checkpoint_directory(root: Path, model_name: str) -> Path:
    """Resolve either a checkpoint root or a direct model directory."""
    root = Path(root).expanduser().resolve()
    candidates = [root, root / model_name]
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate
    expected = root / model_name / "config.yaml"
    raise FileNotFoundError(
        f"Missing Kimodo checkpoint at {expected}. Fetch on a connected machine with "
        "python scripts/fetch_kimodo.py --only checkpoint, then rsync "
        "checkpoints/kimodo/ to this machine."
    )


def _verify_checkpoint(root: Path, model_name: str) -> tuple[Path, str | None]:
    """Check the local model shape and, when present, the generated manifest."""
    root = Path(root).expanduser().resolve()
    model_dir = _checkpoint_directory(root, model_name)
    required = [
        model_dir / "config.yaml", model_dir / "model.safetensors",
        model_dir / "stats/motion/body/mean.npy",
        model_dir / "stats/motion/body/std.npy",
        model_dir / "stats/motion/global_root/mean.npy",
        model_dir / "stats/motion/global_root/std.npy",
        model_dir / "stats/motion/local_root/mean.npy",
        model_dir / "stats/motion/local_root/std.npy",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty Kimodo checkpoint file: {path}")
    manifest_root = root.parent if root.name == model_name else root
    manifest_path = manifest_root / "kimodo.manifest.json"
    if not manifest_path.is_file():
        # Keep the inference boundary usable for a manually copied, complete
        # checkpoint, but report the absence so provenance is never implied.
        return model_dir, None
    manifest = json.loads(manifest_path.read_text())
    expected_source = _lock()["checkpoint"]
    if manifest.get("source") != expected_source or not manifest.get("files"):
        raise ValueError(f"Kimodo asset manifest does not match the lock: {manifest_path}")
    actual_files = {str(path.relative_to(model_dir)) for path in model_dir.rglob("*")
                    if path.is_file() and ".cache" not in path.relative_to(model_dir).parts}
    if actual_files != set(manifest["files"]):
        raise ValueError(f"Kimodo files differ from the completed manifest: {manifest_path}")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid Kimodo asset path in manifest: {name}")
        path = model_dir / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Kimodo asset changed or incomplete: {path}")
    return model_dir, sha256(manifest_path)


def _kimodo_joint_names(xml_path: Path, converter: Any) -> tuple[str, ...]:
    """Read the converter's actual MuJoCo hinge order, never assume qpos order."""
    root = ET.parse(xml_path).getroot().find("worldbody")
    names = [] if root is None else [
        joint.attrib["name"] for joint in root.findall(".//joint")
        if joint.attrib.get("name") and joint.attrib.get("type", "hinge") != "free"
    ]
    if len(names) != 29:
        # Kimodo keeps the parsed order on the converter. This fallback handles
        # XMLs that use a freejoint element and therefore omit the base from the
        # regular joint list.
        parsed = getattr(converter, "_mujoco_joint_including_root_list", None)
        if parsed is not None:
            names = [str(name).replace("_skel", "_joint") for name in list(parsed)[1:]]
    names = tuple(names)
    if len(names) != 29 or set(names) != set(ISAACLAB_JOINT_NAMES):
        raise ValueError(
            "Kimodo converter XML must expose exactly the 29 canonical G1 joints; "
            f"got {len(names)} names: {names}"
        )
    return names


def _resample_reference(reference: ReferenceSequence, target_fps: float = 50.0) -> ReferenceSequence:
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive")
    end = float(reference.times[-1])
    count = max(2, int(np.floor(end * target_fps + 1e-9)) + 1)
    # Keep an exact 50 Hz grid and stop before the 30 Hz clip endpoint when
    # that endpoint does not itself land on a 50 Hz frame.
    times = np.arange(count, dtype=np.float64) / target_fps
    return reference.sample(times)


class KimodoService:
    """Load one frozen local Kimodo model and export shared 50 Hz references."""

    def __init__(self, args):
        import torch

        self.args = args
        self.torch = torch
        _JointRotationConstraint.torch = torch
        self.device = device_for(args.device)
        self.text_device = device_for(args.text_device)
        self.lock = _lock()
        self.model_name = str(self.lock["model"])
        self.kimodo_repo = Path(args.kimodo_repo).expanduser().resolve()
        try:
            self.upstream_commit = checked_checkout(
                self.kimodo_repo, self.lock["source"]["commit"])
        except Exception as error:
            raise RuntimeError(
                f"Kimodo source is unavailable or not pinned at {self.kimodo_repo}. "
                "Run python scripts/fetch_kimodo.py --only source on a connected "
                "machine, or rsync third_party/kimodo/ here. "
                f"({type(error).__name__}: {error})"
            ) from error
        if not (self.kimodo_repo / "kimodo" / "__init__.py").is_file():
            raise FileNotFoundError(f"Incomplete Kimodo source checkout: {self.kimodo_repo}")
        self.model_dir, self.manifest_sha256 = _verify_checkpoint(
            args.kimodo_assets, self.model_name)

        # Kimodo's loader otherwise attempts Hugging Face resolution. The
        # checkpoint and the shared LLM2Vec assets are required to be local.
        os.environ["CHECKPOINT_DIR"] = str(self.model_dir.parent)
        os.environ["LOCAL_CACHE"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TEXT_ENCODER_DEVICE"] = args.text_device
        sys.path.insert(0, str(self.kimodo_repo))

        from baseline.text_encoder import LocalTextEncoder, check_transformers_version

        check_transformers_version()
        start = time.perf_counter()
        print("Loading the shared local LLM2Vec encoder for Kimodo...", file=sys.stderr, flush=True)
        # The encoder has the same (features, lengths) contract as Kimodo's
        # native LLM2VecEncoder, so existing ARDY text assets are reused once.
        sys.path.insert(0, str(Path(args.ardy_repo).expanduser().resolve()))
        self.encoder = LocalTextEncoder(args.assets, dtype=args.text_dtype,
                                        device=str(self.text_device))
        synchronize(self.text_device)
        self.text_load_seconds = time.perf_counter() - start

        from kimodo import load_model
        from kimodo.exports.mujoco import MujocoQposConverter

        print(f"Loading frozen {self.model_name} on {self.device}...", file=sys.stderr, flush=True)
        self.model = load_model(self.model_name, device=str(self.device),
                                text_encoder=self.encoder, eval_mode=True)
        self.model.eval()
        self.fps = float(self.model.fps)
        if not np.isfinite(self.fps) or self.fps != float(self.lock["source_fps"]):
            raise ValueError(
                f"Expected Kimodo source FPS {self.lock['source_fps']}, got {self.fps}"
            )
        self.converter = MujocoQposConverter(self.model.skeleton)
        # The pinned converter's qpos->motion helper runs FK through its
        # skeleton. Keep a CPU-only skeleton/converter for measured-state
        # constraint construction; the model-owned skeleton may live on MUSA.
        self.cpu_skeleton = type(self.model.skeleton)(
            folder=str(getattr(self.model.skeleton, "folder", "")) or None)
        self.cpu_converter = MujocoQposConverter(
            self.cpu_skeleton, xml_path=str(self.converter.xml_path))
        self.joint_names = _kimodo_joint_names(Path(self.converter.xml_path), self.converter)
        self.max_frames = int(self.lock["max_frames"])
        self.projector = None
        if not args.kimodo_no_projection:
            from baseline.kimodo_projection import KimodoConstraintProjector
            self.projector = KimodoConstraintProjector(self.converter.xml_path, self.joint_names)
        self.steps = int(args.kimodo_diffusion_steps)
        self.constraint_guidance = float(args.kimodo_constraint_guidance)
        if not np.isfinite(self.constraint_guidance) or self.constraint_guidance <= 0:
            raise ValueError("kimodo constraint guidance must be finite and positive")
        if self.steps < 1:
            raise ValueError("kimodo diffusion steps must be positive")
        synchronize(self.device)
        self.motion_load_seconds = time.perf_counter() - start - self.text_load_seconds
        gc.collect()

    @staticmethod
    def _seed(torch, seed: int, device) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "musa":
            torch.musa.manual_seed_all(seed)

    @staticmethod
    def _numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _kimodo_matrix(self):
        # The pinned converter owns the exact z-up/x-forward -> y-up/z-forward
        # transform. Never duplicate an axis convention in this integration.
        return self._numpy(self.converter.mujoco_to_kimodo_matrix).astype(np.float32)

    def _frame_indices(self, constraints, frames):
        indices = np.asarray(constraints["frame_indices"], dtype=np.int64)
        if (indices.ndim != 1 or len(indices) == 0 or indices[0] < 0
                or (np.diff(indices) <= 0).any()):
            raise ValueError("Invalid shared pose constraint frame indices")
        # Shared ARDY constraints are authored at its 25 Hz motion clock. Kimodo
        # is generated at the pinned 30 Hz source rate; map through seconds.
        mapped = np.rint(indices.astype(np.float64) / 25.0 * self.fps).astype(np.int64)
        if mapped[0] < 0 or mapped[-1] >= frames or (np.diff(mapped) <= 0).any():
            raise ValueError("Shared pose constraints cannot be represented on the Kimodo frame grid")
        return mapped

    def _cpu_motion_from_qpos(self, qpos):
        """Decode measured qpos on CPU with enough frames for Kimodo smoothing."""
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 1 or not np.isfinite(qpos).all():
            raise ValueError("qpos must be finite [N,36]")
        if len(qpos) < 8:
            qpos = np.concatenate([qpos, np.repeat(qpos[-1:], 8 - len(qpos), axis=0)], axis=0)
        # Shared measured history uses IsaacLab order; the pinned converter
        # expects its XML traversal order. The inverse boundary needs the same
        # explicit-name permutation as the generated-reference boundary.
        ordered = qpos.copy()
        ordered[:, 7:] = qpos[:, 7:][:, [ISAACLAB_JOINT_NAMES.index(name)
                                        for name in self.joint_names]]
        return self.cpu_converter.qpos_to_motion_dict(ordered, source_fps=self.fps)

    def _standing_motion(self, standing_qpos):
        qpos = np.asarray(standing_qpos, dtype=np.float32)
        if qpos.shape != (36,) or not np.isfinite(qpos).all():
            raise ValueError("standing_qpos must be finite [36]")
        motion = self._cpu_motion_from_qpos(qpos[None, :])
        return {key: self._numpy(value) for key, value in motion.items()}

    def _shared_constraints(self, constraints, frames, standing_qpos):
        """Translate the common MuJoCo pose schema to native Kimodo objects.

        The execution layer still creates one simulator-grounded constraint
        dictionary for both generators. Only this boundary knows Kimodo's
        Y-up/XZ representation and its end-effector constraint objects.
        """
        if constraints is None:
            return [], {}
        if isinstance(constraints, (str, Path)):
            from kimodo.constraints import load_constraints_lst
            loaded = load_constraints_lst(str(constraints), self.model.skeleton,
                                          device=self.device)
            return loaded, {"source": str(constraints)}
        if not isinstance(constraints, dict):
            raise TypeError("Kimodo pose constraints must be a path or shared constraint dictionary")
        if (constraints.get("schema_version") != 1
                or constraints.get("coordinate_frame") != "mujoco_world"
                or constraints.get("units") != "SI"):
            raise ValueError("Unsupported shared pose constraint schema/frame/units")
        from kimodo.constraints import EndEffectorConstraintSet, Root2DConstraintSet

        # Kimodo's EndEffectorConstraintSet constructor creates its constant
        # joint-index tensors on CPU.  The model later calls crop_move() for
        # each generation segment; that creates another instance and combines
        # those CPU indices with MUSA frame indices in create_pairs().  Keep
        # the pinned upstream checkout untouched, but make the adapter's
        # constraint class device-stable, including all cropped instances.
        class _DeviceEndEffectorConstraintSet(EndEffectorConstraintSet):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.to(device=self.frame_indices.device)

        torch = self.torch
        matrix = self._kimodo_matrix()
        frame_indices = self._frame_indices(constraints, frames)
        info = {"source": "shared_mujoco_constraints", "frame_indices": frame_indices.tolist()}
        # The shared rotation delta is relative to this standing anchor, not
        # the latest executed history. Mixing the two changes the requested
        # wrist orientation and the stationary foot/left-hand conditions.
        standing = np.asarray(constraints.get("standing_qpos", standing_qpos), dtype=np.float32)
        if standing.shape != (36,) or not np.isfinite(standing).all():
            raise ValueError("standing_qpos must be finite [36]")
        if constraints.get("kind") == "root_path":
            xy = np.asarray(constraints["root_positions_xy"], dtype=np.float32)
            heading = np.asarray(constraints["root_heading_rad"], dtype=np.float32)
            if (xy.shape != (len(frame_indices), 2) or heading.shape != (len(frame_indices),)
                    or not np.isfinite(xy).all() or not np.isfinite(heading).all()):
                raise ValueError("Invalid shared root-path constraint dimensions")
            root0 = standing[:3]
            world_delta = np.c_[xy - root0[:2], np.zeros(len(xy), dtype=np.float32)]
            kimodo_delta = world_delta @ matrix.T
            smooth_root_2d = kimodo_delta[:, [0, 2]]
            # Kimodo's heading pair is [cos(theta), sin(theta)] in its XZ
            # plane; the shared yaw is around MuJoCo +Z and maps to the same
            # signed angle after the converter's axis transform.
            global_heading = np.c_[np.cos(heading), np.sin(heading)].astype(np.float32)
            root_constraint = Root2DConstraintSet(
                self.model.skeleton,
                torch.as_tensor(frame_indices, dtype=torch.long, device=self.device),
                torch.as_tensor(smooth_root_2d, dtype=torch.float32, device=self.device),
                global_root_heading=torch.as_tensor(global_heading, dtype=torch.float32,
                                                    device=self.device),
            )
            if hasattr(root_constraint, "to"):
                root_constraint.to(device=self.device)
            result = [root_constraint]
            info.update(kind="root2d", constrained_fields=["root_positions_xy", "root_heading_rad"])
            return result, info

        kind = constraints.get("kind", "wrist")
        if kind not in {"wrist", "wrist_pose"}:
            raise ValueError(f"Unsupported shared pose constraint kind: {kind}")
        positions = np.asarray(constraints["wrist_positions"], dtype=np.float32)
        rotations = np.asarray(constraints.get("wrist_rotations", constraints.get("wrist_rotation")),
                                dtype=np.float32)
        if positions.shape != (len(frame_indices), 3) or not np.isfinite(positions).all():
            raise ValueError("Invalid shared wrist-position constraint dimensions")
        if rotations.shape == (3, 3):
            rotations = np.repeat(rotations[None], len(frame_indices), axis=0)
        if rotations.shape != (len(frame_indices), 3, 3) or not np.isfinite(rotations).all():
            raise ValueError("Invalid shared wrist-rotation constraint dimensions")
        anchor_rotation = np.asarray(constraints.get("standing_wrist_rotation"), dtype=np.float32)
        if anchor_rotation.shape != (3, 3) or not np.isfinite(anchor_rotation).all():
            raise ValueError("Invalid standing wrist rotation dimensions")
        for rotation_matrix in (rotations, anchor_rotation[None]):
            if (not np.allclose(np.matmul(rotation_matrix.transpose(0, 2, 1), rotation_matrix),
                                np.eye(3), atol=1e-5)
                    or not np.allclose(np.linalg.det(rotation_matrix), 1., atol=1e-5)):
                raise ValueError("Shared wrist rotations must be proper rotation matrices")

        standing_motion = self._standing_motion(standing)
        base_pos = np.asarray(standing_motion["posed_joints"], dtype=np.float32)[0]
        base_rot = np.asarray(standing_motion["global_rot_mats"], dtype=np.float32)[0]
        root0_k = matrix @ standing[:3]
        # Kimodo canonicalizes only the horizontal root (X/Z); Y remains the
        # absolute hip height. Preserve that convention for hand and torso
        # targets so the generated clip starts at the measured root height.
        base_pos = base_pos.copy()
        base_pos[:, [0, 2]] -= root0_k[[0, 2]][None, :]
        target_pos = positions @ matrix.T
        target_pos[:, [0, 2]] -= root0_k[[0, 2]][None, :]
        target_rot = np.einsum("ab,tbc,cd->tad", matrix, rotations, matrix.T)

        wrist_name = "right_wrist_yaw_skel"
        if wrist_name not in self.model.skeleton.bone_index:
            raise ValueError(f"Kimodo G1 skeleton has no {wrist_name} bone")
        wrist = self.model.skeleton.bone_index[wrist_name]
        hand_names = tuple(getattr(self.model.skeleton, "right_hand_joint_names", (wrist_name,)))
        hand_indices = [self.model.skeleton.bone_index[name] for name in hand_names]
        anchor_rotation = matrix @ anchor_rotation @ matrix.T
        delta = np.einsum("tij,jk->tik", target_rot, anchor_rotation.T)
        joint_pos = np.repeat(base_pos[None], len(frame_indices), axis=0)
        joint_rot = np.repeat(base_rot[None], len(frame_indices), axis=0)
        for index in hand_indices:
            offset = base_pos[index] - base_pos[wrist]
            joint_pos[:, index] = target_pos + np.einsum("tij,j->ti", delta, offset)
            joint_rot[:, index] = np.einsum("tij,jk->tik", delta, base_rot[index])
        ee_constraint = _DeviceEndEffectorConstraintSet(
            self.model.skeleton,
            torch.as_tensor(frame_indices, dtype=torch.long, device=self.device),
            torch.as_tensor(joint_pos, dtype=torch.float32, device=self.device),
            torch.as_tensor(joint_rot, dtype=torch.float32, device=self.device),
            None,
            joint_names=["RightHand", "LeftHand", "LeftFoot", "RightFoot", "Hips"],
        )
        # The Kimodo constructor creates alias index tensors on CPU; move the
        # complete constraint object so those indices match a MUSA model.
        if hasattr(ee_constraint, "to"):
            ee_constraint.to(device=self.device)
        result = [ee_constraint]
        torso_specs = (
            ("torso_yaw_rad", "waist_yaw_joint", "waist_yaw_skel"),
            ("torso_roll_rad", "waist_roll_joint", "waist_roll_skel"),
            ("torso_pitch_rad", "waist_pitch_joint", "waist_pitch_skel"),
        )
        torso_names = []
        torso_qpos = np.repeat(standing[None],
                               len(frame_indices), axis=0)
        for key, joint_name, skeleton_name in torso_specs:
            if key not in constraints:
                continue
            values = np.asarray(constraints[key], dtype=np.float32)
            bound = .52 if key == "torso_pitch_rad" else np.pi
            if (values.shape != (len(frame_indices),) or not np.isfinite(values).all()
                    or (np.abs(values) > bound).any()):
                raise ValueError(f"Invalid {key}: expected one bounded radian value per frame")
            joint_index = ISAACLAB_JOINT_NAMES.index(joint_name)
            torso_qpos[:, 7 + joint_index] = values
            torso_names.append(skeleton_name)
        if torso_names:
            torso_motion = self._cpu_motion_from_qpos(torso_qpos)
            torso_global_rots = np.asarray(torso_motion["global_rot_mats"], dtype=np.float32)
            torso_joint_indices = [self.model.skeleton.bone_index[name] for name in torso_names]
            torso_global_rots = torso_global_rots[:len(frame_indices), torso_joint_indices]
            torso_constraint = _JointRotationConstraint(
                self.model.skeleton,
                torch.as_tensor(frame_indices, dtype=torch.long, device=self.device),
                torch.as_tensor(torso_global_rots, dtype=torch.float32, device=self.device),
                torso_names, torch)
            result.append(torso_constraint)
        info.update(kind="end-effector", constrained_fields=["wrist_positions", "wrist_rotations"],
                    additional_rotation_fields=torso_names)
        return result, info

    def _constraints(self, constraints, frames, standing_qpos=None):
        if constraints is None:
            return [], {}
        if isinstance(constraints, (str, Path)):
            from kimodo.constraints import load_constraints_lst
            loaded = load_constraints_lst(str(constraints), self.model.skeleton,
                                          device=self.device)
            if not loaded:
                raise ValueError("Kimodo constraint file contains no constraints")
            for constraint in loaded:
                indices = constraint.frame_indices
                if (indices.numel() == 0 or int(indices.min()) < 0
                        or int(indices.max()) >= frames):
                    raise ValueError("Kimodo constraint frame indices must index the requested clip")
            return loaded, {"source": str(constraints)}
        if standing_qpos is None:
            raise ValueError("Kimodo shared pose constraints require measured standing_qpos")
        return self._shared_constraints(constraints, frames, standing_qpos)

    def _generate_model(self, prompt: str, frames: int, constraints, first_heading_angle):
        # The pinned Kimodo FK helper allocates its boolean parent mask on CPU.
        # With MUSA constraints, the model's unused ``posed_joints`` decode would
        # therefore fail before qpos conversion. We only consume local rotations
        # and root positions below, so decode posed joints from the generated
        # position features instead; this avoids changing the frozen upstream
        # checkout or the shared execution semantics.
        inverse = self.model.motion_rep.inverse
        self.model.motion_rep.inverse = partial(inverse, posed_joints_from="positions")
        try:
            return self.model(
                [prompt], [frames], constraint_lst=constraints,
                num_denoising_steps=self.steps, num_samples=1, multi_prompt=True,
                cfg_weight=[2.0, self.constraint_guidance],
                first_heading_angle=first_heading_angle,
                num_transition_frames=5,
                post_processing=False,  # shared execution performs its own checks
                return_numpy=True,
            )
        finally:
            self.model.motion_rep.inverse = inverse

    def generate(self, prompt, duration, seed, output, history_qpos=None,
                 pose_constraints=None, constraints_path=None):
        import torch

        if not str(prompt).strip():
            raise ValueError("prompt must not be empty")
        if not np.isfinite(duration) or duration < 0.08:
            raise ValueError("duration must be finite and at least 0.08 seconds")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running", "stage": "kimodo_text_to_reference", "prompt": str(prompt),
            "physics_executed": False, "sonic_executed": False, "task_success": None,
            "upstream_commit": self.upstream_commit,
            "checkpoint": self.lock["checkpoint"],
            "checkpoint_manifest_sha256": self.manifest_sha256,
            "lock_sha256": sha256(KIMODO_LOCK), "device": str(self.device),
            "text_device": str(self.text_device), "text_dtype": self.args.text_dtype,
            "seed": int(seed), "history_conditioning": False,
            "shared_transition": "SonicSimulation.install", "kimodo_model": self.model_name,
            "classifier_free_guidance": [2.0, self.constraint_guidance],
        }
        try:
            torch.set_num_threads(self.args.threads)
            self._seed(torch, int(seed), self.device)
            history = None if history_qpos is None else np.asarray(history_qpos, dtype=np.float32)
            if history is not None:
                if history.ndim != 2 or history.shape[1] != 36 or not np.isfinite(history).all():
                    raise ValueError("history_qpos must be finite [N,36]")
                np.savez_compressed(output / "executed_history.npz", qpos=history,
                                    joint_names=np.array(ISAACLAB_JOINT_NAMES), fps=25)
            report["executed_history_frames"] = 0 if history is None else len(history)
            frames = max(2, int(duration * self.fps))
            if frames > self.max_frames:
                raise ValueError(
                    f"Kimodo G1 is limited to {self.max_frames} frames at {self.fps:g} Hz "
                    f"({self.max_frames / self.fps:g} seconds); requested {frames} frames"
                )
            selected_constraints = pose_constraints if pose_constraints is not None else constraints_path
            if isinstance(selected_constraints, dict):
                # Retain the actual shared goals, not only native field names,
                # so frame/coordinate conversion can be audited per phase.
                write_json(output / "pose_constraints.json", selected_constraints)
                report["pose_constraints_sha256"] = sha256(output / "pose_constraints.json")
            standing_qpos = None if history is None else history[-1]
            constraints, constraint_report = self._constraints(selected_constraints, frames, standing_qpos)
            report["constraints"] = constraint_report
            if isinstance(selected_constraints, (str, Path)):
                constraint_path = Path(selected_constraints)
                report["constraints_sha256"] = sha256(constraint_path)
                (output / "constraints.json").write_bytes(constraint_path.read_bytes())
            elif constraint_report:
                write_json(output / "constraints.json", constraint_report)
            first_heading_angle = 0.0
            if standing_qpos is not None:
                standing_motion = self._standing_motion(standing_qpos)
                heading = np.asarray(standing_motion["global_root_heading"], dtype=np.float32)
                if heading.ndim >= 2 and heading.shape[-1] == 2:
                    first_heading_angle = float(np.arctan2(heading[0, 1], heading[0, 0]))
                elif heading.size:
                    # Some pinned Kimodo motion-representation revisions expose
                    # the same heading as a scalar angle rather than [cos, sin].
                    first_heading_angle = float(heading.reshape(-1)[0])
            report["first_heading_angle_rad"] = first_heading_angle
            synchronize(self.device)
            start = time.perf_counter()
            report["stage"] = "motion_generate"
            with torch.no_grad():
                generated = self._generate_model(str(prompt), frames, constraints,
                                                  first_heading_angle)
            synchronize(self.device)
            report["motion_generate_seconds"] = time.perf_counter() - start

            qpos = np.asarray(self.converter.dict_to_qpos(generated, device="cpu"),
                              dtype=np.float32)
            if qpos.ndim == 3:
                if qpos.shape[0] != 1:
                    raise ValueError(f"Kimodo service expects one generated clip, got {qpos.shape}")
                qpos = qpos[0]
            if qpos.shape != (frames, 36) or not np.isfinite(qpos).all():
                raise ValueError(f"Unexpected/nonfinite Kimodo qpos: {qpos.shape}")
            np.savez_compressed(output / "raw_kimodo_qpos.npz", qpos=qpos,
                                fps=self.fps, joint_names=np.array(self.joint_names))
            report["constraint_projection"] = {"applied": False, "reason": "disabled"}
            if self.projector is not None:
                start = time.perf_counter()
                qpos, projection_report = self.projector.project(qpos, selected_constraints, self.fps)
                report["constraint_projection"] = projection_report
                report["constraint_projection_seconds"] = time.perf_counter() - start
                qpos = qpos.astype(np.float32)
            self.converter.save_csv(qpos, str(output / "motion.csv"))
            np.savez_compressed(output / "kimodo_qpos.npz", qpos=qpos,
                                fps=self.fps, joint_names=np.array(self.joint_names))
            write_json(output / "joint_names.json", self.joint_names)
            source = ReferenceSequence.from_named_joints(
                np.arange(frames, dtype=np.float64) / self.fps,
                qpos[:, 7:], qpos[:, 3:7], self.joint_names)
            reference = _resample_reference(source, 50.0)
            np.savez_compressed(
                output / "reference.npz", times=reference.times, joint_pos=reference.joint_pos,
                joint_vel=reference.velocities(), body_quat=reference.body_quat,
                joint_names=np.array(ISAACLAB_JOINT_NAMES), fps=50)
            (output / "reference.packet").write_bytes(JointStreamEncoder().encode(reference))
            report.update(
                status="passed", stage="complete", source_fps=self.fps,
                target_fps=50, frames=frames, diffusion_steps=self.steps,
                motion_seconds=frames / self.fps,
                generated_motion_seconds_per_wall_second=(frames / self.fps) /
                report["motion_generate_seconds"], source_xml_sha256=sha256(self.converter.xml_path),
                outputs={p.name: sha256(p) for p in output.iterdir()
                         if p.is_file() and p.name != "report.json"})
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc())
            raise
        finally:
            write_json(output / "report.json", report)
        return report
