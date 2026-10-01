"""Persistent text-to-ARDY reference service.

The service loads the frozen text encoder and ARDY model once, then accepts
JSONL requests on stdin. Each request produces the same offline reference
files as ``run_ardy.py`` without reloading the 8B text model.
"""

import gc
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.adapters.sonic import JointStreamEncoder
from baseline.common import LOCK, ROOT, checked_checkout, sha256, verify_assets, write_json
from baseline.runtime import device_for, synchronize


class ArdyService:
    def __init__(self, args):
        import torch

        from baseline.text_encoder import LocalTextEncoder, check_transformers_version

        self.args = args
        self.torch = torch
        self.device = device_for(args.device)
        self.text_device = device_for(args.text_device)
        check_transformers_version()
        lock = json.loads(LOCK.read_text())
        self.lock = lock
        self.upstream_commit = checked_checkout(args.ardy_repo, lock["ardy"]["commit"])
        self.asset_hashes = {
            key: verify_assets(args.assets, key)
            for key in ("ardy", "llama_base", "text_base", "text_adapter")
        }
        sys.path.insert(0, str(args.ardy_repo.resolve()))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TEXT_ENCODER_DEVICE"] = args.text_device

        from ardy.exports.mujoco import MujocoQposConverter
        from ardy.model import load_model

        start = time.perf_counter()
        print("Loading frozen local LLM2Vec encoder...", file=sys.stderr, flush=True)
        self.encoder = LocalTextEncoder(args.assets, dtype=args.text_dtype,
                                         device=str(self.text_device))
        synchronize(self.text_device)
        self.text_load_seconds = time.perf_counter() - start

        model_name = lock["ardy"]["repo_id"].split("/")[-1]
        print(f"Loading frozen {model_name} on {self.device}...", file=sys.stderr, flush=True)
        self.model = load_model(model_name, device=str(self.device), text_encoder=False,
                                checkpoints_dir=str(args.assets.resolve() / "ardy"))
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.autoencoder.eval()
        self.fps = float(self.model.motion_rep.fps)
        if self.fps != 25 or self.model.gen_horizon_len != 8:
            raise ValueError("Expected the frozen 25 FPS / Horizon8 G1 model")
        self.converter = MujocoQposConverter(self.model.skeleton)
        world = ET.parse(self.converter.xml_path).getroot().find("worldbody")
        self.joint_names = [joint.attrib["name"] for joint in world.findall(".//joint")]
        if len(self.joint_names) != 29 or set(self.joint_names) != set(ISAACLAB_JOINT_NAMES):
            raise ValueError("Converter XML does not define exactly the 29 expected G1 joints")
        synchronize(self.device)
        self.motion_load_seconds = time.perf_counter() - start - self.text_load_seconds
        self.steps = int(self.model.diffusion.num_base_steps)
        gc.collect()

    def history_features(self, history_qpos):
        """Convert measured G1 poses in IsaacLab order into normalized ARDY history."""
        from scipy.spatial.transform import Rotation
        torch = self.torch
        qpos = np.asarray(history_qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 4 or len(qpos) % 4:
            raise ValueError("ARDY history needs a multiple of four [N,36] measured poses")
        if not np.isfinite(qpos).all(): raise ValueError("Nonfinite executed history")
        convert = self.converter
        coordinate = convert.mujoco_to_ardy_matrix.numpy()
        local = np.tile(np.eye(3), (len(qpos), self.model.skeleton.nbjoints, 1, 1))
        local[:, 0] = coordinate @ Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix() @ coordinate.T
        named_order = [ISAACLAB_JOINT_NAMES.index(n) for n in self.joint_names]
        angles = qpos[:, 7:][:, named_order]
        axes = convert._mujoco_joint_axis_values_ardy_space.numpy()
        dof_rot = Rotation.from_rotvec((angles[..., None] * axes).reshape(-1, 3)).as_matrix()
        dof_rot = dof_rot.reshape(len(qpos), 29, 3, 3)
        indexes = convert._mujoco_indices_to_ardy_indices.numpy()
        offsets = convert._rot_offsets_f2q.numpy()[indexes]
        local[:, indexes] = np.swapaxes(offsets, -1, -2) @ dof_rot
        root = qpos[:, :3] @ coordinate.T
        # Verify the inverse against the exact upstream export before using history.
        rebuilt = convert.to_qpos(torch.tensor(local[None], dtype=torch.float32),
                                  torch.tensor(root[None], dtype=torch.float32)).numpy()[0]
        if not np.allclose(rebuilt[:, 7:], angles, atol=2e-5):
            raise ValueError("Executed-history joint conversion failed its round-trip")
        if not np.allclose(rebuilt[:, :3], qpos[:, :3], atol=2e-5):
            raise ValueError("Executed-history root conversion failed its round-trip")
        return self.model.motion_rep(torch.tensor(local[None], dtype=torch.float32, device=self.device),
                                     torch.tensor(root[None], dtype=torch.float32, device=self.device),
                                     to_normalize=True)

    def generate(self, prompt, duration, seed, output, history_qpos=None):
        import torch

        from ardy.motion_rep.tools import length_to_mask
        from ardy.tools import to_numpy

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not np.isfinite(duration) or duration < 0.08:
            raise ValueError("duration must be finite and at least 0.08 seconds")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running", "stage": "ardy_text_to_reference", "prompt": prompt,
            "physics_executed": False, "sonic_executed": False, "task_success": None,
            "upstream_commit": self.upstream_commit, "assets": self.asset_hashes,
            "lock_sha256": sha256(LOCK), "torch": str(torch.__version__),
            "device": str(self.device), "text_device": str(self.text_device),
            "text_dtype": self.args.text_dtype, "seed": seed,
        }
        try:
            torch.set_num_threads(self.args.threads)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "musa" or self.text_device.type == "musa":
                torch.musa.manual_seed_all(seed)

            start = time.perf_counter()
            report["stage"] = "text_encode"
            with torch.no_grad():
                features, lengths = self.encoder([prompt])
                features = features.detach().to(device="cpu", dtype=torch.float32)
            synchronize(self.text_device)
            if features.shape != (1, 1, 4096) or lengths != [1] or not torch.isfinite(features).all():
                raise ValueError("Unexpected/nonfinite LLM2Vec embedding")
            report["text_encode_seconds"] = time.perf_counter() - start
            np.savez_compressed(output / "text_embedding.npz", features=features.numpy(), text=prompt)

            frames = int(duration * self.fps)
            init_history = None if history_qpos is None else self.history_features(history_qpos)
            history_count = 0 if init_history is None else init_history.shape[1]
            total_frames = frames + history_count
            if history_qpos is not None:
                np.savez_compressed(output / "executed_history.npz", qpos=history_qpos,
                                    joint_names=np.array(ISAACLAB_JOINT_NAMES), fps=25)
            report["executed_history_frames"] = history_count
            history = ((int(10 * self.fps) // self.model.num_frames_per_token *
                        self.model.num_frames_per_token - self.model.gen_horizon_len) //
                       self.model.num_frames_per_token) * self.model.num_frames_per_token
            motion_lengths = torch.tensor([total_frames], device=self.device)
            features = features.to(device=self.device)
            synchronize(self.device)
            start = time.perf_counter()
            report["stage"] = "motion_generate"
            with torch.no_grad():
                motion = self.model(
                    [prompt], total_frames, num_denoising_steps=self.steps,
                    pad_mask=length_to_mask(motion_lengths),
                    first_heading_angle=None if init_history is not None else torch.zeros(1, device=self.device),
                    motion_mask=None, observed_motion=None,
                    cfg_weight=(2.0, 2.0), crop_history_length=None if init_history is not None else history,
                    init_history_sequence=init_history,
                    text_feat=features,
                    text_pad_mask=torch.ones(1, 1, dtype=torch.bool, device=self.device),
                )
                output_motion = to_numpy(self.model.motion_rep.inverse(motion, is_normalized=True))
            synchronize(self.device)
            report["motion_generate_seconds"] = time.perf_counter() - start

            qpos = self.converter.dict_to_qpos(output_motion, device="cpu")[0][history_count:]
            if qpos.shape != (frames, 36) or not np.isfinite(qpos).all():
                raise ValueError(f"Unexpected/nonfinite qpos: {qpos.shape}")
            self.converter.save_csv(qpos, str(output / "motion.csv"))
            write_json(output / "joint_names.json", self.joint_names)
            reference = ReferenceSequence.from_named_joints(
                np.arange(frames) / self.fps, qpos[:, 7:], qpos[:, 3:7], self.joint_names)
            reference = reference.sample(np.arange(int(round(reference.times[-1] * 50)) + 1) / 50)
            np.savez_compressed(
                output / "reference.npz", times=reference.times, joint_pos=reference.joint_pos,
                joint_vel=reference.velocities(), body_quat=reference.body_quat,
                joint_names=np.array(ISAACLAB_JOINT_NAMES), fps=50)
            (output / "reference.packet").write_bytes(JointStreamEncoder().encode(reference))
            report.update(
                status="passed", stage="complete", model=self.lock["ardy"]["repo_id"].split("/")[-1],
                fps=self.fps, frames=frames, source_xml_sha256=sha256(self.converter.xml_path),
                diffusion_steps=self.steps, history_frames=history, cfg_weight=[2.0, 2.0],
                motion_seconds=frames / self.fps,
                generated_motion_seconds_per_wall_second=(frames / self.fps) /
                report["motion_generate_seconds"],
                outputs={p.name: sha256(p) for p in output.iterdir()
                         if p.is_file() and p.name != "report.json"})
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(output / "report.json", report)
        return report

