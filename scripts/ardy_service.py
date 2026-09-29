"""Persistent text-to-ARDY reference service.

The service loads the frozen text encoder and ARDY model once, then accepts
JSONL requests on stdin. Each request produces the same offline reference
files as ``run_ardy.py`` without reloading the 8B text model.
"""

import argparse
import gc
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.adapters.sonic import JointStreamEncoder
from baseline.common import LOCK, ROOT, checked_checkout, sha256, verify_assets, write_json
from scripts.run_ardy import device_for, synchronize


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

    def generate(self, prompt, duration, seed, output):
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
            history = ((int(10 * self.fps) // self.model.num_frames_per_token *
                        self.model.num_frames_per_token - self.model.gen_horizon_len) //
                       self.model.num_frames_per_token) * self.model.num_frames_per_token
            motion_lengths = torch.tensor([frames], device=self.device)
            features = features.to(device=self.device)
            synchronize(self.device)
            start = time.perf_counter()
            report["stage"] = "motion_generate"
            with torch.no_grad():
                motion = self.model(
                    [prompt], frames, num_denoising_steps=self.steps,
                    pad_mask=length_to_mask(motion_lengths),
                    first_heading_angle=torch.zeros(1, device=self.device),
                    motion_mask=None, observed_motion=None,
                    cfg_weight=(2.0, 2.0), crop_history_length=history,
                    text_feat=features,
                    text_pad_mask=torch.ones(1, 1, dtype=torch.bool, device=self.device),
                )
                output_motion = to_numpy(self.model.motion_rep.inverse(motion, is_normalized=True))
            synchronize(self.device)
            report["motion_generate_seconds"] = time.perf_counter() - start

            qpos = self.converter.dict_to_qpos(output_motion, device="cpu")[0]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="musa")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--out-root", type=Path, default=ROOT / "artifacts/ardy-service")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)
    service = ArdyService(args)
    print("READY", file=sys.stderr, flush=True)
    request_index = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        try:
            request = json.loads(line) if line.startswith("{") else {"prompt": line}
            prompt = request["prompt"]
            request_index += 1
            name = request.get("name", f"request-{request_index:04d}")
            name = Path(str(name)).name
            output = args.out_root / name
            report = service.generate(
                prompt, float(request.get("duration", 2.0)), int(request.get("seed", 0)), output)
            print(json.dumps({"status": "passed", "output": str(output),
                              "motion_seconds": report["motion_seconds"]}), flush=True)
        except Exception as error:
            print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
                  flush=True)


if __name__ == "__main__":
    main()
