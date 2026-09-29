"""Frozen Horizon8 text-to-reference bring-up, with explicit CPU/MUSA devices.

Produces real ARDY output, not a physical baseline result. No socket or robot control.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.common import LOCK, ROOT, checked_checkout, sha256, verify_assets, write_json


def device_for(name):
    import torch
    if name.startswith("musa"):
        import torch_musa  # noqa: F401
        if not torch.musa.is_available():
            raise RuntimeError("MUSA requested but unavailable; no automatic CPU fallback")
    device = torch.device(name)
    if device.type not in ("cpu", "musa"):
        raise ValueError("This bring-up entry supports cpu or musa[:index]")
    torch.zeros(1, device=device)
    return device


def synchronize(device):
    if device.type == "musa":
        import torch
        torch.musa.synchronize(device)


def generate(args, report):
    import numpy as np
    import torch
    from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
    from baseline.adapters.reference import ReferenceSequence
    from baseline.adapters.sonic import JointStreamEncoder

    lock = json.loads(LOCK.read_text())
    report["upstream_commit"] = checked_checkout(args.ardy_repo, lock["ardy"]["commit"])
    report["assets"] = {key: verify_assets(args.assets, key)
                        for key in ("ardy", "text_base", "text_adapter")}
    report["lock_sha256"] = sha256(LOCK)
    sys.path.insert(0, str(args.ardy_repo.resolve()))
    # Local snapshots only. Never silently fetch a different model during inference.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TEXT_ENCODER_DEVICE"] = args.text_device
    os.environ.pop("TEXT_ENCODERS_DIR", None)
    from ardy.model import LLM2VecEncoder, load_model
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import to_numpy

    device = device_for(args.device)
    text_device = device_for(args.text_device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "musa" or text_device.type == "musa":
        torch.musa.manual_seed_all(args.seed)
    report.update(torch=str(torch.__version__), device=str(device), text_device=str(text_device),
                  text_dtype=args.text_dtype, acceleration="eager", seed=args.seed)
    if device.type == "musa" or text_device.type == "musa":
        import torch_musa
        report["torch_musa"] = str(torch_musa.__version__)

    start = time.perf_counter()
    print("Loading frozen local LLM2Vec encoder...", flush=True)
    encoder = LLM2VecEncoder(str(args.assets.resolve() / "text_base"),
                            str(args.assets.resolve() / "text_adapter"),
                            dtype=args.text_dtype, llm_dim=4096, device=str(text_device))
    synchronize(text_device)
    report["text_load_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    with torch.no_grad():
        features, lengths = encoder([args.prompt])
        features = features.detach().to(device="cpu", dtype=torch.float32)
    synchronize(text_device)
    if features.shape != (1, 1, 4096) or lengths != [1] or not torch.isfinite(features).all():
        raise ValueError("Unexpected/nonfinite LLM2Vec embedding")
    report["text_encode_seconds"] = time.perf_counter() - start
    np.savez_compressed(args.out / "text_embedding.npz", features=features.numpy(), text=args.prompt)
    # Free the 8B text encoder before loading the motion model on the selected device.
    del encoder
    gc.collect()
    if text_device.type == "musa":
        torch.musa.empty_cache()
    start = time.perf_counter()
    model_name = lock["ardy"]["repo_id"].split("/")[-1]
    print(f"Loading frozen {model_name} on {device}...", flush=True)
    model = load_model(model_name, device=str(device), text_encoder=False,
                       checkpoints_dir=str(args.assets.resolve() / "ardy"))
    model.requires_grad_(False)
    model.eval()
    model.autoencoder.eval()
    fps = float(model.motion_rep.fps)
    if fps != 25 or model.gen_horizon_len != 8:
        raise ValueError("Expected the frozen 25 FPS / Horizon8 G1 model")
    synchronize(device)
    report["motion_load_seconds"] = time.perf_counter() - start
    frames = int(args.duration * fps)
    steps = int(model.diffusion.num_base_steps)
    patch = model.num_frames_per_token
    history = ((int(10 * fps) // patch * patch - model.gen_horizon_len) // patch) * patch
    motion_lengths = torch.tensor([frames], device=device)
    observed_motion, motion_mask = None, None
    if args.constraints:
        from ardy.constraints import load_constraints_lst
        constraints = load_constraints_lst(str(args.constraints), model.skeleton)
        if not constraints:
            raise ValueError("Constraint file was supplied but contains no constraints")
        if any(int(c.frame_indices.min()) < 0 or int(c.frame_indices.max()) >= frames for c in constraints):
            raise ValueError("Constraint frame lies outside requested motion")
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraints, motion_lengths, to_normalize=True, device=str(device))
        report["constraints_sha256"] = sha256(args.constraints)
    features = features.to(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        motion = model([args.prompt], frames, num_denoising_steps=steps,
                       pad_mask=length_to_mask(motion_lengths),
                       first_heading_angle=torch.zeros(1, device=device),
                       motion_mask=motion_mask, observed_motion=observed_motion,
                       cfg_weight=(2.0, 2.0), crop_history_length=history,
                       text_feat=features, text_pad_mask=torch.ones(1, 1, dtype=torch.bool, device=device))
        output = to_numpy(model.motion_rep.inverse(motion, is_normalized=True))
    synchronize(device)
    report["motion_generate_seconds"] = time.perf_counter() - start
    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device="cpu")[0]
    if qpos.shape != (frames, 36) or not np.isfinite(qpos).all():
        raise ValueError(f"Unexpected/nonfinite qpos: {qpos.shape}")
    # Read the same XML and traversal as the upstream CSV converter; no guessed order.
    world = ET.parse(converter.xml_path).getroot().find("worldbody")
    names = [joint.attrib["name"] for joint in world.findall(".//joint")]
    if len(names) != 29 or set(names) != set(ISAACLAB_JOINT_NAMES):
        raise ValueError("Converter XML does not define exactly the 29 expected G1 joints")
    converter.save_csv(qpos, str(args.out / "motion.csv"))
    write_json(args.out / "joint_names.json", names)
    reference = ReferenceSequence.from_named_joints(np.arange(frames) / fps, qpos[:, 7:], qpos[:, 3:7], names)
    reference = reference.sample(np.arange(int(round(reference.times[-1] * 50)) + 1) / 50)
    np.savez_compressed(args.out / "reference.npz", times=reference.times,
                        joint_pos=reference.joint_pos, joint_vel=reference.velocities(),
                        body_quat=reference.body_quat, joint_names=np.array(ISAACLAB_JOINT_NAMES), fps=50)
    (args.out / "reference.packet").write_bytes(JointStreamEncoder().encode(reference))
    report.update(status="passed", model=model_name, fps=fps, frames=frames,
                  source_xml_sha256=sha256(converter.xml_path), diffusion_steps=steps,
                  history_frames=history, cfg_weight=[2.0, 2.0],
                  motion_seconds=frames / fps,
                  generated_motion_seconds_per_wall_second=(frames / fps) / report["motion_generate_seconds"],
                  outputs={p.name: sha256(p) for p in args.out.iterdir()
                           if p.is_file() and p.name != "report.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--text-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--prompt", default="A person stands still.")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--ardy-repo", type=Path, default=ROOT / "third_party/ardy")
    parser.add_argument("--assets", type=Path, default=ROOT / "checkpoints/baseline")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import math
    if not math.isfinite(args.duration) or args.duration < 0.08 or args.threads < 1 or not args.prompt.strip():
        parser.error("Need a nonempty prompt, positive threads and finite duration >= 0.08 s")
    if args.out.exists():
        parser.error("Choose a fresh --out directory")
    args.out.mkdir(parents=True)
    report = {"status": "running", "stage": "ardy_text_to_reference", "prompt": args.prompt,
              "physics_executed": False, "sonic_executed": False, "task_success": None}
    try:
        generate(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
