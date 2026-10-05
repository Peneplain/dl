"""Convert audited nominal rollouts and verified pairs into Risk/Residual windows.

Input is a JSON source plan with `parents`: parent_id, split, prompt_group,
scene_seed, nominal attempt path, and optional pair.json path. Split ownership is
declared before extraction. This tool refuses invented contacts or teacher data.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from baseline.adapters.joints import ARM_INDICES, ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.common import LOCK, sha256, write_json
from baseline.rollout import SavedRollout
from risk_residual.config import CONTEXT_DIM, NOMINAL_DIM, PHASES, SCHEMA, STATE_DIM, ModelConfig
from risk_residual.data import WindowDataset, audit_parents, nominal_risk_labels


JOINTS = tuple(ISAACLAB_JOINT_NAMES)
GROUPS = (
    tuple(i for i, name in enumerate(JOINTS) if name.startswith("left_") and i in ARM_INDICES),
    tuple(i for i, name in enumerate(JOINTS) if name.startswith("right_") and i in ARM_INDICES),
    tuple(i for i, name in enumerate(JOINTS) if name.startswith("waist_")),
    tuple(i for i in range(29) if i not in ARM_INDICES and not JOINTS[i].startswith("waist_")),
)
FIELDS = {
    "history": (16, STATE_DIM), "nominal": (8, NOMINAL_DIM), "context": (8, CONTEXT_DIM),
    "history_times": (16,), "nominal_times": (8,), "decision_time": (),
    "future_valid": (8, 4), "contact_mask": (8, 4),
    "track_target": (8, 4), "contact_target": (8, 4), "balance_target": (8, 4),
    "intervention_valid": (), "intervention_target": (),
    "offset_target": (8, 29), "residual_valid": (8,),
    "correction_sample": (), "stable_sample": (), "parent_id": (),
    "risk_source": (), "teacher_verified": (), "nominal_snapshot": (), "teacher_snapshot": (),
}


def window_arrays(windows):
    arrays = {key: np.asarray([window[key] for window in windows]) for key in FIELDS}
    for key, shape in FIELDS.items():
        if arrays[key].shape != (len(windows), *shape):
            raise ValueError(f"Unexpected {key} shape: {arrays[key].shape}")
    return arrays


def read_context(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        finger_names = [name.removeprefix("finger_ref:") for name in reader.fieldnames or ()
                        if name.startswith("finger_ref:")]
        rows = list(reader)
    if len(rows) < 16:
        raise ValueError(f"Not enough 50 Hz context rows: {path}")
    names = list(JOINTS)
    def vector(row, prefix):
        return np.asarray([float(row[f"{prefix}:{name}"]) for name in names])
    def slots(row, prefix, dimensions):
        return np.asarray([[float(row[f"{prefix}:{i:02d}:{name}"]) for name in dimensions]
                           for i in range(10)])
    parsed = []
    for row in rows:
        parsed.append({
            "time": float(row["sim_time"]), "frame": int(row["frame_index"]),
            "phase": row["phase"].split("_replan_", 1)[0],
            "q": vector(row, "state_q"), "dq": vector(row, "state_dq"),
            "root": np.asarray([float(row[f"root_{axis}"]) for axis in ("qw", "qx", "qy", "qz")]),
            "root_z": float(row["root_z"]),
            "gyro": np.asarray([float(row[f"root_w{axis}"]) for axis in "xyz"]),
            "fingers": np.asarray([float(row[f"finger_ref:{name}"])
                                   for name in finger_names]),
            "slot_times": np.asarray([float(row[f"nominal_time_offset:{i:02d}"])
                                      for i in range(10)]),
            "pos": slots(row, "nominal_pos", names),
            "vel": slots(row, "nominal_vel", names),
            "quat": slots(row, "nominal_quat", ("w", "x", "y", "z")),
        })
    times = np.asarray([row["time"] for row in parsed])
    frames = np.asarray([row["frame"] for row in parsed])
    if (not np.isfinite(times).all() or not np.allclose(np.diff(times), .02, atol=1e-6, rtol=0)
            or not np.array_equal(np.diff(frames), np.ones(len(frames) - 1, dtype=int))):
        raise ValueError("Context must be a contiguous 50 Hz episode")
    for row in parsed:
        if (row["phase"] not in PHASES or row["fingers"].shape != (14,)
                or not np.isfinite(np.concatenate((row["q"], row["dq"], row["root"],
                                                   [row["root_z"]], row["gyro"], row["fingers"],
                                                   row["slot_times"], row["pos"].ravel(),
                                                   row["vel"].ravel(), row["quat"].ravel()))).all()
                or not np.allclose(row["slot_times"], np.arange(10) * .1, atol=1e-6)):
            raise ValueError("Invalid phase, finger commands or nominal reference slots")
    return parsed


def planned(row, times):
    reference = ReferenceSequence(row["time"] + row["slot_times"], row["pos"], row["quat"])
    sample = reference.sample(times)
    return np.concatenate((sample.joint_pos, sample.velocities(), sample.body_quat), axis=-1)


def measured_features(attempt, rows):
    saved = SavedRollout(attempt)
    model = saved.model
    joints = [int(model.joint(name).id) for name in JOINTS]
    joint_q = model.jnt_qposadr[joints]
    joint_v = model.jnt_dofadr[joints]
    root_joint = int(model.joint("floating_base_joint").id)
    root_q = int(model.jnt_qposadr[root_joint])
    root_v = int(model.jnt_dofadr[root_joint])
    block = int(model.joint("task_block_free").qposadr[0])
    block_geom = int(model.geom("task_block_geom").id)
    wrists = [int(model.body(f"{side}_wrist_yaw_link").id) for side in ("left", "right")]
    # Mirrored virtual palm-center offsets in the two wrist-yaw body frames.
    centers = (np.array([.125, -.035, 0.]), np.array([.125, .035, 0.]))
    state_index = {int(frame): index for index, frame in enumerate(saved.frame_index)}
    output = []
    for row in rows:
        index = state_index.get(row["frame"] - 1)
        if index is None:
            raise ValueError("Missing pre-decision MuJoCo state")
        data = saved.restore(index)
        if abs(float(data.time) - row["time"]) > 1e-6:
            raise ValueError("Context and replay state clocks differ")
        if (not np.allclose(data.qpos[joint_q], row["q"], atol=1e-5)
                or not np.allclose(data.qvel[joint_v], row["dq"], atol=1e-5)
                or not np.allclose(data.qpos[root_q + 3:root_q + 7], row["root"], atol=1e-5)
                or abs(float(data.qpos[root_q + 2]) - row["root_z"]) > 1e-5
                or not np.allclose(data.qvel[root_v + 3:root_v + 6], row["gyro"], atol=1e-5)):
            raise ValueError("Context state differs from its pre-decision rollout state")
        block_pos = data.qpos[block:block + 3]
        block_xyzw = np.roll(data.qpos[block + 3:block + 7], -1)
        object_rotation = Rotation.from_quat(block_xyzw)
        relative, contacts = [], np.zeros(2, dtype=np.float32)
        for side, (body, center) in enumerate(zip(wrists, centers)):
            rotation = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
            palm_pos = data.xpos[body] + rotation.apply(center)
            xyz = rotation.inv().apply(block_pos - palm_pos)
            wxyz = np.roll((rotation.inv() * object_rotation).as_quat(), 1)
            relative.extend((*xyz, *wxyz))
            prefix = ("left_hand_", "left_wrist_yaw_link") if side == 0 else (
                "right_hand_", "right_wrist_yaw_link")
            for contact in data.contact[:data.ncon]:
                if contact.dist > 0 or block_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == block_geom else contact.geom1
                name = model.body(int(model.geom_bodyid[other])).name or ""
                if name.startswith(prefix[0]) or name == prefix[1]:
                    contacts[side] = 1
                    break
        phase = np.eye(len(PHASES), dtype=np.float32)[PHASES.index(row["phase"])]
        measured = np.concatenate((row["q"], row["dq"], row["root"], row["gyro"],
                                   relative, contacts, np.abs(row["pos"][0] - row["q"]), phase))
        if measured.shape != (STATE_DIM,) or not np.isfinite(measured).all():
            raise ValueError("Invalid measured history feature vector")
        output.append((measured.astype(np.float32), contacts,
                       hashlib.sha256(saved.states[index].tobytes()).hexdigest()))
    return output, saved.metadata["model_sha256"]


def pair_details(path):
    if path is None:
        return None
    pair_path = Path(path).resolve()
    pair = json.loads(pair_path.read_text())
    if not pair.get("pair_state_verified"):
        raise ValueError("Unverified teacher/nominal activation pair")
    branches = pair["branches"]
    for name in ("nominal", "teacher"):
        report = Path(branches[name]["report"])
        if sha256(report) != branches[name]["report_sha256"]:
            raise ValueError("Paired branch report changed since collection")
        recorded = json.loads(report.read_text())
        if (recorded.get("status") != branches[name]["status"] or
                bool(recorded.get("task_success")) != branches[name]["task_success"] or
                bool(recorded.get("physics_executed")) != branches[name]["physics_executed"] or
                bool(recorded.get("sonic_executed")) != branches[name]["sonic_executed"]):
            raise ValueError("Paired branch outcome disagrees with its report")
        branch = report.parent
        if (sha256(branch / "scene.xml") != branches[name]["scene_sha256"] or
                json.loads((branch / "rollout/metadata.json").read_text())["model_sha256"]
                != branches[name]["model_sha256"]):
            raise ValueError("Paired scene or physics model changed since collection")
    if (branches["teacher"]["scene_sha256"] != branches["nominal"]["scene_sha256"] or
            branches["teacher"]["model_sha256"] != branches["nominal"]["model_sha256"]):
        raise ValueError("Teacher and nominal branches used different physics models")
    teacher_activation = branches["teacher"]["activation"]
    nominal_activation = branches["nominal"]["activation"]
    if (teacher_activation["snapshot_sha256"] != nominal_activation["snapshot_sha256"] or
            teacher_activation["frame_index"] != nominal_activation["frame_index"] or
            abs(teacher_activation["time"] - nominal_activation["time"]) > 1e-8):
        raise ValueError("Teacher did not start from the nominal decision state")
    if pair.get("teacher_verified") != (branches["teacher"]["task_success"] and
                                       branches["teacher"]["status"] == "passed"):
        raise ValueError("Teacher verification disagrees with the clean branch")
    recovery = (pair["teacher_verified"] and
                branches["nominal"]["status"] in {"passed", "stopped"} and
                branches["nominal"]["physics_executed"] and
                branches["nominal"]["sonic_executed"] and
                not branches["nominal"]["task_success"])
    if pair.get("recovery_verified") != recovery:
        raise ValueError("Recovery verification disagrees with branch outcomes")
    return pair


def read_task_contacts(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Grasp task contact stream is empty")
    times = np.asarray([float(row["sim_time"]) for row in rows])
    forces = np.asarray([[float(row["thumb_force_n"]), float(row["finger_force_n"])]
                         for row in rows])
    opposing = np.all(forces > .01, axis=1)
    if not np.isfinite(times).all() or not np.isfinite(forces).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Task contact times must increase")
    return times, opposing


def extract_parent(parent, thresholds, *, context_file=None):
    pair = pair_details(parent.get("pair"))
    attempt = Path(parent["nominal"]).resolve()
    if pair is not None and attempt != Path(pair["branches"]["nominal"]["report"]).parent.resolve():
        raise ValueError("Parent nominal attempt differs from the paired branch")
    report = json.loads((attempt / "report.json").read_text())
    if (report.get("request", {}).get("task") != "grasp" or
            not (attempt / "rollout/metadata.json").exists() or
            not (attempt / "task.csv").exists()):
        raise ValueError("Training source must be a completed grasp rollout")
    request = report["request"]
    if str(parent["scene_seed"]) != str(request["seed"]):
        raise ValueError("Declared scene seed differs from the nominal request")
    stream = context_file or ("effective_context.csv" if pair else "nominal_context.csv")
    rows = read_context(attempt / stream)
    measured, scene_hash = measured_features(attempt, rows)
    task_times, opposing_contact = read_task_contacts(attempt / "task.csv")
    teacher_row = None
    activation_frame = None
    if pair is not None:
        activation_frame = pair["branches"]["nominal"]["activation"]["frame_index"]
        teacher = Path(pair["branches"]["teacher"]["report"]).parent
        teacher_rows = read_context(teacher / "effective_context.csv")
        by_frame = {row["frame"]: row for row in teacher_rows}
        teacher_row = by_frame.get(activation_frame)
        nominal_row = next((row for row in rows if row["frame"] == activation_frame), None)
        if teacher_row is None or nominal_row is None or abs(teacher_row["time"] - nominal_row["time"]) > 1e-6:
            raise ValueError("Teacher reference is absent at the paired decision frame")
    result = []
    # Select 10 Hz decisions for Risk; always include the activation frame for
    # a paired correction even if it falls between the regular decision ticks.
    candidates = set(range(15, len(rows), 5))
    if activation_frame is not None:
        candidates.add(next(i for i, row in enumerate(rows) if row["frame"] == activation_frame))
    for i in sorted(candidates):
        if i < 15 or i >= len(rows):
            continue
        row = rows[i]
        future_indices = i + 2 * np.arange(8)
        valid_steps = future_indices < len(rows)
        if not valid_steps.any():
            continue
        # The recorded current phase/finger context must describe the whole
        # nominal prediction window; never borrow future phase observations.
        if any(rows[j]["phase"] != row["phase"] or
               not np.allclose(rows[j]["fingers"], row["fingers"], atol=1e-6)
               for j in future_indices[valid_steps]):
            continue
        valid = np.broadcast_to(valid_steps[:, None], (8, 4)).copy()
        future_times = row["time"] + np.arange(8) * .04
        nominal = planned(row, future_times)
        context = np.zeros((8, CONTEXT_DIM), dtype=np.float32)
        context[:, PHASES.index(row["phase"])] = 1
        context[:, len(PHASES):] = row["fingers"]
        track, balance, lost, required = [np.zeros((8, 4), dtype=np.float32) for _ in range(4)]
        for h, future_index in enumerate(future_indices[valid_steps]):
            future = rows[future_index]
            for body, joint_indices in enumerate(GROUPS):
                track[h, body] = np.mean(np.abs(nominal[h, list(joint_indices)] - future["q"][list(joint_indices)]))
            quat = future["root"]
            upright = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
            balance[h, :] = float(future["root_z"] < .35 or upright < .4)
            if future["phase"] in {"lift", "hold"}:
                contact_index = np.searchsorted(task_times, future["time"], side="right") - 1
                if contact_index >= 0 and future["time"] - task_times[contact_index] <= .005001:
                    required[h, 1] = 1
                    lost[h, 1] = float(not opposing_contact[contact_index])
        labels = nominal_risk_labels(track, lost, balance, valid, required.astype(bool), thresholds)
        correction = bool(pair and row["frame"] == activation_frame and
                          pair["recovery_verified"] and labels["intervention_target"])
        offset = np.zeros((8, 29), dtype=np.float32)
        residual_valid = np.zeros(8, dtype=bool)
        if correction:
            teacher_action = planned(teacher_row, future_times)
            nonarms = [j for j in range(29) if j not in ARM_INDICES]
            if (not np.allclose(teacher_action[:, 58:], nominal[:, 58:], atol=1e-5)
                    or not np.allclose(teacher_action[:, nonarms], nominal[:, nonarms], atol=1e-5)
                    or not np.allclose(teacher_action[:, np.asarray(nonarms) + 29],
                                       nominal[:, np.asarray(nonarms) + 29], atol=1e-5)):
                raise ValueError("Teacher changed root, velocity context or non-arm coordinates")
            offset = (teacher_action[:, :29] - nominal[:, :29]).astype(np.float32)
            if np.max(np.abs(offset)) > .150001 or np.max(np.abs(offset)) < 1e-5:
                raise ValueError("Teacher offset is zero or outside the correction bound")
            residual_valid[:] = True
        stable = bool(not correction and report.get("task_success") and valid.all() and
                      not labels["intervention_target"])
        if stable:
            residual_valid[:] = True
        snapshot = (pair["branches"]["nominal"]["activation"]["snapshot_sha256"]
                    if correction else measured[i][2])
        result.append({
            "history": np.stack([measured[j][0] for j in range(i - 15, i + 1)]),
            "nominal": nominal.astype(np.float32), "context": context,
            "history_times": np.asarray([rows[j]["time"] for j in range(i - 15, i + 1)]),
            "nominal_times": future_times, "decision_time": row["time"],
            **labels, "offset_target": offset, "residual_valid": residual_valid,
            "correction_sample": correction, "stable_sample": stable,
            "parent_id": parent["parent_id"], "risk_source": "nominal",
            "teacher_verified": correction, "nominal_snapshot": snapshot,
            "teacher_snapshot": snapshot if correction else "",
        })
    source_files = {name: sha256(attempt / name) for name in
                    ("report.json", "task.csv", "rollout/metadata.json", "rollout/states.npz",
                     stream)}
    if pair is not None:
        source_files["pair.json"] = sha256(parent["pair"])
        teacher = Path(pair["branches"]["teacher"]["report"]).parent
        # The clean branch is an independent successful nominal rollout for
        # identity supervision. It retains the same parent split and carries
        # no teacher action or corrected future as an inference input.
        clean_parent = {**parent, "nominal": str(teacher)}
        clean_parent.pop("pair")
        clean_rows, clean_scene_hash, clean_provenance = extract_parent(
            clean_parent, thresholds, context_file="effective_context.csv")
        if clean_scene_hash != scene_hash:
            raise ValueError("Clean identity and perturbed branches use different scenes")
        result.extend(clean_rows)
        source_files.update({"clean/" + name: digest for name, digest in
                             clean_provenance["source_files_sha256"].items()})
    return result, scene_hash, {"parent_id": parent["parent_id"],
                                "prompt_identity": json.dumps(
                                    [request.get("prompt"), request.get("phase_prompts", {}),
                                     request.get("prompt_profile")], sort_keys=True),
                                "source_files_sha256": source_files,
                                "decision_time_bounds":
                                [float(min(row["decision_time"] for row in result)),
                                 float(max(row["decision_time"] for row in result))] if result else None}


def build(source_plan, output, *, thresholds):
    source_plan, output = Path(source_plan).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh dataset directory")
    plan = json.loads(source_plan.read_text())
    parents = plan["parents"]
    audit_parents(parents)
    if np.shape(thresholds) != (4,) or not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("Four positive body-group tracking thresholds are required")
    split_rows = {split: [] for split in ("train", "val", "test")}
    scenes = set()
    source_provenance = []
    prompt_owners = {}
    for parent in parents:
        rows, scene_hash, provenance = extract_parent(parent, thresholds)
        if not rows:
            raise ValueError("Parent has no eligible decision windows")
        split_rows[parent["split"]].extend(rows)
        scenes.add(scene_hash)
        identity = provenance.pop("prompt_identity")
        if identity in prompt_owners and prompt_owners[identity] != parent["split"]:
            raise ValueError("Exact prompt instructions leak across parent splits")
        prompt_owners[identity] = parent["split"]
        source_provenance.append(provenance)
    if any(not rows for rows in split_rows.values()):
        raise ValueError("Train, val and test each need at least one parent with windows")
    output.mkdir(parents=True)
    try:
        for split, windows in split_rows.items():
            np.savez_compressed(output / f"{split}.npz", **window_arrays(windows))
        manifest = {"schema": SCHEMA, "synthetic_inputs": False,
                    "parents": [{key: parent[key] for key in
                                 ("parent_id", "split", "prompt_group", "scene_seed")}
                                for parent in parents],
                    "windows": {split: f"{split}.npz" for split in split_rows},
                    "provenance": {"baseline_lock_sha256": sha256(LOCK),
                                   "scene_hashes": sorted(scenes),
                                   "label_thresholds": {"tracking": list(map(float, thresholds))},
                                   "collector_revision": sha256(__file__),
                                   "source_plan_sha256": sha256(source_plan),
                                   "source_files": source_provenance}}
        write_json(output / "manifest.json", manifest)
        for split in split_rows:
            WindowDataset(output / "manifest.json", split, ModelConfig())
        report = {"status": "passed", "source_plan": str(source_plan),
                  "windows": {split: len(rows) for split, rows in split_rows.items()},
                  "correction_samples": {split: sum(bool(row["correction_sample"]) for row in rows)
                                         for split, rows in split_rows.items()},
                  "stable_samples": {split: sum(bool(row["stable_sample"]) for row in rows)
                                      for split, rows in split_rows.items()},
                  "physics_executed_by_collector": False}
        write_json(output / "report.json", report)
        return report
    except Exception:
        write_json(output / "report.json", {"status": "failed", "source_plan": str(source_plan)})
        raise


def inspect(source_plan, *, thresholds):
    """Audit and summarize pilot parents without claiming a trainable split."""
    plan = json.loads(Path(source_plan).read_text())
    parents = plan["parents"]
    audit_parents(parents)
    if np.shape(thresholds) != (4,) or not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("Four positive body-group tracking thresholds are required")
    result = []
    prompt_owners = {}
    for parent in parents:
        rows, scene_hash, provenance = extract_parent(parent, thresholds)
        identity = provenance["prompt_identity"]
        if identity in prompt_owners and prompt_owners[identity] != parent["split"]:
            raise ValueError("Exact prompt instructions leak across parent splits")
        prompt_owners[identity] = parent["split"]
        if rows:
            with tempfile.TemporaryDirectory() as temporary:
                temp = Path(temporary)
                split = parent["split"]
                np.savez_compressed(temp / f"{split}.npz", **window_arrays(rows))
                write_json(temp / "manifest.json", {
                    "schema": SCHEMA, "parents": [{key: parent[key] for key in
                         ("parent_id", "split", "prompt_group", "scene_seed")}],
                    "windows": {split: f"{split}.npz"},
                    "provenance": {"baseline_lock_sha256": sha256(LOCK),
                                   "scene_hashes": [scene_hash],
                                   "label_thresholds": {"tracking": list(map(float, thresholds))},
                                   "collector_revision": sha256(__file__)}})
                WindowDataset(temp / "manifest.json", split, ModelConfig())
        result.append({"parent_id": parent["parent_id"], "split": parent["split"],
                       "scene_sha256": scene_hash, "windows": len(rows),
                       "risk_positive": sum(bool(row["intervention_target"]) for row in rows),
                       "correction_samples": sum(bool(row["correction_sample"]) for row in rows),
                       "stable_samples": sum(bool(row["stable_sample"]) for row in rows),
                       "decision_time_bounds": provenance["decision_time_bounds"]})
    return {"status": "inspected", "dataset_written": False, "parents": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-only", action="store_true",
                        help="Validate and summarize one or more parents without writing a dataset")
    parser.add_argument("--tracking-thresholds", type=float, nargs=4, required=True,
                        metavar=("LEFT", "RIGHT", "TORSO", "LOWER"))
    args = parser.parse_args()
    thresholds = np.asarray(args.tracking_thresholds, dtype=float)
    if args.inspect_only:
        if args.out is not None:
            parser.error("--out cannot be combined with --inspect-only")
        result = inspect(args.sources, thresholds=thresholds)
    else:
        if args.out is None:
            parser.error("--out is required unless --inspect-only is set")
        result = build(args.sources, args.out, thresholds=thresholds)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
