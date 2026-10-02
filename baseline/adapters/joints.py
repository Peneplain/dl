"""SONIC G1 joint order verified against upstream b042411f (see docs/integration.md)."""

import numpy as np

ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# Explicit anatomical membership; never infer arm coordinates from contiguous slices.
ARM_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint" for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
)
ARM_INDICES = tuple(ISAACLAB_JOINT_NAMES.index(name) for name in ARM_JOINT_NAMES)


def to_isaaclab(values: np.ndarray, source_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Reorder by explicit names, never assume the order of a CSV/qpos array."""
    values = np.asarray(values)
    if len(set(source_names)) != len(source_names) or values.shape[-1] != len(source_names):
        raise ValueError("Duplicate joint names or incompatible array width")
    missing = set(ISAACLAB_JOINT_NAMES) - set(source_names)
    if missing:
        raise ValueError(f"Missing required G1 joints: {sorted(missing)}")
    return values[..., [source_names.index(name) for name in ISAACLAB_JOINT_NAMES]].copy()
