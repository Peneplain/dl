"""Versioned input layout and initial (uncalibrated) training settings."""

import math
from dataclasses import asdict, dataclass

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES

PHASES = ("stand", "approach", "settle", "prepare", "reach", "lower", "close", "lift", "hold")
BODY_GROUPS = ("left_arm_hand", "right_arm_hand", "torso", "lower_body")
# All state fields are measured at or BEFORE the decision time. Relative poses
# are object-in-palm xyz (m) + wxyz for left then right hand. Contacts are binary.
STATE_FIELDS = (
    ("joint_pos_rad", 29), ("joint_vel_rad_s", 29), ("root_wxyz", 4),
    ("base_angular_velocity_rad_s", 3), ("object_in_palms_xyz_wxyz", 14),
    ("hand_object_contacts", 2), ("tracking_error_rad", 29), ("phase_one_hot", len(PHASES)),
)
STATE_DIM = sum(width for _, width in STATE_FIELDS)
# Future nominal q, dq and root wxyz plus shared planned phase and 14 fingers.
NOMINAL_DIM = 29 + 29 + 4
CONTEXT_DIM = len(PHASES) + 14
SCHEMA = {
    "version": 2, "state_fields": [list(field) for field in STATE_FIELDS],
    "joint_names": list(ISAACLAB_JOINT_NAMES), "body_groups": list(BODY_GROUPS),
    "phases": list(PHASES), "history_dt": .02, "risk_dt": .04,
    "nominal_fields": ["joint_pos_rad", "joint_vel_rad_s", "root_wxyz"],
    "context_fields": ["phase_one_hot", "left_fingers_rad", "right_fingers_rad"],
}


@dataclass(frozen=True)
class ModelConfig:
    history_steps: int = 16  # 300 ms from oldest to newest at 50 Hz
    horizon: int = 8
    body_groups: int = 4
    token_dim: int = 32
    width: int = 256
    layers: int = 4
    heads: int = 8
    dropout: float = .1
    max_offset: float = .15  # radians; pilot setting, calibrate before experiments

    def __post_init__(self):
        if not 11 <= self.history_steps <= 26:
            raise ValueError("History must cover 200–500 ms at 50 Hz")
        if (self.horizon, self.body_groups, self.token_dim) != (8, 4, 32):
            raise ValueError("Proposal requires H=8, B=4, D=32")
        if (self.width < 1 or self.layers < 1 or self.heads < 1
                or self.width % self.heads or not 0 <= self.dropout < 1
                or not math.isfinite(self.max_offset) or self.max_offset <= 0):
            raise ValueError("Invalid Transformer or correction settings")

    def to_dict(self):
        return asdict(self)
