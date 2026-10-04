"""Shared arm-correction limits independent of learned models or checkpoints."""

import numpy as np

from baseline.adapters.joints import ARM_INDICES, ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence


class ReferenceChecks:
    """Limit offsets, committing only the current control sample each tick."""

    def __init__(self, joint_ranges, *, max_offset=.15, offset_rate=.5, control_dt=.02):
        self.ranges = np.asarray(joint_ranges, dtype=float)
        if (self.ranges.shape != (29, 2) or not np.isfinite(self.ranges).all()
                or (self.ranges[:, 0] >= self.ranges[:, 1]).any()):
            raise ValueError("Expected finite increasing joint ranges [29,2]")
        if not all(np.isfinite(v) and v > 0 for v in (max_offset, offset_rate, control_dt)):
            raise ValueError("Correction limits and control_dt must be finite and positive")
        self.max_offset, self.offset_rate, self.control_dt = max_offset, offset_rate, control_dt
        self.mask = np.zeros(29, dtype=bool)
        self.mask[list(ARM_INDICES)] = True
        self.reset()

    def reset(self):
        self.last_time = None
        self.offset = np.zeros(29)

    def apply(self, nominal, desired):
        desired = np.asarray(desired)
        if desired.shape != nominal.joint_pos.shape or not np.isfinite(desired).all():
            raise ValueError("Offset must be finite [N,29]")
        if np.any(desired[:, ~self.mask] != 0):
            raise ValueError("Correction requested outside the 14 arm joints")
        now = float(nominal.times[0])
        elapsed = self.control_dt if self.last_time is None else now - self.last_time
        if elapsed <= 0 or not np.isfinite(elapsed):
            raise ValueError("Reference checks require strictly advancing simulation time")
        lower = np.maximum(-self.max_offset, self.ranges[:, 0] - nominal.joint_pos)
        upper = np.minimum(self.max_offset, self.ranges[:, 1] - nominal.joint_pos)
        lower[:, ~self.mask] = upper[:, ~self.mask] = 0
        applied = np.empty_like(nominal.joint_pos)
        previous = self.offset.copy()
        for i, dt in enumerate(np.r_[elapsed, np.diff(nominal.times)]):
            lo = np.maximum(lower[i], previous - self.offset_rate * dt)
            hi = np.minimum(upper[i], previous + self.offset_rate * dt)
            if np.any(lo > hi + 1e-7):
                raise ValueError("Joint limits and correction rate cannot both be satisfied")
            applied[i] = np.clip(desired[i], lo, hi)
            previous = applied[i]
        checked = ReferenceSequence(nominal.times.copy(), nominal.joint_pos + applied,
                                    nominal.body_quat.copy())
        self.offset = applied[0].copy()
        self.last_time = now
        return checked
