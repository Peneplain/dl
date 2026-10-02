from dataclasses import dataclass

import numpy as np

from .joints import to_isaaclab


class BufferUnderrun(RuntimeError):
    """Executor must hold/stop identically across methods and log this event."""


@dataclass
class ReferenceSequence:
    times: np.ndarray
    joint_pos: np.ndarray
    body_quat: np.ndarray  # root wxyz; multi-body fields require FK after correction

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.joint_pos = np.asarray(self.joint_pos, dtype=np.float32)
        self.body_quat = np.asarray(self.body_quat, dtype=np.float32)
        n = len(self.times)
        if self.times.shape != (n,) or n < 2 or self.joint_pos.shape != (n, 29):
            raise ValueError("Expected >=2 timestamps and [N,29] joint positions")
        if self.body_quat.shape != (n, 4):
            raise ValueError("Expected root quaternion [N,4] in wxyz order")
        if not all(np.isfinite(a).all() for a in (self.times, self.joint_pos, self.body_quat)):
            raise ValueError("Reference contains nonfinite values")
        if (np.diff(self.times) <= 0).any():
            raise ValueError("Reference timestamps must strictly increase")
        if not np.allclose(np.linalg.norm(self.body_quat, axis=-1), 1, atol=1e-3):
            raise ValueError("Quaternion must be unit length; convert xyzw explicitly before import")

    @classmethod
    def from_named_joints(cls, times, positions, root_wxyz, names):
        return cls(times, to_isaaclab(positions, names), root_wxyz)

    def sample(self, times):
        """Joint linear interpolation and shortest-path quaternion SLERP; no extrapolation."""
        times = np.asarray(times, float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
            raise ValueError("Need at least two finite query timestamps")
        if times[0] < self.times[0] - 1e-8 or times[-1] > self.times[-1] + 1e-8:
            raise BufferUnderrun("Requested horizon extends beyond buffered motion")
        if (np.diff(times) <= 0).any():
            raise ValueError("Query times must strictly increase")
        positions = np.stack([np.interp(times, self.times, self.joint_pos[:, j])
                              for j in range(29)], axis=-1)
        idx = np.clip(np.searchsorted(self.times, times, side="right") - 1, 0, len(self.times) - 2)
        fraction = ((times - self.times[idx]) / (self.times[idx + 1] - self.times[idx]))[:, None]
        q0, q1 = self.body_quat[idx], self.body_quat[idx + 1].copy()
        dot = (q0 * q1).sum(-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        angle = np.arccos(np.clip(np.abs(dot), 0, 1))
        sine = np.sin(angle).clip(1e-8)
        slerp = (np.sin((1 - fraction) * angle) * q0 + np.sin(fraction * angle) * q1) / sine
        quat = np.where(np.abs(dot) > 0.9995, (1 - fraction) * q0 + fraction * q1, slerp)
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        return ReferenceSequence(times, positions, quat)

    def velocities(self):
        """Recompute AFTER corrections/resampling. Units: rad/s."""
        return np.gradient(self.joint_pos, self.times, axis=0).astype(np.float32)

    def sample_with_terminal_hold(self, times):
        """Explicit finite-clip hold; caller must log missing future coverage.

        This is not silent extrapolation: only the terminal pose is repeated.
        Derivatives must be computed from the returned, resampled positions.
        """
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or (np.diff(times) <= 0).any()):
            raise ValueError("Need increasing finite query timestamps")
        if times[0] < self.times[0] - 1e-8:
            raise BufferUnderrun("Query precedes buffered motion")
        clipped = np.clip(times, self.times[0], self.times[-1])
        unique, inverse = np.unique(clipped, return_inverse=True)
        if len(unique) == 1:
            pos = np.repeat(self.joint_pos[-1:], len(times), axis=0)
            quat = np.repeat(self.body_quat[-1:], len(times), axis=0)
        else:
            sampled = self.sample(unique)
            pos, quat = sampled.joint_pos[inverse], sampled.body_quat[inverse]
        return ReferenceSequence(times, pos, quat)


class ReferenceBuffer:
    """Timestamped append/replan buffer; shared by baseline and corrected methods."""

    def __init__(self, max_gap=0.041):
        if not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be positive")
        self.max_gap = max_gap
        self.sequence = None
        self.underruns = 0

    def push(self, chunk: ReferenceSequence):
        if (np.diff(chunk.times) > self.max_gap).any():
            raise ValueError("Motion chunk contains an unobserved time gap")
        if self.sequence is None:
            self.sequence = chunk
            return
        old = self.sequence
        if chunk.times[0] - old.times[-1] > self.max_gap:
            raise BufferUnderrun("New chunk leaves a gap; do not interpolate across missing motion")
        # A newer plan replaces the overlapping future rather than duplicating timestamps.
        keep = old.times < chunk.times[0] - 1e-8
        self.sequence = ReferenceSequence(np.concatenate((old.times[keep], chunk.times)),
                                          np.concatenate((old.joint_pos[keep], chunk.joint_pos)),
                                          np.concatenate((old.body_quat[keep], chunk.body_quat)))

    def sample(self, now, count, dt):
        if (not isinstance(count, (int, np.integer)) or count < 2
                or not np.isfinite(now) or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Need at least two samples and positive dt")
        try:
            if self.sequence is None:
                raise BufferUnderrun("No motion buffered")
            return self.sequence.sample(now + np.arange(count) * dt)
        except BufferUnderrun:
            self.underruns += 1
            raise

    def discard_before(self, now):
        if self.sequence is not None:
            s = self.sequence
            start = min(max(0, np.searchsorted(s.times, now) - 1), len(s.times) - 2)
            self.sequence = ReferenceSequence(s.times[start:], s.joint_pos[start:], s.body_quat[start:])
