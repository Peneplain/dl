"""Controlled arm-reference perturbation for paired simulator teacher branches.

The clean and perturbed branches run from the same episode reset with the same
saved ARDY references. An activation fingerprint rejects pairs whose complete
observable execution/controller prefix has diverged before the intervention.
"""

import hashlib

import mujoco
import numpy as np

from baseline.adapters.joints import ARM_INDICES, ISAACLAB_JOINT_NAMES


def activation_fingerprint(simulation, reference):
    """Hash physical and controller state immediately before the first offset."""
    digest = hashlib.sha256()

    def add(value):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())

    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(simulation.model, spec), dtype=np.float64)
    mujoco.mj_getState(simulation.model, simulation.data, state, spec)
    add(state)
    add(np.asarray([simulation.frame_index, simulation.end_time], dtype=np.float64))
    add(simulation.current_ref)
    add(simulation.current_quat)
    add(simulation.finger_target)
    add(simulation.policy.last_action)
    add(simulation.policy.alignment.as_quat())
    add(simulation.reference_checks.offset)
    add(np.asarray([simulation.reference_checks.last_time if
                    simulation.reference_checks.last_time is not None else -1.,
                    simulation.buffer.underruns, int(simulation.holding),
                    int(simulation.terminal_hold_logged)], dtype=np.float64))
    for item in simulation.policy.history:
        for name in sorted(item):
            digest.update(name.encode())
            add(item[name])
    for timestamp, pose in simulation.pose_history:
        add(np.asarray([timestamp]))
        add(pose)
    sequence = simulation.buffer.sequence
    if sequence is not None:
        for value in (sequence.times, sequence.joint_pos, sequence.body_quat):
            add(value)
    for value in (reference.times, reference.joint_pos, reference.body_quat):
        add(value)
    digest.update(simulation.task_phase.encode())
    return digest.hexdigest()


class ScheduledArmOffset:
    """Perturb only one arm joint after a phase-local delay; clean mode stays zero."""

    def __init__(self, *, phase, delay, joint, amplitude, perturb):
        if (phase not in {"reach", "lower", "lift"} or joint not in ISAACLAB_JOINT_NAMES
                or ISAACLAB_JOINT_NAMES.index(joint) not in ARM_INDICES
                or not np.isfinite(delay) or delay < 0
                or not np.isfinite(amplitude) or not 0 < abs(amplitude) <= .15):
            raise ValueError("Invalid bounded arm perturbation")
        self.phase, self.delay, self.joint = phase, float(delay), joint
        self.amplitude, self.perturb = float(amplitude), bool(perturb)
        self.joint_index = ISAACLAB_JOINT_NAMES.index(joint)
        self.reset()

    def reset(self):
        self.phase_start = None
        self.activation = None

    def invalidate_plan(self):
        # Phase timing belongs to the episode, not to one installed reference.
        pass

    def request(self, simulation, nominal):
        desired = np.zeros_like(nominal.joint_pos)
        if simulation.task_phase != self.phase:
            return desired
        now = float(simulation.data.time)
        if self.phase_start is None:
            self.phase_start = now
        if now + 1e-8 < self.phase_start + self.delay:
            return desired
        if self.activation is None:
            self.activation = {"time": now, "frame_index": simulation.frame_index,
                               "phase": self.phase,
                               "snapshot_sha256": activation_fingerprint(simulation, nominal)}
            simulation.event("teacher_pair_activation", **self.activation,
                             perturb=self.perturb, joint=self.joint,
                             amplitude=self.amplitude if self.perturb else 0.)
        if self.perturb:
            desired[:, self.joint_index] = self.amplitude
        return desired
