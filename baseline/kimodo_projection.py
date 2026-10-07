"""Project nominal Kimodo wrist constraints onto the source robot's arm hinges.

This generator/export boundary uses frozen-source FK and common pose goals,
not an execution controller, learned residual, teacher or future rollout.
SONIC and the shared checks still execute the resulting nominal references.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES

RIGHT_ARM_JOINTS = tuple(name for name in ISAACLAB_JOINT_NAMES
                         if name.startswith("right_") and any(
                             part in name for part in ("shoulder", "elbow", "wrist")))


class KimodoConstraintProjector:
    """Fit seven right-arm hinges while preserving all other qpos fields."""

    VERSION = 2
    MAX_OFFSET_RAD = .7
    REGULARIZATION = .05
    MAX_EVALUATIONS = 60
    POSITION_WEIGHT = 20.
    ROTATION_WEIGHT = .2

    def __init__(self, xml_path: Path, joint_names):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 29 or set(self.joint_names) != set(ISAACLAB_JOINT_NAMES):
            raise ValueError("Projection requires the complete named 29-joint G1 stream")
        self.q_indices = np.array([int(self.model.joint(name).qposadr[0])
                                   for name in self.joint_names])
        self.arm_columns = np.array([7 + self.joint_names.index(name)
                                     for name in RIGHT_ARM_JOINTS])
        self.arm_q_indices = np.array([int(self.model.joint(name).qposadr[0])
                                       for name in RIGHT_ARM_JOINTS])
        self.arm_limits = np.array([self.model.joint(name).range
                                    for name in RIGHT_ARM_JOINTS])
        if (len(self.arm_columns) != 7 or self.model.nq != 36
                or not np.array_equal(np.sort(self.q_indices), np.arange(7, 36))
                or np.any(self.arm_limits[:, 0] >= self.arm_limits[:, 1])):
            raise ValueError("Projection requires a free-root G1 and seven bounded right-arm hinges")
        self.wrist_body = self.model.body("right_wrist_yaw_link").id

    def _pose(self, row, arm=None):
        self.data.qpos[:7] = row[:7]
        self.data.qpos[self.q_indices] = row[7:]
        if arm is not None:
            self.data.qpos[self.arm_q_indices] = arm
        self.mujoco.mj_forward(self.model, self.data)
        return (self.data.xpos[self.wrist_body].copy(),
                self.data.xmat[self.wrist_body].reshape(3, 3).copy())

    @staticmethod
    def _goals(constraints, frames, fps):
        if (constraints.get("schema_version") != 1
                or constraints.get("coordinate_frame") != "mujoco_world"
                or constraints.get("units") != "SI"):
            raise ValueError("Projection requires the shared SI MuJoCo-world pose schema")
        indices = np.asarray(constraints["frame_indices"], dtype=np.float64)
        if (indices.ndim != 1 or len(indices) < 1 or indices[0] < 0
                or not np.isfinite(indices).all() or (np.diff(indices) <= 0).any()):
            raise ValueError("Invalid projection goal frame indices")
        # Preserve the shared 25 Hz timestamps, rather than stretching them
        # to the last sample of Kimodo's different 30 Hz source grid.
        times = indices / 25.
        requested = np.clip(np.arange(frames) / fps, times[0], times[-1])
        positions = np.asarray(constraints["wrist_positions"], dtype=np.float64)
        rotations = np.asarray(constraints.get("wrist_rotations", constraints.get("wrist_rotation")),
                               dtype=np.float64)
        if rotations.shape == (3, 3):
            rotations = np.repeat(rotations[None], len(indices), axis=0)
        anchor = np.asarray(constraints["standing_qpos"], dtype=np.float64)
        if (positions.shape != (len(indices), 3)
                or rotations.shape != (len(indices), 3, 3) or anchor.shape != (36,)
                or not all(np.isfinite(a).all() for a in (positions, rotations, anchor))):
            raise ValueError("Invalid projection goal dimensions or values")
        if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)):
            raise ValueError("Projection goals must use proper rotation matrices")
        positions = np.stack([np.interp(requested, times, positions[:, axis])
                              for axis in range(3)], axis=-1)
        if len(indices) == 1:
            rotations = np.repeat(rotations, frames, axis=0)
        else:
            rotations = Slerp(times, Rotation.from_matrix(rotations))(requested).as_matrix()
        return positions, rotations, anchor

    def project(self, qpos, constraints, fps):
        qpos = np.asarray(qpos, dtype=np.float64)
        if (qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 1
                or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0):
            raise ValueError("Projection requires finite [N,36] qpos and positive FPS")
        if not isinstance(constraints, dict) or constraints.get("kind", "wrist") not in {"wrist", "wrist_pose"}:
            return qpos.copy(), {"applied": False, "reason": "no_shared_wrist_goals"}
        positions, rotations, anchor = self._goals(constraints, len(qpos), fps)
        offset = np.asarray(constraints.get("wrist_effector_offset_m", [0., 0., 0.]), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .4:
            raise ValueError("Projection effector offset must be a finite local SI 3-vector")
        targets = positions + np.einsum("tij,j->ti", rotations, offset)
        projected = qpos.copy()
        before, after, rotation_after, evaluations = [], [], [], []
        effector_before, effector_after = [], []
        for frame, row in enumerate(qpos):
            # Kimodo canonicalizes horizontal root translation only. The
            # shared stream has no root XYZ, but world-space FK needs it.
            world = row.copy()
            world[:2] += anchor[:2]
            nominal = row[self.arm_columns]
            lower = np.maximum(self.arm_limits[:, 0], nominal - self.MAX_OFFSET_RAD)
            upper = np.minimum(self.arm_limits[:, 1], nominal + self.MAX_OFFSET_RAD)
            if np.any(lower >= upper):
                raise ValueError("Nominal right-arm reference is outside the projection bound")
            start_position, start_rotation = self._pose(world)
            before.append(float(np.linalg.norm(start_position - positions[frame])))
            effector_before.append(float(np.linalg.norm(
                start_position + start_rotation @ offset - targets[frame])))

            def residual(arm):
                position, rotation = self._pose(world, arm)
                rotation_error = Rotation.from_matrix(rotations[frame].T @ rotation).as_rotvec()
                # Fit the declared hand-center point when available. A hard
                # wrist orientation can be infeasible on seven physical
                # hinges; retain it as a soft, reported nominal preference.
                effector_error = position + rotation @ offset - targets[frame]
                return np.r_[self.POSITION_WEIGHT * effector_error,
                             self.ROTATION_WEIGHT * rotation_error,
                             self.REGULARIZATION * (arm - nominal)]

            fit = least_squares(residual, np.clip(nominal, lower, upper),
                                bounds=(lower, upper), max_nfev=self.MAX_EVALUATIONS,
                                ftol=1e-6, xtol=1e-6, gtol=1e-6)
            if not np.isfinite(fit.x).all():
                raise ValueError("Nonfinite nominal constraint projection")
            projected[frame, self.arm_columns] = fit.x
            position, rotation = self._pose(world, fit.x)
            after.append(float(np.linalg.norm(position - positions[frame])))
            effector_after.append(float(np.linalg.norm(
                position + rotation @ offset - targets[frame])))
            rotation_after.append(float(np.linalg.norm(
                Rotation.from_matrix(rotations[frame].T @ rotation).as_rotvec())))
            evaluations.append(int(fit.nfev))
        untouched = np.ones(36, dtype=bool)
        untouched[self.arm_columns] = False
        if not np.array_equal(projected[:, untouched], qpos[:, untouched]):
            raise AssertionError("Projection changed a non-right-arm coordinate")
        offsets = projected[:, self.arm_columns] - qpos[:, self.arm_columns]
        if np.max(np.abs(offsets)) > self.MAX_OFFSET_RAD + 1e-8:
            raise AssertionError("Projection exceeded its arm-only bound")
        return projected, {
            "applied": True, "version": self.VERSION,
            "kind": "nominal_actuator_axis_wrist_projection",
            "joint_names": list(RIGHT_ARM_JOINTS), "max_offset_rad": self.MAX_OFFSET_RAD,
            "regularization": self.REGULARIZATION, "max_evaluations": self.MAX_EVALUATIONS,
            "position_weight": self.POSITION_WEIGHT, "rotation_weight": self.ROTATION_WEIGHT,
            "effector_offset_wrist_m": offset.tolist(),
            "effector_error_before_m_mean": float(np.mean(effector_before)),
            "effector_error_after_m_mean": float(np.mean(effector_after)),
            "effector_error_after_m_max": float(np.max(effector_after)),
            "effector_error_after_m_terminal": effector_after[-1],
            "physics_stepped": False, "execution_feedback_used": False, "teacher_used": False,
            "position_error_before_m_mean": float(np.mean(before)),
            "position_error_before_m_max": float(np.max(before)),
            "position_error_after_m_mean": float(np.mean(after)),
            "position_error_after_m_max": float(np.max(after)),
            "rotation_error_after_rad_max": float(np.max(rotation_after)),
            "position_error_after_m_terminal": after[-1],
            "rotation_error_after_rad_terminal": rotation_after[-1],
            "observed_max_offset_rad": float(np.max(np.abs(offsets))),
            "solver_evaluations_max": max(evaluations),
        }
