"""Shared free-base MuJoCo execution for the frozen SONIC controller."""

from collections import deque
import csv
from pathlib import Path
import time

import mujoco
import numpy as np

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import BufferUnderrun, ReferenceBuffer, ReferenceSequence
from baseline.common import sha256
from baseline.sonic_policy import SonicPolicy


class SimulationStop(RuntimeError):
    pass


class SonicSimulation:
    def __init__(self, assets, repo, output, threads=2):
        self.policy = SonicPolicy(assets, repo, threads)
        self.scene = repo / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.model.opt.timestep = .005
        self.data = mujoco.MjData(self.model)
        self.output = Path(output)
        self.events = (self.output / "events.jsonl").open("w", buffering=1)
        self.trajectory_file = (self.output / "trajectory.csv").open("w", buffering=1)
        self.trajectory = csv.writer(self.trajectory_file)
        self.trajectory.writerow(["sim_time", "wall_time", "frame_index", "root_x", "root_y", "root_z",
                                  "root_qw", "root_qx", "root_qy", "root_qz"] +
                                 [f"q:{n}" for n in ISAACLAB_JOINT_NAMES] +
                                 [f"ref:{n}" for n in ISAACLAB_JOINT_NAMES] +
                                 [f"torque:{n}" for n in ISAACLAB_JOINT_NAMES])
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if root < 0 or self.model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The robot must have a free base")
        self.root_q = int(self.model.jnt_qposadr[root])
        self.root_v = int(self.model.jnt_dofadr[root])
        joints = [self.model.joint(n).id for n in ISAACLAB_JOINT_NAMES]
        self.q_indices = self.model.jnt_qposadr[joints]
        self.v_indices = self.model.jnt_dofadr[joints]
        self.ranges = self.model.jnt_range[joints].copy()
        by_joint = {int(self.model.actuator_trnid[i, 0]): i for i in range(self.model.nu)}
        self.actuators = np.array([by_joint[j] for j in joints])
        if not np.allclose(self.model.actuator_gear[self.actuators, 0], 1):
            raise ValueError("Expected direct joint torque actuators")
        body_joints = set(joints)
        fingers = [j for j in range(self.model.njnt)
                   if "_hand_" in (self.model.joint(j).name or "") and j in by_joint]
        self.finger_q = self.model.jnt_qposadr[fingers]
        self.finger_v = self.model.jnt_dofadr[fingers]
        self.finger_actuators = np.array([by_joint[j] for j in fingers])
        if len(fingers) != 14 or len(body_joints) != 29:
            raise ValueError("Expected 29 body joints and 14 articulated finger joints")
        self.finger_target = np.clip(np.zeros(14), self.model.jnt_range[fingers, 0],
                                     self.model.jnt_range[fingers, 1])
        self.started = time.perf_counter()
        self.latencies = []
        self.tracking_errors = []
        self.deadline_misses = 0
        self.limit_clamps = 0
        self.reset_count = 0
        self.frame_index = 0
        self.reset()

    def event(self, event, **fields):
        import json
        self.events.write(json.dumps({"event": event, "sim_time": float(self.data.time),
                                     "wall_time": time.perf_counter() - self.started,
                                     **fields}, allow_nan=False) + "\n")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.q_indices] = self.policy.parameters.default
        self.data.qpos[self.finger_q] = self.finger_target
        mujoco.mj_forward(self.model, self.data)
        # Set initial ground clearance once, never constrain or teleport the base during execution.
        feet = [i for i in range(self.model.ngeom)
                if "ankle_roll" in (self.model.body(int(self.model.geom_bodyid[i])).name or "")
                and self.model.geom_contype[i]]
        bottoms = [self.data.geom_xpos[i, 2] -
                   np.abs(self.data.geom_xmat[i].reshape(3, 3)[2]) @ self.model.geom_size[i]
                   for i in feet if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
        if bottoms: self.data.qpos[self.root_q + 2] += .003 - min(bottoms)
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset()
        self.buffer = ReferenceBuffer(max_gap=.021)
        self.current_ref = self.policy.parameters.default.copy()
        self.current_quat = self.data.qpos[self.root_q + 3:self.root_q + 7].copy()
        self.end_time = 0.0
        self.holding = True
        self.pose_history = deque(maxlen=200)
        self.reset_count += 1
        self.event("reset", reset_index=self.reset_count)

    def body_pose(self):
        return np.concatenate([self.data.qpos[self.root_q:self.root_q + 7],
                               self.data.qpos[self.q_indices]])

    def history(self, count=16):
        """Return recent executed qpos at 25 Hz in ISAACLAB joint order."""
        if count < 4 or count % 4:
            raise ValueError("ARDY history count must be a multiple of four and at least four")
        if not self.pose_history:
            return np.repeat(self.body_pose()[None], count, axis=0)
        timestamps, poses = zip(*self.pose_history)
        end = timestamps[-1]
        samples = end - np.arange(count - 1, -1, -1) / 25
        poses = np.asarray(poses)
        # Use the same shortest-path SLERP as the reference adapter.
        if len(poses) < 2: return np.repeat(poses[-1:], count, axis=0)
        ref = ReferenceSequence(np.array(timestamps), poses[:, 7:], poses[:, 3:7])
        times = np.clip(samples, timestamps[0], end)
        q = np.stack([np.interp(times, timestamps, poses[:, j]) for j in range(36)], axis=-1)
        unique, inverse = np.unique(times, return_inverse=True)
        if len(unique) > 1: q[:, 3:7] = ref.sample(unique).body_quat[inverse]
        return q

    def history_qpos(self, count=16):
        """Explicit alias used by the live ARDY request path."""
        return self.history(count)

    @staticmethod
    def load_reference(path):
        """Load an ARDY-exported 50 Hz reference with its explicit schema."""
        data = np.load(path)
        names = tuple(str(x) for x in data["joint_names"].tolist())
        if names != tuple(ISAACLAB_JOINT_NAMES):
            raise ValueError("Reference joint_names must be the verified ISAACLAB order")
        if int(data["fps"]) != 50:
            raise ValueError("SONIC references must be sampled at 50 Hz")
        return ReferenceSequence(data["times"], data["joint_pos"], data["body_quat"])

    def validate_reference(self, reference):
        lower, upper = self.ranges.T
        excess = np.maximum(lower - reference.joint_pos, reference.joint_pos - upper)
        if excess.max() > .05:
            f, j = np.unravel_index(np.argmax(excess), excess.shape)
            raise ValueError(f"Reference violates {ISAACLAB_JOINT_NAMES[j]} by {excess[f,j]:.3f} rad")
        return ReferenceSequence(reference.times,
                                 np.clip(reference.joint_pos, lower, upper), reference.body_quat)

    def install(self, reference, transition=.6):
        if not np.isfinite(transition) or transition < .1:
            raise ValueError("Transition must be at least .1 seconds")
        reference = self.validate_reference(reference)
        now = float(self.data.time)
        n = max(2, int(round(transition / .02)))
        # Current checked nominal pose to first generated pose. Recompute velocities below.
        blend = ReferenceSequence([now, now + n * .02],
                                  [self.current_ref, reference.joint_pos[0]],
                                  [self.current_quat, reference.body_quat[0]])
        prefix = blend.sample(now + np.arange(n) * .02)
        sequence = ReferenceSequence(np.concatenate([prefix.times, now + n * .02 +
                                                      reference.times - reference.times[0]]),
                                     np.concatenate([prefix.joint_pos, reference.joint_pos]),
                                     np.concatenate([prefix.body_quat, reference.body_quat]))
        self.buffer = ReferenceBuffer(max_gap=.021)
        self.buffer.push(sequence)
        self.velocities = sequence.velocities()
        self.end_time = float(sequence.times[-1])
        self.holding = False
        self.policy.align(self.data.qpos[self.root_q + 3:self.root_q + 7], sequence.body_quat[0])
        self.event("reference_installed", frames=len(sequence.times), end_time=self.end_time,
                   transition_seconds=n * .02)

    def lookahead(self):
        times = self.data.time + np.arange(10) * self.policy.future_step * .02
        sequence = self.buffer.sequence
        if sequence is None or self.data.time > self.end_time + 1e-8:
            if not self.holding:
                self.buffer.underruns += 1
                self.holding = True
                self.event("buffer_underrun", behavior="hold_terminal_nominal_pose")
            return (np.repeat(self.current_ref[None], 10, axis=0), np.zeros((10, 29)),
                    np.repeat(self.current_quat[None], 10, axis=0))
        # Future samples outside a completed finite clip explicitly hold its endpoint.
        clipped = np.minimum(times, self.end_time)
        unique, inverse = np.unique(clipped, return_inverse=True)
        if len(unique) == 1:
            p = np.repeat(sequence.joint_pos[-1:], 10, axis=0)
            q = np.repeat(sequence.body_quat[-1:], 10, axis=0)
        else:
            sampled = sequence.sample(unique)
            p, q = sampled.joint_pos[inverse], sampled.body_quat[inverse]
        v = np.stack([np.interp(clipped, sequence.times, self.velocities[:, j])
                      for j in range(29)], axis=-1)
        v[times > self.end_time] = 0
        self.current_ref, self.current_quat = p[0].copy(), q[0].copy()
        return p, v, q

    def tick(self):
        start = time.perf_counter()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise SimulationStop("nonfinite_state")
        quat = self.data.qpos[self.root_q + 3:self.root_q + 7].copy()
        upright = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
        if self.data.qpos[self.root_q + 2] < .35 or upright < .4:
            raise SimulationStop("fall")
        if np.max(np.abs(self.data.qvel[self.v_indices])) > 35:
            raise SimulationStop("joint_velocity_limit")
        self.policy.observe(self.data.qpos[self.q_indices].copy(),
                            self.data.qvel[self.v_indices].copy(), quat,
                            self.data.qvel[self.root_v + 3:self.root_v + 6].copy())
        p, v, q = self.lookahead()
        inference = time.perf_counter()
        target, action = self.policy.act(p, v, q, quat)
        self.latencies.append((time.perf_counter() - inference) * 1000)
        clipped = np.clip(target, self.ranges[:, 0], self.ranges[:, 1])
        self.limit_clamps += int(np.count_nonzero(clipped != target))
        target = clipped
        params = self.policy.parameters
        for _ in range(4):
            torque = params.kp * (target - self.data.qpos[self.q_indices]) - params.kd * self.data.qvel[self.v_indices]
            self.data.ctrl[self.actuators] = np.clip(torque,
                self.model.actuator_ctrlrange[self.actuators, 0], self.model.actuator_ctrlrange[self.actuators, 1])
            finger_torque = 4 * (self.finger_target - self.data.qpos[self.finger_q]) - .2 * self.data.qvel[self.finger_v]
            self.data.ctrl[self.finger_actuators] = np.clip(finger_torque,
                self.model.actuator_ctrlrange[self.finger_actuators, 0],
                self.model.actuator_ctrlrange[self.finger_actuators, 1])
            mujoco.mj_step(self.model, self.data)
        error = float(np.sqrt(np.mean((self.data.qpos[self.q_indices] - p[0]) ** 2)))
        self.tracking_errors.append(error)
        self.pose_history.append((float(self.data.time), self.body_pose()))
        self.trajectory.writerow([self.data.time, time.perf_counter() - self.started, self.frame_index] +
                                  self.data.qpos[self.root_q:self.root_q + 7].tolist() +
                                  self.data.qpos[self.q_indices].tolist() + p[0].tolist() +
                                  self.data.ctrl[self.actuators].tolist())
        self.frame_index += 1
        elapsed = time.perf_counter() - start
        self.deadline_misses += int(elapsed > .02)
        return elapsed

    def summary(self):
        return {"physics_executed": self.frame_index > 0, "sonic_executed": self.frame_index > 0,
                "task_success": None, "free_base": True, "elastic_support": False,
                "control_hz": 50, "physics_hz": 200,
                "sonic_g1_lookahead_seconds": self.policy.lookahead_seconds,
                "frames": self.frame_index, "simulation_seconds": float(self.data.time),
                "wall_seconds": time.perf_counter() - self.started,
                "control_deadline_misses": self.deadline_misses,
                "target_limit_clamps": self.limit_clamps,
                "tracking_joint_rmse_rad": float(np.mean(self.tracking_errors)) if self.tracking_errors else None,
                "sonic_latency_ms_p50": float(np.percentile(self.latencies, 50)) if self.latencies else None,
                "sonic_latency_ms_p95": float(np.percentile(self.latencies, 95)) if self.latencies else None,
                "sonic_source": self.policy.source_commit, "sonic_manifest_sha256": self.policy.asset_hash,
                "scene_sha256": sha256(self.scene)}

    def close(self):
        self.events.close()
        self.trajectory_file.close()
