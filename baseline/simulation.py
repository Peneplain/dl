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
from baseline.reference_checks import ReferenceChecks
from baseline.sonic_policy import SonicPolicy
from baseline.rollout import RolloutRecorder


class SimulationStop(RuntimeError):
    pass


class SonicSimulation:
    def __init__(self, assets, repo, output, threads=2, policy=None,
                 gui=False, scene=None, initial_body_reference=None, finger_kp=6., finger_kd=.4,
                 correction_provider=None):
        if not np.isfinite([finger_kp, finger_kd]).all() or finger_kp <= 0 or finger_kd < 0:
            raise ValueError("Finger gains must be finite, with positive stiffness and nonnegative damping")
        self.finger_kp, self.finger_kd = float(finger_kp), float(finger_kd)
        self.policy = policy if policy is not None else SonicPolicy(assets, repo, threads)
        self.scene = Path(scene) if scene is not None else repo / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.model.opt.timestep = .005
        self.data = mujoco.MjData(self.model)
        self.output = None
        self.events = None
        self.trajectory_file = None
        self.trajectory = None
        self.context_file = None
        self.context = None
        self.effective_context_file = None
        self.effective_context = None
        self.task_phase = "stand"
        self.recording = None
        self.viewer = None
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if root < 0 or self.model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The robot must have a free base")
        self.root_q = int(self.model.jnt_qposadr[root])
        self.root_v = int(self.model.jnt_dofadr[root])
        joints = [self.model.joint(n).id for n in ISAACLAB_JOINT_NAMES]
        self.q_indices = self.model.jnt_qposadr[joints]
        self.v_indices = self.model.jnt_dofadr[joints]
        self.ranges = self.model.jnt_range[joints].copy()
        self.correction_provider = correction_provider
        self.reference_checks = ReferenceChecks(self.ranges) if correction_provider is not None else None
        self.initial_body_reference = np.array(
            self.policy.parameters.default if initial_body_reference is None else initial_body_reference,
            dtype=float, copy=True)
        if (self.initial_body_reference.shape != (29,)
                or not np.isfinite(self.initial_body_reference).all()
                or (self.initial_body_reference < self.ranges[:, 0]).any()
                or (self.initial_body_reference > self.ranges[:, 1]).any()):
            raise ValueError("Initial body reference must contain 29 finite, bounded joint angles")
        by_joint = {int(self.model.actuator_trnid[i, 0]): i for i in range(self.model.nu)}
        self.actuators = np.array([by_joint[j] for j in joints])
        if not np.allclose(self.model.actuator_gear[self.actuators, 0], 1):
            raise ValueError("Expected direct joint torque actuators")
        body_joints = set(joints)
        fingers = [j for j in range(self.model.njnt)
                   if "_hand_" in (self.model.joint(j).name or "") and j in by_joint]
        self.finger_q = self.model.jnt_qposadr[fingers]
        self.finger_v = self.model.jnt_dofadr[fingers]
        self.finger_names = tuple(self.model.joint(j).name for j in fingers)
        self.finger_ranges = self.model.jnt_range[fingers].copy()
        self.finger_actuators = np.array([by_joint[j] for j in fingers])
        if len(fingers) != 14 or len(body_joints) != 29:
            raise ValueError("Expected 29 body joints and 14 articulated finger joints")
        self.finger_open = np.clip(np.zeros(14), self.finger_ranges[:, 0],
                                   self.finger_ranges[:, 1])
        self.finger_target = self.finger_open.copy()
        self.started = time.perf_counter()
        self.latencies = []
        self.tracking_errors = []
        self.deadline_misses = 0
        self.limit_clamps = 0
        self.risk_decisions = 0
        self.risk_activations = 0
        self.risk_probabilities = []
        self.risk_correction_norms = []
        self.risk_latencies = []
        self.residual_latencies = []
        self.controller_latencies = []
        self.risk_history_gaps = 0
        self.reset_count = 0
        self.frame_index = 0
        # Set only after ReferenceChecks accepts a nonzero offset that is
        # actually sampled into the SONIC reference slots.
        self.learned_correction_used = False
        if output is None:
            self.started = time.perf_counter()
            self.reset()
        else:
            self.start_run(output)
        if gui:
            try:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
                self.viewer.sync()
            except Exception as error:
                self.close()
                raise RuntimeError(
                    "MuJoCo GUI could not connect to the virtual Xorg/GLX display. "
                    "Check the VNC startup logs and GUI image before retrying: "
                    f"{type(error).__name__}: {error}") from error

    def event(self, event, **fields):
        if event == "risk_residual":
            self.risk_decisions += 1
            if fields.get("active"):
                self.risk_activations += 1
            for key, target in (("probability", self.risk_probabilities),
                                ("desired_offset_norm", self.risk_correction_norms),
                                ("risk_ms", self.risk_latencies),
                                ("residual_ms", self.residual_latencies),
                                ("controller_ms", self.controller_latencies)):
                value = fields.get(key)
                if value is not None and np.isfinite(value):
                    target.append(float(value))
        elif event == "risk_history_gap":
            self.risk_history_gaps += 1
        if self.events is None:
            return
        import json
        self.events.write(json.dumps({"event": event, "sim_time": float(self.data.time),
                                     "wall_time": time.perf_counter() - self.started,
                                     **fields}, allow_nan=False) + "\n")

    def start_run(self, output):
        if self.output is not None:
            raise RuntimeError("A simulation output run is already active")
        self.output = Path(output)
        if not self.output.is_dir():
            raise FileNotFoundError(f"Run output directory does not exist: {self.output}")
        self.events = (self.output / "events.jsonl").open("x", buffering=1)
        self.trajectory_file = (self.output / "trajectory.csv").open("x", buffering=1)
        self.trajectory = csv.writer(self.trajectory_file)
        self.trajectory.writerow(["sim_time", "wall_time", "frame_index", "root_x", "root_y", "root_z",
                                  "root_qw", "root_qx", "root_qy", "root_qz"] +
                                 [f"q:{n}" for n in ISAACLAB_JOINT_NAMES] +
                                 [f"ref:{n}" for n in ISAACLAB_JOINT_NAMES] +
                                 [f"torque:{n}" for n in ISAACLAB_JOINT_NAMES])
        self.context_file = (self.output / "nominal_context.csv").open("x", buffering=1)
        self.context = csv.writer(self.context_file)
        context_header = ["sim_time", "wall_time", "frame_index", "phase",
                          "root_x", "root_y", "root_z", "root_qw", "root_qx",
                          "root_qy", "root_qz", "root_vx", "root_vy", "root_vz",
                          "root_wx", "root_wy", "root_wz"]
        context_header += [f"state_q:{name}" for name in ISAACLAB_JOINT_NAMES]
        context_header += [f"state_dq:{name}" for name in ISAACLAB_JOINT_NAMES]
        context_header += [f"finger_ref:{name}" for name in self.finger_names]
        context_header += [f"nominal_time_offset:{i:02d}" for i in range(self.policy.future_count)]
        context_header += [f"nominal_pos:{i:02d}:{name}"
                           for i in range(self.policy.future_count)
                           for name in ISAACLAB_JOINT_NAMES]
        context_header += [f"nominal_vel:{i:02d}:{name}"
                           for i in range(self.policy.future_count)
                           for name in ISAACLAB_JOINT_NAMES]
        context_header += [f"nominal_quat:{i:02d}:{axis}" for i in range(self.policy.future_count)
                           for axis in ("w", "x", "y", "z")]
        self.context.writerow(context_header)
        if self.correction_provider is not None:
            self.effective_context_file = (self.output / "effective_context.csv").open("x", buffering=1)
            self.effective_context = csv.writer(self.effective_context_file)
            self.effective_context.writerow(context_header)
        self.started = time.perf_counter()
        self.latencies = []
        self.tracking_errors = []
        self.deadline_misses = 0
        self.limit_clamps = 0
        self.reset_count = 0
        self.frame_index = 0
        self.reset()
        self.recording = RolloutRecorder(self.model, self.output, scene=self.scene)
        self.recording.record(self.data, -1, time.perf_counter() - self.started)

    def end_run(self):
        if self.recording is not None:
            self.recording.close()
            self.recording = None
        if self.events is not None:
            self.events.close()
            self.events = None
        if self.trajectory_file is not None:
            self.trajectory_file.close()
            self.trajectory_file = None
            self.trajectory = None
        if self.context_file is not None:
            self.context_file.close()
            self.context_file = None
            self.context = None
        if self.effective_context_file is not None:
            self.effective_context_file.close()
            self.effective_context_file = None
            self.effective_context = None
        self.output = None

    def sync_viewer(self):
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            raise SimulationStop("viewer_closed")
        self.viewer.sync()

    def reset(self):
        if self.recording is not None:
            raise RuntimeError("End the recorded episode before resetting simulation time")
        mujoco.mj_resetData(self.model, self.data)
        self.finger_target = self.finger_open.copy()
        self.data.qpos[self.q_indices] = self.initial_body_reference
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
        if self.reference_checks is not None:
            self.reference_checks.reset()
            self.correction_provider.reset()
        self.buffer = ReferenceBuffer(max_gap=.021)
        self.current_ref = self.initial_body_reference.copy()
        self.current_quat = self.data.qpos[self.root_q + 3:self.root_q + 7].copy()
        self.end_time = 0.0
        self.holding = True
        self.terminal_hold_logged = False
        self.learned_correction_used = False
        self.risk_decisions = 0
        self.risk_activations = 0
        self.risk_probabilities = []
        self.risk_correction_norms = []
        self.risk_latencies = []
        self.residual_latencies = []
        self.controller_latencies = []
        self.risk_history_gaps = 0
        self.task_phase = "stand"
        self.pose_history = deque(maxlen=200)
        self.reset_count += 1
        self.event("reset", reset_index=self.reset_count)

    def body_pose(self):
        return np.concatenate([self.data.qpos[self.root_q:self.root_q + 7],
                               self.data.qpos[self.q_indices]])

    def history_qpos(self, count=16):
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

    @staticmethod
    def load_reference(path):
        """Load an ARDY-exported 50 Hz reference with its explicit schema."""
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(x) for x in data["joint_names"].tolist())
            if names != tuple(ISAACLAB_JOINT_NAMES):
                raise ValueError("Reference joint_names must be the verified ISAACLAB order")
            if float(data["fps"]) != 50:
                raise ValueError("SONIC references must be sampled at 50 Hz")
            reference = ReferenceSequence(data["times"], data["joint_pos"], data["body_quat"])
        if not np.allclose(np.diff(reference.times), .02, atol=1e-6, rtol=0):
            raise ValueError("Reference timestamps must match the declared 50 Hz rate")
        return reference

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
        self.end_time = float(sequence.times[-1])
        self.holding = False
        self.terminal_hold_logged = False
        self.policy.align(self.data.qpos[self.root_q + 3:self.root_q + 7], sequence.body_quat[0])
        if getattr(self, "correction_provider", None) is not None:
            self.correction_provider.invalidate_plan()
        self.event("reference_installed", frames=len(sequence.times), end_time=self.end_time,
                   transition_seconds=n * .02)

    def hold_measured_pose(self, reason, transition=.3):
        """End a task segment with a checked pose hold through frozen SONIC.

        Only the reference buffer changes. Physics, free joints, and policy
        history continue; no executed pose or object state is overwritten.
        """
        pose = self.body_pose()
        reference = ReferenceSequence([0., .02], np.repeat(pose[None, 7:], 2, axis=0),
                                      np.repeat(pose[None, 3:7], 2, axis=0))
        self.install(reference, transition=transition)
        self.event("measured_pose_hold", reason=reason, body_qpos=pose.tolist(),
                   transition_seconds=float(transition))

    def lookahead(self):
        count = self.policy.future_count
        times = self.data.time + np.arange(count) * self.policy.future_step * .02
        sequence = self.buffer.sequence
        if sequence is None or self.data.time > self.end_time + 1e-8:
            if sequence is not None:
                # The final control tick can land just beyond the clip endpoint.
                # Hold the actual endpoint, not the previous tick's reference.
                self.current_ref = sequence.joint_pos[-1].copy()
                self.current_quat = sequence.body_quat[-1].copy()
            if not self.holding:
                self.buffer.underruns += 1
                self.holding = True
                self.event("buffer_underrun", behavior="hold_terminal_nominal_pose")
            p = np.repeat(self.current_ref[None], count, axis=0)
            q = np.repeat(self.current_quat[None], count, axis=0)
            v = np.zeros((count, 29))
        else:
            # Future samples outside a completed finite clip hold its endpoint.
            if times[-1] > self.end_time + 1e-8 and not self.terminal_hold_logged:
                self.event("reference_terminal_hold", buffered_end=self.end_time,
                           requested_end=float(times[-1]), behavior="hold_terminal_nominal_pose")
                self.terminal_hold_logged = True
            sampled = sequence.sample_with_terminal_hold(times)
            p, q = sampled.joint_pos, sampled.body_quat
            v = sampled.velocities()
            v[times >= self.end_time - 1e-8] = 0
        self.current_ref, self.current_quat = p[0].copy(), q[0].copy()
        self._nominal_lookahead = (p, v, q)
        if getattr(self, "correction_provider", None) is None:
            return p, v, q
        dense_count = (count - 1) * self.policy.future_step + 1
        dense_times = self.data.time + np.arange(dense_count) * .02
        if sequence is None or self.data.time > self.end_time + 1e-8:
            dense = ReferenceSequence(dense_times, np.repeat(p[:1], dense_count, axis=0),
                                      np.repeat(q[:1], dense_count, axis=0))
        else:
            dense = sequence.sample_with_terminal_hold(dense_times)
        desired = self.correction_provider.request(self, dense)
        checked = self.reference_checks.apply(dense, desired)
        slots = np.arange(count) * self.policy.future_step
        # Preserve B0's nominal velocity convention exactly. Re-differentiating
        # the entire dense reference changes sparse SONIC velocities even for
        # a zero correction. Only the checked correction contributes an extra
        # derivative, including the shared ramp and joint-limit clipping.
        offset = checked.joint_pos - dense.joint_pos
        # Only count a correction once it survives ReferenceChecks and is
        # sampled into the references passed to SONIC. A requested offset that
        # is clipped to zero, or one that exists only outside the sampled
        # control slots, must not be reported as executed.
        applied_offset = offset[slots]
        if np.any(np.abs(applied_offset) > 1e-8):
            self.learned_correction_used = True
        offset_velocity = np.gradient(offset, dense.times, axis=0).astype(np.float32)
        return checked.joint_pos[slots], v + offset_velocity[slots], checked.body_quat[slots]

    def _write_nominal_context(self, positions, velocities, quaternions):
        """Persist the causal P input tuple at the SONIC control clock.

        Object/contact labels stay in ``task.csv`` so the baseline remains
        model-independent; both files share ``sim_time`` and ``frame_index``.
        Future executed states are never written as nominal inputs.
        """
        self._write_context(self.context, positions, velocities, quaternions)

    def _write_context(self, writer, positions, velocities, quaternions):
        if writer is None:
            return
        root = self.data.qpos[self.root_q:self.root_q + 7]
        root_velocity = self.data.qvel[self.root_v:self.root_v + 6]
        offsets = np.arange(self.policy.future_count, dtype=float) * self.policy.future_step * .02
        row = [self.data.time, time.perf_counter() - self.started, self.frame_index,
               self.task_phase, *root, *root_velocity,
               *self.data.qpos[self.q_indices], *self.data.qvel[self.v_indices],
               *self.finger_target, *offsets, *positions.reshape(-1),
               *velocities.reshape(-1), *quaternions.reshape(-1)]
        writer.writerow(row)

    def command_fingers(self, targets, max_rate=2.5):
        """Named hand commands under a shared radians/second rate limit."""
        if not np.isfinite(max_rate) or max_rate <= 0:
            raise ValueError("Finger rate must be positive and finite")
        requested = self.finger_target.copy()
        for name, value in targets.items():
            index = self.finger_names.index(name)
            if not np.isfinite(value) or not self.finger_ranges[index, 0] <= value <= self.finger_ranges[index, 1]:
                raise ValueError(f"Invalid finger target for {name}: {value}")
            requested[index] = value
        self.finger_target += np.clip(requested - self.finger_target, -max_rate * .02,
                                      max_rate * .02)

    def tick(self, after_step=None):
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
        self._write_nominal_context(*getattr(self, "_nominal_lookahead", (p, v, q)))
        self._write_context(self.effective_context, p, v, q)
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
            finger_torque = (self.finger_kp * (self.finger_target - self.data.qpos[self.finger_q])
                             - self.finger_kd * self.data.qvel[self.finger_v])
            self.data.ctrl[self.finger_actuators] = np.clip(finger_torque,
                self.model.actuator_ctrlrange[self.finger_actuators, 0],
                self.model.actuator_ctrlrange[self.finger_actuators, 1])
            mujoco.mj_step(self.model, self.data)
            if after_step is not None:
                after_step(self)
        error = float(np.sqrt(np.mean((self.data.qpos[self.q_indices] - p[0]) ** 2)))
        self.tracking_errors.append(error)
        self.pose_history.append((float(self.data.time), self.body_pose()))
        if self.trajectory is not None:
            self.trajectory.writerow([self.data.time, time.perf_counter() - self.started,
                                      self.frame_index] +
                                     self.data.qpos[self.root_q:self.root_q + 7].tolist() +
                                     self.data.qpos[self.q_indices].tolist() + p[0].tolist() +
                                     self.data.ctrl[self.actuators].tolist())
        if self.recording is not None:
            self.recording.record(self.data, self.frame_index, time.perf_counter() - self.started)
        self.frame_index += 1
        self.sync_viewer()
        elapsed = time.perf_counter() - start
        self.deadline_misses += int(elapsed > .02)
        return elapsed

    def summary(self):
        return {"physics_executed": self.frame_index > 0, "sonic_executed": self.frame_index > 0,
                "task_success": None, "free_base": True, "elastic_support": False,
                "control_hz": 50, "physics_hz": 200,
                "finger_kp_nm_per_rad": self.finger_kp, "finger_kd_nm_s_per_rad": self.finger_kd,
                "sonic_g1_lookahead_seconds": self.policy.lookahead_seconds,
                "frames": self.frame_index, "simulation_seconds": float(self.data.time),
                "wall_seconds": time.perf_counter() - self.started,
                "control_deadline_misses": self.deadline_misses,
                "buffer_underruns": self.buffer.underruns,
                "target_limit_clamps": self.limit_clamps,
                "tracking_joint_rmse_rad": float(np.mean(self.tracking_errors)) if self.tracking_errors else None,
                "sonic_latency_ms_p50": float(np.percentile(self.latencies, 50)) if self.latencies else None,
                "sonic_latency_ms_p95": float(np.percentile(self.latencies, 95)) if self.latencies else None,
                "sonic_source": self.policy.source_commit, "sonic_manifest_sha256": self.policy.asset_hash,
                "scene_sha256": sha256(self.scene),
                "rollout": self.recording.summary() if self.recording else None,
                "learned_correction_used": bool(getattr(self, "learned_correction_used", False)),
                "risk_decisions": int(self.risk_decisions),
                "risk_activations": int(self.risk_activations),
                "risk_activation_rate": (self.risk_activations / self.risk_decisions
                                          if self.risk_decisions else None),
                "risk_history_gaps": int(self.risk_history_gaps),
                "risk_probability_p50": float(np.percentile(self.risk_probabilities, 50)) if self.risk_probabilities else None,
                "risk_probability_p95": float(np.percentile(self.risk_probabilities, 95)) if self.risk_probabilities else None,
                "correction_norm_p50": float(np.percentile(self.risk_correction_norms, 50)) if self.risk_correction_norms else None,
                "correction_norm_p95": float(np.percentile(self.risk_correction_norms, 95)) if self.risk_correction_norms else None,
                "risk_latency_ms_p50": float(np.percentile(self.risk_latencies, 50)) if self.risk_latencies else None,
                "risk_latency_ms_p95": float(np.percentile(self.risk_latencies, 95)) if self.risk_latencies else None,
                "residual_latency_ms_p50": float(np.percentile(self.residual_latencies, 50)) if self.residual_latencies else None,
                "residual_latency_ms_p95": float(np.percentile(self.residual_latencies, 95)) if self.residual_latencies else None,
                "controller_latency_ms_p50": float(np.percentile(self.controller_latencies, 50)) if self.controller_latencies else None,
                "controller_latency_ms_p95": float(np.percentile(self.controller_latencies, 95)) if self.controller_latencies else None,
                "nominal_context": "nominal_context.csv" if self.output is not None else None,
                "video": None}

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            finally:
                self.viewer = None
        self.end_run()
