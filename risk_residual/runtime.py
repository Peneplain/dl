"""Selective inference and adapter for the SHARED baseline executor."""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from baseline.adapters.reference import ReferenceSequence
from baseline.runtime import synchronize
from risk_residual.config import CONTEXT_DIM, NOMINAL_DIM, PHASES, STATE_DIM
from risk_residual.models import risk_features


@dataclass
class InferenceWindow:
    decision_time: float
    history_times: np.ndarray
    history: np.ndarray
    nominal_times: np.ndarray
    nominal: np.ndarray
    context: np.ndarray

    def validate(self, config):
        n, h = config.history_steps, config.horizon
        for value, shape in ((self.history_times, (n,)), (self.history, (n, STATE_DIM)),
                              (self.nominal_times, (h,)), (self.nominal, (h, NOMINAL_DIM)),
                              (self.context, (h, CONTEXT_DIM))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid inference input; expected {shape}")
        if (not np.isfinite(self.decision_time)
                or not np.isclose(self.history_times[-1], self.decision_time, atol=1e-6, rtol=0)
                or not np.allclose(np.diff(self.history_times), .02, atol=1e-6, rtol=0)
                or not np.isclose(self.nominal_times[0], self.decision_time, atol=1e-6, rtol=0)
                or not np.allclose(np.diff(self.nominal_times), .04, atol=1e-6, rtol=0)):
            raise ValueError("Past execution and future nominal clocks are not aligned")


class PredictiveController:
    """No teacher/labels at inference. Low risk does not call the residual at all."""
    def __init__(self, risk, residual, normalizer, *, threshold, interface="P", no_gate=False,
                 temperature=1.0):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Gate threshold must be a validation-selected probability")
        if interface not in {"B1", "B2", "I1", "I2", "I3", "I4", "P", "pooled"}:
            raise ValueError("Unknown interface")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Calibration temperature must be positive and finite")
        if risk.config != residual.config:
            raise ValueError("Risk and residual model configurations differ")
        self.risk = risk.eval().requires_grad_(False)
        self.residual = residual.eval().requires_grad_(False)
        self.normalizer = normalizer.eval()
        self.threshold, self.interface, self.no_gate = threshold, interface, no_gate
        self.temperature = float(temperature)
        self.config = risk.config

    @torch.inference_mode()
    def predict(self, window, *, reactive_trigger=None):
        controller_start = time.perf_counter()
        window.validate(self.config)
        device = next(self.risk.parameters()).device
        if next(self.residual.parameters()).device != device:
            raise ValueError("Risk and residual devices differ")
        batch = {key: torch.as_tensor(getattr(window, key), dtype=torch.float32,
                                      device=device)[None] for key in ("history", "nominal", "context")}
        inputs = self.normalizer(batch)
        synchronize(device)  # Exclude queued input preparation from model latency.
        probability, risk_ms, output = None, 0.0, None
        if self.interface not in {"B1", "B2"}:
            start = time.perf_counter()
            output = self.risk(**inputs)
            probability = float((output.intervention_logit / self.temperature).sigmoid().item())
            risk_ms = (time.perf_counter() - start) * 1000
            if not np.isfinite(probability) or not torch.isfinite(output.tokens).all():
                raise ValueError("Nonfinite risk output")
        if self.interface == "B2" and not isinstance(reactive_trigger, (bool, np.bool_)):
            raise ValueError("B2 requires the shared current-state threshold trigger")
        active = (self.no_gate or self.interface == "B1" or
                  (bool(reactive_trigger) if self.interface == "B2" else probability >= self.threshold))
        desired = np.zeros((self.config.horizon, 29), dtype=np.float32)
        residual_ms = 0.0
        if active:
            start = time.perf_counter()
            features = (inputs["history"].new_zeros((1, 8, 4, 32)) if output is None
                        else risk_features(output, self.interface))
            desired = self.residual(**inputs, features=features)[0].cpu().numpy()
            residual_ms = (time.perf_counter() - start) * 1000
            if not np.isfinite(desired).all():
                raise ValueError("Nonfinite residual output")
        return desired, {"probability": probability, "active": active,
                         "risk_ms": risk_ms, "residual_ms": residual_ms,
                         "controller_ms": (time.perf_counter() - controller_start) * 1000,
                         "temperature": self.temperature}


class SimulationHistoryBuilder:
    """Read causal simulation feedback using the exact rollout-converter layout.

    This observes the shared simulation; it does not advance physics or command
    a robot. Contact zeros are measured absence, never substitute sensor values.
    Future phase/finger context repeats the currently commanded values, matching
    converter windows that exclude changing context. No future executed states,
    evaluator outcomes, teacher references or labels are inputs.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.config.history_steps)

    def __call__(self, simulation, risk_reference):
        from scipy.spatial.transform import Rotation

        model, data = simulation.model, simulation.data
        now = float(data.time)
        if not np.isclose(now, risk_reference.times[0], atol=1e-6, rtol=0):
            raise ValueError("Observation and nominal clocks differ")
        if self.history:
            dt = now - self.history[-1][0]
            if dt <= 0:
                raise ValueError("Measured history clock must strictly increase")
            if not np.isclose(dt, .02, atol=1e-6, rtol=0):
                self.history.clear()
                simulation.event("risk_history_gap", gap_seconds=dt, behavior="zero_during_warmup")
        phase_name = simulation.task_phase.split("_replan_", 1)[0]
        if phase_name not in PHASES:
            raise ValueError("Unknown task phase for measured inference")
        phase = np.eye(len(PHASES), dtype=np.float32)[PHASES.index(phase_name)]
        block = int(model.joint("task_block_free").qposadr[0])
        block_geom = int(model.geom("task_block_geom").id)
        object_rotation = Rotation.from_quat(np.roll(data.qpos[block + 3:block + 7], -1))
        relative, contacts = [], np.zeros(2, dtype=np.float32)
        centers = (np.array([.125, -.035, 0.]), np.array([.125, .035, 0.]))
        for index, (side, center) in enumerate(zip(("left", "right"), centers)):
            body = int(model.body(f"{side}_wrist_yaw_link").id)
            rotation = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
            palm = data.xpos[body] + rotation.apply(center)
            xyz = rotation.inv().apply(data.qpos[block:block + 3] - palm)
            quat = np.roll((rotation.inv() * object_rotation).as_quat(), 1)
            relative.extend((*xyz, *quat))
            for contact in data.contact[:data.ncon]:
                if contact.dist > 0 or block_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == block_geom else contact.geom1
                name = model.body(int(model.geom_bodyid[other])).name or ""
                if name.startswith(f"{side}_hand_") or name == f"{side}_wrist_yaw_link":
                    contacts[index] = 1
                    break
        positions = data.qpos[simulation.q_indices]
        measured = np.concatenate((positions, data.qvel[simulation.v_indices],
                                   data.qpos[simulation.root_q + 3:simulation.root_q + 7],
                                   data.qvel[simulation.root_v + 3:simulation.root_v + 6],
                                   relative, contacts,
                                   np.abs(risk_reference.joint_pos[0] - positions), phase))
        fingers = np.asarray(simulation.finger_target)
        if (measured.shape != (STATE_DIM,) or fingers.shape != (14,)
                or not np.isfinite(measured).all() or not np.isfinite(fingers).all()):
            raise ValueError("Missing or nonfinite measured inference sensors")
        self.history.append((now, measured.astype(np.float32)))
        if len(self.history) < self.config.history_steps:
            return None
        nominal = np.concatenate((risk_reference.joint_pos, risk_reference.velocities(),
                                  risk_reference.body_quat), axis=-1)
        context = np.repeat(np.concatenate((phase, fingers))[None], self.config.horizon, axis=0)
        window = InferenceWindow(now, np.asarray([row[0] for row in self.history]),
                                 np.stack([row[1] for row in self.history]),
                                 risk_reference.times.copy(), nominal, context.astype(np.float32))
        window.validate(self.config)
        return window


class ReferenceCorrectionProvider:
    """Optional injection into SonicSimulation, with an explicit observation builder.

    builder(simulation, risk_reference) -> InferenceWindow or None during warmup.
    It must sample measured state on the 50 Hz history clock and provide actual
    object/contact state plus planned phase/finger commands. Missing task sensors
    must NOT be filled with invented zeros. No builder is assumed for run_live.
    """
    def __init__(self, controller, builder, *, update_hz=10, reactive=None):
        if not np.isfinite(update_hz) or not 10 <= update_hz <= 20:
            raise ValueError("Risk/residual updates must target 10–20 Hz")
        self.controller, self.builder, self.reactive = controller, builder, reactive
        self.period = 1 / update_hz
        self.reset()

    def reset(self):
        self.invalidate_plan()
        if hasattr(self.builder, "reset"):
            self.builder.reset()

    def invalidate_plan(self):
        self.last_update = None
        self.plan_times = None
        self.plan = None

    def request(self, simulation, nominal):
        now = float(nominal.times[0])
        # The builder is called EVERY 50 Hz tick so histories don't become 10 Hz histories.
        times = now + np.arange(self.controller.config.horizon) * .04
        # The training converter reconstructs its nominal input from SONIC's
        # logged sparse lookahead slots, then resamples on the Risk clock. Use
        # those same uncorrected slots rather than a denser, unseen trajectory.
        slots = getattr(simulation, "_nominal_lookahead", None)
        if slots is None:
            risk_reference = nominal.sample(times)
        else:
            positions, _, quaternions = slots
            slot_times = now + np.arange(len(positions)) * simulation.policy.future_step * .02
            risk_reference = ReferenceSequence(slot_times, positions, quaternions).sample(times)
        window = self.builder(simulation, risk_reference)
        if window is None:
            self.last_update = None
            self.plan = None
            return np.zeros_like(nominal.joint_pos)
        window.validate(self.controller.config)
        expected = np.concatenate((risk_reference.joint_pos, risk_reference.velocities(),
                                   risk_reference.body_quat), axis=-1)
        if (not np.isclose(window.decision_time, now, atol=1e-6, rtol=0)
                or not np.allclose(window.nominal, expected, atol=1e-5, rtol=0)):
            raise ValueError("Builder must use the supplied FUTURE NOMINAL reference")
        if self.last_update is None or now - self.last_update >= self.period - 1e-8:
            trigger = self.reactive(simulation) if self.reactive else None
            self.plan, report = self.controller.predict(window, reactive_trigger=trigger)
            self.plan_times = np.asarray(window.nominal_times).copy()
            self.last_update = now
            simulation.event("risk_residual", **report,
                             desired_offset_norm=float(np.linalg.norm(self.plan[0])),
                             history_start=float(window.history_times[0]),
                             history_end=float(window.history_times[-1]),
                             nominal_end=float(window.nominal_times[-1]),
                             schema_version=2)
        desired = np.stack([np.interp(nominal.times, self.plan_times, self.plan[:, j],
                                      left=0, right=0) for j in range(29)], axis=-1)
        # Shared ReferenceChecks handles the ramp at the gate/horizon boundaries.
        return desired
