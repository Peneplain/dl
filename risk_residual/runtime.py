"""Selective inference and adapter for the SHARED baseline executor."""

import time
from dataclasses import dataclass

import numpy as np
import torch

from risk_residual.config import CONTEXT_DIM, NOMINAL_DIM, STATE_DIM
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
    def __init__(self, risk, residual, normalizer, *, threshold, interface="P", no_gate=False):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Gate threshold must be a validation-selected probability")
        if interface not in {"B1", "B2", "I1", "I2", "I3", "I4", "P", "pooled"}:
            raise ValueError("Unknown interface")
        if risk.config != residual.config:
            raise ValueError("Risk and residual model configurations differ")
        self.risk = risk.eval().requires_grad_(False)
        self.residual = residual.eval().requires_grad_(False)
        self.normalizer = normalizer.eval()
        self.threshold, self.interface, self.no_gate = threshold, interface, no_gate
        self.config = risk.config

    @torch.inference_mode()
    def predict(self, window, *, reactive_trigger=None):
        window.validate(self.config)
        device = next(self.risk.parameters()).device
        if next(self.residual.parameters()).device != device:
            raise ValueError("Risk and residual devices differ")
        batch = {key: torch.as_tensor(getattr(window, key), dtype=torch.float32,
                                      device=device)[None] for key in ("history", "nominal", "context")}
        inputs = self.normalizer(batch)
        probability, risk_ms, output = None, 0.0, None
        if self.interface not in {"B1", "B2"}:
            start = time.perf_counter()
            output = self.risk(**inputs)
            probability = float(output.probability.item())  # synchronizes the score
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
                             "risk_ms": risk_ms, "residual_ms": residual_ms}


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
        risk_reference = nominal.sample(now + np.arange(8) * .04)
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
                             desired_offset_norm=float(np.linalg.norm(self.plan[0])))
        desired = np.stack([np.interp(nominal.times, self.plan_times, self.plan[:, j],
                                      left=0, right=0) for j in range(29)], axis=-1)
        # Shared ReferenceChecks handles the ramp at the gate/horizon boundaries.
        return desired
