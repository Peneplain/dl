"""Causal input equivalence and shared-executor pilot contracts, without GPU jobs."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceBuffer, ReferenceSequence
from baseline.rollout import RolloutRecorder
from baseline.reference_checks import ReferenceChecks
from baseline.simulation import SonicSimulation
from experiments.build_dataset import measured_features, planned
from experiments.run_policy import correction_provider, prepare_plan, run
from risk_residual.config import ModelConfig
from risk_residual.runtime import ReferenceCorrectionProvider, SimulationHistoryBuilder


def simulation_fixture():
    names = list(ISAACLAB_JOINT_NAMES)
    bodies = {"left_wrist_yaw_link": 0, "right_wrist_yaw_link": 1,
              "right_hand_index_0_link": 2}

    def joint(name):
        if name == "task_block_free":
            return SimpleNamespace(qposadr=np.array([36]))
        return SimpleNamespace(id=0 if name == "floating_base_joint" else 1 + names.index(name))

    def body(key):
        return SimpleNamespace(id=bodies[key]) if isinstance(key, str) else SimpleNamespace(
            name=next(name for name, index in bodies.items() if index == key))

    model = SimpleNamespace(joint=joint, body=body,
                            geom=lambda name: SimpleNamespace(id=0),
                            geom_bodyid=np.array([0, 2]),
                            jnt_qposadr=np.r_[0, np.arange(7, 36)],
                            jnt_dofadr=np.r_[0, np.arange(6, 35)])
    data = SimpleNamespace(time=0., qpos=np.zeros(43), qvel=np.zeros(35),
                           xpos=np.array([[0., 0., 1.], [0., -.2, 1.], [0., 0., 0.]]),
                           xmat=np.tile(np.eye(3).reshape(1, 9), (3, 1)),
                           contact=[SimpleNamespace(geom1=0, geom2=1, dist=-.001)], ncon=1)
    data.qpos[2:7] = [1., 1., 0., 0., 0.]
    data.qpos[36:43] = [.2, -.2, 1., 1., 0., 0., 0.]
    return SimpleNamespace(model=model, data=data, q_indices=np.arange(7, 36),
                           v_indices=np.arange(6, 35), root_q=0, root_v=0,
                           task_phase="lower_replan_01", finger_target=np.arange(14) * .01,
                           policy=SimpleNamespace(future_step=5),
                           events=[], event=lambda *args, **kwargs: None)


def reference(now, dt=.04, count=8):
    times = now + np.arange(count) * dt
    positions = np.repeat((times * times)[:, None], 29, axis=1)
    return ReferenceSequence(times, positions, np.tile([1., 0., 0., 0.], (count, 1)))


class ControllerFixture:
    config = ModelConfig(width=32, layers=1, heads=4)

    def __init__(self):
        self.windows = []

    def predict(self, window, **kwargs):
        self.windows.append(window)
        plan = np.zeros((8, 29))
        plan[:, 15] = np.arange(8) * .01 + .02
        return plan, {"probability": .9, "active": True, "risk_ms": 1., "residual_ms": 1.}


class PolicyRuntimeTests(unittest.TestCase):
    def test_zero_provider_preserves_every_nominal_sonic_input(self):
        times = np.arange(61) * .02
        nonlinear = ReferenceSequence(times, np.repeat(np.sin(times * 4)[:, None] * .1, 29, axis=1),
                                      np.tile([1., 0., 0., 0.], (len(times), 1)))
        def simulation(provider):
            sim = SonicSimulation.__new__(SonicSimulation)
            sim.data = SimpleNamespace(time=0.)
            sim.policy = SimpleNamespace(future_count=10, future_step=5)
            sim.buffer = ReferenceBuffer()
            sim.buffer.push(nonlinear)
            sim.end_time, sim.holding, sim.terminal_hold_logged = 1.2, False, False
            sim.current_ref, sim.current_quat = nonlinear.joint_pos[0], nonlinear.body_quat[0]
            sim.event = lambda *a, **k: None
            sim.correction_provider = provider
            sim.reference_checks = ReferenceChecks(np.tile([-3., 3.], (29, 1)))
            sim.finger_target = np.arange(14) * .01
            return sim
        zero = SimpleNamespace(request=lambda sim, nominal: np.zeros_like(nominal.joint_pos))
        b0, p = simulation(None), simulation(zero)
        for now in (0., .02, .30, .92, 1.22):
            b0.data.time = p.data.time = now
            nominal, corrected = b0.lookahead(), p.lookahead()
            for original, effective in zip(nominal, corrected):
                np.testing.assert_array_equal(original, effective)
            np.testing.assert_array_equal(b0.finger_target, p.finger_target)
            self.assertEqual(b0.data.time, p.data.time)
            self.assertEqual(b0.buffer.underruns, p.buffer.underruns)

    def test_executed_trial_survives_diagnostic_failure_and_exception_status(self):
        with tempfile.TemporaryDirectory() as directory:
            request = {"index": 1, "seed": 42000}
            plan = {"baseline_plan": {"requests": [request], "plan_sha256": "baseline"},
                    "plan_sha256": "experiment", "paired_request_sha256": "requests",
                    "controller_artifacts": {}}
            args = SimpleNamespace(method="B0", out=Path(directory) / "run")
            def execute(request, output, digest):
                (output / "events.jsonl").write_text("malformed JSON\n")
                return {"status": "stopped", "task_success": False, "physics_executed": True}
            with patch("experiments.run_policy.prepare_plan", return_value=(SimpleNamespace(), plan)), \
                    patch("baseline.execution.ExecutionRuntime") as factory:
                factory.return_value.run.side_effect = execute
                summary = run(args)
                self.assertEqual(summary["completed"], 1)
                self.assertEqual(summary["success_rate"], 0)
                self.assertIn("diagnostic_error", summary["attempts"][0])
            args.out = Path(directory) / "startup-error"
            with patch("experiments.run_policy.prepare_plan", return_value=(SimpleNamespace(), plan)), \
                    patch("baseline.execution.ExecutionRuntime") as factory:
                factory.return_value.preload.side_effect = RuntimeError("operator failure fixture")
                with self.assertRaises(RuntimeError):
                    run(args)
                saved = json.loads((args.out / "summary.json").read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertIn("operator failure fixture", saved["error"])

    @unittest.skipUnless(hasattr(mujoco, "MjModel"), "Requires real MuJoCo for compiled-state parity")
    def test_real_compiled_mujoco_replay_matches_live_feedback(self):
        # A sensor fixture is enough to verify MuJoCo state/FK/clock parity.
        # This is not a G1/SONIC model check or physical grasp result.
        bodies = []
        for index, name in enumerate(ISAACLAB_JOINT_NAMES):
            body = ("left_wrist_yaw_link" if index == 0 else
                    "right_wrist_yaw_link" if index == 1 else f"fixture_body_{index}")
            bodies.append(f'<body name="{body}" pos="0 {index * .03} 0">'
                          f'<joint name="{name}" axis="0 1 0"/>'
                          '<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/>'
                          '</body>')
        xml = ('<mujoco><option timestep=".005" gravity="0 0 0"/>'
               '<worldbody><body name="root" pos="0 0 1"><freejoint name="floating_base_joint"/>'
               '<geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>'
               + ''.join(bodies) + '</body><body name="task_block" pos=".2 -.2 1">'
               '<freejoint name="task_block_free"/><geom name="task_block_geom" type="box" '
               'size=".03 .03 .03" mass=".08" contype="0" conaffinity="0"/>'
               '</body></worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        joints = [int(model.joint(name).id) for name in ISAACLAB_JOINT_NAMES]
        root = int(model.joint("floating_base_joint").id)
        simulation = SimpleNamespace(model=model, data=data, q_indices=model.jnt_qposadr[joints],
                                     v_indices=model.jnt_dofadr[joints], root_q=int(model.jnt_qposadr[root]),
                                     root_v=int(model.jnt_dofadr[root]), task_phase="lower",
                                     finger_target=np.arange(14) * .01, event=lambda *a, **k: None)
        data.qvel[3:6] = [.01, .02, .03]
        builder = SimulationHistoryBuilder(ModelConfig())
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            recorder = RolloutRecorder(model, Path(directory))
            for frame in range(16):
                if frame:
                    for _ in range(4):
                        mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                recorder.record(data, frame - 1, float(data.time))
                slots = reference(float(data.time), .1, 10)
                window = builder(simulation, slots.sample(float(data.time) + np.arange(8) * .04))
                rows.append({"frame": frame, "time": float(data.time),
                             "q": data.qpos[simulation.q_indices].copy(),
                             "dq": data.qvel[simulation.v_indices].copy(),
                             "root": data.qpos[3:7].copy(), "gyro": data.qvel[3:6].copy(),
                             "root_z": float(data.qpos[2]), "phase": "lower", "pos": slots.joint_pos,
                             "quat": slots.body_quat, "slot_times": np.arange(10) * .1})
            recorder.close()
            offline, _ = measured_features(Path(directory), rows)
        np.testing.assert_array_equal(window.history, np.stack([row[0] for row in offline]))
        np.testing.assert_array_equal(window.nominal.astype(np.float32),
                                      planned(rows[-1], window.nominal_times).astype(np.float32))

    def test_online_features_exactly_match_converter_replay(self):
        simulation = simulation_fixture()
        builder = SimulationHistoryBuilder(ModelConfig())
        rows, snapshots = [], []
        for frame in range(16):
            now = frame * .02
            simulation.data.time = now
            simulation.data.qpos[simulation.q_indices] = now * .2
            simulation.data.qvel[simulation.v_indices] = .2
            simulation.data.qvel[3:6] = [.01, .02, .03]
            slots = reference(now, .1, 10)
            future = slots.sample(now + np.arange(8) * .04)
            window = builder(simulation, future)
            rows.append({"frame": frame, "time": now, "q": simulation.data.qpos[simulation.q_indices].copy(),
                         "dq": simulation.data.qvel[simulation.v_indices].copy(), "root": np.array([1., 0., 0., 0.]),
                         "gyro": simulation.data.qvel[3:6].copy(), "root_z": 1., "phase": "lower",
                         "pos": slots.joint_pos, "quat": slots.body_quat,
                         "slot_times": np.arange(10) * .1})
            snapshots.append(deepcopy(simulation.data))
        saved = SimpleNamespace(model=simulation.model, metadata={"model_sha256": "fixture"},
                                frame_index=np.arange(-1, 15), states=np.arange(16)[:, None],
                                restore=lambda index: snapshots[index])
        with patch("experiments.build_dataset.SavedRollout", return_value=saved):
            offline, _ = measured_features(Path("unused-fixture"), rows)
        np.testing.assert_array_equal(window.history, np.stack([row[0] for row in offline]))
        np.testing.assert_allclose(window.nominal, planned(rows[-1], window.nominal_times))
        self.assertEqual(window.history.shape, (16, 119))
        # Measured right-hand contact remains 1; missing left contact remains 0.
        np.testing.assert_array_equal(window.history[-1, 79:81], [0., 1.])
        np.testing.assert_array_equal(window.context[:, 9:],
                                      np.tile(simulation.finger_target.astype(np.float32), (8, 1)))

    def test_warmup_gap_and_reset_never_fabricate_history(self):
        simulation = simulation_fixture()
        builder = SimulationHistoryBuilder(ModelConfig())
        for frame in range(15):
            simulation.data.time = frame * .02
            self.assertIsNone(builder(simulation, reference(simulation.data.time)))
        simulation.data.time = .30
        self.assertIsNotNone(builder(simulation, reference(.30)))
        simulation.data.time = .34  # Missing the .32 observation restarts warmup.
        self.assertIsNone(builder(simulation, reference(.34)))
        self.assertEqual(len(builder.history), 1)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            builder(simulation, reference(.34))
        builder.reset()
        self.assertEqual(len(builder.history), 0)

    def test_provider_uses_logged_sparse_nominal_and_updates_receding_horizon(self):
        simulation = simulation_fixture()
        controller = ControllerFixture()
        provider = ReferenceCorrectionProvider(controller, SimulationHistoryBuilder(controller.config))
        for frame in range(21):
            now = frame * .02
            simulation.data.time = now
            slots = reference(now, .1, 10)
            simulation._nominal_lookahead = (slots.joint_pos, slots.velocities(), slots.body_quat)
            desired = provider.request(simulation, reference(now, .02, 46))
            if frame < 15:
                np.testing.assert_array_equal(desired, 0)
            if frame == 15:
                self.assertAlmostEqual(desired[0, 15], .02)
            if frame == 16:
                self.assertAlmostEqual(desired[0, 15], .025)
        self.assertEqual(len(controller.windows), 2)  # .30/.40 decisions, 50 Hz observations.
        last = controller.windows[-1]
        np.testing.assert_allclose(last.nominal, planned({"time": .4, "slot_times": np.arange(10) * .1,
                                                        "pos": slots.joint_pos, "quat": slots.body_quat},
                                                       last.nominal_times))
        np.testing.assert_allclose(last.history_times, np.arange(5, 21) * .02)
        self.assertAlmostEqual(simulation.data.time, .4)  # Observation never advances physics.
        provider.reset()
        self.assertEqual(len(provider.builder.history), 0)

    def test_b0_requires_no_learning_controller_and_paired_plan_is_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "requests.json"
            requests.write_text(json.dumps([{"seed": 41000, "cube_xy": [.4, -.2]}]))
            files = [root / name for name in ("risk.pt", "residual.pt", "gate.json")]
            for file in files:
                file.write_text("hash-only fixture; never loaded")
            args = SimpleNamespace(method="B0", requests=requests, risk=None, residual=None,
                                   gate=None, baseline_options=["--", "--ardy", "--device", "cpu",
                                                                "--hold-seconds", "10"],
                                   learning_device="cpu", update_hz=10)
            b0, plan0 = prepare_plan(args)
            with patch("risk_residual.checkpoints.load_controller", side_effect=AssertionError("must not load")):
                self.assertIsNone(correction_provider(args))
            args.method, args.risk, args.residual, args.gate = "P", *files
            p, planp = prepare_plan(args)
            self.assertEqual(plan0["baseline_plan"], planp["baseline_plan"])
            self.assertEqual(plan0["paired_request_sha256"], planp["paired_request_sha256"])
            self.assertEqual(vars(b0), vars(p))
            self.assertEqual(planp["evaluation_scope"], "pilot")
            self.assertFalse(planp["teacher_used"])

    def test_paired_plan_rejects_generator_and_controller_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "requests.json"
            requests.write_text(json.dumps([{"seed": 41000, "cube_xy": [.4, -.2]}]))
            files = [root / name for name in ("risk.pt", "residual.pt", "gate.json")]
            for file in files:
                file.write_text("hash-only fixture; never loaded")
            for method in ("B0", "P"):
                args = SimpleNamespace(method=method, requests=requests,
                                       risk=files[0] if method == "P" else None,
                                       residual=files[1] if method == "P" else None,
                                       gate=files[2] if method == "P" else None,
                                       baseline_options=["--", "--kimodo"],
                                       learning_device="cpu", update_hz=10)
                with self.subTest(method=method, selector="--kimodo"), \
                        self.assertRaisesRegex(ValueError, "requires ARDY"):
                    prepare_plan(args)
                for option, value in (("--risk-checkpoint", files[0]), ("--risk", files[0]),
                                      ("--residual-checkpoint", files[1]), ("--residual", files[1]),
                                      ("--risk-gate", files[2]), ("--gate", files[2]),
                                      ("--risk-interface", "P"), ("--risk-device", "cpu"),
                                      ("--risk-update-hz", "20")):
                    for flags in ([option, str(value)], [f"{option}={value}"]):
                        args.baseline_options = ["--", "--ardy", *flags]
                        with self.subTest(method=method, flags=flags), \
                                self.assertRaisesRegex(ValueError, "controlled by this experiment entry point"):
                            prepare_plan(args)
                abbreviations = (["--risk-check", str(files[0]),
                                  "--residual-check", str(files[1]), "--risk-gat", str(files[2])],
                                 ["--bat", "99"], ["--out", str(root / "other")], ["--plan-on"])
                for flags in abbreviations:
                    args.baseline_options = ["--", "--ardy", *flags]
                    with self.subTest(method=method, spelling="abbreviated", flags=flags), \
                            redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        prepare_plan(args)


if __name__ == "__main__":
    unittest.main()
