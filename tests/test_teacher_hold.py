"""Hold teacher scheduling and unchanged counterfactual guards, without rollouts."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from baseline.adapters.joints import ARM_INDICES, ISAACLAB_JOINT_NAMES
from baseline.adapters.reference import ReferenceSequence
from baseline.reference_checks import ReferenceChecks
from risk_residual import teacher
from risk_residual.teacher import PAIR_PHASES, ScheduledArmOffset


class HoldTeacherTests(unittest.TestCase):
    def reference(self, now=1.):
        return ReferenceSequence(now + np.arange(8) * .04, np.zeros((8, 29)),
                                 np.tile([1., 0., 0., 0.], (8, 1)))

    def simulation(self, phase='hold', now=1.):
        events = []
        return SimpleNamespace(
            task_phase=phase, data=SimpleNamespace(time=now, integration=np.array([now, .8, 0.])),
            model=object(), frame_index=int(round(now / .02)), end_time=30.,
            current_ref=np.zeros(29), current_quat=np.array([1., 0., 0., 0.]),
            finger_target=np.zeros(14),
            policy=SimpleNamespace(last_action=np.zeros(29), alignment=SimpleNamespace(as_quat=lambda: np.array([0., 0., 0., 1.])), history=[{'q': np.zeros(29)}]),
            reference_checks=SimpleNamespace(offset=np.zeros(29), last_time=None),
            buffer=SimpleNamespace(sequence=self.reference(now), underruns=0),
            holding=True, terminal_hold_logged=True, pose_history=[(now, np.zeros(7))],
            event=lambda name, **values: events.append((name, values)), events=events)

    def integration_api(self):
        return SimpleNamespace(
            mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
            mj_stateSize=lambda model, spec: 3,
            mj_getState=lambda model, data, target, spec: np.copyto(target, data.integration))

    def test_hold_delay_clean_zero_and_one_preoffset_fingerprint(self):
        nominal = ScheduledArmOffset(phase='hold', delay=.2, joint='right_wrist_pitch_joint', amplitude=-.06, perturb=True)
        clean = ScheduledArmOffset(phase='hold', delay=.2, joint='right_wrist_pitch_joint', amplitude=-.06, perturb=False)
        a, b = self.simulation(), self.simulation()
        reference = self.reference()
        with patch.object(teacher, 'mujoco', self.integration_api()):
            for now in (1., 1.1, 1.18):
                a.data.time = b.data.time = now
                self.assertFalse(nominal.request(a, reference).any())
                self.assertFalse(clean.request(b, reference).any())
                self.assertIsNone(nominal.activation)
            a.data.time = b.data.time = 1.2
            a.frame_index = b.frame_index = 60
            offset = nominal.request(a, reference)
            self.assertFalse(clean.request(b, reference).any())
            np.testing.assert_allclose(offset[:, ISAACLAB_JOINT_NAMES.index('right_wrist_pitch_joint')], -.06)
            self.assertEqual(nominal.activation, clean.activation)
            self.assertEqual(nominal.activation['phase'], 'hold')
            a.data.time = b.data.time = 1.22
            nominal.request(a, reference)
            clean.request(b, reference)
        self.assertEqual(sum(name == 'teacher_pair_activation' for name, _ in a.events), 1)
        self.assertEqual(sum(name == 'teacher_pair_activation' for name, _ in b.events), 1)

    def test_hold_never_activates_before_entering_hold(self):
        provider = ScheduledArmOffset(phase='hold', delay=.04, joint='right_wrist_pitch_joint', amplitude=.06, perturb=True)
        sim = self.simulation('lift')
        with patch.object(teacher, 'activation_fingerprint', return_value='same'):
            self.assertFalse(provider.request(sim, self.reference()).any())
            sim.data.time = 5.
            sim.task_phase = 'hold'
            self.assertFalse(provider.request(sim, self.reference(5.)).any())
            sim.data.time = 5.04
            self.assertTrue(provider.request(sim, self.reference(5.04)).any())
        self.assertEqual(provider.activation['time'], 5.04)

    def test_invalidate_plan_keeps_episode_local_hold_clock_reset_clears_it(self):
        provider = ScheduledArmOffset(phase='hold', delay=.04, joint='right_wrist_pitch_joint', amplitude=.06, perturb=True)
        sim = self.simulation()
        with patch.object(teacher, 'activation_fingerprint', return_value='same'):
            provider.request(sim, self.reference())
            provider.invalidate_plan()
            sim.data.time = 1.04
            self.assertTrue(provider.request(sim, self.reference(1.04)).any())
            provider.reset()
            self.assertFalse(provider.request(sim, self.reference(1.04)).any())
        self.assertIsNone(provider.activation)

    def test_existing_phases_and_authority_limits_stay_validated(self):
        self.assertEqual(PAIR_PHASES, ('reach', 'lower', 'lift', 'hold'))
        for phase in PAIR_PHASES:
            for sign in (-1, 1):
                ScheduledArmOffset(phase=phase, delay=0., joint='right_wrist_pitch_joint', amplitude=sign * .15, perturb=True)
        for kwargs in ({'phase': 'close'}, {'phase': 'lower_replan_01'}, {'joint': 'left_knee_joint'}, {'joint': 'right_hand_index_0_joint'}, {'amplitude': .15001}, {'amplitude': 0.}, {'amplitude': np.nan}, {'delay': -1.}, {'delay': np.inf}):
            defaults = dict(phase='hold', delay=.2, joint='right_wrist_pitch_joint', amplitude=.06, perturb=True)
            defaults.update(kwargs)
            with self.assertRaises(ValueError):
                ScheduledArmOffset(**defaults)

    def test_shared_limiter_preserves_nonarms_and_ramps_hold_entry_exit(self):
        provider = ScheduledArmOffset(phase='hold', delay=0., joint='right_wrist_pitch_joint', amplitude=-.06, perturb=True)
        checks = ReferenceChecks(np.tile([-1., 1.], (29, 1)))
        sim = self.simulation()
        index = ISAACLAB_JOINT_NAMES.index('right_wrist_pitch_joint')
        nonarms = [i for i in range(29) if i not in ARM_INDICES]
        with patch.object(teacher, 'activation_fingerprint', return_value='same'):
            reference = self.reference()
            checked = checks.apply(reference, provider.request(sim, reference))
            self.assertAlmostEqual(float(checked.joint_pos[0, index]), -.01, places=6)
            np.testing.assert_array_equal(checked.joint_pos[:, nonarms], reference.joint_pos[:, nonarms])
            np.testing.assert_array_equal(checked.body_quat, reference.body_quat)
            # Leaving hold requests zero; the shared rate limit is still .5 rad/s.
            sim.task_phase = 'stand'
            sim.data.time = 1.02
            reference = self.reference(1.02)
            zero = provider.request(sim, reference)
            self.assertFalse(zero.any())
            checked = checks.apply(reference, zero)
            self.assertAlmostEqual(float(checked.joint_pos[0, index]), 0., places=6)

    def test_full_fingerprint_rejects_preoffset_drift(self):
        baseline = self.simulation()
        with patch.object(teacher, 'mujoco', self.integration_api()):
            first = teacher.activation_fingerprint(baseline, self.reference())
            self.assertEqual(first, teacher.activation_fingerprint(copy.deepcopy(baseline), self.reference()))
            for field in ('finger', 'physics', 'history', 'limiter', 'phase', 'reference'):
                changed = copy.deepcopy(baseline)
                reference = self.reference()
                if field == 'finger': changed.finger_target[0] = .01
                elif field == 'physics': changed.data.integration[1] += .01
                elif field == 'history': changed.policy.history[0]['q'][0] = .01
                elif field == 'limiter': changed.reference_checks.offset[0] = .01
                elif field == 'phase': changed.task_phase = 'lift'
                else: reference.joint_pos[0, 0] = .01
                self.assertNotEqual(first, teacher.activation_fingerprint(changed, reference), field)

    def collector_fixture(self):
        execution = ModuleType('baseline.execution'); execution.ExecutionRuntime = object
        session = ModuleType('baseline.session'); session.make_plan = lambda config, requests: {'plan_sha256': 'plan'}
        scripts = ModuleType('scripts'); scripts.__path__ = []
        run = ModuleType('scripts.run'); run.config_for = lambda args: {}; run.parse_args = lambda args: args
        path = Path(__file__).resolve().parents[1] / 'experiments/collect_pair.py'
        spec = importlib.util.spec_from_file_location('hold_collector_fixture', path)
        module = importlib.util.module_from_spec(spec)
        # Use a unique module name so mocked heavy backends never contaminate other tests.
        with patch.dict(sys.modules, {'baseline.execution': execution, 'baseline.session': session, 'scripts': scripts, 'scripts.run': run}):
            spec.loader.exec_module(module)
        return module

    def test_cli_accepts_hold_and_preserves_lower_default(self):
        module = self.collector_fixture()
        result = dict(status='paired', pair_state_verified=True, teacher_verified=True, recovery_verified=False, nominal_task_success=True)
        for suffix, expected in [(['--phase', 'hold'], 'hold'), ([], 'lower')]:
            with patch.object(sys, 'argv', ['collect_pair', '--source', '/source', '--out', '/fresh', *suffix]), patch.object(module, 'collect', return_value=result) as collect, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(module.main(), 0)
                self.assertEqual(collect.call_args.kwargs['phase'], expected)

    def collect_fixture(self, *, nominal_status='passed', nominal_physics=True, clean_success=True, drift=False, missing=False):
        module = self.collector_fixture()
        fixture = self
        class Runtime:
            def __init__(self, args, output, correction_provider):
                self.correction_provider = correction_provider
            def run(self, request, branch, plan_hash):
                branch.mkdir(exist_ok=True)
                # Hold is a planned terminal-reference hold: replay only the exact lift artifact.
                self.generate('same saved prompt', 2., request['seed'], branch / 'ardy/lift')
                (branch / 'rollout').mkdir()
                (branch / 'scene.xml').write_text('<same-scene/>')
                (branch / 'rollout/metadata.json').write_text('{"model_sha256":"same-model"}')
                sim = fixture.simulation(now=1.)
                if drift and branch.name == 'nominal': sim.finger_target[0] = .01
                provider = self.correction_provider
                provider.request(sim, fixture.reference())
                sim.data.time = 1.2; sim.frame_index = 60
                provider.request(sim, fixture.reference(1.2))
                clean = branch.name == 'teacher'
                report = {'status': 'passed' if clean else nominal_status, 'task_success': clean and clean_success, 'physics_executed': True if clean else nominal_physics, 'sonic_executed': True}
                (branch / 'report.json').write_text(json.dumps(report))
                return report
            def close(self): pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'source'; source.mkdir()
            (source / 'report.json').write_text('{"source":"immutable"}')
            if not missing:
                (source / 'ardy/lift').mkdir(parents=True)
                (source / 'ardy/lift/reference.npz').write_bytes(b'exact-saved-lift-reference')
            with patch.object(module, 'ExecutionRuntime', Runtime), patch.object(module, 'source_args', return_value=(SimpleNamespace(), {'seed': 50000})), patch.object(teacher, 'mujoco', self.integration_api()):
                if missing:
                    with self.assertRaisesRegex(FileNotFoundError, 'Missing exact saved ARDY reference'):
                        module.collect(source, root / 'pair', phase='hold', delay=.2, joint='right_wrist_pitch_joint', amplitude=-.06)
                    self.assertFalse((root / 'pair/pair.json').exists())
                    return None
                result = module.collect(source, root / 'pair', phase='hold', delay=.2, joint='right_wrist_pitch_joint', amplitude=-.06)
                self.assertEqual((root / 'pair/teacher/ardy/lift/reference.npz').read_bytes(), b'exact-saved-lift-reference')
                self.assertEqual((root / 'pair/nominal/ardy/lift/reference.npz').read_bytes(), b'exact-saved-lift-reference')
                self.assertFalse((root / 'pair/teacher/ardy/hold').exists())
                return result

    def test_hold_pair_preserves_real_execution_and_same_state_gates(self):
        pair = self.collect_fixture()
        self.assertTrue(pair['pair_state_verified'])
        self.assertTrue(pair['teacher_verified'])
        self.assertTrue(pair['recovery_verified'])
        a, b = pair['branches']['teacher'], pair['branches']['nominal']
        self.assertEqual(a['activation'], b['activation'])
        self.assertEqual(a['activation']['phase'], 'hold')

    def test_missing_exact_reference_has_no_generation_fallback(self):
        self.collect_fixture(missing=True)

    def test_hold_recovery_never_accepts_drift_failed_clean_or_invalid_nominal(self):
        for kwargs in ({'drift': True}, {'clean_success': False}, {'nominal_physics': False}, {'nominal_status': 'failed'}):
            with self.subTest(**kwargs):
                pair = self.collect_fixture(**kwargs)
                self.assertFalse(pair['recovery_verified'])
                if kwargs.get('drift'): self.assertFalse(pair['pair_state_verified'])
                if kwargs.get('clean_success') is False: self.assertFalse(pair['teacher_verified'])


if __name__ == '__main__':
    unittest.main()
