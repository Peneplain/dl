"""Exercise the interactive viewer startup without loading model weights."""

from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from baseline.adapters.reference import ReferenceBuffer, ReferenceSequence
from baseline.simulation import SonicSimulation
from scripts.run_live import finish_run, parse_request, run_interactive


class InteractiveViewerStartupTests(unittest.TestCase):
    def test_executor_holds_actual_endpoint_and_logs_coverage_once(self):
        simulation = SonicSimulation.__new__(SonicSimulation)
        simulation.policy = SimpleNamespace(future_count=10, future_step=5)
        simulation.data = SimpleNamespace(time=.19)
        simulation.buffer = ReferenceBuffer()
        simulation.buffer.sequence = ReferenceSequence(
            [0, .1, .2], np.tile([0, .1, .2], (29, 1)).T,
            np.tile([1, 0, 0, 0], (3, 1)))
        simulation.end_time = .2
        simulation.current_ref = np.zeros(29)
        simulation.current_quat = np.array([1, 0, 0, 0])
        simulation.holding = False
        simulation.terminal_hold_logged = False
        events = []
        simulation.event = lambda name, **fields: events.append(name)
        simulation.lookahead()
        simulation.lookahead()
        self.assertEqual(events.count("reference_terminal_hold"), 1)
        simulation.data.time = .21
        positions, velocities, _ = simulation.lookahead()
        np.testing.assert_allclose(positions, .2)
        np.testing.assert_array_equal(velocities, np.zeros((10, 29)))
        simulation.lookahead()
        self.assertEqual(simulation.buffer.underruns, 1)
        self.assertEqual(events.count("buffer_underrun"), 1)

    def test_declared_reference_fps_must_match_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            from baseline.adapters.joints import ISAACLAB_JOINT_NAMES
            for fps, times in [(50.5, [0, .02]), (50, [0, .04])]:
                np.savez(path, fps=fps, times=times, joint_names=ISAACLAB_JOINT_NAMES,
                         joint_pos=np.zeros((2, 29)), body_quat=[[1, 0, 0, 0]] * 2)
                with self.subTest(fps=fps, times=times), self.assertRaises(ValueError):
                    SonicSimulation.load_reference(path)

    def test_invalid_duration_is_rejected_before_generation(self):
        for duration in [-1, 0, .01, float("nan"), float("inf")]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                parse_request('{"prompt":"stand","duration":' + str(duration) + '}', 1)

    def test_runtime_and_render_failures_are_not_reported_as_passed(self):
        import json
        for error in [None, "RuntimeError: controller failed"]:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                simulation = SimpleNamespace(video_path=run / "video.mp4",
                                             summary=lambda: {"physics_executed": True},
                                             end_run=lambda: None)
                final = {"runtime_error": error}
                with patch("baseline.rendering.render_rollout", side_effect=RuntimeError("no GL")):
                    finish_run(simulation, final, run)
                report = json.loads((run / "report.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["execution_status"], "failed" if error else "passed")
                self.assertIn("no GL", report["render_error"])

    def test_simulation_stop_is_distinct_from_success(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            simulation = SimpleNamespace(video_path=None,
                                         summary=lambda: {"physics_executed": True},
                                         end_run=lambda: None)
            final = {"stop_reason": "fall"}
            finish_run(simulation, final, run)
            report = json.loads((run / "report.json").read_text())
            self.assertEqual(report["status"], "stopped")
            self.assertEqual(report["execution_status"], "stopped")

    def test_idle_viewer_is_created_and_refreshed_before_first_prompt(self):
        order = []
        viewer_created = threading.Event()
        idle_refreshed = threading.Event()

        class InputAfterViewer:
            def __iter__(self):
                viewer_created.wait(timeout=3)
                idle_refreshed.wait(timeout=3)
                yield "quit\n"

        class Service:
            def __init__(self, args):
                order.append("ardy")

        class Policy:
            def __init__(self, *args, **kwargs):
                order.append("sonic")

        class Simulation:
            def __init__(self, *args, **kwargs):
                order.append("viewer")
                viewer_created.set()

            def sync_viewer(self):
                idle_refreshed.set()

            def close(self):
                order.append("closed")

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                out=Path(directory) / "session", video=None, assets=Path("assets"),
                sonic_repo=Path("sonic"), gui=True, prompt=None,
            )
            with patch("scripts.run_live.ArdyService", Service), \
                    patch("scripts.run_live.SonicPolicy", Policy), \
                    patch("scripts.run_live.SonicSimulation", Simulation), \
                    patch("scripts.run_live.sys.stdin", InputAfterViewer()):
                run_interactive(args)

        self.assertEqual(order, ["ardy", "sonic", "viewer", "closed"])
        self.assertTrue(idle_refreshed.is_set())


if __name__ == "__main__":
    unittest.main()
