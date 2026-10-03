"""Session scheduling, input exclusion and readable results; no grasp-success claims."""

from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from baseline.common import write_json
from baseline.console import ManualConsole
from baseline.session import (make_plan, session_directory, next_attempt, latest_result,
                              summarize, result_label)
from baseline.grasp import phase_prompt
from scripts.run import parse_args, parse_request, config_for, main
from scripts.render import select_attempts


class SessionTests(unittest.TestCase):
    def test_second_precision_session_collision_preserves_both_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("baseline.session.timestamp", return_value="261003-031351"):
                with session_directory(directory, {"first": True}, mode="batch") as first:
                    self.assertEqual(first.name, "batch-261003-031351")
            with patch("baseline.session.timestamp", side_effect=["261003-031351", "261003-031352"]), \
                    patch("baseline.session.time.sleep") as sleep:
                with session_directory(directory, {"second": True}, mode="batch") as second:
                    self.assertNotEqual(first, second)
                sleep.assert_called_once()
            self.assertEqual(json.loads((first / "plan.json").read_text()), {"first": True})
            self.assertEqual(json.loads((second / "plan.json").read_text()), {"second": True})

    def test_empty_defaults_and_task_flag(self):
        args = parse_args(["batch"])
        self.assertFalse(args.grasp)
        self.assertFalse(args.gui)
        self.assertIsNone(args.prompt)
        request = parse_request({}, 1, args)
        self.assertIsNone(request["cube_xy"])
        self.assertIsNone(request["task"])
        self.assertEqual(args.batch, 1)
        for payload in ({"prompt": ""}, {"duration": float("nan")}, {"duration": 31},
                        {"seed": True}, {"typo": "text"}, {"cube_xy": [.4, -.2]}):
            with self.assertRaises(ValueError):
                parse_request(payload, 1, args)
        for argv in (["batch", "--gui"], ["batch", "--batch", "0"], ["manual", "--batch", "2"]):
            with self.assertRaises(SystemExit):
                parse_args(argv)
        with self.assertRaises(SystemExit):
            parse_args(["batch", "--direct-start"])
        self.assertTrue(parse_args(["batch", "--grasp", "--direct-start"]).direct_start)

    def test_grasp_starts_at_table_unless_walk_is_requested(self):
        for mode in ("batch", "manual"):
            args = parse_args([mode, "--grasp"])
            self.assertTrue(args.direct_start)
            self.assertFalse(args.walk)
            walking = parse_args([mode, "--grasp", "--walk"])
            self.assertFalse(walking.direct_start)
            self.assertTrue(walking.walk)
        with redirect_stdout(io.StringIO()):
            for argv in (["batch", "--walk"],
                         ["batch", "--grasp", "--walk", "--direct-start"]):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_grasp_records_full_hold_after_success_and_still_stops_on_failure(self):
        from baseline.execution import ExecutionRuntime
        from baseline.simulation import SimulationStop

        for fail_in_hold in (False, True):
            with self.subTest(fail_in_hold=fail_in_hold), \
                    tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                args = parse_args(["batch", "--grasp"])
                runtime = ExecutionRuntime(args, directory)
                simulation = Mock()
                simulation.data = SimpleNamespace(time=0.)
                simulation.model.qpos0 = np.zeros(14)
                simulation.model.joint.return_value.qposadr = [7]
                simulation.root_q = 0
                simulation.body_pose.return_value = np.r_[np.zeros(3), 1., np.zeros(32)]
                simulation.install.side_effect = lambda *a, **kw: setattr(
                    simulation, "end_time", simulation.data.time + .1)
                simulation.summary.side_effect = lambda: {"simulation_seconds": simulation.data.time}
                runtime.simulation = simulation
                runtime.ensure_simulation = Mock(return_value=simulation)
                runtime.generate = Mock(return_value={"status": "passed"})
                evaluator = Mock(phase="stand", failure=None, success_time=None,
                                 opposing_contact_seconds=.2)
                evaluator.summary.side_effect = lambda: {
                    "task_success": evaluator.success_time is not None and evaluator.failure is None,
                    "success_sim_time": evaluator.success_time,
                    "failure_reason": evaluator.failure}
                hold_times = []
                def tick(_):
                    simulation.data.time = round(simulation.data.time + .02, 8)
                    if evaluator.phase == "lift" and evaluator.success_time is None:
                        evaluator.success_time = simulation.data.time
                    if evaluator.phase == "hold":
                        hold_times.append(simulation.data.time)
                        if fail_in_hold and len(hold_times) == 10:
                            evaluator.failure = "fall"
                            raise SimulationStop("fall")
                runtime.tick = Mock(side_effect=tick)
                try:
                    with patch("baseline.execution.build_scene", return_value={"robot_start_xy_m": [.09, -.08]}), \
                            patch("baseline.execution.GraspEvaluator", return_value=evaluator), \
                            patch("baseline.execution.approach_state", return_value={"ready": True}), \
                            patch("baseline.execution.ground_scene", return_value={}), \
                            patch("baseline.execution.wrist_pose", return_value=(np.zeros(3), np.eye(3))), \
                            patch("baseline.execution.phase_constraints", return_value={}), \
                            patch("baseline.execution.hand_alignment", return_value={"ready": True}):
                        result = runtime.run(parse_request({}, 1, args), Path(directory), "synthetic-plan")
                    self.assertLess(result["success_sim_time"], hold_times[0])
                    self.assertEqual([entry["phase"] for entry in result["phase_reports"]],
                                     ["reach", "lower", "close", "lift", "hold"])
                    if fail_in_hold:
                        self.assertFalse(result["task_success"])
                        self.assertEqual(result["failure_reason"], "fall")
                        self.assertEqual(len(hold_times), 10)
                    else:
                        self.assertTrue(result["task_success"])
                        self.assertEqual(len(hold_times), 150)
                        self.assertAlmostEqual(hold_times[-1] - hold_times[0] + .02, 3.)
                        self.assertAlmostEqual(result["simulation_seconds"], hold_times[-1])
                finally:
                    runtime.close()

    def test_deterministic_requests_flat_attempts_and_failure_denominator(self):
        args = parse_args(["batch", "--grasp", "--batch", "3", "--seed", "42"])
        requests = [parse_request({}, i, args) for i in range(1, 4)]
        self.assertEqual([r["seed"] for r in requests], [42, 43, 44])
        self.assertEqual(requests[0], parse_request({}, 1, args))
        plan = make_plan(config_for(args), requests)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            with session_directory(output, plan):
                first = next_attempt(output, requests[0])
                write_json(first / "report.json", {"request": requests[0], "status": "stopped",
                                                    "task_success": False, "failure_reason": "fall"})
                second = next_attempt(output, requests[1])
                write_json(second / "report.json", {"request": requests[1], "status": "interrupted",
                                                     "task_success": False})
            with session_directory(output, plan, resume=True):
                retry = next_attempt(output, requests[1])
                self.assertEqual(retry.name, "attempt-00002-retry-02")
                write_json(retry / "report.json", {"request": requests[1], "status": "passed",
                                                   "task_success": True})
                summary, text = summarize(output, plan, "running")
                self.assertEqual((summary["completed"], summary["pending"]), (2, 1))
                self.assertEqual(summary["success_rate"], .5)
                self.assertIn("SUCCESS", text)
                self.assertIn("FAILED", text)
                self.assertEqual((output / "failures.txt").read_text(), "attempt-00001\n")
                self.assertEqual((output / "successes.txt").read_text(), retry.name + "\n")
                self.assertTrue((output / "summary.md").is_file())
                self.assertTrue((output / "results.csv").is_file())
                with self.assertRaises(RuntimeError):
                    with session_directory(output, plan, resume=True):
                        pass

    def test_completion_without_task_is_not_reported_as_grasp_success(self):
        self.assertEqual(result_label({"status": "passed", "task_success": None}), "COMPLETED")
        self.assertEqual(result_label({"status": "passed", "task_success": False}), "FAILED")

    def test_plan_only_and_resume_preserve_configuration(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch("baseline.execution.ExecutionRuntime", side_effect=AssertionError("models loaded")):
                main(["batch", "--output-root", directory, "--grasp", "--batch", "2", "--plan-only"])
            output = next(Path(directory).glob("batch-*"))
            args = parse_args(["batch", "--resume", str(output)])
            self.assertEqual(args.batch, 2)
            self.assertTrue(args.grasp)
            original = json.loads((output / "plan.json").read_text())
            self.assertEqual(make_plan(config_for(args), original["requests"]), original)
            with self.assertRaises(SystemExit):
                parse_args(["batch", "--resume", str(output), "--seed", "9"])

    def test_legacy_plan_keeps_its_original_walking_selection(self):
        for grasp, direct_start in ((False, False), (True, False), (True, True)):
            with self.subTest(grasp=grasp, direct_start=direct_start), \
                    tempfile.TemporaryDirectory() as directory:
                args = parse_args(["batch", "--grasp"] if grasp else ["batch"])
                config = config_for(args)
                config.pop("walk")
                config["direct_start"] = direct_start
                write_json(Path(directory) / "plan.json", {"schema_version": 2, "config": config})
                resumed = parse_args(["batch", "--resume", directory])
                self.assertEqual(resumed.walk, grasp and not direct_start)
                self.assertEqual(resumed.direct_start, direct_start)

    def test_resume_rejects_changed_source_or_completed_evidence(self):
        from baseline.common import sha256
        args = parse_args(["batch"])
        request = parse_request({}, 1, args)
        plan = make_plan(config_for(args), [request])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            with session_directory(output, plan):
                attempt = next_attempt(output, request)
                state = attempt / "evidence.txt"
                state.write_text("original")
                write_json(attempt / "report.json", {"status": "passed", "request": request,
                           "outputs_sha256": {"evidence.txt": sha256(state)}})
                latest_result(output, 1, verify=True)
                state.write_text("changed")
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    latest_result(output, 1, verify=True)
            changed = {**plan, "source_sha256": {"different": "source"}}
            with self.assertRaisesRegex(ValueError, "source/configuration"):
                with session_directory(output, changed, resume=True):
                    pass

    def test_failed_generation_does_not_destroy_worker_for_next_attempt(self):
        from baseline.execution import ExecutionRuntime
        runtime = ExecutionRuntime(parse_args(["batch"]), Path("unused"))
        runtime.service = Mock()
        runtime.service.generate.side_effect = [ValueError("invalid generated motion"), {"status": "passed"}]
        try:
            with self.assertRaisesRegex(ValueError, "invalid generated"):
                runtime.generate("first")
            self.assertEqual(runtime.generate("second"), {"status": "passed"})
        finally:
            runtime.close()

    def test_preload_reuses_models_and_does_not_generate_motion(self):
        from baseline.execution import ExecutionRuntime
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as terminal:
            runtime = ExecutionRuntime(parse_args(["batch"]), directory)
            runtime.ensure_simulation = Mock()
            with patch("baseline.ardy.ArdyService") as service:
                try:
                    runtime.preload()
                    service.assert_called_once()
                    service.return_value.generate.assert_not_called()
                    self.assertIn("ALL MODELS LOADED", terminal.getvalue())
                    report = json.loads((Path(directory) / "startup.json").read_text())
                    self.assertEqual(report["status"], "ready")
                    self.assertFalse(report["physics_executed"])
                    runtime.generate("first prompt")
                    service.assert_called_once()
                    service.return_value.generate.assert_called_once_with("first prompt")
                finally:
                    runtime.close()

    def test_manual_loads_before_ready_or_reading_input_with_and_without_gui(self):
        for gui in (False, True):
            with self.subTest(gui=gui), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as terminal:
                runtime, console = Mock(), Mock()
                console.busy.return_value = nullcontext(lambda: None)
                def preload():
                    self.assertNotIn("READY >", terminal.getvalue())
                    console.read_line.assert_not_called()
                runtime.preload.side_effect = preload
                def read_line(idle):
                    runtime.preload.assert_called_once()
                    return "quit"
                console.read_line.side_effect = read_line
                with patch("baseline.execution.ExecutionRuntime", return_value=runtime), patch("scripts.run.ManualConsole", return_value=console):
                    self.assertEqual(main(["manual", "--output-root", directory] + (["--gui"] if gui else [])), 0)
                runtime.run.assert_not_called()

    def test_startup_failure_never_accepts_or_executes_requests(self):
        for mode in ("batch", "manual"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as terminal:
                runtime, console = Mock(), Mock()
                runtime.preload.side_effect = RuntimeError("model load failed")
                console.busy.return_value = nullcontext(lambda: None)
                with patch("baseline.execution.ExecutionRuntime", return_value=runtime), patch("scripts.run.ManualConsole", return_value=console):
                    with self.assertRaisesRegex(RuntimeError, "model load failed"):
                        main([mode, "--output-root", directory])
                self.assertNotIn("READY >", terminal.getvalue())
                runtime.run.assert_not_called()
                console.read_line.assert_not_called()
                summary = json.loads(next(Path(directory).glob("*/summary.json")).read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["completed"], 0)

    def test_batch_stops_on_missing_models_and_keeps_remaining_pending(self):
        runtime = Mock()
        def fail(request, output, plan_hash):
            runtime.preload.assert_called_once()
            result = {"request": request, "status": "failed", "stage": "reach",
                      "model_load_failed": True, "failure_reason": "reach_error",
                      "runtime_error": "missing ARDY checkpoint", "task_success": False}
            write_json(output / "report.json", result)
            return result
        runtime.run.side_effect = fail
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch("baseline.execution.ExecutionRuntime", return_value=runtime):
                code = main(["batch", "--output-root", directory, "--grasp", "--batch", "3"])
            self.assertEqual(code, 1)
            self.assertEqual(runtime.run.call_count, 1)
            summary = json.loads(next(Path(directory).glob("batch-*/summary.json")).read_text())
            self.assertEqual((summary["completed"], summary["pending"], summary["failures"]), (1, 2, 1))

    def test_manual_discards_input_received_while_busy(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd) as stream:
                console = ManualConsole(stream)
                os.write(write_fd, b'{"prompt":"first"}\n')
                self.assertIn("first", console.read_line())
                with console.busy() as drain:
                    os.write(write_fd, b'{"prompt":"must not execute"}\n')
                    drain()
                    os.write(write_fd, b'partial prompt')
                self.assertGreater(console.discarded, 0)
                os.write(write_fd, b'{"prompt":"second"}\n')
                self.assertEqual(json.loads(console.read_line())["prompt"], "second")
        finally:
            os.close(write_fd)

    def test_tty_echo_is_restored_and_busy_keys_are_discarded(self):
        import pty
        import termios
        master, slave = pty.openpty()
        try:
            with os.fdopen(slave) as stream:
                original = termios.tcgetattr(stream.fileno())
                console = ManualConsole(stream)
                with console.busy():
                    self.assertFalse(termios.tcgetattr(stream.fileno())[3] & termios.ECHO)
                    os.write(master, b'buffered while busy\n')
                self.assertEqual(termios.tcgetattr(stream.fileno()), original)
                os.write(master, b'ready\n')
                self.assertEqual(console.read_line(), "ready")
        finally:
            os.close(master)

    def test_render_selection_includes_failures_but_success_filter_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, success in enumerate((True, False, None), 1):
                path = root / f"attempt-{index:05d}"
                path.mkdir()
                write_json(path / "report.json", {"task_success": success})
            self.assertEqual(len(select_attempts(root, all_attempts=True)), 3)
            self.assertEqual(len(select_attempts(root, attempts=["1", "3"])), 2)
            self.assertEqual([p.name for p in select_attempts(root, successes=True)], ["attempt-00001"])
            with self.assertRaises(ValueError):
                select_attempts(root, attempts=["../outside"])
            with self.assertRaises(ValueError):
                select_attempts(root)

    def test_focused_prompts_use_current_phase_and_custom_overrides(self):
        prompt = phase_prompt(None, "reach")
        self.assertNotIn("lift the block", prompt)
        self.assertIn("right", prompt)
        self.assertIn("feet", prompt)
        self.assertIn("Current step:", phase_prompt(None, "reach", "legacy"))
        args = parse_args(["manual", "--grasp"])
        request = parse_request({"phase_prompts": {"reach": "A person bends the right elbow."}}, 1, args)
        self.assertEqual(request["phase_prompts"]["reach"], "A person bends the right elbow.")


if __name__ == "__main__":
    unittest.main()
