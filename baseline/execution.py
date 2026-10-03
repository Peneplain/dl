"""Shared frozen-model execution for both batch and manual sessions."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import time

from baseline.common import sha256, write_json
from baseline.grasp import (GraspEvaluator, PHASES, DIRECT_START_PHASES, RIGHT_CLOSED,
                            build_scene, phase_constraints, phase_prompt, wrist_pose,
                            ground_scene, approach_constraints, approach_state,
                            APPROACH_POSITION_TOLERANCE, SETTLE_HOLD_SECONDS,
                            HOLD_PHASES, RIGHT_PRESHAPED, initial_body_reference, hand_alignment)
from baseline.simulation import SimulationStop, SonicSimulation


class ExecutionRuntime:
    """Reuse models and an optional viewer; reset all episode state per attempt."""

    def __init__(self, args, session):
        self.args = args
        self.session = Path(session)
        self.policy = None
        self.service = None
        self.simulation = None
        self.model_load_failed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ardy")
        self.discard_input = lambda: None

    def ensure_simulation(self):
        if self.simulation is None:
            from baseline.sonic_policy import SonicPolicy
            print("[LOAD] Loading frozen SONIC ...", flush=True)
            self.policy = SonicPolicy(self.args.assets, self.args.sonic_repo, threads=2)
            scene = None
            if self.args.grasp:
                scene = self.session / "scene.xml"
                build_scene(self.args.sonic_repo, scene, self.args.cube_xy or [.40, -.22],
                            0.0 if self.args.direct_start else self.args.start_back,
                            self.args.table_standoff, direct_start=self.args.direct_start)
            self.simulation = SonicSimulation(self.args.assets, self.args.sonic_repo, None,
                                             policy=self.policy, scene=scene, gui=self.args.gui,
                                             initial_body_reference=initial_body_reference(
                                                 self.policy.parameters.default) if self.args.grasp else None)
        return self.simulation

    def idle(self):
        if self.simulation:
            self.simulation.sync_viewer()

    def ensure_service(self):
        """Load ARDY and its text encoder on the generation worker thread."""
        if self.service is None:
            from baseline.ardy import ArdyService
            try:
                self.service = ArdyService(self.args)
            except Exception:
                self.model_load_failed = True
                raise
        return self.service

    def preload(self):
        """Prepare all session models before accepting or executing prompts."""
        started = time.perf_counter()
        report = {"status": "loading", "physics_executed": False}
        write_json(self.session / "startup.json", report)
        try:
            self.ensure_simulation()
            print("[LOAD] SONIC loaded; simulation initialized.", flush=True)
            print("[LOAD] Loading frozen ARDY and text encoder; input disabled ...", flush=True)
            self.run_worker(self.ensure_service, "LOAD")
            report["status"] = "ready"
            print("\n============================================================\n"
                  "[READY] ALL MODELS LOADED: ARDY + text encoder + SONIC\n"
                  "============================================================", flush=True)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["wall_seconds"] = time.perf_counter() - started
            write_json(self.session / "startup.json", report)

    def generate(self, *args, **kwargs):
        return self.run_worker(lambda: self.ensure_service().generate(*args, **kwargs), "GENERATE")

    def run_worker(self, work, stage):
        future = self.executor.submit(work)
        last_update = time.perf_counter()
        try:
            while not future.done():
                self.idle()
                self.discard_input()
                if time.perf_counter() - last_update >= 15:
                    print(f"[{stage}] Still working; simulation paused ...", flush=True)
                    last_update = time.perf_counter()
                time.sleep(.02)
            return future.result()
        except (KeyboardInterrupt, SystemExit):
            # No next command or model reset may race an in-flight generation.
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise

    def tick(self, evaluator):
        started = time.perf_counter()
        self.discard_input()
        self.simulation.tick(after_step=evaluator.observe if evaluator else None)
        if evaluator and evaluator.failure:
            raise SimulationStop(evaluator.failure)
        if self.args.mode == "manual":
            time.sleep(max(0, .02 - (time.perf_counter() - started)))

    def run(self, request, output, plan_hash):
        started = time.perf_counter()
        self.model_load_failed = False
        final = {"schema_version": 2, "method": "B0", "status": "running", "stage": "initializing",
                 "request": request, "plan_sha256": plan_hash, "command": sys.argv,
                 "physics_executed": False, "sonic_executed": False,
                 "task_success": False if self.args.grasp else None,
                 "user_corrections": 0, "teacher_used": False, "expert_valid": False,
                 "generation_mode": "offline; simulated time paused during generation",
                 "phase_reports": [], "dependency_versions": {}}
        for package in ("torch", "torch_musa", "mujoco", "numpy", "scipy", "onnxruntime", "transformers"):
            try:
                final["dependency_versions"][package] = version(package)
            except PackageNotFoundError:
                final["dependency_versions"][package] = None
        evaluator = None
        active = False
        def stage(name):
            final["stage"] = name
            write_json(output / "report.json", final)
            print(f"[{output.name}] {name}", flush=True)
        stage("initializing")
        try:
            simulation = self.ensure_simulation()
            if self.args.grasp:
                scene = output / "scene.xml"
                start_back = 0.0 if self.args.direct_start else self.args.start_back
                settings = build_scene(self.args.sonic_repo, scene, request["cube_xy"],
                                       start_back, self.args.table_standoff,
                                       direct_start=self.args.direct_start)
                settings["prompt_profile"] = request["prompt_profile"]
                write_json(output / "settings.json", settings)
                block = int(simulation.model.joint("task_block_free").qposadr[0])
                simulation.model.qpos0[block:block + 2] = request["cube_xy"]
                simulation.model.qpos0[simulation.root_q:simulation.root_q + 2] = settings["robot_start_xy_m"]
                simulation.scene = scene
            simulation.start_run(output)
            active = True
            if self.args.grasp:
                evaluator = GraspEvaluator(simulation, output)
            stage("standing")
            settled_since = None
            direct_start_state = None
            while simulation.data.time < 2.0 - 1e-8:
                self.tick(evaluator)
                if evaluator and self.args.direct_start:
                    direct_start_state = approach_state(simulation, self.args.table_standoff)
                    if direct_start_state["ready"]:
                        if settled_since is None:
                            settled_since = float(simulation.data.time)
                    else:
                        settled_since = None
            if self.args.grasp:
                if self.args.direct_start:
                    if direct_start_state is None:
                        direct_start_state = approach_state(simulation, self.args.table_standoff)
                    direct_start_state = dict(direct_start_state)
                    direct_start_state["continuous_settle_seconds"] = (
                        float(simulation.data.time) - settled_since if settled_since is not None else 0.0)
                    final["direct_start_result"] = direct_start_state
                    simulation.event("direct_start_assessed", **direct_start_state)
                    if (not direct_start_state["ready"] or settled_since is None or
                            simulation.data.time - settled_since < SETTLE_HOLD_SECONDS - 1e-8):
                        raise SimulationStop("direct_start_not_settled")
                standing = simulation.body_pose().copy()
                _, standing_rotation = wrist_pose(simulation)
                phases = DIRECT_START_PHASES if self.args.direct_start else PHASES
            else:
                phases = (("motion", request["duration"]),) if request["prompt"] or request.get("reference") else ()
            for phase_index, (phase, duration) in enumerate(phases):
                stage(phase)
                planned_hold = evaluator is not None and phase in HOLD_PHASES
                if evaluator:
                    evaluator.phase = phase
                    grounding = ground_scene(simulation, self.args.table_standoff)
                    write_json(output / "grounding" / f"{phase}.json", grounding)
                    if phase == "approach":
                        constraints = approach_constraints(simulation, duration, self.args.table_standoff)
                        print(f"  approach target: {grounding['approach_target_xy']} m; "
                              f"distance: {grounding['position_error_m']:.3f} m (MuJoCo state)", flush=True)
                    elif planned_hold:
                        constraints = None
                    else:
                        # Each phase is grounded in the arrived, measured pose.
                        standing = simulation.body_pose().copy()
                        _, standing_rotation = wrist_pose(simulation)
                        constraints = phase_constraints(simulation, phase, duration, standing, standing_rotation)
                    prompt = request.get("phase_prompts", {}).get(phase) or phase_prompt(
                        request["prompt"], phase, request["prompt_profile"])
                else:
                    prompt, constraints = request["prompt"], None
                simulation.event("task_phase", phase=phase, prompt=prompt)
                if planned_hold:
                    print(f"[HOLD] {phase}: preserving the current checked reference; no new motion sample.", flush=True)
                    final["phase_reports"].append({"phase": phase, "mode": "planned_reference_hold",
                                                   "prompt_used_for_generation": False})
                else:
                    print(f"  prompt: {prompt or '(saved reference)'}", flush=True)
                if planned_hold:
                    reference = None
                elif request.get("reference"):
                    if sha256(request["reference"]) != request["reference_sha256"]:
                        raise ValueError("Saved reference changed after the request was recorded")
                    reference = simulation.load_reference(request["reference"])
                else:
                    reference_dir = output / "ardy" / phase
                    report = self.generate(prompt, duration, (request["seed"] + phase_index) % 2**32,
                                           reference_dir, history_qpos=simulation.history_qpos(self.args.history_frames),
                                           pose_constraints=constraints)
                    final["phase_reports"].append({"phase": phase, **report})
                    reference = simulation.load_reference(reference_dir / "reference.npz")
                phase_start = float(simulation.data.time)
                if planned_hold:
                    end = min(30., phase_start + duration)
                else:
                    simulation.install(reference, transition=(.2 if phase == "approach" else .4) if evaluator else .6)
                    final.setdefault("command_to_first_reference_wall_seconds", time.perf_counter() - started)
                    end = min(30.0, simulation.end_time + ((.6 if phase == "approach" else .2) if evaluator else .02))
                settled_since = None
                aligned = False
                while simulation.data.time < end - 1e-8:
                    if evaluator:
                        simulation.command_fingers(RIGHT_CLOSED if phase in {"close", "lift", "hold"}
                                                   else RIGHT_PRESHAPED if phase in {"reach", "lower"}
                                                   else {n: 0. for n in RIGHT_CLOSED})
                    self.tick(evaluator)
                    if evaluator and phase == "lower":
                        alignment = hand_alignment(simulation)
                        if alignment["ready"]:
                            final["grasp_alignment"] = alignment
                            simulation.event("grasp_alignment_reached", **alignment)
                            simulation.hold_measured_pose("grasp_alignment_reached")
                            aligned = True
                            print("[ALIGN] Block entered the hand acquisition region. Holding pose and closing fingers.", flush=True)
                            break
                    if evaluator and phase in {"approach", "settle", "prepare"}:
                        arrival = approach_state(simulation, self.args.table_standoff)
                        if arrival["root_to_table_front_m"] < self.args.table_standoff - .10:
                            final["approach_result"] = arrival
                            raise SimulationStop("approach_too_close")
                        if phase in {"settle", "prepare"}:
                            if not arrival["ready"]:
                                settled_since = None
                            elif settled_since is None:
                                settled_since = float(simulation.data.time)
                if evaluator and phase in {"settle", "prepare"}:
                    final["approach_result" if phase == "settle" else "prepare_result"] = arrival
                    arrival["continuous_settle_seconds"] = (float(simulation.data.time) - settled_since) if settled_since is not None else 0.
                    simulation.event(f"{phase}_assessed", **arrival)
                    if arrival["position_error_m"] > APPROACH_POSITION_TOLERANCE:
                        raise SimulationStop("approach_target_missed" if phase == "settle" else "prepare_target_drift")
                    if arrival["continuous_settle_seconds"] < SETTLE_HOLD_SECONDS - 1e-8:
                        raise SimulationStop("approach_not_settled" if phase == "settle" else "prepare_not_settled")
                    if phase == "settle":
                        print("[APPROACH] Target reached; both feet stable. Starting upper-body preparation.", flush=True)
                    else:
                        print("[PREPARE] Both feet stable after preparation. Starting reach.", flush=True)
                if evaluator and phase == "lower" and not aligned:
                    final["grasp_alignment"] = hand_alignment(simulation)
                    raise SimulationStop("grasp_alignment_missed")
                if evaluator and phase == "close" and evaluator.opposing_contact_seconds < .1:
                    raise SimulationStop("grasp_not_acquired")
            if evaluator:
                stage("hold")
                evaluator.phase = "hold"
                # The configured hold phase above is always recorded in full,
                # including the tail after the two-second success threshold.
                # If it did not succeed there, continue holding until the
                # common 30-second timeout so a late success remains possible.
                while simulation.data.time < 30 - 1e-8 and evaluator.success_time is None:
                    simulation.command_fingers(RIGHT_CLOSED)
                    self.tick(evaluator)
            elif not phases:
                while simulation.data.time < max(2., request["duration"]) - 1e-8:
                    self.tick(None)
            final["status"] = "passed"
        except SimulationStop as error:
            final.update(status="stopped", failure_reason=str(error))
        except KeyboardInterrupt:
            final.update(status="interrupted", failure_reason="interrupted")
            raise
        except Exception as error:
            final.update(status="failed", failure_reason=final["stage"] + "_error",
                         runtime_error=f"{type(error).__name__}: {error}",
                         model_load_failed=self.model_load_failed)
        finally:
            if evaluator:
                failure = final.get("failure_reason")
                final.update(evaluator.summary())
                if final["status"] != "passed":
                    final.update(task_success=False, failure_reason=failure)
                evaluator.close()
            if active:
                try:
                    final.update({k: v for k, v in self.simulation.summary().items() if k != "task_success"})
                except Exception as error:
                    final.update(status="failed", failure_reason="recording_error",
                                 runtime_error=f"{type(error).__name__}: {error}")
                    if self.args.grasp:
                        final["task_success"] = False
                finally:
                    try:
                        self.simulation.end_run()
                    except Exception as error:
                        final.update(status="failed", failure_reason="cleanup_error", runtime_error=str(error))
                        if self.args.grasp:
                            final["task_success"] = False
            final["execution_wall_seconds"] = time.perf_counter() - started
            final["outputs_sha256"] = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                                        if p.is_file() and p != output / "report.json"}
            write_json(output / "report.json", final)
        return final

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.simulation:
            self.simulation.close()
