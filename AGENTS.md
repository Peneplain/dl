# Language

Write all project content in English. This includes documentation, code comments,
docstrings, prompts, CLI and GUI messages, generated summaries and reports, test
fixtures, and filenames. User-facing conversation may be in Chinese when requested.
Translate existing
Chinese project text when editing it. Keep upstream dependencies, model assets,
and immutable experiment measurements intact; translate presentation text
without changing recorded results or provenance.

# Project specification

At the start of a new chat in this repository, read `docs/project_context.md`
for the latest project handoff, then inspect Git status and task-relevant files.
After material changes, update that handoff with verified facts and remaining
work. Keep secrets and raw chat transcripts out of the handoff.

`docs/proposal.tex` is the source of truth for research scope, architecture,
training and evaluation. `README.md` explains that design and the current
implementation; keep it consistent with the proposal. Distinguish planned
components from implemented and verified behavior. Record agreed design changes
in the relevant documents instead of silently changing the experiment.

The project studies text-driven tabletop block grasping with a simulated
Unitree G1. Compare one frozen ARDY–SONIC baseline (B0) with the same system
augmented by predictive Risk + Residual correction (P), followed by the
proposal's controls and ablations. All task data, evaluation and demonstrations
come from simulation. Supervised correction is the core learning approach;
reinforcement learning, real hardware, camera perception and the proposal's
future VLA directions are outside the core study.

# Architecture and control

- Keep ARDY-G1-RP-25FPS-Horizon8 and SONIC frozen. Pin source revisions,
  checkpoint revisions and the matching SONIC encoder/decoder/config set.
- Use the same task sequencer, simulator-state grounding, pose constraints,
  finger controller, reference adapter, checks, physics and evaluation for
  every method. ARDY receives text, pose history and constraints.
- Decode and map references by explicit joint names. Preserve quaternion
  conventions, units, timestamps and frame indices at module boundaries.
- Insert learned correction between the nominal reference adapter and the
  shared checks. Learn arm-joint reference offsets only; root, torso, leg and
  finger coordinates are not correction outputs. Recompute velocities and
  dependent reference fields after corrections and resampling.
- Risk consumes execution history and future nominal references, with common
  task-phase and planned finger-command context. Residual consumes history,
  nominal references and predicted risk features. Never supply future executed
  states or teacher outputs as inference inputs.
- Apply the arm mask and correction bounds. At low risk, skip residual inference
  and request zero offset; ramp any existing offset to zero under the shared
  rate limit. Keep limits, clearance checks and failure stops identical to B0.
- Use a timestamped buffer for SONIC's 50 Hz target and Risk/Residual's 10–20 Hz
  target. Derive SONIC lookahead from its observation configuration separately
  from the risk horizon. Log underruns and invoke the shared hold behavior.
- Keep G1's root free and the block dynamic. Use articulated fingers and
  frictional contacts for lifting; do not replace contact physics with object
  attachment, scripted block motion or a fixed robot base.

# Code organization

- Keep reusable frozen-model integration, reference interfaces and shared
  execution behavior under `baseline/`. B0 must run without learned correction
  checkpoints, training configurations or learning-package imports.
- Put Risk/Residual models, training, data tooling and experiment entry points
  in separate modules. They may depend on `baseline/`; the dependency must not
  run in the opposite direction. Extend shared interfaces instead of copying
  separate controllers or simulators for each method.
- Implement the full proposal as work progresses. Do not treat a baseline-only
  milestone as a permanent restriction on repository contents.
- Keep the baseline upload archive limited to its declared baseline files.
  Give later training or experiment releases their own explicit packaging
  scope; adding a module must not silently expand an existing source archive.
- Keep real weights, rollout data, generated artifacts, local environments and
  credentials out of Git. Store source, configurations, schemas, tests and
  reproducible commands; keep large outputs in the ignored artifact locations.

# Data and learning

- Split parent episodes, including nominal, counterfactual and teacher branches,
  before extracting windows. Keep held-out prompt paraphrases and scene seeds
  separate as specified in the proposal.
- Obtain nominal risk labels from nominal future execution. Restore simulator
  state, controller history, reference buffers and disturbance seeds when
  branching. Corrected outcomes must not replace nominal counterfactual labels.
- Verify teacher arm corrections through frozen SONIC and the shared hand
  controller from the same perturbed state. Failed recoveries supervise risk
  only. The teacher must not participate in evaluation.
- Train risk first. Freeze it, then train residuals using predicted risk tokens.
  Joint fine-tuning is deferred. Use the proposal's risk horizon and body/time
  representation; do not conflate them with ARDY's generation horizon.
- Keep recoverable-perturbed correction targets and stable-nominal identity
  targets disjoint. Mask unavailable future labels and make contact labels
  task-phase dependent. Open-hand phases are not grasp failures.
- Train bounded residual predictions before gating. Keep loss definitions,
  sampling ratios, normalization, checkpoint provenance and training budgets
  explicit and reproducible.

# Experiments and evidence

- Follow the proposal's B0/B1/B2/I1/I2/I3/I4/P definitions. B1/B2/I1 reuse one
  residual checkpoint; I1–I4/P share a frozen risk encoder and gate. Preserve
  matched feature slots, model capacity and data budgets for risk-interface
  comparisons.
- Change one factor per component ablation. The no-gate variant reuses P's
  trained networks without retraining; the other listed component/loss
  ablations follow the proposal's retraining rules. Prioritize B0/B1/I2/P and
  report unfinished experiments rather than substituting unplanned variants.
- Use paired instructions, initial states, perturbations and ARDY seeds across
  methods. Calibrate thresholds on validation data, then freeze them. Follow
  the planned 20 paired episodes per core condition and configuration; B1, I2
  and P target three training seeds. Treat other single-seed controls as
  exploratory and report deviations from the planned budget.
- A successful trial lifts the block's lowest point at least 5 cm above the
  table and holds it for 2 s, within 30 s of simulated time, without a fall or
  prohibited collision. Include failures and timeouts in the denominator.
- Keep autonomous trials free of extra user corrections. The interactive
  subset allows at most two corrections under the proposal's fixed rule.
- Report success with Wilson 95% intervals and paired differences with an
  episode-level bootstrap. Windows are not independent trials. Preserve
  tracking, contact, balance, intervention, correction and latency metrics,
  along with seeds, prompts, model/scene hashes, trajectories and videos.
- Separate simulation time, wall-clock latency and simulation speed. Motion
  sampling rate alone is not evidence of real-time inference. Keep synthetic
  fixtures, model operator checks and physical task results clearly distinct.

# Validation and documentation

Run checks appropriate to the changed interface and record what they establish.
Documentation-only edits need consistency and formatting checks, not model runs.

- After reference or checkpoint-interface changes, run
  `python scripts/smoke.py --out output/<fresh-run>` and the relevant tests.
  The smoke check uses synthetic references and packet conversion; it is not
  evidence of model compatibility or grasp success.
- Use `python scripts/check_baseline.py --device musa` to check dependencies,
  pinned sources and complete local assets in the supported cluster environment.
  Run `python scripts/check_backend.py --device musa`, then actual ARDY and
  SONIC checks. Primitive checks do not establish full-model compatibility.
- Preserve the matched vendor `torch`/`torch_musa` stack. Do not assume that
  CUDA, TensorRT or a passing CPU run implies MUSA compatibility.
- Validate the simulation in stages: free-base standing, known-reference
  tracking, ARDY standing/arm motion, then contact grasp-and-lift. Record failed
  trials and their causes as well as successful ones.
- Use fresh output directories. Record commands, dependency versions, hashes,
  settings and failures with each run. Update `docs/verification.md` with actual
  evidence and remaining gaps; do not present a planned check as completed.
- Keep README architecture diagrams about the two main systems. Document
  comparison controls and ablations in a separate section, and update current
  implementation status as components become available.
