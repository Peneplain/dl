# Project context for new Codex chats

Read this with `AGENTS.md` before changing the repository. It is a concise
continuity note; `docs/proposal.tex` is the research source of truth and
`docs/verification.md` records measured evidence. Update this note when the
project state materially changes. Do not include passwords, credentials,
private session text or restricted robot data.

## Current project

- Course: HKU DASC 7606C Track 4, Group 3. The required deliverable is a
  text-driven simulated Unitree G1 tabletop block grasp using SONIC as the
  low-level controller, two distinct methods, repeated-trial evaluation,
  report, presentation, videos and a public reproducible repository.
- Canonical working copy: `/home/group3/dl` on the S4000 host. Preserve
  server-side changes when reconciling older local copies. `DL-proposal` is
  separate and is not the implementation repository.
- Current study: frozen ARDY–SONIC B0 versus predictive structured Risk plus
  bounded arm-reference Residual P. GR00T and bimanual plans from older notes
  are not the current core implementation. Physical robot deployment and
  real-robot data are outside this simulation study.
- All project files and reports are in English. Conversation may be Chinese.

## What exists

- B0 supports batch/manual text input, simulator-state task grounding, free-base
  G1, a dynamic table block, articulated fingers, frozen ARDY generation and
  SONIC execution, episode logs and offline video rendering. See `README.md`,
  `docs/grasp.md` and `docs/verification.md` for exact behavior and evidence.
- Recent B0 changes include the nominal 50 Hz context logger, per-attempt scene
  compilation, grasp alignment/hold diagnostics and solver calibration. Do not
  replace these with older local `baseline/` files.
- `risk_residual/`, `experiments/` and `configs/learning/` provide a versioned
  learning schema, models, losses, dataset checks, synthetic two-stage training
  check and optional correction provider. A controlled paired-teacher pilot
  and rollout-to-window converter passed one same-state simulation pair and
  single-parent loader audit; see `docs/learning.md` and `docs/verification.md`.
  The final fixed-split dataset and two full 30-epoch Risk runs are now
  complete; see the latest completion handoff below. Arbitrary mid-episode
  branching and general teacher recovery remain limited. Residual training
  and the matched P trial entry point are implemented; new physical training
  and evaluation outcomes remain to be measured.
- The October 5 merge passed 78 server tests, pinned-asset/MUSA operator checks,
  baseline reference smoke and reduced-width synthetic Risk/Residual training
  on MUSA. An unprompted two-second B0 episode executed 100 SONIC frames in
  MuJoCo. These checks do not measure P grasp success.
- The paired-teacher/data-conversion update passed 84 server tests, a new
  baseline reference smoke, one matched successful-clean/failed-perturbed
  simulation pair, and a single-parent loader audit. At that pilot stage,
  the available prompt groups did not support a leakage-free three-split dataset.
- Optional frozen K0 is selected with `--kimodo`; ARDY/B0 stays the default
  with unchanged generator, controller, physics and evaluator. Pinned Kimodo
  assets are downloaded/transferred and offline-verified. New K0 grasp plans
  default to raw output and the explicit wrist-target calibration
  `[0.125, 0.035, 0.080]` metres; ARDY retains its measured-site default.
  The final-source 20-attempt batch completed with 11/20 contact-grasp
  successes (55%, Wilson 95% 34.2%--74.2%). These tuned seeds are not held-out
  evaluation or paired B0/P evidence. Optional nominal projection remains
  experimental and has no verified physical grasp success. Kimodo has no
  native ARDY history conditioning; current-state anchors and transitions
  are shared. See the October 7 final handoff and verification entry.
- Real weights, large outputs, environments and upstream source checkouts stay
  outside Git. Use `output/` for fresh experiment artifacts.

## Working rules

1. Inspect `git status` and current evidence before modifying code; preserve
   uncommitted server work.
2. Keep B0 independent of the learning package. Compare methods through one
   shared simulator, controller, hand logic and evaluator.
3. Distinguish synthetic tests, operator checks and measured physical trials.
   Never report P grasp performance from a synthetic fixture.
4. Run relevant tests, `./run.sh check` when dependencies change, and record
   new measured evidence in `docs/verification.md`.
5. Check this note and `docs/learning.md` at the start of a new chat. At the
   end of material work, update their status so later chats have a handoff.

## Storage handoff — 2026-10-05

- `output` on the S4000 host is a symlink to `/data/group3/dl-output` on the
  larger data filesystem. `docker/run-musa.sh` bind-mounts an external output
  symlink target automatically. Ordinary `./run.sh batch ...` commands need no
  additional output argument. Verify with `df -h output` on the host.
- The former 943-file output tree remains at
  `/home/group3/dl-output-backup-20261005`. Its copy under `/data` passed a
  checksum-mode comparison. Do not delete the backup without separately
  deciding that retention is no longer needed.
- A container mount check passed and 85 server unit tests passed (three
  optional RGB tests skipped). This storage change collected no new episodes.
  A leakage-free train/validation/test dataset and general corrective teacher
  remain outstanding.

## Data tooling handoff — 2026-10-05

- `experiments.plan_data plan` creates immutable B0 batch plans with prompt
  groups and scene seeds assigned to train/val/test before collection. The
  proposed `configs/data/collection.json` budgets 120/40/40 episodes in six
  authored prompt groups. No prompt-performance result is claimed.
- The code-only plan at `output/data-collection-261005-code-ready` contains
  `collection-plan.json`, generated prompts and six commands in `commands.txt`.
  No batch command was executed while preparing it. Run those commands from
  the host when ready; indexing and teacher collection are separate commands.
- `experiments.plan_data index` validates completed B0 batches and writes source
  candidates plus explicit pending/unusable reasons. `experiments.collect_pairs`
  schedules bounded saved-reference pairs from successes and supports resume
  without silently retrying completed failures. All variants retain the parent
  split. General recovery of naturally failed states remains unimplemented.
- The converter supports `--skip-ineligible` and supervision readiness reports.
  It still rejects corruption and changed provenance. `experiments.audit_dataset`
  counts valid positive/negative/censored Risk labels, correction/stable samples,
  contributing parents and phases. Training requires both Risk classes or both
  Residual supervision categories in train and val. See `docs/learning.md` for
  the complete command sequence and current limitations.
- The offline baseline asset check passed with the installed S4000 stack; these
  tools need no new packages or model downloads. Tests use temporary fixtures
  and mocked pair execution; they do not establish a new physical recovery rate.
- The final server suite ran 98 tests with three optional RGB tests skipped;
  all remaining tests passed. Exact logs and code-only plan evidence are in
  `docs/verification.md`.

## Acquisition repair handoff — 2026-10-05

- The user-run `data-collection-261005-code-ready/batches/train-a` completed
  60/60 attempts with zero successes: 52 timeouts, seven missing grasps and
  one prohibited robot-table contact. All recorded acquisition positions were
  near the old lower vertical gate boundary, -4.50 to -3.82 cm from the hand
  center. Most fingers contacted the block's upper edge and lost it on lift.
- `hand_alignment` now uses one-third of the block half extent vertically
  (1 cm for this block), retaining the original lateral margins, controller,
  motor limits, physics, success test and timeout. A matched three-seed pilot
  with the final default code retained seed 10000 through a complete ten-second
  hold; seeds 10001/10002 still failed acquisition/retention. This is a small
  calibration result, not a validated batch success rate.
- A separate terminal-rotation position-goal change failed its three physical
  pilots and was withdrawn. Its source and reports remain in output. No new
  IK, object attachment, policy training or large collection was performed.
- Use the fresh plan `output/data-collection-261005-acquisition-fixed` and its
  `commands.txt`. Earlier `code-ready` and `data-collection-261005-fixed` plans
  have different baseline source hashes and must not resume with current code.
  Preserve the old failed episodes and keep controller-version cohorts explicit.
- The final suite ran 98 tests, with three optional RGB tests skipped and all
  remaining tests passing. Three original failure videos and one successful
  diagnostic video were rendered; see `docs/verification.md` for artifact paths.

## Completed collection and output cleanup — 2026-10-06

- All six user-run groups completed: train-a 23/60, train-b 19/60, val-a 10/20,
  val-b 3/20, test-a 9/20 and test-b 5/20 retained successes, totaling 69/200.
  These are B0 collection outcomes, not trained Risk/Residual performance.
  No learning dataset manifest, teacher-pair proof or training checkpoint was
  present in output when inspected.
- The user requested renaming the completed collection to
  `output/data-collection-261005`. All episode files and plan hashes were
  preserved. `commands.txt` uses the new path; `relocation.json` records the
  original location. Load episodes from their current parent directories;
  historical reports still record their original execution-time paths.
- The relocated `index/` passed the source-evidence and split audit: 200
  planned parents, 196 candidates, 69 teacher candidates, zero pending and four
  unusable execution failures. Window eligibility, paired recovery and training
  readiness still require the later converter/teacher steps.
- Output retains that collection, `batch-261003-122608` (all 20 attempt videos)
  and both `manual-261005-152355` and `manual-261005-155455` video sessions.
  Protected video SHA-256 values were checked after cleanup.
- Obsolete plans, repair pilots, smoke artifacts and temporary logs were
  removed at the user's request, freeing 9,529,565,184 allocated bytes. Their
  compact reports/configurations are under
  `/home/group3/dl-output-history/cleanup-261006/reports`, with the explicit
  deletion list and video hashes in the sibling `cleanup-receipt.json`.
  Historical verification links into removed directories refer to those
  archived reports. Raw removed rollouts were not archived. The separate
  original storage-migration backup was outside this cleanup request.

## Real Risk dataset and MUSA training — 2026-10-06

- The 196 eligible collection parents were converted without additional skips
  into `output/dataset-261006-risk/manifest.json`: 24,163 train, 8,589 val and
  9,235 test windows. Train/val include both intervention classes and pass
  Risk availability checks. The exploratory tracking thresholds remain
  `.06 .06 .1 .1`; test data did not choose settings or checkpoints.
- The dataset has zero verified corrections in every split, so Residual is
  not ready. It has 639/236/212 stable identity windows, respectively.
  Collect verified teacher pairs before rebuilding for Residual; extra
  ordinary B0 rollouts do not supply same-state correction supervision.
- The full-width model initially failed MUSA's SDPA descriptor setup. A
  MUSA-only explicit-attention encoder path now retains identical parameter
  names, dimensions, layer count and dropout. CPU uses the ordinary PyTorch
  path. The frozen ARDY/SONIC execution chain was not changed.
- A real full-size batch passed forward, backward, optimizer update and eval
  on physical GPU 2. A subsequent full epoch passed 378 updates, validation
  and checkpoint writes under `output/risk-epoch-check-261006`; validation
  loss was 0.1403171413. These are training checks, not physical P evaluation.
- The suite passed 100 tests with three optional RGB skips. The log is
  `output/risk-training-suite-261006.log`. Attention regression checks compare
  parameter keys, output and gradients with the ordinary encoder and exercise
  the full-width dropout path while SDPA is unavailable.
- `scripts/train_risk_multiseed.sh` launches one independent Risk seed per
  physical GPU, default 0/2/3, each using the existing 30-epoch config. It is
  not DDP. Use the documented `nohup` command to continue after disconnect;
  fresh run directories contain per-seed logs/checkpoints and a final summary.
  GPU 1 was occupied at inspection. No new nominal batch was started.
- The launcher itself passed a simultaneous one-epoch run on all three GPUs,
  saved at `output/risk-multiseed-261006-023521`. All three reports and the
  final summary passed. At that check, the ordinary 30-epoch run was
  prepared but had not started. The final teacher-derived manifest now has
  completed Risk runs, as recorded below. Residual provenance checks require
  the matching final Risk checkpoint and dataset.

## Authorized background teacher-to-Risk workflow — 2026-10-06

- The user authorized pushing current server code, starting teacher pairs,
  automatically converting/auditing the final dataset, then training Risk on
  all available cards. Existing main `aa3b252` was already pushed and clean.
- `scripts/teacher_risk_pipeline.py` implements that sequence using the host
  standard library and existing MUSA container. Preferred teacher GPU is 4;
  collection remains serial. The current fixed plan has 207 candidates from
  69 successful parents (42 train, 13 val, 14 test), with no missing required
  phase references. These are candidates, not verified recovery counts.
- Conversion is followed by a separate Risk AND Residual readiness audit.
  Failed/incomplete audits stop training. Rejected individual collection
  attempts are preserved, and completion must be established from the report.
- Risk selection rechecks zero memory/utilization and listed processes before
  starting one 30-epoch independent seed on each idle physical GPU. GPU 1's
  existing 16.5 GiB Python process was retained. Seven cards were idle at
  preflight; actual selection is recorded at training time. No DDP is claimed.
- Atomic `state.json`, per-stage logs, PIDs, source hashes and unique event IDs
  provide monitoring and notification evidence. The detached server process
  chains stages directly. Desktop scheduled monitoring is only for notices.
- Five workflow tests passed locally. Physical GPU 7 also passed a real
  one-epoch check of the launcher's explicit output-directory argument, saved
  at `output/pipeline-launch-check-261006`; validation loss was 0.1403171413.
  This is execution compatibility, not a learned-control result.
- Workflow code was pushed as `0c2e00a`, then the fresh
  `output/teacher-risk-261006` run was detached with parent PID 1390625,
  SID 1390625 and PPID 1. Collection used physical GPU 4. The first
  post-disconnect check recorded seven of 207 candidates completed, two
  verified recoveries and one rejected execution. The final completion
  evidence is recorded below.
- The existing `dl-teacher-data-finish-and-run` heartbeat was updated to
  `DL teacher → dataset → Risk monitor`, checking every 15 minutes for this
  exact run. It keeps quiet at healthy unchanged progress, reports new stage
  completions/failures, and sends deduplicated `[DL]: ` Outlook notices to
  `wentao_gu@outlook.com`. It pauses after terminal notices are delivered.
  The record-sync and generic mail-queue automations remain paused.

## Completed teacher-to-Risk workflow — 2026-10-06

- The exact run `output/teacher-risk-261006` completed at
  2026-10-05 21:26:36 UTC (2026-10-06 05:26:36 Asia/Shanghai).
  `state.json` records schema `dl-teacher-risk-pipeline-v1`, status
  `complete`, and four stage-completion events. The parent exited after
  writing the terminal state; this was a completed workflow.
- All 207 teacher candidates were processed: 48 verified controlled
  recoveries, 68 failed/rejected pairs and zero pending. Individual failures
  remain in the artifacts. General recovery of naturally failed states is
  still outside the implemented teacher's scope.
- Conversion and the independent availability audit both passed. Final
  train/val/test windows are 53,485/17,268/20,222, with 28/7/13 correction
  windows and 1,686/624/723 stable identity windows. No source parents were
  skipped by conversion. Parent variants share split ownership; window
  counts are not independent episode counts.
- Risk ran independent seeds 0 and 1 on physical GPUs 2 and 3, respectively,
  using the existing 30-epoch configuration. Other cards had active workloads
  at training startup. Each model completed 25,080 updates with 3,227,812
  parameters on MUSA. The best validation losses were 0.0865046687 and
  0.0998180839, both at epoch 3. Each `best.pt` hash matches its final report.
- Use `output/teacher-risk-261006/dataset/manifest.json` for the final
  fixed dataset. Per-seed checkpoints, metrics and reports are in
  `output/teacher-risk-261006/risk/seed-0` and `risk/seed-1`; the overall
  result is `risk/summary.json`. Read `docs/verification.md` for hashes.
- Readiness establishes supervision availability only. Correction coverage
  remains small and Risk labels are heavily imbalanced. Later epochs did
  not improve the best validation loss. No final test evaluation, gate
  calibration, Residual training or physical P trial was performed.
  Next work is validation-based Risk assessment/calibration, additional
  verified correction coverage if needed, then frozen-Risk Residual training
  and paired B0/P evaluation. No grasp improvement or DDP result is claimed.


## Risk quality work in progress — 2026-10-06

- The user authorized stepwise label diagnosis, validation/gate metrics,
  overfitting controls, independent teacher coverage, frozen-Risk Residual
  training and a paired B0/P pilot. Three implementation/review agents were
  used; long jobs use atomic state and a thread heartbeat.
- Train/validation-only diagnosis found no confirmed unit, joint-mapping or
  time-alignment error. Continuous 5 ms physics evidence identifies 6,712
  train and 2,234 validation stable hold windows from 35/12 original episodes.
  Every one currently triggers the tracking proxy. Standing arm medians
  0.07343/0.07575 rad exceed the exploratory 0.06 rad thresholds.
- The diagnostic report is `output/risk-quality-261006-checks/label-diagnosis.json`.
  Repeated pair clean replays are correlated; parent variants are not new
  independent episodes. Verified recovery labels must remain positive even
  when tracking/contact/balance short-horizon proxies do not trigger.
- `experiments/evaluate_risk.py` now exports masked class metrics, both-class
  PR/AP, phase/body summaries, clustered intervals and a current-state
  comparator with a separate future-onset subset. Thresholds are selected
  on validation only. Temperature fitting remains a diagnostic; runtime
  requires raw sigmoid and temperature 1 with matching validation hashes.
- `configs/learning/risk-quality-pilot.json` changes only early stopping
  (patience 5) while preserving architecture, losses, natural sampling and
  the maximum 30-epoch budget. Balanced intervention sampling is a separate
  optional experiment, not an automatic simultaneous change.
- A fairness bug was fixed in the P reference path: preserve the B0 sparse
  nominal velocity and add only the checked correction derivative. A zero
  provider now preserves nominal SONIC position/velocity/quaternion inputs.
  Old dense-velocity teacher replay results remain immutable and belong to
  the previous control-source cohort; they do not prove current P performance.
  The baseline source hash changed, so do not bypass old batch resume locks.
- `scripts/risk_quality_pipeline.py` runs old Risk seeds 0/1 validation,
  then 10 train and 5 validation parents at fresh scene seeds 12000–12004,
  13000–13004 and 22000–22004, followed by at most three initial teacher
  candidates. It stops at a measured pilot-review gate. New teacher plans
  record current source hashes. No new test scenes are collected.
- Server regression: 133 tests passed with 3 skips; real compiled MuJoCo
  feedback parity and zero-provider B0/P equality passed. No new Risk,
  Residual or physical P outcome is yet established by these code tests.

- The bounded run `output/risk-quality-261006` was launched detached at
  2026-10-06 15:17:39 Asia/Shanghai with parent PID 1010421 and source
  commit `9c382dc085531dfb6d9dc661ed1b2621b33fec97`. Old validation
  completed at 15:18:45; the parent exit-code interruption and safe resume
  are described below. State schema is
  `dl-risk-quality-pipeline-v1`; monitoring checks every 15 minutes and
  continues the authorized quality stages after measured gates.
- Old seed 0/1 gates are 0.18/0.31, validation balanced accuracy
  0.887097/0.891694, specificity 0.781726/0.792724. In 2,216
  current-proxy-negative future-onset windows, recall is
  0.932302/0.893617 and balanced accuracy 0.857014/0.843171.
  Existing exploratory proxy labels still need physical calibration.


## Quality workflow pause and authorized resume — 2026-10-06

- The first quality run completed train-a seeds 12000–12004: 5/5 physical
  trials, 2 successes (12002/12003), 2 timeouts (12000/12001) and 1 late
  grasp loss (12004). All outcomes and source evidence are preserved in
  `output/risk-quality-261006/parents/train-a/batch-261006-151846`.
- The host runner stopped because B0 batch returns exit 1 for evaluated
  task failures. That return code must be assessed together with the
  complete per-attempt physics/SONIC reports; it is not proof of a broken
  model load or simulation. Parent PID 1010421 and child 1014707 exited.
- The user requested save/stop, then explicitly authorized continuing.
  Local source/artifact archives and a server STOPPED-HANDOFF.md were saved.
  Resume must preserve old events/state, verify input and evidence hashes,
  and reuse valid old validation plus the completed five trials. An
  explicit upgrade of only the runner's source pin requires the old Git
  blob to match the previously recorded hash. Baseline/source locks remain
  strict; incomplete or altered artifacts are rejected.
- The reviewed follow-up code adds train-only exploratory label calibration
  and the bounded continuation chain. Physical recovery and label-validation
  gates precede fresh Risk/Residual training. Exact commands, source hashes,
  budgets and model selection are recorded in `docs/learning.md`.
- The repaired server suite passed 154 tests with 3 optional RGB skips in
  6.289 seconds. Real saved train-a reports, fixed physical configuration,
  phase prompts and every hashed episode output passed the reuse preflight.
  Regression tests verify validation/train-a reuse without duplicate trials,
  Git-verified runner upgrade, active-PID rejection and unchanged evidence.
- The train-only calibration artifact is frozen at
  `output/risk-quality-261006-checks/label-calibration.json`, SHA-256
  `b6f2be232e9c6d0850a0ed6035fd36e6a8699cd11de6962f5645528b61fbe1bf`.
  It proposes `.14 .14 .10 .12`, writes no labels and uses no test evidence.
  The continuation must audit the rebuilt dataset and satisfy its physical
  evidence gates before starting Risk and Residual training. Use the exact
  run's `state.json` for current stage/PID rather than the original launch PID.

## Kimodo final-source grasp handoff — 2026-10-07

- Canonical implementation remains `/home/group3/dl` on the worker. No older
  local repository was copied over it. Downloaded pinned Kimodo source and
  checkpoint were transferred with rsync; `./run.sh shell -c 'python
  scripts/fetch_kimodo.py --offline'` passed with no network access required.
- `./run.sh batch --kimodo --grasp --batch 20 --seed 0 --device musa:0
  --text-device musa:0 --output-root output/kimodo-final-batch20-261007`
  completed all seeds 0--19 at
  `output/kimodo-final-batch20-261007/batch-261007-104935`. It uses 100 steps, guidance `[2,2]`, raw nominal
  rotations and saved wrist offset `[0.125,0.035,0.080]` metres.
- Final result: 11 successes, 5 alignment misses, 1 acquisition failure,
  2 timeouts and 1 loss after the hold threshold. All attempts remain in the
  denominator; loss-after-threshold is not counted as success. Successful
  trials preserve free base, dynamic contact lift, final retention, opposing
  fingers, the 5 cm/2 second/30 second evaluator and no teacher, learned
  correction or additional user intervention.
- The preceding raw `.08` calibration run also obtained 11/20 on these same
  seeds; the raw `.04`, original and named-order-only 20-trial batches obtained
  0/20. Failed projection and guidance pilots remain saved. Do not pool
  repeated seeded trials or report a held-out or matched generator advantage.
- Full server suite: 184 tests passed; clean staged-index snapshot: 173 tests
  passed, both including RGB/MP4 tests. Fresh
  synthetic reference smoke, baseline/MUSA checks and Kimodo offline asset
  verification passed. Final batch source hashes match the current worktree.
- A fresh default ARDY text-action regression passed and executed 229 SONIC
  frames. Its separate seed-0 grasp trial completed the shared runtime with
  no runtime error but timed out at 30 seconds; report it as a task failure.
  `baseline/ardy.py`, shared grounding/fingers/physics and frozen assets have
  no changes in this K0 work.
- Three representative final-source trials passed offline rendering (3/3):
  attempt 1 (alignment failure), attempt 2 (seed-1 success), and attempt 4
  (seed-3 success), each with third-person and wrist cameras. Rendering is
  replay evidence, not a new simulation or performance trial.
- Default B0 source packaging excludes Kimodo files. Use `--scope kimodo`
  for the explicit K0 package. Real weights, outputs and credentials remain
  outside Git. Preserve unrelated uncommitted Risk/Residual continuation,
  review and learning-document edits; only K0 changes belong in its commit.
- Limitations: grasp tuning reused seeds 0--19, walking/grasp GUI validation
  and held-out paired Kimodo evaluation are unfinished, native history
  conditioning is absent, and optional arm projection has only failed pilots.
  Follow `docs/verification.md` for exact commands, hashes and retained data.
