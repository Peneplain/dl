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
  A disjoint train/validation/test dataset, arbitrary mid-episode branching,
  general teacher recovery and the P trial entry point remain future work.
- The October 5 merge passed 78 server tests, pinned-asset/MUSA operator checks,
  baseline reference smoke and reduced-width synthetic Risk/Residual training
  on MUSA. An unprompted two-second B0 episode executed 100 SONIC frames in
  MuJoCo. These checks do not measure P grasp success.
- The paired-teacher/data-conversion update passed 84 server tests, a new
  baseline reference smoke, one matched successful-clean/failed-perturbed
  simulation pair, and a single-parent loader audit. The available prompt
  groups do not support a leakage-free three-split dataset yet.
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
  final summary passed. The ordinary 30-epoch run is prepared but has not
  started. After teacher conversion changes the manifest, Risk must be
  trained on the final fixed dataset again before Residual because provenance
  checks require matching Risk/Residual datasets.
