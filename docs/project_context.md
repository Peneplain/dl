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
