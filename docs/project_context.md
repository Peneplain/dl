# Current Project Context

Read this page, inspect Git status and then read task-relevant source/artifacts
before starting work. Keep this a concise description of the current system;
per-run progress and immutable diagnostics belong in output artifacts.

## Canonical workspace

- Server repository: `/home/group3/dl` on the S4000 worker.
- Generated artifacts: repository `output/`, resolving to
  `/data/group3/dl-output/` on the larger storage volume.
- Supported entry point: `./run.sh` from the host with the existing vendor
  MUSA stack and `.venv-baseline-musa`.
- Keep weights, rollout data, videos, upstream source trees and credentials
  outside Git. Preserve measured evidence before any cleanup.

## Implemented system

`--ardy` and `--kimodo` select distinct frozen motion generators. The shared
executor provides simulator-state grounding, phased English prompts, named
G1 references, SONIC, articulated fingers, free-root MuJoCo physics, recording,
rendering and the evaluator. ARDY remains the default for old commands/plans.

B0 is nominal ARDY–SONIC; P adds the trained Predictive Risk and arm-only
Residual controller through the same batch/manual entry. Kimodo K0 is an
additional frozen generator comparator. P currently supports ARDY only and
requires a matched Risk, supervised Residual and validation gate. B0 does not
require learned artifacts.

The current study is simulation-only and uses privileged object state.
Camera perception, GR00T N1.7 integration, hardware deployment, general
natural-failure teacher recovery and joint fine-tuning are not implemented.
`docs/proposal.tex` stays frozen; document implementation and experimental
limits in the other pages.

## Current trained artifacts

All paths below are under `output/`:

```text
risk-quality-final-261007f/dataset/manifest.json
risk-quality-retrain-261007f/risk/seed-0/best.pt
risk-quality-retrain-261007f/residual/best.pt
risk-quality-retrain-261007f/eval-val/gate.json
risk-quality-retrain-261007f/final-label-gate.json
```

The runtime gate is `eval-val/gate.json` at 0.31; `final-label-gate.json` is a
separate label/data audit certificate. Selected checkpoints and the manifest
must retain their recorded hashes. [Verification](verification.md) lists exact
hashes, actual training epochs, offline metrics and physical trial results.

The final dataset has 82,060/29,248/29,497 train/validation/test windows and
30/9/13 correction samples. Related branches are not independent original
parents. Risk seeds 0 and 1 completed, seed 0 was selected on validation,
and supervised P Residual training with frozen Risk completed.

The original batch integration executed actual SONIC/physics and nonzero
bounded corrections. The retained 60-pair repeat achieved 23/60 for both B0
and P, so no aggregate grasp-success improvement is established. The tuned
Kimodo 20-trial result is 11/20 and is not a matched held-out comparison.

## Working invariants and next research

Keep ARDY, Kimodo and SONIC sources/assets frozen. Preserve English project
content, named mappings, causal history/lookahead, original-parent split
ownership, missing-label masks and model/gate/data provenance. Invalid replay
execution must remain excluded, and all valid physical failures/timeouts remain
in the evaluation denominator. Do not relax success criteria or select only
favorable seed groups.

Next useful work is to diagnose the discordant B0/P trials and increase
independent recovery supervision before testing a predeclared validation
change. The proposal's controls, multi-trained-seed/perturbation budget and
ablations remain incomplete. For the course submission, also finish a fair
ARDY/Kimodo comparison, representative videos, Archon workflow evidence and
the report/demo. Use [commands](commands.md), [learning](learning.md) and
[requirements](track4_requirements.md) as the operational references.
