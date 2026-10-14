# Current Project Context

Read this page, inspect Git status and then read task-relevant source and
artifacts. Keep this summary about current capabilities and constraints;
per-run results and diagnostics belong in ignored output artifacts.

## Workspace and entry points

- Server repository: `/home/group3/dl` on the S4000 worker.
- Generated artifacts: `output/`, resolving to `/data/group3/dl-output/`.
- Host entry point: `./run.sh`, with the matched vendor MUSA environment.
- Keep model assets, upstream checkouts, rollouts, videos and credentials out
  of Git. Preserve original manifests and measured evidence.

## Implemented system

Frozen ARDY is the default generator; `--ardy` and `--kimodo` select distinct
frozen models. Both use simulator-state task grounding, phased English goals,
named G1 references, shared checks, frozen SONIC, articulated fingers and
free-root MuJoCo physics with a dynamic block.

B0 is nominal ARDY + SONIC. P adds Predictive Risk and arm-only Residual
correction through the shared executor. Risk reads measured history, future
nominal references and causal phase/finger context; it produces body/time
features and an intervention score. Residual uses predicted Risk features and
modifies only the arm reference. Low risk skips Residual inference, requests
zero offset and returns any existing correction to zero under the shared rate
limit. SONIC continues to execute lower-body and nominal whole-body motion.

The learning package supports masked nominal-future Risk supervision,
controlled same-state teacher pairs, stable identity targets and supervised
Residual training with frozen Risk. Checkpoints embed training-input
normalizers and bind dataset, schema, source and Risk provenance. The canonical
loader rejects incompatible bundles; native sidecar artifacts also require
their exact architecture and adapter source.

## Scope and evidence constraints

The implemented task is simulated right-hand tabletop block grasp/lift with
privileged object grounding. GR00T integration, camera perception, bimanual
transport, hardware deployment, arbitrary natural-failure teacher recovery and
joint fine-tuning are outside the current implementation. The approved
`docs/proposal.tex` remains frozen.

Keep related branches under their original parent split. Preserve raw recorded
targets, unavailable-label masks, normalizer provenance, frozen gate decisions,
shared limits and task criteria. The teacher supplies training supervision and
must not participate in evaluation. Distinguish nominal anchor rejections,
attempted-arm operational failures and completed physical outcomes.

Reusable risk interfaces and no-gate controls exist, but source availability
is not evidence of a completed fair ablation. Remaining course work includes a
matched ARDY/Kimodo comparison, timing and correction metrics, representative
videos, Archon workflow evidence and report/demo publication. Per-run training
losses and task rates remain in their immutable artifact records.

Use [commands](commands.md), [learning](learning.md), [verification](verification.md)
and [requirements](track4_requirements.md) for the relevant workflow.
