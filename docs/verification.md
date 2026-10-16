# Verification and Evidence Boundaries

This page documents what a verification step establishes and how to preserve
its evidence. Exact training losses, dataset sizes, success rates, artifact
hashes and per-run failures belong in ignored output artifacts and the course
report. Source availability and implementation status alone do not establish
an experimental result.

## Source and artifact compatibility

Pin frozen ARDY/Kimodo sources and checkpoints plus the matching SONIC
encoder, decoder and configuration. A learned bundle also binds the tensor
schema, dataset manifest/window hashes, Risk/Residual architecture, normalizer
and gate calibration. Keep source archives within their declared packaging
scope. Keep weights, bulk datasets, raw rollouts and credentials outside Git.
Publish manuscript assets under `report/`, curated demonstrations under
`videos/`, and numerical summaries under `docs/results/`, preserving the relevant
source and video manifests.

The canonical loader verifies the frozen Risk hash used by Residual, shared
data/configuration and normalizer tensors. It accepts the gate's recorded raw
sigmoid contract and validation window identity. Native sidecar checkpoints
need their own matching source/adapter and target schema; dimensions and
renamed state keys do not establish equivalence.

## Levels of evidence

| Check | Establishes | Does not establish |
| --- | --- | --- |
| Formatting, imports or synthetic fixtures | Source/contract consistency within the checked scope | Model assets, MUSA execution or physical success |
| Primitive device operators | Those operators on the selected environment | Full ARDY, SONIC or learned-model compatibility |
| Frozen-model execution | The pinned model path executes in its recorded runtime | A completed grasp or fair method comparison |
| Dataset audit | Split, availability, target and provenance consistency | Sufficient recovery coverage or improved policy quality |
| Training/validation completion | Declared optimization and checkpoint selection finished | Improved physical task success |
| Running process or launch receipt | Dispatch or health at the recorded observation | Completed physics, admitted outcomes or finished media |
| Independently admitted trial accounting | Outcomes under the pinned task/protocol and endpoint definitions | A different cohort, controller or untested ablation |
| Saved-state rendering | Visualization of recorded states | New physics, missing hold time or an independent evaluation |

Use the matched vendor `torch`/`torch_musa` stack. CPU or CUDA success cannot
stand in for MUSA compatibility. Apply the checks appropriate to changed
interfaces; documentation edits need consistency and link checks, not model
runs. See [commands](commands.md) and `AGENTS.md` for the relevant entry points.

## Physical endpoint accounting

Preserve every scheduled original parent and paired instruction/scene/seed.
An anchor rejected before any scored arm is generated leaves a common null.
An attempted arm with a runtime/reference guard failure is an operational
failure: unsuccessful end-to-end, with no completed physical grasp endpoint.
A completed grasp failure or timeout remains a physical failure. Startup and
render failures have their own operational records and cannot become scored
physical failures.

Report end-to-end and completed physical denominators explicitly, including
null origins, operational failures, recoveries and harms. Preserve original
namespaces when a later fresh run or repair is authorized. Windows, sampled
draws and related branches are not independent trials. Follow the declared
uncertainty and multiplicity plan; a small descriptive cohort must retain its
descriptive scope.

Use the same evaluator, scene, hand control, nominal constraints, reference
checks and frozen SONIC for paired comparisons. Keep language-generator
comparisons distinct from Risk/Residual/interface comparisons. A tuned Kimodo
run under different conditions does not establish a fair ranking against
ARDY. Lower validation loss alone cannot establish improved grasp success.

## Timing, instruction and media evidence

Distinguish simulation time, wall-clock response time and inference latency.
Define the response-time endpoint before comparing methods. Model timing
events do not automatically measure text-to-first-motion latency. Report user
corrections under the actual scripted or interactive protocol.

Bind video captions to recorded user instructions or generated phase goals
and label which source supplied the text. Preserve whole-episode outcome
labels as such. A saved-state clip uses recorded control timestamps and does
not add higher-frequency contact/collision evidence. Outcome-conditioned
rescue, harm and both-failure clips illustrate mechanisms; estimate success
rates from the complete admitted cohort.

Keep failed renders, dispatch receipts and transport errors. An interrupted
archive transfer leaves remote job/archive status unknown until a later exact
receipt establishes it. Verify archive contents and native artifact hashes
before claiming a reproducible package has been delivered.

## Reported experiments and remaining scope

The [course report](https://github.com/Peneplain/dl/tree/main/report) documents
the shared 100-scene ARDY/Kimodo comparison, an independent paired ARDY-based
controller confirmation, and a separate same-weight invocation study.
[Numerical summaries](https://github.com/Peneplain/dl/tree/main/docs/results)
provide the response and correction metrics with source provenance;
[demonstration videos](https://github.com/Peneplain/dl/tree/main/videos) show
representative successes and failures. The report also describes the Archon
workflow. Keep these cohorts and their endpoint definitions distinct.

The full risk-interface, component/loss and multiple-training-seed comparisons
remain separate from the completed invocation study. Source controls alone do
not establish those experiments. Camera perception, hardware deployment,
bimanual transport, joint fine-tuning and a LeRobot data export remain outside
the reported implementation.

Keep the approved proposal frozen, describe implementation changes in
[README.md](../README.md) and the relevant existing documents, and preserve
experimental results in their original artifact records. Acknowledge external
models, code, datasets and LLM assistance according to the course policy.
