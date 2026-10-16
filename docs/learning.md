# Risk and Residual Learning

The learning package adds predictive correction to the frozen ARDY + SONIC
simulation workflow. Training and deployment use separate stages: train Risk,
freeze it, then train Residual with predicted Risk features. Joint fine-tuning
is deferred. This page documents the reusable learning interfaces; the
[course report](https://github.com/Peneplain/dl/tree/main/report) records the
selected native training, data counts, checkpoint selection and physical
evaluation. [Numerical summaries](https://github.com/Peneplain/dl/tree/main/docs/results)
retain the public results and source provenance. Preserve complete per-run results
and selected-artifact provenance in their original output artifacts.

## Package boundaries

| Module | Responsibility |
| --- | --- |
| `risk_residual/config.py` | Tensor schema, clocks and model settings |
| `risk_residual/models.py` | Structured Risk and bounded arm-only Residual Transformers |
| `risk_residual/losses.py` | Masked auxiliary, intervention, correction, identity and smoothness losses |
| `risk_residual/data.py` | Input/target audit and train-only normalization |
| `risk_residual/checkpoints.py` | Model, dataset, normalizer and gate compatibility |
| `risk_residual/runtime.py` | Causal features, gate and receding-horizon corrections |
| `baseline/reference_checks.py` | Shared arm mask, joint, offset and rate limits |
| `experiments/plan_data.py` | Original-parent split and immutable collection plans |
| `experiments/collect_pair.py`, `collect_pairs.py` | Controlled verified teacher pairs |
| `experiments/build_dataset.py`, `audit_dataset.py` | Window extraction and supervision readiness |
| `experiments/train.py`, `evaluate_risk.py` | Staged training and validation-selected gates |
| `experiments/run_policy.py`, `statistics.py` | Closed-loop evaluation and episode-level analysis |

B0 requires no learning-package imports or correction checkpoints. Learning
uses the shared baseline reference, controller, finger, physics and evaluator
interfaces.

## Causal tensors and clocks

Risk receives 16 measured samples at 50 Hz, spanning 300 ms, and eight nominal
samples 40 ms apart, spanning 0–280 ms. These are separate from ARDY's generation
horizon and SONIC's observation-configured lookahead.

| Tensor | Shape per sample | Contents |
| --- | --- | --- |
| History | `16 × 119` | Joint positions/velocities, root wxyz quaternion, angular velocity, object-relative poses, hand contacts, tracking errors and phase |
| Nominal future | `8 × 62` | 29 joint positions, 29 velocities and root wxyz quaternion |
| Planned context | `8 × 23` | Nine phase indicators and 14 finger commands |
| Structured Risk | `8 × 4 × 32` | Future-time features for left arm/hand, right arm/hand, torso and lower body |
| Residual | `8 × 29` | Fourteen arm coordinates; all other offsets are zero |

Joint order is `ISAACLAB_JOINT_NAMES`. Inference uses current/past execution
and future planned references. Future executed states, teacher actions,
correction targets and ground-truth risk labels never enter inference inputs.
The canonical converter keeps causal phase/finger context fixed over an
eligible window; runtime repeats the current commanded context to match it.

Risk predicts tracking, phase-dependent contact, balance and intervention.
Auxiliary labels come from nominal future execution. A verified same-state
recovery may add an intervention positive without replacing those auxiliary
labels. Open-hand phases do not supervise grasp loss. Missing future targets
are masked; an incomplete future with no observed violation is censored.

## Data and teacher contract

Split original parents before extracting windows. Keep nominal, perturbation
and teacher branches together; preserve held-out scene seeds and reviewed
prompt families. Window counts and repeated sampler exposure do not add
independent episodes.

The canonical teacher pairs a saved clean ARDY reference with a perturbed
nominal branch under the same SONIC, fingers and physics. A pre-divergence
activation fingerprint binds physical state, controller history, reference
buffers and limiter state. Recovery supervision requires matching activation
state, actual branch execution, clean task success and perturbed task failure.
Both-success pairs do not supply recovery targets; failed clean branches do
not supply corrective demonstrations. This controlled corruption workflow
does not establish recovery from an arbitrary already-diverged failure.

The correction target is the recorded teacher-minus-nominal arm reference.
Stable nominal samples have an identity target of zero, and correction and
identity sets are disjoint. Preserve recorded targets and availability masks.
The loss compares predictions with `offset_target` without clipping or
normalizing the target. The canonical `WindowDataset` requires available
correction targets to satisfy its declared magnitude, arm mask and identity
contract and rejects violations. A raw native dataset with another target
scale needs its own declared loader/architecture contract; silently clipping
its labels changes the learning problem.

Plan and index a collection through the supported host container:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.plan_data plan \
  --config configs/data/collection.json --out output/data-collection-new

MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.plan_data index \
  --collection output/data-collection-new/collection-plan.json \
  --out output/data-index-new

MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.collect_pairs \
  --sources output/data-index-new/sources.json \
  --out output/teacher-pairs-new --limit 3
```

Planning runs no physics. Execute the saved immutable commands with their
recorded source/configuration contract. A resumed collection checks those
pins. Completed failures, invalid reference replays and rejected candidates
remain recorded; unavailable replans cannot be invented to create recoveries.

Freeze train-only label calibration before conversion. Pass the four tracking
thresholds explicitly in left-arm, right-arm, torso and lower-body order.
Do not tune labels on validation/test or change them to force class balance.
Convert into a fresh dataset directory and run its audit before training.
Risk readiness requires observed positive and negative labels; Residual also
requires verified correction and stable identity samples in train/validation.
Readiness establishes availability, not adequate data coverage.

## Staged training and checkpoint selection

Inside `./run.sh shell`, use a fresh audited dataset and fresh output paths:

```bash
python -m experiments.train --stage risk --device musa:0 --seed 0 \
  --config configs/learning/risk-quality-pilot.json \
  --data output/dataset-new/manifest.json --out output/risk-new

python -m experiments.evaluate_risk --device musa:0 --split val \
  --risk output/risk-new/best.pt --data output/dataset-new/manifest.json \
  --out output/eval-val-new

python -m experiments.train --stage residual --interface P --device musa:0 --seed 0 \
  --config configs/learning/p.json --risk output/risk-new/best.pt \
  --data output/dataset-new/manifest.json --out output/residual-new

python -m experiments.evaluate_risk --device musa:0 --split test \
  --risk output/risk-new/best.pt --data output/dataset-new/manifest.json \
  --calibration output/eval-val-new/gate.json --out output/eval-test-new
```

Residual training verifies the Risk configuration, manifest and window hashes,
sets Risk to evaluation mode with gradients disabled, and computes predicted
features under `no_grad`. P uses the complete learned body/time tokens. The
Residual projects these features into the future nominal tokens of its
Temporal Transformer; this is additive conditioning, not a separate
cross-attention decoder. Training optimizes bounded predictions before runtime
gating, using correction, zero-offset identity and temporal smoothness losses.

Only history and nominal continuous inputs use fitted normalization. Fit it
on the full natural training inputs, preserve categorical/binary encoding and
context units, and store the tensors inside each checkpoint. Residual reuses
the frozen Risk normalizer. A runtime bundle must match its baseline lock,
schema, dataset, Risk hash, normalizer and gate calibration; a label-audit
certificate is a separate artifact from a deployment gate.

Select checkpoints by the declared full validation objective. `budget_epochs`
is a cap, `actual_epochs` records completed work, and `best_epoch` identifies
`best.pt`. Record actual updates, stopping reason, learning rates and component
losses. Validation-only schedulers and early stopping change the completed
budget; they do not change the objective. Test data is reserved for final
reporting.

## Sampling and interface controls

Risk defaults to natural sampling. Optional `balanced_intervention` changes
valid positive/negative exposure. Optional `verified_recovery` assigns a
declared recovery mass equally across original origins, then equally across
each origin's windows; the background retains the declared remaining mass.
These options preserve labels, auxiliary masks, the natural training
normalizer and full validation windows. Censored samples retain their stated
supervision scope. Record sampling mode, draw budget, group mass and expected
repeats; expectations are not realized draws or new independent recoveries.

Residual samples the disjoint correction and identity sets with the configured
mass and replacement. `training.residual_samples_per_epoch`, when supplied,
sets its draw count. Validation uses all eligible correction/identity windows
without resampling. A lower draw budget changes optimizer exposure and compute;
it is not additional data or an equal-compute comparison.

Risk-interface adapters preserve the same body/time feature slots and decoder
capacity. Binary-gate, scalar, body-wise, time-wise and learned-token interfaces
need matched training/data/gate protocols before their outcomes are compared.
The no-gate control reuses the trained networks without retraining: Risk still
runs and Residual receives its full predicted token tensor. The report's
invocation study shares weights between its selective and always-on arms; its
checkpoints and cohort are distinct from the independent confirmation.
Available source interfaces are not completed ablation results.

## Runtime and portability

The canonical runtime updates Risk/Residual at 10–20 Hz against shared 50 Hz
SONIC references. Only the current correction sample is committed; the next
cycle observes the new robot state. Low risk skips Residual inference and
requests zero offset. The shared limiter brings any existing offset back to
zero while respecting arm-only authority, joint limits, magnitude and rate.
Canonical defaults are 0.15 rad magnitude and 0.5 rad/s offset rate; checkpoint
and runtime settings must remain bound to their declared configuration.
Velocities and dependent reference fields are recomputed after correction.

Native sidecar checkpoints may use distinct encoder pooling, heads, output
bounds or raw-target schemas. Preserve the matching source and adapter rather
than converting keys or copying metadata to force canonical compatibility.
A passing shape or primitive operator check does not establish compatibility
of a full motion generator, SONIC or a learned controller on MUSA.

See [commands](commands.md#load-the-trained-p-controller) for host execution
and [verification](verification.md) for evidence accounting. Record gate
activation, input/inference latency, history gaps and desired/applied offsets
in the ignored run artifacts. General natural-failure recovery, joint
fine-tuning, perception and hardware deployment remain separate work.
