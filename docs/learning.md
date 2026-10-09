# Risk and Residual Learning

The learned controller augments the frozen ARDY–SONIC reference workflow.
Risk is trained first, then frozen; Residual is trained with **predicted Risk
features**, not ground-truth labels. Both models are trained and physically
integrated. [Verification](verification.md) records the current artifacts and
results. Training availability and offline accuracy do not establish a grasp
success advantage.

## Package boundaries

| Module | Responsibility |
| --- | --- |
| `risk_residual/config.py` | Versioned tensor schema and model settings |
| `risk_residual/models.py` | Risk and bounded arm-only Residual Transformers |
| `risk_residual/losses.py` | Masked auxiliary, correction, identity and smoothness losses |
| `risk_residual/data.py` | Split audit, labels and train-only normalization |
| `risk_residual/checkpoints.py` | Baseline/schema/dataset/checkpoint provenance |
| `risk_residual/runtime.py` | Causal features, gating and online corrections |
| `baseline/reference_checks.py` | Shared arm mask, joint limits and offset/rate limits |
| `experiments/plan_data.py` | Split planning and batch indexing |
| `experiments/collect_pair.py`, `collect_pairs.py` | Verified controlled clean/perturbed pairs |
| `experiments/build_dataset.py`, `audit_dataset.py` | NPZ extraction and supervision readiness |
| `experiments/diagnose_labels.py`, `calibrate_labels.py` | Continuous label diagnosis and train-only calibration |
| `experiments/train.py`, `evaluate_risk.py` | Training, frozen validation gate and test metrics |
| `experiments/run_policy.py`, `statistics.py` | Closed-loop trials and episode-level uncertainty |

Baseline-only execution does not require learned checkpoints. The learning
package uses the shared baseline interfaces; it must not copy a separate
physics controller, task evaluator or finger policy.

## Causal tensor and clock contract

Risk receives 16 measured samples at 50 Hz (300 ms history) and eight nominal
samples 40 ms apart (0–280 ms). This is separate from SONIC's ten-sample
0–0.9 s lookahead and ARDY's generation horizon.

| Tensor | Shape | Contents |
| --- | --- | --- |
| History | `16 × 119` | 29 joint positions, 29 velocities, root wxyz quaternion, 3 angular velocities, 14 object-relative pose fields, 2 hand contacts, 29 tracking errors, 9 phase indicators |
| Nominal future | `8 × 62` | 29 joint positions, 29 velocities, root wxyz quaternion |
| Planned context | `8 × 23` | 9 phase indicators and 14 planned finger commands |
| Learned risk tokens | `8 × 4 × 32` | Time/body features for left arm/hand, right arm/hand, torso and lower body |
| Residual | `8 × 29` with arm mask | Only 14 arm coordinates may be nonzero |

Joint order is `ISAACLAB_JOINT_NAMES`. The phase vocabulary is stand,
approach, settle, prepare, reach, lower, close, lift and hold; lower replans map
to lower. Inference sees current/past executed state and current planned
reference only. No future executed state, teacher action, correction target,
future contact or ground-truth Risk label enters the models.

`nominal_context.csv` records causal inputs at 50 Hz. `task.csv` contains
200 Hz privileged object/contact/clearance labels. Pre-decision MuJoCo states
provide current relative poses, contact and balance information. Pair
collection records `effective_context.csv` separately for perturbed references
actually delivered through shared limits. The converter pins report, context,
task, rollout and pair hashes rather than silently mixing these sources.

Risk has tracking, phase-dependent contact, balance and intervention heads.
Labels use nominal future execution; a verified same-state recovery adds a
separate direct intervention positive without replacing auxiliary targets.
Open-hand phases are not contact-loss positives. Missing future targets are
masked; a partial future with no observed violation is censored, not a negative.
Normalization is fitted on train inputs only.

## Dataset and split rules

A dataset contains `manifest.json`, `train.npz`, `val.npz`, `test.npz` and
conversion/audit reports. The manifest pins schema, source parents, split
assignment, evidence and window hashes. NPZ files hold model inputs,
supervision targets, availability masks and provenance identifiers.

Declare splits **before window extraction**. Keep every original episode and
its nominal, counterfactual and clean teacher branches in the same split.
Hold out scene seeds and reviewed prompt paraphrase families. Effective text
comparison catches identical instructions under renamed groups; it cannot
prove semantic independence, so paraphrase grouping still needs review.
Source-entry and window counts must not be reported as independent episodes.

The final dataset preserves older dense-velocity rollout sources and current
sparse-velocity sources with their cohort provenance. Calibration used
train-only stable episodes with the current controls. This mixed cohort and
its narrower no-walk task distribution must be reported; do not relabel old
control/replay results as current matched B0/P trials.

## Prepare a fresh collection

Run data modules through the supported container from the host:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.plan_data plan \
  --config configs/data/collection.json --out output/data-collection-new
```

The configuration fixes six English prompt groups and 200 parent episodes:
120 train, 40 validation and 40 test. It uses ARDY, no walking, .20 m standoff
and 10 s hold. Review/pilot the prompt groups before launching a large batch.
The planner writes immutable batch plans and `commands.txt`; planning itself
runs no physics. Execute the saved batch commands, which use `--resume` with
no overrides. Create a fresh plan after source changes; do not edit old locks,
move completed parents between splits or retry completed failures until success.

Index the completed plans:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.plan_data index \
  --collection output/data-collection-new/collection-plan.json \
  --out output/data-index-new
```

The index verifies exact seeds, instructions and evidence. It reports pending,
too-short, malformed and invalid attempts. Complete physical task failures with
sufficient history remain Risk candidates. Success only marks an episode as a
teacher candidate; it does not create a correction target.

## Collect verified teacher pairs

The supported teacher uses the **clean frozen ARDY reference** from a completed
source. A nominal branch introduces a bounded arm perturbation; a clean branch
replays the same saved references through SONIC and the shared hand/physics
system. Both full episodes begin from the same reset. An activation fingerprint
checks physical state, SONIC history, buffers and correction limiter immediately
before perturbation.

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.collect_pairs \
  --sources output/data-index-new/sources.json \
  --out output/teacher-pairs-new --limit 3

MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.collect_pairs \
  --sources output/data-index-new/sources.json \
  --out output/teacher-pairs-new --resume --limit 20
```

The default perturbation plan proposes bounded right-shoulder offsets in lower
and a right-elbow offset in lift. A lock prevents concurrent queue writers.
Resume verifies unchanged sources, references, perturbations and collector
hashes. Completed failed/rejected candidates remain recorded.

A usable recovery requires matching activation fingerprints, actual SONIC and
physics execution, clean branch task success and perturbed branch task failure.
At the verified activation, the correction target is clean minus perturbed
arm reference. Both-success pairs supply no recovery target. Failed clean
branches supply no Residual correction. Successful clean nominal futures may
supply zero-offset identity samples, under the same original-parent split.

This is controlled corruption supervision at a matched pre-divergence state.
It does not restore an arbitrary mid-episode state or demonstrate recovery from
an already-diverged natural failure. Feedback-driven nominal replay may ask
for a saved replan that does not exist; such execution is invalid and excluded.
Do not invent the missing reference or call the invalid branch a verified
recovery. Missing references produce nonrandom replay censoring, so report
attempted, valid, rejected and independent recovery-parent counts separately.

## Calibrate labels, convert and audit

Diagnose continuous execution signals on train/validation without modifying
rollouts. The label calibrator uses train per-original-episode maximum-horizon
tracking p95, then across-episode p95, rounded upward to .01 rad and never below
existing floors. Require at least ten independent stable train episodes with
current controls. Freeze this exploratory artifact before conversion and
check the unchanged validation physical stable-hold guard. Do not adjust
thresholds to force class balance or choose them on validation/test.

The current frozen candidate is `[.14, .15, .10, .13]` rad in left-arm,
right-arm, torso, lower-body order. After freezing a source plan and candidate, convert into fresh output paths.
The example below uses the present candidate and placeholder source paths:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.build_dataset \
  --sources output/teacher-pairs-new/sources.json --out output/dataset-new \
  --tracking-thresholds .14 .15 .10 .13 --skip-ineligible --require residual

MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.audit_dataset --data output/dataset-new/manifest.json \
  --require residual --out output/dataset-audit-new.json
```

These conversion and supervision-availability commands do not run the physical
stable-hold label guard. Verify that guard separately before quality-gated training.

For a new collection, repeat train-only calibration and preserve its artifact;
these numerical values are not a universal label setting. `--inspect-only`
inspects windows before writing a dataset. `--skip-ineligible` reports short
or windowless sources; malformed streams or changed pair evidence still fail.
Conversion does not run physics.

Risk readiness requires observed positives and negatives in train/validation.
Residual readiness also requires verified corrections and stable identity
samples in both. Availability alone does not establish adequate data size.
A new manifest invalidates prior model/gate compatibility and requires
retraining before supervised Residual training.

## Train Risk, freeze, then train Residual

Inside `./run.sh shell`, with a fresh verified dataset:

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

Risk uses natural sampling and early stopping on validation total loss with a
30-epoch maximum in `risk-quality-pilot.json`. Record actual epochs, best epoch
and stop reason; the budget is not the completed epoch count. Residual uses
supervised correction MSE plus zero-offset identity and temporal smoothness,
with explicit balanced sampling of correction/identity supervision. Risk is
frozen and supplies predicted tokens throughout Residual training.

For separate seeds on idle physical GPUs, run from the host:

```bash
bash scripts/train_risk_multiseed.sh 2,3 output/dataset-new/manifest.json \
  configs/learning/risk-quality-pilot.json output/risk-multiseed-new
```

This launches independent models, not DDP. Select a Risk seed by minimum best
**validation** total loss. The gate selects maximum validation F1 on a .01
probability grid, ties favoring fewer activations then higher threshold. Keep
raw sigmoid, the selected threshold, checkpoint, train normalizer and manifest
hashes frozen for test/physical evaluation. Test data is used only for final
reporting. Optional probability scaling, sampling changes or dropout experiments
are separate ablations.

## Online integration and remaining research

The original batch/manual workflow loads the matched Risk/Residual/gate set
with `--ardy`; see [commands](commands.md#load-the-trained-p-controller).
Risk/Residual updates at 10–20 Hz. Low risk skips Residual inference and requests
zero offset; the shared limiter ramps existing correction back to zero.
Offsets are limited to .15 rad, use the 14-arm mask and the shared .5 rad/s
rate limit. Checks recompute velocities and preserve nominal non-arm fields.
Record gate decisions, latency, history gaps and desired/applied corrections.

The proposal's B1/B2/I1/I2/I3/I4 controls, risk-interface and loss/component
ablations remain a separate evaluation budget. Available interfaces are not
completed comparison results. Joint fine-tuning, real hardware, camera
perception, general natural-failure teacher recovery and geometric clearance
prediction remain outside the verified implementation. Current closed-loop
P has not improved aggregate matched grasp success.
