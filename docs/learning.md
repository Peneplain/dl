# Risk and Residual implementation status

`docs/proposal.tex` defines the research experiment. This package adds the
learning architecture and auditable training interfaces; it does not establish
that the proposed controller improves a physical grasp. Settings in
`configs/learning/p.json` are initial defaults pending pilot calibration.

## Implemented code

| Path | Responsibility |
| --- | --- |
| `risk_residual/config.py` | Versioned tensor schema, clocks and model settings |
| `risk_residual/models.py` | Predictive Risk and bounded arm-only Residual Transformers |
| `risk_residual/losses.py` | Masked risk, correction, identity and smoothness losses |
| `risk_residual/data.py` | Parent-split audit, nominal labels and normalization |
| `risk_residual/checkpoints.py` | Schema, baseline and checkpoint provenance validation |
| `risk_residual/runtime.py` | Gated prediction and optional correction provider |
| `baseline/reference_checks.py` | Shared arm mask, joint and offset-rate limits |
| `experiments/train.py` | Risk training, then Residual with predicted frozen-Risk tokens |
| `experiments/calibrate.py` | Validation-only gate-threshold selection |
| `experiments/statistics.py` | Wilson intervals and paired episode bootstrap |
| `experiments/smoke.py` | Small synthetic two-stage pipeline check |
| `risk_residual/teacher.py` | Bounded arm perturbation and activation-state fingerprint |
| `experiments/collect_pair.py` | Two-branch, saved-reference teacher collection pilot |
| `experiments/build_dataset.py` | Audited nominal rollout to versioned NPZ windows |

Risk uses 16 execution samples at 50 Hz, spanning 300 ms, and eight nominal
samples 40 ms apart, spanning 0–280 ms. The latter is distinct from SONIC's
longer lookahead and ARDY's generation horizon. The schema has 119 history
fields: 29 joint positions, 29 joint velocities, a root quaternion, 3 angular
velocities, 14 hand/object relative-pose coordinates, 2 contacts, 29 tracking
errors and 9 phase indicators. The nominal reference has 62 fields; future
context has 9 phase indicators and 14 planned finger commands. Joint order is
`baseline.adapters.joints.ISAACLAB_JOINT_NAMES`. The phase vocabulary is
`stand`, `approach`, `settle`, `prepare`, `reach`, `lower`, `close`, `lift`,
`hold`; a future observation builder must map `lower_replan_*` to `lower`.

Risk produces 8 x 4 x 32 learned tokens plus auxiliary tracking, contact,
balance and intervention predictions. Residual outputs bounded corrections
for only the 14 arm joints. The optional provider samples nominal references
on the risk clock; shared checks apply arm and rate limits and recompute
velocities before SONIC. Inference inputs contain no future executed state,
teacher action or risk labels.

## Data contract and remaining integration

The dataset is `manifest.json` plus `train.npz`, `val.npz` and `test.npz`.
Parents and all nominal/counterfactual/teacher branches must be split before
window extraction. Prompt groups and scene seeds remain disjoint across splits.
The loader requires nominal-branch risk labels and matching verified teacher
snapshots for correction targets. Contact labels are phase dependent; missing
future negative labels are masked. Normalization is fitted on training inputs
only. Synthetic fixtures carry `synthetic_inputs=true` and cannot be loaded as
task checkpoints by the production loader.

The first rollout-to-window converter and controlled paired-teacher pilot are
implemented and have been checked on one real simulation pair. A full
train/validation/test dataset has not been collected. Arbitrary mid-episode
checkpoint restoration, a general corrective teacher, and the P evaluation
entry point remain unimplemented.
`scripts/run.py` does not yet launch P trials. No P success rate, latency or
physical result is claimed. Shared checks limit offsets but do not yet
implement a geometric clearance check.

### Collecting a paired pilot

Choose a completed B0 grasp attempt that contains saved ARDY references in
`ardy/<phase>/reference.npz`. In the supported S4000 container, run:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.collect_pair \
  --source output/EXISTING/attempt-00001 \
  --out output/teacher-pair-NEW \
  --phase lower --delay .2 --joint right_wrist_pitch_joint --amplitude .1
```

The command runs two full episodes from the same initial condition with the
same frozen SONIC and saved ARDY references. The `teacher` branch keeps the arm
reference clean; the `nominal` branch introduces the bounded offset at the
selected decision. Both branches pass through the same reference checks,
SONIC, hand controller and MuJoCo scene. `pair.json` contains each report hash,
the intervention frame and a fingerprint of the physical state, SONIC history,
reference buffer and correction limiter immediately before the offset. A pair
is usable only when the fingerprints match. `teacher_verified` records clean
task success. `recovery_verified` additionally requires the perturbed nominal
branch to fail in an otherwise valid physics/SONIC trial. At the matching
decision, that verified outcome supplies a direct positive intervention label
and an arm-only correction target, even if the short nominal future does not
cross an auxiliary tracking/contact/balance threshold. Auxiliary targets still
come only from nominal execution. The clean branch is also converted as an
independent nominal rollout under the same parent split, providing stable
zero-offset identity windows when its future remains low risk. If both branches
succeed, the pair can provide Risk and stable identity windows but no recovery
target. A failed clean branch likewise supplies no Residual target.
This replay-verified pilot only covers one controlled arm perturbation from a
matched pre-intervention state. It does not restore an arbitrary mid-episode
checkpoint or prove that a teacher can recover an already diverged state.

The per-branch `nominal_context.csv` records the uncorrected reference;
`effective_context.csv` records the reference delivered to SONIC after the
shared limits. The latter is used as the nominal input for the perturbed
counterfactual. Neither stream contains future executed states or teacher
actions. Both branches save `task.csv` and the 50 Hz MuJoCo rollout.

### Converting to training windows

Declare parent splits **before** extraction. Each split needs at least one
parent, and prompt paraphrase groups and scene seeds must not cross splits.
For example, save a source plan at an ignored output path:

```json
{
  "parents": [
    {"parent_id":"episode-a","split":"train","prompt_group":"group-a","scene_seed":12,"nominal":"output/PAIR-A/nominal","pair":"output/PAIR-A/pair.json"},
    {"parent_id":"episode-b","split":"val","prompt_group":"group-b","scene_seed":16,"nominal":"output/PAIR-B/nominal","pair":"output/PAIR-B/pair.json"},
    {"parent_id":"episode-c","split":"test","prompt_group":"group-c","scene_seed":18,"nominal":"output/PAIR-C/nominal","pair":"output/PAIR-C/pair.json"}
  ]
}
```

Use real, distinct prompt groups and scene seeds; the placeholders above are
not evidence of a valid split. `scene_seed` must equal the nominal request
seed, and the converter rejects identical instructions assigned to different
splits. Paraphrase grouping still requires human review. For Risk-only data,
omit `pair` and point
`nominal` to a B0 attempt with `nominal_context.csv`, `task.csv` and a complete
rollout. A single-parent pilot can be audited without creating a dataset:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.build_dataset --sources output/pilot-sources.json \
  --tracking-thresholds .02 .02 .1 .1 --inspect-only
```

This reports window, positive-risk, correction and stable-identity counts. The
`.02` and `.1` values are exploratory inspection thresholds, not frozen gate
calibration. A single-parent inspection does not meet the train/validation/test
split requirement. Once distinct,
audited parents exist for all three splits, convert with explicit tracking
thresholds in radians:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.build_dataset \
  --sources output/source-plan.json --out output/dataset-NEW \
  --tracking-thresholds .1 .1 .1 .1
```

The `.1` thresholds are an illustrative input, not calibrated values. The
converter reads 16 past 50 Hz states (300 ms span) and resamples the current
10-slot SONIC reference to eight 40 ms predictions (280 ms span). It rejects
phase or finger-command changes within a window instead of inserting future
observations into model inputs. Pre-decision MuJoCo states supply block pose,
virtual palm-relative poses and hand contacts. The palm centers are fixed
mirrored offsets from wrist-yaw frames, not measured tactile sites. Nominal
future execution supplies body tracking, balance and right-hand grasp-contact
labels. Contact loss is supervised only during lift and hold. Unavailable
future labels are masked; partial futures cannot establish negative
intervention labels. Only the matching intervention decision can carry a
verified teacher-minus-nominal arm offset. Other successful stable windows
can supply zero-offset identity targets, disjoint from correction windows.

The output is `manifest.json`, `train.npz`, `val.npz`, `test.npz` and
`report.json`. The manifest records source and scene hashes, split parents,
thresholds, source file hashes and decision-time bounds. The converter loads
all three splits through `WindowDataset` before reporting success. Review
`correction_samples` and `stable_samples` in `report.json` before training;
the existence of an NPZ file alone does not establish usable correction data.

## Checks and training commands

```bash
python -m unittest discover -s tests -v
python -m experiments.smoke --device cpu --out output/p-smoke-new
```

Keep the existing vendor `torch`/`torch_musa` stack on S4000. After a real
verified dataset exists, run the training stages in the supported environment:

```bash
python -m experiments.train --stage risk --device musa --seed 0 \
  --data datasets/task/manifest.json --out output/risk-seed0
python -m experiments.calibrate --device musa \
  --risk output/risk-seed0/best.pt --data datasets/task/manifest.json \
  --out output/risk-seed0/gate.json
python -m experiments.train --stage residual --interface P --device musa --seed 0 \
  --data datasets/task/manifest.json --risk output/risk-seed0/best.pt \
  --out output/p-seed0
```

The initial gate rule selects maximum validation F1 on the 0–1 grid at 0.01
steps, resolving ties toward fewer activations then a higher threshold. Freeze
the rule and threshold before testing. B1/B2/I1 use matched zero risk-feature
slots; I2/I3/I4/P use scalar, body, body-time and learned features with the
same decoder capacity. These are model/training interfaces, not completed
comparison results.
