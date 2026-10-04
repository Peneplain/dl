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

The full rollout collector, restorable branch state, verified correction
teacher and real observation-window builder are **not implemented**. B0's
`nominal_context.csv` contains useful 50 Hz inputs but is not itself a
training dataset. `ReferenceCorrectionProvider` is an optional interface;
`scripts/run.py` does not yet construct it or launch P trials. No P success
rate, latency or physical result is claimed. Shared checks limit offsets but
do not yet implement a geometric clearance check. Complete these items before
a fair B0/P physical comparison.

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
