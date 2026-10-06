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
| `experiments/plan_data.py` | Fixed split planning and indexing existing B0 batches |
| `experiments/collect_pairs.py` | Resumable multi-source controlled pair collection |
| `experiments/audit_dataset.py` | Per-split supervision counts and readiness checks |
| `scripts/train_risk_multiseed.sh` | Independent Risk seeds on separate physical MUSA GPUs |

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

The rollout-to-window converter and controlled paired-teacher pilot are
implemented and have been checked on one real simulation pair. Batch planning,
indexing, resumable pairing and supervision audits are implemented separately
from B0. These tools do not establish that new prompt groups or perturbations
produce successful physical trials. The October 6 nominal collection now
provides real train/validation/test Risk windows, but no verified correction
samples for Residual training. Arbitrary mid-episode
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

## Batch data workflow

The commands below run from `/home/group3/dl` on the host. Planning, indexing,
inspection and auditing load no policy and execute no physics. Pair collection
executes two physical/controller branches for each selected source and variant;
invoke it only when ready to collect. The tools use existing Python dependencies
and local frozen assets; they do not download models. Run the existing offline
asset check before collection when asset availability is uncertain.

### 1. Fix splits before collection

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.plan_data plan \
  --config configs/data/collection.json --out output/data-collection-NEW
```

The proposed configuration contains 200 parent episodes: 120 train, 40 val and
40 test, in six explicitly authored English phase-prompt groups. Seed ranges
are 10000--10059 and 11000--11059 for train, 20000--20019 and 21000--21019 for
val, and 30000--30019 and 31000--31019 for test. The physical settings use the
current no-walk .20 m standoff and a 10-second final hold. This is a collection
plan, not a measured dataset or a validated prompt-performance claim. Review
the generated `prompts/` and pilot the groups before a large collection.

The planner writes `collection-plan.json`, standard `batches/<group>/plan.json`
files and `commands.txt`. Run the individual `./run.sh batch --resume ...`
commands in that file yourself. They retain the existing B0 runner's source,
model-lock, completed-attempt and interrupted-attempt checks. Planning does not
run these commands. Do not change a saved plan to relabel a seed or move a
completed parent between splits.

After changing baseline code, create a fresh collection plan. Old plans pin
the earlier source hashes and cannot resume with the repaired controller.
Keep earlier failed episodes as evidence; do not overwrite their reports or
reuse their output directories. The October 5 acquisition repair has a fresh
plan at `output/data-collection-261005` (renamed on October 6); its batch commands are in
`commands.txt`. No large collection was started during the repair.

The user subsequently completed all 200 planned episodes. The October 6
relocation retained the original collection and parent plan hashes, updated
`commands.txt`, and recorded the path change in `relocation.json`. Original
reports retain their execution-time paths. Use the current parent directories
and the relocated collection plan for indexing and teacher collection.

### 2. Index completed batches

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.plan_data index \
  --collection output/data-collection-NEW/collection-plan.json \
  --out output/data-index-NEW
```

`sources.json` contains candidate original episodes. `report.json` lists pending,
too-short, incomplete, unsupported and invalid-execution attempts. Stopped
grasp failures with valid physics/SONIC and enough history remain Risk candidates.
The index verifies recorded evidence hashes and exact requests. It audits all
planned seeds and prompt groups, including pending attempts. Successful
episodes are marked as teacher candidates, not automatically as correction data.

Existing ordinary B0 batches can instead be assigned before window extraction
with a JSON file passed as `--batches`:

```json
{"batches":[
  {"run":"output/BATCH-A","split":"train","prompt_group":"train-a"},
  {"run":"output/BATCH-B","split":"val","prompt_group":"val-a"},
  {"run":"output/BATCH-C","split":"test","prompt_group":"test-a"}
]}
```

The runs must have distinct held-out seed and prompt groups. The tool compares
the effective phase instructions after filling implicit baseline defaults and
normalizing whitespace/case. Explicitly writing the default text, renaming a
group, or reformatting its JSON cannot bypass instruction leakage checks.
Semantic paraphrase-family assignment still needs review; string comparison
cannot prove semantic independence. A single default-prompt batch can be
adopted as train and supplemented by separate held-out groups.

### 3. Collect verified controlled pairs from successes

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.collect_pairs --sources output/data-index-NEW/sources.json \
  --out output/teacher-pairs-NEW --limit 3

# Continue exactly the saved source/reference/perturbation plan.
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.collect_pairs --sources output/data-index-NEW/sources.json \
  --out output/teacher-pairs-NEW --resume --limit 20
```

`configs/data/perturbations.json` proposes bounded positive/negative right-shoulder
offsets during lower and a right-elbow offset during lift. Each successful
source is replayed clean and perturbed with its saved ARDY references. Resume
requires unchanged source reports, reference hashes, settings, collector and
baseline sources. Completed failures are recorded and are not silently retried
until success; an interrupted attempt is preserved before a new attempt is made.
`--limit` caps newly attempted pairs in this invocation. A lock prevents concurrent
processes from writing the same pair queue.

The output `sources.json` preserves each parent's split and replaces eligible
originals with their verified paired branches; the converter includes their
clean identity windows. Sources without a valid pair remain original Risk
candidates. `report.json` lists recovery counts, failed/unpaired/invalid jobs,
pending jobs and error messages. Both-success pairs supply no recovery target.
A missing saved replan reference or a mismatched activation cannot be repaired
by inventing a teacher action. Failed pair outputs remain available for diagnosis.

This is controlled corruption supervision: the clean frozen baseline supplies
the teacher reference before nominal divergence. General recovery of natural
failures from arbitrary saved states still requires simulator/controller/RNG
restoration and a separate verified corrective teacher. The batch wrapper does
not implement or claim that capability.

### 4. Convert and check supervision

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.build_dataset --sources output/teacher-pairs-NEW/sources.json \
  --out output/dataset-NEW --tracking-thresholds .06 .06 .1 .1 \
  --skip-ineligible --require residual

MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh \
  python -m experiments.audit_dataset --data output/dataset-NEW/manifest.json \
  --require residual --out output/dataset-audit-NEW.json
```

The threshold values above are exploratory; inspect train/validation behavior,
choose and record thresholds, then freeze them before testing. Use
`--inspect-only --skip-ineligible` on the converter to inspect source counts
before writing windows. Use the original index `sources.json` for Risk-only data.

`--skip-ineligible` reports too-short and no-window parents; malformed streams,
changed evidence and invalid pair provenance still stop conversion. Missing
future targets remain masked. A partial future with no observed violation is
censored, not a negative. Dataset reports count positive/negative/censored Risk
windows, correction/stable windows, contributing parents and phases per split.
Missing splits write a diagnostic report without claiming a complete dataset.

`--require residual` returns nonzero if train or val lacks either verified
corrections or stable identity samples. Converted files remain available with a
`not-ready` report for inspection. Risk training likewise requires valid positive
and negative intervention samples in both train and val; Residual training
requires both supervision categories in both splits. Training checks these
conditions before optimization. Passing this availability gate does not prove
sufficient sample size or independent physical trials. Test supervision is
reported but never used to select thresholds or checkpoints.

## Checks and training commands

### Current real-data Risk pilot — 2026-10-06

The completed 200-parent B0 collection was indexed and converted into
`output/dataset-261006-risk/manifest.json`. Four execution failures were
ineligible at indexing; the remaining 196 parents passed conversion without
additional skips. The supervision availability audit passed for Risk:

| Split | Parents | Windows | Risk positive | Risk negative | Censored | Stable identity | Corrections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 116 | 24,163 | 22,358 | 1,746 | 59 | 639 | 0 |
| val | 40 | 8,589 | 7,965 | 610 | 14 | 236 | 0 |
| test | 40 | 9,235 | 8,639 | 584 | 12 | 212 | 0 |

This is enough to run an initial Risk training pilot. Windows overlap and are
not independent episodes; positive labels dominate. Availability does not
establish final sample adequacy or grasp improvement. The initial tracking
thresholds are `.06 .06 .1 .1`; later calibration must use train/validation
evidence only. Residual is not ready: both train and val have zero verified
correction samples. Collect and verify teacher pairs before rebuilding data
for that stage. No additional nominal batch is needed just to start Risk.
When teacher conversion produces a new manifest, train Risk again on that
fixed dataset before Residual: checkpoint loading requires matching Risk and
Residual dataset provenance. The current Risk run is an initial pilot.

From the server repository, launch the fixed 30-epoch budget in the background:

```bash
cd /home/group3/dl
nohup bash scripts/train_risk_multiseed.sh 0,2,3 > output/risk-overnight-261006.log 2>&1 < /dev/null &
```

These are three independent models with seeds 0/1/2 on physical GPUs 0/2/3,
not distributed training of one model. GPU 1 was occupied at inspection.
Each container sees its selected physical device as logical MUSA device 0.
The launcher uses the installed `dl-musa-render:latest` environment, audits the
existing default dataset before starting, and converts it if missing. A lock
prevents duplicate launchers. It preserves incomplete conversion directories
and never regenerates ARDY rollouts. Custom GPU, manifest and config paths are
optional positional arguments; custom data still passes the training loader's
audit and provenance checks.

The outer log prints a fresh `output/risk-multiseed-<timestamp>` directory.
Each `seed-<n>.log` records epochs/errors; each `seed-<n>/` holds `best.pt`,
`last.pt`, `metrics.json` and `report.json`. `launch.json` records GPU/seed and
configuration/manifest hashes; `summary.json` reports all jobs after they exit.
Do not assume a shell PID means training succeeded: check reports and logs.

```bash
tail -n 30 /home/group3/dl/output/risk-overnight-261006.log
```

The installed MUSA SDPA operator rejected the full-width model's dropout
descriptor. `PortableEncoderLayer` uses explicit attention matrix products,
softmax and dropout on MUSA only. Parameters, checkpoint keys, layer count,
width and dropout remain unchanged. CPU keeps the standard PyTorch path.
Real-data forward, backward, optimizer update, full-epoch training, validation
and checkpoint saving passed with the full 256-wide, four-layer model and batch
size 64; see `docs/verification.md`.
The actual launcher also passed a simultaneous one-epoch run on all three
GPUs; the default 30-epoch run still requires the start command above.

### Server background teacher-to-Risk workflow

`scripts/teacher_risk_pipeline.py` chains collection, conversion, a separate
dataset audit and independent Risk training seeds without requiring a desktop
agent to stay online between stages. It uses host Python's standard library
and the installed MUSA container. Start it once with a fresh output directory:

```bash
nohup python3 /home/group3/dl/scripts/teacher_risk_pipeline.py \
  --out output/teacher-risk-261006 --teacher-gpu 4 \
  > /home/group3/dl/output/teacher-risk-261006.log 2>&1 < /dev/null &
```

The indexed 69 successful parents supply 207 candidate pairs under the fixed
three-perturbation configuration: 126 train, 39 val and 42 test. Candidate
counts do not guarantee verified recovery targets. Teacher collection is
serial, initially on physical GPU 4 if idle. It preserves completed rejected
pairs as evidence. A collector exit code 1 caused by rejected individual pairs
can advance only when its report proves every planned job finished. Pending
or incomplete collection stops the workflow.

After conversion, the independent audit must pass both Risk and Residual
supervision availability before training begins. Thresholds stay fixed at
`.06 .06 .1 .1`; no test data selects settings. If correction supervision is
missing in train or val, the workflow records failure and does not launch
final Risk training. Passing these gates remains an initial data pilot, not
proof of enough independent examples or physical controller performance.

At the training stage the workflow queries `mthreads-gmi` again and uses every
device with zero memory use, zero utilization and no listed process. Each
selected card trains one independent seed under the 30-epoch config; this is
not DDP. It waits when no GPU is idle and leaves existing processes alone.
Source and configuration hashes are checked before each stage. A global lock
prevents duplicate pipeline launches. Use a fresh output directory for a new
run; this wrapper does not automatically resume a failed run.

The run directory contains `pairs/`, `dataset/`, `dataset-audit.json`, `risk/`,
stage logs, GPU snapshots and an atomically updated `state.json`. The latter
records parent/child PIDs, stage, teacher progress, source hashes and unique
completion/failure events. The outer log is adjacent to the run directory.
Risk seed folders contain the ordinary checkpoints and training reports.
The multiseed launcher accepts an optional fourth argument selecting its
fresh run directory; repository-relative paths cross the Docker mount safely.

Codex monitoring checks the exact run state on a schedule and sends stage
completion/failure notices through the authorized Outlook account. Local
scheduled monitoring needs the desktop computer on and the app running;
the detached server workflow continues when the local app is closed.
The retired record-sync and general mail-queue automations remain paused.

### Individual stage commands

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


## Stepwise quality pilot

Use `experiments.diagnose_labels` on train/validation only. Physically stable
hold evidence checks the continuous 5 ms task stream, not just sampled labels.
Fit exploratory tracking limits from train evidence and preserve verified
same-state recovery, contact/balance labels and censored future masks. Do not
select label thresholds to force class balance or tune them on test data.

`experiments.evaluate_risk` exports raw validation probabilities, class/phase
metrics, both-class PR/AP, clustered bootstrap intervals, auxiliary-head
metrics and a separate current-proxy-negative future-onset comparison.
The historical validation F1 grid/tie rule remains unchanged. Temperature
reports do not produce runtime gates; controller loading rejects a scaled
gate and mismatching validation-window hashes. Test evaluation requires a
matching pre-frozen validation gate.

The early-stopping pilot config preserves the original model and loss setup.
Natural and balanced-intervention sampling must be compared as separate
experiments. Every report records actual/budget epochs and the stopping reason.

Start the bounded first stage on the host, detached by the operator:

```bash
python3 scripts/risk_quality_pipeline.py --out output/risk-quality-261006 --gpu 2
```

Its `state.json` contains source hashes, PID, child PID, logs and completion
events. It evaluates the two old best checkpoints, generates 15 fresh parent
rollouts, indexes their declared train/validation splits, and checks three
teacher candidates. The terminal `pilot-review` status requires a measured
review before expanding pairs, creating a fresh manifest and training. This
command uses installed assets only. Never resume an old source-locked batch
under changed baseline source or overwrite the old dataset/checkpoints.

`experiments.run_policy` supports matched B0/P pilot requests with explicit
scene seeds. P uses real current/past simulator feedback, the same causal
nominal resampling as conversion, frozen Risk predictions and a matching
Residual/gate. Compare the same requests/options under the same current
control source; old dense-velocity replay outcomes are a separate cohort.
The full proposal evaluation budget remains outstanding.

### Frozen exploratory label calibration and continuation

`experiments.calibrate_labels` derives a candidate from the train diagnosis
only: each body's across-original-episode p95 of per-episode p95 maximum
horizon error, rounded upward to .01 rad and floored at the old threshold.
The current evidence gives `[.14,.14,.10,.12]`. This is an exploratory
proxy calibration, not intervention ground truth. Its artifact pins the
diagnosis, original manifest/source plan and calibrator code. Validation
statistics do not select the values; test is not used.

```bash
python -m experiments.calibrate_labels \
  --diagnosis output/risk-quality-261006-checks/label-diagnosis.json \
  --expected-diagnosis-sha256 33c9a6798272a28382a3fffabf2c645bbfbea8f767ce3e0a935fb4173829a649 \
  --out output/risk-quality-261006-checks/label-calibration.json
```

After terminal first-stage pilot review, the separately tested continuation
uses the same bounded teacher plan, preserves source cohorts and prior
evidence, and requires independent added recovery parents in train and val.
It validates the frozen candidate on continuous physical hold evidence
before training; a validation hold tracking-positive fraction above 10%
stops the stage rather than retuning on held-out data.

```bash
python3 scripts/risk_quality_continue.py \
  --first output/risk-quality-261006 \
  --calibration output/risk-quality-261006-checks/label-calibration.json \
  --diagnosis output/risk-quality-261006-checks/label-diagnosis.json \
  --out output/risk-quality-next-261006
```

Two independent Risk seeds use the early-stopping pilot config; choose the
lowest best total validation loss (ties: lower seed), then its raw
validation max-F1 gate. One supervised P Residual seed uses the original
`p.json` budget and the matching frozen Risk/manifest. Three matched
B0/P validation-family trials are an integration pilot, not the proposal's
20-pair, multiple-condition, three-seed evaluation. Report low correction
coverage, unobserved balance positives and narrow perturbation recovery
limits rather than treating availability checks as performance proof.

### Safely resuming the interrupted first stage

After the corrected runner and tests are deployed, the saved failed run uses:

```bash
python3 scripts/risk_quality_pipeline.py --resume \
  --out output/risk-quality-261006 --gpu 2
```

Use the explicit resume mode for this existing run. It validates the prior
terminal state, dead processes, source provenance and completed artifacts;
backs up the previous state and records an artifact receipt; and reuses
completed validation and parent trials. It rejects incomplete existing
stage directories and changes to frozen baseline/collector inputs.
Parent task failures can return exit 1, accepted only when all five actual
reports establish completed physics and SONIC execution.
