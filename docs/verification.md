# Measured Results and Verification Limits

This page summarizes retained evidence and current conclusions. Per-attempt
measurements, source hashes, original plans, logs and rendered states remain
in the ignored output storage; they are not a chronological repository log.
Paths below are relative to `output/`, whose canonical server target is
`/data/group3/dl-output/`.

## What the evidence establishes

- Frozen ARDY and Kimodo generate named G1 body references and run through the
  same SONIC/MuJoCo task harness.
- The final real-rollout dataset passes Risk/Residual supervision availability
  and the frozen physical label guard.
- Risk training, frozen-Risk Residual training and validation gate selection
  completed on MUSA.
- P runs through the original batch entry, producing causal gated arm
  corrections during actual SONIC/physics execution.
- The fixed 61020–61079 B0/P cohort shows **no aggregate grasp-success improvement**.

Operator checks, synthetic tests, offline model metrics, visual replay and
physical task outcomes establish different things. A passing readiness audit
or small supervised validation loss is not physical task success.

## Dataset evidence

Artifacts:

```text
risk-quality-final-261007f/dataset/manifest.json
risk-quality-final-261007f/dataset/report.json
risk-quality-final-261007f/dataset-audit.json
risk-quality-retrain-261007f/final-label-gate.json
```

| Split | Windows | Source parent entries | Risk positive | Risk negative | Censored | Correction samples | Stable identity samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 82,060 | 286 | 24,606 | 56,759 | 695 | 30 | 29,185 |
| Validation | 29,248 | 99 | 7,688 | 21,299 | 261 | 9 | 13,026 |
| Test | 29,497 | 97 | 7,924 | 21,323 | 250 | 13 | 13,620 |

Both supervision-availability gates pass. Parent entries include related
clean/perturbed branches; these are not 482 independent original episodes.
Correction entries are also not independent recovery trials merely because
they have distinct branch IDs. Use original-parent/episode-cluster identifiers
for uncertainty and disclose independent recovery coverage.

The manifest preserves older dense-velocity and current sparse-velocity
cohorts. The current training-only tracking thresholds are
`[.14, .15, .10, .13]` rad. The frozen stable-hold tracking-positive guard is
at most 10% on untouched validation. It measured 690/9,721 (7.10%) on train
and 378/4,276 (8.84%) on validation. No test data was used to select this
candidate. Passing this exploratory guard does not prove that every proxy
label corresponds to a future physical failure.

The controlled teacher verifies matching pre-perturbation state and a clean
success versus perturbed failure. It is not general natural-failure recovery.
Unsupported missing-reference replays remain excluded and introduce nonrandom
censoring. The main correction supervision is still small despite the number
of overlapping windows.

## Training and selected artifacts

Run root: `risk-quality-retrain-261007f/`.

| Model | Seed | Maximum epochs | Actual epochs | Best epoch | Best validation total loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Risk | 0 | 30 | 9 | 4 | 0.2157395087 |
| Risk | 1 | 30 | 7 | 2 | 0.2307007945 |
| Residual P, with frozen Risk seed 0 | 0 | 30 | 30 | 11 | 0.0006613362 |

Risk used independent seeds on physical GPUs 2 and 3, natural sampling and
validation patience 5. The actual epochs differ from the 30-epoch maximum.
Residual completed its fixed budget and was supervised by corrections and
stable identity samples using predicted frozen-Risk tokens. These were
independent models, not single-model DDP. Offline reports record
`physics_executed=false`; physical evidence is documented separately below.

Risk seed 0 was selected by minimum best validation total loss. Its raw
validation max-F1 gate is threshold **0.31**, validation F1 **0.8802951**.
Tie-breaking favors fewer activations, then higher threshold. The threshold
was frozen before test; no temperature transform was fitted for production.

| Artifact | Server path under `output/` | SHA-256 |
| --- | --- | --- |
| Dataset manifest | `risk-quality-final-261007f/dataset/manifest.json` | `9b004bad3f425e69ca3d91aebc50e668cfa7424965f342d86285182ee51d27a6` |
| Selected Risk | `risk-quality-retrain-261007f/risk/seed-0/best.pt` | `07dc7498b24cc28570fe00cd5e57292fc8e46607215e243ce8d357b403395512` |
| Residual P | `risk-quality-retrain-261007f/residual/best.pt` | `0ada52f5686a62ecdd32b511d29ac346d401c10a7f6297292f5554f4500ba095` |
| Runtime gate | `risk-quality-retrain-261007f/eval-val/gate.json` | `8e6cf670449ca1c24d6c47d55a4d2a1ef657eb57e55377b213bb746a441ebcd5` |

`final-label-gate.json` is a dataset/label audit certificate, not the runtime
intervention gate. Keep these artifacts matched; loader/session provenance
checks reject changed checkpoint, normalizer, dataset or gate evidence.

## Offline Risk test metrics

Source: `risk-quality-retrain-261007f/eval-test/report.json`.
The frozen gate evaluates 29,247 observed windows and masks 250 censored ones.

| Metric | All observed intervention windows | Current-proxy-negative future-onset subset |
| --- | ---: | ---: |
| Available windows | 29,247 | 22,479 |
| Positive windows | 7,924 | 1,156 |
| Recall | 0.8727 | 0.7413 |
| False-positive rate | 0.0520 | 0.0520 |
| Balanced accuracy | 0.9103 | 0.8447 |
| F1 | 0.8672 | 0.5490 |
| Average precision | 0.9510 | 0.4990 |
| ROC AUC | 0.9759 | 0.9506 |

Of all positives, 6,755 already have a current tracking/contact/balance proxy
violation, 1,156 are future-only proxy positives, and 13 are separate direct
verified-recovery labels. This distinction matters: overall intervention
metrics mix reactive and future-onset cases. On all observed windows the
current-proxy reactive diagnostic achieves F1 0.9204, exceeding the learned
gate's 0.8672. On the future-onset subset that proxy cannot trigger, but proxy
onset is not proof of future physical task failure. Do not infer overall
predictive-control superiority from window accuracy or the onset subset.

## Original batch integration

Artifact:
`risk-quality-batch-integration-261008/batch-261008-003745/attempt-00001/`.
Its report records `physics_executed=true`, `sonic_executed=true` and
`task_success=true`, with the selected matched model/gate hashes. The episode
recorded 254 Risk decisions, 13 activations and no history gaps, with nonzero
bounded arm correction.

This verifies loading, causal runtime control and physical execution through
`./run.sh batch --ardy`. One successful episode is not a success-rate estimate.
The shared baseline had no learned-artifact requirement, and P retained the
same contact, fall, reference and task-success criteria.

## Matched B0/P physical evaluation: seeds 61020–61079

Artifacts:

```text
risk-quality-policy-paired3x20-261007-seeds61020-61079/manifest.json
risk-quality-policy-paired3x20-261007-seeds61020-61079/audit.json
risk-quality-policy-paired3x20-261007-seeds61020-61079/paired-results.json
```

Three fixed seed groups each contain 20 matched trials. Each pair shares
request, scene, baseline parameters, physics, SONIC and evaluator. Risk,
Residual and the validation gate remained frozen. All six method batches
completed 20 valid physical trials, with zero invalid executions. The audit
is `status=pass`, with no errors.

| Seeds | B0 successes | P successes |
| --- | ---: | ---: |
| 61020–61039 | 5/20 (25%) | 7/20 (35%) |
| 61040–61059 | 10/20 (50%) | 5/20 (25%) |
| 61060–61079 | 8/20 (40%) | 11/20 (55%) |
| **All 60 pairs** | **23/60 (38.33%)** | **23/60 (38.33%)** |

Each aggregate Wilson 95% interval is [27.09%, 50.98%]. Pair outcomes are
seven P-only successes, seven B0-only successes, 16 both-success and 30
both-failure. P minus B0 is 0 percentage points; its paired bootstrap 95%
interval is [-11.67, +11.67] percentage points. Exact two-sided McNemar p=1.0.
Retaining only the first or third batch would misrepresent the pooled result.

| Failure reason | B0 | P |
| --- | ---: | ---: |
| `grasp_not_acquired` | 12 | 12 |
| `timeout` | 15 | 15 |
| `grasp_lost_after_success` | 9 | 7 |
| `prohibited_robot_table_contact` | 0 | 2 |
| `block_floor_contact` | 1 | 1 |

P made 12,276 Risk decisions with 4,517 Residual activations (36.80%) and no
history gaps across these trials. Means of the per-trial p95 latency were
4.174 ms for Risk, 3.464 ms for Residual and 9.170 ms for the controller.
These include the report's gate-dependent sampling and are not one pooled
latency percentile. They do not establish real-time motion-generation control.

This is a single learned-checkpoint exploratory repeat with changed scene
seeds. It does not complete the proposal's multiple trained seeds, perturbation
conditions, B1/I2 controls or full ablation budget. Analyze the discordant pairs
and freeze a targeted validation change before a new independent experiment;
do not select seeds after inspecting their outcomes.

## Incomplete additional repeat

The separate `risk-quality-policy-paired4x20-261008-seeds61100-61159-rerun1/`
run planned 60 requests per method with a 5 s hold phase. It did not finish
all branches or produce a passing paired audit. Its six batch summaries record
54 completed B0 attempts (53 valid executions, 19 successes) and 47 completed P
attempts (45 valid executions, 18 successes). There are 45 request-matched pairs
with valid physical execution on both sides: 9 both succeed, 19 both fail,
9 P-only successes and 8 B0-only successes.

Three executions were invalid because generated references violated shared
joint limits during `lower`: B0 and P at seed 61113 and P at seed 61132. Preserve
those reports and the uncompleted requests. Do not count them as completed valid
trials, choose only favorable batches, or pool this partial, differently
configured run with the fully audited 61020–61079 cohort. Its partial evidence
is not a replacement for a predeclared completed comparison.


## Kimodo physical evidence

Artifact: `kimodo-final-batch20-261007/batch-261007-104935/summary.json`.
The pinned final-source/raw-output batch completed 20 attempts at seeds 0–19
and retained **11/20 successes (55%; Wilson 95% [34.21%, 74.18%])**. Failures
were five alignment misses, one acquisition failure, two timeouts and one
post-threshold loss. It used the tuned wrist offset, 100 denoising steps and
constraint guidance 2, with the shared free-root/dynamic-block/SONIC evaluator.

The tuning and final batches reuse the same seeds; they are not 40 independent
trials. Static-FK projection pilots failed and are not part of this raw-output
result. This calibrated K0 batch has different seeds/selection context from
B0/P and does not establish superiority to either. A fair generator comparison
requires predeclared common instructions, conditions and seeds.

## Regression and remaining evidence

Run `./run.sh tests` in the supported image for source contracts and RGB/MP4
checks. Asset/operator checks, frozen model probes and shared physical
execution remain distinct acceptance gates. Environment-dependent skips must
be reported rather than treated as passed model runs. The cleanup's check
results are stored with its output receipt.

Remaining research/deliverables include the proposal's controls and ablations,
a fair held-out ARDY/Kimodo comparison, representative success/failure videos
for each submission method, Archon-session evidence and the final report/demo.
Current results do not establish P task improvement, walking grasp,
real-robot transfer or GR00T integration.
