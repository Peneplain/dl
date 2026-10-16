# Recorded results

The [report](https://github.com/Peneplain/dl/blob/main/report/report.pdf) follows three distinct evaluations: a shared-scene ARDY/Kimodo benchmark, an independent confirmation of the ARDY-based learned controller, and a same-weight intervention-gate ablation. Their cohorts remain separate.

## ARDY versus Kimodo: 100 scenes per method

Both generators execute the same 100 scheduled scene requests, seeds 60000–60099, with matching scene and SONIC manifests. Each method uses its frozen adapter and wrist calibration. All attempts remain in the denominator, including stopped or failed execution outcomes.

| Metric | ARDY | Kimodo |
|---|---:|---:|
| Successful grasps | 40/100 | 55/100 |
| Initial clearance/contact threshold reached | 58/100 | 58/100 |
| Later sticky losses among threshold-reaching trials | 18/58 (31.03%) | 3/58 (5.17%) |
| Final success among threshold-reaching trials | 40/58 (68.97%) | 55/58 (94.83%) |
| Success rate | 40.00% | 55.00% |
| 95% Wilson interval | [30.94%, 49.80%] | [45.24%, 64.39%] |
| Median command-to-first-reference wall time | 4.3564 s | 7.4403 s |
| Mean joint tracking RMSE | 0.09241 rad | 0.09552 rad |
| Additional user corrections | 0 | 0 |

There are 23 shared successes, 28 shared failures, 32 Kimodo-only successes, and 17 ARDY-only successes. Kimodo has an observed **15 percentage-point** success advantage; ARDY responds faster. Command response includes the recorded command-to-first-reference path and is not a model-forward benchmark. Automated alignment replans are part of both deployed systems.

The clearest aggregate distinction is retention: each method reaches the initial clearance/contact threshold in 58 of its own 100 trials, but ARDY subsequently loses the object in 18 and Kimodo in 3. ARDY records 16 terminal `grasp_lost_after_success` reasons; the remaining two sticky losses terminate with prohibited contact. Equal threshold counts with unequal final success show why the entire lift/hold sequence matters. ARDY's slightly lower joint-tracking RMSE does not ensure better object retention. Kimodo's dominant remaining failure category is missed alignment (28 trials).

Source: [public numerical results](https://github.com/Peneplain/dl/blob/main/docs/results/public-results.json), including all 200 trial rows, retention flags and response arrays, with a [source SHA-256 manifest](https://github.com/Peneplain/dl/blob/main/docs/results/public-source-manifest.json). The [video supplement](https://github.com/Peneplain/dl/tree/main/videos) includes one successful and one failed example from each generator.

## Risk-guided residual controller: independent confirmation

The controller is frozen before evaluating 128 scheduled prospective origins. The reference policy is **symmetric hybrid**: covered requests use a shared frozen reference inventory, while uncovered requests generate natively from each branch's measured state under the same rule. Treatment order alternates by original origin index. Thus the comparison measures the deployed ARDY-based feedback pipeline.

| Outcome | ARDY baseline | Ours |
|---|---:|---:|
| Complete end-to-end pairs | 124 | 124 |
| Successful grasps | 54 | 74 |
| Success rate | 43.55% | 59.68% |

The paired counts are **49 shared successes, 45 shared failures, 25 rescues, and 5 harms**. The gain is **+16.13 percentage points**. The exact two-sided paired McNemar result is **p = 0.0003249142318964**, below the prespecified multiplicity-controlled threshold **α = 0.00625**. The nominal 95% paired-origin percentile bootstrap interval is **[+8.06, +24.19] percentage points**, using 100,000 resamples and RNG seed 261010. This interval is separate from the multiplicity-controlled test.

Four scheduled origins have common nominal-reference rejection, leaving both scored arms unattempted. One candidate operational failure remains unsuccessful in the end-to-end endpoint. Across all 128 scheduled origins, the empirical missing-outcome effect bounds are **[+12.50, +18.75] percentage points**; these are identification bounds, not confidence intervals.

This is evidence of improvement over the paired ARDY baseline under the recorded hybrid protocol. The 59.68% controller result and Kimodo's 55% generator result come from different cohorts and do not establish a controller-versus-Kimodo ranking.

The [public confirmation summary](https://github.com/Peneplain/dl/blob/main/docs/results/public-results.json) includes all scheduled-origin rows and the recorded inferential results. The [source manifest](https://github.com/Peneplain/dl/blob/main/docs/results/public-source-manifest.json) identifies original report bytes by SHA-256. Full original archives remain preserved separately; the public JSON is a derived numerical view.

## Training and implemented models

All experiments, including simulation data collection, model training, and evaluation, were conducted on the course-provided **Moore Threads S4000 cluster**.

Both Risk and Residual use temporal Transformer encoders with **2 layers, width 128, 8 attention heads, and feed-forward width 512**. Risk predicts structured **8 × 4 × 32** body/time tokens. Residual receives their projected values by addition to its future tokens and requests corrections for 14 arm joints. The other 15 body joints receive zero learned offset. A scalar intervention head controls invocation; it is separate from the full structured conditioning.

| Training summary | Risk | Residual |
|---|---:|---:|
| Parameters | 431,077 | 438,030 |
| TRAIN parent origins | 3,163 | 1,917 |
| TRAIN windows | 695,381 | 489,888 |
| Fixed VAL windows | 1,946 | 1,832 |
| Executed epochs | 19 | 45 |
| Retained checkpoint epoch | 7 | 33 |
| Optimizer updates | 4,864 | 23,040 |
| Best validation objective | 0.18812064 | 0.00034409 |

The Residual TRAIN view contains 1,055 verified-correction windows, 311,945 causal teacher behavior-cloning windows, and 176,888 stable-identity windows. Overlapping windows are training samples; they are not independent grasp trials. Residual uses predicted tokens from its matching frozen Risk model. The normalization statistics are fitted on Risk TRAIN nominal inputs.

Learned updates run at 10 Hz, execution/state recording at 50 Hz, and physics/task evaluation at 200 Hz. Requested offsets use a tanh output; the shared limiter preserves the 0.15 rad magnitude and 0.5 rad/s rate limits. The complete recorded epoch curves appear in [learning-curves.pdf](https://github.com/Peneplain/dl/blob/main/report/figures/learning-curves.pdf). These supervised objectives use different units and are not grasp-success probabilities.

## Structured Risk-conditioned control: invocation study

This separate development cohort uses 20 scheduled origins and fixed learned weights. Both learned variants run Risk and retain its full structured tokens. The changed factor is selective versus always-on Residual invocation.

| Metric | ARDY baseline | Selective | Always-on |
|---|---:|---:|---:|
| Successful origins | 3/20 | 10/20 | 14/20 |
| End-to-end success | 15% | 50% | 70% |
| Risk calls | — | 4,172 | 4,075 |
| Residual calls | — | 1,575 | 4,075 |
| Residual invocation | — | 37.75% | 100% |
| Mean learned-controller time per decision | — | 4.1515 ms | 5.1978 ms |

Selective invocation skips **62.25%** of Residual forward calls on its own decisions and reduces mean learned-controller time by **20.13%** (1.0463 ms per decision) relative to always-on. Both learned modes consume the complete predicted Risk tokens: always-on applies this conditioned correction at every learned update; selective invocation uses the additional scalar scheduling head. Both modes fit the 100 ms learned-decision interval. The experiment varies invocation while preserving the central predictive Risk-to-Residual interface.

The timing scope covers recorded learned-controller preparation and model work on each arm's evolving trajectory. It does not measure generator latency, total simulation throughput, or energy. The ablation is descriptive and has no additional significance test.

Sources: [public invocation and efficiency results](https://github.com/Peneplain/dl/blob/main/docs/results/public-results.json), the [ablation chart](https://github.com/Peneplain/dl/blob/main/report/figures/gating.pdf), and the [source SHA-256 manifest](https://github.com/Peneplain/dl/blob/main/docs/results/public-source-manifest.json).

## Videos and reproducibility

The [video supplement](https://github.com/Peneplain/dl/tree/main/videos) presents matched scene views and generator examples with text instructions, stage, simulation time, and method labels. Success, failure, recovery, and harmful correction are all represented. Presentation copies use clear method names; original clip bytes and source hashes are preserved in the package's provenance.

Use the [repository README](https://github.com/Peneplain/dl#edit-the-report-in-overleaf) for Overleaf compilation and the report/video layout.
