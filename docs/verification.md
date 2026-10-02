# Verification — 2026-10-03

Evidence for the new two-mode workflow is retained under
[`output/review-20261003-024719`](../output/review-20261003-024719/).
Generated evidence stays outside Git; this document records the result and its
limits. Commands were run on the supported S4000 host through the existing
MUSA containers and `.venv-baseline-musa`.

## Environment and checks

`./run.sh check` passed dependency checks, pinned-source cleanliness, all five
local asset manifests and MUSA linear/layer-norm/attention primitives.
Versions: torch 2.9.1, torch_musa 2.9.1+a18d871, MuJoCo 3.2.7,
NumPy 1.26.4, ONNX Runtime 1.23.2, Transformers 5.8.1, PEFT 0.19.1.
ARDY source is `693f74d13b3d04a0a22ce127ee79c929dd89756b`; SONIC source is
`b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`. Full asset hashes remain in the
lock/manifests and each physical report. Primitive checks alone establish
neither complete model execution nor task success.

- `./run.sh tests`: **47 tests passed** (`tests-final.log`). These cover
  named reference conversion, pose-constraint transforms, task assessment,
  deterministic seeds, default empty tasks, immutable resume plans, evidence
  tampering, failure denominators, model-load failures, worker reuse, pipe/PTY
  busy-input exclusion, render selection, actual RGB/MP4, sampled state mapping,
  failure preservation, source packaging and offline text adapters.
- `./run.sh smoke --out output/review-20261003-024719/smoke-final`: passed,
  converting 9 synthetic frames to 17 at 50 Hz. It runs no model or physics.
- Shell syntax, Python parsing, local documentation links and Git diff whitespace
  are checked as part of the final source review. The LaTeX source is updated;
  a PDF was not rebuilt because no TeX compiler is installed.

## Actual execution and visualization

`./run.sh batch` produced
[`batch-20261003-030857-346254`](../output/batch-20261003-030857-346254/summary.md):
one 2-second free-base SONIC standing attempt. It had no prompt, task objects or
GUI, and reported `COMPLETED`, with `task_success=null`. This is not grasp success.

`./run.sh manual --gui` produced
[`manual-20261003-030857-536362`](../output/manual-20261003-030857-536362/summary.md).
An automated stdin driver waited for READY, submitted `{}`, sent an extra prompt
while BUSY, then submitted a second `{}` after READY and quit. The actual GLX
MuJoCo viewer ran both independent two-second attempts, with seeds 0 and 1.
The busy prompt was discarded, never executed or recorded as an accepted request.
`manual-gui-check.json` records the assertions. The remote Mac VNC client was
not retested during this change; the server viewer was exercised.

A second GUI check submitted a real text JSON, “A person stands upright and
slowly raises the right hand.” (duration 1 s, seed 42), to frozen ARDY on MUSA,
then tracked it through SONIC in MuJoCo. It completed 3.58 simulated seconds
including standing/transition, returned to READY, and discarded the injected
busy command. Evidence is
[`manual-20261003-031351-875357`](../output/manual-20261003-031351-875357/summary.md)
and `manual-prompt-gui-check.json`. This establishes a working text-to-GUI
execution path, not a quantitative motion-fidelity or grasp result.

The actual `render --attempts 2 3` command rendered both short collision trials
from the prompt pilot, including failures, into RGB plus MP4. `render --all`
rendered both GUI standing attempts. Logs are `render-selected.log` and
`render-all.log`; receipts are in their session directories. These checks used
160×120, 25 FPS, third-person, OSMesa. Unit tests additionally exercise multiple
named cameras. Replays restore saved states without policy inference or physics
integration, and do not provide independent task verification.

## Prompt pilot: no success improvement established

[`batch-20261003-030410-331002`](../output/batch-20261003-030410-331002/summary.md)
contains a four-run paired diagnostic. `prompt_pilot.py` in the review directory
reuses one frozen model instance and records the exact requests/provenance.
Each pair uses the same scene position, seed, controller, spatial constraints,
finger targets and success/failure checks; only prompt profile differs.

| Attempt | Seed | Profile | Block XY (m) | Result | Simulated seconds |
| --- | --- | --- | --- | --- | --- |
| 00001 | 42 | legacy | .437396, -.238557 | timeout | 30.00 |
| 00002 | 42 | focused | .437396, -.238557 | prohibited robot-table contact | 2.98 |
| 00003 | 44 | legacy | .372257, -.263864 | prohibited robot-table contact | 2.94 |
| 00004 | 44 | focused | .372257, -.263864 | timeout | 30.00 |

Both profiles succeeded in **0/2** trials. This is a small exploratory pilot,
not the proposal's evaluation suite or evidence that the focused profile is
better. The default focused text provides a clearer stage-specific interface;
its physical benefit is unverified. The legacy profile remains selectable.

Both collision reports identify `right_hand_index_1_link` contacting
`task_table` during reach (first contact at 2.965 s and 2.935 s).
Both timeouts reached all five phases but logged zero opposing-finger contact
steps and zero continuous hold time. Maximum block clearance across all four
was about 1.75 mm from initial settling, far below the required 5 cm.
These observations point to reach clearance and hand/block alignment as further
calibration needs; they do not isolate a single controller cause. Changing text
alone has not solved the physical failures.

The current task uses a free root, dynamic 80 g / 6 cm block, articulated
fingers and frictional contacts. It checks collision/hold conditions at 200 Hz
and saves full states at 50 Hz. Success requires the block's lowest point at
least 5 cm above the table for 2 s within 30 s, with pilot opposing contact
above .01 N and no prohibited contact/fall. This contact threshold still needs
validation. Safety checks were not relaxed for the pilot.

## Review and cleanup

Execution scheduling is unified in `scripts/run.py` and
`baseline/execution.py`; session reporting/resume and manual input are separate
small shared modules. Rendering is consolidated in `scripts/render.py` plus
`baseline/rendering.py`. Redundant live/grasp frontends, the extra GUI wrapper,
obsolete batch helpers and unused simulation video arguments were removed.

Review fixed generation-worker shutdown after an ordinary per-attempt error,
repeated batch attempts after missing model assets, the uint32 seed upper bound,
summary persistence during cleanup, reference-file integrity, and preservation
of original state indices when RGB is subsampled. Regression checks cover the
failure and mapping cases. Completed failures remain in the denominator and
are never automatically retried to obtain a success.

At the user's request, the previous `artifacts/` output tree, unused installed
wheel cache and stale generated proposal PDF were removed: **4,720,087,667 bytes**
(about 4.40 GiB of file contents). `cleanup.json` records the inventory.
The current proposal source, pinned sources, model assets, environment and
Git history remain available. All new results use ignored `output/` paths.

Before removal, the earlier 20-trial summary was retained as
`previous-batch.json`: 0/20 successes, 14 robot-table collision failures and
6 timeouts; Wilson 95% interval [0, .161125]. Those old full trajectories and
videos are no longer available. Historical paths must not be interpreted as
current evidence locations.

## Remaining work

No physical G1 grasp/lift success is established. Hand/contact geometry,
reference tracking and table-edge clearance need controlled validation.
General instruction parsing, predictive clearance checks, real-time generation,
Risk/Residual learning, counterfactual controller snapshots, verified teacher
corrections, train/evaluation splits and the full comparative suite remain
unimplemented or unverified. `expert_valid=false` on these B0 records.
The manual interface resets each JSON attempt; it does not implement the
proposal's within-episode, at-most-two-correction interactive evaluation subset.

## English-only project text

Project documentation, terminal messages, generated summary labels, and the
existing readable output summaries/logs were translated to English.
`AGENTS.md` now requires English for project content and communication.
Existing experiment measurements, request data, source hashes, and model
provenance were preserved. Historical readable logs are translated presentation
copies, rather than byte-identical original console output. The previous source
archive was replaced with an English-only source archive.

Validation: `tests-english.log` records the regression suite. A Unicode Han scan
covers first-party source, documentation, configuration, tests, and generated
text outputs, excluding Git internals, installed dependencies, upstream sources,
model assets, and binary recordings. No Chinese text remains in that scope.

## Model loading before prompt acceptance or execution

Both batch and manual entry points now preload SONIC, ARDY, and the text encoder
before accepting or executing prompts. Completion prints a prominent
`ALL MODELS LOADED` banner. Manual input stays disabled until loading finishes;
batch attempts start afterward. Startup status and wall time are written to
`startup.json`; a loading failure accepts no prompt and executes no attempt.
Planning-only mode still loads no models. Loading uses the same worker thread
as subsequent generation, reusing the initialized service.

`./run.sh tests` passed all 50 tests; the log is `output/tests-preload.log`.
New tests verify model reuse without generating a warm-up motion, manual
startup ordering with and without GUI, and failure behavior in both modes.
Batch ordering is also asserted. These startup tests use mocked model loaders;
full model inference was not rerun for this orchestration-only change. Earlier
physical and GUI evidence above remains historical. Whitespace checks passed.

## Grounded approach without vision — 2026-10-03

The grasp protocol now starts farther back and inserts approach/settle before
the five hand phases. The default scene uses root X=-.36 m, table front X=.31 m,
and a target root X=-.11 m: 0.25 m of approach with a 0.42 m final root standoff.
Positions are initial conditions or ARDY constraints, never root teleports during
execution. Grounding reads current MuJoCo table/block/root state and logs world
and base-relative coordinates. No image is supplied to ARDY or SONIC.

ARDY receives its pinned Root2DConstraintSet with explicit MuJoCo XY to ARDY XZ
mapping and history-offset frame indices. Walking does not pin the feet. The
settling check requires actual position/heading tolerance, low root speed, and
both feet on the floor continuously before reach. Standing anchors are captured
again after arrival. The unchanged 30-second trial budget includes approach.
This is a new pilot scene/protocol, not a prompt-only improvement claim.

Evidence is under
[`review-approach-20261003-0406`](../output/review-approach-20261003-0406/).
`check.log` records passing complete-asset and MUSA operator checks;
`smoke/report.json` records a passing synthetic reference smoke check.
`tests-final.log` records 54 passing tests, including translated-scene grounding,
state-dependent targets, initial root placement, root waypoint axes/headings,
future frame indices, and position/heading/speed/two-foot arrival checks.

Two physical trials used `--grasp --batch 2 --seed 42 --cube-xy .40 -.22` on
MUSA device 1 with the frozen checkpoints:
[`batch-20261003-040615-951835`](../output/batch-20261003-040615-951835/summary.md).

| Seed | Approach result | Final task result |
| --- | --- | --- |
| 42 | Passed the configured near-table limit at 4.72 s; safely stopped | Failed: approach_too_close |
| 43 | Arrived and settled; root error .05368 m, heading error .15858 rad, speed .00601 m/s, both feet on floor | Failed at 13.12 s in lower: right_hand_middle_1_link contacted table |

The seed-43 rollout was also rendered at 5 FPS for inspection:
[video](../output/batch-20261003-040615-951835/attempt-00002/vision/video.mp4).
`render.log` records 66 RGB/video frames; the renderer restored saved states and
did not supply images to either model.

There were **0/2 grasp successes**. Seed 43 establishes one physical
approach-and-settle example; it does not establish general approach reliability
or successful grasping. Its subsequent reach finished without a prohibited
contact, but lowering still collided. Initial proximity is therefore not a
sufficient explanation for all table collisions.

For seed 42, ARDY's generated path ended near X=-.11486 m, close to the requested
X=-.11 m, while the executed root reached X=-.00468 m before the standoff stop.
The pinned SONIC G1 observation mode does not consume world root XY; a correct
ARDY root path does not guarantee exact executed global displacement. This is
why physical arrival checks are required.

A temporary variant switched to settle immediately upon entering the target
region. It was tested with the same seeds and position in
[`batch-20261003-041130-432721`](../output/batch-20261003-041130-432721/summary.md).
Seed 42 still crossed the standoff limit (4.76 s); seed 43 stopped 8.35 cm from
the target (6.56 s) and failed arrival. This variant was removed because it
interrupted gait without improving arrival. The final implementation executes
the approach clip with the standoff guard, then verifies settlement before reach.
All four trial records are retained, including the rejected variant's failures.

Walking and hand-clearance calibration remain necessary. No model weights,
SONIC observation configuration, finger gains, or prohibited-contact rules were
changed, and no grasp-success or training-data eligibility claim is made.


## Preparation, second-precision names and clearer video — 2026-10-03

Implemented `approach -> settle -> prepare -> reach -> lower -> close -> lift -> hold`.
Preparation is requested only after .4 s of continuous arrival/heading/speed and
double-foot-contact acceptance. It conditions an 8-degree waist pitch through
ARDY, holds the wrists back, then checks continuous stability again before
reaching. The pinned G1 has no movable neck. This implements the phase sequence
and a bounded nominal posture goal, not verified gaze targeting or grasp success.
Torso targets remain shared nominal inputs; learned corrections remain arm-only.

Session names now use Asia/Shanghai `batch-YYMMDD-HHMMSS` and
`manual-YYMMDD-HHMMSS`. Atomic directory allocation waits for the next available
second on a collision; it cannot overwrite a previous session. Existing evidence
paths are preserved. Repeated render receipts retain a finer unique suffix.
Default video output is 640x480 at 25 FPS with H.264 CRF 18. Collection and
rendering remain separate.

Evidence is under [`review-prepare-261003`](../output/review-prepare-261003/).
`commands.json` records the commands; individual plans and reports preserve
source/model hashes, settings and dependency versions.

- `tests-final.log`: 56 tests passed, including torso conditioning frame/joint
  mapping, unchanged hand goals during preparation, invalid torso inputs, and
  preservation of both plans when second-resolution session names collide.
- `smoke/`: synthetic 9-to-17-frame reference conversion passed. This does not
  validate physical grasp success.
- `check.log`: complete local assets/pinned sources and MUSA primitive checks
  passed. Actual full-model execution is evidenced separately below.
- Python syntax parsing, English-content scan and `git diff --check` passed.
  A host bytecode-writing check encountered container-owned cache permissions;
  syntax was checked without writing bytecode and container tests passed.

Two actual frozen ARDY/SONIC trials with block XY=(.40,-.22), seeds 42/43 and the
historical .42 m standoff are saved in
[`batch-261003-050532`](../output/batch-261003-050532/summary.md): **0/2 successes**.
Seed 42 stopped at 4.72 s for approach overshoot. Seed 43 passed the first
stability gate at 7.96 s, then stopped at 10.14 s after preparation: position
error rose from .0537 to .0824 m and heading error from .1586 to .2532 rad.
Both feet contacted the floor, but the complete second gate failed, so reach
was never generated. This run predates the more specific `prepare_target_drift`
label; its original report retains `approach_target_missed`.
The final requested waist goal was .1396 rad; the final generated joint
reference was .3112 rad and the measured joint was .0050 rad. Preparation
tracking is therefore **not verified**. Do not describe the goal angle as an
achieved physical inclination. No limits or gates were relaxed to pass this run.

A separate closer-standoff pilot, .22 m with the same seeds/block XY, is saved in
[`batch-261003-051057`](../output/batch-261003-051057/summary.md): **0/2 successes**.
Seed 42 overshot at 4.70 s; seed 43 missed the arrival target at 7.96 s. This
candidate is not adopted as the default or presented as an improvement. The
configurable standoff range now includes .20-.55 m for validation experiments;
the existing .42 m default still needs workspace calibration. These four trials
are interface/feasibility pilots, not the proposal's paired method evaluation.

`reachability.json` analyzes the earlier
`batch-20261003-040615-951835/attempt-00002` at 7.96 s. The right
shoulder-to-wrist chain length sum is .4104 m; distances from that shoulder to
the nominal reach/lower goals are .4679/.5395 m. With that fixed shoulder,
these goals exceed even a geometric reach upper bound. Joint limits, hand
orientation and collisions would restrict reach further. This diagnoses one
recorded stance; it does not certify alternative stances or a complete workspace.
Initial backoff and final grasp distance must be calibrated separately.

The new-resolution [review video](../output/batch-261003-050532/attempt-00002/vision/video.mp4)
was rendered with `--fps 5` for review speed while using the new default 640x480
resolution. All 51 H.264 frames were decoded successfully; duration 10.2 s,
CRF 18, saved in `video-decode.json` and the render report. `video-review.jpg`
contains inspected frames. The normal render default remains 25 FPS.

The assignment-aligned improvement plan is recorded in [grasp.md](grasp.md):
calibrate shared approach reachability, grasp frame and complete finger clearance;
then evaluate learned predictive arm correction under the proposal's frozen
nominal/target-bias/action-latency conditions. Contact-dependent alignment gates,
teacher branching, learning and perturbation evaluation remain planned.
No successful physical grasp has been established by these changes.

## State-based acquisition checkpoint — 2026-10-03

Evidence is under `output/review-b0-recovery-261003/`. This is calibration,
not the planned paired B0/P evaluation. The earlier failures above are unchanged.

The frozen SONIC controller followed an official pinned walking reference in
`known_walk` without falling (mean body-joint RMSE .0961 rad), and a static arm
reference in `static_arm` (RMSE .0561 rad). These controller diagnostics bypassed
ARDY and are not complete task trials. Root-path and explicit-footstep ARDY
pilots exposed under-travel: satisfying a generated root path does not establish
physical arrival. The unused experimental footstep and guidance-weight interfaces
were removed from production; their diagnostic source snapshot is retained with
the outputs.

Two **hand-only** frozen ARDY–SONIC trials passed the existing physical assessment:

| Trial | Lowest-point maximum clearance | Continuous hold | Video |
| --- | --- | --- | --- |
| `measured-hold-ardy-wrist-2` | .1266 m | 5.315 s | [Hand trial 2](../output/review-b0-recovery-261003/measured-hold-ardy-wrist-2/vision/video.mp4) |
| `measured-hold-ardy-wrist-3` | .1311 m | 5.310 s | [Hand trial 3](../output/review-b0-recovery-261003/measured-hold-ardy-wrist-3/vision/video.mp4) |

Both use a free robot root, the original dynamic 6 cm/80 g block, articulated
fingers, frozen models, and the original contact/clearance/hold requirements.
There is no attachment, teacher, direct IK body execution, or camera input.
All 435/432 H.264 frames were decoded at 640x480 and 25 FPS; receipts are in each
`vision/video-validation.json`. Close-up stills were inspected separately.
The same hand-only pilot also includes seed 0 failing to retain the block and
seed 1 contacting the table: **2/4**, with walking omitted. Its `task_success`
field assesses only the recorded hand scope. Neither positive case establishes
complete B0 success or verified expert-data eligibility.

The initial complete protocol pilots `full-pipeline-00/02/03/04` produced **0/4**.
Seeds 0 and 4 passed walking, continuous double-foot settlement, and preparation,
but missed hand alignment. Seeds 2 and 3 missed arrival. Seed 0's measured arrival
error was .0201 m after settlement and .0364 m after preparation. Its
[partial full-task video](../output/review-b0-recovery-261003/full-pipeline-00/vision/video.mp4)
shows that progress and the failed hand approach. Four lower-goal calibration
trials also failed (two missed alignment, two contacted the table). Two trials
using canonical full-body preparation constraints failed preparation settlement;
that variant was not adopted. These ten full-task pilots vary calibration
settings and must not be pooled into a claimed held-out success rate.

The integrated checkpoint adopts initial parked arms, .45 m extra backoff,
.22 m final standoff, a faster two-step prompt, measured standing anchors,
pre-curled fingers, a wrist-frame acquisition gate, and a checked measured-pose
hold through SONIC before closing. Settle/close/hold no longer resample motion.
Lift preserves the measured grasp orientation and requests a .14 m wrist rise.
A missing acquisition or sustained opposing contact ends the trial explicitly.
These are shared nominal task rules; they neither modify the frozen policy's
normalization pose nor expand the future learned arm mask. Full-task success
of this integrated checkpoint is still unverified.

Checkpoint validation: `checkpoint-tests-final.log` records **59 passing tests**,
including RGB/MP4 rendering, acquisition geometry, immutable policy defaults,
and a measured reference hold that does not mutate physical state.
`checkpoint-smoke/` passed synthetic 9-to-17-frame reference conversion; it does
not establish model compatibility or task success. The first new hold test used
an inappropriate exact comparison across float64-to-float32 conversion; the
corrected tolerance check passed. English-content and diff-whitespace checks
passed. The asset/backend and actual frozen-model evidence remains distinct
from these synthetic checks.
