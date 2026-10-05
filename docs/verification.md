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

## Direct-start manipulation calibration — 2026-10-03

A direct grasp calibration uses `--direct-start` to place the free base at the
grounded approach target in the scene's initial condition and checks position,
heading, speed, and double-foot support throughout a two-second stand. It must
remain ready for at least .4 continuous seconds before manipulation starts, then
omits approach, settle, and preparation. SONIC still executes all subsequent
motion; the robot root remains free. This calibration does not measure walking
or establish complete B0 success. The block remains dynamic and camera images
remain outside inference.

Evidence is in `output/batch-261003-083431/` (standoff .22 m),
`output/batch-261003-083920/` and `output/batch-261003-084351/` (standoff .20 m).
All four closer-standoff trials used block XY=(.40,-.22), frozen ARDY and SONIC,
and seeds 42–45. Their combined result was **2/4** (Wilson 95% interval
[15.0%, 85.0%]):

| Seed | Result | Evidence |
| --- | --- | --- |
| 42 | Success; 10.2 cm maximum block clearance, 2.01 s continuous hold | [Video](../output/batch-261003-083920/attempt-00001/vision/video.mp4) |
| 43 | Stopped during close after right middle finger contacted the table; 1.1 mm penetration | [Video](../output/batch-261003-083920/attempt-00002/vision/video.mp4) |
| 44 | Stopped during close after right middle finger contacted the table; 1.3 mm penetration | [Video](../output/batch-261003-084351/attempt-00001/vision/video.mp4) |
| 45 | Success; 11.7 cm maximum block clearance, 2.01 s continuous hold | [Video](../output/batch-261003-084351/attempt-00002/vision/video.mp4) |

Both successes maintained opposing finger contact throughout the required hold;
no slipping was observed. Finger gains and targets were unchanged. The failures
share one prohibited contact, so collision rules were not relaxed. At .22 m,
the seed-42/43 pair failed before acquisition: the block was 4.8–6.1 cm below
the wrist frame, outside the existing 4 cm lower acquisition bound, with no
thumb opposition. Reducing standoff to .20 m allowed all four later trials to
reach acquisition, but did not eliminate close-phase finger/table collisions.
No grip-force increase was evaluated because neither successful trial slipped.
Matched seed-43/44 variants tested distal index/middle targets held at their
1.3-rad preshape, then wrist roll of +90 degrees and -90 degrees during reach
and lower. The distal-curl variant retained the same middle-finger/table
collision (`batch-261003-085937`). The +90-degree roll moved the block outside
the positive wrist-lateral acquisition region (`batch-261003-090500`); the
-90-degree roll entered the region but did not establish sustained opposing
contact (`batch-261003-090919`). None was adopted. The selected .20 m setup
retains the original closure targets and upright wrist orientation.

The shared validator, target clipping, torque clipping, and velocity gate apply
to all 29 body joints; no per-joint exception was introduced. A test checks
reference clipping across all 29 joints. `./run.sh tests` passed with **61 tests**
after restoring the selected controller settings. MUSA loaded the pinned full
models for all twelve direct-start attempts, including the unsuccessful
variants. Rendering for seeds 42 and 43 completed at 640x480 and 10 FPS, with
receipt `batch-261003-083920/render-261003-091951-990222.json`. Seeds 44 and 45
were rendered at 640x480 and 25 FPS; their receipt is
`batch-261003-084351/render-261003-084814-808165.json`. The six
selected-configuration attempts are exploratory calibration evidence only, not
the planned paired evaluation; the two successes cannot be counted as full B0
trials.

## Allowed hand contact, optional walking and complete hold recording — 2026-10-03

The agreed pilot contact policy now permits both hands, fingers, and palms to
contact the tabletop and table legs. The pinned G1 attaches its palm meshes to
the wrist-yaw bodies, which are included in this allowance. Contact physics
remains active; forearm, torso, leg, block-floor, non-foot robot-floor, and
non-right-hand robot-block contact stops remain. Allowed hand-table contact is
counted per 200 Hz physics sample and cannot replace finger-block opposition
in the success criterion. Historical results above retain their original rules.

`--grasp` now starts the free robot at the grounded table target by default,
checks continuous stability during the two-second stand, then runs the five
hand phases. `--walk` adds the existing approach, settle, and preparation path;
`--direct-start` remains a compatibility alias for the default start. Scene
settings list the actual phases and declare the hand-table contact allowance.
Legacy plan parsing preserves whether walking was originally selected; changed
source hashes still prevent cross-version resume.

The earlier seed-42 recording in `batch-261003-083920` ended at 14.20 s,
with only .02 s in its configured hold phase. Seed 45 in
`batch-261003-084351` ended at 13.98 s while still in lift. Both stopped when
the two-second success threshold was reached. The executor now completes lift
and the configured three-second hold before ending a successful trial, while
continuing shared failure checks and respecting the 30-second timeout. A later
fall or prohibited contact invalidates an earlier success. Rendering cannot
restore motion absent from the old saved states.

Validation artifacts are in `output/review-contact-hold-261003-CvjmWg/`.
`tests-final.log` records **65 passing tests**, including actual MuJoCo contact
fixtures for both hands/palms, prohibited non-hand table contacts, default and
optional walking selection, full hold scheduling after early success, later
failure invalidation, and RGB/MP4 fixture rendering. Scheduler fixtures do not
establish model compatibility or physical task success. Bash syntax,
documentation consistency, English-content, and diff-whitespace checks passed.
The launch source snapshot matches every source hash in the collection plan.

The actual frozen ARDY/text stack ran on MUSA, with frozen SONIC on ONNX Runtime
CPU, in a fresh two-episode manipulation pilot:

```bash
./run.sh batch --grasp --batch 2 --seed 42 --cube-xy .40 -.22 --table-standoff .20 --output-root output/review-contact-hold-261003-CvjmWg
./run.sh render --run output/review-contact-hold-261003-CvjmWg/batch-261003-121121 --all
```

Both episodes used the same .20 m calibration standoff as the earlier
seed-42/43 pair, without walking, teacher intervention, or user corrections.
The block and root remained free; fingers and frictional contacts performed
the lift. This is **2/2** exploratory manipulation success (Wilson 95% interval
[34.2%, 100.0%]), not the planned held-out evaluation or a walking result.

| Seed | Trial length | Continuous successful hold | Configured final hold | Allowed hand-table samples | Maximum block clearance |
| --- | --- | --- | --- | --- | --- |
| 42 | 17.18 s | 4.41 s | 14.18–17.18 s | 0 | .1021 m |
| 43 | 17.76 s | 5.31 s | 14.76–17.76 s | 471 | .1538 m |

Seed 43 now completes rather than stopping on the historical finger/table
contact. Each final hold contains 600 physics samples, covering its full
three-second duration. Reports preserve dependency versions, commands, seeds,
model/scene hashes, reference artifacts, task trajectories, and full saved
states. Dependencies were torch/torch_musa 2.9.1, MuJoCo 3.2.7, NumPy 1.26.4,
SciPy 1.15.3, ONNX Runtime 1.23.2, and Transformers 5.8.1.

Both 640x480, 25 FPS H.264 videos decoded successfully: **430 frames / 17.20 s**
for [seed 42](../output/review-contact-hold-261003-CvjmWg/batch-261003-121121/attempt-00001/vision/video.mp4),
and **445 frames / 17.80 s** for
[seed 43](../output/review-contact-hold-261003-CvjmWg/batch-261003-121121/attempt-00002/vision/video.mp4).
`hold-validation.json` verifies full three-second hold intervals, terminal saved
states, decoded frame counts, simulation-time video mappings, launch source
hashes, and free robot/block joints. The last selected video state is within
one 25 Hz frame of the final saved state. Hold onset, midpoint, and final decoded
frames were visually inspected in
[hold-review.png](../output/review-contact-hold-261003-CvjmWg/hold-review.png).
The block remains above the tabletop in all selected hold frames. Rendering
restored saved states without stepping physics; its wall time is recorded
separately from simulation time and model latency. The renderer receipt is
`batch-261003-121121/render-261003-121504-843333.json`.

## Failure audit, contact drift and open-hand approach — 2026-10-03

The available `output/batch-261003-122608` contains 20 physical episodes, seeds
0–19, sampled block positions, .22 m standoff, historical finger gains 4/.2,
distal-finger preshape and three-second final hold. Read-only analysis found:

| Recorded outcome | Episodes | Diagnosis |
| --- | --- | --- |
| `grasp_alignment_missed` | 8 | Block center remains below the acquisition region |
| `grasp_not_acquired` | 5 | Closing produces transient rather than sustained opposition |
| `timeout` | 3 | Opposition acquired on the table is lost during lift; block does not clear 5 cm |
| Recorded success | 4 | Seeds 2, 12, 16, 18 reach two seconds, but 12/16/18 later drop |

The original labels and trajectories are preserved. Only seed 2 remains held at
the original final frame: **1/20 retained**, versus **4/20 threshold events**.
`audit/audit.json` stores original report/task hashes and per-phase forces and
motion metrics. All new artifacts are in `output/grasp-review-261003-IR6tcT/`;
the English [analysis report](../output/grasp-review-261003-IR6tcT/analysis.md)
and [clearance/force plot](../output/grasp-review-261003-IR6tcT/recorded-drops.png)
show the distinction.

During lift the wrist and block oscillate together. During subsequent hold,
wrist displacement is only about 7–11 mm, while the three dropped blocks move
about 10–11 cm relative to the wrist. Opposing normal force is approximately
19 N before release. A restored seed-16 hold snapshot has tangential loads of
about .4 N at each opposing contact with sliding coefficient 1. This supports
soft-contact drift as a material cause of slow slip, rather than inadequate
normal force alone. MuJoCo's [official slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)
recommends elliptic cones, larger impedance ratio and accurate Newton solves
for this numerical issue. Simulator calibration is not evidence of real motor
or material behavior.

Diagnostics reran from reset through frozen SONIC, shared checks, articulated
fingers and a dynamic block. In the initial four-seed ten-second hold control,
all four historical threshold successes eventually dropped. Larger stiffness
and damping alone did not prevent that. Uniformly curling all closure targets
20% further also failed; those targets were not adopted. A deeper wrist goal
and stricter acquisition threshold failed all eight later centered-target
pilots; those goals were not adopted either. Gaussian arm-reference smoothing
had mixed wrist vibration results. Slower saved-reference playback increased
wrist vibration in the two tested clips, so neither filter nor replay retiming
was adopted. `comparison.json` records the measured vibration metrics and the
high-pass definition.

An elliptic Newton solver with `impratio=10`, `tolerance=1e-10` and no NoSlip
post-processing eliminated later drops for the acquired seed-16/18 grasps in
the fixed-reference controls: continuous hold exceeded 12 s. Acquisition still
failed for some references because altered ground/contact dynamics change
tracking. Those controls are explicitly saved-reference diagnostics; they do
not establish closed-loop ARDY task success. Their references are hashed and
controller history/buffers are recreated by re-executing from episode reset.
No visual replay state was treated as a complete controller branch checkpoint.

A fresh paired calibration then used all original 20 prompts, block positions
and seeds. ARDY generated every moving phase from current executed history;
there were no saved-reference substitutions, teacher calls or user corrections.
Both configurations used .20 m standoff, 6/.4 finger gains, elliptic contacts,
unchanged closure targets/acquisition region, 4.8 s generated lift, and a
complete ten-second final hold within the existing 30-second timeout. The only
factor between them was the reach/lower finger posture:

| Approach posture | Retained success | Wilson 95% interval | Successful seeds |
| --- | --- | --- | --- |
| Historical distal preshape | 2/20 (10%) | [2.8%, 30.1%] | 3, 7 |
| Fully open until alignment | 6/20 (30%) | [14.5%, 51.9%] | 0, 4, 7, 10, 12, 19 |

The paired difference is **+20 percentage points**, with an episode bootstrap
95% percentile interval **[0, +40] percentage points** (20,000 resamples,
seed 20261003). The interval includes zero. This exploratory calibration
supports the selected default but does not establish a universal advantage or
replace held-out evaluation. The six open-hand successes complete the entire
ten-second hold. Its remaining failures are nine alignment failures, four
timeouts and one later drop, all retained in the denominator.

Nominal ARDY and physical SONIC errors both contribute to remaining alignment
failures. In three stopped episodes, nominal wrist FK is roughly 4–6 cm above
the requested goal and the executed wrist is another 2–4 cm above that nominal
FK. The requested wrist height is about .715 m, while execution remains at
.79–.80 m. Finger strength cannot fix this. These position checks used restored
executed root pose and named reference body joints; they are diagnostics of the
recorded state, not proof of universal workspace limits. Lift oscillation
remains a limitation; the selected hand/contact changes do not claim to
eliminate all frozen-model tracking vibration.

The selected defaults use open fingers, stiffness 6 and damping .4 under the
original motor and joint bounds, elliptic contacts, .20 m standoff, a 4.8 s lift
and a five-second final hold. The shorter default hold allows the walking
variant to fit the shared 30-second budget. New reports retain the original
`success_threshold_reached` event but require `retained_at_end` for task success;
`grasp_lost_after_success` exposes a later drop onto the table. All future
methods must share these calibrated scene, controller and assessment settings.
Finger limits, mass, sliding/torsional/rolling coefficients, timestep, free-base
and dynamic-block behavior are unchanged. No attachment or extra support force
was introduced. Protocol versions with different physics or retention rules
must not be pooled without identifying the change.

Commands and source snapshots are stored alongside the runs. For the paired
calibration:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh python scripts/calibrate_grasp.py --source output/batch-261003-122608 --out output/grasp-review-261003-IR6tcT/paired20 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --variants strong open --standoff .20 --lift-seconds 4.8 --fresh-motion
```

`paired20-source/` matches the complete source hash set in each paired plan.
The final default command, `./run.sh batch --grasp --batch 3 --seed 0
--output-root output/grasp-review-261003-IR6tcT/default`, independently verified
the new CLI defaults: seed 0 succeeded at 20.50 s with 8.53 s continuous hold;
seeds 1 and 2 failed alignment. The command correctly returned status 1 for
those physical failures. `default-source/` matches its launch plan. Both
default and paired reports contain dependency versions, model/scene hashes,
prompts, seeds, trajectories, fingers, forces, full states and simulation/wall
time. `tests-final.log` records **67 passing tests**, including late-drop
assessment, contact-solver configuration, CLI bounds and actual RGB/MP4 fixtures.
`smoke/` passed synthetic 9-to-17-frame packet/reference conversion, which is
not a model or physical grasp acceptance test. The diagnostic scripts remain
outside the declared baseline upload archive.

The retained seed-12 paired rollout was rendered with `./run.sh render --run
output/grasp-review-261003-IR6tcT/paired20/open-seed-012 --distance 2.2`.
The decoded H.264 video has 657 frames at 25 FPS (26.28 s), covers the last
saved simulation state at 26.24 s, and includes the complete ten-second final
hold. `improved-hold-review.png` visually confirms the block remains raised
at hold start, five seconds into hold, and the last saved frame.
`evidence-validation.json` verifies all 20 original report/task hashes remain
unchanged, all 40 paired source hash sets match their snapshots, and the
compiled scene preserves motor/joint limits, friction, body masses, timestep,
and free root/block joints. Seed 12 has opposing contact throughout all 2,000
final-hold physics samples and 13.515 s maximum continuous hold. The validator
command and SHA-256, video hash, and frame/duration measurements are recorded
in that receipt. These checks establish recording completeness and physical
configuration consistency; they do not remove the remaining tracking errors.

## Causal P context and Track4 video set - 2026-10-04

The adaptive torso-constraint calibration was run with fresh motion generation
and four held-out seeds:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh python scripts/calibrate_grasp.py \
  --source output/batch-261003-122608 --out output/grasp-review-261004-torso \
  --seeds 1 2 5 6 --variants open --standoff .20 --lift-seconds 4.8 \
  --hold-seconds 5 --fresh-motion
```

All four trials reached the alignment/contact phases but timed out at 30 s;
none reached the 5 cm clearance threshold, with maximum clearance about
0.00175 m and zero retained hold samples. This is a physical failure result,
not a successful grasp claim. The run is useful because the new torso limits
remove the earlier waist-roll reference violations while exposing the remaining
tracking/contact problem: joint tracking RMSE is 0.088--0.097 rad, close/lift
contacts are intermittent, and all four trials have zero contact force in the
terminal hold after the block settles back on the table.

The read-only analyzer accepted all four `open-seed-*` directories directly
under this calibration run. Each `nominal_context.csv` has 1,500 rows at 50 Hz,
strictly increasing frame/time values, ten causal lookahead offsets from 0 to
0.9 s, 290 position columns, 290 velocity columns and 40 quaternion columns.
The context contains current execution history, phase and planned finger
commands plus nominal future references; it does not contain future executed
states or teacher outputs. These checks establish the data contract needed by
the next Risk/Residual implementation, not model quality.

Per-attempt scene compilation was also checked directly from the recorded XML:
the elliptic scene has `cone=1`, `impratio=10` and `tolerance=1e-10`; the legacy
scene has `cone=0`, `impratio=1` and `tolerance=1e-8`. The execution runtime now
recompiles the exact per-episode scene before simulation, so random block
positions and contact settings cannot be silently replaced by the preload
model.

The Track4 B0 video set is available in the ignored output artifacts:

* Success, complete hold: `output/grasp-review-261003-IR6tcT/paired20/open-seed-012/vision/video.mp4` (657 frames, 25 FPS, 26.28 s).
* Success, independent example: `output/grasp-review-261003-IR6tcT/paired20/open-seed-004/vision/video.mp4` (640 frames, 25 FPS, 25.60 s).
* Failure with late loss: `output/grasp-review-261003-IR6tcT/paired20/open-seed-003/vision/video.mp4` (318 frames, 12.5 FPS, 25.44 s).
* Fresh timeout after torso constraints: `output/grasp-review-261004-torso/open-seed-001/vision/video.mp4` (376 frames, 12.5 FPS, 30.08 s).

The additional renders were produced with `./run.sh render --run <attempt>
--distance 2.2`; the failure and fresh-timeout clips used `--fps 12.5` to keep
software rendering practical. Rendering restores recorded states and does not
step physics or alter the reports.

The first three are B0 examples; the fresh timeout is the current diagnostic
failure. P videos are deliberately not labeled as available because the P
networks and training/evaluation entry point are not implemented yet. The exact
implementation gate, causal inputs, branch restoration rules, arm-only output
mask, bounded residual and shared evaluator are specified in
`docs/track4_requirements.md` and `docs/p_interface.md`. Once P exists, its
success and failure videos must use the same scene, SONIC, evaluator and
recording path as B0.

After these changes, `./run.sh tests` completed with 67 passing tests and
`./run.sh smoke --out output/grasp-review-261004-smoke-final` passed the
synthetic reference/packet check. Neither check establishes full ARDY/SONIC
compatibility or physical grasp success on the supported MUSA stack.
The required `./run.sh check` also passed the MUSA device, pinned ARDY/SONIC
source revisions, all local manifests, and the linear/layer-norm/attention
backend primitives. `./run.sh sonic --out output/grasp-review-261004-sonic-check
--repeats 5` passed the frozen SONIC encoder and decoder ONNX operator probe on
CPU; its synthetic inputs and disconnected graphs are not a control-loop or
physics result.

## Risk/Residual source merge and synthetic checks — 2026-10-05

The local `risk_residual/` learning source was integrated into the canonical
server tree without replacing the newer B0 grasp, execution, scene and context
logging changes. Its phase encoding was updated for the server's nine-phase
task vocabulary; the versioned history schema is now 119 fields. An optional
correction provider and shared arm-only reference checks were added. B0 does
not import the learning package. `docs/learning.md` lists the missing collector,
verified teacher, observation builder and P trial entry point.

On the S4000 host, `./run.sh tests` passed **78 tests**; the log is
`output/merge-261005-tests.log`. `./run.sh check` passed pinned assets and MUSA
operator checks (`output/merge-261005-check.log`). The baseline synthetic
reference smoke passed (`output/merge-261005-reference-smoke/report.json`).
The reduced-width synthetic Risk then Residual training check passed on MUSA
(`output/merge-261005-p-musa/report.json`). Its fixture contains no measured
grasp data or physics. These checks establish a working learning software
pipeline and optional correction interface; they do not establish task success,
teacher validity or full P control-loop behavior.

An actual unprompted B0 standing episode also completed after the merge:
`output/batch-261005-042037/attempt-00001/report.json` records frozen model
loading, 100 SONIC control frames, 2.00 s of MuJoCo physics, and no task
success value because no grasp task was requested. The separate scene helper
completed against the pinned SONIC assets (`output/merge-261005-scene.log`).

## Paired teacher and rollout conversion pilot — 2026-10-05

The first data collector replays saved ARDY references from a successful B0
attempt through two full SONIC/MuJoCo episodes. It changes one bounded arm
reference only in the perturbed nominal branch. A fingerprint of MuJoCo's
integration state, SONIC history, reference buffer, planned reference and
offset limiter at the activation decision must match before a pair is accepted.
The scene XML and compiled model hashes must also match. This is a verified
replay from reset, not an arbitrary mid-episode checkpoint restore.

The source attempt was
`output/grasp-review-261003-IR6tcT/paired20/open-seed-012`. The first pilot,
`output/teacher-pair-261005-pilot-01`, used a 0.10 rad right-wrist-pitch
offset during `lower`. The state pair matched and both branches succeeded;
therefore it is not evidence of needed corrective intervention. This run used
the initial collector version before the explicit `recovery_verified` field.

The second pilot,
`output/teacher-pair-261005-pilot-02/pair.json`, used a 0.15 rad
right-shoulder-pitch offset after 0.2 s in `lower`. The activation states and
physics models matched. The clean branch completed the task; the perturbed
branch timed out after a maximum continuous hold of 1.875 s, below the 2 s
success criterion. Its maximum block clearance was 0.1228 m. The pair reports
`pair_state_verified=true`, `teacher_verified=true` and
`recovery_verified=true`. These are two exploratory runs from one source seed,
not a success-rate estimate or a test of the learned P controller.
These existing branch reports retain the executor's old `method=B0` tag;
`pair.json` and `effective_context.csv` identify the actual injected reference
offset. Subsequent provider runs use `method=reference_correction_pilot`.

The single-parent source plan is
`output/teacher-pilot-source-plan-261005.json`. Running
`experiments.build_dataset --inspect-only` loads both executed branches,
restores recorded pre-decision MuJoCo states, builds 16-step history and
eight-step future windows, and validates the generated arrays through the
production `WindowDataset` loader in a temporary directory. With exploratory
tracking thresholds `[.06, .06, .1, .1]` rad, the pilot had 524 windows,
494 positive-risk labels, one recoverable correction sample and 10 stable
zero-offset samples. With `[.02, .02, .1, .1]`, all 524 windows were positive
and no stable sample remained. These values show threshold sensitivity; no
threshold has been calibrated or frozen. No persistent train/validation/test
dataset was written. Existing recorded attempts with complete context use one
exact instruction identity, so they cannot form disjoint prompt groups across
all three splits without new collection.

After the final recovery and clean-identity changes, `./run.sh tests` passed
all **84 tests**, including the GL rendering fixtures; the log is
`output/teacher-pair-261005-tests.log`. The baseline reference smoke passed
at `output/teacher-pair-261005-reference-smoke/report.json`. Its inputs are
synthetic and its report correctly records that no physics was executed. The
physical pair and single-parent loader audit above supply the separate
measured evidence for this data collector. Full split conversion and learned
P deployment remain unverified.

## External output storage migration — 2026-10-05

The S4000 host's `/home` filesystem had only about 15 GB available while
`/data` had about 12 TB available. The administrator-created
`/data/group3/dl-output` directory is owned by `group3:group3` with mode 2770.
The existing `output/` tree was copied there with `rsync`: 943 regular files,
4,147,245,327 logical bytes. A checksum-mode dry run reported no differences,
and both trees had 943 regular files and approximately 3.9 GB allocated.
The original tree remains at `/home/group3/dl-output-backup-20261005` and has
not been deleted. `/home/group3/dl/output` is now a symlink to the `/data`
directory.

`docker/run-musa.sh` detects an external output symlink and mounts its resolved
target at the same absolute path in the container. A no-simulation container
check resolved `output` to `/data/group3/dl-output`, read the historical batch
summary, confirmed it was writable, and showed the 13 TB `/data` filesystem.
`python -m unittest discover -s tests -v` in `dl-musa-render:latest` passed
85 tests with three optional RGB tests skipped. No new batch, training run, or
risk/residual outcome was generated during this migration.

## Batch data tooling checks — 2026-10-05

The fixed-split planner, completed-batch indexer, resumable controlled pair
wrapper and dataset supervision audit were added without changing B0 control.
`configs/data/collection.json` proposes six English prompt groups and disjoint
seed ranges for 120 train, 40 val and 40 test parent episodes. The planner's
actual CLI wrote `output/data-collection-261005-code-ready`, with plan hash
`690d6b1224d12743fbe445b7500a2c853abd278eb603e5d786572df114006ff1`.
It wrote six standard B0 resume commands and reported `physics_executed=false`.
None of those batch commands was executed during this code change.

The supported S4000 container ran 98 unit tests in 5.103 seconds, with three
optional RGB tests skipped and all remaining tests passing. The final log is
`output/data-code-check-261005/unittest-final.log`. New regression tests cover
seed overlap, implicit/explicit prompt identity, altered evidence, short and
pending episodes, stopped grasp failures, pair resume without duplicate
execution, interrupted and failed pair retention, zero teacher candidates,
source history/backend settings, masked supervision counts, and a converted
NPZ write/load round trip with an explicit not-ready Residual report. Pair
execution in these tests is mocked; converted windows are temporary synthetic
fixtures. These checks do not establish physical recovery, grasp success or
generalization of the proposed prompt/perturbation configurations.

With `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`,
`python scripts/check_baseline.py --device musa` passed the installed dependency,
pinned source, local weight and text-encoder compatibility checks. Its log is
`output/data-code-check-261005/assets.log`. No new packages or models were
downloaded. The converter now reports/skips only short or no-window parents
when explicitly requested, while changed evidence and malformed streams still
fail. Training refuses missing Risk classes or missing correction/identity
categories in train or val before optimization. A real disjoint dataset,
new physical pairs and the general corrective teacher remain unverified.

## Premature acquisition repair — 2026-10-05

The user collected `output/data-collection-261005-code-ready/batches/train-a`
with the saved 60-episode B0 plan (seeds 10000--10059). All attempts completed;
none succeeded. Reasons were 52 timeouts, seven `grasp_not_acquired` stops and
one prohibited robot-table contact. The recorded acquisition wrist-frame Z
ranged from -0.044987 to -0.038154 m, median -0.044182 m. The old lower gate
was -0.045 m. Only one episode cleared 5 cm, for 0.87 s, below the required
continuous two seconds. Task CSVs show contact during close followed by loss
during lift and no opposing contact in terminal hold. These are genuine
physical failures; reports and success criteria were preserved.

`hand_alignment` now scales the vertical half-extent margin by one-third,
instead of 1.5. For the 6 cm block, descent stops within 1 cm of the hand center
instead of 4.5 cm. Lateral bounds, frozen models, finger commands/gains, torque
limits, contact solver, free root/block, 5 cm/two-second success test, retention
check and 30-second timeout are unchanged. The regression fixture rejects the
observed premature -4.3 cm acquisition while accepting a centered block.

Matched exploratory pilots used `--grasp --batch 3 --seed 10000
--hold-seconds 10`, so all three compare the same train seeds and block poses:

| Configuration | Run | Retained successes |
| --- | --- | --- |
| Original gate | Original train-a attempts 1--3 | 0/3 |
| Original source, `--acquisition-z-min -.01` | `output/grasp-fix-261005-z01-pilot/batch-261005-225145` | 1/3 |
| Tight gate plus terminal-rotation position-goal candidate | `output/grasp-fix-261005-final-pilot/batch-261005-225720` | 0/3 |
| Final default tight gate, original position goals | `output/grasp-fix-261005-acquisition-pilot/batch-261005-230417` | 1/3 |

The final seed-10000 trial cleared about 9.82 cm and held continuously for
12.56 s, remaining held at the 25.28 s end after the full configured ten-second
hold phase. Seed 10001 failed to acquire stable opposing contacts; seed 10002
reached the success threshold but later lost the block, and remains a failure.
The final 1/3 Wilson 95% interval is approximately [6.1%, 79.2%]. This pilot
diagnoses a common premature-close failure, not held-out generalization or a
reliable production success rate. More lateral alignment and retention
calibration remain necessary. Test seeds were not used for calibration.

The position-goal candidate was withdrawn after physical verification: it
retained no successful grasp and one generated lower replan violated the
left-knee limit by 0.070 rad. The original checker correctly rejected it;
the limit was not relaxed. Candidate source is preserved at
`output/grasp-fix-261005-rejected-source/grasp.py`. Failed pilot reports remain
intact. No large batch or Risk/Residual training was launched during the repair.

The final server suite passed 98 tests with three optional RGB tests skipped,
log `output/grasp-fix-261005-acquisition-suite.log`. A fresh collection plan
`output/data-collection-261005-acquisition-fixed` contains the same six prompt
groups and 200 parent budget, pinned to the final source. It executed no physics.
Its plan hash is
`9c9195bfc3404159f80e04befc179ebd00ce02e6abbd7102f45b090268cbd8d9`.
The older code-ready and withdrawn-candidate plans cannot resume against this
source; do not edit them to bypass their provenance checks.

Offline RGB/MP4 rendering passed for original train-a attempts 1, 20 and 5
(376, 376 and 120 frames respectively, 640x480 at 12.5 FPS), and for the
successful Z-bound diagnostic attempt 1 (317 frames). MP4s are under each
attempt's `vision/video.mp4`; receipts are
`train-a/render-261005-225456-895898.json` and
`grasp-fix-261005-z01-pilot/batch-261005-225145/render-261005-230503-688081.json`
under output. Videos show simulation time, not model-generation wall latency.
They were copied to the user's local `outputs/grasp-fix-videos` for inspection.
