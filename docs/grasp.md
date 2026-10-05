# B0 tabletop trials

Use `./run.sh batch --grasp` or `./run.sh manual --grasp` from the S4000 host.
Without `--grasp`, neither task objects nor automatic grasp prompts are present.
The two modes use the same frozen ARDY, frozen SONIC and `SonicSimulation`.
Commands, output structure, resume and rendering are documented in
[commands.md](commands.md); prompt settings are in [prompts.md](prompts.md).

The default XY range is X=[.36,.46], Y=[-.30,-.16] metres, constrained to the
supported table bounds X=[.34,.55], Y=[-.38,-.08]. A block is 6 cm wide, 80 g,
and starts on the 0.70 m tabletop. Each attempt seed determines its XY sample;
phase generation seeds add phase indices to that seed (modulo uint32).
Models persist, but physics, fingers, SONIC history and buffers reset per attempt.

## Execution and assessment

The sequencer starts with a two-second SONIC stand at the grounded table target,
then runs reach (3.2 s), lower (3.2 s maximum), close (2 s), lift (4.8 s),
and hold (5 s). With `--walk`, the robot starts farther back and adds approach
(2 s), settle (.8 s), and prepare (1.2 s) before reaching.
Approach transitions take .2 s; generated arm references take .4 s.
Acquisition switches to a measured-pose hold through SONIC over .1 s. The whole trial,
including approach, remains limited to 30 simulated seconds.

The default root starts at X=.11 m, 0.20 m before the table's front edge,
with lateral position aligned to the block. `--table-standoff .20` controls
that target distance. With `--walk`, `--start-back .45` adds 0.45 m of initial
separation and starts the root at X=-.34 m. The initial root position
is part of the scene/reset only. All subsequent displacement must result from
SONIC torques and free-base physics.

The default manipulation path omits approach, settle, and preparation;
`--direct-start` remains a compatibility alias for this no-walk mode.
The normal two-second stand must
still pass the shared position, heading, speed, and double-foot stability gate
before reach begins. Direct-start results assess grasp mechanics only and must
not be reported as full walk-and-grasp success or pooled with full-task trials.

`ground_scene` reads the actual table geometry, dynamic block pose, and robot
pose from MuJoCo. It computes world-space and root-relative object positions,
a table-facing heading, and an approach target aligned with the robot's right
side. These are privileged simulator-state observations, not vision. Snapshots
are written to `grounding/PHASE.json`; no future executed state is supplied.

Approach passes a timed root path and heading to ARDY's pinned
`Root2DConstraintSet`, mapping MuJoCo XY to ARDY XZ and offsetting future frame
indices by history length. Feet are free during locomotion. Settle holds the terminal walking reference
through SONIC while checking actual double-foot support; it does not generate
a new motion. The initial arm reference parks the hands behind the table,
with shoulder pitch .4 rad and elbow .8 rad, without changing policy defaults.
Preparation starts only after the measured root stays within 6 cm of the approach
target, within 12 degrees of the table-facing heading, below .12 m/s horizontal
speed, and with both feet contacting the floor for .4 continuous seconds.
Crossing closer than the configured standoff minus .10 m stops the attempt.
A missed target or an unstable stop ends the trial with an explicit failure.
The configurable standoff range is .20-.55 m; the current .20 m pilot default
replaces the unreachable historical .42 m stance. Complete task success with
the new defaults is still being calibrated. These thresholds are pilot
settings that must be calibrated and shared across
methods. There is no retry-until-success loop.

The frozen SONIC G1 mode consumes body joints, velocities, and root orientation;
it does not consume world root XY. The path conditions ARDY's gait, while actual
arrival is checked from simulation. This is not a claim of precise SONIC global
position tracking. A correctly generated path can still fail physical arrival.
Standing constraints are rebuilt from the arrived pose before preparation and
again before reach, so feet and pelvis are not pulled back toward the distant
starting position.

The pinned G1 has no articulated neck. Preparation gently inclines the upper
body using an 8-degree waist-pitch target, holding both wrists at their measured
positions. Named-joint forward kinematics produces a torso orientation condition
for ARDY; it does not overwrite generated references or simulator state.
ARDY still generates the motion and SONIC executes it. This is a posture cue,
not visual inspection or a claim that the head's optical axis points at the block.
The same .4-second continuous arrival/stability gate is applied after preparation;
`prepare_not_settled` stops an unstable trial. Subsequent hand phases condition
the torso on its measured prepared pose. Actual torso tracking remains a physical
validation question. The shared nominal torso condition is available to every
method; future learned residual outputs remain restricted to the arms.

The subsequent hand phases ground goals in current simulator block state. Sparse wrist
positions first raise the hanging hand before crossing the table edge; lower,
and lift use smooth interpolation. Close and hold preserve their existing
checked reference instead of sampling another motion. The lift goal is .14 m
above the measured wrist at phase entry, with that wrist orientation retained. Wrist orientation rotates gradually
from the measured pose to a side grasp.
Standing feet, pelvis and unused hand goals are re-anchored to measured state
at each generated hand phase.
`ArdyService.generate(..., pose_constraints=...)` maps those positions and
rotations from MuJoCo world axes to the pinned ARDY skeleton, offsets frame
indices by measured history length, and passes normalized observations/masks
through the upstream constraint interface. It saves both inputs for inspection.
ARDY generates the moving body references. A shared acquisition hold uses the
measured body pose as a constant SONIC reference; it changes no simulator state.
No IK trajectory or teacher body controller replaces generated motion.

The shared reference validator and timestamped 50 Hz buffer feed SONIC's
configured 0.9-second lookahead. The task transitions described above blend
checked references;
velocities are recomputed after resampling/transitions. Fingers receive named,
bounded targets with a shared 2.5 rad/s rate limit and the existing torque PD.
The current torque gains are 6 Nm/rad stiffness and .4 Nm s/rad damping
(historical gains: 4 and .2). Upstream motor torque and finger joint limits are
retained; a larger gain does not bypass a saturated motor. The tabletop contact
profile uses elliptic cones, Newton, `impratio=10`, `tolerance=1e-10`, and no
NoSlip post-processing. The `legacy` profile reproduces pyramidal cones and
unit impedance ratio for calibration comparisons. Both profiles use the same
friction coefficients, mass, timestep, free joints, controller, and checks.
The root remains free and the block has a free joint. Lifting depends on hand
frictional contacts, with no weld, attachment, scripted object motion or support
forces. Physics advances at 200 Hz. Models stay loaded across batch episodes,
while robot, object, hand targets, controller history and buffers reset.

During reach/lower, the default `--hand-approach open` commands all fingers open
until the measured alignment gate passes, then uses the bounded closure targets.
`--hand-approach preshaped` reproduces the historical 1.3-rad distal index/middle
preshape for comparisons. The two approaches use the same geometry-derived
acquisition gate, closure rate, motor limits and opposing-contact check. The gate
is recomputed from the current MuJoCo wrist pose, the configured
`task_grasp_center` site and the current block half extents, with a 1.5x extent
margin. An optional `--acquisition-z-min` only tightens its lower wrist-frame
bound. The optional `--wrist-offset` is a calibration override for the nominal
ARDY goal and is distinct from the physical gate.
When descent misses the gate, up to the configured measured-state lower replans
request fresh ARDY references from current execution history. At acquisition,
the controller holds the measured pose through a configurable transition and
closes the fingers, including .6 rad thumb opposition. Missing alignment after
those replans or less than .1 s of continuous opposing contact at the end of
close stops the trial explicitly.
These are shared nominal task rules, not learned residuals or user corrections.
All thresholds remain calibration settings, not frozen evaluation results.

Every generated hand phase also conditions ARDY on the measured waist yaw,
roll, and pitch. The preparation phase changes only the pitch trajectory; the
other two axes stay at their measured values. This prevents an unconstrained
torso rotation from creating a reference that the shared joint-limit checker
must reject after a valid hand alignment, while keeping the check itself
unchanged.

This collector pauses **simulated time** during ARDY generation. Batch physics
runs without wall-clock pacing; manual execution is paced for viewing. It records generation, loading, SONIC latency,
simulation time and execution wall time separately. It does not demonstrate
real-time streaming or delayed-action robustness.

At each physics step, assessment uses the block's actual free-joint pose and
rotation to compute its lowest point. Success requires at least 5 cm clearance
above the 0.70 m tabletop, maintained for two continuous seconds within 30 s,
with opposing thumb and index/middle contacts (>0.01 N normal force) and no
fall/prohibited collision. Interrupted contact resets the hold interval.
Open-hand phases cannot trigger a missing-grasp failure. The collector ends
after the complete configured five-second hold phase, at a failure stop, or
at the 30 s timeout. Reaching the success threshold during lift or hold does
not truncate these phases. If the configured hold finishes without success,
the collector continues holding until success or timeout. A later prohibited
contact or fall still invalidates an earlier success. Video duration follows
recorded simulation time. `success_threshold_reached` preserves the original
two-second event; `retained_at_end` reports whether the block remains above the
clearance threshold with opposing contact at the final assessment. Both are
required for `task_success`. Otherwise, `grasp_lost_after_success` identifies
a later drop, including a drop onto the table that causes no prohibited contact.
Historical success labels remain intact; the audit tool reports their retention
separately. There is no fixed 14-second cap. Old recordings that
ended early require a new rollout to include the missing hold motion.
The added contact criterion is a conservative pilot operationalization of
“held”; contact thresholds/hand geometry need validation before evaluation.

Both hands, palms, and fingers may touch the tabletop or table legs. Their
contact physics remains active. Pilot prohibited pairs are non-hand robot–table,
block–floor,
non-right-hand robot–block and non-foot robot–floor. Existing shared fall,
joint-velocity and reference-limit checks still apply. Robot self-clearance and
pre-execution geometric clearance prediction remain unimplemented. The pilot
contact policy must be reused by future methods and frozen after validation.

## Results and limits

`summary.txt` and `summary.md` identify every successful/failed attempt;
`results.csv` provides the same rows for analysis. `report.json` records the
execution status separately from `task_success`: a timeout can finish execution
with `status=passed` while being labeled **FAILED** in all summaries.
Completed failures and timeouts remain in the success denominator, including
on resume. Interrupted attempts remain pending and use a new retry directory.
No completed physical failure is automatically retried or refined.

For clarity, the prohibited-contact labels mean:

| Label | Contact that stops the trial |
| --- | --- |
| non-hand robot-table | Any robot body except either hand touching the tabletop or table legs |
| non-right-hand robot-block | Any robot body except the right hand touching the block |
| non-foot robot-floor | Any robot body except the feet touching the floor |
| block-floor | The block touching the floor |

These pilot rules are shared across methods. The hand allowance includes
`left_hand_*` and `right_hand_*` bodies plus the two wrist-yaw bodies carrying
the pinned G1's palm meshes. Wrist-pitch, wrist-roll, forearm, torso, and leg
contacts with the table remain prohibited. `hand_table_contact_steps` counts
200 Hz samples with an allowed contact; these contacts cannot satisfy the
opposing finger-block contact requirement. Historical results retain their
original contact policy and must not be pooled with the revised pilot rules.
Detailed first-contact body names and penetration are stored in report.json.

The recorded state includes a compiled scene and full 50 Hz integration state.
`render` restores those saved states without new physics or policy inference;
it is visual replay, not an independent SONIC acceptance trial. The original
B0 rollout already executes through SONIC. A future separate teacher generator
must verify its corrections through SONIC from the same perturbed state.

`expert_valid` remains false. These attempts are physical B0 records, not a
verified teacher training dataset. A separate paired pilot can replay a saved
ARDY reference from reset and check an equal controller/physics fingerprint at
one arm perturbation decision. It cannot branch arbitrarily after divergence.
General controller-history checkpoint restoration and teacher recovery remain
planned. RGB is output only, not an inference observation.

The four-run paired prompt pilot still had no successes; see
[verification.md](verification.md). Prompt settings, hand geometry and contact
thresholds need validation before freezing comparative evaluation. A pilot
batch does not replace the proposal's paired conditions or learned-method
training seeds. The baseline archive retains an explicit source allowlist.


## Improving grasping while preserving a fair learned comparison

The assignment permits stepwise instructions, but requires two distinct motion
generation methods through the same SONIC controller. The intended comparison
is B0 (frozen ARDY with shared grounding and constraints) versus P (the same
nominal generator followed by learned predictive Risk + Residual arm-reference
correction). Explain P as a learned corrected-reference generation pipeline,
with its own inputs, outputs, training and ablations. Prompt variants alone are
not the second method. Course acceptance of this interpretation is not itself
established by this implementation.

A measured diagnostic from the earlier seed-43 settled pose found a 0.539 m
shoulder-to-lower-wrist-goal distance versus a 0.410 m sum of arm segment lengths.
That goal is unreachable while preserving that shoulder position, even before
joint limits or collisions are considered. This is one scene/state, not a
workspace certification; see [verification.md](verification.md). The initial
backoff and final grasp standoff must be calibrated separately.

Improve the common execution pipeline on pilot/validation episodes first:

1. Calibrate the final approach distance jointly with arm reachability and the
   complete open-hand collision geometry. More distance reduces early table
   overlap but can put the block outside the feasible arm workspace. Use the
   actual settled pose, not the requested root endpoint.
2. Validate the wrist-to-grasp-center transform and hand orientation against the
   articulated collision meshes. Check the swept fingers during raise, transfer,
   lower and closure; wrist height alone cannot establish table clearance.
3. Measure generated-reference error separately from SONIC tracking error.
   A feasible target, a feasible generated pose and a feasible executed pose
   are different checks. Validate the nominal hand approach before learning.
4. Calibrate shared phase transitions using measured wrist alignment and
   opposing finger contacts before lifting. Any bounded wait must fit the same
   30-second budget. Such alignment/contact gates and geometric clearance
   prediction are recommendations, not implemented features in this collector.

Freeze this calibrated common pipeline before B0/P comparison. Do not deliberately
weaken B0, relax P's contact rules, or select test scenes using P's outcomes.
A useful baseline should demonstrate actual contact grasps in nominal validation
scenes. No target success percentage is guaranteed or used to select test results.

The proposal's primary robustness conditions are nominal, grasp-target bias and
action latency. Choose recoverable perturbation magnitudes on validation data,
then freeze them before held-out paired tests. An offline generation pause is
not an action-latency perturbation; a real delayed-command buffer would be needed.
Risk should anticipate prohibited arm/table contact or grasp degradation from execution
history and upcoming references; Residual should make small bounded arm offsets
and return to zero at low risk. This offers a testable source of improvement
without intentionally damaging nominal execution. Stable nominal identity
targets help preserve already successful behavior.

Walking overshoot, an unstable lower body, an unreachable target, or unsuitable
finger mechanics cannot generally be repaired by an arm-only residual. Report
failure causes, arrival/stability rates, and overall grasp success separately,
while keeping every failed approach in the overall denominator. Teacher arm
corrections must succeed through frozen SONIC from the same perturbed state;
otherwise retain risk supervision only. Split parent episodes before windows,
then use matched prompts, scene/ARDY seeds and disturbances, the proposal's
20 paired episodes per core condition, Wilson intervals and paired bootstrap
comparisons. General teacher branching and this full evaluation remain planned;
the paired clean-versus-arm-perturbed pilot is described in
[learning.md](learning.md).
