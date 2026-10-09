# Tabletop Grasp Task

The same task sequencer and assessment run with `--ardy`, `--kimodo` and
ARDY-based learned P control. [Commands](commands.md) contains runnable examples.
`--grasp` enables the right-hand block task; ordinary text motion has no task
success label.

## Scene and timing

A 6 cm, 80 g dynamic block starts on a 0.70 m table. Default block XY sampling
is X=[.36,.46], Y=[-.30,-.16] m; supported positions are X=[.34,.55],
Y=[-.38,-.08]. Each attempt seed fixes its sampled position and generator
phase seeds. Robot, block, fingers, SONIC history and reference buffers reset
between attempts.

The default manipulation path starts at the grounded table target, 0.20 m
before its front edge, and runs a two-second stand, reach, lower, close, lift
and hold. Generated phase defaults are reach 3.2 s, lower up to 3.2 s, close
2 s, lift 4.8 s and hold 5 s. The data/paired-evaluation commands can explicitly
use a 10 s hold. `--walk` starts farther back and adds approach, settle and
prepare. `--direct-start` retains compatibility with the default no-walk path.
All paths share a 30 s simulation timeout.

The root is free and the block has a free joint. Neither is moved by scripted
attachment or fixed-base control. SONIC produces body torques, and the shared
finger controller uses articulated joints and frictional contact. MuJoCo steps
at 200 Hz; body reference/recording ticks are 50 Hz.

## Grounding and generated motion

`ground_scene` reads current table geometry, block pose and robot pose from
MuJoCo. It produces root targets and wrist goals in world and generator frames.
This is privileged simulator-state grounding, not camera perception or a
language model inferring hidden coordinates. Snapshots are saved in
`grounding/<phase>.json`.

The default stand checks measured root position, heading, speed and both-foot
support before reaching. Walking must arrive within 6 cm of the target and
12 degrees of heading, below .12 m/s, with both feet contacting the floor for
.4 continuous seconds. A separate stability check follows preparation. The
G1 has no articulated neck; preparation is an 8-degree waist-pitch cue, not
visual inspection. SONIC does not consume world root XY, so generated paths
still require actual arrival assessment.

Moving hand phases condition the selected frozen generator on current pose,
wrist goals and standing body constraints. Reach raises the hand before
crossing the table edge. Lower approaches the block, and lift requests a
vertical wrist displacement from measured phase entry. Close/hold reuse the
checked reference rather than sampling new motion. Measured-pose transitions
and acquisition holds are sent through SONIC; they do not overwrite physics.

ARDY uses measured pose history. Kimodo uses measured state anchors and shared
transitions without native ARDY-style history conditioning. Differences in
these generator interfaces and their recorded wrist defaults must be disclosed
when comparing them.

## Acquisition and fingers

The default `--hand-approach open` keeps fingers open during reach/lower.
The physical alignment gate is derived from the measured wrist, the configured
`task_grasp_center` site and block half extents. Its lateral margin is 1.5×
and its vertical margin one-third of the half extent. `--acquisition-z-min`
can tighten the lower bound. `--wrist-offset` instead calibrates the nominal
generator goal; it does not relax the physical gate.

If descent misses alignment, a bounded number of lower replans requests fresh
motion from current state. Acquisition blends into a measured-pose SONIC hold,
then closes named fingers, including thumb opposition. Failure to align after
replans or to maintain opposing contact for .1 s at the end of close stops
the attempt. These are common nominal rules, not learned corrections or
additional user input.

Finger targets use a 2.5 rad/s rate limit and torque PD with current gains
6 Nm/rad and .4 Nm·s/rad. Upstream joint and torque limits remain active.
The tabletop contact profile uses elliptic cones, Newton, `impratio=10`,
`tolerance=1e-10` and no NoSlip post-processing. `legacy` selects the earlier
contact profile as an explicit calibration condition. Keep hand/contact
settings identical within a B0/P comparison.

## Success and failure

A successful trial must satisfy all of the following:

- The block's **lowest point**, computed using actual position and rotation,
  stays at least 5 cm above the tabletop for two continuous seconds.
- Opposing thumb and index/middle contacts exceed .01 N normal force.
- The threshold is reached within 30 s of simulated time, without a fall or
  prohibited collision.
- Clearance and opposing contact are retained at final assessment. A later
  loss invalidates an earlier threshold event.

Reaching the two-second threshold does not truncate the configured hold.
`success_threshold_reached` records the event; `retained_at_end` records final
retention. If hold finishes without success, execution continues until success
or timeout. A later drop is `grasp_lost_after_success` even if the block lands
back on the table. Incomplete old recordings require recollection; rendering
cannot add missing motion.

Hands/palms/fingers may touch the table with contact physics active.
Prohibited pairs are non-hand robot–table, block–floor, non-right-hand
robot–block and non-foot robot–floor. Existing fall, reference-limit and
joint-velocity stops also apply. General robot self-clearance and
pre-execution geometric clearance prediction remain unimplemented.

Execution status and task success are separate: a timeout may have a completed
`status=passed` execution while `task_success=false`. Report physical failures
and timeouts in the denominator. Invalid model/physics execution is disclosed
separately; it must not be reclassified as an ordinary task failure or recovery.

## English phase prompts

The `focused` profile describes only the current phase; `legacy` repeats the
whole mission. `--prompt` prepends a common prefix, and `--phase-prompts` replaces
specified phase text using `configs/grasp-prompts.json` as a template.

| Phase | Intended instruction |
| --- | --- |
| approach | Walk toward the supplied target, then stop |
| settle | Stand still with both feet supported |
| prepare | Gently incline the upper body and pause |
| reach | Raise the open hand before extending above the table |
| lower | Lower beside the block without restarting the reach |
| close | Keep the wrist steady while closing the fingers |
| lift | Raise the hand vertically while keeping grip |
| hold | Maintain the raised hand and grip |

Language specifies the action; coordinates belong to spatial constraints.
Inspect the actual per-phase generator report and event log when diagnosing
instruction following. Compare prompt changes with matched seeds/settings,
retain failures, calibrate on training/validation and freeze before final test.
A plausible motion or better wording alone does not establish grasp success.

## Measurement limits

Generation pauses simulated time. Record loading/generation wall time,
controller latency, physics execution wall time and simulation time separately;
sampling FPS is not evidence of real-time streaming. No-walk contact-grasp
results do not establish walking grasp or real-robot transfer. Current measured
results, confidence intervals and failure counts are in
[verification](verification.md).
