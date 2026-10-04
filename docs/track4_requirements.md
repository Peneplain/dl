# Track4 Delivery Requirements

This checklist is derived from the Track4 brief supplied with the project. It
keeps the course deliverables separate from the B0 pilot status and identifies
what must be completed before a final demonstration.

## Required evidence

| Requirement | Repository evidence | Current boundary |
| --- | --- | --- |
| Explain the supervised Archon session | Report and presentation section covering inputs, recorded data, model outputs, and robot control flow | The visit is external to this repository; no visit is claimed here. |
| Use SONIC as the low-level controller | `baseline/sonic_policy.py`, `baseline/simulation.py`, `docs/integration.md` | B0 uses the pinned SONIC encoder/decoder/config and records provenance; full-model MUSA evidence is kept in `docs/verification.md`. |
| Text-driven tabletop grasp and lift | `baseline/execution.py`, `baseline/grasp.py`, `docs/grasp.md` | B0 has a simulator-state-grounded pilot with a dynamic block, articulated fingers, and a free root. Exploratory failures remain in the denominator. |
| Compare two distinct motion-generation methods | B0 is implemented; the proposal defines Risk + Residual (P) as the second method | P networks, training, and the shared correction hook are not implemented yet. Prompt variants or finger variants must not be presented as a second motion generator. |
| Shared repeated evaluation | `docs/proposal.tex` defines paired seeds, prompts, conditions, success, Wilson intervals, and bootstrap differences | The existing 20-episode calibration is B0 exploratory evidence, not the final B0/P comparison. |
| Show success and failure videos for both methods | `scripts/render.py` restores recorded states without stepping physics | B0 examples are listed below. P videos require P implementation and matched rollouts; do not relabel B0 variants as P. |
| Public reproducible repository | Source, configs, tests, commands, and provenance documentation | Keep checkpoints, credentials, generated rollouts, and restricted Archon data out of Git. |

The report must also discuss instruction following, spatial precision, physical
feasibility, object interaction, response time, user corrections, failures, and
lower-body stabilization. The current executor records simulation time,
wall-clock generation/inference time, correction count, contact outcomes,
tracking error, and free-root state so those quantities can be reported without
mixing clocks or treating windows as independent trials.

## Video evidence

These B0 recordings are rendered from complete MuJoCo state archives. The
request JSON next to each video supplies the text-driven phase prompts; the
report must show that text beside the corresponding video rather than implying
that the renderer performed inference.

| B0 case | Video | Recorded outcome |
| --- | --- | --- |
| Open-hand success | `output/grasp-review-261003-IR6tcT/paired20/open-seed-012/vision/video.mp4` | Retained lift/hold in the paired calibration. |
| Open-hand success | `output/grasp-review-261003-IR6tcT/paired20/open-seed-004/vision/video.mp4` | Retained success example. |
| Open-hand failure | `output/grasp-review-261003-IR6tcT/paired20/open-seed-003/vision/video.mp4` | `grasp_lost_after_success` in the historical protocol. |
| Open-hand failure | `output/grasp-review-261004-torso/open-seed-001/vision/video.mp4` | Fresh torso-constrained run timed out after contact loss. |

The final submission needs the same success/failure selection for P under the
same prompts, scene seeds, physics, controller, and video settings. A B0-only
video set is useful for debugging but does not satisfy the method-comparison
deliverable by itself.

## P implementation gate

The frozen B0 logger now writes `nominal_context.csv` at the SONIC 50 Hz clock.
Each row contains the measured current root/body state, current planned finger
targets and phase, plus the 0--0.9 s nominal joint position/velocity/quaternion
lookahead. Object, clearance, contact, and threshold labels stay in
`task.csv` at 200 Hz. This is the intended causal extraction boundary for P:
future executed states and teacher outputs are absent from the input file.

Before claiming P results, implement and verify the following in a separate
package that depends on `baseline/`:

1. Split parent episodes, including nominal and counterfactual branches, before
   extracting windows; hold out prompt paraphrases and scene seeds.
2. Restore simulator state, SONIC history, reference buffers, controller
   history, and disturbance seeds for every counterfactual branch.
3. Train Risk first from nominal future execution, freeze it, then train bounded
   arm-only Residual offsets using predicted risk features. Do not feed future
   executed states or teacher corrections at inference time.
4. Apply the shared arm mask, correction bounds, rate limit, clearance checks,
   and low-risk zero-offset gate before the existing shared checks. Recompute
   velocities and dependent fields after correction and resampling.
5. Reuse the B0 sequencer, articulated hand controller, scene, evaluator,
   timestamps, and video renderer. Record model hashes, thresholds, latency,
   gate decisions, corrections, and branch provenance with every P episode.
6. Run the planned B0/B1/I2/P conditions with paired instructions, seeds, and
   perturbations; report Wilson intervals and episode-level bootstrap intervals.

Until this gate is satisfied, the repository status is an honest B0 pilot plus
P-ready causal recording, not a completed two-method study.
