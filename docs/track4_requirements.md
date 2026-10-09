# Track 4 Submission Requirements

The project uses the assignment's Unitree G1 simulation and frozen SONIC
control requirement. ARDY and Kimodo are distinct implemented motion generators.
The Risk/Residual extension studies selective correction of ARDY and is
reported separately. Implementation evidence and unresolved limits are in
[verification](verification.md); the approved research plan is
[proposal.tex](proposal.tex).

## Evidence to include

| Requirement | Repository support | Submission work |
| --- | --- | --- |
| Supervised Archon session | Simulation code does not replace the visit | Book at least two days ahead; explain real demonstration recording and policy-inference inputs, data, outputs and control flow. Keep restricted data private. |
| SONIC integration | `baseline/`, model/source locks and [integration contract](baseline.md) | Explain named reference conversion, G1 representation, frozen SONIC tracking, free-root/lower-body behavior and separate fingers. |
| Text-driven grasp/lift | English phase prompts and shared physical tabletop task | Show text and corresponding grasp/lift behavior. Disclose simulator-state grounding and manipulation-only versus walking conditions. |
| Two distinct motion-generation methods | Frozen ARDY (`--ardy`) and Kimodo (`--kimodo`) | Compare with predeclared common instructions, seeds, scene/controller settings and repeated trials. Changing prompts alone is not a second generator. |
| Evaluation and analysis | Reports, seeds, task states, hashes and matched B0/P results | Report success/uncertainty, response time, user corrections, failures, instruction following, precision, physical feasibility and object interaction. |
| Demonstration videos | State-based RGB/MP4 renderer | Retain representative successes and failures for both submitted generator methods, and P if claiming the extension. Show actual instructions. |
| Public reproducible source | Setup, commands, configs, data tooling and tests | Supply the repository link; exclude credentials, model weights, generated bulk data and restricted third-party/robot material. |

Kimodo's tuned 11/20 batch and the 60-pair B0/P experiment have different seeds
and selection context. They cannot be presented as one fair three-method
comparison. The existing B0/P repeat is a single learned-checkpoint experiment,
not completion of all proposal controls and ablations.

## Deliverables

- A 5–8 page report excluding references, covering the Archon workflow,
  simulation system, compared generators, evaluation, failures and extensions.
- A 10-minute presentation with required demonstration, followed by 5 minutes
  of Q&A. Recorded demonstrations are acceptable.
- Videos with text instructions and corresponding behavior, including success
  and failure for both methods.
- A public GitHub repository with setup, configurations and evaluation code;
  include data access and conversion scripts when claiming the data extension.

If claiming the optional data-collection extension, provide synchronized
images/actions/states converted to LeRobot with instructions, timestamps and
episode boundaries. Current rollout/NPZ training files are not automatically
LeRobot exports. Real-robot deployment is an optional supervised activity that
requires the course's prior review/approval; the simulation results do not
establish it.

Acknowledge external models/code/data and LLM assistance under the course's
policy. Report unsuccessful trials and unfinished research honestly. A focused
comparison with reproducible evidence is valuable even without a high success
rate or an improvement from P.
