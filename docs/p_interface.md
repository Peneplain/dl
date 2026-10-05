# Risk + Residual Input Contract

This is the implementation contract for the next P method. It keeps the
learned modules downstream of the frozen baseline and makes causal inputs
reviewable before training starts.

## Recorded streams

`nominal_context.csv` is sampled once per 50 Hz SONIC control tick. The
`sim_time` and `frame_index` columns identify the current execution sample.
The row contains:

- current free-root pose and velocity, named 29-joint body state and velocity;
- current named finger targets and task phase;
- ten nominal samples at offsets 0, 0.1, ..., 0.9 seconds, each with 29 joint
  positions, 29 joint velocities, and a root quaternion in wxyz order.

`task.csv` is sampled at 200 Hz and remains the label stream. It contains the
dynamic block pose, lowest clearance, wrist pose, contact forces, held time,
finger state and finger target. Join labels to a context row by timestamp using
the documented causal interval; never append future task rows to model inputs.

The two streams deliberately separate nominal inputs from privileged labels.
The logger never writes future executed states, corrected references, teacher
outputs, or camera observations into the nominal context. A phase string and
planned finger targets are common context for Risk and Residual, so open-hand
samples are not mislabeled as grasp failures.

## Window and split rules

Create a parent-episode manifest before making windows. Keep all windows from a
nominal episode, its counterfactual branches, and its teacher branch in the
same split. Hold out complete scene seeds and prompt paraphrase groups. Store
the manifest hash, source report/task/context hashes, time bounds, and label
availability mask with each extracted window.

Nominal risk labels come from the nominal future execution. The first paired
pilot replays two episodes from the same reset and verifies a matching
physical/controller-state fingerprint immediately before the controlled arm
perturbation. This supports one same-decision clean/perturbed comparison.
General branching from an arbitrary saved decision still requires explicit
restoration of SONIC history, reference buffers, finger state, contact solver
warm start and disturbance RNG; the pilot does not claim this capability.
Corrected outcomes never replace nominal risk labels. Failed clean teacher
branches supervise Risk only; they do not become Residual targets.

## Model and executor boundary

Use an explicit interface similar to:

```text
risk = risk_model(history, nominal_lookahead, phase, planned_fingers)
offset = residual_model(history, nominal_lookahead, phase, planned_fingers, risk)
```

Risk is trained and frozen before Residual training. The Residual output is a
bounded offset for the named arm joints only. Root, torso, legs, and fingers
remain controlled by the shared B0 path. At low risk, return a zero offset and
ramp any existing offset to zero under the common rate limit. Recompute joint
velocities and dependent reference fields after applying and resampling the
offset, then run the unchanged checks and stop conditions.

The P executor must log the risk features, threshold decision, arm mask,
bounded offset, rate-limited offset, and inference latency without exposing any
teacher or future executed state to the online model. Its output should be
drop-in compatible with the existing `ReferenceSequence` and
`ReferenceBuffer`, allowing B0 and P to share the same video and evaluator.
