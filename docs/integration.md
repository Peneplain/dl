# ARDY, SONIC and MuJoCo integration

The `baseline/` package contains model-independent reference, provenance and
frozen SONIC execution utilities. `scripts/run.py` connects text input,
ARDY reference generation, the timestamped reference buffer and a free-base
MuJoCo G1 loop. Batch and manual modes share `baseline/execution.py`; an
explicit `--grasp` enables the pilot tabletop task.
The [B0 bring-up guide](baseline.md) now supplies pinned downloads, an explicit
MUSA/CPU ARDY generation entry and a CPU SONIC ONNX graph check. These are
separate acceptance gates. The [pilot grasp collector](grasp.md) reuses this
executor with a dynamic table/block scene, fixed phases and ARDY spatial
constraints; physical grasp/lift acceptance remains incomplete.

## Upstream interfaces checked

- [ARDY official code](https://github.com/nv-tlabs/ardy):
  `ARDY-G1-RP-25FPS-Horizon8`, G1 MuJoCo-qpos CSV export, text and kinematic constraints.
- [SONIC streaming protocol](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html):
  protocol v1 / encode mode 0, full 29-joint position/velocity references.
- SONIC source reviewed at commit `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`:
  [G1 ordering](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic/envs/manager_env/robots/g1.py),
  [C++ packet receiver](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp),
  [motion endpoint](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp).

Source and model revisions are recorded in `configs/baseline.lock.json`. The
downloader records hashes for the matching default SONIC encoder, decoder and
observation configuration. A source pin establishes the inspected wire contract;
physical behavior still needs validation. Store asset hashes in run manifests.

## 1. Establish the external baseline

Use the official environment instructions for each upstream project only when
you need an upstream diagnostic. The repository's supported B0 path puts ARDY,
SONIC simulation dependencies and MuJoCo in the single `.venv-baseline-musa`;
you do not need to create the upstream `.venv_sim` for `scripts/run.py`.
Confirm the available inference host supports the actual SONIC backend; an S4000
training allocation does not imply TensorRT compatibility.

From the SONIC checkout, the upstream-only documented simulation entry is:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

The documented deployment entry, from its `gear_sonic_deploy` directory, is:

```bash
bash deploy.sh --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose sim
```

These are upstream commands, not scripts supplied by this repository. Complete
upstream installation and the documented start/streaming controls before using
them. Verify standing and a known reference in MuJoCo before adding generated motion.

Build a reachable tabletop scene with a free-root G1, dynamic block, articulated
hand and frictional contacts. Do not weld the object to the hand or fix the base.
Record the scene XML, contact settings, actuator gains and joint limits. Prove
one physical baseline grasp/lift before collecting learned correction data.

## 2. Convert and check reference data

The official ARDY batch generator exports G1 motion as MuJoCo qpos CSV. Obtain
the exact source joint names from the selected skeleton/robot asset. A JSON list
of those 29 names is required; never relabel columns based on an assumed index.

```bash
python scripts/prepare_reference.py \
  --qpos-csv /path/to/ardy-output.csv \
  --joint-names /path/to/source-joint-names.json \
  --source-fps 25 --out output/ardy-reference.npz --packet
```

This converter explicitly expects 36 columns: root xyz, root quaternion **wxyz**,
then 29 body-joint angles. If an upstream export has a different layout, adapt
it explicitly rather than passing it through. The root translation is not a v1
packet field. Root orientation is preserved by the reference adapter.

The converter remaps names, samples at 50 Hz, recomputes velocities and writes an
offline packet file. It opens no socket. It does not establish joint limits,
standing feasibility, object clearance or grasp behavior; those checks belong
to the shared physical reference pipeline.

Wire fields are `joint_pos [N,29]`, `joint_vel [N,29]`, `body_quat [N,4]` in wxyz,
and increasing `frame_index [N]`. Bytes use a `pose` topic prefix, a null-padded
**1280-byte** JSON header and contiguous little-endian arrays. Some upstream
comments still mention 1024; both the inspected constant and C++ receiver use
1280. The endpoint accepts `body_quat` and `body_quat_w` aliases. Optional separate
Dex3 hand commands have seven joints per hand; they are never learned offsets.

`JointStreamEncoder` is an append-only frame allocator for converted chunks.
Live overlapping replans must replace future references by their absolute frame
indices, rather than assigning new indices to duplicate future times. Complete
that scheduling/transport integration against the pinned receiver before live use.

## 3. Run the B0 control loop

The shared executor is `baseline/execution.py`, invoked by `scripts/run.py`. It:

1. Timestamps measured poses and ARDY requests.
2. Converts ARDY output to the canonical reference. Manual mode disables and
   discards new input during generation/execution, then waits for the next JSON.
3. Buffers the lookahead required by the selected SONIC observation configuration.
4. At each 50 Hz reference tick, checks limits from the active G1 asset using
   elapsed simulation time, then recomputes velocity and dependent kinematics.
5. Applies the shared joint checks, passes the reference to frozen SONIC and
   issues the separate finger commands.
6. Holds or stops on underrun or invalid references, logging the event. Preserves
   clocks and frame indices, and resets controller and buffer state per episode.

`ReferenceBuffer`, the ONNX observation adapter, MuJoCo executor and rollout
logger are available in `baseline/`. `baseline/grasp.py` now supplies a pilot
table/block scene, simulator-state root approach goals and arrival checks, a
right-hand sequencer, finger targets and physical contact assessment. The pinned
SONIC G1 mode does not consume world root XY; root paths condition ARDY gait,
and actual position is assessed separately before reaching. General instruction parsing, pre-execution clearance
checks and verified grasp/lift still need work.
The logger saves full MuJoCo state at 50 Hz plus a compiled scene under
`rollout/`. The separate `scripts/render.py` restores frames
without advancing physics, producing camera RGB in `vision/images.npz` and
optional MP4. The default camera is fixed third-person; named cameras from the
recorded scene provide head/wrist views in the tabletop collector. These images are recording
outputs and do not change the privileged-state inference interfaces.
Additional body orientations or positions, if supplied, must be consistent with
joint references.

The planned Risk + Residual version will insert bounded arm offsets before step
4, while reusing the baseline executor. It is documented separately in the
[README architecture](../README.md#architecture); no learned controller is
required to run the B0 scripts.

## Integration acceptance evidence

- Text-conditioned G1 output with verified joint and quaternion conversion.
- Receiver-decoded fields and matched SONIC checkpoint/config hashes.
- Free-base standing and tracking of known and ARDY-generated references.
- Calibrated fingers, frictional lift and a contact-verified hold.
- Clock and underrun logs, reproducible resets, and complete trial records.

The 25 FPS ARDY sampling rate is a motion representation rate, not a measured
generation throughput. Record text encoding, generation, transport and module
latencies separately from simulation time and real-time factor.


## 4. Optional Kimodo generator boundary

Kimodo is a separate frozen generator option, not a second executor. The
`--kimodo` selection is resolved in the shared execution runtime; ARDY remains
the default and is not imported through the Kimodo path. `baseline/kimodo.py`
loads the independently pinned source/checkpoint, reuses the local LLM2Vec text
encoder contract, translates the shared MuJoCo constraint schema using the
Kimodo converter's coordinate transform, and emits the same named-joint,
wxyz-quaternion `ReferenceSequence` consumed by SONIC.

Kimodo's native 30 Hz/36-column qpos is never assumed to have SONIC's order.
The converter XML is checked for the exact 29 joint names, the reference is
resampled to 50 Hz, and velocities are recomputed after resampling. The shared
`SonicSimulation.install`, limit checks, timestamped buffer, underrun behavior,
finger controller, dynamic block, free root and evaluator are then used without
branch-specific physics. Kimodo does not receive future executed states;
measured history is retained only for matched logging and current-state
constraint anchoring.

New K0 grasp plans record a method-specific wrist-target calibration
`[0.125, 0.035, 0.080]` metres. The shared grounding algorithm is unchanged;
ARDY's measured-site default and saved resume settings are preserved. This
calibration was selected using exploratory physical trials, not validation or
held-out evidence for the proposal's learned methods.

Raw nominal output is the default. An explicit `--kimodo-project-constraints`
option invokes static MuJoCo FK plus bounded seven-hinge right-arm fitting
before shared conversion/checks. It preserves the other qpos fields, records
raw/projected artifacts and solver/error metadata, and never steps physics,
reads future executions, attaches the object or uses a teacher. Version 2
fits the hand-center point with a soft orientation preference. Projection
has only failed physical pilots and is excluded from the successful raw K0
configuration.

The canonical worker has the complete pinned source and checkpoint. Offline
verification, MUSA model loading and text-to-motion execution passed. A full
raw K0 tuning batch obtained 11/20 contact-grasp successes; final-source batch
and ARDY regressions are recorded in `docs/verification.md`. Synthetic tests
and execution-only checks remain distinct from physical task evidence.
