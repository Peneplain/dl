# ARDY, SONIC and MuJoCo integration

The `baseline/` package contains model-independent reference, provenance and
frozen SONIC execution utilities. `scripts/run_live.py` connects text input,
ARDY reference generation, the timestamped reference buffer and a free-base
MuJoCo G1 loop. It is a standing/arm-motion bring-up, not a grasp demo.
The [B0 bring-up guide](baseline.md) now supplies pinned downloads, an explicit
MUSA/CPU ARDY generation entry and a CPU SONIC ONNX graph check. These are
separate acceptance gates; tabletop task sequencing and grasp execution remain
outside the current bring-up executor.

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
you do not need to create the upstream `.venv_sim` for `scripts/run_live.py`.
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
  --source-fps 25 --out artifacts/ardy-reference.npz --packet
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

The current standing/arm-motion executor is `scripts/run_live.py`. It:

1. Timestamps measured poses and ARDY requests.
2. Converts ARDY output to the canonical reference and accepts further text
   requests while the simulation runs.
3. Buffers the lookahead required by the selected SONIC observation configuration.
4. At each 50 Hz reference tick, checks limits from the active G1 asset using
   elapsed simulation time, then recomputes velocity and dependent kinematics.
5. Applies the shared joint checks, passes the reference to frozen SONIC and
   issues the separate finger commands.
6. Holds or stops on underrun or invalid references, logging the event. Preserves
   clocks and frame indices, and resets controller and buffer state per episode.

`ReferenceBuffer`, the ONNX observation adapter, MuJoCo executor and rollout
logger are available in `baseline/`. The tabletop scene, task sequencer,
clearance/collision policy and contact hand state machine are still pending.
The logger saves full MuJoCo state at 50 Hz plus a compiled scene under
`rollout/`. The separate `scripts/render_expert_rollout.py` restores frames
without advancing physics, producing camera RGB in `vision/images.npz` and
optional MP4. The default camera is fixed third-person; named cameras from the
recorded scene can provide future head/wrist views. These images are recording
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
