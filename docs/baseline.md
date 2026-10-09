# Frozen Models and Shared Integration

[Commands](commands.md) contains executable workflows. This page defines the
asset and reference contracts shared by ARDY, Kimodo and learned P control.

## Supported runtime

The S4000 host uses the vendor's matching `torch` / `torch_musa` stack and the
repository-local `.venv-baseline-musa` environment. The launcher enters the
container with that environment on `PATH`. Preserve the vendor pair; CUDA
wheels, TensorRT availability or a passing CPU operator test do not establish
MUSA model compatibility.

| Image | Purpose |
| --- | --- |
| `dl-musa:latest` | Control and state recording |
| `dl-musa-render:latest` | Default headless OSMesa RGB and MP4 rendering |
| `dl-musa-gui:latest` | Virtual GLX viewer and tunneled VNC |

Software graphics libraries are isolated from vendor graphics libraries.
Containers use private IPC and configurable shared memory. Images contain no
local model weights. The 8B text backbone uses MUSA bfloat16 by default; a CPU
float32 load requires about 32 GB for weights before overhead. Device selection
is explicit. SONIC uses ONNX Runtime's CPU provider in this implementation.

## Pinned assets

`configs/baseline.lock.json` fixes ARDY, SONIC, the Llama backbone and both
LLM2Vec adapters. `configs/kimodo.lock.json` separately fixes Kimodo.

| Location | Contents |
| --- | --- |
| `third_party/ardy/`, `third_party/sonic/` | Exact upstream source revisions |
| `third_party/kimodo/` | Separately pinned Kimodo source |
| `checkpoints/baseline/ardy/` | ARDY Horizon8 checkpoint and statistics |
| `checkpoints/baseline/sonic/` | Matching SONIC encoder, decoder and observation configuration |
| `checkpoints/baseline/llama_base/` | Complete Llama 3 8B Instruct backbone |
| `checkpoints/baseline/text_base/`, `text_adapter/` | Tokenizer and frozen LLM2Vec adapters |
| `checkpoints/kimodo/` | Kimodo G1 checkpoint and asset manifest |

Asset manifests verify file hashes and original download metadata. Access to
Llama requires an authorized Hugging Face account. For a host without external
network access, download pinned assets on a connected machine, transfer the
complete source and checkpoint directories, and register them offline. Do not
copy credentials or mark an incomplete cache as verified. See
[offline setup](commands.md#offline-asset-transfer).

## Generator boundaries

### ARDY (`--ardy`)

ARDY-G1-RP-25FPS-Horizon8 consumes a text embedding, measured pose history and
upstream-format kinematic constraints. The local text encoder preserves the
pinned tokenization, prompt format, bidirectional Llama behavior and pooling.
ARDY generates at 25 FPS; this is its motion sampling rate, not measured
wall-clock throughput.

The service saves actual text, measured history, constraints, generated qpos,
references and hashes under `ardy/<phase>/`. Shared feedback-driven lower
replans generate from the latest measured state. No teacher participates in
evaluation.

### Kimodo (`--kimodo`)

Kimodo-G1-RP-v1 is a distinct frozen motion generator. It uses the shared text
phases, simulator grounding, reference checks, SONIC, hand controller, physics
and evaluator. Outputs are saved under `kimodo/<phase>/`.

It exports named 36-column MuJoCo qpos at 30 Hz and limits one generated clip
to 300 frames. Named mappings convert its XML joint order to SONIC order.
Measured history supplies state anchors and transition boundaries; Kimodo has
no native ARDY history conditioning. Reports record
`history_conditioning=false`.

The exploratory grasp default uses wrist offset `[0.125, 0.035, 0.080]` m and
raw generated rotations, 100 denoising steps and constraint guidance 2. ARDY
keeps its measured hand-center default. `--wrist-offset` overrides either
explicitly. `--kimodo-project-constraints` enables an experimental bounded
nominal right-arm projection; it has not demonstrated physical grasp success
and is excluded from the measured K0 result. It is separate from learned P
residual control.

## Reference and SONIC contracts

A motion export contains root xyz, root quaternion **wxyz**, then 29 named
body-joint positions. Convert by explicit names into
`baseline.adapters.joints.ISAACLAB_JOINT_NAMES`; never assume that ARDY,
Kimodo, MuJoCo XML and SONIC use the same column order.

`ReferenceSequence` and `ReferenceBuffer` preserve timestamps and absolute
frame indices, resample to 50 Hz, and recompute velocities and dependent
kinematics after transitions or corrections. SONIC's configured ten-sample
0–0.9 s lookahead is separate from ARDY's generation horizon and Risk's
280 ms horizon. The selected SONIC G1 mode consumes joint positions,
velocities and root orientation, but not world root XY. Root paths condition
the generator; actual arrival is checked in physics.

The offline SONIC packet converter writes named joint positions/velocities,
body quaternion and increasing frame indices. Protocol v1 uses a `pose` topic
prefix, a 1280-byte JSON header and little-endian contiguous arrays. Conversion
opens no socket and issues no robot command. Deployment CSV and packet fixtures
do not establish live transport or hardware acceptance.

## Shared control

`baseline/execution.py` runs the task sequencer, reference buffer, frozen
SONIC, articulated fingers and MuJoCo assessment. The G1 root and block remain
free. Lifting requires frictional contacts; object attachments, scripted block
motion and support forces are not used.

B0 runs without learned checkpoints. P supplies a matched frozen Risk,
Residual and validation-gate set to the same executor. The correction provider
acts after nominal adaptation and before shared checks, changes only the 14
arm coordinates and preserves non-arm references. The common limiter bounds
offsets, enforces their rate and active joint ranges, and recomputes velocities.
Existing fall, collision, velocity and reference stops remain active.
Pre-execution geometric clearance and general self-clearance prediction are
not implemented; bounded corrections are not a collision-free guarantee.

## Recording and rendering

Physics steps at 200 Hz; the recorder saves the initial state and every 50 Hz
control state. `rollout/` stores the compiled scene, names, MuJoCo version,
`mjSTATE_INTEGRATION` states and hashes, including solver warm start, controls,
applied forces, articulated fingers and dynamic objects. Initial state index
is -1; later indices align with `trajectory.csv`.

`nominal_context.csv` stores causal measured history and uncorrected nominal
lookahead at 50 Hz. `task.csv` stores object/contact/clearance labels at 200 Hz.
Teacher collection also records `effective_context.csv` for the reference
actually delivered after controlled perturbation and shared limits. These
streams remain distinguishable in dataset provenance.

The renderer verifies the scene and version, restores states and calls
`mj_forward` without stepping physics or rerunning SONIC. RGB/MP4 files are
visual replay, not new independent trials. Head/wrist images are recording
outputs and are not model inputs. Simulation time pauses during generation;
loading, generation, controller latency, physics wall time and simulation time
are reported separately.
