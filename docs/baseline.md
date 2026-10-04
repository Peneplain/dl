# Frozen Baseline

Use the [command guide](commands.md) as the single source for executable project
commands. It covers setup, asset downloads, checks, ARDY generation, SONIC
execution, state recording, rendering, GUI sessions and packaging through
`./run.sh` from the S4000 host. This document describes the implementation and
its boundaries; it is not a second command list.

## Scope and status

B0 combines the pinned `ARDY-G1-RP-25FPS-Horizon8` generator, frozen SONIC
ONNX policy and a free-base G1 MuJoCo scene. The verified executor supports
standing and arm-motion bring-up. The [B0 grasp collector](grasp.md) adds a
pilot table/block scene, GT grounding, a default initial position before the
table, an optional `--walk` root-path approach and settling checks, right-hand
phases, spatial constraints, fingers, collision stops and episode statistics.
Manipulation-only contact grasp/lift has pilot evidence; complete walk-and-grasp
success and calibrated evaluation settings remain unverified.
Risk/Residual models and their training pipeline are not implemented yet.

All task data and future demonstrations are simulation-only. Risk/Residual
learning remains separate from `baseline/`, which must run without learned
checkpoints or imports. The proposal remains the research source of truth.

## Runtime and assets

The pinned vendor image supplies the matching `torch` and `torch_musa` stack
for the S4000 driver. The project preserves that pair and installs MuJoCo,
ARDY/SONIC simulation dependencies and the frozen text stack into the shared
`.venv-baseline-musa` environment.

Three image targets share the same vendor base:

| Image | Purpose |
| --- | --- |
| `dl-musa:latest` | Control and state recording; no offscreen RGB backend |
| `dl-musa-render:latest` | Headless OSMesa RGB and MP4 rendering |
| `dl-musa-gui:latest` | Virtual GLX display, viewer and SSH-tunneled VNC |

The `render` target extracts OSMesa libraries into a separate directory rather
than replacing vendor graphics libraries. The GUI target similarly isolates
software GLX. The launcher uses private IPC by default, with configurable
16 GiB shared memory, so separate containers do not share OpenMP registration
state. Images do not contain the local model weights.

Source revisions and asset identifiers are pinned in
[`configs/baseline.lock.json`](../configs/baseline.lock.json). Clean upstream
checkouts live in `third_party/ardy` and `third_party/sonic`. The matching model
assets live under `checkpoints/baseline/`:

| Directory | Contents |
| --- | --- |
| `ardy/` | Horizon8 model checkpoint and statistics |
| `sonic/` | Matching encoder, decoder and observation configuration |
| `llama_base/` | Full Meta Llama 3 8B Instruct backbone |
| `text_base/` | LLM2Vec MNTP adapter and tokenizer |
| `text_adapter/` | LLM2Vec supervised adapter |

Per-asset manifests record source provenance and file hashes. Llama access
requires an authorized Hugging Face account. Generated assets, credentials,
checkpoints and source checkouts are excluded from Git.

The frozen text model has 8 billion parameters. CPU float32 backbone weights
need about 32 GB before loading overhead. The supported default uses MUSA and
bfloat16; device selection is explicit and there is no silent CPU fallback.
SONIC inference uses ONNX Runtime's CPU provider in this implementation.

## Model interfaces

ARDY consumes a text embedding, history and optional upstream-format kinematic
constraints. The local text path loads the backbone and both LLM2Vec adapters
from verified local assets, preserving pinned ARDY tokenization, prompt
formatting and pooling. ARDY generates at 25 FPS with Horizon8. Its sampling
rate is not a wall-clock throughput measurement.

The adapter maps explicitly named joints into the canonical 29-joint G1 order,
preserves root quaternion wxyz, resamples to 50 Hz and recomputes velocities.
The SONIC observation adapter follows the pinned default G1 encoder mode and
observation configuration. SONIC's 0.9 s configured lookahead is independent
of any future Risk horizon.

The executor checks references against active joint limits, uses the same
SONIC policy and articulated-finger controller, and keeps the robot root free.
It logs reference installation, buffer coverage, holds, stops, wall/simulation
clocks, tracking and model provenance. Buffer underruns hold the terminal
nominal pose and are recorded. Reference rejection and safety stops are
failures, not successful task outcomes.

At the 50 Hz SONIC clock, each run also writes `nominal_context.csv`. It
contains the causal executed body state and finger targets, the current task
phase, and the complete nominal position, velocity, quaternion, and time-offset
lookahead supplied to SONIC. `sim_time` and `frame_index` align this file with
`trajectory.csv`; grasp object, wrist, and contact labels remain in the 200 Hz
`task.csv`. This is a frozen B0 recording interface for later Risk/Residual
window extraction: it never contains future executed states or teacher outputs
and adds no learning-package dependency.

Reference converters are offline tools. SONIC packet output is a serialization
fixture; it does not publish a socket or DDS command. Deploy CSV conversion does
not start the upstream C++ deployment reader.

## State recording and video

Each run saves the initial state and every completed 50 Hz control state. The
compiled scene, joint names/addresses, MuJoCo version and state hashes are
stored with the rollout. Full `mjSTATE_INTEGRATION` includes free-root,
articulated fingers, dynamic objects present in the scene, controls, applied
forces, mocap state and solver warm-start data. Physics still steps at 200 Hz;
internal substeps are not separate recorded frames.

The offline renderer loads the compiled scene, verifies hashes and MuJoCo
version, restores each frame, calls `mj_forward` to update derived geometry,
then uses `mujoco.Renderer`. It never calls `mj_step` and loads no policy. The
default camera is fixed in world-space third-person. Named cameras work when
present in the scene at recording time. The upstream G1 scene defines a head
camera; the tabletop collector configures head and wrist cameras.

`vision/images.npz` contains uint8 RGB shaped `[N,C,H,W,3]`, camera names,
simulation timestamps, state indices and original control-frame indices.
The CLI defaults to 25 FPS RGB sampling from 50 Hz states; `--fps 50` includes
every saved state. Optional H.264 MP4 uses the first camera and a
uniform simulation-time grid, mapping each presentation time to the nearest
saved state. Video metadata records that mapping. Rendering and encoding wall
time are separate from simulation time and model latency.

RGB is atomically published before video encoding. An encoder failure keeps
the complete RGB archive and rollout while marking `video_status` failed.
The MP4 is encoded to a temporary file and published only after ffmpeg exits
successfully. A failed renderer writes its own report and can be retried with
an automatically timestamped output directory. Large RGB runs need disk space: one
30-second, 50 Hz, 640x480 camera is about 1.38 GB uncompressed; each additional
camera adds a similar amount.

CSV-only historical runs cannot be faithfully reconstructed: they lack full
finger, object, controller and solver state. Visual snapshots are not sufficient
counterfactual branch checkpoints either. Those experiments must additionally
restore SONIC history, reference buffers, controller state and disturbance RNG.
Rendered images are recording outputs, not inference inputs; camera perception
remains outside the core study.

## Evidence

Synthetic fixtures validate state restoration, named cameras, timestamps,
RGB/MP4 generation, failure reports and overwrite protection. Actual checks and
their limits are recorded in [verification.md](verification.md). The optional
tabletop task has been physically exercised, including collision stops and
timeouts.
Historical direct-start pilots have achieved grasp/lift. Those measurements
predate the allowed hand-table contact policy and complete hold recording.
Two fresh manipulation pilots verified the revised collection behavior;
their evidence is recorded separately in the verification document.
Synthetic rendering fixtures do not establish physical G1 grasp success.
