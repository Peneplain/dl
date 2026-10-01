# Running the frozen baseline

B0 uses `ARDY-G1-RP-25FPS-Horizon8` to generate motion references for a frozen
SONIC controller and a free-base G1 in MuJoCo. The scripts below cover model
setup, inference checks and the free-base standing/arm-motion control loop. The
tabletop task and contact grasping remain under development; see
[integration.md](integration.md) for the interfaces.

## Setup

On the S4000 host, update the repository and enter the existing MUSA container:

```bash
git pull --ff-only origin main
./docker/run-musa.sh bash
```

The launcher uses `dl-musa:latest` by default. Set `MUSA_IMAGE` if the image has a
different name. Inside the container, create one environment that inherits the
vendor PyTorch installation:

```bash
# The vendor image has the venv module but omits ensurepip.
python -m virtualenv --system-site-packages .venv-baseline-musa
source .venv-baseline-musa/bin/activate
bash scripts/install_baseline.sh
```

If `virtualenv` is missing, install it in the container with
`python -m pip install virtualenv` before creating the environment. The
installer adds ARDY and SONIC simulation dependencies to this same
`.venv-baseline-musa`; it never replaces the vendor `torch`/`torch_musa` pair.
The environment lives in the mounted project directory and survives container
restarts. When it exists, `docker/run-musa.sh` sets `VIRTUAL_ENV` and puts its
`bin` directory first on the container PATH, including for `run-musa.sh bash`.
For an already-open container or a shell that resets PATH, run
`source .venv-baseline-musa/bin/activate` before invoking Python.

Direct SSH X11 forwarding reaches XQuartz, but the Mac setup here does not
provide the GLX framebuffer configurations required by MuJoCo/GLFW. Build the
runtime and its optional GUI target on the S4000 host from the repository root.
The second build selects the GUI target and reuses the shared runtime layers:

```bash
cd ~/dl
docker build -f Dockerfile.musa -t dl-musa:latest .
docker build --target gui -f Dockerfile.musa -t dl-musa-gui:latest .
```

To start the VNC display, preload ARDY and SONIC, and open the interactive
prompt session, run this single command from a Mac terminal. It uses the S4000
address used by the current setup; change it if the host address changes:

```bash
ssh -tt -o ExitOnForwardFailure=yes \
  -L 5901:127.0.0.1:5900 \
  group3@10.123.0.39 \
  'cd ~/dl && ./docker/run-mujoco-gui.sh'
```

Enter the server's SSH password. Keep this terminal open: it carries both the
SSH tunnel and the prompt session. After the server reports that the viewer is
open, open Screen Sharing from a second Mac terminal:

```bash
open vnc://localhost:5901
```

Leave the username blank and enter the fixed VNC password `group3`. The robot
window opens after weights load and stays visible at its standing pose while
waiting for prompts. Enter prompts in the first Mac terminal, one per line:

```json
{"prompt":"raise both arms slowly, then return to a balanced standing posture","duration":4,"seed":1}
{"prompt":"walk slowly forward with alternating steps","duration":6,"seed":2}
quit
```

Each prompt gets its own `run-NNNN` directory and MP4 under the timestamped
`artifacts/interactive-gui-*` directory. The viewer remains open between
prompts; simulated time is paused while waiting. If Mac port `5901` is occupied,
change the local side and URL to `5902` while leaving the remote side at `5900`.
The VNC server binds only to server localhost, and the SSH tunnel encrypts the
connection. Press `Ctrl-C` in the first terminal to stop the session and
container.

The regular `dl-musa:latest` image remains headless. Omit `--gui` for headless
physics and MP4 recording.

The installer preserves the installed `torch` and `torch_musa` versions and
records dependencies in `artifacts/baseline-install-*`. Its backend check tests
inference primitives without importing any learning modules; ARDY and SONIC
are checked separately below.

It installs both `requirements-musa.txt` (including MuJoCo) and
`requirements-baseline.txt` plus `requirements-sonic-sim.txt`, so ARDY, SONIC
and this repository use one environment.

Source and model revisions are pinned in `configs/baseline.lock.json`. Downloads
go to `third_party/` and `checkpoints/baseline/`, both excluded from Git. The
Hugging Face client uses an existing login or `HF_TOKEN` when authentication is
needed.

Use the following local layout; none of these directories belongs in Git or
the baseline source archive:

```text
checkpoints/baseline/
  ardy/ARDY-G1-RP-25FPS-Horizon8/
  sonic/                   Matching encoder, decoder and observation config
  llama_base/              Full Meta Llama backbone
  text_base/               MNTP adapter, model config and tokenizer
  text_adapter/            Supervised adapter
  <asset-key>.manifest.json
third_party/
  ardy/                    Clean checkout at the locked commit
  sonic/                   Clean checkout at the locked commit
artifacts/
  imports/                 Offline archives, Git bundles and checksum sidecars
  <fresh-run>/             Check reports and generated references
```

An offline SONIC **weights** archive does not provide the SONIC source checkout.
Use `python scripts/fetch_baseline.py --only sources` to fetch both pinned
sources, or import a source Git bundle at the locked commit. Keep archives and
their checksum sidecars together in `artifacts/imports/`; check them there with
`sha256sum -c <archive-name>.sha256`. Preserve backups until they are no longer
needed. Do not move a model directory while a downloader is writing to it.

## Check SONIC on CPU

The pinned upstream C++ deployment requires TensorRT and CUDA. For the first
S4000 check, run the frozen ONNX models with ONNX Runtime's CPU provider:

```bash
python scripts/fetch_baseline.py --only sonic
python scripts/check_sonic_onnx.py --out artifacts/sonic-cpu-01
```

This downloads the default encoder, decoder and matching observation config
(about 91 MB of model weights), verifies their hashes, and measures inference
latency. The encoder accepts 1,762 inputs and returns 64 tokens; the decoder
accepts 994 inputs and returns 29 actions.

The two graphs run independently with zero-valued inputs. A passing report
establishes operator support, but does not validate the observation adapter or
robot control. Results and errors are saved to `report.json`; failures also
return a nonzero exit code.

## Generate an ARDY reference

```bash
python scripts/fetch_baseline.py --only sources
python scripts/fetch_baseline.py --only ardy
python scripts/fetch_baseline.py --only text
python scripts/check_baseline.py --device musa
python scripts/run_ardy.py \
  --device musa --text-device cpu --text-dtype float32 \
  --prompt "A person stands still." --duration 2 --seed 0 \
  --out artifacts/ardy-stand-01
```

The text encoder needs three separate downloads. The directory names are kept
compatible with earlier installations:

| Directory under `checkpoints/baseline/` | Contents |
| --- | --- |
| `llama_base/` | `meta-llama/Meta-Llama-3-8B-Instruct`, the full 8B backbone |
| `text_base/` | McGill's `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` adapter and tokenizer |
| `text_adapter/` | McGill's `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` adapter |

The McGill repositories contain adapters, not the Llama backbone. `--only text`
downloads all three; `--only llama` fetches just the backbone. Access to the
Meta repository requires accepting its license and using an authorized Hugging
Face account. Inference loads Llama locally, merges MNTP, then applies the
supervised adapter. It does not resolve the adapter's remote base-model path.

If Llama is already downloading with `hf download`, let it finish. Put its
download directory at `checkpoints/baseline/llama_base`, or symlink that path to
the existing directory. Keep the `.cache/huggingface` metadata (or the original
Hugging Face snapshot and blob links). Register it without network access:

```bash
python scripts/fetch_baseline.py --only llama --offline
python scripts/check_baseline.py --device musa
```

Offline registration checks the pinned revision, every shard, and the download
ETags before writing a manifest. It refuses an incomplete download or missing
provenance. To resume a download in place against the pinned revision, run
`python scripts/fetch_baseline.py --only llama` with network access. Completed
manifests are verified and reused; a failed network retry does not delete them.

LLM2Vec has 8B parameters. CPU float32 weights alone require roughly 32 GB of
memory, with additional space needed during loading. If host memory is limited,
try `--text-device musa --text-dtype bfloat16` on a card with sufficient memory.
That path still needs an operator check on the target machine. The script
releases the text encoder before loading ARDY.

The entry point uses the official Python API with frozen parameters, eager
inference and the full diffusion schedule. It loads verified local snapshots
and explicitly selects Horizon8. In the pinned upstream registry, the shorthand
`g1` selects Horizon52, and the upstream generation CLI does not select MUSA.

Use the pinned `transformers==5.8.1` dependency. `baseline/llama.py` supplies
the bidirectional padding mask explicitly because this Transformers version
no longer calls the upstream model's `_update_causal_mask` override. The local
tests check both future-token attention and padding isolation. Tokenization,
prompt formatting and pooling still use the pinned ARDY implementation.

Each run requires a fresh output directory and writes:

| File | Contents |
| --- | --- |
| `motion.csv` | Root xyz, root quaternion in wxyz order, and 29 body joints at 25 FPS |
| `joint_names.json` | Source joint order read from the converter's XML asset |
| `reference.npz` | SONIC joint order, resampled to 50 Hz with recomputed velocities |
| `reference.packet` | Offline SONIC v1 packet; no network transmission |
| `text_embedding.npz` | The prompt embedding |
| `report.json` | Devices, versions, hashes, timings and any failure |

Failures include the stage (`preflight`, `text_load`, `text_encode`,
`motion_load`, `motion_generate` or `reference_export`). A passing preflight
checks installation and files; run both actual model checks afterward.

After standing motion works, try a standing arm-raise prompt. Use `--constraints`
to supply a constraint file in the upstream format. Generated references still
need joint-limit, collision and standing-feasibility checks before execution.

### Keep the models loaded

For several prompts, start the persistent service once. It loads LLM2Vec and
ARDY a single time and then reads one prompt per JSONL line from standard input:

```bash
python scripts/ardy_service.py \
  --device musa --text-device musa --text-dtype bfloat16 \
  --out-root artifacts/ardy-service
```

Submit requests to the already running process, one JSON object per line:

```json
{"name":"stand","prompt":"A person stands still.","duration":2,"seed":0}
{"name":"left-hand","prompt":"A person moves up the left hand.","duration":2,"seed":1}
```

Each request gets its own directory under `--out-root` and writes the same
motion and reference files as the single-run entry point. Send `quit` to stop
the service. The service keeps both models loaded between requests.

### Prepare a reference for the SONIC deploy reader

The offline `reference.packet` is a serialization fixture; it does not publish
DDS commands to a running simulator. The upstream C++ deploy reader instead
expects one motion directory containing CSV arrays. Convert an ARDY result with:

```bash
python scripts/prepare_deploy_motion.py \
  --reference artifacts/ardy-service/kick/reference.npz \
  --motion-csv artifacts/ardy-service/kick/motion.csv \
  --out-dir artifacts/sonic-deploy-motion \
  --name kick
```

This writes the 29-joint position/velocity arrays, root position/orientation,
root velocities and metadata at 50 Hz. It prepares deploy input only; it does
not start MuJoCo or claim a tracking result. The C++ deploy executable and its
matching observation/checkpoint configuration must be built and verified before
the directory can drive `run_sim_loop.py` through DDS.

## Run the simulation

The headless B0 executor accepts commands while the 50 Hz SONIC loop runs:

```bash
python scripts/run_live.py --out artifacts/live-b0-01 \
  --prompt "Stand still and raise both arms" --duration 2 --sim-seconds 30 \
  --video artifacts/live-b0-01/baseline.mp4
```

The first request loads the frozen ARDY text and motion models in a background
worker. In default real-time mode, MuJoCo continues to run the nominal SONIC
hold while it is loading or generating. After generation, the 50 Hz timestamped
buffer blends from the measured pose and executes the reference. Additional
lines can be entered as plain text, or as JSON such as
`{"prompt":"lower both arms","duration":2,"seed":1}`;
`quit` ends the run. Add `--fast` for a non-real-time headless check. In fast
mode, the simulation clock pauses while a queued ARDY request is being
generated, so model latency does not stretch the requested simulated
choreography; plain real-time mode keeps the nominal SONIC hold running during
that generation. `--keep-alive` also uses the paused-clock behavior so waiting
for later interactive prompts is not recorded as motion time. This path does
not yet build the tabletop/block task or claim a grasp result.

`--video` writes a display-free MP4 from MuJoCo's computed G1 visual meshes and
geom poses. It is a deterministic software mesh reconstruction, not an OpenGL
camera render; the current vendor image has no EGL/OSMesa/GLX runtime. The
video metadata is recorded in `report.json`. Omit this option when only control
metrics are needed.

Use `--keep-alive` (or `--interactive`) to start a persistent session. With
`--gui`, the model and passive viewer are created after the frozen weights load,
before input is accepted. The same viewer and MuJoCo model are reused across
prompts; the state resets to standing between prompt runs while each gets
independent logs and an MP4. Do not use `--fast` with `--gui` when you want to
watch motion at real-time speed. ARDY, LLM2Vec, and SONIC are loaded once for
the session. Each JSONL line creates an independent `run-0001`,
`run-0002`, ... directory under `--out`; its prompt can change `duration` and
`seed`, and its reference, video, events, trajectory, and report stay inside
that directory. The existing session path is never reused. Enter `quit` to
finish:

```bash
./docker/run-mujoco-gui.sh
```

Then enter, one line at a time:

```json
{"prompt":"side-step right and return to standing","duration":4,"seed":1}
{"prompt":"raise both arms, then lower them","duration":3,"seed":2}
quit
```

The resulting layout is:

```text
artifacts/interactive-b0-01/
  run-0001/{baseline.mp4,events.jsonl,trajectory.csv,report.json,ardy/}
  run-0002/{baseline.mp4,events.jsonl,trajectory.csv,report.json,ardy/}
```

If ARDY produces a reference that violates the shared joint limits, the
executor rejects that reference and keeps the robot in its default standing
hold. The run still records the simulation and video, but `report.json` is
marked `"status": "failed"` with `ardy_errors` instead of being reported as a
successful baseline run.

The current executor establishes free-base standing and reference tracking. The
remaining work for the proposal's physical task is:

1. Extend the verified SONIC observation adapter checks against upstream's C++
   gather outputs. Keep G1 mode, history, future reference samples, relative
   orientations, joint order and action scaling pinned; SONIC's lookahead must
   remain independent of the risk model's eight-frame horizon.
2. Broaden the free-base evidence with longer known-reference and ARDY standing
   and arm-motion trials. Record state, torque, tracking error, falls, simulation
   time, wall time and video.
3. Add a reachable table, a dynamic block and articulated fingers. Use simulator
   object state to set wrist and standing constraints for approach, close,
   lift and hold. Calibrate finger control and contact parameters once and
   share them across methods. Keep the object free to move under contact forces.
4. Record every trial, including failures. A successful grasp lifts the target
   block's lowest point at least 5 cm above the table and holds it for 2 s,
   within 30 s of simulated time, without a fall or prohibited collision.

Keep model and scene hashes, prompts, seeds, trajectories and failure reasons
with each run. Begin rollout collection for learning once this execution path
is reproducible.

## Logs and local checks

Keep the installation log and both inference reports when diagnosing a failed
run. Use a new output directory for each attempt.

```bash
python -m unittest discover -s tests -v
python scripts/smoke.py --out artifacts/smoke-b0-01
```

The smoke run checks reference conversion and packet fields on synthetic data.
It does not train a model or provide physical grasping evidence. Recorded local checks are listed in
[verification.md](verification.md).

For hosts where Git access is unavailable, `scripts/package_baseline.py` can
create a source archive from the baseline file allowlist, including uncommitted
baseline files. Learning modules, weights, datasets, environments and run outputs
are excluded:

```bash
python3 scripts/package_baseline.py --out artifacts/b0-source.tar.gz
```
