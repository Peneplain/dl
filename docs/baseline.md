# Running the frozen baseline

B0 uses `ARDY-G1-RP-25FPS-Horizon8` to generate motion references for a frozen
SONIC controller and a free-base G1 in MuJoCo. The scripts below cover model
setup and inference checks. The simulation control loop and grasping task are
still under development; see [integration.md](integration.md) for the interfaces.

## Setup

On the S4000 host, update the repository and enter the existing MUSA container:

```bash
git pull --ff-only origin main
./docker/run-musa.sh bash
```

The launcher uses `dl-musa:latest` by default. Set `MUSA_IMAGE` if the image has a
different name. Inside the container, create an environment that inherits the
vendor PyTorch installation:

```bash
python -m venv --system-site-packages .venv-baseline-musa
source .venv-baseline-musa/bin/activate
bash scripts/install_baseline.sh
```

If the image lacks `venv`, use
`python -m virtualenv --system-site-packages .venv-baseline-musa` instead.
The environment lives in the mounted project directory and survives container
restarts. Activate it again when opening a new container.

The installer preserves the installed `torch` and `torch_musa` versions and
records dependencies in `artifacts/baseline-install-*`. Its backend check tests
inference primitives without importing any learning modules; ARDY and SONIC
are checked separately below.

It installs both `requirements-musa.txt` (including MuJoCo) and
`requirements-baseline.txt`, so setup also covers the unmodified vendor image.

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

## Connect the simulation

The remaining work is to connect inference to physical execution:

1. Implement SONIC's observation adapter from the pinned YAML and C++ gather
   functions. Match G1 mode, history, future reference samples, relative
   orientations, joint order and action scaling against upstream outputs.
   SONIC's lookahead must come from its observation config, independently of
   the risk model's eight-frame horizon.
2. Load the G1 asset in MuJoCo with a free root, the correct actuator mapping,
   default pose and PD gains. Establish standing with a known reference, then
   track ARDY standing and arm motion. Record state, torque, tracking error,
   falls, simulation time, wall time and video.
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
