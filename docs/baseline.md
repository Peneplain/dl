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

Source and model revisions are pinned in `configs/baseline.lock.json`. Downloads
go to `third_party/` and `checkpoints/baseline/`, both excluded from Git. The
Hugging Face client uses an existing login or `HF_TOKEN` when authentication is
needed.

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
python scripts/run_ardy.py \
  --device musa --text-device cpu --text-dtype float32 \
  --prompt "A person stands still." --duration 2 --seed 0 \
  --out artifacts/ardy-stand-01
```

LLM2Vec has 8B parameters. CPU float32 weights alone require roughly 32 GB of
memory, with additional space needed during loading. If host memory is limited,
try `--text-device musa --text-dtype bfloat16` on a card with sufficient memory.
That path still needs an operator check on the target machine. The script
releases the text encoder before loading ARDY.

The entry point uses the official Python API with frozen parameters, eager
inference and the full diffusion schedule. It loads verified local snapshots
and explicitly selects Horizon8. In the pinned upstream registry, the shorthand
`g1` selects Horizon52, and the upstream generation CLI does not select MUSA.

Each run requires a fresh output directory and writes:

| File | Contents |
| --- | --- |
| `motion.csv` | Root xyz, root quaternion in wxyz order, and 29 body joints at 25 FPS |
| `joint_names.json` | Source joint order read from the converter's XML asset |
| `reference.npz` | SONIC joint order, resampled to 50 Hz with recomputed velocities |
| `reference.packet` | Offline SONIC v1 packet; no network transmission |
| `text_embedding.npz` | The prompt embedding |
| `report.json` | Devices, versions, hashes, timings and any failure |

After standing motion works, try a standing arm-raise prompt. Use `--constraints`
to supply a constraint file in the upstream format. Generated references still
need joint-limit, collision and standing-feasibility checks before execution.

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
