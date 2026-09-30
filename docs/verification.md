# Baseline verification

Local checks were run on September 29, 2026, using macOS arm64, Python 3.11.2,
PyTorch 2.5.0 and NumPy 1.26.4. They cover the standalone `baseline/` package.

## Source and reference checks

- Five tests passed for asset hash/release validation, archive exclusions,
  symlink rejection, dirty upstream rejection and failure reporting. The
  archive test explicitly excludes learning code, training scripts and configs.
- `python scripts/check_backend.py --device cpu` passed linear projection,
  layer normalization and scaled dot-product attention in inference mode.
  This check imports no Risk/Residual model and runs no optimizer.
- `python scripts/smoke.py --out artifacts/baseline-only-smoke-20260929` passed.
  A synthetic CSV with reversed joint order was converted from 25 to 50 Hz;
  positions, velocities, names, packet metadata and every binary field matched
  the expected reference. No model weights or physics were used.
- Shell syntax and diff whitespace checks passed.

## Frozen SONIC graphs

The default SONIC weights were downloaded at Hugging Face revision
`6733128a3d8a523b1418b06bca3cdf61c8b0987f`. Source and model pins are recorded in
`configs/baseline.lock.json`; downloaded files have SHA256 manifests under
`checkpoints/baseline/`.

After the package split, `scripts/check_sonic_onnx.py` passed again using ONNX
Runtime 1.30.0 with CPUExecutionProvider, four threads, three warm-ups and 20
measured executions. The encoder maps `[1,1762]` to `[1,64]`; the decoder maps
`[1,994]` to `[1,29]`. All outputs were finite. The report is saved locally at
`artifacts/sonic-baseline-only-20260929/report.json`.

The graphs were checked independently with zero-valued synthetic inputs. This
validates CPU operator execution, not the connected controller, observation
adapter or tracking behavior.

## Remaining validation

Full prompt-to-motion inference with the 8B Llama backbone, MUSA execution, the SONIC observation adapter and
MuJoCo control still need validation on the target environment. No physical
standing, tracking or grasping result is reported. See [baseline.md](baseline.md)
for the model checks and [integration.md](integration.md) for the control loop.

## September 30 local review

The review used an isolated macOS arm64 environment with Python 3.12,
PyTorch 2.9.1, Transformers 5.8.1, PEFT 0.19.1 and ONNX Runtime 1.30.0.

- Fifteen tests passed, including offline download registration, complete shard
  checks, interrupted downloads, source directory validation and packet timing.
- Tiny locally generated Llama and LoRA checkpoints exercised the actual
  Transformers/PEFT loader with network connections blocked. Both adapter
  deltas matched their expected values. Prompt formatting and pooled embeddings
  matched ARDY's single-device encoding path. These are interface tests, not
  tests of the released 8B text model.
- The pinned upstream `LlamaBiModel` with Transformers 5.8.1 produced no change
  in an earlier token when a later token changed: its legacy mask override was
  not called. The baseline compatibility class passed future-token dependence
  and padding-isolation tests using an explicit bidirectional mask.
- The real Horizon8 motion weights ran on CPU with synthetic zero text features.
  Eight motion frames were generated through the full ten-step schedule and
  exported to CSV, a 50 Hz reference and a SONIC packet. The report explicitly
  marks the synthetic text fixture at
  `artifacts/ardy-motion-interface-review-20260930/report.json`.
  Additional checks passed for the minimum duration (0.08 s, two frames) and
  the requested five-second duration (125 frames), including non-multiples of
  the model's four-frame token size.
- The released McGill tokenizer/config and both LoRA configurations loaded
  locally with network connections blocked. Their HF download metadata and
  file ETags also passed the offline registration checks.
- The reference smoke check and frozen SONIC CPU probes passed again at
  `artifacts/smoke-review-20260930/` and `artifacts/sonic-review-20260930/`.

The local machine has no full Llama backbone or MuJoCo installed in the review
environment. `check_baseline.py --device cpu` correctly reports these missing
components and exits nonzero. It verifies both upstream sources and the other
four model manifests successfully. The S4000 Dockerfile installs MuJoCo;
the complete ARDY text path must be tested there after the Llama download.


## September 30 S4000 completeness audit

This audit ran in the existing `dl-musa:latest` container with the mounted
`.venv-baseline-musa` environment: Linux x86_64, Python 3.10.12, vendor
PyTorch 2.9.1 / torch_musa 2.9.1+a18d871, MuJoCo 3.2.7, Transformers 5.8.1,
PEFT 0.19.1 and ONNX Runtime 1.23.2. No dependency was installed or replaced.
The fresh output directory is `artifacts/baseline-audit-20260930-010307/`.
`commands.json`, `cleanup-commands.json`, numbered logs and `environment.json`
record commands, exit codes and installed versions. Model reports and the
existing asset manifests record model/config hashes.

### Local installation and frozen assets

| Component | Observed status |
| --- | --- |
| Vendor MUSA stack and baseline Python dependencies | Imports passed; MUSA device available; MuJoCo installed |
| ARDY source | Clean checkout at `693f74d13b3d04a0a22ce127ee79c929dd89756b` |
| ARDY Horizon8 weights, tokenizer and statistics | Complete layout and manifest hashes passed |
| SONIC default encoder, decoder and observation config | Complete matching release and manifest hashes passed |
| MNTP adapter/tokenizer and supervised LLM2Vec adapter | Both complete layouts and manifest hashes passed |
| Full Meta Llama 3 8B Instruct backbone | Not present at the canonical path during the audit; download reported in progress by the user |
| SONIC source | `third_party/sonic/` missing; the local SONIC archive contains weights only |

`python scripts/check_baseline.py --device musa` exited 1 with exactly two
failed checks: SONIC source and Llama backbone manifest. The ARDY inference
entry point was also invoked with a two-second standing prompt, seed 0, MUSA
motion device and CPU text device. It stopped at `preflight` for the missing
`llama_base.manifest.json`; `ardy/report.json` preserves that failure. No full
8B text inference or ARDY MUSA motion inference was established by this audit.

After the Llama download finishes in `checkpoints/baseline/llama_base/`, run
these commands inside the activated container environment:

```bash
python scripts/fetch_baseline.py --only llama --offline
python scripts/fetch_baseline.py --only sources
python scripts/check_baseline.py --device musa
```

Registration requires revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, all
indexed shards and their original Hugging Face metadata. If revision or metadata
checks fail, resume with `python scripts/fetch_baseline.py --only llama` against
the locked revision. Then run the actual ARDY command in [baseline.md](baseline.md)
with a fresh output directory. Source download requires network access; no
SONIC source download was attempted in this audit.

### Checks completed

- All 15 unit tests passed, including tiny offline text-model/adapter loading,
  bidirectional attention, download integrity, packaging boundaries and timing.
- MUSA linear, layer-normalization and scaled-dot-product-attention primitives
  passed. The backend emitted its float32 SDPA dtype warning; this is not a
  full-model compatibility result.
- The real SONIC encoder and decoder passed independent CPU ONNX probes with
  synthetic zero inputs and finite outputs. Four threads, three warm-ups and
  20 repetitions were used; p50/p95 were approximately 0.652/0.670 ms for the
  encoder and 0.462/0.469 ms for the decoder. Graphs were not connected to physics.
- The reference smoke passed with 9 synthetic source frames converted to
  17 reference frames at 50 Hz. This is packet/reference evidence only.
- The installer's combined MUSA and baseline requirements passed an offline
  `pip install --dry-run --no-index` using the existing vendor constraints.
  This verifies resolution in the installed environment, not a fresh install.
- Shell syntax checks passed. The baseline packager produced a 35-file source
  archive using its unchanged allowlist; local weights and transfer packages
  remain outside its scope.

### Repository cleanup

The ARDY source bundle and SONIC weights archive, together with both SHA256
sidecars, were moved from the repository root to `artifacts/imports/`.
Checksums matched before and after moving; `relocations.json` records both
paths and hashes. Existing model directories, upstream checkouts and backups
were retained. Llama downloads were not moved or restarted.

Git and Docker ignore rules now cover local caches and these transfer packages;
Docker also excludes safetensors and environment-secret files. The installer
now includes `requirements-musa.txt`, ensuring that MuJoCo is requested when
using the original vendor image as well as the project image. Vendor package
constraints remain in effect. Repository documentation and scripts use English.

### B0 implementation still required

Downloading the remaining assets does not complete the proposal's B0 executor.
The following shared components are still missing or not connected:

| Component | Remaining work |
| --- | --- |
| SONIC observation/action adapter | Match the pinned observation config, history, lookahead, frames, joint order and action scaling; connect encoder and decoder |
| MuJoCo body-control loop | Free-root G1, actuator mapping, default pose, gains, physics stepping and reproducible reset |
| Tabletop scene and hand controller | Versioned table/dynamic block/articulated-finger assets, frictional contacts, closure/release and phase-dependent contact checks |
| Task sequencer and online ARDY history | Simulator-state grounding, wrist/standing constraints, phase transitions and executed pose history for replanning |
| Shared reference checks | Joint/rate bounds, clearance/collision checks, dependent-field updates and common failure stops |
| Live scheduling and underrun handling | Connect the existing timestamped buffer to the 50 Hz consumer, derive SONIC lookahead from its config, and implement logged hold behavior |
| Episode logging and evaluation | Paired seeds/prompts/resets, trajectories/videos, model/scene provenance, timing, failures/timeouts, 5 cm / 2 s / 30 s success rule and episode-level statistics |

Validate these in order: free-base standing, known-reference tracking, ARDY
standing/arm motion, then contact grasp-and-lift. No physical trial was run in
this audit. Risk/Residual models, training and rollout/teacher data tooling are
separate planned components; their absence does not prevent B0 model checks.

### README architecture diagrams

The proposed diagram retains B0's six modules and adds only separate Risk and
Residual blocks. Explicit edges show nominal references `A`, execution history
`S`, risk tokens `Z`, gate score `p` and the corrected reference `A + delta A`.
Masking, bounds and nominal-reference addition are grouped with Residual;
shared checks still apply the common rate limits and validation. The separate
finger-control path is preserved. The Risk output branches explicitly:
`p >= tau` enables Residual with `Z` and `p`; `p < tau` skips Residual and
selects nominal `A` with a zero-offset request for the shared checks. Any
existing offset still ramps to zero under the shared rate limit. Task/hand
context and the other shared feedback paths are explained in the text.

Both Mermaid diagrams were rendered to PNG with Mermaid Ink and visually
inspected. Edge ordering was adjusted to place Risk and Residual before SONIC
in the displayed flow. The diagrams and data flow were checked against the
proposal; `git diff --check` passed. This documentation-only change required no
model or simulation runs.

## September 30 final local bring-up

After the offline SONIC source and mesh transfer, the matching MUSA container
ran the following checks without downloading or replacing model files:

- `python scripts/check_backend.py --device musa` passed MUSA linear,
  layer-normalization and scaled-dot-product attention operators.
- `python scripts/check_baseline.py --device musa` passed all dependency,
  pinned-source and five local asset-manifest checks. The upstream SONIC mesh
  files are hydrated Git LFS files; provenance validation accepts those files
  while continuing to reject ordinary source changes.
- `python scripts/smoke.py --out artifacts/baseline-final-smoke-20260930`
  passed the synthetic 25-to-50 Hz reference conversion.
- `python scripts/check_sonic_onnx.py --out
  artifacts/baseline-final-sonic-20260930` passed both frozen CPU ONNX graphs.
  The encoder and decoder produced finite outputs with the expected shapes.
- `python scripts/run_ardy.py --device musa --text-device musa
  --text-dtype bfloat16 --duration 5 --seed 42` passed for the kick prompt in
  `artifacts/ardy-final-kick-20260930/`. It generated 125 ARDY frames and the
  50 Hz SONIC reference; text loading took 117.2 s, text encoding 18.1 s and
  motion generation 5.3 s on the selected MUSA device.
- `scripts/prepare_deploy_motion.py` converted that reference into 249 frames
  at 50 Hz under `artifacts/sonic-deploy-motion-final-20260930/kick/`, with all
  required C++ deploy CSV files and metadata.
- A headless `run_sim_loop.py --interface bond0 --no-enable-onscreen` process
  remained alive for a 20-second smoke window. It initialized MuJoCo and then
  was stopped by the test timeout. The known duplicate Unitree DDS
  initialization message was printed; no tracking or task result was claimed.

These checks establish local model loading, reference conversion and simulator
startup. They do not establish SONIC tracking, contact physics or kick/grasp
success. The C++ deploy executable and its matching TensorRT release policy are
still required to publish `rt/lowcmd` commands from the converted motion into
the MuJoCo bridge.
