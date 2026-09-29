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
