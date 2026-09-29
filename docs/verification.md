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

ARDY and LLM2Vec inference, MUSA execution, the SONIC observation adapter and
MuJoCo control still need validation on the target environment. No physical
standing, tracking or grasping result is reported. See [baseline.md](baseline.md)
for the model checks and [integration.md](integration.md) for the control loop.
