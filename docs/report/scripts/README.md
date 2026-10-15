# Rebuilding the report figures

Run these commands from `docs/report`:

```bash
python3 -m pip install -r scripts/requirements.txt
python3 scripts/make_figures.py
```

The entry point rebuilds the following figures from published inputs. It performs
no simulator execution, model inference, training, or new inferential tests.

| Script | Input | Outputs in `figures/` |
| --- | --- | --- |
| `make_result_figures.py` | `results/public-results.json` | `generator-comparison`, `confirmation`, `gating`, `learning-curves`, `failure-profile`, `failure-profile-column` (`.pdf` and `.png`) |
| `draw_horizontal_system.py` | Authored diagram definitions in the script | `system`, `generator_pipeline` (`.pdf`, `.png`, `.svg`) |
| `draw_horizontal_architecture.py` | Authored diagram definitions in the script | `architecture`, `risk_residual_architecture` (`.pdf`, `.png`, `.svg`) |

Each script can also be run individually. The result-chart script accepts
`--results PATH` and `--output DIRECTORY`; its defaults are relative to this
report directory, so it also works when invoked from another working directory.

The result figures preserve the manuscript's definitions and values:

- Generator comparison: final success out of 100 trials per generator, published
  95% Wilson intervals, conditional retention in the 58 method-specific
  threshold-reaching trials, and recorded median first-reference response.
- Confirmation: the 124 complete end-to-end origin pairs and their recorded
  paired outcome table. The script counts the published rows and checks them
  against the published summary; it does not recompute the primary test.
- Invocation study: the 20-origin end-to-end success counts, Residual call
  fractions, and recorded mean controller latencies. Both learned modes retain
  the full structured Risk conditioning.
- Learning curves: all 19 recorded Risk epochs and 45 recorded Residual epochs;
  retained checkpoints are epochs 7 and 33. No smoothing is applied. These are
  separate supervised objectives, not grasp-success curves.
- Failure profiles: recorded terminal failure reasons out of the 100 scheduled
  trials per generator. Sticky post-threshold loss can differ from the terminal
  reason when a later prohibited contact ends the episode.

The qualitative behavior panels (`baseline-qualitative*`, `qualitative`, and
`retention-example`) are supplied as rendered report assets. This minimal public
entry point does **not** regenerate those panels or re-extract video frames.
The accompanying public videos show the recorded behavior. Do not use the old
private discovery, SSH, or video-labeling scripts as public rebuild commands.

The scripts write figure-level regeneration receipts with public input/source
hashes. Original source references and result provenance are listed in
`results/public-source-manifest.json`; no private evidence directory is needed.
Matplotlib's bundled DejaVu Sans is used. Rendering metadata, font rasterization,
and PDF bytes may differ across Matplotlib/Pillow versions while numerical
values, layout definitions, and dimensions remain fixed.
