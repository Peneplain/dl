# Commands

Run `./run.sh` from the repository on the S4000 **host**, outside the container.
It reuses the vendor MUSA stack, `.venv-baseline-musa` and the rendering image.
Use fresh output directories for every new experiment.

## Select ARDY or Kimodo

`--ardy` and `--kimodo` are mutually exclusive generator selectors. ARDY remains
the compatibility default; use an explicit selector in new commands and plans.
B0/P name controller conditions, while the flags name motion generators.

```bash
cd ~/dl
./run.sh batch --ardy --grasp --batch 20 --seed 42
./run.sh batch --kimodo --grasp --batch 20 --seed 42
./run.sh batch --ardy --prompt 'A person stands upright and slowly raises the right hand.'
./run.sh batch --kimodo --prompt 'A person slowly raises the right hand.' --duration 2
```

Without `--grasp`, execution has no tabletop block task. Without `--batch`, a
batch contains one attempt. `--seed 42 --batch 20` fixes seeds 42–61; block XY
is sampled separately by each seed. The default XY range is X=[.36,.46],
Y=[-.30,-.16] m. Models stay loaded across attempts; physics and controller
state reset for each attempt. Physical failures remain in the denominator and
do not stop the rest of a standard batch. A nonzero command exit can indicate
completed failed trials; inspect `summary.json` before assuming interruption.

## GPU selection

Select physical devices before entering the container. With one visible
physical device, its in-container index is `musa:0`:

```bash
MTHREADS_VISIBLE_DEVICES=2 ./run.sh batch --ardy --grasp --batch 20 --seed 42 \
  --device musa:0 --text-device musa:0
```

Use a separate output directory and an idle GPU for each concurrent job.
Multi-GPU Risk training runs independent model seeds, not one distributed model.
Two SSH clients may use the same account; avoid concurrent writers to the same
session, teacher queue, dataset, checkpoint or Git worktree.

## Grasp settings and paired trials

```bash
./run.sh batch --ardy --grasp --batch 20 --seed 42 \
  --table-standoff .20 --hold-seconds 10 --output-root output/b0-new
./run.sh batch --kimodo --grasp --batch 20 --seed 42 \
  --table-standoff .20 --hold-seconds 10 --output-root output/kimodo-new
./run.sh batch --ardy --grasp --cube-xy .40 -.22 --seed 42
./run.sh batch --ardy --grasp --walk --start-back .45 --table-standoff .20 --seed 42
./run.sh batch --ardy --grasp --phase-prompts configs/grasp-prompts.json --seed 42
```

The default starts at the grounded table target without walking.
`--direct-start` is a compatibility alias for that default. `--walk` adds
approach, settling and preparation within the same 30 s simulation budget.
Manipulation-only trials must be reported separately from full walking trials.

`--cube-xy X Y` fixes block position. `--xy-range XMIN XMAX YMIN YMAX` changes
sampling bounds. `--wrist-offset X Y Z` is a generator-goal calibration; it does
not relax the geometry-derived acquisition or success checks. Kimodo's tuned
wrist default differs from ARDY's measured-site default and is recorded in the
plan. Keep all settings fixed across each intended matched comparison.

Default fingers remain open during reach/lower, then close after measured
alignment. `--hand-approach preshaped` is a historical calibration control.
`--finger-kp`, `--finger-kd`, `--contact-profile`, `--lift-seconds`,
`--hold-seconds` and `--alignment-replans` expose shared settings; changing one
creates a new experimental condition. Longer hold settings do not extend the
30 s timeout. [Grasp](grasp.md) defines the assessment and prompt semantics.

## Load the trained P controller

Supply all three matching artifacts with `--ardy`:

```bash
MTHREADS_VISIBLE_DEVICES=2 ./run.sh batch --ardy --grasp --batch 20 --seed 62000 \
  --device musa:0 --text-device musa:0 --risk-device musa:0 --risk-update-hz 10 \
  --table-standoff .20 --hold-seconds 10 \
  --risk-checkpoint output/risk-quality-retrain-261007f/risk/seed-0/best.pt \
  --residual-checkpoint output/risk-quality-retrain-261007f/residual/best.pt \
  --risk-gate output/risk-quality-retrain-261007f/eval-val/gate.json \
  --output-root output/p-new
```

`eval-val/gate.json` is the runtime intervention gate. The separate
`final-label-gate.json` certifies dataset/label audit and is not a runtime gate.
The plan and controller metadata pin model, dataset, validation-window and
learning-source hashes. Mixing artifacts, synthetic checkpoints or Kimodo
with this ARDY-trained P set is rejected. Checkpoint/gate details and the
absence of aggregate B0/P improvement are in [verification](verification.md).

For a matched B0 run use the same seed range, task settings and physical
conditions, omit the three learned artifacts and the `--risk-device` /
`--risk-update-hz` options, and select another idle GPU and fresh output root. Predeclare all batches; keep every success, timeout and
failure. Do not keep only the batch with the largest P improvement.

## Plan and resume

```bash
./run.sh batch --ardy --grasp --batch 20 --seed 42 --plan-only
./run.sh batch --kimodo --grasp --batch 20 --seed 42 --plan-only
./run.sh batch --resume output/batch-YYMMDD-HHMMSS
```

A resume command supplies **only `--resume PATH`**. The generator and every
setting come from the saved plan; a selector or override on resume is rejected.
Completed attempts are skipped. Interrupted attempts retain their original
seeds and write sibling retry directories; completed failures are not retried
until success. Changes to pinned source, assets, configuration or evidence
prevent resume. Create a fresh plan after an implementation change instead of
editing old plans or reports.

## Manual operation and GUI

```bash
./run.sh manual --ardy --grasp
./run.sh manual --kimodo --grasp --gui
```

Wait for model loading and the `READY >` prompt before submitting one JSON
request. For example:

```json
{"seed":42,"cube_xy":[0.40,-0.22],"phase_prompts":{"reach":"A person slowly raises the open right hand above the table."}}
```

General motion without a grasp task accepts `prompt`, `duration` and `seed`.
Other supported request keys are `cube_xy`, `phase_prompts` and `reference`;
check `./run.sh manual --help` for the configured mode. Input is disabled while
busy. Waiting at READY does not advance physics. Use `quit` or `exit` at READY
to stop. Manual attempts are recorded but not rendered automatically.

A GUI image must exist. Forward the configured VNC port over SSH and use the
service's configured authentication; keep passwords out of project documents:

```bash
ssh -tt -o ExitOnForwardFailure=yes -L 5901:127.0.0.1:5900 group3 \
  'cd ~/dl && ./run.sh manual --ardy --grasp --gui'
```

## Render selected attempts

```bash
./run.sh render --run output/batch-YYMMDD-HHMMSS --attempts 1 3 5
./run.sh render --run output/batch-YYMMDD-HHMMSS --successes
./run.sh render --run output/batch-YYMMDD-HHMMSS --all
./run.sh render --run output/batch-YYMMDD-HHMMSS --attempts 1 \
  --camera third_person --camera head_camera --camera wrist_camera \
  --width 640 --height 480 --fps 25
```

Substitute the directory printed by execution. Full retry names and individual
attempt paths are accepted. The default third-person view is 640×480 at 25 FPS;
`--fps 50` selects every control frame and `--no-video` retains RGB only.
The first selected camera supplies the video. A 30 s raw RGB sequence at these
defaults is about 0.69 GB per camera, so render selected examples when space is
limited. Rendering restores recorded states; it cannot add missing hold time
or alter a failed physical result.

## Setup and diagnostics

Reuse the existing environment on an installed server:

```bash
./run.sh check
./run.sh tests
./run.sh batch --help
./run.sh manual --help
./run.sh render --help
```

For an initial installation:

```bash
./run.sh build
./run.sh build-gui
./run.sh shell
# Inside the container, create this environment only if it does not exist.
python -m virtualenv --system-site-packages .venv-baseline-musa
source .venv-baseline-musa/bin/activate
bash scripts/install_baseline.sh
exit
# Back on the host.
./run.sh fetch --only all
./run.sh check
```

Install `virtualenv` first if necessary. Authorized Llama access is required;
keep account authentication outside Git. `fetch --only` accepts
`sources|sonic|ardy|text|llama`. Kimodo has a separate pinned downloader:

```bash
./run.sh shell
python scripts/fetch_kimodo.py --only all
exit
```

### Offline asset transfer

On a connected machine, use the pinned fetch tools and preserve complete
Hugging Face download metadata. Transfer the resulting directories to the
server, including manifests and source `.git` metadata:

```bash
rsync -a --progress third_party/ardy/ group3:~/dl/third_party/ardy/
rsync -a --progress third_party/sonic/ group3:~/dl/third_party/sonic/
rsync -a --progress third_party/kimodo/ group3:~/dl/third_party/kimodo/
rsync -a --progress checkpoints/baseline/ group3:~/dl/checkpoints/baseline/
rsync -a --progress checkpoints/kimodo/ group3:~/dl/checkpoints/kimodo/
```

Then register and verify in the server container:

```bash
./run.sh fetch --offline
./run.sh shell
python scripts/fetch_kimodo.py --offline
exit
./run.sh check
```

Use `df -h output` before a large collection. A symlink from `output/` to a
writable `/data` directory is mounted at its resolved path by the launcher.
Credentials, checkpoints, upstream source trees and output artifacts stay
outside Git.

### Scope of checks

```bash
./run.sh smoke --out output/smoke-new
./run.sh sonic --out output/sonic-new
./run.sh ardy --prompt 'A person stands still.' --duration 2 --seed 42 --out output/ardy-new
./run.sh package --scope b0 --out output/b0-source-new.tar.gz
./run.sh package --scope kimodo --out output/kimodo-source-new.tar.gz
```

`check` verifies assets and MUSA operators. `smoke` uses synthetic references;
neither proves grasp success. `ardy` and `sonic` are frozen-model diagnostics.
`convert`, `deploy` and `service` are module utilities, not hardware deployment
approval. Packages contain source only. Set `MUSA_IMAGE` to change the supported
container target; learning/data commands are in [learning](learning.md).
