# Text-Driven Humanoid Manipulation

HKU DASC 7606C Deep Learning · Track 4, Group 3

This repository runs text-driven tabletop block grasping with a simulated
Unitree G1, frozen motion generators and frozen SONIC control. It implements
Predictive Risk and Residual models and loads them through the same execution
workflow as the nominal baseline. All task data and demonstrations are from
simulation; images are recorded outputs, and object grounding uses simulator
state.

## Start here

- [Course report](https://github.com/Peneplain/dl/tree/main/report): ICRA-format
  PDF and standalone Overleaf source.
- [Demonstration videos](https://github.com/Peneplain/dl/tree/main/videos):
  22 annotated success, failure and paired-controller videos.
- [Numerical results](https://github.com/Peneplain/dl/tree/main/docs/results):
  benchmark, confirmation, training and invocation-study summaries.
- [Commands](docs/commands.md): setup, `--ardy` / `--kimodo`, GPU selection,
  learned control, resume and video rendering.
- [Frozen-model integration](docs/baseline.md): assets, adapters, SONIC and
  recording interfaces.
- [Grasp task](docs/grasp.md): language grounding, contact physics and success
  criteria.
- [Learning](docs/learning.md): data collection, verified teacher pairs,
  tensor contracts, training and checkpoint provenance.
- [Verification guide](docs/verification.md): source and checkpoint compatibility,
  evidence boundaries and physical-trial accounting.

The [research proposal](docs/proposal.tex) is the approved, frozen source for
research scope. Implementation status is documented separately. GR00T N1.7,
camera perception, real-robot deployment and joint fine-tuning are not
implemented in this simulation workflow.

## Motion generator and controller

Select a generator explicitly with `--ardy` or `--kimodo`. These flags are
mutually exclusive. ARDY is the compatibility default when neither is supplied.
Generator selection and learned control are separate choices:

| Execution system | Generator | Learned controller | Evidence |
| --- | --- | --- | --- |
| B0 | Frozen ARDY-G1-RP-25FPS-Horizon8 | None | Shared nominal baseline |
| K0 | Frozen Kimodo-G1-RP-v1 (`--kimodo`) | None | Shared 100-scene generator comparison |
| P | Frozen ARDY (`--ardy`) | Matched trained Risk + Residual and declared gate settings | Integrated in batch/manual execution |

P currently requires ARDY. Its checkpoints were trained on the ARDY data
interface, so Kimodo cannot be combined with these learned artifacts. Kimodo
is an additional generator comparison; it does not replace the proposal's
B0/P learned-control study or its planned controls and ablations.

## Architecture

The task sequencer combines English phase instructions with current table,
block and robot poses. The selected frozen generator produces nominal body
references. A named-joint adapter maps them into SONIC's G1 representation.
SONIC controls the 29 body joints; a separate shared controller operates the
articulated fingers.

### Nominal baseline

```mermaid
flowchart TB
    T[Language + simulator-state task grounding] --> M[Frozen ARDY or Kimodo]
    M --> A[Named-joint reference adapter + buffer]
    A --> C[Shared reference checks]
    C --> S[Frozen SONIC]
    S --> G[G1 in MuJoCo: free base + dynamic block]
    T --> H[Shared articulated-finger controller]
    H --> G
```

### Predictive Risk-guided Residual control

```mermaid
flowchart TB
    T[Language + simulator-state task grounding] --> A[Frozen ARDY + nominal reference adapter]
    A --> C[Shared reference checks]
    A --> R[Predictive Risk Transformer]
    G[G1 in MuJoCo] -. Execution history .-> R
    R --> Z[8 x 4 x 32 structured risk tokens + intervention score]
    Z --> V[Declared intervention gate]
    V -->|High risk| P[Residual Transformer]
    Z --> P
    A --> P
    G -. Execution history .-> P
    P -->|Bounded arm offsets| C
    V -->|Low risk: zero desired offset| C
    C --> S[Frozen SONIC]
    S --> G
    T --> H[Shared articulated-finger controller]
    H --> G
```

Risk uses 300 ms of execution history and a 280 ms nominal future. It predicts
tracking, contact, balance and intervention information. Residual reads the
predicted representation and changes only the 14 arm-reference coordinates.
Root, torso, legs and fingers remain under the common nominal system. In
selective mode, low risk skips Residual inference and any existing offset ramps
to zero. Always-on mode retains Risk inference and the full predicted tokens,
while invoking Residual at every learned-controller decision.

Physics runs at 200 Hz, SONIC references at 50 Hz, and Risk/Residual updates at
10–20 Hz. Correction bounds, rate limits, physical stops, the scene and success
assessment are shared between B0 and P. This executor pauses simulation during
motion generation; it does not establish real-time deployment.

## Implementation and evidence boundaries

The learning package implements staged supervised training: train Risk, freeze
it, then train Residual using predicted body/time features. The normalizer is
fitted on training inputs and stored in each checkpoint. Runtime loads a
matched Risk, Residual and gate bundle and checks its dataset, schema and
normalizer provenance.

The canonical Residual predicts bounded arm offsets. Its loss uses the stored
teacher-minus-nominal target without clipping that target, and the shared
executor enforces joint, magnitude and rate limits. Native training sidecars
may use a different architecture or target contract; their matching source and
adapter must accompany the artifacts. Copying such weights into the canonical
loader does not establish compatibility.

Training completion, synthetic checks, full-model execution and admitted
physical results are separate kinds of evidence. Keep exact dataset sizes,
losses, trial outcomes and selected-checkpoint manifests in the corresponding
original output artifacts. The `report/` directory contains manuscript assets,
`videos/` contains curated demonstrations, and `docs/results/` contains numerical
summaries and source provenance. The implementation supports controlled teacher
pairs and shared execution; it does not establish general recovery from
arbitrary natural failures or a complete comparison of all risk interfaces.
The [report](https://github.com/Peneplain/dl/tree/main/report) documents
the shared 100-scene generator comparison and an independent paired evaluation
of the selected ARDY-based learned controller, with separate cohorts and
reported uncertainty.

## Quick start on the S4000 host

Run from the repository on the host, outside the container:

```bash
./run.sh batch --ardy --grasp --batch 20 --seed 42
./run.sh batch --kimodo --grasp --batch 20 --seed 42
./run.sh render --run output/batch-YYMMDD-HHMMSS --attempts 1 3
```

Each batch uses a fresh timestamped directory. Render only selected attempts
when raw RGB storage is limited. [Commands](docs/commands.md) includes the
matched checkpoint set for P and the offline asset-transfer procedure.

## Edit the report in Overleaf

Upload the contents of `report/` as an Overleaf project, select `main.tex` as
the main document, and use the pdfLaTeX compiler. The project contains a single
TeX source with its figures, bibliography and conference class. References use
BibTeX with the standard `IEEEtran` bibliography style supplied by the TeX
distribution. Videos and numerical result files remain in their separate
repository directories.

## Repository layout

```text
baseline/           Shared frozen generators, reference adapters, physics and recording
risk_residual/      Models, losses, causal dataset, provenance and inference
experiments/        Data planning, teacher collection, training and evaluation
scripts/            Host workflows, asset tools, rendering and guarded pipelines
configs/            Source/model locks, prompts and learning/data configurations
requirements/       CPU and S4000 dependency lists for models and simulation
docker/             Vendor MUSA container launcher and rendering/GUI support
tests/              Contracts, regression checks and physical/rendering probes
docs/               Setup, learning, verification, numerical results and frozen proposal
report/             Standalone Overleaf manuscript, figures and compiled report
videos/             Curated annotated demonstrations and video manifest
checkpoints/        Local model assets and manifests (ignored)
third_party/        Pinned upstream checkouts (ignored)
output/             Datasets, checkpoints, trials and rendered videos (ignored)
```

Keep original model download metadata and generated evidence outside Git.
`output/` may resolve to a shared `/data` volume; the launcher mounts its target
inside the container. Source archives use an explicit packaging scope.

External code, models and datasets retain their upstream licenses. Source and
checkpoint references are pinned in `configs/baseline.lock.json` and
`configs/kimodo.lock.json`; acknowledge them and LLM assistance in course
submissions.
