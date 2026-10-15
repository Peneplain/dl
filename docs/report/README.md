# Text-driven humanoid grasping report

Authored as **DASC7606C Group 3**, *Text-Driven Humanoid Grasping: Motion Generators and Structured Predictive Residual Control* compares ARDY and Kimodo through a shared SONIC controller and presents the **Risk-guided residual controller** as an execution-side extension to ARDY. Its central learned interface transmits structured future-time/body Risk tokens to Residual; both selective and always-on invocation retain this conditioning. All experiments were conducted on the course-provided Moore Threads S4000 cluster. The final document uses the supplied ICRA 2027 `ieeeconf` format: **8 pages of main text, 1 page of references, and 3 appendix pages**.

## Report and supplementary material

- [Complete report PDF](https://github.com/Peneplain/dl/blob/main/docs/report/report.pdf)
- [Main text and references only](https://github.com/Peneplain/dl/blob/main/docs/report/report-main.pdf) and [separate appendix](https://github.com/Peneplain/dl/blob/main/docs/report/appendix.pdf)
- [Portable LaTeX source archive](https://github.com/Peneplain/dl/blob/main/docs/report/latex-source.zip)
- [Video demonstrations](https://github.com/Peneplain/dl/tree/main/docs/report/videos): instructions, stages, simulation time, successful grasps, and failures
- [Editable LaTeX manuscript](https://github.com/Peneplain/dl/blob/main/docs/report/main.tex), [generator results](https://github.com/Peneplain/dl/blob/main/docs/report/generator-results.tex), [contribution allocation](https://github.com/Peneplain/dl/blob/main/docs/report/contributions.tex), and [appendix](https://github.com/Peneplain/dl/blob/main/docs/report/appendix.tex)
- [Results summary](https://github.com/Peneplain/dl/blob/main/docs/report/RESULTS.md)
- [Figure assets](https://github.com/Peneplain/dl/tree/main/docs/report/figures/) and [numerical results](https://github.com/Peneplain/dl/tree/main/docs/report/results)

The gallery contains examples from both motion generators and paired ARDY baseline/controller trials. The images and videos are illustrations of recorded trials; the quantitative results use the complete corresponding cohorts.

## Architecture and scholarly context

Figure 2 follows the [implementation architecture](https://github.com/Peneplain/dl#predictive-risk-guided-residual-control): measured history and future nominal motion enter both learned models; the full structured body/time Risk tensor conditions Residual; invocation control forms a separate branch; shared limits and checks act before SONIC execution. The diagram is supplied as PDF, PNG, and editable SVG.

The report includes 17 cited research and official technical sources, an Archon whole-body intelligence discussion, and limitations covering task/sensing scope, asynchronous deployment, validation diversity, and the remaining risk-interface comparisons. Experimental results and original video bytes are preserved.

## Principal results

| Evaluation | Result |
|---|---|
| Shared 100-scene generator comparison | ARDY **40/100**; Kimodo **55/100** |
| Retention after the initial threshold | Each method reaches the threshold in **58/100** trials; ARDY later loses **18/58**, Kimodo **3/58** |
| Independent ARDY-based controller confirmation | Baseline **54/124**; Ours **74/124**; **+16.13 percentage points** |
| Paired confirmation test | Exact two-sided McNemar **p = 0.0003249142**, below the prespecified **0.00625** threshold |
| Separate same-weight gate ablation, 20 origins | Baseline **3/20**; selective **10/20**; always-on **14/20** |
| Selective controller efficiency | **62.25%** of Residual calls skipped on selective decisions; **20.13%** lower mean learned-controller time than always-on |

The generator benchmark and controller confirmation are different studies. The controller's 59.68% result is compared with its paired ARDY baseline; it is not a direct comparison against Kimodo's 55% in the generator benchmark. See [RESULTS.md](https://github.com/Peneplain/dl/blob/main/docs/report/RESULTS.md) for accounting, uncertainty, training details, and the interpretation of the ablation.

## Build the report

Keep the manuscript, `ieeeconf.cls`, bibliography, included `.tex` files, and `figures/` together. From this directory, build with a current Tectonic installation:

```sh
mkdir -p build
tectonic --outdir build main.tex
cp build/main.pdf report.pdf
```

A conventional TeX installation can use:

```sh
mkdir -p build
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
cp build/main.pdf report.pdf
```

The document uses US Letter, 10 pt, two columns, and the supplied conference class. The document class is included with the manuscript source.

## Regenerate figures

From this report directory:

```sh
python3 -m pip install -r scripts/requirements.txt
python3 scripts/make_figures.py
```

This public entry point reads the [published numerical data](https://github.com/Peneplain/dl/blob/main/docs/report/results/public-results.json) to rebuild six result charts and the two horizontal diagrams. It runs no training, simulation, model inference, or new inferential tests. The [script guide](https://github.com/Peneplain/dl/blob/main/docs/report/scripts/README.md) lists the inputs and outputs. The diagram sources are also supplied as editable SVG files.

The qualitative screenshot panels and 22 demonstration videos are included as finished assets. They are not regenerated by this entry point. The [source SHA-256 manifest](https://github.com/Peneplain/dl/blob/main/docs/report/results/public-source-manifest.json) identifies the original numerical evidence.

## Public results and provenance

The [public numerical summaries](https://github.com/Peneplain/dl/tree/main/docs/report/results) contain the quantities used in the report, recorded optimization histories, and source SHA-256 identifiers. The [video supplement](https://github.com/Peneplain/dl/tree/main/docs/report/videos) contains the 22 selected clips and a manifest with relative filenames and hashes. Full experiment archives remain preserved separately; the public package contains the report and its supporting assets.
