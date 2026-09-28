# Project scope

Use `docs/proposal.tex` and `README.md` as the current project specification.
The instructor-approved scope is **one frozen ARDY baseline plus the same
baseline with Risk + Residual and ablations, entirely in G1 simulation**.
The older planning document in the separate proposal workspace described
GR00T and real hardware; those are superseded for this repository.

- Frozen ARDY-G1-RP-25FPS-Horizon8 -> reference adapter -> optional learned
  correction -> shared checks -> frozen SONIC -> free-base G1 in MuJoCo.
- Learn arm-joint reference offsets only; never learn root, leg or finger offsets.
- Share task grounding, hand control, limits, physics and evaluation across methods.
- Keep synthetic pipeline fixtures explicitly separate from physical task results.
- Split parent episodes, including counterfactual/teacher branches, before windowing.
- Train risk first. Freeze it and train residuals using predicted risk features.
- B1/B2/I1 reuse one residual checkpoint. No-gate reuses P without retraining.
- Keep real model weights, rollout data, generated artifacts and credentials out of Git.

# Validation

Run `python -m unittest discover -s tests -v` for model/control/data changes.
Run `python scripts/smoke.py --out artifacts/<fresh-run>` after training or
checkpoint-interface changes. The CPU smoke run is not evidence of grasp success.
Check actual S4000 operators with `scripts/check_backend.py --device musa` in
the cluster's supported environment; do not assume CUDA/TensorRT compatibility.
