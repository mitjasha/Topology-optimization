# Topology Lab — local GUI

## Purpose

Topology Lab is a local, bilingual (English / Русский) graphical workbench for
topology-optimization research. It uses the same FEM/OC and scientific ML
command-line implementations as the project; it does not replace, mock, or
simplify the solvers. It starts real Python processes and retains outputs.

![No remote account required](https://img.shields.io/badge/Execution-local%20Python-informational)

## Installation

From the repository root on Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gui.txt
python -m streamlit run topopt/mlnext/gui.py
```

The local web UI opens in your browser; if it doesn't, open
`http://localhost:8501`.

Use the sidebar **Language / Язык** selector to switch between English and
Russian. Streamlit itself is served locally. No GPU is required for small experiments. Training and diffusion can optionally
use CUDA if PyTorch detects a supported GPU; the paired OC/Ridge/ML FEM benchmark
runs on CPU to keep timing comparisons controlled. Research-scale datasets can
require substantial computation and storage.

## Guided workflow

1. **Datasets** — select size 8×4, 16×8, 32×16 or larger; configure the
   number of FEM cases, seed and maximum OC iterations. Every case is
   numerically solved. Nonconvergence is recorded in the manifest.
2. **Training** — choose U-Net, MeshGNN, FNO or learned update; configure
   epochs, early stopping, batch size, seed and (U-Net only) differentiable
   compliance loss. Neural models write `best.pt` and `training.json`.
3. **Comparative study** — choose architectures and independent seeds.
   Each seed runs a complete train-then-compare sequence against OC and Ridge
   on the same held-out test cases. Or evaluate existing checkpoints.
4. **Generative design** — train convolutional DDPM or a compact diffusion
   transformer. On evaluation, generate several candidates, rank by actual
   FEM compliance and refine with OC. A diffusion sample is not a
   validated engineering design.
5. **Neural FEM** — train a nodal displacement predictor and compare
   neural-start Jacobi-PCG with ordinary CG and an independent sparse-direct
   numerical check.
6. **Results & analysis** — examine old Ridge/U-Net evidence, new CSV and
   JSON summaries, learning curves, FEM solve counts, compliance ratios,
   fallback counts, and per-case densities. Export CSV and JSON directly.
7. **Runs & logs** — read the active process log, elapsed stage history,
   generated files and final status. The browser can be refreshed or closed
   without killing the Python worker. Use Cancel to request graceful
   termination of the active subprocess.

## Files and safety

The app writes to `artifacts/mlnext/gui/<run-id>/`. Each run has
`run.json` (machine-readable status/config), `process.log`, trained
weights and/or reports. Dataset files may also live under this tree.
Historical `evidence_v2/` datasets are read-only in the GUI.

No arbitrary shell execution, arbitrary path selection, or remote service
connections are exposed by the GUI. The job backend permits only known
`python -m topopt.mlnext.*` commands and confines output to
`artifacts/mlnext`. Do not expose this developer workbench directly
to an untrusted public network: Streamlit has no project-specific
authentication in this application.

One local experiment is active at a time. Multiple architectures and seeds
can run sequentially as a single study. The job manager records failed
steps rather than inventing success. Interrupted worker processes are marked
as interrupted when detected.

### Run states

* `queued` — worker launched.
* `running` — at least one command is executing.
* `completed` — all commands exited with code 0.
* `failed` — a command returned a nonzero exit code.
* `cancel_requested` / `cancelled` — cancellation flag and termination.
* `interrupted` — the manager found an active record whose worker exited.

## Scientific evidence and limitations

The existing reported Ridge baseline is from a prior narrow cantilever
study (48 train, 12 validation, 24 held-out cases), not evidence that any
new model is superior. New study accuracy, generalization and timing must
be measured on the newly held-out support/geometry families. The GUI
deliberately separates training losses from final FEM quality gates.
A visual improvement is not proof of higher stiffness or accuracy.

The current GUI supports CPU execution and the parameters exposed by the
underlying CLIs. Training-resume from an arbitrary optimizer checkpoint,
distributed/multi-GPU scheduling, arbitrary CAD imports and cloud orchestration
are not implemented. Extending these capabilities requires extending
the numerical tools and validation protocol, not merely adding UI controls.

## Troubleshooting

* **No datasets:** create one in Datasets. If reference FEM fails to converge,
  inspect the manifest and the process log.
* **No checkpoints:** train a model first. Only `best.pt` checkpoints
  with accompanying `training.json` are listed.
* **Busy:** one experiment is active; inspect Runs & logs or cancel it.
* **Interrupted:** a worker exited unexpectedly. The run directory and
  logs remain available for inspection.
* **Missing PyTorch or Streamlit:** install the requirements with the same
  Python interpreter used to start the GUI.
