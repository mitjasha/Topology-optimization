# Topology Lab

**Open-source structural topology optimization, finite-element analysis and scientific machine learning**

[![Numerical verification](https://github.com/mitjasha/Topology-optimization/actions/workflows/verify.yml/badge.svg?branch=master)](https://github.com/mitjasha/Topology-optimization/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Topology Lab develops reproducible numerical methods for material distribution in elastic structures. The project brings together finite-element analysis (FEA), density-based topology optimization, trainable surrogate models and physics-checked numerical experiments. It supports classical optimization and investigates when machine learning can reduce the *complete cost* of obtaining an acceptable structural solution.

The software and its experimental protocols evolve independently. Numerical assumptions, failure conditions, reproducible results and known limitations are part of the engineering method, not afterthoughts.

**Current development focus:** extending the established 2D workflow to volumetric 3D FEM; developing comparable Ridge, U-Net, graph and Fourier-operator baselines; and measuring end-to-end efficiency under physical acceptance gates.

## Capabilities

| Area | 2D | 3D |
|---|---|---|
| Linear elasticity | Q4, plane stress, 2 DOF/node | Hex8, spatial elasticity, 3 DOF/node |
| Structural optimization | SIMP, density filtering, optimality criteria (OC) | SIMP, density filtering, shared OC |
| Finite-element backend | Sparse SciPy; dense reference for small systems | Sparse SciPy |
| Individual engineering case | GUI and CLI, configurable mesh/load/volume | GUI and CLI, configurable box dimensions/mesh/load/volume |
| Dataset generation | FEM labels, train/validation/held-out test | FEM labels, volumetric fields, held-out aspect ratios/load positions |
| Learning models | Ridge, residual U-Net, grid GNN, FNO, learned update | Ridge, U-Net3D, voxel GNN, FNO3D |
| Other experimental learning | Diffusion / DiT, neural FEM initial guesses, differentiable compliance | Extension under investigation |
| Visualization | Density plots and convergence history | Volume view, Z slices, ParaView VTK |
| Research workflow | Bilingual local GUI, training/evaluation, process logs, saved outputs | Same GUI with 2D/3D selector |

The 3D implementation currently targets a **rectangular, left-clamped cantilever**, with a force acting on the right face. It is an actual Hex8 3D elasticity formulation, not an extrusion of 2D plane-stress results. Support for arbitrary CAD geometries, unstructured meshes, stress and buckling constraints, and industrial-scale solvers is ongoing work.

## Install and run

Use Python 3.12. Create a virtual environment if possible.

```bash
git clone https://github.com/mitjasha/Topology-optimization.git
cd Topology-optimization
python -m pip install -r requirements-gui.txt
python -m streamlit run topopt/mlnext/gui.py
```

The local application opens at **http://localhost:8501**. Select **English / Русский** and **2D / 3D** in the sidebar.

For Windows there is `run_topology_lab.cmd`; for Linux/macOS use `bash run_topology_lab.sh` after installing dependencies. The interface is intended for trusted local execution and does not provide public-server authentication.

The main development line is `master`; new algorithms and research studies are developed on feature branches and integrated after numerical verification.

Documentation: [GUI — English](docs/GUI_EN.md) · [GUI — Русский](docs/GUI_RU.md) · [2D/3D guide — English](docs/2D_3D_EN.md) · [2D/3D — Русский](docs/2D_3D_RU.md) · [Scientific ML study protocol](docs/scientific-ml-roadmap.md).

### Numerical case without the GUI

```bash
# Full 3D Hex8 FEM + SIMP optimization; writes densities, compliance history and VTK.
python -m topopt.mlnext.case_run --dimension 3 --nx 8 --ny 4 --nz 4 --volume .5 --output artifacts/mlnext/example3d

# Retain the 2D plane-stress workflow.
python -m topopt.mlnext.case_run --dimension 2 --nx 16 --ny 8 --volume .5 --output artifacts/mlnext/example2d
```

These modest meshes are suitable for initial numerical and workflow checks, not for mesh-independent structural design.

## Numerical evidence and baseline results

The established 2D learning study comprises **84 cases (48 train, 12 validation, 24 held-out test)** and compares uniform-start OC with supervised warm starts. The 24 test problems were evaluated with three network seeds and three timing repeats; 216 evaluations per method do **not** represent 216 distinct cases.

| 2D method | Mean FEM solutions per test evaluation | Final quality gates |
|---|---:|---:|
| Uniform-start OC | 48.25 | 216 / 216 |
| Ridge + FEM/OC refinement | **27.00** | 216 / 216 |
| Small U-Net + FEM/OC refinement | 36.28 | 216 / 216 |

Within **this specific test distribution**, the Ridge warm start required approximately 44% fewer online FEM solves than uniform OC, while the small U-Net used approximately 25% fewer. These figures do not establish superiority for general structures or in 3D; offline data generation and model training are separate costs.

Sources: [per-run measurements](evidence_v2/families/results.csv) · [summary](evidence_v2/families/summary.json) · [splits and provenance](evidence_v2/families/manifest.json).

Other verified and unresolved aspects:

- Classical numerical tests cover equilibrium, energy, sensitivity gradients, dense/sparse consistency and failed acceptance conditions.
- CalculiX comparisons for the earlier 2D foundation show good agreement for the thinnest studied cases and documented discrepancies at thicker settings: [independent solver study](evidence_v2/binary_correlation/summary.json).
- Mesh-independent performance **has not yet been established**: the earlier refinement study still finds non-negligible changes in compliance at its final mesh level: [refinement evidence](evidence_v2/refinement/summary.json).
- 3D neural performance is **not yet established**. Volumetric architecture implementations, dataset generation and verification tests need successful independent integration runs and mesh studies.

The acceptance pipeline requires FEM equilibrium, volume compliance, optimizer convergence and engineering-quality checks. Any learned initialization may be rejected and replaced by a conventional solve, with that cost reported. Numerical solver correlation is not experimental validation.

## Scientific ML studies

The research code includes conditioned residual U-Net models, mesh-neighbor graph learning, Fourier neural operators, physics-aware losses, trainable density updates, neural initial guesses for iterative FEM and conditional generative models.

The relevant research question is not whether a network makes a visually plausible shape: it is **under which conditions an ML-assisted process is faster, sufficiently accurate, reproducible and robust relative to classical optimization and numerical solution methods.**

Comparisons should be controlled for mesh, boundary conditions, held-out test family, optimizer stopping criteria, number of FEM solves, inference cost, training cost and failure/recovery rate. Results are retained even when an ML approach does not improve the baseline.

Start with:

```bash
python -m pip install -r requirements-mlnext.txt
python -m pytest mltests/test_mlnext.py mltests/test_3d.py -v
python scripts/mlnext_experiment.py --quick --dry-run
```

The `--quick` profile is only a small end-to-end engineering check; it is not a performance study. Large independent 2D/3D experiments and multi-seed analyses are planned and documented in [the research protocol](docs/scientific-ml-roadmap.md).

## Architecture and repository

| Path | Purpose |
|---|---|
| [`topopt/solver.py`](topopt/solver.py) | Established Q4 FEM, filtering, SIMP and OC optimization |
| [`topopt/solver3d.py`](topopt/solver3d.py) | Volumetric Hex8 finite-element implementation |
| [`topopt/mlnext/`](topopt/mlnext) | Scientific ML architectures, dataset tools, training, validation and GUI |
| [`tests/`](tests) and [`mltests/`](mltests) | Foundation and optional neural/numerical regression checks |
| [`evidence_v2/`](evidence_v2) | Preserved experimental evidence and provenance |
| [`docs/`](docs) | Model assumptions, validation methods and future research questions |
| [`results/`](results), [`mesh_study/`](mesh_study), [`correlation/`](correlation) | Earlier numerical studies, retained as historical baselines |

Computed experiment outputs belong under `artifacts/` and are not treated as proof of a model improvement without independent checks.

## Development directions

- **Numerical verification:** energy, force/reaction equilibrium, sensitivity gradients, mesh convergence and independent FEM correlation.
- **Structural models:** extended supports, multiple and distributed loads, arbitrary design domains, external mesh exchange, stress and vibration objectives, and manufacturing constraints.
- **Scalability:** iterative solvers, preconditioning, sparse assembly optimizations and larger volumetric discretizations.
- **Scientific ML:** data efficiency, out-of-distribution transfer, mesh-conditioned graph/operator learning, learned FEM acceleration and 3D generative optimization.
- **Reproducibility and usability:** experiment manifests, comparable multi-seed benchmarks, diagnostics, bilingual GUI and explicit model limitations.

Contributions are welcome when supported by clear numerical tests, documented assumptions and reproducible results. Proposed functionality and known limitations are tracked in the documentation and future development work.

## Origins, methods and attribution

The original [2019 notebook](20190505_Shakhrai_DS_Topology_Optimization.ipynb) is preserved with its historical context and limitations. It explored image-based neural architectures. The current Q4/Hex8 solvers and generated datasets are independent later implementations, not restored versions of unavailable historical dependencies.

Density-based topology optimization, SIMP and OC are established numerical methods. A methodological reference is Andreassen et al., *Efficient topology optimization in MATLAB using 88 lines of code* (2011), [DTU project page](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab).

Engineering code and documentation were developed with AI assistance and require the same numerical scrutiny, code review and independent replication as any other research contribution.

Licensed under MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution and third-party material.
