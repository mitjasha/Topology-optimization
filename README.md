# Verification-first topology optimization

**Can a learned warm start reduce optimization effort while preserving structural performance?**

This project connects a historical U-Net/LinkNet experiment with a new, auditable CAE benchmark: **early SIMP iterations → supervised prediction → FEM verification → OC refinement**.

The deliverable is an engineering decision supported by numerical evidence. Image similarity alone is not an acceptance criterion.

> **Status: research demonstrator, foundation release.** The runnable implementation below is new; it does not reproduce the original 2019 dataset or neural-network scores. Its scope is deliberately small and its limitations are explicit. It is not qualified for safety-critical or manufacturing decisions.

## Start here

- **[Windows launch instructions in Russian](RUN_WINDOWS_RU.md)** — setup, commands and result interpretation.
- [Mesh/binary study](docs/mesh-binary-study.md) and [visual report](mesh_study/report.html).
- [Visual benchmark report](results/report.html) — download/open locally; GitHub displays HTML source.
- [Measured results](results/test_results.csv), [experiment manifest](results/manifest.json), [verification evidence](results/verification.json).
- [Engineering specification](docs/engineering-spec.md).
- [Independent CalculiX correlation](docs/calculix-correlation.md) and [cross-solver report](correlation/report.html).
- [Review of the historical notebook](docs/legacy-review.md).
- [Portfolio roadmap in Russian](docs/portfolio-roadmap-ru.md).

## Current evidence

The checked-in run uses 24 load positions on one 16×8 cantilever mesh. Mirror-equivalent cases are kept in the same partition: **14 training cases / 4 validation cases / 6 test cases**, corresponding to **7 / 2 / 3 independent symmetry groups**. This is a tiny within-family benchmark, including held-out edge positions; it does not establish generalization to new geometries or boundary conditions.

| Observation | Evidence from the foundation run |
|---|---|
| Direct prediction can be unacceptable | Worst direct compliance ≈1.244× the OC reference |
| Refinement recovers performance in this test | All 6 held-out cases pass the defined continuous-density gate |
| FEM effort is reduced in this test | 35–41 reference solves vs 23–26 hybrid solves |
| Refined compliance | Approximately 0.991–0.992× reference; a different local solution, not proof of global optimality |
| Binary reanalysis | 1/16 candidates passes the demonstrator gate; manufacturing readiness is not established |
| Mesh independence | Not established: frozen-field compliance changes 5.42% on the final refinement |
| Numerical checks | 19 automated checks, including binary and patch-load interfaces |
| Independent solver comparison | CalculiX 2.23: three fields, compliance difference 0.17–0.27% in the thin-membrane limit; thickness diagnostic failures retained |

Wall-time measurements are available in the CSV. They are single-run, sub-second CPU timings on a dense reference backend, not a robust industrial speedup claim. Offline data-generation and training costs are reported separately. No claim is made that learning pays off for this small problem.

## Reproduce

Tested with Python 3.12 and NumPy 2.3.5. No solver license, training download or GPU is required.

```bash
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m topopt.benchmark --output results
python -m topopt.mesh_study --output my_mesh_study
```

To reproduce the checked-in numerical environment, install `requirements-reproduce.txt` instead. The broader requirements express intended compatibility, not a tested version matrix. GitHub Actions is configured for Windows/Linux verification; consult the PR checks for the hosted run status.

The benchmark writes its own dataset, fitted ridge parameters, train/validation/test IDs, convergence history, source hashes, all test results and an HTML report. Run from the repository root. The historical notebook has separate, unavailable dependencies and is **not** part of this runnable path.

An independent CalculiX comparison is available via `python -m topopt.correlation --ccx /path/to/ccx --output correlation`. The solver is an external optional dependency. See the correlation method document for the essential thickness/formulation distinction. The package does not bundle the CalculiX executable.

## Engineering model

- Linear isotropic plane stress; small displacement; nondimensional E=1 and unit thickness.
- Bilinear Q4 elements with 2×2 Gauss integration.
- Left-edge clamping; unit downward force distributed over adjacent right-edge nodes to represent a continuous load position.
- Minimum compliance under a physical-density volume fraction of 0.5.
- SIMP exponent 3, minimum modulus 10⁻⁶, normalized linear density filter.
- OC updates with move limit 0.2 and design-variable lower bound 0.001.
- Physical filter radius is fixed in geometry coordinates.
- Dense NumPy solver: transparent and portable, intentionally limited to small meshes.

The ML benchmark fields are filtered continuous densities. The separate mesh study reanalyzes binary fields without filtering. Neither is CAD geometry. No stress, buckling, fatigue, minimum-member-size guarantee, overhang constraint or mesh-independent optimized-design qualification is claimed.

## Experiment design

1. Generate independent FEM optimization trajectories and retain convergence status.
2. Group mirror-equivalent cases before splitting; perform no augmentation.
3. Learn a mapping from design variables after five OC updates to final design variables.
4. Fit normalization only on training cases; select ridge regularization by validation compliance.
5. Recompute early iterations for test cases and predict without reading their reference targets.
6. Project predictions to the physical volume constraint; evaluate by FEM.
7. Refine with the same OC algorithm and stopping rule as the reference.
8. Report every test result, including failures; count early solves and final checks.

The ridge model is an interpretable **supervised ML baseline**, not a neural network. It is included so that a future U-Net must demonstrate incremental engineering value over a simpler method.

## Why this is a CAE leadership case study

The project makes model assumptions, acceptance criteria, data lineage and release limitations reviewable. It includes a negative result (unsafe-to-accept direct prediction), traceable numerical checks, and a planned escalation path for failed predictions. A future production workflow would fall back to a full solve when a gate fails; that fallback is not implemented or timed here.

This repository alone does not demonstrate leadership of people, industrial delivery, or nonlinear structural-analysis expertise. Those claims require separate professional evidence.

## Historical work and attribution

The original [2019 notebook](20190505_Shakhrai_DS_Topology_Optimization.ipynb) is preserved unchanged. It explores U-Net, LinkNet and transfer learning on 32×32 density fields. The original `libtopology`, `X.npz` and trained checkpoints are not present in the public repository as inspected on 2026-10-13. The new solver and dataset must not be represented as recovered historical artifacts.

Methodological reference: Andreassen et al., *Efficient topology optimization in MATLAB using 88 lines of code* (2011), [DTU project page](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab). The current FEM code is independently implemented using isoparametric integration; the referenced MATLAB source is not bundled or copied. SIMP, density filtering and OC are established methods, not original inventions of this project.

The foundation code/documentation was prepared with AI assistance and reviewed using the included numerical checks. The owner should review and understand the implementation before presenting it as portfolio work. No license has been assigned by this update; the repository owner should choose one before inviting reuse.
