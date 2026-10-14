# Run and verify topology optimization on Windows

## Get the current implementation

Extract the project archive into a folder such as `C:\CAE\Topology-optimization`. The root should contain `README.md`, `requirements-reproduce.txt`, `run_checks.cmd` and the `topopt` directory.

Download the default branch as a ZIP from GitHub or clone the repository:

```powershell
git clone https://github.com/mitjasha/Topology-optimization.git
cd Topology-optimization
```

## Install Python

Use **64-bit Python 3.12** to match the tested environment. If needed, the [Python 3.12.10 release page](https://www.python.org/downloads/release/python-31210/) provides a Windows installer. This is a specific installer version for reproducibility, not a claim that it is the latest security release. Include the Python Launcher (`py`) and open a new PowerShell window after installation.

```powershell
py -3.12 --version
```

## Create the environment

Open PowerShell in the project root and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproduce.txt
```

The commands explicitly use the environment's Python, so activation and PowerShell ExecutionPolicy changes are unnecessary. The core dependencies are NumPy 2.3.5 and SciPy 1.16.3. Installation needs internet access; the core calculations can then run offline. A GPU, Jupyter, paid solver licenses and the historical `X.npz` and `libtopology` files are unnecessary for this implementation.

## Run the complete workflow

```powershell
.\run_checks.cmd
```

The script runs automated tests, the ML benchmark, and the mesh and binary study. It stops on a program error and writes new results to `my_results` and `my_mesh_study`. Repeated runs overwrite those output folders. The checked-in evidence remains in `results` and `mesh_study`.

The original preparation machine completed these tasks in seconds to tens of seconds; another CPU may take longer. The solver uses dense matrices. Keep several gigabytes of memory available and avoid increasing the mesh beyond 48×24 in this dense-backend workflow. The extended sparse experiments have separate commands in `docs/development-v2.md`.

```powershell
Start-Process .\my_results\report.html
Start-Process .\my_mesh_study\report.html
```

Alternatively, double-click the HTML files. Keep each report with its accompanying SVG, CSV and JSON files.

## Run individual steps

```powershell
# Numerical and interface tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 24 cases, a supervised ML baseline and FEM refinement
.\.venv\Scripts\python.exe -m topopt.benchmark --output my_results

# 8x4, 16x8, 32x16 and 48x24 meshes, followed by binary reanalysis
.\.venv\Scripts\python.exe -m topopt.mesh_study --output my_mesh_study
```

## Optional CalculiX comparison

Download [CalculiX 2.23 for Windows from its author's website](https://www.dhondt.de/calculix_2.23_4win.zip), extract it into a folder such as `C:\CAE\CalculiX`, and locate `ccx_static.exe`. Substitute its actual path below:

```powershell
.\.venv\Scripts\python.exe -m topopt.correlation --ccx "C:\CAE\CalculiX\calculix_2.23_4win\ccx_static.exe" --output my_correlation
Start-Process .\my_correlation\report.html
```

No system installation or PATH change is required. CalculiX is not bundled. This experiment compares three density fields and thickness effects; it does not independently verify all binary shapes in the mesh study.

## Interpret the results

- The test suite should finish with `OK`; this revision contains 31 tests (one optional neural check is skipped without PyTorch).
- The ML split contains 14 training, 4 validation and 6 test cases. Mirror-equivalent pairs remain in the same partition. Numerical results should be close to the saved evidence; timings may vary.
- Six thin CalculiX models pass the gates. Three unit-thickness models are retained as failed diagnostic comparisons; these are numerical outcomes rather than execution errors.
- The mesh study separately evaluates the solid model, a frozen density field and reoptimized designs. Comparing different designs does not isolate FEM convergence.
- A binary `FAIL` means at least one predefined criterion is unmet. Inspect volume, compliance, connectivity and sensitivity to void stiffness.
- A result with `converged=False` is not a completed optimization.
- The main evidence command fails when any ML benchmark quality gate fails. Diagnostic mesh and thickness studies retain numerical failures as research outcomes; review their reports.

The historical 2019 notebook is an archive and is outside this runnable workflow. `notebooks/01_benchmark_review.ipynb` optionally displays saved evidence; Jupyter is unnecessary for the main experiments.

## Troubleshooting

**`py` is missing or Python 3.12 is unavailable:** install the launcher and interpreter, then reopen PowerShell. If Python was installed without the launcher, substitute its full executable path, for example `& "C:\...\python.exe"`.

**`No module named topopt`:** run from the repository root, which contains the `topopt` folder.

**`No module named numpy`:** install the requirements and use `.venv\Scripts\python.exe`.

**Network errors during installation:** check PyPI access and your organization's proxy settings. Keep TLS verification enabled.

**CalculiX is not found:** `--ccx` must point to the executable, rather than a ZIP or directory. Quote paths containing spaces.

**Missing report images:** confirm the calculation completed and keep the report beside its supporting files.

When reporting an error, include the command and full error text. The CSV and `manifest.json` files are sufficient for comparing numerical outcomes.
