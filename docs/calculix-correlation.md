# Independent CalculiX correlation

## Scope and decision

The custom NumPy Q4 FEM was compared with **CalculiX 2.23**, using its CPS4 element and SPOOLES linear solver. Three density fields were tested on a 16×8 mesh: solid, graded, and OC-optimized. Geometry L/H=2, normalized load position y/H=0.37, left-edge in-plane clamping and a downward unit resultant were matched.

For thickness 0.001, relative compliance differences are approximately **0.169%, 0.176% and 0.274%**, respectively. Global displacement relative L2 errors are **0.056%, 0.080% and 0.294%**. These runs and the three runs at thickness 0.01 pass the predefined demonstrator gates.

This provides independent numerical support for the implemented linear membrane analysis on the tested cases. It is not experimental validation, certification, or verification of every generated topology.

## Crucial formulation distinction

CalculiX CPS4 is internally expanded into C3D8. It is not algebraically identical to the direct 2D plane-stress Q4 formulation. A unit-thickness CalculiX model of a height-one domain is not a thin-plate limit; its compliance discrepancy reaches 8.1% for the optimized field.

The thickness study therefore preserves **E_CCX·t_CCX = E_reference·1** for each element, with unchanged Poisson ratio, mesh, nodal forces and in-plane constraints. Thicknesses 1, 0.01 and 0.001 are retained. This preserves membrane stiffness, not a physically identical 3D material/geometry pair. The residual differences at small thickness remain reported.

Changing thickness from 0.01 to 0.001 changes CalculiX compliance by less than 0.002% in these cases. This is a thickness-sensitivity observation, not an in-plane mesh-convergence study. The normalized unit-load linear response can be rescaled; a physical application must use load levels consistent with small strains.

## Gates declared before the comparison

| Quantity | Criterion |
|---|---:|
| Compliance | absolute relative error ≤1% |
| Entire in-plane displacement field | relative L2 error ≤1% |
| Constrained in-plane nodal reactions | relative L2 error ≤5% |
| Force-resultant balance | absolute error ≤10⁻⁵ with unit load |

These tolerances were saved in `acceptance.json` before solver execution. Unit-thickness failures are retained; the thresholds were not relaxed to make them pass. Printed CalculiX DAT values have finite precision, which limits comparison near roundoff.

Compliance is computed independently from CalculiX displacements as Fᵀu. Only constrained in-plane RF values are interpreted as support reactions: the RF output includes external forces at loaded nodes as well. A global L2 error is not a bound on each nodal relative error.

## Reproduce

Download CalculiX from the [author's website](https://www.dhondt.de/). The checked run used the standalone `ccx_static.exe` from [CalculiX 2.23 for Windows](https://www.dhondt.de/calculix_2.23_4win.zip). No system installation or registry change is needed. The solver executable is external and not included in the portfolio archive.

```bash
python -m topopt.correlation --ccx /absolute/path/to/ccx --output correlation
python -m unittest discover -s tests -v
```

The command exports decks, executes the external solver with one thread, checks completion and errors, parses complete displacement/reaction tables, saves paired arrays and emits CSV/HTML reports. The checked output has nine runs: three fields × three thicknesses. Executable, source and input hashes are recorded in the manifest. CI runs interface/unit tests; it does not install or run CalculiX automatically.

## Evidence

- `correlation/results.csv`: all runs, including failed thickness diagnostics.
- `correlation/report.html`: readable comparison.
- `correlation/manifest.json`: parameters, density design variables, executable and code hashes.
- Each `.inp`, `.log`, `.dat` and `_fields.npz`: complete reproducibility trail.

Method documentation: [CalculiX CPS4 expansion](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node41.html), [solid-section thickness](https://www.feacluster.com/CalculiX/ccx_2.18/doc/ccx/node326.html), [nodal output](https://www.feacluster.com/CalculiX/ccx_2.18/doc/ccx/node307.html).

## Remaining work

In-plane mesh study with constant physical filter radius; binary reanalysis; additional load/geometry families; contact/nonlinear extensions if relevant; and tests against physical measurements if experimental validation is claimed. The present correlation does not remove these limitations.
