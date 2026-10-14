# Extended numerical verification and learning benchmark

## Run the release checks

Install `requirements-reproduce.txt` in a fresh Python 3.12 environment, then run from the repository root:

```bash
python -m topopt.evidence --output my_results
```

This command runs the test suite, writes `verification.json`, generates the full 24-case benchmark and validates its numerical gates and partition records. A failed gate terminates the command unsuccessfully. The Windows and Linux CI jobs run this same path and retain artifacts even on failure.

You can also run the benchmark directly with `--strict` and inspect persisted evidence separately:

```bash
python -m topopt.benchmark --output my_results --backend sparse --strict
python -m topopt.gates my_results
```

The default benchmark backend is now sparse. The dense backend remains available as a small-mesh reference; existing historical result folders are frozen evidence from the original PR revision, not regenerated outputs of the current sources. New results belong in `evidence_v2/` or a separate `my_...` directory.

## Sparse finite element analysis

`Model(case, backend="sparse")` assembles stiffness into a COO matrix, combines duplicates into CSR and solves the free-DOF system with SciPy's sparse direct solver. The density filter is also sparse: a spatial tree identifies neighbours inside the physical radius, then each row is normalized. Neither the global dense stiffness matrix nor the all-pairs dense distance matrix is required. Dense/sparse displacement, reaction, compliance and gradient comparisons are covered by automated tests, along with a small optimization trajectory comparison.

Reference: [SciPy spsolve documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).

## Mesh refinement and projected binary designs

```bash
python -m topopt.refinement --output my_refinement
```

The extended study evaluates a frozen 8×4 field on seven meshes up to 128×64. The coarse field is generated anew with sparse FEM; it is not an exact reuse of the historical dense-optimization field. A fixed physical load patch is maintained. The project-specific mesh gate requires less than 1% change in each of the last two refinements. Failure is retained in `summary.json`; a completed experiment does not imply mesh independence.

The same command optimizes 32×16 and 64×32 designs with a smooth Heaviside projection at beta values 1, 2, 4, 8 and 16. Both compliance and physical-volume gradients include the projection derivative. Each beta stage maintains the physical-volume constraint. The final field is converted into a binary mask by retaining the largest densities within the volume budget.

Binary reanalysis uses physical masks directly, without refiltering. Acceptance requires a converged final projection stage, volume at most 50%, compliance at most 105% of the projected continuous parent, a support-to-load path and less than 1% change when void modulus is reduced from 1e-6 to 1e-8. Intermediate stage convergence is reported as well. These checks do not establish stress, buckling or manufacturing constraints.

## Independent binary comparison

```bash
python -m topopt.binary_correlation --ccx /path/to/ccx --inputs my_refinement --output my_binary_correlation
```

The exported decks use the exact binary mask and preserve membrane stiffness E·t. The study runs both void moduli at thicknesses 0.01, 0.001 and 0.0001, retaining all failures. `--strict` returns an unsuccessful exit status if any of these comparisons fails. This is intentionally stronger than reporting only the thin-limit subset.

The existing compliance, displacement, reaction and resultant-balance tolerances remain unchanged. Thick-model failures must not be interpreted as a thin-limit pass, or hidden by relaxing the tolerances. The source and executable hashes, decks, logs and DAT tables accompany the summary. CalculiX is an external dependency and is not bundled.

## U-Net and ridge on held-out geometry

Install the optional ML dependencies, then run:

```bash
python -m pip install -r requirements-ml.txt
python -m topopt.families --output my_families --strict
```

The complete experiment contains 84 cases. Training uses L/H=1.5 and 2.0 with three volume fractions; validation uses L/H=1.75 with two volume fractions; the 24 test cases hold out L/H=2.5 with three volume fractions. Load position varies within each group, with a fixed load patch. All cases still have rectangular domains and cantilever supports. This tests a new aspect-ratio family, not a new support type, arbitrary geometry or industrial generalization.

The partition and numerical contract are written before generation. Mirror-equivalent pairs remain together. There is no augmentation. Converged training and validation labels are eligible; test reference results are used only for reporting quality, never for fitting or checkpoint selection.

Both learned methods receive the early design field and fixed-scale case metadata. Ridge regularization is selected using validation FEM compliance. The small U-Net uses two downsampling stages and skip connections; its checkpoint is selected by validation FEM compliance every ten epochs. The default run uses 80 full-batch CPU epochs and three initialization seeds. These are initialization seeds, not independent dataset draws. The deterministic ridge fit is repeated for timing comparability.

Each test case is evaluated three times per seed with randomized paired method order. The compared methods are uniform-start OC, five-update OC with fallback, ridge plus refinement and U-Net plus refinement. **216 evaluations per method are repetitions of 24 distinct test cases**, not 216 independent problems.

## Cost accounting and runtime fallback

Online timing includes fresh model/filter setup, early FEM updates, prediction, volume projection, direct FEM checking, refinement, final reanalysis and any actual fallback solve. Interpreter startup, imports and report serialization are excluded. Dataset generation and training/validation tuning are reported separately; the test reference solves are an evaluation audit cost rather than a deployable prerequisite.

The runtime fallback decision uses only convergence, volume, residual and compliance no worse than the already computed early state. It does not use an unavailable test reference. Reference compliance within 105% is an additional evaluation criterion enforced by `--strict`; the runtime gate cannot guarantee that relative quality in an arbitrary new problem.

If the runtime gate fails, the implementation executes a fresh uniform-start OC solve and includes its cost. Five-update OC is always non-converged under the specified stopping rule, so it falls back. The report explicitly distinguishes attempt acceptance, fallback and final reference-quality acceptance.

Break-even estimates divide training-label generation plus model training/tuning cost by measured average paired time savings. They apply only to this measured problem distribution and machine; negative savings produce no break-even claim. Compare solve counts as well as short wall-time measurements.

The manually triggered extended GitHub workflow runs refinement and the ML benchmark and saves their artifacts. Independent CalculiX comparisons remain a separately executed study, not an automatic hosted check.

## License and preserved work

Original project software and documentation are licensed under MIT; see `LICENSE`. Third-party material keeps its owners' rights; see `NOTICE`. Historical notebook bytes are preserved. AI assistance in this implementation remains disclosed in the README.
