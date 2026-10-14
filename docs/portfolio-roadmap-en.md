# Develop the topology optimization project into a CAE portfolio

## Position the project

The central question is whether ML can reduce the cost of obtaining a verified topology optimization solution. The project should demonstrate problem definition, numerical credibility, reproducibility and an engineering decision about the method's applicability.

## Completed foundation

- An independent linear Q4 FEM, SIMP, density filter and OC implementation.
- Numerical checks for equilibrium, energy, sensitivities and other properties.
- A small new dataset split by cases and mirror-equivalent groups.
- A ridge baseline against which a future U-Net can demonstrate added value.
- Direct prediction, FEM verification, refinement and cost comparison.
- English README and engineering specification, with reports and raw results.

This first verifiable implementation supports a small research demonstration. It does not establish a universal speedup or replace industrial CAE experience.

## Extend numerical verification

The first independent comparison with CalculiX 2.23 covers a solid cantilever, a graded density field and an optimized field. Because CPS4 is internally expanded into C3D8, the thickness study preserves membrane stiffness E·t. Thin-model compliance differs by approximately 0.17–0.27%. Failed large-thickness comparisons remain in the evidence. This is numerical verification rather than experimental validation; see `docs/calculix-correlation.md` and `correlation/`.

Four mesh levels and 16 binary candidates have been analyzed. One binary candidate meets the demonstrator gates, while mesh independence remains unproven. The next stage is sparse FEM, density projection within optimization and independent verification of binary shapes. See `docs/mesh-binary-study.md`.

## Restore or replace the neural experiment

If the original `libtopology` and `X.npz` files survive, use them for a documented historical reproduction. Otherwise, retain the old notebook as an archive and train a new model using the new generator under a distinct experiment name.

Start with a small U-Net and compare it with ridge, nearest-neighbor prediction and ordinary optimization stopped early. Split before augmentation. Transform geometry, density fields, loads and constraints consistently. Select the method using validation data and evaluate the final test set after fixing the method.

Measure compliance, volume, rejected-solution rate, iterations, FEM solves and total runtime. Report density-field error separately. Include failure cases and the conditions for reverting to conventional optimization.

## Demonstrate engineering judgment

Add two or three problem families, a risk matrix, broader independent solver comparisons and a short engineering decision memo. Explain where ML is justified, where it is not yet justified, and how many problems are needed to amortize data generation and training. Generalization requires separate tests on new geometries, loads and volume fractions.

Distinguish the author's problem definition and engineering decisions from established algorithms and AI assistance. Be prepared to explain FEM assembly, filtered sensitivities, data leakage, physical constraints and why direct ML prediction is rejected in some cases.

## Readiness for presentation

Another engineer should be able to run the project using the README, obtain comparable numerical results, understand the applicability limits and trace each conclusion to its data. Claims of industrial applicability additionally require broader independent verification and benchmarks.

## Publish the foundation

The work is in `portfolio/verification-first-cae` and PR #1, with the historical notebook preserved. Before accepting the PR, complete the author's technical review, choose a license and verify the workflow in a clean environment. Present this release as a research demonstrator within its tested scope.
