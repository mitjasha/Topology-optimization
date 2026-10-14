# Engineering specification and acceptance contract

## Intended decision

Determine whether a learned initial density design reduces the total cost of obtaining an acceptable minimum-compliance solution on a defined family of linear-elastic problems.

This foundation release supports an experiment, not production adoption. The reference is a local OC solution. A lower hybrid compliance can indicate a better local minimum or stopping-rule effects; it does not establish global optimality.

## Governing equations

Minimize C(x)=Fᵀu subject to K(ρ)u=F and mean(ρ)≤V*, where ρ=Hx and each row of H sums to one.

Element modulus: E_e=E_min+(1−E_min)ρ_e^p. The compliance gradient is evaluated through the filter transpose, including nonuniform boundary row normalization. The volume gradient is Hᵀ1/N.

The Poisson ratio is 0.3; p=3; E_min=10⁻⁶; thickness=1. The force resultant is −1 vertically. Load interpolation preserves its location along the right edge. Left-edge translations are fixed in both directions. Units are nondimensional and consistent.

## Stopping and acceptance

The demonstration stops when the maximum design-variable change is below 0.02 for three consecutive updates, after at least 15 updates, with a cap of 180. The cap is explicitly recorded as non-convergence. This is not a KKT optimality proof.

The test gate requires reference and refined runs to converge; C_hybrid/C_ref≤1.05; absolute volume error <10⁻⁶; and free-DOF force residual norm <10⁻⁷ (unit resultant). These are demonstrator choices, not regulatory or industry acceptance criteria.

Four-neighbour connectivity at density 0.5 and the fraction with 0.1<ρ<0.9 are reported as diagnostics. Connectivity does not certify stiffness or manufacturability. A binary shape requires its own reanalysis.

## Data contract

Each case has geometry, mesh, material, load position, filter radius and solver controls. The NPZ contains early and final **design variables**, split IDs, symmetry-group IDs and label convergence status. Density filtering is applied during physical evaluation. No silent thresholding is performed.

Mirror pairs belong to one partition. There is no train/test augmentation leakage. However, only a one-parameter load family is sampled, so this protocol is not an out-of-distribution assessment. Test endpoints extend beyond the training range slightly; this still is not a new-family test.

Only converged training and validation labels are eligible. Failed test references remain reported and cause the quality gate to fail. Source hashes, environment versions, seed and complete parameters are stored in the manifest.

## Verification versus validation

Implemented numerical checks: Q4 symmetry and rigid modes; analytical constant-strain element energy; global force balance and energy/work equality; filtered compliance gradient versus central differences; volume and objective behavior; load/filter invariants; disjoint case and mirror groups; explicit iteration-limit status; and a solid-cantilever mesh-refinement comparison.

The beam comparison uses Euler–Bernoulli tip compliance 4L³/H³ with L/H=4. The continuum plane-stress solution includes effects absent from the beam approximation, so exact agreement is not expected. It is a sanity check, not proof of asymptotic convergence. The separate [mesh/binary study](mesh-binary-study.md) is completed; mesh independence is not established.

Independent correlation with CalculiX 2.23 has been performed on solid, graded and optimized fields, including a thickness sensitivity study. See [the correlation method](calculix-correlation.md): CPS4 is internally expanded to C3D8 and is not algebraically identical to the direct 2D formulation. No experimental validation or commercial-solver correlation has been performed.

## Cost accounting

Reference: common model setup + all OC FEM solves + final FEM reanalysis.

Hybrid: common model setup + five early OC updates and final check + prediction/projection/FEM check + OC refinement and final reanalysis.

Offline label generation and training/validation-sweep costs are separate. Timings are single-run and may be sensitive to cache, BLAS and CPU state. Solve counts are more stable evidence at this scale. No fallback timing, deployment overhead or amortization crossover is claimed.

## Release gates for a stronger portfolio version

1. Initial independent solver correlation completed with CalculiX; expand it across meshes and boundary/load families.
2. Four-level mesh/binary study completed; resolve remaining mesh sensitivity and independently verify selected binary shapes.
3. Several load/volume/geometry families and explicit held-out-family tests.
4. U-Net compared with ridge and non-ML early termination at matched quality.
5. Multiple seeds, paired repeated timings, failure rates and full fallback accounting.
6. Owner review, attribution/license decision and clean-environment execution before publication.
