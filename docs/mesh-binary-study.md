# Mesh refinement and binary reanalysis

Four meshes (8×4, 16×8, 32×16, 48×24) use constant geometry, volume fraction 0.5 and physical filter radius 0.19. A uniform traction acts over a fixed patch of height 0.25H centered at y/H=0.5. Consistent nodal loads preserve the unit resultant and patch footprint. This is a new case, not the concentrated-load benchmark.

Three experiments are separated: solid-body FEM refinement; exact subdivision of the frozen 8×4 physical-density field without smoothing; and a fresh OC optimization on every mesh. Reoptimization changes both the design space and FEM discretization and is not a pure FEM error estimate.

| Mesh | Solid C | Frozen C | Reoptimized C | Gray fraction |
|---|---:|---:|---:|---:|
| 8×4 | 36.528 | 84.126 | 84.126 | 31.25% |
| 16×8 | 37.702 | 102.125 | 104.214 | 73.44% |
| 32×16 | 38.097 | 116.703 | 100.995 | 69.92% |
| 48×24 | 38.185 | 123.023 | 101.965 | 69.10% |

The last refinement changes solid compliance by 0.23%, frozen-field compliance by 5.42%, and reoptimized compliance by 0.96%. **Mesh independence is not established.** The small reoptimized change cannot substitute for the fixed-structure comparison. No asymptotic error estimate is claimed.

The 8×4 radius is only 0.76 element widths and its filter is effectively inactive. This coarse diagnostic does not resolve the physical filter scale. The other meshes represent the radius with 1.52, 3.04 and 4.56 element widths.

All four OC runs meet the specified update stopping rule (<0.01 for three consecutive updates after at least 20, maximum 240). This is not a global-optimality or KKT certificate.

## Binary reanalysis

Physical densities are binarized at thresholds 0.3, 0.5 and 0.7, or by selecting the largest densities at a fixed 50% material budget. Each mask is analyzed **without filtering again**. Void modulus is 10⁻⁶, with a sensitivity rerun at 10⁻⁸.

The gate requires: converged parent optimization; material ≤50%; compliance ≤105% of the continuous parent; four-neighbour support-to-load connectivity; and <1% sensitivity to void modulus. It does not establish stress, buckling, minimum feature size or manufacturability.

**One of 16 candidates passes:** volume-preserving selection on 48×24, C=65.907 versus continuous C=101.965, with 50% material. Penalized intermediate material is replaced by solid material and the binary field is outside the filtered-design family, so a lower compliance is possible. This is not an ML improvement or a manufacturing qualification.

The same selection increases compliance by 22.25% on 32×16 and 56.92% on 16×8. Threshold 0.5 retains too much material (54.51% on 48×24), so its apparently good stiffness fails the volume criterion. High thresholds can break load paths. All failures are retained.

Stable element-index tie breaking can break symmetry. Four-neighbour connectivity rejects corner-only contact even if ersatz FEM returns finite values. Disconnected structures with finite artificial stiffness are not meaningful load-bearing designs.

## Reproduce

```bash
python -m topopt.mesh_study --output my_mesh_study
```

Open the generated HTML report. CSV files, NPZ density masks, iteration histories, configuration and source hashes are included. The current CalculiX correlation covers continuous fields; it does not independently verify these binary shapes.

Next: sparse FEM for larger meshes, projected-density optimization with continuation, and independent refined/CalculiX analysis of selected binary geometry.
