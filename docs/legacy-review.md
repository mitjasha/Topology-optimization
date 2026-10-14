# Historical notebook: scope of reuse

The original notebook is preserved unchanged for provenance. The current package is a separate reconstruction of a related experiment, not a repair that reproduces historical metrics.

| Historical issue | Treatment in the foundation |
|---|---|
| Missing `libtopology` and `X.npz` | New, explicitly identified Q4/SIMP implementation and generated data |
| Augmentation before random split | Whole trajectories and mirror-equivalent cases grouped before split |
| `np.rot90(train, 2)` defaults to axes (0,1) | No augmentation in baseline; future spatial transforms must use (1,2) for NHWC and transform BC/load metadata |
| `1:4` described as iterations 2–5 | New experiment explicitly takes the state after five OC updates |
| Incorrect unused R²/DIC definitions; ambiguous stateful IoU | No reuse; primary evaluation is compliance/volume/residual |
| Best checkpoint saved but not restored before evaluation | New ridge baseline is explicitly fitted after validation selection; neural checkpoint handling remains a future migration task |
| Fixed 50th iteration treated as target | Explicit convergence flag and maximum-iteration status |
| No FEM reanalysis of ML output | Direct prediction and refined prediction are both analyzed |
| No complete runtime accounting | Early iterations, projection/check and refinement counted |

Historical IoU scores should not be used as current verified results. The new package has not modernized or rerun the historical TensorFlow/Keras models. Restoring those models should follow restoration or documented replacement of the data generator, followed by a new leakage-safe experiment.

Preserved source: `mitjasha/Topology-optimization`, master, original notebook Git blob SHA `de1cd4c85c7e1a118b14121b2e59ad1a14ab9f5b` as inspected 2026-10-13. This identifies a file blob, not a repository commit.
