# Two-dimensional and three-dimensional topology optimization

The repository now contains two separately testable finite-element solvers
that use the **same OC topology-optimization algorithm**.

| | 2D | 3D |
|---|---|---|
| Elements | 4-node Q4 | 8-node Hex8 |
| Displacement DOFs | 2 per node | 3 per node |
| Material | Plane stress | Full 3D isotropic linear elasticity |
| Field shape | (ny, nx) | (nz, ny, nx) |
| Machine learning | Ridge, U-Net, GNN, FNO and further experimental methods | Ridge, U-Net3D, MeshGNN3D, FNO3D |
| Visualization | 2D field | Interactive 3D cells, density threshold, Z slice and VTK |
| Data | Historical 2D NPZ/CSV | Independent 3D NPZ/CSV/JSON and ParaView VTK |

## Solve an individual 2D or 3D engineering case

Both dimensions have a **Single-case solver** GUI page. Set the mesh,
geometry, volume fraction, load position and filter radius. The
calculation writes a numerical quality report, density field and
convergence history; 3D cases additionally export a ParaView VTK.

```bash
python -m topopt.mlnext.case_run --dimension 3 --nx 8 --ny 4 --nz 4 --volume .5 --output artifacts/mlnext/manual3d
python -m topopt.mlnext.case_run --dimension 2 --nx 16 --ny 8 --volume .5 --output artifacts/mlnext/manual2d
```

Run the local bilingual GUI:

```bash
python -m pip install -r requirements-gui.txt
python -m pytest mltests/test_3d.py -v
python -m streamlit run topopt/mlnext/gui.py
```

Choose **Dimension → 2D / 3D** in the sidebar. Both paths are available
without changing branches. Existing 2D baseline evidence remains unchanged.

To create a small, real three-dimensional FEM dataset:

```bash
python -m topopt.mlnext.data3d --output artifacts/mlnext/solid3d/dataset3d.npz --count 60 --nx 8 --ny 4 --nz 4
python -m topopt.mlnext.train3d train --dataset artifacts/mlnext/solid3d/dataset3d.npz --model unet3d --output artifacts/mlnext/solid3d/unet3d --epochs 400
python -m topopt.mlnext.train3d evaluate --dataset artifacts/mlnext/solid3d/dataset3d.npz --output artifacts/mlnext/solid3d/comparison3d.csv --checkpoint unet3d=artifacts/mlnext/solid3d/unet3d/best.pt
```

The 3D solver uses 2x2x2-integrated Hex8 elements on a regular
hexahedral mesh, a clamped left end, a downward nodal force on the right
face, a cone-weighted physical density filter, SIMP material interpolation,
a sparse stiffness matrix, and OC updates. Geometry and force positions
can vary, but the current support/load families are not arbitrary CAD.

The neural networks are truly volumetric: U-Net uses Conv3d,
the Fourier operator uses 3D spectral kernels, and the graph model uses
six-neighbor 3D voxel-element adjacency. Their outputs are physically
volume-projected, independently FEM-audited, then refined via OC with
reference-free runtime fallback. The held-out 3D cases have longer
cantilevers and extreme load locations not seen in training.

The standard 3D training supports CPU and, when available, CUDA.
Sparse FEM, dataset generation and benchmark numerical checks use CPU.

**No unmeasured accuracy or speed claims:** model checkpoints and
performance numbers must be generated and audited on real 3D datasets.
Coarse 8x4x4 meshes are for basic correctness checks. Next studies need
mesh convergence, new BC families, other geometries, stress-based constraints,
large-scale iterative solvers and potentially learned FEM preconditioners.

3D diffusion and neural-CG are not yet available; the earlier
2D implementations remain in the 2D workspace.
