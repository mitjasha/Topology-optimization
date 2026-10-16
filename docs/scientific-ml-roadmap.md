# Scientific machine learning for topology optimization

**Status:** experimental software on `feature/scientific-ml-suite`. All previous evidence under
`evidence_v2/` remains the historical baseline. Implementations below are not claimed speedups
until the independent tests and the full held-out benchmark are successful.

## What is implemented

| Development stage | Implementation | Evidence required |
|---|---|---|
| 1. Modern U-Net (AdamW, residual blocks, LR scheduler, early stopping, long training) | `topopt/mlnext/models.py`, `train.py` | learning curves; model selection exclusively on validation |
| 2. Physically located features | `data.py`: coordinates, loads, fixed masks, volume, filter, domain | ablation versus no BC channels |
| 3. Broader FEM examples and separated OOD test | `GeneralModel`: clamped-left/bottom, corner pins, point load, passive notches/holes, aspect ratios | converge reference; hold out geometry and support; mesh sensitivity |
| 4. Mesh GNN and FNO | `models.py`: graph message passing over element adjacency; spectral neural operator | identical train/validation/test splits and refiner |
| 5. Physically regularized learning and updates | differentiable SIMP/Q4 compliance, projected mass, `LearnedUpdate` | numeric gradient checks, physically meaningful loss, ablation |
| 6. Neural FEM acceleration | `fem_accel.py`: nodal displacement prediction plus Jacobi-PCG | count solver iterations, inference overhead, equilibrium residual, direct-solver checks |
| 7. Conditional diffusion and DiT | `diffusion.py`: DDPM denoisers, physical ranking, OC refinement | sample diversity, mass, compliance, costs, fallback |

## Run

Python 3.12 is recommended. CPU and PyTorch are sufficient for correctness tests.

```bash
python -m pip install -r requirements-mlnext.txt
python -m pytest mltests/test_mlnext.py -v
python scripts/mlnext_experiment.py --quick --dry-run
python scripts/mlnext_experiment.py --quick
```

The **quick** mode uses 24 coarse cases and five training epochs. It is only a smoke test.
It cannot support architecture-comparison conclusions. All iterations solve actual FEM equations.

The longer reproducibility run generates **900 FEM cases on 32×16 grids**, with 400 supervised
training epochs and 50 diffusion steps. It requires nontrivial offline CPU time and has not
been executed/verified in this commit:

```bash
python scripts/mlnext_experiment.py --dry-run
python scripts/mlnext_experiment.py
```

To conduct higher-resolution experiments separately:

```bash
python -m topopt.mlnext.data --output artifacts/mlnext/grid64/dataset.npz --count 1200 --nx 64 --ny 32
python -m topopt.mlnext.train train --dataset artifacts/mlnext/grid64/dataset.npz --model fno --output artifacts/mlnext/grid64/fno --epochs 600 --patience 120
python -m topopt.mlnext.train train --dataset artifacts/mlnext/grid64/dataset.npz --model gnn --output artifacts/mlnext/grid64/gnn --epochs 600 --patience 120
```

Note: physically regularized differentiable **dense** FEM loss is capped at 512 elements
because it is cubic in the unconstrained DOF count. Turn it off on 64×32 meshes
(the default is off) or replace it with an adjoint/implicit differentiation scheme.

Single-command comparative benchmark after completing training:

```bash
python -m topopt.mlnext.comparison \
  --dataset artifacts/mlnext/dataset.npz \
  --output artifacts/mlnext/comparison.csv \
  --checkpoint unet=artifacts/mlnext/unet/best.pt \
  --checkpoint gnn=artifacts/mlnext/gnn/best.pt \
  --checkpoint fno=artifacts/mlnext/fno/best.pt \
  --checkpoint learned_update=artifacts/mlnext/learned_update/best.pt
```

## Reproducibility and acceptance

* Train, validation and OOD test partitions are defined **before** solving; reference
  nonconvergence leads to an explicit omission and a manifest failure, never a fabricated label.
* Test introduces a **new support/load family, longer aspect ratios and central-hole geometry**.
  Training supports rectangles with an optional small boundary notch. This is not unrestricted
  CAD geometry. Cases all use Q4 plane-stress linear elasticity and SIMP.
* All networks share the same early-five-iteration inputs and subsequent OC optimizer.
  All online evaluations include model setup, early FEM iterations, inference, direct FEM
  prediction audit, optimization and an actual full-OC fallback when a reference-free gate fails.
* Ridge regularization is selected on validation; U-Net, GNN, FNO and update models use
  validation density MSE for checkpointing; diffusion uses validation noise MSE.
* Neural FEM is a distinct solver experiment, trained only on **converged design densities**.
  It cannot yet demonstrate gains for arbitrary intermediate optimization iterates.
  Jacobi-preconditioned CG starts are compared against each other with the **same assembled
  stiffness matrix** and the direct numerical solve is used for verification.
* Diffusion can be much more expensive than simple warm starts. Selecting from N samples
  performs N real FEM calls, accounted for in `diffusion.py`.
* Numerical quality requirements: converged OC reference, normalized volume error below 1e-6,
  free equilibrium residual below 1e-7, and final compliance <=1.05 times the independently
  computed test reference. Runtime fallback must never access the held-out reference.
* Evaluate sensitivity to three or more learning seeds, multiple meshes and training-set sizes.
  A run at one grid size or one seed must not be presented as generalization.
* Report reference generation, training, inference and full paired computation costs. Track
  a distribution of per-case speedups, rather than claiming speedups from network forward time.
* Check mesh independence of physical compliance and manufacturability before using outputs as
  engineering designs. Numerical FEM correlation does not mean experimental validation.

## Study design

The planned research sequence:

1. Reproduce old Ridge/U-Net experiment without changing `evidence_v2`.
2. Train residual U-Net for 300–1000 epochs; inspect early-stopping and data scaling.
3. Ablate coordinate, BC, force, domain-mask and early-field input channels.
4. Scale datasets and perform strictly held-out boundary/geometry and mesh-family studies.
5. Compare Ridge, residual U-Net, FNO, MeshGNN and learned-density update under identical budgets.
6. Test physics-regularized compliance against density-only MSE with controlled weight sweeps.
7. Benchmark neural FEM PCG starts; only integrate into OC if stable cost advantages emerge.
8. Train diffusion and DiT, measure candidate diversity, compliance, oc refinement and extra FEM calls.
9. Decide whether any neural system materially improves end-to-end time, robustness or quality.

## Limitations

The graph model uses four-neighbor adjacency of Q4 **element cells**, not a general tetrahedral
mesh library. The FNO uses regular-grid FFT and is not geometry-invariant by default.
The passive-void model describes holes through `E_min`; it does not remove hole degrees of
freedom or enforce manufacturability. The simplified notch/hole parametrization is an
initial controlled generalization test, not arbitrary geometry. Diffusion uses a compact
cell-token transformer, not a pretrained vision DiT. All models are trained from scratch.

These constraints are intentional and documented so findings can be improved by subsequent
physics/method extensions rather than overstated.
