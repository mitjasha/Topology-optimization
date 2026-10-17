"""Reproducible multiboundary-condition FEM datasets; no test-set training leakage."""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from topopt.solver import Case, Model, optimize


@dataclass(frozen=True)
class Problem:
    case: Case
    support: str = "left"
    load_edge: str = "right"
    load_axis: str = "y"
    load_position: float = 0.5
    geometry: str = "solid"


class GeneralModel(Model):
    """Extend the reference Q4 discretization without changing its verified solver."""
    def __init__(self, problem):
        self.problem = problem
        super().__init__(problem.case, backend="sparse")
        yy, xx = np.mgrid[:problem.case.ny, :problem.case.nx]
        xn = (xx + .5) / problem.case.nx
        yn = (yy + .5) / problem.case.ny
        if problem.geometry == "solid":
            mask = np.ones_like(xn)
        elif problem.geometry == "bottom-notch":
            mask = np.where((xn > .35) & (xn < .65) & (yn < .26), 0., 1.)
        elif problem.geometry == "central-hole":
            mask = np.where((xn > .32) & (xn < .68) & (yn > .29) & (yn < .71), 0., 1.)
        else:
            raise ValueError("Unknown geometry")
        self.domain = mask.ravel()
        if problem.case.volume >= self.domain.mean():
            raise ValueError("Requested material volume exceeds available design domain")
        self.dv = np.maximum(self.filter.T @ self.domain / self.ne, 1e-12)
        c = problem.case
        nx, ny = c.nx, c.ny

        def node(i, j):
            return j * (nx + 1) + i

        if problem.support == "left":
            nodes = [node(0, j) for j in range(ny + 1)]
            fixed = np.ravel(np.column_stack((2 * np.array(nodes), 2 * np.array(nodes) + 1)))
        elif problem.support == "bottom":
            nodes = [node(i, 0) for i in range(nx + 1)]
            fixed = np.ravel(np.column_stack((2 * np.array(nodes), 2 * np.array(nodes) + 1)))
        elif problem.support == "corner-pins":
            fixed = [2 * node(0, 0), 2 * node(0, 0) + 1, 2 * node(nx, 0) + 1]
        else:
            raise ValueError("Unknown support: " + problem.support)
        self.fixed = np.unique(fixed).astype(int)
        self.free = np.setdiff1d(np.arange(self.nd), self.fixed)
        if problem.load_axis not in ("x", "y") or problem.load_edge not in ("top", "right"):
            raise ValueError("load_axis must be x/y and load_edge top/right")
        if not (0 <= problem.load_position <= 1):
            raise ValueError("Load position must belong to [0,1]")
        self.f[:] = 0
        if problem.load_edge == "right":
            s = problem.load_position * ny
            k = min(int(s), ny - 1)
            ids = (node(nx, k), node(nx, k + 1))
        else:
            s = problem.load_position * nx
            k = min(int(s), nx - 1)
            ids = (node(k, ny), node(k + 1, ny))
        a = int(problem.load_axis == "y")
        self.f[2 * ids[0] + a] = -(1 - (s - k))
        self.f[2 * ids[1] + a] = -(s - k)
        if not np.any(self.f[self.free]):
            raise ValueError("Applied load lies entirely on fixed degrees of freedom")

    def physical(self, design):
        return np.asarray(super().physical(design)) * self.domain

    def analyze(self, design):
        x = np.asarray(design, dtype=float).reshape(self.ne)
        if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
            raise ValueError("Design variables must be finite and in [0,1]")
        result = self.analyze_density(self.physical(x))
        result["gradient"] = self.filter.T @ (result["density_gradient"] * self.domain)
        return result


FEATURE_NAMES = ("early_density", "x", "y", "fixed_fraction", "force_x",
                 "force_y", "volume", "aspect", "load_span", "filter_radius", "domain_mask")


def cell_features(problem, model, early):
    """Physical geometry and load maps are supplied explicitly to every architecture."""
    c = problem.case
    ny, nx = c.ny, c.nx
    y, x = np.mgrid[:ny, :nx]
    fixed_node = np.zeros((ny + 1) * (nx + 1))
    fixed_node[np.unique(model.fixed // 2)] = 1
    fixed = fixed_node[model.nodes].mean(axis=1).reshape(ny, nx)
    nodal_force = model.f.reshape(ny + 1, nx + 1, 2)
    fx = (nodal_force[:-1, :-1, 0] + nodal_force[:-1, 1:, 0] +
          nodal_force[1:, :-1, 0] + nodal_force[1:, 1:, 0])
    fy = (nodal_force[:-1, :-1, 1] + nodal_force[:-1, 1:, 1] +
          nodal_force[1:, :-1, 1] + nodal_force[1:, 1:, 1])
    constant = lambda v: np.full((ny, nx), v, dtype=float)
    return np.stack((np.asarray(early).reshape(ny, nx),
                     (x + .5) / nx, (y + .5) / ny, fixed, fx, fy,
                     constant(c.volume), constant(c.length / c.height / 4),
                     constant(c.load_span), constant(c.radius / c.height),
                     model.domain.reshape(ny, nx))).astype("float32")


def nodal_features(model, density):
    """Nodal density, coordinates, forces and both fixed-DOF masks for FEM warm starts."""
    nx, ny = model.case.nx, model.case.ny
    sums = np.zeros((nx + 1) * (ny + 1))
    counts = np.zeros_like(sums)
    np.add.at(sums, model.nodes.ravel(), np.repeat(np.asarray(density), 4))
    np.add.at(counts, model.nodes.ravel(), 1)
    rho = (sums / np.maximum(counts, 1)).reshape(ny + 1, nx + 1)
    xx, yy = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))
    forces = model.f.reshape(ny + 1, nx + 1, 2)
    fixed = np.zeros(model.nd)
    fixed[model.fixed] = 1
    fixed = fixed.reshape(ny + 1, nx + 1, 2)
    return np.stack((rho, xx, yy, forces[:, :, 0], forces[:, :, 1],
                     fixed[:, :, 0], fixed[:, :, 1])).astype("float32")


def sample_problems(count=90, nx=16, ny=8, seed=17):
    """Whole boundary-condition families are reserved for the final OOD test."""
    if count < 12 or nx < 4 or ny < 4 or nx % 4 or ny % 4:
        raise ValueError("count >= 12 and grid counts must be multiples of four >= 4")
    rng = np.random.default_rng(seed)
    families = {"train": (("left", "right", "y"), ("bottom", "top", "x")),
                "validation": (("left", "right", "y"), ("bottom", "top", "x")),
                "test": (("corner-pins", "top", "y"), ("left", "top", "x"))}
    counts = {"train": int(.65 * count), "validation": int(.15 * count)}
    counts["test"] = count - counts["train"] - counts["validation"]
    for split, n in counts.items():
        for i in range(n):
            support, edge, axis = families[split][i % 2]
            length = (rng.uniform(1.45, 2.1) if split == "train" else
                      rng.uniform(1.65, 2.2) if split == "validation" else
                      rng.uniform(2.35, 2.8))
            c = Case(nx=nx, ny=ny, length=float(length),
                     volume=float(rng.uniform(.35, .65)),
                     load_y=.5, load_span=0.0,
                     radius=max(.19, 1.1 * length / nx))
            geometry = ("central-hole" if split == "test" else
                        "bottom-notch" if support == "left" and i % 3 == 0 else "solid")
            yield split, Problem(c, support, edge, axis, float(rng.uniform(.2, .8)), geometry)


def generate_dataset(output, count=90, nx=16, ny=8, seed=17,
                     max_iter=400, tolerance=.02):
    """Compute actual converged OC labels; reject nonconverged cases, never fabricate."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    from time import perf_counter
    started = perf_counter()
    records = []
    arrays = {k: [] for k in ("features", "targets", "nodal", "displacements",
                              "reference_compliance", "early_compliance")}
    failures = []
    for split, problem in sample_problems(count, nx, ny, seed):
        model = GeneralModel(problem)
        ref = optimize(model, max_iter=max_iter, min_iter=15, tolerance=tolerance)
        if not ref["converged"]:
            failures.append({"problem": {"case": asdict(problem.case),
                              "support": problem.support, "load_edge": problem.load_edge,
                              "load_axis": problem.load_axis, "load_position": problem.load_position,
                              "geometry": problem.geometry},
                             "reason": "reference_not_converged"})
            continue
        early = ref["snapshots"][4] if len(ref["snapshots"]) >= 5 else optimize(
            model, max_iter=5, min_iter=15)["x"]
        analysis = model.analyze(ref["x"])
        arrays["features"].append(cell_features(problem, model, early))
        arrays["targets"].append(ref["x"].reshape(ny, nx).astype("float32"))
        arrays["nodal"].append(nodal_features(model, model.physical(ref["x"])))
        arrays["displacements"].append(analysis["displacement"].reshape(ny + 1, nx + 1, 2)
                                        .transpose(2, 0, 1).astype("float32"))
        arrays["reference_compliance"].append(ref["compliance"])
        arrays["early_compliance"].append(model.analyze(early)["compliance"])
        records.append({"split": split, "case": asdict(problem.case),
                        "support": problem.support, "load_edge": problem.load_edge,
                        "load_axis": problem.load_axis, "load_position": problem.load_position,
                        "geometry": problem.geometry, "fe_solves": ref["fe_solves"]})
    if not records or not all(any(r["split"] == split for r in records)
                              for split in ("train", "validation", "test")):
        raise RuntimeError("Insufficient converged cases per split; increase max_iter")
    np.savez_compressed(output, **{k: np.asarray(v, dtype="float32")
                                   for k, v in arrays.items()},
                        split=np.asarray([r["split"] for r in records]),
                        cases=np.asarray([json.dumps(r) for r in records]))
    manifest = {"requested": count, "usable": len(records), "failures": failures,
                "generation_seconds": perf_counter() - started,
                "seed": seed, "max_iter": max_iter, "tolerance": tolerance,
                "feature_names": FEATURE_NAMES,
                "ood_test": "held-out supports, load families, aspect ratios and central-hole geometry",
                "warning": "FEM numerical references are not experimental validation",
                "counts": {s: sum(r["split"] == s for r in records)
                           for s in ("train", "validation", "test")}}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def problem_from_record(record):
    r = json.loads(str(record))
    return Problem(Case(**r["case"]), r["support"], r["load_edge"],
                   r["load_axis"], r["load_position"], r.get("geometry", "solid"))


def load_dataset(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlnext/dataset.npz"))
    parser.add_argument("--count", type=int, default=90)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-iter", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.output, args.count, args.nx, args.ny,
                                     args.seed, args.max_iter), indent=2))
