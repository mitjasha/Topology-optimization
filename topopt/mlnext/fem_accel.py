"""A learned displacement guess for physically exact preconditioned CG FEM solves.

This does not replace FEM equilibrium: every output is solved to tolerance.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve
from torch.nn import functional as F

from .data import GeneralModel, load_dataset, problem_from_record
from .models import NeuralFEM


def assemble_free(model, design):
    """Assemble the same SIMP stiffness as the Q4 reference solver."""
    rho = model.physical(design)
    c = model.case
    elastic_modulus = c.emin + (1 - c.emin) * rho ** c.penal
    values = (elastic_modulus[:, None, None] * model.ke[None, :, :]).ravel()
    k = coo_matrix((values, (model.rows, model.cols)),
                   shape=(model.nd, model.nd)).tocsr()
    free = model.free
    return k[free][:, free].tocsr(), model.f[free]


def train(dataset, output, *, epochs=300, seed=17, batch_size=8, device="cpu"):
    a = load_dataset(dataset)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    ids = {s: np.flatnonzero(a["split"] == s)
           for s in ("train", "validation", "test")}
    if not len(ids["train"]) or not len(ids["validation"]):
        raise ValueError("Train and validation cases required")
    scale = float(np.percentile(np.abs(a["displacements"][ids["train"]]), 95))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid displacement targets")
    inputs = torch.as_tensor(a["nodal"], dtype=torch.float32, device=device)
    targets = torch.as_tensor(a["displacements"] / scale,
                              dtype=torch.float32, device=device)
    model = NeuralFEM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    clock = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = np.random.permutation(ids["train"])
        for batch in np.array_split(permutation, max(1, int(np.ceil(len(permutation) / batch_size)))):
            opt.zero_grad()
            loss = F.mse_loss(model(inputs[batch]), targets[batch])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = float(F.mse_loss(model(inputs[ids["validation"]]),
                                     targets[ids["validation"]]))
        history.append({"epoch": epoch, "validation_mse": score})
        if score < best - 1e-8:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "scale": scale, "seed": seed},
                       output / "best.pt")
        else:
            stale += 1
        if stale >= 70:
            break
    report = {"best_validation_mse": best, "train_seconds": perf_counter()-clock,
              "epochs": len(history), "trained_on": "converged reference designs",
              "test_not_used_for_training": True, "history": history}
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


def run_cg(k, f, x0=None, rtol=1e-8):
    if rtol <= 0:
        raise ValueError("rtol must be positive")
    diagonal = k.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("Non-positive FEM stiffness diagonal")
    preconditioner = diags(1 / diagonal)
    counter = [0]
    def callback(_):
        counter[0] += 1
    t0 = perf_counter()
    sol, info = cg(k, f, x0=x0, M=preconditioner, rtol=rtol, atol=0,
                   maxiter=max(500, 12*len(f)), callback=callback)
    return sol, info, counter[0], perf_counter() - t0


def evaluate(dataset, checkpoint, output, device="cpu", rtol=1e-8):
    a = load_dataset(dataset)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    net = NeuralFEM().to(device)
    net.load_state_dict(state["state_dict"])
    net.eval()
    rows = []
    for i in np.flatnonzero(a["split"] == "test"):
        model = GeneralModel(problem_from_record(a["cases"][i]))
        design = a["targets"][i].ravel()
        k, f = assemble_free(model, design)
        # Benchmark both starts on EXACTLY the same assembled operator and RHS.
        cold, cold_info, cold_iters, cold_time = run_cg(k, f, rtol=rtol)
        tick = perf_counter()
        with torch.no_grad():
            predicted = net(torch.as_tensor(a["nodal"][i:i+1], device=device))
        guessed = predicted[0].cpu().numpy().transpose(1, 2, 0).ravel() * state["scale"]
        guess_time = perf_counter() - tick
        warm, warm_info, warm_iters, warm_time = run_cg(k, f, guessed[model.free], rtol=rtol)
        # Audit against independent sparse direct solve, counting fallback where needed.
        direct = spsolve(k.tocsc(), f)
        cold_rel = np.linalg.norm(cold - direct) / max(np.linalg.norm(direct), 1e-15)
        warm_rel = np.linalg.norm(warm - direct) / max(np.linalg.norm(direct), 1e-15)
        cold_res = np.linalg.norm(k @ cold - f) / max(np.linalg.norm(f), 1e-15)
        warm_res = np.linalg.norm(k @ warm - f) / max(np.linalg.norm(f), 1e-15)
        accepted = bool(warm_info == 0 and np.isfinite(warm_rel) and
                        warm_res < max(5*rtol, 1e-10))
        rows.append({"case_id": int(i), "cold_iters": cold_iters, "warm_iters": warm_iters,
                     "cold_info": int(cold_info), "warm_info": int(warm_info),
                     "cold_solve_seconds": cold_time, "warm_solve_seconds": warm_time,
                     "network_seconds": guess_time, "cold_relative_error": float(cold_rel),
                     "warm_relative_error": float(warm_rel), "cold_residual": float(cold_res),
                     "warm_residual": float(warm_res), "fallback_to_direct": not accepted,
                     "time_advantage": bool(accepted and warm_time+guess_time<cold_time)})
    if not rows:
        raise ValueError("No test cases")
    summary = {"cases": len(rows), "mean_cold_iterations": float(np.mean([r["cold_iters"] for r in rows])),
               "mean_warm_iterations": float(np.mean([r["warm_iters"] for r in rows])),
               "warm_accepted": sum(not r["fallback_to_direct"] for r in rows),
               "faster_including_inference": sum(r["time_advantage"] for r in rows),
               "results": rows,
               "scope": "Same-mesh final-design test cases; no claimed improvement for intermediate SIMP iterates.",
               "caveat": "Excludes stiffness assembly common to both methods; direct-solver fallback required when CG fails"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    t.add_argument("--output", default="artifacts/mlnext/fem")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    e = sub.add_parser("evaluate")
    e.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    e.add_argument("--output", default="artifacts/mlnext/fem_eval.json")
    a = p.parse_args()
    if a.action == "train":
        print(json.dumps(train(a.dataset, a.output, epochs=a.epochs,
                               device=a.device), indent=2))
    else:
        print(json.dumps(evaluate(a.dataset, a.checkpoint, a.output,
                                  device=a.device), indent=2))
