"""Supervised and physics-regularized neural warm starts with measurable OC fallback."""
import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from topopt.solver import optimize
from .data import GeneralModel, cell_features, load_dataset, problem_from_record
from .models import build_model


def volume_project_torch(x, model):
    """Differentiable clipped shift subject to filtered material-volume constraint.

    Bisection selects a shift without differentiating through the search; gradients
    propagate through the unsaturated material densities.
    """
    x = x.flatten().to(torch.float64)
    filt = torch.as_tensor(model.filter.toarray(), dtype=x.dtype, device=x.device)
    lo = (-x.max() - 1).detach()
    hi = (1 - x.min()).detach()
    with torch.no_grad():
        for _ in range(45):
            mid = (lo + hi) / 2
            trial = (x.detach() + mid).clamp(.001, 1)
            if (filt @ trial).mean() > model.case.volume:
                hi = mid
            else:
                lo = mid
        shift = (lo + hi) / 2
    return (x + shift).clamp(.001, 1), filt


def differentiable_compliance(x, model):
    """Dense differentiable FEM; opt-in small-mesh training loss, not a faster solver."""
    projected, filt = volume_project_torch(x, model)
    rho = filt @ projected
    case = model.case
    moduli = case.emin + (1 - case.emin) * rho.pow(case.penal)
    ke = torch.as_tensor(model.ke, dtype=rho.dtype, device=rho.device)
    values = (moduli[:, None, None] * ke[None]).reshape(-1)
    rows = torch.as_tensor(model.rows, dtype=torch.long, device=rho.device)
    cols = torch.as_tensor(model.cols, dtype=torch.long, device=rho.device)
    k = torch.zeros((model.nd, model.nd), dtype=rho.dtype, device=rho.device)
    k.index_put_((rows, cols), values, accumulate=True)
    free = torch.as_tensor(model.free, dtype=torch.long, device=rho.device)
    f = torch.as_tensor(model.f[model.free], dtype=rho.dtype, device=rho.device)
    solution = torch.linalg.solve(k[free][:, free], f)
    return f @ solution, rho.mean()


def prepare_update_input(features, model):
    """Two FEM-derived channels improve the early design: sensitivity and energy."""
    early = model.project_volume(features[0].flatten())
    solution = model.analyze(early)
    gradient = solution["gradient"].reshape(features.shape[1:])
    # Robust across very different load scales, without access to final labels.
    gscale = max(float(np.max(np.abs(gradient))), 1e-12)
    sensitivity = (gradient / gscale).astype("float32")
    return np.concatenate([features, sensitivity[None],
                           np.full_like(sensitivity[None],
                                        np.log1p(solution["compliance"]) / 30)], axis=0)


def prediction_from_network(model, feats, network, architecture, device):
    if architecture == "learned_update":
        feats = prepare_update_input(feats, model)
    inp = torch.as_tensor(feats[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = network(inp)[0, 0]
        if architecture != "learned_update":
            prediction = prediction.sigmoid()
    return model.project_volume(prediction.cpu().numpy().ravel())


def train(dataset, architecture="unet", output="artifacts/mlnext/unet", *,
          epochs=400, patience=90, lr=.001, batch_size=8, seed=17,
          physics_weight=0.0, physics_every=10, device="cpu"):
    if epochs < 1 or patience < 1 or batch_size < 1 or physics_every < 1:
        raise ValueError("Epochs, patience, batch size and physics_every must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dataset = load_dataset(dataset)
    split = dataset["split"].astype(str)
    indices = {s: np.flatnonzero(split == s)
               for s in ("train", "validation", "test")}
    if any(not len(indices[s]) for s in indices):
        raise ValueError("A non-empty train, validation and test split is required")
    feats = dataset["features"]
    targets = dataset["targets"]
    channels = feats.shape[1] + (2 if architecture == "learned_update" else 0)
    if architecture == "learned_update":
        enhanced = []
        for i, record in enumerate(dataset["cases"]):
            enhanced.append(prepare_update_input(feats[i],
                                                 GeneralModel(problem_from_record(record))))
        feats = np.stack(enhanced)
    net = build_model(architecture, channels).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=25)
    inputs = torch.from_numpy(feats).to(device)
    truth = torch.from_numpy(targets).to(device).unsqueeze(1)
    models = {}
    if physics_weight:
        if feats.shape[-1] * feats.shape[-2] > 512:
            raise ValueError("Differentiable dense FEM loss limited to <=512 elements")
        models = {int(i): GeneralModel(problem_from_record(dataset["cases"][i]))
                  for i in indices["train"]}
    best, stale, history = float("inf"), 0, []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        shuffled = np.random.permutation(indices["train"])
        total = 0.0
        for batch in np.array_split(shuffled, max(1, int(np.ceil(len(shuffled) / batch_size)))):
            optimizer.zero_grad()
            predicted = net(inputs[batch])
            if architecture != "learned_update":
                predicted = predicted.sigmoid()
            loss = F.mse_loss(predicted, truth[batch])
            if physics_weight and (epoch % physics_every == 0):
                # One independent training-only FEM case per batch.
                i = int(batch[0])
                compliance, volume = differentiable_compliance(
                    predicted[0, 0], models[i])
                ref = float(dataset["reference_compliance"][i])
                penalty = F.relu(compliance / ref - 1).square() + (volume - models[i].case.volume).square()
                loss = loss + physics_weight * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        net.eval()
        with torch.no_grad():
            score = 0.0
            for batch in np.array_split(indices["validation"], max(1, int(np.ceil(len(indices["validation"]) / batch_size)))):
                prediction = net(inputs[batch])
                if architecture != "learned_update":
                    prediction = prediction.sigmoid()
                score += float(F.mse_loss(prediction, truth[batch],
                                          reduction="sum")) / targets.shape[1] / targets.shape[2]
            score /= len(indices["validation"])
        scheduler.step(score)
        history.append({"epoch": epoch, "train_loss": total / len(indices["train"]),
                        "val_mse": score, "lr": optimizer.param_groups[0]["lr"]})
        if score < best - 1e-8:
            best, stale = score, 0
            torch.save({"state_dict": net.state_dict(), "architecture": architecture,
                        "channels": channels, "epoch": epoch, "seed": seed},
                       output / "best.pt")
        else:
            stale += 1
        if epoch % 25 == 0 or epoch == 1 or stale >= patience:
            print(f"{architecture} epoch={epoch} val_mse={score:.6g} best={best:.6g}", flush=True)
        if stale >= patience:
            break
    report = {"architecture": architecture, "best_validation_mse": best,
              "trained_epochs": len(history), "seed": seed,
              "train_cases": len(indices["train"]), "validation_cases": len(indices["validation"]),
              "test_cases_reserved": len(indices["test"]),
              "train_seconds": perf_counter() - start, "physics_weight": physics_weight,
              "physics_every": physics_every, "history": history,
              "selection": "Validation density MSE only; test never used for model selection"}
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    net = build_model(state["architecture"], state["channels"]).to(device)
    net.load_state_dict(state["state_dict"])
    return net.eval(), state["architecture"]


def evaluate(dataset, checkpoint, output, max_iter=400, tolerance=.02, device="cpu"):
    """Report both attempt/fallback and final OC quality on disjoint held-out cases."""
    dataset = load_dataset(dataset)
    indices = np.flatnonzero(dataset["split"] == "test")
    net, architecture = load_checkpoint(checkpoint, device)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in indices:
        problem = problem_from_record(dataset["cases"][i])
        model = GeneralModel(problem)
        start = perf_counter()
        early = optimize(model, max_iter=5, min_iter=15, tolerance=tolerance)
        features = cell_features(problem, model, early["x"])
        prediction = prediction_from_network(model, features, net, architecture, device)
        pred_analysis = model.analyze(prediction)
        attempt = optimize(model, initial=prediction, max_iter=max_iter,
                           tolerance=tolerance, min_iter=15)
        ceiling = model.analyze(early["x"])["compliance"]
        ref = float(dataset["reference_compliance"][i])
        # Runtime gate deliberately knows neither test reference nor training target.
        valid = (attempt["converged"] and np.isfinite(attempt["compliance"])
                 and attempt["compliance"] <= ceiling * (1 + 1e-8)
                 and abs(attempt["volume"] - problem.case.volume) < 1e-6
                 and attempt["equilibrium_error"] < 1e-7)
        solves = early["fe_solves"] + 1 + attempt["fe_solves"] + 1
        result = attempt
        if not valid:
            result = optimize(model, max_iter=max_iter, tolerance=tolerance, min_iter=15)
            solves += result["fe_solves"]
        elapsed = perf_counter() - start
        original = json.loads(str(dataset["cases"][i]))
        quality = (result["converged"] and result["compliance"] <= 1.05 * ref
                   and abs(result["volume"] - problem.case.volume) < 1e-6
                   and result["equilibrium_error"] < 1e-7)
        rows.append({"case_id": int(i), "support": original["support"],
                     "architecture": architecture, "fallback": not valid,
                     "quality_gate": bool(quality), "fe_solves": solves,
                     "seconds": elapsed, "baseline_fe_solves": original["fe_solves"],
                     "prediction_compliance_ratio": pred_analysis["compliance"] / ref,
                     "final_compliance_ratio": result["compliance"] / ref})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"architecture": architecture, "cases": len(rows),
               "quality_passes": sum(r["quality_gate"] for r in rows),
               "fallbacks": sum(r["fallback"] for r in rows),
               "mean_fe_solves": float(np.mean([r["fe_solves"] for r in rows])),
               "mean_baseline_fe_solves": float(np.mean([r["baseline_fe_solves"] for r in rows])),
               "mean_online_seconds": float(np.mean([r["seconds"] for r in rows])),
               "caveat": "Stored reference FEM solves, fresh ML online timings; do not mix with different hardware"}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    tr.add_argument("--model", choices=["unet", "gnn", "fno", "learned_update"], default="unet")
    tr.add_argument("--output", default="artifacts/mlnext/unet")
    tr.add_argument("--epochs", type=int, default=400)
    tr.add_argument("--patience", type=int, default=90)
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--physics-weight", type=float, default=0.)
    tr.add_argument("--physics-every", type=int, default=10)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--output", default="artifacts/mlnext/evaluation.csv")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.dataset, args.model, args.output,
                              epochs=args.epochs, patience=args.patience,
                              batch_size=args.batch_size, seed=args.seed,
                              physics_weight=args.physics_weight,
                              physics_every=args.physics_every), indent=2))
    else:
        print(json.dumps(evaluate(args.dataset, args.checkpoint, args.output), indent=2))
