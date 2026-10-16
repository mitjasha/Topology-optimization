"""Paired benchmark: OC, dual Ridge, and any trained model on identical OOD cases."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from topopt.learning import RidgeSurrogate
from topopt.solver import optimize
from .data import GeneralModel, cell_features, load_dataset, problem_from_record
from .train import load_checkpoint, prediction_from_network


def ridge_fit(dataset):
    """Select alpha on validation FEM, keeping every test label sealed."""
    a = dataset
    xs = a["features"].reshape(len(a["features"]), -1)
    ys = a["targets"].reshape(len(a["targets"]), -1)
    tr = np.flatnonzero(a["split"] == "train")
    va = np.flatnonzero(a["split"] == "validation")
    scores = []
    start = perf_counter()
    for alpha in (.0001, .001, .01, .1, 1., 10.):
        reg = RidgeSurrogate(alpha).fit(xs[tr], ys[tr])
        compliance = []
        for i in va:
            model = GeneralModel(problem_from_record(a["cases"][i]))
            guess = model.project_volume(reg.predict(xs[i:i+1])[0])
            compliance.append(model.analyze(guess)["compliance"] /
                              float(a["reference_compliance"][i]))
        scores.append((float(np.mean(compliance)), alpha))
    alpha = min(scores)[1]
    return RidgeSurrogate(alpha).fit(xs[tr], ys[tr]), alpha, perf_counter()-start


def run(dataset, checkpoints, output, *, max_iter=400, tolerance=.02, seed=17):
    data = load_dataset(dataset)
    rng = np.random.default_rng(seed)
    ridge, alpha, ridge_seconds = ridge_fit(data)
    nets = {}
    train_times = {"ridge": ridge_seconds}
    for name, path in checkpoints.items():
        net, model_name = load_checkpoint(path)
        if name != model_name:
            raise ValueError(f"{name} checkpoint is {model_name}")
        nets[name] = net
        timing_path = Path(path).parent / "training.json"
        train_times[name] = (float(json.loads(timing_path.read_text())["train_seconds"])
                             if timing_path.is_file() else None)
    methods = ["oc", "ridge"] + list(nets)
    rows = []
    for i in np.flatnonzero(data["split"] == "test"):
        case = problem_from_record(data["cases"][i])
        reference = float(data["reference_compliance"][i])
        for method in rng.permutation(methods):
            started = perf_counter()
            model = GeneralModel(case)
            if method == "oc":
                attempt = optimize(model, max_iter=max_iter, min_iter=15, tolerance=tolerance)
                result = attempt
                nsolve = attempt["fe_solves"]
                direct_ratio = None
                fall = False
                ceiling = float("inf")
            else:
                early = optimize(model, max_iter=5, min_iter=15, tolerance=tolerance)
                features = cell_features(case, model, early["x"])
                ceiling = model.analyze(early["x"])["compliance"]
                if method == "ridge":
                    guess = model.project_volume(ridge.predict(
                        features.reshape(1, -1))[0])
                else:
                    guess = prediction_from_network(model, features, nets[method], method, "cpu")
                direct = model.analyze(guess)
                direct_ratio = direct["compliance"] / reference
                attempt = optimize(model, initial=guess, max_iter=max_iter,
                                   min_iter=15, tolerance=tolerance)
                # Both the ceiling and candidate checks are FEM analyses. The learned
                # update model additionally requires a sensitivity FEM analysis.
                nsolve = (early["fe_solves"] + 2 + attempt["fe_solves"] +
                          int(method == "learned_update"))
                accepted = (attempt["converged"] and np.isfinite(attempt["compliance"])
                            and attempt["compliance"] <= ceiling * (1+1e-8)
                            and abs(attempt["volume"]-case.case.volume) < 1e-6
                            and attempt["equilibrium_error"] < 1e-7)
                fall = not accepted
                result = attempt
                if fall:
                    result = optimize(model, max_iter=max_iter, min_iter=15, tolerance=tolerance)
                    nsolve += result["fe_solves"]
            passed = bool(result["converged"] and result["compliance"] <= 1.05*reference
                          and abs(result["volume"] - case.case.volume) < 1e-6
                          and result["equilibrium_error"] < 1e-7)
            rows.append({"case_id": int(i), "method": method,
                         "seconds": perf_counter()-started, "fe_solves": nsolve,
                         "fallback": fall, "final_gate": passed,
                         "direct_compliance_ratio": direct_ratio,
                         "final_compliance_ratio": result["compliance"]/reference})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base = {r["case_id"]: r for r in rows if r["method"] == "oc"}
    meta_path = Path(dataset).with_suffix(".json")
    label_cost = (float(json.loads(meta_path.read_text()).get("generation_seconds", 0))
                  if meta_path.is_file() else None)
    summary = {"dataset": str(dataset), "test_cases": len(base), "ridge_alpha": alpha,
               "model_summary": {}, "explanation": "Paired test on OOD, with fresh OC reference, no test-based runtime gate"}
    for method in methods:
        sample = [r for r in rows if r["method"] == method]
        saving = float(np.mean([base[r["case_id"]]["seconds"]-r["seconds"] for r in sample]))
        train_cost = train_times.get(method, 0.) if method != "oc" else 0.
        cost = (None if train_cost is None or label_cost is None else
                train_cost + label_cost)
        summary["model_summary"][method] = {
            "mean_seconds": float(np.mean([r["seconds"] for r in sample])),
            "mean_fe_solves": float(np.mean([r["fe_solves"] for r in sample])),
            "mean_paired_seconds_saved": saving,
            "fallbacks": sum(r["fallback"] for r in sample),
            "quality_passes": sum(r["final_gate"] for r in sample),
            "training_and_tuning_seconds": train_cost,
            "shared_dataset_generation_seconds": label_cost if method != "oc" else 0,
            "estimated_break_even_new_cases": (
                cost / saving if method != "oc" and cost is not None and saving > 0 else None)}
    summary["caveat"] = ("Timing is hardware-specific; dataset generation is shared among architectures; "
                          "the break-even estimate assumes the same problem distribution and no retraining.")
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    parser.add_argument("--output", default="artifacts/mlnext/comparison.csv")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="name=path/to/best.pt (repeat for each trained architecture)")
    args = parser.parse_args()
    cp = dict(entry.split("=", 1) for entry in args.checkpoint)
    print(json.dumps(run(args.dataset, cp, args.output), indent=2))
