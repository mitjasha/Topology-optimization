"""Conditional DDPM with convolutional or transformer denoiser, FEM-ranked sampling.

Research generator: test leakage and quality selection are explicitly disallowed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from topopt.solver import optimize
from .data import GeneralModel, cell_features, load_dataset, problem_from_record
from .models import ModernUNet


class ConvDenoiser(nn.Module):
    def __init__(self, channels=10, width=24):
        super().__init__()
        self.net = ModernUNet(channels + 2, width)

    def forward(self, noisy, condition, time):
        t = time.view(-1, 1, 1, 1).expand(-1, 1, *noisy.shape[-2:])
        return self.net(torch.cat([noisy, condition, t], dim=1))


class TinyDiT(nn.Module):
    """Cell-token diffusion transformer with explicit spatial conditioning."""
    def __init__(self, channels=10, width=64, depth=3, heads=4):
        super().__init__()
        self.embed = nn.Linear(channels + 2, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
                                           dim_feedforward=4 * width,
                                           batch_first=True, dropout=0, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def forward(self, noisy, condition, time):
        b, _, ny, nx = noisy.shape
        t = time.view(b, 1, 1, 1).expand(-1, 1, ny, nx)
        x = torch.cat([noisy, condition, t], dim=1)
        tokens = x.permute(0, 2, 3, 1).reshape(b, ny * nx, -1)
        return self.out(self.norm(self.encoder(self.embed(tokens)))).transpose(1, 2).reshape(b, 1, ny, nx)


def denoiser(kind, channels):
    if kind == "conv":
        return ConvDenoiser(channels)
    if kind == "dit":
        return TinyDiT(channels)
    raise ValueError("Denoiser must be conv or dit")


def schedule(steps, device):
    if steps < 2:
        raise ValueError("At least two diffusion steps needed")
    beta = torch.linspace(1e-4, .02, steps, device=device)
    alpha = 1 - beta
    abar = torch.cumprod(alpha, dim=0)
    return beta, alpha, abar


def train(dataset, output, *, architecture="dit", epochs=400, steps=50,
          batch_size=8, seed=17, lr=2e-4, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    a = load_dataset(dataset)
    train_ids = np.flatnonzero(a["split"] == "train")
    val_ids = np.flatnonzero(a["split"] == "validation")
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Train and validation sets required")
    x = torch.as_tensor(a["targets"][:, None] * 2 - 1, device=device)
    cond = torch.as_tensor(a["features"], device=device)
    net = denoiser(architecture, cond.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    _, _, abar = schedule(steps, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    records = []
    clock = perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        perm = np.random.permutation(train_ids)
        losses = []
        for ids in np.array_split(perm, max(1, math.ceil(len(perm) / batch_size))):
            xt = x[ids]
            t = torch.randint(steps, (len(ids),), device=device)
            noise = torch.randn_like(xt)
            z = abar[t].view(-1, 1, 1, 1).sqrt() * xt + (1 - abar[t]).view(-1, 1, 1, 1).sqrt() * noise
            opt.zero_grad()
            loss = F.mse_loss(net(z, cond[ids], t.float() / steps), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        # Validation is noise prediction using independently drawn validation noise.
        net.eval()
        with torch.no_grad():
            ids = val_ids
            t = torch.randint(steps, (len(ids),), device=device)
            noise = torch.randn_like(x[ids])
            z = abar[t].view(-1, 1, 1, 1).sqrt() * x[ids] + (1 - abar[t]).view(-1, 1, 1, 1).sqrt() * noise
            validation = float(F.mse_loss(net(z, cond[ids], t.float() / steps), noise))
        if validation < best:
            best = validation
            torch.save({"architecture": architecture, "channels": cond.shape[1],
                        "steps": steps, "seed": seed, "state_dict": net.state_dict()},
                       output / "best.pt")
        records.append({"epoch": epoch, "train_noise_mse": float(np.mean(losses)),
                        "validation_noise_mse": validation})
        if epoch % 25 == 0 or epoch == 1:
            print(f"diffusion/{architecture}: epoch={epoch} validation={validation:.5f}", flush=True)
    report = {"architecture": architecture, "train_cases": len(train_ids),
              "validation_cases": len(val_ids), "test_cases_reserved": int(np.sum(a["split"] == "test")),
              "epochs": epochs, "steps": steps, "best_val_noise_mse": best,
              "seconds": perf_counter() - clock, "history": records,
              "warning": "Generation quality and FEM speedup have not been established by denoising loss"}
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def sample(model, condition, steps, n=4, seed=42, device="cpu"):
    """Full ancestral DDPM: no shortened schedule or unstated diffusion approximation."""
    gen = torch.Generator(device=device).manual_seed(seed)
    beta, alpha, abar = schedule(steps, device)
    cond = torch.as_tensor(condition, dtype=torch.float32, device=device).unsqueeze(0)
    cond = cond.repeat(n, 1, 1, 1)
    z = torch.randn((n, 1, *condition.shape[-2:]), generator=gen, device=device)
    for t in reversed(range(steps)):
        time = torch.full((n,), t / steps, device=device)
        eps = model(z, cond, time)
        mean = (z - beta[t] * eps / (1 - abar[t]).sqrt()) / alpha[t].sqrt()
        if t > 0:
            posterior_var = beta[t] * (1 - abar[t-1]) / (1 - abar[t])
            z = mean + posterior_var.sqrt() * torch.randn(z.shape, generator=gen, device=device)
        else:
            z = mean
    return ((z.clamp(-1, 1) + 1) / 2)[:, 0].cpu().numpy()


def evaluate(dataset, checkpoint, output, *, samples=4, seed=42, device="cpu",
             max_iter=400):
    """Physics-rank samples then refine top candidate; no test reference in selection."""
    data = load_dataset(dataset)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    net = denoiser(state["architecture"], state["channels"]).to(device)
    net.load_state_dict(state["state_dict"])
    net.eval()
    records = []
    for i in np.flatnonzero(data["split"] == "test"):
        problem = problem_from_record(data["cases"][i])
        model = GeneralModel(problem)
        tick = perf_counter()
        early = optimize(model, max_iter=5, min_iter=15, tolerance=.02)
        condition = cell_features(problem, model, early["x"])
        candidates = sample(net, condition, state["steps"], samples, seed + int(i), device)
        ranked = []
        for j, raw in enumerate(candidates):
            density = model.project_volume(raw.ravel())
            quality = model.analyze(density)
            ranked.append((quality["compliance"], j, density))
        score, index, best = min(ranked, key=lambda x: x[0])
        result = optimize(model, initial=best, max_iter=max_iter,
                          min_iter=15, tolerance=.02)
        ceiling = model.analyze(early["x"])["compliance"]
        accepted = (result["converged"] and result["compliance"] <= ceiling * (1 + 1e-8)
                    and abs(result["volume"] - problem.case.volume) < 1e-6
                    and result["equilibrium_error"] < 1e-7)
        solves = early["fe_solves"] + len(candidates) + 1 + result["fe_solves"]
        if not accepted:
            result = optimize(model, max_iter=max_iter, min_iter=15, tolerance=.02)
            solves += result["fe_solves"]
        ref = float(data["reference_compliance"][i])
        records.append({"case_id": int(i), "architecture": state["architecture"],
                        "selected_sample": index, "best_direct_compliance": score,
                        "samples": samples, "fallback": not accepted,
                        "fe_solves": solves, "seconds": perf_counter() - tick,
                        "reference_ratio": result["compliance"] / ref,
                        "quality_gate": bool(result["converged"] and
                            result["compliance"] <= 1.05 * ref and
                            abs(result["volume"] - problem.case.volume) < 1e-6 and
                            result["equilibrium_error"] < 1e-7)})
    if not records:
        raise ValueError("No test cases")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"architecture": state["architecture"], "evaluated": len(records),
               "quality_passes": sum(r["quality_gate"] for r in records),
               "fallbacks": sum(r["fallback"] for r in records),
               "mean_fe_solves": float(np.mean([r["fe_solves"] for r in records])),
               "mean_seconds": float(np.mean([r["seconds"] for r in records])),
               "warning": "Generation incurs one FEM analysis per sample; do not claim speedup absent paired baseline"}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    t.add_argument("--output", default="artifacts/mlnext/dit")
    t.add_argument("--model", choices=["conv", "dit"], default="dit")
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--steps", type=int, default=50)
    v = sub.add_parser("evaluate")
    v.add_argument("--dataset", default="artifacts/mlnext/dataset.npz")
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--output", default="artifacts/mlnext/diffusion.csv")
    v.add_argument("--samples", type=int, default=4)
    a = p.parse_args()
    if a.action == "train":
        print(json.dumps(train(a.dataset, a.output, architecture=a.model,
                               epochs=a.epochs, steps=a.steps), indent=2))
    else:
        print(json.dumps(evaluate(a.dataset, a.checkpoint, a.output,
                                  samples=a.samples), indent=2))
