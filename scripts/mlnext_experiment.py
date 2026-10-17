"""Orchestrate the seven scientific-ML experiments. Use --dry-run before spending compute."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def plan(root, quick=False):
    root = Path(root)
    data = root / "dataset.npz"
    counts = (24, 8, 4, 5, 5) if quick else (900, 32, 16, 400, 50)
    n, nx, ny, epochs, steps = counts
    py = sys.executable
    tasks = [("data", [py, "-m", "topopt.mlnext.data", "--output", str(data),
                        "--count", str(n), "--nx", str(nx), "--ny", str(ny)])]
    for name in ("unet", "gnn", "fno", "learned_update"):
        cmd = [py, "-m", "topopt.mlnext.train", "train", "--dataset", str(data),
               "--model", name, "--output", str(root / name),
               "--epochs", str(epochs), "--patience", str(max(5, epochs//4))]
        if name == "unet":
            cmd += ["--physics-weight", ".01", "--physics-every", "10" if not quick else "1"]
        tasks.append(("train_" + name, cmd))
    tasks += [
        ("comparison", [py, "-m", "topopt.mlnext.comparison", "--dataset", str(data),
                        "--output", str(root / "comparison.csv"),
                        *sum((["--checkpoint", f"{name}={root/name/'best.pt'}"]
                              for name in ("unet", "gnn", "fno", "learned_update")), [])]),
        ("diffusion_conv", [py, "-m", "topopt.mlnext.diffusion", "train",
                            "--dataset", str(data), "--output", str(root / "diffusion_conv"),
                            "--model", "conv", "--epochs", str(epochs), "--steps", str(steps)]),
        ("diffusion_dit", [py, "-m", "topopt.mlnext.diffusion", "train",
                           "--dataset", str(data), "--output", str(root / "diffusion_dit"),
                           "--model", "dit", "--epochs", str(epochs), "--steps", str(steps)]),
        ("diffusion_eval", [py, "-m", "topopt.mlnext.diffusion", "evaluate",
                            "--dataset", str(data),
                            "--checkpoint", str(root / "diffusion_dit" / "best.pt"),
                            "--output", str(root / "diffusion_eval.csv"), "--samples", "4"]),
        ("fem_train", [py, "-m", "topopt.mlnext.fem_accel", "train",
                       "--dataset", str(data), "--output", str(root / "fem"),
                       "--epochs", str(max(epochs, 5))]),
        ("fem_eval", [py, "-m", "topopt.mlnext.fem_accel", "evaluate",
                      "--dataset", str(data), "--checkpoint", str(root / "fem" / "best.pt"),
                      "--output", str(root / "fem_eval.json")])
    ]
    return tasks


def run(root, quick=False, dry_run=False):
    root = Path(root)
    tasks = plan(root, quick)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"profile": "quick" if quick else "research",
                "tasks": [{"id": name, "command": command} for name, command in tasks],
                "results": [], "warning": "Experimental methods must pass held-out physics gates"}
    (root / "commands.json").write_text(json.dumps(manifest, indent=2))
    for name, command in tasks:
        print(f"[{name}] " + shlex.join(command), flush=True)
        if dry_run:
            continue
        result = subprocess.run(command, check=False)
        manifest["results"].append({"task": name, "returncode": result.returncode})
        (root / "commands.json").write_text(json.dumps(manifest, indent=2))
        if result.returncode:
            raise SystemExit(f"Experiment {name} failed (code {result.returncode})")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/mlnext"))
    p.add_argument("--quick", action="store_true",
                   help="Small synthetic research smoke run (not a quality benchmark)")
    p.add_argument("--dry-run", action="store_true", help="Print and persist the full plan")
    args = p.parse_args()
    run(args.output, args.quick, args.dry_run)
