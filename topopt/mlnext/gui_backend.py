"""Local, resumable GUI job manager for the scientific ML command-line tools.

No shell, remote queue, database, or cloud services. Commands run in a separate
local worker process and survive Streamlit reruns/browser tab closure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "artifacts" / "mlnext"
RUNS = EXPERIMENTS / "gui"
PYTHON = sys.executable
SAFE_MODULES = frozenset({
    "topopt.mlnext.data",
    "topopt.mlnext.train",
    "topopt.mlnext.comparison",
    "topopt.mlnext.diffusion",
    "topopt.mlnext.fem_accel",
    "topopt.mlnext.data3d",
    "topopt.mlnext.train3d",
})
TRAINABLE = ("unet", "gnn", "fno", "learned_update")
GENERATIVE = ("conv", "dit")


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".status-", suffix=".json", delete=False
    ) as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)
        temp = out.name
    os.replace(temp, path)


def _inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def _strpath(path):
    path = Path(path).resolve()
    if not _inside(path, EXPERIMENTS):
        raise ValueError("Only paths inside artifacts/mlnext are allowed")
    return str(path)


def datasets():
    if not EXPERIMENTS.is_dir():
        return []
    return sorted((p for p in EXPERIMENTS.rglob("*.npz")
                   if p.name == "dataset.npz" and p.is_file()
                   and _inside(p, EXPERIMENTS)),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def checkpoints(kind=None):
    if not EXPERIMENTS.is_dir():
        return []
    found = []
    for path in EXPERIMENTS.rglob("best.pt"):
        if not path.is_file():
            continue
        report = path.parent / "training.json"
        try:
            meta = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model = meta.get("architecture")
        # FEM checkpoints carry no architecture in training.json.
        if model is None and "best_validation_mse" in meta:
            model = "fem"
        if kind is None or model == kind:
            found.append((path, model))
    return sorted(found, key=lambda p: p[0].stat().st_mtime, reverse=True)


def dataset_info(path):
    """Read only the manifest. Data arrays and model weights are not deserialized."""
    path = Path(_strpath(path))
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = path.with_suffix(".json")
    if not manifest.is_file():
        return {"path": str(path), "warning": "No manifest"}
    info = json.loads(manifest.read_text(encoding="utf-8"))
    info["path"] = str(path)
    return info


def dataset_task(folder, *, count=24, nx=8, ny=4, seed=17, max_iter=400):
    if count < 12 or count > 100000:
        raise ValueError("Dataset count must be 12..100000")
    if nx < 4 or ny < 4 or nx % 4 or ny % 4:
        raise ValueError("Mesh counts must be multiples of four and >=4")
    if nx * ny > 16384:
        raise ValueError("Grid exceeds interactive safety limit")
    if max_iter < 15 or max_iter > 10000:
        raise ValueError("max_iter must be 15..10000")
    output = _strpath(Path(folder) / "dataset.npz")
    return [PYTHON, "-m", "topopt.mlnext.data", "--output", output,
            "--count", str(int(count)), "--nx", str(int(nx)), "--ny", str(int(ny)),
            "--seed", str(int(seed)), "--max-iter", str(int(max_iter))]


def train_task(dataset, folder, architecture, *, epochs=400, patience=90,
               batch_size=8, seed=17, physics_weight=0.0, physics_every=10,
               lr=0.001, device="cpu"):
    if architecture not in TRAINABLE:
        raise ValueError("Unknown supervised architecture")
    if not 1 <= epochs <= 20000 or not 1 <= patience <= 20000:
        raise ValueError("Epoch or patience out of bounds")
    if not 1 <= batch_size <= 1024 or physics_weight < 0 or physics_weight > 100:
        raise ValueError("Invalid batch size or physics weight")
    if physics_every < 1 or not 0 < lr <= 1:
        raise ValueError("Invalid physics interval or learning rate")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    return [PYTHON, "-m", "topopt.mlnext.train", "train",
            "--dataset", _strpath(dataset), "--model", architecture,
            "--output", _strpath(folder), "--epochs", str(int(epochs)),
            "--patience", str(int(patience)), "--batch-size", str(int(batch_size)),
            "--seed", str(int(seed)), "--physics-weight", str(float(physics_weight)),
            "--physics-every", str(int(physics_every)), "--lr", str(float(lr)),
            "--device", device]


def comparison_task(dataset, output, model_paths=None):
    cmd = [PYTHON, "-m", "topopt.mlnext.comparison",
           "--dataset", _strpath(dataset), "--output", _strpath(output)]
    for name, path in sorted((model_paths or {}).items()):
        if name not in TRAINABLE:
            raise ValueError("Unknown checkpoint architecture")
        cmd += ["--checkpoint", name + "=" + _strpath(path)]
    return cmd


def diffusion_task(dataset, folder, architecture="dit", epochs=100, steps=30, device="cpu"):
    if architecture not in GENERATIVE or not 1 <= epochs <= 20000 or not 2 <= steps <= 1000:
        raise ValueError("Invalid diffusion configuration")
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown device")
    return [PYTHON, "-m", "topopt.mlnext.diffusion", "train",
            "--dataset", _strpath(dataset), "--output", _strpath(folder),
            "--model", architecture, "--epochs", str(int(epochs)), "--steps", str(int(steps)),
            "--device", device]


def diffusion_eval_task(dataset, checkpoint, output, samples=4, device="cpu"):
    if not 1 <= samples <= 128:
        raise ValueError("samples must be 1..128")
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown device")
    return [PYTHON, "-m", "topopt.mlnext.diffusion", "evaluate",
            "--dataset", _strpath(dataset), "--checkpoint", _strpath(checkpoint),
            "--output", _strpath(output), "--samples", str(int(samples)),
            "--device", device]


def fem_task(dataset, folder, epochs=300, device="cpu"):
    if not 1 <= epochs <= 20000:
        raise ValueError("epochs out of bounds")
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown device")
    return [PYTHON, "-m", "topopt.mlnext.fem_accel", "train",
            "--dataset", _strpath(dataset), "--output", _strpath(folder),
            "--epochs", str(int(epochs)), "--device", device]


def fem_eval_task(dataset, checkpoint, output, device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown device")
    return [PYTHON, "-m", "topopt.mlnext.fem_accel", "evaluate",
            "--dataset", _strpath(dataset), "--checkpoint", _strpath(checkpoint),
            "--output", _strpath(output), "--device", device]


def make_study(dataset, folder, architectures, *, seeds=(17,), epochs=400,
               patience=90, batch_size=8, physics_weight=0.0,
               lr=.001, device="cpu"):
    """For each seed: train all selected models, then compare with OC + Ridge.

    Every architecture within a seed uses the same dataset and evaluation gate.
    """
    if not architectures or len(set(architectures)) != len(architectures):
        raise ValueError("Select distinct architectures")
    if len(seeds) > 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Use 1..10 distinct seeds")
    steps = []
    for seed in seeds:
        paths = {}
        for arch in architectures:
            target = Path(folder) / f"seed_{int(seed)}" / arch
            paths[arch] = target / "best.pt"
            steps.append((f"{arch} / seed {seed}",
                          train_task(dataset, target, arch, epochs=epochs,
                                     patience=patience, batch_size=batch_size,
                                     seed=int(seed), lr=lr, device=device,
                                     physics_weight=physics_weight if arch == "unet" else 0.0)))
        steps.append((f"Comparison / seed {seed}",
                      comparison_task(dataset, Path(folder) / f"seed_{int(seed)}" /
                                      "comparison.csv", paths)))
    return steps


def _check_commands(steps):
    if not steps or len(steps) > 100:
        raise ValueError("Invalid number of steps")
    for name, command in steps:
        if not name or not isinstance(command, list) or len(command) < 4:
            raise ValueError("Malformed step")
        if command[:2] != [PYTHON, "-m"] or command[2] not in SAFE_MODULES:
            raise ValueError("Unapproved command")
        # Every file argument was constructed through _strpath.
        for i, arg in enumerate(command[:-1]):
            if arg in ("--dataset", "--output", "--checkpoint"):
                value = command[i + 1]
                if arg == "--checkpoint" and "=" in value:
                    value = value.split("=", 1)[1]
                _strpath(value)


def new_run(kind, steps_builder, config):
    """Create a run. steps_builder takes the unique, reserved run directory."""
    if active_runs():
        raise RuntimeError("Another local experiment is running; finish or cancel it first.")
    RUNS.mkdir(parents=True, exist_ok=True)
    ident = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    folder = RUNS / ident
    folder.mkdir(exist_ok=False)
    try:
        steps = steps_builder(folder)
        _check_commands(steps)
        record = {
            "id": ident, "kind": kind, "status": "queued", "created": utcnow(),
            "started": None, "finished": None, "step_index": 0,
            "steps": [{"name": name, "argv": args} for name, args in steps],
            "config": config, "returncode": None, "pid": None,
            "directory": str(folder), "error": None,
        }
        atomic_json(folder / "run.json", record)
        proc = subprocess.Popen(
            [PYTHON, "-m", "topopt.mlnext.gui_worker", "--run", str(folder)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True,
            **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True}),
        )
        # The worker may already have advanced from queued -> running.
        # Never overwrite its status, logs or stage results with stale state.
        try:
            latest = get_run(ident)
            if latest["status"] == "queued":
                latest["pid"] = proc.pid
                atomic_json(folder / "run.json", latest)
        except (OSError, ValueError):
            pass
        return ident
    except Exception:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        raise


def list_runs(limit=120):
    if not RUNS.is_dir():
        return []
    result = []
    for path in sorted(RUNS.glob("*/run.json"), reverse=True):
        try:
            result.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if len(result) >= limit:
            break
    return result


def get_run(run_id):
    if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_id):
        raise ValueError("Invalid run id")
    file = RUNS / run_id / "run.json"
    if not _inside(file, RUNS):
        raise ValueError("Invalid run location")
    return json.loads(file.read_text(encoding="utf-8"))


def _pid_running(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_runs():
    """Exclude workers known to have exited; classify them as interrupted."""
    active = []
    for record in list_runs():
        if record["status"] not in ("queued", "running", "cancel_requested"):
            continue
        pid = record.get("pid")
        age_seconds = (datetime.now(timezone.utc) -
            datetime.fromisoformat(record["created"])).total_seconds()
        if pid and not _pid_running(pid) or (not pid and age_seconds > 20):
            record.update(status="interrupted", finished=utcnow(),
                          error="Worker process is no longer running")
            atomic_json(RUNS / record["id"] / "run.json", record)
        else:
            active.append(record)
    return active


def request_cancel(run_id):
    state = get_run(run_id)
    if state["status"] not in ("queued", "running", "cancel_requested"):
        return False
    (RUNS / run_id / "cancel.flag").write_text("cancel", encoding="utf-8")
    return True


def tail_log(run_id, max_bytes=32000):
    get_run(run_id)
    path = RUNS / run_id / "process.log"
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell()-max_bytes))
        return stream.read().decode("utf-8", "replace")


def summarize_run_files(run):
    folder = Path(run["directory"])
    return sorted([p for p in folder.rglob("*") if p.is_file()
                   and p.name not in ("run.json", "cancel.flag")
                   and not p.name.startswith(".status-")],
                  key=lambda p: str(p))


def read_json(path):
    path = Path(path)
    if not _inside(path, EXPERIMENTS):
        raise ValueError("Outside experiment folder")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Volumetric Hex8 experiment commands. Kept separate from 2D metadata/schema.
# ---------------------------------------------------------------------------
THREED_ARCHITECTURES = ("unet3d", "gnn3d", "fno3d")


def datasets3d():
    if not EXPERIMENTS.is_dir():
        return []
    return sorted(
        [p for p in EXPERIMENTS.rglob("dataset3d.npz")
         if p.is_file() and _inside(p, EXPERIMENTS)],
        key=lambda p: p.stat().st_mtime, reverse=True)


def dataset_task3d(folder, *, count=24, nx=8, ny=4, nz=4, seed=17, max_iter=400):
    if not 12 <= count <= 100000:
        raise ValueError("3D dataset case count must be between 12 and 100000")
    if min(nx, ny, nz) < 4 or nx*ny*nz > 8192:
        raise ValueError("3D mesh dimensions >=4 and at most 8192 Hex8 elements")
    if not 15 <= max_iter <= 10000:
        raise ValueError("Invalid FEM optimization iteration limit")
    return [PYTHON, "-m", "topopt.mlnext.data3d",
            "--output", _strpath(Path(folder)/"dataset3d.npz"),
            "--count", str(int(count)), "--nx", str(int(nx)),
            "--ny", str(int(ny)), "--nz", str(int(nz)),
            "--seed", str(int(seed)), "--max-iter", str(int(max_iter))]


def train_task3d(dataset, folder, architecture, *, epochs=400, patience=90,
                 batch_size=4, seed=17, lr=.001, device="cpu"):
    if architecture not in THREED_ARCHITECTURES:
        raise ValueError("Unknown 3D architecture")
    if not 1 <= epochs <= 20000 or not 1 <= patience <= 20000 or not 1 <= batch_size <= 1024:
        raise ValueError("Invalid 3D training parameters")
    if not 0 < lr <= 1 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid learning rate or device")
    return [PYTHON, "-m", "topopt.mlnext.train3d", "train",
            "--dataset", _strpath(dataset), "--model", architecture,
            "--output", _strpath(folder), "--epochs", str(int(epochs)),
            "--patience", str(int(patience)), "--batch-size", str(int(batch_size)),
            "--seed", str(int(seed)), "--lr", str(float(lr)), "--device", device]


def comparison_task3d(dataset, output, model_paths=None):
    cmd=[PYTHON, "-m", "topopt.mlnext.train3d", "evaluate",
         "--dataset", _strpath(dataset), "--output", _strpath(output)]
    for name,path in sorted((model_paths or {}).items()):
        if name not in THREED_ARCHITECTURES:
            raise ValueError("Unknown 3D checkpoint architecture")
        cmd.extend(["--checkpoint", name+"="+_strpath(path)])
    return cmd


def make_study3d(dataset, folder, architectures, *, seeds=(17,),
                 epochs=400, patience=90, batch_size=4, lr=.001, device="cpu"):
    if not architectures or len(architectures)!=len(set(architectures)):
        raise ValueError("Choose distinct 3D architectures")
    if not 1 <= len(seeds) <= 10 or len(set(seeds))!=len(seeds):
        raise ValueError("Use 1–10 distinct random seeds")
    steps=[]
    for seed in seeds:
        cp={}
        for architecture in architectures:
            target=Path(folder)/f"seed_{int(seed)}"/architecture
            cp[architecture]=target/"best.pt"
            steps.append((f"{architecture} / seed {seed}",
                         train_task3d(dataset,target,architecture,epochs=epochs,
                                      patience=patience,batch_size=batch_size,
                                      seed=int(seed),lr=lr,device=device)))
        steps.append((f"3D comparison / seed {seed}",
                      comparison_task3d(dataset,Path(folder)/f"seed_{int(seed)}"/
                                        "comparison3d.csv",cp)))
    return steps
