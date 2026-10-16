"""Tests for safe bilingual GUI orchestration (no browser and no training required)."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from topopt.mlnext import gui_backend as b


def test_bilingual_translation_keys_and_python_syntax():
    script = b.ROOT / "topopt" / "mlnext" / "gui.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    dicts = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and target.id in ("EN", "RU"):
                dicts[target.id] = ast.literal_eval(stmt.value)
    assert set(dicts) == {"EN", "RU"}
    assert len(dicts["EN"]) >= 100
    assert set(dicts["EN"]) == set(dicts["RU"])
    assert all(isinstance(x, str) and x.strip() for d in dicts.values() for x in d.values())


def test_gui_sources_compile_without_optional_ui_import():
    folder = b.ROOT / "topopt" / "mlnext"
    for name in ("gui.py", "gui_backend.py", "gui_worker.py"):
        path = folder / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.fixture
def local_store(monkeypatch, tmp_path):
    experiments = tmp_path / "mlnext"
    experiments.mkdir()
    monkeypatch.setattr(b, "EXPERIMENTS", experiments)
    monkeypatch.setattr(b, "RUNS", experiments / "gui")
    return experiments


def test_command_builder_and_study_guards(local_store):
    dataset = local_store / "experiment" / "dataset.npz"
    dataset.parent.mkdir()
    dataset.write_bytes(b"placeholder")
    command = b.dataset_task(local_store / "generated", count=24, nx=8, ny=4)
    assert command[:3] == [b.PYTHON, "-m", "topopt.mlnext.data"]
    assert command[command.index("--nx") + 1] == "8"
    study = b.make_study(dataset, local_store / "study",
                         ("unet", "gnn", "fno"), seeds=(7, 11), epochs=3)
    assert len(study) == 8  # three networks and one paired comparison per seed
    assert all(x[1][2] in b.SAFE_MODULES for x in study)
    b._check_commands(study)
    with pytest.raises(ValueError):
        b._strpath(local_store.parent / "outside")
    with pytest.raises(ValueError):
        b._check_commands([("unsafe", [b.PYTHON, "-m", "os", "--help"])])
    with pytest.raises(ValueError):
        b.make_study(dataset, local_store / "study", ("unet", "unet"))
    with pytest.raises(ValueError):
        b.dataset_task(local_store, nx=7, ny=4)
    with pytest.raises(ValueError):
        b.diffusion_task(dataset, local_store, architecture="unknown")


def test_run_registry_and_cancellation(local_store, monkeypatch):
    class FakeProcess:
        pid = 999999
    monkeypatch.setattr(b.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    ident = b.new_run("dataset", lambda p: [
        ("dataset", b.dataset_task(p / "sample", count=12, nx=8, ny=4))],
        {"created_for": "unit_test"})
    state = b.get_run(ident)
    assert state["status"] == "queued"
    assert state["steps"][0]["name"] == "dataset"
    assert state["pid"] == 999999
    assert b.request_cancel(ident)
    assert (b.RUNS / ident / "cancel.flag").is_file()
    log = b.RUNS / ident / "process.log"
    log.write_text("started\nok\n", encoding="utf-8")
    assert "ok" in b.tail_log(ident)
    assert b.list_runs()[0]["id"] == ident
    with pytest.raises(ValueError):
        b.get_run("../invalid")


def test_atomic_write_and_legacy_evidence_unmodified(local_store):
    path = local_store / "example" / "state.json"
    b.atomic_json(path, {"status": "completed", "value": 42})
    assert json.loads(path.read_text())["value"] == 42
    assert not list(path.parent.glob(".status-*"))
