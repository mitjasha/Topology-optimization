"""Single-case and 3D GUI orchestration tests without a Streamlit runtime."""
import ast
import json
from pathlib import Path

import pytest

from topopt.mlnext import gui_backend as b
from topopt.mlnext.case_run import run


def test_2d_and_3d_single_cases(tmp_path):
    case2=run(tmp_path/"two",dimension=2,nx=8,ny=4,length=2,volume=.5,
              radius=.2,max_iter=100,tolerance=.04)
    case3=run(tmp_path/"three",dimension=3,nx=4,ny=4,nz=4,length=2,
              volume=.5,radius=.55,max_iter=140,tolerance=.04)
    for summary,folder,dimension in [
        (case2,tmp_path/"two",2),(case3,tmp_path/"three",3)]:
        assert summary["dimension"]==dimension
        assert summary["converged"]
        assert summary["quality_checks"]["equilibrium"]
        assert summary["quality_checks"]["volume"]
        assert summary["compliance"]<summary["initial_compliance"]
        assert (folder/"case.json").is_file()
        assert (folder/"density.npz").is_file()
        assert (folder/"history.csv").is_file()
    assert (tmp_path/"two"/"density.svg").is_file()
    assert (tmp_path/"three"/"density.vtk").is_file()


def test_3d_gui_command_construction(tmp_path,monkeypatch):
    base=tmp_path/"mlnext"
    base.mkdir()
    monkeypatch.setattr(b,"EXPERIMENTS",base)
    monkeypatch.setattr(b,"RUNS",base/"gui")
    ds=base/"dataset3d.npz"
    ds.write_bytes(b"dummy")
    steps=[
        ("single",b.single_case_task(base/"case",3,nx=8,ny=4,nz=4)),
        ("dataset",b.dataset_task3d(base/"study",count=12,nx=8,ny=4,nz=4)),
        ("training",b.train_task3d(ds,base/"fit","fno3d",epochs=2)),
        ("comparison",b.comparison_task3d(ds,base/"compare.csv",{}))]
    b._check_commands(steps)
    with pytest.raises(ValueError):
        b.single_case_task(base/"large",3,nx=128,ny=128,nz=32)
    with pytest.raises(ValueError):
        b.dataset_task3d(base/"too_big",nx=32,ny=16,nz=8)
    with pytest.raises(ValueError):
        b.train_task3d(ds,base/"bad","unet")
    with pytest.raises(ValueError):
        b.comparison_task3d(ds,base/"bad.csv",{"bad":base/"x.pt"})


def test_3d_study_seeds_and_bilingual_scripts():
    root=Path(__file__).resolve().parents[1]
    for name in ("gui.py","gui3d.py","gui_solver.py"):
        path=root/"topopt"/"mlnext"/name
        tree=ast.parse(path.read_text(encoding="utf-8"))
        names={}
        for stmt in tree.body:
            if isinstance(stmt,ast.Assign) and len(stmt.targets)==1:
                t=stmt.targets[0]
                if isinstance(t,ast.Name) and t.id in ("TEXT","EN","RU"):
                    names[t.id]=ast.literal_eval(stmt.value)
        if name=="gui.py":
            assert set(names["EN"])==set(names["RU"])
            assert "dimension" in names["EN"] and "solver" in names["EN"]
        else:
            assert set(names["TEXT"]["English"])==set(names["TEXT"]["Русский"])
