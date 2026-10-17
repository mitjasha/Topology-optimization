"""Reproducible 3D Hex8 topology-optimization datasets.

Dimension-specific NPZ schema: inputs [N, C, nz, ny, nx], targets [N, nz, ny, nx].
Train and validation share short/aspect-ratio families; the held-out test
contains longer cantilevers and load positions not copied from training.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from topopt.solver import optimize
from topopt.solver3d import Case3D, Model3D, write_vtk

FEATURE_NAMES = ("early_density", "x", "y", "z", "fixed_face", "load_plane",
                 "volume_fraction", "length_ratio", "width_ratio", "load_y", "load_z")


def features3d(model, early):
    c = model.case
    nz,ny,nx=c.nz,c.ny,c.nx
    zz,yy,xx=np.mgrid[:nz,:ny,:nx]
    const=lambda value:np.full((nz,ny,nx),value,dtype=np.float32)
    fixed_face=(xx==0).astype(np.float32)
    load_plane=np.zeros((nz,ny,nx),dtype=np.float32)
    load_plane[:,:,-1]=1.
    return np.stack((
        np.asarray(early).reshape(nz,ny,nx),(xx+.5)/nx,
        (yy+.5)/ny,(zz+.5)/nz,fixed_face,load_plane,
        const(c.volume),const(c.length/4),const(c.width/2),
        const(c.load_y),const(c.load_z))).astype(np.float32)


def sample_cases(count=90,nx=8,ny=4,nz=4,seed=17):
    if count < 12 or nx < 4 or ny < 4 or nz < 4:
        raise ValueError("At least 12 cases, grid dimensions >=4")
    rng=np.random.default_rng(seed)
    counts={"train":int(.65*count),"validation":int(.15*count)}
    counts["test"]=count-counts["train"]-counts["validation"]
    for split, amount in counts.items():
        for _ in range(amount):
            if split == "train":
                length=rng.uniform(1.5,2.15)
                load_y=rng.uniform(.2,.8)
                load_z=rng.uniform(.25,.75)
            elif split == "validation":
                length=rng.uniform(1.6,2.2)
                load_y=rng.uniform(.25,.75)
                load_z=rng.uniform(.25,.75)
            else:
                length=rng.uniform(2.4,2.9)
                load_y=rng.uniform(.08,.18) if rng.random()<.5 else rng.uniform(.82,.92)
                load_z=rng.uniform(.1,.2) if rng.random()<.5 else rng.uniform(.8,.9)
            width=float(rng.uniform(.8,1.2))
            c=Case3D(nx=nx,ny=ny,nz=nz,length=float(length),width=width,
                     volume=float(rng.uniform(.35,.65)),
                     load_y=float(load_y),load_z=float(load_z),
                     radius=max(.19,1.15*float(length)/nx))
            yield split,c


def generate_dataset(output,count=90,nx=8,ny=4,nz=4,seed=17,
                     max_iter=400,tolerance=.02,write_examples=True):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    arrays={"features":[],"targets":[],"reference_compliance":[],"reference_fe_solves":[]}
    records=[]
    failures=[]
    for split,case in sample_cases(count,nx,ny,nz,seed):
        model=Model3D(case)
        reference=optimize(model,max_iter=max_iter,min_iter=15,tolerance=tolerance)
        if not reference["converged"]:
            failures.append({"case":asdict(case),"split":split,"reason":"not_converged"})
            continue
        early=reference["snapshots"][4]
        arrays["features"].append(features3d(model,early))
        arrays["targets"].append(reference["x"].reshape(nz,ny,nx).astype(np.float32))
        arrays["reference_compliance"].append(reference["compliance"])
        arrays["reference_fe_solves"].append(reference["fe_solves"])
        records.append({"split":split,"case":asdict(case),"converged":True,
                        "iterations":reference["iterations"]})
        if write_examples and split=="test" and sum(r["split"]=="test" for r in records)<=2:
            write_vtk(output.parent/f"reference_test_{sum(r['split']=='test' for r in records)}.vtk",
                      case,reference["rho"])
    if not records or any(not any(r["split"]==s for r in records)
                          for s in ("train","validation","test")):
        raise RuntimeError("Not enough converged Hex8 cases for every partition")
    np.savez_compressed(output,**{k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()},
                        split=np.asarray([r["split"] for r in records]),
                        cases=np.asarray([json.dumps(r) for r in records]),
                        dimension=np.asarray(3))
    manifest={"dimension":3,"input_channels":FEATURE_NAMES,
              "requested":count,"usable":len(records),
              "counts":{s:sum(r["split"]==s for r in records)
                        for s in ("train","validation","test")},
              "failures":failures,"generation_seconds":perf_counter()-started,
              "seed":seed,"solver":"Hex8 sparse SIMP plane 3D elasticity",
              "test_policy":"held-out aspect ratios and more extreme y/z load positions",
              "scope":"rectangular cantilever, left face fixed, force in negative z; no arbitrary CAD",
              "physical_gates":"convergence, volume, equilibrium, compliance",
              "output":"FEM labels, NOT experimentally validated structural performance"}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def load3d(path):
    with np.load(path,allow_pickle=False) as data:
        result={k:data[k].copy() for k in data.files}
    if int(result.get("dimension",0))!=3:
        raise ValueError("Expected dimension=3 Hex8 dataset")
    return result


def case_from_record(record):
    return Case3D(**json.loads(str(record))["case"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("artifacts/mlnext/dataset3d.npz"))
    p.add_argument("--count",type=int,default=90)
    p.add_argument("--nx",type=int,default=8)
    p.add_argument("--ny",type=int,default=4)
    p.add_argument("--nz",type=int,default=4)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--max-iter",type=int,default=400)
    a=p.parse_args()
    print(json.dumps(generate_dataset(a.output,a.count,a.nx,a.ny,a.nz,a.seed,a.max_iter),
                     indent=2))
