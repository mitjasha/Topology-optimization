"""One-case 2D/3D FEM + SIMP optimization from GUI or CLI.

Keeps all exports with the numerical conditions that generated them.
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from topopt.solver import Case, Model, optimize
from topopt.solver3d import Case3D, Model3D, write_vtk


def run(output,dimension=3,*,nx=8,ny=4,nz=4,length=2.,
        width=1.,height=1.,volume=.5,load_y=.5,load_z=.5,
        radius=.38,max_iter=400,tolerance=.02):
    if dimension not in (2,3):
        raise ValueError("dimension must be 2 or 3")
    if dimension==2:
        case=Case(nx=nx,ny=ny,length=length,height=height,
                  volume=volume,load_y=load_y,radius=radius)
        model=Model(case,backend="sparse")
    else:
        case=Case3D(nx=nx,ny=ny,nz=nz,length=length,width=width,
                    height=height,volume=volume,load_y=load_y,
                    load_z=load_z,radius=radius)
        model=Model3D(case)
    original=model.analyze(model.project_volume(np.full(model.ne,volume)))
    result=optimize(model,max_iter=max_iter,tolerance=tolerance,min_iter=15)
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    shape=(ny,nx) if dimension==2 else (nz,ny,nx)
    np.savez_compressed(folder/"density.npz",
                        dimension=dimension,design=result["x"].reshape(shape),
                        physical_density=result["rho"].reshape(shape))
    if dimension==3:
        write_vtk(folder/"density.vtk",case,result["rho"])
    else:
        from topopt.benchmark import density_svg
        density_svg(result["rho"],nx,ny,folder/"density.svg")
    with (folder/"history.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("iteration","compliance","volume","change"))
        writer.writeheader()
        writer.writerows(result["history"])
    summary={"dimension":dimension,"case":asdict(case),"converged":bool(result["converged"]),
             "iterations":result["iterations"],"fe_solves":result["fe_solves"],
             "initial_compliance":original["compliance"],"compliance":result["compliance"],
             "volume":result["volume"],"equilibrium_error":result["equilibrium_error"],
             "seconds":result["seconds"],
             "quality_checks":{"volume":abs(result["volume"]-case.volume)<1e-6,
                               "equilibrium":result["equilibrium_error"]<1e-7,
                               "converged":bool(result["converged"])},
             "scope":"linear elastic rectangular cantilever with left fixed edge/face"}
    (folder/"case.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--dimension",type=int,choices=(2,3),default=3)
    p.add_argument("--nx",type=int,default=8)
    p.add_argument("--ny",type=int,default=4)
    p.add_argument("--nz",type=int,default=4)
    p.add_argument("--length",type=float,default=2.)
    p.add_argument("--width",type=float,default=1.)
    p.add_argument("--height",type=float,default=1.)
    p.add_argument("--volume",type=float,default=.5)
    p.add_argument("--load-y",type=float,default=.5)
    p.add_argument("--load-z",type=float,default=.5)
    p.add_argument("--radius",type=float,default=.38)
    p.add_argument("--max-iter",type=int,default=400)
    p.add_argument("--tolerance",type=float,default=.02)
    a=p.parse_args()
    print(json.dumps(run(**{k.replace("-","_"):v for k,v in vars(a).items()}),indent=2))
