"""Extended fixed-field convergence and projected binary design experiment."""
import argparse
import csv
from dataclasses import asdict,replace
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
from .solver import Case,Model,optimize
from .projected import continuation
from .mesh_study import volume_binary
from .benchmark import connected_load_path,density_svg


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    base=Case(nx=8,ny=4,load_span=0.25)
    coarse=optimize(Model(base,backend='sparse'),max_iter=240,tolerance=0.01,min_iter=20)
    field=coarse['rho'].reshape(4,8)
    rows=[]
    for nx in (8,16,32,48,64,96,128):
        case=replace(base,nx=nx,ny=nx//2);m=Model(case,backend='sparse')
        rho=np.repeat(np.repeat(field,nx//8,axis=0),nx//8,axis=1).ravel()
        frozen=m.analyze_density(rho);solid=m.analyze_density(np.ones(m.ne))
        change=None if not rows else frozen['compliance']/rows[-1]['frozen_compliance']-1
        rows.append({'nx':nx,'ny':nx//2,'frozen_compliance':frozen['compliance'],
                     'solid_compliance':solid['compliance'],'frozen_relative_change':change})
        print(rows[-1],flush=True)
    with (output/'refinement.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    binaries=[]
    for nx in (32,64):
        case=replace(base,nx=nx,ny=nx//2)
        model,result=continuation(case)
        binary=volume_binary(result['rho'],case.volume)
        direct=model.analyze_density(binary)
        soft=Model(replace(case,emin=1e-8),backend='sparse').analyze_density(binary)
        sensitivity=abs(soft['compliance']/direct['compliance']-1)
        connected=connected_load_path(binary,model)
        ratio=direct['compliance']/result['compliance']
        passed=bool(result['converged'] and connected and ratio<=1.05 and direct['volume']<=case.volume+1e-12 and sensitivity<0.01)
        row={'nx':nx,'ny':nx//2,'parent_converged':result['converged'],'binary_compliance':direct['compliance'],
             'projected_compliance':result['compliance'],'ratio':ratio,'volume':direct['volume'],
             'connected':connected,'void_sensitivity':sensitivity,'gate_pass':passed,'stages':result['stages']}
        binaries.append(row)
        np.savez_compressed(output/f'binary_{nx}.npz',binary=binary,physical_density=result['rho'],design=result['x'])
        density_svg(binary,case.nx,case.ny,output/f'binary_{nx}.svg')
        (output/f'binary_{nx}_case.json').write_text(json.dumps(asdict(case),indent=2),encoding='utf-8')
        print(row,flush=True)
    summary={'meshes':rows,'binary_candidates':binaries,'mesh_gate':{'tolerance':0.01,'successive_refinements':2},
             'mesh_independence_gate_pass':all(abs(r['frozen_relative_change'])<0.01 for r in rows[-2:]),
             'scope':'One newly optimized fixed coarse field and two projected designs; not a general convergence proof.',
             'python':platform.python_version(),'numpy':np.__version__,
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('my_refinement'));a=p.parse_args();run(a.output)
