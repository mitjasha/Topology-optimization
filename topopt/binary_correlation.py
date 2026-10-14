"""Independent reanalysis of saved physical binary masks, without refiltering."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .solver import Case,Model
from .correlation import write_deck,parse_dat,evaluate


def run(ccx,inputs,output,thicknesses=(0.01,0.001,0.0001)):
    ccx=ccx.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in sorted(inputs.glob('binary_*.npz')):
        case=Case(**json.loads(source.with_name(source.stem+'_case.json').read_text(encoding='utf-8')))
        binary=np.load(source)['binary']
        if not np.all((binary==0)|(binary==1)):
            raise ValueError('Independent binary comparison requires a 0/1 mask')
        for emin in (1e-6,1e-8):
            model=Model(replace(case,emin=emin),backend='sparse')
            for thickness in thicknesses:
                name=f'{source.stem}_e{emin:g}_t{thickness:g}'.replace('.','p')
                deck=output/(name+'.inp')
                write_deck(deck,model,binary,thickness,physical_density=True)
                # Prevent reuse of stale solver output after a failed execution.
                dat=output/(name+'.dat')
                if dat.exists():dat.unlink()
                env=dict(os.environ,OMP_NUM_THREADS='1',CCX_NPROC_RESULTS='1')
                proc=subprocess.run([str(ccx),'-i',name],cwd=output,env=env,capture_output=True,text=True,timeout=180)
                log=proc.stdout+proc.stderr
                (output/(name+'.log')).write_text(log,encoding='utf-8')
                if proc.returncode or 'Job finished' not in log or '*ERROR' in log:
                    raise RuntimeError(name+' failed; inspect the log')
                metrics,_=evaluate(model,binary,parse_dat(dat,model.nd//2),physical_density=True)
                banner=next((s.strip() for s in log.splitlines() if 'CalculiX Version' in s),'unknown')
                row={'candidate':source.stem,'thickness':thickness,'emin':emin,'solver_banner':banner,**metrics}
                rows.append(row);print(row,flush=True)
    if not rows:raise ValueError('No binary candidates found')
    result={'runs':rows,'all_correlation_gates_pass':all(r['gate_pass'] for r in rows),
            'executable_sha256':hashlib.sha256(ccx.read_bytes()).hexdigest(),
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.inp')},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'note':'Binary densities exported directly. Cross-solver verification does not establish manufacturability.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ccx',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('my_binary_correlation'))
    p.add_argument('--thicknesses',type=float,nargs='+',default=[0.01,0.001,0.0001])
    p.add_argument('--strict',action='store_true');a=p.parse_args();result=run(a.ccx,a.inputs,a.output,a.thicknesses)
    if a.strict and not result['all_correlation_gates_pass']:raise SystemExit(2)
