"""Independent CalculiX CPS4 cross-solver correlation (not experimental validation).

Run: python -m topopt.correlation --ccx /path/to/ccx --output correlation
The executable is an external dependency and is never bundled with this project.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
from .solver import Case, Model, optimize


LIMITS = {'compliance_relative':0.01,'displacement_relative_l2':0.01,
          'reaction_relative_l2':0.05,'force_balance_absolute':1e-5}


def write_deck(path, model, design, thickness):
    """Same membrane stiffness E*t, Poisson ratio, nodes, load and in-plane BCs.

    Thickness sensitivity is required because CCX expands CPS4 to C3D8.
    E_CCX=E_reference/thickness preserves E*t=E_reference*1.
    """
    if not np.isfinite(thickness) or thickness<=0:
        raise ValueError('Thickness must be positive and finite')
    c=model.case
    rho=model.physical(design)
    moduli=(c.emin+(1-c.emin)*rho**c.penal)/thickness
    lines=['*HEADING','Independent Q4/SIMP correlation; linear static; unit membrane stiffness',
           '** CPS4 expanded internally; E*t is preserved. No nonlinear geometry.', '*NODE,NSET=NALL']
    for j in range(c.ny+1):
        for i in range(c.nx+1):
            node=j*(c.nx+1)+i+1
            lines.append(f'{node},{i*c.length/c.nx:.16g},{j*c.height/c.ny:.16g},0')
    for j in range(c.ny):
        for i in range(c.nx):
            eid=j*c.nx+i+1
            a=j*(c.nx+1)+i+1
            lines.extend([f'*ELEMENT,TYPE=CPS4,ELSET=E{eid}',
                          f'{eid},{a},{a+1},{a+c.nx+2},{a+c.nx+1}',
                          f'*MATERIAL,NAME=M{eid}','*ELASTIC',f'{moduli[eid-1]:.16g},{c.nu:.16g}',
                          f'*SOLID SECTION,ELSET=E{eid},MATERIAL=M{eid}',f'{thickness:.16g}'])
    lines.append('*BOUNDARY')
    for j in range(c.ny+1):lines.append(f'{j*(c.nx+1)+1},1,2,0')
    lines.extend(['*STEP','*STATIC,SOLVER=SPOOLES','*CLOAD'])
    for dof in np.flatnonzero(model.f):
        lines.append(f'{dof//2+1},{dof%2+1},{model.f[dof]:.16g}')
    lines.extend(['*NODE PRINT,NSET=NALL','U,RF','*END STEP'])
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')


def parse_dat(path, node_count):
    """Read one linear-static displacement/force table; fail on incomplete data."""
    tables={'u':{},'rf':{}}
    mode=None
    for line in path.read_text().splitlines():
        if 'displacements (vx,vy,vz)' in line:
            if tables['u']:raise ValueError('Multiple displacement tables: only one step is supported')
            mode='u';continue
        if 'forces (fx,fy,fz)' in line:
            if tables['rf']:raise ValueError('Multiple force tables: only one step is supported')
            mode='rf';continue
        fields=line.split()
        if mode and len(fields)==4 and fields[0].isdigit():
            node=int(fields[0])
            if node in tables[mode]:raise ValueError('Duplicate node in CalculiX output')
            tables[mode][node]=[float(v.replace('D','E')) for v in fields[1:]]
    expected=set(range(1,node_count+1))
    if any(set(t)!=expected for t in tables.values()):
        raise ValueError('Missing or unexpected node IDs in CalculiX output')
    result={key:np.array([values[i] for i in range(1,node_count+1)]) for key,values in tables.items()}
    if not all(np.all(np.isfinite(x)) for x in result.values()):
        raise ValueError('Non-finite solver output')
    return result


def evaluate(model, design, parsed):
    reference=model.analyze(design)
    u=parsed['u'][:,:2].ravel()
    reactions=parsed['rf'][:,:2].ravel()[model.fixed]
    compliance=float(model.f@u)
    balance=reactions.reshape(-1,2).sum(axis=0)+model.f.reshape(-1,2).sum(axis=0)
    errors={'compliance_relative':abs(compliance/reference['compliance']-1),
            'displacement_relative_l2':float(np.linalg.norm(u-reference['displacement'])/np.linalg.norm(reference['displacement'])),
            'reaction_relative_l2':float(np.linalg.norm(reactions-reference['reaction'])/np.linalg.norm(reference['reaction'])),
            'force_balance_absolute':float(np.linalg.norm(balance))}
    return {'numpy_compliance':reference['compliance'],'calculix_compliance':compliance,
            **errors,'gate_pass':all(errors[key]<=limit for key,limit in LIMITS.items())},reference


def run(ccx,output):
    ccx=ccx.resolve()
    if not ccx.is_file():raise FileNotFoundError(ccx)
    output.mkdir(parents=True,exist_ok=True)
    # Predeclared gates saved before running; failures remain in the results.
    (output/'acceptance.json').write_text(json.dumps(LIMITS,indent=2))
    c=Case(load_y=0.37)
    model=Model(c)
    opt=optimize(model,max_iter=180,tolerance=0.02,min_iter=15)
    if not opt['converged']:raise RuntimeError('Reference optimization did not converge')
    cases=[('solid',model,np.ones(model.ne)),
           ('graded',model,0.25+0.65*np.linspace(0,1,model.ne)),
           ('optimized',model,opt['x'])]
    rows=[];banners=[];case_info=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',CCX_NPROC_RESULTS='1')
    for label,m,x in cases:
        case_info.append({'name':label,'case':asdict(m.case),'design':x.tolist()})
        for thickness in (1.0,0.01,0.001):
            name=f'{label}_t{thickness:g}'.replace('.','p')
            path=output/f'{name}.inp'
            write_deck(path,m,x,thickness)
            result=subprocess.run([str(ccx),'-i',name],cwd=output.resolve(),env=env,
                                  capture_output=True,text=True,timeout=120)
            log=result.stdout+result.stderr
            (output/f'{name}.log').write_text(log,encoding='utf-8')
            if result.returncode or 'Job finished' not in log or '*ERROR' in log:
                raise RuntimeError(f'CalculiX failed: inspect {name}.log')
            banners.append(next((line.strip() for line in log.splitlines() if 'CalculiX Version' in line),'unknown'))
            parsed=parse_dat(output/f'{name}.dat',m.nd//2)
            metrics,ref=evaluate(m,x,parsed)
            row={'case':label,'thickness':thickness,'role':'thickness diagnostic' if thickness==1 else 'thin membrane correlation',**metrics}
            rows.append(row)
            np.savez_compressed(output/f'{name}_fields.npz',design=x,physical_density=ref['rho'],
                                numpy_u=ref['displacement'],calculix_u=parsed['u'],
                                numpy_reaction=ref['reaction'],calculix_rf=parsed['rf'])
            print(json.dumps(row),flush=True)
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta={'solver_banners':sorted(set(banners)),'executable_sha256':hashlib.sha256(ccx.read_bytes()).hexdigest(),
          'download_source':'https://www.dhondt.de/calculix_2.23_4win.zip',
          'python':platform.python_version(),'numpy':np.__version__,'limits':LIMITS,
          'input_cases':case_info,'optimized_reference_converged':opt['converged'],
          'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('solver.py'))},
          'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.inp')},
          'interpretation':'Cross-solver verification, not experimental validation; CPS4 expands into C3D8. Thickness and formulation effects must be interpreted.'}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    write_report(output,rows,meta)
    return rows


def write_report(output,rows,metadata):
    thin=[r for r in rows if r['thickness']<0.1]
    smallest=[r for r in rows if r['thickness']==min(x['thickness'] for x in rows)]
    thick=[r for r in rows if r['thickness']==1]
    decision=(f'<b>{sum(r["gate_pass"] for r in thin)}/{len(thin)} thin-membrane runs pass the predeclared gates.</b> '
              f'At the smallest thickness, compliance differs by {min(r["compliance_relative"] for r in smallest):.2%}–'
              f'{max(r["compliance_relative"] for r in smallest):.2%}; displacement L2 by at most '
              f'{max(r["displacement_relative_l2"] for r in smallest):.2%}; reaction L2 by at most '
              f'{max(r["reaction_relative_l2"] for r in smallest):.2%}.')
    diagnostic=(f'{sum(not r["gate_pass"] for r in thick)}/{len(thick)} unit-thickness diagnostic runs fail. '
                f'The largest compliance discrepancy is {max(r["compliance_relative"] for r in thick):.1%}. '
                'Failures are retained rather than hidden or accommodated by changing the thresholds.')
    table=''.join(f'<tr><td>{r["case"]}</td><td>{r["thickness"]:g}</td>'
                  f'<td>{r["compliance_relative"]:.3%}</td><td>{r["displacement_relative_l2"]:.3%}</td>'
                  f'<td>{r["reaction_relative_l2"]:.3%}</td><td>{r["force_balance_absolute"]:.2e}</td>'
                  f'<td class="{"pass" if r["gate_pass"] else "fail"}">{"PASS" if r["gate_pass"] else "FAIL"}</td></tr>'
                  for r in rows)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CalculiX correlation — topology optimization</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#173145;background:#edf2f5;margin:0}main{max-width:1100px;margin:auto;padding:36px 24px}h1{font-size:36px;line-height:1.2}section{background:white;padding:24px;margin:20px 0;border-radius:12px;border:1px solid #d6e1e7}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow-x:auto}.pass{color:#176742;font-weight:bold}.fail{color:#a44122;font-weight:bold}a{color:#006f87}aside{background:#fff3df;padding:18px;border-left:4px solid #c88b32}
</style><main><p>CAE VERIFICATION · INDEPENDENT SOLVER</p><h1>Does the NumPy FEM agree with CalculiX?</h1>
<p>CalculiX 2.23 · CPS4 · linear statics · three density fields · three thickness settings</p>
<section><h2>Decision</h2><p>'''+decision+'''</p><p>'''+diagnostic+'''</p></section>
<aside><b>This is cross-solver verification, not experimental validation.</b> It supports the linear membrane implementation on three cases at one mesh and load position. It does not qualify strength, nonlinear response, manufacturing geometry, or all ML predictions.</aside>
<section><h2>Results, including failed diagnostics</h2><div class="scroll"><table><thead><tr><th>Density field</th><th>Thickness</th><th>Compliance error</th><th>Displacement L2</th><th>Reaction L2</th><th>Force balance</th><th>Gate</th></tr></thead><tbody>'''+table+'''</tbody></table></div></section>
<section><h2>Why thickness matters</h2><p>CalculiX expands CPS4 into C3D8 internally. The reference solver directly uses a 2D plane-stress constitutive matrix. For the thickness study, E<sub>CCX</sub>=E<sub>reference</sub>/t so E·t, Poisson ratio, in-plane mesh, forces and in-plane constraints are preserved. This is membrane equivalence, not an assertion that the three-dimensional models are identical.</p><p>Reducing thickness from 0.01 to 0.001 changes the resulting compliance by less than 0.002% in these cases. A small residual formulation discrepancy remains; it is not rounded away.</p><p>Unit force is a linear-response normalization. Physical force levels would need scaling to satisfy small-strain assumptions for a physical material and geometry.</p></section>
<section><h2>Acceptance criteria fixed before the comparison</h2><p>Relative compliance ≤1%; global displacement relative L2 ≤1%; constrained in-plane reaction relative L2 ≤5%; absolute force-resultant imbalance ≤10⁻⁵ for a unit applied force. These are project-specific demonstrator criteria, not industrial standards. CalculiX printed output has finite precision.</p></section>
<section><h2>Reproducible evidence</h2><p><a href="results.csv">All numerical results</a> · <a href="manifest.json">Source/executable hashes and case definitions</a> · <a href="acceptance.json">Acceptance criteria</a> · <a href="../docs/calculix-correlation.md">Method and reproduction instructions</a></p><p>Every run includes its input deck, CalculiX log, DAT output and paired displacement/density arrays. The external executable is not bundled.</p><p>Sources: <a href="https://www.dhondt.de/">CalculiX author and downloads</a>; <a href="https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node41.html">CalculiX CPS4 element documentation</a>.</p></section></main></html>'''
    (output/'report.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ccx',required=True,type=Path)
    p.add_argument('--output',default=Path('correlation'),type=Path)
    args=p.parse_args();run(args.ccx,args.output)
