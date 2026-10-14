"""Fixed-physical-filter mesh study and unfiltered binary FEM reanalysis."""
import argparse
import csv
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from .solver import Case,Model,optimize
from .benchmark import density_svg,connected_load_path


def volume_binary(rho,volume):
    """Select floor(V*N) largest physical densities. Stable index tie breaking.

    Volume never exceeds the requested fraction; ties can break symmetry and are
    not resolved using knowledge of the mechanical response.
    """
    rho=np.asarray(rho).ravel()
    if not 0<=volume<=1 or not np.all(np.isfinite(rho)):
        raise ValueError('Invalid volume or density')
    n=int(np.floor(volume*len(rho)+1e-12))
    binary=np.zeros_like(rho,dtype=float)
    binary[np.argsort(-rho,kind='stable')[:n]]=1
    return binary


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    meshes=[(8,4),(16,8),(32,16),(48,24)]
    settings={'max_iter':240,'tolerance':0.01,'min_iter':20}
    config={'meshes':meshes,'physical_radius':0.19,'load_y':0.5,'load_span':0.25,
            'stopping':settings,'thresholds':[0.3,0.5,0.7],
            'binary_gate':{'max_compliance_ratio':1.05,'max_volume':0.5,'requires_connected_path':True,
                           'max_void_modulus_sensitivity':0.01,'requires_converged_parent':True},
            'void_moduli':[1e-6,1e-8], 'note':'Gates fixed before analysis. Not manufacturing qualification.'}
    (output/'study_config.json').write_text(json.dumps(config,indent=2))
    mesh_rows=[];binary_rows=[];frozen=None;cases=[]
    for nx,ny in meshes:
        c=Case(nx=nx,ny=ny,load_y=0.5,load_span=0.25,radius=0.19)
        cases.append(asdict(c));model=Model(c)
        solid=model.analyze_density(np.ones(model.ne))
        optimized=optimize(model,**settings)
        rho=optimized['rho']
        if frozen is None:frozen=rho.reshape(ny,nx).copy()
        factor=nx//meshes[0][0]
        fixed=np.repeat(np.repeat(frozen,factor,axis=0),factor,axis=1).ravel()
        fixed_result=model.analyze_density(fixed)
        row={'nx':nx,'ny':ny,'elements':model.ne,'filter_radius':c.radius,
             'filter_radius_in_elements':c.radius/(c.length/nx),
             'solid_compliance':solid['compliance'],'frozen_compliance':fixed_result['compliance'],
             'optimized_compliance':optimized['compliance'],'volume':optimized['volume'],
             'gray_fraction':float(np.mean((rho>0.1)&(rho<0.9))),
             'iterations':optimized['iterations'],'converged':optimized['converged'],
             'max_final_design_change':optimized['history'][-1]['change'],
             'seconds':optimized['seconds']}
        if mesh_rows:
            for key in ('solid_compliance','frozen_compliance','optimized_compliance'):
                row[key+'_change_from_previous']=row[key]/mesh_rows[-1][key]-1
        else:
            for key in ('solid_compliance','frozen_compliance','optimized_compliance'):
                row[key+'_change_from_previous']=None
        mesh_rows.append(row)
        density_svg(rho,nx,ny,output/f'mesh_{nx}x{ny}_continuous.svg')
        masks={f'threshold_{t:g}':(rho>=t).astype(float) for t in config['thresholds']}
        masks['volume_preserving']=volume_binary(rho,c.volume)
        saved={'design':optimized['x'],'physical_density':rho,'frozen_density':fixed}
        for name,binary in masks.items():
            result=model.analyze_density(binary)
            soft=Model(replace(c,emin=1e-8)).analyze_density(binary)
            connected=connected_load_path(binary,model)
            ratio=result['compliance']/optimized['compliance']
            sensitivity=abs(soft['compliance']/result['compliance']-1)
            passed=bool(optimized['converged'] and connected and ratio<=1.05
                        and result['volume']<=c.volume+1e-12 and sensitivity<0.01)
            binary_rows.append({'nx':nx,'ny':ny,'method':name,'volume':result['volume'],
                                'compliance':result['compliance'],'continuous_compliance':optimized['compliance'],
                                'compliance_ratio':ratio,'load_path_connected':connected,
                                'void_modulus_sensitivity':sensitivity,'gate_pass':passed})
            saved[name]=binary
            density_svg(binary,nx,ny,output/f'mesh_{nx}x{ny}_{name}.svg')
        np.savez_compressed(output/f'mesh_{nx}x{ny}.npz',**saved)
        (output/f'mesh_{nx}x{ny}_history.json').write_text(json.dumps(optimized['history'],indent=2))
        print(json.dumps(row),flush=True)
    for name,rows in [('mesh_results',mesh_rows),('binary_results',binary_rows)]:
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta={'python':platform.python_version(),'numpy':np.__version__,'cases':cases,
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
          'interpretation':'Frozen field tests FEM refinement; reoptimization also changes the design space. Four meshes do not prove mesh independence. The 8x4 filter radius is below one element width and acts as identity.'}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
    report(output,mesh_rows,binary_rows)
    print(json.dumps({'binary_passes':sum(r['gate_pass'] for r in binary_rows),'binary_cases':len(binary_rows)},indent=2))


def report(output,mesh_rows,binary_rows):
    table=''.join(f'<tr><td>{r["nx"]}×{r["ny"]}</td><td>{r["solid_compliance"]:.3f}</td>'
                  f'<td>{r["frozen_compliance"]:.3f}</td><td>{r["optimized_compliance"]:.3f}</td>'
                  f'<td>{r["gray_fraction"]:.1%}</td><td>{r["converged"]}</td></tr>' for r in mesh_rows)
    sections=[]
    for mesh in mesh_rows:
        nx,ny=mesh['nx'],mesh['ny']
        pictures=f'<figure><img src="mesh_{nx}x{ny}_continuous.svg"><figcaption>Continuous density</figcaption></figure>'
        rows=[]
        for r in binary_rows:
            if r['nx']!=nx:continue
            pictures+=f'<figure><img src="mesh_{nx}x{ny}_{r["method"]}.svg"><figcaption>{r["method"]}</figcaption></figure>'
            rows.append(f'<tr><td>{r["method"]}</td><td>{r["volume"]:.1%}</td><td>{r["compliance_ratio"]:.3f}</td>'
                        f'<td>{r["load_path_connected"]}</td><td>{r["void_modulus_sensitivity"]:.2%}</td><td>{"PASS" if r["gate_pass"] else "FAIL"}</td></tr>')
        sections.append(f'<section><h2>{nx}×{ny}: binary reanalysis</h2><div class="fields">{pictures}</div>'
                        '<div class="scroll"><table><tr><th>Method</th><th>Material</th><th>C / Ccontinuous</th><th>Load path</th><th>Emin sensitivity</th><th>Gate</th></tr>'
                        +''.join(rows)+'</table></div></section>')
    last=mesh_rows[-1]
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mesh and binary verification</title><style>
body{font:16px/1.6 system-ui;color:#163044;background:#eef2f4;margin:0}main{max-width:1160px;margin:auto;padding:30px 24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}.fields{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}figure{margin:0}img{width:100%;border:1px solid #ccd6da}figcaption{font-size:12px}.scroll{overflow:auto}aside{background:#fff2db;padding:18px;border-left:4px solid #b87a28}a{color:#00647a}@media(max-width:750px){.fields{grid-template-columns:repeat(2,1fr)}}
</style><main><p>CAE VERIFICATION · MESH & BINARY DESIGN</p><h1>Does the result survive mesh refinement and thresholding?</h1>
<aside>New mesh-study load case: uniform traction over a fixed right-edge patch of height 0.25H, centered at 0.5H. Unit resultant; physical filter radius 0.19. Previous point-load benchmark results are not directly comparable.</aside>
<section><h2>Three separate questions</h2><p><b>Solid:</b> reference FEM refinement. <b>Frozen:</b> the same coarse physical-density field, exactly subdivided without smoothing. <b>Reoptimized:</b> a new OC optimization on each mesh, with constant physical radius. Only the first two isolate FEM discretization effects.</p><table><tr><th>Mesh</th><th>Solid C</th><th>Frozen C</th><th>Reoptimized C</th><th>Gray fraction</th><th>OC converged</th></tr>'''+table+'''</table>'''
    html+=f'<p>Last refinement: frozen-field compliance changes by {last["frozen_compliance_change_from_previous"]:.2%}; reoptimized compliance changes by {last["optimized_compliance_change_from_previous"]:.2%}. Four levels and an update-based stopping rule do not prove mesh independence or global optimality. At 8×4 the radius is smaller than one element width: the filter is effectively inactive. Keep this coarse case as a diagnostic, not evidence of a resolved physical length scale.</p></section>'
    html+=''.join(sections)
    html+='''<section><h2>Binary acceptance contract</h2><p>Reanalyze physical 0/1 densities without density filtering. Require material fraction ≤50%, compliance ≤105% of the continuous result, a four-neighbour support-to-load path, converged parent optimization and &lt;1% compliance sensitivity when void modulus changes from 10⁻⁶ to 10⁻⁸. These are demonstrator criteria, not manufacturing qualification.</p><p>Volume-preserving selection keeps the largest densities; equal values are resolved by stable element index. This can break symmetry. Low compliance obtained by adding material is not accepted. Disconnected shapes may still return finite numbers because void is represented by weak material.</p><p>Evidence: <a href="mesh_results.csv">mesh results</a> · <a href="binary_results.csv">binary results</a> · <a href="study_config.json">configuration</a> · <a href="manifest.json">provenance</a>.</p></section></main></html>'''
    (output/'report.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('mesh_study'))
    args=p.parse_args();run(args.output)
