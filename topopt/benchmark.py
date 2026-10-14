"""Run: python -m topopt.benchmark --output results

Small end-to-end experiment, including negative results. Test cases are never
used for hyperparameter selection. Timing is single-run, CPU, reference backend.
"""
import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from .solver import Case, Model, optimize
from .learning import RidgeSurrogate, split_cases


def compact(result):
    return {k: result[k] for k in ('compliance','volume','equilibrium_error',
                                  'converged','iterations','fe_solves','seconds')}


def density_svg(rho, nx, ny, path):
    # Exact density cells, no smoothing or interpolated appearance.
    cells=[]
    for j,row in enumerate(np.asarray(rho).reshape(ny,nx)[::-1]):
        for i,value in enumerate(row):
            gray=round(255*(1-float(value)))
            cells.append(f'<rect x="{i*12}" y="{j*12}" width="12" height="12" fill="rgb({gray},{gray},{gray})"/>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {nx*12} {ny*12}">' + ''.join(cells)+'</svg>', encoding='utf-8')


def connected_load_path(rho, model, threshold=0.5):
    """4-neighbour pixel diagnostic only, not a mechanical safety certificate."""
    mask=rho.reshape(model.case.ny,model.case.nx)>=threshold
    seen={(j,0) for j in range(mask.shape[0]) if mask[j,0]}
    todo=list(seen)
    while todo:
        j,i=todo.pop()
        for p,q in ((j-1,i),(j+1,i),(j,i-1),(j,i+1)):
            if 0<=p<mask.shape[0] and 0<=q<mask.shape[1] and mask[p,q] and (p,q) not in seen:
                seen.add((p,q)); todo.append((p,q))
    loaded=np.flatnonzero(model.f[1::2])
    for node in loaded:
        j=int(node)//(model.case.nx+1)
        adjacent=[row for row in (j-1,j) if 0<=row<model.case.ny]
        if not any((row,model.case.nx-1) in seen for row in adjacent):
            return False
    return True


def run(output, n_cases=24, backend='sparse'):
    output.mkdir(parents=True,exist_ok=True)
    # A symmetric cantilever at y and 1-y has an equivalent mirrored solution.
    # Keep both trajectories together, even though no augmentation is used.
    groups=np.minimum(np.arange(n_cases), np.arange(n_cases)[::-1])
    split=split_cases(n_cases, groups=groups)
    # Uniformly spaced right-edge load positions, fixed geometry/volume/material.
    # This is interpolation on ONE problem family, not out-of-distribution proof.
    cases=[Case(load_y=float(y)) for y in np.linspace(0.08,0.92,n_cases)]
    settings={'max_iter':180,'tolerance':0.02,'min_iter':15}
    inputs=[]; targets=[]; references=[]; generation_start=perf_counter()
    for i,case in enumerate(cases):
        result=optimize(Model(case,backend=backend),**settings)
        inputs.append(result['snapshots'][4])
        targets.append(result['x'])
        references.append(result)
        print(f'Case {i+1}/{n_cases}: C={result["compliance"]:.3f}, iterations={result["iterations"]}, converged={result["converged"]}',flush=True)
    generation_seconds=perf_counter()-generation_start
    inputs,targets=np.asarray(inputs),np.asarray(targets)
    train,valid,test=(split[k] for k in ('train','validation','test'))
    usable_train=np.array([i for i in train if references[i]['converged']],dtype=int)
    usable_valid=np.array([i for i in valid if references[i]['converged']],dtype=int)
    if len(usable_train)<3 or len(usable_valid)<1:
        raise RuntimeError('Not enough converged training/validation labels; investigate solver stopping first')
    tuning=[]; training_start=perf_counter()
    for alpha in (0.001,0.01,0.1,1.0,10.0):
        reg=RidgeSurrogate(alpha).fit(inputs[usable_train],targets[usable_train])
        ratios=[]
        for i,pred in zip(usable_valid,reg.predict(inputs[usable_valid])):
            m=Model(cases[i],backend=backend)
            ratios.append(m.analyze(m.project_volume(pred))['compliance']/references[i]['compliance'])
        tuning.append({'alpha':alpha,'validation_mean_compliance_ratio':float(np.mean(ratios))})
    chosen=min(tuning,key=lambda item:item['validation_mean_compliance_ratio'])['alpha']
    reg=RidgeSurrogate(chosen).fit(inputs[usable_train],targets[usable_train])
    training_seconds=perf_counter()-training_start
    np.savez_compressed(output/'dataset.npz',early_design=inputs,final_design=targets,
                        train_ids=train,validation_ids=valid,test_ids=test,
                        group_ids=groups, converged=np.array([r['converged'] for r in references]))
    np.savez_compressed(output/'ridge_model.npz',mean=reg.mean,scale=reg.scale,
                        train=reg.train,target_mean=reg.target_mean,weights=reg.weights,alpha=chosen)
    rows=[]
    for i in test:
        start=perf_counter(); model=Model(cases[i],backend=backend); setup_seconds=perf_counter()-start
        # Recompute test early iterations; do not reuse the label-generation time.
        early=optimize(model,max_iter=5,tolerance=settings['tolerance'],min_iter=15)
        start=perf_counter()
        predicted=model.project_volume(reg.predict(early['x'])[0])
        direct=model.analyze(predicted)
        prediction_and_check_seconds=perf_counter()-start
        warm=optimize(model,initial=predicted,**settings)
        reference=references[i]
        ratio=warm['compliance']/reference['compliance']
        # Conservative demo gate; thresholds are project choices, not standards.
        gate=bool(reference['converged'] and warm['converged'] and ratio<=1.05
                  and abs(warm['volume']-cases[i].volume)<1e-6
                  and warm['equilibrium_error']<1e-7)
        baseline_seconds=setup_seconds+reference['seconds']
        hybrid_seconds=setup_seconds+early['seconds']+prediction_and_check_seconds+warm['seconds']
        row={'case_id':int(i),'load_y':cases[i].load_y,
             'reference_compliance':reference['compliance'],
             'early_compliance_ratio':early['compliance']/reference['compliance'],
             'direct_compliance_ratio':direct['compliance']/reference['compliance'],
             'hybrid_compliance_ratio':ratio,'reference_converged':reference['converged'],
             'hybrid_equilibrium_error':warm['equilibrium_error'],
             'hybrid_converged':warm['converged'],'quality_gate_pass':gate,
             'reference_fe_solves':reference['fe_solves'],
             'hybrid_fe_solves':early['fe_solves']+1+warm['fe_solves'],
             'reference_seconds':baseline_seconds,'hybrid_seconds':hybrid_seconds,
             'speedup':baseline_seconds/hybrid_seconds,
             'volume_error':warm['volume']-cases[i].volume,
             'thresholded_path_exists':connected_load_path(warm['rho'],model),
             'gray_fraction':float(np.mean((warm['rho']>0.1)&(warm['rho']<0.9)))}
        rows.append(row)
        for name,rho in [('early',early['rho']),('prediction',direct['rho']),('hybrid',warm['rho']),('reference',reference['rho'])]:
            density_svg(rho,cases[i].nx,cases[i].ny,output/f'case_{i}_{name}.svg')
        (output/f'case_{i}_history.json').write_text(json.dumps({'reference':reference['history'],'hybrid':warm['history']},indent=2))
    with (output/'test_results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    source_hashes={str(p.relative_to(Path(__file__).parent)):sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))}
    metadata={'experiment':'small-family load-position holdout baseline; not a production qualification',
              'seed':3017,'python':platform.python_version(),'numpy':np.__version__,
              'platform':platform.platform(),'solver':backend+' Q4 plane-stress',
              'cases':[asdict(c) for c in cases], 'split':{k:v.tolist() for k,v in split.items()},
              'group_ids':groups.tolist(), 'group_definition':'mirror pair y/H and 1-y/H',
              'usable_train_ids':usable_train.tolist(),'usable_validation_ids':usable_valid.tolist(),
              'stopping':settings,'early_updates':5,'tuning':tuning,'chosen_alpha':chosen,
              'generation_seconds':generation_seconds,'training_and_tuning_seconds':training_seconds,
              'reference_results':[compact(r) for r in references], 'source_sha256':source_hashes,
              'timing_note':'Single run, no parallelism. Setup included equally. Offline dataset and training reported separately. No fallback is executed or included.'}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report(output,rows,metadata)
    print(json.dumps({'test_cases':len(rows),'quality_gate_passes':sum(r['quality_gate_pass'] for r in rows),
                      'median_speedup':float(np.median([r['speedup'] for r in rows])),
                      'max_compliance_ratio':max(r['hybrid_compliance_ratio'] for r in rows)},indent=2))
    return rows


def report(output,rows,metadata):
    cards=[]
    for r in rows:
        i=r['case_id']
        images=''.join(f'<figure><img src="case_{i}_{stage}.svg" alt="{stage} density field"><figcaption>{label}</figcaption></figure>'
                       for stage,label in [('early','After 5 OC updates'),('prediction','Ridge + volume projection'),('hybrid','Ridge + OC refinement'),('reference','Uniform-start OC reference')])
        cards.append(f'<section><h2>Held-out case {i} · load y/H = {r["load_y"]:.3f}</h2><div class="fields">{images}</div>'
                     f'<p>Direct C/Cref: <b>{r["direct_compliance_ratio"]:.3f}</b> · Refined C/Cref: <b>{r["hybrid_compliance_ratio"]:.3f}</b> · '
                     f'Wall-time speedup: <b>{r["speedup"]:.2f}×</b> · Quality gate: <b>{"PASS" if r["quality_gate_pass"] else "FAIL"}</b></p>'
                     f'<p>FE solves: {r["reference_fe_solves"]} reference / {r["hybrid_fe_solves"]} hybrid. '
                     f'Volume error: {r["volume_error"]:.2e}. Gray fraction (0.1&lt;ρ&lt;0.9): {r["gray_fraction"]:.1%}. '
                     f'Connected load path at ρ≥0.5: {r["thresholded_path_exists"]}.</p></section>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Topology optimization — verification-first benchmark</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#193042;background:#eef2f4;margin:0}main{max-width:1150px;margin:auto;padding:45px 25px}h1{font-size:38px;line-height:1.15}h2{font-size:21px}section{background:white;padding:24px;margin:22px 0;border:1px solid #d7e0e4;border-radius:12px}.fields{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}figure{margin:0}img{width:100%;border:1px solid #ccd6dc}figcaption{font-size:13px}aside{border-left:4px solid #cc8b30;padding:12px 20px;background:#fff6e6}a{color:#006785}@media(max-width:700px){.fields{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}}
</style><main><p>CAE / DATA-DRIVEN DESIGN · RESEARCH DEMONSTRATOR</p><h1>Can an ML warm start reduce topology-optimization cost?</h1>
<p>Plane-stress Q4 FEM · SIMP · density filter · optimality criteria · supervised ridge baseline</p>
<aside><b>Scope:</b> small 16×8 mesh, one cantilever family, 24 load positions by default, fixed volume and material. This is not an industrial validation or a reproduction of the 2019 U-Net results. A speedup below 1 means slower. Failed cases are retained.</aside>
<section><h2>Experiment contract</h2><p>Entire FEM cases and their mirror-equivalent partners are grouped before splitting. Ridge regularization is selected on validation compliance only. Five early updates, inference, volume projection, FEM checking and refinement are all included in hybrid timing. The reference is a converged local OC solution, not a certified global optimum.</p>
<p>Quality gate: both runs converged, compliance ≤105% of reference, volume error &lt;10⁻⁶, free-DOF residual &lt;10⁻⁷. Pixel connectivity is a separate diagnostic. Physical density fields are displayed without thresholding.</p>
<p>Raw evidence: <a href="test_results.csv">test_results.csv</a> · <a href="manifest.json">manifest.json</a> · <a href="dataset.npz">dataset.npz</a></p></section>'''
    test_groups=len(set(metadata['group_ids'][r['case_id']] for r in rows))
    html+=f'<section><h2>Measured outcome</h2><p><b>{sum(r["quality_gate_pass"] for r in rows)}/{len(rows)} quality gates passed</b>, from {test_groups} independent symmetry groups. Worst direct prediction: {max(r["direct_compliance_ratio"] for r in rows):.3f}× reference compliance. Refinement is essential for the weakest predictions.</p><p>Geometry: L/H=2, left edge clamped, downward unit load on the right edge. Volume fraction=0.5. Black=solid (ρ=1), white=void (ρ=0), gray=intermediate density.</p></section>'
    html+=f'<p>Offline dataset generation: {metadata["generation_seconds"]:.1f} s. Training and validation sweep: {metadata["training_and_tuning_seconds"]:.2f} s. Selected ridge α={metadata["chosen_alpha"]}.</p>'
    html+=''.join(cards)
    html+='''<section><h2>What this does not establish</h2><p>No new geometry family, 3D, nonlinear material, contact, fatigue, manufacturing constraint, binary reanalysis or commercial-solver correlation is included. Timing is single-run CPU on a reference FEM backend. Dataset generation/training is offline and its amortization must be considered before deployment. A failed quality gate requires investigation or a full reference solve; fallback timing has not been measured.</p></section></main></html>'''
    (output/'report.html').write_text(html,encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--cases',type=int,default=24)
    parser.add_argument('--backend',choices=('dense','sparse'),default='sparse')
    parser.add_argument('--strict',action='store_true',help='Return exit code 2 if any numerical gate fails')
    args=parser.parse_args(argv)
    rows=run(args.output,args.cases,args.backend)
    if args.strict and not all(r['quality_gate_pass'] for r in rows):
        raise SystemExit(2)


if __name__=='__main__':
    main()
