"""Held-out geometry benchmark with repeated timing and actual fallback costs."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from .solver import Case,Model,optimize
from .learning import RidgeSurrogate


SETTINGS={'max_iter':400,'tolerance':0.02,'min_iter':15}


def define_cases():
    cases=[];split={'train':[],'validation':[],'test':[]};groups=[]
    definitions={'train':((1.5,2.0),(0.35,0.5,0.65),8),
                 'validation':((1.75,),(0.4,0.6),6),'test':((2.5,),(0.35,0.5,0.65),8)}
    group_offset=0
    for part,(lengths,volumes,count) in definitions.items():
        for length in lengths:
            for volume in volumes:
                for i,y in enumerate(np.linspace(0.15,0.85,count)):
                    split[part].append(len(cases))
                    cases.append(Case(length=length,volume=volume,load_y=float(y),load_span=0.2))
                    groups.append(group_offset+min(i,count-1-i))
                group_offset+=count//2
    return cases,split,groups


def features(early,case):
    # Fixed physical scaling, not normalization fitted using evaluation data.
    shape=(case.ny,case.nx)
    return np.stack([early.reshape(shape),*[np.full(shape,value) for value in
                     (case.length/3,case.volume,case.load_y,case.load_span)]])


def acceptable(result,reference,case):
    values=(result['compliance'],result['volume'],result['equilibrium_error'],reference['compliance'])
    return bool(np.all(np.isfinite(values)) and reference['converged'] and result['converged']
                and 0<result['compliance']<=1.05*reference['compliance']
                and abs(result['volume']-case.volume)<1e-6 and result['equilibrium_error']<1e-7)


def operationally_acceptable(result,case,ceiling):
    """Reference-free runtime gate; reference ratios are evaluation-only."""
    values=(result['compliance'],result['volume'],result['equilibrium_error'],ceiling)
    return bool(np.all(np.isfinite(values)) and result['converged']
                and 0<result['compliance']<=ceiling*(1+1e-8)
                and abs(result['volume']-case.volume)<1e-6 and result['equilibrium_error']<1e-7)


def evaluate_method(case,reference,predictor=None,early_only=False):
    # Time includes fresh model/filter setup, early updates, inference, checks,
    # refinement, and an actual full uniform-start solve on a failed gate.
    start=perf_counter();model=Model(case,backend='sparse');solves=0
    if predictor is None and not early_only:
        result=optimize(model,**SETTINGS)
        solves=result['fe_solves'];attempt=result
        ceiling=result['history'][0]['compliance']
    else:
        early=optimize(model,max_iter=5,tolerance=0.02,min_iter=15)
        solves+=early['fe_solves']
        ceiling=early['compliance']
        if early_only:
            result=early
        else:
            predicted=model.project_volume(predictor(features(early['x'],case)[None])[0])
            direct=model.analyze(predicted);solves+=1
            result=optimize(model,initial=predicted,**SETTINGS);solves+=result['fe_solves']
        attempt=result
    gate=acceptable(attempt,reference,case)
    operational_gate=operationally_acceptable(attempt,case,ceiling)
    fallback=not operational_gate
    if fallback:
        result=optimize(model,**SETTINGS);solves+=result['fe_solves']
    return {'seconds':perf_counter()-start,'fe_solves':solves,'attempt_gate_pass':gate,'fallback':fallback,
            'operational_gate_pass':operational_gate,
            'attempt_compliance_ratio':attempt['compliance']/reference['compliance'],
            'final_compliance_ratio':result['compliance']/reference['compliance'],
            'final_gate_pass':acceptable(result,reference,case)}


def run(output,epochs=80,seeds=(3017,4017,5017),repeats=3):
    if epochs<1 or repeats<1 or not seeds:raise ValueError('Positive epochs, repeats and at least one seed required')
    from .neural import UNetSurrogate
    import torch
    output.mkdir(parents=True,exist_ok=True)
    cases,split,groups=define_cases()
    (output/'experiment_contract.json').write_text(json.dumps({'cases':[asdict(c) for c in cases],
        'split':split,'groups':groups,'seeds':list(seeds),'repeats':repeats,'epochs':epochs,'stopping':SETTINGS,
        'gate':{'compliance_ratio':1.05,'volume_error':1e-6,'equilibrium_residual':1e-7}},indent=2),encoding='utf-8')
    inputs=[];targets=[];references=[];generation_times=[]
    for i,case in enumerate(cases):
        start=perf_counter();result=optimize(Model(case,backend='sparse'),**SETTINGS)
        generation_times.append(perf_counter()-start);references.append(result)
        inputs.append(features(result['snapshots'][4],case));targets.append(result['x'])
        print(f'Generate {i+1}/{len(cases)} converged={result["converged"]}',flush=True)
    inputs,targets=np.asarray(inputs),np.asarray(targets)
    train=[i for i in split['train'] if references[i]['converged']]
    valid=[i for i in split['validation'] if references[i]['converged']]
    if len(train)<3 or not valid:raise RuntimeError('Insufficient converged labels')
    np.savez_compressed(output/'dataset.npz',features=inputs,targets=targets,groups=groups,
                        **{k+'_ids':v for k,v in split.items()})
    rows=[];models=[]
    for seed in seeds:
        start=perf_counter();flat=inputs.reshape(len(inputs),-1);choices=[]
        for alpha in (0.001,0.01,0.1,1,10):
            reg=RidgeSurrogate(alpha).fit(flat[train],targets[train]);scores=[]
            for i,pred in zip(valid,reg.predict(flat[valid])):
                m=Model(cases[i],backend='sparse')
                scores.append(m.analyze(m.project_volume(pred))['compliance']/references[i]['compliance'])
            choices.append((float(np.mean(scores)),alpha))
        alpha=min(choices)[1];ridge=RidgeSurrogate(alpha).fit(flat[train],targets[train])
        ridge_seconds=perf_counter()-start
        unet=UNetSurrogate().fit(inputs[train],targets[train],inputs[valid],
                                  [cases[i] for i in valid],[references[i] for i in valid],epochs,seed)
        torch.save(unet.model.state_dict(),output/f'unet_{seed}.pt')
        np.savez_compressed(output/f'ridge_{seed}.npz',mean=ridge.mean,scale=ridge.scale,train=ridge.train,
                            target_mean=ridge.target_mean,weights=ridge.weights,alpha=alpha)
        models.append({'seed':seed,'ridge_alpha':alpha,'ridge_training_tuning_seconds':ridge_seconds,
                       'unet_training_tuning_seconds':unet.seconds,'unet_best_epoch':unet.best_epoch,'unet_history':unet.history})
        methods={'oc':None,'early_oc':None,
                 'ridge':lambda x:ridge.predict(x.reshape(len(x),-1)),'unet':unet.predict}
        rng=np.random.default_rng(seed)
        for repeat in range(repeats):
            for i in split['test']:
                # Paired timing, randomized method order to reduce order bias.
                for name in rng.permutation(list(methods)):
                    metrics=evaluate_method(cases[i],references[i],methods[name],name=='early_oc')
                    rows.append({'seed':seed,'repeat':repeat,'case_id':i,'method':name,**metrics})
        print(f'Seed {seed}: completed {len(rows)} evaluations',flush=True)
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    label_cost=sum(generation_times[i] for i in split['train']+split['validation'])
    paired={(r['seed'],r['repeat'],r['case_id']):r for r in rows if r['method']=='oc'}
    summary={}
    for name in ('oc','early_oc','ridge','unet'):
        selected=[r for r in rows if r['method']==name]
        savings=[paired[(r['seed'],r['repeat'],r['case_id'])]['seconds']-r['seconds'] for r in selected]
        training=0 if name in ('oc','early_oc') else float(np.mean([m[name+'_training_tuning_seconds'] for m in models]))
        mean_saving=float(np.mean(savings))
        summary[name]={'evaluations':len(selected),'attempt_gate_passes':sum(r['attempt_gate_pass'] for r in selected),
                       'fallbacks':sum(r['fallback'] for r in selected),'final_gate_passes':sum(r['final_gate_pass'] for r in selected),
                       'median_seconds':float(np.median([r['seconds'] for r in selected])),
                       'mean_seconds':float(np.mean([r['seconds'] for r in selected])),
                       'mean_fe_solves':float(np.mean([r['fe_solves'] for r in selected])),
                       'mean_paired_seconds_saved':mean_saving,'offline_label_seconds':label_cost if name in ('ridge','unet') else 0,
                       'offline_training_tuning_seconds':training,
                       'break_even_cases':(label_cost+training)/mean_saving if name in ('ridge','unet') and mean_saving>0 else None}
    manifest={'cases':[asdict(c) for c in cases],'split':split,'groups':groups,'seeds':list(seeds),'repeats':repeats,
              'epochs':epochs,'stopping':SETTINGS,'models':models,'usable_train_ids':train,'usable_validation_ids':valid,
              'reference_converged':[r['converged'] for r in references],
              'train_validation_label_seconds':label_cost,'test_reference_audit_seconds':sum(generation_times[i] for i in split['test']),
              'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'scope':'Three training/validation aspect ratios; test holds out L/H=2.5. All cases share cantilever supports. No industrial generalization claim.',
              'timing':'CPU one torch thread; paired randomized order; setup, early iterations, inference, checks, refinement and actual fallback included. Offline costs separate. Interpreter/import startup and report serialization excluded.',
              'fallback':'Runtime gate uses convergence, volume, residual and compliance no worse than the early state; no test reference is used in the fallback decision. Reference compliance ratios are evaluation-only and enforced by --strict.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('my_families'))
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[3017,4017,5017]);p.add_argument('--strict',action='store_true')
    a=p.parse_args();rows=run(a.output,a.epochs,a.seeds,a.repeats)
    if a.strict and not all(r['final_gate_pass'] for r in rows):raise SystemExit(2)
