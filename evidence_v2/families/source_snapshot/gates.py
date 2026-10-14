"""Fail closed when persisted benchmark evidence is absent or unacceptable."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def verify_benchmark(output):
    required=('test_results.csv','manifest.json','dataset.npz','ridge_model.npz','report.html')
    missing=[name for name in required if not (output/name).is_file()]
    if missing:
        raise ValueError('Missing benchmark evidence: '+', '.join(missing))
    meta=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    with (output/'test_results.csv').open(encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    split=meta['split']
    ids=[set(split[k]) for k in ('train','validation','test')]
    if any(not s for s in ids) or ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:
        raise ValueError('Invalid or overlapping split')
    if set.union(*ids)!=set(range(len(meta['cases']))):
        raise ValueError('Split does not cover the cases')
    groups=[set(meta['group_ids'][i] for i in s) for s in ids]
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError('Symmetry-group leakage')
    if len(rows)!=len(ids[2]) or {int(r['case_id']) for r in rows}!=ids[2]:
        raise ValueError('Missing or duplicated test cases')
    for r in rows:
        numbers=[float(r[k]) for k in ('hybrid_compliance_ratio','volume_error','reference_compliance','hybrid_equilibrium_error')]
        if not np.all(np.isfinite(numbers)) or numbers[0]<=0 or numbers[0]>1.05 or abs(numbers[1])>=1e-6 or numbers[2]<=0:
            raise ValueError('Unacceptable numerical result for case '+r['case_id'])
        if numbers[3]<0 or numbers[3]>=1e-7:
            raise ValueError('Unacceptable equilibrium residual')
        if any(r[k]!='True' for k in ('quality_gate_pass','reference_converged','hybrid_converged')):
            raise ValueError('Failed or non-converged case '+r['case_id'])
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
    try:
        print('Verified benchmark cases:',verify_benchmark(args.output))
    except (ValueError,KeyError,TypeError,OSError) as error:
        p.exit(2,str(error)+'\n')
