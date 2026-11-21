"""Tabulate per-master-seed effects, game-family curves and measured run costs."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from .verify import REGIMES


def write(path, header, rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summarize(root):
    cfg=json.loads((root/'manifest.json').read_text())['config']
    grouped=defaultdict(list)
    families=defaultdict(list)
    runtime=[]
    for block in range(cfg['blocks']):
        path=root/f'block_{block:03d}'
        data=json.loads(path.with_suffix('.json').read_text())
        runtime.append([data['seconds'][key] for key in ('prior','teacher','evaluation','total')])
        with np.load(path.with_suffix('.npz')) as raw:
            for regime in REGIMES:
                arm = raw[regime+'_assessment']
                for j,method in enumerate(raw['methods']):
                    grouped[data['master_seed'],regime,str(method)].append(data[regime]['auc'][method])
                    for family in np.unique(raw['families']):
                        chosen = arm[j][:,raw['families']==family]
                        families[regime,str(method),str(family)].append(chosen.mean(axis=(1,2)))
    out=root/'analysis'
    master_rows=[]
    for (master,regime,method),values in grouped.items():
        per_block=np.asarray(values)
        learned=np.asarray(grouped[master,regime,'learned'])
        master_rows.append((master,regime,method,len(values),per_block.mean(),
                            per_block.std(ddof=1) if len(values)>1 else 0.,(learned-per_block).mean()))
    write(out/'master_seeds.csv',['master_seed','regime','method','blocks','mean_auc','block_sd','learned_minus_method'],master_rows)
    family_rows=[]
    for (regime,method,family),values in families.items():
        for k,value in zip(cfg['ks'],np.mean(values,axis=0),strict=True):
            family_rows.append((regime,method,family,k,value))
    write(out/'family_curves.csv',['regime','method','family','k','mean'],family_rows)
    write(out/'runtime_by_block.csv',['block','prior_seconds','teacher_seconds','eval_seconds','total_seconds'],
          [(i,*row) for i,row in enumerate(runtime)])
    return dict(first_block_seconds=runtime[0][-1],median_parallel_block_seconds=float(np.median(runtime[1:],axis=0)[-1]),
                total_block_elapsed_seconds=float(np.sum(runtime,axis=0)[-1]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    result=summarize(args.output)
    (args.output/'analysis'/'runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
