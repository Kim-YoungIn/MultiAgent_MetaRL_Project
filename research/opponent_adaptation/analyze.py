"""Paired block-level contrasts for focal meta-training and counterpart learning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .study import BUDGETS, METHODS, FAMILIES
from .data import CONDITIONS

ENDPOINTS = ('meta_minus_pooled_learning', 'opponent_moderation_of_extra_adaptation')


def effects(value):
    k = BUDGETS.index(8)
    absolute = value[:,1,0,k]-value[:,1,1,k]
    extra = (value[:,:,0,k]-value[:,:,2,k])-(value[:,:,1,k]-value[:,:,3,k])
    return np.array([absolute.mean(),(extra[:,1]-extra[:,0]).mean()])


def save_csv(file, rows):
    with file.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def analyze(root):
    count=json.loads((root/'manifest.json').read_text())['config']['blocks']
    block_values,curves,rollouts,opponents=[],[],[],[]
    for j in range(count):
        folder=root/f'block_{j:03d}'
        with np.load(folder/'results.npz') as r:
            val=r['value']
            ret=r['deploy_rewards']
            drift=r['deploy_q_delta']
        with np.load(folder/'test.npz') as d:
            family=d['family']
            policy_distance=.5*abs(d['opp_p'][:,1]-d['opp_p'][:,0]).sum(axis=-1)
            oracle=d['truth'].max(axis=-1)
            different=(d['b'][:,0]!=d['b'][:,1])
            change=d['q_delta']
        block_values.append(effects(val))
        for name in ('all',*FAMILIES):
            mask=np.ones(len(family),dtype=bool) if name=='all' else family==FAMILIES.index(name)
            for e,condition in enumerate(CONDITIONS):
                for m,method in enumerate(METHODS):
                    for t,k in enumerate(BUDGETS):
                        curves.append(dict(block=j,family=name,opponent=condition,method=method,k=k,
                                           value=float(val[mask,e,m,t].mean()),
                                           regret=float((oracle[mask,e,k]-val[mask,e,m,t]).mean())))
                for m,method in enumerate(('meta_adapt','pool_adapt','scratch')):
                    x=ret[mask,e,m]
                    rollouts.append(dict(block=j,family=name,opponent=condition,method=method,
                                         return_24=x[:,:,0].sum(1).mean(),first4=x[:,:4,0].mean(),
                                         last4=x[:,-4:,0].mean(),opponent_return=x[:,:,1].sum(1).mean(),
                                         opponent_q_change=drift[mask,e,m].sum(1).mean()))
            for k in BUDGETS:
                opponents.append(dict(block=j,family=name,k=k,total_variation=policy_distance[mask,k].mean(),
                    divergent_move_fraction=different[mask,:k].mean() if k else 0.,
                    cumulative_q_change=change[mask,1,:k].sum(1).mean()))
    block_values=np.stack(block_values)
    rng=np.random.default_rng(384916)
    idx=rng.integers(count,size=(20000,count))
    bounds=np.quantile(block_values[idx].mean(1),[.0125,.9875],axis=0)
    summary=dict(blocks=count, primary=[dict(name=key,estimate=block_values[:,j].mean(),
                lower=bounds[0,j],upper=bounds[1,j]) for j,key in enumerate(ENDPOINTS)])
    save_csv(root/'curves.csv',curves)
    save_csv(root/'deployment.csv',rollouts)
    save_csv(root/'opponents.csv',opponents)
    save_csv(root/'effects.csv',[dict(block=i,**dict(zip(ENDPOINTS,row))) for i,row in enumerate(block_values)])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    print(json.dumps(analyze(p.parse_args().root),indent=2))
