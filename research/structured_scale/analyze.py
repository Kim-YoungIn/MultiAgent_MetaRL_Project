"""Compute paired block inference and descriptive curves for structured-scale v1."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METHODS=('meta_adapt','pooled_adapt','meta_frozen','scratch','no_state','no_long_history')
SETTINGS=((3,1),(3,3),(6,5))
FAMILIES=('resource','sequence','reciprocity','mixed')
BUDGETS=(0,4,8,16)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    out=p.parse_args().output
    manifest=json.loads((out/'manifest.json').read_text())
    values=[];rewards=[];changes=[];probe_changes=[]
    for i in range(manifest['blocks']):
        path=out/f'block_{i:03d}'
        assert (path/'complete.json').exists()
        with np.load(path/'results.npz') as data:
            assert data['values'].shape==(3,24,6,4)
            assert data['rewards'].shape==(3,24,6,24,2)
            values.append(data['values']);rewards.append(data['rewards'])
            changes.append(data['changes']);probe_changes.append(data['probe_changes'])
    values=np.array(values);rewards=np.array(rewards);changes=np.array(changes)
    difference=(values[:,2,:,0,2]-values[:,2,:,1,2]).mean(axis=1)
    rng=np.random.default_rng(248291)
    draws=rng.integers(0,len(difference),size=(10000,len(difference)))
    low,high=np.quantile(difference[draws].mean(axis=1),[.025,.975])
    summary=dict(primary=dict(contrast='meta_adapt minus pooled_adapt, n=6 L=5 k=8',
                              estimate=float(difference.mean()),interval_95=[float(low),float(high)]),
                 blocks=manifest['blocks'],test_games_per_setting_per_block=24,
                 seconds=sum(json.loads((out/f'block_{i:03d}'/'complete.json').read_text())['seconds'] for i in range(manifest['blocks'])),
                 probe_opponent_q_change=float(np.mean(probe_changes)),
                 deployment_opponent_q_change=float(np.mean(changes)))
    (out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['n','depth','family','method','k','expected_reward'])
        for s,(n,depth) in enumerate(SETTINGS):
            for family in ('all',*FAMILIES):
                mask=np.ones(24,dtype=bool) if family=='all' else np.arange(24)%4==FAMILIES.index(family)
                for m,method in enumerate(METHODS):
                    for k,budget in enumerate(BUDGETS):
                        w.writerow([n,depth,family,method,budget,values[:,s,mask,m,k].mean()])
    with (out/'deployment.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['n','depth','family','method','return_24','first_8_mean','last_8_mean','opponent_return_24','opponent_q_absolute_change'])
        for s,(n,depth) in enumerate(SETTINGS):
            for family in ('all',*FAMILIES):
                mask=np.ones(24,dtype=bool) if family=='all' else np.arange(24)%4==FAMILIES.index(family)
                for m,method in enumerate(METHODS):
                    d=rewards[:,s,mask,m]
                    w.writerow([n,depth,family,method,d[:,:,:,0].sum(axis=2).mean(),
                                d[:,:,:8,0].mean(),d[:,:,16:,0].mean(),d[:,:,:,1].sum(axis=2).mean(),
                                changes[:,s,mask,m].mean()])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True)
        for s,(n,depth) in enumerate(SETTINGS):
            for m,c in [(0,'#225999'),(1,'#ba6539'),(2,'#6c9b75'),(3,'#666666')]:
                axes[s].plot(BUDGETS,values[:,s,:,m].mean(axis=(0,1)),marker='o',color=c,label=METHODS[m])
            axes[s].set_title(f'{n} actions, depth {depth}')
            axes[s].set_xlabel('Probe interactions')
        axes[0].set_ylabel('Expected one-step reward')
        axes[-1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'curves.png',dpi=160)
    except ImportError:
        print('matplotlib unavailable')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
