"""Post-evaluation descriptive summaries and figure from the held-out blocks."""
import argparse
import csv
from pathlib import Path

import numpy as np
from .study import BUDGETS, METHODS, histories


def summarize(root, output):
    curves, loss, policy_gap, deployed = [], [], [], []
    chosen_differ = []
    for path in sorted(root.glob('block_*')):
        with np.load(path/'results.npz') as results, np.load(path/'test.npz') as data:
            curves.append(results['value'].mean(0))
            chosen_differ.append((results['decision'][:,:,0] != results['decision'][:,:,1]).mean(0))
            deployed.append(results['deploy_rewards'][...,0].sum(-1).mean(0))
            policy_gap.append((.5*abs(data['opp_p'][:,1]-data['opp_p'][:,0]).sum(-1)).mean(0))
            # Conditional mean-squared prediction error, averaging all three candidate actions.
            w = results['w']
            errors = []
            truth = data['truth'].reshape(-1,33,3)
            for k,(_, z, offset) in enumerate(histories(data)):
                if k in BUDGETS:
                    pred = z @ w.T + offset[...,None]
                    sq = (pred-truth[:,k,:,None])**2
                    errors.append(sq.reshape(-1,2,3,2).mean((0,2)))
            loss.append(errors)
    curves=np.stack(curves)
    deployed=np.stack(deployed)
    loss=np.array(loss)
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(421973)
    bootstrap=rng.integers(len(curves),size=(10000,len(curves)))
    rows=[]
    def record(condition, budget, metric, values):
        lo,hi=np.quantile(values[bootstrap].mean(1),[.025,.975])
        rows.append([condition,budget,metric,values.mean(),lo,hi])
    for e,name in enumerate(('fixed','learning')):
        for j,k in enumerate(BUDGETS):
            for m,method in enumerate(METHODS):
                record(name,k,method,curves[:,e,m,j])
            record(name,k,'retaining_meta_updates',curves[:,e,0,j]-curves[:,e,2,j])
            record(name,k,'retaining_pool_updates',curves[:,e,1,j]-curves[:,e,3,j])
            record(name,k,'meta_minus_pool',curves[:,e,0,j]-curves[:,e,1,j])
            record(name,k,'meta_minus_pool_prediction_mse',loss[:,j,e,0]-loss[:,j,e,1])
            record(name,k,'action_disagreement',np.array(chosen_differ)[:,e,j])
        record(name,24,'deployment_difference',deployed[:,e,0]-deployed[:,e,1])
    with (output/'descriptive.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['opponent','budget','metric','mean','descriptive_95_low','descriptive_95_high'])
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    for m,color,linestyle in [(0,'#14696a','-'),(1,'#ac453c','--'),(2,'#14696a',':'),(3,'#ac453c',':')]:
        label=('Meta, updating','Pooled, updating','Meta, frozen','Pooled, frozen')[m]
        left.plot(BUDGETS,curves[:,1,m].mean(0),marker='o',ms=3,label=label,c=color,ls=linestyle)
    left.set_title('Against a learning opponent')
    left.set_ylabel('Expected payoff of selected action')
    for e,name,color in [(0,'Fixed opponent','#535577'),(1,'Learning opponent','#197672')]:
        d=curves[:,e,0]-curves[:,e,1]
        lo,hi=np.quantile(d[bootstrap].mean(1),[.025,.975],axis=0)
        right.plot(BUDGETS,d.mean(0),marker='o',ms=3,c=color,label=name)
        right.fill_between(BUDGETS,lo,hi,color=color,alpha=.12)
    right.axhline(0,c='#393939',lw=.8)
    right.set_title('Meta-training advantage')
    right.set_ylabel('Meta minus pooled expected payoff')
    for ax in (left,right):
        ax.set_xlabel('Interactions with the new opponent')
        ax.set_xticks(BUDGETS)
        ax.legend(frameon=False,fontsize=8)
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'adaptation.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    summarize(args.run,args.output)
