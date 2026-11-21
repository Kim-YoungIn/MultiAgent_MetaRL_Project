"""Join sealed D-031 rule coverage to matched trial-level adaptation outcomes."""
import argparse
import csv
from collections import defaultdict
import gzip
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from experiment.artifacts import verify_evaluation
from experiment.math_audit import strategy_losses
from experiment.panels import evaluation_entries, training_entries
from experiment.statistics import _rank, _spearman
from mamrl.meta_exploration import decode_rule
from mamrl.rules import compile_rule

METRICS = ('strategy_similarity', 'payoff_similarity')
OUTCOMES = ('meta_minus_reward', 'meta_minus_random', 'meta_gain', 'retained_q_gain')


def tensor(entry):
    return np.asarray(compile_rule(decode_rule(entry['rule'])).values).reshape(2,3,3,3,3)


def payoff_similarity(train, test):
    train_values = np.stack(list(map(tensor, train)))
    eval_values = np.stack(list(map(tensor, test)))
    n = train_values.shape[-1]
    smallest = np.full(len(test), np.inf)
    for perm in permutations(range(n)):
        adjusted = train_values
        for axis in range(2, 6):
            adjusted = np.take(adjusted, perm, axis=axis)
        square = ((eval_values[:, None]-adjusted[None])**2).mean(axis=(2,3,4,5,6))
        smallest = np.minimum(smallest, square.min(axis=1))
    return -np.sqrt(smallest)


def trial_values(folder, method):
    verify_evaluation(folder, method)
    observations = {}
    with gzip.open(folder/f'{method}.trials.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line)
            if r['split']!='fresh_rules' or r['profile'] not in ('steady','sarsa'):
                continue
            key = (r['rule'],r['profile'],r['rep'])
            curve = r['curve']
            budget = [p['k'] for p in curve]
            if budget != [0,3,6,12,24] or key in observations:
                raise ValueError(f'unexpected trial or budget {folder} {key}')
            y=np.array([p['return'] for p in curve])
            auc=np.diff(budget) @ ((y[1:]+y[:-1])/2-y[0])/budget[-1]
            np.testing.assert_allclose(auc,r['auc'],atol=1e-12,rtol=0.)
            gain = np.array([point['G'] for point in curve])
            retained = np.diff(budget) @ ((gain[1:]+gain[:-1])/2)/budget[-1]
            observations[key] = (auc, y[0], r['seed'], retained)
    if len(observations)!=64*2*8:
        raise ValueError(f'incomplete pool in {folder}')
    return observations


def merge(study, audit):
    panel=study/'panels'
    test=evaluation_entries(panel,3)
    rows = []
    cache = {}
    paths = sorted((study/'cells').glob('primary/c*/j*/n3/M.receipt.json'))
    for receipt in paths:
        folder=receipt.parent
        if not (folder/'losses.json').is_file() or not all(
            (folder/f'{method}.receipt.json').is_file() for method in ('P', 'random')
        ):
            continue
        catalog=int(folder.parent.parent.name[1:])
        replica=int(folder.parent.name[1:])
        pred={m:trial_values(folder,m) for m in ('M','P','random')}
        if not pred['M'].keys()==pred['P'].keys()==pred['random'].keys():
            raise ValueError(f'unpaired outcomes {folder}')
        offsets=defaultdict(list)
        for key,(meta,start,seed,retained) in pred['M'].items():
            pooled,p_start,p_seed,_=pred['P'][key]
            random,r_start,r_seed,_=pred['random'][key]
            if seed != p_seed or seed != r_seed:
                raise ValueError(f'unpaired trial seeds in {folder}: {key}')
            np.testing.assert_array_equal([p_start,r_start],[start,start])
            offsets[key[0]].append((meta-pooled,meta-random,meta,retained))
        if catalog not in cache:
            train=training_entries(panel,3,'broad',catalog,32)
            distances=payoff_similarity(train,test)
            loss=json.loads((folder/'losses.json').read_text())
            if audit:
                checked=strategy_losses(train,test)
                np.testing.assert_allclose([checked[t['id']] for t in test],[loss[t['id']] for t in test],atol=1e-12)
            cache[catalog]=(distances,loss)
        distances,loss=cache[catalog]
        for row,d in zip(test,distances,strict=True):
            rule=row['id']
            if rule in offsets:
                rows.append(dict(catalog=catalog,replica=replica,rule=rule,family=row['family'],
                                 strategy_similarity=-loss[rule],payoff_similarity=d,
                                 **dict(zip(OUTCOMES,np.mean(offsets[rule],axis=0)))))
    if not rows:
        raise ValueError('no complete primary M/P/random pairing')
    return rows


def write_csv(out,rows):
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def demonstrate(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Average replicas within catalog, then center ranks within catalog and family.
    groups=defaultdict(list)
    for row in rows:
        groups[row['catalog'],row['family'],row['rule']].append(row)
    facets=defaultdict(list)
    for (c,f,_),g in groups.items():
        facets[c,f].append([g[0]['strategy_similarity'],g[0]['payoff_similarity'],
                             np.mean([r['meta_minus_reward'] for r in g])])
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained',sharey=True)
    colors=plt.get_cmap('tab10')
    families=sorted({key[1] for key in facets})
    for index,ax in enumerate(axes):
        for i,family in enumerate(families):
            x,y=[],[]
            for (_,f),values in facets.items():
                if f==family:
                    data=np.array(values)
                    x.extend(np.array(_rank(data[:,index]))/(len(data)-1))
                    y.extend(data[:,2]-data[:,2].mean())
            ax.scatter(x,y,color=colors(i),alpha=.7,s=21,label=family)
        ax.axhline(0,color='#666666',lw=.7)
        ax.set_xlabel(['Strategy similarity rank within family','Payoff similarity rank within family'][index])
        ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Within-family adaptation AUC difference (M − P)')
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(out,dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--verify-strategy-loss',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=merge(args.study,args.verify_strategy_loss)
    groups=defaultdict(list)
    for row in rows:
        groups[row['catalog'],row['replica'],row['family']].append(row)
    correlations=[]
    for (c,r,f),group in sorted(groups.items()):
        for metric in METRICS:
            for outcome in OUTCOMES:
                score=_spearman([row[metric] for row in group],[row[outcome] for row in group])
                correlations.append(dict(catalog=c,replica=r,family=f,n_rules=len(group),
                                         metric=metric,outcome=outcome,spearman=score))
    summary=dict(study=str(args.study),cells=len({key[:2] for key in groups}),rule_rows=len(rows),
                 scope='Opened-panel descriptive estimates; fixed shared rules and paired profiles/repeats.',
                 estimates=[])
    for metric in METRICS:
        for outcome in OUTCOMES:
            by_catalog=defaultdict(list)
            for row in correlations:
                if row['metric']==metric and row['outcome']==outcome and row['spearman'] is not None:
                    by_catalog[row['catalog']].append(row['spearman'])
            values=np.array([np.mean(v) for v in by_catalog.values()])
            summary['estimates'].append(dict(metric=metric,outcome=outcome,catalogs=len(values),
                mean_spearman=float(values.mean()) if len(values) else None))
    write_csv(args.output/'games.csv',rows)
    write_csv(args.output/'correlations.csv',correlations)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    demonstrate(rows,args.output/'similarity.png')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
