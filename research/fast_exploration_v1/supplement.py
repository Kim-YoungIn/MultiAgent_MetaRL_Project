<<<<<<< HEAD
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
=======
"""Summarize block-level seed, family, runtime and paired reward diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .report import METHODS, REGIMES, bootstrap_indices, load, summary, write_csv


def paired_differences(paired, regime, family, metric, values):
    for method, comparator in zip(METHODS[1:], values[1:], strict=True):
        paired[regime, family, metric, method].append(values[0] - comparator)


def summarize(root, output=None):
    manifest, blocks = load(root)
    ks = manifest['config']['ks']
    grouped = defaultdict(list)
    families = defaultdict(list)
    paired = defaultdict(list)
    runtime = []
    for block in blocks:
        index = block['block']
        path = root / f'block_{index:03d}.npz'
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != block['raw_sha256']:
                raise ValueError(f'raw reward checksum mismatch at {path}')
        seconds = block['seconds']
        runtime.append(dict(block=index, prior_seconds=seconds['prior'],
                            teacher_seconds=seconds['teacher'], eval_seconds=seconds['evaluation'],
                            total_seconds=seconds['total']))
        with np.load(path) as raw:
            if tuple(raw['methods']) != METHODS or raw['ks'].tolist() != ks:
                raise ValueError(f'unexpected methods or budgets in {path}')
            for regime in REGIMES:
                pay = raw[f'{regime}_assessment']
                family_curves = {
                    str(family): pay[:, :, raw['families'] == family].mean(axis=(2, 3))
                    for family in np.unique(raw['families'])
                }
                for j, method in enumerate(METHODS):
                    grouped[block['master_seed'], regime, method].append(block[regime]['auc'][method])
                    for family, curve in family_curves.items():
                        families[regime, method, family].append(curve[j])
                for family, curve in family_curves.items():
                    auc = np.trapezoid(curve, ks, axis=1) / ks[-1]
                    paired_differences(paired, regime, family, 'absolute_auc', auc)
                if regime == 'reward_auc':
                    exploration = raw[f'{regime}_exploration'].mean(axis=2)
                    assessment = pay.sum(axis=3).mean(axis=2)
                    for metric, values in (
                        ('exploration_reward', exploration),
                        ('assessment_reward', assessment),
                        ('total_return', exploration + assessment),
                    ):
                        paired_differences(paired, regime, 'all', metric, values)

    output = root / 'analysis' if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    master_rows = []
    for (master, regime, method), values in grouped.items():
        learned = np.asarray(grouped[master, regime, 'learned'])
        master_rows.append(dict(master_seed=master, regime=regime, method=method, blocks=len(values),
                                mean_auc=np.mean(values), block_sd=np.std(values, ddof=1) if len(values) > 1 else 0.,
                                learned_minus_method=(learned - values).mean()))
    write_csv(output / 'master_seeds.csv', master_rows)
    write_csv(output / 'family_curves.csv', [
        dict(regime=regime, method=method, family=family, k=k, mean=value)
        for (regime, method, family), values in families.items()
        for k, value in zip(ks, np.mean(values, axis=0), strict=True)
    ])
    write_csv(output / 'runtime_by_block.csv', runtime)

    resamples = bootstrap_indices(blocks)
    paired_rows = []
    for (regime, family, metric, method), values in paired.items():
        if len(values) != len(blocks):
            raise ValueError(f'{family} has incomplete block coverage')
        block_values = np.array(values).reshape(len(blocks), -1)
        budgets = ['all'] if metric == 'absolute_auc' else ks
        for k, per_block in zip(budgets, block_values.T, strict=True):
            paired_rows.append(dict(regime=regime, family=family, metric=metric,
                                    contrast=f'learned_minus_{method}', k=k,
                                    **summary(per_block, resamples)))
    write_csv(output / 'paired_contexts.csv', paired_rows)

    times = np.array([row['total_seconds'] for row in runtime])
    result = dict(first_block_seconds=times[0],
                  median_parallel_block_seconds=float(np.median(times[1:])) if len(times) > 1 else None,
                  total_block_elapsed_seconds=float(times.sum()))
    (output / 'runtime.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.output), indent=2))
>>>>>>> 0c5c8a8 (ver.0.8)
