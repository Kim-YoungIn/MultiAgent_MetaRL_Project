"""Summarize and plot each master seed from a completed scaled run."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .report import METHODS, REGIMES, bootstrap_indices, load, write_csv

COLORS = ('#176d76', '#a54b2f', '#517a49', '#66568a')
NAMES = ('Learned', 'Belief-Blind', 'Random', 'Greedy')


def interval(values, draws):
    """Return mean and pointwise limits with block as the first axis."""
    if not np.isfinite(values).all():
        raise ValueError('non-finite block metric')
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
    return np.stack([values.mean(axis=0), lo, hi], axis=-1)


def collect(records, ks, step_k, h):
    groups = defaultdict(list)
    for record in records:
        groups[record['master_seed']].append(record)

    axes = {
        'absolute_auc': ['all'],
        'paired_auc': ['all'],
        'assessment_mean': ks,
        'total_return': ks,
        'slope': [f'{a}–{b}' for a, b in zip(ks, ks[1:])],
        'assessment_turn': list(range(1, h+1)),
    }
    values = {}
    rows = []
    for master, blocks in groups.items():
        draws = bootstrap_indices(blocks)
        per_seed = values[master] = {}
        for regime in REGIMES:
            records_at_regime = [block[regime] for block in blocks]
            if any(r['ks'] != ks for r in records_at_regime):
                raise ValueError('inconsistent exploration budgets')
            auc = np.array([[r['auc'][m] for m in METHODS] for r in records_at_regime])
            curves, totals = (np.array([[r[key][m] for m in METHODS] for r in records_at_regime])
                              for key in ('curves', 'total_return'))
            turns = np.array([[r['assessment_steps'][m][str(step_k)] for m in METHODS]
                              for r in records_at_regime])
            if turns.shape[-1] != h:
                raise ValueError('inconsistent assessment horizon')
            per_metric = dict(absolute_auc=auc[..., None], paired_auc=(auc[:, :1]-auc[:, 1:])[..., None],
                              assessment_mean=curves, total_return=totals,
                              slope=np.diff(curves, axis=2)/np.diff(ks), assessment_turn=turns)
            for metric, array in per_metric.items():
                result = interval(array, draws)
                if regime == 'reward_auc':
                    per_seed[metric] = result
                labels = [f'learned_minus_{m}' for m in METHODS[1:]] if metric == 'paired_auc' else METHODS
                sd = array.std(axis=0, ddof=1)
                for method, stats, spread in zip(labels, result, sd, strict=True):
                    for x, (mean, low, high), block_sd in zip(axes[metric], stats, spread, strict=True):
                        rows.append(dict(master_seed=master, blocks=len(blocks), regime=regime,
                                         method=method, metric=metric, x=x,
                                         probe_k=step_k if metric=='assessment_turn' else '',
                                         mean=mean, block_sd=block_sd, ci_low=low, ci_high=high))
    return groups, values, rows


def lines(ax, values, x):
    for name, color, points in zip(NAMES, COLORS, values, strict=True):
        avg, low, high = points.T
        ax.plot(x, avg, color=color, marker='o' if len(x)<10 else None, ms=3, label=name)
        ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)


def dots(ax, values, names, colors):
    for j, ((mean, lo, hi), color) in enumerate(zip(values, colors, strict=True)):
        ax.hlines(j, lo, hi, color=color, linewidth=1.8)
        ax.plot(mean, j, 'o', color=color, ms=5)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()


def limits(values, metric):
    values = [group[metric][..., 1:] for group in values.values()]
    lo = min(np.min(group) for group in values)
    hi = max(np.max(group) for group in values)
    if metric=='paired_auc':
        lo, hi = min(lo, 0), max(hi, 0)
    margin = max(.07*(hi-lo), 1e-8)
    return lo-margin, hi+margin


def plots(output, groups, values, ks, h, step_k):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    specs = [('assessment_mean', 'Student adaptation curve', 'Reward per assessment turn', ks),
             ('total_return', 'Total reward', 'Exploration + assessment reward', ks),
             ('slope', 'Adaptation slope', 'Reward change per added probe', (np.array(ks[1:])+ks[:-1])/2),
             ('assessment_turn', f'Assessment at K={step_k}', 'Realized reward', np.arange(1, h+1))]
    overview, panels = plt.subplots(2,len(groups),figsize=(4.25*len(groups),6.2),layout='constrained',squeeze=False)
    for i, (seed, data) in enumerate(values.items()):
        fig, axes = plt.subplots(2,3,figsize=(12.4,6.8),layout='constrained')
        for ax, (metric, title, ylabel, x) in zip(axes.ravel(),specs,strict=False):
            lines(ax,data[metric],x)
            ax.set_ylim(limits(values,metric))
            ax.set(title=title, ylabel=ylabel, xlabel='Assessment turn' if metric=='assessment_turn' else 'Exploration turns (K)')
            if metric in ('assessment_mean','total_return'):
                ax.set_xticks(ks)
            if metric=='slope':
                ax.set_xticks(x,[f'{a}–{b}' for a,b in zip(ks,ks[1:])])
        for ax, metric, title, names, color in [
            (axes[1,1],'absolute_auc','Absolute adaptation AUC',NAMES,COLORS),
            (axes[1,2],'paired_auc','Learned minus comparator',NAMES[1:],COLORS[1:]),
        ]:
            dots(ax,data[metric][:,0],names,color)
            ax.set_xlim(limits(values,metric))
            ax.set(title=title,xlabel='AUC' if metric=='absolute_auc' else 'Paired AUC difference')
        axes[1,2].axvline(0,color='#717171',lw=.8)
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle(f'Master seed {seed} · {len(groups[seed])} independent blocks')
        for ax in axes.ravel():
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15)
        fig.savefig(output/f'seed_{seed}.png',dpi=170)
        plt.close(fig)

        for row, (metric, _, ylabel, x) in enumerate(specs[:2]):
            ax = panels[row,i]
            lines(ax,data[metric],x)
            ax.set_ylim(limits(values,metric))
            ax.set_xticks(ks)
            if not row:
                ax.set_title(f'Seed {seed}')
            else:
                ax.set_xlabel('Exploration turns (K)')
            if i==0:
                ax.set_ylabel(ylabel)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15)
    panels[0,0].legend(frameon=False,fontsize=8)
    overview.savefig(output/'overview.png',dpi=170)
    plt.close(overview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--assessment-k', type=int, default=8)
    args = parser.parse_args()
    manifest, records = load(args.run)
    cfg = manifest['config']
    ks = cfg['ks']
    if args.assessment_k not in ks:
        parser.error(f'assessment K must be in {ks}')
    groups, values, rows = collect(records, ks, args.assessment_k, cfg['h'])
    if (list(groups) != manifest['master_seeds'] or
        any(sorted(r['seed_block'] for r in block) != list(range(cfg['blocks_per_master']))
            for block in groups.values())):
        raise ValueError('unexpected seed assignments or incomplete seed group')
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'seed_metrics.csv',rows)
    for seed in groups:
        write_csv(args.output/f'seed_{seed}.csv',[r for r in rows if r['master_seed']==seed])
    plots(args.output, groups, values, ks, cfg['h'], args.assessment_k)
    print(', '.join(f'{seed}: {len(blocks)} blocks' for seed, blocks in groups.items()))


if __name__ == '__main__':
    main()
