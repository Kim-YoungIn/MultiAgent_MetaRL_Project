"""Plot Student adaptation vs exploration budget K and within-assessment rewards.

All confidence intervals resample *independent training/test blocks*, not turns
or individual games.  Curves are actual held-out results, not illustrative.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ('learned', 'blind', 'random', 'greedy')
DISPLAY = {'learned': 'Bayesian Teacher', 'blind': 'Belief-blind Teacher',
           'random': 'Random', 'greedy': 'Greedy'}


def _interval(block_values, rng, draws=10000):
    """Pointwise percentile bootstrap of mean; first axis is independent block."""
    values = np.asarray(block_values, dtype=float)
    if values.ndim < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('expected finite values from at least one independent block')
    mean = values.mean(axis=0)
    resamples = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[resamples].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return mean, lo, hi


def _write_csv(path, fields, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(output, *, k_for_steps=8):
    """Create three figures and machine-readable CSV from sealed block records."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    paths = sorted(output.glob('block_*.json'))
    if not paths:
        raise ValueError(f'no completed block_*.json in {output}')
    manifest = output / 'manifest.json'
    if manifest.exists():
        expected = int(json.loads(manifest.read_text())['config']['blocks'])
        actual = [p.stem for p in paths]
        required = [f'block_{i:03d}' for i in range(expected)]
        if actual != required:
            raise ValueError(f'incomplete run: found {len(actual)} of {expected} blocks')
        # Figures from a sealed run should reject corrupted weights and records.
        import hashlib
        for path in paths:
            record = json.loads(path.read_text())
            weights = path.with_suffix('.pt')
            if not weights.is_file() or record.get('weights_sha256') != hashlib.sha256(weights.read_bytes()).hexdigest():
                raise ValueError(f'checkpoint integrity failure for {path}')
    blocks = [json.loads(p.read_text())['reward_auc'] for p in paths]
    ks = np.asarray(blocks[0]['ks'], dtype=int)
    if len(ks) < 2 or ks[0] != 0 or not np.all(np.diff(ks) > 0):
        raise ValueError('invalid adaptation budget grid')
    if k_for_steps not in ks:
        raise ValueError(f'K={k_for_steps} not in {ks.tolist()}')
    if any(b['ks'] != ks.tolist() for b in blocks):
        raise ValueError('inconsistent probe budgets across blocks')
    methods = [m for m in METHODS if m in blocks[0]['curves']]
    if any(set(b['curves']) != set(methods) for b in blocks):
        raise ValueError('inconsistent methods across blocks')
    output_dir = output / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(992271)
    colors = {'learned': 'tab:blue', 'blind': 'tab:orange',
              'random': 'tab:green', 'greedy': 'tab:red'}
    mean_curve, lo_curve, hi_curve, auc_summary, curve_rows, slope_rows = {}, {}, {}, {}, [], []
    for method in methods:
        block_curves = np.array([b['curves'][method] for b in blocks], dtype=float)
        if block_curves.shape != (len(blocks), len(ks)):
            raise ValueError('invalid adaptation curve length')
        # Do not silently trust saved AUC: verify it agrees with the graphed curve.
        auc_blocks = np.trapezoid(block_curves, x=ks, axis=1) / ks[-1]
        saved = np.array([b['auc'][method] for b in blocks])
        if not np.allclose(auc_blocks, saved, rtol=1e-8, atol=1e-10):
            raise ValueError('saved AUC does not match step-by-step Student reward curve')
        mean, lo, hi = _interval(block_curves, rng)
        auc_mean, auc_lo, auc_hi = _interval(auc_blocks, rng)
        mean_curve[method], lo_curve[method], hi_curve[method] = mean, lo, hi
        auc_summary[method] = (float(auc_mean), float(auc_lo), float(auc_hi))
        for j, k in enumerate(ks):
            curve_rows.append(dict(method=method,k=int(k),mean=float(mean[j]),
                                   ci_low=float(lo[j]),ci_high=float(hi[j])))
        for j in range(1, len(ks)):
            slopes = (block_curves[:,j]-block_curves[:,j-1])/(ks[j]-ks[j-1])
            sm, sl, sh = _interval(slopes, rng)
            slope_rows.append(dict(method=method,k_from=int(ks[j-1]),k_to=int(ks[j]),
                                   slope=float(sm),ci_low=float(sl),ci_high=float(sh)))
    _write_csv(output_dir/'adaptation_curve.csv', ['method','k','mean','ci_low','ci_high'],curve_rows)
    _write_csv(output_dir/'adaptation_slopes.csv',
               ['method','k_from','k_to','slope','ci_low','ci_high'],slope_rows)
    _write_csv(output_dir/'auc.csv', ['method','absolute_auc','ci_low','ci_high'],
               [dict(method=m, absolute_auc=auc_summary[m][0],ci_low=auc_summary[m][1],
                     ci_high=auc_summary[m][2]) for m in methods])

    fig, ax = plt.subplots(figsize=(10.0,5.7))
    for method in methods:
        ax.plot(ks,mean_curve[method],marker='o',linewidth=2.1,
                color=colors[method],label=f'{DISPLAY[method]}  (AUC {auc_summary[method][0]:.4f})')
        ax.fill_between(ks,lo_curve[method],hi_curve[method],alpha=.13,color=colors[method])
    if 'learned' in methods:
        ax.fill_between(ks,0,mean_curve['learned'],alpha=.10,color=colors['learned'],
                        label='Bayesian Teacher area to zero')
    ax.axhline(0,color='gray',linewidth=.75,linestyle=':')
    ax.set_xticks(ks)
    ax.set_xlim(ks[0],ks[-1])
    ax.set_xlabel('Teacher exploration experience (K interactions)')
    ax.set_ylabel('Mean Student reward per turn after handoff')
    ax.set_title(f'Adaptation curves on held-out games  ({len(blocks)} independent fits)')
    ax.grid(alpha=.15)
    ax.legend(loc='best',fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir/'adaptation_curve.png',dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5,5.2))
    for method in methods:
        rows=[r for r in slope_rows if r['method']==method]
        mid=np.array([(r['k_from']+r['k_to'])/2 for r in rows])
        y=np.array([r['slope'] for r in rows]); low=np.array([r['ci_low'] for r in rows]); high=np.array([r['ci_high'] for r in rows])
        ax.plot(mid,y,marker='o',color=colors[method],label=DISPLAY[method])
        ax.fill_between(mid,low,high,color=colors[method],alpha=.12)
    ax.axhline(0,color='gray',linestyle=':',linewidth=.8)
    ax.set_xlabel('Exploration experience interval midpoint (K)')
    ax.set_ylabel('Change in Student reward / additional probe')
    ax.set_title('Local adaptation slopes (adjacent K checkpoints)')
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir/'adaptation_slopes.png',dpi=175)
    plt.close(fig)

    step_rows=[]
    fig, ax = plt.subplots(figsize=(10,5.5))
    for method in methods:
        per_block=np.array([b['assessment_steps'][method][str(k_for_steps)] for b in blocks])
        avg,lo,hi=_interval(per_block,rng)
        t=np.arange(1,len(avg)+1)
        ax.plot(t,avg,color=colors[method],linewidth=1.7,label=DISPLAY[method])
        ax.fill_between(t,lo,hi,color=colors[method],alpha=.12)
        for i in range(len(t)):
            step_rows.append(dict(method=method,k=k_for_steps,assessment_step=i+1,
                                  mean=float(avg[i]),ci_low=float(lo[i]),ci_high=float(hi[i])))
    ax.axhline(0,color='gray',linestyle=':',linewidth=.8)
    ax.set_xlabel('Student assessment turn after handoff')
    ax.set_ylabel('Mean realized Student reward at this turn')
    ax.set_title(f'Student reward over assessment time  (after K={k_for_steps} probes)')
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir/f'assessment_reward_k{k_for_steps}.png',dpi=175)
    plt.close(fig)
    _write_csv(output_dir/f'assessment_reward_k{k_for_steps}.csv',
               ['method','k','assessment_step','mean','ci_low','ci_high'],step_rows)
    return dict(blocks=len(blocks), figures=[str(output_dir/p) for p in
        ('adaptation_curve.png','adaptation_slopes.png',f'assessment_reward_k{k_for_steps}.png')],
        auc={m:auc_summary[m][0] for m in methods})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--k-for-steps',type=int,default=8)
    args=parser.parse_args()
    print(json.dumps(render(args.output,k_for_steps=args.k_for_steps),indent=2))

if __name__ == '__main__':
    main()
