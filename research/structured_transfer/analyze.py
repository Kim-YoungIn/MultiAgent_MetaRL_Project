"""Summarize predeclared block-level contrasts and descriptive outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METHODS = ('meta_mixed', 'meta_resource', 'frozen_mixed', 'frozen_resource',
           'pooled', 'pooled_adapt', 'scratch')
BUDGETS = (0, 4, 8, 16)
FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')


def bootstrap(x, rng):
    x = np.asarray(x)
    draws = rng.integers(0, len(x), (10000, len(x)))
    return [float(x.mean()), *np.quantile(x[draws].mean(axis=1), [.025, .975]).tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / 'manifest.json').read_text())
    vals, deployed, changes, probeschanges, families = [], [], [], [], []
    for i in range(manifest['blocks']):
        folder = out / f'block_{i:03d}'
        assert (folder / 'complete.json').exists(), f'incomplete block {i}'
        with np.load(folder / 'results.npz') as data:
            assert data['values'].shape == (32, len(METHODS), len(BUDGETS))
            assert data['rewards'].shape == (32, len(METHODS), 32, 2)
            vals.append(data['values'])
            deployed.append(data['rewards'])
            changes.append(data['changes'])
            probeschanges.append(data['probe_changes'])
            families.append(data['families'])
    vals = np.asarray(vals)
    deployed = np.asarray(deployed)
    changes = np.asarray(changes)
    rng = np.random.default_rng(470919)
    gains = vals[:, :, :, 2] - vals[:, :, :, 0]
    primary = (gains[:, :, 0] - gains[:, :, 6]).mean(axis=1)
    diversity = (gains[:, :, 0] - gains[:, :, 1]).mean(axis=1)
    report = {'primary_meta_vs_scratch_gain': bootstrap(primary, rng),
              'descriptive_diversity_gain': bootstrap(diversity, rng),
              'blocks': manifest['blocks'], 'test_games_per_block': 32,
              'probe_opponent_q_absolute_change': float(np.mean(probeschanges)),
              'deploy_opponent_q_absolute_change_by_method': dict(zip(METHODS, changes.mean(axis=(0,1)).tolist())),
              'time_seconds_sum_blocks': sum(json.loads((out / f'block_{i:03d}' / 'complete.json').read_text())['seconds'] for i in range(manifest['blocks']))}
    (out / 'analysis.json').write_text(json.dumps(report, indent=2))
    with (out / 'curves.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['family', 'method', 'budget', 'mean_value', 'gain_from_zero'])
        for fam in ('all', *FAMILIES):
            mask = np.ones(32, bool) if fam == 'all' else families[0] == fam
            for j, method in enumerate(METHODS):
                for k, budget in enumerate(BUDGETS):
                    writer.writerow([fam, method, budget, vals[:, mask, j, k].mean(),
                                     (vals[:, mask, j, k] - vals[:, mask, j, 0]).mean()])
    with (out / 'deployment.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['family', 'method', 'return_32', 'first_8_mean', 'last_8_mean',
                         'opponent_return_32', 'opponent_q_absolute_change'])
        for fam in ('all', *FAMILIES):
            mask = np.ones(32, bool) if fam == 'all' else families[0] == fam
            for j, method in enumerate(METHODS):
                d = deployed[:, mask, j]
                writer.writerow([fam, method, d[:,:,:,0].sum(axis=2).mean(),
                                 d[:,:,:8,0].mean(), d[:,:,24:,0].mean(),
                                 d[:,:,:,1].sum(axis=2).mean(), changes[:,mask,j].mean()])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
        for ax, family in zip(axes, FAMILIES):
            mask = families[0] == family
            for method, color in [('meta_mixed', '#244e95'), ('meta_resource', '#bb613a'),
                                  ('scratch', '#666666'), ('frozen_mixed', '#5c9774')]:
                j = METHODS.index(method)
                y = vals[:,mask,j].mean(axis=(0,1))
                ax.plot(BUDGETS, y-y[0], marker='o', color=color, label=method)
            ax.axhline(0, color='#999999', linewidth=.6)
            ax.set_title(family)
            ax.set_xlabel('Probe interactions')
        axes[0].set_ylabel('Expected reward change from k=0')
        axes[-1].legend(loc='best', fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'curves.png', dpi=160)
    except ImportError:
        print('matplotlib unavailable, curves.csv still written')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
