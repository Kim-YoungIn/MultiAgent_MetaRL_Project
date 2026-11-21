"""Plot measured effects from raw block outputs of both primary experiments."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_band(ax, x, samples, indices, label, color, marker):
    central = samples.mean(axis=0)
    lower, upper = np.quantile(samples[indices].mean(axis=1), [.025, .975], axis=0)
    ax.fill_between(x, lower, upper, color=color, alpha=.12, linewidth=0)
    ax.plot(x, central, marker=marker, markersize=4, color=color, label=label, linewidth=1.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controlled', type=Path)
    parser.add_argument('structured', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.3), layout='constrained')
    shifts = (.75, .5, .25)
    advantages = {p: [] for p in shifts}
    for path in sorted(args.controlled.glob('block_*/metrics.npz')):
        with np.load(path) as data:
            names = data['methods'].tolist()
            d = data['value'][names.index('g128_p75_mix'), :, :, 1] - data['value'][names.index('independent'), :, :, 1]
            same = data['signs'][:, 0] == data['signs'][:, 1]
            for p in shifts:
                advantages[p].append(p*d[same].mean(axis=0) + (1-p)*d[~same].mean(axis=0))
            budgets = data['budgets']
    rng = np.random.default_rng(214230)
    n = len(advantages[shifts[0]])
    indices = rng.integers(n, size=(5000, n))
    for p, color, marker in zip(shifts, ['#196974','#746C66','#A43B35'], ['o','s','^']):
        plot_band(left, budgets, np.array(advantages[p]), indices, f'{int(100*p)}% same sign', color, marker)
    left.set_title('Transfer to an unvisited arena', loc='left', weight='bold')
    left.set_ylabel('Expected payoff advantage over independent filter')
    left.set_xticks(budgets)
    left.legend(frameon=False, fontsize=9)
    histories = []
    for path in sorted(args.structured.glob('block_*/results.npz')):
        with np.load(path) as data:
            histories.append(data['values'].mean(axis=0))
    histories = np.array(histories)
    indices = rng.integers(len(histories), size=(5000, len(histories)))
    for i, label, color, marker in [(0,'Learned prior + updates','#196974','o'), (6,'Scratch + updates','#A43B35','^'),
                                   (5,'Pooled prior + updates','#60488D','D'), (2,'Learned prior, frozen','#746C66','s')]:
        plot_band(right, [0,4,8,16], histories[:, i], indices, label, color, marker)
    right.set_title('Four structured game families', loc='left', weight='bold')
    right.set_ylabel('Expected payoff of selected action')
    right.set_xticks([0,4,8,16])
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.axhline(0, lw=.7, color='#777777')
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Shared probe interactions')
        ax.grid(axis='y', alpha=.15)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
