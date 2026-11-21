"""Exploratory payoff-rule similarity versus retained-experience gain."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from research.structured_scale.run import FAMILIES, game
from research.structured_oracle.run import RewardKernel
from mamrl.structured_games import StructuredEnv

PERMS = list(itertools.permutations(range(3)))


def signature(seed, family):
    rule = game(int(seed), FAMILIES[int(family)], 3, 3)
    env = StructuredEnv(rule, max_steps=1)
    env.reset(seed=0)
    kernel = RewardKernel(rule)
    result = np.empty((3, 3, 3, 3))
    for a in range(3):
        for b in range(3):
            env._history = ((a, b), (0, 0), (0, 0))
            result[a, b] = kernel(env)
    return result


def main():
    root = Path('runs/opponent_adaptation_v1')
    output = Path('runs/rule_similarity_structured')
    output.mkdir(exist_ok=True, parents=True)
    rows = []
    block_correlations = []
    for folder in sorted(root.glob('block_*')):
        with np.load(folder/'train.npz') as train, np.load(folder/'test.npz') as test, np.load(folder/'results.npz') as result:
            train_signatures = np.stack([signature(s, f) for s, f in zip(train['seed'], train['family'])])
            test_signatures = np.stack([signature(s, f) for s, f in zip(test['seed'], test['family'])])
            # One shared relabeling applies to both players and each time index.
            nearest = np.full(len(test['seed']), np.inf)
            for perm in PERMS:
                permuted = train_signatures[:, perm, :, :, :][:, :, perm, :, :][:, :, :, perm, :][:, :, :, :, perm]
                distance = ((test_signatures[:, None] - permuted[None])**2).mean(axis=(2, 3, 4, 5))**.5
                nearest = np.minimum(nearest, distance.min(axis=1))
            similarity = -nearest
            gain = result['value'][:, 1, 0, 3] - result['value'][:, 1, 2, 3]
            zero = result['value'][:, 1, 0, 0]
            adjusted = result['value'][:, 1, 0, 3] - zero
            for f in range(4):
                mask = test['family'] == f
                for y, name in [(gain, 'adaptive_minus_frozen'), (adjusted, 'k8_minus_k0')]:
                    block_correlations.append(dict(block=folder.name, family=FAMILIES[f], outcome=name,
                                                   correlation=float(np.corrcoef(similarity[mask], y[mask])[0, 1])))
            for i in range(len(similarity)):
                rows.append(dict(block=folder.name, family=FAMILIES[test['family'][i]], seed=int(test['seed'][i]),
                                 nearest_rmse=float(nearest[i]), similarity=float(similarity[i]),
                                 adaptive_minus_frozen=float(gain[i]), k8_minus_k0=float(adjusted[i])))
    for name, data in [('games.csv', rows), ('block_correlations.csv', block_correlations)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    rng = np.random.default_rng(197001)
    summaries = {}
    for outcome in ('adaptive_minus_frozen', 'k8_minus_k0'):
        values = np.array([[r['correlation'] for r in block_correlations if r['block'] == f'block_{i:03d}' and r['outcome'] == outcome] for i in range(24)]).mean(axis=1)
        draws = rng.integers(0, len(values), (20000, len(values)))
        summaries[outcome] = dict(mean=float(values.mean()), interval_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
    (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
