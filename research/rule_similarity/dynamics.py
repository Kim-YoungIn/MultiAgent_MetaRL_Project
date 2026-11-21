"""Exploratory within-family correlations with reward and transition distances."""
import copy
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from mamrl.structured_games import StructuredEnv
from research.structured_oracle.run import Gate
from research.structured_scale.run import FAMILIES, game

PERMS = tuple(itertools.permutations(range(3)))
# Histories contain prior joint actions, most recent first. Shared across games.
CONTEXTS = (
    (((0, 0), (0, 1), (1, 2)), (0, 0, 0), (0, 0)),
    (((0, 1), (2, 0), (1, 1)), (1, 1, 1), (1, 2)),
    (((1, 2), (1, 2), (0, 0)), (2, 0, 1), (2, 0)),
    (((2, 1), (0, 2), (2, 1)), (0, 2, 2), (1, 1)),
    (((1, 1), (2, 2), (0, 1)), (1, 0, 2), (0, 2)),
    (((2, 2), (1, 0), (1, 0)), (2, 2, 0), (2, 1)),
)


def signatures(seed, family):
    rule = game(int(seed), FAMILIES[int(family)], 3, 3)
    gated = [t.probability for t in rule.transitions if 0 < t.probability < 1]
    if len(gated) > 1:
        raise ValueError('one stochastic gate at most')
    probability = gated[0] if gated else 1.
    env = StructuredEnv(rule, max_steps=1)
    env.reset(seed=0)
    result = np.zeros((6, len(CONTEXTS), 5))
    for j, perm in enumerate(PERMS):
        for k, (history, state, actions) in enumerate(CONTEXTS):
            env._history = tuple((perm[a], perm[b]) for a, b in history)
            env._state = {v.name: state[i] for i, v in enumerate(rule.states)}
            values = []
            for u in (0., (1 + probability)/2):
                branch = copy.copy(env)
                branch._rng = Gate(u)
                outcome = branch.step(perm[actions[0]], perm[actions[1]])
                states = [branch._state[v.name] for v in rule.states]
                values.append([*outcome.rewards, *(states + [0]*(3-len(states)))])
            result[j, k] = probability*np.array(values[0]) + (1-probability)*np.array(values[1])
    return result


def main():
    root = Path('runs/opponent_adaptation_v1')
    output = Path('runs/rule_similarity_dynamics')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in sorted(root.glob('block_*')):
        with np.load(folder/'train.npz') as tr, np.load(folder/'test.npz') as te, np.load(folder/'results.npz') as rs:
            train = np.stack([signatures(s, f)[0] for s, f in zip(tr['seed'], tr['family'])])
            test = np.stack([signatures(s, f) for s, f in zip(te['seed'], te['family'])])
            # Signatures have independently permuted test labels. State coordinates
            # refer to named generator variables. Reward and state are kept separate.
            for i in range(len(te['seed'])):
                eligible = train[tr['family'] == te['family'][i]]
                diff = test[i, :, None] - eligible[None]
                rd = (diff[..., :2]**2).mean(axis=(2, 3))**.5
                td = (diff[..., 2:]**2).mean(axis=(2, 3))**.5
                combined = (rd**2 + (td/2)**2)**.5
                v = rs['value'][i, 1, :, 3]
                rows.append(dict(block=folder.name, family=FAMILIES[te['family'][i]], seed=int(te['seed'][i]),
                    reward_similarity=-float(rd.min()), transition_similarity=-float(td.min()),
                    combined_similarity=-float(combined.min()),
                    retained_gain=float(v[0]-v[2]), meta_extra_gain=float((v[0]-v[2])-(v[1]-v[3]))))
    with (output/'games.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(81021)
    index = rng.integers(24, size=(20000, 24))
    summary = {}
    strata = []
    for metric in ('reward_similarity','transition_similarity','combined_similarity'):
        for outcome in ('retained_gain','meta_extra_gain'):
            values = []
            for block in range(24):
                correlations = []
                for family in FAMILIES:
                    subset = [r for r in rows if r['block'] == f'block_{block:03d}' and r['family'] == family]
                    x = np.array([r[metric] for r in subset])
                    y = np.array([r[outcome] for r in subset])
                    if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                        correlations.append(np.corrcoef(x, y)[0, 1])
                for family, correlation in zip(FAMILIES, [
                    np.corrcoef([r[metric] for r in subset], [r[outcome] for r in subset])[0, 1]
                    if np.std([r[metric] for r in subset]) > 1e-12 and np.std([r[outcome] for r in subset]) > 1e-12 else None
                    for subset in [[r for r in rows if r['block'] == f'block_{block:03d}' and r['family'] == f] for f in FAMILIES]
                ]):
                    strata.append(dict(block=block, family=family, metric=metric, outcome=outcome, correlation=correlation))
                values.append(np.mean(correlations) if correlations else np.nan)
            values = np.array(values)
            finite = np.isfinite(values)
            if finite.all():
                ci = np.quantile(values[index].mean(1), [.025,.975]).tolist()
            else:
                ci = None
            summary[f'{metric}:{outcome}'] = dict(mean=float(np.mean(values[finite])) if finite.any() else None, interval_95=ci, valid_blocks=int(finite.sum()))
    with (output/'strata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=strata[0])
        writer.writeheader()
        writer.writerows(strata)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
