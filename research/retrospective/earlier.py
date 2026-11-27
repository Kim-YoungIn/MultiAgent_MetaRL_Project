"""Post-hoc paired analyses of three opened final panels.

All intervals resample whole independent training blocks. Within-block game
summaries are paired between methods and are never treated as training seeds.
"""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


ROOT = Path('runs')
FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
OUT = Path('research/retrospective/evidence/earlier.csv')


def checked(block, filename):
    expected = json.loads((block / 'record.json').read_text())['sha256'][filename]
    actual = hashlib.sha256((block / filename).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'checksum mismatch for {block / filename}')
    with np.load(block / filename, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def opponents():
    root = ROOT / 'opponent_adaptation_v1'
    assert json.loads((root / 'manifest.json').read_text())['config']['blocks'] == 24
    rows = []
    for i in range(24):
        folder = root / f'block_{i:03d}'
        data, result = checked(folder, 'test.npz'), checked(folder, 'results.npz')
        assert result['value'].shape == (128, 2, 5, 6)
        for env, env_label in ((0, 'frozen_opponent'), (1, 'learning_opponent')):
            for family, mask in [('all', np.ones(128, bool)),
                                 *[(f, data['family'] == j) for j, f in enumerate(FAMILIES)]]:
                assert mask.sum() == (128 if family == 'all' else 32)
                values = result['value'][mask, env]
                decisions = result['decision'][mask, env]
                # Same probe history and opponent state, k=8 (budget index 3).
                diff = values[:, 0, 3] - values[:, 1, 3]
                disagreement = decisions[:, 0, 3] != decisions[:, 1, 3]
                if not np.all(np.abs(diff[~disagreement]) <= 1e-12):
                    raise ValueError('equal query actions must have equal expected value')
                deploy = result['deploy_rewards'][mask, env]
                focal = deploy[:, :, :, 0]
                other = deploy[:, :, :, 1]
                metrics = {
                    'common_meta_minus_pooled_k0': (values[:, 0, 0] - values[:, 1, 0]).mean(),
                    'common_meta_minus_pooled_k8': diff.mean(),
                    'common_action_disagreement_k8': disagreement.mean(),
                    'common_meta_better_k8': (diff > 1e-12).mean(),
                    'common_pooled_better_k8': (diff < -1e-12).mean(),
                    'common_tied_value_k8': (np.abs(diff) <= 1e-12).mean(),
                    'common_meta_vs_pool_gain_k8': ((values[:, 0, 3] - values[:, 0, 0])
                                                     - (values[:, 1, 3] - values[:, 1, 0])).mean(),
                    'deploy_meta_minus_pooled_first8': (focal[:, 0, :8] - focal[:, 1, :8]).sum(1).mean(),
                    'deploy_meta_minus_pooled_last16': (focal[:, 0, 8:] - focal[:, 1, 8:]).sum(1).mean(),
                    'deploy_meta_minus_pooled_total24': (focal[:, 0] - focal[:, 1]).sum(1).mean(),
                    'deploy_opponent_meta_minus_pooled_total24': (other[:, 0] - other[:, 1]).sum(1).mean(),
                    'deploy_meta_minus_pooled_win_fraction': ((focal[:, 0].sum(1) - focal[:, 1].sum(1)) > 1e-12).mean(),
                    'deploy_q_delta_meta_minus_pooled': (result['deploy_q_delta'][mask, env, 0] -
                                                        result['deploy_q_delta'][mask, env, 1]).sum(1).mean(),
                }
                rows += [(i, 'opponent_adaptation', env_label, family, k, float(v)) for k, v in metrics.items()]
    return rows


def cross_context():
    root = ROOT / 'adapt_v1_final'
    assert json.loads((root / 'manifest.json').read_text())['config']['blocks'] == 32
    rows = []
    for i in range(32):
        folder = root / f'block_{i:03d}'
        for pop in ('metrics', 'shift'):
            data = checked(folder, pop + '.npz')
            methods = data['methods'].tolist()
            left = methods.index('g128_p75_mix')
            right = methods.index('independent')
            target = data['value'][left, :, 3, 1] - data['value'][right, :, 3, 1]
            np.testing.assert_allclose(data['value'][left, :, 0, 1],
                                       data['value'][right, :, 0, 1], atol=1e-12, rtol=0)
            same = data['signs'][:, 0] == data['signs'][:, 1]
            for label, mask in (('concordant', same), ('discordant', ~same),
                                ('fast_concordant', same & np.isclose(data['rate'], .65)),
                                ('slow_concordant', same & np.isclose(data['rate'], .12)),
                                ('fast_discordant', ~same & np.isclose(data['rate'], .65)),
                                ('slow_discordant', ~same & np.isclose(data['rate'], .12))):
                if not mask.any():
                    raise ValueError(f'empty {label} in {folder}/{pop}')
                rows.append((i, 'cross_context', pop, label, 'learned_minus_independent_other_arena_k4',
                             float(target[mask].mean())))
            disagree = data['pred'][left, :, 3, 1] - data['pred'][right, :, 3, 1]
            rows.append((i, 'cross_context', pop, 'all', 'mean_absolute_prediction_difference',
                         float(np.abs(disagree).mean())))
    return rows


def absolute():
    root = ROOT / 'absolute_exploration_v1'
    assert json.loads((root / 'manifest.json').read_text())['config']['blocks'] == 24
    rows = []
    for i in range(24):
        data = checked(root / f'block_{i:03d}', 'eval.npz')
        assert data['openers'].shape == (512, 5)
        n = len(data['motif'])
        # Terminal actions use (game, opening action, feedback, environment).
        selected = np.take_along_axis(data['next_value'], data['terminal_actions'], axis=3)
        first = data['first_reward']
        terminal = selected[:, :, 0, 1]
        total = first + terminal
        ii = np.arange(n)
        chosen_return, chosen_myopic = data['openers'][:, 0], data['openers'][:, 3]
        for family, mask in [('all', np.ones(n, bool)),
                             *[(f, data['motif'] == j) for j, f in enumerate(FAMILIES)]]:
            if mask.sum() != (n if family == 'all' else n // 4):
                raise ValueError('unbalanced motif panel')
            a, b = chosen_return[mask], chosen_myopic[mask]
            ix = ii[mask]
            diff = total[ix, a] - total[ix, b]
            different = a != b
            if not np.all(np.abs(diff[~different]) <= 1e-12):
                raise ValueError('the same opening must lead to the same next decision')
            delta_first = first[ix, a] - first[ix, b]
            delta_terminal = terminal[ix, a] - terminal[ix, b]
            metrics = {
                'total_minus_myopic': diff.mean(),
                'opening_minus_myopic': delta_first.mean(),
                'terminal_minus_myopic': delta_terminal.mean(),
                'different_opening_fraction': different.mean(),
                'better_total_fraction': (diff > 1e-12).mean(),
                'worse_total_fraction': (diff < -1e-12).mean(),
                'tied_total_fraction': (np.abs(diff) <= 1e-12).mean(),
                # Aggregate numerator and denominator separately for conditional effect.
                'different_opening_sum_delta_per_game': (diff * different).mean(),
                'different_opening_sum_count_per_game': different.mean(),
            }
            rows += [(i, 'absolute_exploration', 'two_step', family, k, float(v))
                     for k, v in metrics.items()]
    return rows


def main():
    samples = opponents() + cross_context() + absolute()
    fields = ('block', 'study', 'regime', 'subgroup', 'metric', 'value')
    block_path = OUT.with_name('earlier_blocks.csv')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with block_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(samples)
    grouped = defaultdict(list)
    for block, study, regime, subgroup, metric, value in samples:
        grouped[study, regime, subgroup, metric].append((block, value))
    rng = np.random.default_rng(20261127)
    bootstrap = {n: rng.integers(n, size=(20000, n)) for n in (24, 32)}
    with OUT.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('study', 'regime', 'subgroup', 'metric', 'blocks', 'mean', 'ci_low', 'ci_high',
                         'blocks_positive', 'blocks_negative'))
        for (study, regime, subgroup, metric), values in sorted(grouped.items()):
            values.sort()
            y = np.array([v for _, v in values])
            assert len(y) in bootstrap
            ci = np.quantile(y[bootstrap[len(y)]].mean(axis=1), [.025, .975])
            writer.writerow((study, regime, subgroup, metric, len(y), y.mean(), *ci,
                             (y > 0).sum(), (y < 0).sum()))
    print(f'Analyzed {len(samples)} block-level estimates; wrote {OUT} and {block_path}')


if __name__ == '__main__':
    main()
