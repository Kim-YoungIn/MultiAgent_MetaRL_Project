"""Summarize paired structured-game interventions at the independent block level."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from research.fast_exploration_v2.analyze import bootstrap_indices, statistics
from research.fast_exploration_v1.scale import digest

FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
LIVE = ('pooled_fixed_total', 'pooled_ucb', 'pooled_greedy')
CONTROLLED = ('pooled_full', 'pooled_masked', 'zero')
PROBE = CONTROLLED[:2]
DISTANCES = (0.0, 0.3, 0.6)
LAGS = (0, 1)


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _group_mean(values, families):
    """The last axis is episode, and all gives each game family equal weight."""
    means = [values[..., families == family].mean(axis=-1) for family in FAMILIES]
    return np.stack([np.mean(means, axis=0), *means], axis=-1)


def _check(raw):
    np.testing.assert_allclose(raw['distances'], DISTANCES, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(raw['lags'], LAGS)
    np.testing.assert_array_equal(raw['live_methods'], LIVE)
    np.testing.assert_array_equal(raw['controlled_methods'], CONTROLLED)
    np.testing.assert_array_equal(raw['probe_methods'], PROBE)
    families = raw['families']
    if set(families.tolist()) != set(FAMILIES):
        raise ValueError('each block must contain all four families')
    episodes = len(families)
    if raw['action_to_role'].shape != (episodes, 3):
        raise ValueError('invalid action-to-role mapping')
    np.testing.assert_array_equal(np.sort(raw['action_to_role'], axis=1),
                                  np.broadcast_to(np.arange(3), (episodes, 3)))
    for prefix, methods, turns in (('live', LIVE, 96), ('controlled', CONTROLLED, 32),
                                   ('probe', PROBE, 8)):
        actions, rewards = raw[f'{prefix}_actions'], raw[f'{prefix}_rewards']
        shape = (2, 3, len(methods), episodes, turns, 2)
        if actions.shape != shape or rewards.shape != shape:
            raise ValueError(f'invalid {prefix} shapes {actions.shape}, {rewards.shape}')
        if not np.isfinite(rewards).all() or not np.all((0 <= actions) & (actions < 3)):
            raise ValueError(f'invalid {prefix} values')
    return families


def block_metrics(raw):
    families = _check(raw)
    metrics = {}
    def add(name, values):
        metrics[name] = _group_mean(values, families)

    controlled = raw['controlled_rewards'][..., 0].mean(axis=-1)
    for index, method in enumerate(CONTROLLED):
        add(f'controlled_mean:{method}', controlled[:, :, index])
    add('controlled_mean:full_minus_masked', controlled[:, :, 0] - controlled[:, :, 1])
    add('controlled_mean:full_minus_zero', controlled[:, :, 0] - controlled[:, :, 2])
    add('controlled_mean:masked_minus_zero', controlled[:, :, 1] - controlled[:, :, 2])

    rewards = raw['live_rewards'][..., 0]
    for start, stop, label in ((0, 8, 'probe'), (8, 48, 'tail_to_48'),
                               (8, 96, 'tail_to_96'), (0, 48, 'total_48'),
                               (0, 96, 'total_96')):
        totals = rewards[..., start:stop].sum(axis=-1)
        for index, method in enumerate(LIVE):
            add(f'{label}:{method}', totals[:, :, index])
        add(f'{label}:teacher_minus_ucb', totals[:, :, 0] - totals[:, :, 1])
        add(f'{label}:teacher_minus_student', totals[:, :, 0] - totals[:, :, 2])

    probe_rewards = raw['probe_rewards'][..., 0].sum(axis=-1)
    for index, method in enumerate(PROBE):
        add(f'controlled_probe_return:{method}', probe_rewards[:, :, index])
    add('controlled_probe_return:full_minus_masked', probe_rewards[:, :, 0] - probe_rewards[:, :, 1])
    probe_actions = raw['probe_actions']
    probe_mapping = np.broadcast_to(raw['action_to_role'][None, None, None, :, None, None, :],
                                    (*probe_actions.shape, 3))
    probe_roles = np.take_along_axis(probe_mapping, probe_actions[..., None], axis=-1)[..., 0]
    for player in (0, 1):
        for role, role_name in enumerate(('invest', 'harvest', 'signal')):
            shares = (probe_roles[..., player] == role).mean(axis=-1)
            for index, method in enumerate(PROBE):
                add(f'controlled_probe_role:player{player}_{role_name}:{method}', shares[:, :, index])
    probe_joint = (probe_roles[..., 0] == 0) & (probe_roles[..., 1] == 0)
    for index, method in enumerate(PROBE):
        add(f'controlled_probe_joint_invest:{method}', probe_joint[:, :, index].mean(axis=-1))

    # A shared action permutation determines roles for both players in a game.
    actions = raw['live_actions']
    mapping = np.broadcast_to(raw['action_to_role'][None, None, None, :, None, None, :],
                              (*actions.shape, 3))
    roles = np.take_along_axis(mapping, actions[..., None], axis=-1)[..., 0]
    for start, stop, label in ((0, 8, 'probe'), (8, 96, 'tail')):
        part = roles[..., start:stop, :]
        for player in (0, 1):
            for role, role_name in enumerate(('invest', 'harvest', 'signal')):
                shares = (part[..., player] == role).mean(axis=-1)
                for index, method in enumerate(LIVE):
                    add(f'role_{label}:player{player}_{role_name}:{method}', shares[:, :, index])
        joint_invest = (part[..., 0] == 0) & (part[..., 1] == 0)
        # Opportunities depend on the previous turns, including across handoff.
        all_joint = (roles[..., 0] == 0) & (roles[..., 1] == 0)
        first = max(start, 2)
        current = all_joint[..., first:stop]
        lag1 = current & all_joint[..., first - 1:stop - 1]
        lag2 = current & all_joint[..., first - 2:stop - 2]
        both_lags = lag1 & all_joint[..., first - 2:stop - 2]
        for index, method in enumerate(LIVE):
            add(f'joint_invest_{label}:{method}', joint_invest[:, :, index].mean(axis=-1))
            add(f'current_and_lag1_invest_{label}:{method}', lag1[:, :, index].mean(axis=-1))
            add(f'current_and_lag2_invest_{label}:{method}', lag2[:, :, index].mean(axis=-1))
            add(f'current_and_both_lags_invest_{label}:{method}', both_lags[:, :, index].mean(axis=-1))
    return metrics


def load(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    paths = sorted(root.glob('block_*.json'))
    if len(paths) != complete['blocks'] or complete['manifest_sha256'] != digest(root / 'manifest.json'):
        raise ValueError('completed block count or manifest hash does not match receipts')
    if manifest['blocks'] != list(range(len(paths))):
        raise ValueError('manifest must describe the same contiguous blocks')
    records, blocks = [], []
    seen_assignments = set()
    names = None
    for expected, receipt in enumerate(paths):
        record = json.loads(receipt.read_text())
        if record['block'] != expected or receipt.stem != f'block_{expected:03d}':
            raise ValueError(f'noncontiguous block identity {receipt}')
        assignment = (record['master_seed'], record['seed_block'])
        if assignment in seen_assignments:
            raise ValueError(f'duplicate training block assignment in {receipt}')
        seen_assignments.add(assignment)
        path = receipt.with_suffix('.npz')
        if digest(path) != record['raw_sha256']:
            raise ValueError(f'raw checksum failure {path}')
        with np.load(path, allow_pickle=False) as raw:
            current = block_metrics(raw)
        if names is not None and set(current) != names:
            raise ValueError(f'inconsistent metrics in {path}')
        names = set(current)
        records.append(record)
        blocks.append(current)
    return manifest, records, {name: np.stack([block[name] for block in blocks]) for name in blocks[0]}


def _contrast_rows(values):
    """Yield series of paired block values for every predeclared intervention."""
    for name, values_by_block in values.items():
        for fi, family in enumerate(('all', *FAMILIES)):
            x = values_by_block[..., fi]
            for li, lag in enumerate(LAGS):
                for di, distance in enumerate(DISTANCES):
                    yield name, family, f'lag{lag}_distance{distance:g}', x[:, li, di]
                yield name, family, f'lag{lag}_distance0.6_minus_0', x[:, li, 2] - x[:, li, 0]
                yield name, family, f'lag{lag}_distance0.3_minus_0', x[:, li, 1] - x[:, li, 0]
            for di, distance in enumerate(DISTANCES):
                yield name, family, f'lag1_minus_0_distance{distance:g}', x[:, 1, di] - x[:, 0, di]
            yield (name, family, 'distance0.6_minus_0_lag1_minus_lag0',
                   (x[:, 1, 2] - x[:, 1, 0]) - (x[:, 0, 2] - x[:, 0, 0]))


def summarize(records, values, output, evidence=None):
    strata = bootstrap_indices(records)
    master_seeds = sorted({record['master_seed'] for record in records})
    groups = {seed: [i for i, record in enumerate(records) if record['master_seed'] == seed]
              for seed in master_seeds}
    group_draws = {seed: bootstrap_indices([records[i] for i in indices])
                   for seed, indices in groups.items()}
    summary, by_seed, by_block = [], [], []
    for metric, family, comparison, block in _contrast_rows(values):
        prefix = dict(metric=metric, family=family, comparison=comparison)
        summary.append(dict(**prefix, blocks=len(records), **statistics(block, strata)))
        for seed, indices in groups.items():
            subset = block[indices]
            by_seed.append(dict(**prefix, master_seed=seed, blocks=len(indices),
                                **statistics(subset, group_draws[seed])))
        # Keep a compact table of block values for all primary and outcome contrasts.
        if metric.startswith(('controlled_mean:', 'total_48:', 'total_96:', 'probe:', 'tail_to_')):
            by_block.extend(dict(**prefix, block=record['block'], master_seed=record['master_seed'], value=float(value))
                            for record, value in zip(records, block, strict=True))
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / 'summary.csv', summary)
    _csv(output / 'by_seed.csv', by_seed)
    _csv(output / 'by_block.csv', by_block)
    primary = next(row for row in summary if row['metric'] == 'controlled_mean:full_minus_masked'
                   and row['family'] == 'all' and row['comparison'] == 'lag0_distance0.6_minus_0')
    (output / 'summary.json').write_text(json.dumps({
        'blocks': len(records), 'master_seeds': master_seeds, 'bootstrap_draws': 20000,
        'primary': primary,
        'interpretation': 'Block-stratified percentile intervals are pointwise, not simultaneous.'
    }, indent=2, allow_nan=False) + '\n')
    plot(summary, output)
    plot_seeds(by_seed, output)
    if evidence is not None:
        evidence.mkdir(parents=True, exist_ok=True)
        selected = {
            'controlled_mean:pooled_full', 'controlled_mean:pooled_masked',
            'controlled_mean:zero', 'controlled_mean:full_minus_masked',
            'controlled_mean:full_minus_zero', 'controlled_mean:masked_minus_zero',
            'total_48:teacher_minus_ucb', 'total_48:teacher_minus_student',
            'total_96:teacher_minus_ucb', 'total_96:teacher_minus_student',
            'probe:teacher_minus_ucb', 'probe:teacher_minus_student',
            'tail_to_96:teacher_minus_ucb', 'tail_to_96:teacher_minus_student',
        }
        _csv(evidence / 'summary.csv', [row for row in summary if row['metric'] in selected])
        _csv(evidence / 'by_seed.csv', [row for row in by_seed if row['metric'] in selected])
        for name in ('summary.json', 'outcomes.png', 'primary_by_seed.png'):
            (evidence / name).write_bytes((output / name).read_bytes())


def plot_seeds(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = [row for row in rows if row['metric'] == 'controlled_mean:full_minus_masked'
              and row['family'] == 'all' and row['comparison'] == 'lag0_distance0.6_minus_0']
    if not points or any(row['ci_low'] is None for row in points):
        return
    fig, ax = plt.subplots(figsize=(6, 3.6), layout='constrained')
    for i, row in enumerate(points):
        ax.plot([row['ci_low'], row['ci_high']], [i, i], color='#286090', linewidth=1.7)
        ax.plot(row['mean'], i, 'o', color='#286090')
    ax.axvline(0, color='.5', linewidth=.8)
    ax.set_yticks(range(len(points)), [str(row['master_seed']) for row in points])
    ax.set(xlabel='Change in full − masked assessment reward', ylabel='Master seed',
           title='Primary contrast by master seed')
    ax.grid(axis='x', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / 'primary_by_seed.png', dpi=170)
    plt.close(fig)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def get(metric, family, comparison):
        return next(row for row in rows if row['metric'] == metric and row['family'] == family
                    and row['comparison'] == comparison)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for li, (lag, color) in enumerate(((0, '#286090'), (1, '#bb6b26'))):
        points = [get('controlled_mean:full_minus_masked', 'all', f'lag{lag}_distance{d:g}')
                  for d in DISTANCES]
        axes[0].errorbar(DISTANCES, [p['mean'] for p in points],
                         yerr=[[p['mean'] - p['ci_low'] for p in points],
                               [p['ci_high'] - p['mean'] for p in points]],
                         color=color, marker='o', capsize=3, label=f'Extra lag {lag}')
    axes[0].axhline(0, color='.5', linewidth=.8)
    axes[0].set(xlabel='Transition intervention strength', ylabel='Full minus masked\nmean assessment reward')
    axes[0].legend(frameon=False)
    for method, color, label in (('teacher_minus_ucb', '#bb6b26', 'Teacher − UCB'),
                                 ('teacher_minus_student', '#286090', 'Teacher − Student')):
        points = [get(f'total_96:{method}', 'all', f'lag0_distance{d:g}') for d in DISTANCES]
        axes[1].errorbar(DISTANCES, [p['mean'] for p in points],
                         yerr=[[p['mean'] - p['ci_low'] for p in points],
                               [p['ci_high'] - p['mean'] for p in points]],
                         color=color, marker='o', capsize=3, label=label)
    axes[1].axhline(0, color='.5', linewidth=.8)
    axes[1].set(xlabel='Transition intervention strength', ylabel='96-turn cumulative reward difference')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.18)
    fig.savefig(output / 'outcomes.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='full local analysis directory')
    parser.add_argument('--evidence', type=Path, help='optional compact evidence directory')
    args = parser.parse_args()
    manifest, records, values = load(args.run)
    if len(records) < 2:
        raise ValueError('at least two independent blocks are needed for intervals')
    if args.evidence is not None and args.output.resolve() == args.evidence.resolve():
        parser.error('--evidence must differ from --output')
    summarize(records, values, args.output, args.evidence)
    print(f"Analyzed {len(records)} blocks from {manifest.get('protocol', args.run)}")


if __name__ == '__main__':
    main()
