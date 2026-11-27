"""Reweight the four game families and summarize both players' realized payoffs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.fast_exploration_v2.analyze import bootstrap_indices, file_sha256, statistics, write_csv


FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
METHODS = ('Teacher', 'UCB', 'Student')
PAIRS = ((0, 2), (1, 2), (0, 1))


def read_json(path):
    return json.loads(path.read_text())


def select(raw, conditions):
    return [conditions.index(x) for x in raw]


def panel_block(root, i, ucb_report, source_receipt):
    """Yield common-game tapes in Teacher, UCB, Student order for each native panel."""
    filename = f'block_{i:03d}.npz'
    same_weights = source_receipt['weights_sha256']
    for panel, directory in (('main48', 'fast_exploration_v2_final'), ('horizon96', 'fast_exploration_v2_horizon_final'),
                             ('native96', 'dynamics_transfer_v1')):
        path = root / directory / filename
        receipt = read_json(path.with_suffix('.json'))
        if (receipt['master_seed'] != source_receipt['master_seed'] or receipt['raw_sha256'] != file_sha256(path)
                or (panel != 'main48' and receipt['teacher_sha256'] != same_weights)):
            raise ValueError(f'block mismatch at {path}')
        with np.load(path) as raw:
            families = raw['families']
            roles = raw['action_to_role']
            resets = raw['reset_seeds']
            if panel == 'main48':
                idx = select(('pooled_fixed_total', 'pooled_ucb', 'pooled_greedy'), list(raw['methods']))
                ki = list(raw['ks']).index(8)
                rewards = raw['fixed_rewards'][idx, [ki, ki, 0]]
                actions = raw['fixed_actions'][idx, [ki, ki, 0]]
            elif panel == 'horizon96':
                conditions = list(zip(raw['methods'], raw['ks'], strict=True))
                ti, si = select((('pooled_fixed_total', 8), ('pooled_greedy', 0)), conditions)
                ucb = root / 'fast_exploration_v2_horizon_ucb' / filename
                if file_sha256(ucb) != ucb_report['blocks'][filename]:
                    raise ValueError(f'UCB hash mismatch at {ucb}')
                with np.load(ucb) as extra:
                    for field in ('families', 'reset_seeds', 'game_seeds'):
                        np.testing.assert_array_equal(extra[field], raw[field])
                    k = list(extra['ks']).index(8)
                    reward = raw['rewards']
                    action = raw['actions']
                    rewards = np.stack([reward[ti], extra['rewards'][k], reward[si]])
                    actions = np.stack([action[ti], extra['actions'][k], action[si]])
            else:
                pos = (list(raw['lags']).index(0), list(raw['distances']).index(0.))
                idx = select(('pooled_fixed_total', 'pooled_ucb', 'pooled_greedy'), list(raw['live_methods']))
                rewards = raw['live_rewards'][pos][idx]
                actions = raw['live_actions'][pos][idx]
            assert actions.shape == rewards.shape == (3, len(resets), 48 if panel == 'main48' else 96, 2)
            assert len(resets) == 2 * len(raw['game_seeds'])
            np.testing.assert_array_equal(families[::2], families[1::2])
            assert all(np.sum(families == f) == len(families) // 4 for f in FAMILIES)
            invest = (roles == 0).argmax(axis=-1)
            together = (actions == invest[None, :, None, None]).all(axis=-1)
            yield panel, families[::2], rewards, together


def collect(root):
    trained = root / 'fast_exploration_v2_final'
    horizon = root / 'fast_exploration_v2_horizon_final'
    ucb_report = read_json(root / 'fast_exploration_v2_horizon_ucb/report.json')
    assert file_sha256(horizon / 'manifest.json') == ucb_report['parent_manifest_sha256']
    assert file_sha256(trained / 'manifest.json') == read_json(horizon / 'manifest.json')['trained_source_sha256']
    assert file_sha256(trained / 'manifest.json') == read_json(root / 'dynamics_transfer_v1/manifest.json')['source_manifest_sha256']
    records, metrics = [], defaultdict(list)
    for i in range(36):
        record = read_json(trained / f'block_{i:03d}.json')
        assert record['block'] == i
        records.append(record)
        for panel, families, rewards, together in panel_block(root, i, ucb_report, record):
            h = rewards.shape[-2]
            for t in (8, 48, 96):
                if t > h:
                    continue
                by_game = rewards[:, :, :t].sum(axis=2).reshape(3, len(families), 2, 2).mean(axis=2)
                joint = together[:, :, :t].mean(axis=2).reshape(3, len(families), 2).mean(axis=2)
                for left, right in PAIRS:
                    name = f'{METHODS[left]}-{METHODS[right]}'
                    delta = by_game[left] - by_game[right]
                    both_up = (delta > 1e-10).all(axis=1)
                    both_down = (delta < -1e-10).all(axis=1)
                    columns = dict(focal=delta[:, 0], counterpart=delta[:, 1], sum=delta.sum(axis=1),
                                   both_up=both_up, both_down=both_down, joint_invest=joint[left] - joint[right])
                    for family in ('all', *FAMILIES):
                        mask = np.ones(len(families), dtype=bool) if family == 'all' else families == family
                        for measure, values in columns.items():
                            metrics[panel, t, name, family, measure].append(values[mask].mean())
    return records, metrics


def run(root, output):
    records, metrics = collect(root)
    indices = bootstrap_indices(records)
    rows, mix = [], []
    for (panel, t, name, group, measure), values in metrics.items():
        rows.append(dict(panel=panel, turns=t, contrast=name, family=group, measure=measure,
                         **statistics(values, indices)))
        if group != 'all' or measure != 'focal' or t == 8:
            continue
        matrix = np.array([metrics[panel, t, name, f, measure] for f in FAMILIES])
        np.testing.assert_allclose(matrix.mean(axis=0), values, atol=1e-12)
        for i, family in enumerate(FAMILIES):
            others = np.delete(matrix, i, axis=0).mean(axis=0)
            for weight in (0., .1, .25, .5, .75, 1.):
                contrast = weight * matrix[i] + (1 - weight) * others
                mix.append(dict(panel=panel, turns=t, contrast=name, varied_family=family, share=weight,
                                **statistics(contrast, indices)))
    output.mkdir(exist_ok=True, parents=True)
    write_csv(output / 'joint_returns.csv', rows)
    write_csv(output / 'family_weights.csv', mix)
    plot(rows, mix, output)


def plot(rows, mixes, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    contrasts = ('Teacher-Student', 'UCB-Student', 'Teacher-UCB')
    for i, name in enumerate(contrasts):
        for j, (outcome, color) in enumerate((('focal', '#154c70'), ('counterpart', '#ac542e'))):
            point = next(r for r in rows if (r['panel'], r['turns'], r['family'], r['contrast'], r['measure'])
                         == ('horizon96', 96, 'all', name, outcome))
            pos = i + (j - .5) * .19
            left.plot([point['ci_low'], point['ci_high']], [pos, pos], color=color, lw=2)
            left.plot(point['mean'], pos, 'o', color=color, label=outcome.capitalize() if i == 0 else None)
    left.set_yticks(range(len(contrasts)), [s.replace('-', ' − ') for s in contrasts])
    left.invert_yaxis()
    left.axvline(0, lw=.8, color='0.4')
    left.set(xlabel='96-turn cumulative reward difference', title='Who receives the reward?')
    left.legend(frameon=False)
    for panel, label, color in (('horizon96', '96-turn panel', '#154c70'), ('native96', 'Native transfer panel', '#ac542e')):
        data = [r for r in mixes if (r['panel'], r['turns'], r['varied_family'], r['contrast'])
                == (panel, 96, 'reciprocity', 'Teacher-Student')]
        x = [r['share'] for r in data]
        right.plot(x, [r['mean'] for r in data], marker='o', label=label, color=color)
        right.fill_between(x, [r['ci_low'] for r in data], [r['ci_high'] for r in data], color=color, alpha=.12)
    right.axvline(.25, color='0.55', ls=':', lw=1)
    right.axhline(0., color='0.4', lw=.8)
    right.set(xlabel='Reciprocity share in test mixture', ylabel='Teacher − Student, focal 96-turn return',
              title='Same policies, different test mix')
    right.legend(frameon=False)
    for ax in (left, right):
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / 'joint_returns.png', dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, default=Path('research/retrospective/evidence'))
    args = parser.parse_args()
    run(args.runs, args.out)
