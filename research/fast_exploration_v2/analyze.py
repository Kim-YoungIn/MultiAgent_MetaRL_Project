"""Analyze completed independent blocks of the Fast Exploration v2 study."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_blocks(root):
    """Load only complete, hash-verified block artifacts."""
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['config']
    count = config['blocks']
    records = []
    seen = set()
    for path in sorted(root.glob('block_*.json')):
        record = json.loads(path.read_text())
        index = record['block']
        if index in seen or not 0 <= index < count or path.stem != f'block_{index:03d}':
            raise ValueError(f'invalid block identity at {path}')
        seen.add(index)
        expected_master = manifest['master_seeds'][index // config['blocks_per_master']]
        if record['master_seed'] != expected_master or record['seed_block'] != index % config['blocks_per_master']:
            raise ValueError(f'inconsistent block assignment at {path}')
        for suffix, key in (('.npz', 'raw_sha256'), ('.pt', 'weights_sha256')):
            if file_sha256(path.with_suffix(suffix)) != record[key]:
                raise ValueError(f'artifact checksum mismatch at {path.with_suffix(suffix)}')
        records.append((record, path.with_suffix('.npz')))
    if not records:
        raise ValueError('no complete blocks')
    return manifest, records


def bootstrap_indices(records, *, draws=20000, seed=882734):
    """Preserve the observed block counts within each master seed."""
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record['master_seed']].append(i)
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(indices, size=(draws, len(indices)), replace=True)
        for indices in groups.values()
    ], axis=1)


def statistics(values, indices):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('block estimates must be finite scalar values')
    if len(values) < 2:
        return dict(mean=float(values.mean()), block_sd=None, ci_low=None, ci_high=None)
    low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), block_sd=float(values.std(ddof=1)),
                ci_low=float(low), ci_high=float(high))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'no rows for {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows, output):
    """Render absolute adaptation curves with pointwise block intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curves = [r for r in rows if r['metric'] == 'assessment_mean'
              and r['regime'] == 'controlled' and r['prior'] == 'pooled'
              and r['family'] == 'all']
    if not curves:
        return
    methods = list(dict.fromkeys(r['method'] for r in curves))
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    for method in methods:
        selected = sorted((r for r in curves if r['method'] == method), key=lambda r: r['k'])
        x = np.array([r['k'] for r in selected])
        y = np.array([r['mean'] for r in selected])
        line, = ax.plot(x, y, marker='o', label=method)
        if selected[0]['ci_low'] is not None:
            ax.fill_between(x, [r['ci_low'] for r in selected],
                            [r['ci_high'] for r in selected], color=line.get_color(), alpha=.13)
    ax.set(xlabel='Exploration turns (K)', ylabel='Mean assessment reward',
           title='Pooled prior, frozen assessment')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / 'adaptation_pooled_controlled.png', dpi=160)
    plt.close(fig)

    fixed = [r for r in rows if r['metric'] == 'fixed_total_return'
             and r['regime'] == 'fixed' and r['family'] == 'all'
             and r['method'] in ('pooled_fixed_value', 'pooled_fixed_total', 'pooled_greedy', 'pooled_random')]
    if not fixed:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    for method in dict.fromkeys(r['method'] for r in fixed):
        selected = sorted((r for r in fixed if r['method'] == method), key=lambda r: r['k'])
        x = [r['k'] for r in selected]
        line, = ax.plot(x, [r['mean'] for r in selected], marker='o', label=method)
        if selected[0]['ci_low'] is not None:
            ax.fill_between(x, [r['ci_low'] for r in selected], [r['ci_high'] for r in selected],
                            color=line.get_color(), alpha=.13)
    ax.set(xlabel='Exploration turns (K)', ylabel='Total reward in fixed interaction budget',
           title='Continuous fixed-budget deployment')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / 'fixed_budget.png', dpi=160)
    plt.close(fig)


def add_value(store, key, value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'non-finite metric {key}')
    store[key].append(value)


def action_summary(store, key, actions, roles, k):
    for player, index in (('focal', 0), ('opponent', 1)):
        selected = actions[..., index]
        if selected.shape[1] > 1:
            add_value(store, (*key, f'{player}_repeat', k), (selected[:, 1:] == selected[:, :-1]).mean())
        chosen_roles = np.take_along_axis(roles, selected, axis=1)
        for role, name in enumerate(('invest', 'harvest', 'signal')):
            add_value(store, (*key, f'{player}_{name}', k), (chosen_roles == role).mean())


def collect_block(data, config, store):
    methods = tuple(str(m) for m in data['methods'])
    ks = np.asarray(data['ks'], dtype=int)
    if len(set(methods)) != len(methods) or len(ks) < 2 or ks[0] != 0 or np.any(np.diff(ks) <= 0):
        raise ValueError('unexpected method labels or exploration budgets')
    if 'ks' in config and ks.tolist() != config['ks']:
        raise ValueError('budget grid differs from manifest')
    samples = len(data['reset_seeds'])
    families = np.asarray(data['families'])
    roles = np.asarray(data['action_to_role'])
    if families.shape != (samples,) or roles.shape != (samples, 3):
        raise ValueError('invalid sample metadata')
    if not np.array_equal(np.sort(roles, axis=1), np.broadcast_to(np.arange(3), roles.shape)):
        raise ValueError('invalid action-to-role mapping')
    selection = {'all': np.ones(samples, dtype=bool)}
    selection.update({str(family): families == family for family in np.unique(families)})
    probe = data['probe_rewards']
    probe_actions = data['probe_actions']
    expected_prefix = (len(methods), len(ks), samples)
    if probe.shape != (*expected_prefix, int(ks[-1]), 2) or probe_actions.shape != probe.shape:
        raise ValueError('invalid exploration data shape')
    outcomes, behavior = {}, {}
    for regime in ('controlled', 'natural', 'frozen', 'fixed'):
        rewards = data[f'{regime}_rewards']
        actions = data[f'{regime}_actions']
        if rewards.ndim != 5 or rewards.shape[:3] != expected_prefix or rewards.shape[-1] != 2 or actions.shape != rewards.shape:
            raise ValueError(f'invalid {regime} data shape')
        if not np.isfinite(rewards).all() or not np.isin(actions, [0, 1, 2]).all():
            raise ValueError(f'invalid {regime} reward or action')
        outcomes[regime] = rewards
        if regime in ('natural', 'fixed'):
            behavior[regime] = actions

    query = data['query_values'] if 'query_values' in data else None
    if query is not None and (query.shape != expected_prefix or not np.isfinite(query).all()):
        raise ValueError('invalid query values')
    intervention_methods = tuple(str(m) for m in data['intervention_methods']) if 'intervention_methods' in data else ()
    interventions = {}
    for regime in ('retain_both', 'reset_student'):
        if regime + '_rewards' in data:
            reward = data[regime + '_rewards']
            action = data[regime + '_actions']
            expected = (len(intervention_methods), len(ks), samples, outcomes['controlled'].shape[3], 2)
            if reward.shape != expected or action.shape != expected:
                raise ValueError(f'invalid {regime} intervention shape')
            if not np.isfinite(reward).all() or not np.isin(action, [0, 1, 2]).all():
                raise ValueError(f'invalid {regime} intervention reward or action')
            interventions[regime] = reward
    for ki, k in enumerate(ks):
        if (not np.isfinite(probe[:, ki, :, :k]).all() or
            not np.isnan(probe[:, ki, :, k:]).all() or
            not np.isin(probe_actions[:, ki, :, :k], [0, 1, 2]).all() or
            np.any(probe_actions[:, ki, :, k:] != -1)):
            raise ValueError(f'invalid exploration padding at K={k}')
    for mi, method in enumerate(methods):
        prior = method.split('_', 1)[0]
        for family, mask in selection.items():
            for regime, rewards in outcomes.items():
                focal = rewards[mi, :, mask, :, 0]
                # NumPy advanced indexing moves the sample axis to the front.
                if focal.shape != (int(mask.sum()), len(ks), rewards.shape[3]):
                    raise ValueError(f'unexpected {regime} sample indexing')
                curve = focal.mean(axis=(0, 2))
                metric = 'fixed_total_return' if regime == 'fixed' else 'assessment_mean'
                if regime == 'fixed':
                    curve = focal.sum(axis=2).mean(axis=0)
                else:
                    auc = np.trapezoid(curve, ks) / ks[-1]
                    add_value(store, (regime, prior, method, family, 'absolute_auc', 'all'), auc)
                    add_value(store, (regime, prior, method, family, 'adaptation_gain_auc', 'all'), auc - curve[0])
                    for ki, k in enumerate(ks):
                        add_value(store, (regime, prior, method, family, 'first_assessment', int(k)),
                                  focal[:, ki, 0].mean())
                for ki, k in enumerate(ks):
                    add_value(store, (regime, prior, method, family, metric, int(k)), curve[ki])
                    if ki and regime != 'fixed':
                        slope = (curve[ki] - curve[ki-1]) / (ks[ki]-ks[ki-1])
                        add_value(store, (regime, prior, method, family, 'slope', int(k)), slope)
                    if regime == 'natural':
                        probe_sum = probe[mi, ki, mask, :k, 0].sum(axis=1).mean()
                        add_value(store, (regime, prior, method, family, 'probe_return', int(k)), probe_sum)
                        add_value(store, (regime, prior, method, family, 'total_return', int(k)),
                                  probe_sum + focal[:, ki].sum(axis=1).mean())
                    elif regime == 'fixed':
                        for label, turns in (('probe', slice(0, k)), ('assessment', slice(k, None))):
                            component = focal[:, ki, turns].sum(axis=1).mean()
                            add_value(store, (regime, prior, method, family, f'fixed_{label}_return', int(k)),
                                      component)
            if query is not None:
                for ki, k in enumerate(ks):
                    add_value(store, ('query', prior, method, family, 'query_value', int(k)),
                              query[mi, ki, mask].mean())

            if method in intervention_methods and len(interventions) == 2:
                ii = intervention_methods.index(method)
                retained = outcomes['controlled'][mi, :, mask, :, 0].mean(axis=(0, 2))
                both = interventions['retain_both'][ii, :, mask, :, 0].mean(axis=(0, 2))
                opponent = interventions['reset_student'][ii, :, mask, :, 0].mean(axis=(0, 2))
                baseline = retained[0]
                interaction = both - retained - opponent + baseline
                effects = {'student_state': retained - baseline,
                           'opponent_state': opponent - baseline,
                           'student_opponent_interaction': interaction}
                for name, values in effects.items():
                    add_value(store, ('intervention', prior, method, family, name + '_auc', 'all'),
                              np.trapezoid(values, ks) / ks[-1])
                    for ki, k in enumerate(ks):
                        add_value(store, ('intervention', prior, method, family, name, int(k)), values[ki])

            for ki, k in enumerate(ks):
                contexts = [('assessment', behavior['natural'][mi, ki, mask]),
                            ('fixed_assessment', behavior['fixed'][mi, ki, mask, k:])]
                if k:
                    contexts.extend((('probe', probe_actions[mi, ki, mask, :k]),
                                     ('fixed_probe', behavior['fixed'][mi, ki, mask, :k])))
                for phase, actions in contexts:
                    action_summary(store, (phase, prior, method, family), actions, roles[mask], int(k))


def paired_metrics(metrics, n_blocks):
    pairs = {}

    def add(label, left, right):
        if left not in metrics or right not in metrics:
            return
        x, y = np.asarray(metrics[left]), np.asarray(metrics[right])
        if x.shape != (n_blocks,) or y.shape != (n_blocks,):
            raise ValueError(f'incomplete observations for {label}')
        pairs[label] = x - y

    families = sorted({key[3] for key in metrics})
    for family in families:
        for regime in ('controlled', 'natural', 'frozen'):
            for prior in ('pooled', 'ips'):
                prefix = (regime, prior)
                full, masked = f'{prior}_full', f'{prior}_masked'
                for opponent in (masked, f'{prior}_random', f'{prior}_greedy', f'{prior}_ucb',
                                 'pooled_immediate' if prior == 'pooled' else masked):
                    left = (*prefix, full, family, 'absolute_auc', 'all')
                    right = (*prefix, opponent, family, 'absolute_auc', 'all')
                    add((regime, family, 'absolute_auc', f'{full}_minus_{opponent}', 'all'), left, right)
                    for key in metrics:
                        if key[:5] != (regime, prior, full, family, 'assessment_mean'):
                            continue
                        k = key[5]
                        add((regime, family, 'assessment_mean', f'{full}_minus_{opponent}', k),
                            key, (regime, prior, opponent, family, 'assessment_mean', k))
                key = (regime, family, 'absolute_auc', f'{prior}_full_minus_{prior}_masked', 'all')
                if regime == 'controlled' and prior == 'pooled':
                    pooled_key = key
                if regime == 'controlled' and prior == 'ips':
                    ips_key = key
        if pooled_key in pairs and ips_key in pairs:
            pairs[('controlled', family, 'prior_feedback_interaction',
                   'pooled_full_minus_masked_less_ips_full_minus_masked', 'all')] = pairs[pooled_key] - pairs[ips_key]
        for metric in ('student_state_auc', 'opponent_state_auc', 'student_opponent_interaction_auc'):
            add(('intervention', family, metric, 'pooled_full_minus_pooled_masked', 'all'),
                ('intervention', 'pooled', 'pooled_full', family, metric, 'all'),
                ('intervention', 'pooled', 'pooled_masked', family, metric, 'all'))
        for key in metrics:
            regime, prior, method, group, metric, k = key
            if (regime, prior, method, group, metric) == ('query', 'pooled', 'pooled_full', family, 'query_value'):
                add(('query', family, metric, 'pooled_full_minus_pooled_masked', k), key,
                    ('query', 'pooled', 'pooled_masked', family, metric, k))
        for prior in ('pooled', 'ips'):
            frozen_key = ('frozen', family, 'absolute_auc', f'{prior}_full_minus_{prior}_masked', 'all')
            controlled_key = ('controlled', family, 'absolute_auc', f'{prior}_full_minus_{prior}_masked', 'all')
            if frozen_key in pairs and controlled_key in pairs:
                pairs[('controlled_vs_frozen_probe', family, 'opponent_learning_interaction',
                       f'{prior}_feedback_gap_difference', 'all')] = pairs[controlled_key] - pairs[frozen_key]
        fixed_metrics = ('fixed_total_return', 'fixed_probe_return', 'fixed_assessment_return')
        for method, opponent in (('pooled_fixed_total', 'pooled_fixed_value'),
                                 ('pooled_fixed_total', 'pooled_greedy'),
                                 ('pooled_full', 'pooled_greedy'),
                                 ('pooled_immediate', 'pooled_greedy')):
            for key in metrics:
                regime, prior, name, group, metric, k = key
                if (regime, prior, name, group) != ('fixed', 'pooled', method, family) or metric not in fixed_metrics:
                    continue
                left = key
                right = ('fixed', 'pooled', opponent, family, metric, k)
                add(('fixed', family, metric, f'{method}_minus_{opponent}', k), left, right)
                initial = ('fixed', 'pooled', method, family, metric, 0)
                add(('fixed', family, metric, f'{method}_minus_own_k0', k), left, initial)
    return pairs


def analyze(root, output):
    manifest, loaded = load_blocks(root)
    records = [record for record, _ in loaded]
    collected = defaultdict(list)
    for record, path in loaded:
        with np.load(path, allow_pickle=False) as data:
            collect_block(data, manifest['config'], collected)
    for key, values in collected.items():
        if len(values) != len(records):
            raise ValueError(f'missing family or metric in blocks for {key}')
    indices = bootstrap_indices(records)
    result_rows, block_rows, pair_rows, seed_rows = [], [], [], []
    for (regime, prior, method, family, metric, k), values in sorted(collected.items(), key=lambda item: str(item[0])):
        result_rows.append(dict(regime=regime, prior=prior, method=method, family=family, metric=metric,
                                k=k, blocks=len(records), **statistics(values, indices)))
        for record, value in zip(records, values, strict=True):
            block_rows.append(dict(block=record['block'], master_seed=record['master_seed'],
                                   seed_block=record['seed_block'], regime=regime, prior=prior, method=method,
                                   family=family, metric=metric, k=k, value=value))
    pairs = paired_metrics(collected, len(records))
    for (regime, family, metric, contrast, k), values in sorted(pairs.items(), key=lambda item: str(item[0])):
        pair_rows.append(dict(regime=regime, family=family, metric=metric, contrast=contrast, k=k,
                              blocks=len(records), **statistics(values, indices)))
        for seed in manifest['master_seeds']:
            chosen = [i for i, record in enumerate(records) if record['master_seed'] == seed]
            if chosen:
                subset = np.asarray(values)[chosen]
                rng = np.random.default_rng(882734)
                seed_indices = rng.integers(0, len(chosen), size=(20000, len(chosen)))
                seed_rows.append(dict(master_seed=seed, regime=regime, family=family, metric=metric,
                                      contrast=contrast, k=k, blocks=len(chosen),
                                      **statistics(subset, seed_indices)))
    if not pair_rows:
        raise ValueError('no paired contrasts present')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'metrics.csv', result_rows)
    write_csv(output / 'paired.csv', pair_rows)
    write_csv(output / 'paired_by_seed.csv', seed_rows)
    write_csv(output / 'block_metrics.csv', block_rows)
    render(result_rows, output)
    required = manifest['config']['blocks']
    completed = len(records)
    primary = [row for row in pair_rows if row['regime'] == 'controlled' and row['family'] == 'all'
               and row['contrast'] == 'pooled_full_minus_pooled_masked' and row['metric'] == 'absolute_auc']
    report = dict(blocks=completed, planned_blocks=required, complete=completed == required,
                  master_seed_counts={str(seed): sum(r['master_seed'] == seed for r in records)
                                      for seed in manifest['master_seeds']},
                  bootstrap=dict(draws=20000, seed=882734, unit='independent block',
                                 stratified_by='master_seed', level=.95,
                                 partial_run='Preserves observed seed counts'),
                  primary=primary[0] if primary else None,
                  source_manifest_sha256=file_sha256(root/'manifest.json'))
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))


if __name__ == '__main__':
    main()
