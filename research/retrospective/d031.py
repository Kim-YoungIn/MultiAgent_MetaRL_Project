"""Retrospective paired comparisons from D-031's frozen, opened evaluation panel."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from research.d031_completion.verify_completion import verify_receipt

METHODS = ('M', 'P', 'O', 'random', 'epsilon', 'balanced')
KS = (0, 3, 6, 12, 24)
FAMILIES = ('contest', 'hidden', 'reciprocity', 'social')
PROFILES = ('steady', 'sarsa')


def load(folder, method):
    verify_receipt(folder / f'{method}.receipt.json')
    with gzip.open(folder / f'{method}.json.gz', 'rt') as stream:
        rows = json.load(stream)
    result = {(r['rule'], r['profile']): r for r in rows if r['split'] == 'fresh_rules'}
    if len(result) != 64 * 3:
        raise ValueError(f'Unexpected fresh rule count: {folder}/{method}')
    return result


def check_raw(folder, method, summaries):
    """Independently recalculate eight-repetition means from a full raw file."""
    groups = defaultdict(list)
    with gzip.open(folder / f'{method}.trials.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['split'] == 'fresh_rules':
                groups[(r['rule'], r['profile'])].append(r)
    assert groups.keys() == summaries.keys()
    for key, trials in groups.items():
        assert sorted(t['rep'] for t in trials) == list(range(8))
        summary = summaries[key]
        np.testing.assert_allclose(summary['auc'], np.mean([t['auc'] for t in trials]), atol=1e-12, rtol=0)
        for field in ('curve', 'fixed_budget'):
            if field not in summary:
                continue
            for i, point in enumerate(summary[field]):
                np.testing.assert_allclose(point['return'], np.mean([t[field][i]['return'] for t in trials]), atol=1e-12, rtol=0)
                np.testing.assert_allclose(point['game'], np.mean([t[field][i]['game'] for t in trials]), atol=1e-12, rtol=0)


def bootstrap(values, indices):
    """Resample catalogs then paired replicas within each chosen catalog."""
    x = np.asarray(values, dtype=float).reshape(5, 5)
    samples = x.ravel()[indices].mean(axis=1)
    return float(x.mean()), *map(float, np.quantile(samples, [.025, .975])), int((x > 0).sum())


def weighted(rows, profiles, families, accessor):
    means = []
    for profile in profiles:
        for family in families:
            group = [accessor(r) for r in rows if r['profile'] == profile and r['family'] == family]
            if not group:
                raise ValueError(f'missing profile/family {profile}/{family}')
            means.append(np.mean(group))
    return float(np.mean(means))


def run(root, out):
    block_data = defaultdict(list)
    rule_data = defaultdict(list)
    reference = None
    for cat in range(1, 6):
        for replica in range(1, 6):
            path = root / 'cells' / 'primary' / f'c{cat}' / f'j{replica}' / 'n3'
            methods = {m: load(path, m) for m in METHODS}
            if reference is None:
                reference = set(methods['M'])
            for method, rows in methods.items():
                if set(rows) != reference:
                    raise ValueError(f'panel differs in {path}/{method}')
                if cat == replica == 1:
                    check_raw(path, method, rows)
            # Rule draws are shared across every policy and trained block.
            for family in FAMILIES:
                rules = sorted({key[0] for key, row in methods['P'].items()
                                if row['profile'] in PROFILES and row['family'] == family})
                if len(rules) != 16:
                    raise ValueError('Expected 16 rules per family')
                for left, right in (('P', 'M'), ('P', 'random'), ('M', 'random')):
                    for metric in ('auc', 'fixed_k24'):
                        differences = []
                        for rule in rules:
                            sample = []
                            for profile in PROFILES:
                                l, r = methods[left][(rule, profile)], methods[right][(rule, profile)]
                                if metric == 'auc':
                                    sample.append(l['auc'] - r['auc'])
                                else:
                                    sample.append(l['fixed_budget'][4]['game'] - r['fixed_budget'][4]['game'])
                            differences.append(np.mean(sample))
                        rule_data[(family, metric, f'{left}_minus_{right}')].append(differences)
            profiles = (*PROFILES, 'sarsa_heldout')
            for p in ('mixed', *profiles):
                selected_profiles = PROFILES if p == 'mixed' else (p,)
                for f in ('all', *FAMILIES):
                    selected_families = FAMILIES if f == 'all' else (f,)
                    for m in METHODS:
                        rows = list(methods[m].values())
                        select = lambda accessor: weighted(rows, selected_profiles, selected_families, accessor)
                        block_data[(p, f, 'adaptation_auc', m, 'all')].append(select(lambda r: r['auc']))
                        for k in KS:
                            index = KS.index(k)
                            block_data[(p, f, 'assessment_return', m, k)].append(select(lambda r: r['curve'][index]['return']))
                            if m in ('M', 'P', 'O', 'random'):
                                block_data[(p, f, 'fixed_game_return', m, k)].append(select(lambda r: r['fixed_budget'][index]['game']))
    contrasts = [('P','M'), ('P','random'), ('M','random'), ('O','random'), ('P','epsilon'), ('P','balanced'), ('P','O')]
    keys = list(block_data)
    for profile, family, metric, method, k in keys:
        for left, right in contrasts:
            if method == left:
                other = (profile, family, metric, right, k)
                if other in block_data:
                    block_data[(profile, family, metric, f'{left}_minus_{right}', k)] = (
                        np.asarray(block_data[(profile, family, metric, left, k)]) -
                        np.asarray(block_data[other])
                    )
    rng = np.random.default_rng(71001)
    cats = rng.integers(0, 5, size=(20000, 5))
    replicas = rng.integers(0, 5, size=(20000, 5, 5))
    indices = (cats[..., None] * 5 + replicas).reshape(20000, 25)
    rows = []
    for (profile, family, metric, method, k), values in block_data.items():
        mean, low, high, positive = bootstrap(values, indices)
        rows.append(dict(profile=profile, family=family, metric=metric, method=method, k=k,
                         mean=mean, ci_low=low, ci_high=high, positive_blocks=positive,
                         blocks=25, bootstrap='catalog_then_replicas'))
    # Compare opponent profiles within each paired training block.
    for method in ('P_minus_M', 'P_minus_random'):
        for family in ('all', *FAMILIES):
            for metric in ('adaptation_auc', 'assessment_return'):
                budgets = ('all',) if metric == 'adaptation_auc' else KS
                for k in budgets:
                    a = np.asarray(block_data[('sarsa', family, metric, method, k)])
                    b = np.asarray(block_data[('steady', family, metric, method, k)])
                    mean, low, high, positive = bootstrap(a - b, indices)
                    rows.append(dict(profile='sarsa_minus_steady', family=family, metric=metric,
                                     method=method, k=k, mean=mean, ci_low=low, ci_high=high,
                                     positive_blocks=positive, blocks=25, bootstrap='catalog_then_replicas'))
    # Conditional intervals vary trained blocks; joint intervals also resample
    # the 16 common rules inside each family for the entire paired comparison.
    for method in ('P_minus_M', 'P_minus_random', 'M_minus_random'):
        for metric, label in (('auc', 'adaptation_auc'), ('fixed_k24', 'fixed_game_return')):
            family_samples = []
            for family in FAMILIES:
                matrix = np.asarray(rule_data[(family, metric, method)])
                assert matrix.shape == (25, 16)
                rule_means = matrix.mean(axis=0)
                rows.append(dict(profile='mixed', family=family, metric=label + '_rule_win_fraction',
                                 method=method, k='all' if metric == 'auc' else 24,
                                 mean=float((rule_means > 0).mean()), ci_low='', ci_high='',
                                 positive_blocks=int((rule_means > 0).sum()), blocks=25,
                                 bootstrap='none_16_shared_rules'))

                rule_rng = np.random.default_rng(17383 + FAMILIES.index(family))
                rule_indices = rule_rng.integers(0, 16, size=(len(indices), 16))
                draws = np.empty(len(indices))
                for start in range(0, len(indices), 500):
                    end = min(start + 500, len(indices))
                    selected = matrix[indices[start:end, :, None], rule_indices[start:end, None, :]]
                    draws[start:end] = selected.mean(axis=(1, 2))
                family_samples.append(draws)
                low, high = np.quantile(draws, [.025, .975])
                rows.append(dict(profile='mixed', family=family, metric=label, method=method,
                                 k='all' if metric == 'auc' else 24, mean=matrix.mean(),
                                 ci_low=low, ci_high=high, positive_blocks=int((matrix.mean(1) > 0).sum()),
                                 blocks=25, bootstrap='catalog_replicas_and_family_rules'))
            values = np.mean(family_samples, axis=0)
            low, high = np.quantile(values, [.025, .975])
            matrices = [np.asarray(rule_data[(family, metric, method)]).mean(1) for family in FAMILIES]
            individual = np.mean(matrices, axis=0)
            rows.append(dict(profile='mixed', family='all', metric=label, method=method,
                             k='all' if metric == 'auc' else 24, mean=individual.mean(),
                             ci_low=low, ci_high=high, positive_blocks=int((individual > 0).sum()),
                             blocks=25, bootstrap='catalog_replicas_and_family_rules'))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} estimates from 25 matched blocks; 150 receipts checked.')
    print('Recomputed all raw repetitions of all six methods in block c1/j1.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('runs/d031_completion_v2/study'))
    parser.add_argument('--output', type=Path, default=Path('research/retrospective/evidence/d031.csv'))
    args = parser.parse_args()
    run(args.study, args.output)
