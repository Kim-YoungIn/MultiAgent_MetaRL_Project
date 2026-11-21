"""Paired D-031 analysis over completed model evaluation artifacts."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import numpy as np
from functools import lru_cache


def _mean(rows):
    return statistics.mean(rows) if rows else None


def _weighted(rows):
    """Equal weight for profiles, families, and rules within each family."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['profile']][row['family']].append(row['auc'])
    return _mean([_mean([_mean(values) for values in families.values()])
                  for families in groups.values()])


def _key(row):
    return row['split'], row['rule'], row['profile']


def _rank(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return None
    x, y = _rank(x), _rank(y)
    mx, my = statistics.mean(x), statistics.mean(y)
    denom = math.sqrt(sum((v-mx)**2 for v in x) * sum((v-my)**2 for v in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / denom if denom else None


@lru_cache(maxsize=256)
def _bootstrap_weights(keys, draws, seed):
    catalogs = sorted({catalog for catalog, _ in keys})
    by_catalog = {c: [i for i, (catalog, _) in enumerate(keys) if catalog == c] for c in catalogs}
    rng = np.random.default_rng(seed)
    chosen = rng.integers(len(catalogs), size=(draws, len(catalogs)))
    weights = np.zeros((draws, len(keys)), dtype=float)
    for draw, sample in enumerate(chosen):
        for index in sample:
            members = by_catalog[catalogs[index]]
            selected = rng.choice(members, size=len(members))
            np.add.at(weights[draw], selected, 1.0/(len(catalogs)*len(members)))
    return weights


def _interval(values, *, dev):
    if len(values) < 2:
        return None
    keys = tuple(sorted(values))
    weights = _bootstrap_weights(keys, 1000 if dev else 10000, 71001)
    data = np.asarray([values[key] for key in keys], dtype=float)
    samples = np.sum(weights * data, axis=1)
    return np.quantile(samples, [.025, .975], method='linear').tolist()


def _summarize(values, dev, with_interval=True):
    cats = defaultdict(list)
    for (catalog, replica), value in values.items():
        cats[catalog].append(value)
    means = {str(c): statistics.mean(v) for c, v in cats.items()}
    return {'mean': _mean([statistics.mean(v) for v in cats.values()]), 'confidence_interval': _interval(values, dev=dev) if with_interval else None,
            'n_models': len(values), 'catalog_means': means,
            'leave_one_catalog_out': {str(c): _mean([v for (cc, _), v in values.items() if cc != c])
                                      for c in cats}}


class _ShardReader:
    """Keep only one cell and action-count worth of result rows in memory."""
    def __init__(self):
        self.paths = {}
        self.cache = {}
        self.current = None

    def __setitem__(self, key, path):
        self.paths[key] = path

    def get(self, key, default=None):
        path = self.paths.get(key)
        if path is None:
            return default
        prefix = key[:4]
        if prefix != self.current:
            self.cache.clear()
            self.current = prefix
        if key not in self.cache:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                rows = json.load(stream)
            indexed = {_key(row): row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f'duplicate task rows in {path}')
            self.cache[key] = indexed
        return self.cache[key]

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def items(self):
        for key in self.paths:
            yield key, self[key]


def report(root, cells, *, dev=False, sealed=True):
    from .artifacts import verify_evaluation
    root = Path(root)
    loaded = _ShardReader()
    missing = []
    condition_methods = {}
    for cell in cells:
        condition = cell['condition']
        condition_methods[condition] = ('M', 'P', 'O') if condition == 'primary' else ('M', 'P')
        for n in ((3, 4, 5) if condition in ('n5', 'primary') else (cell['train_n'],)):
            for method in ('M', 'P', 'O', 'initial', 'random', 'epsilon', 'balanced'):
                path = root/'cells'/condition/f"c{cell['catalog']}"/f"j{cell['replica']}"/f'n{n}'/(method+'.json.gz')
                key = (condition, cell['catalog'], cell['replica'], n, method)
                receipt = path.with_name(method + '.receipt.json')
                if not path.exists() or (sealed and not receipt.exists()):
                    if n == cell['train_n'] and method in condition_methods[condition]:
                        missing.append(str(path))
                    continue
                if sealed:
                    verify_evaluation(path.parent, method)
                loaded[key] = path
    result = {'development': dev, 'missing_required_files': missing, 'groups': {}, 'block_contrasts': {},
              'h2_spearman': {}}
    values_by_group = defaultdict(dict)
    for cell in cells:
        condition, catalog, replica = cell['condition'], cell['catalog'], cell['replica']
        ns = (3, 4, 5) if condition in ('n5', 'primary') else (cell['train_n'],)
        for n in ns:
            pair = (condition, catalog, replica, n)
            m = loaded.get((*pair, 'M'))
            if m is None:
                continue
            for method in ('P', 'O', 'initial', 'random', 'epsilon', 'balanced'):
                baseline = loaded.get((*pair, method))
                if baseline is None:
                    continue
                if m.keys() != baseline.keys():
                    raise ValueError(f'task sets differ for {pair} {method}')
                for key in m:
                    if m[key]['zero_hash'] != baseline[key]['zero_hash']:
                        raise ValueError(f'k=0 mismatch for {pair} {key}')
                for split in sorted({key[0] for key in m}):
                    for profiles in (('steady', 'sarsa'), ('steady',), ('sarsa',), ('sarsa_heldout',)):
                        keys = [key for key in m if key[0] == split and key[2] in profiles]
                        if not keys:
                            continue
                        rows = [{**m[key], 'auc': m[key]['auc'] - baseline[key]['auc']} for key in keys]
                        label = f'{condition}/n{n}/{split}/{"mixed" if len(profiles)==2 else profiles[0]}/M-{method}'
                        values_by_group[label][(catalog, replica)] = _weighted(rows)
                        for family in sorted({row['family'] for row in rows}):
                            values_by_group[f'{label}/family/{family}'][(catalog, replica)] = _weighted([row for row in rows if row['family'] == family])
                        for k in sorted({point['k'] for key in keys for point in m[key]['curve']}):
                            points = []
                            for key in keys:
                                ma = next((p for p in m[key]['curve'] if p['k'] == k), None)
                                mb = next((p for p in baseline[key]['curve'] if p['k'] == k), None)
                                if ma is not None and mb is not None:
                                    points.append({**m[key], 'auc': ma['return'] - mb['return']})
                            if points:
                                values_by_group[f'{label}/return/k{k}'][(catalog, replica)] = _weighted(points)
            for method in ('M', 'P', 'O', 'initial', 'random', 'epsilon', 'balanced'):
                source = loaded.get((*pair, method))
                if source is None:
                    continue
                for split in sorted({key[0] for key in source}):
                    for profile_group in (('steady', 'sarsa'), ('steady',), ('sarsa',), ('sarsa_heldout',)):
                        selected = [row for key, row in source.items() if key[0] == split and key[2] in profile_group]
                        if not selected:
                            continue
                        label = f'{condition}/n{n}/{split}/{"mixed" if len(profile_group)==2 else profile_group[0]}/{method}'
                        values_by_group[label][(catalog, replica)] = _weighted(selected)
                        for family in sorted({row['family'] for row in selected}):
                            values_by_group[f'{label}/family/{family}'][(catalog, replica)] = _weighted([row for row in selected if row['family'] == family])
            loss_path = root/'cells'/condition/f'c{catalog}'/f'j{replica}'/f'n{n}'/'losses.json'
            p = loaded.get((*pair, 'P'))
            if loss_path.exists() and p is not None:
                losses = json.loads(loss_path.read_text())
                families = defaultdict(lambda: defaultdict(list))
                for key, row in m.items():
                    if key in p and row['rule'] in losses and row['profile'] in ('steady', 'sarsa'):
                        families[(row['split'], row['family'])][row['rule']].append(row['auc'] - p[key]['auc'])
                for (split, family), rules in families.items():
                    score = _spearman([losses[r] for r in rules], [_mean(v) for v in rules.values()])
                    result['h2_spearman'].setdefault(f'{condition}/n{n}/{split}/{family}', []).append(
                        {'catalog': catalog, 'replica': replica, 'value': score})
    # Preserve the requested curves and fixed-budget return for every method.
    for (condition, catalog, replica, n, method), source in loaded.items():
        for split in sorted({key[0] for key in source}):
            for profile_group in (('steady', 'sarsa'), ('steady',), ('sarsa',), ('sarsa_heldout',)):
                selected = [row for key, row in source.items() if key[0] == split and key[2] in profile_group]
                if not selected:
                    continue
                prefix = f'{condition}/n{n}/{split}/{"mixed" if len(profile_group)==2 else profile_group[0]}/{method}'
                for field in ('return', 'first', 'early3', 'G', 'q_reuse', 'q_revisit', 'explore_mean', 'first_explore_reward', 'opponent_nonuniform_fraction'):
                    for k in sorted({point['k'] for row in selected for point in row['curve']}):
                        points = []
                        for row in selected:
                            point = next((p for p in row['curve'] if p['k'] == k), None)
                            if point is not None and point.get(field) is not None:
                                points.append({**row, 'auc': point[field]})
                        if points:
                            values_by_group[f'{prefix}/{field}/k{k}'][(catalog, replica)] = _weighted(points)
                for k in sorted({point['k'] for row in selected for point in row.get('fixed_budget', [])}):
                    points = []
                    for row in selected:
                        point = next((p for p in row.get('fixed_budget', []) if p['k'] == k), None)
                        if point is not None:
                            points.append({**row, 'auc': point['game']})
                    if points:
                        values_by_group[f'{prefix}/fixed_budget/k{k}'][(catalog, replica)] = _weighted(points)
    result['groups'] = {name: _summarize(values, dev, with_interval='/M-' in name) for name, values in values_by_group.items()}
    primary_key = 'primary/n3/fresh_rules/mixed/M-P'
    primary_values = values_by_group.get(primary_key, {})
    expected = {(c['catalog'], c['replica']) for c in cells if c['condition'] == 'primary'}
    complete = (bool(expected) and set(primary_values) == expected
                and not any('/primary/' in p for p in missing))
    if not dev:
        complete = complete and len(expected) == 25 and len({c for c, _ in expected}) == 5
        for catalog, replica in expected:
            for method in ('M', 'P', 'O'):
                rows = loaded.get(('primary', catalog, replica, 3, method), {})
                fresh = [row for row in rows.values() if row['split'] == 'fresh_rules'
                         and row['profile'] in ('steady', 'sarsa')]
                if len(fresh) != 128 or len({r['rule'] for r in fresh}) != 64:
                    complete = False
    primary = _summarize(primary_values, dev)
    primary['complete'] = complete
    primary['claim'] = 'development only' if dev else 'incomplete'
    if complete and not dev:
        lo, hi = primary['confidence_interval']
        primary['claim'] = 'positive' if lo > 0 else ('negative' if hi < 0 else 'inconclusive')
    result['primary'] = primary
    if complete:
        # Resample evaluation rules once per replicate and share them across all models.
        # Training catalog and seed sampling follow the primary hierarchy.
        rule_scores = {}
        families = defaultdict(set)
        for catalog, replica in expected:
            m = loaded[('primary', catalog, replica, 3, 'M')]
            p = loaded[('primary', catalog, replica, 3, 'P')]
            grouped = defaultdict(list)
            for key, row in m.items():
                if row['split'] == 'fresh_rules' and row['profile'] in ('steady', 'sarsa'):
                    grouped[(row['family'], row['rule'])].append(row['auc'] - p[key]['auc'])
                    families[row['family']].add(row['rule'])
            rule_scores[(catalog, replica)] = {key: statistics.mean(value) for key, value in grouped.items()}
        family_rules = {family: sorted(rules) for family, rules in families.items()}
        keys = tuple(sorted(expected))
        draws = 1000 if dev else 10000
        model_weights = _bootstrap_weights(keys, draws, 71001)
        rng = np.random.default_rng(71002)
        family_arrays = []
        for family, rules in family_rules.items():
            matrix = np.asarray([[rule_scores[key][(family, rule)] for rule in rules] for key in keys])
            chosen_rules = rng.integers(len(rules), size=(draws, len(rules)))
            family_arrays.append((matrix, chosen_rules))
        samples = np.empty(draws)
        for start in range(0, draws, 256):
            end = min(start + 256, draws)
            weights = model_weights[start:end]
            estimates = [(matrix[:, chosen[start:end]].mean(axis=2).T * weights).sum(axis=1)
                         for matrix, chosen in family_arrays]
            samples[start:end] = np.mean(estimates, axis=0)
        primary['rule_sampling_sensitivity_interval'] = np.quantile(samples, [.025, .975], method='linear').tolist()
    for label, low, high in [('rule_count', 'rules_low', 'rules_high'), ('low_to_primary', 'rules_low', 'primary'),
                             ('primary_to_high', 'primary', 'rules_high'), ('family', 'focused', 'primary')]:
        left = values_by_group.get(f'{low}/n3/fresh_rules/mixed/M-P', {})
        right = values_by_group.get(f'{high}/n3/fresh_rules/mixed/M-P', {})
        paired = {key: right[key]-left[key] for key in left.keys() & right.keys()}
        result['block_contrasts'][label] = _summarize(paired, dev)
    q = values_by_group.get('q_only/n3/fresh_rules/mixed/M-P', {})
    s = values_by_group.get('sarsa_only/n3/fresh_rules/mixed/M-P', {})
    mixed = values_by_group.get(primary_key, {})
    paired = {key: mixed[key]-(q[key]+s[key])/2 for key in mixed.keys() & q.keys() & s.keys()}
    result['block_contrasts']['opponent_diversity'] = _summarize(paired, dev)
    return result
