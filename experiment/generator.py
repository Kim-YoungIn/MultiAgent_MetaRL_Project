"""D-031 v2 logical game catalog.

The hidden-action term receives an independently seeded relative coefficient.
All payoff tables are still compiled from relational expressions.
"""
from dataclasses import replace
from functools import lru_cache
from itertools import permutations, product
import random

from mamrl.diverse_catalog import (
    BASE_FAMILIES, _entry, _family, _source, _stream_seed, _validate_config,
)
from mamrl.relational_rules import generate_relational_rule
from mamrl.rule_metrics import common_values, rounded_values, stable_digest


@lru_cache(maxsize=None)
def _permutation_offsets(n):
    points = tuple(product(range(n), repeat=4))
    return tuple(tuple(((perm[p]*n + perm[q])*n + perm[a])*n + perm[b]
                       for p, q, a, b in points)
                 for perm in permutations(range(n)))


def canonical_values(values, n):
    if len(values) != n**4:
        raise ValueError('expected n^4 common payoff entries')
    rounded = rounded_values(values)
    return min(tuple(rounded[i] for i in offsets) for offsets in _permutation_offsets(n))


def candidate(n, seed, family):
    if family == 'hidden':
        source = generate_relational_rule(n=n, seed=seed, family=family)
        multiplier = round(random.Random(f'diverse-hidden-v2/{seed}').uniform(.5, 1.5), 8)
        base, conditional = source.root.children
        yes, no = conditional.children
        conditional = replace(conditional, children=(replace(yes, value=yes.value * multiplier), no))
        source = replace(source, root=replace(source.root, children=(base, conditional)))
    else:
        source, multiplier = _source(n, seed, family)
    values = common_values(source)
    low, high = min(values), max(values)
    if high - low < 1e-10:
        return None
    factor = 2 / (high - low)
    shift = -1 - 2 * low / (high - low)
    played = replace(source, scale=source.scale * factor, shift=source.shift * factor + shift)
    payoffs = common_values(played)
    return source, played, multiplier, factor, shift, payoffs, stable_digest(canonical_values(payoffs, n))


def catalog_rows(*, n, seed, diversity, train_count, fresh_count, structural_count):
    counts = {'train': train_count, 'fresh_rules': fresh_count, 'structural': structural_count}
    _validate_config(n, seed, counts, diversity)
    seen, generated = set(), {}
    for split in ('fresh_rules', 'structural', 'train'):
        rng = random.Random(_stream_seed(seed, n, split))
        rows, attempts = [], 0
        for index in range(counts[split]):
            family = _family(split, index, diversity)
            for _ in range(20000):
                attempts += 1
                source_seed = rng.getrandbits(63)
                item = candidate(n, source_seed, family)
                if item is None or item[-1] in seen:
                    continue
                seen.add(item[-1])
                rows.append(_entry(split, index, source_seed, attempts, family, item))
                break
            else:
                raise RuntimeError(f'cannot generate distinct n={n} {split} family={family} slot={index}')
        generated[split] = rows
    return [row for split in ('train', 'fresh_rules', 'structural') for row in generated[split]]
