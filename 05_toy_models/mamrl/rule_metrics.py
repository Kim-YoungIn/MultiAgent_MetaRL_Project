"""Shared payoff fingerprints and descriptive distances for both rule catalogs.

These metrics describe expressions/tensors, not strategic equivalence.
"""

import hashlib
import itertools
import json

from .rules import compile_rule


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def common_values(rule):
    return tuple(compile_rule(rule).values[:rule.n ** 4])


def rounded_values(values):
    return tuple(0.0 if abs(v) < 1e-12 else round(v, 12) for v in values)


def canonical_values(values, n):
    """Quotient under one common label permutation of all four axes."""
    if len(values) != n ** 4:
        raise ValueError('expected n^4 common payoff entries')
    def offset(indices):
        a, b, c, d = indices
        return ((a*n+b)*n+c)*n+d
    points = tuple(itertools.product(range(n), repeat=4))
    return min(rounded_values(values[offset(tuple(p[i] for i in point))] for point in points)
               for p in itertools.permutations(range(n)))


def standardized_values(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return (0.0,) * len(values)
    return tuple(2 * (v-lo) / (hi-lo)-1 for v in values)


def tensor_distance(left, right, *, canonical=False, standardize=False):
    if left.n != right.n:
        raise ValueError('distances require the same action count')
    a, b = common_values(left), common_values(right)
    if standardize:
        a, b = standardized_values(a), standardized_values(b)
    if canonical:
        # Minimize relative alignment, rather than distance between two separately
        # lexicographic representatives (which need not minimize their distance).
        n = left.n
        points = tuple(itertools.product(range(n), repeat=4))
        def aligned(p):
            return tuple(b[((p[x]*n+p[y])*n+p[z])*n+p[w]] for x,y,z,w in points)
        return min(sum(abs(x-y) for x,y in zip(a, aligned(p))) / len(a)
                   for p in itertools.permutations(range(n)))
    return sum(abs(x-y) for x,y in zip(a,b)) / len(a)


def expression_tokens(serialized):
    from collections import Counter
    result = Counter()
    def walk(value):
        if isinstance(value, dict):
            if 'op' in value:
                result[value['op']] += 1
            for key in ('left', 'right'):
                if value.get(key) is not None:
                    result[str(value[key])] += 1
            for key in ('condition', 'children', 'root'):
                if key in value:
                    walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(serialized)
    return result


def depends_on_axis(values, n, axis):
    for fixed in itertools.product(range(n), repeat=3):
        out = []
        for action in range(n):
            index = list(fixed)
            index.insert(axis, action)
            a,b,c,d = index
            out.append(values[((a*n+b)*n+c)*n+d])
        if max(out) - min(out) > 1e-10:
            return True
    return False


