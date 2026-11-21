"""Exact two-turn Bayes decisions using reachable states of generated games."""
import argparse
import copy
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from mamrl import structured_games as sg

FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
SETTINGS = ((3, 1), (3, 3), (6, 5))
TOL = 1e-10


class RewardKernel:
    """Compile current-action masks once. Dynamic gates read only old state/history."""
    def __init__(self, game):
        self.base = np.array(game.base.values).reshape(2, game.n, game.n, game.n, game.n)[0]
        self.scale = game.reward_scale
        self.terms = []
        a, b = np.indices((game.n, game.n))
        for clause in game.rewards:
            mask = np.full((game.n, game.n), clause.value)
            dynamic = []
            for p in clause.when:
                if p.kind == 'own_is':
                    match = a == p.action
                elif p.kind == 'other_is':
                    match = b == p.action
                elif p.kind == 'both_is':
                    match = (a == p.action) & (b == p.action)
                elif p.kind == 'match':
                    match = a == b
                elif p.kind in ('state_min', 'state_max', 'own_count', 'other_count', 'matching_count', 'always'):
                    dynamic.append(p)
                    continue
                else:
                    raise ValueError(f'unsupported reward predicate in this diagnostic: {p.kind}')
                mask *= ~match if p.negate else match
            self.terms.append((mask, dynamic))

    def __call__(self, env):
        rewards = self.base[env._history[0]].copy()
        for mask, predicates in self.terms:
            if all(sg._check(p, 0, env._state, env._history, (0, 0)) for p in predicates):
                rewards += mask
        return rewards * self.scale


class Gate:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def branch_probabilities(game):
    gates = [t.probability for t in game.transitions if 0 < t.probability < 1]
    if len(gates) > 1 or any(t.probability == 0 for t in game.transitions):
        raise ValueError('diagnostic supports generated games with at most one nontrivial gate')
    if gates:
        return [(gates[0], Gate(0.)), (1-gates[0], Gate((1+gates[0])/2))]
    return [(1., Gate(0.))]


def next_distribution(n, old_joint, a, b, opponent_reward):
    q = np.zeros(n)
    if old_joint == (a, b):
        q[b] = .3 * opponent_reward
    best = np.isclose(q, q.max(), atol=1e-12, rtol=0)
    return .25/n + .75*best/best.sum()


def solve_pair(games, reset_seed):
    n = games[0].n
    first = np.zeros(n)
    groups = [[defaultdict(lambda: np.zeros(n)) for _ in range(n)] for _ in range(2)]
    for game in games:
        kernel = RewardKernel(game)
        env = sg.StructuredEnv(game, max_steps=2)
        env.reset(seed=reset_seed)
        for a in range(n):
            for b in range(n):
                for mass, gate in branch_probabilities(game):
                    # step replaces the history and state dict, leaving the parent untouched.
                    child = copy.copy(env)
                    child._rng = gate
                    transition = child.step(a, b)
                    r, opp_r = transition.rewards
                    weight = mass / (len(games)*n)
                    first[a] += weight*r
                    probs = next_distribution(n, env._history[0], a, b, opp_r)
                    future = kernel(child) @ probs
                    public = (b, transition.observations[0].public_state)
                    groups[0][a][public] += weight*future
                    groups[1][a][(*public, r)] += weight*future
    total = np.array([first + [sum(v.max() for v in buckets.values()) for buckets in actions]
                      for actions in groups])
    greedy = first >= first.max()-TOL
    gaps = [total[1].max()-total[0].max(), total[1].max()-total[1, greedy].max(),
            total[0].max()-total[0, greedy].max()]
    if min(gaps) < -TOL:
        raise AssertionError('information or a larger feasible policy set reduced optimal value')
    return first, total, gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    # Create exclusively, so reruns use a fresh path.
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    rng = np.random.default_rng(61739291)
    results = []
    for motif in FAMILIES:
        for pair in range(24):
            seeds = [int(x) for x in rng.integers(2**53, size=3)]
            for n, depth in SETTINGS:
                games = [sg.generate_structured_rule(n=n, memory_depth=depth, seed=s,
                         family=motif, stochastic=True) for s in seeds[:2]]
                first, totals, gaps = solve_pair(games, seeds[2])
                results.append(dict(family=motif, pair=pair, n=n, depth=depth, seeds=seeds,
                                    fingerprints=[g.fingerprint() for g in games], first=first.tolist(),
                                    masked=totals[0].tolist(), visible=totals[1].tolist(), gaps=gaps))
    (args.output/'pairs.json').write_text(json.dumps(results, indent=1)+'\n')
    with (args.output/'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n','depth','motif','pairs','reward_info_count','first_cost_count',
                         'masked_first_cost_count','mean_reward_info','mean_first_cost','mean_masked_first_cost'])
        for n, depth in SETTINGS:
            for fam in ('all', *FAMILIES):
                g = np.array([r['gaps'] for r in results if r['n']==n and r['depth']==depth and
                              (fam=='all' or fam==r['family'])])
                writer.writerow([n, depth, fam, len(g), *(g>TOL).sum(axis=0), *g.mean(axis=0)])
    here = Path(__file__).parent
    sources = [here/'PROTOCOL.md', Path(__file__), Path(sg.__file__)]
    record = dict(seed=61739291, seconds=time.monotonic()-start, numpy=np.__version__,
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
