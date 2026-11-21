"""Independent brute-force verification of saved two-turn information results."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_toy_models'))
from mamrl.structured_games import StructuredEnv, generate_structured_rule


class FixedGate:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def check(name, actual, expected, tol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=tol, atol=tol):
        error = np.max(np.abs(actual-expected)) if actual.shape == expected.shape else None
        raise AssertionError(f'{name}: {actual.shape} vs {expected.shape}, maximum error {error}')


def reconstruct(record):
    n, depth = record['n'], record['depth']
    games = [generate_structured_rule(n=n, memory_depth=depth, seed=s, family=record['family'], stochastic=True)
             for s in record['seeds'][:2]]
    if [game.fingerprint() for game in games] != record['fingerprints']:
        raise AssertionError('generated fingerprint mismatch')
    first = np.zeros(n)
    totals = np.zeros((2,n))
    for focal_action in range(n):
        observations = [defaultdict(list), defaultdict(list)]
        for game in games:
            env = StructuredEnv(game,max_steps=2)
            obs = env.reset(seed=record['seeds'][2])
            for opponent_action in range(n):
                gates = [clause.probability for clause in game.transitions if 0 < clause.probability < 1]
                if len(gates) > 1:
                    raise AssertionError('more than one stochastic transition')
                branches = [(1., 0.)] if not gates else [(gates[0],0.),(1-gates[0],1.)]
                for branch_mass, uniform in branches:
                    parent = copy.deepcopy(env)
                    parent._rng = FixedGate(uniform)
                    result = parent.step(focal_action,opponent_action)
                    r, opponent_reward = result.rewards
                    weight = branch_mass / (2*n)
                    first[focal_action] += weight*r
                    # Recompute the opponent's Q table from its own perspective.
                    q = np.zeros((n,n,n))
                    old = (obs[1].previous_own_action,obs[1].previous_opponent_action)
                    new = (result.observations[1].previous_own_action,result.observations[1].previous_opponent_action)
                    q[old][opponent_action] += .3*opponent_reward
                    nextq = q[new]
                    best = np.abs(nextq-nextq.max()) <= 1e-12
                    probabilities = np.ones(n)*(.25/n) + .75*best/best.sum()
                    action_values = np.zeros(n)
                    for a2 in range(n):
                        for b2 in range(n):
                            child = copy.deepcopy(parent)
                            action_values[a2] += probabilities[b2]*child.step(a2,b2).rewards[0]
                    public = (opponent_action, result.observations[0].public_state)
                    observations[0][public].append((weight,action_values))
                    observations[1][(*public,r)].append((weight,action_values))
        for view in range(2):
            totals[view,focal_action] = first[focal_action] + sum(
                np.max(sum(weight*value for weight,value in values)) for values in observations[view].values())
    greedy = first >= first.max()-1e-10
    gaps = [totals[1].max()-totals[0].max(),totals[1].max()-totals[1,greedy].max(),
            totals[0].max()-totals[0,greedy].max()]
    return first,totals,gaps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    root=args.output
    records=json.loads((root/'pairs.json').read_text())
    if len(records)!=288:
        raise AssertionError('expected 288 pairs')
    checked=0
    for record in records:
        first=np.array(record['first'])
        masked=np.array(record['masked'])
        full=np.array(record['visible'])
        greedy=first >= first.max()-1e-10
        expected=np.array([full.max()-masked.max(),full.max()-full[greedy].max(),
                           masked.max()-masked[greedy].max()])
        check('pair gap',record['gaps'],expected)
        if np.min(expected) < -1e-10:
            raise AssertionError('negative information or restriction gap')
        if record['pair'] == 0:
            one,two,three=reconstruct(record)
            check('first expected reward',record['first'],one)
            check('masked values',record['masked'],two[0])
            check('visible values',record['visible'],two[1])
            check('pair gaps',record['gaps'],three)
            checked+=1
    with (root/'summary.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=15:
        raise AssertionError('incorrect number of summary rows')
    for row in rows:
        selected=[r for r in records if r['n']==int(row['n']) and r['depth']==int(row['depth'])
                  and (row['motif']=='all' or row['motif']==r['family'])]
        gaps=np.array([r['gaps'] for r in selected])
        check('summary count',int(row['pairs']),len(selected),0)
        counts=(gaps>1e-10).sum(axis=0)
        means=gaps.mean(axis=0)
        for field,value in zip(('reward_info_count','first_cost_count','masked_first_cost_count'),counts):
            check('summary '+field,int(row[field]),value,0)
        for field,value in zip(('mean_reward_info','mean_first_cost','mean_masked_first_cost'),means):
            check('summary '+field,float(row[field]),value)
    manifest=json.loads((root/'manifest.json').read_text())
    files={'PROTOCOL.md':Path(__file__).with_name('PROTOCOL.md'),
           'run.py':Path(__file__).with_name('run.py'),
           'structured_games.py':Path(__file__).resolve().parents[2]/'05_toy_models/mamrl/structured_games.py'}
    mismatches=[name for name,path in files.items() if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['source_hashes'][name]]
    result={'pairs_checked':len(records),'pair_gaps_reconstructed':len(records),'brute_force_pairs':checked,
            'summary_rows_checked':len(rows),'source_hash_mismatches':mismatches,
            'scope':'Brute-force two real environment turns for pair 0 of each motif and setting, enumerating both opponent actions, stochastic gates, and information sets.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
