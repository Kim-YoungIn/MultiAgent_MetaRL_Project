"""Pair fixed UCB against archived policies on the opened 96-turn panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

from experiment.artifacts import atomic_path
from research.fast_exploration_v1 import batch
from research.fast_exploration_v1.scale import digest, save_json
from . import core
from .analyze import bootstrap_indices, load_blocks, statistics, write_csv
from .verify import check_rewards

KS = (8, 24)


def block(output, path, record, beta):
    with np.load(path) as source:
        methods = source['methods'].tolist()
        budgets = source['ks'].tolist()
        joint = source['rewards']
        seeds, resets, families = (source[name] for name in ('game_seeds', 'reset_seeds', 'families'))
        repeats = len(resets) // len(seeds)
        horizon = joint.shape[-2]
        teachers = {k: joint[next(i for i, m in enumerate(methods)
                                  if m == 'pooled_fixed_total' and budgets[i] == k), ..., 0] for k in KS}
        fingerprint = source['test_fingerprints']
    ep = batch.prepare(resets, families, max(KS), horizon, rule_seeds=np.repeat(seeds, repeats))
    np.testing.assert_array_equal([r.fingerprint() for r in ep.rules[::repeats]], fingerprint)
    q = batch.warm_up(ep)
    prior = np.array(record['pooled_prior'])
    actions, rewards, estimates = [], [], []
    for k in KS:
        prefix = core.probe(ep, q, prior, 'pooled_ucb', None, None, k, horizon, beta=beta)
        tail = core.assess(prefix.env, ep.assess_uniform[k:], prefix.student, prefix.q, learn=True)
        a = np.concatenate((prefix.tape.actions, tail.actions), axis=1)
        r = np.concatenate((prefix.tape.rewards, tail.rewards), axis=1)
        check_rewards(ep.rules[0], ep.explore_seed[0], a[0], ep.assess_uniform[:, 0], r[0])
        actions.append(a)
        rewards.append(r)
        delta = teachers[k] - r[..., 0]
        estimates.append([delta[:, :48].sum(axis=1).mean(), delta.sum(axis=1).mean()])
    dest = output / path.name
    with atomic_path(dest) as temp, temp.open('wb') as stream:
        np.savez_compressed(stream, actions=np.stack(actions), rewards=np.stack(rewards),
                            game_seeds=seeds, reset_seeds=resets, families=families, ks=KS)
    return estimates, digest(dest)


def run(trained, panel, output, workers):
    manifest, entries = load_blocks(trained)
    published = json.loads((panel / 'complete.json').read_text())
    if len(entries) != published['blocks'] or len(entries) != manifest['config']['blocks']:
        raise ValueError('complete training and horizon runs are required')
    parent = json.loads((panel / 'manifest.json').read_text())
    if parent['trained_source_sha256'] != digest(trained / 'manifest.json'):
        raise ValueError('horizon study has different trained policies')
    beta = manifest['config']['ucb_beta']
    records = [r for r, _ in entries]
    paths = [panel / p.name for _, p in entries]
    for path, record in zip(paths, records, strict=True):
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['raw_sha256'] != digest(path) or receipt['teacher_sha256'] != record['weights_sha256']:
            raise ValueError(f'horizon block integrity failure at {path}')
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        results = list(pool.map(block, [output] * len(paths), paths, records, [beta] * len(paths)))
    values, checksums = zip(*results, strict=True)
    values = np.array(values)
    indices = bootstrap_indices(records)
    rows = [dict(k=k, horizon=h, blocks=len(paths), **statistics(values[:, ki, hi], indices))
            for ki, k in enumerate(KS) for hi, h in enumerate((48, 96))]
    write_csv(output / 'paired.csv', rows)
    save_json(output / 'report.json', dict(
        scope='exploratory reuse of opened 96-turn games', contrast='total reward Teacher minus UCB',
        ucb_beta=beta, parent_manifest_sha256=digest(panel / 'manifest.json'),
        source_sha256=digest(Path(__file__)), scalar_trajectories=len(paths)*len(KS),
        blocks={p.name: sha for p, sha in zip(paths, checksums, strict=True)}, results=rows,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trained', type=Path)
    parser.add_argument('panel', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args.trained, args.panel, args.output, args.workers)


if __name__ == '__main__':
    main()
