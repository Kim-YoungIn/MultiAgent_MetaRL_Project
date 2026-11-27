"""Measure exposure to the changed transition under two policy-independent scripts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import itertools
import json
import multiprocessing
from pathlib import Path

import numpy as np

from experiment.artifacts import atomic_path
from research.fast_exploration_v1.scale import digest, save_json
from research.fast_exploration_v1.simulator import EpisodeBatch
from research.fast_exploration_v2.analyze import bootstrap_indices, statistics, write_csv
from research.structured_scale.run import FAMILIES, game
from .rules import variant


SCRIPTS = ('uniform_joint_actions', 'four_invest_two_harvest')
HORIZON = 96


def trajectories(rules, seeds, families, actions, gates, extra_lag):
    env = EpisodeBatch(rules, seeds, HORIZON)
    rows = np.arange(len(rules))
    slots = [env.state_index[i][investment_state(rule)] for i, rule in enumerate(rules)]
    ceiling = env.levels[rows, slots] - 1
    invest = np.array([r.roles[0] for r in rules])[:, None]
    sequence = families == 'sequence'
    states = np.empty((len(rows), HORIZON), dtype=np.int8)
    opportunities = np.empty(states.shape, dtype=bool)
    payoffs = np.empty((*states.shape, 2))
    for t in range(HORIZON):
        joint = actions[:, t]
        eligible = (joint == invest).all(axis=1)
        eligible &= ~sequence | (env.history[:, 1] == invest).all(axis=1)
        if extra_lag:
            eligible &= (env.last == invest).all(axis=1)
        before = env.state[rows, slots]
        opportunities[:, t] = eligible & (before < ceiling)
        payoffs[:, t] = env.step(joint[:, 0], joint[:, 1], gates[:, t])
        states[:, t] = env.state[rows, slots]
    return states, opportunities, payoffs


def investment_state(rule):
    return next(clause.state for clause in rule.transitions if clause.delta > 0)


def replay_block(job):
    root, dest, index = job
    stem = f'block_{index:03d}'
    receipt = json.loads((root / f'{stem}.json').read_text())
    panel = root / f'{stem}.npz'
    if digest(panel) != receipt['raw_sha256']:
        raise ValueError(f'panel hash mismatch in {stem}')
    with np.load(panel, allow_pickle=False) as raw:
        seeds, resets, families = (raw[k] for k in ('game_seeds', 'reset_seeds', 'families'))
        lags, distances = raw['lags'], raw['distances']
        fingerprints = raw['variant_fingerprints']
    repeats = len(resets) // len(seeds)
    base = [game(int(seed), str(f), 3, 3) for seed, f in zip(seeds, families[::repeats], strict=True)]
    rng = np.random.default_rng(np.random.SeedSequence([91152732, index]))
    # Both open-loop scripts share reset histories and uniforms for stochastic gates.
    reset_seeds = rng.integers(2**61, size=len(resets))
    gates = rng.random((len(resets), HORIZON))
    uniform = rng.integers(3, size=(len(resets), HORIZON, 2), dtype=np.int8)
    roles = np.repeat([rule.roles for rule in base], repeats, axis=0)
    cyclic = roles[:, (np.arange(HORIZON) % 6 >= 4).astype(int)]
    actions = np.stack((uniform, np.repeat(cyclic[:, :, None], 2, axis=2))).astype(np.int8)
    archive = {}
    for i, lag in enumerate(lags):
        for j, d in enumerate(distances):
            unique_rules = [variant(rule, float(d), int(lag)) for rule in base]
            np.testing.assert_array_equal([r.fingerprint() for r in unique_rules], fingerprints[i, j])
            rules = [r for r in unique_rules for _ in range(repeats)]
            for s in range(len(SCRIPTS)):
                values = trajectories(rules, reset_seeds, families, actions[s], gates, lag)
                for key, value in zip(('state', 'opportunity', 'rewards'), values, strict=True):
                    if key not in archive:
                        archive[key] = np.empty((len(lags), len(distances), len(SCRIPTS), *value.shape), dtype=value.dtype)
                    archive[key][i, j, s] = value
    output = dest / f'{stem}.npz'
    with atomic_path(output) as tmp, tmp.open('wb') as stream:
        np.savez_compressed(stream, **archive, actions=actions, gates=gates, reset_seeds=reset_seeds,
                            scripts=SCRIPTS, game_seeds=seeds, families=families, lags=lags, distances=distances)
    # Compare post-step occupancy under common randomness. Opponent choices are
    # scripted here, so the comparison is conditional on the two joint policies.
    rows = {}
    for i, j, s in itertools.product(range(len(lags)), range(len(distances)), range(len(SCRIPTS))):
        for f in ('all', *FAMILIES):
            mask = np.ones(len(families), dtype=bool) if f == 'all' else families == f
            state = archive['state'][i, j, s, mask]
            reference = archive['state'][0, 0, s, mask]
            rewards = archive['rewards'][i, j, s, mask, :, 0]
            ref_rewards = archive['rewards'][0, 0, s, mask, :, 0]
            key = (int(lags[i]), float(distances[j]), SCRIPTS[s], f)
            rows[key] = dict(mean_state=state.mean(), opportunity=archive['opportunity'][i, j, s, mask].mean(),
                             state_disagreement=(state != reference).mean(),
                             return_difference=(rewards - ref_rewards).sum(axis=-1).mean())
    return receipt, digest(output), rows


def run(panel, output, workers):
    complete = json.loads((panel / 'complete.json').read_text())
    manifest = panel / 'manifest.json'
    if digest(manifest) != complete['manifest_sha256']:
        raise ValueError('panel manifest mismatch')
    ids = json.loads(manifest.read_text())['blocks']
    if complete['blocks'] != len(ids):
        raise ValueError('source panel is incomplete')
    dest = panel / 'scripted_exposure'
    dest.mkdir(exist_ok=True)
    source = dict(panel=digest(manifest), source=digest(Path(__file__)), horizon=HORIZON, scripts=list(SCRIPTS))
    with (dest / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = dest / 'manifest.json'
        if stamp.exists() and json.loads(stamp.read_text()) != source:
            raise ValueError('sealed exposure specification differs')
        save_json(stamp, source)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            blocks = list(pool.map(replay_block, [(panel, dest, i) for i in ids]))
        indices = bootstrap_indices([record for record, _, _ in blocks])
        table = []
        for (lag, distance, script, family), metrics in blocks[0][2].items():
            for name in metrics:
                data = [rows[(lag, distance, script, family)][name] for _, _, rows in blocks]
                table.append(dict(extra_lag=lag, distance=distance, script=script, family=family, metric=name,
                                  **statistics(data, indices)))
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'exposure.csv', table)
        save_json(dest / 'complete.json', dict(source=source, blocks={str(r['block']): h for r, h, _ in blocks},
                                               summary_sha256=digest(output / 'exposure.csv')))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('panel', type=Path)
    cli.add_argument('--output', required=True, type=Path)
    cli.add_argument('--workers', type=int, default=1)
    args = cli.parse_args()
    if args.workers < 1:
        cli.error('workers must be positive')
    run(args.panel, args.output, args.workers)
