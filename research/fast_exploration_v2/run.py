"""Run the Fast Exploration v2 development or final experiment on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiment.artifacts import atomic_path
from research.fast_exploration_v1 import batch, study
from research.fast_exploration_v1.prior import auc_weights
from research.fast_exploration_v1.scale import digest, save_json, save_weights, source_files as v1_sources
from research.structured_scale.run import FAMILIES
from . import core


SPLITS = {
    'smoke': dict(master_seeds=[40811983], blocks_per_master=1, teacher_updates=2,
                  batch=4, prior_games=4, prior_steps=2, test_games=4, test_resets=2),
    'dev': dict(master_seeds=[68419087], blocks_per_master=4, teacher_updates=80,
                batch=32, prior_games=32, prior_steps=160, test_games=32, test_resets=2),
    'final': dict(master_seeds=[17493629, 27510379, 39572317], blocks_per_master=12,
                  teacher_updates=400, batch=32, prior_games=128, prior_steps=160,
                  test_games=128, test_resets=2),
}
COMMON = dict(ks=[0, 4, 8, 16, 24], h=32, budget_h=48, ucb_beta=.25)


def configuration(split):
    data = SPLITS[split].copy()
    masters = data.pop('master_seeds')
    data.update(COMMON)
    data['blocks'] = len(masters) * data['blocks_per_master']
    return data, masters


def source_files():
    project, sources = v1_sources()
    for name in ('research/fast_exploration_v2/__init__.py',
                 'research/fast_exploration_v2/core.py',
                 'research/fast_exploration_v2/run.py',
                 'research/fast_exploration/architecture/V2_PROTOCOL.md'):
        sources[name] = digest(project / name)
    return project, dict(sorted(sources.items()))


def completed(output, index):
    prefix = output / f'block_{index:03d}'
    record = prefix.with_suffix('.json')
    if not record.exists():
        return False
    data = json.loads(record.read_text())
    if data['block'] != index:
        raise ValueError(f'wrong block number: {record}')
    for suffix, key in (('.npz', 'raw_sha256'), ('.pt', 'weights_sha256')):
        if digest(prefix.with_suffix(suffix)) != data[key]:
            raise ValueError(f'block integrity failure: {prefix}{suffix}')
    return True


def make_actors(seed):
    actors = {name: study.BeliefActor(seed=seed) for name in core.TRAINED}
    optimizers = {name: torch.optim.Adam(actor.parameters(), lr=.001)
                  for name, actor in actors.items()}
    return actors, optimizers


def train(output, index, priors, train_seed, cfg, forbidden):
    """Save state, deterministic sampling RNG and exact trajectories every 25 updates."""
    checkpoint = output / f'.block_{index:03d}_progress.pt'
    actors, optimizers = make_actors(train_seed)
    rng = np.random.default_rng(train_seed)
    updates, count = cfg['teacher_updates'], cfg['batch']
    length = max(cfg['ks']) + max(cfg['h'], cfg['budget_h'])
    actions = np.full((updates, len(core.TRAINED), count, length, 2), -1, dtype=np.int8)
    metrics = np.full((updates, len(core.TRAINED)), np.nan, dtype=np.float32)
    rule_seeds = np.empty((updates, count), dtype=np.int64)
    reset_seeds = np.empty((updates, count), dtype=np.int64)
    budgets = np.empty(updates, dtype=np.int16)
    family_names = np.empty((updates, count), dtype='<U11')
    fingerprints = np.empty((updates, count), dtype='<U64')
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for name in core.TRAINED:
            actors[name].load_state_dict(saved['actors'][name])
            optimizers[name].load_state_dict(saved['optimizers'][name])
        rng.bit_generator.state = saved['rng']
        start = saved['update']
        for key, array in (('actions', actions), ('metrics', metrics), ('rule_seeds', rule_seeds),
                           ('reset_seeds', reset_seeds), ('budgets', budgets), ('families', family_names),
                           ('fingerprints', fingerprints)):
            array[:start] = saved[key]
    else:
        start = 0
    weights = auc_weights(tuple(cfg['ks']))[1:]
    weights /= weights.sum()
    for update in range(start, updates):
        k = int(rng.choice(cfg['ks'][1:], p=weights))
        rules = rng.integers(2**62, 2**63-1, size=count, dtype=np.int64)
        resets = rng.integers(0, 2**61, size=count, dtype=np.int64)
        families = [FAMILIES[(update*count+i) % len(FAMILIES)] for i in range(count)]
        episodes = batch.prepare(resets, families, k, max(cfg['h'], cfg['budget_h']), rule_seeds=rules)
        rule_fingerprints = [rule.fingerprint() for rule in episodes.rules]
        if forbidden.intersection(rule_fingerprints):
            raise ValueError(f'training rule overlaps a prior or test rule in block {index}')
        initial_q = batch.warm_up(episodes)
        scores, traces = core.train_update(actors, optimizers, priors, episodes, initial_q,
                                           k, cfg['h'], cfg['budget_h'])
        metrics[update] = scores
        for m, trajectory in enumerate(traces):
            actions[update, m, :, :trajectory.shape[1]] = trajectory
        rule_seeds[update], reset_seeds[update], budgets[update] = rules, resets, k
        family_names[update] = families
        fingerprints[update] = rule_fingerprints
        done = update + 1
        if done % 25 == 0 or done == updates:
            save_weights(checkpoint, dict(
                update=done, rng=rng.bit_generator.state,
                actors={name: actor.state_dict() for name, actor in actors.items()},
                optimizers={name: optimizers[name].state_dict() for name in actors},
                actions=actions[:done], metrics=metrics[:done], rule_seeds=rule_seeds[:done],
                reset_seeds=reset_seeds[:done], budgets=budgets[:done], families=family_names[:done],
                fingerprints=fingerprints[:done],
            ))
    return actors, dict(train_actions=actions, train_metrics=metrics, train_rule_seeds=rule_seeds,
                        train_reset_seeds=reset_seeds, train_k=budgets, train_families=family_names,
                        train_fingerprints=fingerprints)


def fit_block(job):
    output, cfg, master, seed_block, index = job
    if completed(output, index):
        return index
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    train_seed, prior_seed, test_seed = map(
        int, np.random.SeedSequence([master, seed_block]).generate_state(3)
    )
    data = study.prior_samples(prior_seed, cfg['prior_games'])
    ips = study.fit_return_prior(data, steps=cfg['prior_steps'], seed=train_seed, device='cpu')
    _, pooled = study.pooled_fit(data)
    priors = {'ips': ips, 'pooled': pooled}
    rng = np.random.default_rng(test_seed)
    game_seeds = rng.integers(2**61, 2**62, size=cfg['test_games'], dtype=np.int64)
    resets = rng.integers(0, 2**61, size=cfg['test_games']*cfg['test_resets'], dtype=np.int64)
    families = [FAMILIES[i % len(FAMILIES)] for i in range(cfg['test_games'])
                for _ in range(cfg['test_resets'])]
    episodes = batch.prepare(resets, families, max(cfg['ks']), max(cfg['h'], cfg['budget_h']),
                             rule_seeds=np.repeat(game_seeds, cfg['test_resets']))
    fingerprints = [rule.fingerprint() for rule in episodes.rules[::cfg['test_resets']]]
    prior_fingerprints = data['fingerprint'].tolist()
    if len(set(fingerprints)) != cfg['test_games'] or not set(fingerprints).isdisjoint(prior_fingerprints):
        raise ValueError('duplicate test rule or overlap with prior panel')
    prior_seconds = time.perf_counter() - start
    train_start = time.perf_counter()
    actors, training = train(output, index, priors, train_seed, cfg,
                             set(fingerprints).union(prior_fingerprints))
    train_seconds = time.perf_counter() - train_start
    eval_start = time.perf_counter()
    result = core.evaluate(actors, priors, episodes, cfg)
    eval_seconds = time.perf_counter() - eval_start
    prefix = output / f'block_{index:03d}'
    archive = prefix.with_suffix('.npz')
    weights = prefix.with_suffix('.pt')
    with atomic_path(archive) as temp, temp.open('wb') as stream:
        np.savez_compressed(
            stream, **result, **training, game_seeds=game_seeds, reset_seeds=resets,
            families=families, test_fingerprints=fingerprints, prior_fingerprints=prior_fingerprints,
            prior_rule_seeds=data['seed'], prior_families=data['family'],
            prior_warm_actions=data['warm_a'], prior_actions=data['a'], prior_opponent_actions=data['b'],
            prior_rewards=data['y'], prior_features=data['x'], prior_warm_opponent_actions=data['warm_b'],
            prior_reset_seeds=data['reset_seed'], prior_warm_seeds=data['warm_seed'],
        )
    save_weights(weights, {name: actor.state_dict() for name, actor in actors.items()})
    save_json(prefix.with_suffix('.json'), dict(
        block=index, master_seed=master, seed_block=seed_block,
        seeds=[train_seed, prior_seed, test_seed], ips_prior=ips.tolist(), pooled_prior=pooled.tolist(),
        raw_sha256=digest(archive), weights_sha256=digest(weights),
        seconds=dict(prior=prior_seconds, teacher=train_seconds, evaluation=eval_seconds,
                     total=time.perf_counter() - start),
    ))
    (output / f'.block_{index:03d}_progress.pt').unlink(missing_ok=True)
    return index


def execute(output, split, workers, only=None, ucb_beta=None):
    cfg, masters = configuration(split)
    if ucb_beta is not None:
        cfg['ucb_beta'] = ucb_beta
    project, sources = source_files()
    identity = dict(protocol='fast-exploration-v2', split=split, config=cfg, master_seeds=masters,
                    device='cpu', python=platform.python_version(), torch=str(torch.__version__),
                    numpy=np.__version__, source=sources)
    manifest = output / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('sealed configuration differs; use a fresh output directory')
    else:
        if any(output.glob('block_*')):
            raise ValueError('block artifacts exist without a manifest')
        for name, checksum in sources.items():
            content = (project / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise RuntimeError(f'source changed during preparation: {name}')
            dest = output / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        save_json(manifest, identity)
    pairs = [(master, block) for master in masters for block in range(cfg['blocks_per_master'])]
    jobs = [(output, cfg, master, block, index) for index, (master, block) in enumerate(pairs)
            if (only is None or only == index) and not completed(output, index)]
    start = time.perf_counter()
    if workers == 1:
        for job in jobs:
            print(f'block {fit_block(job):03d} saved', flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for result in as_completed([pool.submit(fit_block, job) for job in jobs]):
                print(f'block {result.result():03d} saved', flush=True)
    if only is None:
        test, prior, training = set(), set(), set()
        for index in range(cfg['blocks']):
            if not completed(output, index):
                raise ValueError(f'incomplete block {index}')
            with np.load(output / f'block_{index:03d}.npz') as raw:
                fingerprints = set(raw['test_fingerprints'])
                if test.intersection(fingerprints):
                    raise ValueError('duplicate test rule across blocks')
                test.update(fingerprints)
                prior.update(raw['prior_fingerprints'])
                training.update(raw['train_fingerprints'].flat)
        if test.intersection(prior.union(training)):
            raise ValueError('training and test rule sets overlap across blocks')
        completion = output / 'complete.json'
        if completion.exists():
            previous = json.loads(completion.read_text())
            if (previous['blocks'] != cfg['blocks'] or previous['test_games'] != len(test)
                    or previous['manifest_sha256'] != digest(manifest)):
                raise ValueError('completion receipt does not match the sealed run')
        else:
            save_json(completion, dict(
                blocks=cfg['blocks'], test_games=len(test), workers=workers,
                completing_invocation_seconds=time.perf_counter() - start,
                platform=platform.platform(), manifest_sha256=digest(manifest),
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--only-block', type=int)
    parser.add_argument('--ucb-beta', type=float)
    args = parser.parse_args()
    cfg, _ = configuration(args.split)
    if (args.workers < 1 or (args.only_block is not None and not 0 <= args.only_block < cfg['blocks'])
            or (args.ucb_beta is not None and (not np.isfinite(args.ucb_beta) or args.ucb_beta < 0))):
        parser.error('invalid worker count, block index, or UCB coefficient')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.output, args.split, args.workers, args.only_block, args.ucb_beta)


if __name__ == '__main__':
    main()
