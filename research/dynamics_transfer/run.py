"""Run the preregistered structured-game perturbation panel on frozen v2 policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiment.artifacts import atomic_path
from research.fast_exploration_v1 import batch, study
from research.fast_exploration_v1.scale import digest, save_json
from research.fast_exploration_v1.simulator import StudentBatch
from research.fast_exploration_v2 import core
from research.fast_exploration_v2.analyze import load_blocks
from research.fast_exploration_v2.run import source_files as v2_source_files
from research.fast_exploration_v2.verify import rule_sets
from .rules import variant

LAGS = (0, 1)
DISTANCES = (0., .3, .6)
LIVE_METHODS = ('pooled_fixed_total', 'pooled_ucb', 'pooled_greedy')
CONTROLLED_METHODS = ('pooled_full', 'pooled_masked', 'zero')
PROBE_METHODS = CONTROLLED_METHODS[:2]
HORIZON = 96
ASSESSMENT = 32
K = 8
BETA = .15
RESETS = 2
PROTOCOL = 'research/dynamics_transfer/PROTOCOL.md'


def run_block(source, output, record, count, split):
    """Write a block atomically, recording per-turn actions for both players."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stem = f'block_{record["block"]:03d}'
    archive = output / f'{stem}.npz'
    receipt = output / f'{stem}.json'
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if (saved['block'] != record['block'] or saved['teacher_sha256'] != record['weights_sha256']
                or saved['raw_sha256'] != digest(archive)):
            raise ValueError(f'block integrity failure: {archive}')
        return record['block']
    started = time.perf_counter()
    split_seed = 253721399 if split == 'final' else 184728591
    rng = np.random.default_rng(np.random.SeedSequence([split_seed, record['master_seed'], record['seed_block']]))
    seeds = rng.integers(2**63, 2**64, count, dtype=np.uint64)
    resets = rng.integers(0, 2**61, count * RESETS, dtype=np.int64)
    families = [batch.FAMILIES[i % len(batch.FAMILIES)] for i in range(count) for _ in range(RESETS)]
    original = batch.prepare(resets, families, K, HORIZON, rule_seeds=np.repeat(seeds, RESETS))
    base_rules = original.rules
    base_fingerprints = np.array([r.fingerprint() for r in base_rules[::RESETS]])
    prior = np.asarray(record['pooled_prior'])
    state = torch.load(source / f'{stem}.pt', map_location='cpu', weights_only=True)
    actors = {}
    for name in ('pooled_fixed_total', *PROBE_METHODS):
        model = study.BeliefActor()
        model.load_state_dict(state[name])
        actors[name] = model.eval()

    n = len(resets)
    live_shape = (len(LAGS), len(DISTANCES), len(LIVE_METHODS), n, HORIZON, 2)
    controlled_shape = (len(LAGS), len(DISTANCES), len(CONTROLLED_METHODS), n, ASSESSMENT, 2)
    probe_shape = (len(LAGS), len(DISTANCES), len(PROBE_METHODS), n, K, 2)
    live_actions = np.empty(live_shape, dtype=np.int8)
    live_rewards = np.empty(live_shape)
    controlled_actions = np.empty(controlled_shape, dtype=np.int8)
    controlled_rewards = np.empty(controlled_shape)
    probe_actions = np.empty(probe_shape, dtype=np.int8)
    probe_rewards = np.empty(probe_shape)
    fingerprints = np.empty((len(LAGS), len(DISTANCES), count), dtype='<U64')
    policy_draws = core.policy_uniforms(original, K)
    with torch.no_grad():
        for li, lag in enumerate(LAGS):
            for di, distance in enumerate(DISTANCES):
                # Reusing episode streams pairs all exogenous randomness while the
                # rule and the opponent's own warm-up are changed together.
                original.rules = [variant(rule, distance, lag) for rule in base_rules]
                original.template = None
                fingerprints[li, di] = [rule.fingerprint() for rule in original.rules[::RESETS]]
                q = batch.warm_up(original)
                for mi, method in enumerate(LIVE_METHODS):
                    k = 0 if method == 'pooled_greedy' else K
                    prefix = core.probe(original, q, prior, method, actors.get(method), policy_draws,
                                        k, HORIZON, beta=BETA)
                    tail = core.assess(prefix.env, original.assess_uniform[k:HORIZON],
                                       prefix.student, prefix.q, learn=True)
                    live_actions[li, di, mi] = np.concatenate((prefix.tape.actions, tail.actions), axis=1)
                    live_rewards[li, di, mi] = np.concatenate((prefix.tape.rewards, tail.rewards), axis=1)
                reset_env = core.reset_assessment(original, ASSESSMENT)
                for mi, method in enumerate(PROBE_METHODS):
                    probes = core.probe(original, q, prior, method, actors[method], policy_draws,
                                        K, ASSESSMENT, reset=reset_env.last)
                    assessment = core.assess(core.reset_assessment(original, ASSESSMENT),
                                             original.assess_uniform[:ASSESSMENT], probes.student, q,
                                             learn=False)
                    probe_actions[li, di, mi] = probes.tape.actions
                    probe_rewards[li, di, mi] = probes.tape.rewards
                    controlled_actions[li, di, mi] = assessment.actions
                    controlled_rewards[li, di, mi] = assessment.rewards
                blank = StudentBatch(n, prior)
                assessment = core.assess(core.reset_assessment(original, ASSESSMENT),
                                         original.assess_uniform[:ASSESSMENT], blank, q, learn=False)
                controlled_actions[li, di, 2] = assessment.actions
                controlled_rewards[li, di, 2] = assessment.rewards
    with atomic_path(archive) as tmp, tmp.open('wb') as stream:
        np.savez_compressed(
            stream, live_actions=live_actions, live_rewards=live_rewards,
            controlled_actions=controlled_actions, controlled_rewards=controlled_rewards,
            probe_actions=probe_actions, probe_rewards=probe_rewards,
            live_methods=LIVE_METHODS, controlled_methods=CONTROLLED_METHODS, probe_methods=PROBE_METHODS,
            distances=DISTANCES, lags=LAGS, families=families, game_seeds=seeds, reset_seeds=resets,
            base_fingerprints=base_fingerprints, variant_fingerprints=fingerprints,
            action_to_role=np.argsort([r.roles for r in base_rules]),
        )
    save_json(receipt, dict(block=record['block'], master_seed=record['master_seed'],
                            seed_block=record['seed_block'], teacher_sha256=record['weights_sha256'],
                            raw_sha256=digest(archive), elapsed_seconds=time.perf_counter() - started))
    return record['block']


def check_references(source, reference):
    """Verify old panel receipts before checking its rules for overlap."""
    complete = json.loads((reference / 'complete.json').read_text())
    manifest = json.loads((reference / 'manifest.json').read_text())
    if manifest['trained_source_sha256'] != digest(source / 'manifest.json'):
        raise ValueError('reference panel does not use the requested training run')
    paths = list(reference.glob('block_*.npz'))
    if len(paths) != complete['blocks']:
        raise ValueError('reference panel is incomplete')
    fingerprints = set()
    for path in paths:
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['raw_sha256'] != digest(path):
            raise ValueError(f'reference checksum mismatch: {path}')
        with np.load(path) as raw:
            fingerprints.update(raw['test_fingerprints'])
    if len(fingerprints) != complete['fresh_games']:
        raise ValueError('reference game count does not match completion receipt')
    return fingerprints


def run(source, reference, output, split, workers):
    old_manifest, entries = load_blocks(source)
    completion = json.loads((source / 'complete.json').read_text())
    if (old_manifest['split'] != 'final' or len(entries) != old_manifest['config']['blocks']
            or completion['manifest_sha256'] != digest(source / 'manifest.json')):
        raise ValueError('a complete, verified final v2 training run is required')
    check_references(source, reference)
    records = [record for record, _ in entries]
    if split == 'dev':
        records = records[:2]
    count = 64 if split == 'final' else 8
    project, sources = v2_source_files()
    for name in ('research/dynamics_transfer/run.py', 'research/dynamics_transfer/rules.py', PROTOCOL):
        sources[name] = digest(project / name)
    identity = dict(protocol='dynamics-transfer-v1', split=split, games_per_block=count,
                    blocks=[r['block'] for r in records], resets=RESETS, lags=LAGS, distances=DISTANCES,
                    live_methods=LIVE_METHODS, controlled_methods=CONTROLLED_METHODS, k=K,
                    horizon=HORIZON, assessment=ASSESSMENT, ucb_beta=BETA,
                    source_manifest_sha256=digest(source / 'manifest.json'),
                    source_receipts={str(r['block']): digest(source / f'block_{r["block"]:03d}.json') for r in records},
                    reference_manifest_sha256=digest(reference / 'manifest.json'),
                    python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                    sources=dict(sorted(sources.items())))
    # JSON round-trip normalizes tuples to lists before later comparisons.
    identity = json.loads(json.dumps(identity))
    manifest = output / 'manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('sealed configuration differs; use a fresh output directory')
        for name, expected in sources.items():
            if digest(output / 'source' / name) != expected:
                raise ValueError(f'archived source mismatch: {name}')
    else:
        if any(output.glob('block_*')):
            raise ValueError('block artifacts exist without a manifest')
        for name, expected in sources.items():
            content = (project / name).read_bytes()
            dest = output / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            if digest(dest) != expected:
                raise ValueError(f'source changed while archiving: {name}')
        save_json(manifest, identity)
    started = time.perf_counter()
    jobs = [(source, output, rec, count, split) for rec in records]
    if workers == 1:
        for job in jobs:
            print(f'block {run_block(*job):03d} saved', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for result in as_completed([pool.submit(run_block, *job) for job in jobs]):
                print(f'block {result.result():03d} saved', flush=True)
    seen = set()
    all_variants = set()
    for record in records:
        with np.load(output / f'block_{record["block"]:03d}.npz') as raw:
            current = set(raw['base_fingerprints'])
            if len(current) != count or current.intersection(seen):
                raise ValueError('duplicate base games within or across blocks')
            seen.update(current)
            all_variants.update(raw['variant_fingerprints'].flat)
    reference_rules = check_references(source, reference)
    old_test, old_train = rule_sets(source)
    if (seen | all_variants).intersection(old_test | old_train | reference_rules):
        raise ValueError('new panel overlaps previous test, prior, training, or horizon rules')
    receipt = output / 'complete.json'
    result = dict(blocks=len(records), base_games=len(seen), variants=len(all_variants),
                  manifest_sha256=digest(manifest), workers=workers,
                  completing_invocation_seconds=time.perf_counter() - started)
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if any(previous[k] != result[k] for k in ('blocks', 'base_games', 'variants', 'manifest_sha256')):
            raise ValueError('completion receipt does not match sealed run')
    else:
        save_json(receipt, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'final'), required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.output.resolve() in (args.source.resolve(), args.reference.resolve()):
        parser.error('invalid worker count or output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.source, args.reference, args.output, args.split, args.workers)


if __name__ == '__main__':
    main()
