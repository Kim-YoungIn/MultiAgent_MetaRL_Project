"""Evaluate frozen v2 Teachers under a fixed 96-turn deployment budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from experiment.artifacts import atomic_path
from research.fast_exploration_v1 import batch, study
from research.fast_exploration_v1.scale import digest, save_json
from . import core
from .analyze import bootstrap_indices, load_blocks, statistics, write_csv
from .verify import check_rewards, rule_sets

HORIZON = 96
CONDITIONS = (
    ('pooled_greedy', 0),
    ('pooled_fixed_value', 8),
    ('pooled_fixed_total', 8),
    ('pooled_fixed_value', 24),
    ('pooled_fixed_total', 24),
)
PROTOCOL = Path('research/fast_exploration/architecture/V2_HORIZON.md')


def trajectory(episodes, initial_q, prior, method, k, actor, uniforms, horizon):
    with torch.no_grad():
        prefix = core.probe(episodes, initial_q, prior, method, actor, uniforms, k, horizon)
        tail = core.assess(prefix.env, episodes.assess_uniform[k:horizon],
                           prefix.student, prefix.q, learn=True)
    return (np.concatenate((prefix.tape.actions, tail.actions), axis=1),
            np.concatenate((prefix.tape.rewards, tail.rewards), axis=1))


def block(source, output, record, count, resets_per_game, short):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stem = f'block_{record["block"]:03d}'
    dest, receipt = output / f'{stem}.npz', output / f'{stem}.json'
    if receipt.exists():
        before = json.loads(receipt.read_text())
        if digest(dest) != before['raw_sha256'] or record['weights_sha256'] != before['teacher_sha256']:
            raise ValueError(f'resume integrity failure at {dest}')
        return record['block']
    rng = np.random.default_rng(np.random.SeedSequence([87658412, record['master_seed'], record['seed_block']]))
    seeds = rng.integers(2**63, 2**64, count, dtype=np.uint64)
    resets = rng.integers(0, 2**61, count * resets_per_game, dtype=np.int64)
    families = [batch.FAMILIES[i % len(batch.FAMILIES)] for i in range(count) for _ in range(resets_per_game)]
    ep = batch.prepare(resets, families, max(k for _, k in CONDITIONS), HORIZON,
                       rule_seeds=np.repeat(seeds, resets_per_game))
    fingerprints = [rule.fingerprint() for rule in ep.rules[::resets_per_game]]
    prior = np.array(record['pooled_prior'])
    weights = torch.load(source / f'{stem}.pt', weights_only=True, map_location='cpu')
    actors = {}
    for method, _ in CONDITIONS[1:3]:
        actor = study.BeliefActor()
        actor.load_state_dict(weights[method])
        actors[method] = actor.eval()
    q = batch.warm_up(ep)
    uniforms = core.policy_uniforms(ep, max(k for _, k in CONDITIONS))
    results = []
    for method, k in CONDITIONS:
        long = trajectory(ep, q, prior, method, k, actors.get(method), uniforms, HORIZON)
        if short:
            reference = trajectory(ep, q, prior, method, k, actors.get(method), uniforms, 48)
            for prefix, baseline in zip(long, reference, strict=True):
                np.testing.assert_array_equal(prefix[:, :48], baseline)
        check_rewards(ep.rules[0], ep.explore_seed[0], long[0][0], ep.assess_uniform[:, 0], long[1][0])
        results.append(long)
    actions, rewards = (np.stack(values) for values in zip(*results, strict=True))
    with atomic_path(dest) as tmp, tmp.open('wb') as stream:
        np.savez_compressed(stream, actions=actions, rewards=rewards,
                            methods=[m for m, _ in CONDITIONS], ks=[k for _, k in CONDITIONS],
                            test_fingerprints=fingerprints, game_seeds=seeds, reset_seeds=resets,
                            families=families, action_to_role=np.argsort([r.roles for r in ep.rules]))
    save_json(receipt, dict(block=record['block'], master_seed=record['master_seed'],
                            teacher_sha256=record['weights_sha256'], raw_sha256=digest(dest)))
    return record['block']


def summarize(output, records, expected):
    blocks = []
    families = []
    for rec in records:
        with np.load(output / f'block_{rec["block"]:03d}.npz') as raw:
            blocks.append(raw['rewards'][..., 0])
            families.append(raw['families'])
    values = np.stack(blocks)
    assignment = np.stack(families)
    curves = values.cumsum(axis=-1)
    samples = bootstrap_indices(records)
    rows = []
    curves_out = []
    weights = np.zeros(samples.shape)
    np.add.at(weights, (np.arange(len(samples))[:, None], samples), 1. / len(records))
    for family in ('all', *batch.FAMILIES):
        mask = np.ones_like(assignment, dtype=bool) if family == 'all' else assignment == family
        means = (curves * mask[:, None, :, None]).sum(axis=2) / mask.sum(axis=1)[:, None, None]
        metrics = {}
        for i, (method, k) in enumerate(CONDITIONS):
            metrics[f'{method}_k{k}'] = means[:, i]
            if i:
                metrics[f'{method}_k{k}_minus_student'] = means[:, i] - means[:, 0]
        for a, b, k in ((2, 1, 8), (4, 3, 24)):
            metrics[f'total_minus_value_k{k}'] = means[:, a] - means[:, b]
        for name, series in metrics.items():
            for turns, estimate in ((48, series[:, 47]), (96, series[:, 95]),
                                     ('96_minus_48', series[:, 95] - series[:, 47])):
                rows.append(dict(family=family, contrast=name, turns=turns, blocks=len(records),
                                 **statistics(estimate, samples)))
            if family == 'all':
                limits = np.quantile(weights @ series, [.025, .975], axis=0)
                for t, mean in enumerate(series.mean(axis=0)):
                    curves_out.append(dict(contrast=name, turns=t + 1, mean=mean,
                                           ci_low=limits[0, t], ci_high=limits[1, t]))
    write_csv(output / 'paired.csv', rows)
    write_csv(output / 'curves.csv', curves_out)
    plot_curves(output, curves_out)
    primary = [r for r in rows if r['family'] == 'all' and
               r['contrast'] == 'pooled_fixed_total_k8_minus_student']
    save_json(output / 'report.json', dict(blocks=len(records), complete=len(records) == expected,
                                           primary=primary))


def plot_curves(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for name, label in (('pooled_fixed_total', 'Total-reward Teacher'),
                        ('pooled_fixed_value', 'Downstream-reward Teacher')):
        for k, style in ((8, '-'), (24, '--')):
            group = [r for r in rows if r['contrast'] == f'{name}_k{k}_minus_student']
            t, mean, low, high = ([r[f] for r in group] for f in ('turns', 'mean', 'ci_low', 'ci_high'))
            line, = ax.plot(t, mean, style, label=f'{label} · K={k}')
            ax.fill_between(t, low, high, color=line.get_color(), alpha=.12, linewidth=0)
    ax.axhline(0, color='0.5', linewidth=.8)
    ax.axvline(48, color='0.5', linewidth=.8)
    ax.set(xlabel='Total turns', ylabel='Cumulative reward vs Student-only')
    ax.legend(frameon=False)
    figure.savefig(output / 'horizon_curves.png', dpi=165)
    plt.close(figure)


def run(source, output, workers, references):
    manifest, loaded = load_blocks(source)
    if len(loaded) != manifest['config']['blocks'] or not (source / 'complete.json').is_file():
        raise ValueError('96-turn deployment needs all source Teachers')
    short = manifest['split'] != 'final'
    count, resets = (16 if short else 128), 2
    records = [r for r, _ in loaded]
    project = Path(__file__).resolve().parents[2]
    files = ('research/fast_exploration_v2/horizon.py', 'research/fast_exploration_v2/core.py', str(PROTOCOL))
    identity = dict(protocol='fast-exploration-v2-horizon', split=manifest['split'],
                    trained_source_sha256=digest(source / 'manifest.json'),
                    games_per_block=count, resets=resets, horizon=HORIZON,
                    sources={p: digest(project / p) for p in files}, conditions=[list(c) for c in CONDITIONS])
    signature = output / 'manifest.json'
    if signature.exists():
        if json.loads(signature.read_text()) != identity:
            raise ValueError('new horizon specification requires a fresh output')
    else:
        for name in files:
            archived = output / 'source' / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes((project / name).read_bytes())
        save_json(signature, identity)
    start = time.perf_counter()
    jobs = [(source, output, r, count, resets, short) for r in records]
    if workers == 1:
        for job in jobs:
            print(f'block {block(*job):03d} ready', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for result in pool.map(block, *zip(*jobs, strict=True)):
                print(f'block {result:03d} ready', flush=True)
    seen = set()
    for rec in records:
        with np.load(output / f'block_{rec["block"]:03d}.npz') as raw:
            subset = set(raw['test_fingerprints'])
            if len(subset) != count or seen.intersection(subset):
                raise ValueError('duplicate horizon evaluation rule')
            seen.update(subset)
    for path in (source, *references):
        test, train = rule_sets(path)
        if not seen.isdisjoint(test | train):
            raise ValueError(f'horizon rule overlaps {path}')
    summarize(output, records, manifest['config']['blocks'])
    if not (output / 'complete.json').exists():
        save_json(output / 'complete.json', dict(blocks=len(records), fresh_games=len(seen),
                                                 scalar_trajectories=len(records) * len(CONDITIONS),
                                                 elapsed_seconds=time.perf_counter() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--reference', action='append', type=Path, default=[])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.run.resolve() == args.output.resolve():
        parser.error('--output must differ from the training run')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.run, args.output, args.workers, args.reference)


if __name__ == '__main__':
    main()
