"""Check v2 budgets, state controls, and archived trajectories against the scalar game."""
import argparse
import json
from pathlib import Path

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.fast_exploration_v1 import batch, study
from research.fast_exploration_v1.prior import auc_weights
from research.fast_exploration_v1.scale import digest
from research.structured_oracle.run import Gate
from . import core
from .run import completed


def check_rewards(rule, seed, actions, uniforms, rewards, features=None):
    """Replay the executed joint actions through the original scalar simulator."""
    if not len(actions):
        return
    env = StructuredEnv(rule, max_steps=len(actions))
    observation, _ = env.reset(seed=int(seed))
    actual = []
    for t, ((a, b), (_, gate)) in enumerate(zip(actions, uniforms, strict=True)):
        if features is not None:
            np.testing.assert_allclose(study.features(observation, 3)[a], features[t], atol=1e-12, rtol=0)
        env._rng = Gate(float(gate))
        result = env.step(int(a), int(b))
        actual.append(result.rewards)
        observation = result.observations[0]
    np.testing.assert_allclose(actual, rewards, atol=1e-12, rtol=0)


def check_block(raw, cfg, record):
    methods = raw['methods'].tolist()
    indices = {m: i for i, m in enumerate(methods)}
    ks, h, budget = cfg['ks'], cfg['h'], cfg['budget_h']
    np.testing.assert_array_equal(raw['ks'], ks)
    np.testing.assert_array_equal(methods, core.METHODS)
    groups = [[i for i, m in enumerate(methods) if m.startswith(prior)] for prior in ('pooled_', 'ips_')]
    for name, length in [('controlled', h), ('natural', h), ('frozen', h), ('fixed', budget),
                         ('retain_both', h), ('reset_student', h)]:
        a, r = raw[f'{name}_actions'], raw[f'{name}_rewards']
        assert a.shape == r.shape and a.shape[-2:] == (length, 2)
        assert np.isfinite(r).all() and np.logical_and(a >= 0, a < 3).all()
        for group in groups if name in ('controlled', 'natural', 'frozen', 'fixed') else [list(range(a.shape[0]))]:
            for index in group[1:]:
                np.testing.assert_array_equal(a[group[0], 0], a[index, 0])
                np.testing.assert_array_equal(r[group[0], 0], r[index, 0])
    for phase in ('probe', 'frozen_probe'):
        a, r = raw[f'{phase}_actions'], raw[f'{phase}_rewards']
        for j, k in enumerate(ks):
            assert (a[:, j, :, k:] == -1).all() and np.isnan(r[:, j, :, k:]).all()
            assert np.logical_and(a[:, j, :, :k] >= 0, a[:, j, :, :k] < 3).all()
            assert np.isfinite(r[:, j, :, :k]).all()
    training = raw['train_actions']
    for method, name in enumerate(core.TRAINED):
        lengths = np.full(len(training), budget) if '_fixed_' in name else raw['train_k'] + h
        valid = np.arange(training.shape[-2])[None, :] < lengths[:, None]
        chosen = training[:, method].transpose(0, 2, 1, 3)
        assert np.logical_and(chosen[valid] >= 0, chosen[valid] < 3).all()
        assert (chosen[~valid] == -1).all()
    for prior in ('pooled', 'ips'):
        i = indices[f'{prior}_greedy']
        for k in range(1, len(ks)):
            np.testing.assert_array_equal(raw['fixed_actions'][i, k], raw['fixed_actions'][i, 0])
            np.testing.assert_array_equal(raw['fixed_rewards'][i, k], raw['fixed_rewards'][i, 0])

    # Sample one game per family. Rotate the trained methods and nonzero K.
    checks = 0
    for j, family in enumerate(np.unique(raw['families'])):
        s = int(np.nonzero(raw['families'] == family)[0][0])
        game = int(raw['game_seeds'][s // cfg['test_resets']])
        ep = batch.prepare([int(raw['reset_seeds'][s])], [str(family)], max(ks), max(h, budget),
                           rule_seeds=[game])
        rule = ep.rules[0]
        np.testing.assert_array_equal(raw['action_to_role'][s], np.argsort(rule.roles))
        contexts, action_values = core.common_queries(ep, batch.warm_up(ep))
        for mi in (indices['pooled_full'], indices['pooled_masked'], indices['pooled_fixed_total']):
            ki = 1 + (mi + j) % (len(ks) - 1)
            k = ks[ki]
            x = raw['probe_features'][mi, ki, s, :k]
            model = study.Model(3, record['pooled_prior'])
            for feature, a, r in zip(x, raw['probe_actions'][mi, ki, s, :k, 0],
                                     raw['probe_rewards'][mi, ki, s, :k, 0], strict=True):
                model.observe(np.broadcast_to(feature, (3, 20)), a, r)
            query = np.mean([v[0, model.act(context[0], np.arange(3))]
                             for context, v in zip(contexts, action_values, strict=True)])
            np.testing.assert_allclose(query, raw['query_values'][mi, ki, s], atol=1e-12, rtol=0)
            for phase, seed, draws in (
                ('probe', ep.explore_seed[0], ep.explore_uniform[:k]),
                ('frozen_probe', ep.explore_seed[0], ep.explore_uniform[:k]),
                ('controlled', ep.assess_seed[0], ep.assess_uniform[:h]),
                ('natural', ep.assess_seed[0], ep.assess_uniform[:h]),
                ('frozen', ep.assess_seed[0], ep.assess_uniform[:h]),
                ('fixed', ep.explore_seed[0], ep.assess_uniform[:budget]),
            ):
                n = len(draws)
                check_rewards(rule, seed, raw[f'{phase}_actions'][mi, ki, s, :n],
                              draws[:, 0], raw[f'{phase}_rewards'][mi, ki, s, :n],
                              x if phase == 'probe' else None)
                checks += 1
    return checks


def rule_sets(root):
    test, train = set(), set()
    for p in root.glob('block_*.npz'):
        with np.load(p) as raw:
            test.update(raw['test_fingerprints'])
            train.update(raw['prior_fingerprints'])
            if 'train_fingerprints' in raw:
                train.update(raw['train_fingerprints'].flat)
    return test, train


def verify(root, references):
    manifest = json.loads((root / 'manifest.json').read_text())
    completion_path = root / 'complete.json'
    completion = json.loads(completion_path.read_text())
    assert digest(root / 'manifest.json') == completion['manifest_sha256']
    cfg = manifest['config']
    for p, expected in manifest['source'].items():
        assert digest(root / 'source' / p) == expected, p
    assert completion['blocks'] == cfg['blocks']
    total = 0
    for i in range(cfg['blocks']):
        assert completed(root, i)
        prefix = root / f'block_{i:03d}'
        record = json.loads(prefix.with_suffix('.json').read_text())
        with np.load(prefix.with_suffix('.npz')) as raw:
            total += check_block(raw, cfg, record)
    current_test, current_train = rule_sets(root)
    assert current_test.isdisjoint(current_train)
    for path in references:
        other_test, other_train = rule_sets(path)
        if not other_test:
            raise ValueError(f'no reference games in {path}')
        assert current_test.isdisjoint(other_test | other_train)
        assert other_test.isdisjoint(current_train)
    return dict(blocks=cfg['blocks'], scalar_trajectories=total,
                actual_fixed_budget=cfg['budget_h'], checked_against=[str(p) for p in references])


def calibrate(root):
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['split'] != 'dev':
        raise ValueError('UCB calibration uses only the development panel')
    cfg = manifest['config']
    betas = (.05, .15, .25, .50)
    weights = auc_weights(tuple(cfg['ks']))
    by_block = []
    for block in range(cfg['blocks']):
        if not completed(root, block):
            raise ValueError(f'missing development block {block}')
        prefix = root / f'block_{block:03d}'
        prior = np.array(json.loads(prefix.with_suffix('.json').read_text())['pooled_prior'])
        with np.load(prefix.with_suffix('.npz')) as raw:
            seeds, resets, families = (raw[key] for key in ('game_seeds', 'reset_seeds', 'families'))
        ep = batch.prepare(resets, families, max(cfg['ks']), cfg['h'],
                           rule_seeds=np.repeat(seeds, cfg['test_resets']))
        q = batch.warm_up(ep)
        targets = []
        for beta in betas:
            curve = []
            for k in cfg['ks']:
                env = core.reset_assessment(ep, cfg['h'])
                probes = core.probe(ep, q, prior, 'pooled_ucb', None, None, k, cfg['h'],
                                    beta=beta, reset=env.last)
                tape = core.assess(env, ep.assess_uniform, probes.student, q, learn=False)
                curve.append(tape.rewards[..., 0].mean())
            targets.append(float(weights @ curve))
        by_block.append(targets)
    means = np.mean(by_block, axis=0)
    return dict(split='dev', dev_manifest_sha256=digest(root / 'manifest.json'),
                candidates=betas, block_aucs=by_block,
                mean_aucs=means.tolist(), selected_beta=betas[int(np.argmax(means))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', action='append', default=[], type=Path)
    parser.add_argument('--calibrate', action='store_true', help='score the prespecified UCB grid on development')
    args = parser.parse_args()
    result = calibrate(args.run) if args.calibrate else verify(args.run, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
