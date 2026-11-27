"""Independently replay sampled policies and recompute the primary paired result."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mamrl.structured_games import StructuredEnv
from research.fast_exploration_v1 import batch, study
from research.fast_exploration_v2.verify import rule_sets
from research.structured_oracle.run import Gate
from research.structured_scale.run import Model, Opponent, features, game
from .rules import variant

FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
DISTANCES = (0., .3, .6)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scalar_step(env, observations, opponent, own, draw, learn, next_obs=None, terminal=False):
    other = min(2, int(np.searchsorted(opponent.distribution(observations[1]).cumsum(), draw[0], side='right')))
    env._rng = Gate(float(draw[1]))
    result = env.step(int(own), other)
    if learn:
        opponent.update(observations[1], other, result.rewards[1],
                        result.observations[1] if next_obs is None else next_obs[1], terminal)
    return result, other


def scalar_trajectory(rule, episode, prior, actor, method, controlled):
    """Reconstruct warm-up, policy and both learners without the batch simulator."""
    warm = StructuredEnv(rule, max_steps=32)
    obs = warm.reset(seed=int(episode.warm_seed[0]))
    opponent = Opponent(3)
    for t in range(32):
        result, _ = scalar_step(warm, obs, opponent, episode.warm_action[t, 0],
                                episode.warm_uniform[t, 0], True, terminal=t == 31)
        obs = result.observations
    warm_q = opponent.q.copy()
    np.testing.assert_allclose(warm_q, batch.warm_up(episode)[0], atol=1e-12, rtol=0)
    student = Model(3, prior)
    env = StructuredEnv(rule, max_steps=8 if controlled else 96)
    obs = env.reset(seed=int(episode.explore_seed[0]))
    reset = StructuredEnv(rule, max_steps=32)
    reset_obs = reset.reset(seed=int(episode.assess_seed[0]))
    teacher_draws = np.random.default_rng(episode.teacher_seed[0]).random(8)
    hidden = actor.initial_hidden(n=3, batch=1) if actor is not None else None
    actions, rewards = [], []
    probe_length = 0 if method == 'zero' or method == 'pooled_greedy' else 8
    turns = probe_length if controlled else 96
    with torch.no_grad():
        for t in range(turns):
            x = features(obs[0], 3)
            if t < probe_length and actor is not None:
                masked = method == 'pooled_masked'
                inputs = actor.observation_features(obs[0], student, x, belief=not masked)
                if masked:
                    inputs[:, 1:3] = 0
                    if t == 0:
                        inputs[:, 0] = 0
                logits, _, hidden = actor.forward_features(torch.from_numpy(inputs[None]), hidden)
                cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1).numpy()[0]
                action = min(2, int((cdf <= teacher_draws[t]).sum()))
            elif t < probe_length and method == 'pooled_ucb':
                means = np.einsum('ad,ad->a', student.mean, x)
                variance = np.einsum('ad,ade,ae->a', x, student.inverse, x)
                action = int(np.argmax(means + .15 * np.sqrt(np.maximum(variance, 0))))
            else:
                action = student.act(x, np.arange(3))
            draw = episode.explore_uniform[t, 0] if controlled else episode.assess_uniform[t, 0]
            result, other = scalar_step(env, obs, opponent, action, draw, True,
                                        next_obs=reset_obs if controlled and t == probe_length-1 else None,
                                        terminal=(not controlled and t == 95))
            student.observe(x, action, result.rewards[0])
            actions.append((action, other))
            rewards.append(result.rewards)
            obs = result.observations
    if not controlled:
        return np.asarray(actions), np.asarray(rewards)
    # The probe updates Student; assessment starts from reset state and warmed Q.
    probe_actions, probe_rewards = np.asarray(actions, dtype=int).reshape(-1, 2), np.asarray(rewards).reshape(-1, 2)
    opponent.q = warm_q.copy()
    obs = reset_obs
    actions, rewards = [], []
    for t in range(32):
        x = features(obs[0], 3)
        action = student.act(x, np.arange(3))
        result, other = scalar_step(reset, obs, opponent, action, episode.assess_uniform[t, 0], False)
        actions.append((action, other))
        rewards.append(result.rewards)
        obs = result.observations
    return (probe_actions, probe_rewards), (np.asarray(actions), np.asarray(rewards))


def verify(root, source, reference, analysis):
    torch.set_num_threads(1)
    manifest = json.loads((root / 'manifest.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    assert complete['manifest_sha256'] == digest(root / 'manifest.json')
    assert manifest['source_manifest_sha256'] == digest(source / 'manifest.json')
    assert manifest['reference_manifest_sha256'] == digest(reference / 'manifest.json')
    for name, expected in manifest['sources'].items():
        assert digest(root / 'source' / name) == expected, name
    old_test, old_train = rule_sets(source)
    old_horizon = set()
    reference_complete = json.loads((reference / 'complete.json').read_text())
    reference_files = list(reference.glob('block_*.npz'))
    assert len(reference_files) == reference_complete['blocks']
    for path in reference_files:
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == receipt['raw_sha256']
        with np.load(path, allow_pickle=False) as raw:
            old_horizon.update(raw['test_fingerprints'])
    assert len(old_horizon) == reference_complete['fresh_games']
    assert old_test and old_train and old_horizon
    seen, variants, values, masters = set(), set(), [], []
    trajectories = 0
    blocks = manifest['blocks']
    assert len(blocks) == complete['blocks']
    for block in blocks:
        stem = f'block_{block:03d}'
        record = json.loads((root / f'{stem}.json').read_text())
        old = json.loads((source / f'{stem}.json').read_text())
        assert old['block'] == record['block'] == block
        assert (record['master_seed'], record['seed_block']) == (old['master_seed'], old['seed_block'])
        assert digest(source / f'{stem}.json') == manifest['source_receipts'][str(block)]
        assert digest(source / f'{stem}.pt') == old['weights_sha256'] == record['teacher_sha256']
        assert digest(source / f'{stem}.npz') == old['raw_sha256']
        assert digest(root / f'{stem}.npz') == record['raw_sha256']
        weights = torch.load(source / f'{stem}.pt', map_location='cpu', weights_only=True)
        actors = {}
        for name in ('pooled_full', 'pooled_masked', 'pooled_fixed_total'):
            actors[name] = study.BeliefActor()
            actors[name].load_state_dict(weights[name])
            actors[name].eval()
        with np.load(root / f'{stem}.npz', allow_pickle=False) as raw:
            count = manifest['games_per_block']
            rng = np.random.default_rng(np.random.SeedSequence([
                253721399 if manifest['split'] == 'final' else 184728591,
                record['master_seed'], record['seed_block']]))
            np.testing.assert_array_equal(raw['game_seeds'], rng.integers(2**63, 2**64, count, dtype=np.uint64))
            np.testing.assert_array_equal(raw['reset_seeds'], rng.integers(0, 2**61, count * 2, dtype=np.int64))
            assert len(set(raw['reset_seeds'])) == count * 2
            families = raw['families']
            assert list(families) == [FAMILIES[i % 4] for i in range(count) for _ in range(2)]
            controlled = raw['controlled_rewards']
            assert controlled.shape == (2, 3, 3, count * 2, 32, 2)
            assert raw['live_rewards'].shape == (2, 3, 3, count * 2, 96, 2)
            assert raw['probe_rewards'].shape == (2, 3, 2, count * 2, 8, 2)
            for name in ('live', 'controlled', 'probe'):
                a, r = raw[name + '_actions'], raw[name + '_rewards']
                assert a.shape == r.shape and np.isfinite(r).all() and ((a >= 0) & (a < 3)).all()
            for i, seed in enumerate(raw['game_seeds']):
                base = game(int(seed), FAMILIES[i % 4], 3, 3)
                fingerprint = base.fingerprint()
                assert fingerprint == raw['base_fingerprints'][i] and fingerprint not in seen
                seen.add(fingerprint)
                for li in range(2):
                    for di, d in enumerate(DISTANCES):
                        fp = variant(base, d, li).fingerprint()
                        assert fp == raw['variant_fingerprints'][li, di, i]
                        variants.add(fp)
            # Balanced family means, then paired difference in differences.
            r = controlled[..., 0].mean(axis=-1)
            family_values = []
            for family in FAMILIES:
                selected = families == family
                means = r[:, :, :, selected].mean(axis=-1)
                family_values.append((means[0, 2, 0] - means[0, 2, 1]) -
                                     (means[0, 0, 0] - means[0, 0, 1]))
            values.append(float(np.mean(family_values)))
            masters.append(record['master_seed'])
            for fi, family in enumerate(FAMILIES):
                # Rotate the intervention and reset across blocks while covering every family.
                index = 2 * fi + ((block + fi) % 2)
                li, di = divmod((block + fi) % 6, 3)
                base = game(int(raw['game_seeds'][index // 2]), family, 3, 3)
                rule = variant(base, DISTANCES[di], li)
                ep = batch.prepare([int(raw['reset_seeds'][index])], [family], 8, 96,
                                   rule_seeds=[int(raw['game_seeds'][index // 2])])
                ep.rules = [rule]
                for mi, name in enumerate(('pooled_full', 'pooled_masked', 'zero')):
                    probe, assessment = scalar_trajectory(rule, ep, old['pooled_prior'], actors.get(name), name, True)
                    if mi < 2:
                        np.testing.assert_array_equal(probe[0], raw['probe_actions'][li, di, mi, index])
                        np.testing.assert_allclose(probe[1], raw['probe_rewards'][li, di, mi, index], atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(assessment[0], raw['controlled_actions'][li, di, mi, index])
                    np.testing.assert_allclose(assessment[1], raw['controlled_rewards'][li, di, mi, index], atol=1e-12, rtol=0)
                    trajectories += 1
                for mi, name in enumerate(('pooled_fixed_total', 'pooled_ucb', 'pooled_greedy')):
                    actions, rewards = scalar_trajectory(rule, ep, old['pooled_prior'], actors.get(name), name, False)
                    np.testing.assert_array_equal(actions, raw['live_actions'][li, di, mi, index])
                    np.testing.assert_allclose(rewards, raw['live_rewards'][li, di, mi, index], atol=1e-12, rtol=0)
                    trajectories += 1
    assert len(seen) == complete['base_games'] == len(blocks) * manifest['games_per_block']
    assert len(variants) == complete['variants'] == 2 * len(DISTANCES) * len(seen)
    assert (seen | variants).isdisjoint(old_test | old_train | old_horizon)
    values = np.array(values)
    rng = np.random.default_rng(882734)
    groups = [np.flatnonzero(np.array(masters) == seed) for seed in dict.fromkeys(masters)]
    draws = np.concatenate([rng.choice(group, (20000, len(group)), replace=True) for group in groups], axis=1)
    ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
    result = json.loads((analysis / 'summary.json').read_text())['primary']
    np.testing.assert_allclose([values.mean(), values.std(ddof=1), *ci],
                               [result['mean'], result['block_sd'], result['ci_low'], result['ci_high']], atol=1e-12, rtol=0)
    return dict(manifest_sha256=digest(root / 'manifest.json'),
                summary_sha256=digest(analysis / 'summary.json'),
                blocks=len(blocks), base_games=len(seen), variant_fingerprints=len(variants),
                sampled_scalar_trajectories=trajectories, scalar_sampling='one episode per family and block; intervention and reset rotate by block/family; 3 controlled and 3 continuous methods',
                scalar_scope='full warm-up, Teacher and masked inputs, Student and Q-learning actions, resets and both players rewards',
                all_source_receipts_and_raw_hashes_checked=True, previous_panel_fingerprint_overlap=0,
                primary_mean=float(values.mean()), primary_block_sd=float(values.std(ddof=1)), primary_ci95=ci.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.source, args.reference, args.analysis)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
