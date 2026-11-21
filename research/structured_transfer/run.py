"""Run predeclared structured-game empirical-Bayes transfer experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
from mamrl.structured_games import StructuredEnv, generate_structured_rule

FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
METHODS = ('meta_mixed', 'meta_resource', 'frozen_mixed', 'frozen_resource',
           'pooled', 'pooled_adapt', 'scratch')
BUDGETS = (0, 4, 8, 16)
D = 20


def features(obs):
    result = np.zeros(D)
    result[0] = 1
    result[1:3] = np.asarray(obs.public_state[:2], dtype=float) / 2
    result[3] = obs.previous_own_reward
    result[4] = obs.episode_start_flag
    result[5 + 3 * obs.action_history[0][0] + obs.action_history[0][1]] = 1
    result[14 + obs.action_history[1][0]] = 1
    result[17 + obs.action_history[2][0]] = 1
    return result


class Opponent:
    def __init__(self):
        self.q = np.zeros((3, 3, 3))
        self.change = 0.0
        self.updates = 0

    @staticmethod
    def key(obs):
        return obs.previous_own_action, obs.previous_opponent_action

    def distribution(self, obs):
        vals = self.q[self.key(obs)]
        best = np.isclose(vals, vals.max(), rtol=0, atol=1e-12)
        return np.full(3, .25 / 3) + .75 * best / best.sum()

    def update(self, obs, action, reward, next_obs, terminal):
        key = self.key(obs)
        target = reward + (0 if terminal else .8 * self.q[self.key(next_obs)].max())
        delta = .3 * (target - self.q[key][action])
        self.q[key][action] += delta
        self.change += abs(delta)
        self.updates += 1


def draw(probs, u):
    return min(int(np.searchsorted(np.cumsum(probs), u, side='right')), 2)


def create_game(seed, family):
    return generate_structured_rule(n=3, memory_depth=3, seed=int(seed), family=family,
                                    state_levels=(3, 3, 3), stochastic=True)


def step(env, opponent, obs, action, u):
    opp_action = draw(opponent.distribution(obs[1]), u)
    outcome = env.step(action, opp_action)
    opponent.update(obs[1], opp_action, outcome.rewards[1], outcome.observations[1],
                    outcome.truncated)
    return outcome, opp_action


def collect(game, seed, action_rng, opp_rng, steps):
    env = StructuredEnv(game, max_steps=steps)
    obs = env.reset(seed=int(seed))
    opp = Opponent()
    rows = []
    for _ in range(steps):
        x = features(obs[0])
        a = int(action_rng.integers(3))
        outcome, b = step(env, opp, obs, a, float(opp_rng.random()))
        rows.append((x, a, b, outcome.rewards[0], outcome.rewards[1]))
        obs = outcome.observations
    return rows


def fit(tasks):
    means = np.zeros((3, D))
    pooled = np.zeros((3, D))
    for a in range(3):
        mat = np.eye(D) * .01
        vec = np.zeros(D)
        pm = np.eye(D) * .01
        pv = np.zeros(D)
        for rows in tasks:
            selected = [r for r in rows if r[1] == a]
            if not selected:
                continue
            x = np.asarray([r[0] for r in selected])
            y = np.asarray([r[3] for r in selected])
            s = x.T @ x
            v = x.T @ y
            transform = np.linalg.solve(np.eye(D) + s, np.eye(D))
            mat += transform @ s
            vec += transform @ v
            pm += s
            pv += v
        means[a] = np.linalg.solve(mat, vec)
        pooled[a] = np.linalg.solve(pm, pv)
    return means, pooled


class Model:
    def __init__(self, mean, update):
        self.mean = mean.copy()
        self.inverse = np.repeat(np.eye(D)[None, :, :], 3, axis=0)
        self.update_enabled = update

    def act(self, x):
        return int(np.argmax(self.mean @ x))

    def observe(self, x, a, reward):
        if self.update_enabled:
            inv_x = self.inverse[a] @ x
            self.mean[a] += inv_x * ((reward - x @ self.mean[a]) / (1 + x @ inv_x))
            self.inverse[a] -= np.outer(inv_x, inv_x) / (1 + x @ inv_x)


def models(mixed, resource, pooled):
    return [Model(mixed, True), Model(resource, True), Model(mixed, False),
            Model(resource, False), Model(pooled, False), Model(pooled, True),
            Model(np.zeros((3, D)), True)]


def values_at(env, opp, obs):
    probabilities = opp.distribution(obs[1])
    values = np.zeros(3)
    for a in range(3):
        for b in range(3):
            values[a] += probabilities[b] * copy.deepcopy(env).step(a, b).rewards[0]
    return values


def evaluate(game, seed, rng, mixed, resource, pooled):
    # Probe includes a query at 16, hence the environment's horizon exceeds it.
    env = StructuredEnv(game, max_steps=17)
    obs = env.reset(seed=seed)
    opp = Opponent()
    learners = models(mixed, resource, pooled)
    values = np.zeros((len(METHODS), len(BUDGETS)))
    actions = np.zeros((len(METHODS), len(BUDGETS)), dtype=int)
    oracle = np.zeros((len(BUDGETS), 3))
    probe = []
    random_actions = rng.integers(3, size=16)
    uniforms = rng.random(16)
    states = []
    for t in range(17):
        x = features(obs[0])
        if t in BUDGETS:
            k = BUDGETS.index(t)
            truth = values_at(env, opp, obs)
            oracle[k] = truth
            states.append({'public': list(obs[0].public_state),
                           'history': list(obs[0].action_history),
                           'opponent_probs': opp.distribution(obs[1]).tolist()})
            for m, learner in enumerate(learners):
                actions[m, k] = learner.act(x)
                values[m, k] = truth[actions[m, k]]
        if t == 16:
            break
        a = int(random_actions[t])
        outcome, b = step(env, opp, obs, a, float(uniforms[t]))
        for learner in learners:
            learner.observe(x, a, outcome.rewards[0])
        probe.append((x.tolist(), a, b, *outcome.rewards, float(uniforms[t])))
        obs = outcome.observations
    probe_change = opp.change
    # Match random streams and reset seed across methods, with isolated states.
    rollout_a = np.zeros((len(METHODS), 32), dtype=int)
    rollout_b = np.zeros_like(rollout_a)
    rewards = np.zeros((len(METHODS), 32, 2))
    rollout_states = np.zeros((len(METHODS), 32, 2), dtype=int)
    changes = np.zeros(len(METHODS))
    uniforms = rng.random(32)
    for m, learner in enumerate(models(mixed, resource, pooled)):
        env = StructuredEnv(game, max_steps=32)
        obs = env.reset(seed=seed + 1)
        opp = Opponent()
        for t in range(32):
            x = features(obs[0])
            a = learner.act(x)
            outcome, b = step(env, opp, obs, a, float(uniforms[t]))
            learner.observe(x, a, outcome.rewards[0])
            rollout_a[m, t], rollout_b[m, t] = a, b
            rewards[m, t] = outcome.rewards
            rollout_states[m, t] = outcome.observations[0].public_state[:2]
            obs = outcome.observations
        changes[m] = opp.change
    return dict(values=values, actions=actions, oracle=oracle, probe=probe,
                probe_states=states, probe_change=probe_change,
                rollout_a=rollout_a, rollout_b=rollout_b, rewards=rewards,
                rollout_states=rollout_states, changes=changes,
                rollout_uniforms=uniforms)


def json_save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def block(index, base_seed, out):
    started = time.monotonic()
    root_rng = np.random.default_rng(np.random.SeedSequence([base_seed, index]))
    seeds = root_rng.integers(0, 2**62, size=32 * 3, dtype=np.int64)
    fingerprints = set()
    catalog = {}
    trained = {}
    for arm, offset in (('mixed', 0), ('resource', 32)):
        tasks, specs = [], []
        for i in range(32):
            seed = int(seeds[offset + i])
            family = FAMILIES[i % 4] if arm == 'mixed' else 'resource'
            game = create_game(seed, family)
            digest = game.fingerprint()
            assert digest not in fingerprints
            fingerprints.add(digest)
            r = np.random.default_rng(np.random.SeedSequence([base_seed, index, offset + i, 10]))
            rows = collect(game, seed + 1, r, r, 32)
            tasks.append(rows)
            specs.append({'seed': seed, 'fingerprint': digest, 'game': game.to_dict(),
                          'rows': [[x.tolist(), a, b, y, z] for x, a, b, y, z in rows]})
        trained[arm] = fit(tasks)
        catalog[arm] = specs
    mixed, pooled = trained['mixed']
    resource, _ = trained['resource']
    tests = []
    all_values, all_actions, all_oracles, all_ra, all_rb = [], [], [], [], []
    all_rewards, all_states, all_changes, all_probe_changes = [], [], [], []
    for i in range(32):
        seed = int(seeds[64 + i])
        family = FAMILIES[i % 4]
        game = create_game(seed, family)
        digest = game.fingerprint()
        assert digest not in fingerprints
        fingerprints.add(digest)
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, index, i, 30]))
        r = evaluate(game, seed + 1, rng, mixed, resource, pooled)
        tests.append({'seed': seed, 'fingerprint': digest, 'game': game.to_dict(),
                      'probe': r['probe'], 'probe_states': r['probe_states'],
                      'rollout_uniforms': r['rollout_uniforms'].tolist()})
        all_values.append(r['values']); all_actions.append(r['actions'])
        all_oracles.append(r['oracle']); all_ra.append(r['rollout_a']); all_rb.append(r['rollout_b'])
        all_rewards.append(r['rewards']); all_states.append(r['rollout_states'])
        all_changes.append(r['changes']); all_probe_changes.append(r['probe_change'])
    folder = out / f'block_{index:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    json_save(folder / 'train.json', catalog)
    json_save(folder / 'test.json', tests)
    np.savez_compressed(folder / 'results.npz', values=all_values, actions=all_actions,
        oracle=all_oracles, rollout_a=all_ra, rollout_b=all_rb, rewards=all_rewards,
        rollout_states=all_states, changes=all_changes, probe_changes=all_probe_changes,
        mixed=mixed, resource=resource, pooled=pooled,
        families=np.array([FAMILIES[i % 4] for i in range(32)]))
    json_save(folder / 'complete.json', {'index': index, 'seconds': time.monotonic()-started,
                                        'fingerprints': len(fingerprints)})
    return index


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--blocks', type=int, default=16)
    parser.add_argument('--seed', type=int, default=67834219)
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('blocks must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
               Path('05_toy_models/mamrl/structured_games.py'),
               Path('05_toy_models/mamrl/rules.py'), Path('05_toy_models/mamrl/tensor.py')]
    manifest = {'seed': args.seed, 'blocks': args.blocks, 'methods': METHODS,
                'budgets': BUDGETS, 'python': sys.version, 'numpy': np.__version__,
                'platform': platform.platform(), 'source_sha256': {str(x): sha(x) for x in sources}}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('configuration or source mismatch on resume')
    else:
        json_save(manifest_path, manifest)
    for i in range(args.blocks):
        path = out / f'block_{i:03d}'
        if (path / 'complete.json').exists():
            print('reusing', i, flush=True)
            continue
        if path.exists():
            raise RuntimeError(f'incomplete block retained at {path}; move it before resuming')
        print('starting', i, flush=True)
        block(i, args.seed, out)
    print('complete', flush=True)


if __name__ == '__main__':
    main()
