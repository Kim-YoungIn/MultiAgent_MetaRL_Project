"""Independent raw replay and statistical verification of active-probe outcomes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from mamrl.structured_games import StructuredEnv, generate_structured_rule

FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')


def close(actual, expected, label, tol=2e-12):
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol, err_msg=label)


def visible(obs):
    state = list(obs.public_state[:2])
    return state + [0] * (2-len(state))


def probs(q, obs):
    values = q[obs.previous_own_action, obs.previous_opponent_action]
    best = np.isclose(values, np.max(values), rtol=0, atol=1e-12)
    return .25 / 3 + .75 * best / np.sum(best)


def advance(q, before, action, result):
    row = (before.previous_own_action, before.previous_opponent_action)
    next_row = (result.observations[1].previous_own_action,
                result.observations[1].previous_opponent_action)
    target = result.rewards[1]
    if not result.truncated:
        target += .8 * np.max(q[next_row])
    delta = .3 * (target - q[row + (action,)])
    q[row + (action,)] += delta
    return delta


def action_from_uniform(dist, uniform):
    return min(int(np.searchsorted(np.cumsum(dist), uniform, side='right')), 2)


class FixedGate:
    def __init__(self, uniform):
        self.uniform = uniform

    def random(self):
        return self.uniform


def action_values(env, distribution):
    values = np.zeros(3)
    for a in range(3):
        for b in range(3):
            clone = copy.deepcopy(env)
            values[a] += distribution[b] * clone.step(a, b).rewards[0]
    return values


def verify_row(data, index):
    motif = int(data['motif'][index])
    rule = generate_structured_rule(n=3, memory_depth=3,
             seed=int(data['game_seed'][index]), family=FAMILIES[motif],
             state_levels=(3, 3, 3), stochastic=True)
    prefix = data['prefix_a'].shape[1]
    env = StructuredEnv(rule, max_steps=prefix+2)
    obs = env.reset(seed=int(data['reset_seed'][index]))
    q = np.zeros((3, 3, 3))
    close(data['init_hist'][index], obs[0].action_history, 'reset history')
    close(data['prefix_public'][index, 0], visible(obs[0]), 'reset state')
    for t in range(prefix):
        p = probs(q, obs[1])
        b = action_from_uniform(p, data['prefix_u'][index, t])
        assert b == data['prefix_b'][index, t], 'simultaneous opponent draw'
        result = env.step(int(data['prefix_a'][index, t]), b)
        close(result.rewards, (data['prefix_reward'][index, t],
                               data['prefix_opp_reward'][index, t]), 'prefix reward')
        advance(q, obs[1], b, result)
        obs = result.observations
        close(data['prefix_public'][index, t+1], visible(obs[0]), 'prefix public state')
    initial_p = probs(q, obs[1])
    close(data['opp_p'][index], initial_p, 'pre-probe opponent distribution')
    first_b = action_from_uniform(initial_p, data['first_u'][index])
    assert first_b == data['first_b'][index]
    first_values = action_values(env, initial_p)
    close(data['first_value'][index], first_values, 'opening expected reward')
    for a in range(3):
        branch = copy.deepcopy(env)
        branch._rng = FixedGate(float(data['gate_u'][index]))
        q_branch = q.copy()
        result = branch.step(a, first_b)
        close(result.rewards, [data['first_reward'][index, a],
                               data['first_opp_reward'][index, a]], 'opening reward')
        close(visible(result.observations[0]), data['post_public'][index, a], 'post-opening state')
        delta = advance(q_branch, obs[1], first_b, result)
        if 'opponent_q_update' in data:
            close(abs(delta), data['opponent_q_update'][index, a], 'opponent Q update')
        p_post = probs(q_branch, result.observations[1])
        close(data['post_opp_p'][index, a], p_post, 'post-opening Q distribution')
        close(data['next_value'][index, a, 0], action_values(branch, p_post), 'live terminal')
        close(data['next_value'][index, a, 1], first_values, 'full rewind terminal')
        if 'next_value_no_update' in data:
            p_frozen = probs(q, result.observations[1])
            close(data['next_value_no_update'][index, a], action_values(branch, p_frozen),
                  'no-update opponent terminal')


def verify_observations(data):
    # Perturb unavailable variables, checking that neither model can receive them.
    from research.active_probe.policy import opening_features, terminal_features
    original_open = opening_features(data)
    original_terminal = terminal_features(data)
    hidden = {name: values.copy() for name, values in data.items()}
    hidden['game_seed'][:] = 42
    hidden['motif'][:] = 3
    hidden['first_opp_reward'][:] += 7
    hidden['opponent_q_update'][:] += 9
    hidden['next_value'][:] += 11
    hidden['first_value'][:] -= 13
    np.testing.assert_array_equal(opening_features(hidden), original_open)
    np.testing.assert_array_equal(terminal_features(hidden), original_terminal)
    changed = {name: values.copy() for name, values in data.items()}
    changed['first_reward'] += 2
    np.testing.assert_array_equal(opening_features(changed), original_open)
    changed_terminal = terminal_features(changed)
    np.testing.assert_array_equal(changed_terminal[:, :, 0], original_terminal[:, :, 0])
    if not np.any(changed_terminal[:, :, 1] != original_terminal[:, :, 1]):
        raise AssertionError('visible reward perturbation did not reach terminal inputs')
    permutation = np.array([1, 2, 0])
    relabeled = {name: values[:min(len(values), 8)].copy() for name, values in data.items()}
    for name in ('prefix_a', 'prefix_b', 'init_hist', 'first_b'):
        relabeled[name] = permutation[relabeled[name]]
    for name in ('first_reward', 'post_public'):
        old = relabeled[name].copy()
        relabeled[name][:, permutation] = old
    open2 = opening_features(relabeled)
    np.testing.assert_array_equal(open2[:, permutation], original_open[:len(open2)])
    term2 = terminal_features(relabeled)
    term2 = np.take(np.take(term2, permutation, axis=1), permutation, axis=3)
    np.testing.assert_array_equal(term2, original_terminal[:len(term2)])


def verify_data(data):
    assert all(len(value) == len(data['game_seed']) for value in data.values())
    verify_observations(data)
    for m in range(4):
        rows = np.flatnonzero(data['motif'] == m)
        assert len(rows)
        for i in rows[:min(3, len(rows))]:
            verify_row(data, int(i))
    first_expected = data['first_value'][:, None, :]
    close(data['next_value'][:, :, 1, :], np.broadcast_to(first_expected,
          data['next_value'][:, :, 1, :].shape), 'rewind values across all rows')
    return dict(samples=len(data['motif']), prefix=data['prefix_a'].shape[1],
                full_independent_replay=min(3, len(np.flatnonzero(data['motif']==0)))*4)


def network(state, x):
    # Re-evaluate weights independently with NumPy, including output normalization.
    def silu(v):
        return v / (1 + np.exp(-v))
    output = x
    for i in (0, 2, 4):
        weight = state[f'layers.{i}.weight'].detach().cpu().numpy()
        bias = state[f'layers.{i}.bias'].detach().cpu().numpy()
        output = output @ weight.T + bias
        if i < 4:
            output = silu(output)
    return output * state['scale'].detach().cpu().numpy() + state['center'].detach().cpu().numpy()


def check_predictions(data, weights):
    from research.active_probe.policy import opening_features, terminal_features
    opening = network(weights['opening'], opening_features(data))
    close(data['opening_scores'], opening, 'saved opening predictions', tol=1e-6)
    predicted = network(weights['terminal'], terminal_features(data))
    terminal = predicted.argmax(axis=3).transpose(0, 1, 3, 2)
    np.testing.assert_array_equal(data['terminal'], terminal)
    expected = []
    for j in (1, 3):
        immediate = opening[:, :, 0]
        blind = immediate + opening[:, :, j]
        expected.append(np.stack([immediate.argmax(1), blind.argmax(1),
                                  (blind+opening[:, :, j+1]).argmax(1)], axis=-1))
    expected = np.stack(expected, axis=1)
    np.testing.assert_array_equal(data['opening'], expected)
    indicator = data['prefix_a'][:, :, None] == np.arange(3)
    total = (indicator*data['prefix_reward'][:, :, None]).sum(axis=1)
    count = 1+indicator.sum(axis=1)
    opener = (total/count).argmax(axis=1)
    rows = np.arange(len(opener))
    total[rows, opener] += data['first_reward'][rows, opener]
    count[rows, opener] += 1
    np.testing.assert_array_equal(data['empirical_opening'], opener)
    np.testing.assert_array_equal(data['empirical_terminal'],
                                  np.stack([opener, (total/count).argmax(axis=1)], axis=1))


def primary(data):
    rows = np.arange(len(data['motif']))
    def value(regime, opening_index, channel):
        action = data['opening'][:, regime, opening_index]
        terminal = data['terminal'][rows, action, regime, channel]
        return data['first_reward'][rows, action] + data['next_value'][rows, action, regime, terminal]
    primary_live = np.mean(value(0, 2, 1) - value(0, 0, 1))
    primary_interaction = np.mean((value(1, 2, 1)-value(1, 1, 1))
                                  -(value(1, 2, 0)-value(1, 1, 0)))
    return np.array([primary_live, primary_interaction])


def verify_run(output):
    import torch
    blocks = sorted(output.glob('block_[0-9][0-9][0-9]'))
    manifest = json.loads((output/'manifest.json').read_text())
    expected_blocks = manifest['config']['blocks']
    assert len(blocks) == expected_blocks, 'missing independent blocks'
    root = Path(__file__).resolve().parents[2]
    for path, expected in manifest['source'].items():
        actual = hashlib.sha256((root/path).read_bytes()).hexdigest()
        archived = hashlib.sha256((output/'source'/path).read_bytes()).hexdigest()
        assert actual == expected and archived == expected, f'source mismatch: {path}'
    results = []
    identifiers = set()
    fingerprints = set()
    checked = []
    for block in blocks:
        record = json.loads((block/'record.json').read_text())
        for path, expected in record['sha256'].items():
            assert hashlib.sha256((block/path).read_bytes()).hexdigest() == expected, f'checksum mismatch: {block/path}'
        weights = torch.load(block/'weights.pt', map_location='cpu', weights_only=True)
        for name in ('terminal_train.npz', 'opening_train.npz', 'eval_k4.npz', 'eval_k0.npz', 'eval_k8.npz'):
            path = block/name
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            unique = set(data['game_seed'].tolist())
            assert identifiers.isdisjoint(unique), f'game seed overlap at {path}'
            identifiers.update(unique)
            fps, counts = np.unique(data['game_fingerprint'], return_counts=True)
            expected_resets = manifest['config']['train_resets'] if 'train' in name else 2
            assert np.all(counts == expected_resets), f'incorrect game resets: {path}'
            assert fingerprints.isdisjoint(set(fps.tolist())), f'exact fingerprint overlap: {path}'
            fingerprints.update(fps.tolist())
            info = verify_data(data)
            info['file'] = str(path)
            checked.append(info)
            if name.startswith('eval_'):
                check_predictions(data, weights)
            if name == 'eval_k4.npz':
                results.append(primary(data))
    differences = np.array(results)
    draws = np.random.default_rng(2817601).integers(0, len(differences), size=(20000, len(differences)))
    means = differences.mean(axis=0)
    intervals = np.quantile(differences[draws].mean(axis=1), [.0125, .9875], axis=0).T
    summary = json.loads((output/'summary.json').read_text())
    for i, item in enumerate(summary['primary']):
        close([item['estimate'], item['lower'], item['upper']],
              [means[i], *intervals[i]], 'reported primary interval')
    return dict(blocks=len(blocks), game_seeds=len(identifiers),
                exact_fingerprints=len(fingerprints), source_and_block_hashes_match=True,
                analysis_primary_intervals_match=True,
                checked=checked, primary_estimates=means.tolist(),
                bootstrap_97_5_intervals=intervals.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='saved raw NPZ dataset or run directory')
    args = parser.parse_args()
    if args.data.is_dir():
        report = verify_run(args.data)
        (args.data/'independent_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    else:
        with np.load(args.data, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        report = verify_data(data)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
