"""Contract checks for paired structured-game opponent trajectories."""
import copy
import unittest

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.opponent_adaptation.data import HORIZON, sample
from research.structured_oracle.run import Gate
from research.structured_scale.run import FAMILIES, Opponent, features, game, step


class DatasetContracts(unittest.TestCase):
    def test_validation_and_reproducibility(self):
        for args in [(-1, 4), (1, 0), (1, 5)]:
            with self.assertRaises(ValueError):
                sample(*args)
        first, second = sample(917, 4), sample(917, 4)
        self.assertEqual(set(first), set(second))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)
        self.assertEqual(len(set(first['fingerprint'])), 4)
        np.testing.assert_array_equal(first['family'], np.arange(4))

    def test_matched_initial_state_and_real_warmup(self):
        data = sample(6271, 8)
        np.testing.assert_array_equal(data['x'][:, 0, 0], data['x'][:, 1, 0])
        np.testing.assert_array_equal(data['opp_p'][:, 0, 0], data['opp_p'][:, 1, 0])
        np.testing.assert_array_equal(data['truth'][:, 0, 0], data['truth'][:, 1, 0])
        self.assertTrue(np.all(data['reset_seed'] != data['warm_seed']))
        self.assertTrue(np.all(data['q_delta'][:, 0] == 0))
        self.assertGreater(data['q_delta'][:, 1].sum(), 0)
        for i in range(8):
            rule = game(int(data['seed'][i]), FAMILIES[int(data['family'][i])], 3, 3)
            self.assertEqual(rule.fingerprint(), data['fingerprint'][i])
            env = StructuredEnv(rule, max_steps=HORIZON)
            obs = env.reset(seed=int(data['warm_seed'][i]))
            opponent = Opponent(3)
            for t in range(HORIZON):
                outcome, b = step(env, opponent, obs, int(data['warm_a'][i, t]),
                                  float(data['warm_u'][i, t]))
                self.assertEqual(b, data['warm_b'][i, t])
                self.assertAlmostEqual(outcome.rewards[1], data['warm_r'][i, t])
                obs = outcome.observations
            np.testing.assert_allclose(opponent.q, data['warm_q'][i], atol=1e-12)

    def test_evaluation_queries_and_opponent_update(self):
        data = sample(735, 4)
        for i in range(4):
            rule = game(int(data['seed'][i]), FAMILIES[int(data['family'][i])], 3, 3)
            for condition in (0, 1):
                env = StructuredEnv(rule, max_steps=HORIZON + 1)
                obs = env.reset(seed=int(data['reset_seed'][i]))
                opponent = Opponent(3)
                opponent.q = data['warm_q'][i].copy()
                original = opponent.q.copy()
                for t in range(HORIZON + 1):
                    p = opponent.distribution(obs[1])
                    np.testing.assert_allclose(p, data['opp_p'][i, condition, t], atol=1e-12)
                    np.testing.assert_allclose(features(obs[0], 3), data['x'][i, condition, t])
                    # Query the actual simulator for every simultaneous action.
                    exact = np.empty((3, 3))
                    for a in range(3):
                        for b in range(3):
                            exact[a, b] = copy.deepcopy(env).step(a, b).rewards[0]
                    np.testing.assert_allclose(exact @ p, data['truth'][i, condition, t], atol=1e-12)
                    if t == HORIZON:
                        break
                    b = min(int(np.searchsorted(np.cumsum(p), data['u'][i, t], side='right')), 2)
                    self.assertEqual(b, data['b'][i, condition, t])
                    env._rng = Gate(float(data['gate_u'][i, t]))
                    result = env.step(int(data['a'][i, t]), b)
                    self.assertAlmostEqual(result.rewards[0], data['y'][i, condition, t])
                    self.assertAlmostEqual(result.rewards[1], data['opp_y'][i, condition, t])
                    old = opponent.q.copy()
                    if condition:
                        opponent.update(obs[1], b, result.rewards[1], result.observations[1], result.truncated)
                    self.assertAlmostEqual(np.abs(opponent.q-old).sum(), data['q_delta'][i, condition, t])
                    obs = result.observations
                if condition == 0:
                    np.testing.assert_array_equal(opponent.q, original)

    def test_frozen_policy_reacts_to_state(self):
        data = sample(619, 16)
        # Q remains frozen, while the input state can select a different Q row.
        variation = np.ptp(data['opp_p'][:, 0], axis=1)
        self.assertTrue(np.any(variation > 1e-12))
        self.assertTrue(np.all(data['q_delta'][:, 0] == 0))


if __name__ == '__main__':
    unittest.main()
