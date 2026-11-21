"""Contracts for reproducible paired worlds and real opponent updates."""
import copy
import unittest

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.active_probe.data import generate
from research.structured_oracle.run import Gate
from research.structured_scale.run import FAMILIES, Opponent, game, step


class DatasetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate(889193, 12, 2)

    def test_reproducibility_and_balancing(self):
        second = generate(889193, 12, 2)
        self.assertEqual(second.keys(), self.data.keys())
        for key in second:
            np.testing.assert_array_equal(self.data[key], second[key], err_msg=key)
        self.assertEqual(np.bincount(self.data['motif']).tolist(), [6, 6, 6, 6])
        self.assertEqual(len(set(self.data['game_seed'][::2])), 12)
        self.assertEqual(len(set(self.data['game_fingerprint'][::2])), 12)
        np.testing.assert_array_equal(self.data['game_fingerprint'][::2], self.data['game_fingerprint'][1::2])
        np.testing.assert_array_equal(self.data['game_seed'][::2], self.data['game_seed'][1::2])

    def test_rejects_invalid_catalogs(self):
        for seed, games, repeats in [(0, 3, 1), (-1, 4, 1), (0, 4, 0)]:
            with self.assertRaises(ValueError):
                generate(seed, games, repeats)

    def test_alternative_prefix_lengths(self):
        for length in (0, 8):
            data = generate(29, 4, 1, prefix=length)
            self.assertEqual(data['prefix_a'].shape, (4, length))
            self.assertEqual(data['prefix_public'].shape, (4, length + 1, 2))
            self.assertTrue(np.isfinite(data['next_value']).all())
        with self.assertRaises(ValueError):
            generate(1, 4, 1, prefix=-1)

    def test_replay_and_exact_values_against_real_simulator(self):
        data = self.data
        # Includes all motifs and independent game/reset samples.
        for row in range(0, 8):
            rule = game(int(data['game_seed'][row]), FAMILIES[data['motif'][row]], 3, 3)
            self.assertEqual(data['game_fingerprint'][row], rule.fingerprint())
            env = StructuredEnv(rule, max_steps=6)
            obs = env.reset(seed=int(data['reset_seed'][row]))
            np.testing.assert_equal(data['init_hist'][row], obs[0].action_history)
            opponent = Opponent(3)
            for t in range(4):
                outcome, b = step(env, opponent, obs, int(data['prefix_a'][row, t]),
                                  float(data['prefix_u'][row, t]))
                self.assertEqual(b, data['prefix_b'][row, t])
                self.assertAlmostEqual(outcome.rewards[0], data['prefix_reward'][row, t])
                self.assertAlmostEqual(outcome.rewards[1], data['prefix_opp_reward'][row, t])
                obs = outcome.observations
                public = list(obs[0].public_state[:2])
                public += [0] * (2-len(public))
                np.testing.assert_array_equal(data['prefix_public'][row, t+1], public)
            probs = opponent.distribution(obs[1])
            np.testing.assert_allclose(data['opp_p'][row], probs)
            expected = np.zeros(3)
            for a in range(3):
                for b in range(3):
                    clone = copy.deepcopy(env)
                    expected[a] += probs[b] * clone.step(a, b).rewards[0]
            np.testing.assert_allclose(data['first_value'][row], expected, atol=1e-14)
            np.testing.assert_allclose(data['next_value'][row, :, 1], np.tile(expected, (3, 1)))
            for a in range(3):
                branch = copy.deepcopy(env)
                branch._rng = Gate(float(data['gate_u'][row]))
                opp = copy.deepcopy(opponent)
                outcome, b = step(branch, opp, obs, a, float(data['first_u'][row]))
                self.assertEqual(b, data['first_b'][row])
                self.assertAlmostEqual(outcome.rewards[0], data['first_reward'][row, a])
                self.assertAlmostEqual(outcome.rewards[1], data['first_opp_reward'][row, a])
                post = opp.distribution(outcome.observations[1])
                np.testing.assert_allclose(data['post_opp_p'][row, a], post)
                self.assertAlmostEqual(data['opponent_q_update'][row, a], opp.change-opponent.change)
                frozen = opponent.distribution(outcome.observations[1])
                no_update = np.zeros(3)
                for c in range(3):
                    for d in range(3):
                        future = copy.deepcopy(branch)
                        no_update[c] += frozen[d] * future.step(c, d).rewards[0]
                np.testing.assert_allclose(data['next_value_no_update'][row, a], no_update, atol=1e-14)
                terminal = np.zeros(3)
                for c in range(3):
                    for d in range(3):
                        future = copy.deepcopy(branch)
                        terminal[c] += post[d] * future.step(c, d).rewards[0]
                np.testing.assert_allclose(data['next_value'][row, a, 0], terminal, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
