"""Batch rollout equivalence with the source simulator and learners."""
import unittest

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.fast_exploration_v1.simulator import (
    EpisodeBatch, StudentBatch, opponent_distribution, opponent_update,
)
from research.structured_oracle.run import Gate
from research.structured_scale.run import FAMILIES, Model, Opponent, features, game


class BatchEquivalence(unittest.TestCase):
    def test_real_games_transitions_and_updates(self):
        rules = [game(300+i, FAMILIES[i % 4], 3, 3) for i in range(64)]
        batched = EpisodeBatch(rules, range(64), 32)
        envs = [StructuredEnv(rule, max_steps=32) for rule in rules]
        obs = [env.reset(seed=i) for i, env in enumerate(envs)]
        q = np.zeros((64, 3, 3, 3))
        opponents = [Opponent(3) for _ in rules]
        students = [Model(3, np.zeros(20)) for _ in rules]
        batched_student = StudentBatch(64, np.zeros(20))
        rng = np.random.default_rng(99)
        for t in range(32):
            x = batched.features()
            for i in range(64):
                np.testing.assert_allclose(x[i], features(obs[i][0], 3), atol=1e-12)
            a = rng.integers(3, size=64)
            u, gates = rng.random(64), rng.random(64)
            probs = opponent_distribution(q, batched.last)
            b = np.array([np.searchsorted(np.cumsum(probs[i]), u[i], side='right') for i in range(64)])
            previous = batched.last.copy()
            reward = batched.step(a, b, gates)
            opponent_update(q, previous, batched.last, b, reward[:,1], t == 31)
            batched_student.observe(x, a, reward[:,0])
            for i, env in enumerate(envs):
                np.testing.assert_allclose(probs[i], opponents[i].distribution(obs[i][1]))
                env._rng = Gate(gates[i])
                result = env.step(int(a[i]), int(b[i]))
                np.testing.assert_allclose(reward[i], result.rewards, atol=1e-12)
                opponents[i].update(obs[i][1], int(b[i]), result.rewards[1],
                                    result.observations[1], result.truncated)
                students[i].observe(x[i], int(a[i]), result.rewards[0])
                np.testing.assert_allclose(q[i], opponents[i].q, atol=1e-12)
                np.testing.assert_allclose(batched_student.mean[i], students[i].mean, atol=1e-12)
                np.testing.assert_allclose(batched_student.inverse[i], students[i].inverse, atol=1e-12)
                obs[i] = result.observations
        with self.assertRaises(RuntimeError):
            batched.step(a, b, gates)


if __name__ == '__main__':
    unittest.main()
