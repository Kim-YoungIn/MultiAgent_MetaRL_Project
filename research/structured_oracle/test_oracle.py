import copy
import unittest

import numpy as np
from mamrl import structured_games as sg
from mamrl.tensor import RewardTensor
from research.structured_oracle.run import FAMILIES, RewardKernel, next_distribution, solve_pair


class TwoTurnTests(unittest.TestCase):
    def test_learning_opponent(self):
        np.testing.assert_allclose(next_distribution(3, (0,1), 1,1, 4), np.full(3,1/3))
        np.testing.assert_allclose(next_distribution(3, (1,1), 1,1, 4), [1/12, 10/12, 1/12])

    def test_reward_identification(self):
        zero = RewardTensor(3, [0.]*(2*3**4))
        worlds = [sg.StructuredGame(3, 1, zero, rewards=(sg.RewardClause(1.,
                  (sg.Predicate('own_is', action=i),)),)) for i in (0, 1)]
        _, totals, gaps = solve_pair(worlds, 123)
        np.testing.assert_allclose(totals.max(axis=1), [1., 1.5])
        np.testing.assert_allclose(gaps, [.5, 0., 0.])

    def test_investment_without_information(self):
        zero = RewardTensor(3, [0.]*(2*3**4))
        invest = (sg.Predicate('own_is', action=2),)
        harvest = (sg.Predicate('own_is', action=1), sg.Predicate('state_min', state='stock', threshold=1))
        game = sg.StructuredGame(3, 1, zero, states=(sg.StateVariable('stock', 2),),
            rewards=(sg.RewardClause(-.1, invest), sg.RewardClause(1., harvest)),
            transitions=(sg.TransitionClause('stock', 1, invest, scope='both'),))
        _, _, gaps = solve_pair([game, game], 987)
        np.testing.assert_allclose(gaps, [0., 1/3-.1, 1/3-.1])

    def test_compiled_immediate_reward_matches_extension(self):
        for i, fam in enumerate(FAMILIES):
            game = sg.generate_structured_rule(n=4, memory_depth=4, seed=711+i, family=fam, stochastic=True)
            env = sg.StructuredEnv(game, max_steps=16)
            env.reset(seed=17)
            kernel = RewardKernel(game)
            for t in range(15):
                direct = [[copy.deepcopy(env).step(a, b).rewards[0] for b in range(4)] for a in range(4)]
                np.testing.assert_allclose(kernel(env), direct, atol=1e-12)
                a = game.roles[(t//3)%3]
                env.step(a, a)


if __name__ == '__main__':
    unittest.main()
