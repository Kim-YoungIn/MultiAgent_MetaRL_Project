"""Regression checks for observation masking and fixed-budget phase boundaries."""
import unittest

import numpy as np

from research.fast_exploration_v1 import batch
from research.fast_exploration_v1.simulator import StudentBatch
from research.fast_exploration_v2 import core


class V2Interventions(unittest.TestCase):
    def setUp(self):
        self.episodes = batch.prepare([11, 12], ['resource', 'reciprocity'], 8, 48,
                                      rule_seeds=[123, 456])
        self.prior = np.zeros(20)

    def test_mask_hides_reset_token_and_exogenous_feedback(self):
        env = self.episodes.episode(self.episodes.explore_seed, 8)
        student = StudentBatch(2, self.prior)
        before = core.inputs(env, student, env.features(), masked=True)
        np.testing.assert_array_equal(before[..., 0], 0)
        env.step([1, 1], [0, 2], [.1, .9])
        student.observe(env.features(), [1, 1], [.8, -.7])
        visible = core.inputs(env, student, env.features(), masked=False)
        hidden = core.inputs(env, student, env.features(), masked=True)
        self.assertFalse(np.array_equal(visible[0], visible[1]))
        np.testing.assert_array_equal(hidden[0], hidden[1])
        np.testing.assert_array_equal(hidden[..., 1:3], 0.)
        np.testing.assert_array_equal(hidden[..., 5:], 0.)

    def test_continuous_greedy_rollout_is_invariant_to_handoff(self):
        q = batch.warm_up(self.episodes)
        first = core.probe(self.episodes, q, self.prior, 'pooled_greedy', None, None, 0, 48)
        second = core.probe(self.episodes, q, self.prior, 'pooled_greedy', None, None, 8, 48)
        baseline = core.assess(first.env, self.episodes.assess_uniform, first.student,
                               first.q, learn=True)
        alternate = core.assess(second.env, self.episodes.assess_uniform[8:], second.student,
                                second.q, learn=True)
        for full, pre, post in ((baseline.actions, second.tape.actions, alternate.actions),
                                (baseline.rewards, second.tape.rewards, alternate.rewards)):
            np.testing.assert_array_equal(full, np.concatenate((pre, post), axis=1))
