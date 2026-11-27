from collections import defaultdict
import itertools
import unittest

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.dynamics_transfer.exposure import trajectories
from research.dynamics_transfer.rules import investment_clause, variant
from research.structured_oracle.run import Gate
from research.structured_scale.run import FAMILIES, game


def state_law(rule, history, level, joint):
    """Integrate the two outcomes of the scalar environment's sole random gate."""
    clause = next(t for t in rule.transitions if t.delta > 0)
    probability = clause.probability
    outcomes = defaultdict(float)
    for uniform, mass in ((0., probability), (1., 1. - probability)):
        env = StructuredEnv(rule, max_steps=1)
        env.reset(seed=11)
        env._history = list(history)
        env._state[clause.state] = level
        env._rng = Gate(uniform)
        env.step(*joint)
        outcomes[tuple(env._state.values())] += mass
    return outcomes


class DynamicsTests(unittest.TestCase):
    def test_scalar_transition_distance_and_ineligible_contexts(self):
        for family in FAMILIES:
            base = game(17, family, 3, 3)
            invest, harvest, _ = base.roles
            self.assertIs(variant(base, 0., 0), base)
            for lag in (0, 1):
                native = variant(base, 0., lag)
                changed = variant(base, .173, lag)
                for level, prev1, prev2, opponent in itertools.product((0, 2), (invest, harvest),
                                                                      (invest, harvest), (invest, harvest)):
                    history = [(prev1, prev1), (prev2, prev2), (harvest, harvest)]
                    left, right = [state_law(r, history, level, (invest, opponent)) for r in (native, changed)]
                    tv = .5 * sum(abs(left[k] - right[k]) for k in set(left) | set(right))
                    observable = (level < 2 and opponent == invest and (not lag or prev1 == invest)
                                  and (family != 'sequence' or prev2 == invest))
                    self.assertAlmostEqual(tv, .173 if observable else 0.)

    def test_role_scripts_against_original_scalar_games(self):
        families = np.array(FAMILIES)
        base = [game(94 + i, f, 3, 3) for i, f in enumerate(FAMILIES)]
        resets = np.arange(103, 107)
        gates = np.random.default_rng(71).random((4, 96))
        roles = np.array([r.roles for r in base])
        choices = roles[:, (np.arange(96) % 6 >= 4).astype(int)]
        actions = np.repeat(choices[:, :, None], 2, axis=2)
        for lag in (0, 1):
            rules = [variant(r, .3, lag) for r in base]
            states, opportunity, rewards = trajectories(rules, resets, families, actions, gates, lag)
            for i, rule in enumerate(rules):
                env = StructuredEnv(rule, max_steps=96)
                env.reset(seed=int(resets[i]))
                name = investment_clause(base[i])[1].state
                invest = (rule.roles[0],) * 2
                for t in range(96):
                    joint = tuple(map(int, actions[i, t]))
                    eligible = (joint == invest and env._state[name] < 2 and
                                (lag == 0 or env._history[0] == invest) and
                                (rule.family != 'sequence' or env._history[1] == invest))
                    self.assertEqual(opportunity[i, t], eligible)
                    env._rng = Gate(gates[i, t])
                    step = env.step(*joint)
                    self.assertEqual(states[i, t], env._state[name])
                    np.testing.assert_allclose(rewards[i, t], step.rewards, rtol=0, atol=1e-12)
