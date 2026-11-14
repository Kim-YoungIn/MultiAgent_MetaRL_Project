"""Small online tabular learners for the inner multi-agent trial.

The policy sees only its own four-field observation. Its table is local to one
trial and carries across the episode reset within that trial.
"""

from __future__ import annotations

import hashlib
import math
import random

from .common import finite_reward, validate_action, validate_n, validate_seed
from .environment import Observation


class OnlineQ:
    """Epsilon-greedy online Q learner with local RNG and inspectable audits."""

    algorithm = 'q_learning'

    def __init__(self, n, seed, alpha=.5, gamma=.95, epsilon=.2):
        self.n = validate_n(n)
        if validate_seed(seed) is None:
            raise TypeError('online learner requires an explicit seed')
        self.alpha = self._unit_interval(alpha, 'alpha')
        self.gamma = self._unit_interval(gamma, 'gamma')
        self.epsilon = self._unit_interval(epsilon, 'epsilon')
        self._rng = random.Random(seed)
        self._q = {}
        self.updates = 0
        self.parameter_changes = 0
        self.absolute_q_change = 0.0
        self.policy_changes = 0
        self.policy_l1_change = 0.0

    @staticmethod
    def _unit_interval(value, name):
        result = finite_reward(value)
        if not 0.0 <= result <= 1.0:
            raise ValueError(f'{name} must be between zero and one')
        return result

    def _key(self, observation):
        if not isinstance(observation, Observation):
            raise TypeError('observation must be an Observation')
        own = validate_action(observation.previous_own_action, self.n)
        other = validate_action(observation.previous_opponent_action, self.n)
        reward = finite_reward(observation.previous_own_reward)
        if type(observation.episode_start_flag) is not bool:
            raise TypeError('episode_start_flag must be bool')
        return (own, other, reward, observation.episode_start_flag)

    def values(self, observation):
        """Return a copy; reading an unseen state does not alter the table."""
        return tuple(self._q.get(self._key(observation), (0.0,) * self.n))

    def distribution(self, observation, epsilon=None):
        exploration = self.epsilon if epsilon is None else self._unit_interval(epsilon, 'epsilon')
        values = self.values(observation)
        best = max(values)
        ties = sum(value == best for value in values)
        return tuple(exploration / self.n + ((1.0 - exploration) / ties if value == best else 0.0)
                     for value in values)

    def act(self, observation, epsilon=None):
        probabilities = self.distribution(observation, epsilon)
        draw = self._rng.random()
        cumulative = 0.0
        for action, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative or action == self.n - 1:
                return action
        raise RuntimeError('invalid action probabilities')

    def update(self, observation, action, reward, next_obs, terminal=False):
        if type(terminal) is not bool:
            raise TypeError('terminal must be bool')
        key = self._key(observation)
        self._key(next_obs)
        action = validate_action(action, self.n)
        reward = finite_reward(reward)
        old_values = self.values(observation)
        before_policy = self.distribution(observation)
        bootstrap = 0.0 if terminal else self._bootstrap(next_obs)
        new_value = finite_reward(old_values[action] + self.alpha *
                                  (reward + self.gamma * bootstrap - old_values[action]))
        delta = new_value - old_values[action]
        if not math.isfinite(delta):
            raise ValueError('Q change must be finite')
        values = list(old_values)
        values[action] = new_value
        self._q[key] = values
        self.updates += 1
        if delta != 0.0:
            self.parameter_changes += 1
        self.absolute_q_change = finite_reward(self.absolute_q_change + abs(delta))
        after_policy = self.distribution(observation)
        policy_delta = sum(abs(a-b) for a,b in zip(before_policy, after_policy))
        if policy_delta > 1e-12:
            self.policy_changes += 1
        self.policy_l1_change = finite_reward(self.policy_l1_change + policy_delta)
        return new_value

    def _bootstrap(self, observation):
        return max(self.values(observation))

    def erase_values(self):
        """Erase Q knowledge while preserving RNG, hyperparameters and audit counters."""
        self._q.clear()

    def clone(self):
        """Copy all agent state, including independent RNG and the complete Q table."""
        result = type(self)(self.n, 0, self.alpha, self.gamma, self.epsilon)
        result._rng.setstate(self._rng.getstate())
        result._q = {key: list(values) for key,values in self._q.items()}
        result.updates = self.updates
        result.parameter_changes = self.parameter_changes
        result.absolute_q_change = self.absolute_q_change
        result.policy_changes = self.policy_changes
        result.policy_l1_change = self.policy_l1_change
        return result

    def snapshot(self):
        """Return detached JSON-safe state/audits; RNG is a reproducible digest."""
        entries = [{'observation': {'previous_own_action': key[0],
                                     'previous_opponent_action': key[1],
                                     'previous_own_reward': key[2],
                                     'episode_start_flag': key[3]},
                    'values': list(self._q[key])} for key in sorted(self._q)]
        digest = hashlib.sha256(repr(self._rng.getstate()).encode('ascii')).hexdigest()
        return {'n':self.n, 'algorithm':self.algorithm, 'alpha':self.alpha, 'gamma':self.gamma, 'epsilon':self.epsilon,
                'updates':self.updates, 'parameter_changes':self.parameter_changes,
                'absolute_q_change':self.absolute_q_change, 'policy_changes':self.policy_changes,
                'policy_l1_change':self.policy_l1_change, 'entries':entries, 'rng_digest':digest}


class ExpectedSARSA(OnlineQ):
    """Same local observations, but on-policy expected TD instead of max-Q TD."""

    algorithm = 'expected_sarsa'

    def _bootstrap(self, observation):
        return sum(prob*value for prob,value in zip(self.distribution(observation),self.values(observation)))
