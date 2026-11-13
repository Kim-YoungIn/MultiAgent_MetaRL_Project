"""Fixed memory-one opponent policies, with no access to current learner actions."""

import random

from .common import validate_action, validate_n, validate_seed
from .environment import Observation
from .policy import PolicyOutput, RandomPolicy


class CopyPreviousPolicy(RandomPolicy):
    def act(self, observation: Observation, state: random.Random, *, n: int):
        validate_n(n)
        action = validate_action(observation.previous_opponent_action, n)
        return PolicyOutput(action, state)


class AvoidPreviousPolicy(RandomPolicy):
    def act(self, observation: Observation, state: random.Random, *, n: int):
        n = validate_n(n)
        excluded = validate_action(observation.previous_opponent_action, n)
        if not isinstance(state, random.Random):
            raise TypeError("opponent state must be a local random.Random")
        action = state.randrange(n-1)
        return PolicyOutput(action + (action >= excluded), state)


def make_opponent(kind: str):
    if kind == 'random':
        return RandomPolicy()
    if kind == 'copy_previous':
        return CopyPreviousPolicy()
    if kind == 'avoid_previous':
        return AvoidPreviousPolicy()
    raise ValueError(f'unknown opponent kind: {kind!r}')
