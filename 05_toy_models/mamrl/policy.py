"""Explicit policy state permits recurrent agents without exposing the game."""

from dataclasses import dataclass
import random
from typing import Generic, Protocol, TypeVar

from .common import validate_n, validate_seed
from .environment import Observation

State = TypeVar("State")


@dataclass(frozen=True, slots=True)
class PolicyOutput(Generic[State]):
    action: int
    state: State


class Policy(Protocol[State]):
    """Episode-local state contract; no assumptions about a neural architecture."""

    def initial_state(self, *, n: int, seed: int | None = None) -> State: ...

    def act(
        self, observation: Observation, state: State, *, n: int
    ) -> PolicyOutput[State]: ...


class RandomPolicy:
    """Parameter-free policy with separate, explicitly owned RNG state per agent."""

    def initial_state(self, *, n: int, seed: int | None = None) -> random.Random:
        validate_n(n)
        return random.Random(validate_seed(seed))

    def act(
        self, observation: Observation, state: random.Random, *, n: int
    ) -> PolicyOutput[random.Random]:
        n = validate_n(n)
        if not isinstance(state, random.Random):
            raise TypeError("RandomPolicy state must be a local random.Random")
        return PolicyOutput(state.randrange(n), state)
