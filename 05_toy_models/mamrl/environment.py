"""Tensor-driven simultaneous game with an unobserved time limit."""

from dataclasses import dataclass
import random

from .common import positive_integer, validate_action, validate_seed
from .tensor import RewardTensor


@dataclass(frozen=True, slots=True)
class Observation:
    previous_own_action: int
    previous_opponent_action: int
    previous_own_reward: float
    episode_start_flag: bool


@dataclass(frozen=True, slots=True)
class StepResult:
    observations: tuple[Observation, Observation]
    rewards: tuple[float, float]
    terminated: bool
    truncated: bool


class MemoryOneEnv:
    """One fixed reward tensor per episode; reset may replace it with the same n.

    The policy interface does not receive this object or its private counter.
    This is an information boundary, not protection against process introspection.
    """

    def __init__(self, rule: RewardTensor, *, max_steps: int):
        if not isinstance(rule, RewardTensor):
            raise TypeError("rule must be a RewardTensor")
        self._max_steps = positive_integer(max_steps, "max_steps")
        self._rule = rule
        self._rng = random.Random()
        self._previous: tuple[int, int] | None = None
        self._elapsed = 0
        self._active = False

    @property
    def n(self) -> int:
        return self._rule.n

    @staticmethod
    def _observations(
        previous: tuple[int, int], rewards: tuple[float, float], start: bool
    ) -> tuple[Observation, Observation]:
        a0, a1 = previous
        return (
            Observation(a0, a1, rewards[0], start),
            Observation(a1, a0, rewards[1], start),
        )

    def reset(
        self, *, seed: int | None = None, rule: RewardTensor | None = None
    ) -> tuple[Observation, Observation]:
        seed = validate_seed(seed)
        if rule is not None:
            if not isinstance(rule, RewardTensor):
                raise TypeError("rule must be a RewardTensor")
            if rule.n != self.n:
                raise ValueError("reset rule must have the same action count")
        if seed is not None:
            self._rng.seed(seed)
        if rule is not None:
            self._rule = rule
        self._previous = (self._rng.randrange(self.n), self._rng.randrange(self.n))
        self._elapsed = 0
        self._active = True
        return self._observations(self._previous, (0.0, 0.0), True)

    def step(self, action_0: int, action_1: int) -> StepResult:
        if not self._active or self._previous is None:
            raise RuntimeError("reset is required before stepping this episode")
        # Validate both actions before changing any state or counter.
        action_0 = validate_action(action_0, self.n)
        action_1 = validate_action(action_1, self.n)
        p0, p1 = self._previous
        rewards = (
            self._rule[0, p0, p1, action_0, action_1],
            self._rule[1, p0, p1, action_0, action_1],
        )
        self._previous = (action_0, action_1)
        self._elapsed += 1
        truncated = self._elapsed == self._max_steps
        self._active = not truncated
        return StepResult(
            self._observations(self._previous, rewards, False),
            rewards,
            terminated=False,
            truncated=truncated,
        )
