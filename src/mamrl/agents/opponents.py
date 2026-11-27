"""Multi-agent opponents: fixed reactive strategy and online tabular Q-learner."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np

from mamrl.envs.base import Observation
from .base import Opponent


class ReactiveOpponent(Opponent):
    """Fixed strategic opponent that responds to focal's previous action."""

    def __init__(self, n_actions: int = 4, copy_prob: float = 0.6, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.copy_prob = copy_prob
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> int:
        if obs.is_start or obs.prev_opp_action < 0:
            return int(self._rng.integers(0, self.n_actions))
        if float(self._rng.random()) < self.copy_prob:
            return int(obs.prev_opp_action)
        return int(self._rng.integers(0, self.n_actions))

    def update(self, obs: Observation, action: int, reward: float, next_obs: Observation) -> None:
        return

    def snapshot(self) -> Any:
        return deepcopy(self._rng.bit_generator.state)

    def restore(self, state: Any) -> None:
        self._rng.bit_generator.state = deepcopy(state)


class QLearningOpponent(Opponent):
    """Online tabular Q-learning counterpart conditioned on last joint action."""

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.35,
        gamma: float = 0.90,
        epsilon: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._q: dict[tuple[int, int], np.ndarray] = {}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._q = {}

    def _key(self, obs: Observation) -> tuple[int, int]:
        if obs.is_start:
            return (-1, -1)
        return (obs.prev_self_action, obs.prev_opp_action)

    def _row(self, key: tuple[int, int]) -> np.ndarray:
        row = self._q.get(key)
        if row is None:
            row = np.zeros(self.n_actions, dtype=np.float64)
            self._q[key] = row
        return row

    def act(self, obs: Observation) -> int:
        if float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        vals = self._row(self._key(obs))
        best = np.flatnonzero(vals == np.max(vals))
        return int(self._rng.choice(best))

    def update(self, obs: Observation, action: int, reward: float, next_obs: Observation) -> None:
        s = self._key(obs)
        ns = self._key(next_obs)
        q_s = self._row(s)
        target = reward + self.gamma * float(np.max(self._row(ns)))
        q_s[action] += self.alpha * (target - q_s[action])

    def snapshot(self) -> Any:
        return (
            {k: v.copy() for k, v in self._q.items()},
            deepcopy(self._rng.bit_generator.state),
        )

    def restore(self, state: Any) -> None:
        q_copy, rng_state = state
        self._q = {k: v.copy() for k, v in q_copy.items()}
        self._rng.bit_generator.state = deepcopy(rng_state)


def build_opponent(kind: str, n_actions: int = 4, seed: int = 0) -> Opponent:
    if kind == "reactive":
        return ReactiveOpponent(n_actions=n_actions, seed=seed)
    if kind == "q_learning":
        return QLearningOpponent(n_actions=n_actions, seed=seed)
    raise ValueError(f"Unknown opponent kind: {kind}")
