"""Clean abstract interfaces for Opponents, Students, and Explorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mamrl.envs.base import Observation


class Opponent(ABC):
    """Counterpart agent with explicit snapshot/restore for controlled evaluation."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        ...

    @abstractmethod
    def act(self, obs: Observation) -> int:
        ...

    @abstractmethod
    def update(self, obs: Observation, action: int, reward: float, next_obs: Observation) -> None:
        ...

    @abstractmethod
    def snapshot(self) -> Any:
        ...

    @abstractmethod
    def restore(self, state: Any) -> None:
        ...


class Student(ABC):
    """Within-task adaptation learner that exploits collected exploration experience."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        ...

    @abstractmethod
    def act(self, obs: Observation) -> int:
        ...

    @abstractmethod
    def update(self, obs: Observation, action: int, reward: float, next_obs: Observation) -> None:
        ...

    @abstractmethod
    def action_values_and_uncertainty(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_estimate, uncertainty_std) per action for UCB / diagnostics."""
        ...


class Explorer(ABC):
    """Policy governing the K-turn exploration phase before handing off to Student."""

    name: str = "explorer"

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        ...

    @abstractmethod
    def act(self, obs: Observation, student: Student, step_idx: int) -> int:
        ...
