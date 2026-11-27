"""Within-task Exploitation Students (Bayesian belief & Tabular Q)."""

from __future__ import annotations

import numpy as np

from mamrl.envs.active_probe_game import ActiveProbeSpec
from mamrl.envs.base import Observation
from .base import Student


class BayesianRewardStudent(Student):
    """Exact-family Bayesian posterior Student over theta in {0, 1}.

    Avoids tabular Q-state sparsity bottlenecks so exploration quality is measured
    cleanly. Falls back to empirical per-action Gaussian updates if no spec prior
    is supplied.
    """

    def __init__(
        self,
        n_actions: int = 4,
        spec: ActiveProbeSpec | None = None,
        obs_sigma: float = 0.08,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self._spec = spec
        self.obs_sigma = obs_sigma
        self._rng = np.random.default_rng(seed)
        self._focal_R = spec.payoff_matrices()[0] if spec is not None else None
        self.belief = np.array([0.5, 0.5], dtype=np.float64)
        self.opp_counts = np.ones(n_actions, dtype=np.float64)
        self._emp_sum = np.zeros(n_actions, dtype=np.float64)
        self._emp_cnt = np.zeros(n_actions, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.belief = np.array([0.5, 0.5], dtype=np.float64)
        self.opp_counts = np.ones(self.n_actions, dtype=np.float64)
        self._emp_sum = np.zeros(self.n_actions, dtype=np.float64)
        self._emp_cnt = np.zeros(self.n_actions, dtype=np.float64)

    def update(self, obs: Observation, action: int, reward: float, next_obs: Observation) -> None:
        opp_a = next_obs.prev_opp_action
        if 0 <= opp_a < self.n_actions:
            self.opp_counts[opp_a] += 1.0
        self._emp_sum[action] += reward
        self._emp_cnt[action] += 1.0

        if self._focal_R is not None and 0 <= opp_a < self.n_actions:
            # Exact Bayesian belief update over theta in {0, 1} from observed (a, b, r)
            pred = self._focal_R[:, action, opp_a]
            log_lik = -0.5 * ((reward - pred) / self.obs_sigma) ** 2
            log_post = np.log(np.clip(self.belief, 1e-12, 1.0)) + log_lik
            log_post -= float(np.max(log_post))
            post = np.exp(log_post)
            self.belief = post / float(np.sum(post))

    def action_values_and_uncertainty(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        if self._focal_R is not None:
            opp_dist = self.opp_counts / float(np.sum(self.opp_counts))
            # Expected payoff per (theta, a) under empirical opponent distribution
            exp_by_th = self._focal_R @ opp_dist  # (2, n_actions)
            means = self.belief @ exp_by_th       # (n_actions,)
            # Epistemic std across theta + count bonus
            var_th = self.belief @ ((exp_by_th - means[None, :]) ** 2)
            count_bonus = 1.0 / np.sqrt(self._emp_cnt + 1.0)
            unc = np.sqrt(np.maximum(var_th, 0.0)) + 0.25 * count_bonus
            return means, unc

        means = np.where(self._emp_cnt > 0, self._emp_sum / np.maximum(self._emp_cnt, 1.0), 0.0)
        unc = 1.0 / np.sqrt(self._emp_cnt + 1.0)
        return means, unc

    def act(self, obs: Observation) -> int:
        means, _ = self.action_values_and_uncertainty(obs)
        best = np.flatnonzero(np.isclose(means, np.max(means), atol=1e-9))
        return int(self._rng.choice(best))


class TabularQStudent(Student):
    """Online tabular Q-learning Student (without exact-float reward state bloat)."""

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.90,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros(n_actions, dtype=np.float64)
        self._cnt = np.zeros(n_actions, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._q = np.zeros(self.n_actions, dtype=np.float64)
        self._cnt = np.zeros(self.n_actions, dtype=np.float64)

    def update(self, obs: Observation, action: int, reward: float, next_obs: Observation) -> None:
        self._cnt[action] += 1.0
        target = reward + self.gamma * float(np.max(self._q))
        self._q[action] += self.alpha * (target - self._q[action])

    def action_values_and_uncertainty(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        unc = 1.0 / np.sqrt(self._cnt + 1.0)
        return self._q.copy(), unc

    def act(self, obs: Observation) -> int:
        best = np.flatnonzero(np.isclose(self._q, np.max(self._q), atol=1e-9))
        return int(self._rng.choice(best))


def build_student(
    kind: str,
    n_actions: int = 4,
    spec: ActiveProbeSpec | None = None,
    seed: int = 0,
) -> Student:
    if kind == "bayesian":
        return BayesianRewardStudent(n_actions=n_actions, spec=spec, seed=seed)
    if kind == "tabular_q":
        return TabularQStudent(n_actions=n_actions, seed=seed)
    raise ValueError(f"Unknown student kind: {kind}")
