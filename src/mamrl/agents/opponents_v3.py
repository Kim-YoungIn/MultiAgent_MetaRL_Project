"""Vectorized Online Learning Opponents for SPEC v3.0 Exp 2A (R3 Non-Stationarity).

Implements two distinct tabular online learners operating on the public state q in {0..7}
and opponent symbol space y in {0..3} using exact opponent utility r_O(q, a, y; theta):
1. VectorizedQLearningOpponent: off-policy max-backup Q-learning with epsilon-greedy exploration.
2. VectorizedExpectedSARSAOpponent: on-policy expected-backup SARSA with Boltzmann softmax
   exploration (strictly unseen opponent algorithm and exploration distribution at test time).

Supports:
- Finite pre-episode warmup (so opponents start partially experienced and continue learning online).
- Exact snapshot() / restore(mask=...) of Q-tables (and optional RNG state) for controlled
  Restored vs. Continuous intervention audits.
- Exact diagnostics: q_l1_distance_from_init(), greedy_honest_rate().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from mamrl.envs.delayed_response import (
    N_ACTIONS,
    N_OPP_SYMBOLS,
    build_opp_reward_table,
    build_response_table,
)


@dataclass(frozen=True)
class OnlineOpponentConfig:
    """Declarative specification for an online learning opponent."""

    kind: str = "q_learning"  # "q_learning" or "expected_sarsa"
    regime: str = "R3"
    alpha: float = 0.35
    gamma: float = 0.50
    epsilon: float = 0.12        # used by q_learning
    temperature: float = 0.45    # used by expected_sarsa (Boltzmann softmax)
    warmup_steps: int = 48
    warmup_epsilon: float = 0.50  # exploration during warmup (for q_learning)
    warmup_temperature: float = 0.80  # exploration during warmup (for expected_sarsa)


class VectorizedOnlineOpponent:
    """Base vectorized tabular learning opponent for B parallel DelayedResponseEnv games."""

    def __init__(self, cfg: OnlineOpponentConfig, batch_size: int, seed: int = 0) -> None:
        if cfg.kind not in ("q_learning", "expected_sarsa"):
            raise ValueError(f"Unsupported opponent kind: {cfg.kind}")
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(seed)
        self._f_table = build_response_table(cfg.regime)
        self._r_opp = build_opp_reward_table(cfg.regime)

        self.theta = np.zeros(self.batch_size, dtype=np.int64)
        self.q_table = np.zeros((self.batch_size, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
        self.q_init = np.zeros_like(self.q_table)

    def _softmax_probs(self, q_vals: np.ndarray, temperature: float) -> np.ndarray:
        scaled = q_vals / max(1e-6, float(temperature))
        scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
        ex = np.exp(scaled)
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def _select_action_from_q(
        self,
        q_state_vals: np.ndarray,
        *,
        epsilon: float,
        temperature: float,
    ) -> np.ndarray:
        b = q_state_vals.shape[0]
        if self.cfg.kind == "q_learning":
            # Random tie-breaking among equal Q-values
            noise = self._rng.uniform(0.0, 1e-9, size=q_state_vals.shape)
            greedy = np.argmax(q_state_vals + noise, axis=-1).astype(np.int64)
            rand_y = self._rng.integers(0, N_OPP_SYMBOLS, size=b, dtype=np.int64)
            return np.where(self._rng.random(b) < epsilon, rand_y, greedy).astype(np.int64)

        # Boltzmann softmax policy for Expected SARSA
        probs = self._softmax_probs(q_state_vals, temperature=temperature)
        cdf = np.cumsum(probs, axis=-1)
        u = self._rng.random((b, 1))
        return np.minimum(N_OPP_SYMBOLS - 1, (u >= cdf).sum(axis=-1)).astype(np.int64)

    def _td_update(
        self,
        q_prev: np.ndarray,
        y_act: np.ndarray,
        r_opp: np.ndarray,
        q_next: np.ndarray,
        *,
        temperature: float,
    ) -> None:
        idx = np.arange(self.batch_size)
        next_q_all = self.q_table[idx, q_next, :]  # (B, Y)
        if self.cfg.kind == "q_learning":
            bootstrap = np.max(next_q_all, axis=-1)
        else:
            probs = self._softmax_probs(next_q_all, temperature=temperature)
            bootstrap = np.sum(probs * next_q_all, axis=-1)

        cur_q = self.q_table[idx, q_prev, y_act]
        td_target = r_opp + self.cfg.gamma * bootstrap
        self.q_table[idx, q_prev, y_act] = cur_q + self.cfg.alpha * (td_target - cur_q)

    def reset(self, theta: np.ndarray, seed: int | None = None) -> None:
        """Initialize opponent Q-tables for new episodes and run finite pre-episode warmup."""
        arr = np.asarray(theta, dtype=np.int64)
        if arr.shape != (self.batch_size,):
            raise ValueError(f"Expected theta shape ({self.batch_size},), got {arr.shape}")
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.theta = arr.copy()
        self.q_table.fill(0.0)

        # Pre-episode warmup against random focal play using the opponent's own learning rule
        q_cur = np.zeros(self.batch_size, dtype=np.int64)  # starts at S = 0
        idx = np.arange(self.batch_size)
        for _ in range(self.cfg.warmup_steps):
            q_vals = self.q_table[idx, q_cur, :]
            y = self._select_action_from_q(
                q_vals,
                epsilon=self.cfg.warmup_epsilon,
                temperature=self.cfg.warmup_temperature,
            )
            a_rand = self._rng.integers(0, N_ACTIONS, size=self.batch_size, dtype=np.int64)
            ro = self._r_opp[self.theta, q_cur, a_rand, y]
            self._td_update(
                q_prev=q_cur,
                y_act=y,
                r_opp=ro,
                q_next=a_rand,
                temperature=self.cfg.warmup_temperature,
            )
            q_cur = a_rand

        self.q_init = self.q_table.copy()

    def act(self, q_cur: np.ndarray) -> np.ndarray:
        """Select opponent symbols y_t in {0..3} simultaneously given public state q_{t-1}."""
        idx = np.arange(self.batch_size)
        q_vals = self.q_table[idx, q_cur, :]
        return self._select_action_from_q(
            q_vals,
            epsilon=self.cfg.epsilon,
            temperature=self.cfg.temperature,
        )

    def update(self, q_prev: np.ndarray, y_act: np.ndarray, r_opp: np.ndarray, q_next: np.ndarray) -> None:
        """Perform one online tabular TD update using realized (q_{t-1}, y_t, r_O, q_t)."""
        self._td_update(
            q_prev=q_prev,
            y_act=y_act,
            r_opp=r_opp,
            q_next=q_next,
            temperature=self.cfg.temperature,
        )

    def snapshot(self) -> dict[str, Any]:
        """Capture exact Q-table and RNG bit-generator state for Restored interventions."""
        return {
            "q_table": self.q_table.copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def restore(self, snap: dict[str, Any] | None = None, mask: np.ndarray | None = None) -> None:
        """Restore opponent Q-table (from snap or episode-start q_init) for all or masked episodes."""
        src_q = self.q_init if snap is None else snap["q_table"]
        if mask is None:
            self.q_table = src_q.copy()
            if snap is not None and "rng_state" in snap:
                self._rng.bit_generator.state = snap["rng_state"]
        else:
            m = np.asarray(mask, dtype=np.bool_)
            self.q_table[m] = src_q[m]

    def q_l1_distance_from_init(self) -> np.ndarray:
        """Return per-episode mean L1 change in opponent Q-table since turn 0: shape (B,)."""
        return np.mean(np.abs(self.q_table - self.q_init), axis=(1, 2))

    def greedy_honest_rate(self) -> np.ndarray:
        """Fraction of the 8 states q where opponent's greedy argmax matches honest f_theta(q)."""
        honest = self._f_table[self.theta]  # (B, 8)
        # Check strict argmax (ties count as non-converged unless unique or matching max)
        max_q = np.max(self.q_table, axis=-1, keepdims=True)
        is_max = np.abs(self.q_table - max_q) <= 1e-12
        idx_b = np.arange(self.batch_size)[:, None]
        idx_q = np.arange(N_ACTIONS)[None, :]
        honest_is_max = is_max[idx_b, idx_q, honest]
        n_ties = np.sum(is_max, axis=-1)
        # Expected probability of picking honest action under uniform tie-break
        p_honest_greedy = np.where(honest_is_max, 1.0 / np.maximum(n_ties, 1), 0.0)
        return np.mean(p_honest_greedy, axis=-1)


def make_online_opponent(cfg: OnlineOpponentConfig, batch_size: int, seed: int = 0) -> VectorizedOnlineOpponent:
    """Factory helper creating a configured vectorized online learning opponent."""
    return VectorizedOnlineOpponent(cfg=cfg, batch_size=batch_size, seed=seed)
