"""Simultaneous Delayed-Response Multi-Agent Repeated Game (Matched R0 / R1 / R3).

Shared across R0, R1, R3:
- 8 focal actions: S(0), P(1), P0(2), P1(3), X00(4), X01(5), X10(6), X11(7)
- 4-symbol opponent action/observation alphabet: y in {0, 1, 2, 3}
- 4 equiprobable hidden opponent types: theta = (b, c) in {0,1}^2 -> id = 2*b + c in {0,1,2,3}
- Public state q_t = previous focal action (q_0 = S = 0); simultaneous move; opponent observes q_{t-1}
- Focal reward r_F(q, a, y): S -> +0.2; P/P0/P1 -> -0.1; Xij -> +1.0 if (q == Xij and y == 1) else -2.0
- Opponent reward r_O(q, a, y; theta): 2.0 * 1[y == f_theta(q)] + 0.5 * 1[a == q in X and y == 1]

Only the opponent's response function f_theta(q) varies across regimes:
- R0 (passive):  f_theta(S) = theta; probes -> 0; Xij -> 1[theta == (i,j)]
- R1 (1-step):   f_theta(S) = 0; f_theta(P) = theta; P0/P1 -> 0; Xij -> 1[theta == (i,j)]
- R3 (2-step):   f_theta(S) = 0; f_theta(P) = b; f_theta(P0) = 1[b=0,c=1]; f_theta(P1) = 1[b=1,c=1]; Xij -> 1[theta == (i,j)]
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import numpy as np

ACT_S = 0
ACT_P = 1
ACT_P0 = 2
ACT_P1 = 3
ACT_X00 = 4
ACT_X01 = 5
ACT_X10 = 6
ACT_X11 = 7

ACTION_NAMES = ("S", "P", "P0", "P1", "X00", "X01", "X10", "X11")
N_ACTIONS = 8
N_OPP_SYMBOLS = 4
N_TYPES = 4
EXPLOIT_ACTIONS = (ACT_X00, ACT_X01, ACT_X10, ACT_X11)
PROBE_ACTIONS = (ACT_P, ACT_P0, ACT_P1)
VALID_REGIMES = ("R0", "R1", "R3")


def build_response_table(regime: str) -> np.ndarray:
    """Return exact integer table f_table[theta, q] -> y in {0, 1, 2, 3}."""
    if regime not in VALID_REGIMES:
        raise ValueError(f"Unsupported regime '{regime}'. Expected one of {VALID_REGIMES}.")

    table = np.zeros((N_TYPES, N_ACTIONS), dtype=np.int64)
    for theta in range(N_TYPES):
        b = theta // 2
        c = theta % 2
        if regime == "R0":
            table[theta, ACT_S] = theta
        elif regime == "R1":
            table[theta, ACT_P] = theta
        elif regime == "R3":
            table[theta, ACT_P] = b
            table[theta, ACT_P0] = 1 if (b == 0 and c == 1) else 0
            table[theta, ACT_P1] = 1 if (b == 1 and c == 1) else 0

        # Exploitation confirmation responses are identical across all regimes
        for idx, act_x in enumerate(EXPLOIT_ACTIONS):
            table[theta, act_x] = 1 if theta == idx else 0

    return table


def focal_reward_rational(q: int, a: int, y: int) -> Fraction:
    """Exact rational focal reward r_F(q, a, y)."""
    if a == ACT_S:
        return Fraction(1, 5)
    if a in PROBE_ACTIONS:
        return Fraction(-1, 10)
    if a in EXPLOIT_ACTIONS:
        return Fraction(1, 1) if (q == a and y == 1) else Fraction(-2, 1)
    raise ValueError(f"Invalid focal action: {a}")


def build_focal_reward_table() -> np.ndarray:
    """Return float64 table R_focal[q, a, y] of shape (8, 8, 4)."""
    r = np.zeros((N_ACTIONS, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
    for q in range(N_ACTIONS):
        for a in range(N_ACTIONS):
            for y in range(N_OPP_SYMBOLS):
                r[q, a, y] = float(focal_reward_rational(q, a, y))
    return r


def build_opp_reward_table(regime: str) -> np.ndarray:
    """Return float64 table R_opp[theta, q, a, y] of shape (4, 8, 8, 4)."""
    f_table = build_response_table(regime)
    r = np.zeros((N_TYPES, N_ACTIONS, N_ACTIONS, N_OPP_SYMBOLS), dtype=np.float64)
    for th in range(N_TYPES):
        for q in range(N_ACTIONS):
            honest_y = int(f_table[th, q])
            for a in range(N_ACTIONS):
                for y in range(N_OPP_SYMBOLS):
                    val = 2.0 if y == honest_y else 0.0
                    if a == q and q >= ACT_X00 and y == 1:
                        val += 0.5
                    r[th, q, a, y] = val
    return r


@dataclass(frozen=True)
class DelayedObs:
    """Public single-step observation for the focal player (zero hidden-type leakage)."""

    prev_self_action: int  # equals current public state q (0=S at turn 0)
    prev_opp_action: int   # -1 at start / right after reset
    prev_self_reward: float
    is_start: bool
    reset_flag: bool
    turn: int


class DelayedResponseEnv:
    """Exact scalar reference environment for the Simultaneous Delayed-Response Game."""

    def __init__(self, regime: str, theta: int, horizon: int = 24) -> None:
        if regime not in VALID_REGIMES:
            raise ValueError(f"Unsupported regime: {regime}")
        if not (0 <= theta < N_TYPES):
            raise ValueError(f"Invalid theta: {theta}")
        self.regime = regime
        self.theta = int(theta)
        self.horizon = int(horizon)
        self._f_table = build_response_table(regime)
        self._r_focal = build_focal_reward_table()
        self._r_opp = build_opp_reward_table(regime)
        self.q = ACT_S
        self.turn = 0

    def honest_opp_action(self) -> int:
        return int(self._f_table[self.theta, self.q])

    def reset(self, reset_flag: bool = False) -> DelayedObs:
        self.q = ACT_S
        self.turn = 0
        return DelayedObs(
            prev_self_action=ACT_S,
            prev_opp_action=-1,
            prev_self_reward=0.0,
            is_start=True,
            reset_flag=reset_flag,
            turn=0,
        )

    def restore_public_state(self) -> DelayedObs:
        """Restore public state q -> S while keeping hidden theta and incrementing nothing."""
        self.q = ACT_S
        return DelayedObs(
            prev_self_action=ACT_S,
            prev_opp_action=-1,
            prev_self_reward=0.0,
            is_start=False,
            reset_flag=True,
            turn=self.turn,
        )

    def step(self, focal_action: int, opp_action: int | None = None) -> tuple[DelayedObs, float, float, int]:
        if not (0 <= focal_action < N_ACTIONS):
            raise ValueError(f"Focal action out of range: {focal_action}")
        y = self.honest_opp_action() if opp_action is None else int(opp_action)
        if not (0 <= y < N_OPP_SYMBOLS):
            raise ValueError(f"Opponent action out of range: {y}")

        q_prev = self.q
        rf = float(self._r_focal[q_prev, focal_action, y])
        ro = float(self._r_opp[self.theta, q_prev, focal_action, y])
        self.q = int(focal_action)
        self.turn += 1
        obs = DelayedObs(
            prev_self_action=self.q,
            prev_opp_action=y,
            prev_self_reward=rf,
            is_start=False,
            reset_flag=False,
            turn=self.turn,
        )
        return obs, rf, ro, y


class VectorizedDelayedResponseEnv:
    """Fast vectorized NumPy simulator for B parallel games (exact match to DelayedResponseEnv)."""

    def __init__(self, regime: str, batch_size: int, horizon: int = 24, seed: int = 0) -> None:
        if regime not in VALID_REGIMES:
            raise ValueError(f"Unsupported regime: {regime}")
        self.regime = regime
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self._rng = np.random.default_rng(seed)
        self._f_table = build_response_table(regime)
        self._r_focal = build_focal_reward_table()
        self._r_opp = build_opp_reward_table(regime)
        self.theta = np.zeros(self.batch_size, dtype=np.int64)
        self.q = np.full(self.batch_size, ACT_S, dtype=np.int64)
        self.turn = 0

    def reset(self, theta: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if theta is None:
            self.theta = self._rng.integers(0, N_TYPES, size=self.batch_size, dtype=np.int64)
        else:
            arr = np.asarray(theta, dtype=np.int64)
            if arr.shape != (self.batch_size,):
                raise ValueError(f"Expected theta shape ({self.batch_size},), got {arr.shape}")
            self.theta = arr.copy()
        self.q = np.full(self.batch_size, ACT_S, dtype=np.int64)
        self.turn = 0
        return {
            "prev_self_action": self.q.copy(),
            "prev_opp_action": np.full(self.batch_size, -1, dtype=np.int64),
            "prev_self_reward": np.zeros(self.batch_size, dtype=np.float64),
            "is_start": np.ones(self.batch_size, dtype=np.bool_),
            "reset_flag": np.zeros(self.batch_size, dtype=np.bool_),
            "turn": np.zeros(self.batch_size, dtype=np.int64),
        }

    def honest_opp_actions(self) -> np.ndarray:
        return self._f_table[self.theta, self.q]

    def step(
        self,
        focal_actions: np.ndarray,
        opp_actions: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(focal_actions, dtype=np.int64)
        y = self.honest_opp_actions() if opp_actions is None else np.asarray(opp_actions, dtype=np.int64)
        q_prev = self.q
        rf = self._r_focal[q_prev, a, y]
        ro = self._r_opp[self.theta, q_prev, a, y]
        self.q = a.copy()
        self.turn += 1
        obs = {
            "prev_self_action": self.q.copy(),
            "prev_opp_action": y.copy(),
            "prev_self_reward": rf.copy(),
            "is_start": np.zeros(self.batch_size, dtype=np.bool_),
            "reset_flag": np.zeros(self.batch_size, dtype=np.bool_),
            "turn": np.full(self.batch_size, self.turn, dtype=np.int64),
        }
        return obs, rf, ro, y
