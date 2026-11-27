"""3-Layer Nested Exploration-Barrier Delayed-Response Game (Exp 4).

Extends the Simultaneous Delayed-Response Game to long horizons (T=64) with 3 nested
epistemic barriers (8 hidden opponent types theta = (b, c, d) in {0, 1}^3, 22 focal
actions, 3 opponent symbols y in {0, 1, 2}):

- Tier 0 (Safe Floor, 1 action):
    S (0): +0.2/turn, f_theta(S) = 0
- Tier 1 (Shallow Barrier, 1-step probe + 2 exploits):
    P1 (1): -0.1/turn, f_theta(P1) = 1 + b in {1, 2}
    T1_0..T1_1 (2..3): +0.7 if (q == a and y == 1) else -1.0; f_theta(T1_i) = 1[b == i]
- Tier 2 (Intermediate Barrier, b-gated 2-step probe + 4 exploits):
    P2_0..P2_1 (4..5): -0.1/turn, f_theta(P2_i) = (1 + c) if b == i else 0
    T2_00..T2_11 (6..9): +1.3 if (q == a and y == 1) else -3.0; f_theta(T2_ij) = 1[(b,c) == (i,j)]
- Tier 3 (Deep Barrier / Jackpot, (b,c)-gated 3-step probe + 8 exploits):
    P3_00..P3_11 (10..13): -0.1/turn, f_theta(P3_ij) = (1 + d) if (b,c) == (i,j) else 0
    T3_000..T3_111 (14..21): +2.2 if (q == a and y == 1) else -10.0; f_theta(T3_ijk) = 1[theta == (i,j,k)]

Includes an exact rational-arithmetic top-down memoized Gate-0 DP solver verifying
horizon-dependent optimal barrier depth (e.g. Tier 1 at T=10, Tier 2 at T=16, Tier 3 at T=64).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import numpy as np

MB_N_TYPES = 8
MB_N_SYMBOLS = 3
MB_N_ACTIONS = 22

MB_ACT_S = 0
MB_ACT_P1 = 1
MB_ACT_T1_START = 2    # 2..3 (2 actions)
MB_ACT_P2_START = 4    # 4..5 (2 actions)
MB_ACT_T2_START = 6    # 6..9 (4 actions)
MB_ACT_P3_START = 10   # 10..13 (4 actions)
MB_ACT_T3_START = 14   # 14..21 (8 actions)

MB_ACTION_NAMES = (
    "S",
    "P1",
    "T1_0", "T1_1",
    "P2_0", "P2_1",
    "T2_00", "T2_01", "T2_10", "T2_11",
    "P3_00", "P3_01", "P3_10", "P3_11",
    "T3_000", "T3_001", "T3_010", "T3_011", "T3_100", "T3_101", "T3_110", "T3_111",
)

MB_FULL_BELIEF_MASK = (1 << MB_N_TYPES) - 1  # 255


def mb_action_tier(a: int) -> int:
    """Return exploit tier {0, 1, 2, 3} or -1 for probe actions."""
    if a == MB_ACT_S:
        return 0
    if MB_ACT_T1_START <= a < MB_ACT_P2_START:
        return 1
    if MB_ACT_T2_START <= a < MB_ACT_P3_START:
        return 2
    if MB_ACT_T3_START <= a < MB_N_ACTIONS:
        return 3
    return -1  # probe


def build_mb_response_table() -> np.ndarray:
    """Exact integer table f_table[theta, q] -> y in {0, 1, 2} of shape (8, 22)."""
    table = np.zeros((MB_N_TYPES, MB_N_ACTIONS), dtype=np.int64)
    for th in range(MB_N_TYPES):
        b = (th >> 2) & 1
        c = (th >> 1) & 1
        d = th & 1
        bc = (b << 1) | c

        table[th, MB_ACT_S] = 0
        table[th, MB_ACT_P1] = 1 + b

        # Tier 1 exploits
        table[th, MB_ACT_T1_START + b] = 1

        # Tier 2 gated probes & exploits
        table[th, MB_ACT_P2_START + b] = 1 + c
        table[th, MB_ACT_T2_START + bc] = 1

        # Tier 3 gated probes & exploits
        table[th, MB_ACT_P3_START + bc] = 1 + d
        table[th, MB_ACT_T3_START + th] = 1

    return table


def mb_focal_reward_rational(q: int, a: int, y: int) -> Fraction:
    """Exact rational focal reward r_F(q, a, y)."""
    if a == MB_ACT_S:
        return Fraction(1, 5)  # +0.2
    if a == MB_ACT_P1 or (MB_ACT_P2_START <= a < MB_ACT_T2_START) or (MB_ACT_P3_START <= a < MB_ACT_T3_START):
        return Fraction(-1, 10)  # -0.1
    if MB_ACT_T1_START <= a < MB_ACT_P2_START:
        return Fraction(7, 10) if (q == a and y == 1) else Fraction(-1, 1)
    if MB_ACT_T2_START <= a < MB_ACT_P3_START:
        return Fraction(13, 10) if (q == a and y == 1) else Fraction(-3, 1)
    if MB_ACT_T3_START <= a < MB_N_ACTIONS:
        return Fraction(22, 10) if (q == a and y == 1) else Fraction(-10, 1)
    raise ValueError(f"Invalid multi-barrier action: {a}")


def build_mb_focal_reward_table() -> np.ndarray:
    r = np.zeros((MB_N_ACTIONS, MB_N_ACTIONS, MB_N_SYMBOLS), dtype=np.float64)
    for q in range(MB_N_ACTIONS):
        for a in range(MB_N_ACTIONS):
            for y in range(MB_N_SYMBOLS):
                r[q, a, y] = float(mb_focal_reward_rational(q, a, y))
    return r


def build_mb_opp_reward_table() -> np.ndarray:
    f_table = build_mb_response_table()
    r = np.zeros((MB_N_TYPES, MB_N_ACTIONS, MB_N_ACTIONS, MB_N_SYMBOLS), dtype=np.float64)
    for th in range(MB_N_TYPES):
        for q in range(MB_N_ACTIONS):
            honest_y = int(f_table[th, q])
            is_exploit_q = mb_action_tier(q) >= 1
            for a in range(MB_N_ACTIONS):
                for y in range(MB_N_SYMBOLS):
                    val = 2.0 if y == honest_y else 0.0
                    if is_exploit_q and a == q and y == 1:
                        val += 0.5
                    r[th, q, a, y] = val
    return r


class VectorizedMultiBarrierEnv:
    """Fast vectorized NumPy simulator for B parallel 3-Layer Multi-Barrier games."""

    def __init__(self, batch_size: int, horizon: int = 64, seed: int = 0) -> None:
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self._rng = np.random.default_rng(seed)
        self._f_table = build_mb_response_table()
        self._r_focal = build_mb_focal_reward_table()
        self._r_opp = build_mb_opp_reward_table()
        self.theta = np.zeros(self.batch_size, dtype=np.int64)
        self.q = np.full(self.batch_size, MB_ACT_S, dtype=np.int64)
        self.turn = 0

    def reset(self, theta: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if theta is None:
            self.theta = self._rng.integers(0, MB_N_TYPES, size=self.batch_size, dtype=np.int64)
        else:
            arr = np.asarray(theta, dtype=np.int64)
            if arr.shape != (self.batch_size,):
                raise ValueError(f"Expected theta shape ({self.batch_size},), got {arr.shape}")
            self.theta = arr.copy()
        self.q = np.full(self.batch_size, MB_ACT_S, dtype=np.int64)
        self.turn = 0
        return {
            "prev_self_action": self.q.copy(),
            "prev_opp_action": np.full(self.batch_size, -1, dtype=np.int64),
            "prev_self_reward": np.zeros(self.batch_size, dtype=np.float64),
            "is_start": np.ones(self.batch_size, dtype=np.bool_),
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
            "turn": np.full(self.batch_size, self.turn, dtype=np.int64),
        }
        return obs, rf, ro, y


class ExactMultiBarrierDP:
    """Exact rational-arithmetic Gate-0 DP solver for the 3-Layer Multi-Barrier Game.

    Supports restricting allowed max_tier in {0, 1, 2, 3} so we can prove exact tier ceilings
    (Tier-0 Blind, Tier-1 Cap, Tier-2 Cap, Tier-3 Full Bayes-Opt) for any horizon T!
    """

    def __init__(self, max_tier: int = 3) -> None:
        if max_tier not in (0, 1, 2, 3):
            raise ValueError(f"max_tier must be in {{0,1,2,3}}, got {max_tier}")
        self.max_tier = int(max_tier)
        self._f_table = build_mb_response_table()
        self._r_rat = [
            [[mb_focal_reward_rational(q, a, y) for y in range(MB_N_SYMBOLS)] for a in range(MB_N_ACTIONS)]
            for q in range(MB_N_ACTIONS)
        ]
        # Allowed actions up to max_tier
        allowed = [MB_ACT_S]
        if max_tier >= 1:
            allowed.extend([MB_ACT_P1, MB_ACT_T1_START, MB_ACT_T1_START + 1])
        if max_tier >= 2:
            allowed.extend(range(MB_ACT_P2_START, MB_ACT_P3_START))
        if max_tier >= 3:
            allowed.extend(range(MB_ACT_P3_START, MB_N_ACTIONS))
        self.allowed_actions = tuple(allowed)

    def _split_belief(self, q: int, mask: int) -> tuple[tuple[int, int, int], ...]:
        """Return tuple of (y, sub_mask, count) for each y with non-zero surviving types."""
        sub = [0] * MB_N_SYMBOLS
        cnt = [0] * MB_N_SYMBOLS
        for th in range(MB_N_TYPES):
            if (mask >> th) & 1:
                y = int(self._f_table[th, q])
                sub[y] |= 1 << th
                cnt[y] += 1
        return tuple((y, sub[y], cnt[y]) for y in range(MB_N_SYMBOLS) if cnt[y] > 0)

    @lru_cache(maxsize=None)
    def value(self, rem: int, q: int, mask: int) -> Fraction:
        if rem <= 0:
            return Fraction(0, 1)
        splits = self._split_belief(q, mask)
        total = sum(c for _, _, c in splits)
        best = Fraction(-10**9, 1)
        for a in self.allowed_actions:
            exp_val = Fraction(0, 1)
            for y, sub_m, c in splits:
                exp_val += Fraction(c, total) * (self._r_rat[q][a][y] + self.value(rem - 1, a, sub_m))
            if exp_val > best:
                best = exp_val
        return best

    def optimal_action(self, rem: int, q: int, mask: int) -> int:
        splits = self._split_belief(q, mask)
        total = sum(c for _, _, c in splits)
        best_a = MB_ACT_S
        best_v = Fraction(-10**9, 1)
        for a in self.allowed_actions:
            exp_val = Fraction(0, 1)
            for y, sub_m, c in splits:
                exp_val += Fraction(c, total) * (self._r_rat[q][a][y] + self.value(rem - 1, a, sub_m))
            if exp_val > best_v:
                best_v = exp_val
                best_a = a
        return best_a

    def opening_q_values(self, horizon: int) -> dict[int, Fraction]:
        splits = self._split_belief(MB_ACT_S, MB_FULL_BELIEF_MASK)
        total = sum(c for _, _, c in splits)
        out: dict[int, Fraction] = {}
        for a in self.allowed_actions:
            exp_val = Fraction(0, 1)
            for y, sub_m, c in splits:
                exp_val += Fraction(c, total) * (self._r_rat[MB_ACT_S][a][y] + self.value(horizon - 1, a, sub_m))
            out[a] = exp_val
        return out


def run_mb_gate0_report(horizons: tuple[int, ...] = (10, 16, 24, 48, 64)) -> dict[str, dict]:
    """Compute exact rational tier ceilings and optimal barrier depth for each horizon T."""
    solvers = {tier: ExactMultiBarrierDP(max_tier=tier) for tier in (0, 1, 2, 3)}
    report: dict[str, dict] = {}
    for h in horizons:
        vals = {f"tier{t}_ceiling": float(solvers[t].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK)) for t in (0, 1, 2, 3)}
        # Determine lowest tier that achieves the full Tier-3 Bayes-Optimal value
        full_v = solvers[3].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK)
        opt_tier = min(t for t in (0, 1, 2, 3) if solvers[t].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK) == full_v)
        q_open = solvers[3].opening_q_values(h)
        best_open = max(q_open, key=lambda a: (q_open[a], -a))
        open_margin = float(q_open[best_open] - q_open[MB_ACT_S])
        report[f"T={h}"] = {
            **vals,
            "bayes_optimal": float(full_v),
            "optimal_barrier_tier": int(opt_tier),
            "optimal_opening": MB_ACTION_NAMES[best_open],
            "opening_margin_over_S": open_margin,
            "tier3_over_tier2_gain": float(solvers[3].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK) - solvers[2].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK)),
            "tier2_over_tier1_gain": float(solvers[2].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK) - solvers[1].value(h, MB_ACT_S, MB_FULL_BELIEF_MASK)),
        }
    return report
