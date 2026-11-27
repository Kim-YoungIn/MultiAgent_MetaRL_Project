"""Compact 6-Action 3-Layer Sequential Stage-Lock Delayed-Response Game (Exp 4B, Option A).

Eliminates the exponential flat-action explosion of 22-action MultiBarrier while preserving
genuine multi-layer epistemic exploration barriers over long horizons (T=48):

- Compact Focal Action Space (|A| = 6, smaller than R3's 8):
    0: S   (Safe floor: +0.2/turn, zero passive information leakage y=0)
    1: P   (Probe current stage key at -0.2; OR if current stage is unlocked & stage < 2,
            cross barrier into stage+1 at upfront cost -0.8 / -1.5 and probe new key)
    2..5: A0..A3 (4 symmetric arms shared across ALL 3 stages)
- Opponent Symbols (|Y| = 5): y in {0, 1, 2, 3, 4}
- Hidden Episode Keys: theta = (k_0, k_1, k_2) i.i.d. uniform in {0, 1, 2, 3}^3 (64 types).
- Public Deterministic Progression (uniquely determined by public (a, y, r) history):
    stage in {0, 1, 2}, unlocked in {0, 1}, q = prev_self_action (init stage=0, unlocked=0, q=S).
- Delayed-Response Honest Opponent f(stage, k_stage, q):
    q == S   -> 0
    q == P   -> 1 + k_stage in {1, 2, 3, 4}
    q == A_i -> 1 if i == k_stage else 0
- Focal Reward r_F(stage, unlocked, q, a, y):
    a == S: +0.2
    a == P:
        if unlocked == 1 and stage < 2: -0.8 (0->1) or -1.5 (1->2), stage += 1, unlocked = 0
        else: -0.2
    a == A_i:
        if q == a and y == 1: +0.7 (stage 0), +1.5 (stage 1), +2.5 (stage 2), unlocked = 1
        else: -0.6 (stage 0), -0.9 (stage 1), -1.2 (stage 2)

Includes an exact rational-arithmetic Gate-0 DP solver (`ExactStageLockDP`) proving
horizon-dependent optimal barrier depth: Tier 0 (T=4) -> Tier 1 (T=8) -> Tier 2 (T=16) -> Tier 3 (T=48).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import numpy as np

SL_N_ACTIONS = 6
SL_N_SYMBOLS = 5
SL_N_KEYS = 4
SL_N_STAGES = 3

SL_ACT_S = 0
SL_ACT_P = 1
SL_ACT_A0 = 2

SL_ACTION_NAMES = ("S", "P", "A0", "A1", "A2", "A3")
SL_FULL_KEY_MASK = (1 << SL_N_KEYS) - 1  # 15

# Exact rational payoff parameters
_R_SAFE = Fraction(1, 5)          # +0.2
_R_PROBE = Fraction(-1, 5)        # -0.2
_R_BARRIER = (Fraction(-4, 5), Fraction(-3, 2))                  # -0.8 (0->1), -1.5 (1->2)
_R_EXPLOIT_HIT = (Fraction(7, 10), Fraction(3, 2), Fraction(5, 2))   # +0.7, +1.5, +2.5
_R_EXPLOIT_MISS = (Fraction(-3, 5), Fraction(-9, 10), Fraction(-6, 5))  # -0.6, -0.9, -1.2

_R_BARRIER_F = tuple(float(x) for x in _R_BARRIER)
_R_EXPLOIT_HIT_F = tuple(float(x) for x in _R_EXPLOIT_HIT)
_R_EXPLOIT_MISS_F = tuple(float(x) for x in _R_EXPLOIT_MISS)


def sl_honest_response(k_stage: int, q: int) -> int:
    """Honest delayed response y = f(k_stage, q) in {0..4}."""
    if q == SL_ACT_S:
        return 0
    if q == SL_ACT_P:
        return 1 + int(k_stage)
    return 1 if (q - SL_ACT_A0) == int(k_stage) else 0


def sl_step_transition_rational(
    stage: int,
    unlocked: int,
    q: int,
    a: int,
    y: int,
) -> tuple[Fraction, int, int, bool]:
    """Exact rational (reward, next_stage, next_unlocked, reset_key_belief)."""
    if a == SL_ACT_S:
        return _R_SAFE, stage, unlocked, False
    if a == SL_ACT_P:
        if unlocked == 1 and stage < SL_N_STAGES - 1:
            return _R_BARRIER[stage], stage + 1, 0, True
        return _R_PROBE, stage, unlocked, False
    # Exploit arm A0..A3
    if q == a and y == 1:
        return _R_EXPLOIT_HIT[stage], stage, 1, False
    return _R_EXPLOIT_MISS[stage], stage, unlocked, False


class VectorizedStageLockEnv:
    """Fast vectorized NumPy simulator for B parallel 6-Action 3-Layer Stage-Lock games."""

    def __init__(self, batch_size: int, horizon: int = 48, seed: int = 0) -> None:
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self._rng = np.random.default_rng(seed)

        self.keys = np.zeros((self.batch_size, SL_N_STAGES), dtype=np.int64)
        self.stage = np.zeros(self.batch_size, dtype=np.int64)
        self.unlocked = np.zeros(self.batch_size, dtype=np.int64)
        self.q = np.full(self.batch_size, SL_ACT_S, dtype=np.int64)
        self.turn = 0

        # Precompute lookup table f_table[k, q] -> y (4, 6)
        self._f_table = np.zeros((SL_N_KEYS, SL_N_ACTIONS), dtype=np.int64)
        for k in range(SL_N_KEYS):
            for q in range(SL_N_ACTIONS):
                self._f_table[k, q] = sl_honest_response(k, q)

        self._hit_lut = np.array(_R_EXPLOIT_HIT_F, dtype=np.float64)
        self._miss_lut = np.array(_R_EXPLOIT_MISS_F, dtype=np.float64)
        self._bar_lut = np.array([_R_BARRIER_F[0], _R_BARRIER_F[1], float(_R_PROBE)], dtype=np.float64)

    def reset(self, keys: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if keys is None:
            self.keys = self._rng.integers(0, SL_N_KEYS, size=(self.batch_size, SL_N_STAGES), dtype=np.int64)
        else:
            arr = np.asarray(keys, dtype=np.int64)
            if arr.shape != (self.batch_size, SL_N_STAGES):
                raise ValueError(f"Expected keys shape ({self.batch_size}, {SL_N_STAGES}), got {arr.shape}")
            self.keys = arr.copy()
        self.stage = np.zeros(self.batch_size, dtype=np.int64)
        self.unlocked = np.zeros(self.batch_size, dtype=np.int64)
        self.q = np.full(self.batch_size, SL_ACT_S, dtype=np.int64)
        self.turn = 0
        return {
            "prev_self_action": self.q.copy(),
            "prev_opp_action": np.full(self.batch_size, -1, dtype=np.int64),
            "prev_self_reward": np.zeros(self.batch_size, dtype=np.float64),
            "is_start": np.ones(self.batch_size, dtype=np.bool_),
            "turn": np.zeros(self.batch_size, dtype=np.int64),
        }

    def honest_opp_actions(self) -> np.ndarray:
        idx = np.arange(self.batch_size)
        cur_k = self.keys[idx, self.stage]
        return self._f_table[cur_k, self.q]

    def step(
        self,
        focal_actions: np.ndarray,
        opp_actions: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(focal_actions, dtype=np.int64)
        honest_y = self.honest_opp_actions()
        y = honest_y if opp_actions is None else np.asarray(opp_actions, dtype=np.int64)

        q_prev = self.q
        st = self.stage
        ul = self.unlocked

        is_s = a == SL_ACT_S
        is_p = a == SL_ACT_P
        is_exp = a >= SL_ACT_A0

        cross = is_p & (ul == 1) & (st < SL_N_STAGES - 1)
        hit = is_exp & (q_prev == a) & (y == 1)
        miss = is_exp & (~hit)

        rf = np.zeros(self.batch_size, dtype=np.float64)
        rf[is_s] = float(_R_SAFE)
        rf[is_p & (~cross)] = float(_R_PROBE)
        rf[cross] = self._bar_lut[st[cross]]
        rf[hit] = self._hit_lut[st[hit]]
        rf[miss] = self._miss_lut[st[miss]]

        # Incentive-compatible opponent utility (honest response + cooperative match bonus)
        ro = np.where(y == honest_y, 2.0, 0.0) + np.where(hit, 0.5, 0.0)

        # State transitions
        next_stage = np.where(cross, st + 1, st)
        next_unlocked = np.where(cross, 0, np.where(hit, 1, ul))

        self.stage = next_stage
        self.unlocked = next_unlocked
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


class ExactStageLockDP:
    """Exact rational-arithmetic Gate-0 DP solver for 6-Action 3-Layer Stage-Lock.

    Supports max_tier in {0, 1, 2, 3}:
    - max_tier == 0: Safe S only (blind floor)
    - max_tier == 1: Allowed to exploit Stage 0, forbidden to cross barrier into Stage 1
    - max_tier == 2: Allowed up to Stage 1, forbidden to cross into Stage 2
    - max_tier == 3: Full 3-layer Bayes-Optimal policy
    """

    def __init__(self, max_tier: int = 3) -> None:
        if max_tier not in (0, 1, 2, 3):
            raise ValueError(f"max_tier must be in {{0,1,2,3}}, got {max_tier}")
        self.max_tier = int(max_tier)
        self._f_table = [
            [sl_honest_response(k, q) for q in range(SL_N_ACTIONS)]
            for k in range(SL_N_KEYS)
        ]

    def _allowed_actions(self, stage: int, unlocked: int) -> tuple[int, ...]:
        if self.max_tier == 0:
            return (SL_ACT_S,)
        # Forbid barrier crossing beyond allowed max_stage = max_tier - 1
        max_stage = self.max_tier - 1
        if unlocked == 1 and stage >= max_stage:
            return (SL_ACT_S, SL_ACT_A0, SL_ACT_A0 + 1, SL_ACT_A0 + 2, SL_ACT_A0 + 3)
        return tuple(range(SL_N_ACTIONS))

    def _split_belief(self, q: int, mask: int) -> tuple[tuple[int, int, int], ...]:
        sub = [0] * SL_N_SYMBOLS
        cnt = [0] * SL_N_SYMBOLS
        for k in range(SL_N_KEYS):
            if (mask >> k) & 1:
                y = self._f_table[k][q]
                sub[y] |= 1 << k
                cnt[y] += 1
        return tuple((y, sub[y], cnt[y]) for y in range(SL_N_SYMBOLS) if cnt[y] > 0)

    @lru_cache(maxsize=None)
    def value(self, rem: int, stage: int, unlocked: int, q: int, mask: int) -> Fraction:
        if rem <= 0:
            return Fraction(0, 1)
        splits = self._split_belief(q, mask)
        total = sum(c for _, _, c in splits)
        best = Fraction(-10**9, 1)
        for a in self._allowed_actions(stage, unlocked):
            ev = Fraction(0, 1)
            for y, sub_m, c in splits:
                r, ns, nu, reset_b = sl_step_transition_rational(stage, unlocked, q, a, y)
                nm = SL_FULL_KEY_MASK if reset_b else sub_m
                ev += Fraction(c, total) * (r + self.value(rem - 1, ns, nu, a, nm))
            if ev > best:
                best = ev
        return best

    def optimal_action(self, rem: int, stage: int, unlocked: int, q: int, mask: int) -> int:
        splits = self._split_belief(q, mask)
        total = sum(c for _, _, c in splits)
        best_a = SL_ACT_S
        best_v = Fraction(-10**9, 1)
        for a in self._allowed_actions(stage, unlocked):
            ev = Fraction(0, 1)
            for y, sub_m, c in splits:
                r, ns, nu, reset_b = sl_step_transition_rational(stage, unlocked, q, a, y)
                nm = SL_FULL_KEY_MASK if reset_b else sub_m
                ev += Fraction(c, total) * (r + self.value(rem - 1, ns, nu, a, nm))
            if ev > best_v:
                best_v = ev
                best_a = a
        return best_a

    def opening_q_values(self, horizon: int) -> dict[int, Fraction]:
        splits = self._split_belief(SL_ACT_S, SL_FULL_KEY_MASK)
        total = sum(c for _, _, c in splits)
        out: dict[int, Fraction] = {}
        for a in self._allowed_actions(0, 0):
            ev = Fraction(0, 1)
            for y, sub_m, c in splits:
                r, ns, nu, reset_b = sl_step_transition_rational(0, 0, SL_ACT_S, a, y)
                nm = SL_FULL_KEY_MASK if reset_b else sub_m
                ev += Fraction(c, total) * (r + self.value(horizon - 1, ns, nu, a, nm))
            out[a] = ev
        return out


def run_sl_gate0_report(horizons: tuple[int, ...] = (4, 8, 16, 24, 48)) -> dict[str, dict]:
    """Compute exact rational tier ceilings and optimal barrier depth for each horizon T."""
    solvers = {tier: ExactStageLockDP(max_tier=tier) for tier in (0, 1, 2, 3)}
    report: dict[str, dict] = {}
    for h in horizons:
        vals = {
            f"tier{t}_ceiling": float(solvers[t].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK))
            for t in (0, 1, 2, 3)
        }
        full_v = solvers[3].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK)
        opt_tier = min(t for t in (0, 1, 2, 3) if solvers[t].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK) == full_v)
        q_open = solvers[3].opening_q_values(h)
        best_open = max(q_open, key=lambda a: (q_open[a], -a))
        open_margin = float(q_open[best_open] - q_open[SL_ACT_S])
        report[f"T={h}"] = {
            **vals,
            "bayes_optimal": float(full_v),
            "optimal_barrier_tier": int(opt_tier),
            "optimal_opening": SL_ACTION_NAMES[best_open],
            "opening_margin_over_S": open_margin,
            "must_change_opening": bool(open_margin > 1e-9),
            "tier3_over_tier2_gain": float(
                solvers[3].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK)
                - solvers[2].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK)
            ),
            "tier2_over_tier1_gain": float(
                solvers[2].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK)
                - solvers[1].value(h, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK)
            ),
        }
    return report
