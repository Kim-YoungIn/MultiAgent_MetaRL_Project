"""Exact Rational Arithmetic DP for Gate-0 Verification and Layer-1 Bayes Audits.

Computes exact values (using fractions.Fraction — zero floating-point drift) for the
Simultaneous Delayed-Response Game across regimes R0, R1, R3:
1. Bayes-Optimal adaptive value V*_h(q, B) and Q*(prior, a) for all opening actions.
2. Clairvoyant task-aware value V_clairvoyant(q, B) (upper bound knowing true theta).
3. Blind / commitment value C_h(q, B) (optimal open-loop sequence with zero feedback).
4. Best K-turn open-loop prefix with retrospective disclosure at K (relaxed upper bound
   for masked-prefix policies, matching independent review §3).
5. Layer-1 posterior belief update and restored decision-relevant values for any prefix tau_{1:K}.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable

from mamrl.envs.delayed_response import (
    ACTION_NAMES,
    ACT_S,
    N_ACTIONS,
    N_TYPES,
    build_response_table,
    focal_reward_rational,
)

FULL_BELIEF_MASK = (1 << N_TYPES) - 1  # 0b1111 = all 4 types equiprobable


def _mask_members(mask: int) -> tuple[int, ...]:
    return tuple(th for th in range(N_TYPES) if (mask >> th) & 1)


def _partition_belief(f_table, q: int, mask: int) -> dict[int, int]:
    """Partition belief bitmask by exact opponent symbol y = f_theta(q)."""
    buckets: dict[int, int] = {}
    for th in _mask_members(mask):
        y = int(f_table[th, q])
        buckets[y] = buckets.get(y, 0) | (1 << th)
    return buckets


class ExactDelayedDP:
    """Exact rational dynamic programming solver for fixed reactive opponent regimes."""

    def __init__(self, regime: str, horizon: int = 24) -> None:
        self.regime = regime
        self.horizon = int(horizon)
        self._f_table = build_response_table(regime)
        self._v_adaptive: dict[tuple[int, int, int], Fraction] = {}
        self._v_blind: dict[tuple[int, int, int], Fraction] = {}
        self._v_clairvoyant: dict[tuple[int, int], Fraction] = {}
        self._solve_all()

    def _solve_all(self) -> None:
        # 1. Adaptive Bayes-Optimal DP: V_h(q, B)
        for q in range(N_ACTIONS):
            for mask in range(1, 1 << N_TYPES):
                self._v_adaptive[(0, q, mask)] = Fraction(0)
                self._v_blind[(0, q, mask)] = Fraction(0)

        for h in range(1, self.horizon + 1):
            for q in range(N_ACTIONS):
                for mask in range(1, 1 << N_TYPES):
                    n_b = mask.bit_count()
                    parts = _partition_belief(self._f_table, q, mask)

                    best_adapt: Fraction | None = None
                    best_blind: Fraction | None = None
                    for a in range(N_ACTIONS):
                        # Adaptive continuation updates belief B -> B_y
                        val_adapt = Fraction(0)
                        val_blind = Fraction(0)
                        for y, sub_mask in parts.items():
                            prob = Fraction(sub_mask.bit_count(), n_b)
                            r = focal_reward_rational(q, a, y)
                            val_adapt += prob * (r + self._v_adaptive[(h - 1, a, sub_mask)])
                            val_blind += prob * r
                        # Blind continuation keeps identical belief mask (cannot condition on y)
                        val_blind += self._v_blind[(h - 1, a, mask)]

                        if best_adapt is None or val_adapt > best_adapt:
                            best_adapt = val_adapt
                        if best_blind is None or val_blind > best_blind:
                            best_blind = val_blind

                    assert best_adapt is not None and best_blind is not None
                    self._v_adaptive[(h, q, mask)] = best_adapt
                    self._v_blind[(h, q, mask)] = best_blind

        # 2. Clairvoyant DP per known theta (singleton mask 1 << theta)
        for h in range(self.horizon + 1):
            for q in range(N_ACTIONS):
                for th in range(N_TYPES):
                    self._v_clairvoyant[(h, q, th)] = self._v_adaptive[(h, q, 1 << th)]

    def adaptive_value(self, remaining_turns: int, q: int = ACT_S, belief_mask: int = FULL_BELIEF_MASK) -> Fraction:
        return self._v_adaptive[(remaining_turns, q, belief_mask)]

    def commitment_value(self, remaining_turns: int, q: int = ACT_S, belief_mask: int = FULL_BELIEF_MASK) -> Fraction:
        """Optimal no-new-feedback (open-loop) value C_h(q, B)."""
        return self._v_blind[(remaining_turns, q, belief_mask)]

    def clairvoyant_value(self, remaining_turns: int, q: int = ACT_S, belief_mask: int = FULL_BELIEF_MASK) -> Fraction:
        """Expected optimal value if true theta in belief_mask were revealed immediately."""
        members = _mask_members(belief_mask)
        return sum((self._v_clairvoyant[(remaining_turns, q, th)] for th in members), Fraction(0)) / len(members)

    def opening_q_values(self, horizon: int | None = None, q: int = ACT_S, belief_mask: int = FULL_BELIEF_MASK) -> dict[int, Fraction]:
        """Exact Q*(q, B, a) for each opening action under optimal adaptive continuation."""
        h = self.horizon if horizon is None else int(horizon)
        n_b = belief_mask.bit_count()
        parts = _partition_belief(self._f_table, q, belief_mask)
        q_vals: dict[int, Fraction] = {}
        for a in range(N_ACTIONS):
            val = Fraction(0)
            for y, sub_mask in parts.items():
                prob = Fraction(sub_mask.bit_count(), n_b)
                val += prob * (focal_reward_rational(q, a, y) + self._v_adaptive[(h - 1, a, sub_mask)])
            q_vals[a] = val
        return q_vals

    def myopic_immediate_values(self, q: int = ACT_S, belief_mask: int = FULL_BELIEF_MASK) -> dict[int, Fraction]:
        n_b = belief_mask.bit_count()
        parts = _partition_belief(self._f_table, q, belief_mask)
        vals: dict[int, Fraction] = {}
        for a in range(N_ACTIONS):
            vals[a] = sum(
                (Fraction(sub_mask.bit_count(), n_b) * focal_reward_rational(q, a, y) for y, sub_mask in parts.items()),
                Fraction(0),
            )
        return vals

    def best_open_loop_prefix_value(self, k: int, horizon: int | None = None) -> Fraction:
        """Exact best K-turn open-loop action sequence with retrospective disclosure at K.

        This is a relaxed upper bound for any masked-prefix policy over turns 1..K (which
        cannot retrospectively view turns 1..K). Solved exactly via DP over
        (remaining_prefix_steps, q, partition_signature).
        """
        h = self.horizon if horizon is None else int(horizon)
        if k < 0 or k > h:
            raise ValueError(f"Invalid prefix length k={k} for horizon={h}")

        # Partition of the 4 equiprobable types represented as sorted tuple of bitmasks
        init_partition = (FULL_BELIEF_MASK,)
        memo: dict[tuple[int, int, tuple[int, ...]], Fraction] = {}

        def solve(step: int, q: int, partition: tuple[int, ...]) -> Fraction:
            if step == k:
                rem = h - k
                return sum(
                    (Fraction(m.bit_count(), N_TYPES) * self._v_adaptive[(rem, q, m)] for m in partition),
                    Fraction(0),
                )
            key = (step, q, partition)
            if key in memo:
                return memo[key]

            # All types in each block emit y = f_theta(q), refining the partition once observed at K
            new_blocks: list[int] = []
            for m in partition:
                sub = _partition_belief(self._f_table, q, m)
                new_blocks.extend(sub.values())
            next_partition = tuple(sorted(new_blocks))

            best: Fraction | None = None
            for a in range(N_ACTIONS):
                # Expected immediate reward across all 4 equiprobable types
                imm = sum(
                    (focal_reward_rational(q, a, int(self._f_table[th, q])) for th in range(N_TYPES)),
                    Fraction(0),
                ) / N_TYPES
                val = imm + solve(step + 1, a, next_partition)
                if best is None or val > best:
                    best = val
            assert best is not None
            memo[key] = best
            return best

        return solve(0, ACT_S, init_partition)

    def exact_posterior_mask(self, trajectory: Iterable[tuple[int, int]]) -> int:
        """Given realized (q_prev, y_observed) pairs under fixed reactive opponent, return surviving belief mask."""
        mask = FULL_BELIEF_MASK
        for q_prev, y_obs in trajectory:
            new_mask = 0
            for th in _mask_members(mask):
                if int(self._f_table[th, q_prev]) == int(y_obs):
                    new_mask |= 1 << th
            mask = new_mask
        return mask

    def audit_prefix_layer1(
        self,
        trajectory_qy: list[tuple[int, int]],
        prefix_reward: float,
        remaining_turns: int,
        q_current: int,
    ) -> dict[str, float | int]:
        """Layer-1 policy-agnostic exact Bayes decision-value audit of a realized prefix."""
        mask = self.exact_posterior_mask(trajectory_qy)
        if mask == 0:
            raise ValueError("Inconsistent trajectory: posterior support is empty.")
        n_surv = mask.bit_count()
        entropy_bits = math.log2(n_surv)
        c_restored = float(self.commitment_value(remaining_turns, ACT_S, mask))
        v_restored = float(self.adaptive_value(remaining_turns, ACT_S, mask))
        v_clairvoyant_restored = float(self.clairvoyant_value(remaining_turns, ACT_S, mask))
        v_live = float(self.adaptive_value(remaining_turns, q_current, mask))
        return {
            "posterior_support_size": n_surv,
            "posterior_entropy_bits": entropy_bits,
            "rule_identified": 1.0 if n_surv == 1 else 0.0,
            "prefix_reward": float(prefix_reward),
            "restored_commitment_value": c_restored,
            "restored_adaptive_value": v_restored,
            "restored_opportunity_gap": v_clairvoyant_restored - v_restored,
            "net_restored_commitment_return": float(prefix_reward) + c_restored,
            "net_restored_adaptive_return": float(prefix_reward) + v_restored,
            "live_bayes_continuation_value": v_live,
        }


@dataclass(frozen=True)
class Gate0Report:
    regime: str
    horizon: int
    bayes_optimal_value: Fraction
    clairvoyant_value: Fraction
    blind_value: Fraction
    feedback_value: Fraction
    myopic_openings: tuple[int, ...]
    optimal_openings: tuple[int, ...]
    opening_q_values: dict[str, Fraction]
    opening_margin: Fraction
    must_change_opening: bool
    open_loop_bounds: dict[int, Fraction]
    feedback_conditioning_headroom: dict[int, Fraction]

    def to_summary_dict(self) -> dict:
        return {
            "regime": self.regime,
            "horizon": self.horizon,
            "bayes_optimal_value": float(self.bayes_optimal_value),
            "clairvoyant_value": float(self.clairvoyant_value),
            "blind_value": float(self.blind_value),
            "feedback_value": float(self.feedback_value),
            "myopic_openings": [ACTION_NAMES[a] for a in self.myopic_openings],
            "optimal_openings": [ACTION_NAMES[a] for a in self.optimal_openings],
            "opening_q_values": {k: float(v) for k, v in self.opening_q_values.items()},
            "opening_margin": float(self.opening_margin),
            "must_change_opening": self.must_change_opening,
            "open_loop_bounds": {str(k): float(v) for k, v in self.open_loop_bounds.items()},
            "feedback_conditioning_headroom": {
                str(k): float(v) for k, v in self.feedback_conditioning_headroom.items()
            },
        }


def run_gate0_audit(regime: str, horizon: int = 24, audit_ks: tuple[int, ...] = (2, 4, 8)) -> Gate0Report:
    """Run exact rational Gate-0 verification for a regime."""
    dp = ExactDelayedDP(regime=regime, horizon=horizon)
    q_vals = dp.opening_q_values(horizon=horizon)
    myopic_vals = dp.myopic_immediate_values()

    max_myopic = max(myopic_vals.values())
    myopic_openings = tuple(a for a, v in myopic_vals.items() if v == max_myopic)

    v_opt = max(q_vals.values())
    optimal_openings = tuple(a for a, v in q_vals.items() if v == v_opt)

    best_under_myopic = max(q_vals[a] for a in myopic_openings)
    margin = v_opt - best_under_myopic

    v_blind = dp.commitment_value(horizon)
    v_clair = dp.clairvoyant_value(horizon)

    ol_bounds: dict[int, Fraction] = {}
    headroom: dict[int, Fraction] = {}
    for k in audit_ks:
        if k <= horizon:
            b_val = dp.best_open_loop_prefix_value(k, horizon=horizon)
            ol_bounds[k] = b_val
            headroom[k] = v_opt - b_val

    return Gate0Report(
        regime=regime,
        horizon=horizon,
        bayes_optimal_value=v_opt,
        clairvoyant_value=v_clair,
        blind_value=v_blind,
        feedback_value=v_opt - v_blind,
        myopic_openings=myopic_openings,
        optimal_openings=optimal_openings,
        opening_q_values={ACTION_NAMES[a]: q_vals[a] for a in range(N_ACTIONS)},
        opening_margin=margin,
        must_change_opening=margin > Fraction(0),
        open_loop_bounds=ol_bounds,
        feedback_conditioning_headroom=headroom,
    )
