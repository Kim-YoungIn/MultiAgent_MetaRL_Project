"""Symmetric 2-Player 4-Action Repeated Games for SPEC v3.0 Exp 2B.

Two game-theoretic families with ZERO dedicated PROBE buttons and strictly disjoint
Train vs. Held-Out Unseen Test rule splits (balanced across all 4 action slots):

1. 'comp_rps' (Hidden Cyclic-Dominance Zero-Sum Competition):
   4 actions = 1 Neutral Anchor (0.0 vs all) + 3-action Rock-Paper-Scissors (+1/-1/0).
   8 strictly distinct symmetric zero-sum payoff matrices:
   - Train (4 rules): (Anchor=0, CW), (Anchor=1, CCW), (Anchor=2, CW), (Anchor=3, CCW)
   - Unseen Test (4 rules): (Anchor=0, CCW), (Anchor=1, CW), (Anchor=2, CCW), (Anchor=3, CW)
   Every action is Anchor 25% of the time in both splits; zero matrix overlap.

2. 'coop_trap' (Hidden Coordination / Mixed-Motive Stag-Hunt + Trap):
   4 actions = (Safe, Coop, Defect, Trap) permuted across all 4! = 24 strictly distinct
   symmetric matrices:
   - Mutual Coop (c*, c*): (+1.2, +1.2) [Pareto-optimal target]
   - Safe (s, *): +0.3 risk-free baseline
   - Defect (d) vs Coop (c*): (+0.6, -0.5); Mutual Defect: (-0.2, -0.2); d vs Safe: (+0.1, +0.3)
   - Trap (x): -0.8 vs non-Trap, -1.0 vs Trap (costly blind exploration hazard)
   Latin-balanced split: 16 Train rules / 8 Held-Out Unseen Test rules (every action holds
   each role exactly 25% of the time in both splits; zero overlap).
"""

from __future__ import annotations

from itertools import permutations
import numpy as np

SYM_N_ACTIONS = 4
VALID_SYM_GAMES = ("comp_rps", "coop_trap")

# Canonical role-space payoff matrix R1[role_p1, role_p2] for coop_trap:
# Roles: 0=Safe, 1=Coop, 2=Defect, 3=Trap
COOP_TRAP_CANONICAL_R1 = np.array(
    [
        [+0.3, +0.3, +0.3, +0.3],  # Safe vs {Safe, Coop, Defect, Trap}
        [-0.5, +1.2, -0.5, -0.5],  # Coop vs {Safe, Coop, Defect, Trap}
        [+0.1, +0.6, -0.2, -0.6],  # Defect vs {Safe, Coop, Defect, Trap}
        [-0.8, -0.8, -0.8, -1.0],  # Trap vs {Safe, Coop, Defect, Trap}
    ],
    dtype=np.float64,
)

# Two disjoint Latin squares of S_4 (8 permutations total) -> strictly balanced Held-Out Test split
_COOP_TEST_PERMS = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (2, 1, 0, 3),
    (3, 2, 1, 0),
)


def _build_comp_rps_matrix(anchor: int, clockwise: bool) -> np.ndarray:
    """Build exact 4x4 zero-sum payoff matrix for Player 1 given Anchor index and orientation."""
    r1 = np.zeros((SYM_N_ACTIONS, SYM_N_ACTIONS), dtype=np.float64)
    cycle = [a for a in range(SYM_N_ACTIONS) if a != anchor]
    if not clockwise:
        cycle = [cycle[0], cycle[2], cycle[1]]
    # cycle[0] beats cycle[1], cycle[1] beats cycle[2], cycle[2] beats cycle[0]
    for i in range(3):
        winner = cycle[i]
        loser = cycle[(i + 1) % 3]
        r1[winner, loser] = +1.0
        r1[loser, winner] = -1.0
    return r1


def _build_coop_trap_matrix(role_of_action: tuple[int, int, int, int]) -> np.ndarray:
    """Build exact 4x4 symmetric payoff matrix for Player 1 given role_of_action[a] in {0..3}."""
    r1 = np.zeros((SYM_N_ACTIONS, SYM_N_ACTIONS), dtype=np.float64)
    for a1 in range(SYM_N_ACTIONS):
        for a2 in range(SYM_N_ACTIONS):
            r1[a1, a2] = COOP_TRAP_CANONICAL_R1[role_of_action[a1], role_of_action[a2]]
    return r1


def build_symmetric_rule_bank(game: str) -> dict[str, dict[str, np.ndarray]]:
    """Return {'train': ..., 'test': ...} rule banks with payoffs R1[K,4,4], R2[K,4,4], and role metadata.

    Guarantees:
    - R2[k, a1, a2] == R1[k, a2, a1] (strict symmetry).
    - Zero payoff matrix overlap between 'train' and 'test'.
    """
    if game not in VALID_SYM_GAMES:
        raise ValueError(f"Unsupported symmetric game '{game}'. Expected one of {VALID_SYM_GAMES}.")

    if game == "comp_rps":
        train_specs = [(0, True), (1, False), (2, True), (3, False)]
        test_specs = [(0, False), (1, True), (2, False), (3, True)]
        out = {}
        for split_name, specs in (("train", train_specs), ("test", test_specs)):
            r1_list = [_build_comp_rps_matrix(anc, cw) for anc, cw in specs]
            r1 = np.stack(r1_list, axis=0)
            r2 = np.transpose(r1, (0, 2, 1)).copy()
            anchors = np.array([anc for anc, _ in specs], dtype=np.int64)
            out[split_name] = {
                "r1": r1,
                "r2": r2,
                "anchor": anchors,
                "n_rules": np.int64(len(specs)),
            }
        return out

    # coop_trap: all 24 permutations partitioned into 16 train / 8 Latin-balanced test
    test_set = set(_COOP_TEST_PERMS)
    all_perms = list(permutations(range(SYM_N_ACTIONS)))
    train_perms = [p for p in all_perms if p not in test_set]
    test_perms = list(_COOP_TEST_PERMS)

    out = {}
    for split_name, plist in (("train", train_perms), ("test", test_perms)):
        r1 = np.stack([_build_coop_trap_matrix(p) for p in plist], axis=0)
        r2 = np.transpose(r1, (0, 2, 1)).copy()
        safe_act = np.array([p.index(0) for p in plist], dtype=np.int64)
        coop_act = np.array([p.index(1) for p in plist], dtype=np.int64)
        defect_act = np.array([p.index(2) for p in plist], dtype=np.int64)
        trap_act = np.array([p.index(3) for p in plist], dtype=np.int64)
        out[split_name] = {
            "r1": r1,
            "r2": r2,
            "safe": safe_act,
            "coop": coop_act,
            "defect": defect_act,
            "trap": trap_act,
            "n_rules": np.int64(len(plist)),
        }
    return out


class VectorizedSymmetricGameEnv:
    """Fast vectorized simulator for B parallel symmetric 2-player 4-action repeated games."""

    def __init__(
        self,
        game: str,
        split: str,
        batch_size: int,
        horizon: int = 24,
        seed: int = 0,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split: {split}")
        self.game = game
        self.split = split
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self._rng = np.random.default_rng(seed)

        bank = build_symmetric_rule_bank(game)[split]
        self.bank = bank
        self.r1_bank = bank["r1"]  # (K, 4, 4)
        self.r2_bank = bank["r2"]  # (K, 4, 4)
        self.n_rules = int(self.r1_bank.shape[0])

        self.rule_ids = np.zeros(self.batch_size, dtype=np.int64)
        self.turn = 0

    def reset(self, rule_ids: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if rule_ids is None:
            self.rule_ids = self._rng.integers(0, self.n_rules, size=self.batch_size, dtype=np.int64)
        else:
            arr = np.asarray(rule_ids, dtype=np.int64)
            if arr.shape != (self.batch_size,):
                raise ValueError(f"Expected rule_ids shape ({self.batch_size},), got {arr.shape}")
            self.rule_ids = arr.copy()
        self.turn = 0
        obs1 = {
            "prev_self_action": np.full(self.batch_size, -1, dtype=np.int64),
            "prev_opp_action": np.full(self.batch_size, -1, dtype=np.int64),
            "prev_self_reward": np.zeros(self.batch_size, dtype=np.float64),
            "is_start": np.ones(self.batch_size, dtype=np.bool_),
            "turn": np.zeros(self.batch_size, dtype=np.int64),
        }
        obs2 = {k: v.copy() for k, v in obs1.items()}
        return obs1, obs2

    def step(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
        a1_arr = np.asarray(a1, dtype=np.int64)
        a2_arr = np.asarray(a2, dtype=np.int64)
        r1 = self.r1_bank[self.rule_ids, a1_arr, a2_arr]
        r2 = self.r2_bank[self.rule_ids, a1_arr, a2_arr]
        self.turn += 1
        obs1 = {
            "prev_self_action": a1_arr.copy(),
            "prev_opp_action": a2_arr.copy(),
            "prev_self_reward": r1.copy(),
            "is_start": np.zeros(self.batch_size, dtype=np.bool_),
            "turn": np.full(self.batch_size, self.turn, dtype=np.int64),
        }
        obs2 = {
            "prev_self_action": a2_arr.copy(),
            "prev_opp_action": a1_arr.copy(),
            "prev_self_reward": r2.copy(),
            "is_start": np.zeros(self.batch_size, dtype=np.bool_),
            "turn": np.full(self.batch_size, self.turn, dtype=np.int64),
        }
        return obs1, obs2, r1, r2
