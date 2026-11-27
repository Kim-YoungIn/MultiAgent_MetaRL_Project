"""Semantically Diverse Symmetric 2-Player 4-Action Repeated Games for SPEC v3.0 Exp 3B.

Unlike pure action-label relabeling, this benchmark provides genuinely distinct strategic
game families (different Nash equilibria and payoff dynamics) separated into:
- Training semantic families (4 canonical structures, N_sem=2 subset vs N_sem=4 full):
  1. 'stag_hunt'         : Risk-reward mutual coordination (+1.2/+1.2 vs Safe +0.3, sucker -0.5, trap -0.8)
  2. 'dilemma_tempt'     : Iterated Prisoner's Dilemma (+0.9/+0.9 coop, +1.3/-0.6 tempt, -0.3/-0.3 clash, Safe +0.2, trap -0.8)
  3. 'chicken_anticoord' : Complementary anti-coordination (+1.0/+1.0 when playing opposite A/B, -0.7/-0.7 clash, Safe +0.25, trap -0.8)
  4. 'cyclic_rps'        : Zero-sum 3-action cyclic dominance (+1.0/-1.0) + Neutral Anchor (0.0)
- Held-out UNSEEN semantic test suites:
  - 'unseen_params'  : 4 new payoff parameterizations (1 per family) never seen during training
  - 'unseen_family'  : Brand-new 5th strategic family 'dual_equilibrium_battle' (two distinct symmetric
                       coordination equilibria +1.1/+1.1 and +0.8/+0.8 with -0.4/-0.4 miscoordination,
                       Safe +0.2, Trap -0.9) — NEVER seen in any training episode!

Action-label permutations are also partitioned cleanly (16 train perms vs 8 Latin-balanced
held-out test perms) so we can separately test:
- 'in_dist'         : Seen semantic families x Seen permutations
- 'unseen_perm'     : Seen semantic families x Unseen permutations (anti-memorization test)
- 'unseen_params'   : Unseen payoff parameterizations x Unseen permutations
- 'unseen_family'   : Unseen 5th strategic family x Unseen permutations
"""

from __future__ import annotations

from itertools import permutations
import numpy as np

SEM_N_ACTIONS = 4

# Canonical 4x4 symmetric payoff matrices R1[a1, a2] (Player 2 payoff is R1.T)
CANONICAL_TRAIN_FAMILIES: dict[str, np.ndarray] = {
    "stag_hunt": np.array(
        [
            [+0.3, +0.3, +0.3, +0.3],  # Safe
            [-0.5, +1.2, -0.5, -0.5],  # Coop (Stag)
            [+0.1, +0.6, -0.2, -0.6],  # Mild defect
            [-0.8, -0.8, -0.8, -1.0],  # Trap
        ],
        dtype=np.float64,
    ),
    "dilemma_tempt": np.array(
        [
            [+0.2, +0.2, +0.2, +0.2],  # Safe
            [-0.6, +0.9, -0.6, -0.6],  # Coop
            [+0.1, +1.3, -0.3, -0.6],  # Strong Temptation / Defect
            [-0.8, -0.8, -0.8, -1.0],  # Trap
        ],
        dtype=np.float64,
    ),
    "chicken_anticoord": np.array(
        [
            [+0.25, +0.25, +0.25, +0.25],  # Safe
            [-0.2, -0.7, +1.15, -0.5],     # Hawk-like anti-coord (wants opp to play Dove!)
            [-0.2, +0.85, -0.55, -0.5],    # Dove-like anti-coord (wants opp to play Hawk!)
            [-0.8, -0.8, -0.8, -1.0],      # Trap
        ],
        dtype=np.float64,
    ),
    "cyclic_rps": np.array(
        [
            [0.0, 0.0, 0.0, 0.0],     # Neutral Anchor
            [0.0, 0.0, -1.1, +0.9],   # Weighted zero-sum cyclic triad (all 4 roles distinct -> 24 unique matrices)
            [0.0, +1.1, 0.0, -1.0],
            [0.0, -0.9, +1.0, 0.0],
        ],
        dtype=np.float64,
    ),
}

CANONICAL_UNSEEN_PARAMS: dict[str, np.ndarray] = {
    "stag_hunt_highstakes": np.array(
        [
            [+0.25, +0.25, +0.25, +0.25],
            [-0.7, +1.4, -0.7, -0.7],
            [+0.05, +0.75, -0.3, -0.7],
            [-0.9, -0.9, -0.9, -1.1],
        ],
        dtype=np.float64,
    ),
    "dilemma_harsh": np.array(
        [
            [+0.15, +0.15, +0.15, +0.15],
            [-0.75, +1.1, -0.75, -0.75],
            [+0.0, +1.45, -0.45, -0.7],
            [-0.9, -0.9, -0.9, -1.1],
        ],
        dtype=np.float64,
    ),
    "chicken_asym_bos": np.array(
        [
            [+0.2, +0.2, +0.2, +0.2],
            [-0.3, -0.8, +1.25, -0.6],
            [-0.3, +0.85, -0.6, -0.6],
            [-0.9, -0.9, -0.9, -1.1],
        ],
        dtype=np.float64,
    ),
    "cyclic_rps_flipped_weighted": np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, +1.2, -1.0],
            [0.0, -1.2, 0.0, +1.0],
            [0.0, +1.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    ),
}

# Brand-new 5th strategic structure: Dual symmetric equilibria (High-Coord +1.1 & Alt-Coord +0.8,
# miscoordination between them gives -0.4, Safe +0.2, Trap -0.9). Never in train!
CANONICAL_UNSEEN_FAMILY: dict[str, np.ndarray] = {
    "dual_eq_battle": np.array(
        [
            [+0.2, +0.2, +0.2, +0.2],    # Safe
            [-0.4, +1.1, -0.4, -0.6],    # Primary Coord Equilibrium
            [-0.4, -0.4, +0.8, -0.6],    # Secondary Coord Equilibrium
            [-0.9, -0.9, -0.9, -1.1],    # Trap
        ],
        dtype=np.float64,
    ),
}

# Latin-balanced held-out 8 permutations vs 16 train permutations (same exact clean split as Exp 2B)
_TEST_PERMS = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (2, 1, 0, 3),
    (3, 2, 1, 0),
)
_ALL_PERMS = tuple(permutations(range(SEM_N_ACTIONS)))
_TRAIN_PERMS = tuple(p for p in _ALL_PERMS if p not in set(_TEST_PERMS))


def permute_payoff_matrix(r1: np.ndarray, role_of_action: tuple[int, int, int, int]) -> np.ndarray:
    """Return permuted 4x4 payoff matrix where action a has semantic role role_of_action[a]."""
    idx = np.asarray(role_of_action, dtype=np.int64)
    return r1[np.ix_(idx, idx)].copy()


def build_semantic_bank(
    *,
    n_sem_train: int = 4,
    eval_suite: str = "in_dist",
) -> dict[str, np.ndarray]:
    """Construct exact symmetric bank {'r1': (K,4,4), 'r2': (K,4,4), 'max_welfare': (K,)} for a suite.

    eval_suite in:
    - 'in_dist'       : first n_sem_train train families x 16 train permutations
    - 'unseen_perm'   : first n_sem_train train families x 8 held-out test permutations
    - 'unseen_params' : 4 unseen payoff parameterizations x 8 held-out test permutations
    - 'unseen_family' : 1 unseen 5th strategic family ('dual_eq_battle') x 8 held-out test permutations
    """
    train_order = ["stag_hunt", "dilemma_tempt", "chicken_anticoord", "cyclic_rps"]
    if n_sem_train not in (2, 4):
        raise ValueError(f"n_sem_train must be 2 or 4, got {n_sem_train}")

    if eval_suite == "in_dist":
        mats = [CANONICAL_TRAIN_FAMILIES[k] for k in train_order[:n_sem_train]]
        perms = _TRAIN_PERMS
    elif eval_suite == "unseen_perm":
        mats = [CANONICAL_TRAIN_FAMILIES[k] for k in train_order[:n_sem_train]]
        perms = _TEST_PERMS
    elif eval_suite == "unseen_params":
        mats = list(CANONICAL_UNSEEN_PARAMS.values())
        perms = _TEST_PERMS
    elif eval_suite == "unseen_family":
        mats = list(CANONICAL_UNSEEN_FAMILY.values())
        perms = _TEST_PERMS
    else:
        raise ValueError(f"Unsupported eval_suite: {eval_suite}")

    r1_list, max_w_list = [], []
    for base_r1 in mats:
        for p in perms:
            pm = permute_payoff_matrix(base_r1, p)
            r1_list.append(pm)
            max_w_list.append(float(np.max(pm + pm.T)))

    r1 = np.stack(r1_list, axis=0)
    r2 = np.transpose(r1, (0, 2, 1)).copy()
    return {
        "r1": r1,
        "r2": r2,
        "max_step_welfare": np.asarray(max_w_list, dtype=np.float64),
        "n_rules": np.int64(len(r1_list)),
    }


class VectorizedSemanticGameEnv:
    """Fast vectorized simulator for B parallel semantically diverse 2-player 4-action games."""

    def __init__(
        self,
        *,
        n_sem_train: int = 4,
        eval_suite: str = "in_dist",
        batch_size: int = 64,
        horizon: int = 24,
        seed: int = 0,
    ) -> None:
        self.n_sem_train = int(n_sem_train)
        self.eval_suite = str(eval_suite)
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self._rng = np.random.default_rng(seed)

        self.bank = build_semantic_bank(n_sem_train=n_sem_train, eval_suite=eval_suite)
        self.r1_bank = self.bank["r1"]
        self.r2_bank = self.bank["r2"]
        self.max_step_welfare = self.bank["max_step_welfare"]
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
