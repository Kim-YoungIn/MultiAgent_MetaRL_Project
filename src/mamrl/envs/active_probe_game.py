"""Oracle-verifiable Active-Probe Multi-Agent Repeated Game.

Supports two matched regimes sharing identical action count and exploitation payoffs:
- mode='active': SAFE gives an uninformative constant payoff; costly PROBE is strictly
  required to identify hidden regime theta (Oracle must_change_opening = True).
- mode='passive': SAFE carries a small diagnostic signature with the same prior mean;
  observing reward has high VoI without needing to change the opening action.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import Observation, StepResult

ROLE_SAFE = 0
ROLE_EXPLOIT_0 = 1
ROLE_EXPLOIT_1 = 2
ROLE_PROBE = 3


@dataclass(frozen=True)
class ActiveProbeSpec:
    """Immutable specification for one two-player Active-Probe game instance."""

    task_id: str
    mode: str  # "active" or "passive"
    theta: int  # Hidden regime in {0, 1}
    n_actions: int
    role_to_action: tuple[int, int, int, int]  # (safe, exploit_0, exploit_1, probe)
    r_safe: float
    passive_signal: float
    probe_cost: float
    probe_signal: float
    exploit_hit_coord: float
    exploit_hit_solo: float
    exploit_miss: float
    opp_alignment: int  # +1 cooperative, -1 competitive
    reward_noise_std: float = 0.0

    def payoff_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact (focal_R, opp_R) of shape (2, n_actions, n_actions) indexed by [theta, a, b]."""
        n = self.n_actions
        a_safe, a_e0, a_e1, a_probe = self.role_to_action
        focal = np.zeros((2, n, n), dtype=np.float64)
        opp = np.zeros((2, n, n), dtype=np.float64)

        for th in (0, 1):
            # SAFE row
            if self.mode == "passive":
                safe_val = self.r_safe + (self.passive_signal if th == 0 else -self.passive_signal)
            else:
                safe_val = self.r_safe
            focal[th, a_safe, :] = safe_val
            opp[th, a_safe, :] = self.r_safe

            # EXPLOIT rows (interaction-dependent payoffs)
            good_a, bad_a = (a_e0, a_e1) if th == 0 else (a_e1, a_e0)
            for b in range(n):
                is_coord = b in (good_a, a_safe)
                focal[th, good_a, b] = self.exploit_hit_coord if is_coord else self.exploit_hit_solo
                focal[th, bad_a, b] = self.exploit_miss

                if self.opp_alignment >= 0:
                    # Cooperative counterpart benefits from coordinating with the good action
                    opp[th, good_a, b] = 0.70 if b == good_a else (0.35 if b == a_safe else 0.05)
                    opp[th, bad_a, b] = 0.25 if b == a_safe else -0.15
                else:
                    # Competitive counterpart benefits from countering / avoiding focal's good action
                    opp[th, good_a, b] = -0.25 if b == good_a else 0.55
                    opp[th, bad_a, b] = 0.45 if b == good_a else 0.15

            # PROBE row: costly immediate action emitting a theta- and opponent-modulated diagnostic signal
            sign = 1.0 if th == 0 else -1.0
            for b in range(n):
                strength = 1.0 if b in (good_a, a_safe) else 0.7
                focal[th, a_probe, b] = -self.probe_cost + sign * self.probe_signal * strength
                opp[th, a_probe, b] = 0.20 if b == a_safe else 0.10

        return focal, opp

    def action_to_role(self) -> dict[int, int]:
        return {action: role for role, action in enumerate(self.role_to_action)}


class ActiveProbeGame:
    """Simultaneous two-player game environment executing an ActiveProbeSpec."""

    def __init__(self, spec: ActiveProbeSpec, seed: int = 0) -> None:
        self.spec = spec
        self.focal_R, self.opp_R = spec.payoff_matrices()
        self._rng = np.random.default_rng(seed)
        self.turn = 0
        self._prev_focal = -1
        self._prev_opp = -1
        self._prev_rf = 0.0
        self._prev_ro = 0.0

    @property
    def n_actions(self) -> int:
        return self.spec.n_actions

    def reset(self, seed: int | None = None) -> tuple[Observation, Observation]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.turn = 0
        self._prev_focal = -1
        self._prev_opp = -1
        self._prev_rf = 0.0
        self._prev_ro = 0.0
        obs_f = Observation(-1, -1, 0.0, True, 0)
        obs_o = Observation(-1, -1, 0.0, True, 0)
        return obs_f, obs_o

    def step(self, focal_action: int, opp_action: int) -> StepResult:
        if not (0 <= focal_action < self.n_actions and 0 <= opp_action < self.n_actions):
            raise ValueError(f"Actions out of range: focal={focal_action}, opp={opp_action}")

        th = self.spec.theta
        rf = float(self.focal_R[th, focal_action, opp_action])
        ro = float(self.opp_R[th, focal_action, opp_action])
        if self.spec.reward_noise_std > 0.0:
            rf += float(self._rng.normal(0.0, self.spec.reward_noise_std))
            ro += float(self._rng.normal(0.0, self.spec.reward_noise_std))

        self.turn += 1
        self._prev_focal = focal_action
        self._prev_opp = opp_action
        self._prev_rf = rf
        self._prev_ro = ro

        focal_obs = Observation(focal_action, opp_action, rf, False, self.turn)
        opp_obs = Observation(opp_action, focal_action, ro, False, self.turn)
        return StepResult(focal_action, opp_action, rf, ro, focal_obs, opp_obs, self.turn)


def generate_task_suite(
    n_tasks: int,
    *,
    mode: str = "active",
    diversity: str = "low",
    seed: int = 0,
    reward_noise_std: float = 0.0,
) -> list[ActiveProbeSpec]:
    """Generate a balanced suite of ActiveProbeSpec tasks.

    Diversity levels (aligned with H1):
    - 'low': fixed role indices (0=SAFE, 1=E0, 2=E1, 3=PROBE), narrow payoff jitter, cooperative opponent.
    - 'medium': moderate payoff & cost jitter + mixed cooperative/competitive opponents (probe index fixed).
    - 'high': full action-label permutations + wide payoff jitter + mixed cooperative/competitive opponents.
    """
    if mode not in ("active", "passive"):
        raise ValueError(f"Unsupported mode: {mode}")
    if diversity not in ("low", "medium", "high"):
        raise ValueError(f"Unsupported diversity: {diversity}")

    rng = np.random.default_rng(seed)
    tasks: list[ActiveProbeSpec] = []
    for i in range(n_tasks):
        theta = i % 2  # Exactly balanced latent regimes
        if diversity == "low":
            roles = (0, 1, 2, 3)
            r_safe = 0.25 + float(rng.uniform(-0.02, 0.02))
            probe_cost = 0.15 + float(rng.uniform(-0.01, 0.01))
            hit_coord = 1.00 + float(rng.uniform(-0.04, 0.04))
            hit_solo = 0.80 + float(rng.uniform(-0.04, 0.04))
            miss = -0.70 + float(rng.uniform(-0.04, 0.04))
            opp_align = 1
        elif diversity == "medium":
            roles = (0, 1, 2, 3)
            r_safe = 0.25 + float(rng.uniform(-0.04, 0.04))
            probe_cost = 0.15 + float(rng.uniform(-0.03, 0.03))
            hit_coord = 1.00 + float(rng.uniform(-0.08, 0.08))
            hit_solo = 0.80 + float(rng.uniform(-0.06, 0.06))
            miss = -0.70 + float(rng.uniform(-0.08, 0.08))
            opp_align = 1 if (i // 2) % 2 == 0 else -1
        else:
            perm = tuple(int(x) for x in rng.permutation(4))
            roles = (perm[0], perm[1], perm[2], perm[3])
            r_safe = 0.25 + float(rng.uniform(-0.04, 0.04))
            probe_cost = 0.15 + float(rng.uniform(-0.03, 0.03))
            hit_coord = 1.00 + float(rng.uniform(-0.08, 0.08))
            hit_solo = 0.80 + float(rng.uniform(-0.06, 0.06))
            miss = -0.70 + float(rng.uniform(-0.08, 0.08))
            opp_align = 1 if (i // 2) % 2 == 0 else -1

        tasks.append(
            ActiveProbeSpec(
                task_id=f"{mode}_{diversity}_{seed}_{i:04d}",
                mode=mode,
                theta=theta,
                n_actions=4,
                role_to_action=roles,
                r_safe=round(r_safe, 4),
                passive_signal=0.04,
                probe_cost=round(probe_cost, 4),
                probe_signal=0.10,
                exploit_hit_coord=round(hit_coord, 4),
                exploit_hit_solo=round(hit_solo, 4),
                exploit_miss=round(miss, 4),
                opp_alignment=opp_align,
                reward_noise_std=reward_noise_std,
            )
        )
    return tasks
