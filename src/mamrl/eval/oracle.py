"""Gate-0 Exact Two-Turn Oracle Audit.

Separates two fundamentally distinct properties of a task distribution:
1. Value of Information (VoI): does observing turn-0 reward improve 2-turn optimal value?
2. Active Probing Necessity (must_change_opening): to capture that VoI, must the agent
   choose a turn-0 action that is strictly suboptimal when reward is masked?
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mamrl.envs.active_probe_game import ActiveProbeSpec


@dataclass(frozen=True)
class OracleAuditResult:
    task_id: str
    mode: str
    masked_value: float
    informed_value: float
    value_keeping_masked_opening: float
    voi: float
    must_change_opening: bool
    loss_keeping_masked_opening: float
    masked_opt_actions: tuple[int, ...]
    informed_opt_actions: tuple[int, ...]


def exact_two_turn_audit(
    spec: ActiveProbeSpec,
    opp_dist: np.ndarray | None = None,
    tol: float = 1e-9,
) -> OracleAuditResult:
    """Compute exact 2-turn Value of Information and active-opening necessity.

    Assumes uniform prior P(theta=0) = P(theta=1) = 0.5, noiseless reward observation
    (exact payoff table), and known opponent action distribution opp_dist (defaults to uniform).
    """
    focal_R, _ = spec.payoff_matrices()  # (2, n, n)
    n = spec.n_actions
    if opp_dist is None:
        opp_dist = np.full(n, 1.0 / n, dtype=np.float64)
    opp_dist = np.asarray(opp_dist, dtype=np.float64)

    # Expected immediate payoff E_b[R(theta, a, b)] of shape (2, n)
    exp_r = focal_R @ opp_dist
    prior_r = 0.5 * (exp_r[0] + exp_r[1])  # (n,)

    # Best turn-1 value under no new info (uniform prior)
    v1_masked = float(np.max(prior_r))
    q_masked = prior_r + v1_masked  # (n,)
    masked_val = float(np.max(q_masked))
    masked_opt = tuple(int(a) for a in np.flatnonzero(q_masked >= masked_val - tol))

    # Informed 2-turn value per opening action a0:
    # Observe (b0, r0 = focal_R[theta, a0, b0]) and update exact posterior over theta in {0, 1}.
    q_informed = np.zeros(n, dtype=np.float64)
    for a0 in range(n):
        total = 0.0
        for b0 in range(n):
            p_b = float(opp_dist[b0])
            if p_b == 0.0:
                continue
            r_th0 = float(focal_R[0, a0, b0])
            r_th1 = float(focal_R[1, a0, b0])
            if abs(r_th0 - r_th1) > tol:
                # Reward uniquely identifies theta!
                v1_0 = float(np.max(exp_r[0]))
                v1_1 = float(np.max(exp_r[1]))
                total += p_b * (0.5 * (r_th0 + v1_0) + 0.5 * (r_th1 + v1_1))
            else:
                # Uninformative observation -> posterior remains 50/50
                total += p_b * (0.5 * (r_th0 + r_th1) + v1_masked)
        q_informed[a0] = total

    informed_val = float(np.max(q_informed))
    informed_opt = tuple(int(a) for a in np.flatnonzero(q_informed >= informed_val - tol))
    val_keep_masked = float(np.max(q_informed[list(masked_opt)]))
    loss_keep = max(0.0, informed_val - val_keep_masked)
    must_change = loss_keep > tol

    return OracleAuditResult(
        task_id=spec.task_id,
        mode=spec.mode,
        masked_value=masked_val,
        informed_value=informed_val,
        value_keeping_masked_opening=val_keep_masked,
        voi=informed_val - masked_val,
        must_change_opening=must_change,
        loss_keeping_masked_opening=loss_keep,
        masked_opt_actions=masked_opt,
        informed_opt_actions=informed_opt,
    )


def audit_task_suite(tasks: list[ActiveProbeSpec]) -> dict[str, float | int]:
    """Summarize Gate-0 Oracle audit across a suite of tasks."""
    results = [exact_two_turn_audit(t) for t in tasks]
    n = len(results)
    if n == 0:
        raise ValueError("Task suite must be non-empty")
    return {
        "n_tasks": n,
        "positive_voi_rate": float(np.mean([r.voi > 1e-9 for r in results])),
        "must_change_opening_rate": float(np.mean([r.must_change_opening for r in results])),
        "mean_voi": float(np.mean([r.voi for r in results])),
        "mean_loss_keeping_masked_opening": float(np.mean([r.loss_keeping_masked_opening for r in results])),
    }
