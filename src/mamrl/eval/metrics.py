"""Distribution-aware adaptation metrics (Mean, Median, Win-Rate, Shaping, TTT)."""

from __future__ import annotations

import numpy as np


def _time_to_threshold(
    curves: np.ndarray,
    threshold: float,
    window: int = 3,
) -> tuple[float, float]:
    """Return (mean_ttt_among_reached, censored_rate) using a moving-average window."""
    # curves: (n_episodes, assess_steps)
    n_ep, h = curves.shape
    reached: list[int] = []
    censored = 0
    for i in range(n_ep):
        found = None
        for t in range(h - window + 1):
            if float(np.mean(curves[i, t : t + window])) >= threshold:
                found = t + 1
                break
        if found is None:
            censored += 1
        else:
            reached.append(found)
    mean_ttt = float(np.mean(reached)) if reached else float(h)
    return mean_ttt, float(censored / max(1, n_ep))


def summarize_method(
    total_returns: np.ndarray,
    explore_returns: np.ndarray,
    assess_returns: np.ndarray,
    counterpart_returns: np.ndarray,
    assess_curves: np.ndarray,
    early_turns: int = 8,
    threshold: float = 0.60,
) -> dict[str, float]:
    """Compute mandatory distribution-aware metrics for one method."""
    q25, q50, q75 = np.percentile(total_returns, [25, 50, 75])
    early_auc = float(np.mean(assess_curves[:, :early_turns])) if assess_curves.shape[1] > 0 else 0.0
    ttt_mean, ttt_censored = _time_to_threshold(assess_curves, threshold=threshold)
    return {
        "total_mean": float(np.mean(total_returns)),
        "total_median": float(q50),
        "total_iqr": float(q75 - q25),
        "explore_mean": float(np.mean(explore_returns)),
        "assess_mean": float(np.mean(assess_returns)),
        "early_auc": early_auc,
        "counterpart_mean": float(np.mean(counterpart_returns)),
        "ttt_mean": ttt_mean,
        "ttt_censored_rate": ttt_censored,
    }


def summarize_paired_contrast(
    method_a: np.ndarray,
    method_b: np.ndarray,
    counterpart_a: np.ndarray,
    counterpart_b: np.ndarray,
) -> dict[str, float]:
    """Paired game-level comparison (A - B) exposing mean/median skew and win rate."""
    diff = method_a - method_b
    opp_diff = counterpart_a - counterpart_b
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_median": float(np.median(diff)),
        "win_rate": float(np.mean(diff > 1e-9)),
        "loss_rate": float(np.mean(diff < -1e-9)),
        "counterpart_diff_mean": float(np.mean(opp_diff)),
        "both_up_rate": float(np.mean((diff > 1e-9) & (opp_diff > 1e-9))),
        "both_down_rate": float(np.mean((diff < -1e-9) & (opp_diff < -1e-9))),
    }
