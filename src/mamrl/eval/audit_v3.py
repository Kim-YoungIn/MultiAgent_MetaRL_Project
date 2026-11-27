"""SPEC v3.0 Evaluation, Layer-1/Layer-3 Decomposition, Behavioral Diagnostics & Hierarchical Bootstrap."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np

from mamrl.agents.baselines_v3 import BatchPolicyV3
from mamrl.envs.delayed_response import (
    ACTION_NAMES,
    ACT_P,
    ACT_P0,
    ACT_P1,
    ACT_S,
    ACT_X00,
    N_ACTIONS,
    N_TYPES,
    VectorizedDelayedResponseEnv,
)
from mamrl.eval.exact_dp import ExactDelayedDP


@dataclass
class TrajectoryBundle:
    regime: str
    policy_name: str
    theta: np.ndarray          # (E,) int64
    actions: np.ndarray        # (E, T) int64
    opp_actions: np.ndarray    # (E, T) int64
    focal_rewards: np.ndarray  # (E, T) float64
    opp_rewards: np.ndarray    # (E, T) float64
    hidden_states: np.ndarray | None = None  # (E, T, H) float32


def rollout_policy_v3(
    policy: BatchPolicyV3,
    *,
    regime: str,
    n_episodes: int = 512,
    horizon: int = 24,
    seed: int = 0,
    record_hidden: bool = False,
) -> TrajectoryBundle:
    """Evaluate a BatchPolicyV3 on exact balanced theta episodes (n_episodes divisible by 4)."""
    if n_episodes % N_TYPES != 0:
        raise ValueError(f"n_episodes ({n_episodes}) must be divisible by {N_TYPES} for balanced types.")

    thetas = np.tile(np.arange(N_TYPES, dtype=np.int64), n_episodes // N_TYPES)
    env = VectorizedDelayedResponseEnv(regime=regime, batch_size=n_episodes, horizon=horizon, seed=seed)
    obs = env.reset(theta=thetas)
    policy.reset(batch_size=n_episodes, seed=seed + 13, true_theta=thetas if policy.is_privileged else None)

    actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    opp_actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    focal_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    opp_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    hidden_list: list[np.ndarray] | None = [] if record_hidden else None

    for t in range(horizon):
        a = policy.act(obs, step_t=t)
        if hidden_list is not None:
            h_cur = policy.last_hidden()
            if h_cur is not None:
                hidden_list.append(h_cur.astype(np.float32))
        obs, rf, ro, y = env.step(a)
        actions[:, t] = a
        opp_actions[:, t] = y
        focal_rewards[:, t] = rf
        opp_rewards[:, t] = ro

    hidden_arr = np.stack(hidden_list, axis=1) if (hidden_list is not None and len(hidden_list) == horizon) else None
    return TrajectoryBundle(
        regime=regime,
        policy_name=policy.name,
        theta=thetas,
        actions=actions,
        opp_actions=opp_actions,
        focal_rewards=focal_rewards,
        opp_rewards=opp_rewards,
        hidden_states=hidden_arr,
    )


def _action_category_dist(actions_slice: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    """Compute shares of S, P, P0, P1, X_match, X_mismatch across (E, L) action slice."""
    n_total = float(actions_slice.size)
    if n_total == 0:
        return {k: 0.0 for k in ("S", "P", "P0", "P1", "X_match", "X_mismatch")}
    match_act = (ACT_X00 + theta)[:, None]
    is_x = actions_slice >= ACT_X00
    is_x_match = actions_slice == match_act
    return {
        "S": float(np.sum(actions_slice == ACT_S) / n_total),
        "P": float(np.sum(actions_slice == ACT_P) / n_total),
        "P0": float(np.sum(actions_slice == ACT_P0) / n_total),
        "P1": float(np.sum(actions_slice == ACT_P1) / n_total),
        "X_match": float(np.sum(is_x_match) / n_total),
        "X_mismatch": float(np.sum(is_x & (~is_x_match)) / n_total),
    }


def _time_to_threshold(rewards: np.ndarray, threshold: float = 0.8, window: int = 3) -> tuple[float, float]:
    n_ep, t_len = rewards.shape
    reached: list[int] = []
    censored = 0
    for i in range(n_ep):
        found = None
        for t in range(t_len - window + 1):
            if float(np.mean(rewards[i, t : t + window])) >= threshold:
                found = t + 1
                break
        if found is None:
            censored += 1
        else:
            reached.append(found)
    return (float(np.mean(reached)) if reached else float(t_len)), float(censored / max(1, n_ep))


def _disjoint_linear_probe_accuracy(hidden_states: np.ndarray, theta: np.ndarray, max_turns: int = 8) -> dict[str, float]:
    """Diagnostic-only ridge linear probe from h_t -> theta on disjoint 50/50 train/test episode split."""
    n_ep, t_len, h_dim = hidden_states.shape
    half = n_ep // 2
    if half < 8:
        return {}
    # Stratified split (even indices vs odd indices preserve exact balanced theta!)
    tr_idx = np.arange(0, n_ep, 2)
    te_idx = np.arange(1, n_ep, 2)
    y_tr = np.eye(N_TYPES, dtype=np.float64)[theta[tr_idx]]
    y_te = theta[te_idx]

    accs: dict[str, float] = {}
    reg = 1e-2 * np.eye(h_dim + 1, dtype=np.float64)
    for t in range(min(max_turns, t_len)):
        x_tr = np.hstack([hidden_states[tr_idx, t, :].astype(np.float64), np.ones((len(tr_idx), 1))])
        x_te = np.hstack([hidden_states[te_idx, t, :].astype(np.float64), np.ones((len(te_idx), 1))])
        w = np.linalg.solve(x_tr.T @ x_tr + reg, x_tr.T @ y_tr)
        preds = np.argmax(x_te @ w, axis=-1)
        accs[f"t{t + 1}"] = float(np.mean(preds == y_te))
    return accs


def compute_episode_audit(
    bundle: TrajectoryBundle,
    *,
    audit_ks: tuple[int, ...] = (2, 4, 8),
    primary_k: int = 4,
) -> dict:
    """Compute all 4 SPEC v3.0 measurement axes from a TrajectoryBundle."""
    n_ep, horizon = bundle.actions.shape
    dp = ExactDelayedDP(regime=bundle.regime, horizon=horizon)
    bayes_opt_total = float(dp.adaptive_value(horizon))

    ep_total = bundle.focal_rewards.sum(axis=1)
    ep_opp_total = bundle.opp_rewards.sum(axis=1)
    q25, q50, q75 = np.percentile(ep_total, [25, 50, 75])
    ttt_mean, ttt_cens = _time_to_threshold(bundle.focal_rewards, threshold=0.8, window=3)

    # 1. End-to-End Distribution & Speed
    by_theta = {}
    for th in range(N_TYPES):
        mask = bundle.theta == th
        vals = ep_total[mask]
        by_theta[str(th)] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }

    end_to_end = {
        "total_mean": float(np.mean(ep_total)),
        "total_median": float(q50),
        "total_iqr": float(q75 - q25),
        "total_std": float(np.std(ep_total)),
        "bayes_opt_ceiling": bayes_opt_total,
        "bayes_opt_gap": bayes_opt_total - float(np.mean(ep_total)),
        "opp_total_mean": float(np.mean(ep_opp_total)),
        "early_auc_1_4": float(np.mean(bundle.focal_rewards[:, : min(4, horizon)])),
        "mid_auc_5_12": float(np.mean(bundle.focal_rewards[:, 4 : min(12, horizon)])) if horizon > 4 else 0.0,
        "late_auc_13_end": float(np.mean(bundle.focal_rewards[:, 12:])) if horizon > 12 else 0.0,
        "ttt_mean": ttt_mean,
        "ttt_censored_rate": ttt_cens,
        "turn_curve": [float(x) for x in np.mean(bundle.focal_rewards, axis=0)],
        "by_theta": by_theta,
    }

    # 2. Behavioral & Action Repetition Diagnostics
    k_p = min(primary_k, horizon)
    act_exp = bundle.actions[:, :k_p]
    act_expt = bundle.actions[:, k_p:]
    rep_exp = float(np.mean(act_exp[:, 1:] == act_exp[:, :-1])) if k_p > 1 else 0.0
    rep_expt = float(np.mean(bundle.actions[:, k_p:] == bundle.actions[:, k_p - 1 : -1])) if horizon > k_p else 0.0
    rep_all = float(np.mean(bundle.actions[:, 1:] == bundle.actions[:, :-1])) if horizon > 1 else 0.0
    uniq_exp = float(np.mean([len(set(row.tolist())) for row in act_exp]))
    open_probe_rate = float(np.mean((bundle.actions[:, 0] >= ACT_P) & (bundle.actions[:, 0] <= ACT_P1)))

    # Top-5 K-turn prefix sequences
    seq_counts = Counter(
        "-".join(ACTION_NAMES[int(a)] for a in row) for row in act_exp
    )
    top_prefixes = [
        {"sequence": seq, "share": float(cnt / n_ep)}
        for seq, cnt in seq_counts.most_common(5)
    ]

    # Feedback-branching sensitivity (TV distance of turn-3 action distribution given y_2=0 vs y_2=1)
    y2 = bundle.opp_actions[:, 1] if horizon >= 3 else np.zeros(n_ep, dtype=np.int64)
    a3 = bundle.actions[:, 2] if horizon >= 3 else np.zeros(n_ep, dtype=np.int64)
    m0 = y2 == 0
    m1 = y2 == 1
    if int(np.sum(m0)) > 0 and int(np.sum(m1)) > 0:
        p0 = np.bincount(a3[m0], minlength=N_ACTIONS) / float(np.sum(m0))
        p1 = np.bincount(a3[m1], minlength=N_ACTIONS) / float(np.sum(m1))
        k4_branching_tv = float(0.5 * np.sum(np.abs(p0 - p1)))
    else:
        k4_branching_tv = 0.0

    behavior = {
        "action_dist_explore": _action_category_dist(act_exp, bundle.theta),
        "action_dist_exploit": _action_category_dist(act_expt, bundle.theta),
        "repeat_rate_explore": rep_exp,
        "repeat_rate_exploit": rep_expt,
        "repeat_rate_all": rep_all,
        "unique_actions_explore": uniq_exp,
        "opening_probe_rate": open_probe_rate,
        "k4_branching_tv": k4_branching_tv,
        "k4_branch_counts": {"y2_eq_0": int(np.sum(m0)), "y2_eq_1": int(np.sum(m1))},
        "top_prefix_sequences": top_prefixes,
    }

    # 3. Layer 1 (Exact Bayes Decision-Value Audit at K in audit_ks) + Layer 3 (Non-OOD Live Gap)
    layer1: dict[str, dict[str, float]] = {}
    ep_arrays: dict[str, np.ndarray] = {
        "total_return": ep_total,
        "opp_total_return": ep_opp_total,
    }

    for k in audit_ks:
        if k >= horizon:
            continue
        rem = horizon - k
        c_vals = np.zeros(n_ep, dtype=np.float64)
        v_vals = np.zeros(n_ep, dtype=np.float64)
        gap_vals = np.zeros(n_ep, dtype=np.float64)
        pref_r = np.zeros(n_ep, dtype=np.float64)
        net_c = np.zeros(n_ep, dtype=np.float64)
        net_v = np.zeros(n_ep, dtype=np.float64)
        ent_vals = np.zeros(n_ep, dtype=np.float64)
        id_vals = np.zeros(n_ep, dtype=np.float64)
        live_gap_vals = np.zeros(n_ep, dtype=np.float64)

        for i in range(n_ep):
            # Reconstruct exact (q_{t-1}, y_t) pairs for t = 0..k-1
            traj_qy: list[tuple[int, int]] = []
            q_prev = ACT_S
            for t in range(k):
                traj_qy.append((q_prev, int(bundle.opp_actions[i, t])))
                q_prev = int(bundle.actions[i, t])
            pr = float(np.sum(bundle.focal_rewards[i, :k]))
            post_r = float(np.sum(bundle.focal_rewards[i, k:]))
            aud = dp.audit_prefix_layer1(traj_qy, pr, remaining_turns=rem, q_current=q_prev)
            c_vals[i] = float(aud["restored_commitment_value"])
            v_vals[i] = float(aud["restored_adaptive_value"])
            gap_vals[i] = float(aud["restored_opportunity_gap"])
            pref_r[i] = pr
            net_c[i] = float(aud["net_restored_commitment_return"])
            net_v[i] = float(aud["net_restored_adaptive_return"])
            ent_vals[i] = float(aud["posterior_entropy_bits"])
            id_vals[i] = float(aud["rule_identified"])
            live_gap_vals[i] = float(aud["live_bayes_continuation_value"]) - post_r

        layer1[f"K={k}"] = {
            "restored_commitment_mean": float(np.mean(c_vals)),
            "restored_commitment_median": float(np.median(c_vals)),
            "restored_adaptive_mean": float(np.mean(v_vals)),
            "restored_adaptive_median": float(np.median(v_vals)),
            "restored_opportunity_gap_mean": float(np.mean(gap_vals)),
            "prefix_reward_mean": float(np.mean(pref_r)),
            "net_restored_commitment_mean": float(np.mean(net_c)),
            "net_restored_adaptive_mean": float(np.mean(net_v)),
            "posterior_entropy_bits_mean": float(np.mean(ent_vals)),
            "rule_identified_rate": float(np.mean(id_vals)),
            "live_continuation_gap_mean": float(np.mean(live_gap_vals)),
            "live_continuation_gap_median": float(np.median(live_gap_vals)),
        }
        if k == primary_k:
            ep_arrays["layer1_commitment_k4"] = c_vals
            ep_arrays["layer1_adaptive_k4"] = v_vals
            ep_arrays["layer1_net_adaptive_k4"] = net_v
            ep_arrays["layer3_live_gap_k4"] = live_gap_vals

    layer3 = {
        "live_continuation_gap_k4_mean": layer1[f"K={primary_k}"]["live_continuation_gap_mean"],
        "live_continuation_gap_k4_median": layer1[f"K={primary_k}"]["live_continuation_gap_median"],
        "hidden_state_linear_probe_acc": (
            _disjoint_linear_probe_accuracy(bundle.hidden_states, bundle.theta, max_turns=8)
            if bundle.hidden_states is not None
            else {}
        ),
    }

    return {
        "end_to_end": end_to_end,
        "behavior": behavior,
        "layer1": layer1,
        "layer3": layer3,
        "episode_arrays": ep_arrays,
    }


def hierarchical_paired_bootstrap(
    values_a_by_seed: np.ndarray,
    values_b_by_seed: np.ndarray,
    *,
    practical_threshold: float,
    n_boot: int = 10000,
    seed: int = 0,
    tol: float = 1e-9,
) -> dict[str, float | str]:
    """Paired hierarchical bootstrap with training seed as independent unit (SPEC v3.0 §5/§7).

    Inputs have shape (n_seeds, n_episodes). Reports mean diff, median diff, win/tie/loss
    rates (ties separate), 95% CI, and 4-way pre-registered verdict.
    """
    a = np.asarray(values_a_by_seed, dtype=np.float64)
    b = np.asarray(values_b_by_seed, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Expected matching 2D (n_seeds, n_episodes) arrays, got {a.shape} vs {b.shape}")

    diff = a - b  # (S, E)
    n_seeds, n_ep = diff.shape
    seed_means = np.mean(diff, axis=1)
    point_mean = float(np.mean(seed_means))
    point_median = float(np.median(diff))
    win_rate = float(np.mean(diff > tol))
    loss_rate = float(np.mean(diff < -tol))
    tie_rate = float(np.mean(np.abs(diff) <= tol))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    if n_seeds == 1:
        # Single-seed fallback (dev only): episode-level bootstrap
        ep_idx = rng.integers(0, n_ep, size=(n_boot, n_ep))
        boot_means = np.mean(diff[0, ep_idx], axis=1)
    else:
        # Hierarchical bootstrap: resample seeds with replacement, then episodes within each seed
        seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
        for i in range(n_boot):
            ep_idx = rng.integers(0, n_ep, size=(n_seeds, n_ep))
            sampled = diff[seed_idx[i][:, None], ep_idx]
            boot_means[i] = float(np.mean(sampled))

    ci_low, ci_high = [float(x) for x in np.percentile(boot_means, [2.5, 97.5])]
    d = float(practical_threshold)
    if ci_low > d:
        verdict = "SUPPORTED_POS"
    elif ci_high < -d:
        verdict = "SUPPORTED_NEG"
    elif ci_low >= -d and ci_high <= d:
        verdict = "RESOLVED_NULL"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n_seeds": int(n_seeds),
        "n_episodes_per_seed": int(n_ep),
        "diff_mean": point_mean,
        "diff_median": point_median,
        "seed_diff_sd": float(np.std(seed_means, ddof=1)) if n_seeds > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "loss_rate": loss_rate,
        "practical_threshold": d,
        "verdict": verdict,
    }
