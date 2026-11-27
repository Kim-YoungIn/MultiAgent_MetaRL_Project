"""SPEC v3.0 Legacy Bridge A (Exp 3A): Evaluation-Only Reliability & Dynamics Stress Test.

Tests two strictly separated perturbations across d in {0.0, 0.3, 0.6} on EXISTING sealed
5-seed checkpoints (zero retraining, zero environment tuning):

1. Track A — 'r3_obs_reliability':
   Discrete symmetric observation corruption on y_obs in R3 with true payoff mechanism
   r_F(q, a, y_true) unchanged (matches independent review §2 R2 definition).
   Evaluates fixed-trained Exp 1 policies, online-trained Exp 2A policies, structural models,
   online baselines, and a recomputed shifted-likelihood Bayesian diagnostic reference
   ('shifted_bayes_ref', privileged=True) — NEVER using fixed-model zs_bayes_optimal as ceiling.

2. Track B — 'coop_trap_dynamics':
   Exact legacy 'dynamics_transfer' analogue on strictly unseen coop_trap rules:
   when both players coordinate on hidden (c*, c*), the joint cooperative outcome slips
   with probability d to (-0.2, -0.2) instead of (+1.2, +1.2).
   Tests whether meta_drqn_aux_yr's welfare advantage over Non-Meta and meta_vanilla survives
   or erodes as cooperative transition reliability drops.

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_reliability_sweep --config configs/exp3a_reliability_sweep.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch

from mamrl.agents.baselines_v3 import (
    ArmUCBPolicyV3,
    BatchPolicyV3,
    ContextualUCBPolicyV3,
    GreedyPolicyV3,
    RandomPolicyV3,
    StructuralModelPolicyV3,
)
from mamrl.agents.drqn_rl2 import NeuralDRQNPolicyV3, RecurrentQNet
from mamrl.agents.opponents_v3 import OnlineOpponentConfig, make_online_opponent
from mamrl.agents.ppo_rl2 import OBS_DIM
from mamrl.envs.delayed_response import (
    ACT_S,
    ACT_X00,
    N_ACTIONS,
    N_OPP_SYMBOLS,
    N_TYPES,
    VectorizedDelayedResponseEnv,
    build_focal_reward_table,
    build_response_table,
)
from mamrl.envs.symmetric_ma import SYM_N_ACTIONS, VectorizedSymmetricGameEnv
from mamrl.eval.audit_v3 import hierarchical_paired_bootstrap
from mamrl.study_exp2b import (
    META_METHODS,
    SYM_OBS_DIM,
    SymContextualUCBPolicy,
    SymNeuralDRQNPolicy,
    SymPolicy,
    SymQLearningPolicy,
    SymUCBPolicy,
    SymmetricRecurrentQNet,
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class ShiftedChannelBayesRefV3(BatchPolicyV3):
    """Privileged shifted-channel Bayesian diagnostic reference for R3 observation noise d.

    Uses exact shifted likelihood P_d(y_obs | q, theta) = (1-d)*1[y=f_theta(q)] + (d/3)*1[y!=f_theta(q)]
    plus exact reward likelihood 1[r == r_F(q, a, f_theta(q))] and 2-step belief lookahead.
    Flagged is_privileged=True (diagnostic reference, never compared as an unprivileged baseline).
    """

    name = "shifted_bayes_ref"
    is_privileged = True

    def __init__(self, regime: str = "R3", horizon: int = 24, noise_d: float = 0.0) -> None:
        self.regime = regime
        self.horizon = int(horizon)
        self.noise_d = float(noise_d)
        self._f_table = build_response_table(regime)  # (4, 8)
        self._r_focal = build_focal_reward_table()    # (8, 8, 4)

        # True deterministic reward R_true[theta, q, a]
        self._r_true = np.zeros((N_TYPES, N_ACTIONS, N_ACTIONS), dtype=np.float64)
        for th in range(N_TYPES):
            for q in range(N_ACTIONS):
                y_true = int(self._f_table[th, q])
                self._r_true[th, q, :] = self._r_focal[q, :, y_true]

        # Observation channel P_obs[theta, q, y_obs]
        self._p_obs = np.full((N_TYPES, N_ACTIONS, N_OPP_SYMBOLS), self.noise_d / (N_OPP_SYMBOLS - 1), dtype=np.float64)
        for th in range(N_TYPES):
            for q in range(N_ACTIONS):
                self._p_obs[th, q, int(self._f_table[th, q])] = 1.0 - self.noise_d

        # Precompute exact per-theta optimal values & cross-theta policy returns (same fast DP as StructuralModel)
        self._v_th = np.zeros((N_TYPES, self.horizon + 1, N_ACTIONS), dtype=np.float64)
        for th in range(N_TYPES):
            for h in range(1, self.horizon + 1):
                self._v_th[th, h] = np.max(self._r_true[th] + self._v_th[th, h - 1][None, :], axis=-1)

        self._policy_ret = np.zeros((N_TYPES, self.horizon + 1, N_ACTIONS, N_TYPES), dtype=np.float64)
        for target_th in range(N_TYPES):
            for h in range(1, self.horizon + 1):
                a_stars = np.argmax(self._r_true[target_th] + self._v_th[target_th, h - 1][None, :], axis=-1)
                for q in range(N_ACTIONS):
                    a_s = int(a_stars[q])
                    self._policy_ret[target_th, h, q] = (
                        self._r_true[:, q, a_s] + self._policy_ret[target_th, h - 1, a_s]
                    )

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._batch = batch_size
        self._belief = np.full((batch_size, N_TYPES), 1.0 / N_TYPES, dtype=np.float64)
        self._prev_q = np.full(batch_size, ACT_S, dtype=np.int64)
        self._last_a = np.zeros(batch_size, dtype=np.int64)

    def _lookahead_action(self, q: int, belief: np.ndarray, rem: int) -> int:
        best_a = ACT_S
        best_val = -1e18
        for a in range(N_ACTIONS):
            imm = float(np.sum(belief * self._r_true[:, q, a]))
            if rem <= 1:
                val = imm
            else:
                # Expected value over joint (y_obs, r_true) observed at this step
                cont = 0.0
                for y_obs in range(N_OPP_SYMBOLS):
                    # Weight per theta for observing y_obs
                    w_y = belief * self._p_obs[:, q, y_obs]
                    p_y = float(np.sum(w_y))
                    if p_y < 1e-12:
                        continue
                    # Group by distinct realized reward values under (q, a)
                    r_vals = self._r_true[:, q, a]
                    for r_u in np.unique(r_vals):
                        m_r = np.abs(r_vals - r_u) < 1e-9
                        w_yr = np.where(m_r, w_y, 0.0)
                        p_yr = float(np.sum(w_yr))
                        if p_yr < 1e-12:
                            continue
                        b1 = w_yr / p_yr
                        # 1-more-step lookahead or immediate commitment
                        best_a2 = -1e18
                        for a2 in range(N_ACTIONS):
                            imm2 = float(np.sum(b1 * self._r_true[:, a, a2]))
                            if rem <= 2:
                                v2 = imm2
                            else:
                                c2 = 0.0
                                for y2 in range(N_OPP_SYMBOLS):
                                    w2 = b1 * self._p_obs[:, a, y2]
                                    p2 = float(np.sum(w2))
                                    if p2 < 1e-12:
                                        continue
                                    b2 = w2 / p2
                                    v_commit = max(
                                        float(np.sum(b2 * self._policy_ret[t_th, rem - 2, a2]))
                                        for t_th in range(N_TYPES)
                                    )
                                    c2 += p2 * v_commit
                                v2 = imm2 + c2
                            if v2 > best_a2:
                                best_a2 = v2
                        cont += p_yr * best_a2
                val = imm + cont
            if val > best_val + 1e-12:
                best_val = val
                best_a = a
        return best_a

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        q_cur = obs["prev_self_action"]
        if step_t > 0:
            prev_y = obs["prev_opp_action"]
            prev_r = obs["prev_self_reward"]
            lik_y = self._p_obs[:, self._prev_q, prev_y].T  # (B, 4)
            r_expected = self._r_true[:, self._prev_q, self._last_a].T  # (B, 4)
            lik_r = (np.abs(r_expected - prev_r[:, None]) < 1e-6).astype(np.float64)
            post = self._belief * lik_y * lik_r
            denom = np.sum(post, axis=1, keepdims=True)
            self._belief = np.where(denom > 1e-12, post / np.maximum(denom, 1e-12), self._belief)
        self._prev_q = q_cur.copy()

        rem = self.horizon - step_t
        # Exact vectorized 2-step belief lookahead (degree-1 homogeneity -> ~200x faster than Python loops)
        r_q = np.transpose(self._r_true[:, q_cur, :], (1, 0, 2))  # (B, M, A)
        imm1 = np.einsum("bm,bma->ba", self._belief, r_q)  # (B, A)
        if rem <= 1:
            acts = np.argmax(imm1, axis=-1).astype(np.int64)
            self._last_a = acts.copy()
            return acts

        p_obs_q = np.transpose(self._p_obs[:, q_cur, :], (1, 2, 0))  # (B, Y, M)
        w1 = self._belief[:, None, :] * p_obs_q  # (B, Y, M)
        imm2 = np.einsum("bym,maq->byaq", w1, self._r_true)  # (B, Y, A1, A2)
        if rem <= 2:
            v2 = imm2
        else:
            w2 = w1[:, :, None, None, :] * np.transpose(self._p_obs, (1, 2, 0))[None, None, :, :, :]
            cret = self._policy_ret[:, rem - 2, :, :]  # (M_t, A2, M)
            val_c = np.einsum("byazm,tqm->byazqt", w2, cret)
            c2 = np.sum(np.max(val_c, axis=-1), axis=3)
            v2 = imm2 + c2

        cont1 = np.sum(np.max(v2, axis=-1), axis=1)  # (B, A1)
        # Account for exact reward reveal when a1 == q_cur >= ACT_X00 (splits match vs mismatch theta!)
        in_x = q_cur >= ACT_X00
        if np.any(in_x) and rem >= 2:
            # For a1 = q_cur in X: if true theta == q-4 (prob b[th]), agent knows theta with certainty!
            idx_x = np.where(in_x)[0]
            qx = q_cur[idx_x]
            th_match = qx - ACT_X00
            p_hit = self._belief[idx_x, th_match]
            v_hit = self._v_th[th_match, rem - 1, qx]
            # Miss branch: zero out th_match in w1
            w1_miss = w1[idx_x].copy()
            w1_miss[np.arange(len(idx_x)), :, th_match] = 0.0
            imm2_m = np.einsum("bym,maq->byaq", w1_miss, self._r_true)
            if rem <= 2:
                v2_m = imm2_m
            else:
                w2_m = w1_miss[:, :, None, None, :] * np.transpose(self._p_obs, (1, 2, 0))[None, None, :, :, :]
                val_cm = np.einsum("byazm,tqm->byazqt", w2_m, cret)
                v2_m = imm2_m + np.sum(np.max(val_cm, axis=-1), axis=3)
            # Only take slice a1 == qx
            best_miss = np.sum(np.max(v2_m[np.arange(len(idx_x)), :, qx, :], axis=-1), axis=1)
            cont1[idx_x, qx] = p_hit * v_hit + best_miss

        acts = np.argmax(imm1 + cont1, axis=-1).astype(np.int64)
        self._last_a = acts.copy()
        return acts


def rollout_r3_obs_corruption(
    policy: BatchPolicyV3,
    *,
    noise_d: float,
    n_episodes: int = 512,
    horizon: int = 24,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate policy on R3 with symmetric categorical observation corruption d (true payoff unchanged)."""
    thetas = np.tile(np.arange(N_TYPES, dtype=np.int64), n_episodes // N_TYPES)
    env = VectorizedDelayedResponseEnv(regime="R3", batch_size=n_episodes, horizon=horizon, seed=seed)
    obs = env.reset(theta=thetas)
    policy.reset(batch_size=n_episodes, seed=seed + 13, true_theta=thetas if policy.is_privileged else None)
    corrupt_rng = np.random.default_rng(seed + 503)

    actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    focal_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    opp_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    obs_y_hist = np.zeros((n_episodes, horizon), dtype=np.int64)
    corrupted_flags = np.zeros((n_episodes, horizon), dtype=np.float64)

    for t in range(horizon):
        a = policy.act(obs, step_t=t)
        obs, rf, ro, y_true = env.step(a)
        if noise_d > 0.0:
            flip = corrupt_rng.random(n_episodes) < noise_d
            shift = corrupt_rng.integers(1, N_OPP_SYMBOLS, size=n_episodes, dtype=np.int64)
            y_obs = np.where(flip, (y_true + shift) % N_OPP_SYMBOLS, y_true)
            obs["prev_opp_action"] = y_obs.copy()
            corrupted_flags[:, t] = flip.astype(np.float64)
        else:
            y_obs = y_true
        actions[:, t] = a
        focal_rewards[:, t] = rf
        opp_rewards[:, t] = ro
        obs_y_hist[:, t] = y_obs

    ep_tot = focal_rewards.sum(axis=1)
    ep_opp = opp_rewards.sum(axis=1)
    q25, q50, q75 = np.percentile(ep_tot, [25, 50, 75])
    match_act = (ACT_X00 + thetas)[:, None]
    expt = actions[:, 4:]
    s_rate = float(np.mean(expt == ACT_S))
    x_match = float(np.mean(expt == match_act))
    x_mismatch = float(np.mean((expt >= ACT_X00) & (expt != match_act)))

    # Policy-agnostic exact shifted posterior entropy & rule-ID at K=4 from realized (q, y_obs, r)
    f_table = build_response_table("R3")
    r_table = build_focal_reward_table()
    p_obs = np.full((N_TYPES, N_ACTIONS, N_OPP_SYMBOLS), noise_d / (N_OPP_SYMBOLS - 1), dtype=np.float64)
    for th in range(N_TYPES):
        for q in range(N_ACTIONS):
            p_obs[th, q, int(f_table[th, q])] = 1.0 - noise_d
    r_true_tbl = np.zeros((N_TYPES, N_ACTIONS, N_ACTIONS), dtype=np.float64)
    for th in range(N_TYPES):
        for q in range(N_ACTIONS):
            r_true_tbl[th, q, :] = r_table[q, :, int(f_table[th, q])]

    post = np.full((n_episodes, N_TYPES), 1.0 / N_TYPES, dtype=np.float64)
    q_prev = np.full(n_episodes, ACT_S, dtype=np.int64)
    for t in range(min(4, horizon)):
        a_t = actions[:, t]
        y_t = obs_y_hist[:, t]
        r_t = focal_rewards[:, t]
        lik_y = p_obs[:, q_prev, y_t].T
        lik_r = (np.abs(r_true_tbl[:, q_prev, a_t].T - r_t[:, None]) < 1e-6).astype(np.float64)
        post = post * lik_y * lik_r
        denom = np.sum(post, axis=1, keepdims=True)
        post = np.where(denom > 1e-12, post / np.maximum(denom, 1e-12), post)
        q_prev = a_t
    p_safe = np.clip(post, 1e-12, 1.0)
    k4_post_ent = float(np.mean(-np.sum(post * np.log2(p_safe), axis=1)))
    k4_map_acc = float(np.mean(np.argmax(post, axis=1) == thetas))

    metrics = {
        "total_mean": float(np.mean(ep_tot)),
        "total_median": float(q50),
        "total_iqr": float(q75 - q25),
        "opp_total_mean": float(np.mean(ep_opp)),
        "early_auc_1_4": float(np.mean(focal_rewards[:, :4])),
        "late_auc_13_end": float(np.mean(focal_rewards[:, 12:])),
        "expt_S": s_rate,
        "expt_X_match": x_match,
        "expt_X_mismatch": x_mismatch,
        "realized_corrupt_rate": float(np.mean(corrupted_flags)),
        "k4_shifted_post_entropy": k4_post_ent,
        "k4_shifted_map_acc": k4_map_acc,
    }
    return metrics, {"total_return": ep_tot, "opp_total_return": ep_opp}


def rollout_coop_dynamics_shift(
    p1: SymPolicy,
    p2: SymPolicy,
    *,
    slip_d: float,
    slip_payoff: float = -0.2,
    n_episodes: int = 512,
    horizon: int = 24,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate coop_trap unseen_rules when mutual (c*, c*) slips to slip_payoff with probability slip_d."""
    env = VectorizedSymmetricGameEnv(game="coop_trap", split="test", batch_size=n_episodes, horizon=horizon, seed=seed)
    n_rules = env.n_rules
    rule_ids = np.tile(np.arange(n_rules, dtype=np.int64), n_episodes // n_rules)
    coop_act = env.bank["coop"][rule_ids]
    safe_act = env.bank["safe"][rule_ids]
    trap_act = env.bank["trap"][rule_ids]

    obs1, obs2 = env.reset(rule_ids=rule_ids)
    p1.reset(n_episodes, seed=seed + 13)
    p2.reset(n_episodes, seed=seed + 29)
    slip_rng = np.random.default_rng(seed + 701)

    a1_hist = np.zeros((n_episodes, horizon), dtype=np.int64)
    a2_hist = np.zeros((n_episodes, horizon), dtype=np.int64)
    r1_hist = np.zeros((n_episodes, horizon), dtype=np.float64)
    r2_hist = np.zeros((n_episodes, horizon), dtype=np.float64)

    for t in range(horizon):
        a1 = p1.act(obs1, step_t=t)
        a2 = p2.act(obs2, step_t=t)
        obs1, obs2, r1, r2 = env.step(a1, a2)
        if slip_d > 0.0:
            is_mc = (a1 == coop_act) & (a2 == coop_act)
            slipped = is_mc & (slip_rng.random(n_episodes) < slip_d)
            if np.any(slipped):
                r1 = np.where(slipped, slip_payoff, r1)
                r2 = np.where(slipped, slip_payoff, r2)
                obs1["prev_self_reward"] = r1.copy()
                obs2["prev_self_reward"] = r2.copy()
        a1_hist[:, t] = a1
        a2_hist[:, t] = a2
        r1_hist[:, t] = r1
        r2_hist[:, t] = r2

    p1_tot = r1_hist.sum(axis=1)
    p2_tot = r2_hist.sum(axis=1)
    welfare = p1_tot + p2_tot
    margin = p1_tot - p2_tot
    q25, q50, q75 = np.percentile(p1_tot, [25, 50, 75])
    mc_late = float(np.mean((a1_hist[:, 4:] == coop_act[:, None]) & (a2_hist[:, 4:] == coop_act[:, None])))
    ms_late = float(np.mean((a1_hist[:, 4:] == safe_act[:, None]) & (a2_hist[:, 4:] == safe_act[:, None])))
    trap_all = float(np.mean((a1_hist == trap_act[:, None]) | (a2_hist == trap_act[:, None])))

    metrics = {
        "p1_mean": float(np.mean(p1_tot)),
        "p1_median": float(q50),
        "p1_iqr": float(q75 - q25),
        "p2_mean": float(np.mean(p2_tot)),
        "welfare_mean": float(np.mean(welfare)),
        "welfare_median": float(np.median(welfare)),
        "margin_mean": float(np.mean(margin)),
        "p1_early_auc_1_4": float(np.mean(r1_hist[:, :4])),
        "mutual_coop_late": mc_late,
        "mutual_safe_late": ms_late,
        "trap_rate": trap_all,
    }
    return metrics, {"p1_return": p1_tot, "welfare": welfare, "margin": margin}


def _load_r3_policy(name: str, seed: int, d_val: float, cfg: dict) -> BatchPolicyV3:
    horizon = int(cfg.get("horizon", 24))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    exp1_dir = Path(cfg.get("exp1_runs_dir", "runs/core_v3_sealed"))
    exp2a_dir = Path(cfg.get("exp2a_runs_dir", "runs/exp2a_online_opp"))

    exp1_map = {
        "exp1_drqn_aux_yr": ("R3_drqn_aux_yr", "rl2"),
        "exp1_drqn_vanilla": ("R3_drqn_vanilla", "rl2"),
        "exp1_drqn_aux_yr_masked": ("R3_drqn_aux_yr_masked", "rl2_masked"),
        "exp1_drqn_prior_only": ("R3_drqn_prior_only", "prior_only"),
    }
    exp2a_map = {
        "exp2a_mt_drqn_aux_yr": ("mt_drqn_aux_yr", "rl2"),
        "exp2a_mt_drqn_vanilla": ("mt_drqn_vanilla", "rl2"),
    }

    if name in exp1_map:
        prefix, variant = exp1_map[name]
        ckpt = exp1_dir / "fits" / f"{prefix}_seed{seed}" / "checkpoint.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        qnet = RecurrentQNet(OBS_DIM, hidden_dim, N_ACTIONS)
        qnet.load_state_dict(payload["state_dict"])
        qnet.eval()
        return NeuralDRQNPolicyV3(qnet, name=name, variant=variant, horizon=horizon, mask_k=4)

    if name in exp2a_map:
        prefix, variant = exp2a_map[name]
        ckpt = exp2a_dir / "fits" / f"{prefix}_seed{seed}" / "checkpoint.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        qnet = RecurrentQNet(OBS_DIM, hidden_dim, N_ACTIONS)
        qnet.load_state_dict(payload["state_dict"])
        qnet.eval()
        return NeuralDRQNPolicyV3(qnet, name=name, variant=variant, horizon=horizon, mask_k=4)

    if name == "exp1_structural_model":
        pol = StructuralModelPolicyV3(horizon=horizon).fit_offline("R3", n_episodes=1024, seed=seed + 777)
        pol.name = name
        return pol

    if name == "exp2a_mt_structural_model":
        opp_cfg = OnlineOpponentConfig(kind="q_learning", regime="R3", alpha=0.35, gamma=0.50, epsilon=0.08, warmup_steps=80, warmup_epsilon=1.0)
        pol = StructuralModelPolicyV3(horizon=horizon).fit_offline(
            "R3", n_episodes=1024, seed=seed + 777, opponent_factory=lambda b, s: make_online_opponent(opp_cfg, b, s)
        )
        pol.name = name
        return pol

    if name == "shifted_bayes_ref":
        return ShiftedChannelBayesRefV3(regime="R3", horizon=horizon, noise_d=d_val)
    if name == "greedy":
        return GreedyPolicyV3()
    if name == "arm_ucb":
        return ArmUCBPolicyV3(beta=1.0)
    if name == "contextual_ucb":
        return ContextualUCBPolicyV3(beta=1.0)
    raise ValueError(f"Unknown R3 sweep policy: {name}")


def _load_coop_policy(name: str, seed: int, cfg: dict) -> SymPolicy:
    exp2b_dir = Path(cfg.get("exp2b_runs_dir", "runs/exp2b_symmetric_ma"))
    if name in META_METHODS:
        variant, _ = META_METHODS[name]
        ckpt = exp2b_dir / "fits" / f"coop_trap_{name}_seed{seed}" / "checkpoint.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        qnet = SymmetricRecurrentQNet(SYM_OBS_DIM, int(cfg.get("hidden_dim", 64)), SYM_N_ACTIONS)
        qnet.load_state_dict(payload["state_dict"])
        qnet.eval()
        return SymNeuralDRQNPolicy(qnet, name=name, variant=variant, horizon=int(cfg.get("horizon", 24)), mask_k=4)
    if name == "q_learner":
        return SymQLearningPolicy(alpha=0.35, gamma=0.50, epsilon=0.12)
    if name == "ucb":
        return SymUCBPolicy(beta=1.0)
    if name == "contextual_ucb":
        return SymContextualUCBPolicy(beta=1.0)
    raise ValueError(f"Unknown coop_trap policy: {name}")


def run_reliability_sweep(cfg: dict) -> dict:
    """Execute Exp 3A evaluation-only reliability & dynamics sweeps and export compact tables."""
    study_id = str(cfg["study_id"])
    out_dir = Path(cfg.get("output_dir", f"runs/{study_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in cfg["seeds"]]
    d_values = [float(d) for d in cfg.get("d_values", [0.0, 0.3, 0.6])]
    n_eval = int(cfg.get("n_eval_episodes", 512))
    horizon = int(cfg.get("horizon", 24))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta = float(cfg.get("delta_return", 0.30))
    eval_seed_base = int(cfg.get("eval_seed_base", 98000))

    r3_methods = [str(m) for m in cfg["r3_methods"]]
    coop_matchups = [
        ("q_learner_vs_q_learner", "q_learner", "q_learner", False),
        ("ucb_vs_ucb", "ucb", "ucb", False),
        ("meta_aux_yr_vs_q_learner", "meta_drqn_aux_yr", "q_learner", False),
        ("meta_aux_yr_vs_ucb", "meta_drqn_aux_yr", "ucb", False),
        ("meta_aux_yr_vs_meta_aux_yr_xseed", "meta_drqn_aux_yr", "meta_drqn_aux_yr", True),
        ("meta_vanilla_vs_meta_vanilla_xseed", "meta_drqn_vanilla", "meta_drqn_vanilla", True),
        ("meta_aux_yr_vs_meta_masked_xseed", "meta_drqn_aux_yr", "meta_drqn_masked", True),
    ]

    r3_rows: list[dict] = []
    coop_rows: list[dict] = []
    contrast_rows: list[dict] = []
    summary_r3: dict[str, dict] = {}
    summary_coop: dict[str, dict] = {}

    # Track A: R3 observation reliability sweep
    for d_idx, d_val in enumerate(d_values):
        d_key = f"d={d_val:.1f}"
        arrays_by_m: dict[str, np.ndarray] = {}
        agg_m: dict[str, dict] = {}

        for m in r3_methods:
            recs = []
            tot_list = []
            for s in seeds:
                pol = _load_r3_policy(m, s, d_val, cfg)
                md, ad = rollout_r3_obs_corruption(
                    pol,
                    noise_d=d_val,
                    n_episodes=n_eval,
                    horizon=horizon,
                    seed=eval_seed_base + d_idx * 1000 + s * 17,
                )
                recs.append(md)
                tot_list.append(ad["total_return"])
            mat = np.stack(tot_list, axis=0)
            arrays_by_m[m] = mat
            seed_means = np.mean(mat, axis=1)
            agg = {
                "is_privileged": m == "shifted_bayes_ref",
                "total_mean": float(np.mean(seed_means)),
                "total_seed_sd": float(np.std(seed_means, ddof=1)) if len(seeds) > 1 else 0.0,
                "total_median": float(np.median(mat)),
                "total_iqr": float(np.percentile(mat, 75) - np.percentile(mat, 25)),
                "opp_total_mean": float(np.mean([r["opp_total_mean"] for r in recs])),
                "early_auc_1_4": float(np.mean([r["early_auc_1_4"] for r in recs])),
                "expt_S": float(np.mean([r["expt_S"] for r in recs])),
                "expt_X_match": float(np.mean([r["expt_X_match"] for r in recs])),
                "expt_X_mismatch": float(np.mean([r["expt_X_mismatch"] for r in recs])),
                "k4_shifted_post_entropy": float(np.mean([r["k4_shifted_post_entropy"] for r in recs])),
                "k4_shifted_map_acc": float(np.mean([r["k4_shifted_map_acc"] for r in recs])),
            }
            agg_m[m] = agg
            r3_rows.append(
                {
                    "track": "r3_obs_reliability",
                    "d": d_val,
                    "method": m,
                    "privileged": agg["is_privileged"],
                    "total_mean": round(agg["total_mean"], 4),
                    "total_seed_sd": round(agg["total_seed_sd"], 4),
                    "total_median": round(agg["total_median"], 4),
                    "total_iqr": round(agg["total_iqr"], 4),
                    "opp_total_mean": round(agg["opp_total_mean"], 4),
                    "early_auc_1_4": round(agg["early_auc_1_4"], 4),
                    "expt_S": round(agg["expt_S"], 4),
                    "expt_X_match": round(agg["expt_X_match"], 4),
                    "expt_X_mismatch": round(agg["expt_X_mismatch"], 4),
                    "k4_post_ent": round(agg["k4_shifted_post_entropy"], 4),
                    "k4_map_acc": round(agg["k4_shifted_map_acc"], 4),
                }
            )

        contrasts = {}
        for a, b in (
            ("exp1_drqn_aux_yr", "exp1_drqn_prior_only"),
            ("exp1_drqn_aux_yr", "exp1_drqn_aux_yr_masked"),
            ("exp1_drqn_aux_yr", "exp1_structural_model"),
            ("exp2a_mt_drqn_aux_yr", "exp1_drqn_aux_yr"),
            ("exp2a_mt_drqn_aux_yr", "exp1_drqn_prior_only"),
            ("exp2a_mt_drqn_aux_yr", "exp2a_mt_structural_model"),
            ("exp2a_mt_drqn_aux_yr", "exp2a_mt_drqn_vanilla"),
        ):
            if a in arrays_by_m and b in arrays_by_m:
                res = hierarchical_paired_bootstrap(arrays_by_m[a], arrays_by_m[b], practical_threshold=delta, n_boot=n_boot, seed=77001)
                c_key = f"{a}_minus_{b}"
                contrasts[c_key] = res
                contrast_rows.append(
                    {
                        "track": "r3_obs_reliability",
                        "d": d_val,
                        "contrast": c_key,
                        "metric": "total_return",
                        "diff_mean": round(float(res["diff_mean"]), 4),
                        "diff_median": round(float(res["diff_median"]), 4),
                        "ci_low": round(float(res["ci_low"]), 4),
                        "ci_high": round(float(res["ci_high"]), 4),
                        "win_rate": round(float(res["win_rate"]), 4),
                        "tie_rate": round(float(res["tie_rate"]), 4),
                        "loss_rate": round(float(res["loss_rate"]), 4),
                        "threshold": delta,
                        "verdict": res["verdict"],
                    }
                )
        summary_r3[d_key] = {"methods": agg_m, "paired_contrasts": contrasts}

    # Track B: coop_trap cooperative-success dynamics shift on unseen_rules
    slip_payoff = float(cfg.get("coop_slip_payoff", -0.2))
    for d_idx, d_val in enumerate(d_values):
        d_key = f"d={d_val:.1f}"
        arrs_by_lbl: dict[str, dict[str, np.ndarray]] = {}
        agg_c: dict[str, dict] = {}

        for lbl, p1_n, p2_n, xseed in coop_matchups:
            recs = []
            w_list, p1_list = [], []
            for i_s, s in enumerate(seeds):
                s2 = seeds[(i_s + 1) % len(seeds)] if (xseed and len(seeds) > 1) else s
                p1 = _load_coop_policy(p1_n, s, cfg)
                p2 = _load_coop_policy(p2_n, s2, cfg)
                md, ad = rollout_coop_dynamics_shift(
                    p1,
                    p2,
                    slip_d=d_val,
                    slip_payoff=slip_payoff,
                    n_episodes=n_eval,
                    horizon=horizon,
                    seed=eval_seed_base + 50000 + d_idx * 1000 + s * 17,
                )
                recs.append(md)
                w_list.append(ad["welfare"])
                p1_list.append(ad["p1_return"])
            w_mat = np.stack(w_list, axis=0)
            p1_mat = np.stack(p1_list, axis=0)
            arrs_by_lbl[lbl] = {"welfare": w_mat, "p1_return": p1_mat}
            seed_w = np.mean(w_mat, axis=1)
            seed_p1 = np.mean(p1_mat, axis=1)
            agg = {
                "p1_mean": float(np.mean(seed_p1)),
                "p1_seed_sd": float(np.std(seed_p1, ddof=1)) if len(seeds) > 1 else 0.0,
                "p1_median": float(np.median(p1_mat)),
                "p1_iqr": float(np.percentile(p1_mat, 75) - np.percentile(p1_mat, 25)),
                "welfare_mean": float(np.mean(seed_w)),
                "welfare_seed_sd": float(np.std(seed_w, ddof=1)) if len(seeds) > 1 else 0.0,
                "welfare_median": float(np.median(w_mat)),
                "margin_mean": float(np.mean([r["margin_mean"] for r in recs])),
                "p1_early_auc_1_4": float(np.mean([r["p1_early_auc_1_4"] for r in recs])),
                "mutual_coop_late": float(np.mean([r["mutual_coop_late"] for r in recs])),
                "mutual_safe_late": float(np.mean([r["mutual_safe_late"] for r in recs])),
                "trap_rate": float(np.mean([r["trap_rate"] for r in recs])),
            }
            agg_c[lbl] = agg
            coop_rows.append(
                {
                    "track": "coop_trap_dynamics",
                    "d": d_val,
                    "matchup": lbl,
                    "p1_mean": round(agg["p1_mean"], 4),
                    "p1_seed_sd": round(agg["p1_seed_sd"], 4),
                    "p1_median": round(agg["p1_median"], 4),
                    "p1_iqr": round(agg["p1_iqr"], 4),
                    "welfare_mean": round(agg["welfare_mean"], 4),
                    "welfare_median": round(agg["welfare_median"], 4),
                    "margin_mean": round(agg["margin_mean"], 4),
                    "p1_early_auc_1_4": round(agg["p1_early_auc_1_4"], 4),
                    "mutual_coop_late": round(agg["mutual_coop_late"], 4),
                    "mutual_safe_late": round(agg["mutual_safe_late"], 4),
                    "trap_rate": round(agg["trap_rate"], 4),
                }
            )

        contrasts_c = {}
        for a, b in (
            ("meta_aux_yr_vs_meta_aux_yr_xseed", "ucb_vs_ucb"),
            ("meta_aux_yr_vs_meta_aux_yr_xseed", "q_learner_vs_q_learner"),
            ("meta_aux_yr_vs_meta_aux_yr_xseed", "meta_vanilla_vs_meta_vanilla_xseed"),
            ("meta_aux_yr_vs_q_learner", "q_learner_vs_q_learner"),
            ("meta_aux_yr_vs_ucb", "ucb_vs_ucb"),
        ):
            res = hierarchical_paired_bootstrap(
                arrs_by_lbl[a]["welfare"], arrs_by_lbl[b]["welfare"], practical_threshold=delta, n_boot=n_boot, seed=77002
            )
            c_key = f"{a}_minus_{b}"
            contrasts_c[c_key] = res
            contrast_rows.append(
                {
                    "track": "coop_trap_dynamics",
                    "d": d_val,
                    "contrast": c_key,
                    "metric": "welfare",
                    "diff_mean": round(float(res["diff_mean"]), 4),
                    "diff_median": round(float(res["diff_median"]), 4),
                    "ci_low": round(float(res["ci_low"]), 4),
                    "ci_high": round(float(res["ci_high"]), 4),
                    "win_rate": round(float(res["win_rate"]), 4),
                    "tie_rate": round(float(res["tie_rate"]), 4),
                    "loss_rate": round(float(res["loss_rate"]), 4),
                    "threshold": delta,
                    "verdict": res["verdict"],
                }
            )
        summary_coop[d_key] = {"matchups": agg_c, "paired_contrasts": contrasts_c}

    summary = {
        "study_id": study_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "seeds": seeds,
        "d_values": d_values,
        "r3_obs_reliability": summary_r3,
        "coop_trap_dynamics": summary_coop,
    }
    _atomic_write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "tables" / "r3_obs_reliability.csv", r3_rows)
    _write_csv(out_dir / "tables" / "coop_trap_dynamics.csv", coop_rows)
    _write_csv(out_dir / "tables" / "paired_contrasts.csv", contrast_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 3A Evaluation-Only Reliability & Dynamics Sweep.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    summary = run_reliability_sweep(cfg)
    print(f"[{cfg['study_id']}] Complete in {time.perf_counter() - t0:.1f}s")
    for dk, data in summary["r3_obs_reliability"].items():
        print(f"\n=== R3 Obs Reliability ({dk}) ===")
        for m, md in data["methods"].items():
            print(
                f"  {m:26s} | Ret={md['total_mean']:+6.2f} (sd={md['total_seed_sd']:4.2f}) med={md['total_median']:+6.2f} "
                f"| S={md['expt_S']:.2f} Xm={md['expt_X_match']:.2f} Xbad={md['expt_X_mismatch']:.2f} | K4_MAP={md['k4_shifted_map_acc']:.2f}"
            )
    for dk, data in summary["coop_trap_dynamics"].items():
        print(f"\n=== coop_trap Unseen Dynamics ({dk}) ===")
        for lbl, md in data["matchups"].items():
            print(
                f"  {lbl:34s} | P1={md['p1_mean']:+6.2f} W={md['welfare_mean']:+6.2f} (med={md['welfare_median']:+6.2f}) "
                f"| coopL={md['mutual_coop_late']:.2f} safeL={md['mutual_safe_late']:.2f} trap={md['trap_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
