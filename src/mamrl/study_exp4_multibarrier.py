"""SPEC v3.0 Exp 4: 3-Layer Nested Exploration Barriers (T=64) + Auxiliary Task Discovery.

Evaluates whether recurrent Meta-RL climbs multiple nested epistemic barriers
(Tier 0 +0.2 -> Tier 1 +0.7/-1.0 -> Tier 2 +1.3/-3.0 -> Tier 3 +2.2/-10.0) over long
horizon T=64 (22 actions, 8 hidden types) AND whether first-order bilevel meta-gradient
Auxiliary Task Discovery ('drqn_aux_discovery') automatically selects useful public
prediction signals (next_opp_y, next_reward) while suppressing pure distractor targets
(random categorical noise1, Gaussian noise2) without hand-selection.

Strictly unprivileged (only 'bayes_optimal' is privileged=True as exact rational DP ceiling).
Respects strict step_t > mask_k masking boundary (fixed per TASK-002 audit).

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_exp4_multibarrier --config configs/exp4_multibarrier.json --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn

from mamrl.envs.multi_barrier import (
    MB_ACT_P1,
    MB_ACT_P2_START,
    MB_ACT_P3_START,
    MB_ACT_S,
    MB_ACT_T1_START,
    MB_ACT_T2_START,
    MB_ACT_T3_START,
    MB_FULL_BELIEF_MASK,
    MB_N_ACTIONS,
    MB_N_SYMBOLS,
    MB_N_TYPES,
    ExactMultiBarrierDP,
    VectorizedMultiBarrierEnv,
    build_mb_response_table,
    mb_action_tier,
    run_mb_gate0_report,
)
from mamrl.eval.audit_v3 import hierarchical_paired_bootstrap

MB_OBS_DIM = MB_N_ACTIONS + MB_N_SYMBOLS + 3  # 22 + 3 + 3 = 28
CAND_NAMES = ("cand_y", "cand_r", "cand_noise1", "cand_noise2")
N_CANDS = 4


def encode_mb_obs_batch(
    obs: dict[str, np.ndarray],
    *,
    horizon: int,
    step_t: int,
    variant: str = "rl2",
    mask_k: int = 8,
) -> np.ndarray:
    """Encode public Multi-Barrier observation into float32 (B, 28). Strictly step_t > mask_k."""
    b = int(obs["prev_self_action"].shape[0])
    x = np.zeros((b, MB_OBS_DIM), dtype=np.float32)
    idx = np.arange(b)

    pa = obs["prev_self_action"]
    valid_a = (pa >= 0) & (pa < MB_N_ACTIONS)
    x[idx[valid_a], pa[valid_a]] = 1.0

    allow_feedback = (variant == "rl2") or (variant == "rl2_masked" and step_t > mask_k)
    if allow_feedback:
        py = obs["prev_opp_action"]
        valid_y = (py >= 0) & (py < MB_N_SYMBOLS)
        x[idx[valid_y], MB_N_ACTIONS + py[valid_y]] = 1.0
        x[:, MB_N_ACTIONS + MB_N_SYMBOLS] = (obs["prev_self_reward"] / 10.0).astype(np.float32)

    x[:, MB_N_ACTIONS + MB_N_SYMBOLS + 1] = obs["is_start"].astype(np.float32)
    x[:, MB_N_ACTIONS + MB_N_SYMBOLS + 2] = float(step_t) / max(1.0, float(horizon))
    return x


class MultiBarrierQNet(nn.Module):
    """GRU-96 Recurrent Q-Network with 4 candidate public auxiliary heads (y, r, noise1, noise2)."""

    def __init__(self, obs_dim: int = MB_OBS_DIM, hidden_dim: int = 96, n_actions: int = MB_N_ACTIONS) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, n_actions)
        # 4 candidate self-supervised auxiliary heads
        self.pred_y = nn.Linear(hidden_dim + n_actions, MB_N_SYMBOLS)
        self.pred_r = nn.Linear(hidden_dim + n_actions, 1)
        self.pred_noise1 = nn.Linear(hidden_dim + n_actions, MB_N_SYMBOLS)
        self.pred_noise2 = nn.Linear(hidden_dim + n_actions, 1)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x_seq)
        return self.q_head(out), out

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_next = self.gru(x_t, h)
        return self.q_head(out.squeeze(1)), h_next

    def candidate_aux_losses(
        self,
        hidden_seq: torch.Tensor,
        act_seq: torch.Tensor,
        y_seq: torch.Tensor,
        r_seq: torch.Tensor,
        z1_seq: torch.Tensor,
        z2_seq: torch.Tensor,
        *,
        start_t: int = 0,
    ) -> list[torch.Tensor]:
        """Return 4 individual scalar candidate auxiliary losses [L_y, L_r, L_noise1, L_noise2]."""
        hs = hidden_seq[:, start_t:, :]
        ax = act_seq[:, start_t:]
        a_oh = nn.functional.one_hot(ax, num_classes=self.n_actions).float()
        ha = torch.cat([hs, a_oh], dim=-1)

        l_y = nn.functional.cross_entropy(self.pred_y(ha).reshape(-1, MB_N_SYMBOLS), y_seq[:, start_t:].reshape(-1))
        l_r = nn.functional.mse_loss(self.pred_r(ha).squeeze(-1), r_seq[:, start_t:] / 10.0) * 10.0
        l_n1 = nn.functional.cross_entropy(self.pred_noise1(ha).reshape(-1, MB_N_SYMBOLS), z1_seq[:, start_t:].reshape(-1))
        l_n2 = nn.functional.mse_loss(self.pred_noise2(ha).squeeze(-1), z2_seq[:, start_t:])
        return [l_y, l_r, l_n1, l_n2]


@dataclass
class MBTrainConfig:
    method: str = "drqn_aux_yr"
    variant: str = "rl2"      # "rl2", "rl2_masked", "prior_only"
    aux_strategy: str = "yr"  # "none", "y", "r", "yr", "uniform", "discovery"
    mask_k: int = 8
    horizon: int = 64
    hidden_dim: int = 96
    batch_episodes: int = 48
    n_iters: int = 450
    warmup_iters: int = 90
    replay_capacity: int = 6144
    grad_steps_per_iter: int = 6
    minibatch_episodes: int = 96
    gamma: float = 0.99       # slight discount stabilizes long-horizon T=64 TD while preserving exact Tier-3 optimality
    lr: float = 3e-3
    polyak_tau: float = 0.05
    aux_scale: float = 2.0    # total normalized auxiliary weight budget across simplex
    meta_gate_lr: float = 0.35
    eps_start: float = 0.90
    eps_end: float = 0.05
    max_grad_norm: float = 5.0
    log_interval: int = 25
    seed: int = 0
    device: str = "cpu"


def _static_gate_weights(aux_strategy: str, variant: str) -> np.ndarray:
    if variant == "prior_only" or aux_strategy == "none":
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if aux_strategy == "y":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if aux_strategy == "r":
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    if aux_strategy == "yr":
        return np.array([0.65, 0.35, 0.0, 0.0], dtype=np.float64)
    if aux_strategy == "uniform":
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    if aux_strategy == "discovery":
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    raise ValueError(f"Unknown aux_strategy: {aux_strategy}")


def train_mb_drqn(cfg: MBTrainConfig) -> tuple[MultiBarrierQNet, list[dict[str, float]], int, int, list[float]]:
    """Train MultiBarrierQNet with optional First-Order Bilevel Gradient-Alignment Auxiliary Discovery.

    Returns (qnet, train_curve, total_transitions, extra_outer_grad_steps, final_gate_weights).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    qnet = MultiBarrierQNet(MB_OBS_DIM, cfg.hidden_dim, MB_N_ACTIONS).to(device)
    tnet = MultiBarrierQNet(MB_OBS_DIM, cfg.hidden_dim, MB_N_ACTIONS).to(device)
    tnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)

    b = cfg.batch_episodes
    t_len = cfg.horizon
    env = VectorizedMultiBarrierEnv(batch_size=b, horizon=t_len, seed=cfg.seed + 17)
    rng = np.random.default_rng(cfg.seed + 31)

    cap = cfg.replay_capacity
    buf_obs = np.zeros((cap, t_len, MB_OBS_DIM), dtype=np.float32)
    buf_act = np.zeros((cap, t_len), dtype=np.int64)
    buf_y = np.zeros((cap, t_len), dtype=np.int64)
    buf_rew = np.zeros((cap, t_len), dtype=np.float32)
    # Candidate distractors generated purely online as public random streams (no hidden info!)
    buf_z1 = np.zeros((cap, t_len), dtype=np.int64)
    buf_z2 = np.zeros((cap, t_len), dtype=np.float32)
    ptr, size = 0, 0

    gate_logits = np.zeros(N_CANDS, dtype=np.float64)
    static_w = _static_gate_weights(cfg.aux_strategy, cfg.variant)
    is_discovery = (cfg.aux_strategy == "discovery") and (cfg.variant != "prior_only")
    gru_params = list(qnet.gru.parameters())

    total_transitions = 0
    extra_outer_steps = 0
    train_curve: list[dict[str, float]] = []
    start_t = cfg.mask_k if cfg.variant == "rl2_masked" else 0

    for it in range(1, cfg.n_iters + 1):
        if it <= cfg.warmup_iters:
            eps = 1.0
        else:
            decay_span = max(1.0, 0.8 * float(cfg.n_iters - cfg.warmup_iters))
            eps = max(cfg.eps_end, cfg.eps_start * (1.0 - float(it - cfg.warmup_iters) / decay_span))

        obs = env.reset()
        h = torch.zeros((1, b, cfg.hidden_dim), dtype=torch.float32, device=device)
        ep_o = np.zeros((b, t_len, MB_OBS_DIM), dtype=np.float32)
        ep_a = np.zeros((b, t_len), dtype=np.int64)
        ep_y = np.zeros((b, t_len), dtype=np.int64)
        ep_r = np.zeros((b, t_len), dtype=np.float32)
        ep_z1 = rng.integers(0, MB_N_SYMBOLS, size=(b, t_len), dtype=np.int64)
        ep_z2 = rng.standard_normal((b, t_len)).astype(np.float32)

        qnet.eval()
        with torch.no_grad():
            for t in range(t_len):
                x = encode_mb_obs_batch(obs, horizon=t_len, step_t=t, variant=cfg.variant, mask_k=cfg.mask_k)
                ep_o[:, t] = x
                qv, h = qnet.step(torch.from_numpy(x).to(device).unsqueeze(1), h)
                greedy_a = qv.argmax(dim=-1).cpu().numpy()
                rand_a = rng.integers(0, MB_N_ACTIONS, size=b)
                a = np.where(rng.random(b) < eps, rand_a, greedy_a).astype(np.int64)
                obs, rf, _ro, y = env.step(a)
                ep_a[:, t] = a
                ep_y[:, t] = y
                ep_r[:, t] = rf.astype(np.float32)

        total_transitions += b * t_len
        for k in range(b):
            buf_obs[ptr] = ep_o[k]
            buf_act[ptr] = ep_a[k]
            buf_y[ptr] = ep_y[k]
            buf_rew[ptr] = ep_r[k]
            buf_z1[ptr] = ep_z1[k]
            buf_z2[ptr] = ep_z2[k]
            ptr = (ptr + 1) % cap
            size = min(cap, size + 1)

        # Current simplex gate weights
        if is_discovery:
            ex = np.exp(gate_logits - np.max(gate_logits))
            cur_w = ex / np.sum(ex)
        else:
            cur_w = static_w

        qnet.train()
        td_vals: list[float] = []
        mb_n = min(cfg.minibatch_episodes, size)

        # Outer-loop bilevel gradient-alignment update for Auxiliary Task Discovery (once per iter after half warmup)
        if is_discovery and it >= (cfg.warmup_iters // 2) and size >= 2 * mb_n:
            perm_all = rng.choice(size, size=2 * mb_n, replace=False)
            tr_idx, val_idx = perm_all[:mb_n], perm_all[mb_n:]

            # 1. Validation extrinsic TD gradient w.r.t. shared GRU parameters
            ox_v = torch.from_numpy(buf_obs[val_idx]).to(device)
            ax_v = torch.from_numpy(buf_act[val_idx]).to(device)
            rx_v = torch.from_numpy(buf_rew[val_idx]).to(device)
            q_v, _ = qnet(ox_v)
            q_sa_v = q_v.gather(-1, ax_v.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                na_v = q_v[:, 1:, :].argmax(dim=-1, keepdim=True)
                tq_v, _ = tnet(ox_v)
                nq_v = tq_v[:, 1:, :].gather(-1, na_v).squeeze(-1)
                tgt_v = rx_v.clone()
                tgt_v[:, :-1] += cfg.gamma * nq_v
            val_td = nn.functional.mse_loss(q_sa_v, tgt_v)
            g_val = torch.autograd.grad(val_td, gru_params, retain_graph=False)
            g_val_flat = torch.cat([g.reshape(-1) for g in g_val]).detach()
            g_val_norm = float(torch.norm(g_val_flat).item()) + 1e-8

            # 2. Inner-batch candidate auxiliary gradients w.r.t. shared GRU parameters -> cosine alignment
            ox_t = torch.from_numpy(buf_obs[tr_idx]).to(device)
            ax_t = torch.from_numpy(buf_act[tr_idx]).to(device)
            yx_t = torch.from_numpy(buf_y[tr_idx]).to(device)
            rx_t = torch.from_numpy(buf_rew[tr_idx]).to(device)
            z1_t = torch.from_numpy(buf_z1[tr_idx]).to(device)
            z2_t = torch.from_numpy(buf_z2[tr_idx]).to(device)
            _, hs_t = qnet(ox_t)
            cand_losses = qnet.candidate_aux_losses(hs_t, ax_t, yx_t, rx_t, z1_t, z2_t, start_t=start_t)

            aligns = np.zeros(N_CANDS, dtype=np.float64)
            for ci, c_loss in enumerate(cand_losses):
                g_c = torch.autograd.grad(c_loss, gru_params, retain_graph=(ci < N_CANDS - 1))
                g_c_flat = torch.cat([g.reshape(-1) for g in g_c]).detach()
                g_c_norm = float(torch.norm(g_c_flat).item()) + 1e-8
                # Cosine alignment: positive when aux gradient decreases held-out extrinsic TD loss!
                aligns[ci] = float(torch.dot(g_val_flat, g_c_flat).item()) / (g_val_norm * g_c_norm)

            # Mirror-ascent / exponentiated-gradient update on simplex logits (clipped for stability)
            gate_logits = np.clip(gate_logits + cfg.meta_gate_lr * aligns, -5.0, 5.0)
            ex = np.exp(gate_logits - np.max(gate_logits))
            cur_w = ex / np.sum(ex)
            extra_outer_steps += 1

        for _ in range(cfg.grad_steps_per_iter):
            idx = rng.integers(0, size, size=mb_n)
            ox = torch.from_numpy(buf_obs[idx]).to(device)
            ax = torch.from_numpy(buf_act[idx]).to(device)
            yx = torch.from_numpy(buf_y[idx]).to(device)
            rx = torch.from_numpy(buf_rew[idx]).to(device)
            z1x = torch.from_numpy(buf_z1[idx]).to(device)
            z2x = torch.from_numpy(buf_z2[idx]).to(device)

            q_all, hs = qnet(ox)
            q_sa = q_all.gather(-1, ax.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_a = q_all[:, 1:, :].argmax(dim=-1, keepdim=True)
                tq, _ = tnet(ox)
                next_q = tq[:, 1:, :].gather(-1, next_a).squeeze(-1)
                tgt = rx.clone()
                tgt[:, :-1] += cfg.gamma * next_q

            td_loss = nn.functional.mse_loss(q_sa, tgt)
            if float(np.sum(cur_w)) > 0.0 and start_t < t_len:
                cand_losses = qnet.candidate_aux_losses(hs, ax, yx, rx, z1x, z2x, start_t=start_t)
                aux_total = sum(float(cfg.aux_scale * cur_w[ci]) * cand_losses[ci] for ci in range(N_CANDS) if cur_w[ci] > 0.0)
            else:
                aux_total = torch.tensor(0.0, device=device)

            loss = td_loss + aux_total
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                tau = cfg.polyak_tau
                for p, tp in zip(qnet.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

            td_vals.append(float(td_loss.item()))

        if it == 1 or it == cfg.n_iters or (it % cfg.log_interval == 0):
            train_curve.append(
                {
                    "update": float(it),
                    "transitions": float(total_transitions),
                    "eps": float(eps),
                    "mean_return": float(np.mean(ep_r.sum(axis=1))),
                    "td_loss": float(np.mean(td_vals)),
                    "w_y": float(cur_w[0]),
                    "w_r": float(cur_w[1]),
                    "w_noise1": float(cur_w[2]),
                    "w_noise2": float(cur_w[3]),
                }
            )

    qnet.eval()
    return qnet, train_curve, total_transitions, extra_outer_steps, [float(x) for x in cur_w]


def rollout_mb_policy(
    policy_fn,
    *,
    n_episodes: int = 512,
    horizon: int = 64,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate a Multi-Barrier policy on exact balanced 8-type episodes."""
    if n_episodes % MB_N_TYPES != 0:
        raise ValueError(f"n_episodes ({n_episodes}) must be divisible by {MB_N_TYPES}.")
    thetas = np.tile(np.arange(MB_N_TYPES, dtype=np.int64), n_episodes // MB_N_TYPES)
    env = VectorizedMultiBarrierEnv(batch_size=n_episodes, horizon=horizon, seed=seed)
    obs = env.reset(theta=thetas)
    policy_fn.reset(n_episodes, seed=seed + 13)

    actions = np.zeros((n_episodes, horizon), dtype=np.int64)
    focal_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)
    opp_rewards = np.zeros((n_episodes, horizon), dtype=np.float64)

    for t in range(horizon):
        a = policy_fn.act(obs, step_t=t)
        obs, rf, ro, _y = env.step(a)
        actions[:, t] = a
        focal_rewards[:, t] = rf
        opp_rewards[:, t] = ro

    ep_tot = focal_rewards.sum(axis=1)
    ep_opp = opp_rewards.sum(axis=1)
    q25, q50, q75 = np.percentile(ep_tot, [25, 50, 75])

    # Classify each episode's highest successfully exploited barrier tier (earned positive match reward!)
    # Tier 3 match: r == +2.2; Tier 2 match: r == +1.3; Tier 1 match: r == +0.7; else Tier 0
    hit_t3 = np.any(focal_rewards > 2.0, axis=1)
    hit_t2 = (~hit_t3) & np.any((focal_rewards > 1.1) & (focal_rewards < 1.5), axis=1)
    hit_t1 = (~hit_t3) & (~hit_t2) & np.any((focal_rewards > 0.5) & (focal_rewards < 0.9), axis=1)
    hit_t0 = (~hit_t3) & (~hit_t2) & (~hit_t1)

    # Late-phase action shares (t >= 16)
    late_r = focal_rewards[:, min(16, horizon - 1) :]
    metrics = {
        "total_mean": float(np.mean(ep_tot)),
        "total_median": float(q50),
        "total_iqr": float(q75 - q25),
        "opp_total_mean": float(np.mean(ep_opp)),
        "early_auc_1_8": float(np.mean(focal_rewards[:, : min(8, horizon)])),
        "late_auc_17_end": float(np.mean(late_r)),
        "tier3_unlock_rate": float(np.mean(hit_t3)),
        "tier2_max_rate": float(np.mean(hit_t2)),
        "tier1_max_rate": float(np.mean(hit_t1)),
        "tier0_only_rate": float(np.mean(hit_t0)),
        "t3_match_turn_share_late": float(np.mean(late_r > 2.0)),
        "t2_match_turn_share_late": float(np.mean((late_r > 1.1) & (late_r < 1.5))),
        "t1_match_turn_share_late": float(np.mean((late_r > 0.5) & (late_r < 0.9))),
    }
    return metrics, {"total_return": ep_tot, "tier3_unlock": hit_t3.astype(np.float64), "opp_total_return": ep_opp}


class _MBNeuralPolicy:
    def __init__(self, model: MultiBarrierQNet, variant: str, horizon: int, mask_k: int) -> None:
        self.model = model
        self.variant = variant
        self.horizon = horizon
        self.mask_k = mask_k
        self._h: torch.Tensor | None = None

    def reset(self, batch_size: int, seed: int) -> None:
        self._h = torch.zeros((1, batch_size, self.model.hidden_dim), dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        assert self._h is not None
        x = encode_mb_obs_batch(obs, horizon=self.horizon, step_t=step_t, variant=self.variant, mask_k=self.mask_k)
        qv, self._h = self.model.step(torch.from_numpy(x).unsqueeze(1), self._h)
        return qv.argmax(dim=-1).numpy().astype(np.int64)


class _MBOnlineUCBPolicy:
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = float(beta)

    def reset(self, batch_size: int, seed: int) -> None:
        self._b = batch_size
        self._rng = np.random.default_rng(seed)
        self._sum = np.zeros((batch_size, MB_N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, MB_N_ACTIONS), dtype=np.float64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        if step_t > 0:
            pa = obs["prev_self_action"]
            pr = obs["prev_self_reward"]
            idx = np.arange(self._b)
            self._sum[idx, pa] += pr
            self._cnt[idx, pa] += 1.0
        means = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1.0), 0.0)
        bonus = self.beta * np.sqrt(math.log(step_t + 2.0) / (self._cnt + 1.0))
        return np.argmax(means + bonus + self._rng.uniform(0.0, 1e-9, size=means.shape), axis=-1).astype(np.int64)


class _MBGreedyPolicy:
    def reset(self, batch_size: int, seed: int) -> None:
        self._b = batch_size
        self._rng = np.random.default_rng(seed)
        self._sum = np.zeros((batch_size, MB_N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, MB_N_ACTIONS), dtype=np.float64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        if step_t > 0:
            pa = obs["prev_self_action"]
            pr = obs["prev_self_reward"]
            idx = np.arange(self._b)
            self._sum[idx, pa] += pr
            self._cnt[idx, pa] += 1.0
        means = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1.0), 0.0)
        return np.argmax(means + self._rng.uniform(0.0, 1e-9, size=means.shape), axis=-1).astype(np.int64)


class _MBBayesOptimalPolicy:
    """Privileged exact rational DP ceiling."""

    def __init__(self, horizon: int = 64) -> None:
        self.horizon = horizon
        self.solver = ExactMultiBarrierDP(max_tier=3)
        self._f_tbl = build_mb_response_table()
        self._cache: dict[tuple[int, int, int], int] = {}

    def reset(self, batch_size: int, seed: int) -> None:
        self._b = batch_size
        self._masks = np.full(batch_size, MB_FULL_BELIEF_MASK, dtype=np.int64)
        self._prev_q = np.full(batch_size, MB_ACT_S, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        q_cur = obs["prev_self_action"]
        if step_t > 0:
            py = obs["prev_opp_action"]
            for i in range(self._b):
                m, pq, yv = int(self._masks[i]), int(self._prev_q[i]), int(py[i])
                nm = 0
                for th in range(MB_N_TYPES):
                    if ((m >> th) & 1) and int(self._f_tbl[th, pq]) == yv:
                        nm |= 1 << th
                if nm != 0:
                    self._masks[i] = nm
        self._prev_q = q_cur.copy()
        rem = self.horizon - step_t
        acts = np.zeros(self._b, dtype=np.int64)
        for i in range(self._b):
            key = (rem, int(q_cur[i]), int(self._masks[i]))
            if key not in self._cache:
                self._cache[key] = self.solver.optimal_action(*key)
            acts[i] = self._cache[key]
        return acts


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


MB_NEURAL_SPECS = {
    "drqn_aux_discovery": ("rl2", "discovery"),
    "drqn_aux_uniform": ("rl2", "uniform"),
    "drqn_aux_yr": ("rl2", "yr"),
    "drqn_aux_y": ("rl2", "y"),
    "drqn_aux_r": ("rl2", "r"),
    "drqn_vanilla": ("rl2", "none"),
    "drqn_aux_yr_masked": ("rl2_masked", "yr"),
    "drqn_prior_only": ("prior_only", "none"),
}


def _run_or_load_mb_unit(out_dir_str: str, method: str, seed: int, cfg: dict) -> tuple[str, int, dict, dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    fit_dir = out_dir / "fits" / f"{method}_seed{seed}"
    receipt_path = fit_dir / "receipt.json"
    eval_path = fit_dir / "eval_metrics.json"
    arr_path = fit_dir / "eval_arrays.npz"

    horizon = int(cfg.get("horizon", 64))
    mask_k = int(cfg.get("mask_k", 8))
    n_eval = int(cfg.get("n_eval_episodes", 512))
    eval_seed = int(cfg.get("eval_seed_base", 99500)) + seed * 1009

    if receipt_path.exists() and eval_path.exists() and arr_path.exists():
        rec = json.loads(receipt_path.read_text(encoding="utf-8"))
        if rec.get("arrays_sha256") == _sha256_file(arr_path):
            md = json.loads(eval_path.read_text(encoding="utf-8"))
            npz = np.load(arr_path)
            arrs = {k: npz[k].astype(np.float64) for k in ("total_return", "tier3_unlock", "opp_total_return")}
            return method, seed, {"receipt": rec, "metrics": md}, arrs

    fit_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    off_trans = 0
    extra_outer = 0
    final_gates = [0.0, 0.0, 0.0, 0.0]
    is_priv = False
    ckpt_sha = None

    if method in MB_NEURAL_SPECS:
        variant, aux_strat = MB_NEURAL_SPECS[method]
        t_cfg = MBTrainConfig(
            method=method,
            variant=variant,
            aux_strategy=aux_strat,
            mask_k=mask_k,
            horizon=horizon,
            hidden_dim=int(cfg.get("hidden_dim", 96)),
            batch_episodes=int(cfg.get("batch_episodes", 48)),
            n_iters=int(cfg.get("n_iters", 450)),
            warmup_iters=int(cfg.get("warmup_iters", 90)),
            replay_capacity=int(cfg.get("replay_capacity", 6144)),
            grad_steps_per_iter=int(cfg.get("grad_steps_per_iter", 6)),
            minibatch_episodes=int(cfg.get("minibatch_episodes", 96)),
            gamma=float(cfg.get("gamma", 0.99)),
            lr=float(cfg.get("lr", 3e-3)),
            polyak_tau=float(cfg.get("polyak_tau", 0.05)),
            aux_scale=float(cfg.get("aux_scale", 2.0)),
            meta_gate_lr=float(cfg.get("meta_gate_lr", 0.35)),
            log_interval=int(cfg.get("log_interval", 25)),
            seed=seed,
            device=str(cfg.get("device", "cpu")),
        )
        qnet, curve, off_trans, extra_outer, final_gates = train_mb_drqn(t_cfg)
        ckpt_path = fit_dir / "checkpoint.pt"
        torch.save({"state_dict": qnet.state_dict(), "config": t_cfg.__dict__, "final_gates": final_gates}, ckpt_path)
        ckpt_sha = _sha256_file(ckpt_path)
        _write_csv(fit_dir / "train_curve.csv", curve)
        pol = _MBNeuralPolicy(qnet, variant=variant, horizon=horizon, mask_k=mask_k)
    elif method == "arm_ucb":
        pol = _MBOnlineUCBPolicy(beta=1.0)
    elif method == "greedy":
        pol = _MBGreedyPolicy()
    elif method == "bayes_optimal":
        pol = _MBBayesOptimalPolicy(horizon=horizon)
        is_priv = True
    else:
        raise ValueError(f"Unknown Exp 4 method: {method}")

    md, arrs = rollout_mb_policy(pol, n_episodes=n_eval, horizon=horizon, seed=eval_seed)
    np.savez_compressed(arr_path, **{k: v.astype(np.float64) for k, v in arrs.items()})
    _atomic_write_json(eval_path, md)
    rec = {
        "method": method,
        "seed": int(seed),
        "is_privileged": is_priv,
        "offline_train_transitions": int(off_trans),
        "extra_outer_meta_steps": int(extra_outer),
        "final_gates": [round(x, 4) for x in final_gates],
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "checkpoint_sha256": ckpt_sha,
        "arrays_sha256": _sha256_file(arr_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(receipt_path, rec)
    return method, seed, {"receipt": rec, "metrics": md}, arrs


def run_study_exp4(cfg: dict, workers: int = 1) -> dict:
    study_id = str(cfg["study_id"])
    out_dir = Path(cfg.get("output_dir", f"runs/{study_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [str(m) for m in cfg["methods"]]
    seeds = [int(s) for s in cfg["seeds"]]
    horizon = int(cfg.get("horizon", 64))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta = float(cfg.get("delta_return", 0.30))

    gate0 = run_mb_gate0_report(horizons=(6, 10, 16, 24, 48, horizon))
    manifest = {
        "study_id": study_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gate0": gate0,
        "config": cfg,
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)

    jobs = [(m, s) for m in methods for s in seeds]
    collected: dict[tuple[str, int], tuple[dict, dict[str, np.ndarray]]] = {}
    if workers <= 1:
        for m, s in jobs:
            m_, s_, rec, arrs = _run_or_load_mb_unit(str(out_dir), m, s, cfg)
            collected[(m_, s_)] = (rec, arrs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_or_load_mb_unit, str(out_dir), m, s, cfg) for m, s in jobs]
            done = 0
            for fut in as_completed(futs):
                m_, s_, rec, arrs = fut.result()
                collected[(m_, s_)] = (rec, arrs)
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print(f"  [{study_id}] progress: {done}/{len(jobs)} units done", flush=True)

    agg_methods: dict[str, dict] = {}
    arrs_by_m: dict[str, dict[str, np.ndarray]] = {}
    method_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for m in methods:
        recs = [collected[(m, s)][0] for s in seeds]
        arrs = {
            k: np.stack([collected[(m, s)][1][k] for s in seeds], axis=0)
            for k in ("total_return", "tier3_unlock", "opp_total_return")
        }
        arrs_by_m[m] = arrs
        seed_means = np.mean(arrs["total_return"], axis=1)
        gates_arr = np.array([r["receipt"]["final_gates"] for r in recs], dtype=np.float64)
        mean_gates = np.mean(gates_arr, axis=0)
        agg = {
            "is_privileged": recs[0]["receipt"]["is_privileged"],
            "offline_train_transitions": recs[0]["receipt"]["offline_train_transitions"],
            "extra_outer_meta_steps": recs[0]["receipt"]["extra_outer_meta_steps"],
            "total_mean": float(np.mean(seed_means)),
            "total_seed_sd": float(np.std(seed_means, ddof=1)) if len(seeds) > 1 else 0.0,
            "total_median": float(np.median(arrs["total_return"])),
            "total_iqr": float(np.percentile(arrs["total_return"], 75) - np.percentile(arrs["total_return"], 25)),
            "opp_total_mean": float(np.mean(arrs["opp_total_return"])),
            "early_auc_1_8": float(np.mean([r["metrics"]["early_auc_1_8"] for r in recs])),
            "late_auc_17_end": float(np.mean([r["metrics"]["late_auc_17_end"] for r in recs])),
            "tier3_unlock_rate": float(np.mean([r["metrics"]["tier3_unlock_rate"] for r in recs])),
            "tier2_max_rate": float(np.mean([r["metrics"]["tier2_max_rate"] for r in recs])),
            "tier1_max_rate": float(np.mean([r["metrics"]["tier1_max_rate"] for r in recs])),
            "tier0_only_rate": float(np.mean([r["metrics"]["tier0_only_rate"] for r in recs])),
            "final_gates_mean": {CAND_NAMES[i]: round(float(mean_gates[i]), 4) for i in range(N_CANDS)},
        }
        agg_methods[m] = agg
        method_rows.append(
            {
                "method": m,
                "privileged": agg["is_privileged"],
                "offline_transitions": agg["offline_train_transitions"],
                "outer_meta_steps": agg["extra_outer_meta_steps"],
                "total_mean": round(agg["total_mean"], 4),
                "total_seed_sd": round(agg["total_seed_sd"], 4),
                "total_median": round(agg["total_median"], 4),
                "total_iqr": round(agg["total_iqr"], 4),
                "opp_total_mean": round(agg["opp_total_mean"], 4),
                "early_auc_1_8": round(agg["early_auc_1_8"], 4),
                "tier3_unlock": round(agg["tier3_unlock_rate"], 4),
                "tier2_max": round(agg["tier2_max_rate"], 4),
                "tier1_max": round(agg["tier1_max_rate"], 4),
                "tier0_only": round(agg["tier0_only_rate"], 4),
                "gate_y": round(float(mean_gates[0]), 4),
                "gate_r": round(float(mean_gates[1]), 4),
                "gate_noise1": round(float(mean_gates[2]), 4),
                "gate_noise2": round(float(mean_gates[3]), 4),
            }
        )

    contrasts: dict[str, dict] = {}
    for a, b in (
        ("drqn_aux_discovery", "drqn_aux_uniform"),
        ("drqn_aux_discovery", "drqn_vanilla"),
        ("drqn_aux_discovery", "drqn_aux_yr"),
        ("drqn_aux_discovery", "drqn_aux_r"),
        ("drqn_aux_yr", "drqn_vanilla"),
        ("drqn_aux_yr", "drqn_aux_yr_masked"),
        ("drqn_aux_yr", "drqn_prior_only"),
        ("drqn_aux_yr", "drqn_aux_r"),
        ("drqn_aux_yr", "drqn_aux_y"),
        ("drqn_aux_y", "drqn_vanilla"),
        ("drqn_aux_discovery", "arm_ucb"),
        ("drqn_aux_discovery", "greedy"),
    ):
        if a not in arrs_by_m or b not in arrs_by_m:
            continue
        c_key = f"{a}_minus_{b}"
        res = hierarchical_paired_bootstrap(
            arrs_by_m[a]["total_return"], arrs_by_m[b]["total_return"], practical_threshold=delta, n_boot=n_boot, seed=99001
        )
        contrasts[c_key] = res
        contrast_rows.append(
            {
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

    summary = {
        "study_id": study_id,
        "git_commit": manifest["git_commit"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon,
        "seeds": seeds,
        "gate0": gate0,
        "methods": agg_methods,
        "paired_contrasts": contrasts,
    }
    _atomic_write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "tables" / "methods.csv", method_rows)
    _write_csv(out_dir / "tables" / "paired_contrasts.csv", contrast_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 4 Multi-Barrier (T=64) + Auxiliary Task Discovery.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 1))
    t0 = time.perf_counter()
    summary = run_study_exp4(cfg, workers=workers)
    g = summary["gate0"][f"T={summary['horizon']}"]
    print(f"[{cfg['study_id']}] Complete in {time.perf_counter() - t0:.1f}s")
    print(f"=== T={summary['horizon']} Ceilings: T0={g['tier0_ceiling']:.2f} | T1={g['tier1_ceiling']:.2f} | T2={g['tier2_ceiling']:.2f} | T3(Opt)={g['tier3_ceiling']:.2f} ===")
    for m, md in summary["methods"].items():
        gs = md["final_gates_mean"]
        print(
            f"  {m:20s} | Ret={md['total_mean']:+7.2f} (sd={md['total_seed_sd']:5.2f}, med={md['total_median']:+7.2f}) "
            f"| Tiers(3/2/1/0)={md['tier3_unlock_rate']:.2f}/{md['tier2_max_rate']:.2f}/{md['tier1_max_rate']:.2f}/{md['tier0_only_rate']:.2f} "
            f"| gates(y/r/n1/n2)={gs['cand_y']:.2f}/{gs['cand_r']:.2f}/{gs['cand_noise1']:.2f}/{gs['cand_noise2']:.2f}"
        )


if __name__ == "__main__":
    main()
