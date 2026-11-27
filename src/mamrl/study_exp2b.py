"""SPEC v3.0 Secondary Study B (Exp 2B): Symmetric 2-Player 4-Action Competition & Cooperation.

Compares strictly unprivileged agents (zero access to payoff matrices) across:
- Non-Meta vs. Non-Meta: q_learner, ucb, contextual_ucb
- Meta vs. Non-Meta: meta_drqn_aux_yr, meta_drqn_vanilla, meta_drqn_masked vs Non-Meta learners
- Meta vs. Meta: cross-seed independent partners (seed_i vs seed_{(i+1)%S}), same-seed self-play,
  and direct head-to-head Meta ablations (aux_yr vs vanilla, aux_yr vs masked)

Evaluated on BOTH:
- in_dist (train rule split with fresh paired seeds)
- unseen_rules (strictly disjoint held-out test rule split, balanced across all 4 action slots)

Usage:
    PYTHONPATH=src .venv/bin/python -m mamrl.study_exp2b --config configs/exp2b_symmetric_ma.json --workers 6
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
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

from mamrl.envs.symmetric_ma import SYM_N_ACTIONS, VectorizedSymmetricGameEnv, build_symmetric_rule_bank
from mamrl.eval.audit_v3 import hierarchical_paired_bootstrap

SYM_OBS_DIM = SYM_N_ACTIONS + SYM_N_ACTIONS + 3  # 4 + 4 + 3 = 11

META_METHODS = {
    "meta_drqn_aux_yr": ("rl2", "yr"),
    "meta_drqn_vanilla": ("rl2", "none"),
    "meta_drqn_masked": ("rl2_masked", "yr"),
}


def encode_sym_obs_batch(
    obs: dict[str, np.ndarray],
    *,
    horizon: int,
    step_t: int,
    variant: str = "rl2",
    mask_k: int = 4,
) -> np.ndarray:
    """Encode public symmetric-game observation into float32 (B, 11)."""
    b = int(obs["prev_self_action"].shape[0])
    x = np.zeros((b, SYM_OBS_DIM), dtype=np.float32)
    idx = np.arange(b)

    pa = obs["prev_self_action"]
    valid_a = (pa >= 0) & (pa < SYM_N_ACTIONS)
    x[idx[valid_a], pa[valid_a]] = 1.0

    allow_feedback = (variant == "rl2") or (variant == "rl2_masked" and step_t >= mask_k)
    if allow_feedback:
        po = obs["prev_opp_action"]
        valid_o = (po >= 0) & (po < SYM_N_ACTIONS)
        x[idx[valid_o], SYM_N_ACTIONS + po[valid_o]] = 1.0
        x[:, 2 * SYM_N_ACTIONS] = obs["prev_self_reward"].astype(np.float32)

    x[:, 2 * SYM_N_ACTIONS + 1] = obs["is_start"].astype(np.float32)
    x[:, 2 * SYM_N_ACTIONS + 2] = float(step_t) / max(1.0, float(horizon))
    return x


class SymmetricRecurrentQNet(nn.Module):
    """GRU-64 Recurrent Q-Network with optional self-supervised next-(opp_a, reward) heads."""

    def __init__(self, obs_dim: int = SYM_OBS_DIM, hidden_dim: int = 64, n_actions: int = SYM_N_ACTIONS) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, n_actions)
        self.pred_opp = nn.Linear(hidden_dim + n_actions, n_actions)
        self.pred_r = nn.Linear(hidden_dim + n_actions, 1)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x_seq)
        return self.q_head(out), out

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_next = self.gru(x_t, h)
        return self.q_head(out.squeeze(1)), h_next


class SymPolicy(ABC):
    """Unified symmetric-player batch policy interface (strictly unprivileged)."""

    name: str = "sym_policy"

    @abstractmethod
    def reset(self, batch_size: int, seed: int) -> None:
        ...

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        ...


class SymRandomPolicy(SymPolicy):
    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        return self._rng.integers(0, SYM_N_ACTIONS, size=self._batch, dtype=np.int64)


class SymUCBPolicy(SymPolicy):
    """Unprivileged online arm-independent UCB1 learner."""

    name = "ucb"

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = float(beta)
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        self._sum = np.zeros((batch_size, SYM_N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, SYM_N_ACTIONS), dtype=np.float64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        if step_t > 0:
            pa = obs["prev_self_action"]
            pr = obs["prev_self_reward"]
            idx = np.arange(self._batch)
            self._sum[idx, pa] += pr
            self._cnt[idx, pa] += 1.0
        means = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1.0), 0.0)
        bonus = self.beta * np.sqrt(math.log(step_t + 2.0) / (self._cnt + 1.0))
        noise = self._rng.uniform(0.0, 1e-9, size=means.shape)
        return np.argmax(means + bonus + noise, axis=-1).astype(np.int64)


class SymContextualUCBPolicy(SymPolicy):
    """Unprivileged online contextual UCB conditioned on opponent's last action (state in {0..4})."""

    name = "contextual_ucb"

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = float(beta)
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        # 5 contexts: 0..3 = opponent's previous action, 4 = start
        self._sum = np.zeros((batch_size, SYM_N_ACTIONS + 1, SYM_N_ACTIONS), dtype=np.float64)
        self._cnt = np.zeros((batch_size, SYM_N_ACTIONS + 1, SYM_N_ACTIONS), dtype=np.float64)
        self._prev_ctx = np.full(batch_size, SYM_N_ACTIONS, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        idx = np.arange(self._batch)
        if step_t > 0:
            pa = obs["prev_self_action"]
            pr = obs["prev_self_reward"]
            self._sum[idx, self._prev_ctx, pa] += pr
            self._cnt[idx, self._prev_ctx, pa] += 1.0
            ctx = obs["prev_opp_action"]
        else:
            ctx = np.full(self._batch, SYM_N_ACTIONS, dtype=np.int64)
        self._prev_ctx = ctx.copy()

        s_c = self._sum[idx, ctx]
        c_c = self._cnt[idx, ctx]
        means = np.where(c_c > 0, s_c / np.maximum(c_c, 1.0), 0.0)
        bonus = self.beta * np.sqrt(math.log(step_t + 2.0) / (c_c + 1.0))
        noise = self._rng.uniform(0.0, 1e-9, size=means.shape)
        return np.argmax(means + bonus + noise, axis=-1).astype(np.int64)


class SymQLearningPolicy(SymPolicy):
    """Unprivileged online tabular Q-learner conditioned on opponent's previous action."""

    name = "q_learner"

    def __init__(self, alpha: float = 0.35, gamma: float = 0.50, epsilon: float = 0.12) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int) -> None:
        self._batch = batch_size
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros((batch_size, SYM_N_ACTIONS + 1, SYM_N_ACTIONS), dtype=np.float64)
        self._prev_ctx = np.full(batch_size, SYM_N_ACTIONS, dtype=np.int64)

    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        idx = np.arange(self._batch)
        if step_t > 0:
            pa = obs["prev_self_action"]
            po = obs["prev_opp_action"]
            pr = obs["prev_self_reward"]
            next_max = np.max(self._q[idx, po, :], axis=-1)
            cur_q = self._q[idx, self._prev_ctx, pa]
            self._q[idx, self._prev_ctx, pa] = cur_q + self.alpha * (pr + self.gamma * next_max - cur_q)
            ctx = po
        else:
            ctx = np.full(self._batch, SYM_N_ACTIONS, dtype=np.int64)
        self._prev_ctx = ctx.copy()

        q_vals = self._q[idx, ctx, :]
        noise = self._rng.uniform(0.0, 1e-9, size=q_vals.shape)
        greedy = np.argmax(q_vals + noise, axis=-1).astype(np.int64)
        rand_a = self._rng.integers(0, SYM_N_ACTIONS, size=self._batch, dtype=np.int64)
        return np.where(self._rng.random(self._batch) < self.epsilon, rand_a, greedy).astype(np.int64)


class SymNeuralDRQNPolicy(SymPolicy):
    """Evaluation wrapper for meta-trained SymmetricRecurrentQNet."""

    def __init__(
        self,
        model: SymmetricRecurrentQNet,
        *,
        name: str,
        variant: str,
        horizon: int = 24,
        mask_k: int = 4,
        eps_eval: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.name = name
        self.variant = variant
        self.horizon = int(horizon)
        self.mask_k = int(mask_k)
        self.eps_eval = float(eps_eval)
        self.device = torch.device(device)
        self._h: torch.Tensor | None = None
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int) -> None:
        self._batch = batch_size
        self._h = torch.zeros((1, batch_size, self.model.hidden_dim), dtype=torch.float32, device=self.device)
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        assert self._h is not None
        x_np = encode_sym_obs_batch(
            obs,
            horizon=self.horizon,
            step_t=step_t,
            variant=self.variant,
            mask_k=self.mask_k,
        )
        qv, self._h = self.model.step(torch.from_numpy(x_np).to(self.device).unsqueeze(1), self._h)
        greedy = qv.argmax(dim=-1).cpu().numpy().astype(np.int64)
        if self.eps_eval <= 0.0:
            return greedy
        rand_a = self._rng.integers(0, SYM_N_ACTIONS, size=self._batch, dtype=np.int64)
        return np.where(self._rng.random(self._batch) < self.eps_eval, rand_a, greedy).astype(np.int64)


@dataclass
class SymDRQNConfig:
    game: str = "coop_trap"
    variant: str = "rl2"       # "rl2", "rl2_masked"
    aux_mode: str = "yr"       # "none", "yr"
    mask_k: int = 4
    horizon: int = 24
    hidden_dim: int = 64
    batch_episodes: int = 64
    n_iters: int = 400
    warmup_iters: int = 60
    replay_capacity: int = 8192
    grad_steps_per_iter: int = 8
    minibatch_episodes: int = 128
    gamma: float = 1.0
    lr: float = 3e-3
    polyak_tau: float = 0.05
    aux_opp_weight: float = 1.5
    aux_r_weight: float = 1.0
    eps_start: float = 0.85
    eps_end: float = 0.05
    max_grad_norm: float = 5.0
    log_interval: int = 20
    seed: int = 0
    device: str = "cpu"


def train_sym_meta_drqn(cfg: SymDRQNConfig) -> tuple[SymmetricRecurrentQNet, list[dict[str, float]], int]:
    """Meta-train SymmetricRecurrentQNet strictly on the 'train' rule split against a mixed partner pool.

    Each batch of B episodes pairs the focal Meta agent with:
    - 50% Polyak target self-play (tnet, small exploration eps=0.05) -> teaches Meta vs Meta adaptation
    - 25% Online tabular Q-learner (fresh per episode)
    - 25% Online UCB1 learner (fresh per episode)
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    qnet = SymmetricRecurrentQNet(SYM_OBS_DIM, cfg.hidden_dim, SYM_N_ACTIONS).to(device)
    tnet = SymmetricRecurrentQNet(SYM_OBS_DIM, cfg.hidden_dim, SYM_N_ACTIONS).to(device)
    tnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)

    b = cfg.batch_episodes
    t_len = cfg.horizon
    n_self = b // 2
    n_ql = b // 4
    n_ucb = b - n_self - n_ql

    env = VectorizedSymmetricGameEnv(cfg.game, split="train", batch_size=b, horizon=t_len, seed=cfg.seed + 17)
    ql_opp = SymQLearningPolicy(alpha=0.35, gamma=0.50, epsilon=0.12)
    ucb_opp = SymUCBPolicy(beta=1.0)
    rng = np.random.default_rng(cfg.seed + 31)

    cap = cfg.replay_capacity
    buf_obs = np.zeros((cap, t_len, SYM_OBS_DIM), dtype=np.float32)
    buf_act = np.zeros((cap, t_len), dtype=np.int64)
    buf_opp = np.zeros((cap, t_len), dtype=np.int64)
    buf_rew = np.zeros((cap, t_len), dtype=np.float32)
    ptr, size = 0, 0

    total_transitions = 0
    train_curve: list[dict[str, float]] = []

    for it in range(1, cfg.n_iters + 1):
        if it <= cfg.warmup_iters:
            eps = 1.0
        else:
            decay_span = max(1.0, 0.8 * float(cfg.n_iters - cfg.warmup_iters))
            frac = float(it - cfg.warmup_iters) / decay_span
            eps = max(cfg.eps_end, cfg.eps_start * (1.0 - frac))

        obs1, obs2 = env.reset()
        ql_opp.reset(n_ql, seed=cfg.seed + it * 101 + 1)
        ucb_opp.reset(n_ucb, seed=cfg.seed + it * 101 + 2)

        h1 = torch.zeros((1, b, cfg.hidden_dim), dtype=torch.float32, device=device)
        h2_self = torch.zeros((1, n_self, cfg.hidden_dim), dtype=torch.float32, device=device)

        ep_o = np.zeros((b, t_len, SYM_OBS_DIM), dtype=np.float32)
        ep_a = np.zeros((b, t_len), dtype=np.int64)
        ep_opp = np.zeros((b, t_len), dtype=np.int64)
        ep_r = np.zeros((b, t_len), dtype=np.float32)

        qnet.eval()
        tnet.eval()
        with torch.no_grad():
            for t in range(t_len):
                x1 = encode_sym_obs_batch(obs1, horizon=t_len, step_t=t, variant=cfg.variant, mask_k=cfg.mask_k)
                ep_o[:, t] = x1
                qv1, h1 = qnet.step(torch.from_numpy(x1).to(device).unsqueeze(1), h1)
                greedy1 = qv1.argmax(dim=-1).cpu().numpy()
                rand1 = rng.integers(0, SYM_N_ACTIONS, size=b)
                a1 = np.where(rng.random(b) < eps, rand1, greedy1).astype(np.int64)

                # Partner actions across 3 slices: [0:n_self] target self-play, [n_self:n_self+n_ql] Q-learner, rest UCB
                a2 = np.zeros(b, dtype=np.int64)
                obs2_self = {k: v[:n_self] for k, v in obs2.items()}
                x2_self = encode_sym_obs_batch(obs2_self, horizon=t_len, step_t=t, variant=cfg.variant, mask_k=cfg.mask_k)
                qv2, h2_self = tnet.step(torch.from_numpy(x2_self).to(device).unsqueeze(1), h2_self)
                g2 = qv2.argmax(dim=-1).cpu().numpy()
                r2_rand = rng.integers(0, SYM_N_ACTIONS, size=n_self)
                a2[:n_self] = np.where(rng.random(n_self) < 0.05, r2_rand, g2)

                obs2_ql = {k: v[n_self : n_self + n_ql] for k, v in obs2.items()}
                a2[n_self : n_self + n_ql] = ql_opp.act(obs2_ql, step_t=t)

                obs2_ucb = {k: v[n_self + n_ql :] for k, v in obs2.items()}
                a2[n_self + n_ql :] = ucb_opp.act(obs2_ucb, step_t=t)

                obs1, obs2, r1, _r2 = env.step(a1, a2)
                ep_a[:, t] = a1
                ep_opp[:, t] = a2
                ep_r[:, t] = r1.astype(np.float32)

        total_transitions += b * t_len
        for k in range(b):
            buf_obs[ptr] = ep_o[k]
            buf_act[ptr] = ep_a[k]
            buf_opp[ptr] = ep_opp[k]
            buf_rew[ptr] = ep_r[k]
            ptr = (ptr + 1) % cap
            size = min(cap, size + 1)

        qnet.train()
        td_vals, ao_vals, ar_vals = [], [], []
        mb_n = min(cfg.minibatch_episodes, size)
        start_t = cfg.mask_k if cfg.variant == "rl2_masked" else 0

        for _ in range(cfg.grad_steps_per_iter):
            idx = rng.integers(0, size, size=mb_n)
            ox = torch.from_numpy(buf_obs[idx]).to(device)
            ax = torch.from_numpy(buf_act[idx]).to(device)
            oppx = torch.from_numpy(buf_opp[idx]).to(device)
            rx = torch.from_numpy(buf_rew[idx]).to(device)

            q_all, hs = qnet(ox)
            q_sa = q_all.gather(-1, ax.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_a = q_all[:, 1:, :].argmax(dim=-1, keepdim=True)
                tq, _ = tnet(ox)
                next_q = tq[:, 1:, :].gather(-1, next_a).squeeze(-1)
                tgt = rx.clone()
                tgt[:, :-1] += cfg.gamma * next_q

            td_loss = nn.functional.mse_loss(q_sa, tgt)
            if cfg.aux_mode == "yr" and start_t < t_len:
                a_oh = nn.functional.one_hot(ax[:, start_t:], num_classes=SYM_N_ACTIONS).float()
                ha = torch.cat([hs[:, start_t:, :], a_oh], dim=-1)
                opp_loss = nn.functional.cross_entropy(qnet.pred_opp(ha).reshape(-1, SYM_N_ACTIONS), oppx[:, start_t:].reshape(-1))
                r_loss = nn.functional.mse_loss(qnet.pred_r(ha).squeeze(-1), rx[:, start_t:])
            else:
                opp_loss = torch.tensor(0.0, device=device)
                r_loss = torch.tensor(0.0, device=device)

            loss = td_loss + cfg.aux_opp_weight * opp_loss + cfg.aux_r_weight * r_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                tau = cfg.polyak_tau
                for p, tp in zip(qnet.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

            td_vals.append(float(td_loss.item()))
            ao_vals.append(float(opp_loss.item()))
            ar_vals.append(float(r_loss.item()))

        if it == 1 or it == cfg.n_iters or (it % cfg.log_interval == 0):
            ep_tot = ep_r.sum(axis=1)
            train_curve.append(
                {
                    "update": float(it),
                    "transitions": float(total_transitions),
                    "eps": float(eps),
                    "mean_return": float(np.mean(ep_tot)),
                    "median_return": float(np.median(ep_tot)),
                    "td_loss": float(np.mean(td_vals)),
                    "aux_opp_loss": float(np.mean(ao_vals)),
                    "aux_r_loss": float(np.mean(ar_vals)),
                }
            )

    qnet.eval()
    return qnet, train_curve, total_transitions


def rollout_symmetric_matchup(
    p1: SymPolicy,
    p2: SymPolicy,
    *,
    game: str,
    split: str,
    n_episodes: int = 512,
    horizon: int = 24,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate paired 2-player symmetric matchup on balanced rule_ids across split ('train' or 'test')."""
    env = VectorizedSymmetricGameEnv(game=game, split=split, batch_size=n_episodes, horizon=horizon, seed=seed)
    n_rules = env.n_rules
    if n_episodes % n_rules != 0:
        raise ValueError(f"n_episodes ({n_episodes}) must be divisible by n_rules ({n_rules}) for {game}/{split}.")
    rule_ids = np.tile(np.arange(n_rules, dtype=np.int64), n_episodes // n_rules)

    obs1, obs2 = env.reset(rule_ids=rule_ids)
    p1.reset(n_episodes, seed=seed + 13)
    p2.reset(n_episodes, seed=seed + 29)

    a1_hist = np.zeros((n_episodes, horizon), dtype=np.int64)
    a2_hist = np.zeros((n_episodes, horizon), dtype=np.int64)
    r1_hist = np.zeros((n_episodes, horizon), dtype=np.float64)
    r2_hist = np.zeros((n_episodes, horizon), dtype=np.float64)

    for t in range(horizon):
        a1 = p1.act(obs1, step_t=t)
        a2 = p2.act(obs2, step_t=t)
        obs1, obs2, r1, r2 = env.step(a1, a2)
        a1_hist[:, t] = a1
        a2_hist[:, t] = a2
        r1_hist[:, t] = r1
        r2_hist[:, t] = r2

    p1_tot = r1_hist.sum(axis=1)
    p2_tot = r2_hist.sum(axis=1)
    welfare = p1_tot + p2_tot
    margin = p1_tot - p2_tot
    k_p = min(4, horizon)

    q25, q50, q75 = np.percentile(p1_tot, [25, 50, 75])
    metrics: dict[str, float] = {
        "p1_mean": float(np.mean(p1_tot)),
        "p1_median": float(q50),
        "p1_iqr": float(q75 - q25),
        "p2_mean": float(np.mean(p2_tot)),
        "p2_median": float(np.median(p2_tot)),
        "welfare_mean": float(np.mean(welfare)),
        "welfare_median": float(np.median(welfare)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "p1_win_ep_rate": float(np.mean(margin > 1e-9)),
        "p1_tie_ep_rate": float(np.mean(np.abs(margin) <= 1e-9)),
        "p1_loss_ep_rate": float(np.mean(margin < -1e-9)),
        "p1_early_auc_1_4": float(np.mean(r1_hist[:, :k_p])),
        "p1_mid_auc_5_12": float(np.mean(r1_hist[:, 4 : min(12, horizon)])) if horizon > 4 else 0.0,
        "p1_late_auc_13_end": float(np.mean(r1_hist[:, 12:])) if horizon > 12 else 0.0,
        "p1_rep_explore": float(np.mean(a1_hist[:, 1:k_p] == a1_hist[:, : k_p - 1])) if k_p > 1 else 0.0,
        "p1_rep_exploit": float(np.mean(a1_hist[:, k_p:] == a1_hist[:, k_p - 1 : -1])) if horizon > k_p else 0.0,
    }

    if game == "coop_trap":
        coop_a = env.bank["coop"][rule_ids][:, None]
        safe_a = env.bank["safe"][rule_ids][:, None]
        trap_a = env.bank["trap"][rule_ids][:, None]
        mc = (a1_hist == coop_a) & (a2_hist == coop_a)
        ms = (a1_hist == safe_a) & (a2_hist == safe_a)
        tr = (a1_hist == trap_a) | (a2_hist == trap_a)
        metrics["mutual_coop_early"] = float(np.mean(mc[:, :k_p]))
        metrics["mutual_coop_late"] = float(np.mean(mc[:, k_p:])) if horizon > k_p else 0.0
        metrics["mutual_safe_late"] = float(np.mean(ms[:, k_p:])) if horizon > k_p else 0.0
        metrics["trap_rate_all"] = float(np.mean(tr))
        metrics["p1_win_turn_late"] = 0.0
        metrics["p1_anchor_late"] = 0.0
        metrics["p1_entropy_late"] = 0.0
    else:
        anchor_a = env.bank["anchor"][rule_ids][:, None]
        late_r1 = r1_hist[:, k_p:] if horizon > k_p else r1_hist
        late_a1 = a1_hist[:, k_p:] if horizon > k_p else a1_hist
        metrics["p1_win_turn_late"] = float(np.mean(late_r1 > 1e-9))
        metrics["p1_anchor_late"] = float(np.mean(late_a1 == anchor_a))
        # Per-episode late action entropy averaged across episodes
        ents = []
        for i in range(n_episodes):
            p = np.bincount(late_a1[i], minlength=SYM_N_ACTIONS) / max(1.0, float(late_a1.shape[1]))
            p_nz = p[p > 0]
            ents.append(float(-np.sum(p_nz * np.log2(p_nz))))
        metrics["p1_entropy_late"] = float(np.mean(ents))
        metrics["mutual_coop_early"] = 0.0
        metrics["mutual_coop_late"] = 0.0
        metrics["mutual_safe_late"] = 0.0
        metrics["trap_rate_all"] = 0.0

    arrays = {
        "p1_return": p1_tot,
        "p2_return": p2_tot,
        "welfare": welfare,
        "margin": margin,
    }
    return metrics, arrays


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


def _train_or_load_sym_fit(
    *,
    out_dir_str: str,
    game: str,
    method: str,
    seed: int,
    cfg: dict,
) -> tuple[str, str, int, dict]:
    """Train (or resume) one (game, meta_method, seed) fit strictly on train rules."""
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    fit_dir = out_dir / "fits" / f"{game}_{method}_seed{seed}"
    receipt_path = fit_dir / "receipt.json"
    ckpt_path = fit_dir / "checkpoint.pt"

    if receipt_path.exists() and ckpt_path.exists():
        rec = json.loads(receipt_path.read_text(encoding="utf-8"))
        if rec.get("checkpoint_sha256") == _sha256_file(ckpt_path):
            return game, method, seed, rec

    fit_dir.mkdir(parents=True, exist_ok=True)
    variant, aux_mode = META_METHODS[method]
    t0 = time.perf_counter()
    d_cfg = SymDRQNConfig(
        game=game,
        variant=variant,
        aux_mode=aux_mode,
        mask_k=int(cfg.get("primary_k", 4)),
        horizon=int(cfg.get("horizon", 24)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        batch_episodes=int(cfg.get("drqn_batch_episodes", 64)),
        n_iters=int(cfg.get("drqn_iters", 400)),
        warmup_iters=int(cfg.get("drqn_warmup_iters", 60)),
        replay_capacity=int(cfg.get("drqn_replay_capacity", 8192)),
        grad_steps_per_iter=int(cfg.get("drqn_grad_steps", 8)),
        minibatch_episodes=int(cfg.get("drqn_minibatch", 128)),
        gamma=float(cfg.get("gamma", 1.0)),
        lr=float(cfg.get("drqn_lr", 3e-3)),
        polyak_tau=float(cfg.get("drqn_polyak_tau", 0.05)),
        aux_opp_weight=float(cfg.get("aux_opp_weight", 1.5)),
        aux_r_weight=float(cfg.get("aux_r_weight", 1.0)),
        log_interval=int(cfg.get("drqn_log_interval", 20)),
        seed=seed,
        device=str(cfg.get("device", "cpu")),
    )
    qnet, curve, off_trans = train_sym_meta_drqn(d_cfg)
    torch.save({"state_dict": qnet.state_dict(), "config": d_cfg.__dict__}, ckpt_path)
    _write_csv(fit_dir / "train_curve.csv", curve)
    rec = {
        "game": game,
        "method": method,
        "seed": int(seed),
        "offline_train_transitions": int(off_trans),
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "checkpoint_sha256": _sha256_file(ckpt_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(receipt_path, rec)
    return game, method, seed, rec


def _load_sym_policy(
    out_dir: Path,
    game: str,
    name: str,
    seed: int,
    cfg: dict,
) -> SymPolicy:
    if name in META_METHODS:
        variant, _ = META_METHODS[name]
        ckpt_path = out_dir / "fits" / f"{game}_{name}_seed{seed}" / "checkpoint.pt"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        qnet = SymmetricRecurrentQNet(SYM_OBS_DIM, int(cfg.get("hidden_dim", 64)), SYM_N_ACTIONS)
        qnet.load_state_dict(payload["state_dict"])
        qnet.eval()
        return SymNeuralDRQNPolicy(
            qnet,
            name=name,
            variant=variant,
            horizon=int(cfg.get("horizon", 24)),
            mask_k=int(cfg.get("primary_k", 4)),
            eps_eval=0.0,
            device="cpu",
        )
    if name == "q_learner":
        return SymQLearningPolicy(alpha=0.35, gamma=0.50, epsilon=0.12)
    if name == "ucb":
        return SymUCBPolicy(beta=1.0)
    if name == "contextual_ucb":
        return SymContextualUCBPolicy(beta=1.0)
    if name == "random":
        return SymRandomPolicy()
    raise ValueError(f"Unknown symmetric policy: {name}")


def run_study_exp2b(cfg: dict, workers: int = 1) -> dict:
    """Train Meta agents on Train rules and evaluate full Non-Meta / Meta matchup matrix on Train & Unseen Test rules."""
    study_id = str(cfg["study_id"])
    out_dir = Path(cfg.get("output_dir", f"runs/{study_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    games = [str(g) for g in cfg.get("games", ["comp_rps", "coop_trap"])]
    meta_methods = [str(m) for m in cfg.get("meta_methods", ["meta_drqn_aux_yr", "meta_drqn_vanilla", "meta_drqn_masked"])]
    seeds = [int(s) for s in cfg["seeds"]]
    horizon = int(cfg.get("horizon", 24))
    n_eval = int(cfg.get("n_eval_episodes", 512))
    eval_seed_base = int(cfg.get("eval_seed_base", 97000))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta_return = float(cfg.get("delta_return", 0.30))

    rule_counts = {
        g: {
            "n_train_rules": int(build_symmetric_rule_bank(g)["train"]["n_rules"]),
            "n_unseen_test_rules": int(build_symmetric_rule_bank(g)["test"]["n_rules"]),
        }
        for g in games
    }
    manifest = {
        "study_id": study_id,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "rule_counts": rule_counts,
        "config": cfg,
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)

    # Phase 1: Parallel Meta-Training on Train Rules ONLY
    jobs = [(g, m, s) for g in games for m in meta_methods for s in seeds]
    if workers <= 1:
        for g, m, s in jobs:
            _train_or_load_sym_fit(out_dir_str=str(out_dir), game=g, method=m, seed=s, cfg=cfg)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_train_or_load_sym_fit, out_dir_str=str(out_dir), game=g, method=m, seed=s, cfg=cfg)
                for g, m, s in jobs
            ]
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 6 == 0 or done == len(jobs):
                    print(f"  [{study_id}] training progress: {done}/{len(jobs)} fits done", flush=True)

    # Phase 2: Full Paired Matchup Evaluation (Non-Meta vs Non-Meta, Meta vs Non-Meta, Meta vs Meta)
    # Matchup tuples: (label, category, p1_name, p2_name, cross_seed_partner)
    matchups = [
        ("q_learner_vs_q_learner", "Non-Meta_vs_Non-Meta", "q_learner", "q_learner", False),
        ("ucb_vs_ucb", "Non-Meta_vs_Non-Meta", "ucb", "ucb", False),
        ("contextual_ucb_vs_contextual_ucb", "Non-Meta_vs_Non-Meta", "contextual_ucb", "contextual_ucb", False),
        ("q_learner_vs_ucb", "Non-Meta_vs_Non-Meta", "q_learner", "ucb", False),
        ("meta_aux_yr_vs_q_learner", "Meta_vs_Non-Meta", "meta_drqn_aux_yr", "q_learner", False),
        ("meta_aux_yr_vs_ucb", "Meta_vs_Non-Meta", "meta_drqn_aux_yr", "ucb", False),
        ("meta_aux_yr_vs_contextual_ucb", "Meta_vs_Non-Meta", "meta_drqn_aux_yr", "contextual_ucb", False),
        ("meta_vanilla_vs_q_learner", "Meta_vs_Non-Meta", "meta_drqn_vanilla", "q_learner", False),
        ("meta_vanilla_vs_ucb", "Meta_vs_Non-Meta", "meta_drqn_vanilla", "ucb", False),
        ("meta_masked_vs_q_learner", "Meta_vs_Non-Meta", "meta_drqn_masked", "q_learner", False),
        ("meta_masked_vs_ucb", "Meta_vs_Non-Meta", "meta_drqn_masked", "ucb", False),
        ("meta_aux_yr_vs_meta_aux_yr_xseed", "Meta_vs_Meta", "meta_drqn_aux_yr", "meta_drqn_aux_yr", True),
        ("meta_aux_yr_vs_meta_aux_yr_self", "Meta_vs_Meta", "meta_drqn_aux_yr", "meta_drqn_aux_yr", False),
        ("meta_vanilla_vs_meta_vanilla_xseed", "Meta_vs_Meta", "meta_drqn_vanilla", "meta_drqn_vanilla", True),
        ("meta_aux_yr_vs_meta_vanilla_xseed", "Meta_vs_Meta", "meta_drqn_aux_yr", "meta_drqn_vanilla", True),
        ("meta_aux_yr_vs_meta_masked_xseed", "Meta_vs_Meta", "meta_drqn_aux_yr", "meta_drqn_masked", True),
    ]
    # Filter matchups to only methods configured
    active_set = set(meta_methods) | {"q_learner", "ucb", "contextual_ucb", "random"}
    matchups = [m for m in matchups if m[2] in active_set and m[3] in active_set]

    contrast_specs = [
        ("meta_aux_yr_vs_q_learner", "q_learner_vs_q_learner"),
        ("meta_aux_yr_vs_ucb", "ucb_vs_ucb"),
        ("meta_aux_yr_vs_contextual_ucb", "contextual_ucb_vs_contextual_ucb"),
        ("meta_aux_yr_vs_q_learner", "meta_vanilla_vs_q_learner"),
        ("meta_aux_yr_vs_q_learner", "meta_masked_vs_q_learner"),
        ("meta_aux_yr_vs_ucb", "meta_vanilla_vs_ucb"),
        ("meta_aux_yr_vs_ucb", "meta_masked_vs_ucb"),
        ("meta_aux_yr_vs_meta_aux_yr_xseed", "q_learner_vs_q_learner"),
        ("meta_aux_yr_vs_meta_aux_yr_xseed", "ucb_vs_ucb"),
        ("meta_aux_yr_vs_meta_aux_yr_xseed", "meta_vanilla_vs_meta_vanilla_xseed"),
        ("meta_aux_yr_vs_meta_aux_yr_xseed", "meta_aux_yr_vs_q_learner"),
    ]

    results_by_game: dict[str, dict] = {}
    matchup_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for g_idx, game in enumerate(games):
        results_by_game[game] = {"splits": {}}
        for s_idx, (split_label, env_split) in enumerate((("in_dist", "train"), ("unseen_rules", "test"))):
            agg_matchups: dict[str, dict] = {}
            arrays_by_matchup: dict[str, dict[str, np.ndarray]] = {}

            for label, category, p1_name, p2_name, cross_seed in matchups:
                per_seed_metrics = []
                per_seed_arrs: dict[str, list[np.ndarray]] = {k: [] for k in ("p1_return", "p2_return", "welfare", "margin")}
                for i_s, seed in enumerate(seeds):
                    p2_seed = seeds[(i_s + 1) % len(seeds)] if (cross_seed and len(seeds) > 1) else seed
                    p1 = _load_sym_policy(out_dir, game, p1_name, seed, cfg)
                    p2 = _load_sym_policy(out_dir, game, p2_name, p2_seed, cfg)
                    eval_seed = eval_seed_base + g_idx * 10000 + s_idx * 1000 + seed * 17
                    m_d, a_d = rollout_symmetric_matchup(
                        p1,
                        p2,
                        game=game,
                        split=env_split,
                        n_episodes=n_eval,
                        horizon=horizon,
                        seed=eval_seed,
                    )
                    per_seed_metrics.append(m_d)
                    for k in per_seed_arrs:
                        per_seed_arrs[k].append(a_d[k])

                arr_stack = {k: np.stack(v, axis=0) for k, v in per_seed_arrs.items()}
                arrays_by_matchup[label] = arr_stack

                seed_p1 = np.mean(arr_stack["p1_return"], axis=1)
                seed_w = np.mean(arr_stack["welfare"], axis=1)
                seed_m = np.mean(arr_stack["margin"], axis=1)
                agg = {
                    "category": category,
                    "p1": p1_name,
                    "p2": p2_name,
                    "cross_seed": cross_seed,
                    "p1_mean": float(np.mean(seed_p1)),
                    "p1_seed_sd": float(np.std(seed_p1, ddof=1)) if len(seeds) > 1 else 0.0,
                    "p1_median": float(np.median(arr_stack["p1_return"])),
                    "p1_iqr": float(np.percentile(arr_stack["p1_return"], 75) - np.percentile(arr_stack["p1_return"], 25)),
                    "p2_mean": float(np.mean(arr_stack["p2_return"])),
                    "welfare_mean": float(np.mean(seed_w)),
                    "welfare_seed_sd": float(np.std(seed_w, ddof=1)) if len(seeds) > 1 else 0.0,
                    "welfare_median": float(np.median(arr_stack["welfare"])),
                    "margin_mean": float(np.mean(seed_m)),
                    "margin_seed_sd": float(np.std(seed_m, ddof=1)) if len(seeds) > 1 else 0.0,
                    "p1_win_ep_rate": float(np.mean([m["p1_win_ep_rate"] for m in per_seed_metrics])),
                    "p1_tie_ep_rate": float(np.mean([m["p1_tie_ep_rate"] for m in per_seed_metrics])),
                    "p1_loss_ep_rate": float(np.mean([m["p1_loss_ep_rate"] for m in per_seed_metrics])),
                    "mutual_coop_early": float(np.mean([m["mutual_coop_early"] for m in per_seed_metrics])),
                    "mutual_coop_late": float(np.mean([m["mutual_coop_late"] for m in per_seed_metrics])),
                    "mutual_safe_late": float(np.mean([m["mutual_safe_late"] for m in per_seed_metrics])),
                    "trap_rate_all": float(np.mean([m["trap_rate_all"] for m in per_seed_metrics])),
                    "p1_win_turn_late": float(np.mean([m["p1_win_turn_late"] for m in per_seed_metrics])),
                    "p1_anchor_late": float(np.mean([m["p1_anchor_late"] for m in per_seed_metrics])),
                    "p1_entropy_late": float(np.mean([m["p1_entropy_late"] for m in per_seed_metrics])),
                    "p1_early_auc_1_4": float(np.mean([m["p1_early_auc_1_4"] for m in per_seed_metrics])),
                    "p1_late_auc_13_end": float(np.mean([m["p1_late_auc_13_end"] for m in per_seed_metrics])),
                }
                agg_matchups[label] = agg
                matchup_rows.append(
                    {
                        "game": game,
                        "split": split_label,
                        "category": category,
                        "matchup": label,
                        "p1_mean": round(agg["p1_mean"], 4),
                        "p1_seed_sd": round(agg["p1_seed_sd"], 4),
                        "p1_median": round(agg["p1_median"], 4),
                        "p2_mean": round(agg["p2_mean"], 4),
                        "welfare_mean": round(agg["welfare_mean"], 4),
                        "welfare_median": round(agg["welfare_median"], 4),
                        "margin_mean": round(agg["margin_mean"], 4),
                        "p1_win_ep": round(agg["p1_win_ep_rate"], 4),
                        "p1_tie_ep": round(agg["p1_tie_ep_rate"], 4),
                        "p1_loss_ep": round(agg["p1_loss_ep_rate"], 4),
                        "mutual_coop_late": round(agg["mutual_coop_late"], 4),
                        "mutual_safe_late": round(agg["mutual_safe_late"], 4),
                        "trap_rate": round(agg["trap_rate_all"], 4),
                        "p1_win_turn_late": round(agg["p1_win_turn_late"], 4),
                        "p1_anchor_late": round(agg["p1_anchor_late"], 4),
                        "p1_entropy_late": round(agg["p1_entropy_late"], 4),
                    }
                )

            # Paired hierarchical bootstrap contrasts
            contrasts: dict[str, dict] = {}
            available = set(arrays_by_matchup.keys())
            metrics_to_test = ("p1_return", "welfare") if game == "coop_trap" else ("p1_return", "margin")
            for a_lbl, b_lbl in contrast_specs:
                if a_lbl not in available or b_lbl not in available:
                    continue
                c_key = f"{a_lbl}_minus_{b_lbl}"
                contrasts[c_key] = {}
                for m_key in metrics_to_test:
                    res_b = hierarchical_paired_bootstrap(
                        arrays_by_matchup[a_lbl][m_key],
                        arrays_by_matchup[b_lbl][m_key],
                        practical_threshold=delta_return,
                        n_boot=n_boot,
                        seed=56789,
                    )
                    contrasts[c_key][m_key] = res_b
                    contrast_rows.append(
                        {
                            "game": game,
                            "split": split_label,
                            "contrast": c_key,
                            "metric": m_key,
                            "diff_mean": round(float(res_b["diff_mean"]), 4),
                            "diff_median": round(float(res_b["diff_median"]), 4),
                            "ci_low": round(float(res_b["ci_low"]), 4),
                            "ci_high": round(float(res_b["ci_high"]), 4),
                            "win_rate": round(float(res_b["win_rate"]), 4),
                            "tie_rate": round(float(res_b["tie_rate"]), 4),
                            "loss_rate": round(float(res_b["loss_rate"]), 4),
                            "threshold": delta_return,
                            "verdict": res_b["verdict"],
                        }
                    )

            # Also direct head-to-head margin test (p1 - p2 vs 0) for asymmetric Meta vs Non-Meta & Meta vs Meta
            for lbl in (
                "meta_aux_yr_vs_q_learner",
                "meta_aux_yr_vs_ucb",
                "meta_aux_yr_vs_contextual_ucb",
                "meta_aux_yr_vs_meta_vanilla_xseed",
                "meta_aux_yr_vs_meta_masked_xseed",
            ):
                if lbl not in available:
                    continue
                zeros = np.zeros_like(arrays_by_matchup[lbl]["margin"])
                res_m = hierarchical_paired_bootstrap(
                    arrays_by_matchup[lbl]["margin"],
                    zeros,
                    practical_threshold=delta_return,
                    n_boot=n_boot,
                    seed=67890,
                )
                c_key = f"head2head_margin::{lbl}"
                contrasts[c_key] = {"margin": res_m}
                contrast_rows.append(
                    {
                        "game": game,
                        "split": split_label,
                        "contrast": c_key,
                        "metric": "margin",
                        "diff_mean": round(float(res_m["diff_mean"]), 4),
                        "diff_median": round(float(res_m["diff_median"]), 4),
                        "ci_low": round(float(res_m["ci_low"]), 4),
                        "ci_high": round(float(res_m["ci_high"]), 4),
                        "win_rate": round(float(res_m["win_rate"]), 4),
                        "tie_rate": round(float(res_m["tie_rate"]), 4),
                        "loss_rate": round(float(res_m["loss_rate"]), 4),
                        "threshold": delta_return,
                        "verdict": res_m["verdict"],
                    }
                )

            results_by_game[game]["splits"][split_label] = {
                "matchups": agg_matchups,
                "paired_contrasts": contrasts,
            }

    summary = {
        "study_id": study_id,
        "git_commit": manifest["git_commit"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "rule_counts": rule_counts,
        "games": results_by_game,
    }
    _atomic_write_json(out_dir / "summary.json", summary)
    tables_dir = out_dir / "tables"
    _write_csv(tables_dir / "matchups.csv", matchup_rows)
    _write_csv(tables_dir / "paired_contrasts.csv", contrast_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPEC v3.0 Exp 2B (Symmetric 4-Action Competition & Cooperation).")
    parser.add_argument("--config", required=True, help="Path to Exp 2B JSON config.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count.")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 1))
    t0 = time.perf_counter()
    summary = run_study_exp2b(cfg, workers=workers)
    out_dir = cfg.get("output_dir", f"runs/{cfg['study_id']}")
    print(f"[{cfg['study_id']}] Complete in {time.perf_counter() - t0:.1f}s -> {out_dir}/summary.json")
    for game, gdata in summary["games"].items():
        for sp in ("in_dist", "unseen_rules"):
            print(f"\n=== {game} | {sp} ===")
            for lbl, md in gdata["splits"][sp]["matchups"].items():
                if game == "coop_trap":
                    extra = f"coopL={md['mutual_coop_late']:.2f} safeL={md['mutual_safe_late']:.2f} trap={md['trap_rate_all']:.2f}"
                else:
                    extra = f"winT={md['p1_win_turn_late']:.2f} ancL={md['p1_anchor_late']:.2f} H={md['p1_entropy_late']:.2f}"
                print(
                    f"  {lbl:34s} | P1={md['p1_mean']:+6.2f} (sd={md['p1_seed_sd']:4.2f}) P2={md['p2_mean']:+6.2f} "
                    f"| W={md['welfare_mean']:+6.2f} M={md['margin_mean']:+6.2f} "
                    f"| W/T/L={md['p1_win_ep_rate']:.2f}/{md['p1_tie_ep_rate']:.2f}/{md['p1_loss_ep_rate']:.2f} | {extra}"
                )


if __name__ == "__main__":
    main()
