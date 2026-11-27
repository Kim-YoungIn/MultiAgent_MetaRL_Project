"""SPEC v3.1 Block 2 (Exp 4B, Option A): R3+Distractor Gate Validity + Compact 6-Action Stage-Lock.

Two tightly connected experiments resolving Exp 4's action-space confound and gating validity:
1. Track 1 — `r3_distractor` (Gate-Validity Precondition, T=24, 8 actions, 5 seeds):
   Tests whether local cosine gradient-alignment auxiliary gating (`drqn_aux_discovery`)
   correctly up-weights `cand_y` over 2 pure public noise distractors (`cand_noise1`, `cand_noise2`)
   on R3 where `cand_y` is proven decisive (`+4.59 -> +17.52`).
2. Track 2 — `stage_lock_t48` (Compact 6-Action 3-Layer Sequential Stage-Lock, T=48, 6 actions, 5 seeds):
   Replaces Exp 4's 22-action exponential tree with 6 actions `{S, P, A0, A1, A2, A3}` across
   3 sequential hidden-key layers (Tier 0 +9.6 -> Tier 1 +30.9 -> Tier 2 +60.1 -> Tier 3 +90.1).
   Compares unprivileged DRQN ablations, `drqn_aux_discovery`, `drqn_aux_yr_masked` (strict
   `step_t > mask_k`), `drqn_prior_only`, unprivileged `structural_model_sl`, `arm_ucb`, `greedy`,
   and privileged `bayes_optimal`. Reports `k/5` seed sign counts + floor-effect guard.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
      .venv/bin/python -m mamrl.study_exp4b_compact --config configs/exp4b_compact_multibarrier.json --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn

from mamrl.envs.delayed_response import (
    N_ACTIONS as R3_N_ACTIONS,
    N_OPP_SYMBOLS as R3_N_SYMBOLS,
    N_TYPES as R3_N_TYPES,
    VectorizedDelayedResponseEnv,
)
from mamrl.envs.stage_lock import (
    SL_ACT_P,
    SL_ACT_S,
    SL_FULL_KEY_MASK,
    SL_N_ACTIONS,
    SL_N_KEYS,
    SL_N_STAGES,
    SL_N_SYMBOLS,
    ExactStageLockDP,
    VectorizedStageLockEnv,
    run_sl_gate0_report,
)
from mamrl.eval.audit_v3 import hierarchical_paired_bootstrap

CAND_NAMES = ("cand_y", "cand_r", "cand_noise1", "cand_noise2")
N_CANDS = 4


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Shared Recurrent Q-Net with 4 Candidate Public Aux Heads (y, r, noise1, noise2)
# ---------------------------------------------------------------------------

def encode_public_obs(
    obs: dict[str, np.ndarray],
    *,
    n_actions: int,
    n_symbols: int,
    r_scale: float,
    horizon: int,
    step_t: int,
    variant: str = "rl2",
    mask_k: int = 8,
) -> np.ndarray:
    """Strictly unprivileged public observation encoder respecting `step_t > mask_k`."""
    b = int(obs["prev_self_action"].shape[0])
    obs_dim = n_actions + n_symbols + 3
    x = np.zeros((b, obs_dim), dtype=np.float32)
    idx = np.arange(b)

    pa = obs["prev_self_action"]
    valid_a = (pa >= 0) & (pa < n_actions)
    x[idx[valid_a], pa[valid_a]] = 1.0

    allow_fb = (variant == "rl2") or (variant == "rl2_masked" and step_t > mask_k)
    if allow_fb:
        py = obs["prev_opp_action"]
        valid_y = (py >= 0) & (py < n_symbols)
        x[idx[valid_y], n_actions + py[valid_y]] = 1.0
        x[:, n_actions + n_symbols] = (obs["prev_self_reward"] / r_scale).astype(np.float32)

    x[:, n_actions + n_symbols + 1] = obs["is_start"].astype(np.float32)
    x[:, n_actions + n_symbols + 2] = float(step_t) / max(1.0, float(horizon))
    return x


class AuxDiscoveryQNet(nn.Module):
    """GRU Recurrent Q-Network with 4 candidate public auxiliary heads (y, r, noise1, noise2)."""

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int, n_symbols: int, r_scale: float) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_symbols = n_symbols
        self.r_scale = float(r_scale)
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, n_actions)
        self.pred_y = nn.Linear(hidden_dim + n_actions, n_symbols)
        self.pred_r = nn.Linear(hidden_dim + n_actions, 1)
        self.pred_noise1 = nn.Linear(hidden_dim + n_actions, n_symbols)
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
        hs = hidden_seq[:, start_t:, :]
        ax = act_seq[:, start_t:]
        a_oh = nn.functional.one_hot(ax, num_classes=self.n_actions).float()
        ha = torch.cat([hs, a_oh], dim=-1)

        l_y = nn.functional.cross_entropy(self.pred_y(ha).reshape(-1, self.n_symbols), y_seq[:, start_t:].reshape(-1))
        l_r = nn.functional.mse_loss(self.pred_r(ha).squeeze(-1), r_seq[:, start_t:] / self.r_scale) * self.r_scale
        l_n1 = nn.functional.cross_entropy(self.pred_noise1(ha).reshape(-1, self.n_symbols), z1_seq[:, start_t:].reshape(-1))
        l_n2 = nn.functional.mse_loss(self.pred_noise2(ha).squeeze(-1), z2_seq[:, start_t:])
        return [l_y, l_r, l_n1, l_n2]


def _fixed_aux_weights(strategy: str) -> np.ndarray:
    if strategy == "none":
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if strategy == "y":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if strategy == "r":
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    if strategy == "yr":
        return np.array([0.67, 0.33, 0.0, 0.0], dtype=np.float64)
    if strategy == "uniform":
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    if strategy == "discovery":
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    raise ValueError(f"Unknown aux_strategy: {strategy}")


def compute_recurrent_double_dqn_targets(
    q_all: torch.Tensor,
    tnet: nn.Module,
    tx: torch.Tensor,
    tr: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute exact full-history Double-DQN targets (preserving turn-0 hidden state)."""
    with torch.no_grad():
        next_a = torch.argmax(q_all.detach()[:, 1:, :], dim=-1, keepdim=True)
        q_target_full, _ = tnet(tx)
        next_v = q_target_full[:, 1:, :].gather(-1, next_a).squeeze(-1)
        targets = tr.clone()
        targets[:, :-1] = tr[:, :-1] + gamma * next_v
    return targets


@dataclass
class TrainSpec:
    env_kind: str        # "r3" or "stage_lock"
    method: str
    variant: str         # "rl2", "rl2_masked", "prior_only"
    aux_strategy: str    # "none", "y", "r", "yr", "uniform", "discovery"
    horizon: int
    mask_k: int
    hidden_dim: int
    batch_episodes: int
    n_iters: int
    warmup_iters: int
    replay_capacity: int
    grad_steps_per_iter: int
    minibatch_episodes: int
    gamma: float
    lr: float
    aux_scale: float
    meta_gate_lr: float
    seed: int


def _train_aux_drqn(spec: TrainSpec) -> tuple[AuxDiscoveryQNet, dict]:
    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed + 37)

    if spec.env_kind == "r3":
        n_actions, n_symbols, r_scale = R3_N_ACTIONS, R3_N_SYMBOLS, 2.0
        env = VectorizedDelayedResponseEnv(regime="R3", batch_size=spec.batch_episodes, horizon=spec.horizon, seed=spec.seed + 19)
    else:
        n_actions, n_symbols, r_scale = SL_N_ACTIONS, SL_N_SYMBOLS, 2.5
        env = VectorizedStageLockEnv(batch_size=spec.batch_episodes, horizon=spec.horizon, seed=spec.seed + 19)

    obs_dim = n_actions + n_symbols + 3
    qnet = AuxDiscoveryQNet(obs_dim, spec.hidden_dim, n_actions, n_symbols, r_scale)
    tnet = AuxDiscoveryQNet(obs_dim, spec.hidden_dim, n_actions, n_symbols, r_scale)
    tnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=spec.lr)

    cap = spec.replay_capacity
    t_len = spec.horizon
    b = spec.batch_episodes
    buf_x = np.zeros((cap, t_len, obs_dim), dtype=np.float32)
    buf_a = np.zeros((cap, t_len), dtype=np.int64)
    buf_y = np.zeros((cap, t_len), dtype=np.int64)
    buf_r = np.zeros((cap, t_len), dtype=np.float32)
    buf_z1 = np.zeros((cap, t_len), dtype=np.int64)
    buf_z2 = np.zeros((cap, t_len), dtype=np.float32)
    ptr, size = 0, 0

    gate_logits = np.zeros(N_CANDS, dtype=np.float64)
    gate_w = _fixed_aux_weights(spec.aux_strategy)
    shared_params = list(qnet.gru.parameters())

    # Training-time coverage mediators (for Stage-Lock)
    warmup_pos_hits = [0, 0, 0]
    total_pos_hits = [0, 0, 0]
    gate_trace: list[list[float]] = []

    for it in range(1, spec.n_iters + 1):
        if it <= spec.warmup_iters:
            eps = 1.0
        else:
            decay = max(1.0, 0.80 * float(spec.n_iters - spec.warmup_iters))
            eps = max(0.05, 0.90 * (1.0 - float(it - spec.warmup_iters) / decay))

        obs = env.reset()
        h = torch.zeros((1, b, spec.hidden_dim), dtype=torch.float32)
        ep_x = np.zeros((b, t_len, obs_dim), dtype=np.float32)
        ep_a = np.zeros((b, t_len), dtype=np.int64)
        ep_y = np.zeros((b, t_len), dtype=np.int64)
        ep_r = np.zeros((b, t_len), dtype=np.float32)
        ep_z1 = rng.integers(0, n_symbols, size=(b, t_len), dtype=np.int64)
        ep_z2 = rng.standard_normal((b, t_len)).astype(np.float32)

        qnet.eval()
        with torch.no_grad():
            for t in range(t_len):
                x_np = encode_public_obs(
                    obs,
                    n_actions=n_actions,
                    n_symbols=n_symbols,
                    r_scale=r_scale,
                    horizon=t_len,
                    step_t=t,
                    variant=spec.variant,
                    mask_k=spec.mask_k,
                )
                ep_x[:, t, :] = x_np
                q_vals, h = qnet.step(torch.from_numpy(x_np).unsqueeze(1), h)
                greedy_a = torch.argmax(q_vals, dim=-1).numpy().astype(np.int64)
                if eps >= 1.0:
                    acts = rng.integers(0, n_actions, size=b, dtype=np.int64)
                else:
                    explore = rng.random(b) < eps
                    rand_a = rng.integers(0, n_actions, size=b, dtype=np.int64)
                    acts = np.where(explore, rand_a, greedy_a)

                st_before = env.stage.copy() if spec.env_kind == "stage_lock" else None
                obs, rf, _ro, y = env.step(acts)
                ep_a[:, t] = acts
                ep_y[:, t] = y
                ep_r[:, t] = rf.astype(np.float32)
                if st_before is not None:
                    hits = rf > 0.25
                    for s_idx in range(SL_N_STAGES):
                        c = int(np.sum(hits & (st_before == s_idx)))
                        total_pos_hits[s_idx] += c
                        if it <= spec.warmup_iters:
                            warmup_pos_hits[s_idx] += c

        # Insert into circular replay buffer
        if ptr + b <= cap:
            buf_x[ptr : ptr + b] = ep_x
            buf_a[ptr : ptr + b] = ep_a
            buf_y[ptr : ptr + b] = ep_y
            buf_r[ptr : ptr + b] = ep_r
            buf_z1[ptr : ptr + b] = ep_z1
            buf_z2[ptr : ptr + b] = ep_z2
        else:
            first = cap - ptr
            buf_x[ptr:] = ep_x[:first]
            buf_a[ptr:] = ep_a[:first]
            buf_y[ptr:] = ep_y[:first]
            buf_r[ptr:] = ep_r[:first]
            buf_z1[ptr:] = ep_z1[:first]
            buf_z2[ptr:] = ep_z2[:first]
            rem = b - first
            buf_x[:rem] = ep_x[first:]
            buf_a[:rem] = ep_a[first:]
            buf_y[:rem] = ep_y[first:]
            buf_r[:rem] = ep_r[first:]
            buf_z1[:rem] = ep_z1[first:]
            buf_z2[:rem] = ep_z2[first:]
        ptr = (ptr + b) % cap
        size = min(cap, size + b)

        if size < spec.minibatch_episodes:
            continue

        qnet.train()
        start_t = spec.mask_k if spec.variant == "rl2_masked" else 0
        for g_step in range(spec.grad_steps_per_iter):
            mb_idx = rng.integers(0, size, size=spec.minibatch_episodes)
            tx = torch.from_numpy(buf_x[mb_idx])
            ta = torch.from_numpy(buf_a[mb_idx])
            ty = torch.from_numpy(buf_y[mb_idx])
            tr = torch.from_numpy(buf_r[mb_idx])
            tz1 = torch.from_numpy(buf_z1[mb_idx])
            tz2 = torch.from_numpy(buf_z2[mb_idx])

            q_all, h_seq = qnet(tx)
            q_sa = q_all.gather(-1, ta.unsqueeze(-1)).squeeze(-1)
            targets = compute_recurrent_double_dqn_targets(q_all, tnet, tx, tr, spec.gamma)

            td_loss = nn.functional.smooth_l1_loss(q_sa, targets)

            aux_loss_total = torch.tensor(0.0)
            if spec.aux_strategy != "none" and spec.variant != "prior_only":
                cand_losses = qnet.candidate_aux_losses(h_seq, ta, ty, tr, tz1, tz2, start_t=start_t)

                # Local cosine gradient-alignment auxiliary gating (every iter on g_step == 0)
                if spec.aux_strategy == "discovery" and g_step == 0:
                    g_td = torch.autograd.grad(td_loss, shared_params, retain_graph=True, allow_unused=True)
                    v_td = torch.cat([g.reshape(-1) for g in g_td if g is not None])
                    td_norm = float(torch.norm(v_td).item()) + 1e-8
                    cos_scores = np.zeros(N_CANDS, dtype=np.float64)
                    for ci, cl in enumerate(cand_losses):
                        g_c = torch.autograd.grad(cl, shared_params, retain_graph=True, allow_unused=True)
                        v_c = torch.cat([g.reshape(-1) for g in g_c if g is not None])
                        c_norm = float(torch.norm(v_c).item()) + 1e-8
                        cos_scores[ci] = float(torch.dot(v_td, v_c).item()) / (td_norm * c_norm)
                    gate_logits = np.clip(gate_logits + spec.meta_gate_lr * cos_scores, -4.0, 4.0)
                    ex = np.exp(gate_logits - np.max(gate_logits))
                    gate_w = ex / np.sum(ex)

                w_t = gate_w * spec.aux_scale
                for ci in range(N_CANDS):
                    if w_t[ci] > 0.0:
                        aux_loss_total = aux_loss_total + float(w_t[ci]) * cand_losses[ci]

            loss = td_loss + aux_loss_total
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
            opt.step()

            # Polyak target update
            with torch.no_grad():
                for p, tp in zip(qnet.parameters(), tnet.parameters()):
                    tp.data.mul_(0.95).add_(p.data, alpha=0.05)

        if it % 50 == 0 or it == spec.n_iters:
            gate_trace.append([float(x) for x in gate_w])

    qnet.eval()
    return qnet, {
        "final_gate_weights": [float(x) for x in gate_w],
        "gate_trace": gate_trace,
        "warmup_pos_hits_by_stage": warmup_pos_hits,
        "total_pos_hits_by_stage": total_pos_hits,
        "train_transitions": int(spec.n_iters * spec.batch_episodes * spec.horizon),
    }


@torch.no_grad()
def _eval_r3_model(qnet: AuxDiscoveryQNet, *, variant: str, horizon: int, n_eval: int, seed: int) -> dict:
    env = VectorizedDelayedResponseEnv(regime="R3", batch_size=n_eval, horizon=horizon, seed=seed)
    thetas = np.tile(np.arange(R3_N_TYPES, dtype=np.int64), n_eval // R3_N_TYPES + 1)[:n_eval]
    obs = env.reset(theta=thetas)
    h = torch.zeros((1, n_eval, qnet.hidden_dim), dtype=torch.float32)
    tot_r = np.zeros(n_eval, dtype=np.float64)
    tot_o = np.zeros(n_eval, dtype=np.float64)
    early_r = np.zeros(n_eval, dtype=np.float64)

    for t in range(horizon):
        x_np = encode_public_obs(
            obs,
            n_actions=R3_N_ACTIONS,
            n_symbols=R3_N_SYMBOLS,
            r_scale=2.0,
            horizon=horizon,
            step_t=t,
            variant=variant,
            mask_k=4,
        )
        q_vals, h = qnet.step(torch.from_numpy(x_np).unsqueeze(1), h)
        acts = torch.argmax(q_vals, dim=-1).numpy().astype(np.int64)
        obs, rf, ro, _y = env.step(acts)
        tot_r += rf
        tot_o += ro
        if t < 4:
            early_r += rf

    return {
        "total_return": tot_r,
        "opp_return": tot_o,
        "early_auc": early_r / 4.0,
    }


@torch.no_grad()
def _eval_stage_lock_model(
    qnet: AuxDiscoveryQNet,
    *,
    variant: str,
    mask_k: int,
    horizon: int,
    n_eval: int,
    seed: int,
) -> dict:
    env = VectorizedStageLockEnv(batch_size=n_eval, horizon=horizon, seed=seed)
    # Exact balanced 64 key tuples tiled across n_eval
    all_keys = np.array(
        [[k0, k1, k2] for k0 in range(SL_N_KEYS) for k1 in range(SL_N_KEYS) for k2 in range(SL_N_KEYS)],
        dtype=np.int64,
    )
    keys = np.tile(all_keys, (n_eval // 64 + 1, 1))[:n_eval]
    obs = env.reset(keys=keys)
    h = torch.zeros((1, n_eval, qnet.hidden_dim), dtype=torch.float32)

    tot_r = np.zeros(n_eval, dtype=np.float64)
    tot_o = np.zeros(n_eval, dtype=np.float64)
    early_r = np.zeros(n_eval, dtype=np.float64)
    ul_s0 = np.zeros(n_eval, dtype=np.bool_)
    ul_s1 = np.zeros(n_eval, dtype=np.bool_)
    ul_s2 = np.zeros(n_eval, dtype=np.bool_)
    acts_hist = np.zeros((n_eval, horizon), dtype=np.int64)

    for t in range(horizon):
        x_np = encode_public_obs(
            obs,
            n_actions=SL_N_ACTIONS,
            n_symbols=SL_N_SYMBOLS,
            r_scale=2.5,
            horizon=horizon,
            step_t=t,
            variant=variant,
            mask_k=mask_k,
        )
        q_vals, h = qnet.step(torch.from_numpy(x_np).unsqueeze(1), h)
        acts = torch.argmax(q_vals, dim=-1).numpy().astype(np.int64)
        st_b = env.stage.copy()
        obs, rf, ro, _y = env.step(acts)
        acts_hist[:, t] = acts
        tot_r += rf
        tot_o += ro
        if t < 8:
            early_r += rf
        ul_s0 |= (st_b == 0) & (rf > 0.25)
        ul_s1 |= (st_b == 1) & (rf > 0.25)
        ul_s2 |= (st_b == 2) & (rf > 0.25)

    max_stage = env.stage.copy()
    return {
        "total_return": tot_r,
        "opp_return": tot_o,
        "early_auc": early_r / 8.0,
        "reach_stage0_unlock": ul_s0.astype(np.float64),
        "reach_stage1": (max_stage >= 1).astype(np.float64),
        "reach_stage1_unlock": ul_s1.astype(np.float64),
        "reach_stage2": (max_stage >= 2).astype(np.float64),
        "reach_stage2_unlock": ul_s2.astype(np.float64),
        "max_stage": max_stage.astype(np.float64),
        "tier0_only": np.all(acts_hist == SL_ACT_S, axis=1).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Unprivileged Baselines & Privileged Bayes-Optimal Ceiling on Stage-Lock
# ---------------------------------------------------------------------------

def _eval_stage_lock_baseline(method: str, *, horizon: int, n_eval: int, seed: int) -> dict:
    env = VectorizedStageLockEnv(batch_size=n_eval, horizon=horizon, seed=seed)
    all_keys = np.array(
        [[k0, k1, k2] for k0 in range(SL_N_KEYS) for k1 in range(SL_N_KEYS) for k2 in range(SL_N_KEYS)],
        dtype=np.int64,
    )
    keys = np.tile(all_keys, (n_eval // 64 + 1, 1))[:n_eval]
    obs = env.reset(keys=keys)
    rng = np.random.default_rng(seed + 71)

    tot_r = np.zeros(n_eval, dtype=np.float64)
    tot_o = np.zeros(n_eval, dtype=np.float64)
    early_r = np.zeros(n_eval, dtype=np.float64)
    ul_s0 = np.zeros(n_eval, dtype=np.bool_)
    ul_s1 = np.zeros(n_eval, dtype=np.bool_)
    ul_s2 = np.zeros(n_eval, dtype=np.bool_)
    acts_hist = np.zeros((n_eval, horizon), dtype=np.int64)

    if method == "bayes_optimal":
        dp = ExactStageLockDP(max_tier=3)
        masks = np.full(n_eval, SL_FULL_KEY_MASK, dtype=np.int64)
        stages = np.zeros(n_eval, dtype=np.int64)
        unlocked = np.zeros(n_eval, dtype=np.int64)
        for t in range(horizon):
            q_prev = obs["prev_self_action"].copy()
            acts = np.array(
                [dp.optimal_action(horizon - t, int(stages[i]), int(unlocked[i]), int(q_prev[i]), int(masks[i])) for i in range(n_eval)],
                dtype=np.int64,
            )
            st_b = env.stage.copy()
            obs, rf, ro, y = env.step(acts)
            acts_hist[:, t] = acts
            tot_r += rf
            tot_o += ro
            if t < 8:
                early_r += rf
            ul_s0 |= (st_b == 0) & (rf > 0.25)
            ul_s1 |= (st_b == 1) & (rf > 0.25)
            ul_s2 |= (st_b == 2) & (rf > 0.25)
            for i in range(n_eval):
                ai, qi, yi = int(acts[i]), int(q_prev[i]), int(y[i])
                nm = 0
                for k in range(SL_N_KEYS):
                    if ((int(masks[i]) >> k) & 1) and dp._f_table[k][qi] == yi:
                        nm |= 1 << k
                if ai == SL_ACT_P and unlocked[i] == 1 and stages[i] < SL_N_STAGES - 1:
                    stages[i] += 1
                    unlocked[i] = 0
                    masks[i] = SL_FULL_KEY_MASK
                else:
                    if ai >= 2 and qi == ai and yi == 1:
                        unlocked[i] = 1
                    masks[i] = nm
    elif method == "structural_model_sl":
        # Strictly UNPRIVILEGED structural model: collects unlabeled random offline episodes,
        # tracks public stage transitions purely from public (a, r > 0.25), fits 4-cluster EM
        # over (q, y) per public stage segment + empirical reward table, and plans via learned DP!
        n_off = 2048
        off_rng = np.random.default_rng(seed + 505)
        off_env = VectorizedStageLockEnv(batch_size=n_off, horizon=horizon, seed=seed + 509)
        o_obs = off_env.reset()  # NEVER reads off_env.keys!

        # Collect unlabeled transitions + publicly tracked (pub_stage, pub_ul)
        pub_st = np.zeros(n_off, dtype=np.int64)
        pub_ul = np.zeros(n_off, dtype=np.int64)
        r_sum = np.zeros((SL_N_STAGES, 2, SL_N_ACTIONS, SL_N_ACTIONS, SL_N_SYMBOLS), dtype=np.float64)
        r_cnt = np.zeros((SL_N_STAGES, 2, SL_N_ACTIONS, SL_N_ACTIONS, SL_N_SYMBOLS), dtype=np.float64)
        # Collect segment-level (q, y) counts for EM clustering (each segment = contiguous stay in a public stage)
        seg_counts: list[np.ndarray] = []
        cur_seg = np.zeros((n_off, SL_N_ACTIONS, SL_N_SYMBOLS), dtype=np.float64)

        for _t in range(horizon):
            q_prev = o_obs["prev_self_action"].copy()
            acts = off_rng.integers(0, SL_N_ACTIONS, size=n_off, dtype=np.int64)
            o_obs, rf, _ro, y = off_env.step(acts)
            np.add.at(r_sum, (pub_st, pub_ul, q_prev, acts, y), rf)
            np.add.at(r_cnt, (pub_st, pub_ul, q_prev, acts, y), 1.0)
            idx = np.arange(n_off)
            cur_seg[idx, q_prev, y] += 1.0
            # Publicly detectable stage unlock & barrier crossing (zero hidden access!)
            hit = rf > 0.25
            cross = (acts == SL_ACT_P) & (pub_ul == 1) & (pub_st < SL_N_STAGES - 1)
            if np.any(cross):
                for ci in np.flatnonzero(cross):
                    if float(np.sum(cur_seg[ci])) >= 4.0:
                        seg_counts.append(cur_seg[ci].copy())
                    cur_seg[ci] = 0.0
            pub_st = np.where(cross, pub_st + 1, pub_st)
            pub_ul = np.where(cross, 0, np.where(hit, 1, pub_ul))

        for i in range(n_off):
            if float(np.sum(cur_seg[i])) >= 4.0:
                seg_counts.append(cur_seg[i].copy())

        seg_arr = np.stack(seg_counts, axis=0)  # (N_seg, 6, 5)
        r_hat = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1.0), 0.0)

        # Unprivileged 4-cluster EM over segment (q, y) counts
        best_ll = -np.inf
        best_p = None
        for restart in range(6):
            rrng = np.random.default_rng(seed + 773 * (restart + 1))
            pi = np.full(SL_N_KEYS, 1.0 / SL_N_KEYS, dtype=np.float64)
            p_mqy = rrng.dirichlet(np.ones(SL_N_SYMBOLS), size=(SL_N_KEYS, SL_N_ACTIONS))
            for _ in range(35):
                lp = np.log(np.clip(p_mqy, 1e-9, 1.0))
                ll_mat = np.log(np.clip(pi, 1e-9, 1.0))[None, :] + np.einsum("eqy,mqy->em", seg_arr, lp)
                m_ll = np.max(ll_mat, axis=1, keepdims=True)
                lse = m_ll + np.log(np.sum(np.exp(ll_mat - m_ll), axis=1, keepdims=True))
                ll = float(np.sum(lse))
                resp = np.exp(ll_mat - lse)
                pi = (np.sum(resp, axis=0) + 1e-2) / (seg_arr.shape[0] + SL_N_KEYS * 1e-2)
                w = np.einsum("em,eqy->mqy", resp, seg_arr) + 1e-3
                p_mqy = w / np.sum(w, axis=-1, keepdims=True)
            if ll > best_ll:
                best_ll = ll
                best_p = p_mqy.copy()
        assert best_p is not None
        f_hat = np.argmax(best_p, axis=-1)  # (4, 6)

        @lru_cache(maxsize=None)
        def ldp_val(rem: int, st: int, ul: int, q: int, mask: int) -> float:
            if rem <= 0:
                return 0.0
            sub = [0] * SL_N_SYMBOLS
            cnt = [0] * SL_N_SYMBOLS
            for m in range(SL_N_KEYS):
                if (mask >> m) & 1:
                    yy = int(f_hat[m, q])
                    sub[yy] |= 1 << m
                    cnt[yy] += 1
            tot = sum(cnt)
            if tot == 0:
                return 0.0
            splits = [(yy, sub[yy], cnt[yy] / tot) for yy in range(SL_N_SYMBOLS) if cnt[yy] > 0]
            best_v = -1e18
            for a in range(SL_N_ACTIONS):
                ev = 0.0
                for yy, sm, prob in splits:
                    rv = float(r_hat[st, ul, q, a, yy])
                    if a == SL_ACT_P and ul == 1 and st < SL_N_STAGES - 1:
                        ns, nu, nm = st + 1, 0, SL_FULL_KEY_MASK
                    else:
                        ns = st
                        nu = 1 if rv > 0.25 else ul
                        nm = sm
                    ev += prob * (rv + ldp_val(rem - 1, ns, nu, a, nm))
                if ev > best_v:
                    best_v = ev
            return best_v

        def ldp_act(rem: int, st: int, ul: int, q: int, mask: int) -> int:
            sub = [0] * SL_N_SYMBOLS
            cnt = [0] * SL_N_SYMBOLS
            for m in range(SL_N_KEYS):
                if (mask >> m) & 1:
                    yy = int(f_hat[m, q])
                    sub[yy] |= 1 << m
                    cnt[yy] += 1
            tot = sum(cnt)
            if tot == 0:
                return SL_ACT_S
            splits = [(yy, sub[yy], cnt[yy] / tot) for yy in range(SL_N_SYMBOLS) if cnt[yy] > 0]
            best_a = SL_ACT_S
            best_v = -1e18
            for a in range(SL_N_ACTIONS):
                ev = 0.0
                for yy, sm, prob in splits:
                    rv = float(r_hat[st, ul, q, a, yy])
                    if a == SL_ACT_P and ul == 1 and st < SL_N_STAGES - 1:
                        ns, nu, nm = st + 1, 0, SL_FULL_KEY_MASK
                    else:
                        ns = st
                        nu = 1 if rv > 0.25 else ul
                        nm = sm
                    ev += prob * (rv + ldp_val(rem - 1, ns, nu, a, nm))
                if ev > best_v + 1e-12:
                    best_v = ev
                    best_a = a
            return best_a

        masks = np.full(n_eval, SL_FULL_KEY_MASK, dtype=np.int64)
        stages = np.zeros(n_eval, dtype=np.int64)
        unlocked = np.zeros(n_eval, dtype=np.int64)
        for t in range(horizon):
            q_prev = obs["prev_self_action"].copy()
            acts = np.array(
                [ldp_act(horizon - t, int(stages[i]), int(unlocked[i]), int(q_prev[i]), int(masks[i])) for i in range(n_eval)],
                dtype=np.int64,
            )
            st_b = env.stage.copy()
            obs, rf, ro, y = env.step(acts)
            acts_hist[:, t] = acts
            tot_r += rf
            tot_o += ro
            if t < 8:
                early_r += rf
            ul_s0 |= (st_b == 0) & (rf > 0.25)
            ul_s1 |= (st_b == 1) & (rf > 0.25)
            ul_s2 |= (st_b == 2) & (rf > 0.25)
            for i in range(n_eval):
                ai, qi, yi, ri = int(acts[i]), int(q_prev[i]), int(y[i]), float(rf[i])
                nm = 0
                for m in range(SL_N_KEYS):
                    if ((int(masks[i]) >> m) & 1) and int(f_hat[m, qi]) == yi:
                        nm |= 1 << m
                if ai == SL_ACT_P and unlocked[i] == 1 and stages[i] < SL_N_STAGES - 1:
                    stages[i] += 1
                    unlocked[i] = 0
                    masks[i] = SL_FULL_KEY_MASK
                else:
                    if ri > 0.25:
                        unlocked[i] = 1
                    masks[i] = nm if nm != 0 else masks[i]
    elif method in ("arm_ucb", "greedy"):
        sums = np.zeros((n_eval, SL_N_ACTIONS), dtype=np.float64)
        cnts = np.zeros((n_eval, SL_N_ACTIONS), dtype=np.float64)
        last_a = np.zeros(n_eval, dtype=np.int64)
        for t in range(horizon):
            if t > 0:
                idx = np.arange(n_eval)
                sums[idx, last_a] += obs["prev_self_reward"]
                cnts[idx, last_a] += 1.0
            means = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), 0.0)
            bonus = np.sqrt(math.log(t + 2.0) / (cnts + 1.0)) if method == "arm_ucb" else 0.0
            acts = np.argmax(means + bonus + rng.uniform(0.0, 1e-9, size=means.shape), axis=-1).astype(np.int64)
            last_a = acts.copy()
            st_b = env.stage.copy()
            obs, rf, ro, _y = env.step(acts)
            acts_hist[:, t] = acts
            tot_r += rf
            tot_o += ro
            if t < 8:
                early_r += rf
            ul_s0 |= (st_b == 0) & (rf > 0.25)
            ul_s1 |= (st_b == 1) & (rf > 0.25)
            ul_s2 |= (st_b == 2) & (rf > 0.25)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    max_stage = env.stage.copy()
    return {
        "total_return": tot_r,
        "opp_return": tot_o,
        "early_auc": early_r / 8.0,
        "reach_stage0_unlock": ul_s0.astype(np.float64),
        "reach_stage1": (max_stage >= 1).astype(np.float64),
        "reach_stage1_unlock": ul_s1.astype(np.float64),
        "reach_stage2": (max_stage >= 2).astype(np.float64),
        "reach_stage2_unlock": ul_s2.astype(np.float64),
        "max_stage": max_stage.astype(np.float64),
        "tier0_only": np.all(acts_hist == SL_ACT_S, axis=1).astype(np.float64),
    }


def _worker_fit(job: dict) -> dict:
    spec = TrainSpec(**job["spec"])
    out_dir = Path(job["fit_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    qnet, meta = _train_aux_drqn(spec)
    torch.save({"state_dict": qnet.state_dict(), "spec": job["spec"], "meta": meta}, out_dir / "checkpoint.pt")
    if spec.env_kind == "r3":
        ev = _eval_r3_model(qnet, variant=spec.variant, horizon=spec.horizon, n_eval=job["n_eval"], seed=job["eval_seed"])
    else:
        ev = _eval_stage_lock_model(
            qnet, variant=spec.variant, mask_k=spec.mask_k, horizon=spec.horizon, n_eval=job["n_eval"], seed=job["eval_seed"]
        )
    wall = time.perf_counter() - t0
    return {
        "track": job["track"],
        "method": spec.method,
        "seed": spec.seed,
        "wall_s": round(wall, 2),
        "meta": meta,
        "eval": ev,
    }


def _verdict_with_floor_guard(
    a: np.ndarray,
    b: np.ndarray,
    *,
    floor_val: float | None,
    n_boot: int,
    delta: float,
    seed: int = 1234,
) -> dict:
    boot = hierarchical_paired_bootstrap(a, b, n_boot=n_boot, practical_threshold=delta, seed=seed)
    seed_diffs = np.mean(a - b, axis=1)
    verdict = boot["verdict"]
    if floor_val is not None:
        if abs(float(np.mean(a)) - floor_val) < 0.25 and abs(float(np.mean(b)) - floor_val) < 0.25 and boot["tie_rate"] > 0.70:
            verdict = "NOT_TESTABLE_AT_FLOOR"
    return {
        **boot,
        "verdict": verdict,
        "seeds_positive": int(np.sum(seed_diffs > 1e-6)),
        "seeds_negative": int(np.sum(seed_diffs < -1e-6)),
    }


def run_exp4b(cfg: dict, workers: int = 8) -> dict:
    t0 = time.perf_counter()
    seeds = [int(s) for s in cfg["seeds"]]
    n_eval = int(cfg.get("n_eval_episodes", 512))
    n_boot = int(cfg.get("n_bootstrap", 10000))
    delta = float(cfg.get("delta_return", 0.30))
    out_dir = Path(cfg.get("output_dir", "runs/exp4b_compact_multibarrier"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory Gate-0 verification before ANY training
    gate0 = run_sl_gate0_report((4, 8, 16, 24, 48))
    assert gate0["T=48"]["optimal_barrier_tier"] == 3 and gate0["T=48"]["must_change_opening"]

    r3_cfg = cfg["r3_distractor"]
    sl_cfg = cfg["stage_lock"]

    r3_methods = [
        ("drqn_vanilla", "rl2", "none"),
        ("drqn_aux_yr", "rl2", "yr"),
        ("drqn_aux_uniform", "rl2", "uniform"),
        ("drqn_aux_discovery", "rl2", "discovery"),
    ]
    sl_neural_methods = [
        ("drqn_vanilla", "rl2", "none"),
        ("drqn_aux_y", "rl2", "y"),
        ("drqn_aux_r", "rl2", "r"),
        ("drqn_aux_yr", "rl2", "yr"),
        ("drqn_aux_uniform", "rl2", "uniform"),
        ("drqn_aux_discovery", "rl2", "discovery"),
        ("drqn_aux_yr_masked", "rl2_masked", "yr"),
        ("drqn_prior_only", "prior_only", "none"),
    ]

    jobs: list[dict] = []
    for m_name, variant, aux_s in r3_methods:
        for s in seeds:
            jobs.append({
                "track": "r3_distractor",
                "fit_dir": str(out_dir / "fits" / f"r3_{m_name}_seed{s}"),
                "n_eval": n_eval,
                "eval_seed": 91000 + s * 17,
                "spec": dict(
                    env_kind="r3",
                    method=m_name,
                    variant=variant,
                    aux_strategy=aux_s,
                    horizon=int(r3_cfg["horizon"]),
                    mask_k=4,
                    hidden_dim=int(r3_cfg["hidden_dim"]),
                    batch_episodes=int(r3_cfg["batch_episodes"]),
                    n_iters=int(r3_cfg["n_iters"]),
                    warmup_iters=int(r3_cfg["warmup_iters"]),
                    replay_capacity=int(r3_cfg["replay_capacity"]),
                    grad_steps_per_iter=int(r3_cfg["grad_steps_per_iter"]),
                    minibatch_episodes=int(r3_cfg["minibatch_episodes"]),
                    gamma=float(r3_cfg["gamma"]),
                    lr=float(r3_cfg["lr"]),
                    aux_scale=float(r3_cfg["aux_scale"]),
                    meta_gate_lr=float(r3_cfg["meta_gate_lr"]),
                    seed=s,
                ),
            })

    for m_name, variant, aux_s in sl_neural_methods:
        for s in seeds:
            jobs.append({
                "track": "stage_lock_t48",
                "fit_dir": str(out_dir / "fits" / f"sl_{m_name}_seed{s}"),
                "n_eval": n_eval,
                "eval_seed": 92000 + s * 17,
                "spec": dict(
                    env_kind="stage_lock",
                    method=m_name,
                    variant=variant,
                    aux_strategy=aux_s,
                    horizon=int(sl_cfg["horizon"]),
                    mask_k=int(sl_cfg["mask_k"]),
                    hidden_dim=int(sl_cfg["hidden_dim"]),
                    batch_episodes=int(sl_cfg["batch_episodes"]),
                    n_iters=int(sl_cfg["n_iters"]),
                    warmup_iters=int(sl_cfg["warmup_iters"]),
                    replay_capacity=int(sl_cfg["replay_capacity"]),
                    grad_steps_per_iter=int(sl_cfg["grad_steps_per_iter"]),
                    minibatch_episodes=int(sl_cfg["minibatch_episodes"]),
                    gamma=float(sl_cfg["gamma"]),
                    lr=float(sl_cfg["lr"]),
                    aux_scale=float(sl_cfg["aux_scale"]),
                    meta_gate_lr=float(sl_cfg["meta_gate_lr"]),
                    seed=s,
                ),
            })

    print(f"[Exp 4B] Launching {len(jobs)} neural fits across {workers} workers...", flush=True)
    results_map: dict[tuple[str, str, int], dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_fit, j) for j in jobs]
        done_n = 0
        for fut in as_completed(futs):
            res = fut.result()
            done_n += 1
            results_map[(res["track"], res["method"], res["seed"])] = res
            if done_n % 10 == 0 or done_n == len(jobs):
                print(
                    f"  [{done_n}/{len(jobs)}] {res['track']}/{res['method']} s={res['seed']} "
                    f"mean={float(np.mean(res['eval']['total_return'])):.2f} ({res['wall_s']}s)",
                    flush=True,
                )

    # Evaluate Stage-Lock non-neural baselines & privileged Bayes-Optimal
    sl_baselines = ("structural_model_sl", "arm_ucb", "greedy", "bayes_optimal")
    for bm in sl_baselines:
        for s in seeds:
            ev = _eval_stage_lock_baseline(bm, horizon=int(sl_cfg["horizon"]), n_eval=n_eval, seed=92000 + s * 17)
            results_map[("stage_lock_t48", bm, s)] = {
                "track": "stage_lock_t48",
                "method": bm,
                "seed": s,
                "meta": {"final_gate_weights": [0.0, 0.0, 0.0, 0.0], "warmup_pos_hits_by_stage": [0, 0, 0]},
                "eval": ev,
            }

    # Aggregate Track 1: r3_distractor
    r3_rows: list[dict] = []
    r3_mats: dict[str, np.ndarray] = {}
    for m_name, _, _ in r3_methods:
        mats = [results_map[("r3_distractor", m_name, s)]["eval"]["total_return"] for s in seeds]
        mat = np.stack(mats, axis=0)
        r3_mats[m_name] = mat
        s_means = np.mean(mat, axis=1)
        q25, q50, q75 = np.percentile(mat, [25, 50, 75])
        gw = np.mean([results_map[("r3_distractor", m_name, s)]["meta"]["final_gate_weights"] for s in seeds], axis=0)
        opp_m = float(np.mean([np.mean(results_map[("r3_distractor", m_name, s)]["eval"]["opp_return"]) for s in seeds]))
        early_m = float(np.mean([np.mean(results_map[("r3_distractor", m_name, s)]["eval"]["early_auc"]) for s in seeds]))
        r3_rows.append({
            "method": m_name,
            "mean_return": float(np.mean(s_means)),
            "seed_std": float(np.std(s_means, ddof=1)),
            "median_return": float(q50),
            "iqr_return": float(q75 - q25),
            "early_auc_1_4": early_m,
            "opp_return": opp_m,
            "gate_y": float(gw[0]),
            "gate_r": float(gw[1]),
            "gate_noise1": float(gw[2]),
            "gate_noise2": float(gw[3]),
        })

    r3_contrasts: list[dict] = []
    for ma, mb in (
        ("drqn_aux_discovery", "drqn_vanilla"),
        ("drqn_aux_discovery", "drqn_aux_uniform"),
        ("drqn_aux_discovery", "drqn_aux_yr"),
        ("drqn_aux_yr", "drqn_vanilla"),
    ):
        c = _verdict_with_floor_guard(r3_mats[ma], r3_mats[mb], floor_val=4.8, n_boot=n_boot, delta=delta, seed=2025)
        r3_contrasts.append({"track": "r3_distractor", "contrast": f"{ma}_minus_{mb}", **c})

    # Aggregate Track 2: stage_lock_t48
    sl_all_methods = [m[0] for m in sl_neural_methods] + list(sl_baselines)
    sl_rows: list[dict] = []
    sl_mats: dict[str, np.ndarray] = {}
    for m_name in sl_all_methods:
        evs = [results_map[("stage_lock_t48", m_name, s)]["eval"] for s in seeds]
        mat = np.stack([e["total_return"] for e in evs], axis=0)
        sl_mats[m_name] = mat
        s_means = np.mean(mat, axis=1)
        q25, q50, q75 = np.percentile(mat, [25, 50, 75])
        gw = np.mean([results_map[("stage_lock_t48", m_name, s)]["meta"]["final_gate_weights"] for s in seeds], axis=0)
        wh = np.mean([results_map[("stage_lock_t48", m_name, s)]["meta"]["warmup_pos_hits_by_stage"] for s in seeds], axis=0)
        sl_rows.append({
            "method": m_name,
            "is_privileged": bool(m_name == "bayes_optimal"),
            "mean_return": float(np.mean(s_means)),
            "seed_std": float(np.std(s_means, ddof=1)),
            "median_return": float(q50),
            "iqr_return": float(q75 - q25),
            "early_auc_1_8": float(np.mean([np.mean(e["early_auc"]) for e in evs])),
            "opp_return": float(np.mean([np.mean(e["opp_return"]) for e in evs])),
            "reach_s0_unlock": float(np.mean([np.mean(e["reach_stage0_unlock"]) for e in evs])),
            "reach_stage1": float(np.mean([np.mean(e["reach_stage1"]) for e in evs])),
            "reach_s1_unlock": float(np.mean([np.mean(e["reach_stage1_unlock"]) for e in evs])),
            "reach_stage2": float(np.mean([np.mean(e["reach_stage2"]) for e in evs])),
            "reach_s2_unlock": float(np.mean([np.mean(e["reach_stage2_unlock"]) for e in evs])),
            "mean_max_stage": float(np.mean([np.mean(e["max_stage"]) for e in evs])),
            "tier0_only": float(np.mean([np.mean(e["tier0_only"]) for e in evs])),
            "seeds_reach_s2": int(sum(float(np.mean(e["reach_stage2_unlock"])) > 0.5 for e in evs)),
            "gate_y": float(gw[0]),
            "gate_r": float(gw[1]),
            "gate_noise1": float(gw[2]),
            "gate_noise2": float(gw[3]),
            "warmup_hits_s0": float(wh[0]),
            "warmup_hits_s1": float(wh[1]),
            "warmup_hits_s2": float(wh[2]),
        })

    sl_contrasts: list[dict] = []
    for ma, mb in (
        ("drqn_aux_yr", "drqn_vanilla"),
        ("drqn_aux_yr", "drqn_aux_yr_masked"),
        ("drqn_aux_yr", "drqn_prior_only"),
        ("drqn_aux_yr", "structural_model_sl"),
        ("drqn_aux_yr", "arm_ucb"),
        ("drqn_aux_y", "drqn_vanilla"),
        ("drqn_aux_r", "drqn_vanilla"),
        ("drqn_aux_discovery", "drqn_vanilla"),
        ("drqn_aux_discovery", "drqn_aux_uniform"),
        ("drqn_aux_discovery", "drqn_aux_yr"),
        ("drqn_vanilla", "arm_ucb"),
    ):
        c = _verdict_with_floor_guard(sl_mats[ma], sl_mats[mb], floor_val=9.6, n_boot=n_boot, delta=delta, seed=2026)
        sl_contrasts.append({"track": "stage_lock_t48", "contrast": f"{ma}_minus_{mb}", **c})

    wall_s = time.perf_counter() - t0
    summary = {
        "study_id": cfg["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "wall_seconds": round(wall_s, 2),
        "gate0_stage_lock": gate0,
        "r3_distractor_methods": r3_rows,
        "r3_distractor_contrasts": r3_contrasts,
        "stage_lock_methods": sl_rows,
        "stage_lock_contrasts": sl_contrasts,
    }

    for key in ("output_dir", "results_dir"):
        if key not in cfg:
            continue
        root = Path(cfg[key])
        _atomic_write_json(root / "summary.json", summary)
        _write_csv(root / "tables" / "r3_distractor_methods.csv", r3_rows)
        _write_csv(root / "tables" / "r3_distractor_contrasts.csv", r3_contrasts)
        _write_csv(root / "tables" / "stage_lock_methods.csv", sl_rows)
        _write_csv(root / "tables" / "stage_lock_contrasts.csv", sl_contrasts)

    print(f"[DONE] Exp 4B finished in {wall_s:.1f}s -> {cfg.get('results_dir')}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SPEC v3.1 Block 2 (Exp 4B) Compact Stage-Lock & R3+Distractor.")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_exp4b(cfg, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
