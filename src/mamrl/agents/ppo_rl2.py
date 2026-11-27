"""End-to-End Recurrent Meta-RL (RL² / RL²-Masked / Prior-Only) with PPO + GAE.

Strictly unprivileged: observes only public (prev_self_action, prev_opp_action,
prev_self_reward, is_start, reset_flag, turn_frac). Zero hidden-type leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mamrl.envs.delayed_response import (
    ACT_P,
    ACT_P0,
    ACT_P1,
    ACT_S,
    ACT_X00,
    N_ACTIONS,
    N_OPP_SYMBOLS,
    VectorizedDelayedResponseEnv,
)

OBS_DIM = N_ACTIONS + N_OPP_SYMBOLS + 4  # 8 + 4 + 4 = 16


def encode_obs_batch(
    obs: dict[str, np.ndarray],
    *,
    horizon: int,
    step_t: int,
    variant: str,
    mask_k: int = 4,
) -> np.ndarray:
    """Encode vectorized public observation into float32 array of shape (B, 16).

    Variants:
    - 'rl2': full public feedback at every turn.
    - 'rl2_masked': hides opponent action and reward for step_t < mask_k (no retrospective reveal).
    - 'prior_only': hides opponent action and reward at ALL turns (true zero-shot blind baseline).
    """
    b = int(obs["prev_self_action"].shape[0])
    x = np.zeros((b, OBS_DIM), dtype=np.float32)
    idx = np.arange(b)

    prev_a = obs["prev_self_action"]
    valid_a = (prev_a >= 0) & (prev_a < N_ACTIONS)
    x[idx[valid_a], prev_a[valid_a]] = 1.0

    # At step_t, obs['prev_opp_action'] and obs['prev_self_reward'] come from turn step_t (0-indexed: t = step_t - 1).
    # To hide ALL feedback generated during turns 1..mask_k (t = 0..mask_k-1) with zero retrospective revelation,
    # feedback is only enabled for step_t > mask_k (starting at step_t = mask_k + 1, which observes turn mask_k + 1).
    allow_feedback = (variant == "rl2") or (variant == "rl2_masked" and step_t > mask_k)
    if allow_feedback:
        prev_y = obs["prev_opp_action"]
        valid_y = (prev_y >= 0) & (prev_y < N_OPP_SYMBOLS)
        x[idx[valid_y], N_ACTIONS + prev_y[valid_y]] = 1.0
        # Normalize reward gently into ~[-1, 1] scale while preserving exact sign/magnitude ratios
        x[:, N_ACTIONS + N_OPP_SYMBOLS] = (obs["prev_self_reward"] / 2.0).astype(np.float32)

    x[:, N_ACTIONS + N_OPP_SYMBOLS + 1] = obs["is_start"].astype(np.float32)
    x[:, N_ACTIONS + N_OPP_SYMBOLS + 2] = obs["reset_flag"].astype(np.float32)
    x[:, N_ACTIONS + N_OPP_SYMBOLS + 3] = float(step_t) / max(1.0, float(horizon))
    return x


class RecurrentActorCritic(nn.Module):
    """GRU-based Actor-Critic for full-episode recurrent PPO."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization for stable PPO training
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step: x_t (B, 1, D), h (1, B, H) -> logits (B, A), value (B,), h_next (1, B, H)."""
        out, h_next = self.gru(x_t, h)
        feat = out.squeeze(1)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value, h_next

    def forward_sequence(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full sequence: x_seq (B, T, D) -> logits (B, T, A), values (B, T), hidden_seq (B, T, H)."""
        out, _ = self.gru(x_seq)
        logits = self.policy_head(out)
        values = self.value_head(out).squeeze(-1)
        return logits, values, out


@dataclass
class PPOConfig:
    regime: str = "R3"
    variant: str = "rl2"  # "rl2", "rl2_masked", "prior_only"
    mask_k: int = 4
    horizon: int = 24
    hidden_dim: int = 64
    batch_episodes: int = 256
    total_transitions: int = 262144  # 2^18
    ppo_epochs: int = 4
    minibatches: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 1e-3
    entropy_coef: float = 0.015
    entropy_coef_end: float | None = None
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    log_interval: int = 2
    seed: int = 0
    device: str = "cpu"


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-episode GAE with terminal value = 0 at t = T."""
    b, t_len = rewards.shape
    adv = np.zeros((b, t_len), dtype=np.float32)
    last_gae = np.zeros(b, dtype=np.float32)
    for t in reversed(range(t_len)):
        next_val = values[:, t + 1] if (t + 1 < t_len) else np.zeros(b, dtype=np.float32)
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae
        adv[:, t] = last_gae
    returns = adv + values
    return adv, returns


def train_ppo_rl2(cfg: PPOConfig) -> tuple[RecurrentActorCritic, list[dict[str, float]], int]:
    """Train RecurrentActorCritic via full-episode vectorized PPO + GAE."""
    if cfg.variant not in ("rl2", "rl2_masked", "prior_only"):
        raise ValueError(f"Unsupported variant: {cfg.variant}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    model = RecurrentActorCritic(OBS_DIM, cfg.hidden_dim, N_ACTIONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    env = VectorizedDelayedResponseEnv(
        regime=cfg.regime,
        batch_size=cfg.batch_episodes,
        horizon=cfg.horizon,
        seed=cfg.seed + 17,
    )
    steps_per_update = cfg.batch_episodes * cfg.horizon
    n_updates = max(1, (cfg.total_transitions + steps_per_update - 1) // steps_per_update)
    total_transitions_run = 0
    train_curve: list[dict[str, float]] = []

    for upd in range(1, n_updates + 1):
        obs = env.reset()
        obs_buf = np.zeros((cfg.batch_episodes, cfg.horizon, OBS_DIM), dtype=np.float32)
        act_buf = np.zeros((cfg.batch_episodes, cfg.horizon), dtype=np.int64)
        logp_buf = np.zeros((cfg.batch_episodes, cfg.horizon), dtype=np.float32)
        val_buf = np.zeros((cfg.batch_episodes, cfg.horizon), dtype=np.float32)
        rew_buf = np.zeros((cfg.batch_episodes, cfg.horizon), dtype=np.float32)

        h = torch.zeros((1, cfg.batch_episodes, cfg.hidden_dim), dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            for t in range(cfg.horizon):
                x_np = encode_obs_batch(
                    obs,
                    horizon=cfg.horizon,
                    step_t=t,
                    variant=cfg.variant,
                    mask_k=cfg.mask_k,
                )
                obs_buf[:, t, :] = x_np
                x_t = torch.from_numpy(x_np).to(device).unsqueeze(1)
                logits, values, h = model.step(x_t, h)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

                a_np = actions.cpu().numpy()
                act_buf[:, t] = a_np
                logp_buf[:, t] = logp.cpu().numpy()
                val_buf[:, t] = values.cpu().numpy()

                obs, rf, _ro, _y = env.step(a_np)
                rew_buf[:, t] = rf.astype(np.float32)

        total_transitions_run += steps_per_update
        adv_np, ret_np = _compute_gae(rew_buf, val_buf, cfg.gamma, cfg.gae_lambda)
        adv_norm = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        obs_t = torch.from_numpy(obs_buf).to(device)
        act_t = torch.from_numpy(act_buf).to(device)
        old_logp_t = torch.from_numpy(logp_buf).to(device)
        adv_t = torch.from_numpy(adv_norm).to(device)
        ret_t = torch.from_numpy(ret_np).to(device)

        # Explained variance before update
        var_y = float(np.var(ret_np))
        explained_var = 1.0 - float(np.var(ret_np - val_buf)) / (var_y + 1e-8) if var_y > 1e-12 else 0.0

        model.train()
        perm_rng = np.random.default_rng(cfg.seed + upd * 997)
        mb_size = max(1, cfg.batch_episodes // cfg.minibatches)
        kl_vals: list[float] = []
        clip_vals: list[float] = []
        ent_vals: list[float] = []
        vloss_vals: list[float] = []

        frac = float(upd - 1) / max(1.0, float(n_updates - 1))
        cur_ent_coef = (
            cfg.entropy_coef
            if cfg.entropy_coef_end is None
            else (cfg.entropy_coef + frac * (cfg.entropy_coef_end - cfg.entropy_coef))
        )

        for _epoch in range(cfg.ppo_epochs):
            perm = perm_rng.permutation(cfg.batch_episodes)
            for start in range(0, cfg.batch_episodes, mb_size):
                mb = perm[start : start + mb_size]
                logits, values, _ = model.forward_sequence(obs_t[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                log_ratio = new_logp - old_logp_t[mb]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - ret_t[mb]) ** 2).mean()

                loss = policy_loss + cfg.value_coef * value_loss - cur_ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    clip_frac = float((torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean().item())
                    kl_vals.append(approx_kl)
                    clip_vals.append(clip_frac)
                    ent_vals.append(float(entropy.item()))
                    vloss_vals.append(float(value_loss.item()))

        if upd == 1 or upd == n_updates or (upd % cfg.log_interval == 0):
            ep_totals = rew_buf.sum(axis=1)
            open_a = act_buf[:, 0]
            k_split = min(cfg.mask_k, cfg.horizon)
            rep_exp = float(np.mean(act_buf[:, 1:k_split] == act_buf[:, : k_split - 1])) if k_split > 1 else 0.0
            rep_expt = float(np.mean(act_buf[:, k_split:] == act_buf[:, k_split - 1 : -1])) if cfg.horizon > k_split else 0.0
            train_curve.append(
                {
                    "update": float(upd),
                    "transitions": float(total_transitions_run),
                    "mean_return": float(np.mean(ep_totals)),
                    "median_return": float(np.median(ep_totals)),
                    "policy_entropy": float(np.mean(ent_vals)),
                    "approx_kl": float(np.mean(kl_vals)),
                    "clip_frac": float(np.mean(clip_vals)),
                    "value_loss": float(np.mean(vloss_vals)),
                    "explained_var": float(explained_var),
                    "p_open_S": float(np.mean(open_a == ACT_S)),
                    "p_open_P": float(np.mean(open_a == ACT_P)),
                    "p_open_P0": float(np.mean(open_a == ACT_P0)),
                    "p_open_P1": float(np.mean(open_a == ACT_P1)),
                    "p_open_X": float(np.mean(open_a >= ACT_X00)),
                    "repeat_rate_explore": rep_exp,
                    "repeat_rate_exploit": rep_expt,
                }
            )

    model.eval()
    return model, train_curve, total_transitions_run
