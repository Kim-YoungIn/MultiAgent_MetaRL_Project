"""Off-Policy Recurrent Double-DQN (DRQN) with Clean Self-Supervised Auxiliary Signal Ablations.

Strictly unprivileged: observes only public observations via encode_obs_batch().
Supports orthogonal ablations:
- variant: 'rl2' (full feedback), 'rl2_masked' (masked t < mask_k, no retrospective reveal,
  aux loss also masked for t < mask_k), 'prior_only' (no feedback at any turn, aux disabled).
- aux_mode: 'none' (pure reward), 'y' (+next opponent action), 'r' (+next reward), 'yr' (+both).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn

from mamrl.agents.baselines_v3 import BatchPolicyV3
from mamrl.agents.ppo_rl2 import OBS_DIM, encode_obs_batch
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


class RecurrentQNet(nn.Module):
    """GRU-based Recurrent Q-Network with optional self-supervised predictive heads."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, n_actions)
        self.pred_y = nn.Linear(hidden_dim + n_actions, N_OPP_SYMBOLS)
        self.pred_r = nn.Linear(hidden_dim + n_actions, 1)

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x_seq)
        return self.q_head(out), out

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_next = self.gru(x_t, h)
        q_vals = self.q_head(out.squeeze(1))
        return q_vals, h_next


@dataclass
class DRQNConfig:
    regime: str = "R3"
    variant: str = "rl2"      # "rl2", "rl2_masked", "prior_only"
    aux_mode: str = "yr"      # "none", "y", "r", "yr"
    mask_k: int = 4
    horizon: int = 24
    hidden_dim: int = 64
    batch_episodes: int = 64
    n_iters: int = 450        # 450 * 64 * 24 = 691,200 transitions
    warmup_iters: int = 100
    replay_capacity: int = 8192
    grad_steps_per_iter: int = 8
    minibatch_episodes: int = 128
    gamma: float = 1.0
    lr: float = 3e-3
    polyak_tau: float = 0.05
    aux_y_weight: float = 2.0
    aux_r_weight: float = 1.0
    eps_start: float = 0.90
    eps_end: float = 0.05
    max_grad_norm: float = 5.0
    log_interval: int = 10
    seed: int = 0
    device: str = "cpu"
    reset_prob: float = 0.0   # Optional mid-episode restore probability at reset_step (for Exp 2A)
    reset_step: int = 4


def compute_aux_losses(
    qnet: RecurrentQNet,
    hidden_seq: torch.Tensor,
    act_seq: torch.Tensor,
    y_seq: torch.Tensor,
    r_seq: torch.Tensor,
    *,
    variant: str,
    aux_mode: str,
    mask_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute self-supervised next-(y, r) prediction losses respecting masking boundaries."""
    device = hidden_seq.device
    zero = torch.tensor(0.0, device=device)
    if aux_mode == "none" or variant == "prior_only":
        return zero, zero

    t_len = hidden_seq.shape[1]
    start_t = mask_k if variant == "rl2_masked" else 0
    if start_t >= t_len:
        return zero, zero

    hs = hidden_seq[:, start_t:, :]
    ax = act_seq[:, start_t:]
    yx = y_seq[:, start_t:]
    rx = r_seq[:, start_t:]
    a_oh = nn.functional.one_hot(ax, num_classes=N_ACTIONS).float()
    ha = torch.cat([hs, a_oh], dim=-1)

    y_loss = zero
    r_loss = zero
    if aux_mode in ("y", "yr"):
        y_loss = nn.functional.cross_entropy(qnet.pred_y(ha).reshape(-1, N_OPP_SYMBOLS), yx.reshape(-1))
    if aux_mode in ("r", "yr"):
        r_loss = nn.functional.mse_loss(qnet.pred_r(ha).squeeze(-1), rx)
    return y_loss, r_loss


def train_drqn_rl2(
    cfg: DRQNConfig,
    *,
    opponent_factory: Callable[[int, int], Any] | None = None,
) -> tuple[RecurrentQNet, list[dict[str, float]], int]:
    """Train RecurrentQNet via off-policy Double-DQN + optional self-supervised predictive aux."""
    if cfg.variant not in ("rl2", "rl2_masked", "prior_only"):
        raise ValueError(f"Unsupported variant: {cfg.variant}")
    if cfg.aux_mode not in ("none", "y", "r", "yr"):
        raise ValueError(f"Unsupported aux_mode: {cfg.aux_mode}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    qnet = RecurrentQNet(OBS_DIM, cfg.hidden_dim, N_ACTIONS).to(device)
    tnet = RecurrentQNet(OBS_DIM, cfg.hidden_dim, N_ACTIONS).to(device)
    tnet.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)

    b = cfg.batch_episodes
    t_len = cfg.horizon
    env = VectorizedDelayedResponseEnv(cfg.regime, batch_size=b, horizon=t_len, seed=cfg.seed + 17)
    opp = opponent_factory(b, cfg.seed + 53) if opponent_factory is not None else None
    rng = np.random.default_rng(cfg.seed + 31)

    cap = cfg.replay_capacity
    buf_obs = np.zeros((cap, t_len, OBS_DIM), dtype=np.float32)
    buf_act = np.zeros((cap, t_len), dtype=np.int64)
    buf_y = np.zeros((cap, t_len), dtype=np.int64)
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

        obs = env.reset()
        if opp is not None:
            opp.reset(env.theta)
        h = torch.zeros((1, b, cfg.hidden_dim), dtype=torch.float32, device=device)
        ep_o = np.zeros((b, t_len, OBS_DIM), dtype=np.float32)
        ep_a = np.zeros((b, t_len), dtype=np.int64)
        ep_y = np.zeros((b, t_len), dtype=np.int64)
        ep_r = np.zeros((b, t_len), dtype=np.float32)

        qnet.eval()
        with torch.no_grad():
            for t in range(t_len):
                x = encode_obs_batch(obs, horizon=t_len, step_t=t, variant=cfg.variant, mask_k=cfg.mask_k)
                ep_o[:, t] = x
                qv, h = qnet.step(torch.from_numpy(x).to(device).unsqueeze(1), h)
                greedy_a = qv.argmax(dim=-1).cpu().numpy()
                rand_a = rng.integers(0, N_ACTIONS, size=b)
                a = np.where(rng.random(b) < eps, rand_a, greedy_a).astype(np.int64)
                q_prev = env.q.copy()
                opp_a = opp.act(q_prev) if opp is not None else None
                obs, rf, ro, y = env.step(a, opp_actions=opp_a)
                if opp is not None:
                    opp.update(q_prev=q_prev, y_act=y, r_opp=ro, q_next=a)
                if cfg.reset_prob > 0.0 and (t + 1) == cfg.reset_step and (t + 1) < t_len:
                    reset_mask = rng.random(b) < cfg.reset_prob
                    if np.any(reset_mask):
                        obs = env.restore_public_state(obs, mask=reset_mask)
                        if opp is not None:
                            opp.restore(snap=None, mask=reset_mask)
                ep_a[:, t] = a
                ep_y[:, t] = y
                ep_r[:, t] = rf.astype(np.float32)

        total_transitions += b * t_len
        for k in range(b):
            buf_obs[ptr] = ep_o[k]
            buf_act[ptr] = ep_a[k]
            buf_y[ptr] = ep_y[k]
            buf_rew[ptr] = ep_r[k]
            ptr = (ptr + 1) % cap
            size = min(cap, size + 1)

        qnet.train()
        td_vals: list[float] = []
        ay_vals: list[float] = []
        ar_vals: list[float] = []
        mb_n = min(cfg.minibatch_episodes, size)

        for _ in range(cfg.grad_steps_per_iter):
            idx = rng.integers(0, size, size=mb_n)
            ox = torch.from_numpy(buf_obs[idx]).to(device)
            ax = torch.from_numpy(buf_act[idx]).to(device)
            yx = torch.from_numpy(buf_y[idx]).to(device)
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
            y_loss, r_loss = compute_aux_losses(
                qnet,
                hs,
                ax,
                yx,
                rx,
                variant=cfg.variant,
                aux_mode=cfg.aux_mode,
                mask_k=cfg.mask_k,
            )
            loss = td_loss + cfg.aux_y_weight * y_loss + cfg.aux_r_weight * r_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                tau = cfg.polyak_tau
                for p, tp in zip(qnet.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

            td_vals.append(float(td_loss.item()))
            ay_vals.append(float(y_loss.item()))
            ar_vals.append(float(r_loss.item()))

        if it == 1 or it == cfg.n_iters or (it % cfg.log_interval == 0):
            ep_totals = ep_r.sum(axis=1)
            open_a = ep_a[:, 0]
            k_split = min(cfg.mask_k, t_len)
            rep_exp = float(np.mean(ep_a[:, 1:k_split] == ep_a[:, : k_split - 1])) if k_split > 1 else 0.0
            rep_expt = float(np.mean(ep_a[:, k_split:] == ep_a[:, k_split - 1 : -1])) if t_len > k_split else 0.0
            train_curve.append(
                {
                    "update": float(it),
                    "transitions": float(total_transitions),
                    "eps": float(eps),
                    "mean_return": float(np.mean(ep_totals)),
                    "median_return": float(np.median(ep_totals)),
                    "td_loss": float(np.mean(td_vals)),
                    "aux_y_loss": float(np.mean(ay_vals)),
                    "aux_r_loss": float(np.mean(ar_vals)),
                    "p_open_S": float(np.mean(open_a == ACT_S)),
                    "p_open_P": float(np.mean(open_a == ACT_P)),
                    "p_open_P0": float(np.mean(open_a == ACT_P0)),
                    "p_open_P1": float(np.mean(open_a == ACT_P1)),
                    "p_open_X": float(np.mean(open_a >= ACT_X00)),
                    "repeat_rate_explore": rep_exp,
                    "repeat_rate_exploit": rep_expt,
                }
            )

    qnet.eval()
    return qnet, train_curve, total_transitions


class NeuralDRQNPolicyV3(BatchPolicyV3):
    """Greedy evaluation wrapper for trained RecurrentQNet."""

    def __init__(
        self,
        model: RecurrentQNet,
        *,
        name: str,
        variant: str,
        horizon: int = 24,
        mask_k: int = 4,
        offline_train_transitions: int = 0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.name = name
        self.variant = variant
        self.horizon = horizon
        self.mask_k = mask_k
        self.offline_train_transitions = int(offline_train_transitions)
        self.device = torch.device(device)
        self._h: torch.Tensor | None = None

    def reset(self, batch_size: int, seed: int, true_theta: np.ndarray | None = None) -> None:
        self._h = torch.zeros((1, batch_size, self.model.hidden_dim), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        assert self._h is not None
        x_np = encode_obs_batch(
            obs,
            horizon=self.horizon,
            step_t=step_t,
            variant=self.variant,
            mask_k=self.mask_k,
        )
        qv, self._h = self.model.step(torch.from_numpy(x_np).to(self.device).unsqueeze(1), self._h)
        return qv.argmax(dim=-1).cpu().numpy().astype(np.int64)

    def last_hidden(self) -> np.ndarray | None:
        if self._h is None:
            return None
        return self._h.squeeze(0).cpu().numpy().copy()
