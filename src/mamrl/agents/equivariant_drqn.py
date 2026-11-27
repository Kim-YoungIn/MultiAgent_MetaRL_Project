"""Exact S_N Permutation-Equivariant Recurrent Q-Network (EquivariantDRQN).

Anti-Memorization Architecture (SPEC v3.0 / Independent Review §5):
- Maintains one recurrent hidden vector h_t^{(i)} in R^d per action token i in {0..N-1}
  with strictly SHARED weights and ZERO action-ID positional embeddings.
- Uses permutation-invariant mean-pool context h_bar = mean_j h^{(j)} so every action
  token compares its interaction history against all other actions while guaranteeing
  exact S_N permutation equivariance:
      Q(pi(obs_seq)) == pi(Q(obs_seq))
      pred_opp_logits(pi(obs_seq), pi(a)) == pi(pred_opp_logits(obs_seq, a))
      pred_r(pi(obs_seq), pi(a)) == pred_r(obs_seq, a)
  for every permutation pi in S_N.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mamrl.study_exp2b import SymPolicy

TOKEN_FEAT_DIM = 6  # [is_prev_self, is_prev_opp, self_r_if_played, global_r, is_start, turn_frac]


def encode_equivariant_tokens_batch(
    obs: dict[str, np.ndarray],
    *,
    n_actions: int = 4,
    horizon: int = 24,
    step_t: int = 0,
) -> np.ndarray:
    """Encode public observation into permutation-equivariant token features (B, N, 6)."""
    b = int(obs["prev_self_action"].shape[0])
    x = np.zeros((b, n_actions, TOKEN_FEAT_DIM), dtype=np.float32)
    idx = np.arange(b)

    pa = obs["prev_self_action"]
    po = obs["prev_opp_action"]
    pr = obs["prev_self_reward"].astype(np.float32)

    valid_a = (pa >= 0) & (pa < n_actions)
    x[idx[valid_a], pa[valid_a], 0] = 1.0
    x[idx[valid_a], pa[valid_a], 2] = pr[valid_a]

    valid_o = (po >= 0) & (po < n_actions)
    x[idx[valid_o], po[valid_o], 1] = 1.0

    x[:, :, 3] = pr[:, None]
    x[:, :, 4] = obs["is_start"].astype(np.float32)[:, None]
    x[:, :, 5] = float(step_t) / max(1.0, float(horizon))
    return x


class EquivariantRecurrentQNet(nn.Module):
    """Exact S_N permutation-equivariant tokenwise GRU Q-network with equivariant aux heads."""

    def __init__(self, token_dim: int = TOKEN_FEAT_DIM, hidden_dim: int = 32, n_actions: int = 4) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)

        # Shared tokenwise GRUCell receives [x_t^{(i)}, mean_j h_{t-1}^{(j)}] -> h_t^{(i)}
        self.gru_cell = nn.GRUCell(self.token_dim + self.hidden_dim, self.hidden_dim)

        # Shared equivariant Q-value head: [h_t^{(i)}, mean_j h_t^{(j)}] -> Q_t(i)
        self.q_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Shared equivariant next-opponent-action head:
        # For each candidate action j given chosen action a: [h_t^{(j)}, h_t^{(a)}, 1[j==a], mean_h] -> logit(j)
        self.pred_opp_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Shared permutation-invariant next-reward head: [h_t^{(a)}, mean_h] -> r_hat
        self.pred_r_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.n_actions, self.hidden_dim), dtype=torch.float32, device=device)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step: x_t (B, N, F), h (B, N, D) -> q_vals (B, N), h_next (B, N, D)."""
        b, n, _ = x_t.shape
        h_mean = h.mean(dim=1, keepdim=True).expand(b, n, self.hidden_dim)
        inp = torch.cat([x_t, h_mean], dim=-1).reshape(b * n, self.token_dim + self.hidden_dim)
        h_next = self.gru_cell(inp, h.reshape(b * n, self.hidden_dim)).reshape(b, n, self.hidden_dim)
        h_next_mean = h_next.mean(dim=1, keepdim=True).expand(b, n, self.hidden_dim)
        q_vals = self.q_head(torch.cat([h_next, h_next_mean], dim=-1)).squeeze(-1)  # (B, N)
        return q_vals, h_next

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full episode sequence: x_seq (B, T, N, F) -> q_seq (B, T, N), h_seq (B, T, N, D)."""
        b, t_len, n, _ = x_seq.shape
        h = self.init_hidden(b, x_seq.device)
        q_list = []
        h_list = []
        for t in range(t_len):
            q_t, h = self.step(x_seq[:, t], h)
            q_list.append(q_t)
            h_list.append(h)
        return torch.stack(q_list, dim=1), torch.stack(h_list, dim=1)

    def predict_aux(self, h_seq: torch.Tensor, act_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivariant opp-action logits (B, T, N) and invariant reward pred (B, T)."""
        b, t_len, n, d = h_seq.shape
        h_mean = h_seq.mean(dim=2, keepdim=True)  # (B, T, 1, D)
        # Gather chosen-action token hidden h_seq[b, t, a_t, :] -> (B, T, D)
        a_idx = act_seq.unsqueeze(-1).unsqueeze(-1).expand(b, t_len, 1, d)
        h_chosen = h_seq.gather(dim=2, index=a_idx)  # (B, T, 1, D)

        # Equivariant opponent logits for every candidate j in {0..N-1}
        a_oh = nn.functional.one_hot(act_seq, num_classes=n).float().unsqueeze(-1)  # (B, T, N, 1)
        opp_in = torch.cat(
            [h_seq, h_chosen.expand(b, t_len, n, d), a_oh, h_mean.expand(b, t_len, n, d)],
            dim=-1,
        )
        opp_logits = self.pred_opp_head(opp_in).squeeze(-1)  # (B, T, N)

        # Permutation-invariant reward prediction
        r_in = torch.cat([h_chosen.squeeze(2), h_mean.squeeze(2)], dim=-1)  # (B, T, 2D)
        r_pred = self.pred_r_head(r_in).squeeze(-1)  # (B, T)
        return opp_logits, r_pred


class EquivariantDRQNPolicy(SymPolicy):
    """Evaluation wrapper for trained EquivariantRecurrentQNet."""

    def __init__(
        self,
        model: EquivariantRecurrentQNet,
        *,
        name: str,
        horizon: int = 24,
        eps_eval: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.name = name
        self.horizon = int(horizon)
        self.eps_eval = float(eps_eval)
        self.device = torch.device(device)
        self._h: torch.Tensor | None = None
        self._rng = np.random.default_rng(0)

    def reset(self, batch_size: int, seed: int) -> None:
        self._batch = batch_size
        self._h = self.model.init_hidden(batch_size, self.device)
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], step_t: int) -> np.ndarray:
        assert self._h is not None
        x_np = encode_equivariant_tokens_batch(
            obs,
            n_actions=self.model.n_actions,
            horizon=self.horizon,
            step_t=step_t,
        )
        qv, self._h = self.model.step(torch.from_numpy(x_np).to(self.device), self._h)
        # Random tie-break among exactly equal Q-values (important at t=0 when all 4 tokens are identical!)
        q_np = qv.cpu().numpy()
        noise = self._rng.uniform(0.0, 1e-7, size=q_np.shape)
        greedy = np.argmax(q_np + noise, axis=-1).astype(np.int64)
        if self.eps_eval <= 0.0:
            return greedy
        rand_a = self._rng.integers(0, self.model.n_actions, size=self._batch, dtype=np.int64)
        return np.where(self._rng.random(self._batch) < self.eps_eval, rand_a, greedy).astype(np.int64)
