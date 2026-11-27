"""Exploration policies: Greedy, Random, UCB, Oracle, and Recurrent Meta-Explorer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from mamrl.envs.active_probe_game import ActiveProbeGame, ActiveProbeSpec, ROLE_PROBE
from mamrl.envs.base import Observation
from .base import Explorer, Student
from .opponents import build_opponent
from .students import build_student


class GreedyExplorer(Explorer):
    name = "greedy"

    def reset(self, seed: int | None = None) -> None:
        return

    def act(self, obs: Observation, student: Student, step_idx: int) -> int:
        return student.act(obs)


class RandomExplorer(Explorer):
    name = "random"

    def __init__(self, n_actions: int = 4, seed: int = 0) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation, student: Student, step_idx: int) -> int:
        return int(self._rng.integers(0, self.n_actions))


class UCBExplorer(Explorer):
    name = "ucb"

    def __init__(self, beta: float = 0.5, seed: int = 0) -> None:
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation, student: Student, step_idx: int) -> int:
        means, unc = student.action_values_and_uncertainty(obs)
        scores = means + self.beta * unc
        best = np.flatnonzero(np.isclose(scores, np.max(scores), atol=1e-9))
        return int(self._rng.choice(best))


class OracleExplorer(Explorer):
    """Privileged reference upper bound: probes until regime is identified, then exploits."""

    name = "oracle"

    def __init__(self, spec: ActiveProbeSpec) -> None:
        self.spec = spec

    def reset(self, seed: int | None = None) -> None:
        return

    def act(self, obs: Observation, student: Student, step_idx: int) -> int:
        # Probe on turn 0 in active mode if uncertainty remains; otherwise let Bayesian student exploit
        if self.spec.mode == "active" and step_idx == 0:
            return self.spec.role_to_action[ROLE_PROBE]
        return student.act(obs)


class _GRUPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)
        self.hidden_dim = hidden_dim

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_next = self.gru(x, h)
        logits = self.head(h_next)
        return logits, h_next


class RecurrentMetaExplorer(Explorer):
    """Meta-learned GRU exploration policy with optional feedback masking control."""

    def __init__(
        self,
        n_actions: int = 4,
        hidden_dim: int = 32,
        mask_feedback: bool = False,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.mask_feedback = mask_feedback
        self.name = "meta_masked" if mask_feedback else "meta_full"
        torch.manual_seed(seed)
        # Input: one-hot prev_self (n) + one-hot prev_opp (n) + prev_reward (1) + is_start (1) + step_frac (1)
        self.input_dim = 2 * n_actions + 3
        self.net = _GRUPolicyNet(self.input_dim, hidden_dim, n_actions)
        self._h = torch.zeros(1, hidden_dim)
        self._eval_rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self._h = torch.zeros(1, self.hidden_dim)
        if seed is not None:
            self._eval_rng = np.random.default_rng(seed)

    def _encode(self, obs: Observation, step_idx: int, max_k: int = 8) -> torch.Tensor:
        vec = np.zeros(self.input_dim, dtype=np.float32)
        if 0 <= obs.prev_self_action < self.n_actions:
            vec[obs.prev_self_action] = 1.0
        if not self.mask_feedback:
            if 0 <= obs.prev_opp_action < self.n_actions:
                vec[self.n_actions + obs.prev_opp_action] = 1.0
            vec[2 * self.n_actions] = float(obs.prev_self_reward)
        vec[2 * self.n_actions + 1] = 1.0 if obs.is_start else 0.0
        vec[2 * self.n_actions + 2] = float(step_idx) / max(1.0, float(max_k))
        return torch.from_numpy(vec).unsqueeze(0)

    def act_with_log_prob(self, obs: Observation, step_idx: int, max_k: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        x = self._encode(obs, step_idx, max_k=max_k)
        logits, self._h = self.net.step(x, self._h)
        dist = Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def act(self, obs: Observation, student: Student, step_idx: int) -> int:
        x = self._encode(obs, step_idx)
        logits, self._h = self.net.step(x, self._h)
        probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
        return int(self._eval_rng.choice(self.n_actions, p=probs))


def train_meta_explorer(
    train_tasks: list[ActiveProbeSpec],
    *,
    opponent_kind: str = "q_learning",
    student_kind: str = "bayesian",
    explore_steps: int = 4,
    total_horizon: int = 32,
    updates: int = 80,
    batch_size: int = 16,
    lr: float = 1e-2,
    entropy_coef: float = 0.01,
    hidden_dim: int = 32,
    mask_feedback: bool = False,
    objective: str = "total_return",
    seed: int = 0,
) -> tuple[RecurrentMetaExplorer, list[float]]:
    """Train RecurrentMetaExplorer via baseline-reduced REINFORCE across train_tasks."""
    if objective not in ("total_return", "assessment_return"):
        raise ValueError(f"Unsupported objective: {objective}")

    n_actions = train_tasks[0].n_actions
    explorer = RecurrentMetaExplorer(
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        mask_feedback=mask_feedback,
        seed=seed,
    )
    opt = torch.optim.Adam(explorer.net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1000)
    history: list[float] = []

    for upd in range(updates):
        log_probs_batch: list[torch.Tensor] = []
        entropies_batch: list[torch.Tensor] = []
        returns_batch: list[float] = []

        for b_idx in range(batch_size):
            task = train_tasks[int(rng.integers(0, len(train_tasks)))]
            ep_seed = int(rng.integers(0, 2**31 - 1))
            env = ActiveProbeGame(task, seed=ep_seed)
            opp = build_opponent(opponent_kind, n_actions=n_actions, seed=ep_seed + 1)
            student = build_student(student_kind, n_actions=n_actions, spec=task, seed=ep_seed + 2)
            obs_f, obs_o = env.reset(seed=ep_seed)
            opp.reset(seed=ep_seed + 1)
            student.reset(seed=ep_seed + 2)
            explorer.reset()

            ep_log_probs: list[torch.Tensor] = []
            ep_entropies: list[torch.Tensor] = []
            explore_ret = 0.0
            assess_ret = 0.0

            for t in range(total_horizon):
                if t < explore_steps:
                    a_f, lp, ent = explorer.act_with_log_prob(obs_f, t, explore_steps)
                    ep_log_probs.append(lp)
                    ep_entropies.append(ent)
                else:
                    a_f = student.act(obs_f)

                a_o = opp.act(obs_o)
                res = env.step(a_f, a_o)
                student.update(obs_f, a_f, res.focal_reward, res.focal_obs)
                opp.update(obs_o, a_o, res.opp_reward, res.opp_obs)
                obs_f, obs_o = res.focal_obs, res.opp_obs

                if t < explore_steps:
                    explore_ret += res.focal_reward
                else:
                    assess_ret += res.focal_reward

            score = (explore_ret + assess_ret) if objective == "total_return" else assess_ret
            log_probs_batch.append(torch.stack(ep_log_probs).sum())
            entropies_batch.append(torch.stack(ep_entropies).mean())
            returns_batch.append(score)

        ret_t = torch.tensor(returns_batch, dtype=torch.float32)
        adv = ret_t - ret_t.mean()
        std = ret_t.std(unbiased=False)
        if float(std.item()) > 1e-6:
            adv = adv / std

        lp_t = torch.stack(log_probs_batch)
        ent_t = torch.stack(entropies_batch)
        loss = -(lp_t * adv.detach()).mean() - entropy_coef * ent_t.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(explorer.net.parameters(), 2.0)
        opt.step()
        history.append(float(ret_t.mean().item()))

    return explorer, history
