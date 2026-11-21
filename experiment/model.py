"""Shared recurrent action model for D-031."""
import math
import torch
from torch import nn


class ContextActor(nn.Module):
    def __init__(self, hidden_size=32, seed=0, masked=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.masked = masked
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed % (2**63))
            self.gru = nn.GRUCell(5, hidden_size)
            self.actor = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
            self.critic = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    @property
    def device(self):
        return next(self.parameters()).device

    def initial_hidden(self, *, n):
        return torch.zeros(n, self.hidden_size, device=self.device)

    def forward_observations(self, observations, hidden, *, n):
        rows = []
        for obs in observations:
            masked = self.masked and not obs.episode_start_flag
            rows.append([[float(a == obs.previous_own_action),
                          0.0 if masked else float(a == obs.previous_opponent_action),
                          0.0 if masked else math.tanh(obs.previous_own_reward / 4),
                          float(obs.episode_start_flag), 1 / n] for a in range(n)])
        features = torch.tensor(rows, dtype=torch.float32, device=self.device)
        out = self.gru(features.reshape(-1, 5), hidden.reshape(-1, self.hidden_size))
        out = out.reshape(len(observations), n, self.hidden_size)
        mean = out.mean(dim=1)
        logits = self.actor(torch.cat([out, mean[:, None, :].expand_as(out)], dim=-1)).squeeze(-1)
        return logits, self.critic(mean).squeeze(-1), out

    def forward_observation(self, observation, hidden, *, n):
        logits, value, updated = self.forward_observations([observation], hidden.unsqueeze(0), n=n)
        return logits[0], value[0], updated[0]
