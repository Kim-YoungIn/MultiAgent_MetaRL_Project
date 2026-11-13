"""Action-label-equivariant shared recurrent actor-critic.

Each candidate has the same update and scoring functions. The observations
indicate relations to the previous actions, without ordinal action features or
action-specific learned parameters.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import finite_reward, positive_integer, validate_action, validate_n, validate_seed
from .environment import Observation
from .recurrent import RecurrentActorCritic


class ActionSetActorCritic(RecurrentActorCritic):
    """Maintain one shared GRU memory row per available candidate action.

    In the memoryless ablation the same network is trained but every forward
    starts from zero hidden rows; no learned history is carried across actions.
    """

    def __init__(self, hidden_size: int = 32, seed: int = 0, *, memoryless: bool = False):
        # Initialize only this architecture, while remaining a subclass of the
        # existing actor-critic for Policy/collector compatibility.
        nn.Module.__init__(self)
        self.hidden_size = positive_integer(hidden_size, "hidden_size")
        seed = validate_seed(seed)
        if seed is None:
            raise TypeError("model initialization seed must be an integer")
        if type(memoryless) is not bool:
            raise TypeError("memoryless must be bool")
        self.memoryless = memoryless
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed % (2**63))
            self.gru = nn.GRUCell(5, self.hidden_size)
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
            )

    @property
    def architecture_kind(self) -> str:
        return "action_set_ff_v1" if self.memoryless else "action_set_v1"

    def initial_hidden(self, *, n: int) -> torch.Tensor:
        n = validate_n(n)
        return torch.zeros((n, self.hidden_size), dtype=torch.float32, device="cpu")

    def forward_observation(
        self, observation: Observation, hidden: torch.Tensor, *, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = validate_n(n)
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        own = validate_action(observation.previous_own_action, n)
        opponent = validate_action(observation.previous_opponent_action, n)
        reward = finite_reward(observation.previous_own_reward)
        if type(observation.episode_start_flag) is not bool:
            raise TypeError("episode_start_flag must be bool")
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if hidden.shape != (n, self.hidden_size):
            raise ValueError(f"hidden must have shape ({n}, {self.hidden_size})")
        if hidden.device.type != "cpu" or hidden.dtype != torch.float32:
            raise ValueError("hidden must be a float32 CPU tensor")
        if not torch.isfinite(hidden).all().item():
            raise ValueError("hidden must be finite")
        actions = torch.arange(n, device="cpu")
        features = torch.stack((
            (actions == own).float(), (actions == opponent).float(),
            torch.full((n,), math.tanh(reward / 4), dtype=torch.float32),
            torch.full((n,), float(observation.episode_start_flag), dtype=torch.float32),
            torch.full((n,), 1 / n, dtype=torch.float32),
        ), dim=1)
        incoming = torch.zeros_like(hidden) if self.memoryless else hidden
        next_hidden = self.gru(features, incoming)
        logits = self.actor(next_hidden).squeeze(-1)
        value = self.critic(next_hidden.mean(dim=0)).squeeze(-1)
        if not (torch.isfinite(next_hidden).all().item()
                and torch.isfinite(logits).all().item() and torch.isfinite(value).item()):
            raise ValueError("model output must be finite")
        return logits, value, next_hidden
