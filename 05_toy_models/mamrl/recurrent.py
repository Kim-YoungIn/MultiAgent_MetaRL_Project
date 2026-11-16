"""A variable-action recurrent actor-critic and explicit-state policy adapter.

This optional module requires PyTorch. Core environment and random-policy
modules do not import it. Model inputs and RNGs follow the model device.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path

import torch
from torch import nn

from .common import finite_reward, positive_integer, validate_action, validate_n, validate_seed
from .environment import Observation
from .policy import PolicyOutput


class RecurrentActorCritic(nn.Module):
    """GRU state with a shared, per-candidate action scorer.

    Action indices are normalized by n-1. This explicitly gives action labels
    an order; it does not establish cross-n generalization of a trained model.
    """

    def __init__(self, hidden_size: int = 32, seed: int = 0):
        super().__init__()
        self.hidden_size = positive_integer(hidden_size, "hidden_size")
        seed = validate_seed(seed)
        if seed is None:
            raise TypeError("model initialization seed must be an integer")
        # Initialization is reproducible without advancing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed % (2**63))
            self.gru = nn.GRUCell(6, self.hidden_size)
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_size + 3, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
            )

    @property
    def device(self):
        return next(self.parameters()).device

    architecture_kind = "ordinal_v1"

    def initial_hidden(self, *, n: int | None = None) -> torch.Tensor:
        if n is not None:
            validate_n(n)
        return torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)

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
        if hidden.shape != (self.hidden_size,):
            raise ValueError(f"hidden must have shape ({self.hidden_size},)")
        if hidden.device != self.device or hidden.dtype != torch.float32:
            raise ValueError("hidden must match the model device and be float32")
        if not torch.isfinite(hidden).all().item():
            raise ValueError("hidden must be finite")

        features = torch.tensor(
            [own / (n - 1), opponent / (n - 1), float(own == opponent),
             math.tanh(reward / 4), float(observation.episode_start_flag), 1 / n],
            dtype=torch.float32, device=self.device,
        )
        next_hidden = self.gru(features, hidden)
        actions = torch.arange(n, dtype=torch.float32, device=self.device)
        candidates = torch.stack(
            (actions / (n - 1), (actions == own).float(), (actions == opponent).float()), dim=1
        )
        combined = torch.cat((next_hidden.unsqueeze(0).expand(n, -1), candidates), dim=1)
        logits = self.actor(combined).squeeze(-1)
        value = self.critic(next_hidden).squeeze(-1)
        if not (torch.isfinite(next_hidden).all().item()
                and torch.isfinite(logits).all().item() and torch.isfinite(value).item()):
            raise ValueError("model output must be finite")
        return logits, value, next_hidden


@dataclass(frozen=True, slots=True)
class RecurrentState:
    hidden: torch.Tensor
    generator: torch.Generator


class RecurrentPolicy:
    """Existing Policy interface with local sampling RNG and explicit memory."""

    def __init__(self, model: RecurrentActorCritic):
        if not isinstance(model, RecurrentActorCritic):
            raise TypeError("model must be a RecurrentActorCritic")
        self.model = model

    def initial_state(self, *, n: int, seed: int | None = None) -> RecurrentState:
        validate_n(n)
        seed = validate_seed(seed)
        generator = torch.Generator(device=self.model.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed % (2**63))
        return RecurrentState(self.model.initial_hidden(n=n), generator)

    def act(self, observation: Observation, state: RecurrentState, *, n: int) -> PolicyOutput:
        if not isinstance(state, RecurrentState) or not isinstance(state.generator, torch.Generator):
            raise TypeError("state must be a RecurrentState with a local torch.Generator")
        with torch.no_grad():
            logits, _, next_hidden = self.model.forward_observation(observation, state.hidden, n=n)
            probs = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probs, 1, generator=state.generator).item())
        return PolicyOutput(action, RecurrentState(next_hidden.detach(), state.generator))


def _expected_shapes(hidden_size: int, kind: str = "ordinal_v1") -> dict[str, tuple[int, ...]]:
    h = hidden_size
    if kind not in ("ordinal_v1", "action_set_v1", "action_set_ff_v1"):
        raise ValueError("unsupported model architecture kind")
    action_set = kind != "ordinal_v1"
    return {
        "gru.weight_ih": (3 * h, 5 if action_set else 6), "gru.weight_hh": (3 * h, h),
        "gru.bias_ih": (3 * h,), "gru.bias_hh": (3 * h,),
        "actor.0.weight": (h, h if action_set else h + 3), "actor.0.bias": (h,),
        "actor.2.weight": (1, h), "actor.2.bias": (1,),
        "critic.0.weight": (h, h), "critic.0.bias": (h,),
        "critic.2.weight": (1, h), "critic.2.bias": (1,),
    }


def _check_state_dict(state: object, hidden_size: int, kind: str = "ordinal_v1") -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError("state_dict must be a mapping of tensors")
    expected = _expected_shapes(hidden_size, kind)
    if set(state) != set(expected):
        raise ValueError("state_dict keys do not match the model architecture")
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a tensor")
        if value.device.type != "cpu" or value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"state_dict[{name!r}] has wrong shape, dtype, or device")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"state_dict[{name!r}] must be finite")
    return state


def save_checkpoint(model: RecurrentActorCritic, path: str | Path) -> None:
    """Write model weights to a new file without overwriting existing output."""
    if not isinstance(model, RecurrentActorCritic):
        raise TypeError("model must be a RecurrentActorCritic")
    kind = model.architecture_kind
    state = _check_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, model.hidden_size, kind)
    architecture = {"hidden_size": model.hidden_size}
    if kind != "ordinal_v1":
        architecture["kind"] = kind
    payload = {"format_version": 1 if kind == "ordinal_v1" else 2, "architecture": architecture,
               "state_dict": {key: value.detach().clone() for key, value in state.items()}}
    # Serialize first so serialization failures do not create a partial file.
    data = io.BytesIO()
    torch.save(payload, data)
    from .run_files import write_bytes_exclusive
    write_bytes_exclusive(path, data.getvalue())


def load_checkpoint(path: str | Path, device="cpu") -> RecurrentActorCritic:
    """Load only weights, verify the exact format, and return a CPU model."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"format_version", "architecture", "state_dict"}:
        raise ValueError("invalid checkpoint structure")
    if type(payload["format_version"]) is not int or payload["format_version"] not in (1, 2):
        raise ValueError("unsupported checkpoint format_version")
    architecture = payload["architecture"]
    version = payload["format_version"]
    expected_keys = {"hidden_size"} if version == 1 else {"hidden_size", "kind"}
    if not isinstance(architecture, dict) or set(architecture) != expected_keys:
        raise ValueError("invalid checkpoint architecture")
    hidden_size = positive_integer(architecture["hidden_size"], "hidden_size")
    kind = "ordinal_v1" if version == 1 else architecture["kind"]
    if version == 2 and kind not in ("action_set_v1", "action_set_ff_v1"):
        raise ValueError("unsupported checkpoint architecture kind")
    state = _check_state_dict(payload["state_dict"], hidden_size, kind)
    if version == 1:
        model = RecurrentActorCritic(hidden_size=hidden_size, seed=0)
    else:
        from .action_set import ActionSetActorCritic
        model = ActionSetActorCritic(hidden_size=hidden_size, seed=0, memoryless=(kind == "action_set_ff_v1"))
    model.load_state_dict(state, strict=True)
    return model.to(device)
