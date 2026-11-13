"""Finite-trial GAE and clipped PPO loss for recurrent rollout replay."""

from __future__ import annotations

import torch

from .common import finite_reward


def _sequence(value: object, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device.type != "cpu" or value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"{name} must contain CPU real numbers")
        result = value.detach().to(dtype=torch.float32).clone()
    else:
        try:
            copied = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a sequence of real numbers") from exc
        result = torch.tensor([finite_reward(v) for v in copied], dtype=torch.float32)
    if result.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(result).all().item():
        raise ValueError(f"{name} must be finite and representable as float32")
    return result


def compute_gae(
    rewards: object, values: object, *, gamma: float = .99, gae_lambda: float = .95,
    bootstrap_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unmasked generalized advantage estimates over a finite trial.

    Internal episode boundaries have no effect. The caller chooses bootstrap
    semantics, including zero at the finite trial's final boundary.
    """
    rewards = _sequence(rewards, "rewards")
    values = _sequence(values, "values")
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same length")
    gamma = finite_reward(gamma)
    gae_lambda = finite_reward(gae_lambda)
    bootstrap_value = finite_reward(bootstrap_value)
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    bootstrap = torch.tensor(bootstrap_value, dtype=torch.float32)
    if not torch.isfinite(bootstrap).item():
        raise ValueError("bootstrap_value must be representable as float32")
    advantages = torch.empty_like(rewards)
    advantage = torch.zeros((), dtype=torch.float32)
    next_value = bootstrap
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        advantage = delta + gamma * gae_lambda * advantage
        advantages[index] = advantage
        next_value = values[index]
    returns = advantages + values
    if not torch.isfinite(advantages).all().item() or not torch.isfinite(returns).all().item():
        raise ValueError("GAE result must be finite")
    return advantages, returns


def _loss_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional tensor")
    if value.device.type != "cpu" or not value.is_floating_point():
        raise TypeError(f"{name} must be a CPU floating-point tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    return value


def ppo_loss(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, new_values: torch.Tensor,
    returns: torch.Tensor, advantages: torch.Tensor, entropies: torch.Tensor, *,
    clip_ratio: float = .2, value_coefficient: float = .5, entropy_coefficient: float = .01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculate clipped policy loss, half-MSE value loss, and entropy bonus."""
    names = ("new_log_probs", "old_log_probs", "new_values", "returns", "advantages", "entropies")
    inputs = tuple(_loss_tensor(value, name) for value, name in zip(
        (new_log_probs, old_log_probs, new_values, returns, advantages, entropies), names
    ))
    if any(item.shape != inputs[0].shape for item in inputs[1:]):
        raise ValueError("PPO input tensors must have the same shape")
    new_log_probs, old_log_probs, new_values, returns, advantages, entropies = inputs
    clip_ratio = finite_reward(clip_ratio)
    value_coefficient = finite_reward(value_coefficient)
    entropy_coefficient = finite_reward(entropy_coefficient)
    if not 0 <= clip_ratio <= 1 or value_coefficient < 0 or entropy_coefficient < 0:
        raise ValueError("clip_ratio must be in [0,1] and loss coefficients nonnegative")
    # Behavior-policy values and GAE targets are constants, not gradient paths.
    old_log_probs = old_log_probs.detach()
    returns = returns.detach()
    advantages = advantages.detach()
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all().item():
        raise ValueError("PPO probability ratio must be finite")
    policy_loss = -torch.minimum(
        ratio * advantages, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    ).mean()
    value_loss = .5 * (new_values - returns).square().mean()
    entropy = entropies.mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    approx_kl = ((ratio - 1) - log_ratio).mean()
    clip_fraction = ((ratio - 1).abs() > clip_ratio).float().mean()
    metrics_tensors = {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                       "entropy": entropy, "approx_kl": approx_kl, "clip_fraction": clip_fraction}
    if not all(torch.isfinite(value).item() for value in metrics_tensors.values()):
        raise ValueError("PPO loss and metrics must be finite")
    return loss, {name: float(value.detach().item()) for name, value in metrics_tensors.items()}
