"""Episode execution and immutable transition records for future learners."""

from dataclasses import dataclass
from typing import Any, Sequence

from .common import validate_seed
from .environment import MemoryOneEnv, Observation
from .policy import Policy, PolicyOutput


@dataclass(frozen=True, slots=True)
class Transition:
    observations: tuple[Observation, Observation]
    actions: tuple[int, int]
    rewards: tuple[float, float]
    next_observations: tuple[Observation, Observation]
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    transitions: tuple[Transition, ...]
    total_rewards: tuple[float, float]


def run_episode(
    env: MemoryOneEnv,
    policies: Sequence[Policy[Any]],
    *,
    seed: int | None = None,
    policy_seeds: tuple[int | None, int | None] = (None, None),
) -> EpisodeResult:
    policies = tuple(policies)
    policy_seeds = tuple(policy_seeds)
    if len(policies) != 2 or len(policy_seeds) != 2:
        raise ValueError("exactly two policies and two policy seeds are required")
    validate_seed(seed)
    for policy_seed in policy_seeds:
        validate_seed(policy_seed)
    for policy in policies:
        if not callable(getattr(policy, "initial_state", None)) or not callable(
            getattr(policy, "act", None)
        ):
            raise TypeError("policies must implement initial_state and act")
    observations = env.reset(seed=seed)
    states = [
        policy.initial_state(n=env.n, seed=policy_seed)
        for policy, policy_seed in zip(policies, policy_seeds)
    ]
    transitions = []
    totals = [0.0, 0.0]
    while True:
        outputs = [
            policy.act(observations[i], states[i], n=env.n)
            for i, policy in enumerate(policies)
        ]
        if any(not isinstance(output, PolicyOutput) for output in outputs):
            raise TypeError("policy.act must return PolicyOutput")
        actions = (outputs[0].action, outputs[1].action)
        result = env.step(*actions)
        states = [output.state for output in outputs]
        transitions.append(
            Transition(
                observations, actions, result.rewards, result.observations,
                result.terminated, result.truncated,
            )
        )
        for i in (0, 1):
            totals[i] += result.rewards[i]
        observations = result.observations
        if result.terminated or result.truncated:
            return EpisodeResult(tuple(transitions), (totals[0], totals[1]))
