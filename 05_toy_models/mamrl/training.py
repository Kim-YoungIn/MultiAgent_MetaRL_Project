"""Full-trial recurrent PPO against a frozen random opponent.

Episode time limits are internal reset boundaries; only the finite trial ends
return/GAE accumulation. Observation-only boundary events have no discount step.
"""

from dataclasses import dataclass
import math
import random
import statistics
from typing import Any

import torch

from .common import finite_reward, positive_integer, validate_action, validate_n, validate_seed
from .environment import MemoryOneEnv, Observation
from .policy import RandomPolicy
from .opponents import make_opponent
from .ppo import compute_gae, ppo_loss
from .recurrent import RecurrentActorCritic
from .rollout import EpisodeResult, Transition
from .rules import RuleSpec, RuleTerm, compile_rule, generate_rule
from .tensor import RewardTensor


@dataclass(frozen=True)
class TrainConfig:
    n: int = 3
    family: str = "action_bonus"
    seed: int = 7
    fixed_action: int | None = None
    architecture: str = "ordinal_v1"
    opponent_mode: str = "random"
    hidden_size: int = 32
    updates: int = 40
    trials_per_update: int = 8
    episodes_per_trial: int = 2
    max_steps: int = 12
    epochs: int = 3
    learning_rate: float = .003
    gamma: float = .99
    gae_lambda: float = .95
    clip_ratio: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    max_grad_norm: float = .5
    evaluation_trials: int = 16
    evaluation_interval: int = 10

    def __post_init__(self):
        validate_n(self.n)
        if self.seed is None:
            raise TypeError("training requires an explicit nonnegative seed")
        validate_seed(self.seed)
        if self.family not in ("action_bonus", "zero_sum", "coordination", "memory_one", "mixed",
                               "probe_memory", "opponent_response"):
            raise ValueError("unknown training task family")
        if self.family == "probe_memory" and self.n < 3:
            raise ValueError("probe_memory requires n >= 3")
        if self.architecture not in ("ordinal_v1", "action_set_v1", "action_set_ff_v1"):
            raise ValueError("unknown architecture")
        if self.opponent_mode not in ("random", "copy_previous", "avoid_previous", "mixed"):
            raise ValueError("unknown opponent_mode")
        if self.fixed_action is not None:
            validate_action(self.fixed_action, self.n)
            if self.family != "action_bonus":
                raise ValueError("fixed_action is only valid for action_bonus")
        for name in ("hidden_size", "updates", "trials_per_update", "episodes_per_trial",
                     "max_steps", "epochs", "evaluation_trials", "evaluation_interval"):
            positive_integer(getattr(self, name), name)
        for name in ("gamma", "gae_lambda"):
            if not 0 <= finite_reward(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("learning_rate", "max_grad_norm"):
            if finite_reward(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < finite_reward(self.clip_ratio) < 1:
            raise ValueError("clip_ratio must be in (0,1)")
        for name in ("value_coefficient", "entropy_coefficient"):
            if finite_reward(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class TrialRollout:
    spec: RuleSpec
    events: tuple[Observation, ...]
    action_event_indices: tuple[int, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    old_log_probs: tuple[float, ...]
    old_values: tuple[float, ...]
    episodes: tuple[EpisodeResult, ...]
    environment_seeds: tuple[int, ...]
    opponent_seeds: tuple[int, ...]
    learner_seeds: tuple[int, ...]
    reset_event_indices: tuple[int, ...] = (0,)
    opponent_kind: str = "random"


def sample_task(n: int, family: str, seed: int, *, fixed_action: int | None = None) -> RuleSpec:
    validate_n(n)
    validate_seed(seed)
    if fixed_action is not None:
        validate_action(fixed_action, n)
        if family != "action_bonus":
            raise ValueError("fixed_action is only valid for action_bonus")
    if family == "action_bonus":
        target = random.Random(seed).randrange(n) if fixed_action is None else fixed_action
        return RuleSpec(n, (RuleTerm("action_bonus", 1, target),))
    if family == "probe_memory":
        if n < 3:
            raise ValueError("probe_memory requires n >= 3")
        kind = ("probe_repeat", "probe_copy")[random.Random(seed).randrange(2)]
        return RuleSpec(n, (RuleTerm(kind, 1),))
    if family == "opponent_response":
        return RuleSpec(n, (RuleTerm("coordination", 1),))
    return generate_rule(n, seed, family)


def evaluation_specs(config: TrainConfig) -> tuple[RuleSpec, ...]:
    rng = random.Random(config.seed + 1_000_000)
    return tuple(sample_task(config.n, config.family, rng.getrandbits(63), fixed_action=config.fixed_action)
                 for _ in range(config.evaluation_trials))


def evaluation_opponents(config: TrainConfig):
    if config.opponent_mode == "random":
        return None
    if config.opponent_mode == "mixed":
        return tuple(("copy_previous", "avoid_previous")[i % 2] for i in range(config.evaluation_trials))
    return (config.opponent_mode,) * config.evaluation_trials


def collect_trial(model, spec: RuleSpec, *, episodes_per_trial: int,
                  max_steps: int, seed: int, reset_hidden: bool = False,
                  opponent_kind: str = "random") -> TrialRollout:
    """Collect one full finite trial with separate exogenous and action RNGs."""
    positive_integer(episodes_per_trial, "episodes_per_trial")
    positive_integer(max_steps, "max_steps")
    if seed is None:
        raise TypeError("trial collection requires an explicit seed")
    validate_seed(seed)
    if type(reset_hidden) is not bool:
        raise TypeError("reset_hidden must be bool")
    tensor = spec if isinstance(spec, RewardTensor) else compile_rule(spec)
    seeds = random.Random(seed)
    learner_seed = seeds.getrandbits(63)
    generator = torch.Generator(device="cpu").manual_seed(learner_seed)
    env_seeds = tuple(seeds.getrandbits(63) for _ in range(episodes_per_trial))
    opp_seeds = tuple(seeds.getrandbits(63) for _ in range(episodes_per_trial))
    hidden = model.initial_hidden(n=spec.n) if model is not None else None
    events, action_indices, actions, rewards, old_logs, old_values = [], [], [], [], [], []
    episode_records, reset_indices = [], [0]
    opponent = make_opponent(opponent_kind)
    with torch.no_grad():
        for episode_index in range(episodes_per_trial):
            env = MemoryOneEnv(tensor, max_steps=max_steps)
            observations = env.reset(seed=env_seeds[episode_index])
            opponent_state = opponent.initial_state(n=spec.n, seed=opp_seeds[episode_index])
            if episode_index and reset_hidden:
                hidden = model.initial_hidden(n=spec.n) if model is not None else None
                reset_indices.append(len(events))
            transitions, totals = [], [0.0, 0.0]
            for _ in range(max_steps):
                action_indices.append(len(events))
                events.append(observations[0])
                if model is None:
                    logits, value = torch.zeros(spec.n), torch.tensor(0.)
                else:
                    logits, value, hidden = model.forward_observation(observations[0], hidden, n=spec.n)
                log_probs = torch.log_softmax(logits, dim=-1)
                action = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
                # Opponent sees only the same pre-step observation, never this action.
                other = opponent.act(observations[1], opponent_state, n=spec.n)
                opponent_state = other.state
                result = env.step(action, other.action)
                actions.append(action)
                rewards.append(result.rewards[0])
                old_logs.append(float(log_probs[action]))
                old_values.append(float(value))
                transitions.append(Transition(observations, (action, other.action), result.rewards,
                                              result.observations, result.terminated, result.truncated))
                for p in (0, 1):
                    totals[p] += result.rewards[p]
                observations = result.observations
            episode_records.append(EpisodeResult(tuple(transitions), tuple(totals)))
            if episode_index < episodes_per_trial - 1:
                # Preserve the last reward without inventing a new environment step.
                events.append(observations[0])
                if model is not None:
                    _, _, hidden = model.forward_observation(observations[0], hidden, n=spec.n)
    return TrialRollout(spec, tuple(events), tuple(action_indices), tuple(actions), tuple(rewards),
                        tuple(old_logs), tuple(old_values), tuple(episode_records), env_seeds, opp_seeds,
                        (learner_seed,) * episodes_per_trial, tuple(reset_indices), opponent_kind)


def replay_trial(model, trial: TrialRollout):
    """Recompute likelihoods from zero hidden, including all boundary events."""
    if not trial.events or not trial.actions:
        raise ValueError("trial must contain observations and actions")
    if len(trial.action_event_indices) != len(trial.actions):
        raise ValueError("trial action index count must match actions")
    if tuple(sorted(set(trial.action_event_indices))) != trial.action_event_indices:
        raise ValueError("action event indices must be strictly increasing")
    if trial.action_event_indices[0] < 0 or trial.action_event_indices[-1] >= len(trial.events):
        raise ValueError("action event index out of range")
    hidden = model.initial_hidden(n=trial.spec.n)
    log_probs, values, entropies = [], [], []
    action_position = 0
    reset_indices = set(trial.reset_event_indices)
    for index, observation in enumerate(trial.events):
        if index in reset_indices:
            hidden = model.initial_hidden(n=trial.spec.n)
        logits, value, hidden = model.forward_observation(observation, hidden, n=trial.spec.n)
        if action_position < len(trial.actions) and index == trial.action_event_indices[action_position]:
            distribution = torch.log_softmax(logits, dim=-1)
            log_probs.append(distribution[trial.actions[action_position]])
            values.append(value)
            entropies.append(-(distribution.exp() * distribution).sum())
            action_position += 1
    return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


def evaluate(model, specs, *, episodes_per_trial: int, max_steps: int,
             seed: int, reset_hidden: bool = False, opponent_kinds=None) -> dict[str, Any]:
    specs = tuple(specs)
    if not specs:
        raise ValueError("evaluation requires at least one task")
    validate_seed(seed)
    if seed is None:
        raise TypeError("evaluation requires an explicit seed")
    rng = random.Random(seed)
    kinds = tuple(opponent_kinds) if opponent_kinds is not None else ("random",) * len(specs)
    if len(kinds) != len(specs):
        raise ValueError("opponent_kinds must align with specs")
    trials = [collect_trial(model, spec, episodes_per_trial=episodes_per_trial,
                            max_steps=max_steps, seed=rng.getrandbits(63), reset_hidden=reset_hidden,
                            opponent_kind=kind) for spec, kind in zip(specs, kinds)]
    returns = [sum(t.rewards) for t in trials]
    step_count = episodes_per_trial * max_steps
    means = [r / step_count for r in returns]
    return {
        "trials": len(trials),
        "mean_reward_per_step": statistics.mean(means),
        "mean_reward_by_episode": [statistics.mean(t.episodes[e].total_rewards[0] / max_steps for t in trials)
                                   for e in range(episodes_per_trial)],
        "mean_reward_by_turn": [statistics.mean(t.rewards[k] for t in trials) for k in range(step_count)],
        "trial_returns": returns,
        "standard_error": statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0,
    }


@dataclass(frozen=True)
class TrainingResult:
    initial_model: RecurrentActorCritic
    model: RecurrentActorCritic
    history: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    config: TrainConfig


def train(config: TrainConfig, *, checkpoint_callback=None, model_factory=None,
          task_sampler=None, evaluation_tasks=None) -> TrainingResult:
    """Train a recurrent baseline, retaining temporal order in every PPO epoch."""
    from copy import deepcopy

    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable")
    if model_factory is not None and not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if task_sampler is not None and not callable(task_sampler):
        raise TypeError("task_sampler must be callable")

    def checked_task(task):
        if not isinstance(task, (tuple, list)) or len(task) != 2:
            raise TypeError("custom task must be a (rule or tensor, opponent kind) pair")
        rule, kind = task
        if getattr(rule, 'n', None) != config.n:
            raise ValueError("custom task action count must match TrainConfig.n")
        make_opponent(kind)
        return rule, kind
    if model_factory is not None:
        factory = model_factory
    elif config.architecture == "ordinal_v1":
        factory = RecurrentActorCritic
    else:
        from .action_set import ActionSetActorCritic
        def factory(*, hidden_size, seed):
            return ActionSetActorCritic(hidden_size, seed, memoryless=config.architecture == "action_set_ff_v1")
    model = factory(hidden_size=config.hidden_size, seed=config.seed)
    if not isinstance(model, RecurrentActorCritic):
        raise TypeError("model_factory must return a RecurrentActorCritic-compatible model")
    initial_model = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = random.Random(config.seed)
    custom_evaluation = None if evaluation_tasks is None else tuple(checked_task(t) for t in evaluation_tasks)
    if custom_evaluation == ():
        raise ValueError("evaluation_tasks must be nonempty")
    specs = evaluation_specs(config) if custom_evaluation is None else tuple(t[0] for t in custom_evaluation)
    eval_options = dict(episodes_per_trial=config.episodes_per_trial, max_steps=config.max_steps,
                        seed=config.seed + 2_000_000)
    if custom_evaluation is not None:
        eval_options['opponent_kinds'] = tuple(t[1] for t in custom_evaluation)
    elif config.opponent_mode != "random":
        eval_options['opponent_kinds'] = evaluation_opponents(config)
    evaluations = [{"update": 0, **evaluate(model, specs, **eval_options)}]
    history = []
    if checkpoint_callback is not None:
        checkpoint_callback(0, model)
    for update in range(1, config.updates+1):
        batch = []
        for _ in range(config.trials_per_update):
            task_seed = rng.getrandbits(63)
            if task_sampler is not None:
                spec, kind = checked_task(task_sampler(task_seed))
            else:
                spec = sample_task(config.n, config.family, task_seed, fixed_action=config.fixed_action)
                kind = config.opponent_mode
                if kind == "mixed":
                    kind = ("copy_previous", "avoid_previous")[random.Random(task_seed + 314159).randrange(2)]
            batch.append(collect_trial(model, spec, episodes_per_trial=config.episodes_per_trial,
                                       max_steps=config.max_steps, seed=rng.getrandbits(63), opponent_kind=kind))
        advantages, returns = [], []
        for trial in batch:
            advantage, target = compute_gae(trial.rewards, trial.old_values, gamma=config.gamma,
                                            gae_lambda=config.gae_lambda, bootstrap_value=0.0)
            advantages.append(advantage)
            returns.append(target)
        advantage = torch.cat(advantages)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        target = torch.cat(returns)
        old_log_probs = torch.tensor([x for trial in batch for x in trial.old_log_probs])
        epoch_metrics = []
        for _ in range(config.epochs):
            # No saved hidden states, no shuffled individual transitions.
            replay = [replay_trial(model, trial) for trial in batch]
            log_probs = torch.cat([x[0] for x in replay])
            values = torch.cat([x[1] for x in replay])
            entropies = torch.cat([x[2] for x in replay])
            loss, metrics = ppo_loss(log_probs, old_log_probs, values, target, advantage, entropies,
                                     clip_ratio=config.clip_ratio, value_coefficient=config.value_coefficient,
                                     entropy_coefficient=config.entropy_coefficient)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                           error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("optimizer produced non-finite parameters")
            epoch_metrics.append({**metrics, "gradient_norm": float(gradient_norm)})
        metrics = {key: statistics.mean(m[key] for m in epoch_metrics) for key in epoch_metrics[0]}
        history.append({"update": update, **metrics,
                        "mean_reward_per_step": statistics.mean(sum(t.rewards)/len(t.rewards) for t in batch)})
        if update % config.evaluation_interval == 0 or update == config.updates:
            evaluations.append({"update": update, **evaluate(model, specs, **eval_options)})
        if checkpoint_callback is not None:
            checkpoint_callback(update, model)
    return TrainingResult(initial_model, model, history, evaluations, config)
