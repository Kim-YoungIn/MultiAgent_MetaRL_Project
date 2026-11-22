"""Paired reset-state and continuous-budget Teacher–Student rollouts."""
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch

from research.fast_exploration_v1 import batch, study
from research.fast_exploration_v1.simulator import StudentBatch, opponent_distribution, opponent_update

TRAINED = (
    'pooled_full', 'pooled_masked', 'ips_full', 'ips_masked', 'pooled_immediate',
    'pooled_fixed_value', 'pooled_fixed_total',
)
METHODS = TRAINED + (
    'pooled_random', 'pooled_greedy', 'pooled_ucb', 'ips_random', 'ips_greedy', 'ips_ucb',
)
INTERVENTION_METHODS = ('pooled_full', 'pooled_masked', 'pooled_immediate')


@dataclass
class Tape:
    rewards: np.ndarray
    actions: np.ndarray

    @classmethod
    def empty(cls, size, horizon):
        shape = (size, horizon, 2)
        return cls(np.empty(shape), np.empty(shape, dtype=np.int8))

    def step(self, env, q, action, uniforms, turn, *, update_q, next_reset=None, terminal=False):
        previous = env.last
        other = batch.opponent_actions(q, previous, uniforms[:, 0])
        rewards = env.step(action, other, uniforms[:, 1])
        if update_q:
            following = env.last if next_reset is None else next_reset
            opponent_update(q, previous, following, other, rewards[:, 1], terminal=terminal)
        self.actions[:, turn, 0] = action
        self.actions[:, turn, 1] = other
        self.rewards[:, turn] = rewards
        return rewards[:, 0]


@dataclass
class Probes:
    tape: Tape
    student: StudentBatch
    q: np.ndarray
    env: object
    actor_inputs: np.ndarray | None
    chosen_features: np.ndarray | None


def policy_uniforms(episodes, k):
    """Common exogenous sampling uniforms, independent across episode streams."""
    return np.stack([np.random.default_rng(s).random(k) for s in episodes.teacher_seed], axis=1)


def inputs(env, student, x, masked):
    z = batch.actor_inputs(env, student, x, belief=not masked)
    if masked:
        z[..., 1:3] = 0
        if env.elapsed == 0:
            z[..., 0] = 0
    return z


@torch.no_grad()
def probe(episodes, initial_q, prior, method, actor, draws, k, h, *, reset=None,
          opponent_learns=True, beta=.25, record_actor=False, record_features=False):
    size = len(episodes.rules)
    horizon = max(1, k) if reset is not None else h
    env = episodes.episode(episodes.explore_seed, horizon)
    student = StudentBatch(size, prior)
    q = initial_q.copy()
    tape = Tape.empty(size, k)
    actor_inputs = np.empty((size, k, 3, 7), dtype=np.float32) if record_actor else None
    chosen_features = np.empty((size, k, 20)) if record_features else None
    if actor is not None and k:
        hidden = actor.initial_hidden(n=3, batch=size)
    for t in range(k):
        x = env.features()
        if actor is not None:
            z = inputs(env, student, x, method.endswith('_masked'))
            logits, _, hidden = actor.forward_features(torch.from_numpy(z), hidden)
            cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1).numpy()
            action = np.minimum(2, (cdf <= draws[t, :, None]).sum(axis=1))
            if record_actor:
                actor_inputs[:, t] = z
        elif method.endswith('_random'):
            action = episodes.random_actions[t]
        else:
            scores = student.predict(x)
            if method.endswith('_ucb'):
                scores += beta * student.uncertainty(x)
            action = scores.argmax(axis=1)
        if record_features:
            chosen_features[:, t] = x[np.arange(size), action]
        # In the continuous condition absolute turn t always uses the same stream.
        uniforms = episodes.explore_uniform[t] if reset is not None else episodes.assess_uniform[t]
        reward = tape.step(env, q, action, uniforms, t, update_q=opponent_learns,
                           next_reset=reset if t == k - 1 else None)
        student.observe(x, action, reward)
    return Probes(tape, student, q, env, actor_inputs, chosen_features)


def assess(env, uniforms, student, q, *, learn):
    tape = Tape.empty(env.count, len(uniforms))
    for t, draw in enumerate(uniforms):
        x = env.features()
        action = student.predict(x).argmax(axis=1)
        reward = tape.step(env, q, action, draw, t, update_q=learn, terminal=t == len(uniforms) - 1)
        if learn:
            student.observe(x, action, reward)
    return tape


def reset_assessment(episodes, h):
    return episodes.episode(episodes.assess_seed, h)


def train_update(actors, optimizers, priors, episodes, initial_q, k, assessment_h, budget_h):
    draws = policy_uniforms(episodes, k)
    means, actions = [], []
    for method in TRAINED:
        continuous = '_fixed_' in method
        prior = priors[method.split('_', 1)[0]]
        init = None if continuous else reset_assessment(episodes, assessment_h)
        horizon = budget_h if continuous else assessment_h
        with torch.no_grad():
            samples = probe(episodes, initial_q, prior, method, actors[method], draws, k, horizon,
                            reset=None if continuous else init.last, record_actor=True)
            ass = assess(samples.env if continuous else init,
                         episodes.assess_uniform[k:budget_h] if continuous else episodes.assess_uniform[:assessment_h],
                         samples.student, samples.q, learn=True)
        if method == 'pooled_immediate':
            returns = samples.tape.rewards[..., 0].mean(axis=1)
        elif continuous:
            returns = ass.rewards[..., 0].sum(axis=1) / budget_h
            if method == 'pooled_fixed_total':
                returns += samples.tape.rewards[..., 0].sum(axis=1) / budget_h
        else:
            returns = ass.rewards[..., 0].mean(axis=1)
        actor, optimizer = actors[method], optimizers[method]
        loss = study.replay_loss(actor, samples.actor_inputs, samples.tape.actions[..., 0], returns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        means.append(float(returns.mean()))
        actions.append(np.concatenate((samples.tape.actions, ass.actions), axis=1))
    return means, actions


def common_queries(episodes, q):
    """Collect reachable assessment contexts independent of any tested Teacher."""
    context = reset_assessment(episodes, 16)
    draws = np.stack([np.random.default_rng(int(seed) ^ 0xD352AD).random((16, 3))
                      for seed in episodes.assess_seed], axis=1)
    features, values = [], []
    size = context.count
    for t, (u, g, a) in enumerate(draws.transpose(0, 2, 1)):
        last = context.last
        if t % 4 == 0:
            probs = opponent_distribution(q, last)
            expected = np.zeros((size, 3))
            for own in range(3):
                for other in range(3):
                    # Simulator stepping replaces state/history arrays on the copy.
                    raw = copy(context).step(np.full(size, own), np.full(size, other), g)
                    expected[:, own] += raw[:, 0] * probs[:, other]
            features.append(context.features())
            values.append(expected)
        context.step((3 * a).astype(int), batch.opponent_actions(q, last, u), g)
    return features, values


def query_quality(student, queries):
    features, values = queries
    size = np.arange(student.mean.shape[0])
    return np.mean([v[size, student.predict(x).argmax(axis=1)]
                    for x, v in zip(features, values, strict=True)], axis=0)


def allocate(raw, name, shape, turns):
    raw[f'{name}_rewards'] = np.full((*shape, turns, 2), np.nan)
    raw[f'{name}_actions'] = np.full((*shape, turns, 2), -1, dtype=np.int8)


def store(raw, name, index, tape):
    n = tape.actions.shape[1]
    selection = (*index, slice(None), slice(0, n))
    raw[f'{name}_rewards'][selection] = tape.rewards
    raw[f'{name}_actions'][selection] = tape.actions


@torch.no_grad()
def evaluate(actors, priors, episodes, cfg):
    h, budget_h, ks = cfg['h'], cfg['budget_h'], cfg['ks']
    initial_q = batch.warm_up(episodes)
    draws = policy_uniforms(episodes, max(ks))
    queries = common_queries(episodes, initial_q)
    shape = (len(METHODS), len(ks), len(episodes.rules))
    raw = dict(methods=np.array(METHODS), ks=np.array(ks), query_values=np.empty(shape),
               action_to_role=np.array([r.roles for r in episodes.rules], dtype=np.int8).argsort(axis=1),
               intervention_methods=np.array(INTERVENTION_METHODS))
    for phase in ('controlled', 'natural', 'frozen'):
        allocate(raw, phase, shape, h)
    for phase in ('probe', 'frozen_probe'):
        allocate(raw, phase, shape, max(ks))
    allocate(raw, 'fixed', shape, budget_h)
    # Only the paired principal three pooled policies enter the posterior × Q diagnostic.
    for phase in ('retain_both', 'reset_student'):
        allocate(raw, phase, (len(INTERVENTION_METHODS), *shape[1:]), h)
    raw['probe_features'] = np.zeros((*shape, max(ks), 20))

    for mi, method in enumerate(METHODS):
        prior = priors[method.split('_', 1)[0]]
        teacher = actors.get(method)
        blank = StudentBatch(shape[-1], prior)
        for ki, k in enumerate(ks):
            index = (mi, ki)
            settings = dict(beta=cfg['ucb_beta'])
            uniform = episodes.assess_uniform[:h]
            init = reset_assessment(episodes, h)
            assessment_start = init.last
            samples = probe(episodes, initial_q, prior, method, teacher, draws, k, h,
                            reset=assessment_start, record_features=True, **settings)
            store(raw, 'probe', index, samples.tape)
            raw['probe_features'][mi, ki, :, :k] = samples.chosen_features
            raw['query_values'][index] = query_quality(samples.student, queries)
            store(raw, 'controlled', index, assess(reset_assessment(episodes, h), uniform,
                                                   samples.student, initial_q, learn=False))
            if method in INTERVENTION_METHODS:
                ii = (INTERVENTION_METHODS.index(method), ki)
                store(raw, 'retain_both', ii, assess(reset_assessment(episodes, h), uniform,
                                                    samples.student, samples.q, learn=False))
                store(raw, 'reset_student', ii, assess(reset_assessment(episodes, h), uniform,
                                                       blank, samples.q, learn=False))
            store(raw, 'natural', index, assess(init, uniform, samples.student, samples.q, learn=True))

            fixed_opp = probe(episodes, initial_q, prior, method, teacher, draws, k, h, reset=assessment_start,
                              opponent_learns=False, **settings)
            store(raw, 'frozen_probe', index, fixed_opp.tape)
            store(raw, 'frozen', index, assess(reset_assessment(episodes, h), uniform,
                                               fixed_opp.student, initial_q, learn=False))
            full = probe(episodes, initial_q, prior, method, teacher, draws, k, budget_h, **settings)
            tail = assess(full.env, episodes.assess_uniform[k:budget_h], full.student, full.q, learn=True)
            store(raw, 'fixed', index,
                  Tape(np.concatenate((full.tape.rewards, tail.rewards), axis=1),
                       np.concatenate((full.tape.actions, tail.actions), axis=1)))
    return raw
