"""Batched teacher inference/replay; each CPU game and learner remains isolated."""

import torch


def collect_batch(model, specifications, *, steps, assessment_steps=None):
    from .meta_exploration import NeuralExplorer, ExplorationRollout
    from .meta_exploration_rollout import meta_trial_stream
    streams = [meta_trial_stream(rule, seed=seed, steps=steps, opponent_profile=profile,assessment_steps=assessment_steps)
               for rule, seed, profile in specifications]
    pending = [next(stream) for stream in streams]
    ns = {item[1] for item in pending}
    if len(ns) != 1:
        raise ValueError('a teacher batch must have one action count')
    n = ns.pop()
    teachers = [NeuralExplorer(model=model, n=n, seed=item[2]) for item in pending]
    trials = [None] * len(streams)
    for step in range(steps):
        observations = [item[0] for item in pending]
        hidden = torch.stack([teacher.hidden for teacher in teachers])
        with torch.no_grad():
            logits, values, next_hidden = model.forward_observations(observations, hidden, n=n)
            distributions = torch.log_softmax(logits, dim=-1)
            sampled = torch.cat([torch.multinomial(distributions[i].exp(), 1,
                                 generator=teacher.generator) for i, teacher in enumerate(teachers)])
            # Transfer once per batch instead of synchronizing for every trial's
            # action, log-probability and value. Local sampling RNGs stay intact.
            actions = sampled.cpu().tolist()
            behavior = torch.stack((distributions.gather(1, sampled[:, None]).squeeze(1),
                                    values), dim=1).cpu().tolist()
        for i, (teacher, action, stream) in enumerate(zip(teachers, actions, streams)):
            teacher.events.append(observations[i]); teacher.actions.append(action)
            teacher.old_log_probs.append(behavior[i][0])
            teacher.old_values.append(behavior[i][1]); teacher.hidden = next_hidden[i]
            try:
                pending[i] = stream.send(action)
            except StopIteration as result:
                if step != steps - 1:
                    raise RuntimeError('trial ended before its exploration budget')
                trials[i] = result.value
    if any(trial is None for trial in trials):
        raise RuntimeError('trial exceeded exploration budget')
    return [ExplorationRollout(tuple(t.events), tuple(t.actions), tuple(t.old_log_probs),
                              tuple(t.old_values), trial['meta_reward'], trial)
            for t, trial in zip(teachers, trials)]


def replay_batch(model, rollouts):
    """Keep trial-major ordering used by targets and behavior log probabilities."""
    n = rollouts[0].trial['n']
    length = len(rollouts[0].actions)
    if any(len(r.actions) != length or r.trial['n'] != n for r in rollouts):
        raise ValueError('replay requires equal horizons/action counts')
    hidden = torch.stack([model.initial_hidden(n=n) for _ in rollouts])
    logs, values, entropies = [], [], []
    for step in range(length):
        logits, value, hidden = model.forward_observations([r.events[step] for r in rollouts], hidden, n=n)
        distribution = torch.log_softmax(logits, dim=-1)
        actions = torch.tensor([r.actions[step] for r in rollouts], device=model.device)
        logs.append(distribution.gather(1, actions[:, None]).squeeze(1))
        values.append(value)
        entropies.append(-(distribution.exp() * distribution).sum(-1))
    return tuple(torch.stack(items, dim=1).reshape(-1) for items in (logs, values, entropies))
