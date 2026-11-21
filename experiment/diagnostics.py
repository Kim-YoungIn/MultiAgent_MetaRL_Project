"""D-031 history and opponent-state diagnostic comparisons."""
import hashlib
import json
import statistics
from collections import defaultdict

from mamrl.history_study import collect_history_context, replay_history, score_history
from mamrl.meta_exploration import NeuralExplorer, RandomExplorer, decode_rule, neural_factory
from mamrl.meta_exploration_rollout import _make_env, assess_pair


class ResetExplorer(NeuralExplorer):
    def act(self, observation):
        self.hidden = self.model.initial_hidden(n=self.n)
        return super().act(observation)


def diagnose(entries, models, *, dev=False):
    counts = defaultdict(int)
    steps, horizon, reps = (3, 6, 1) if dev else (12, 24, 2)
    factories = {'M': neural_factory(models['M']), 'P': neural_factory(models['P']), 'random': RandomExplorer}
    for entry in entries:
        family = entry['family']
        if entry['split'] != 'fresh_rules' or counts[family] >= (1 if dev else 4):
            continue
        counts[family] += 1
        rule = decode_rule(entry['rule'])
        for profile in ('steady', 'sarsa'):
            for rep in range(reps):
                raw = json.dumps([rule.n, 'fresh_rules', entry['id'], profile, rep, 61001], separators=(',', ':')).encode()
                seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')
                collected = {name: collect_history_context(rule, factory, seed=seed, steps=steps,
                             assessment_steps=horizon, opponent_profile=profile) for name, factory in factories.items()}
                base = {'rule_id': entry['id'], 'family': family, 'profile': profile, 'rep': rep, 'seed': seed}
                for name, (trial, capture) in collected.items():
                    student = replay_history(trial['exploration'], n=rule.n, seed=capture['seeds']['student'])
                    if student.snapshot() != capture['student'].snapshot():
                        raise RuntimeError('student replay mismatch')
                    yield {**base, 'kind': 'history', 'history': name, 'exploration': trial['exploration']}
                    for reference, (_, ref) in collected.items():
                        for mode in ('online', 'frozen'):
                            env, start = _make_env(capture['tensor'], horizon, ref['seeds']['assessment_reset'])
                            events = assess_pair(env, start, student.clone(), ref['opponent'].clone(), horizon,
                                                 learn_student=(mode == 'online'))
                            if name == reference and mode == 'online' and events != trial['assessment']:
                                raise RuntimeError('natural diagonal mismatch')
                            yield {**base, 'kind': 'cross', 'history': name, 'reference': reference,
                                   'mode': mode, 'assessment_mean': statistics.mean(e['rewards'][0] for e in events),
                                   'assessment_curve': [e['rewards'][0] for e in events]}
                    for representation in ('full', 'binned'):
                        result = score_history(trial, capture, capture, representation)
                        yield {**base, 'kind': 'representation', 'history': name,
                               'representation': representation, **result}
                for name in ('M', 'P'):
                    model = models[name]
                    def reset_factory(*, n, seed, student):
                        return ResetExplorer(model=model, n=n, seed=seed)
                    trial, _ = collect_history_context(rule, reset_factory, seed=seed, steps=steps,
                                assessment_steps=horizon, opponent_profile=profile)
                    yield {**base, 'kind': 'hidden_reset', 'history': name,
                           'assessment_mean': trial['assessment_mean'], 'exploration': trial['exploration']}
