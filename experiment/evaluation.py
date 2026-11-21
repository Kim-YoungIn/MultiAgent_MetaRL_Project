"""Paired handoff evaluations with independent episode boundaries for every k."""
from hashlib import sha256
import json
import random
from statistics import mean

from mamrl.adaptive_agents import OnlineQ, ExpectedSARSA
from mamrl.meta_exploration import decode_rule, RandomExplorer, EpsilonExplorer, BalancedExplorer
from mamrl.meta_exploration_rollout import _make_env, assess_pair, OPPONENT_PROFILES, run_meta_trial
from mamrl.rules import compile_rule

PROFILES = ('steady', 'sarsa', 'sarsa_heldout')
BASELINES = {'random': RandomExplorer, 'epsilon': EpsilonExplorer, 'balanced': BalancedExplorer}
KS = (0, 3, 6, 12, 24)


def trial_seed(n, split, rule_id, profile, rep):
    key = json.dumps([n, split, rule_id, profile, rep, 61001], separators=(',', ':'))
    return int.from_bytes(sha256(key.encode()).digest()[:8], 'big')


def empty_trial(tensor, profile, seed, h):
    rng = random.Random(seed)
    _, student_seed, opponent_seed, _, reset_seed = (rng.getrandbits(63) for _ in range(5))
    student = OnlineQ(tensor.n, student_seed, alpha=.5, gamma=.95, epsilon=.2)
    config = dict(OPPONENT_PROFILES[profile])
    cls = ExpectedSARSA if config.pop('algorithm', None) else OnlineQ
    opponent = cls(tensor.n, opponent_seed, **config)
    env, start = _make_env(tensor, h, reset_seed)
    erased_env, erased_start = _make_env(tensor, h, reset_seed)
    erased = assess_pair(erased_env, erased_start, student.clone(), opponent.clone(), h)
    assessment = assess_pair(env, start, student, opponent, h)
    if assessment != erased:
        raise RuntimeError('empty history changed paired assessment')
    return {'assessment': assessment, 'exploration': [], 'meta_reward': 0.,
            'gameplay_mean':mean(e['rewards'][0] for e in assessment)}


def state_key(event):
    obs = event['observations'][0]
    return tuple(obs[name] for name in ('previous_own_action', 'previous_opponent_action',
                                        'previous_own_reward', 'episode_start_flag'))


def trial_row(tensor, factory, profile, seed, k, h):
    trial = (run_meta_trial(tensor, factory, seed=seed, steps=k, assessment_steps=h,
                            opponent_profile=profile) if k else empty_trial(tensor, profile, seed, h))
    exploration, assessment = trial['exploration'], trial['assessment']
    seen = set(map(state_key, exploration))
    curve = [e['rewards'][0] for e in assessment]
    return {'k': k, 'h': h, 'return': mean(curve), 'first': curve[0], 'early3': mean(curve[:3]),
            'G': trial['meta_reward'], 'game': trial['gameplay_mean'],
            'first_explore_action': exploration[0]['actions'][0] if k else None,
            'first_explore_reward': exploration[0]['rewards'][0] if k else None,
            'explore_mean': mean(e['rewards'][0] for e in exploration) if k else None,
            'q_revisit': 1-len(seen)/k if k else None,
            'q_reuse': mean(state_key(e) in seen for e in assessment),
            'opponent_nonuniform_fraction': mean(
                any(abs(p - 1/tensor.n) > 1e-12 for p in e['opponent_action_probabilities'])
                for e in exploration + assessment), 'steps_with_control': k+2*h}


def evaluate(entries, factory, *, reps, fixed_budget, ks=KS, h=24):
    for entry in entries:
        tensor = compile_rule(decode_rule(entry['rule']))
        for profile in PROFILES:
            for rep in range(reps):
                seed = trial_seed(tensor.n, entry['split'], entry['id'], profile, rep)
                record = {'rule': entry['id'], 'family': entry['family'], 'split': entry['split'],
                          'profile': profile, 'rep': rep, 'seed': seed}
                curve = [trial_row(tensor, factory, profile, seed, k, h) for k in ks]
                record['curve'] = curve
                record['auc'] = adaptation_auc(curve)
                if fixed_budget:
                    record['fixed_budget'] = [trial_row(tensor, factory, profile, seed, k, 2*h-k)
                                              if k != h else dict(curve[i], steps_with_control=0)
                                              for i,k in enumerate(ks)]
                yield record


def adaptation_auc(curve):
    if curve[0]['k'] != 0 or len(curve) < 2:
        raise ValueError('a paired curve must start at zero and contain positive experience')
    baseline = curve[0]['return']
    return sum((r['k']-l['k'])*((r['return']+l['return'])/2-baseline)
               for l,r in zip(curve,curve[1:],strict=False))/curve[-1]['k']
