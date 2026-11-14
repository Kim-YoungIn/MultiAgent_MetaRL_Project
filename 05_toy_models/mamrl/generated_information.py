"""Finite-belief information witness for two preselected generated rules.

This is a researcher-side, task-informed reference, never an agent observation.
It uses an equal prior over only the two stated rules and knows the horizon and
opponent. It is not a learned policy, a general oracle, or an H1 result.
"""
from __future__ import annotations

from functools import lru_cache

from .generated_tasks import validate_manifest
from .logical_rules import rule_from_dict
from .rules import compile_rule


_DEFAULT_SEED = 20261114
_RULE_IDS = ('r000', 'r005')
_START = (1, 0)
_HORIZON = 4
# The generated payoff entries are floating-point values. This precision is
# fixed for reproducible inference and prevents machine-level roundoff from
# being mistaken for distinguishable feedback. The pair was found analytically
# in the generated catalog, then frozen before neural results; it is not a
# preregistered representative sample of the generator distribution.
_REWARD_DECIMALS = 9


def information_witness(manifest):
    """Compute the fixed conditional two-rule, copy-opponent information test.

    A uniform prior is placed on the two rule IDs in the default n=3 catalog.
    The previous joint action (1,0) is reachable with probability 1/9 at an
    independently uniform episode reset. The opponent copies learner previous
    action. Both the state and reward after each simultaneous step are visible.
    The control erases only the *first* reward as information; it retains the
    reward in return and allows all later reward observations. Expected returns
    are conditional on the stated initial state and measured over four steps.
    """
    validate_manifest(manifest)
    if manifest['n'] != 3 or manifest['seed'] != _DEFAULT_SEED:
        raise ValueError('information witness requires the default n=3 manifest')
    entries = {entry['id']: entry for entry in manifest['rules']}
    if any(rule_id not in entries for rule_id in _RULE_IDS):
        raise ValueError('preselected information-witness rules are missing')
    tensors = tuple(compile_rule(rule_from_dict(entries[rule_id]['rule']))
                    for rule_id in _RULE_IDS)
    n = manifest['n']
    prior = tuple(range(len(tensors)))

    def groups(hypotheses, previous_own, previous_other, action):
        # P1 copies the learner's previous action and does not see the new action.
        other_action = previous_own
        result = {}
        for i in hypotheses:
            reward = round(tensors[i][0, previous_own, previous_other, action, other_action],
                           _REWARD_DECIMALS)
            result.setdefault(reward, []).append(i)
        return tuple((reward, tuple(group)) for reward, group in sorted(result.items()))

    @lru_cache(maxsize=None)
    def value(hypotheses, previous_own, previous_other, remaining):
        if remaining == 0:
            return 0.0
        return max(qvalue(hypotheses, previous_own, previous_other, remaining, action)
                   for action in range(n))

    def qvalue(hypotheses, previous_own, previous_other, remaining, action):
        return sum(len(group) / len(hypotheses) *
                   (reward + value(group, action, previous_own, remaining - 1))
                   for reward, group in groups(hypotheses, previous_own, previous_other, action))

    x, y = _START
    immediate, observed, blinded, outcomes = [], [], [], []
    for action in range(n):
        feedback = groups(prior, x, y, action)
        first = sum(len(group) / len(prior) * reward for reward, group in feedback)
        immediate.append(first)
        observed.append(qvalue(prior, x, y, _HORIZON, action))
        # The hidden rule posterior remains the prior after the first action.
        # The new joint action (action, x) is still observed by the learner.
        blinded.append(first + value(prior, action, x, _HORIZON - 1))
        outcomes.append([
            {'reward': reward, 'probability': len(group) / len(prior),
             'remaining_rule_ids': [_RULE_IDS[i] for i in group]}
            for reward, group in feedback
        ])
    best_immediate = max(immediate)
    best_total = max(observed)
    optimal_actions = [a for a, result in enumerate(observed)
                       if abs(result - best_total) <= 1e-9]
    information_values = [max(0.0, real - blind) if real - blind > -1e-9 else real - blind
                          for real, blind in zip(observed, blinded)]
    witnesses = [a for a in optimal_actions
                 if immediate[a] < best_immediate - 1e-9 and information_values[a] > 1e-9]
    return {
        'kind': 'preselected_finite_belief_information_witness',
        'manifest_seed': manifest['seed'],
        'n': n,
        'rule_ids': list(_RULE_IDS),
        'rule_fingerprints': [entries[rule_id]['fingerprint'] for rule_id in _RULE_IDS],
        'opponent_kind': 'copy_previous',
        'initial_previous_joint_action': [x, y],
        'initial_state_probability_under_uniform_reset': 1 / n**2,
        'horizon_steps': _HORIZON,
        'prior_probabilities': [1 / len(prior)] * len(prior),
        'reward_grouping_decimal_places': _REWARD_DECIMALS,
        'immediate_expected_rewards_by_action': immediate,
        'optimal_total_values_by_action': observed,
        'first_reward_blinded_total_values_by_action': blinded,
        'first_reward_information_values_by_action': information_values,
        'first_feedback_outcomes_by_action': outcomes,
        'optimal_first_actions': optimal_actions,
        'myopically_suboptimal_informative_optimal_actions': witnesses,
        'demonstrates_positive_information_value': bool(witnesses),
        'control_description': ('First reward is included in return but hidden from inference; '
                                'subsequent rewards and actions remain observable.'),
        'interpretation': ('Conditional on the stated state and known two-rule prior. '
                           'The reference knows the opponent and four-step horizon. '
                           'This does not establish neural learning or generalize to '
                           'arbitrary unseen rules.'),
    }
