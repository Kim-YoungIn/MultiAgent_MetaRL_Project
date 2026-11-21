"""Finite two-step reward-observation audit with a uniform opponent."""
from collections import defaultdict
from itertools import product

from mamrl.meta_exploration import decode_rule
from mamrl.probe_templates import exact_information_audit
from mamrl.rules import compile_rule


def pair_value(left, right, opponent="uniform"):
    tensors = [compile_rule(decode_rule(row['rule'])) for row in (left, right)]
    n = tensors[0].n
    if tensors[1].n != n:
        raise ValueError('pair requires the same action count')
    if opponent not in ('uniform', 'steady', 'sarsa'):
        raise ValueError('unknown opponent')
    from mamrl.adaptive_agents import OnlineQ, ExpectedSARSA
    from mamrl.environment import Observation
    observed_total = masked_total = myopic_total = 0.0
    for p, q in product(range(n), repeat=2):
        candidates = []
        for a in range(n):
            immediate = 0.0
            observed = masked = 0.0
            for b in range(n):
                groups = defaultdict(list)
                for tensor in tensors:
                    reward = tensor[0, p, q, a, b]
                    immediate += reward / (2 * n)
                    probs = [1/n] * n
                    if opponent != 'uniform':
                        agent = (OnlineQ(n, 0, alpha=.3, epsilon=.15) if opponent == 'steady'
                                 else ExpectedSARSA(n, 0, alpha=.5, epsilon=.2))
                        old = Observation(q, p, 0, True)
                        next_obs = Observation(b, a, tensor[1, p, q, a, b], False)
                        agent.update(old, b, next_obs.previous_own_reward, next_obs, terminal=False)
                        probs = agent.distribution(next_obs)
                    groups[reward].append((tensor, probs))
                observed += sum(max(sum(prob * tensor[0, a, b, x, y]
                                        for tensor, probs in group for y, prob in enumerate(probs))
                                    for x in range(n)) / (2 * n) for group in groups.values())
                all_members = [item for group in groups.values() for item in group]
                masked += max(sum(prob * tensor[0, a, b, x, y]
                                  for tensor, probs in all_members for y, prob in enumerate(probs))
                              for x in range(n)) / (2 * n)
            candidates.append((immediate, immediate + observed, immediate + masked))
        observed_total += max(row[1] for row in candidates) / n**2
        masked_total += max(row[2] for row in candidates) / n**2
        best_immediate = max(row[0] for row in candidates)
        myopic_total += max(row[1] for row in candidates if abs(row[0] - best_immediate) < 1e-10) / n**2
    return {'observed': observed_total, 'masked': masked_total,
            'observation_value': observed_total - masked_total,
            'myopic': myopic_total, 'active_value': observed_total - myopic_total,
            'opponent': opponent, 'horizon': 2}


def audit_panel(entries):
    families = defaultdict(list)
    for row in entries:
        families[(row['split'], row['family'])].append(row)
    from mamrl.rule_metrics import expression_tokens
    result = {'probe': exact_information_audit(),
              'probe_short': exact_information_audit(horizon=1),
              'probe_dominant': exact_information_audit(safe_reward=4), 'pairs': [],
              'rules': [{'id': row['id'], 'family': row['family'], 'split': row['split'],
                         'audit': row.get('audit'),
                         'relation_tokens': dict(expression_tokens(row['source_rule']))} for row in entries]}
    for (split, family), rows in sorted(families.items()):
        if len(rows) % 2:
            raise ValueError(f'odd family size for {split}/{family}')
        for i in range(0, len(rows), 2):
            left = compile_rule(decode_rule(rows[i]['rule']))
            right = compile_rule(decode_rule(rows[i+1]['rule']))
            n = left.n
            difference = sum(left[0, p, q, a, b] != right[0, p, q, a, b]
                             for p, q, a, b in product(range(n), repeat=4)) / n**4
            common = 0
            for p, q in product(range(n), repeat=2):
                best = []
                for tensor in (left, right):
                    values = [sum(tensor[0, p, q, a, b] for b in range(n))/n for a in range(n)]
                    maximum = max(values)
                    best.append({a for a, value in enumerate(values) if abs(value-maximum) < 1e-10})
                common += bool(best[0] & best[1])
            result['pairs'].append({'split': split, 'family': family,
                                    'reward_difference_fraction': difference,
                                    'shared_immediate_best_response_fraction': common/n**2,
                                    'rules': [rows[i]['id'], rows[i+1]['id']],
                                    'results': {opponent: pair_value(rows[i], rows[i+1], opponent) for opponent in ('uniform', 'steady', 'sarsa')}})
    result['summary'] = {}
    for (split, family) in sorted(families):
        pairs = [row for row in result['pairs'] if row['split'] == split and row['family'] == family]
        if not pairs:
            continue
        result['summary'][split + '/' + family] = {
            'pairs': len(pairs),
            'identifiable_fraction': sum(row['reward_difference_fraction'] > 0 for row in pairs) / len(pairs),
            'mean_shared_immediate_best_response_fraction': sum(row['shared_immediate_best_response_fraction'] for row in pairs) / len(pairs),
            'zero_fraction': {opponent: {metric: sum(abs(row['results'][opponent][metric]) < 1e-10 for row in pairs) / len(pairs)
                for metric in ('observation_value', 'active_value')} for opponent in ('uniform', 'steady', 'sarsa')},
            'values': {opponent: {metric: [row['results'][opponent][metric] for row in pairs]
                for metric in ('observation_value', 'active_value')} for opponent in ('uniform', 'steady', 'sarsa')}}
    return result


def strategy_losses(training_entries, eval_entries):
    """Minimum three-step loss over training rules and common relabellings."""
    from functools import lru_cache
    from itertools import permutations

    def optimal_policy(tensor, opponent):
        n = tensor.n
        @lru_cache(None)
        def value(t, p, q):
            if t == 0:
                return 0.0
            probs = [float(b == p) for b in range(n)] if opponent == 'copy' else [1/n] * n
            return max(sum(probs[b] * (tensor[0, p, q, a, b] + value(t-1, a, b))
                           for b in range(n)) for a in range(n))
        policies = {}
        for t in range(1, 4):
            for p, q in product(range(n), repeat=2):
                probs = [float(b == p) for b in range(n)] if opponent == 'copy' else [1/n] * n
                values = [sum(probs[b] * (tensor[0, p, q, a, b] + value(t-1, a, b))
                              for b in range(n)) for a in range(n)]
                maximum = max(values)
                policies[t, p, q] = tuple(a for a, score in enumerate(values) if abs(score - maximum) < 1e-10)
        mean = sum(value(3, p, q) for p, q in product(range(n), repeat=2)) / n**2
        return policies, mean

    training = [compile_rule(decode_rule(row['rule'])) for row in training_entries]
    source_policies = [(tensor.n, {opp: optimal_policy(tensor, opp)[0] for opp in ('uniform', 'copy')})
                       for tensor in training]
    result = {}
    for row in eval_entries:
        target = compile_rule(decode_rule(row['rule']))
        n = target.n
        target_values = {opp: optimal_policy(target, opp)[1] for opp in ('uniform', 'copy')}
        best_loss = float('inf')
        for source_n, policies in source_policies:
            if source_n != n:
                raise ValueError('mixed action counts')
            for perm in permutations(range(n)):
                inv = {v: k for k, v in enumerate(perm)}
                total = 0.0
                for opponent in ('uniform', 'copy'):
                    policy = policies[opponent]
                    @lru_cache(None)
                    def transfer(t, p, q):
                        if t == 0:
                            return 0.0
                        probs = [float(b == p) for b in range(n)] if opponent == 'copy' else [1/n] * n
                        actions = [inv[a] for a in policy[t, perm[p], perm[q]]]
                        return sum(sum(probs[b] * (target[0, p, q, a, b] + transfer(t-1, a, b))
                                       for b in range(n)) for a in actions) / len(actions)
                    observed = sum(transfer(3, p, q) for p, q in product(range(n), repeat=2)) / n**2
                    total += (target_values[opponent] - observed) / 2
                best_loss = min(best_loss, total)
        result[row['id']] = best_loss
    return result
