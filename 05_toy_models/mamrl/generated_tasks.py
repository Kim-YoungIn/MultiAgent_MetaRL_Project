"""Auditable logical-game catalogs and shared held-out task manifests.

All metadata here is researcher-only. Learners receive environment observations.
"""

from dataclasses import dataclass
import random

from .common import positive_integer, validate_n, validate_seed
from .logical_rules import LogicalRule, Payoff, generate_logical_rule, rule_to_dict, rule_from_dict


CONDITIONS = ('low_rules_low_opponents', 'high_rules_low_opponents',
              'low_rules_high_opponents', 'high_rules_high_opponents')
TRAIN_OPPONENTS = ('random', 'copy_previous', 'avoid_previous')
SPLITS = ('combination', 'fresh_rules', 'structural', 'novel_opponent')


# Re-export historical names for existing callers and snapshots.
from .rule_metrics import (stable_digest as _digest, rounded_values as _rounded,
                           expression_tokens as _tokens, depends_on_axis as _depends,
                           common_values, canonical_values, standardized_values, tensor_distance)


def structure_distance(left, right):
    a, b = _tokens(rule_to_dict(left)), _tokens(rule_to_dict(right))
    keys = set(a) | set(b)
    union = sum(max(a[k], b[k]) for k in keys)
    return 1 - sum(min(a[k], b[k]) for k in keys) / union if union else 0.0


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    rule_id: str
    rule: LogicalRule
    opponent_kind: str
    split: str
    structure: str


def build_manifest(*, n=3, seed=20261114):
    validate_n(n)
    if n > 5:
        raise ValueError('catalog permutation audit currently supports n<=5')
    if validate_seed(seed) is None:
        raise TypeError('manifest seed must be explicit')
    rng = random.Random(seed)
    rules, seen, audit = [], set(), {'attempts':0, 'constant':0, 'not_joint_dependent':0, 'duplicate':0}
    for split, count in (('train',12), ('fresh_rules',6), ('structural',6)):
        made = 0
        while made < count:
            audit['attempts'] += 1
            if audit['attempts'] > 20000:
                raise RuntimeError('could not construct unique joint-dependent catalog')
            source_seed = rng.getrandbits(63)
            structure = 'nested' if split == 'structural' else ('atom','and','or')[made % 3]
            source = generate_logical_rule(n=n, seed=source_seed, structure=structure)
            values = common_values(source)
            lo, hi = min(values), max(values)
            if hi-lo < 1e-10:
                audit['constant'] += 1
                continue
            if not (_depends(values,n,2) and _depends(values,n,3)):
                audit['not_joint_dependent'] += 1
                continue
            factor, shift = 2/(hi-lo), -1 - 2*lo/(hi-lo)
            rule = LogicalRule(n, Payoff('add', children=(
                Payoff('scale', value=factor, children=(source.root,)), Payoff('const',value=shift))))
            common = common_values(rule)
            canonical = _digest(canonical_values(common,n))
            if canonical in seen:
                audit['duplicate'] += 1
                continue
            seen.add(canonical)
            entry = {'id':f'r{len(rules):03d}', 'split':split, 'structure':structure, 'seed':source_seed,
                     'source_rule':rule_to_dict(source), 'rule':rule_to_dict(rule),
                     'normalization':{'factor':factor,'shift':shift},
                     'common_payoffs':list(common), 'fingerprint':_digest(_rounded(common)),
                     'canonical_fingerprint':canonical}
            rules.append(entry)
            made += 1
    tasks = []
    for entry in rules:
        kinds = TRAIN_OPPONENTS if entry['split'] != 'train' else (*TRAIN_OPPONENTS,'win_stay')
        i = int(entry['id'][1:])
        for j, kind in enumerate(kinds):
            if entry['split'] != 'train':
                split = entry['split']
            elif kind == 'win_stay':
                split = 'novel_opponent'
            elif (i+j)%3 == 0 and kind != 'copy_previous':
                split = 'combination'
            else:
                split = 'train'
            tasks.append({'id':entry['id']+'/'+kind, 'rule_id':entry['id'],
                          'opponent_kind':kind, 'split':split})
    conditions = {}
    for name in CONDITIONS:
        high_rules = name.startswith('high_rules')
        high_opponents = name.endswith('high_opponents')
        conditions[name] = [t['id'] for t in tasks if t['split']=='train'
                            and (high_rules or int(t['rule_id'][1:]) < 4)
                            and (high_opponents or t['opponent_kind']=='copy_previous')]
    result = {'version':1, 'n':n, 'seed':seed, 'rules':rules, 'tasks':tasks, 'conditions':conditions,
              'catalog_audit':audit, 'train_opponents':list(TRAIN_OPPONENTS),
              'test_splits':list(SPLITS), 'distance_note':'rule permutation quotient only; opponents not relabeled'}
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    if manifest.get('version') != 1:
        raise ValueError('unsupported manifest version')
    rules = manifest['rules']
    ids = [r['id'] for r in rules]
    if ids != [f'r{i:03d}' for i in range(24)]:
        raise ValueError('catalog requires the predefined 24 ordered rule IDs')
    canonical = []
    for entry in rules:
        index = int(entry['id'][1:])
        expected_split = 'train' if index<12 else ('fresh_rules' if index<18 else 'structural')
        expected_structure = ('atom','and','or')[index%3] if index<18 else 'nested'
        if entry['split'] != expected_split or entry['structure'] != expected_structure:
            raise ValueError('rule split/structure does not match the predefined catalog')
        rule = rule_from_dict(entry['rule'])
        source = rule_from_dict(entry['source_rule'])
        if rule.n != manifest['n'] or source.n != rule.n:
            raise ValueError('manifest n mismatch')
        values = common_values(rule)
        if _rounded(values) != _rounded(entry['common_payoffs']):
            raise ValueError('manifest payoff mismatch')
        if entry['fingerprint'] != _digest(_rounded(values)):
            raise ValueError('manifest raw fingerprint mismatch')
        fp = _digest(canonical_values(values,rule.n))
        if entry['canonical_fingerprint'] != fp:
            raise ValueError('manifest canonical fingerprint mismatch')
        if min(values) < -1-1e-9 or max(values) > 1+1e-9:
            raise ValueError('manifest rewards must be normalized')
        norm = entry['normalization']
        expected = tuple(norm['factor']*v+norm['shift'] for v in common_values(source))
        if norm['factor'] <= 0 or _rounded(values) != _rounded(expected):
            raise ValueError('invalid affine transform provenance')
        canonical.append(fp)
    if len(canonical) != len(set(canonical)):
        raise ValueError('equivalent rules across manifest')
    tasks = {t['id']:t for t in manifest['tasks']}
    if len(tasks) != len(manifest['tasks']):
        raise ValueError('duplicate task IDs')
    rule_map = {r['id']:r for r in rules}
    expected_ids = {r['id']+'/'+k for r in rules for k in
                    ((*TRAIN_OPPONENTS,'win_stay') if r['split']=='train' else TRAIN_OPPONENTS)}
    if set(tasks) != expected_ids:
        raise ValueError('manifest must contain the complete predefined task table')
    for task in tasks.values():
        if task['rule_id'] not in rule_map or task['split'] not in ('train',*SPLITS):
            raise ValueError('invalid task reference/split')
        if task['id'] != task['rule_id']+'/'+task['opponent_kind']:
            raise ValueError('invalid task identity')
        if task['opponent_kind'] not in (*TRAIN_OPPONENTS,'win_stay'):
            raise ValueError('unknown manifest opponent')
        rs = rule_map[task['rule_id']]['split']
        if (rs != 'train' and task['split'] != rs) or (task['split']=='novel_opponent' and task['opponent_kind']!='win_stay'):
            raise ValueError('inconsistent rule/task split')
        if task['opponent_kind']=='win_stay' and task['split']!='novel_opponent':
            raise ValueError('novel opponent must be held out')
        if rs == 'train':
            kind = task['opponent_kind']
            index = int(task['rule_id'][1:])
            expected = 'novel_opponent' if kind=='win_stay' else (
                'combination' if (index+TRAIN_OPPONENTS.index(kind))%3==0 and kind!='copy_previous' else 'train')
            if task['split'] != expected:
                raise ValueError('task split violates the predefined holdout rule')
    if set(manifest['conditions']) != set(CONDITIONS):
        raise ValueError('missing experiment condition')
    for name, selected in manifest['conditions'].items():
        if not selected or len(selected)!=len(set(selected)) or any(t not in tasks or tasks[t]['split']!='train' for t in selected):
            raise ValueError('held-out or invalid task in training condition')
        expected = {t['id'] for t in tasks.values() if t['split']=='train'
                    and (name.startswith('high_rules') or int(t['rule_id'][1:])<4)
                    and (name.endswith('high_opponents') or t['opponent_kind']=='copy_previous')}
        if set(selected) != expected:
            raise ValueError('training condition violates the predefined diversity design')
    return manifest


def tasks_from_manifest(manifest, *, split=None, condition=None):
    if split is not None and condition is not None:
        raise ValueError('choose split OR condition')
    if condition is not None and condition not in CONDITIONS:
        raise ValueError('unknown condition')
    if split is not None and split not in ('train',*SPLITS):
        raise ValueError('unknown split')
    selected = set(manifest['conditions'][condition]) if condition is not None else None
    rules = {r['id']:r for r in manifest['rules']}
    return tuple(GeneratedTask(t['id'],t['rule_id'],rule_from_dict(rules[t['rule_id']]['rule']),
                               t['opponent_kind'],t['split'],rules[t['rule_id']]['structure'])
                 for t in manifest['tasks'] if (selected is None or t['id'] in selected)
                 and (split is None or t['split']==split))


def distance_manifest(manifest):
    rules = {r['id']:rule_from_dict(r['rule']) for r in manifest['rules']}
    sources = {r['id']:rule_from_dict(r['source_rule']) for r in manifest['rules']}
    result = {}
    for condition in CONDITIONS:
        training_ids = sorted({t.rule_id for t in tasks_from_manifest(manifest,condition=condition)})
        result[condition] = {}
        for rule_id, rule in rules.items():
            distances = [{'train_rule_id':other,
                          'payoff_l1':tensor_distance(sources[rule_id],sources[other]),
                          'standardized_l1':tensor_distance(rule,rules[other],standardize=True),
                          'permutation_l1':tensor_distance(rule,rules[other],canonical=True),
                          'structural_jaccard':structure_distance(rule,rules[other])}
                         for other in training_ids]
            result[condition][rule_id] = {'nearest_'+key:min(d[key] for d in distances)
                                          for key in ('payoff_l1','standardized_l1','permutation_l1','structural_jaccard')}
    return result
