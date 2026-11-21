"""Deterministic, bounded relational game distributions with shared evaluation.

Rules remain expressions compiled by the existing relational compiler.  The
catalog manipulates rule count and recipe coverage; it does not establish that
these manipulations increase strategic diversity or create useful exploration.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import random

from .common import positive_integer, validate_n, validate_seed
from .relational_catalog import audit_relational_rule
from .relational_rules import (describe_relational_rule, generate_relational_rule,
                               relational_from_dict, relational_to_dict)
from .rule_metrics import canonical_values, common_values, rounded_values, stable_digest

BASE_FAMILIES = ('social', 'contest', 'reciprocity', 'hidden')
SPLITS = ('train', 'fresh_rules', 'structural')
MAX_ATTEMPTS_PER_RULE = 20000


def _validate_config(n, seed, counts, diversity):
    validate_n(n)
    if n not in (3, 4, 5):
        raise ValueError('diverse relational catalog requires n in {3,4,5}')
    if validate_seed(seed) is None:
        raise TypeError('explicit catalog seed required')
    for name in SPLITS:
        positive_integer(counts[name], name)
    if diversity not in ('focused', 'broad'):
        raise ValueError('diversity must be focused or broad')


def _stream_seed(seed, n, split):
    return int.from_bytes(hashlib.sha256(f'diverse-relational-v1/{seed}/{n}/{split}'.encode()).digest()[:8], 'big')


def _family(split, index, diversity):
    if split == 'structural':
        return 'exception'
    if split == 'train' and diversity == 'focused':
        return 'contest'
    return BASE_FAMILIES[index % len(BASE_FAMILIES)]


def _source(n, seed, family):
    """Sample a known recipe; contest gets a deterministic continuous coefficient.

    The existing contest recipe has finitely many normalized payoff tables. A
    seeded multiplier of its conditional reward expands its parameter family,
    while preserving the reward expression and its interpretation.
    """
    rule = generate_relational_rule(n=n, seed=seed, family=family)
    if family != 'contest':
        return rule, None
    multiplier = round(random.Random(f'diverse-contest-v1/{seed}').uniform(.5, 1.5), 8)
    base, conditional = rule.root.children
    yes, no = conditional.children
    conditional = replace(conditional, children=(replace(yes, value=yes.value * multiplier), no))
    return replace(rule, root=replace(rule.root, children=(base, conditional))), multiplier


def _candidate(n, seed, family):
    source, multiplier = _source(n, seed, family)
    values = common_values(source)
    low, high = min(values), max(values)
    if high - low < 1e-10:
        return None
    factor = 2 / (high - low)
    shift = -1 - 2 * low / (high - low)
    played = replace(source, scale=source.scale * factor, shift=source.shift * factor + shift)
    payoffs = common_values(played)
    canonical = stable_digest(canonical_values(payoffs, n))
    return source, played, multiplier, factor, shift, payoffs, canonical


def _entry(split, index, seed, attempt, family, candidate):
    source, played, multiplier, factor, shift, values, canonical = candidate
    return {'id': f'{split}-{index:05d}', 'split': split, 'family': family,
            'seed': seed, 'attempt': attempt, 'coefficient_multiplier': multiplier,
            'source_rule': relational_to_dict(source), 'rule': relational_to_dict(played),
            'normalization': {'factor': factor, 'shift': shift},
            'common_payoffs': list(values), 'fingerprint': stable_digest(rounded_values(values)),
            'canonical_fingerprint': canonical, 'description': describe_relational_rule(source),
            'source_audit': audit_relational_rule(source), 'audit': audit_relational_rule(played)}


def _summary(entries):
    result = {}
    for split in SPLITS:
        rows = [row for row in entries if row['split'] == split]
        signatures = Counter(stable_digest(row['audit']['best_responses']) for row in rows)
        result[split] = {
            'families': dict(sorted(Counter(row['family'] for row in rows).items())),
            'distinct_labelled_stage_best_response_signatures': len(signatures),
            'states_without_universal_pure_best_response': sum(
                row['audit']['states_without_universal_pure_best_response'] for row in rows),
        }
    return result


def build_diverse_manifest(*, n=3, seed=20261118, train_rules=32, fresh_rules=16,
                           structural_rules=16, diversity='broad'):
    counts = {'train': train_rules, 'fresh_rules': fresh_rules, 'structural': structural_rules}
    _validate_config(n, seed, counts, diversity)
    # Reserve held-out games before drawing training. Both diversity levels and
    # all training counts therefore see exactly the same held-out rules.
    seen = set()
    generated = {}
    stats = {}
    for split in ('fresh_rules', 'structural', 'train'):
        rng = random.Random(_stream_seed(seed, n, split))
        entries = []
        attempts = 0
        duplicate = 0
        constant = 0
        for index in range(counts[split]):
            family = _family(split, index, diversity)
            while True:
                candidate_seed = rng.getrandbits(63)
                attempts += 1
                if attempts > counts[split] * MAX_ATTEMPTS_PER_RULE:
                    raise RuntimeError(f'could not generate distinct {split} catalog within attempt limit')
                candidate = _candidate(n, candidate_seed, family)
                if candidate is None:
                    constant += 1
                    continue
                if candidate[-1] in seen:
                    duplicate += 1
                    continue
                seen.add(candidate[-1])
                entries.append(_entry(split, index, candidate_seed, attempts, family, candidate))
                break
        generated[split] = entries
        stats[split] = {'attempts': attempts, 'duplicate_rejected': duplicate, 'constant_rejected': constant}
    entries = [row for split in SPLITS for row in generated[split]]
    return {'version': 1, 'catalog': 'diverse_relational', 'n': n, 'seed': seed,
            'diversity': diversity, 'counts': counts, 'rules': entries,
            'catalog_audit': stats, 'summary': _summary(entries),
            'sampling_note': 'Evaluation reserved first. Train count is nested; focused/broad share evaluation for equal seed and counts.',
            'diversity_note': 'Family and labelled stage-response counts are descriptive, not strategic diversity or exploration-value guarantees.',
            'structural_note': 'Exception recipe is held out from training; it reuses the existing relational primitives.'}


def validate_diverse_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('version') != 1 or manifest.get('catalog') != 'diverse_relational':
        raise ValueError('unsupported diverse relational catalog')
    n, seed, diversity, counts = manifest['n'], manifest['seed'], manifest['diversity'], manifest['counts']
    if set(counts) != set(SPLITS):
        raise ValueError('invalid split counts')
    _validate_config(n, seed, counts, diversity)
    entries = manifest['rules']
    if len(entries) != sum(counts.values()):
        raise ValueError('incorrect number of rules')
    # Replay the sampler in actual generation order. This verifies rejected
    # candidates as well as accepted source seeds and the global uniqueness set.
    by_split = {}
    cursor = 0
    for split in SPLITS:
        by_split[split] = entries[cursor:cursor + counts[split]]
        cursor += counts[split]
    seen = set()
    for split in ('fresh_rules', 'structural', 'train'):
        rng = random.Random(_stream_seed(seed, n, split))
        attempts = duplicate = constant = 0
        for index, entry in enumerate(by_split[split]):
            family = _family(split, index, diversity)
            if entry['id'] != f'{split}-{index:05d}' or entry['split'] != split or entry['family'] != family:
                raise ValueError('rule order, family or split mismatch')
            target = positive_integer(entry['attempt'], 'attempt')
            if target <= attempts or target > counts[split] * MAX_ATTEMPTS_PER_RULE:
                raise ValueError('invalid sampler attempt')
            while attempts < target:
                candidate_seed = rng.getrandbits(63)
                attempts += 1
                candidate = _candidate(n, candidate_seed, family)
                if candidate is None:
                    constant += 1
                elif candidate[-1] in seen:
                    duplicate += 1
                elif attempts != target:
                    raise ValueError('sampler skipped a valid candidate')
                else:
                    break
            if (candidate is None or candidate[-1] in seen or entry['seed'] != candidate_seed):
                raise ValueError('source seed or sampler rejection mismatch')
            seen.add(candidate[-1])
            expected = _entry(split, index, candidate_seed, target, family, candidate)
            if entry != expected:
                raise ValueError('rule provenance, payoff, normalization or audit mismatch')
        stats = manifest['catalog_audit'][split]
        if stats != {'attempts': attempts, 'duplicate_rejected': duplicate, 'constant_rejected': constant}:
            raise ValueError('invalid rejection accounting')
    if manifest['summary'] != _summary(entries):
        raise ValueError('catalog summary mismatch')
    return manifest
