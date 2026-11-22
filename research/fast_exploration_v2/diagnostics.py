"""Analyze rule similarity with independently sampled reachable contexts."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.fast_exploration_v1.simulator import EpisodeBatch
from research.structured_scale.run import FAMILIES, game


PERMUTATIONS = np.array(list(itertools.permutations(range(3))), dtype=int)
CONTEXT_SEED = 48390172
CONTEXT_TURNS = (0, 4, 8, 12, 16, 20, 24, 28)
REFERENCE_GAMES_PER_FAMILY = 32


def reachable_contexts():
    """Generate the fixed, public diagnostic contexts without any evaluation data."""
    rng = np.random.default_rng(CONTEXT_SEED)
    result = {}
    for family in FAMILIES:
        candidates = defaultdict(list)
        generated = []
        for _ in range(16):
            rule_seed = int(rng.integers(2**61))
            reset_seed = int(rng.integers(2**61))
            generated.append([rule_seed, reset_seed])
            rule = game(rule_seed, family, 3, 3)
            env = StructuredEnv(rule, max_steps=32)
            env.reset(seed=reset_seed)
            for t in range(32):
                if t in CONTEXT_TURNS:
                    state = tuple(env._state[var.name] for var in rule.states)
                    candidates[state].append({
                        'history': [list(pair) for pair in env._history],
                        'state': list(state),
                    })
                action = rng.integers(3, size=2)
                env.step(int(action[0]), int(action[1]))
        # Include the all-zero state and sample distinct reachable nonzero states
        # before filling remaining slots with other histories from those states.
        states = sorted(candidates, key=lambda state: (sum(state), state))
        if len(states) > len(CONTEXT_TURNS):
            selected = rng.choice(len(states) - 1, len(CONTEXT_TURNS) - 1, replace=False) + 1
            states = [states[0], *[states[i] for i in sorted(selected)]]
        contexts = []
        for i in range(len(CONTEXT_TURNS)):
            group = candidates[states[i % len(states)]]
            contexts.append(group[int(rng.integers(len(group)))])
        result[family] = {'source_seeds': generated, 'contexts': contexts}
    return result


def signatures(rule_seeds, family, contexts, *, permute=True):
    """Evaluate expected rewards and next-state distributions for all joint actions.

    Histories and current actions are permuted together. State coordinates are
    aligned by their names within the same family. The generated grammar has at
    most one stochastic gate; the two gate outcomes give the exact distribution.
    """
    rules = [game(int(seed), family, 3, 3) for seed in rule_seeds]
    if not rules:
        raise ValueError('at least one rule is required')
    permutations = PERMUTATIONS if permute else PERMUTATIONS[:1]
    count, nperm, nctx = len(rules), len(permutations), len(contexts)
    expanded = [rule for rule in rules for _ in permutations]
    batch = EpisodeBatch(expanded, [0] * len(expanded), 1)
    gates = np.array([
        next((clause.probability for clause in rule.transitions if clause.probability < 1), 1.)
        for rule in expanded
    ])
    reward = np.empty((count, nperm, nctx, 9, 2))
    # Slots absent from a family remain constant, so do not affect distances.
    transition = np.empty((count, nperm, nctx, 9, 3, 3))
    actions = np.tile(permutations, (count, 1))
    for ci, context in enumerate(contexts):
        history = np.array(context['history'], dtype=int)
        permuted = actions[np.arange(len(actions))[:, None, None], history[None]]
        state = np.zeros((len(expanded), 3), dtype=int)
        state[:, :len(context['state'])] = context['state']
        for action in range(9):
            a, b = divmod(action, 3)
            left, right = actions[:, a], actions[:, b]
            outcomes = []
            for uniform in (0., 1.):
                batch.state = state.copy()
                batch.history = permuted.copy()
                batch.elapsed = 0
                payoffs = batch.step(left, right, np.full(len(expanded), uniform))
                outcomes.append((payoffs.copy(), batch.state.copy()))
            r0, s0 = outcomes[0]
            r1, s1 = outcomes[1]
            values = gates[:, None] * r0 + (1 - gates[:, None]) * r1
            reward[:, :, ci, action] = values.reshape(count, nperm, 2)
            indicators = np.eye(3)[s0] * gates[:, None, None]
            indicators += np.eye(3)[s1] * (1 - gates[:, None, None])
            transition[:, :, ci, action] = indicators.reshape(count, nperm, 3, 3)
    return reward, transition


def nearest_signatures(train_reward, train_transition, test_reward, test_transition):
    """Find the closest reference with one shared permutation per test game."""
    reward_distance = []
    transition_distance = []
    for rewards, states in zip(test_reward, test_transition, strict=True):
        delta = rewards[:, None] - train_reward[None, :, 0]
        reward_distance.append(np.sqrt(np.mean(delta * delta, axis=tuple(range(2, delta.ndim)))).min())
        delta = states[:, None] - train_transition[None, :, 0]
        transition_distance.append(np.sqrt(np.mean(delta * delta, axis=tuple(range(2, delta.ndim)))).min())
    return -np.asarray(reward_distance), -np.asarray(transition_distance)


def rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(x, y):
    if len(x) < 3:
        return None
    x, y = rank(x), rank(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest['config']
    contexts = reachable_contexts()
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    references = {}
    for index in range(cfg['blocks']):
        stem = root / f'block_{index:03d}'
        record_path = stem.with_suffix('.json')
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        if record['block'] != index:
            raise ValueError(f'misnumbered block record: {stem}')
        for suffix, key in (('.npz', 'raw_sha256'), ('.pt', 'weights_sha256')):
            with stem.with_suffix(suffix).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != record[key]:
                    raise ValueError(f'checksum mismatch: {stem}{suffix}')
        with np.load(stem.with_suffix('.npz')) as raw:
            methods = list(raw['methods'])
            if len(set(methods)) != len(methods):
                raise ValueError(f'duplicate method labels in {stem}')
            full = methods.index('pooled_full')
            masked = methods.index('pooled_masked')
            ks = raw['ks'].astype(int)
            if ks[0] != 0 or not np.all(np.diff(ks) > 0):
                raise ValueError(f'invalid K grid in {stem}')
            rewards = raw['controlled_rewards']
            if rewards.ndim != 5 or rewards.shape[:2] != (len(methods), len(ks)) or rewards.shape[-1] != 2:
                raise ValueError(f'invalid controlled reward shape in {stem}')
            seeds = raw['game_seeds']
            samples = len(raw['reset_seeds'])
            if samples % len(seeds):
                raise ValueError('unequal number of resets per game')
            resets = samples // len(seeds)
            families = np.asarray(raw['families']).reshape(len(seeds), resets)
            if not np.all(families == families[:, :1]):
                raise ValueError('family differs between game resets')
            families = families[:, 0]
            curves = rewards[..., 0].mean(axis=-1).reshape(len(methods), len(ks), len(seeds), resets).mean(axis=-1)
            np.testing.assert_allclose(curves[full, 0], curves[masked, 0], atol=1e-12, rtol=0)
            difference = np.trapezoid(curves[full] - curves[masked], x=ks, axis=0) / ks[-1]
            initial = curves[full, 0]
            training = np.asarray(raw['train_rule_seeds']).reshape(-1)
            if 'train_families' in raw:
                training_families = np.asarray(raw['train_families'])
            else:
                training_families = np.asarray(raw['training_families'])
            if training_families.size != training.size:
                if training.size % training_families.size:
                    raise ValueError('training family labels do not match rule seeds')
                training_families = np.tile(training_families, training.size // training_families.size)
            training_families = training_families.reshape(-1)
            if training_families.size != training.size:
                raise ValueError('training family labels do not match rule seeds')
            ref_rng = np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED, int(record['master_seed']),
                                                                     int(record['seed_block'])]))
            for family in FAMILIES:
                eligible = training[training_families == family]
                tests = np.flatnonzero(families == family)
                if not len(eligible) or not len(tests):
                    raise ValueError(f'insufficient {family} rule coverage in {stem}')
                n_reference = min(len(eligible), REFERENCE_GAMES_PER_FAMILY)
                chosen = ref_rng.choice(len(eligible), size=n_reference, replace=False)
                reference = eligible[chosen]
                references[f'{index:03d}/{family}'] = [int(seed) for seed in reference]
                reference_rules = [game(int(seed), family, 3, 3) for seed in reference]
                reference_ids = {rule.fingerprint() for rule in reference_rules}
                test_ids = {str(raw['test_fingerprints'][j]) for j in tests}
                rebuilt_ids = {game(int(seeds[j]), family, 3, 3).fingerprint() for j in tests}
                if rebuilt_ids != test_ids:
                    raise ValueError(f'evaluation rule fingerprints differ in {stem}')
                if reference_ids.intersection(test_ids):
                    raise ValueError(f'training and evaluation rule overlap in {stem}')
                train_r, train_t = signatures(reference, family, contexts[family]['contexts'], permute=False)
                test_r, test_t = signatures(seeds[tests], family, contexts[family]['contexts'])
                reward_similarity, transition_similarity = nearest_signatures(train_r, train_t, test_r, test_t)
                for local, game_index in enumerate(tests):
                    rows.append(dict(block=index, master_seed=record['master_seed'],
                                     seed_block=record['seed_block'], game_index=int(game_index),
                                     game_seed=int(seeds[game_index]), family=family,
                                     reference_count=n_reference,
                                     reward_similarity=reward_similarity[local],
                                     transition_similarity=transition_similarity[local],
                                     initial_reward=initial[game_index],
                                     pooled_full_minus_masked_auc=difference[game_index]))
    if not rows:
        raise ValueError(f'no completed blocks in {root}')

    groups = defaultdict(list)
    for row in rows:
        groups[row['block'], row['master_seed'], row['family']].append(row)
    strata = []
    for (block, master, family), group in sorted(groups.items()):
        for metric in ('reward_similarity', 'transition_similarity'):
            x = [row[metric] for row in group]
            y = [row['pooled_full_minus_masked_auc'] for row in group]
            strata.append(dict(block=block, master_seed=master, family=family, metric=metric,
                               n_games=len(group), distinct_similarity=len(set(x)),
                               spearman=spearman(x, y)))
    estimates = {}
    for metric in ('reward_similarity', 'transition_similarity'):
        block_scores = defaultdict(list)
        for row in strata:
            if row['metric'] == metric and row['spearman'] is not None:
                block_scores[row['block'], row['master_seed']].append(row['spearman'])
        values = [(block, master, float(np.mean(scores)))
                  for (block, master), scores in sorted(block_scores.items())]
        if values:
            by_master = defaultdict(list)
            for _, master, value in values:
                by_master[master].append(value)
            # Equal master-seed weights retain the intended training distribution.
            mean = float(np.mean([np.mean(v) for v in by_master.values()]))
            interval = None
            if all(len(v) >= 2 for v in by_master.values()):
                rng = np.random.default_rng(48402091)
                draws = np.stack([
                    rng.choice(group, size=(20000, len(group)), replace=True).mean(axis=1)
                    for group in by_master.values()
                ]).mean(axis=0)
                interval = np.quantile(draws, [.025, .975]).tolist()
        else:
            mean, interval = None, None
        estimates[metric] = dict(mean_spearman=mean, interval_95=interval,
                                 valid_blocks=len(values), total_blocks=len({r['block'] for r in rows}),
                                 valid_strata=sum(r['metric'] == metric and r['spearman'] is not None for r in strata),
                                 total_strata=sum(r['metric'] == metric for r in strata))

    _write_csv(output / 'similarity_games.csv', rows, list(rows[0]))
    _write_csv(output / 'similarity_strata.csv', strata, list(strata[0]))
    with manifest_path.open('rb') as stream:
        manifest_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with Path(__file__).open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(scope='descriptive within-family, within-block rank association on opened evaluation rules',
                  run_manifest_sha256=manifest_hash, analysis_source_sha256=source_hash,
                  context_seed=CONTEXT_SEED,
                  contexts=contexts, reference_games_per_family_max=REFERENCE_GAMES_PER_FAMILY,
                  selected_reference_rule_seeds=references,
                  estimates=estimates)
    (output / 'similarity_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    print(json.dumps(result['estimates'], indent=2))


if __name__ == '__main__':
    main()
