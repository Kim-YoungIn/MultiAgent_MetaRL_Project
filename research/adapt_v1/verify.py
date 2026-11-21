"""Independently reconstruct saved adaptation experiment calculations.

Run as python -m research.adapt_v1.verify OUTPUT. Checks operate on raw NPZ
arrays and use separately implemented scalar-game equations. A successful
report does not certify the simulator's scientific model or statistical power.
"""
import argparse
import csv
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np

CANDIDATES = tuple(itertools.product((-1, 1), (-1, 1), (0.12, 0.65)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def close(name, actual, expected, tolerance=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        error = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
        raise AssertionError(f"{name}: shapes {actual.shape}, {expected.shape}; maximum error {error}")


def likelihood(gain, arena, a, b):
    """Log likelihood for eight signed-rate candidates from observed histories."""
    games, resets, horizon = arena.shape
    out = np.zeros((games, 8))
    row = np.arange(games)[:, None]
    for candidate, (sign0, sign1, rate) in enumerate(CANDIDATES):
        signs = np.array([sign0, sign1])
        belief = np.full((games, resets), 0.5)
        for t in range(horizon):
            c = arena[:, :, t].astype(int)
            p = sigmoid(signs[c] * gain[row, c] * belief)
            bt = b[:, :, t]
            out[:, candidate] += np.log(np.where(bt == 1, p, 1 - p)).sum(axis=1)
            belief = (1 - rate) * belief + rate * (2 * a[:, :, t] - 1)
    return out


def em_fit(loglik):
    same = np.array([s0 == s1 for s0, s1, _ in CANDIDATES])
    fast = np.array([rate == 0.65 for _, _, rate in CANDIDATES])
    q = r = 0.5
    for _ in range(80):
        prior = 0.5 * np.where(same, q, 1 - q) * np.where(fast, r, 1 - r)
        weights = loglik + np.log(prior)[None, :]
        weights -= weights.max(axis=1, keepdims=True)
        weights = np.exp(weights)
        weights /= weights.sum(axis=1, keepdims=True)
        q = (weights[:, same].sum() + 1) / (len(weights) + 2)
        r = (weights[:, fast].sum() + 1) / (len(weights) + 2)
    return np.array([q, r])


def probe_predictions(data, q, r, frozen=False):
    """Filter shared probe histories, then predict action one in both arenas."""
    n = len(data['source'])
    gain, source = data['gain'], data['source'].astype(int)
    a, b = data['probe_a'], data['probe_b']
    signs = np.array([row[:2] for row in CANDIDATES])
    rates = np.array([row[2] for row in CANDIDATES])
    same = signs[:, 0] == signs[:, 1]
    fast = rates == .65
    prior = .5 * np.where(same, q, 1 - q) * np.where(fast, r, 1 - r)
    weights = np.broadcast_to(prior, (n, 8)).copy()
    belief = np.full((n, 8), .5)
    result = np.empty((n, len(data['budgets']), 2))
    nrow = np.arange(n)
    for t in range(17):
        if t in data['budgets']:
            ki = int(np.where(data['budgets'] == t)[0][0])
            for location, arena in enumerate((source, 1-source)):
                p = sigmoid(signs[:, arena].T * gain[nrow, arena, None] * belief)
                result[:, ki, location] = (weights * p).sum(axis=1)
        if t == 16:
            break
        p = sigmoid(signs[:, source].T * gain[nrow, source, None] * belief)
        if not frozen:
            weights *= np.where(b[:, t, None] == 1, p, 1-p)
            weights /= weights.sum(axis=1, keepdims=True)
        belief = (1-rates) * belief + rates * (2*a[:, t, None] - 1)
    return result


def probe_truth(data, shifted):
    n = len(data['source'])
    signs, rate, gain = data['signs'], data['rate'], data['gain']
    source = data['source'].astype(int)
    a, b = data['probe_a'], data['probe_b']
    u = data['probe_u']
    row = np.arange(n)
    belief = np.full((n, 2 if shifted else 1), .5)
    result = np.empty((n, len(data['budgets']), 2))
    for t in range(17):
        if t in data['budgets']:
            ki = int(np.where(data['budgets'] == t)[0][0])
            for j, arena in enumerate((source, 1-source)):
                m = belief[row, arena] if shifted else belief[:, 0]
                result[:, ki, j] = sigmoid(signs[row, arena] * gain[row, arena] * m)
        if t == 16:
            break
        m = belief[row, source] if shifted else belief[:, 0]
        p = sigmoid(signs[row, source] * gain[row, source] * m)
        close('probe opponent draws', b[:, t], (u[:, t] < p).astype(int), 0)
        update = (1-rate)*m + rate*(2*a[:, t]-1)
        if shifted:
            belief[row, source] = update
        else:
            belief[:, 0] = update
    return result


def verify_probe(data, fits, coef, shifted):
    truth = probe_truth(data, shifted)
    methods = data['methods'].tolist()
    for i, name in enumerate(methods):
        prediction = data['pred'][i]
        if name in fits:
            close(name + ' posterior prediction', prediction, probe_predictions(data, *fits[name]))
        elif name == 'independent':
            close(name + ' posterior prediction', prediction, probe_predictions(data, .5, .5))
        elif name in ('oracle75', 'oracle50', 'oracle25'):
            q = {'oracle75': .75, 'oracle50': .5, 'oracle25': .25}[name]
            close(name + ' posterior prediction', prediction, probe_predictions(data, q, .5))
        elif name == 'frozen':
            close(name + ' frozen prediction', prediction, np.full(prediction.shape, .5))
        elif name == 'frequency':
            expected = np.full(prediction.shape, .5)
            for ki, budget in enumerate(data['budgets']):
                expected[:, ki, 0] = (1 + data['probe_b'][:, :budget].sum(axis=1))/(2 + budget)
            close(name + ' frequency prediction', prediction, expected)
        elif name == 'pooled':
            close(name + ' prediction', prediction, pooled_probe(data, coef))
        elif name == 'type_oracle':
            close(name + ' prediction', prediction, truth)
        source = data['source'].astype(int)
        bonuses = np.stack((data['bonus'][np.arange(len(source)), source], data['bonus'][np.arange(len(source)), 1-source]), axis=1)
        choose = 2*prediction + bonuses[:, None, :] > 1
        expected = np.where(choose, truth + bonuses[:, None, :], 1-truth)
        brier = truth * (1-prediction)**2 + (1-truth)*prediction**2
        close(name + ' query value', data['value'][i], expected)
        close(name + ' query brier', data['brier'][i], brier)


def pooled_probe(data, coef):
    n = len(data['source'])
    row = np.arange(n)
    source = data['source'].astype(int)
    traces = np.full((n, 2), .5)
    rates = np.array([.12, .65])
    out = np.empty((n, len(data['budgets']), 2))
    for t in range(17):
        if t in data['budgets']:
            k = int(np.where(data['budgets'] == t)[0][0])
            for j, c in enumerate((source, 1-source)):
                x = np.column_stack([np.ones(n), data['gain'][row, c]*traces[:, 0],
                                     data['gain'][row, c]*traces[:, 1]])
                out[:, k, j] = sigmoid((x*coef[c[:, None]*3 + np.arange(3)]).sum(axis=1))
        if t < 16:
            traces = (1-rates)*traces + rates*(2*data['probe_a'][:, t, None]-1)
    return out


def verify_pooled_fit(data):
    arena, action, moves = data['arena'], data['a'], data['b']
    count, resets, horizon = arena.shape
    row = np.arange(count)[:, None]
    traces = np.full((count, resets, 2), .5)
    gradient = -data['pooled'].copy()
    for t in range(horizon):
        c = arena[:, :, t]
        x = np.stack((np.ones_like(c), data['gain'][row, c]*traces[:, :, 0],
                      data['gain'][row, c]*traces[:, :, 1]), axis=-1)
        predicted = sigmoid((x * data['pooled'][c[..., None]*3 + np.arange(3)]).sum(axis=-1))
        for context in (0, 1):
            gradient[3*context:3*context+3] += (x * ((moves[:, :, t]-predicted)*(c == context))[..., None]).sum(axis=(0, 1))
        traces = (1-np.array([.12, .65]))*traces + np.array([.12, .65])*(2*action[:, :, t, None]-1)
    close('pooled optimum gradient', gradient, np.zeros(6), 1e-8)

def verify_online(data, fits, coef, shifted):
    signs, rate, gain = data['signs'], data['rate'], data['gain']
    arenas, uniforms = data['online_arena'], data['online_u']
    row = np.arange(len(rate))
    for j, name in enumerate(data['methods'].tolist()):
        belief = np.full((len(rate), 2 if shifted else 1), .5)
        candidate_m = np.full((len(rate), 8), .5)
        signs8 = np.array([candidate[:2] for candidate in CANDIDATES])
        rates8 = np.array([candidate[2] for candidate in CANDIDATES])
        same = signs8[:, 0] == signs8[:, 1]
        fast = rates8 == .65
        q, r = fits.get(name, ({'oracle75': .75, 'oracle50': .5, 'oracle25': .25}.get(name, .5), .5))
        weights = np.broadcast_to(.5*np.where(same, q, 1-q)*np.where(fast, r, 1-r), (len(rate), 8)).copy()
        counts = np.ones((len(rate), 2))
        totals = np.full((len(rate), 2), 2.)
        for t, c in enumerate(arenas.T):
            m = belief[row, c] if shifted else belief[:, 0]
            truth = sigmoid(signs[row, c] * gain[row, c] * m)
            a, b = data['focal_move'][j, :, t], data['opponent_move'][j, :, t]
            candidate_p = sigmoid(signs8[:, c].T * gain[row, c, None] * candidate_m)
            if name == 'type_oracle':
                pred = truth
            elif name == 'frequency':
                pred = counts[row, c] / totals[row, c]
            elif name == 'frozen':
                pred = np.full(len(rate), .5)
            elif name != 'pooled':
                pred = (weights*candidate_p).sum(axis=1)
            else:
                x = np.column_stack([np.ones(len(rate)), gain[row, c]*candidate_m[:, 0],
                                     gain[row, c]*candidate_m[:, 1]])
                pred = sigmoid((x*coef[c[:, None]*3 + np.arange(3)]).sum(axis=1))
            if pred is not None:
                choose = (2*pred + data['bonus'][row, c] > 1).astype(int)
                close(name + ' online focal decision', a, choose, 0)
            weights *= np.where(b[:, None] == 1, candidate_p, 1-candidate_p)
            weights /= weights.sum(axis=1, keepdims=True)
            candidate_m = (1-rates8)*candidate_m + rates8*(2*a[:, None]-1)
            counts[row, c] += b
            totals[row, c] += 1
            close(name + ' online opponent probability', data['opponent_prob'][j, :, t], truth)
            close(name + ' online opponent draw', b, (uniforms[:, t] < truth).astype(int), 0)
            close(name + ' online focal reward', data['online_reward'][j, :, t],
                  (a == b).astype(float) + data['bonus'][row, c]*a)
            close(name + ' online expected reward', data['online_expected'][j, :, t],
                  np.where(a == 1, truth + data['bonus'][row, c], 1-truth))
            close(name + ' online opponent reward', data['opponent_reward'][j, :, t],
                  (a == b) * signs[row, c] * gain[row, c])
            close(name + ' opponent state', data['field_abs'][j, :, t], abs(m))
            update = (1-rate)*m + rate*(2*a-1)
            if shifted:
                belief[row, c] = update
            else:
                belief[:, 0] = update


def independent_endpoints(blocks):
    results = [[], []]
    for block in blocks:
        with np.load(block / 'metrics.npz', allow_pickle=False) as data:
            names = data['methods'].tolist()
            same = data['signs'][:, 0] == data['signs'][:, 1]
            for j, (left, right, q) in enumerate((('g128_p75_mix', 'independent', .75),
                                                   ('g128_p50_mix', 'g128_p100_mix', .5))):
                li, ri = names.index(left), names.index(right)
                l = data['value'][li, :, 3, 1] - data['value'][li, :, 0, 1]
                r = data['value'][ri, :, 3, 1] - data['value'][ri, :, 0, 1]
                d = l-r
                results[j].append(q*d[same].mean() + (1-q)*d[~same].mean())
    rng = np.random.default_rng(79113)
    estimates, intervals = [], []
    for values in results:
        values = np.array(values)
        estimates.append(float(values.mean()))
        resample = rng.integers(0, len(values), (10000, len(values)))
        intervals.append(np.quantile(values[resample].mean(axis=1), [.0125, .9875]).tolist())
    return estimates, intervals

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        if hashlib.sha256((output / 'source' / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'{name}: archived source checksum mismatch')
    blocks = sorted(output.glob('block_[0-9][0-9][0-9]'))
    if len(blocks) != manifest['config']['blocks']:
        raise AssertionError('block count differs from manifest')
    report = {'blocks': len(blocks), 'training_fits_checked': 0, 'query_sets_checked': 0}
    for block in blocks:
        record = json.loads((block / 'record.json').read_text())
        for name, digest in record['sha256'].items():
            if hashlib.sha256((block / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f'{block / name}: checksum mismatch')
        fits = {}
        all_train = []
        train_files = sorted(block.glob('train_*.npz'))
        if len(train_files) != 12:
            raise AssertionError(f'{block}: expected 12 training arms')
        common_by_count = {}
        for file in train_files:
            with np.load(file, allow_pickle=False) as archive:
                data = dict(archive)
            count, resets, horizon = data['arena'].shape
            if count * resets * horizon != 2048 or horizon != 16:
                raise AssertionError(f'{file}: unexpected training interaction budget')
            arm = file.stem.removeprefix('train_')
            count_spec, p_spec, speed_spec = arm.split('_')
            if count != int(count_spec[1:]):
                raise AssertionError(f'{file}: game count mismatch')
            same = data['signs'][:, 0] == data['signs'][:, 1]
            close(f'{file} concordance', same.mean(), int(p_spec[1:])/100)
            close(f'{file} fast fraction', (data['rate'] == .65).mean(), .5 if speed_spec == 'mix' else 0)
            for sign0 in (-1, 1):
                for same_value in (False, True):
                    for fast_value in (False, True):
                        stratum = ((data['signs'][:, 0] == sign0) & (same == same_value)
                                   & ((data['rate'] == .65) == fast_value))
                        desired = (count * .5 * (int(p_spec[1:])/100 if same_value else 1-int(p_spec[1:])/100)
                                   * ((.5 if speed_spec == 'mix' else 0) if fast_value
                                      else (.5 if speed_spec == 'mix' else 1)))
                        close(f'{file} stratum', stratum.sum(), desired)
            shared = common_by_count.setdefault(count, {key: data[key] for key in ('gain', 'bonus', 'arena', 'a')})
            for key, expected_shared in shared.items():
                close(f'{file} common {key}', data[key], expected_shared, 0)
            all_train.append(data['gain'])
            expected = likelihood(data['gain'], data['arena'], data['a'], data['b'])
            close(str(file) + ' likelihood', data['loglik'], expected)
            close(str(file) + ' fit', data['fit'], em_fit(expected))
            fits[arm] = data['fit']
            if arm == 'g128_p75_mix':
                coef = data['pooled']
                verify_pooled_fit(data)
            report['training_fits_checked'] += 1
        reference_tasks = None
        for filename in ('metrics.npz', 'shift.npz'):
            file = block / filename
            if not file.exists():
                raise AssertionError(f'{file}: missing required panel')
            with np.load(file, allow_pickle=False) as archive:
                data = dict(archive)
            input_keys = ('signs', 'rate', 'gain', 'bonus', 'source', 'probe_a', 'probe_u', 'online_arena', 'online_u')
            if reference_tasks is None:
                reference_tasks = {key: data[key] for key in input_keys}
            else:
                for key in input_keys:
                    close(f'{file}: paired task {key}', data[key], reference_tasks[key], 0)
            if len(data['rate']) != 8 * manifest['config']['reps']:
                raise AssertionError(f'{file}: unexpected evaluation task count')
            for s0, s1, rate in CANDIDATES:
                mask = (data['signs'][:, 0] == s0) & (data['signs'][:, 1] == s1) & (data['rate'] == rate)
                if mask.sum() != manifest['config']['reps']:
                    raise AssertionError(f'{file}: unbalanced sign/rate stratum')
            test_gain = set(map(tuple, data['gain']))
            if any(test_gain.intersection(map(tuple, gain)) for gain in all_train):
                raise AssertionError(f'{file}: public game gain overlap with training')
            verify_probe(data, fits, coef, filename == 'shift.npz')
            verify_online(data, fits, coef, filename == 'shift.npz')
            report['query_sets_checked'] += 1
    report['primary_estimates'], report['bootstrap_97_5_intervals'] = independent_endpoints(blocks)
    inference = output / 'analysis' / 'primary_inference.csv'
    if inference.exists():
        with inference.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        close('analysis primary estimates', [float(row['estimate']) for row in rows], report['primary_estimates'])
        close('analysis lower bootstrap bounds', [float(row['lower_97_5']) for row in rows], [v[0] for v in report['bootstrap_97_5_intervals']])
        close('analysis upper bootstrap bounds', [float(row['upper_97_5']) for row in rows], [v[1] for v in report['bootstrap_97_5_intervals']])
        report['analysis_estimates_and_intervals_match'] = True
    destination = output / 'verification.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
