"""Independently reconstruct labels, neural decisions and paired estimands."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.active_probe.policy import opening_features, terminal_features
from research.active_probe.verify import network, verify_data

METHODS = ('return', 'terminal_only', 'gain', 'myopic')


def close(actual, expected, name, tol=1e-6):
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=0, err_msg=name)


def check_terminal(data, weights):
    x = terminal_features(data)
    scores = network(weights['terminal'], x)
    scores = scores.transpose(0, 1, 4, 2, 3)
    selected = data['terminal_actions']
    chosen = np.take_along_axis(scores, selected[..., None], axis=-1)[..., 0]
    close(chosen, scores.max(axis=-1), 'terminal action')


def check_openers(data, weights):
    x = opening_features(data)
    for i, name in enumerate(METHODS):
        score = network(weights[name], x)[..., 0]
        close(data['scores'][:, :, i], score, name+' scores')
        selected = data['openers'][:, i]
        close(score[np.arange(len(score)), selected], score.max(axis=1), name+' action')
    donor = data['donor']
    assert np.array_equal(np.sort(donor), np.arange(len(donor)))
    assert np.all(donor != np.arange(len(donor)))
    assert np.array_equal(data['motif'][donor], data['motif'])
    donor_scores = data['scores'][donor, :, 0]
    chosen = data['openers'][:, 4]
    close(donor_scores[np.arange(len(chosen)), chosen], donor_scores.max(axis=1), 'donor action')


def targets(data):
    action = data['terminal_actions']
    values = data['next_value']
    selected = np.take_along_axis(values, action, axis=3)
    visible = selected[:, :, 0, 1]
    masked = selected[:, :, 0, 0]
    return np.stack([data['first_value']+visible, visible, visible-masked, data['first_value']], axis=-1)


def contrasts(data):
    n = len(data['openers'])
    index = np.arange(n)
    a = data['openers']
    chosen = data['terminal_actions']
    def value(method, regime, channel):
        opening = a[:, method]
        terminal = chosen[index, opening, regime, channel]
        return data['next_value'][index, opening, regime, terminal]
    def total(method):
        return data['first_reward'][index, a[:, method]] + value(method, 0, 1)
    return np.array([(total(0)-total(2)).mean(), (total(0)-total(3)).mean(),
                     ((value(0,1,1)-value(0,1,0))-(value(3,1,1)-value(3,1,0))).mean()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    manifest = json.loads((output/'manifest.json').read_text())
    blocks = sorted(output.glob('block_[0-9][0-9][0-9]'))
    assert len(blocks) == manifest['config']['blocks']
    fingerprints = set()
    effects = []
    verified = []
    for block in blocks:
        record = json.loads((block/'record.json').read_text())
        for file, expected in record['sha256'].items():
            assert hashlib.sha256((block/file).read_bytes()).hexdigest() == expected
        weights = torch.load(block/'weights.pt', map_location='cpu', weights_only=True)
        for filename in ('terminal_train.npz', 'opening_train.npz', 'eval.npz'):
            with np.load(block/filename, allow_pickle=False) as f:
                data = dict(f)
            fps = set(data['game_fingerprint'].tolist())
            assert fingerprints.isdisjoint(fps), 'game overlap'
            fingerprints.update(fps)
            info = verify_data(data)
            if filename != 'terminal_train.npz':
                check_terminal(data, weights)
            if filename == 'opening_train.npz':
                close(data['targets'], targets(data), 'opening labels')
            if filename == 'eval.npz':
                check_openers(data, weights)
                effects.append(contrasts(data))
            verified.append(dict(block=block.name, dataset=filename, **info))
    effects = np.array(effects)
    rng = np.random.default_rng(912853)
    index = rng.integers(len(effects), size=(20000, len(effects)))
    intervals = np.quantile(effects[index].mean(axis=1), [1/120,119/120], axis=0)
    summary = json.loads((output/'summary.json').read_text())
    for j, item in enumerate(summary['primary']):
        close([item['estimate'],item['lower'],item['upper']],
              [effects[:, j].mean(), intervals[0, j], intervals[1, j]], 'primary interval', 1e-12)
    result = dict(blocks=len(blocks), exact_distinct_fingerprints=len(fingerprints),
                  effects=effects.mean(axis=0).tolist(), intervals=intervals.T.tolist(),
                  raw_sha256_verified=True, checks=verified)
    (output/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__ == '__main__':
    main()
